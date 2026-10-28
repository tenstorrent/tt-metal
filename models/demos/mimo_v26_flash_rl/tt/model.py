# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The all-device MiMo model on the [8, 4] mesh: the hidden state stays on the device from the embedding to the final
norm, every block through ``run_block`` with the device modules of the component steps.

Residual layout (mesh_io.py): each chip holds [1, 1, 160, 4096] bf16, SP row r the chunk positions of
``chunk_positions(start)[r]`` (the server's block-cyclic reshuffle), TP column c rows [c * 160, (c + 1) * 160) of them.

    embed       tokens [5120] natural order (PAD_ID past the real ones) -> reshuffled [8, 1, 640] uint32, SP-sharded,
                TP-replicated (the runner's H2D layout) -> minimum(ids, V - 1) -> embedding (bf16 table, replicated)
                -> mesh_partition(dim 2, TP axis) -> [1, 1, 160, 4096]
    layer i     run_block over the reference's block graph: attn_norm, attention, attn_residual, ffn_norm,
                (router + experts | mlp), mlp_residual
    final_norm  rms_norm on the local rows
    logits      TP all_gather of the rows -> lm_head (vocab / 4 per TP column, bfloat8_b) -> host (harness boundary)

The model's state IS the served KV cache (serving_contract.md "KV cache"): every attention layer's ``LayerKV``. The
per-chunk metadata (slot, start, end, positions, masks) is staged into preallocated device buffers by ``embed``, the
runtime's metadata step, so ``layer`` does no host work.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import torch

import ttnn
from models.demos.mimo_v26_flash_rl.reference._bringup_interface import Ctx, run_block
from models.demos.mimo_v26_flash_rl.reference.model import Layer as RefLayer
from models.demos.mimo_v26_flash_rl.reference.weights import Checkpoint

from . import ccl
from .attention import Attention, ChunkMeta, RopeTables, layer_dims
from .experts import Experts
from .kv_cache import chunk_positions
from .mesh_io import residual_to_host, residual_to_mesh
from .mlp import DenseMLP
from .residual import ResidualAdd
from .rms_norm import RMSNorm
from .router import Router

PAD_ID = 0xFFFFFFFF  # tt-d-gen's pad token (engine/include/engine/runtime/types.hpp)
SLOT = 0  # the slot the bring-up harness runs (the model holds SLOTS)


class Block:
    """The device modules of one decoder layer, keyed by the reference graph's step names."""

    def __init__(self, mesh, ck, layer: int, cfg: dict, shared: dict, max_seq: int, chunk: int, slots: int, kv=None):
        dims = layer_dims(cfg, layer)
        kind = "sliding" if dims["window"] else "full"
        eps = cfg["layernorm_epsilon"]
        p = f"model.layers.{layer}"
        self.is_moe = bool(cfg["moe_layer_freq"][layer])
        self.steps = RefLayer.graph(SimpleNamespace(is_moe=self.is_moe))  # the reference's own block graph
        self.attn_norm = RMSNorm(mesh, ck.raw(f"{p}.input_layernorm.weight"), eps)
        self.ffn_norm = RMSNorm(mesh, ck.raw(f"{p}.post_attention_layernorm.weight"), eps)
        self.attn = Attention(
            mesh,
            ck,
            layer,
            cfg,
            max_seq=max_seq,
            chunk=chunk,
            slots=slots,
            ring=shared["ring"],
            meta=shared["meta"][kind],
            rope=shared["rope"][kind],
            kv=kv,
        )
        self.add = ResidualAdd()
        if self.is_moe:
            sp = tuple(mesh.shape)[0]
            self.router = Router(mesh, ck, layer)
            self.experts = Experts(mesh, ck, layer, rows_per_chip=chunk // sp)
        else:
            self.mlp = DenseMLP(mesh, ck, layer)

    def _route(self, ctx, x):
        xg = ccl.gather_rows(x)  # the column's 640 rows: the experts' dispatch-group input
        out = self.router(xg)
        xg.deallocate(True)
        return out

    def component(self, name: str):
        return {
            "attn_norm": lambda ctx, x: self.attn_norm(x),
            "attention": lambda ctx, x: self.attn(x),
            "attn_residual": lambda ctx, a, b: self.add(a, b),
            "ffn_norm": lambda ctx, x: self.ffn_norm(x),
            "router": self._route,
            "experts": lambda ctx, x, r: self.experts(x, r),
            "mlp": lambda ctx, x: self.mlp(x),
            "mlp_residual": lambda ctx, a, b: self.add(a, b),
        }[name]


class State:
    """The served KV caches of the model's layers (serving_contract.md "KV cache"), slot ``SLOT`` for the harness."""

    def __init__(self, model: "MimoModel"):
        self.model = model

    def formats(self) -> dict:
        return {"key": ttnn.bfloat8_b, "value": ttnn.bfloat8_b}

    def kv_tensors(self, layer: int) -> dict:
        return self.model.blocks[layer].attn.kv.tensors()

    def load_prefix(self, layer: int, tensors: dict, length: int) -> None:
        """Harness boundary: the golden prefix [0, length) of one layer into the served caches; the next ``embed``
        without a start continues at ``length``."""
        self.model.blocks[layer].attn.load_prefix(SLOT, tensors["key"], tensors["value"], length)
        self.model.cursor = length

    def to_torch(self, layer: int, length: int) -> dict:
        """{key: [KV, length, 192], value: [KV, length, 128]} float32 of slot ``SLOT``, global positions (inverse of
        ``Attention.load_prefix``'s block-cyclic layout)."""
        m = self.model
        sp, tp, chunk = m.sp, m.tp, m.chunk
        w = chunk // sp
        g = torch.arange(length)
        row, local = (g // w) % sp, (g // chunk) * w + g % w
        out = {}
        for name, t in self.kv_tensors(layer).items():
            # per chip [slots, local heads, local_seq, width], chips in row-major mesh order (SP row, TP column)
            shards = [ttnn.to_torch(d).float()[SLOT] for d in ttnn.get_device_tensors(t)]
            per_row = [torch.cat(shards[r * tp : (r + 1) * tp], dim=0) for r in range(sp)]  # [KV, local_seq, width]
            host = torch.stack(per_row)  # [sp, KV, local_seq, width]
            out[name] = host[row, :, local].permute(1, 0, 2).contiguous()  # [KV, length, width]
        return out


class MimoModel:
    def __init__(self, mesh, spec, layers, lm_head: bool = True, slots: int = 2, kv_caches: dict | None = None):
        """kv_caches: {layer: LayerKV}, the engine-owned served caches (tt/runners/kv_caches.py); None allocates them."""
        t0 = time.time()
        self.mesh = mesh
        self.sp, self.tp = tuple(mesh.shape)
        self.chunk, self.max_seq = int(spec.get("target.chunk")), int(spec.get("target.seq"))
        self.layers = list(layers)
        ck = Checkpoint(spec.hf_dir, torch.float32)
        cfg = ck.config
        self.vocab = cfg["vocab_size"]

        shared = {"ring": ccl.RingResources(mesh), "meta": {}, "rope": {}}
        for kind, layer in (("full", 0), ("sliding", 1)):  # MiMo: layer 0 full, layer 1 sliding
            dims = layer_dims(cfg, layer)
            rope_dim = int(dims["head_dim"] * cfg["partial_rotary_factor"])  # the reference Layer's inv_freq formula
            inv_freq = 1.0 / (
                dims["rope_theta"] ** (torch.arange(0, rope_dim, 2, dtype=torch.int64).float() / rope_dim)
            )
            shared["meta"][kind] = ChunkMeta(mesh, self.chunk, window=dims["window"])
            shared["rope"][kind] = RopeTables(mesh, inv_freq, self.max_seq)
        self.metas = list(shared["meta"].values())
        kv_caches = kv_caches or {}
        self.blocks = {
            i: Block(mesh, ck, i, cfg, shared, self.max_seq, self.chunk, slots, kv=kv_caches.get(i))
            for i in self.layers
        }

        rep = ttnn.ReplicateTensorToMesh(mesh)
        self.embed_w = ttnn.from_torch(
            ck.raw("model.embed_tokens.weight").to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )
        self.norm = RMSNorm(mesh, ck.raw("model.norm.weight"), cfg["layernorm_epsilon"])
        self.lm_head = None
        if lm_head:  # [V / tp, H] per TP column, bfloat8_b, [out, in] with transpose_b
            w = ck.raw("lm_head.weight").to(torch.bfloat16)
            self.lm_head = ttnn.from_torch(
                w.view(1, 1, self.vocab, -1),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(self.sp, self.tp), dims=(None, 2)),
            )
            self.head_cfg = ttnn.init_device_compute_kernel_config(
                mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
            )
        self.staged = None  # (start, end) of the metadata on the device
        self.staged_slot = SLOT
        self.cursor = 0  # the start of the next chunk when embed is not given one
        self.load_seconds = time.time() - t0

    # ---- the runtime's per-chunk metadata step (host -> device, outside the layers)
    def _stage(self, start: int, end: int, slot: int = SLOT) -> None:
        if self.staged != (start, end) or self.staged_slot != slot:
            for meta in self.metas:
                meta.stage(slot, start, end)
            self.staged, self.staged_slot = (start, end), slot

    def _perm(self) -> torch.Tensor:
        """Natural row index of every device row (SP row, then TP column, then local row) of the staged chunk."""
        start = self.staged[0] if self.staged else 0
        return (chunk_positions(start, self.sp, self.chunk) - start).flatten()

    # ---- harness / serving interface
    def new_state(self, max_seq: int) -> State:
        assert max_seq <= self.max_seq, f"the caches hold {self.max_seq} positions, asked {max_seq}"
        for b in self.blocks.values():
            b.attn.kv.zero()
        self.cursor, self.staged = 0, None
        return State(self)

    def embed(self, tokens: torch.Tensor, start: int | None = None):
        """One chunk of tokens in natural order (PAD_ID past the real ones), its first token at absolute position
        ``start`` (default: right after the previous chunk) -> the residual rows on the device."""
        start = self.cursor if start is None else start
        t = torch.full((self.chunk,), PAD_ID, dtype=torch.int64)
        n = min(tokens.numel(), self.chunk)
        t[:n] = tokens.reshape(-1)[:n].long() & 0xFFFFFFFF
        real = int((t != PAD_ID).sum())
        self._stage(start, start + real)
        ids = ttnn.from_torch(
            t[self._perm()].view(self.sp, 1, self.chunk // self.sp),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=(self.sp, self.tp), dims=(0, None)),
        )
        return self.embed_device(ids)

    def embed_device(self, ids):
        """ids per chip [1, 1, 640] uint32 (the runner's H2D tensor) -> [1, 1, 160, H] bf16."""
        dram = ttnn.DRAM_MEMORY_CONFIG
        safe = ttnn.minimum(ids, self.vocab - 1)  # PAD_ID never indexes the table; pad rows stay pad
        ids.deallocate(True)
        x = ttnn.embedding(safe, self.embed_w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=dram)
        safe.deallocate(True)
        x4 = ttnn.unsqueeze_to_4D(x)
        out = ttnn.mesh_partition(x4, dim=2, cluster_axis=ccl.TP_AXIS, memory_config=dram)
        x4.deallocate(True)
        return out

    def from_host(self, h: torch.Tensor):
        return residual_to_mesh(self.mesh, h[self._perm()])

    def to_host(self, h) -> torch.Tensor:
        dev = residual_to_host(self.mesh, h)
        out = torch.empty_like(dev)
        out[self._perm()] = dev
        return out

    def layer(self, i: int, h, start: int, state: State, end: int | None = None):
        """Block ``i`` on the device rows ``h`` of the chunk at ``start``; rows at or past ``end`` are pad."""
        if self.staged is None or self.staged[0] != start or (end is not None and self.staged[1] != end):
            # a start embed did not stage
            self._stage(start, start + self.chunk if end is None else end, self.staged_slot)
        b = self.blocks[i]
        inner = []
        ctx = Ctx(layer=i, start=start, length=self.chunk, state=state)
        # ttnn runs its pre-operation hooks (what the harness's device-execution proof counts) only outside fast
        # runtime mode; the ops and their results are the same either way
        prev = ttnn.CONFIG.enable_fast_runtime_mode
        ttnn.CONFIG.enable_fast_runtime_mode = False
        try:
            out = run_block(b.steps, b.component, ctx, h, rec=lambda name, t: inner.append((name, t)))
        finally:
            ttnn.CONFIG.enable_fast_runtime_mode = prev
        for name, t in inner:  # every boundary but the block's input and output
            if name not in ("in", "out"):
                t.deallocate(True)
        self.cursor = start + self.chunk
        return out

    def final_norm(self, h):
        return self.norm(h)

    def logits(self, hidden, rows) -> torch.Tensor:
        """[len(rows), V] float32 of the natural-order ``rows`` (harness boundary)."""
        assert self.lm_head is not None, "device_model(lm_head=False) has no lm_head"
        xg = ccl.gather_rows(hidden)  # [1, 1, 640, H]: the SP row's rows
        lg = ttnn.linear(
            xg,
            self.lm_head,
            transpose_b=True,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.head_cfg,
        )
        xg.deallocate(True)
        # [sp, 1, 640, V]: SP rows along dim 0, the TP columns' vocab slices along dim 3
        host = ttnn.to_torch(
            lg, mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh, mesh_shape=(self.sp, self.tp), dims=(0, 3))
        )
        lg.deallocate(True)
        dev = host.reshape(self.chunk, -1)
        nat = torch.empty_like(dev)
        nat[self._perm()] = dev
        return nat[torch.as_tensor(rows).long()]

    def free(self, h) -> None:
        h.deallocate(True)

    def sync(self) -> None:
        ttnn.synchronize_device(self.mesh)
