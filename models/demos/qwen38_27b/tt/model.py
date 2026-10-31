# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The all-device Qwen3.8-27B prefill model on the 8 x 4 mesh: the hidden state stays on the device from the embedding
to the final norm, every block through run_block (reference/_bringup_interface.py).

Residual layout between blocks: [1, 1, 640, 5120] bf16 per chip, the chunk's rows in the engine's chip order
(gdn.chip_order: token start + n on SP row (start + n) / 640 % 8, the ring_sdpa_reshuffle the KV writer expects),
replicated over the 4 TP columns.

State (TtModelState), the served one from the start (serving_contract.md "KV cache"):
  attention layers  one TtKVCache, batch slot * n_attn_layers + attn_layer, bfp8 ND-sharded [1, 1, 32, 256] records
  GDN layers        per (slot, layer) conv tail [1, 4, 2560] bf16 and recurrent [1, 12, 128, 128] fp32 per TP column,
                    the position it is at, and restart snapshots at align_down(end, KV_BLOCK_SIZE) and at
                    max_seq - chunk (the server's two restart points below the state, the "hybrid rule")
"""

import contextlib
import time

import torch

import ttnn

from ..reference._bringup_interface import Ctx, Step, run_block
from ..reference.qwen38_ref import PREFIX, Checkpoint, Config
from .attention import TtGatedAttention
from .gdn import TtGatedDeltaNet, chip_order
from .kv_cache import TtKVCache
from .mlp import TtMLP
from .mlp import compute_config as mlp_compute_config
from .norm import TtRMSNorm
from .residual import TtResidualAdd
from .settings import settings

PAD_ID = 0xFFFFFFFF  # the engine's pad token id (out of vocab)


def block_graph() -> list[Step]:
    """The reference's block graph (Qwen38Reference.block_graph), the same for both block types."""
    return [
        Step("attn_norm", ("in",), "attn_norm", "norm"),
        Step("attention", ("attn_norm",), "attn_out", "attention", stateful=True),
        Step("attn_residual", ("in", "attn_out"), "h_mid", "residual"),
        Step("ffn_norm", ("h_mid",), "ffn_norm", "norm"),
        Step("mlp", ("ffn_norm",), "mlp_out", "mlp"),
        Step("mlp_residual", ("h_mid", "mlp_out"), "out", "residual"),
    ]


@contextlib.contextmanager
def hooked_dispatch():
    """ttnn's fast runtime (FastOperation) skips the pre-operation hooks that count device ops; the standard runtime
    runs the same device ops and calls them."""
    fast = ttnn.CONFIG.enable_fast_runtime_mode
    ttnn.CONFIG.enable_fast_runtime_mode = False
    try:
        yield
    finally:
        ttnn.CONFIG.enable_fast_runtime_mode = fast


class GdnSlot:
    """The GDN state of one (slot, layer): the live state at position pos and the restart snapshots."""

    def __init__(self, state: dict):
        self.state, self.pos, self.snaps = state, 0, {}


class TtModelState:
    """The served state of every layer of the model, for num_slots users (module docstring)."""

    def __init__(self, model: "TtQwen38Model", max_seq: int):
        self.model, self.max_seq = model, max_seq
        self.num_slots = settings.get("NUM_SLOTS")
        cfg = model.cfg
        self.attn_layers = [i for i in model.layers if model.is_attn(i)]
        self.kv = None
        if self.attn_layers:
            self.kv = TtKVCache(
                model.mesh,
                cfg.n_kv,
                cfg.head_dim,
                max_seq,
                model.chunk,
                num_slots=self.num_slots,
                num_layers=len(self.attn_layers),
            )
        self.gdn = {
            (s, i): GdnSlot(TtGatedDeltaNet.clone_state(model.gdn_zero))
            for s in range(self.num_slots)
            for i in model.layers
            if not model.is_attn(i)
        }

    # ---- serving hooks
    def formats(self) -> dict:
        return {"key": ttnn.bfloat8_b, "value": ttnn.bfloat8_b, "conv": ttnn.bfloat16, "recurrent": ttnn.float32}

    def cache_tensors(self, layer: int, slot: int = 0) -> dict:
        if self.model.is_attn(layer):
            return self.kv.cache_tensors()
        return dict(self.gdn[(slot, layer)].state)

    # ---- harness boundary (host I/O allowed here)
    def to_torch(self, layer: int, length: int, slot: int = 0) -> dict:
        if self.model.is_attn(layer):
            return self.kv.to_torch(slot, self.attn_layers.index(layer), length)
        return self.model.blocks[layer].gdn.state_to_torch(self.gdn[(slot, layer)].state)

    def load_prefix(self, layer: int, tensors: dict, length: int, slot: int = 0) -> None:
        if self.model.is_attn(layer):
            self.kv.load_layer_prefix(slot, self.attn_layers.index(layer), tensors["key"], tensors["value"], length)
            return
        g = self.gdn[(slot, layer)]
        TtGatedDeltaNet.free_state(g.state)
        for s in g.snaps.values():
            TtGatedDeltaNet.free_state(s)
        g.state = self.model.blocks[layer].gdn.state_to_device(tensors["conv"], tensors["recurrent"])
        g.pos, g.snaps = length, {}

    def free(self) -> None:
        if self.kv is not None:
            self.kv.free()
        for g in self.gdn.values():
            TtGatedDeltaNet.free_state(g.state)
            for s in g.snaps.values():
                TtGatedDeltaNet.free_state(s)
        self.gdn = {}

    # ---- the GDN restart rule (serving_contract.md "Attention and cache writes")
    def gdn_begin(self, slot: int, layer: int, start: int) -> dict:
        """The state a chunk at start runs from: zeros at 0, the live state at its position, else the snapshot taken
        exactly at start. Any other start would fold the prefix twice or skip part of it: refused."""
        g = self.gdn[(slot, layer)]
        if start == g.pos and start != 0:
            return g.state
        if start == 0:
            fresh = TtGatedDeltaNet.clone_state(self.model.gdn_zero)
        elif start in g.snaps:
            fresh = TtGatedDeltaNet.clone_state(g.snaps[start])
        else:
            raise ValueError(
                f"GDN layer {layer} slot {slot}: a chunk at {start} needs the state at {start}, but the state is at "
                f"{g.pos} and snapshots exist only at {sorted(g.snaps)}"
            )
        TtGatedDeltaNet.free_state(g.state)
        g.state, g.pos = fresh, start
        if start == 0:
            for s in g.snaps.values():
                TtGatedDeltaNet.free_state(s)
            g.snaps = {}
        return g.state

    def snap_points(self, start: int, end: int) -> list[int]:
        """Positions in (start, end] to keep a snapshot at: align_down(end, kv_block_size), and max_seq - chunk when
        this chunk crosses it."""
        pts = {end - end % settings.get("KV_BLOCK_SIZE"), self.max_seq - self.model.chunk}
        return sorted(p for p in pts if start < p <= end)

    def gdn_end(self, slot: int, layer: int, start: int, end: int, state: dict, snaps: dict) -> None:
        """After a chunk [start, end): state is the new live state, snaps the states inside (start, end]. Keeps the
        snapshots at this chunk's restart points (snap_points, and its own start when that is one)."""
        g = self.gdn[(slot, layer)]
        keep = {end - end % settings.get("KV_BLOCK_SIZE"), self.max_seq - self.model.chunk}
        new = {p: s for p, s in g.snaps.items() if p in keep and p <= start}
        if start in keep and start not in new and start != 0:
            new[start] = g.state  # the state this chunk began from, a restart point
        else:
            TtGatedDeltaNet.free_state(g.state)
        for p, s in g.snaps.items():
            if new.get(p) is not s:
                TtGatedDeltaNet.free_state(s)
        for p, s in snaps.items():
            if p in keep and p not in new:
                new[p] = s
            else:
                TtGatedDeltaNet.free_state(s)
        g.state, g.pos, g.snaps = state, end, new


class TtBlock:
    """One decoder layer's device modules (weights loaded once)."""

    def __init__(self, mesh, cfg, ckpt, layer: int, attn: bool, max_seq: int, chunk: int):
        def weights(sub):
            p = f"{PREFIX}layers.{layer}.{sub}."
            return {f"{sub}." + n[len(p) :]: ckpt.get(n, torch.bfloat16) for n in ckpt.index if n.startswith(p)}

        get = lambda n: ckpt.get(f"{PREFIX}layers.{layer}.{n}", torch.bfloat16)
        self.attn_norm = TtRMSNorm(mesh, get("input_layernorm.weight"), cfg.eps)
        self.ffn_norm = TtRMSNorm(mesh, get("post_attention_layernorm.weight"), cfg.eps)
        self.attn = TtGatedAttention(mesh, cfg, weights("self_attn"), max_seq, chunk) if attn else None
        self.gdn = None if attn else TtGatedDeltaNet(mesh, cfg, weights("linear_attn"), chunk=chunk)
        self.mlp = TtMLP(mesh, cfg, weights("mlp"))
        self.add = TtResidualAdd()


class TtQwen38Model:
    """device_model(mesh, spec, layers): the harness / serving interface of bringup/hooks.py's docstring."""

    def __init__(self, mesh, spec, layers, lm_head: bool = True):
        t0 = time.time()
        self.mesh = mesh
        self.sp, self.tp = mesh.shape[0], mesh.shape[1]
        path = str(spec.hf_dir)
        self.cfg = cfg = Config.load(path)
        ckpt = Checkpoint(path)
        self.layers = [int(i) for i in layers]
        self.max_seq, self.chunk = int(spec.get("target")["seq"]), int(spec.get("target")["chunk"])
        self.rows = self.chunk // self.sp
        self.graph = block_graph()
        self.blocks = {i: TtBlock(mesh, cfg, ckpt, i, self.is_attn(i), self.max_seq, self.chunk) for i in self.layers}
        gdn = next((b.gdn for b in self.blocks.values() if b.gdn is not None), None)
        self.gdn_zero = None
        if gdn is not None:
            self.gdn_zero = gdn.state_to_device(
                torch.zeros(cfg.conv_dim, cfg.conv_k), torch.zeros(cfg.lv_heads, cfg.lk_dim, cfg.lv_dim)
            )
        # embedding table [vocab, hidden / tp] bf16 ROW_MAJOR per TP column (hidden split, replicated over SP rows)
        emb = ckpt.get(f"{PREFIX}embed_tokens.weight", torch.bfloat16)
        self.embed_w = ttnn.from_torch(
            emb.reshape(1, 1, *emb.shape),
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(None, 3)),
        )
        del emb
        self.final = TtRMSNorm(mesh, ckpt.get(f"{PREFIX}norm.weight", torch.bfloat16), cfg.eps)
        self.lm_head = None
        if lm_head:
            # [vocab / tp, hidden] bfp8 per TP column ([out, in] with transpose_b, as the other projections)
            w = ckpt.get("lm_head.weight", torch.bfloat16)
            self.lm_head = ttnn.from_torch(
                w.reshape(1, 1, *w.shape),
                device=mesh,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(None, 2)),
            )
            del w
        self.last_start = 0  # the chip order of the latest embedded chunk (to_host undoes it)
        self.load_seconds = time.time() - t0

    def is_attn(self, layer: int) -> bool:
        return self.cfg.layer_types[layer] == "full_attention"

    def new_state(self, max_seq: int) -> TtModelState:
        assert max_seq <= self.max_seq, (max_seq, self.max_seq)
        return TtModelState(self, max_seq)

    # ---- chunk boundary (host I/O allowed)
    def _order(self, start: int) -> torch.Tensor:
        """Chip-order index of each natural row of a chunk at start."""
        return torch.tensor(chip_order(start, self.chunk, self.sp))

    def embed(self, tokens: torch.Tensor, start: int = 0) -> ttnn.Tensor:
        """One chunk of token ids in natural order (PAD_ID past the real tokens) -> the residual layout of a chunk at
        start. The ids are laid out as the engine's payload ([sp, 1, 640] uint32, chip order); the pad id is clamped
        into the vocab on the device (pad rows are never read by a real row)."""
        T = self.chunk
        ids = torch.full((T,), PAD_ID, dtype=torch.int64)
        ids[: tokens.shape[0]] = tokens.reshape(-1).to(torch.int64) & 0xFFFFFFFF
        payload = torch.empty_like(ids)
        payload[self._order(start)] = ids
        self.last_start = start
        dev = ttnn.from_torch(
            payload.reshape(self.sp, 1, self.rows),
            device=self.mesh,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=tuple(self.mesh.shape), dims=(0, None)),
        )
        return self.embed_device(dev)

    def embed_device(self, ids: ttnn.Tensor) -> ttnn.Tensor:
        """[1, 1, 640] uint32 ids per chip (engine payload) -> [1, 1, 640, hidden] bf16 replicated over TP."""
        dram = ttnn.DRAM_MEMORY_CONFIG
        safe = ttnn.minimum(ids, self.cfg.vocab - 1)
        e = ttnn.embedding(safe, self.embed_w, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=dram)
        ttnn.deallocate(safe)
        e = ttnn.reshape(e, (1, 1, self.rows, self.cfg.hidden // self.tp))
        h = ttnn.all_gather(e, dim=3, cluster_axis=1, memory_config=dram)
        ttnn.deallocate(e)
        return h

    def from_host(self, h: torch.Tensor) -> ttnn.Tensor:
        """Host [S, hidden] (natural order of a chunk at last_start) -> the residual layout."""
        x = torch.empty_like(h)
        x[self._order(self.last_start)] = h
        return ttnn.from_torch(
            x.reshape(1, 1, *x.shape).to(torch.bfloat16),
            device=self.mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=tuple(self.mesh.shape), dims=(2, None)),
        )

    def to_host(self, h: ttnn.Tensor) -> torch.Tensor:
        """The residual layout -> host [S, hidden] in natural order (TP column 0's copy)."""
        cat = ttnn.ConcatMesh2dToTensor(self.mesh, mesh_shape=tuple(self.mesh.shape), dims=(2, 3))
        out = ttnn.to_torch(h, mesh_composer=cat)[0, 0, :, : self.cfg.hidden].float()
        return out[self._order(self.last_start)]

    # ---- device forward
    def layer(self, i: int, h: ttnn.Tensor, start: int, state: TtModelState, end: int | None = None, slot: int = 0):
        """One decoder layer on the device for the chunk at start (rows at or past end are pad). Writes the layer's
        state in place; returns the block output (h is left to the caller)."""
        blk = self.blocks[i]
        T = self.chunk
        end = start + T if end is None else end
        snaps = {}
        gdn_state = None

        def attention(ctx, x):
            nonlocal gdn_state, snaps
            if blk.attn is not None:
                kv = state.kv
                return blk.attn(x, start, kv, slot=slot, cache_layer=state.attn_layers.index(i), end=end)
            init = state.gdn_begin(slot, i, start)
            out, gdn_state, snaps = blk.gdn(x, start, init, end=end, snap_at=state.snap_points(start, end))
            return out

        table = {
            "attn_norm": lambda ctx, x: blk.attn_norm(x),
            "attention": attention,
            "attn_residual": lambda ctx, a, b: blk.add(a, b),
            "ffn_norm": lambda ctx, x: blk.ffn_norm(x),
            "mlp": lambda ctx, x: blk.mlp(x),
            "mlp_residual": lambda ctx, a, b: blk.add(a, b),
        }
        inner = {}

        def rec(name, t):
            if name not in ("in", "out"):
                inner[name] = t

        with hooked_dispatch():
            out = run_block(self.graph, lambda name: table[name], Ctx(i, start, T, state), h, rec)
            for t in inner.values():
                ttnn.deallocate(t)
        if blk.gdn is not None:
            state.gdn_end(slot, i, start, end, gdn_state, snaps)
        return out

    def final_norm(self, h: ttnn.Tensor) -> ttnn.Tensor:
        with hooked_dispatch():
            return self.final(h)

    def logits(self, hidden: ttnn.Tensor, rows) -> torch.Tensor:
        """lm_head on the device over this chip's rows (vocab split over TP), gathered over TP; the asked rows (natural
        positions in the chunk) read back."""
        dram = ttnn.DRAM_MEMORY_CONFIG
        part = ttnn.linear(
            hidden,
            self.lm_head,
            transpose_b=True,
            compute_kernel_config=mlp_compute_config(),
            dtype=ttnn.float32,
            memory_config=dram,
        )
        cat = ttnn.ConcatMesh2dToTensor(self.mesh, mesh_shape=tuple(self.mesh.shape), dims=(2, 3))
        full = ttnn.to_torch(part, mesh_composer=cat)[0, 0].float()  # [T, vocab] chip order
        ttnn.deallocate(part)
        return full[self._order(self.last_start)][torch.as_tensor(rows).long()]

    def free(self, h) -> None:
        if h is not None:
            ttnn.deallocate(h)

    def sync(self) -> None:
        ttnn.synchronize_device(self.mesh)
