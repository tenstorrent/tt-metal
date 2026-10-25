# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The all-device Qwen3.8-27B prefill model on the 8x4 mesh (SP 8 on axis 0, TP 4 on axis 1).

The hidden state stays on the device from the embedding to the final norm: [1, 1, 640, 5120] bf16 per chip, the
chunk's rows split over the mesh rows (row r holds chunk offsets [r * 640, (r + 1) * 640), the server's block-cyclic
layout for a chunk-aligned start), replicated over the columns. Every block runs through ``run_block`` with the
validated device modules (tt/norm.py, tt/attention.py, tt/linear_attention.py, tt/mlp.py, tt/residual.py).

    embed       uint32 ids [1, 1, 1, 640] per chip -> ttnn.minimum(ids, V - 1) (the engine's 0xFFFFFFFF pad id) ->
                ttnn.embedding on the replicated bf16 table
    layer i     attn_norm -> attention (GDN or gated full attention) -> attn_residual -> mlp_norm -> mlp -> mlp_residual
    final_norm  ttnn.rms_norm; logits: lm_head column-parallel over TP (vocab / 4 per column)

State (TtState) is the served one: one KV cache for every full-attention layer (bfloat8_b, ND-sharded [1, 1, 32, 256]
ROUND_ROBIN_1D over the DRAM banks, user-major batch slot * n_attn + attention ordinal) and per (slot, linear layer)
the GDN conv history (bf16) and recurrent state (fp32). A chunk at start 0 begins a new request: the linear layers
start from a zero state. actual_start / actual_end reach the GDN ops as replicated uint32 device scalars, built once
per state for the chunk-aligned starts.
"""

from __future__ import annotations

import time

import torch

import ttnn
from models.demos.qwen38_27b.reference._bringup_interface import Ctx, run_block
from models.demos.qwen38_27b.reference.model import (
    ATTN_NAMES,
    COMMON_NAMES,
    LINEAR_NAMES,
    PREFIX,
    Checkpoint,
    Config,
    Reference,
)

from .attention import KvCache, TtGatedAttention, _Ring, blockcyclic_positions, kv_memory_config
from .linear_attention import GdnState, TtGatedDeltaNet
from .mlp import TtMlp
from .norm import TtRmsNorm, compute_config
from .residual import TtResidualAdd
from .runners.kv_cache import Qwen38KvCaches
from .runners.kv_cache import allocate as allocate_caches
from .settings import settings


def rows_to_device(mesh, x: torch.Tensor):
    """A host [S, H] tensor as the residual stream (harness boundary only)."""
    return ttnn.from_torch(
        x.reshape(1, 1, *x.shape[-2:]).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(2, None)),
    )


def rows_to_host(mesh, t) -> torch.Tensor:
    """The residual stream back as [S, H] fp32 (column 0 of each mesh row; the columns are replicas)."""
    rows, cols = tuple(mesh.shape)
    shards = ttnn.get_device_tensors(t)
    out = torch.cat([ttnn.to_torch(shards[r * cols]) for r in range(rows)], dim=-2)
    return out.reshape(out.shape[-2], out.shape[-1]).float()


class TtState:
    """The served state of one model (module docstring). ``slot`` selects the slot the chunks run on. ``caches``: the
    engine-owned serving state (tt/runners/kv_cache.py) to run on; by default the model allocates its own the same
    way, so the ladder and the prefill runner run on one cache format."""

    def __init__(self, model: "TtModel", max_seq: int, caches: Qwen38KvCaches | None = None):
        self.model = model
        chunk = model.chunk
        self.capacity = -(-max_seq // chunk) * chunk
        self.slot = 0
        if caches is None:
            caches = allocate_caches(model.mesh, model.cfg, model.layers, model.slots, chunk, max_seq)
        assert caches.chunk == chunk and caches.capacity == self.capacity, (caches.chunk, caches.capacity)
        assert caches.attn_layers == model.attn_layers and caches.linear_layers == model.linear_layers
        self.caches = caches
        if model.attn_layers:
            _Ring.shared(model.mesh, self.capacity, model.cfg.head_dim)  # the ring SDPA's scratch, at load time
        for p in range(0, self.capacity + 1, chunk):  # actual_start / full-chunk actual_end scalars, at load time
            model.scalar(p)

    @property
    def kv(self) -> KvCache | None:
        return self.caches.kv

    @kv.setter
    def kv(self, value: KvCache | None):
        self.caches.kv = value

    @property
    def gdn(self) -> dict[tuple[int, int], GdnState]:
        """(slot, linear layer) -> GdnState; a chunk replaces its slot's entry with the state after it."""
        return self.caches.gdn

    @gdn.setter
    def gdn(self, value: dict):
        self.caches.gdn = value

    # ---- serving / contract
    def formats(self) -> dict:
        return {"key": ttnn.bfloat8_b, "value": ttnn.bfloat8_b, "conv": ttnn.bfloat16, "recurrent": ttnn.float32}

    def device_tensors(self, layer: int) -> dict:
        if layer in self.model.attn_layers:
            return {"key": self.kv.k, "value": self.kv.v}
        st = self.gdn[(self.slot, layer)]
        return {"conv": st.conv, "recurrent": st.recurrent}

    def free(self):
        if self.kv is not None:
            ttnn.deallocate(self.kv.k)
            ttnn.deallocate(self.kv.v)
            self.kv = None
        for st in self.gdn.values():
            ttnn.deallocate(st.conv)
            ttnn.deallocate(st.recurrent)
        self.gdn = {}

    # ---- host <-> device (harness boundary only)
    def _kv_host(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The whole KV cache in natural order: [slots * n_attn, n_kv, capacity, hd] each."""
        m = self.model
        rows, cols = tuple(m.mesh.shape)
        idx = blockcyclic_positions(rows, m.chunk, self.capacity)
        composer = ttnn.ConcatMesh2dToTensor(m.mesh, mesh_shape=(rows, cols), dims=(2, 1))
        out = []
        for t in (self.kv.k, self.kv.v):
            dev = ttnn.to_torch(t, mesh_composer=composer).float()  # device-major rows
            nat = torch.empty_like(dev)
            nat[:, :, idx] = dev
            out.append(nat)
        return out[0], out[1]

    def to_torch(self, layer: int, length: int) -> dict:
        m = self.model
        if layer in m.attn_layers:
            k, v = self._kv_host()
            b = self.slot * len(m.attn_layers) + m.attn_layers.index(layer)
            return {"key": k[b, :, :length].clone(), "value": v[b, :, :length].clone()}
        mod, st = m.mixers[layer], self.gdn[(self.slot, layer)]
        return {"conv": m.conv_to_host(mod, st.conv), "recurrent": m.rec_to_host(st.recurrent)}

    def load_prefix(self, layer: int, tensors: dict, length: int):
        m = self.model
        if layer in m.attn_layers:
            k, v = self._kv_host()
            b = self.slot * len(m.attn_layers) + m.attn_layers.index(layer)
            k[b, :, :length] = tensors["key"][:, :length].float()
            v[b, :, :length] = tensors["value"][:, :length].float()
            ttnn.deallocate(self.kv.k)
            ttnn.deallocate(self.kv.v)
            self.kv = KvCache(
                k=m.kv_upload(k, self.capacity),
                v=m.kv_upload(v, self.capacity),
                num_layers=self.kv.num_layers,
                capacity=self.capacity,
            )
            return
        old = self.gdn[(self.slot, layer)]
        self.gdn[(self.slot, layer)] = m.mixers[layer].state_from_host(tensors["conv"], tensors["recurrent"])
        ttnn.deallocate(old.conv)
        ttnn.deallocate(old.recurrent)


class TtModel:
    def __init__(self, mesh, hf_dir, layers: list[int], chunk: int, max_seq: int, lm_head: bool = True):
        t0 = time.time()
        self.mesh = mesh
        self.cfg = cfg = Config(hf_dir)
        self.layers = list(layers)
        rows, cols = tuple(mesh.shape)
        assert chunk % (rows * 32) == 0, chunk
        self.chunk, self.local_rows = chunk, chunk // rows
        self.capacity = -(-max_seq // chunk) * chunk
        self.slots = settings.get("KV_SLOTS")
        self.attn_layers = [i for i in self.layers if not cfg.is_linear(i)]
        self.linear_layers = [i for i in self.layers if cfg.is_linear(i)]
        self._scalars: dict[int, ttnn.Tensor] = {}
        ck = Checkpoint(hf_dir)
        dram = ttnn.DRAM_MEMORY_CONFIG

        # embedding: the bf16 table replicated on every chip (ROW_MAJOR, as ttnn.embedding reads it)
        self.embed_w = ttnn.from_torch(
            ck.get(PREFIX + "embed_tokens.weight", torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=dram,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

        self.blocks, self.mixers = {}, {}
        add = TtResidualAdd()
        for i in self.layers:
            w = ck.layer(i, COMMON_NAMES + (LINEAR_NAMES if cfg.is_linear(i) else ATTN_NAMES), torch.float32)
            if cfg.is_linear(i):
                self.mixers[i] = TtGatedDeltaNet(mesh, w, cfg, self.local_rows)
            else:
                self.mixers[i] = TtGatedAttention(mesh, w, cfg, chunk, self.capacity)
            self.blocks[i] = {
                "attn_norm": TtRmsNorm(mesh, w["input_layernorm.weight"], cfg.eps),
                "mlp_norm": TtRmsNorm(mesh, w["post_attention_layernorm.weight"], cfg.eps),
                "mlp": TtMlp(mesh, {k: v for k, v in w.items() if k.startswith("mlp.")}),
                "residual": add,
            }
            del w
        self.final = TtRmsNorm(mesh, ck.get(PREFIX + "norm.weight", torch.float32), cfg.eps)
        self.head = None
        if lm_head:
            self.head = ttnn.from_torch(
                ck.get("lm_head.weight", torch.float32).T.contiguous(),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=dram,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(rows, cols), dims=(None, -1)),
            )
            self.head_cfg = compute_config(settings.get("MLP_FIDELITY"), settings.get("MLP_FP32_ACC"))
        # a new request (start 0) runs its linear layers from this zero state (read only, never freed)
        self.zero_gdn = {i: self.mixers[i].state_from_host(None, None) for i in self.linear_layers}
        self.graph = Reference.block_graph(None, 0)  # the block graph is the same for both block types
        self.load_seconds = time.time() - t0

    # ---- scalars: replicated uint32 device scalars (actual_start / actual_end)
    def scalar(self, value: int) -> ttnn.Tensor:
        if value not in self._scalars:
            self._scalars[value] = ttnn.from_torch(
                torch.tensor([value], dtype=torch.int64),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )
        return self._scalars[value]

    # ---- interface
    def new_state(self, max_seq: int, caches: Qwen38KvCaches | None = None) -> TtState:
        assert max_seq <= self.capacity, f"max_seq {max_seq} above the model's capacity {self.capacity}"
        return TtState(self, max_seq, caches)

    def embed(self, tokens: torch.Tensor, start: int = 0):
        """One chunk of ids in natural order (0xFFFFFFFF past the real tokens) -> the residual stream."""
        rows, cols = tuple(self.mesh.shape)
        S = tokens.shape[-1]
        assert S == self.chunk, f"a chunk of {S} tokens, the model runs {self.chunk}"
        ids = ttnn.from_torch(
            tokens.to(torch.int64).reshape(1, 1, 1, S),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=(rows, cols), dims=(3, None)),
        )
        return self._embed_device(ids)

    def embed_engine_input(self, ids):
        """The engine's H2D chunk (uint32 ROW_MAJOR [sp, 1, chunk / sp], sharded over the mesh rows, replicated over
        the columns; a chunk-aligned start puts chunk offsets [r * 640, (r + 1) * 640) on row r) -> the residual
        stream. Frees ``ids``."""
        return self._embed_device(ttnn.reshape(ids, (1, 1, 1, self.local_rows)))

    def _embed_device(self, ids):
        dram = ttnn.DRAM_MEMORY_CONFIG
        t = ttnn.to_layout(ids, ttnn.TILE_LAYOUT, memory_config=dram)
        ttnn.deallocate(ids)
        c = ttnn.minimum(t, self.cfg.vocab - 1)  # the engine's pad id 0xFFFFFFFF -> a valid row
        ttnn.deallocate(t)
        ids = ttnn.to_layout(c, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram)
        ttnn.deallocate(c)
        h = ttnn.embedding(ids, self.embed_w, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=dram)
        ttnn.deallocate(ids)
        return ttnn.reshape(h, (1, 1, self.local_rows, self.cfg.hidden))

    def from_host(self, h: torch.Tensor):
        return rows_to_device(self.mesh, h)

    def to_host(self, h) -> torch.Tensor:
        return rows_to_host(self.mesh, h)

    def layer(self, i: int, h, start: int, state: TtState, end: int | None = None):
        assert start % self.chunk == 0, f"chunk start {start} (the ring SDPA at fp32 DEST takes whole chunks)"
        end = start + self.chunk if end is None else end
        blk, mixer = self.blocks[i], self.mixers[i]
        ctx = Ctx(i, start, self.chunk, state, {"end": end})

        if i in self.attn_layers:
            ordinal = self.attn_layers.index(i)

            def attention(ctx, x):
                return mixer(x, state.kv, start, slot=state.slot, layer=ordinal, end=end)

        else:

            def attention(ctx, x):
                key = (state.slot, i)
                old = state.gdn[key]
                partial = end < start + self.chunk
                out, new = mixer(
                    x,
                    self.zero_gdn[i] if start == 0 else old,
                    self.scalar(start),
                    valid=end - start if partial else None,
                    actual_end=self.scalar(end) if partial else None,
                )
                state.gdn[key] = new
                ttnn.deallocate(old.conv)
                ttnn.deallocate(old.recurrent)
                return out

        steps = {
            "attn_norm": lambda ctx, x: blk["attn_norm"](x),
            "attention": attention,
            "attn_residual": lambda ctx, a, b: blk["residual"](a, b),
            "mlp_norm": lambda ctx, x: blk["mlp_norm"](x),
            "mlp": lambda ctx, x: blk["mlp"](x),
            "mlp_residual": lambda ctx, a, b: blk["residual"](a, b),
        }
        made = {}
        out = run_block(self.graph, steps.__getitem__, ctx, h, rec=lambda name, t: made.__setitem__(name, t))
        for name, t in made.items():  # the block's intermediates; the caller frees its input
            if name not in ("in", "out"):
                ttnn.deallocate(t)
        return out

    def final_norm(self, h):
        return self.final(h)

    def logits(self, hidden, rows) -> torch.Tensor:
        assert self.head is not None, "built without lm_head"
        lg = ttnn.linear(
            hidden,
            self.head,
            dtype=ttnn.float32,
            compute_kernel_config=self.head_cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, 1, 640, V / cols] per chip
        r, c = tuple(self.mesh.shape)
        shards = ttnn.get_device_tensors(lg)
        full = torch.cat(
            [torch.cat([ttnn.to_torch(shards[a * c + b]) for b in range(c)], dim=-1) for a in range(r)], dim=-2
        )
        ttnn.deallocate(lg)
        full = full.reshape(-1, full.shape[-1])[:, : self.cfg.vocab].float()
        return full[torch.as_tensor(rows).long()]

    def free(self, h):
        ttnn.deallocate(h)

    def sync(self):
        ttnn.synchronize_device(self.mesh)

    # ---- state read-back helpers (harness boundary)
    def kv_upload(self, full: torch.Tensor, capacity: int):
        rows, cols = tuple(self.mesh.shape)
        idx = blockcyclic_positions(rows, self.chunk, capacity)
        dev = full[:, :, idx].contiguous()
        return ttnn.from_torch(
            dev,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=kv_memory_config(self.mesh, self.cfg.head_dim),
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=(rows, cols), dims=(2, 1)),
        )

    def conv_to_host(self, mod: TtGatedDeltaNet, t) -> torch.Tensor:
        """The conv history [1, 3, conv_tp] per column back in the reference's [conv_dim, 3] channel order."""
        cols = tuple(self.mesh.shape)[1]
        shards = ttnn.get_device_tensors(t)
        per = torch.cat([ttnn.to_torch(shards[c]).float().reshape(3, -1) for c in range(cols)], dim=-1)
        q, k, v = [], [], []
        for c in range(cols):
            blk = per[:, c * mod.conv_tp : (c + 1) * mod.conv_tp]
            q.append(blk[:, : mod.kd])
            k.append(blk[:, mod.kd : 2 * mod.kd])
            v.append(blk[:, 2 * mod.kd :])
        return torch.cat(q + k + v, dim=-1).T.contiguous()

    def rec_to_host(self, t) -> torch.Tensor:
        cols = tuple(self.mesh.shape)[1]
        shards = ttnn.get_device_tensors(t)
        return torch.cat([ttnn.to_torch(shards[c]).float() for c in range(cols)], dim=0)
