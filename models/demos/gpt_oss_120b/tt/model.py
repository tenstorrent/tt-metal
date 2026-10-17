# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The all-device gpt-oss-120b prefill model on the [8, 4] mesh (plan.md "Model level"): the hidden state stays on the
device from the embedding to the final norm, and every block runs through ``run_block`` over the reference's block
graph with each step's validated device module (attn_norm, attention, attn_residual, ffn_norm, router, experts,
mlp_residual).

Residual stream per chip (row r, column c): [1, 1, chunk / 8, 2880] bf16 TILE DRAM, the chunk's positions of SP row r
(block-cyclic: row r holds the 640-row block r of the chunk), replicated over the 4 TP columns.

State (``TtKvState``): the served cache (serving_contract.md "KV cache"). One K and one V tensor per chip, [slots x
layers, 2, max_seq / 8, 64] bfloat8_b, DRAM ND-sharded [1, 1, 32, 64] ROUND_ROBIN_1D (tt/kv_cache.py), batch slot =
slot * num_layers + layer position, as gpt_oss_d_p allocate_kv_cache packs it. Full-attention layers also use a K
residual cache of the same layout (ATTN_FULL_K_LO, tt/attention.py); it is prefill-only state, not served.
"""

import torch

import ttnn

from ..reference._bringup_interface import Ctx, run_block
from ..reference.gpt_oss_ref import GptOssReference
from .attention import HEAD_DIM, meta_permute_cols
from .ccl import SP_AXIS
from .kv_cache import bfp8_round, block_cyclic_reorder, cache_memory_config
from .layout import rows_to_device, rows_to_host
from .residual import TtResidualAdd
from .settings import settings

PAD_ID = 0xFFFFFFFF


def block_graph(layer: int):
    """The reference's block graph (GptOssReference.block_graph reads no weights)."""
    return GptOssReference.block_graph(None, layer)


def meta_unpermute_cols(t: torch.Tensor) -> torch.Tensor:
    """Inverse of meta_permute_cols: the head dim from meta (interleaved-pair) back to half-split order."""
    shape = t.shape
    return t.reshape(*shape[:-1], HEAD_DIM // 2, 2).transpose(-1, -2).reshape(shape)


def rotated_positions(start: int, sp: int, chunk_local: int) -> list[list[int]]:
    """positions[r][i]: the global position chip row r's i-th row carries in a chunk starting at ``start`` (any
    tile-aligned start), as update_padded_kv_cache and rotary_embedding_indexed derive it (deepseek_v3_d_p
    mla/utils.rotated_chip_positions). A chunk-aligned start gives row r the block [start + r * chunk_local, ...)."""
    chunk = sp * chunk_local
    split_chunk, split_chip, split_off = start // chunk, (start // chunk_local) % sp, start % chunk_local
    out = []
    for r in range(sp):
        if r < split_chip:
            first = (split_chunk + 1) * chunk_local
        elif r == split_chip:
            first = split_chunk * chunk_local + split_off
        else:
            first = split_chunk * chunk_local
        out.append(
            [
                (lr // chunk_local) * chunk + r * chunk_local + lr % chunk_local
                for lr in range(first, first + chunk_local)
            ]
        )
    return out


class TtKvState:
    """The device model's state: the served K / V caches of every (slot, layer) and the K residual caches of the
    full-attention layers. ``slot`` is the user slot the model's layer calls read and write."""

    def __init__(self, mesh, layers: list[int], k_lo_layers: set, n_kv: int, max_seq: int, chunk: int, slots: int):
        self.mesh, self.layers, self.n_kv, self.max_seq, self.chunk = mesh, list(layers), n_kv, max_seq, chunk
        self.sp, self.tp = tuple(mesh.shape)
        assert max_seq % chunk == 0, f"max_seq {max_seq} must be a multiple of the chunk {chunk}"
        self.slots, self.slot = slots, 0
        self.k_lo_layers = set(k_lo_layers)
        n_batch = slots * len(self.layers)
        self.k = self._zeros(n_batch)
        self.v = self._zeros(n_batch)
        self.k_lo = self._zeros(n_batch) if self.k_lo_layers else None

    def _zeros(self, n_batch: int) -> ttnn.Tensor:
        return ttnn.from_torch(
            torch.zeros(n_batch, self.n_kv, self.max_seq, HEAD_DIM),
            device=self.mesh,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=cache_memory_config(self.mesh, HEAD_DIM),
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=(self.sp, self.tp), dims=(2, 1)),
        )

    def batch(self, layer: int, slot: int | None = None) -> int:
        return (self.slot if slot is None else slot) * len(self.layers) + self.layers.index(layer)

    # ---------------------------------------------------------------- serving contract hooks
    def formats(self) -> dict:
        return {"key": ttnn.bfloat8_b, "value": ttnn.bfloat8_b}

    def cache_tensors(self, layer: int) -> dict:
        assert layer in self.layers, f"layer {layer} is not in this model ({self.layers})"
        return {"key": self.k, "value": self.v}

    # ---------------------------------------------------------------- host side (load time, read-back)
    def load_prefix(self, layer: int, tensors: dict, length: int) -> None:
        """The golden prefix (key / value [length, n_kv, 64], K half-split) into this slot, chunk by chunk through
        update_padded_kv_cache, so it lands exactly where the model's own writes put it. Host work, outside any
        forward."""
        b = self.batch(layer)
        mapper = ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=(self.sp, self.tp), dims=(2, 1))
        for c0 in range(0, length, self.chunk):
            n = min(length, c0 + self.chunk) - c0
            for name, cache in (("key", self.k), ("value", self.v)):
                src = tensors[name][c0 : c0 + n].float()
                if name == "key":
                    src = meta_permute_cols(src)
                slab = torch.zeros(1, self.n_kv, self.chunk, HEAD_DIM)
                slab[0, :, :n] = src.permute(1, 0, 2)
                parts = [(cache, slab)]
                if name == "key" and layer in self.k_lo_layers:
                    parts.append((self.k_lo, slab - bfp8_round(slab)))
                for dst, host in parts:
                    t = ttnn.from_torch(
                        host,
                        device=self.mesh,
                        dtype=ttnn.bfloat8_b,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=mapper,
                    )
                    ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                        dst, t, slot_idx=b, layer_idx=0, num_layers=1, kv_actual_global=c0, cluster_axis=SP_AXIS
                    )
                    t.deallocate(True)

    def _read(self, cache: ttnn.Tensor, b: int) -> torch.Tensor:
        """One batch slot of a cache -> host [n_kv, max_seq, 64] in natural position order."""
        dts = ttnn.get_device_tensors(cache)
        rows = []
        for r in range(self.sp):
            heads = [ttnn.to_torch(dts[r * self.tp + c]).float()[b] for c in range(self.tp)]
            rows.append(torch.cat(heads, dim=0))  # [n_kv, max_seq / sp, 64]
        reordered = torch.cat(rows, dim=1)  # block-cyclic order
        idx = block_cyclic_reorder(torch.arange(self.max_seq).reshape(1, 1, -1, 1), self.chunk // self.sp, self.sp)
        out = torch.empty_like(reordered)
        out[:, idx.flatten()] = reordered
        return out

    def to_torch(self, layer: int, length: int) -> dict:
        b = self.batch(layer)
        k = meta_unpermute_cols(self._read(self.k, b)[:, :length])
        v = self._read(self.v, b)[:, :length]
        return {"key": k.permute(1, 0, 2).contiguous(), "value": v.permute(1, 0, 2).contiguous()}

    def free(self) -> None:
        for t in (self.k, self.v, self.k_lo):
            if t is not None:
                t.deallocate(True)


class TtModel:
    """``blocks[i]``: layer i's validated device modules {"attn_norm": TtRmsNorm, "attention": TtAttention,
    "ffn_norm": TtRmsNorm, "router": TtRouter, "experts": TtExperts}; ``final_norm`` a TtRmsNorm; ``lm_head_w`` the
    [vocab, hidden] head or None."""

    def __init__(self, mesh, cfg: dict, layers, embed_w, final_norm, lm_head_w, blocks: dict, chunk: int):
        self.mesh, self.cfg, self.layers, self.chunk = mesh, cfg, list(layers), int(chunk)
        self.sp, self.tp = tuple(mesh.shape)
        self.chunk_local = self.chunk // self.sp
        self.vocab = int(cfg["vocab_size"])
        self.blocks = blocks
        self.attention = {i: blocks[i]["attention"] for i in self.layers}
        self.steps = {i: self._steps(i) for i in self.layers}
        self.graphs = {i: block_graph(i) for i in self.layers}
        self.embed_w = ttnn.from_torch(
            embed_w.to(torch.bfloat16)[None, None],
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        self.norm = final_norm
        self.lm_head = None
        if lm_head_w is not None:  # vocab over the TP columns, [out, in] bfp8 + transpose_b (blocks along K)
            v_pad = -(-self.vocab // (self.tp * ttnn.TILE_SIZE)) * self.tp * ttnn.TILE_SIZE
            w = torch.zeros(v_pad, lm_head_w.shape[1])
            w[: self.vocab] = lm_head_w.float()
            self.lm_head = ttnn.from_torch(
                w[None, None],
                device=mesh,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(self.sp, self.tp), dims=(None, 2)),
            )
            self.lm_config = ttnn.init_device_compute_kernel_config(
                mesh.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            )
        self.load_seconds = 0.0  # the hooks set the whole load (weights read and uploaded)

    # ---------------------------------------------------------------- state
    def new_state(self, max_seq: int, slots: int | None = None) -> TtKvState:
        k_lo = {i for i in self.layers if self.attention[i].k_lo}
        slots = settings.get("KV_SLOTS") if slots is None else slots
        # at least two chunks of rows: ring SDPA's sliding path is chunked-only (Q shorter than the per-chip K shard),
        # so a one-chunk cache (Q 640 == K 640 rows per chip) is rejected; the rows past max_seq stay zero, unread
        rows = max(max_seq, 2 * self.chunk)
        return TtKvState(self.mesh, self.layers, k_lo, int(self.cfg["num_key_value_heads"]), rows, self.chunk, slots)

    # ---------------------------------------------------------------- forward
    def embed(self, tokens: torch.Tensor, start: int = 0) -> ttnn.Tensor:
        """One chunk of token ids in natural order (PAD_ID past the real tokens), its first position ``start`` ->
        the residual stream. Row r gets the positions update_padded_kv_cache and RoPE give it for that start."""
        assert tokens.shape[-1] == self.chunk, f"one chunk of {self.chunk} ids, got {tokens.shape[-1]}"
        assert start % ttnn.TILE_SIZE == 0, f"start {start} must be tile-aligned"
        ids = tokens.reshape(-1).long()
        if start % self.chunk:
            idx = torch.tensor([p - start for row in rotated_positions(start, self.sp, self.chunk_local) for p in row])
            ids = ids[idx]
        tt_ids = ttnn.from_torch(
            ids.reshape(self.sp, 1, self.chunk_local),
            device=self.mesh,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=(self.sp, self.tp), dims=(0, None)),
        )
        safe = ttnn.minimum(tt_ids, self.vocab - 1)  # PAD_ID rows embed a real row; their KV is zeroed (end=)
        tt_ids.deallocate(True)
        h = ttnn.embedding(
            safe, self.embed_w, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        safe.deallocate(True)
        return ttnn.reshape(h, (1, 1, self.chunk_local, h.shape[-1]))

    def layer(self, i: int, h: ttnn.Tensor, start: int, state: TtKvState, end: int | None = None) -> ttnn.Tensor:
        """Block i on the chunk at ``start``; rows at or past ``end`` are pad (their KV rows are zeroed)."""
        ctx = Ctx(i, start, self.chunk, state, {"end": end})
        made = []

        def step(fn):
            def run(ctx, *x):
                y = fn(ctx, *x)
                made.append(y)
                return y

            return run

        overrides = {name: step(fn) for name, fn in self.steps[i].items()}
        out = run_block(self.graphs[i], lambda n: overrides[n], ctx, h, overrides=overrides)
        for t in made:
            if t is not out:
                t.deallocate(True)
        return out

    def _steps(self, i: int) -> dict:
        """Layer i's step functions fn(ctx, *inputs), one per step of the block graph."""
        m, add = self.blocks[i], TtResidualAdd()

        def router(ctx, x):
            dense = m["router"](x)  # fp32 ROW_MAJOR (ttnn.scatter); the experts' topk reads fp32 TILE
            out = ttnn.to_layout(dense, ttnn.TILE_LAYOUT)
            dense.deallocate(True)
            return out

        return {
            "attn_norm": lambda ctx, x: m["attn_norm"](x),
            "attention": self._attention_step(i),
            "attn_residual": lambda ctx, a, b: add(a, b),
            "ffn_norm": lambda ctx, x: m["ffn_norm"](x),
            "router": router,
            "experts": lambda ctx, x, r: m["experts"](x, r),
            "mlp_residual": lambda ctx, a, b: add(a, b),
        }

    def _attention_step(self, i: int):
        """The attention step of layer i: the served cache slot of (state.slot, i), the K residual cache on full
        layers; rows past ctx.extra["end"] go to every cache it writes as zeros (TtAttention end=)."""
        attn = self.attention[i]

        def run(ctx, x):
            st, b = ctx.state, ctx.state.batch(i)
            extra = {"cache_k_lo": st.k_lo} if attn.k_lo else {}
            end = ctx.extra.get("end")
            if end is not None and end >= ctx.start + self.chunk:
                end = None
            return attn(x, ctx.start, st.k, st.v, batch_idx=b, end=end, **extra)

        return run

    def final_norm(self, h: ttnn.Tensor) -> ttnn.Tensor:
        return self.norm(h)

    # ---------------------------------------------------------------- host boundary (harness only)
    def from_host(self, h: torch.Tensor) -> ttnn.Tensor:
        return rows_to_device(self.mesh, h)

    def to_host(self, h: ttnn.Tensor) -> torch.Tensor:
        return rows_to_host(self.mesh, h)

    def logits(self, hidden: ttnn.Tensor, rows) -> torch.Tensor:
        """Logits of the chosen rows of a final hidden [len(rows), vocab] fp32 on the host (harness only)."""
        assert self.lm_head is not None, "built with lm_head=False"
        x = self.to_host(hidden)[torch.as_tensor(rows)]
        n = x.shape[0]
        n_pad = -(-n // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        xp = torch.zeros(n_pad, x.shape[1])
        xp[:n] = x
        tt_x = ttnn.from_torch(
            xp[None, None],
            device=self.mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        y = ttnn.linear(
            tt_x,
            self.lm_head,
            transpose_b=True,
            dtype=ttnn.float32,
            compute_kernel_config=self.lm_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_x.deallocate(True)
        dts = ttnn.get_device_tensors(y)
        out = torch.cat([ttnn.to_torch(dts[c]).float() for c in range(self.tp)], dim=-1)
        y.deallocate(True)
        return out.reshape(n_pad, -1)[:n, : self.vocab]

    def free(self, h: ttnn.Tensor) -> None:
        h.deallocate(True)

    def sync(self) -> None:
        ttnn.synchronize_device(self.mesh)
