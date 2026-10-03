# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1-Flash one-token decode on a 4x8 Blackhole Galaxy (batch 16, tp_heads), layer by layer.

Each layer is loaded from the checkpoint, seeded with its prefilled cache state (from ``reference/ref_chain.py``),
run for the decode token and released; only the kv-source layers' attention objects stay alive because later layers
read their compressed cache. Embedding, Engram (layers 1 and 14) and the LM head run on the host.
"""

import os
import time

import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.attention import DSV41Attention, DSV41CompressedAttention
from models.demos.blackhole.deepseek_v41_flash.tt.layer import DSV41Layer
from models.demos.blackhole.deepseek_v41_flash.tt.loader import load_layer
from models.demos.blackhole.deepseek_v41_flash.tt.paged_attention import (
    DSV41PagedAttention,
    DSV41PagedCompressedAttention,
    DSV41PagedStepState,
    paged_enabled,
)
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager


class DSV41DecodeChain:
    def __init__(self, mesh_device, users_per_row=4, max_comp=128, log=print):
        self.md = mesh_device
        self.rows, self.cols = tuple(mesh_device.shape)
        self.T = users_per_row
        self.B = self.rows * users_per_row
        self.max_comp = max_comp
        self.log = log
        self.mesh_config = mesh_4x8()
        self.ccl = CCLManager(mesh_device, num_links=2, topology=ttnn.Topology.Ring)
        self.sources = {}  # kv-source layer id -> its attention module
        self.moe_buffers = None  # persistent MoE scratch buffers, built by the first layer and shared by all later ones
        # ---- paged KV (DSV41_PAGED=1): shared row-major pool + per-user page tables, see tt/paged_ops.py
        self.paged = paged_enabled()
        self.kvpool = None
        self.ring_slots = 0
        self.index_owner = {}  # kv-source layer -> its DSV41DecodeIndexer (key slab owner)
        if self.paged:
            from models.demos.blackhole.deepseek_v41_flash.tt.paged_ops import PAGE_TOKENS, PagedKVPool

            self.max_ctx = int(os.environ.get("DSV41_PAGED_CTX", "4096"))
            pages_per_user = -(-self.max_ctx // PAGE_TOKENS)
            self.kvpool = PagedKVPool(
                mesh_device,
                users_per_row,
                num_pages=int(os.environ.get("DSV41_PAGED_PAGES", users_per_row * pages_per_user)),
                n_ring_layers=int(os.environ.get("DSV41_PAGED_RING_LAYERS", "40")),
                max_ctx=self.max_ctx,
                dtype=ttnn.fp8_e4m3 if os.environ.get("DSV41_PAGED_KV", "bf16") == "fp8" else ttnn.bfloat16,
            )
            self.kvpool.stage_begin()
            self.use_indexer = os.environ.get("DSV41_PAGED_INDEXER", "0") == "1"

    def build_layer(self, L, chain, w=None):
        """chain: the saved per-layer dict from ref_chain (state, gate_cutoff). ``w``: preloaded ``load_layer(L)``."""
        import os as _os
        import time as _time

        _prof = _os.environ.get("DSV41_BUILD_PROFILE") == "1"
        t_start = _time.perf_counter()
        w = (
            w
            if w is not None
            else load_layer(
                L, max_seq_len=self.max_ctx + 64 if self.paged else 256, with_indexer=self.paged and self.use_indexer
            )
        )
        meta, state = w["meta"], chain["state"]
        S = chain["S"]
        kw = dict(users_per_row=self.T)
        if self.paged:
            attn = self._build_paged_attention(L, meta, w, state, S)
        elif meta["ratio"] == 0:
            attn = DSV41Attention(self.md, self.mesh_config, self.ccl, w["attn"], w["freqs_cis"], max_seq=256, **kw)
            attn.load_window(state["window"][:, :S])
        elif meta["is_kv_source"]:
            attn = DSV41CompressedAttention(
                self.md,
                self.mesh_config,
                self.ccl,
                w["attn"],
                w["freqs_cis"],
                meta["ratio"],
                w["compressor"],
                max_comp=self.max_comp,
                **kw,
            )
            attn.load_state(
                state["window"], state["comp"], state.get("kv_state"), state.get("score_state"), start_pos=S
            )
            self.sources[L] = attn
        else:
            attn = DSV41CompressedAttention(
                self.md,
                self.mesh_config,
                self.ccl,
                w["attn"],
                w["freqs_cis"],
                meta["ratio"],
                None,
                max_comp=self.max_comp,
                source=self.sources[meta["kv_source"]],
                **kw,
            )
            attn.load_state(state["window"], None, None, None)
        if _prof:
            ttnn.synchronize_device(self.md)
            print(f"BUILDT layer {L}: host load + attention build {_time.perf_counter() - t_start:6.2f} s", flush=True)
        layer = DSV41Layer(
            self.md,
            self.mesh_config,
            self.ccl,
            attn,
            w["norms"],
            w["mhc"],
            w["moe"],
            gate_bias_shift=chain["gate_cutoff"],
            users_per_row=self.T,
            moe_buffers=self.moe_buffers,
        )
        if self.moe_buffers is None:
            self.moe_buffers = layer.moe.decode.buffers
        return layer, attn

    def _build_paged_attention(self, L, meta, w, state, S):
        from models.demos.blackhole.deepseek_v41_flash.tt.indexer import DSV41DecodeIndexer, default_backend

        slot, self.ring_slots = self.ring_slots, self.ring_slots + 1
        pool = self.kvpool
        if not pool.allocs[0].pages:  # first layer: admit every user (prefilled S tokens + the decode lookahead)
            for b in range(self.B):
                pool.admit(b, S + 1, reserve_tokens=int(os.environ.get("DSV41_PAGED_LOOKAHEAD", "128")))
            pool.sync_page_table()
        kw = dict(users_per_row=self.T)
        if meta["ratio"] == 0:
            attn = DSV41PagedAttention(self.md, self.mesh_config, self.ccl, w["attn"], w["freqs_cis"], pool, slot, **kw)
            attn.load_ring(state["window"])
            return attn
        idx = None
        if self.use_indexer and "indexer" in w:
            iw = w["indexer"]
            n_alloc = -(-(self.max_ctx // meta["ratio"] + 32) // 32) * 32
            idx = DSV41DecodeIndexer(
                self.md,
                iw,
                w["freqs_cis"],
                users_per_row=self.T,
                n_alloc=n_alloc,
                ratio=meta["ratio"],
                key_dtype=ttnn.bfloat8_b,
                fp4_q=True,
                backend=default_backend(self.max_ctx // meta["ratio"]),
            )
            if meta["is_kv_source"]:
                idx.set_key_weights(iw["wk"], iw["k_norm"])
                idx.load_keys(state["index_k"] if "index_k" in state else torch.zeros(self.B, 0, 128))
                self.index_owner[L] = idx
            else:
                idx.k_cache = self.index_owner[meta["kv_source"]].k_cache  # layers 24..36 score against layer 20's keys
        if meta["is_kv_source"]:
            attn = DSV41PagedCompressedAttention(
                self.md,
                self.mesh_config,
                self.ccl,
                w["attn"],
                w["freqs_cis"],
                meta["ratio"],
                w["compressor"],
                pool,
                slot,
                L,
                indexer=idx,
                **kw,
            )
            attn.load_state(
                state["window"], state["comp"], state.get("kv_state"), state.get("score_state"), start_pos=S
            )
            self.sources[L] = attn
        else:
            attn = DSV41PagedCompressedAttention(
                self.md,
                self.mesh_config,
                self.ccl,
                w["attn"],
                w["freqs_cis"],
                meta["ratio"],
                None,
                pool,
                slot,
                meta["kv_source"],
                source=self.sources[meta["kv_source"]],
                indexer=idx,
                **kw,
            )
            attn.load_state(state["window"], None, None, None)
        return attn

    def finalize(self):
        """Paged mode: upload the staged pool (all layers' seeded rings / latents) once every layer is built."""
        if self.paged and getattr(self.kvpool, "_stage", None) is not None:
            self.kvpool.stage_commit()

    def step_state(self, attn):
        """Per-kind step-input builder for the decoder (``ss.build(pos)``)."""
        if self.paged:
            return DSV41PagedStepState(
                attn,
                max_pos=self.max_ctx + 64,
                with_indexer=self.use_indexer,
                per_user_valid=os.environ.get("DSV41_PAGED_RAGGED", "0") == "1",
            )
        from models.demos.blackhole.deepseek_v41_flash.tt.step_state import DSV41StepState

        return DSV41StepState(attn)

    def to_dev(self, t, dims_last):
        shard = ttnn.ShardTensor2dMesh(self.md, dims=(0, None), mesh_shape=(self.rows, self.cols))
        shape = (self.B, 1, 4, dims_last // 4) if dims_last > 4 else (self.B, 1, 1, 4)
        return ttnn.from_torch(
            t.float().reshape(*shape),
            device=self.md,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=shard,
        )

    def to_host(self, t, d):
        devs = ttnn.get_device_tensors(t)
        return torch.cat([ttnn.to_torch(devs[r * self.cols]).reshape(-1, d) for r in range(self.rows)]).float()[
            : self.B
        ]

    def run_layer(self, L, chain, x, pre, position, engram=None, engram_hashes=None, w=None):
        """x: [B,1,4,D] torch (bf16/fp32), pre: [B,1,4] -> (x_out, pre_out) torch, after the layer ran on device."""
        t0 = time.time()
        if engram is not None and L in engram.mods:
            x = engram.apply(L, x, engram_hashes)
        layer, attn = self.build_layer(L, chain, w)
        t1 = time.time()
        st = attn.step_inputs(torch.full((self.B,), position))
        tx, tp = self.to_dev(x.reshape(self.B, -1), 4 * 5120), self.to_dev(pre.reshape(self.B, -1), 4)
        forced = None
        if chain.get("routing_idx") is not None and __import__("os").environ.get("DSV41_ORACLE_ROUTING") == "1":
            rs = ttnn.ShardTensor2dMesh(self.md, dims=(0, None), mesh_shape=(self.rows, self.cols))
            B = self.B
            forced = (
                ttnn.from_torch(
                    chain["routing_wt"].reshape(B, 1, 1, -1).to(torch.bfloat16),
                    device=self.md,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=rs,
                ),
                ttnn.from_torch(
                    chain["routing_idx"].reshape(B, 1, 1, -1).to(torch.int32),
                    device=self.md,
                    dtype=ttnn.uint16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=rs,
                ),
            )
        out, nxt = layer.forward(tx, tp, st, forced_routing=forced)
        ttnn.synchronize_device(self.md)
        x_out = self.to_host(out, 4 * 5120).reshape(self.B, 1, 4, 5120)
        pre_out = self.to_host(nxt, 4).reshape(self.B, 1, 4)
        self.log(f"layer {L:2d}: build {t1 - t0:5.1f}s  run+io {time.time() - t1:5.2f}s")
        return x_out, pre_out
