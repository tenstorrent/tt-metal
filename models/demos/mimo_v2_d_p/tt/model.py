# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2.6-Flash text backbone, disaggregated chunked prefill (SP rows x TP cols x EP all chips).

Per chunk of ``chunk_size`` tokens at global offset ``kv_actual``:
  tokens (block-cyclic over SP rows) -> embedding -> layers[first : first+n] -> hidden [1,1,S_local,H]
KV lives in two block-cyclic caches (GA: 4 KV heads, K 192 / V 128; SWA: 8 KV heads, K 192 / V 192-padded)
that decode reads by address (``tt/runners/kv_chunk_table.py``). A rank may own any contiguous layer range
(pipeline parallel): non-first ranks take the hidden state instead of tokens.

Deployments: QuietBox 2x2 (SP2 x TP2, 64 experts/chip); BH Galaxy 8x4 (SP8 x TP4, 8 experts/chip, the
whole 48-layer model at ~10 GB/chip of bf8 experts). Constraints: chunk_size % (32*SP) == 0,
chunk_size // SP >= sliding window (one-hop halo), max_seq_len % chunk_size == 0.
"""

import os

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.utils import rotated_chip_positions
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.mimo_v2_d_p.reference.config import GA, SWA, MiMoTextConfig
from models.demos.mimo_v2_d_p.tt.attention.attention import cache_v_dim, kv_heads_for_col
from models.demos.mimo_v2_d_p.tt.attention.kv_cache import allocate_kv_cache
from models.demos.mimo_v2_d_p.tt.ccl import CCLManager, default_num_links
from models.demos.mimo_v2_d_p.tt.decoder import TtDecoderLayer
from models.demos.mimo_v2_d_p.tt.ffn import TtRMSNorm
from models.demos.mimo_v2_d_p.tt.rope import build_indexed_rope, build_transformation_mat

PAD_TOKEN_ID = 0xFFFFFFFF  # tt-d-gen chunk-tail pad (engine/include/engine/types.hpp PAD_ID)


def clamp_pad_tokens(tok: ttnn.Tensor, vocab: int) -> ttnn.Tensor:
    """Map out-of-vocab ids (the 0xFFFFFFFF pad) to 0 on device. Pad rows sit after the last real token,
    so causal attention never lets them influence real positions; their KV is masked by actual_end."""
    t = ttnn.to_layout(tok, ttnn.TILE_LAYOUT)
    out = ttnn.where(ttnn.lt(t, vocab), t, 0)
    return ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)


def block_cyclic_index(kv_actual: int, sp: int, chunk_local: int) -> torch.Tensor:
    """Global positions of a chunk in device order (SP row major, then local row)."""
    pos = rotated_chip_positions(kv_actual, sp, chunk_local)
    return torch.tensor([pos[c][r] for c in range(sp) for r in range(chunk_local)], dtype=torch.long)


class TtMiMoModel:
    def __init__(self, mesh_device, cfg: MiMoTextConfig, layer_state, *, fabric_config, max_seq_len: int, chunk_size: int,
                 layers: list[int] | None = None, global_state=None, num_users: int = 1, expert_dtype=None,
                 allocate_kv: bool = True, embed: bool = True):
        """``layer_state(i) -> {HF name: tensor}``; ``global_state() -> {embed_tokens.weight, norm.weight}``."""
        self.mesh_device = mesh_device
        self.cfg = cfg
        self.sp, self.tp = tuple(mesh_device.shape)
        self.chunk_size = chunk_size
        self.chunk_local = chunk_size // self.sp
        self.max_seq_len = max_seq_len
        assert chunk_size % (32 * self.sp) == 0 and max_seq_len % chunk_size == 0
        assert self.chunk_local >= cfg.sliding_window, f"chunk/SP ({self.chunk_local}) must be >= sliding window ({cfg.sliding_window})"
        self.layer_ids = list(layers) if layers is not None else list(range(cfg.num_hidden_layers))
        self.sp_topo, self.tp_topo = per_axis_topology(fabric_config)
        self.num_links = default_num_links()
        self.ccl = CCLManager(mesh_device, num_links=self.num_links, topology=self.sp_topo)
        self.vocab = cfg.vocab_size

        self.embed_w = None
        if embed:
            g = global_state()
            self.embed_w = ttnn.from_torch(
                g["embed_tokens.weight"].bfloat16(), device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device), memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        self.rope = {t: build_indexed_rope(mesh_device, cfg.attn_spec(t), max_seq_len=max_seq_len, chunk_size=chunk_size) for t in (GA, SWA)}
        self.trans_mat = build_transformation_mat(mesh_device)

        # One cache per attention type; each layer knows its slot within its type's cache.
        self.cache_layer, counts = {}, {GA: 0, SWA: 0}
        for i in self.layer_ids:
            t = cfg.layer_type(i)
            self.cache_layer[i] = counts[t]
            counts[t] += 1
        self.layer_counts = counts
        self.num_users = num_users
        self.kv = self.allocate_kv_caches(num_users) if allocate_kv else {}

        self.layers = []
        for i in self.layer_ids:
            logger.info(f"loading layer {i} ({cfg.layer_type(i)}, {'moe' if cfg.is_moe(i) else 'dense'})")
            self.layers.append(TtDecoderLayer(mesh_device, cfg, i, layer_state(i), ccl=self.ccl, sp_topology=self.sp_topo,
                                              seq_len_per_chip=self.chunk_local, expert_dtype=expert_dtype, num_links=self.num_links))

    def kv_geometry(self, t):
        spec = self.cfg.attn_spec(t)
        return dict(n_kv_local=len(kv_heads_for_col(0, self.tp, spec.n_q, spec.n_kv)), k_dim=spec.head_dim, v_dim=cache_v_dim(spec))

    def allocate_kv_caches(self, num_users: int | None = None) -> dict:
        """{GA|SWA: MiMoKVCache} sized for this model's layers (engine-owned in serving)."""
        return {
            t: allocate_kv_cache(self.mesh_device, num_layers=n, max_seq_len=self.max_seq_len, num_users=num_users or self.num_users,
                                 **self.kv_geometry(t))
            for t, n in self.layer_counts.items()
            if n
        }

    # -------------------------------------------------------------- chunk
    def tokens_to_device(self, ids_bc: torch.Tensor):
        """ids_bc [chunk] in block-cyclic device order -> uint32 [1,1,S_local] per chip (the H2D layout)."""
        return ttnn.from_torch(ids_bc.view(self.sp, 1, self.chunk_local).to(torch.int64).to(torch.int32), device=self.mesh_device,
                               layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                               mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, mesh_shape=(self.sp, self.tp), dims=(0, None)))

    def embed_device(self, tok):
        tok = clamp_pad_tokens(tok, self.vocab)
        tok = ttnn.reshape(tok, (1, self.chunk_local))
        e = ttnn.embedding(tok, self.embed_w, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        return ttnn.reshape(e, (1, 1, self.chunk_local, self.cfg.hidden_size))

    def forward_device(self, x, kv_actual: int, *, user: int = 0, valid_end: int | None = None, on_layer_complete=None, capture=None):
        """Run this rank's layers on hidden ``x``; ``on_layer_complete(global_layer_idx)`` fires once per layer, in order."""
        for layer in self.layers:
            t = layer.kind
            y = layer(x, self.rope[t], self.trans_mat, self.kv[t], cache_layer=self.cache_layer[layer.layer_idx], kv_actual=kv_actual,
                      user=user, valid_end=valid_end)
            x.deallocate(True)
            x = y
            if capture is not None:
                capture(layer.layer_idx, x, kv_actual)
            if on_layer_complete is not None:
                on_layer_complete(layer.layer_idx)
        return x

    def prefill_chunk(self, ids_chunk: torch.Tensor, kv_actual: int, user: int = 0, capture=None, valid_end=None):
        """ids_chunk [chunk_size] (natural order) at global offset kv_actual -> hidden [1,1,S_local,H] (device, block-cyclic)."""
        idx = block_cyclic_index(kv_actual, self.sp, self.chunk_local) - kv_actual
        x = self.embed_device(self.tokens_to_device(ids_chunk[idx]))
        return self.forward_device(x, kv_actual, user=user, capture=capture, valid_end=valid_end)

    def gather_hidden(self, x, kv_actual):
        """Device hidden [1,1,S_local,H] (block-cyclic) -> host [chunk_size, H] in natural order."""
        dts = ttnn.get_device_tensors(x)
        bc = torch.cat([ttnn.to_torch(dts[s * self.tp]).float()[0, 0] for s in range(self.sp)], 0)
        out = torch.empty_like(bc)
        out[block_cyclic_index(kv_actual, self.sp, self.chunk_local) - kv_actual] = bc
        return out
