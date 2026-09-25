# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Gemma-4 26B-A4B text model, disaggregated chunked prefill (SP rows x TP cols x EP all chips).

Per chunk of ``chunk_size`` tokens at global offset ``kv_actual``:
  tokens (block-cyclic over SP rows) -> embedding * sqrt(H) -> 30 x decoder layer -> hidden [1,1,S_local,H]
KV lives in two block-cyclic caches (sliding: 8x256 heads, full: 2x512) that a decode system reads by
address (see ``KvChunk`` migration tables, TODO). After the last chunk, the final norm + tied LM head
(+ softcap) run on the last valid token only.

Constraints: chunk_size % (32*SP) == 0, chunk_size // SP >= sliding_window when SP > 1 (one-hop halo),
max_seq_len % chunk_size == 0.
"""

import math
import os

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.utils import rotated_chip_positions
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.gemma4_26b_d_p.reference.config import FULL, SLIDING, Gemma4TextConfig
from models.demos.gemma4_26b_d_p.reference.weights import CheckpointReader
from models.demos.gemma4_26b_d_p.tt.attention.attention import kv_heads_for_col
from models.demos.gemma4_26b_d_p.tt.attention.kv_cache import allocate_kv_cache
from models.demos.gemma4_26b_d_p.tt.ccl import CCLManager
from models.demos.gemma4_26b_d_p.tt.decoder import TtDecoderLayer
from models.demos.gemma4_26b_d_p.tt.ffn import TtRMSNorm
from models.demos.gemma4_26b_d_p.tt.rope import build_indexed_rope, build_transformation_mat


PAD_TOKEN_ID = 0xFFFFFFFF  # tt-d-gen chunk-tail pad (engine/include/engine/types.hpp PAD_ID)


def clamp_pad_tokens(tok: ttnn.Tensor, vocab: int) -> ttnn.Tensor:
    """Map out-of-vocab ids (the 0xFFFFFFFF pad) to 0 on device. Pad rows sit after the last real token,
    so causal attention never lets them influence real positions; their KV is masked by actual_end."""
    t = ttnn.to_layout(tok, ttnn.TILE_LAYOUT)
    valid = ttnn.lt(t, vocab)
    out = ttnn.where(valid, t, 0)
    return ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)


def block_cyclic_index(kv_actual: int, sp: int, chunk_local: int) -> torch.Tensor:
    """Global positions of a chunk in device order (SP row major, then local row)."""
    pos = rotated_chip_positions(kv_actual, sp, chunk_local)
    return torch.tensor([pos[c][r] for c in range(sp) for r in range(chunk_local)], dtype=torch.long)


class TtGemma4Model:
    def __init__(self, mesh_device, cfg: Gemma4TextConfig, reader: CheckpointReader, *, fabric_config, max_seq_len: int, chunk_size: int,
                 layers: list[int] | None = None, expert_dtype=ttnn.bfloat8_b, build_lm_head: bool = True, num_users: int = 1,
                 allocate_kv: bool = True):
        self.mesh_device = mesh_device
        self.cfg = cfg
        self.sp, self.tp = tuple(mesh_device.shape)
        self.chunk_size = chunk_size
        self.chunk_local = chunk_size // self.sp
        self.max_seq_len = max_seq_len
        assert chunk_size % (32 * self.sp) == 0 and max_seq_len % chunk_size == 0
        if self.sp > 1:
            assert self.chunk_local >= cfg.sliding_window, f"chunk/SP ({self.chunk_local}) must be >= sliding window ({cfg.sliding_window})"
        self.layer_ids = list(layers) if layers is not None else list(range(cfg.num_hidden_layers))
        self.sp_topo, self.tp_topo = per_axis_topology(fabric_config)
        # GEMMA4_NUM_LINKS: fabric links used by the MoE dispatch/combine + ring CCLs (QuietBox chips have >= 2).
        self.num_links = int(os.environ.get("GEMMA4_NUM_LINKS", "2"))
        self.ccl = CCLManager(mesh_device, num_links=self.num_links, topology=self.sp_topo)

        H = cfg.hidden_size
        emb = reader.get("embed_tokens.weight")
        self.embed_scale = float(torch.tensor(H**0.5, dtype=torch.bfloat16))  # HF casts the scale to the weight dtype
        rep = ttnn.ReplicateTensorToMesh(mesh_device)
        self.embed_w = ttnn.from_torch(emb.bfloat16(), device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=rep,
                                       memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # Tied LM head, vocab sharded over TP cols (replicated over SP rows). Serving prefill only emits KV.
        self.vocab = emb.shape[0]
        self.lm_head_w = None if not build_lm_head else ttnn.from_torch(emb.T.contiguous()[None, None], device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b,
                                         mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(self.sp, self.tp), dims=(None, 3)),
                                         memory_config=ttnn.DRAM_MEMORY_CONFIG)
        del emb
        self.final_norm = TtRMSNorm(mesh_device, reader.get("norm.weight"), cfg.rms_norm_eps)
        self.softcap = cfg.final_logit_softcapping

        self.rope = {
            t: build_indexed_rope(mesh_device, cfg.rope_spec(t), max_seq_len=max_seq_len, chunk_size=chunk_size) for t in (SLIDING, FULL)
        }
        self.trans_mat = build_transformation_mat(mesh_device)

        # Two caches (geometry per layer type); each layer knows its slot within its type's cache.
        self.cache_layer = {}
        counts = {SLIDING: 0, FULL: 0}
        for i in self.layer_ids:
            t = cfg.layer_types[i]
            self.cache_layer[i] = counts[t]
            counts[t] += 1
        self.layer_counts = counts
        self.num_users = num_users
        self.kv = {}
        for t, n in (counts.items() if allocate_kv else ()):
            if n == 0:
                continue
            li = cfg.layer_types.index(t)
            nkv_l = len(kv_heads_for_col(0, self.tp, cfg.num_attention_heads, cfg.layer_kv_heads(li)))
            self.kv[t] = allocate_kv_cache(mesh_device, num_layers=n, max_seq_len=max_seq_len, num_users=num_users, n_kv_local=nkv_l,
                                           head_dim=cfg.layer_head_dim(li))

        self.layers = []
        for i in self.layer_ids:
            logger.info(f"loading layer {i} ({cfg.layer_types[i]})")
            sd = reader.layer_state(i)
            self.layers.append(TtDecoderLayer(mesh_device, cfg, i, sd, ccl=self.ccl, sp_topology=self.sp_topo, tp_topology=self.tp_topo, num_links=self.num_links,
                                              seq_len_per_chip=self.chunk_local, expert_dtype=expert_dtype))

    def allocate_kv_caches(self, num_users: int | None = None) -> dict:
        """{SLIDING|FULL: Gemma4KVCache} sized for this model's layers (engine-owned in serving)."""
        caches = {}
        for t, n in self.layer_counts.items():
            if n == 0:
                continue
            li = self.cfg.layer_types.index(t)
            nkv_l = len(kv_heads_for_col(0, self.tp, self.cfg.num_attention_heads, self.cfg.layer_kv_heads(li)))
            caches[t] = allocate_kv_cache(self.mesh_device, num_layers=n, max_seq_len=self.max_seq_len, num_users=num_users or self.num_users,
                                          n_kv_local=nkv_l, head_dim=self.cfg.layer_head_dim(li))
        return caches

    # -------------------------------------------------------------- chunk
    def tokens_to_device(self, ids_bc: torch.Tensor):
        """ids_bc [chunk] in block-cyclic device order -> uint32 [1,1,S_local] per chip (the H2D layout)."""
        return ttnn.from_torch(ids_bc.view(self.sp, 1, self.chunk_local).to(torch.int32), device=self.mesh_device,
                               layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                               mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, mesh_shape=(self.sp, self.tp), dims=(0, None)))

    def embed_device(self, tok):
        """uint32 tokens [1,1,S_local] (block-cyclic, may hold the 0xFFFFFFFF pad) -> [1,1,S_local,H]."""
        tok = clamp_pad_tokens(tok, self.vocab)
        tok = ttnn.reshape(tok, (1, self.chunk_local))
        e = ttnn.embedding(tok, self.embed_w, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        e = ttnn.reshape(e, (1, 1, self.chunk_local, self.cfg.hidden_size))
        out = ttnn.mul(e, self.embed_scale)
        e.deallocate(True)
        return out

    def embed(self, ids_bc: torch.Tensor):
        return self.embed_device(self.tokens_to_device(ids_bc))

    def forward_device(self, x, kv_actual: int, *, user: int = 0, valid_end: int | None = None, on_layer_complete=None, capture=None):
        """Run every layer on the embedded chunk ``x``; returns the last hidden. ``on_layer_complete(global_layer_idx)``
        fires once per layer, in order (the tt-d-gen layer-ack contract)."""
        for layer in self.layers:
            t = SLIDING if layer.is_sliding else FULL
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
        return self.forward_device(self.embed(ids_chunk[idx]), kv_actual, user=user, capture=capture, valid_end=valid_end)

    def gather_hidden(self, x, kv_actual):
        """Device hidden [1,1,S_local,H] (block-cyclic) -> host [chunk_size, H] in natural order."""
        dts = ttnn.get_device_tensors(x)
        bc = torch.cat([ttnn.to_torch(dts[s * self.tp]).float()[0, 0] for s in range(self.sp)], 0)
        out = torch.empty_like(bc)
        out[block_cyclic_index(kv_actual, self.sp, self.chunk_local) - kv_actual] = bc
        return out

    def logits_from_row(self, h_row: torch.Tensor):
        """Final norm + LM head + softcap for one hidden row [H] -> logits [V]."""
        t = ttnn.from_torch(h_row.view(1, 1, 1, -1).expand(1, 1, 32, -1).contiguous(), device=self.mesh_device, layout=ttnn.TILE_LAYOUT,
                            dtype=ttnn.bfloat16, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device))
        n = self.final_norm(t)
        lg = ttnn.linear(n, self.lm_head_w, dtype=ttnn.bfloat16)
        if self.softcap:
            lg = ttnn.mul(ttnn.tanh(ttnn.mul(lg, 1.0 / self.softcap)), self.softcap)
        dts = ttnn.get_device_tensors(lg)
        return torch.cat([ttnn.to_torch(dts[c]).float()[0, 0, 0] for c in range(self.tp)], -1)[: self.vocab]

    # -------------------------------------------------------------- request
    def prefill(self, ids: torch.Tensor, user: int = 0, capture=None):
        """ids [S] -> (last-token logits [V], last-chunk hidden [chunk, H]). S is padded up to a chunk multiple
        (padding only affects positions after the last real token)."""
        S = ids.shape[0]
        n_chunks = math.ceil(S / self.chunk_size)
        assert n_chunks * self.chunk_size <= self.max_seq_len
        padded = torch.zeros(n_chunks * self.chunk_size, dtype=ids.dtype)
        padded[:S] = ids
        x = None
        for c in range(n_chunks):
            if x is not None:
                x.deallocate(True)
            x = self.prefill_chunk(padded[c * self.chunk_size : (c + 1) * self.chunk_size], c * self.chunk_size, user, capture)
        kv_last = (n_chunks - 1) * self.chunk_size
        h = self.gather_hidden(x, kv_last)
        return self.logits_from_row(h[S - 1 - kv_last]), h
