# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``TtV4PrefillTransformer``: this rank's slice of DeepSeek-V4-Flash for the pure-ttnn prefill.

first rank:  token ids [1, 1, S_l] -> TtParallelEmbedding -> [1, 1, S_l, D_l] -> 4 identical streams
             (reference: ``inputs_embeds.unsqueeze(2).expand(hc_mult)``)
other ranks: the packed streams [1, 1, S_l, 4 D_l] from the upstream rank's D2D socket -> unpack
layers:      TtV4PrefillBlock per global layer index (this rank's contiguous slice)
last rank:   TtHyperHead -> final RMSNorm -> [1, 1, S_l, D_l] (the LM head is not built here: the prefill worker's
             output is the KV; ``kv_only_last_layer`` skips the tail entirely and the last block returns None)
non-last:    the packed streams for the D2D socket.
"""

from __future__ import annotations

from typing import Callable, Optional

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import DeepseekV4RotaryEmbedding
from models.demos.deepseek_v3_d_p.tt.tt_distributed_rms_norm import TtDistributedRmsNorm
from models.demos.deepseek_v3_d_p.tt.tt_parallel_embedding import TtParallelEmbedding
from models.demos.deepseek_v3_d_p.tt.v4.block import HC, TtV4PrefillBlock
from models.demos.deepseek_v3_d_p.tt.v4.hyper_connection import TtHyperHead
from models.demos.deepseek_v3_d_p.tt.v4.pp_pack import pack_streams, unpack_streams


class TtV4PrefillTransformer(LightweightModule):
    def __init__(
        self,
        mesh_device,
        cfg,
        *,
        layer_weights: Callable[[int], dict],
        top_level_weights: Optional[dict],
        first_layer_idx: int,
        num_layers: int,
        is_first_rank: bool,
        is_last_rank: bool,
        kv_only_last_layer: bool,
        chunk_tokens: int,
        sp_axis: int = 0,
        tp_axis: int = 1,
        topology=ttnn.Topology.Linear,
        num_links: int = 2,
        num_routed_experts: Optional[int] = None,
        dispatch_buffer_capacity_factor: int = 2,
        weight_cache_path=None,
    ):
        """``layer_weights(global_idx) -> dict`` (reference names + ``"__experts__"``); ``top_level_weights`` has
        ``model.embed_tokens.weight`` (first rank) and ``model.hc_head.{hc_fn,hc_base,hc_scale}`` + ``model.norm.weight``
        (last rank with a tail)."""
        super().__init__()
        self.mesh_device, self.cfg = mesh_device, cfg
        self.first_layer_idx, self.num_layers = int(first_layer_idx), int(num_layers)
        self.is_first_rank, self.is_last_rank = bool(is_first_rank), bool(is_last_rank)
        self.kv_only_last_layer = bool(kv_only_last_layer)
        self.sp_axis, self.tp_axis = sp_axis, tp_axis
        sp = mesh_device.shape[sp_axis]
        assert chunk_tokens % (32 * sp) == 0, (chunk_tokens, sp)
        self.chunk_tokens = int(chunk_tokens)
        self.rotary_emb = DeepseekV4RotaryEmbedding(cfg)
        tl = top_level_weights or {}
        self.embed = (
            TtParallelEmbedding(
                mesh_device=mesh_device,
                vocab_size=cfg.vocab_size,
                emb_dim=cfg.hidden_size,
                torch_weight=tl.get("model.embed_tokens.weight"),
                sp_axis=sp_axis,
                tp_axis=tp_axis,
                weight_cache_path=weight_cache_path,
            )
            if self.is_first_rank
            else None
        )
        self.layers = []
        for local in range(self.num_layers):
            gidx = self.first_layer_idx + local
            kv_only = self.kv_only_last_layer and self.is_last_rank and local == self.num_layers - 1
            self.layers.append(
                TtV4PrefillBlock(
                    mesh_device,
                    cfg,
                    gidx,
                    layer_weights(gidx),
                    rotary_emb=self.rotary_emb,
                    seq_len_per_chip=self.chunk_tokens // sp,
                    sp_axis=sp_axis,
                    tp_axis=tp_axis,
                    topology=topology,
                    num_links=num_links,
                    kv_only=kv_only,
                    num_routed_experts=num_routed_experts,
                    dispatch_buffer_capacity_factor=dispatch_buffer_capacity_factor,
                    weight_cache_path=weight_cache_path,
                )
            )
        self.build_tail = self.is_last_rank and not self.kv_only_last_layer
        if self.build_tail:
            self.hc_head = TtHyperHead(
                mesh_device,
                hidden=cfg.hidden_size,
                hc_fn=tl["model.hc_head.hc_fn"],
                hc_base=tl["model.hc_head.hc_base"],
                hc_scale=tl["model.hc_head.hc_scale"],
                rms_eps=cfg.rms_norm_eps,
                hc_eps=cfg.hc_eps,
                sp_axis=sp_axis,
                tp_axis=tp_axis,
            )
            self.norm = TtDistributedRmsNorm(
                mesh_device=mesh_device,
                emb_dim=cfg.hidden_size,
                epsilon=cfg.rms_norm_eps,
                torch_weight=tl.get("model.norm.weight"),
                cluster_axis=tp_axis,
                num_links=num_links,
                topology=topology,
                weight_cache_path=weight_cache_path,
                cache_name_prefix="norm",
            )
        else:
            self.hc_head = self.norm = None

    def alloc_states(self, num_users: int, max_seq_len: int) -> None:
        for layer in self.layers:
            layer.alloc_states(num_users, max_seq_len, self.chunk_tokens)

    def forward(
        self,
        x,
        *,
        slot: int,
        caches,
        actual_start: int,
        actual_end: int,
        input_ids=None,
        on_layer_complete: Optional[Callable[[int], None]] = None,
        on_layer_hidden: Optional[Callable[[int, list], None]] = None,
    ):
        if self.is_first_rank:
            h = ttnn.unsqueeze_to_4D(self.embed(x))  # [1, 1, S_l, D_l]
            streams = [h] + [ttnn.clone(h) for _ in range(HC - 1)]
        else:
            streams = unpack_streams(x)
        for layer in self.layers:
            streams = layer(
                streams,
                slot=slot,
                caches=caches,
                actual_start=actual_start,
                actual_end=actual_end,
                input_ids=input_ids,
                on_layer_complete=on_layer_complete,
                on_layer_hidden=on_layer_hidden,
            )
            if streams is None:  # kv_only last layer
                return None
        if not self.is_last_rank:
            return pack_streams(streams)
        if self.build_tail:
            h = self.hc_head(streams)
            return self.norm(h)
        return pack_streams(streams)
