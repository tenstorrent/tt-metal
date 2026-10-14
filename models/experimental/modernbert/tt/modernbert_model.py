# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""TTNN ModernBERT encoder: embeddings, 22 layers, final norm.

Layer types alternate full/sliding/sliding, i.e. full attention on indices
0, 3, 6, ... 21 (global_attn_every_n_layers = 3). The pattern is read from
config.layer_types rather than recomputed, so it cannot drift from HF.

Masks and rotary caches are built once per sequence length and shared across
every layer of the matching type, rather than rebuilt per layer.
"""

import ttnn
from models.experimental.modernbert.tt import model_config as _cfg
from models.experimental.modernbert.tt.model_config import (
    down_projection_core_grid,
    mlp_shard_plan,
    mlp_up_projection_program_config,
    qkv_matmul_program_config,
    sdpa_program_config,
)
from models.experimental.modernbert.tt.modernbert_embeddings import TtnnModernBertEmbeddings
from models.experimental.modernbert.tt.modernbert_layer import TtnnModernBertEncoderLayer
from models.experimental.modernbert.tt.modernbert_masks import build_masks, deallocate_masks
from models.experimental.modernbert.tt.modernbert_rope import TtnnModernBertRotary


class TtnnModernBertModel:
    def __init__(self, parameters, config, device, seq_len, attention_mask=None, batch_size=1):
        self.config = config
        self.device = device
        self.seq_len = seq_len
        self.eps = config.norm_eps
        self.embeddings = TtnnModernBertEmbeddings(parameters["embeddings"], config)
        # Derive batch from attention_mask when present, matching build_masks, so
        # the two cannot disagree about the effective batch size.
        effective_batch = attention_mask.shape[0] if attention_mask is not None else batch_size
        self.batch_size = effective_batch
        self.rotary = TtnnModernBertRotary(config, device, seq_len, batch_size=effective_batch)
        # batch_size is only consulted when attention_mask is None.
        self.masks = build_masks(config, device, seq_len, attention_mask=attention_mask, batch_size=batch_size)
        down_core_grid = down_projection_core_grid(device)
        qkv_pc = qkv_matmul_program_config(device, effective_batch, seq_len, config.hidden_size)
        sdpa_pc = sdpa_program_config(device, seq_len, effective_batch * seq_len)
        mlp_act_pc = mlp_up_projection_program_config(
            device, effective_batch, seq_len, config.hidden_size, config.intermediate_size, fuse_gelu=True
        )
        mlp_gate_pc = mlp_up_projection_program_config(
            device, effective_batch, seq_len, config.hidden_size, config.intermediate_size, fuse_gelu=False
        )
        # None below the work-per-core threshold; see mlp_shard_plan.
        mlp_shard = mlp_shard_plan(device, effective_batch, seq_len, config.hidden_size, config.intermediate_size)
        self.layers = [
            TtnnModernBertEncoderLayer(
                parameters["layers"][i],
                config,
                i,
                down_core_grid=down_core_grid,
                qkv_program_config=qkv_pc,
                sdpa_program_config=sdpa_pc,
                mlp_act_program_config=mlp_act_pc,
                mlp_gate_program_config=mlp_gate_pc,
                mlp_shard=mlp_shard,
            )
            for i in range(config.num_hidden_layers)
        ]
        self.final_norm = parameters["final_norm"]
        # Lets __call__ shard the residual stream once at the top rather than per layer.
        self.mlp_shard = mlp_shard

    def __call__(self, input_ids):
        """input_ids: ttnn uint32 (B, S) ROW_MAJOR.

        Returns last_hidden_state as a ttnn tensor, shape (B, S, hidden). Any
        padding mask was supplied at construction time, since it is fixed for the
        lifetime of a given input shape.
        """
        # Masks, rotary caches and every program config are built for one shape at
        # construction. Feeding another reaches the matmul as `num_blocks_y <=
        # grid.y`, which does not say what is wrong.
        got = tuple(input_ids.shape)[-2:]
        want = (self.batch_size, self.seq_len)
        if got != want:
            raise ValueError(
                f"model was built for (batch, seq) {want}, got {got}; construct a new model for that shape"
            )

        hidden = self.embeddings(input_ids)

        resident = self.mlp_shard is not None and self.mlp_shard.norm is not None and _cfg._SHARD_RESIDUAL_STREAM
        if resident:
            # One reshard in for the whole stack, not one per layer.
            sharded = ttnn.to_memory_config(hidden, self.mlp_shard.hidden_memory)
            ttnn.deallocate(hidden)
            hidden = sharded

        for layer in self.layers:
            hidden = layer(hidden, self.rotary, attn_mask=self.masks[layer.layer_type])

        if resident:
            interleaved = ttnn.to_memory_config(hidden, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(hidden)
            hidden = interleaved

        out = ttnn.layer_norm(hidden, weight=self.final_norm, epsilon=self.eps)
        ttnn.deallocate(hidden)
        return out

    def deallocate(self):
        self.rotary.deallocate()
        deallocate_masks(self.masks)
