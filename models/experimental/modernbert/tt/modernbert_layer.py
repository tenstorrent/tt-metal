# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""TTNN ModernBERT encoder layer: pre-norm attention and MLP with residuals.

    h = h + attn(attn_norm(h))
    h = h + mlp(mlp_norm(h))

Layer 0 has NO attention norm. HF uses nn.Identity() there because the embedding
LayerNorm has already normalised the input, so there is no
`layers.0.attn_norm.weight` in the checkpoint and prepare_weights stores None.
Applying a norm at layer 0 is rejected by strict state_dict loading, which is how
the reference-side negative control detects it.
"""

import ttnn
from models.experimental.modernbert.tt import model_config as _cfg
from models.experimental.modernbert.tt.modernbert_attention import TtnnModernBertAttention
from models.experimental.modernbert.tt.modernbert_mlp import TtnnModernBertMLP


class TtnnModernBertEncoderLayer:
    def __init__(
        self,
        parameters,
        config,
        layer_idx,
        down_core_grid=None,
        qkv_program_config=None,
        sdpa_program_config=None,
        mlp_act_program_config=None,
        mlp_gate_program_config=None,
        mlp_shard=None,
    ):
        self.eps = config.norm_eps
        # None at layer 0 - see module docstring.
        self.attn_norm = parameters["attn_norm"]
        self.mlp_norm = parameters["mlp_norm"]
        self.layer_type = config.layer_types[layer_idx]
        self.attn = TtnnModernBertAttention(
            parameters["attn"],
            config,
            self.layer_type,
            down_core_grid=down_core_grid,
            qkv_program_config=qkv_program_config,
            sdpa_program_config=sdpa_program_config,
        )
        self.mlp = TtnnModernBertMLP(
            parameters["mlp"],
            down_core_grid=down_core_grid,
            act_program_config=mlp_act_program_config,
            gate_program_config=mlp_gate_program_config,
            shard=mlp_shard,
        )

    def __call__(self, hidden_states, rotary, attn_mask=None):
        plan_ = self.mlp.shard
        if plan_ is not None and plan_.norm is not None and _cfg._SHARD_RESIDUAL_STREAM:
            return self._resident(hidden_states, rotary, attn_mask, plan_)

        if self.attn_norm is None:
            normed = hidden_states
        else:
            normed = ttnn.layer_norm(hidden_states, weight=self.attn_norm, epsilon=self.eps)

        attn_out = self.attn(normed, rotary, attn_mask=attn_mask)
        if self.attn_norm is not None:
            ttnn.deallocate(normed)

        hidden_states = ttnn.add(hidden_states, attn_out)
        ttnn.deallocate(attn_out)

        plan = self.mlp.shard
        if plan is not None and plan.norm is not None:
            return self._sharded_mlp_half(hidden_states, plan)

        mlp_normed = ttnn.layer_norm(hidden_states, weight=self.mlp_norm, epsilon=self.eps)
        mlp_out = self.mlp(mlp_normed)
        ttnn.deallocate(mlp_normed)

        out = ttnn.add(hidden_states, mlp_out)
        ttnn.deallocate(hidden_states)
        ttnn.deallocate(mlp_out)
        return out

    def _sharded_mlp_half(self, hidden_states, plan):
        """Norm, GeGLU and the residual add all in L1 on one 6x8 grid.

        The point is the Wi matmuls: with an interleaved in0 they run at 33.9%
        utilisation against the 78% the down-projection reaches reading L1. Paying
        one explicit reshard on the way in buys an L1 in0 for both of them, and the
        residual add then stays sharded too, so only one trip back to interleaved
        is needed for the next layer's attention.
        """
        h = ttnn.to_memory_config(hidden_states, plan.hidden_memory)
        ttnn.deallocate(hidden_states)

        normed = ttnn.layer_norm(
            h,
            weight=self.mlp_norm,
            epsilon=self.eps,
            program_config=plan.norm,
            memory_config=plan.hidden_memory,
        )
        mlp_out = self.mlp(normed)
        ttnn.deallocate(normed)

        out = ttnn.add(h, mlp_out, memory_config=plan.hidden_memory)
        ttnn.deallocate(h)
        ttnn.deallocate(mlp_out)

        interleaved = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out)
        return interleaved

    def _resident(self, h, rotary, attn_mask, plan):
        """Whole layer with the residual stream L1-resident, in and out.

        Still two reshards per layer, but both residual adds and both norms now run
        sharded: the same (8,256,768) add costs 73.1 us interleaved and 4.7 sharded.

        The reshard before attention looks removable - Wqkv could read the shard
        directly - and is not: that costs 5.9%, because a sharded in0
        puts Wqkv on the GeGLU's 48-core grid. See _SHARD_GRID_X.
        """
        if self.attn_norm is None:
            normed_sh = h
        else:
            normed_sh = ttnn.layer_norm(
                h,
                weight=self.attn_norm,
                epsilon=self.eps,
                program_config=plan.norm,
                memory_config=plan.hidden_memory,
            )
        normed = ttnn.to_memory_config(normed_sh, _cfg.attention_interleaved())
        if self.attn_norm is not None:
            ttnn.deallocate(normed_sh)
        attn_out = self.attn(normed, rotary, attn_mask=attn_mask)
        ttnn.deallocate(normed)

        attn_sh = ttnn.to_memory_config(attn_out, plan.hidden_memory)
        ttnn.deallocate(attn_out)
        h2 = ttnn.add(h, attn_sh, memory_config=plan.hidden_memory)
        ttnn.deallocate(h)
        ttnn.deallocate(attn_sh)

        normed2 = ttnn.layer_norm(
            h2,
            weight=self.mlp_norm,
            epsilon=self.eps,
            program_config=plan.norm,
            memory_config=plan.hidden_memory,
        )
        mlp_out = self.mlp(normed2)
        ttnn.deallocate(normed2)

        out = ttnn.add(h2, mlp_out, memory_config=plan.hidden_memory)
        ttnn.deallocate(h2)
        ttnn.deallocate(mlp_out)
        return out
