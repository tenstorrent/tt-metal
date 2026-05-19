# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.wormhole.bge_m3.tt.attention import BgeM3Attention, BgeM3AttentionConfig
from models.demos.wormhole.bge_m3.tt.mlp import BgeM3MLP, BgeM3MLPConfig
from models.demos.wormhole.bge_m3.tt.norm import LayerNorm1D, LayerNorm1DConfig
from models.demos.wormhole.bge_m3.tt.weight_adapter import LayerNormWeights, build_attention_weights, build_mlp_weights


class BgeM3TransformerBlock(LightweightModule):
    """
    Layer-only transformer block that transforms hidden states in-place.
    """

    def __init__(self, args, mesh_device, dtype, state_dict, layer_num, optimizations=None):
        super().__init__()
        self.args = args
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.layer_num = layer_num

        attention_weights = build_attention_weights(state_dict, layer_num, dtype, ttnn.bfloat16)
        mlp_weights = build_mlp_weights(state_dict, layer_num, dtype, ttnn.bfloat16)
        max_seq_len = getattr(args, "max_seq_len", None)
        max_batch_size = getattr(args, "max_batch_size", None)

        self.attention = BgeM3Attention.from_config(
            _build_attention_config(
                args, attention_weights, mesh_device, dtype, max_seq_len, max_batch_size, optimizations
            )
        )
        self.attention_norm = _build_optional_layer_norm(
            attention_weights.layer_norm,
            eps=args.norm_eps,
            mesh_device=mesh_device,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            optimizations=optimizations,
        )

        self.feed_forward = BgeM3MLP.from_config(
            _build_mlp_config(args, mlp_weights, mesh_device, dtype, max_seq_len, max_batch_size, optimizations)
        )
        self.feed_forward_norm = _build_optional_layer_norm(
            mlp_weights.layer_norm,
            eps=args.norm_eps,
            mesh_device=mesh_device,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            optimizations=optimizations,
        )

    def _use_sharded_handoff(self) -> bool:
        """True when both LNs have sharded configs (B1/S512 path)."""
        return (
            self.attention_norm is not None
            and self.feed_forward_norm is not None
            and self.attention_norm.config.sharded_memcfg is not None
        )

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: ttnn.Tensor | None = None,
        residual_sharded: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor | tuple[ttnn.Tensor, ttnn.Tensor | None]:
        """
        residual_sharded (B1/S512 only): pre-sharded copy of hidden_states from the
        previous block's LN2. Saves the residual I->S reshard in attention_norm.
        When sharded handoff is active, returns (hidden_states, next_residual_sharded).
        Otherwise returns hidden_states only.
        """
        attention_output = self.attention(hidden_states, attention_mask=attention_mask)

        # When both LNs have sharded configs (B1/S512), ask LN1 to also return
        # its internal sharded output. Pass that to LN2's residual to skip the
        # residual I->S reshard (-1.1 us/call).
        use_handoff = self._use_sharded_handoff()

        if self.attention_norm is not None:
            if use_handoff:
                # Use pre-sharded residual from previous block if available
                # (saves another I->S reshard, -1.1 us/call).
                attn_residual = residual_sharded if residual_sharded is not None else hidden_states
                hidden_states, mlp_in_sharded = self.attention_norm(
                    attention_output,
                    residual_input_tensor=attn_residual,
                    return_sharded=True,
                )
                if residual_sharded is not None:
                    ttnn.deallocate(residual_sharded)
            else:
                hidden_states = self.attention_norm(attention_output, residual_input_tensor=hidden_states)
                mlp_in_sharded = None
        else:
            hidden_states = ttnn.add(hidden_states, attention_output)
            mlp_in_sharded = None
        ttnn.deallocate(attention_output)

        mlp_in = hidden_states
        mlp_output = self.feed_forward(mlp_in)
        next_residual_sharded = None
        if self.feed_forward_norm is not None:
            if use_handoff:
                residual = mlp_in_sharded if mlp_in_sharded is not None else mlp_in
                hidden_states, next_residual_sharded = self.feed_forward_norm(
                    mlp_output, residual_input_tensor=residual, return_sharded=True
                )
                if mlp_in_sharded is not None:
                    ttnn.deallocate(mlp_in_sharded)
            else:
                residual = mlp_in_sharded if mlp_in_sharded is not None else mlp_in
                hidden_states = self.feed_forward_norm(mlp_output, residual_input_tensor=residual)
                if mlp_in_sharded is not None:
                    ttnn.deallocate(mlp_in_sharded)
        else:
            hidden_states = ttnn.add(mlp_in, mlp_output)
        ttnn.deallocate(mlp_output)

        if use_handoff:
            return hidden_states, next_residual_sharded
        return hidden_states


def _attention_score_dtype(
    dtype: ttnn.DataType,
    max_seq_len: int | None,
    max_batch_size: int | None,
) -> ttnn.DataType:
    max_batch = 1 if max_batch_size is None else max(1, int(max_batch_size))
    if max_seq_len == 512 and max_batch == 1:
        return dtype
    if max_seq_len == 512 and max_batch == 32:
        return dtype
    return ttnn.bfloat16


def _build_attention_config(args, attention_weights, mesh_device, dtype, max_seq_len, max_batch_size, optimizations):
    """Build BgeM3AttentionConfig, overlaying Optimizations.attention fields when provided."""
    config = BgeM3AttentionConfig(
        wqkv=attention_weights.wqkv,
        bqkv=attention_weights.bqkv,
        wo_weight=attention_weights.wo_weight,
        wo_bias=attention_weights.wo_bias,
        hidden_size=args.dim,
        num_heads=args.n_heads,
        head_dim=args.head_dim,
        mesh_device=mesh_device,
        qkv_dtype=dtype,
        score_dtype=_attention_score_dtype(dtype, max_seq_len, max_batch_size),
        output_dtype=dtype,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    if optimizations is not None and optimizations.attention is not None:
        attn_opts = optimizations.attention
        config = replace(
            config,
            qkv_compute_kernel_cfg=attn_opts.qkv_compute_kernel_cfg,
            output_compute_kernel_cfg=attn_opts.output_compute_kernel_cfg,
            score_compute_kernel_cfg=attn_opts.score_compute_kernel_cfg,
            qkv_memcfg=attn_opts.qkv_memcfg,
            create_heads_memcfg=attn_opts.create_heads_memcfg,
            score_memcfg=attn_opts.score_memcfg,
            output_memcfg=attn_opts.output_memcfg,
            core_grid=attn_opts.core_grid,
            qkv_prg_config=attn_opts.qkv_prg_config,
            output_prg_config=attn_opts.output_prg_config,
        )
    return config


def _build_mlp_config(args, mlp_weights, mesh_device, dtype, max_seq_len, max_batch_size, optimizations):
    """Build BgeM3MLPConfig, overlaying Optimizations.mlp fields when provided."""
    config = BgeM3MLPConfig(
        wi_weight=mlp_weights.wi_weight,
        wi_bias=mlp_weights.wi_bias,
        wo_weight=mlp_weights.wo_weight,
        wo_bias=mlp_weights.wo_bias,
        hidden_size=args.dim,
        intermediate_size=args.intermediate_size,
        mesh_device=mesh_device,
        wi_dtype=dtype,
        wo_dtype=dtype,
        activation_dtype=ttnn.bfloat16,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    if optimizations is not None and optimizations.mlp is not None:
        mlp_opts = optimizations.mlp
        config = replace(
            config,
            wi_compute_kernel_cfg=mlp_opts.wi_compute_kernel_cfg,
            wo_compute_kernel_cfg=mlp_opts.wo_compute_kernel_cfg,
            wi_memcfg=mlp_opts.wi_memcfg,
            wo_memcfg=mlp_opts.wo_memcfg,
            activation_memcfg=mlp_opts.activation_memcfg,
            core_grid=mlp_opts.core_grid,
            wi_prg_config=mlp_opts.wi_prg_config,
            wo_prg_config=mlp_opts.wo_prg_config,
            wi_minimal_config=mlp_opts.wi_minimal_config,
        )
    return config


def _build_optional_layer_norm(
    layer_norm_weights: LayerNormWeights | None,
    eps: float,
    mesh_device,
    max_seq_len: int | None = None,
    max_batch_size: int | None = None,
    optimizations=None,
) -> LayerNorm1D | None:
    if layer_norm_weights is None:
        return None

    config = LayerNorm1DConfig(
        weight=layer_norm_weights.weight,
        bias=layer_norm_weights.bias,
        eps=eps,
        mesh_device=mesh_device,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    if optimizations is not None and optimizations.norm is not None:
        norm_opts = optimizations.norm
        config = replace(
            config,
            compute_kernel_config=norm_opts.compute_kernel_config,
            output_memcfg=norm_opts.output_memcfg,
            program_config=norm_opts.program_config,
            sharded_memcfg=norm_opts.sharded_memcfg,
        )
    return LayerNorm1D.from_config(config)
