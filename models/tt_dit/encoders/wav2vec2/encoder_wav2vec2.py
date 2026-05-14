# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import ttnn

from ...layers.linear import ColParallelLinear, Linear, RowParallelLinear
from ...layers.module import Module, ModuleList
from ...layers.normalization import LayerNorm
from ...parallel.config import EncoderParallelConfig
from ...parallel.manager import CCLManager
from .config_wav2vec2 import Wav2Vec2Config

# Note: there is no on-device `Wav2Vec2PositionalConvEmbedding` here. The HF
# pos-conv is a grouped Conv1d with `groups=16, in_per_group=48, kernel=128`.
# `ttnn.experimental.conv3d`'s grouped path requires `C_in_block == in_per_group`
# AND `C_in_block` must be a multiple of TILE_WIDTH (32). Since 48 isn't a tile
# multiple, no valid `C_in_block` satisfies both constraints simultaneously. Hardware
# verified: `C_in_block=48` → PCC 51 %, `C_in_block=32` → PCC 95.6 %,
# `C_in_block=96` → L1 CB overflow. A minimal grouped conv3d test with
# `in_per_group=32=C_in_block` got PCC 99.998 %, proving the kernel itself is
# correct when sizes align. The pos-conv therefore runs on CPU inside
# `Wav2Vec2Encoder.forward` (~3 M FLOPs, one-shot per audio clip).


class Wav2Vec2FeatureProjection(Module):
    """`feature_projection`: LayerNorm + Linear(512 -> 768)."""

    def __init__(
        self,
        config: Wav2Vec2Config,
        *,
        mesh_device: ttnn.MeshDevice,
    ) -> None:
        super().__init__()
        self.config = config
        self.mesh_device = mesh_device

        self.layer_norm = LayerNorm(
            embedding_dim=config.conv_dim[-1],
            norm_eps=config.layer_norm_eps,
            norm_elementwise_affine=True,
            bias=True,
            mesh_device=mesh_device,
        )
        self.projection = Linear(
            in_features=config.conv_dim[-1],
            out_features=config.hidden_size,
            bias=True,
            mesh_device=mesh_device,
        )

    def forward(self, x_BLC: ttnn.Tensor) -> ttnn.Tensor:
        x = self.layer_norm(x_BLC)
        return self.projection(x)


class Wav2Vec2Attention(Module):
    """Multi-head self-attention with bias on q/k/v/o (matches HF Wav2Vec2)."""

    def __init__(
        self,
        config: Wav2Vec2Config,
        *,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config
        self.num_heads = config.num_attention_heads
        self.embed_dim = config.hidden_size
        self.head_dim = config.head_dim

        tp_axis = parallel_config.tensor_parallel.mesh_axis
        self.q_proj = ColParallelLinear(
            in_features=self.embed_dim,
            out_features=self.embed_dim,
            bias=True,
            mesh_device=mesh_device,
            mesh_axis=tp_axis,
        )
        self.k_proj = ColParallelLinear(
            in_features=self.embed_dim,
            out_features=self.embed_dim,
            bias=True,
            mesh_device=mesh_device,
            mesh_axis=tp_axis,
        )
        self.v_proj = ColParallelLinear(
            in_features=self.embed_dim,
            out_features=self.embed_dim,
            bias=True,
            mesh_device=mesh_device,
            mesh_axis=tp_axis,
        )
        self.out_proj = ColParallelLinear(
            in_features=self.embed_dim,
            out_features=self.embed_dim,
            bias=True,
            mesh_device=mesh_device,
            mesh_axis=tp_axis,
        )

        # Softmax compute config: HiFi4 with packer_l1_acc=False (the precision
        # win there is on the exp/normalize, not L1 accumulation).
        self.softmax_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        # Matmul compute config: HiFi4 with packer_l1_acc=True. The default
        # Linear config uses HiFi2 + packer_l1_acc=True for bf16 weights, which
        # accumulates noticeable error across 12 layers. Bumping to HiFi4 (with
        # packer_l1_acc=True to preserve L1 accumulation precision) cuts the
        # final PCC gap.
        self.matmul_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def forward(self, hidden_BLC: ttnn.Tensor) -> ttnn.Tensor:
        # HF Wav2Vec2Attention scales `q` *before* the q·k matmul:
        #   query_states = self.q_proj(hidden_states) * self.scaling
        # Mathematically equivalent to scaling the scores afterwards, but pre-
        # scaling q keeps the matmul inputs smaller and tends to preserve more
        # precision under bf16 accumulation. We mirror HF exactly.
        scaling = 1.0 / math.sqrt(self.head_dim)
        # Use HiFi4 (with fp32 accumulation) for the projection matmuls — the
        # 12-layer trunk accumulates bf16 errors and HiFi4 is required to
        # preserve PCC > 99.95% end-to-end. The audio encoder runs once per
        # clip outside the denoise loop, so the per-step latency cost is
        # acceptable in exchange for the precision win.
        q = ttnn.multiply(
            self.q_proj(hidden_BLC, compute_kernel_config=self.matmul_compute_kernel_config),
            scaling,
        )
        k = self.k_proj(hidden_BLC, compute_kernel_config=self.matmul_compute_kernel_config)
        v = self.v_proj(hidden_BLC, compute_kernel_config=self.matmul_compute_kernel_config)

        qkv = ttnn.concat([q, k, v], dim=-1)
        num_devices = self.parallel_config.tensor_parallel.factor
        num_local_heads = self.num_heads // num_devices
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv, num_heads=num_local_heads, transpose_key=True
        )

        scores = ttnn.matmul(q, k, compute_kernel_config=self.matmul_compute_kernel_config)
        attn = ttnn.softmax(scores, dim=-1, compute_kernel_config=self.softmax_compute_kernel_config)
        attn_out = ttnn.matmul(attn, v, compute_kernel_config=self.matmul_compute_kernel_config)
        attn_out = ttnn.transformer.concatenate_heads(attn_out)

        attn_out = ttnn.unsqueeze(attn_out, 0)
        orig_shape = list(attn_out.shape)
        if num_devices > 1:
            attn_out = self.ccl_manager.all_gather(
                attn_out, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis, use_hyperparams=True
            )

        dense = self.out_proj(attn_out, compute_kernel_config=self.matmul_compute_kernel_config)
        if num_devices > 1:
            dense = self.ccl_manager.all_gather(
                dense, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis, use_hyperparams=True
            )

        dense_shape = list(dense.shape)
        dense_shape[2] = orig_shape[2]
        dense = ttnn.reshape(dense, tuple(dense_shape), dense.shape)
        return ttnn.reshape(dense, tuple(dense.shape)[1:])


class Wav2Vec2FeedForward(Module):
    """`intermediate_dense` + GeLU + `output_dense` with TP on the inner dim."""

    def __init__(
        self,
        config: Wav2Vec2Config,
        *,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        tp_axis = parallel_config.tensor_parallel.mesh_axis
        self.intermediate_dense = ColParallelLinear(
            in_features=config.hidden_size,
            out_features=config.intermediate_size,
            bias=True,
            activation_fn="gelu",
            mesh_device=mesh_device,
            mesh_axis=tp_axis,
        )
        self.output_dense = RowParallelLinear(
            in_features=config.intermediate_size,
            out_features=config.hidden_size,
            bias=True,
            mesh_device=mesh_device,
            mesh_axis=tp_axis,
            ccl_manager=ccl_manager,
        )

        # HiFi4 + fp32 acc for FFN matmuls, matching the attention path. Each
        # FFN has a fan-out of 3072 (and fan-in on the second linear), where
        # bf16 HiFi2 accumulation visibly compounds across 12 layers.
        # packer_l1_acc=True matches the default Linear config and preserves
        # L1 accumulation precision.
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def forward(self, x_BLC: ttnn.Tensor) -> ttnn.Tensor:
        hidden = self.intermediate_dense(x_BLC, compute_kernel_config=self.compute_kernel_config)
        out = self.output_dense(hidden, compute_kernel_config=self.compute_kernel_config)
        # RowParallelLinear emits a reduce-scattered tensor on the TP axis. Gather
        # back to fully replicated form so downstream ops (residual add, next
        # layer's LayerNorm) see the same shape on every device.
        if self.parallel_config.tensor_parallel.factor > 1:
            needs_unsqueeze = len(out.shape) <= 3
            if needs_unsqueeze:
                out = ttnn.unsqueeze(out, 0)
            out = self.ccl_manager.all_gather(
                out, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis, use_hyperparams=True
            )
            if needs_unsqueeze:
                out = ttnn.squeeze(out, 0)
        return out


class Wav2Vec2EncoderLayer(Module):
    """Single Wav2Vec2 transformer encoder block.

    Switches between post-LN (HF ``Wav2Vec2EncoderLayer``, used by base-960h)
    and pre-LN (``Wav2Vec2EncoderLayerStableLayerNorm``, used by
    large-xlsr-53) based on ``config.do_stable_layer_norm``.
    """

    def __init__(
        self,
        config: Wav2Vec2Config,
        *,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.attention = Wav2Vec2Attention(
            config, mesh_device=mesh_device, ccl_manager=ccl_manager, parallel_config=parallel_config
        )
        self.layer_norm = LayerNorm(
            embedding_dim=config.hidden_size,
            norm_eps=config.layer_norm_eps,
            norm_elementwise_affine=True,
            bias=True,
            mesh_device=mesh_device,
        )
        self.feed_forward = Wav2Vec2FeedForward(
            config, mesh_device=mesh_device, ccl_manager=ccl_manager, parallel_config=parallel_config
        )
        self.final_layer_norm = LayerNorm(
            embedding_dim=config.hidden_size,
            norm_eps=config.layer_norm_eps,
            norm_elementwise_affine=True,
            bias=True,
            mesh_device=mesh_device,
        )

    def forward(self, hidden_BLC: ttnn.Tensor) -> ttnn.Tensor:
        if self.config.do_stable_layer_norm:
            # HF `Wav2Vec2EncoderLayerStableLayerNorm` (pre-LN, used by
            # wav2vec2-large-xlsr-53):
            #   r = hidden
            #   hidden = attention(layer_norm(hidden))
            #   hidden = r + hidden
            #   hidden = hidden + feed_forward(final_layer_norm(hidden))
            attn_residual = hidden_BLC
            normed = self.layer_norm(hidden_BLC)
            hidden = self.attention(normed)
            hidden = attn_residual + hidden
            ff_input = self.final_layer_norm(hidden)
            return hidden + self.feed_forward(ff_input)

        # HF `Wav2Vec2EncoderLayer` (post-LN, used by wav2vec2-base-960h):
        #   r = hidden
        #   hidden = attention(hidden)
        #   hidden = layer_norm(r + hidden)
        #   hidden = hidden + feed_forward(hidden)
        #   hidden = final_layer_norm(hidden)
        attn_residual = hidden_BLC
        hidden = self.attention(hidden_BLC)
        hidden = self.layer_norm(attn_residual + hidden)
        hidden = hidden + self.feed_forward(hidden)
        return self.final_layer_norm(hidden)


class Wav2Vec2EncoderStack(Module):
    """``encoder``: N transformer encoder layers.

    Pos-conv runs on CPU (kernel constraint, see ``Wav2Vec2Encoder`` docstring).

    The placement of the encoder's ``layer_norm`` differs between the two HF
    variants:

      * post-LN (base): applied *before* the stack — handled on CPU in
        ``Wav2Vec2Encoder`` and folded into the host pre-conv pre-norm step.
      * pre-LN / stable (large-xlsr): applied *after* the stack — owned by
        this module (``self.layer_norm``) and run on device.
    """

    def __init__(
        self,
        config: Wav2Vec2Config,
        *,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        self.layers = ModuleList(
            Wav2Vec2EncoderLayer(
                config,
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
            )
            for _ in range(config.num_hidden_layers)
        )

        # In stable mode the encoder's ``layer_norm`` follows the stack; we
        # own it on device. In post-LN mode the same-named module is applied
        # *before* the stack and is handled on CPU by ``Wav2Vec2Encoder``.
        self.layer_norm = (
            LayerNorm(
                embedding_dim=config.hidden_size,
                norm_eps=config.layer_norm_eps,
                norm_elementwise_affine=True,
                bias=True,
                mesh_device=mesh_device,
            )
            if config.do_stable_layer_norm
            else None
        )

    def _prepare_torch_state(self, state):
        # Post-LN mode pulls the ``layer_norm.*`` weights via the CPU path —
        # drop them here so the device loader doesn't trip on missing keys.
        if not self.config.do_stable_layer_norm:
            for k in list(state):
                if k.startswith("layer_norm."):
                    state.pop(k)

    def forward(
        self, hidden_BLC: ttnn.Tensor, *, output_hidden_states: bool = False
    ) -> ttnn.Tensor | list[ttnn.Tensor]:
        if output_hidden_states:
            all_states = [hidden_BLC]
            hidden = hidden_BLC
            for layer in self.layers:
                hidden = layer(hidden)
                all_states.append(hidden)
            if self.layer_norm is not None:
                all_states[-1] = self.layer_norm(all_states[-1])
            return all_states

        hidden = hidden_BLC
        for layer in self.layers:
            hidden = layer(hidden)
        if self.layer_norm is not None:
            hidden = self.layer_norm(hidden)
        return hidden
