# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import torch

import ttnn

from ...layers.linear import ColParallelLinear, Linear, RowParallelLinear
from ...layers.module import Module, ModuleList
from ...layers.normalization import LayerNorm
from ...parallel.config import EncoderParallelConfig
from ...parallel.manager import CCLManager
from .config_wav2vec2 import Wav2Vec2Config


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


class Wav2Vec2PositionalConvEmbedding(Module):
    """HF ``Wav2Vec2PositionalConvEmbedding`` ported to ``ttnn.conv1d``.

    Grouped Conv1d (groups=16, in=out=768, kernel=128, padding=64, bias=True)
    + ``SamePadLayer`` (drops 1 trailing frame for the even kernel) + exact GELU.
    The HF conv weight is ``weight_norm``-parametrized; we materialize it once
    in ``_prepare_torch_state``.
    """

    def __init__(
        self,
        config: Wav2Vec2Config,
        *,
        mesh_device: ttnn.MeshDevice,
    ) -> None:
        super().__init__()
        self.config = config
        self.mesh_device = mesh_device
        # Hardcoded to the HF wav2vec2 positional-conv constants (matches every
        # checkpoint we load: base, large, large-xlsr-53). HF defaults to
        # kernel=128, groups=16.
        self.in_channels = config.hidden_size
        self.kernel_size = 128
        self.padding = self.kernel_size // 2
        self.groups = 16
        self._num_pad_remove = 1 if self.kernel_size % 2 == 0 else 0

        # Set in _prepare_torch_state; cached prepared device tensors set on
        # first forward via conv1d's return_weights_and_bias.
        self._weight_host: ttnn.Tensor | None = None
        self._bias_host: ttnn.Tensor | None = None
        self._weight_device: ttnn.Tensor | None = None
        self._bias_device: ttnn.Tensor | None = None

        # Match the precision tuning used elsewhere in this encoder (see
        # Wav2Vec2Attention): HiFi4 + fp32_dest_acc + packer_l1_acc. Critical
        # for PCC parity against the fp32 HF reference.
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.conv_config = ttnn.Conv1dConfig(
            weights_dtype=ttnn.bfloat16,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        g_key = "conv.parametrizations.weight.original0"
        v_key = "conv.parametrizations.weight.original1"
        b_key = "conv.bias"
        if g_key not in state or v_key not in state or b_key not in state:
            return  # missing-key reporting handled by Module parent
        g = state.pop(g_key)  # [1, 1, kernel]
        v = state.pop(v_key)  # [out, in/groups, kernel]
        b = state.pop(b_key)  # [out]
        # weight_norm: w = (g / ||v||_2-along-kernel) * v
        w = torch._weight_norm(v, g, dim=2)  # [out, in/groups, kernel]
        # conv2d weight layout (conv1d delegates): [out, in/groups, kH=1, kW=kernel].
        w_4d = w.unsqueeze(2).contiguous()
        self._weight_host = ttnn.from_torch(w_4d, dtype=ttnn.float32)
        # Conv2d bias layout: [1, 1, 1, out].
        self._bias_host = ttnn.from_torch(b.reshape(1, 1, 1, -1).contiguous(), dtype=ttnn.float32)

    def forward(self, hidden_BTC: ttnn.Tensor) -> ttnn.Tensor:
        if self._weight_host is None:
            raise RuntimeError("Wav2Vec2PositionalConvEmbedding weights not loaded")
        B, T, C = hidden_BTC.shape
        weight = self._weight_device if self._weight_device is not None else self._weight_host
        bias = self._bias_device if self._bias_device is not None else self._bias_host

        out, T_out, (weight_dev, bias_dev) = ttnn.conv1d(
            input_tensor=hidden_BTC,
            weight_tensor=weight,
            bias_tensor=bias,
            device=self.mesh_device,
            in_channels=C,
            out_channels=C,
            batch_size=B,
            input_length=T,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding,
            groups=self.groups,
            dtype=ttnn.bfloat16,
            conv_config=self.conv_config,
            compute_config=self.compute_kernel_config,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        # Cache device-prepared weights so subsequent calls skip the upload+reshard.
        if self._weight_device is None:
            self._weight_device = weight_dev
            self._bias_device = bias_dev

        # conv1d output is sharded, flat-along-T. Move to interleaved + reshape
        # to the standard [B, T_out, C] layout (matches whisper/squeezebert).
        out = ttnn.sharded_to_interleaved(out)
        out = ttnn.reshape(out, (B, T_out, C))
        if self._num_pad_remove > 0:
            keep = T_out - self._num_pad_remove
            out = ttnn.slice(out, [0, 0, 0], [B, keep, C])
        return ttnn.gelu(out, fast_and_approximate_mode=False)


class Wav2Vec2EncoderStack(Module):
    """N transformer encoder layers preceded by on-device pos_conv_embed.

    Owns the HF ``encoder.pos_conv_embed`` (grouped Conv1d), the residual add,
    and the final ``encoder.layer_norm`` (pre-LN/stable variants only —
    post-LN's pre-layers LN is not supported on device yet).
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
        if not config.do_stable_layer_norm:
            raise NotImplementedError(
                "post-LN wav2vec2 variants (e.g. wav2vec2-base) require an on-device "
                "encoder.layer_norm pre-norm step that hasn't been ported. Only "
                "do_stable_layer_norm=True (large-xlsr-53) is supported."
            )

        self.config = config
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        self.pos_conv_embed = Wav2Vec2PositionalConvEmbedding(config, mesh_device=mesh_device)
        self.layers = ModuleList(
            Wav2Vec2EncoderLayer(
                config,
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
            )
            for _ in range(config.num_hidden_layers)
        )

        # Pre-LN / stable mode: final layer_norm follows the stack.
        self.layer_norm = LayerNorm(
            embedding_dim=config.hidden_size,
            norm_eps=config.layer_norm_eps,
            norm_elementwise_affine=True,
            bias=True,
            mesh_device=mesh_device,
        )

    def forward(
        self, hidden_BLC: ttnn.Tensor, *, output_hidden_states: bool = False
    ) -> ttnn.Tensor | list[ttnn.Tensor]:
        pos = self.pos_conv_embed(hidden_BLC)
        hidden = ttnn.add(hidden_BLC, pos)
        ttnn.deallocate(pos)

        if output_hidden_states:
            all_states = [hidden]
            for layer in self.layers:
                hidden = layer(hidden)
                all_states.append(hidden)
            all_states[-1] = self.layer_norm(all_states[-1])
            return all_states

        for layer in self.layers:
            hidden = layer(hidden)
        return self.layer_norm(hidden)
