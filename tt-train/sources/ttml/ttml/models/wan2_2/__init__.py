# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Wan2.2 DiT for ttml. Defaults are Wan2.2-T2V-A14B; one expert per instance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ttnn

import ttml
from ttml.modules import AbstractModuleBase, LinearLayer, ModuleList, Parameter

from .. import RunnerType, memory_efficient_runner
from .conditioning import WanConditioning, timestep_features
from .patch_embed import (
    conv3d_weight_to_linear,
    grid_size,
    patch_features,
    patchify,
    patchify_output_order,
    to_ndhwc,
    unpatchify,
)
from .rope import build_rope_params, build_tables
from .transformer import WanTransformerBlock
from .weights import load_expert_from_safetensors, to_ttml_name

_FINAL_CHUNKS = 2

# ttnn pads the weight's C_in to this and demands the activation match; its default of 32
# would upload every 16-channel latent zero-padded to 32, for the same math.
_CONV3D_ALIGNMENT = 16


def _conv3d_config(device, alignment: int = _CONV3D_ALIGNMENT):
    return ttnn.Conv3dConfig(
        weights_dtype=ttnn.bfloat16,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        T_out_block=1,
        W_out_block=1,
        H_out_block=1,
        C_out_block=32,
        C_in_block=0,
        dilation=[1, 1, 1],
        alignment=alignment,
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
    )


def _read_replicated(tensor) -> np.ndarray:
    """First shard of a replicated parameter. A bare to_numpy() on a mesh tensor raises."""
    native = ttml.autograd.PreferredPrecision.NATIVE
    mesh = ttml.maybe_mesh()
    if mesh is None or mesh.num_devices() == 1:
        return np.asarray(tensor.to_numpy(precision=native), dtype=np.float32)

    device = ttml.autograd.AutoContext.get_instance().get_device()
    composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)
    stacked = np.asarray(tensor.to_numpy(composer=composer, precision=native), dtype=np.float32)
    return stacked[:1]


def prepare_conv3d_patch_weight(weight_host, device, alignment: int = _CONV3D_ALIGNMENT):
    return ttnn.experimental.prepare_conv3d_weights(
        weight_tensor=weight_host, groups=1, C_in_block=0, alignment=alignment, device=device
    )


def conv3d_patch_embed(latent, weight, patch_size, dim, bias=None):
    channels = latent.get_value().shape[-1]
    if channels % _CONV3D_ALIGNMENT:
        # ttnn reports this as the much less obvious "Weight patch size must match input".
        raise ValueError(f"conv3d patch embed needs channels divisible by {_CONV3D_ALIGNMENT}, got {channels}")
    device = latent.get_value().device()
    out = ttnn.experimental.conv3d(
        input_tensor=latent.get_value(),
        weight_tensor=weight,
        device=device,
        config=_conv3d_config(device),
        dtype=ttnn.bfloat16,
        output_channels=dim,
        kernel_size=tuple(patch_size),
        stride=tuple(patch_size),
    )
    batch = out.shape[0]
    out = ttnn.to_layout(ttnn.reshape(out, (batch, 1, -1, dim)), ttnn.Layout.TILE)
    if bias is not None:
        out = ttnn.add(out, bias)  # after tilizing: conv3d emits row-major, tile add is the fast path
    return ttml.autograd.create_tensor(out, False)


def assert_conv3d_patch_embed_is_frozen(model) -> None:
    """No conv3d backward exists, so refuse to run if an adapter was added to patch_embed.

    named_parameters(), not parameters(): the latter's keys are slash-joined and class-prefixed,
    so a "patch_embed." test never matches and the guard silently passes.
    """
    adapted = [name for name, _ in model.named_parameters() if name.startswith("patch_embed.") and "lora" in name]
    if adapted:
        raise RuntimeError(
            f"conv3d patch embed cannot train {adapted}: ttml has no conv3d backward. "
            "Drop patch_embed from the LoRA target set or use the linear patch embed."
        )


@dataclass(frozen=True)
class WanConfig:
    dim: int = 5120
    ffn_dim: int = 13824
    num_layers: int = 40
    num_heads: int = 40
    patch_size: tuple = (1, 2, 2)
    in_channels: int = 16
    out_channels: int = 16
    text_dim: int = 4096
    freq_dim: int = 256
    cross_attn_norm: bool = True
    eps: float = 1e-6
    rope_max_seq_len: int = 1024
    model_type: str = "t2v"
    runner_type: RunnerType = RunnerType.Default
    init_weights: bool = True
    use_tp: bool = False

    def weight_init(self):
        return ttml.init.normal(0.0, 0.02) if self.init_weights else ttml.init.zeros()

    def __post_init__(self) -> None:
        if self.dim % self.num_heads:
            raise ValueError(f"dim {self.dim} must be divisible by num_heads {self.num_heads}")
        if self.model_type not in ("t2v", "i2v"):
            raise ValueError(f"model_type must be t2v or i2v, got {self.model_type!r}")

    @property
    def head_dim(self) -> int:
        return self.dim // self.num_heads


class WanTransformer3D(AbstractModuleBase):
    """One MoE expert. Input and output are patch tokens; see patch_embed for the layout."""

    def __init__(self, config: WanConfig) -> None:
        super().__init__()
        self.config = config

        self.patch_embed = LinearLayer(
            patch_features(config.in_channels, config.patch_size),
            config.dim,
            True,
            weight_init=config.weight_init(),
        )
        self.condition_embedder = WanConditioning(config)
        self.blocks = ModuleList([WanTransformerBlock(config) for _ in range(config.num_layers)])
        self.scale_shift_table = Parameter(ttml.init.zeros()((1, 1, _FINAL_CHUNKS, config.dim)))
        self.proj_out = LinearLayer(
            config.dim,
            patch_features(config.out_channels, config.patch_size),
            True,
            weight_init=config.weight_init(),
        )
        # Set by enable_conv3d_patch_embed(); None keeps the linear patch embed.
        self._conv3d_patch_weight = None
        self._conv3d_patch_bias = None

    def enable_conv3d_patch_embed(self) -> None:
        """Swap the linear patch embed for ttnn conv3d over the raw latent, in place.

        Call after the checkpoint is loaded: the prepared weight is a snapshot, so a later
        load_expert_from_safetensors would not reach it.
        """
        assert_conv3d_patch_embed_is_frozen(self)
        config = self.config
        params = dict(self.named_parameters())
        linear = _read_replicated(params["patch_embed.weight"])
        checkpoint_layout = linear.reshape(config.dim, config.in_channels, *config.patch_size)
        device = ttml.autograd.AutoContext.get_instance().get_device()
        # The weight must reach prepare_conv3d_weights on host; round-trip through from_numpy
        # rather than pull torch into the model package for one conversion.
        on_device = ttml.autograd.Tensor.from_numpy(
            np.ascontiguousarray(checkpoint_layout), ttnn.Layout.ROW_MAJOR, ttnn.bfloat16
        )
        self._conv3d_patch_weight = prepare_conv3d_patch_weight(ttnn.from_device(on_device.get_value()), device)
        self._conv3d_patch_bias = params["patch_embed.bias"].get_value()

    @property
    def uses_conv3d_patch_embed(self) -> bool:
        """True once enable_conv3d_patch_embed() has run, i.e. forward() wants a raw latent."""
        return self._conv3d_patch_weight is not None

    def _embed(self, inputs):
        if self._conv3d_patch_weight is None:
            return self.patch_embed(inputs)
        return conv3d_patch_embed(
            inputs,
            self._conv3d_patch_weight,
            self.config.patch_size,
            self.config.dim,
            bias=self._conv3d_patch_bias,
        )

    def _final_modulation(self, temb):
        """Two chunks here, unlike the six per block. Frozen, so no grad."""
        shifted = ttnn.add(self.scale_shift_table.tensor.get_value(), temb.get_value())
        shift, scale = ttnn.chunk(shifted, _FINAL_CHUNKS, dim=2)
        gamma = ttml.autograd.create_tensor(ttnn.add(scale, 1.0), False)
        beta = ttml.autograd.create_tensor(shift, False)
        return gamma, beta

    def forward(self, inputs, timesteps, text_embed, rope_params, mask=None):
        """-> (B, 1, S, out*prod(patch)) in proj_out order.

        `inputs` is (B, 1, S, in*prod(patch)) patch tokens from patchify, or -- after
        enable_conv3d_patch_embed() -- the raw latent as (B, F, H, W, C) from to_ndhwc.

        Compare against patchify_output_order(target); the model does not unpatchify.
        """
        x = self._embed(inputs)
        timestep_proj, temb, prompt = self.condition_embedder(timesteps, text_embed)

        checkpointed = self.config.runner_type == RunnerType.MemoryEfficient
        for block in self.blocks:
            if checkpointed:
                x = memory_efficient_runner(block, x, mask, prompt, timestep_proj, rope_params)
            else:
                x = block(x, mask, prompt, timestep_proj, rope_params)

        gamma, beta = self._final_modulation(temb)
        x = ttml.ops.layernorm.layernorm(x, gamma, beta)
        return self.proj_out(x)


__all__ = [
    "WanConditioning",
    "WanConfig",
    "WanTransformer3D",
    "WanTransformerBlock",
    "build_rope_params",
    "build_tables",
    "assert_conv3d_patch_embed_is_frozen",
    "conv3d_patch_embed",
    "conv3d_weight_to_linear",
    "grid_size",
    "load_expert_from_safetensors",
    "patch_features",
    "patchify",
    "patchify_output_order",
    "prepare_conv3d_patch_weight",
    "timestep_features",
    "to_ndhwc",
    "to_ttml_name",
    "unpatchify",
]
