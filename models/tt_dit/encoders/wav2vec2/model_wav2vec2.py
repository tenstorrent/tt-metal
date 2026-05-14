# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

import ttnn

from ...layers.module import Module
from ...parallel.config import EncoderParallelConfig
from ...parallel.manager import CCLManager
from ...utils.conv3d import conv_pad_in_channels
from ...utils.tensor import from_torch, to_torch
from .config_wav2vec2 import Wav2Vec2Config
from .encoder_wav2vec2 import Wav2Vec2EncoderStack, Wav2Vec2FeatureProjection
from .feature_extractor_wav2vec2 import Wav2Vec2FeatureExtractor


class Wav2Vec2Encoder(Module):
    """TTNN port of HuggingFace `Wav2Vec2Model` for the WAN 2.2 S2V pipeline.

    On device:
        feature_extractor: 7 Conv1d layers (``ttnn.experimental.conv3d`` with
            ``kernel_size=(k, 1, 1)``; GroupNorm-as-InstanceNorm on layer 0).
        feature_projection: LayerNorm + Linear(512 -> 768).
        encoder.layers: 12 transformer encoder layers.

    On CPU (one short device→host→device hop per audio clip):
        encoder.pos_conv_embed: grouped Conv1d (groups=16, kernel=128).
        encoder.layer_norm: initial LayerNorm before the transformer stack.

    Why pos-conv is on CPU: ``ttnn.experimental.conv3d``'s grouped path
    requires the per-group in_channels to equal ``C_in_block``, which itself
    must be a multiple of TILE_WIDTH (32). With ``in_per_group = 768 / 16 = 48``
    no valid ``C_in_block`` satisfies both constraints. The initial LayerNorm
    runs on CPU too because it directly consumes the pos-conv output. Both ops
    are tiny (~3M FLOPs total) and run once per audio clip outside the denoise
    loop, so the round-trip cost is negligible.

    Weight loading: call ``load_torch_state_dict(hf_model.state_dict())`` and
    ``bind_cpu_modules(hf_model)``.
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

        self.feature_extractor = Wav2Vec2FeatureExtractor(config, mesh_device=mesh_device)
        self.feature_projection = Wav2Vec2FeatureProjection(config, mesh_device=mesh_device)
        self.encoder = Wav2Vec2EncoderStack(
            config,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
        )

        # CPU-only modules — populated by `bind_cpu_modules`.
        self._cpu_pos_conv_embed: torch.nn.Module | None = None
        self._cpu_pre_layer_norm: torch.nn.Module | None = None

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # `masked_spec_embed` is a training-time buffer for SpecAugment.
        # ``encoder.pos_conv_embed.*`` is loaded via ``bind_cpu_modules``.
        # ``encoder.layer_norm.*`` is loaded via ``bind_cpu_modules`` in
        # post-LN mode, but in stable-LN mode it lives in the device stack
        # under ``encoder.layer_norm.*`` — let those keys flow through.
        state.pop("masked_spec_embed", None)
        for k in list(state):
            if k.startswith("encoder.pos_conv_embed."):
                state.pop(k)
            elif k.startswith("encoder.layer_norm.") and not self.config.do_stable_layer_norm:
                state.pop(k)

    def bind_cpu_modules(self, hf_model: torch.nn.Module) -> None:
        """Snapshot the HF modules used on host.

        In post-LN mode (wav2vec2-base), both ``encoder.pos_conv_embed`` and
        ``encoder.layer_norm`` are needed on host. In stable mode
        (wav2vec2-large-xlsr), only ``encoder.pos_conv_embed`` is used on host;
        the final ``encoder.layer_norm`` runs on device inside the stack.
        """
        self._cpu_pos_conv_embed = hf_model.encoder.pos_conv_embed.eval()
        if not self.config.do_stable_layer_norm:
            self._cpu_pre_layer_norm = hf_model.encoder.layer_norm.eval()

    def forward(
        self,
        input_values_torch: torch.Tensor,
        *,
        output_hidden_states: bool = False,
    ) -> ttnn.Tensor | list[ttnn.Tensor]:
        """Run the encoder end-to-end on device.

        Args:
            input_values_torch: CPU `torch.Tensor` shaped `[B, T_raw]` or
                `[B, T_raw, 1]`, already normalized by `Wav2Vec2Processor`.
            output_hidden_states: If `True`, return a list of
                `num_hidden_layers + 1` tensors. Matches HF's
                `output_hidden_states=True`.
        """
        x = input_values_torch.squeeze(-1) if input_values_torch.dim() == 3 else input_values_torch
        B, T_raw = x.shape

        # [B, T_raw] -> [B, T_raw, 1, 1, 1] then pad C to the conv3d alignment.
        # Audio is uploaded at the feature extractor's dtype (fp32 for
        # large-xlsr to bound LayerNorm precision loss; bf16 for base).
        x_BTHWC = x.reshape(B, T_raw, 1, 1, 1)
        x_BTHWC = conv_pad_in_channels(x_BTHWC)
        audio_dev = from_torch(
            x_BTHWC,
            device=self.mesh_device,
            dtype=self.feature_extractor.dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        # 7-layer feature extractor (conv3d). Runs in fp32 for the
        # "layer"-norm variant (wav2vec2-large-xlsr) to bound cumulative
        # per-layer LayerNorm precision loss.
        feats = self.feature_extractor(audio_dev)  # [B, T_out, 1, 1, 512] TILE

        # Collapse the degenerate H, W dims to feed the 3D-shaped projection.
        feats = ttnn.reshape(feats, (B, feats.shape[1], self.config.conv_dim[-1]))

        # Boundary cast: the feature_projection's Linear+LayerNorm carries
        # bf16 weights, so the fp32 feature extractor's output is cast down
        # before entering the bf16 transformer trunk.
        if feats.dtype != ttnn.bfloat16:
            feats = ttnn.typecast(feats, ttnn.bfloat16)

        # Feature projection (LayerNorm + Linear(512 -> 768)) — on device.
        projected = self.feature_projection(feats)

        # Pos-conv — always on CPU. Initial LayerNorm placement depends on
        # the variant:
        #   * post-LN (base): applied here on CPU before the stack.
        #   * pre-LN / stable (large-xlsr): NOT applied here — owned by
        #     `Wav2Vec2EncoderStack.layer_norm` and run on device after the stack.
        if self._cpu_pos_conv_embed is None:
            raise RuntimeError("bind_cpu_modules(hf_model) was not called.")
        projected_torch = to_torch(projected).float().reshape(B, -1, self.config.hidden_size)
        with torch.no_grad():
            pos = self._cpu_pos_conv_embed(projected_torch)
            hidden_torch = projected_torch + pos
            if not self.config.do_stable_layer_norm:
                if self._cpu_pre_layer_norm is None:
                    raise RuntimeError("post-LN variant requires bound _cpu_pre_layer_norm")
                hidden_torch = self._cpu_pre_layer_norm(hidden_torch)
        hidden = from_torch(
            hidden_torch,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )

        # Transformer encoder layers on device. In stable mode the stack also
        # applies the final layer_norm on device.
        return self.encoder(hidden, output_hidden_states=output_hidden_states)
