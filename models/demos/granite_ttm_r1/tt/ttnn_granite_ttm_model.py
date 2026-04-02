# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import torch

from models.demos.granite_ttm_r1.tt.common import TorchModuleFallback, run_model_as_torch_fallback
from models.demos.granite_ttm_r1.tt.ttnn_granite_ttm_adaptive_block import TtnnGraniteTTMEncoderBlock
from models.demos.granite_ttm_r1.tt.ttnn_granite_ttm_embedding import TtnnGraniteTTMEmbedding
from models.demos.granite_ttm_r1.tt.ttnn_granite_ttm_head import TtnnGraniteTTMHead
from models.demos.granite_ttm_r1.tt.ttnn_granite_ttm_patching import TtnnGraniteTTMPatching


class TtnnGraniteTTMModel:
    """Top-level TTNN model for ibm-granite/granite-timeseries-ttm-r1.

    Construction modes
    ------------------
    TTNN path (preferred)
        Pass ``parameters`` (a preprocessed parameter tree from
        ``ttnn.model_preprocessing.preprocess_model_parameters``) and a
        ``GraniteTTMModelConfig`` as ``config``.  The reference HuggingFace
        model is also required as ``reference_model`` so that components
        without a native TTNN implementation can be run via
        ``TorchModuleFallback``.

    Torch fallback path (backward-compatible)
        Pass only ``reference_model``.  The entire forward pass is delegated to
        ``run_model_as_torch_fallback``.

    Architecture wiring (TTNN path)
    --------------------------------
    1. backbone.scaler              – called directly via torch (returns 3-tuple)
    2. backbone.patching            – TtnnGraniteTTMPatching (reshape + permute)
    3. backbone.encoder.patcher     – TtnnGraniteTTMEmbedding (TTNN linear)
    4. backbone.encoder.mlp_mixer_encoder – TtnnGraniteTTMEncoderBlock
                                      (adaptive patching blocks; 3 levels of
                                      time_mixer + channel_mixer + reshape)
    5. decoder.adapter              – TtnnGraniteTTMEmbedding (TTNN linear)
    6. decoder.decoder_block        – TtnnGraniteTTMBlock (2 × TinyTimeMixerLayer)
    7. head                         – TtnnGraniteTTMHead (TTNN linear + reshape)
    8. de-normalisation             – y_hat * scale + loc  (torch)

    Note: Steps 1-4 correspond to backbone.forward, 5-6 to decoder.forward, and
    7-8 to the prediction head plus de-normalisation done in
    TinyTimeMixerForPrediction.forward.
    """

    def __init__(
        self,
        parameters=None,
        config=None,
        reference_model: torch.nn.Module | None = None,
    ):
        self.parameters = parameters
        self.config = config
        self.reference_model = reference_model.eval() if reference_model is not None else None

        # Pure torch-fallback path: no parameter tree supplied.
        self._torch_fallback = parameters is None

        if self._torch_fallback:
            # Nothing else to initialise – __call__ will delegate entirely.
            return

        # ------------------------------------------------------------------ #
        # TTNN path: wire up each sub-component.                              #
        # reference_model is needed for the sub-modules that still use torch. #
        # ------------------------------------------------------------------ #

        named_modules: dict[str, torch.nn.Module] = (
            dict(reference_model.named_modules()) if reference_model is not None else {}
        )

        def _get(path: str) -> torch.nn.Module | None:
            return named_modules.get(path)

        # 1. Scaler parameters – used by _ttnn_std_scaler in forward.
        _scaler_mod = _get("backbone.scaler")
        self._scaler_module = _scaler_mod  # kept for reference; TTNN path uses minimum_scale only
        self._scaler_minimum_scale = float(getattr(_scaler_mod, "minimum_scale", 1e-5))

        # 2. Patching (TinyTimeMixerPatchify) – TTNN reshape + permute.
        #    stride == patch_length is confirmed for this model (non-overlapping).
        self._patching = TtnnGraniteTTMPatching(
            num_patches=config.num_patches,
            patch_length=config.patch_length,
            config=config,
        )

        # 3. backbone.encoder.patcher – TTNN linear projection (patch_length -> d_model)
        patcher_params = _nested_attr(parameters, "backbone.encoder.patcher")
        patcher_mod = _get("backbone.encoder.patcher")
        self._embedding = TtnnGraniteTTMEmbedding(
            parameters=patcher_params,
            config=config,
            torch_module=patcher_mod if patcher_params is None else None,
        )

        # 4. backbone.encoder.mlp_mixer_encoder – TtnnGraniteTTMEncoderBlock.
        #    Implements 3 adaptive patching levels, each running 2 TinyTimeMixerLayer
        #    blocks (time_mixer + channel_mixer) at factor-expanded patch granularity.
        enc_params = _nested_attr(parameters, "backbone.encoder.mlp_mixer_encoder")
        enc_mod = _get("backbone.encoder.mlp_mixer_encoder")
        if enc_params is not None:
            # Each adaptive level has its own factor (e.g. 4, 2, 1 for TTM-R1).
            _factors = (
                [getattr(enc_mod.mixers[i], "adaptive_patch_factor", 1) for i in range(len(enc_mod.mixers))]
                if enc_mod is not None
                else [1]
            )
            self._encoder_block = TtnnGraniteTTMEncoderBlock(
                parameters=enc_params,
                config=config,
                adaptive_patch_factors=_factors,
            )
        else:
            self._encoder_block = None

        # 5. decoder.adapter – TTNN linear (d_model -> decoder_d_model)
        adapter_params = _nested_attr(parameters, "decoder.adapter")
        adapter_mod = _get("decoder.adapter")
        self._decoder_adapter = TtnnGraniteTTMEmbedding(
            parameters=adapter_params,
            config=config,
            torch_module=adapter_mod if adapter_params is None else None,
        )

        # 6. decoder.decoder_block – TtnnGraniteTTMBlock (2 × TinyTimeMixerLayer).
        #    Each mixer in decoder_block.mixers is a TinyTimeMixerLayer with
        #    patch_mixer and feature_mixer subtrees.
        dec_block_params = _nested_attr(parameters, "decoder.decoder_block")
        dec_block_mod = _get("decoder.decoder_block")
        if dec_block_params is not None:
            self._decoder_block = _TtnnDecoderBlock(parameters=dec_block_params, config=config)
        else:
            self._decoder_block = TorchModuleFallback(dec_block_mod) if dec_block_mod is not None else None

        # 7. Head (TinyTimeMixerForPredictionHead) – TTNN
        head_params = _nested_attr(parameters, "head")
        head_mod = _get("head")
        self._head = TtnnGraniteTTMHead(
            parameters=head_params,
            config=config,
            torch_module=head_mod if head_params is None else None,
        )

    # ---------------------------------------------------------------------- #
    # Forward                                                                  #
    # ---------------------------------------------------------------------- #

    def __call__(
        self,
        history,
        *,
        observed_mask=None,
        future_values=None,
        device=None,
        extra_inputs: dict[str, Any] | None = None,
    ):
        if self._torch_fallback:
            return run_model_as_torch_fallback(
                self.reference_model,
                history,
                observed_mask=observed_mask,
                future_values=future_values,
                device=device,
                extra_inputs=extra_inputs,
            )

        return self._forward_ttnn(history, observed_mask=observed_mask, device=device)

    def _forward_ttnn(self, history, *, observed_mask=None, device):
        import ttnn

        # 1. Scaler: [B, T, C] → scaled [B, T, C],  loc [B, 1, C],  scale [B, 1, C]
        #    All ops stay on-device to avoid PCIe round-trips.
        hidden_states, loc, scale = _ttnn_std_scaler(
            history,
            observed_mask=observed_mask,
            minimum_scale=self._scaler_minimum_scale,
            device=device,
        )

        # 2. Patching: [B, T, C] -> [B, C, P, patch_length]
        hidden_states = self._patching(hidden_states, device=device)

        # 3. Patch embedding (patcher Linear): [B, C, P, patch_length] -> [B, C, P, d_model]
        hidden_states = self._embedding(hidden_states, device=device)

        # 4. Encoder (3 × TtnnGraniteTTMAdaptivePatchingBlock): [B, C, P, d_model] -> [B, C, P, d_model]
        if self._encoder_block is not None:
            hidden_states = self._encoder_block(hidden_states, device=device)

        # 5. Decoder adapter Linear: [B, C, P, d_model] -> [B, C, P, decoder_d_model]
        hidden_states = self._decoder_adapter(hidden_states, device=device)

        # 6. Decoder block (2 × TinyTimeMixerLayer at decoder_d_model): [B, C, P, d_dec] -> [B, C, P, d_dec]
        if self._decoder_block is not None:
            hidden_states = self._decoder_block(hidden_states, device=device)

        # 7. Head: [B, C, P, d_dec] -> [B, forecast_len, C]  (scaled space)
        hidden_states = self._head(hidden_states, device=device)

        # 8. De-normalise: y_hat = y_hat * scale + loc  (on-device broadcast)
        #    loc/scale: [B, 1, C];  hidden_states: [B, forecast_len, C]
        hidden_states = ttnn.add(
            ttnn.mul(hidden_states, scale, memory_config=ttnn.L1_MEMORY_CONFIG),
            loc,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        return hidden_states


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


class _TtnnDecoderBlock:
    """Runs all TinyTimeMixerLayer mixers in decoder.decoder_block sequentially.

    decoder.decoder_block is a TinyTimeMixerBlock whose .mixers list contains
    TinyTimeMixerLayer instances, each with a patch_mixer and feature_mixer.
    We reuse TtnnGraniteTTMBlock to run each layer.
    """

    def __init__(self, *, parameters, config):
        from models.demos.granite_ttm_r1.tt.ttnn_granite_ttm_block import TtnnGraniteTTMBlock

        self._layers = [
            TtnnGraniteTTMBlock(parameters=layer_params, config=config) for layer_params in parameters.mixers
        ]

    def __call__(self, hidden_states, *, device=None, **kwargs):
        for layer in self._layers:
            hidden_states = layer(hidden_states, device=device)
        return hidden_states


def _nested_attr(obj: Any, path: str) -> Any | None:
    """Traverse a dotted attribute path on ``obj``, returning None if any
    intermediate attribute is missing."""
    for part in path.split("."):
        if obj is None:
            return None
        obj = getattr(obj, part, None)
    return obj


def _ttnn_std_scaler(
    data,
    *,
    observed_mask=None,
    minimum_scale: float = 1e-5,
    device=None,
):
    """TTNN implementation of TinyTimeMixerStdScaler.

    Input:  data  [B, T, C]  (bfloat16, on device)
    Output: (scaled [B,T,C], loc [B,1,C], scale [B,1,C])  all on device

    Computes per-channel mean and std over the T dimension, using the
    observed_mask if provided (all-ones is the common inference case).
    """
    import ttnn

    if observed_mask is not None:
        mask = observed_mask
    else:
        mask = ttnn.ones_like(data, memory_config=ttnn.L1_MEMORY_CONFIG)

    # denominator = clamp_min(sum(mask, dim=1), 1)
    denom = ttnn.sum(mask, dim=1, keepdim=True, memory_config=ttnn.L1_MEMORY_CONFIG)
    # Clamp denominator ≥ 1 by adding a tiny guard; for all-ones mask this is exact.
    denom = ttnn.maximum(denom, ttnn.ones_like(denom, memory_config=ttnn.L1_MEMORY_CONFIG))

    # loc = sum(data * mask, dim=1) / denom   → [B, 1, C]
    weighted = ttnn.mul(data, mask, memory_config=ttnn.L1_MEMORY_CONFIG)
    loc = ttnn.div(
        ttnn.sum(weighted, dim=1, keepdim=True, memory_config=ttnn.L1_MEMORY_CONFIG),
        denom,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # variance = sum(((data - loc) * mask)^2, dim=1) / denom   → [B, 1, C]
    diff = ttnn.sub(data, loc, memory_config=ttnn.L1_MEMORY_CONFIG)
    masked_diff = ttnn.mul(diff, mask, memory_config=ttnn.L1_MEMORY_CONFIG)
    sq = ttnn.mul(masked_diff, masked_diff, memory_config=ttnn.L1_MEMORY_CONFIG)
    variance = ttnn.div(
        ttnn.sum(sq, dim=1, keepdim=True, memory_config=ttnn.L1_MEMORY_CONFIG),
        denom,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # scale = sqrt(variance + minimum_scale)   → [B, 1, C]
    min_scale_t = ttnn.full_like(variance, fill_value=minimum_scale, memory_config=ttnn.L1_MEMORY_CONFIG)
    scale = ttnn.sqrt(
        ttnn.add(variance, min_scale_t, memory_config=ttnn.L1_MEMORY_CONFIG),
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # scaled = (data - loc) / scale   → [B, T, C]
    scaled = ttnn.div(diff, scale, memory_config=ttnn.L1_MEMORY_CONFIG)

    return scaled, loc, scale
