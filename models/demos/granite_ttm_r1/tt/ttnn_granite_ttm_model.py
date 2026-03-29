# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import torch

from models.demos.granite_ttm_r1.tt.common import (
    TorchModuleFallback,
    run_model_as_torch_fallback,
    to_torch_tensor,
    to_ttnn_tensor,
)
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
    2. backbone.patching            – TtnnGraniteTTMPatching (torch fallback)
    3. backbone.encoder.patcher     – TtnnGraniteTTMEmbedding (TTNN linear if params)
    4. backbone.encoder.mlp_mixer_encoder – TorchModuleFallback
                                      (adaptive patching reshape logic; TinyTimeMixerBlock
                                      returns a 2-tuple, first element extracted as tensor)
    5. decoder.adapter              – TtnnGraniteTTMEmbedding (TTNN linear if params)
    6. decoder.decoder_block        – TorchModuleFallback
    7. head                         – TtnnGraniteTTMHead (TTNN if params)
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

        # 1. Scaler (TinyTimeMixerStdScaler) – called directly via torch to
        #    preserve the full (scaled_data, loc, scale) 3-tuple.
        self._scaler_module = _get("backbone.scaler")

        # 2. Patching (TinyTimeMixerPatchify) – torch fallback
        patching_mod = _get("backbone.patching")
        self._patching = TtnnGraniteTTMPatching(torch_module=patching_mod)

        # 3. backbone.encoder.patcher – TTNN linear projection (patch_length -> d_model)
        patcher_params = _nested_attr(parameters, "backbone.encoder.patcher")
        patcher_mod = _get("backbone.encoder.patcher")
        self._embedding = TtnnGraniteTTMEmbedding(
            parameters=patcher_params,
            config=config,
            torch_module=patcher_mod if patcher_params is None else None,
        )

        # 4. backbone.encoder.mlp_mixer_encoder – TorchModuleFallback
        #    (TinyTimeMixerAdaptivePatchingBlock contains shape-reshape ops).
        #    TinyTimeMixerBlock.forward returns (embedding, hidden_states) — a plain 2-tuple;
        #    TorchModuleFallback will call extract_prediction_tensor and return the first tensor.
        encoder_block_mod = _get("backbone.encoder.mlp_mixer_encoder")
        self._encoder_block = TorchModuleFallback(encoder_block_mod) if encoder_block_mod is not None else None

        # 5. decoder.adapter – TTNN linear (d_model -> decoder_d_model)
        adapter_params = _nested_attr(parameters, "decoder.adapter")
        adapter_mod = _get("decoder.adapter")
        self._decoder_adapter = TtnnGraniteTTMEmbedding(
            parameters=adapter_params,
            config=config,
            torch_module=adapter_mod if adapter_params is None else None,
        )

        # 6. decoder.decoder_block – TorchModuleFallback
        #    (TinyTimeMixerBlock at decoder_d_model; returns a 2-tuple
        #    (last_hidden_state, hidden_states), first element is the tensor).
        decoder_block_mod = _get("decoder.decoder_block")
        self._decoder_block = TorchModuleFallback(decoder_block_mod) if decoder_block_mod is not None else None

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

        # Work in torch for scaler so we can keep loc/scale for de-normalisation.
        torch_history = to_torch_tensor(history)

        # 1. Scaler: [B, T, C] -> ([B, T, C], [B, 1, C], [B, 1, C])
        if self._scaler_module is not None:
            if observed_mask is not None:
                torch_mask = to_torch_tensor(observed_mask)
            else:
                torch_mask = torch.ones_like(torch_history)
            scaled_values, loc, scale = self._scaler_module(torch_history, torch_mask)
        else:
            scaled_values = torch_history
            loc = torch.zeros(torch_history.shape[0], 1, torch_history.shape[2], dtype=torch_history.dtype)
            scale = torch.ones(torch_history.shape[0], 1, torch_history.shape[2], dtype=torch_history.dtype)

        # Convert scaled values back to ttnn for the rest of the pipeline.
        hidden_states = to_ttnn_tensor(scaled_values, device=device, dtype=ttnn.bfloat16)

        # 2. Patching: [B, T, C] -> [B, C, P, patch_length]
        hidden_states = self._patching(hidden_states, device=device)

        # 3. Patch embedding (patcher Linear): [B, C, P, patch_length] -> [B, C, P, d_model]
        hidden_states = self._embedding(hidden_states, device=device)

        # 4. Encoder (adaptive patching mixer blocks): [B, C, P, d_model] -> [B, C, P, d_model]
        #    TinyTimeMixerBlock.forward returns (embedding, hidden_states_list) — a 2-tuple.
        #    TorchModuleFallback with no output_selector calls extract_prediction_tensor, which
        #    returns the first tensor in the tuple (the embedding).
        if self._encoder_block is not None:
            hidden_states = self._encoder_block(hidden_states, device=device)

        # 5. Decoder adapter Linear: [B, C, P, d_model] -> [B, C, P, decoder_d_model]
        hidden_states = self._decoder_adapter(hidden_states, device=device)

        # 6. Decoder block (TinyTimeMixerBlock at decoder_d_model): [B, C, P, d_dec] -> [B, C, P, d_dec]
        #    The module returns a 2-tuple; TorchModuleFallback extracts the first tensor.
        if self._decoder_block is not None:
            hidden_states = self._decoder_block(hidden_states, device=device)

        # 7. Head: [B, C, P, d_dec] -> [B, forecast_len, C]  (scaled space)
        hidden_states = self._head(hidden_states, device=device)

        # 8. De-normalise: y_hat = y_hat * scale + loc
        #    Bring back to torch for the element-wise broadcast, then re-wrap.
        torch_prediction = to_torch_tensor(hidden_states)
        # loc/scale shape: [B, 1, C]; torch_prediction shape: [B, forecast_len, C]
        torch_prediction = torch_prediction * scale + loc

        return to_ttnn_tensor(torch_prediction, device=device, dtype=ttnn.bfloat16)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _nested_attr(obj: Any, path: str) -> Any | None:
    """Traverse a dotted attribute path on ``obj``, returning None if any
    intermediate attribute is missing."""
    for part in path.split("."):
        if obj is None:
            return None
        obj = getattr(obj, part, None)
    return obj
