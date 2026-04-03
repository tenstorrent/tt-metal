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

    Shared-weight path (multi-model serving)
        Use ``TtnnGraniteTTMModel.from_shared_parameters(parameters, config)``
        to create a new model instance that references an existing pre-processed
        parameter tree without copying it.  All instances share the same device
        weight tensors, enabling hundreds of logical model instances per device.

    Trace-compiled path (lowest-latency inference)
        After construction, call ``model.compile(device, batch_size=1)`` to
        record a TTNN execution trace.  Subsequent calls to
        ``model.execute_compiled(history)`` replay the trace with a single host
        command, eliminating Python dispatch overhead for every intermediate op.

    Architecture wiring (TTNN path)
    --------------------------------
    1. backbone.scaler              – _ttnn_std_scaler (on-device)
    2. backbone.patching            – TtnnGraniteTTMPatching (reshape + permute)
    3. backbone.encoder.patcher     – TtnnGraniteTTMEmbedding (TTNN linear)
    4. backbone.encoder.mlp_mixer_encoder – TtnnGraniteTTMEncoderBlock
                                      (adaptive patching blocks; 3 levels of
                                      time_mixer + channel_mixer + reshape)
    5. decoder.adapter              – TtnnGraniteTTMEmbedding (TTNN linear)
    6. decoder.decoder_block        – TtnnGraniteTTMBlock (2 × TinyTimeMixerLayer)
    7. head                         – TtnnGraniteTTMHead (TTNN linear + reshape)
    8. de-normalisation             – y_hat * scale + loc  (on-device)
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
        self._is_compiled = False

        if self._torch_fallback:
            return

        named_modules: dict[str, torch.nn.Module] = (
            dict(reference_model.named_modules()) if reference_model is not None else {}
        )
        self._build_components(named_modules, parameters, config)

    @classmethod
    def from_shared_parameters(
        cls,
        parameters,
        config,
        reference_model: torch.nn.Module | None = None,
    ) -> "TtnnGraniteTTMModel":
        """Create a model instance sharing an existing pre-processed parameter tree.

        This avoids the cost of re-running ``preprocess_model_parameters`` and
        keeps all weight tensors shared on device.  Safe because weight tensors
        are read-only during inference.
        """
        instance = cls.__new__(cls)
        instance.parameters = parameters
        instance.config = config
        instance.reference_model = reference_model.eval() if reference_model is not None else None
        instance._torch_fallback = False
        instance._is_compiled = False

        named_modules: dict[str, torch.nn.Module] = (
            dict(reference_model.named_modules()) if reference_model is not None else {}
        )
        instance._build_components(named_modules, parameters, config)
        return instance

    # ---------------------------------------------------------------------- #
    # Component wiring                                                         #
    # ---------------------------------------------------------------------- #

    def _build_components(
        self,
        named_modules: dict[str, torch.nn.Module],
        parameters,
        config,
    ) -> None:
        """Wire all TTNN sub-components from a pre-processed parameter tree."""

        def _get(path: str) -> torch.nn.Module | None:
            return named_modules.get(path)

        # 1. Scaler parameters
        _scaler_mod = _get("backbone.scaler")
        self._scaler_module = _scaler_mod
        self._scaler_minimum_scale = float(getattr(_scaler_mod, "minimum_scale", 1e-5))

        # 2. Patching
        self._patching = TtnnGraniteTTMPatching(
            num_patches=config.num_patches,
            patch_length=config.patch_length,
            config=config,
        )

        # 3. backbone.encoder.patcher
        patcher_params = _nested_attr(parameters, "backbone.encoder.patcher")
        patcher_mod = _get("backbone.encoder.patcher")
        self._embedding = TtnnGraniteTTMEmbedding(
            parameters=patcher_params,
            config=config,
            torch_module=patcher_mod if patcher_params is None else None,
        )

        # 4. backbone.encoder.mlp_mixer_encoder
        enc_params = _nested_attr(parameters, "backbone.encoder.mlp_mixer_encoder")
        enc_mod = _get("backbone.encoder.mlp_mixer_encoder")
        if enc_params is not None:
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

        # 5. decoder.adapter
        adapter_params = _nested_attr(parameters, "decoder.adapter")
        adapter_mod = _get("decoder.adapter")
        self._decoder_adapter = TtnnGraniteTTMEmbedding(
            parameters=adapter_params,
            config=config,
            torch_module=adapter_mod if adapter_params is None else None,
        )

        # 6. decoder.decoder_block
        dec_block_params = _nested_attr(parameters, "decoder.decoder_block")
        dec_block_mod = _get("decoder.decoder_block")
        if dec_block_params is not None:
            self._decoder_block = _TtnnDecoderBlock(parameters=dec_block_params, config=config)
        else:
            self._decoder_block = TorchModuleFallback(dec_block_mod) if dec_block_mod is not None else None

        # 7. Head
        head_params = _nested_attr(parameters, "head")
        head_mod = _get("head")
        self._head = TtnnGraniteTTMHead(
            parameters=head_params,
            config=config,
            torch_module=head_mod if head_params is None else None,
        )

    # ---------------------------------------------------------------------- #
    # Trace compilation                                                        #
    # ---------------------------------------------------------------------- #

    def compile(self, device, batch_size: int = 1) -> None:
        """Compile the forward pass using TTNN trace capture.

        Records the op graph once (including program compilation).  Subsequent
        calls to ``execute_compiled`` replay the entire graph with a single host
        command, eliminating Python dispatch overhead (~53 µs × ~160 ops).

        Pre-allocates constant tensors used inside ``_ttnn_std_scaler`` to avoid
        any dynamic tensor creation inside the captured trace.

        Args:
            device: TTNN device handle.
            batch_size: Input batch size for which the trace is compiled.
                        A separate ``compile`` call is needed for each batch size.
        """
        import ttnn

        B = batch_size
        T = self.config.context_length
        C = self.config.num_channels

        # --- Pre-allocate constant tensors for the scaler ---
        # These replace the dynamic ones_like / full_like calls inside
        # _ttnn_std_scaler so that no new tensor allocations occur during the
        # trace (which would produce dangling references on replay).
        self._scaler_consts = {
            # Observation mask: all-ones [B, T, C] – used when observed_mask=None
            "mask": ttnn.ones(
                [B, T, C],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            # Denominator clamp: ones [B, 1, C] – for maximum(denom, 1)
            "ones_small": ttnn.ones(
                [B, 1, C],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            ),
            # Minimum scale constant [B, 1, C]
            "min_scale": ttnn.full(
                [B, 1, C],
                fill_value=self._scaler_minimum_scale,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            ),
        }

        # --- Pre-allocate the input buffer (fixed address for trace replay) ---
        dummy = torch.zeros(B, T, C, dtype=torch.float32)
        self._trace_input = ttnn.from_torch(
            dummy,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # --- Warm-up: populate the TTNN program cache (kernel JIT compilation) ---
        _ = self._forward_ttnn(self._trace_input, device=device, _scaler_consts=self._scaler_consts)

        # --- Capture trace ---
        trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        self._trace_output = self._forward_ttnn(self._trace_input, device=device, _scaler_consts=self._scaler_consts)
        ttnn.end_trace_capture(device, trace_id, cq_id=0)

        self._trace_id = trace_id
        self._trace_device = device
        self._trace_batch_size = batch_size
        self._is_compiled = True

    def execute_compiled(
        self,
        history: "torch.Tensor | ttnn.Tensor",
        observed_mask=None,
    ) -> "ttnn.Tensor":
        """Run inference using the pre-compiled trace.

        Copies ``history`` into the pre-allocated input buffer, then replays
        the captured op graph with a single host command.

        Args:
            history: Input tensor [B, T, C].  B must equal the ``batch_size``
                     passed to ``compile()``.
            observed_mask: Ignored (the compiled trace uses an all-ones mask).
                           Pass None for normal inference.

        Returns:
            Output tensor [B, forecast_len, C] (same device buffer every call).
        """
        import ttnn

        if not self._is_compiled:
            raise RuntimeError("Model not compiled. Call compile(device) first.")

        # Convert history to a host-side bfloat16 tensor then copy into the
        # pre-allocated DRAM buffer at the address captured in the trace.
        if isinstance(history, ttnn.Tensor):
            host_tensor = ttnn.from_torch(
                ttnn.to_torch(history).to(torch.float32),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
        else:
            host_tensor = ttnn.from_torch(
                history.to(torch.float32),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
        ttnn.copy_host_to_device_tensor(host_tensor, self._trace_input)

        ttnn.execute_trace(self._trace_device, self._trace_id, cq_id=0, blocking=True)
        return self._trace_output

    def release_trace(self) -> None:
        """Release the captured trace and free associated device memory."""
        import ttnn

        if self._is_compiled:
            ttnn.release_trace(self._trace_device, self._trace_id)
            self._is_compiled = False

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

    def _forward_ttnn(self, history, *, observed_mask=None, device, _scaler_consts=None):
        import ttnn

        # 1. Scaler
        hidden_states, loc, scale = _ttnn_std_scaler(
            history,
            observed_mask=observed_mask,
            minimum_scale=self._scaler_minimum_scale,
            device=device,
            _consts=_scaler_consts,
        )

        # 2. Patching: [B, T, C] -> [B, C, P, patch_length]
        hidden_states = self._patching(hidden_states, device=device)

        # 3. Patch embedding: [B, C, P, patch_length] -> [B, C, P, d_model]
        hidden_states = self._embedding(hidden_states, device=device)

        # 4. Encoder (3 × TtnnGraniteTTMAdaptivePatchingBlock)
        if self._encoder_block is not None:
            hidden_states = self._encoder_block(hidden_states, device=device)

        # 5. Decoder adapter: [B, C, P, d_model] -> [B, C, P, decoder_d_model]
        hidden_states = self._decoder_adapter(hidden_states, device=device)

        # 6. Decoder block
        if self._decoder_block is not None:
            hidden_states = self._decoder_block(hidden_states, device=device)

        # 7. Head: [B, C, P, d_dec] -> [B, forecast_len, C]
        hidden_states = self._head(hidden_states, device=device)

        # 8. De-normalise: y_hat = y_hat * scale + loc  (on-device broadcast)
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
    """Runs all TinyTimeMixerLayer mixers in decoder.decoder_block sequentially."""

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
    _consts: dict | None = None,
):
    """TTNN implementation of TinyTimeMixerStdScaler.

    Input:  data  [B, T, C]  (bfloat16, on device)
    Output: (scaled [B,T,C], loc [B,1,C], scale [B,1,C])  all on device

    Args:
        _consts: Optional dict of pre-allocated constant tensors used during
                 TTNN trace capture to avoid dynamic tensor creation inside the
                 trace.  Keys: ``mask`` [B,T,C], ``ones_small`` [B,1,C],
                 ``min_scale`` [B,1,C].  When None, tensors are created eagerly.
    """
    import ttnn

    if _consts is not None:
        mask = _consts["mask"]
    elif observed_mask is not None:
        mask = observed_mask
    else:
        mask = ttnn.ones_like(data, memory_config=ttnn.L1_MEMORY_CONFIG)

    # denominator = clamp_min(sum(mask, dim=1), 1)
    denom = ttnn.sum(mask, dim=1, keepdim=True, memory_config=ttnn.L1_MEMORY_CONFIG)
    ones_small = (
        _consts["ones_small"] if _consts is not None else ttnn.ones_like(denom, memory_config=ttnn.L1_MEMORY_CONFIG)
    )
    denom = ttnn.maximum(denom, ones_small)

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
    min_scale_t = (
        _consts["min_scale"]
        if _consts is not None
        else ttnn.full_like(variance, fill_value=minimum_scale, memory_config=ttnn.L1_MEMORY_CONFIG)
    )
    scale = ttnn.sqrt(
        ttnn.add(variance, min_scale_t, memory_config=ttnn.L1_MEMORY_CONFIG),
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # scaled = (data - loc) / scale   → [B, T, C]
    scaled = ttnn.div(diff, scale, memory_config=ttnn.L1_MEMORY_CONFIG)

    return scaled, loc, scale
