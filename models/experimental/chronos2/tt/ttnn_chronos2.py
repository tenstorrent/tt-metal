# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0
#
# Public forecasting interface: create_backend(...) returns a Backend whose
# forecast(past_values, past_observed_mask, prediction_length) maps CPU float32
# arrays to {"quantiles": [B, prediction_length, Q]}. The caller owns the TTNN
# device; it is never opened or closed here.

from __future__ import annotations

import math
import os

import numpy as np

from .executor import TTChronos2Executor
from .model_config import (
    DEFAULT_PRECISION,
    DEVICE_OPTIONS,
    PRECISION_POLICIES,
    SUPPORTED_PRECISIONS,
    Chronos2Config,
    resolve_config,
    validate_device_options,
)
from .preprocess import Preprocessor
from .weights import load_weights_fp32

MAX_PREDICTION_LENGTH = 64  # validated horizon; the checkpoint itself allows longer


class Backend:
    precision_policy: dict

    def __init__(self, weights_path, cfg: Chronos2Config, device, precision: str, device_options: dict | None):
        if precision not in SUPPORTED_PRECISIONS:
            raise ValueError(f"precision '{precision}' is not supported; supported: {SUPPORTED_PRECISIONS}")
        self.precision = precision
        self.precision_policy = PRECISION_POLICIES[precision]
        self.cfg = cfg
        self.preprocessor = Preprocessor(cfg)
        # device_options describes how the caller opened `device` (normally
        # with the module-level DEVICE_OPTIONS); trace replay needs a trace region.
        self.device_options = validate_device_options(DEVICE_OPTIONS if device_options is None else device_options)
        use_trace = (
            self.device_options.get("trace_region_size", 0) > 0 and os.environ.get("CHRONOS2_TT_TRACE", "1") != "0"
        )
        weights = load_weights_fp32(weights_path)
        self.executor = TTChronos2Executor(cfg, weights, device, precision, use_trace=use_trace)

    def num_output_patches(self, prediction_length: int) -> int:
        n = math.ceil(prediction_length / self.cfg.output_patch_size)
        return min(n, self.cfg.max_output_patches)

    def release(self) -> None:
        """Release the live trace and its persistent buffers (device stays open)."""
        self.executor.release_trace()

    def weight_bytes(self) -> int:
        """Device bytes held by uploaded weights/biases (per-precision memory report)."""
        return self.executor.weight_bytes

    def forecast(self, past_values, past_observed_mask, prediction_length):
        values = np.asarray(past_values, dtype=np.float32)
        mask = np.asarray(past_observed_mask, dtype=np.float32)
        if values.ndim != 2 or mask.shape != values.shape:
            raise ValueError("past_values/past_observed_mask must be [B,T] numpy float32 of equal shape")
        if not (1 <= int(prediction_length) <= MAX_PREDICTION_LENGTH):
            raise ValueError(f"prediction_length must be in 1..{MAX_PREDICTION_LENGTH}, got {prediction_length}")
        if values.shape[-1] < 1:
            raise ValueError("empty context")

        n_out = self.num_output_patches(int(prediction_length))
        prepared = self.preprocessor.prepare(values, mask, n_out)
        preds_scaled = self.executor.run(prepared)  # [B, 21, n_out*p]

        loc, scale = prepared["loc"], prepared["scale"]
        q = self.cfg.num_quantiles
        horizon = int(prediction_length)
        preds = preds_scaled.reshape(values.shape[0], q, -1)[:, :, :horizon]
        # per-row loc/scale [B,1] -> [B,1,1] to broadcast over [B, Q, horizon]
        preds = self.preprocessor.unscale(preds, loc[:, :, None], scale[:, :, None])
        # [B, prediction_length, Q] in config quantile order
        return {"quantiles": np.ascontiguousarray(preds.transpose(0, 2, 1)).astype(np.float32)}


def create_backend(
    weights_path, config, device, *, precision: str = DEFAULT_PRECISION, device_options: dict | None = None
):
    """Build a Backend on a caller-owned TTNN device.

    config: a dict, a path to config.json, a Chronos2Config, or None (searched
    next to weights_path). precision: "fp32" (default), "bf16" or "bfp8_b"; see
    PRECISION_POLICIES. device_options: the options `device` was opened with
    (defaults to DEVICE_OPTIONS); trace_region_size 0 runs without trace replay.
    Setting CHRONOS2_TT_TRACE=0 also disables trace replay.
    """
    cfg = resolve_config(config, weights_path)
    return Backend(weights_path, cfg, device, precision, device_options)
