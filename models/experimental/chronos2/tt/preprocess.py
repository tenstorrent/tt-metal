# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0
#
# Host-side preprocessing (FP32), replicating chronos-forecasting's
# Chronos2Dataset(TEST) + Chronos2Model.prepare semantics exactly: instance
# scaling (nanmean/nanstd with the reference guards), optional arcsinh,
# left-NaN-padding + non-overlapping patching, time encoding, and the token
# attention mask. No learned values are used here.

from __future__ import annotations

import numpy as np

from .model_config import Chronos2Config


class Preprocessor:
    """FP32 preprocessing identical to Chronos2Dataset(TEST) + Chronos2Model.prepare.

    Callers pass 1.0/0.0 observed masks (values at masked cells are ignored); the
    reference pipeline maps masked cells to NaN internally, so we do the same here.
    Inputs are never mutated.
    """

    def __init__(self, cfg: Chronos2Config):
        self.cfg = cfg

    # -- scaling ---------------------------------------------------------- #

    def fit_scale(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """loc = nanmean, scale = nanstd with the reference's NaN/zero guards (FP32)."""
        xf = x.astype(np.float32)
        valid = np.isfinite(xf)
        n = valid.sum(axis=-1, keepdims=True).astype(np.float32)
        loc = np.where(n > 0, np.nansum(xf, axis=-1, keepdims=True) / np.maximum(n, 1.0), 0.0)
        diff = np.where(valid, xf - loc, 0.0).astype(np.float32)
        var = np.where(n > 0, (diff * diff).sum(axis=-1, keepdims=True) / np.maximum(n, 1.0), np.nan)
        scale = np.sqrt(var).astype(np.float32)
        scale = np.where(np.isnan(scale), np.float32(1.0), scale)  # all-NaN row -> 1.0
        scale = np.where(scale == 0, np.float32(1e-5), scale)  # constant row -> eps
        return loc.astype(np.float32), scale.astype(np.float32)

    def apply_scale(self, x: np.ndarray, loc: np.ndarray, scale: np.ndarray) -> np.ndarray:
        z = (x.astype(np.float32) - loc) / scale
        if self.cfg.use_arcsinh:
            z = np.arcsinh(z).astype(np.float32)
        return z

    def unscale(self, y: np.ndarray, loc: np.ndarray, scale: np.ndarray) -> np.ndarray:
        y = y.astype(np.float32)
        if self.cfg.use_arcsinh:
            y = np.sinh(y).astype(np.float32)
        return (y * scale + loc).astype(np.float32)

    # -- patching --------------------------------------------------------- #

    def _patch(self, x: np.ndarray, patch_size: int) -> np.ndarray:
        """Left-NaN-pad to a multiple of patch_size, then non-overlapping unfold."""
        length = x.shape[-1]
        rem = length % patch_size
        if rem != 0:
            pad = np.full((*x.shape[:-1], patch_size - rem), np.nan, dtype=x.dtype)
            x = np.concatenate([pad, x], axis=-1)
        n = x.shape[-1] // patch_size
        return x.reshape(*x.shape[:-1], n, patch_size)

    # -- full preparation -------------------------------------------------- #

    def prepare(
        self,
        past_values: np.ndarray,
        past_observed_mask: np.ndarray,
        num_output_patches: int,
    ) -> dict:
        """Returns FP32 host features + masks for one forecast call.

        Outputs:
          ctx_features   [B, n_ctx_patches, 3*patch]  (time_enc, value, mask)
          fut_features   [B, n_out, 3*patch]
          attn_mask      [B, n_ctx_patches + reg + n_out] float32 (1 = attend)
          loc, scale     [B, 1]
        """
        cfg = self.cfg
        values = np.array(past_values, dtype=np.float32, copy=True)
        mask = np.array(past_observed_mask, dtype=np.float32, copy=True)
        if values.shape != mask.shape or values.ndim != 2:
            raise ValueError(f"expected 2-D [B,T] values/mask of equal shape, got {values.shape}/{mask.shape}")

        # internal missing-value semantics: masked cells -> NaN
        x = np.where(mask > 0.0, values, np.float32(np.nan)).astype(np.float32)

        # truncate to the model's context window (keeps the most recent points)
        if x.shape[-1] > cfg.context_length:
            x = x[..., -cfg.context_length :]

        # scaling in FP32 on the (unpadded) context
        loc, scale = self.fit_scale(x)
        z = self.apply_scale(x, loc, scale)

        # batch left-pad to the longest row (NaN), as left_pad_and_cat_2D does
        max_len = x.shape[-1]
        # rows are already equal-length here ([B,T] rectangular input); kept explicit:
        if z.shape[-1] < max_len:  # pragma: no cover - defensive
            padw = max_len - z.shape[-1]
            z = np.concatenate([np.full((*z.shape[:-1], padw), np.nan, np.float32), z], axis=-1)

        p = cfg.input_patch_size
        patched_z = self._patch(z, p)  # [B, n, p], NaN pads
        patched_m = np.nan_to_num(self._patch(mask[..., -z.shape[-1] :], p), nan=0.0)  # [B, n, p]
        patched_z = np.where(patched_m > 0.0, patched_z, np.float32(0.0))
        patched_z = np.nan_to_num(patched_z, nan=0.0)  # fully-masked patches -> 0
        attn_ctx = (patched_m.sum(axis=-1) > 0).astype(np.float32)  # [B, n]

        n_ctx_patches = attn_ctx.shape[-1]
        tes = np.float32(cfg.time_encoding_scale)
        t_ctx = np.tile(
            (np.arange(-n_ctx_patches * p, 0, dtype=np.float32) / tes).reshape(1, n_ctx_patches, p),
            (x.shape[0], 1, 1),
        )
        ctx_features = np.concatenate([t_ctx, patched_z, patched_m], axis=-1).astype(np.float32)

        n_out = num_output_patches
        t_fut = np.tile(
            (np.arange(0, n_out * cfg.output_patch_size, dtype=np.float32) / tes).reshape(
                1, n_out, cfg.output_patch_size
            ),
            (x.shape[0], 1, 1),
        )
        fut_features = np.concatenate(
            [
                t_fut,
                np.zeros((x.shape[0], n_out, cfg.output_patch_size), np.float32),
                np.zeros((x.shape[0], n_out, cfg.output_patch_size), np.float32),
            ],
            axis=-1,
        ).astype(np.float32)

        ones_out = np.ones((x.shape[0], n_out), np.float32)
        reg = np.ones((x.shape[0], 1), np.float32) if cfg.use_reg_token else np.ones((x.shape[0], 0), np.float32)
        attn_mask = np.concatenate([attn_ctx, reg, ones_out], axis=-1).astype(np.float32)
        return {
            "ctx_features": ctx_features,
            "fut_features": fut_features,
            "attn_mask": attn_mask,
            "loc": loc,
            "scale": scale,
            "n_ctx_patches": n_ctx_patches,
            "num_output_patches": n_out,
        }
