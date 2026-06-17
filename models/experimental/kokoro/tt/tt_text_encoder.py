# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of Kokoro ``TextEncoder`` (embedding + Conv1d/LayerNorm + BiLSTM).

Reference: ``models.experimental.kokoro.reference.modules.TextEncoder``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn

import ttnn

from .tt_conv import TTConv1dParams, tt_conv1d_nlc, tt_weight_norm_materialize
from .tt_lstm import TTLSTMParams, build_fused_recurrent_weight, preprocess_tt_lstm_1layer, tt_bilstm_nlc


@dataclass(frozen=True)
class TTTextEncoderConvLNBlockParams:
    """Weights for one Conv1d + channel LayerNorm stage (dropout omitted at inference)."""

    conv: TTConv1dParams
    ln_weight: ttnn.Tensor
    ln_bias: ttnn.Tensor


@dataclass(frozen=True)
class TTTextEncoderParams:
    """Preprocessed TTNN parameters for :class:`TTTextEncoder`."""

    embedding_weight: ttnn.Tensor
    blocks: tuple[TTTextEncoderConvLNBlockParams, ...]
    lstm_fwd: TTLSTMParams
    lstm_rev: TTLSTMParams
    # Block-diagonal recurrent weight fusing both BiLSTM directions into one matmul/step
    # (None for a unidirectional LSTM). Halves per-step matmul/activation/elementwise ops on
    # the unpadded path; bit-exact at bf16 state (see ``build_fused_recurrent_weight``).
    lstm_w_h_block: Optional[ttnn.Tensor] = None
    ln_eps: float = 1e-5


def preprocess_tt_text_encoder(
    text_encoder: nn.Module,
    device: ttnn.Device,
    *,
    weights_dtype=ttnn.bfloat16,
) -> TTTextEncoderParams:
    """Upload PyTorch ``TextEncoder`` weights to device for :class:`TTTextEncoder`."""
    emb_w = ttnn.from_torch(
        text_encoder.embedding.weight.detach().cpu(),
        dtype=ttnn.bfloat16,  # ttnn.embedding requires BF16 weights on device
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    block_params: list[TTTextEncoderConvLNBlockParams] = []

    for block in text_encoder.cnn:
        conv = block[0]
        ln = block[1]

        if hasattr(conv, "weight_v") and hasattr(conv, "weight_g"):
            w = tt_weight_norm_materialize(conv.weight_v.detach().cpu(), conv.weight_g.detach().cpu())
        else:
            # PyTorch ``nn.utils.parametrizations.weight_norm`` exposes materialized ``weight``.
            w = conv.weight.detach().cpu()
        b = conv.bias.detach().cpu() if conv.bias is not None else None

        w_tt = ttnn.from_torch(w, dtype=weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        b_tt = (
            ttnn.from_torch(b.reshape(1, 1, 1, -1), dtype=weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
            if b is not None
            else None
        )
        conv_p = TTConv1dParams(
            weight=w_tt,
            bias=b_tt,
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size[0],
            stride=conv.stride[0],
            padding=conv.padding[0],
            groups=conv.groups,
        )

        ln_w = ttnn.from_torch(
            ln.gamma.detach().cpu(),
            dtype=weights_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ln_b = ttnn.from_torch(
            ln.beta.detach().cpu(),
            dtype=weights_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        block_params.append(TTTextEncoderConvLNBlockParams(conv=conv_p, ln_weight=ln_w, ln_bias=ln_b))

    fwd, rev = preprocess_tt_lstm_1layer(text_encoder.lstm, device, weights_dtype=weights_dtype)
    assert rev is not None, "TextEncoder expects a bidirectional LSTM"
    lstm_w_h_block = build_fused_recurrent_weight(text_encoder.lstm, device, weights_dtype=weights_dtype)

    return TTTextEncoderParams(
        embedding_weight=emb_w,
        blocks=tuple(block_params),
        lstm_fwd=fwd,
        lstm_rev=rev,
        lstm_w_h_block=lstm_w_h_block,
        ln_eps=1e-5,
    )


def _mask_keep_nlc(text_mask: torch.Tensor, *, device: ttnn.Device) -> ttnn.Tensor:
    """``text_mask[b,t] == True`` means padded / ignored (same as reference ``masked_fill_``)."""
    keep = (~text_mask).to(torch.float32).unsqueeze(-1)
    return ttnn.from_torch(keep, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def _maybe_interleaved(x: ttnn.Tensor) -> ttnn.Tensor:
    if ttnn.is_tensor_storage_on_device(x):
        try:
            return ttnn.sharded_to_interleaved(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        except Exception:
            return x
    return x


class TTTextEncoderConvLNBlock:
    """One depth-wise CNN stack: Conv1d → LayerNorm → LeakyReLU (matches reference ``Sequential``)."""

    def __init__(
        self,
        *,
        device: ttnn.Device,
        params: TTTextEncoderConvLNBlockParams,
        ln_eps: float,
        compute_kernel_config,
    ) -> None:
        self.device = device
        self.params = params
        self.ln_eps = ln_eps
        self.compute_kernel_config = compute_kernel_config

    def forward(self, x_nlc: ttnn.Tensor, mask_keep: ttnn.Tensor) -> ttnn.Tensor:
        x = tt_conv1d_nlc(
            x_nlc=x_nlc,
            params=self.params.conv,
            device=self.device,
            compute_config=self.compute_kernel_config,
        )
        x = _maybe_interleaved(x)
        x = ttnn.layer_norm(
            x,
            weight=self.params.ln_weight,
            bias=self.params.ln_bias,
            epsilon=self.ln_eps,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x = ttnn.leaky_relu(x, negative_slope=0.2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.multiply(x, mask_keep, memory_config=ttnn.DRAM_MEMORY_CONFIG)


class TTTextEncoder:
    """TTNN ``TextEncoder``: embedding → masked CNN stages → packed-length BiLSTM → ``[B, C, T]``."""

    def __init__(self, device: ttnn.Device, params: TTTextEncoderParams) -> None:
        self.device = device
        self.params = params
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi3,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )
        self._cnn_blocks = tuple(
            TTTextEncoderConvLNBlock(
                device=device,
                params=bp,
                ln_eps=params.ln_eps,
                compute_kernel_config=self.compute_kernel_config,
            )
            for bp in params.blocks
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        input_lengths: torch.LongTensor,
        text_mask: Optional[torch.Tensor] = None,
        *,
        mask_keep_float: Optional[torch.Tensor] = None,
    ) -> ttnn.Tensor:
        """
        Args:
            input_ids: ``[B, T]`` token indices (CPU or CUDA tensor; copied to device).
            input_lengths: ``[B]`` valid length per row (CPU long, as in reference).
            text_mask: ``[B, T]`` bool, ``True`` where positions are masked out.
            mask_keep_float: optional pre-computed ``[B, T, 1]`` float32 keep mask
                (``1.0`` where real, ``0.0`` where padded). When provided, the mask is
                uploaded directly with no torch ops inside forward. When ``None``,
                computed from ``text_mask`` for backward compatibility.

        Returns:
            TTNN tensor ``[B, C, T]`` (channels = ``2 * lstm_hidden``), layout TILE.
        """
        dev = self.device
        B, T = input_ids.shape

        tt_ids = ttnn.from_torch(
            input_ids,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x = ttnn.embedding(tt_ids, self.params.embedding_weight, layout=ttnn.TILE_LAYOUT)
        ttnn.deallocate(tt_ids)

        if mask_keep_float is not None:
            mask_keep = ttnn.from_torch(
                mask_keep_float,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        elif text_mask is not None:
            mask_keep = _mask_keep_nlc(text_mask, device=dev)
        else:
            # Full-length sequence (no padding): keep_mask is all-ones — create on device directly.
            mask_keep = ttnn.ones(
                [B, T, 1],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        x = ttnn.multiply(x, mask_keep, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        for blk in self._cnn_blocks:
            x = blk.forward(x, mask_keep)

        lengths_list: Sequence[int] = input_lengths.detach().cpu().tolist()
        x = tt_bilstm_nlc(
            x_nlc=x,
            fwd=self.params.lstm_fwd,
            rev=self.params.lstm_rev,
            compute_kernel_config=self.compute_kernel_config,
            sequence_lengths=lengths_list,
            w_h_block=self.params.lstm_w_h_block,
            # TextEncoder is the one LSTM that tolerates the gate-sum rounding change; fold the
            # per-step gates_x add into the recurrent matmul bias (one fewer BinaryNg/step).
            fold_gates_bias=True,
        )

        x = ttnn.multiply(x, mask_keep, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(mask_keep)

        return ttnn.permute(x, (0, 2, 1))

    __call__ = forward
