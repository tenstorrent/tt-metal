# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN Kokoro ``Decoder`` ``encode`` + ``decode`` stack (``AdainResBlk1d``), matching ``Decoder.forward``."""

from __future__ import annotations

from typing import Any, List

import ttnn

from .ttnn_adain_resblk_encode import AdainResBlk1d, infer_encode_dims, preprocess_encode_parameters


def preprocess_kokoro_decoder_body_parameters(decoder: Any, device) -> dict[str, Any]:
    """
    Preprocess ``decoder.encode`` and ``decoder.decode`` for :class:`KokoroDecoderBody`.

    Removes weight norm on those submodules in place (same as ``preprocess_encode_parameters``).
    """
    enc = decoder.encode
    di, do, sd = infer_encode_dims(enc)
    enc_entry = {"params": preprocess_encode_parameters(enc, device), "dim_in": di, "dim_out": do, "style_dim": sd}

    dec_entries: list[dict[str, Any]] = []
    for i in range(len(decoder.decode)):
        blk = decoder.decode[i]
        di, do, sd = infer_encode_dims(blk)
        dec_entries.append(
            {"params": preprocess_encode_parameters(blk, device), "dim_in": di, "dim_out": do, "style_dim": sd}
        )

    return {"encode": enc_entry, "decode": dec_entries}


class KokoroDecoderBody:
    """
    Runs ``encode`` then ``decode`` ModuleList with the same concat pattern as ``Decoder.forward``.

    Inputs are ``(B, C, T)`` float32 TILE except ``s`` is ``(B, style_dim)`` TILE.
    """

    def __init__(self, device, body_parameters: dict[str, Any]):
        self.device = device
        enc = body_parameters["encode"]
        self.encode = AdainResBlk1d(
            device,
            enc["params"],
            int(enc["dim_in"]),
            int(enc["dim_out"]),
            int(enc["style_dim"]),
        )
        self.decode_blocks: List[AdainResBlk1d] = []
        self._decode_sets_res_false_after: List[bool] = []
        for d in body_parameters["decode"]:
            self.decode_blocks.append(
                AdainResBlk1d(
                    device,
                    d["params"],
                    int(d["dim_in"]),
                    int(d["dim_out"]),
                    int(d["style_dim"]),
                )
            )
            upsample = bool(d["params"].get("upsample_nearest", False)) or d["params"].get("pool") is not None
            self._decode_sets_res_false_after.append(upsample)

    def __call__(
        self,
        x_asr_f0_n: ttnn.Tensor,
        s: ttnn.Tensor,
        asr_res: ttnn.Tensor,
        f0: ttnn.Tensor,
        n_b: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """
        Args:
            x_asr_f0_n: ``torch.cat([asr, F0, N], dim=1)`` layout ``(B, 514, T)``.
            s: Style ``(B, 128)``.
            asr_res: ``(B, 64, T)``.
            f0: ``(B, 1, T)`` after ``F0_conv``.
            n_b: ``(B, 1, T)`` after ``N_conv``.

        Returns:
            ``x`` after the last decode block ``(B, 512, 2*T)`` when the last block upsamples.
        """
        x = self.encode(x_asr_f0_n, s)
        res = True
        for block, sets_res_false in zip(self.decode_blocks, self._decode_sets_res_false_after):
            if res:
                x = ttnn.concat([x, asr_res, f0, n_b], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
            x = block(x, s)
            if sets_res_false:
                res = False
        return x
