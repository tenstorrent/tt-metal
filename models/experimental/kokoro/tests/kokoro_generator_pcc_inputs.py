# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared decoder-prefix tensors for Kokoro generator PCC tests (TTNN front + body path)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import ttnn

from models.experimental.kokoro.tt import KokoroDecoderBody, KokoroDecoderFront


def decoder_tensors_for_generator(
    dec,
    batch: int,
    time_asr: int,
    seed: int,
    *,
    tt_front: KokoroDecoderFront | None = None,
    tt_body: KokoroDecoderBody | None = None,
    ttnn_device=None,
):
    """Match ``Decoder.forward`` tensor sizes up to the generator call.

    When ``tt_front``, ``tt_body``, and ``ttnn_device`` are set, the full prefix through decode runs on TTNN.

    When only ``tt_front`` and ``ttnn_device`` are set, ``encode``/``decode`` stay PyTorch.
    """
    torch.manual_seed(seed)
    dim_in = dec.asr_res[0].in_channels
    tf = 2 * time_asr
    asr = torch.randn(batch, dim_in, time_asr, dtype=torch.float32)
    f0_curve = torch.randn(batch, tf, dtype=torch.float32) * 100.0 + 120.0
    n = torch.randn(batch, tf, dtype=torch.float32)
    s = torch.randn(batch, 128, dtype=torch.float32)
    l1 = ttnn.L1_MEMORY_CONFIG

    with torch.no_grad():
        if tt_front is not None and tt_body is not None and ttnn_device is not None:
            asr_tt = ttnn.from_torch(
                asr,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=ttnn_device,
                memory_config=l1,
            )
            f0_in = ttnn.from_torch(
                f0_curve.unsqueeze(1),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=ttnn_device,
                memory_config=l1,
            )
            f0_tt = tt_front.f0_conv(f0_in, batch, tf)
            ttnn.deallocate(f0_in)
            n_in = ttnn.from_torch(
                n.unsqueeze(1),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=ttnn_device,
                memory_config=l1,
            )
            n_tt = tt_front.n_conv(n_in, batch, tf)
            ttnn.deallocate(n_in)
            asr_res_tt = tt_front.asr_res(asr_tt, batch, time_asr)
            x0_tt = ttnn.concat([asr_tt, f0_tt, n_tt], dim=1, memory_config=l1)
            s_tt = ttnn.from_torch(
                s,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=ttnn_device,
                memory_config=l1,
            )
            x_tt = tt_body(x0_tt, s_tt, asr_res_tt, f0_tt, n_tt)
            x = ttnn.to_torch(x_tt).reshape(batch, 512, 2 * time_asr)
        elif tt_front is not None and ttnn_device is not None:
            f0_in = ttnn.from_torch(
                f0_curve.unsqueeze(1),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=ttnn_device,
                memory_config=l1,
            )
            f0_tt = tt_front.f0_conv(f0_in, batch, tf)
            f0 = ttnn.to_torch(f0_tt).reshape(batch, 1, time_asr)
            ttnn.deallocate(f0_in)
            ttnn.deallocate(f0_tt)

            n_in = ttnn.from_torch(
                n.unsqueeze(1),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=ttnn_device,
                memory_config=l1,
            )
            n_tt = tt_front.n_conv(n_in, batch, tf)
            n_b = ttnn.to_torch(n_tt).reshape(batch, 1, time_asr)
            ttnn.deallocate(n_in)
            ttnn.deallocate(n_tt)

            asr_in = ttnn.from_torch(
                asr,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=ttnn_device,
                memory_config=l1,
            )
            asr_tt = tt_front.asr_res(asr_in, batch, time_asr)
            asr_res = ttnn.to_torch(asr_tt).reshape(batch, 64, time_asr)
            ttnn.deallocate(asr_in)
            ttnn.deallocate(asr_tt)

            x = torch.cat([asr, f0, n_b], dim=1)
            x = dec.encode(x, s)
            res = True
            for block in dec.decode:
                if res:
                    x = torch.cat([x, asr_res, f0, n_b], dim=1)
                x = block(x, s)
                if block.upsample_type != "none":
                    res = False
        else:
            f0 = dec.F0_conv(f0_curve.unsqueeze(1))
            n_b = dec.N_conv(n.unsqueeze(1))
            asr_res = dec.asr_res(asr)

            x = torch.cat([asr, f0, n_b], dim=1)
            x = dec.encode(x, s)
            res = True
            for block in dec.decode:
                if res:
                    x = torch.cat([x, asr_res, f0, n_b], dim=1)
                x = block(x, s)
                if block.upsample_type != "none":
                    res = False
    return x, s, f0_curve
