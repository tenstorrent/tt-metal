# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the TiRex-2 device PCC tests."""

from __future__ import annotations

import functools

import torch

from models.experimental.tirex2_forecast.reference import dataflow as df
from models.experimental.tirex2_forecast.tt.common import pad_to


@functools.lru_cache(maxsize=1)
def host_weights() -> df.TiRex2W:
    return df.build_weights(df.load_state_dict(df.default_checkpoint()))


def pcc(expected: torch.Tensor, got: torch.Tensor) -> float:
    from tests.ttnn.utils_for_testing import comp_pcc

    return float(comp_pcc(expected.float(), got.float(), 0.0)[1])


def report(name: str, expected: torch.Tensor, got: torch.Tensor, floor: float) -> float:
    p = pcc(expected, got)
    e, g = expected.double(), got.double()
    rel = float((e - g).abs().max() / e.abs().max().clamp_min(1e-30))
    print(f"[PCC] {name}: pcc {p:.6f} max rel err {rel:.3e} (floor {floor})")
    return p


def to_device_seq(device, x: torch.Tensor, seq: int, dtype=None):
    """``[rows, tokens, D]`` -> device ``[pad32(rows), 1, seq, D]`` (zero rows / tokens appended)."""
    from models.experimental.tirex2_forecast.tt.common import upload

    rows, tokens, d = x.shape
    buf = torch.zeros(pad_to(rows), 1, seq, d)
    buf[:rows, 0, :tokens] = x
    return upload(device, buf, dtype=dtype)


def from_device_seq(t, rows: int, tokens: int) -> torch.Tensor:
    import ttnn

    return ttnn.to_torch(t).float()[:rows, 0, :tokens]
