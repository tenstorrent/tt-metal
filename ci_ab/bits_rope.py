# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CI-only bits sweep of the fused rope (copied into python_tests by ci_ab/run_hr.sh, not part of the PR): every bf16
pattern as x_even and as x_odd, against random and special cos/sin, for the 2x2 32-row shape (16-bit DEST) and a 1x2
32-row shape (32-bit DEST), with and without a scale and per-row phases. Saves the device output bits per test to
$BITS_OUT/$BITS_TAG/<id>.pt; ci_ab/bits_cmp.py compares two tags bit for bit."""

import os

import torch
from helpers.llk_params import DestAccumulation
from helpers.param_config import parametrize
from test_rope import TILE_ROWS, _bf16_bits, _dest_tiles, _run

SPECIAL = [0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC0, 0xFFC1, 0x0001, 0x8001, 0x007F, 0x3F80, 0xBF80, 0x7F7F, 0xFF7F, 0x0080]


def _dest(geometry, tiles, chunk, seed):
    g = torch.Generator().manual_seed(seed)
    bits = torch.randint(0, 1 << 16, (tiles * TILE_ROWS, 16), generator=g, dtype=torch.int32)
    # cos/sin rows: random bf16 in [-1, 1] with special patterns sprinkled in
    cs = torch.empty((tiles * TILE_ROWS, 16)).uniform_(-1.0, 1.0, generator=g).to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
    sp = torch.tensor(SPECIAL, dtype=torch.int32)
    pick = torch.randint(0, len(SPECIAL), cs.shape, generator=g)
    mask = torch.rand(cs.shape, generator=g) < 0.05
    cs = torch.where(mask, sp[pick], cs)
    cs_lo = geometry["cos_base"]
    cs_hi = cs_lo + geometry["wt"] * geometry["cs_stride"]
    bits[cs_lo:cs_hi] = cs[cs_lo:cs_hi]
    # x rows: consecutive bf16 patterns, chunk by chunk, even and odd columns offset by half the space
    x_lo = geometry["x_base"]
    x_hi = x_lo + geometry["ht"] * geometry["wt"] * geometry["x_stride"]
    n = (x_hi - x_lo) * 16
    pat = (torch.arange(n, dtype=torch.int32) + chunk * n) & 0xFFFF
    pat = pat.reshape(x_hi - x_lo, 16)
    pat[:, 1::2] = (pat[:, 1::2] + 0x8000 + 0x1235) & 0xFFFF
    bits[x_lo:x_hi] = pat
    return (bits.to(torch.int16)).view(torch.bfloat16)


CASES = []
for per_row in (False, True):
    for scale in (None, 0.7):
        CASES += [("bf16", 2, 2, per_row, scale, c) for c in range(16)]
        CASES += [("fp32", 1, 2, per_row, scale, c) for c in range(0, 32, 4)]


@parametrize(case=CASES)
def test_rope_bits(case):
    if len(case) == 1:
        (case,) = case
    mode, ht, wt, per_row, scale, chunk = case
    geometry = dict(ht=ht, wt=wt, x_base=0, x_stride=64, cos_base=ht * wt * 64, sin_base=ht * wt * 64, cs_stride=64)
    tiles = _dest_tiles(geometry)
    dest_acc = DestAccumulation.Yes if mode == "fp32" else DestAccumulation.No
    dest = _dest(geometry, tiles, chunk, seed=1000 + chunk)
    device = _run(
        geometry,
        tiles,
        dest,
        scale_fp32=None if scale is None else _bf16_bits(scale),
        fused_cos_sin=True,
        tile_h=32,
        cos_sin_per_row=per_row,
        dest_acc=dest_acc,
    )
    out = os.path.join(os.environ["BITS_OUT"], os.environ["BITS_TAG"])
    os.makedirs(out, exist_ok=True)
    name = f"{mode}_h{ht}w{wt}_pr{int(per_row)}_s{scale}_c{chunk}"
    torch.save({"in": dest.view(torch.int16).clone(), "out": device.view(torch.int16 if mode == "bf16" else torch.int32).clone()}, os.path.join(out, name + ".pt"))
