# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Price every way of doing a Speaker-Encoder conv tap.

`_conv1d_device_nlc` turns each reflect-padded dilated conv1d into im2col + one
matmul, and each im2col tap is currently a matmul against a one-hot row selector
(`_tap_by_matmul`). In the traced ECAPA window that is ~46 matmuls and ~470 us —
**29 % of the whole encoder** — spent moving data, against ~167 us for the
convolutions those taps feed:

    entry TDNN  k=5 d=1  C=128   4 shifted taps x 13 us  = 52 us   (conv 384x640x512 = 20 us)
    3 blocks x 7 scales  C=64    2 shifted taps x 10 us  = 20 us   (conv 384x192x64  =  7 us)

`_tap_by_matmul` picked HiFi4 deliberately, so the one-hot copy is bit-exact. The
question this probe asks is whether HiFi4 is the CHEAPEST exact option: on Wormhole
the fidelity setting is a pass count, so HiFi2 is half the math of HiFi4, and a
one-hot matrix might not need HiFi4's mantissa coverage at all.

It also re-prices `slice` and `gather` (both already reachable via
QWEN3_TTS_SE_CONV_SHIFT), and a hand-tuned program config: the shipped call is a
bare `ttnn.matmul`, and auto-routing puts a [384,384] x [384,64] on TWELVE of 64
cores because N is only 2 tiles.

Every arm is scored for BIT-EXACTNESS against torch's own `x[rows]`, because a tap
is a copy and anything less than exact is a bug, not a tradeoff.

    python -m tracy -p -v -r --op-support-count 100000 \
        -m pytest -s -q models/demos/qwen3_tts/tests/test_qwen3_tts_se_tap_probe.py
    python models/demos/qwen3_tts/tests/se_tap_probe_report.py
"""

from __future__ import annotations

import json
import os

import pytest
import torch
import torch.nn.functional as F

import ttnn

TILE = 32
L = 384  # mel T=376 padded to a tile multiple
REPS = 5
MANIFEST = "generated/se_tap_manifest.json"

# (label, channels, kernel, dilation) — every tap shape the traced encoder runs.
CASES = [
    ("res2net d2", 64, 3, 2),
    ("res2net d3", 64, 3, 3),
    ("res2net d4", 64, 3, 4),
    ("entry tdnn", 128, 5, 1),
]
_ARMS: list = []


@pytest.fixture(scope="module")
def device():
    d = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=32768)
    d.enable_program_cache()
    yield d
    ttnn.close_mesh_device(d)
    os.makedirs("generated", exist_ok=True)
    with open(MANIFEST, "w") as f:
        json.dump({"L": L, "reps": REPS, "arms": _ARMS}, f, indent=2)
    print(f"\nwrote {MANIFEST}: {len(_ARMS)} arms x {REPS} reps")


def _tap_rows(length: int, kernel: int, dilation: int):
    """Exactly `_reflect_tap_rows`: the row each output position reads, per tap."""
    pad_total = dilation * (kernel - 1)
    pad_left = pad_total // 2
    idx = (
        F.pad(
            torch.arange(length, dtype=torch.float32).view(1, 1, length),
            (pad_left, pad_total - pad_left),
            mode="reflect",
        )
        .view(-1)
        .long()
    )
    return [idx[j * dilation : j * dilation + length] for j in range(kernel)]


def _row_runs(rows):
    runs, start = [], 0
    for i in range(1, len(rows) + 1):
        if i == len(rows) or int(rows[i]) != int(rows[i - 1]) + 1:
            runs.append((int(rows[start]), int(rows[i - 1]) + 1))
            start = i
    return runs


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_se_tap_alternatives(device, case):
    label, C, k, dil = case
    cg = device.compute_with_storage_grid_size()
    torch.manual_seed(0)
    x = torch.randn(1, L, C, dtype=torch.bfloat16)
    ident = torch.arange(L)
    rows = next(r for r in _tap_rows(L, k, dil) if not torch.equal(r, ident))
    ref = x[0][rows].clone()  # ground truth: torch's own gather
    nruns = len(_row_runs(rows))
    print(f"\n### {label}: C={C} k={k} d={dil}  tap has {nruns} ascending run(s)")

    mc = ttnn.L1_MEMORY_CONFIG
    x_tt = ttnn.from_torch(x, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=mc)
    x_4d = ttnn.reshape(x_tt, (1, 1, L, C))

    perm = torch.zeros(1, 1, L, L, dtype=torch.float32)
    perm[0, 0, torch.arange(L), rows.long()] = 1.0
    p_tt = ttnn.from_torch(
        perm, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    def _kc(fid):
        return ttnn.init_device_compute_kernel_config(device.arch(), math_fidelity=fid)

    def _mm(kc, pc=None):
        return lambda: ttnn.reshape(
            ttnn.matmul(p_tt, x_4d, memory_config=mc, compute_kernel_config=kc, program_config=pc), (1, L, C)
        )

    def _slice():
        runs = _row_runs(rows)
        pieces = [ttnn.slice(x_tt, [0, a, 0], [1, b, C], memory_config=mc) for a, b in runs]
        return pieces[0] if len(pieces) == 1 else ttnn.concat(pieces, dim=1, memory_config=mc)

    idx_tt = ttnn.from_torch(
        rows.view(1, -1, 1).expand(1, L, C).contiguous().to(torch.int32),
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=mc,
    )

    # A 2D config that puts more than 12 cores on it: gy | m_tiles and gy | k_tiles,
    # gx | n_tiles. n_tiles is 2 for C=64, so gx <= 2 and the ceiling really is 12 —
    # this arm asks the question rather than assuming the auto-route was wrong.
    from models.demos.qwen3_tts.tt.linear_1d_program_config import find_2d_mcast_grid, make_linear_2d_program_config

    g2x, g2y = find_2d_mcast_grid(L, L, C, cg.x, cg.y)
    pc2 = make_linear_2d_program_config(L, L, C, g2x, g2y, False)

    arms = [
        ("matmul HiFi4 (shipped)", _mm(_kc(ttnn.MathFidelity.HiFi4))),
        ("matmul HiFi3", _mm(_kc(ttnn.MathFidelity.HiFi3))),
        ("matmul HiFi2", _mm(_kc(ttnn.MathFidelity.HiFi2))),
        ("matmul LoFi", _mm(_kc(ttnn.MathFidelity.LoFi))),
        (f"matmul HiFi2 + 2D {g2x}x{g2y}", _mm(_kc(ttnn.MathFidelity.HiFi2), pc2)),
        ("slice + concat", _slice),
        ("gather", lambda: ttnn.gather(x_tt, dim=1, index=idx_tt)),
    ]

    for name, fn in arms:
        try:
            probe = fn()
            ttnn.synchronize_device(device)
            got = ttnn.to_torch(probe).reshape(L, C)
            exact = bool(torch.equal(got.to(torch.float32), ref.to(torch.float32)))
            maxd = float((got.to(torch.float32) - ref.to(torch.float32)).abs().max())
            ttnn.deallocate(probe)
        except Exception as e:
            why = next((l.strip() for l in str(e).splitlines() if "TT_FATAL" in l or "must" in l), str(e)[:100])
            print(f"  {name:30s} -> REFUSED {why[:100]}")
            continue
        for _ in range(REPS):
            o = fn()
            ttnn.synchronize_device(device)
            ttnn.deallocate(o)
        _ARMS.append(
            {
                "tag": f"{label}:{name}",
                "case": label,
                "C": C,
                "k": k,
                "dil": dil,
                "arm": name,
                "reps": REPS,
                "exact": exact,
                "max_diff": maxd,
                "shipped": name.endswith("(shipped)"),
            }
        )
        print(f"  {name:30s} -> {'BIT-EXACT' if exact else f'DIFFERS max={maxd:.3e}'}")

    ttnn.deallocate(x_tt)
    ttnn.deallocate(p_tt)
    ttnn.deallocate(idx_tt)
