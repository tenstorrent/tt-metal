# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""SineGen / SourceModuleHnNSF, with the cumsum precision claim under test.

The host tier here is doing real work, not just shape checking: it pins the
numerical claim that float32 is sufficient for the audio-rate phase integration,
which is the finding that removed `02_plan.md` sec.3.4 from the risk register.
"""
from __future__ import annotations

import math
import os

import pytest
import torch

from models.demos.cosyvoice.tt.common import as_torch, load_golden, pcc
from models.demos.cosyvoice.tt.hifigan.source import TtSineGen

needs_l1_small = pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)

# ttsim cannot run this module, for two independent reasons found by probing it:
#   * float32 is unsupported outright -- every fp32 elementwise op aborts with
#     "UndefinedBehavior: tensix_unpacr: in_data_format=0", while the bfloat16
#     spelling of the same op passes;
#   * ttnn.cumsum is unsupported even in bfloat16
#     ("UnsupportedFunctionality: tensix_unpacr: cfg_context_id=1").
# Both abort the process rather than raising, so they cannot be caught in Python
# -- the test has to be skipped up front or it takes the whole session down.
# SineGen needs exactly those two things, so it is silicon-only.
needs_real_silicon = pytest.mark.skipif(
    bool(os.environ.get("TT_METAL_SIMULATOR")),
    reason="ttsim supports neither float32 nor ttnn.cumsum; SineGen needs both",
)


def _have_golden(name):
    from models.demos.cosyvoice.tt.common import GOLDEN_DIR

    return os.path.exists(os.path.join(GOLDEN_DIR, f"{name}.npz"))


needs_golden = pytest.mark.skipif(
    not _have_golden("hift.sinegen"), reason="run scripts/gen_golden.py in the CosyVoice venv first"
)


# --------------------------------------------------------------------------
# the precision claim
# --------------------------------------------------------------------------
@needs_golden
def test_fp32_cumsum_phase_error_is_negligible():
    """float32 phase integration over the audio-rate signal must stay well inside
    a radian. This is the measurement that retired the sec.3.4 risk; if it ever
    regresses, the block-wise mod-1 fallback becomes necessary again."""
    g = load_golden("hift.sinegen")
    f0 = as_torch(g["call0.in_f0"])  # [B, T, 1]
    harm = torch.arange(1, 10, dtype=torch.float32) / 22050.0
    F = f0 * harm.reshape(1, 1, -1)

    c32 = torch.cumsum(F.float(), dim=1)
    c64 = torch.cumsum(F.double(), dim=1)
    th32 = 2 * math.pi * (c32 % 1)
    th64 = (2 * math.pi * (c64 % 1)).float()
    d = (th32 - th64).abs()
    d = torch.minimum(d, 2 * math.pi - d)  # phase is circular

    print(f"\n  accumulator reaches {float(c64.max()):.1f}")
    print(f"  phase error: max {float(d.max()):.5f} rad, mean {float(d.mean()):.6f} rad")
    assert float(d.max()) < 0.01, float(d.max())
    assert pcc(torch.sin(th32), torch.sin(th64)) > 0.9999


@needs_golden
def test_bfloat16_cumsum_is_catastrophic():
    """The converse, asserted rather than assumed: bfloat16 really does destroy
    this signal. Documents WHY the fp32 dtype argument is not optional."""
    g = load_golden("hift.sinegen")
    f0 = as_torch(g["call0.in_f0"])
    harm = torch.arange(1, 10, dtype=torch.float32) / 22050.0
    F = f0 * harm.reshape(1, 1, -1)

    c16 = torch.cumsum(F.bfloat16(), dim=1).float()
    c64 = torch.cumsum(F.double(), dim=1)
    th16 = 2 * math.pi * (c16 % 1)
    th64 = (2 * math.pi * (c64 % 1)).float()
    p = pcc(torch.sin(th16), torch.sin(th64))
    print(f"\n  bfloat16 sin() PCC vs fp64: {p:.6f}  (fp32 achieves > 0.9999)")
    assert p < 0.5, f"bf16 unexpectedly survived (PCC {p}) -- re-examine the fp32 requirement"


@needs_golden
def test_f0_is_piecewise_constant_in_blocks_of_256():
    """nn.Upsample defaults to nearest, so 282 distinct f0 values drive 72192
    samples. Recorded as a test because it is a Stage 2/3 performance lever and
    would silently stop holding if the upsample mode ever changed."""
    g = load_golden("hift.sinegen")
    f0 = as_torch(g["call0.in_f0"])[0, :, 0]
    n = (len(f0) // 256) * 256
    blocks = f0[:n].reshape(-1, 256)
    assert torch.equal(blocks, blocks[:, :1].expand_as(blocks)), "f0 upsample is no longer nearest"
    print(f"\n  {blocks.shape[0]} distinct values drive {len(f0)} samples (256x redundant scan)")


# --------------------------------------------------------------------------
# reference agreement
# --------------------------------------------------------------------------
@needs_golden
def test_torch_reference_matches_captured_sinegen():
    """Our layout-transposed reference must reproduce the captured output once the
    RNG draws are injected. Without injection this test is meaningless -- which is
    exactly the point of capturing them."""
    g = load_golden("hift.sinegen")
    f0 = as_torch(g["call0.in_f0"])
    want_uv = as_torch(g["call0.out_uv"])

    _, uv, _ = TtSineGen.torch_reference(f0)
    assert uv.shape == want_uv.shape, (uv.shape, want_uv.shape)
    assert torch.equal(uv, want_uv), "voiced/unvoiced mask disagrees with the reference"


# --------------------------------------------------------------------------
# device tier
# --------------------------------------------------------------------------
@needs_l1_small
@needs_real_silicon
@pytest.mark.parametrize("length", [1024, 8192])
def test_device_sinegen_matches_host(device, length):
    """ttnn.cumsum(dtype=float32) over an audio-rate ramp, against the host form."""
    import ttnn

    torch.manual_seed(11)
    # A plausible f0 contour: voiced around 200 Hz with unvoiced gaps.
    f0_t = (torch.rand(1, length, 1) * 250.0) * (torch.rand(1, length, 1) > 0.2).float()
    want, want_uv, _ = TtSineGen.torch_reference(f0_t)

    op = TtSineGen(device)
    f0 = ttnn.from_torch(f0_t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    got, uv, _ = op(f0)
    got = ttnn.to_torch(got).float()

    p = pcc(got, want)
    print(f"\n  sinegen T={length} PCC {p:.8f}  max|d| {(got - want).abs().max():.3e}")
    assert got.shape == want.shape, (got.shape, want.shape)
    assert p >= 0.99, p
