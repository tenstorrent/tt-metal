# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host only: the FP8 rounding of the scaled-FP8 KVPE latent (``per_token_cast_to_fp8(round_to_nearest=)``).

* ``e4m3_round_host`` / ``quantize_scaled_fp8_host`` (``utils/kv_cache_utils.py``) model the kernel bit for bit
  (the device side of that claim is ``test_cast_to_fp8_rounding_bit_exact`` in the op's unit tests). Here: the
  ``nearest`` model is round-half-away-from-zero on every BF16 value up to 448; ties, negatives, saturation,
  subnormals and zero rows.
* Against decode (``models/experimental/hy4_preview``): the half-step routine of the prefill kernel is the same
  source as decode's append kernel (``HY4A_FP8_NEAREST``), and decode's host emulation (``host_fp8_rows.e4m3``)
  agrees with our truncation everywhere and with our nearest everywhere except exact ties, where it rounds to even.
* The cache carries its rounding: ``MlaKvCache.fp8_rounding`` reaches the op, survives ``dataclasses.replace`` and
  shows in the hand-off descriptor; GLM-5.3 truncates, Hy4 rounds to nearest unless ``HY4_KV_FP8_ROUNDING`` says not.
"""

from __future__ import annotations

import ast
import dataclasses
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import (
    MlaKvCache,
    MlaKvCacheFormat,
    MlaKvCacheGeometry,
    MlaKvFp8Rounding,
    e4m3_round_host,
    quantize_scaled_fp8_host,
)

ROOT = Path(__file__).resolve().parents[4]
PREFILL_KERNEL = (
    ROOT / "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/per_token_cast_to_fp8/device/kernels/compute"
    "/compute_per_token_cast_to_fp8.cpp"
)
DECODE_KERNEL = ROOT / "models/experimental/hy4_preview/fused/layer_attn/kernels/attn_append_compute.cpp"
DECODE_HOST = ROOT / "models/experimental/hy4_preview/fused/layer_attn/tests/host_fp8_rows.py"

NEAREST, TRUNCATE = MlaKvFp8Rounding.NEAREST, MlaKvFp8Rounding.TRUNCATE


def _all_bf16_upto_448(normal_only: bool = True) -> torch.Tensor:
    """Every non-negative finite BF16 value <= 448 (by default only e4m3's normal range, >= 2**-6, where the packer
    is IEEE), then the same negated (as float32)."""
    v = torch.arange(0, 0x7F80, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).float()
    v = v[(v <= 448.0) & ((v >= 2.0**-6) | (not normal_only))]
    return torch.cat([v, -v])


def _step(a: torch.Tensor) -> torch.Tensor:
    """e4m3 step at |a| (float64): 2**(floor(log2 a) - 3), 2**-9 below 2**-6."""
    a = a.double()
    e = torch.floor(torch.log2(a.clamp(min=2.0**-6)))
    return 2.0 ** (e - 3)


def _decode_host_e4m3():
    """Decode's host emulation ``e4m3(v, mode)`` (mode "trunc" or anything else = torch.round), taken from its source:
    the module runs a checkpoint study at import, so the function is compiled on its own."""
    if not DECODE_HOST.is_file():
        pytest.skip(f"no decode host emulation at {DECODE_HOST}")
    tree = ast.parse(DECODE_HOST.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "e4m3")
    scope = {"torch": torch}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(DECODE_HOST), "exec"), scope)
    return scope["e4m3"]


def _cpp_function(path: Path, name: str) -> str:
    """Body of ``inline void name()`` in a kernel source, comments and whitespace removed."""
    src = path.read_text()
    start = src.index(f"inline void {name}()")
    body, depth = [], 0
    for ch in src[src.index("{", start) :]:
        body.append(ch)
        depth += {"{": 1, "}": -1}.get(ch, 0)
        if depth == 0:
            break
    text = re.sub(r"//[^\n]*", "", "".join(body))
    return re.sub(r"\s+", "", text)


# --- the host model ------------------------------------------------------------------------------------------------
def test_nearest_is_round_half_away_on_every_bf16_value():
    v = _all_bf16_upto_448()
    a, step = v.abs().double(), _step(v.abs())
    away = torch.sign(v.double()) * torch.floor(a / step + 0.5) * step
    floor = torch.sign(v.double()) * torch.floor(a / step) * step
    assert torch.equal(e4m3_round_host(v, NEAREST).double(), away)
    assert torch.equal(e4m3_round_host(v, TRUNCATE).double(), floor)
    # PyTorch's float8_e4m3fn cast rounds to nearest EVEN: the two differ on exactly half of the ties
    rne = v.to(torch.float8_e4m3fn).float()
    ties = (a / step - torch.floor(a / step)) == 0.5
    diff = e4m3_round_host(v, NEAREST) != rne
    assert bool((diff <= ties).all()) and int(diff.sum()) * 2 == int(ties.sum())


@pytest.mark.parametrize(
    "q, want_nearest, want_truncate",
    [
        (1.0625, 1.125, 1.0),  # tie between 1.0 and 1.125: away from zero
        (-1.0625, -1.125, -1.0),  # negative tie
        (1.1875, 1.25, 1.125),  # tie whose even neighbour is 1.25: both rules agree
        (1.0624999, 1.0, 1.0),  # just below the tie
        (-1.09375, -1.125, -1.0),  # negative, above the tie
        (15.5, 16.0, 15.0),  # carries into the next binade
        (448.0, 448.0, 448.0),  # saturation: the largest finite e4m3 stays
        (440.0, 448.0, 416.0),  # rounds up to 448, not past it
        (447.9, 448.0, 416.0),
        (-448.0, -448.0, -448.0),
        (2.0**-6, 2.0**-6, 2.0**-6),  # smallest normal
        (17 * 2.0**-10, 18 * 2.0**-10, 2.0**-6),  # tie in the lowest normal binade
        # below 2**-6 the packer writes no IEEE subnormals (measured on device, every BF16 value): [2**-7, 2**-6)
        # becomes exponent field 0 with the value's 3 mantissa bits (IEEE-decoded m * 2**-9), smaller values 0
        (2.0**-6 - 2.0**-12, 2.0**-6, 7 * 2.0**-9),  # nearest carries into the normals
        (7 * 2.0**-9, 7 * 2.0**-9, 6 * 2.0**-9),  # 1.75 * 2**-7: mantissa 110; nearest adds 2**-10 -> 111
        (2.0**-7, 2.0**-9, 0.0),
        (3 * 2.0**-10, 0.0, 0.0),
        (2.0**-10, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    ],
)
def test_rounding_cases(q, want_nearest, want_truncate):
    t = torch.tensor([q], dtype=torch.float32)
    assert float(e4m3_round_host(t, NEAREST)) == want_nearest
    assert float(e4m3_round_host(t, TRUNCATE)) == want_truncate


def test_low_range_matches_the_device_probe():
    """[2**-12, 2**-5) on device (``per_token_cast_to_fp8`` bytes, scale 1, both modes), as recorded in
    ``HY4_KV_FP8_PROBE`` when it is set: the model reproduces every byte."""
    import os

    path = os.environ.get("HY4_KV_FP8_PROBE")
    if not path or not Path(path).is_file():
        pytest.skip("HY4_KV_FP8_PROBE (the device probe's saved bytes) not set")
    d = torch.load(path)
    for mode in ("truncate", "nearest"):
        dev = d[mode].view(torch.float8_e4m3fn).float()
        assert torch.equal(dev, e4m3_round_host(d["vals"], MlaKvFp8Rounding(mode)))


def test_negative_zero_becomes_positive_zero_under_nearest():
    """The kernel adds +2**-10 to -0 (``value < 0`` is false) and the cut gives +0; truncation keeps -0."""
    z = torch.tensor([-0.0])
    assert torch.signbit(e4m3_round_host(z, TRUNCATE)).item()
    assert not torch.signbit(e4m3_round_host(z, NEAREST)).item()


def test_zero_rows_and_saturation_through_the_scale():
    x = torch.zeros(3, 512)
    x[1, :128] = torch.linspace(-3.0, 3.0, 128)  # amax 3: scale 2**-7, 3 / 2**-7 = 384
    x[2, 128:256] = 448.0 * 2.0**-5  # amax exactly 448 * 2**-5: scale 2**-5, the block is 448
    for rounding in (NEAREST, TRUNCATE):
        codes, scale = quantize_scaled_fp8_host(x, rounding)
        assert torch.equal(codes[0], torch.zeros(512))  # a zero row stays zero
        assert float(scale[0, 0]) == 2.0**-22  # clamp(0, 1e-4) / 448 = 2.2e-7, up to 2**-22
        assert float(codes.abs().max()) <= 448.0
        assert float(scale[2, 1]) == 2.0**-5 and torch.equal(codes[2, 128:256], torch.full((128,), 448.0))
        assert float(scale[1, 0]) == 2.0**-7 and float(codes[1, :128].abs().max()) == 384.0


def test_nearest_removes_the_truncation_bias():
    g = torch.Generator().manual_seed(0)
    x = (torch.randn(256, 512, generator=g) * 0.3).bfloat16().float()
    out = {}
    for rounding in (NEAREST, TRUNCATE):
        codes, scale = quantize_scaled_fp8_host(x, rounding)
        y = (codes.reshape(256, 4, 128) * scale[..., None]).reshape(256, 512)
        out[rounding] = float((torch.sign(x) * (y - x)).sum() / x.abs().sum())  # signed magnitude error / |x|
    assert out[TRUNCATE] < -0.02  # about 3 % low
    assert abs(out[NEAREST]) < 0.003  # ties away leave a small positive residue only


# --- against decode ------------------------------------------------------------------------------------------------
def test_half_step_routine_is_decode_s():
    """The SFPU routine the prefill kernel adds before the pack is decode's append routine, token for token."""
    if not DECODE_KERNEL.is_file():
        pytest.skip(f"no decode kernel at {DECODE_KERNEL}")
    ours = _cpp_function(PREFILL_KERNEL, "calculate_e4m3_half_step")
    theirs = _cpp_function(DECODE_KERNEL, "calculate_e4m3_half_step")
    assert ours == theirs
    # and both apply it to the whole block in DST after the multiply, before pack_untilize_dest
    assert "e4m3_half_step_tile(k)" in PREFILL_KERNEL.read_text()


def test_against_decode_host_emulation():
    """Decode's host ``e4m3``: equal to our truncation; equal to our nearest except at exact ties (it uses
    torch.round, ties to even; both device kernels round ties away from zero)."""
    e4m3 = _decode_host_e4m3()
    v = _all_bf16_upto_448()
    assert torch.equal(e4m3(v, "trunc"), e4m3_round_host(v, TRUNCATE))
    theirs, ours = e4m3(v, "near"), e4m3_round_host(v, NEAREST)
    a, step = v.abs().double(), _step(v.abs())
    ties = (a / step - torch.floor(a / step)) == 0.5
    diff = theirs != ours
    assert bool((diff <= ties).all())
    # where they differ, ours is one step further from zero
    assert torch.equal((ours[diff].abs() - theirs[diff].abs()).double(), step[diff])


# --- the cache carries its rounding -------------------------------------------------------------------------------
def _scaled_cache(rounding):
    geometry = MlaKvCacheGeometry(latent_dim=512, rope_dim=64)
    storage = SimpleNamespace(dtype=ttnn.fp8_e4m3, layout=ttnn.ROW_MAJOR_LAYOUT, shape=[1, 1, 32, 656])
    return MlaKvCache(format=MlaKvCacheFormat.SCALED_FP8, storage=storage, geometry=geometry, fp8_rounding=rounding)


@pytest.mark.parametrize("rounding", ["truncate", "nearest"])
def test_cache_passes_its_rounding_to_the_cast(monkeypatch, rounding):
    seen = {}

    def cast(latent, **kw):
        seen.update(kw)
        return "fp8", "scales"

    monkeypatch.setattr(ttnn.experimental.deepseek_prefill, "per_token_cast_to_fp8", cast)
    cache = _scaled_cache(rounding)
    assert cache.fp8_rounding is MlaKvFp8Rounding(rounding)
    cache.prepare_scaled_fp8_inputs("latent", "rope", keep_rope_tiled=True)
    assert seen == {"round_scale_to_power_of_two": True, "round_to_nearest": rounding == "nearest"}
    derived = dataclasses.replace(cache, storage=cache.storage)  # the compact / gathered views mla.py builds
    assert derived.fp8_rounding is cache.fp8_rounding
    assert cache.encoding == f"scaled_fp8/{rounding}"
    assert cache.descriptor()["fp8_rounding"] == rounding and cache.descriptor()["row_width"] == 656
    assert cache.descriptor()["fp8_rule"] == ("ties away from zero" if rounding == "nearest" else "toward zero")


def test_cache_default_is_truncate_and_rejects_unknown(expect_error):
    geometry = MlaKvCacheGeometry(latent_dim=512, rope_dim=64)
    storage = SimpleNamespace(dtype=ttnn.fp8_e4m3, layout=ttnn.ROW_MAJOR_LAYOUT, shape=[1, 1, 32, 656])
    assert MlaKvCache(format=MlaKvCacheFormat.SCALED_FP8, storage=storage, geometry=geometry).fp8_rounding is TRUNCATE
    with expect_error(ValueError, "stochastic"):
        _scaled_cache("stochastic")


def test_adapter_defaults(monkeypatch, expect_error):
    from models.demos.deepseek_v3_d_p.tt.runners.adapters.glm_5_3 import GLM53Adapter
    from models.demos.deepseek_v3_d_p.tt.runners.adapters.hy4_preview import KV_FP8_ROUNDING_ENV, Hy4Adapter

    monkeypatch.delenv(KV_FP8_ROUNDING_ENV, raising=False)
    assert GLM53Adapter().kvpe_fp8_rounding is TRUNCATE
    assert Hy4Adapter().kvpe_fp8_rounding is NEAREST
    monkeypatch.setenv(KV_FP8_ROUNDING_ENV, "truncate")
    assert Hy4Adapter().kvpe_fp8_rounding is TRUNCATE
    monkeypatch.setenv(KV_FP8_ROUNDING_ENV, "Nearest")
    assert Hy4Adapter().kvpe_fp8_rounding is NEAREST
    monkeypatch.setenv(KV_FP8_ROUNDING_ENV, "bfp8")
    with expect_error(ValueError, KV_FP8_ROUNDING_ENV):
        Hy4Adapter().kvpe_fp8_rounding
