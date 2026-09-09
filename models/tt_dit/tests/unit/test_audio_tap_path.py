# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""`depthwise_tap_filter` picks its conv1d formulation and DRAM slice count from the measured tables in
`utils/tap_filter_configs.py`, with the old trial chain behind them. The pure tests pin the lookup chain and the
slice-count derivation with fakes; the device tests walk every table row for this device class (a stale row fails
here, not in a decoder build), and check that unknown shapes and stale rows still fall back to a correct answer."""

from __future__ import annotations

import types

import pytest
import torch

import ttnn

from ...layers import audio_ops
from ...layers.audio_ops import depthwise_tap_filter
from ...utils import tap_filter_configs as tfc
from ...utils.tap_filter_configs import (
    applicable_formulations,
    derive_num_slices,
    register_tap_configs,
    slice_config_for,
    slice_signature,
    tap_device_key,
)

# (C, K, stride, T_pad): the depthwise resample filters MiniMax-H3's BigVGAN runs at a 5 s clip (K=12
# anti-alias taps at stride 1 up / stride 2 down per band, K=7 in the LTX vocoder), from the latent rate
# (T_pad ~ 40-200, where full C cannot fit and C is chunked) to the audio rate.
SHAPES = [
    (512, 12, 1, 166),
    (512, 12, 2, 416),
    (512, 7, 1, 40),
    (256, 12, 1, 832),
    (256, 7, 1, 80),
    (128, 12, 2, 1664),
    (64, 12, 1, 3328),
    (64, 7, 1, 166),
    (32, 12, 2, 3328),
]
SINGLE_DEVICE_PARAMS = [{"l1_small_size": 65536}]


# ---------------------------------------------------------------- pure: derivation and table plumbing


@pytest.mark.parametrize(
    ("T_out", "T_ref", "n_ref", "expected"),
    [
        (160, 160, 4, 4),  # the reference itself
        (320, 160, 4, 8),  # twice the length, twice the slices
        (161, 160, 4, 5),  # rounds up, never down
        (40, 160, 4, 1),  # a quarter of the length: one slice (L1_FULL)
        (3, 160, 4, 1),  # at least one
        (2, 160, 400, 2),  # never more slices than output rows
    ],
)
def test_derive_num_slices(T_out, T_ref, n_ref, expected):
    assert derive_num_slices(T_out, T_ref, n_ref) == expected


def test_derive_num_slices_rejects_empty_output(expect_error):
    with expect_error(ValueError, "T_out must be positive"):
        derive_num_slices(0, 160, 4)


def test_slice_config_for_one_slice_is_l1_full():
    assert slice_signature(slice_config_for(1)) == slice_signature(ttnn.Conv2dL1FullSliceConfig)
    cfg = slice_config_for(3)
    assert cfg.num_slices == 3
    assert slice_signature(cfg) == ("DRAM_WIDTH", 3)
    assert slice_signature(None) is None


@pytest.mark.parametrize(
    ("C", "expected"),
    [
        (512, ["direct", 128, 64, 32]),
        (96, ["direct", 32]),  # 128 >= C, 64 does not divide 96
        (32, ["direct"]),  # every chunk width is >= C
        (16, ["direct"]),
    ],
)
def test_applicable_formulations(C, expected):
    assert applicable_formulations(C) == expected


def test_register_validates_rows(expect_error):
    key = ("unit-test-arch", 1, 1)
    try:
        with expect_error(ValueError, "is not one of"):
            register_tap_configs(key, formulations={(512, 7, 1): 96})
        with expect_error(ValueError, "must be positive"):
            register_tap_configs(key, slices={(128, 7, 1): (0, 4)})
        register_tap_configs(key, formulations={(512, 7, 1): 128}, slices={(128, 7, 1): (160, 4)})
        assert tfc.tap_formulation(key, 512, 7, 1) == 128
        assert tfc.tap_slice_config(key, 128, 7, 1, 320).num_slices == 8
        assert tfc.tap_slice_config(key, 64, 7, 1, 320) is None
    finally:
        tfc.clear_tap_configs(key)


# ---------------------------------------------------------------- pure: the lookup chain, with fakes


class _FakeTensor:
    def __init__(self, shape):
        self.shape = shape


FAKE_KEY = ("fake-arch", 7, 7)


@pytest.fixture
def chain(monkeypatch):
    """Drive `depthwise_tap_filter` without a device: the conv helpers record what they were asked to run and
    raise for the (formulation, slice signature) pairs the test declares as not fitting."""
    state = types.SimpleNamespace(calls=[], failing=set(), warnings=[])

    def conv(x_BTC, weight, *, C, slice_config, **kwargs):
        state.calls.append(("direct", slice_signature(slice_config)))
        if ("direct", slice_signature(slice_config)) in state.failing:
            raise RuntimeError("does not fit")
        return "direct-out"

    def chunked(x_BTC, weight, *, C, chunk, slice_config, **kwargs):
        state.calls.append((chunk, slice_signature(slice_config)))
        if (chunk, slice_signature(slice_config)) in state.failing:
            raise RuntimeError("does not fit")
        return f"chunk{chunk}-out"

    def mac(x_BTC, taps, stride, *, T_out, dtype):
        state.calls.append(("mac", None))
        return "mac-out"

    monkeypatch.setattr(audio_ops, "_depthwise_tap_conv1d", conv)
    monkeypatch.setattr(audio_ops, "_depthwise_tap_conv1d_chunked", chunked)
    monkeypatch.setattr(audio_ops, "_depthwise_tap_mac", mac)
    monkeypatch.setattr(audio_ops, "_tap_weight", lambda taps, channels, dtype, mesh_device: None)
    monkeypatch.setattr(audio_ops, "tap_device_key", lambda mesh_device: FAKE_KEY)
    monkeypatch.setattr(audio_ops.logger, "warning", lambda msg: state.warnings.append(msg))
    audio_ops._TAP_WARNED.clear()
    tfc.clear_tap_configs(FAKE_KEY)
    yield state
    tfc.clear_tap_configs(FAKE_KEY)


def _call(cache=None, *, C=512, K=7, stride=1, T_pad=166):
    cache = {"cc": object()} if cache is None else cache
    x = _FakeTensor((1, T_pad, C))
    out = depthwise_tap_filter(x, [1.0] * K, stride, mesh_device=None, dtype=ttnn.float32, cache=cache)
    return out, cache


def test_table_hit_runs_the_explicit_plan_first_and_only(chain):
    register_tap_configs(FAKE_KEY, formulations={(512, 7, 1): 128}, slices={(128, 7, 1): (160, 4)})
    out, cache = _call()
    assert out == "chunk128-out"
    assert chain.calls == [(128, ("DRAM_WIDTH", 4))]  # T_out = 160 -> the reference count itself
    assert cache[("tap_path", 1, 166, 512, 7, 1)][0] == 128
    assert chain.warnings == []


def test_table_hit_derives_the_slice_count_for_another_length(chain):
    register_tap_configs(FAKE_KEY, formulations={(512, 7, 1): 128}, slices={(128, 7, 1): (160, 4)})
    _call(T_pad=326)  # T_out = 320 -> 8 slices
    assert chain.calls == [(128, ("DRAM_WIDTH", 8))]


def test_cached_plan_is_tried_before_the_table(chain):
    register_tap_configs(FAKE_KEY, formulations={(512, 7, 1): 128}, slices={(128, 7, 1): (160, 4)})
    _, cache = _call()
    chain.calls.clear()
    _call(cache)
    assert chain.calls == [(128, ("DRAM_WIDTH", 4))]
    assert chain.warnings == []


def test_stale_explicit_slices_fall_back_to_auto_then_to_the_chain(chain):
    register_tap_configs(FAKE_KEY, formulations={(512, 7, 1): 128}, slices={(128, 7, 1): (160, 4)})
    chain.failing = {(128, ("DRAM_WIDTH", 4))}
    out, _ = _call()
    assert out == "chunk128-out"
    assert chain.calls == [(128, ("DRAM_WIDTH", 4)), (128, None)]
    assert len(chain.warnings) == 1 and "failed" in chain.warnings[0]


def test_stale_formulation_falls_through_the_whole_chain(chain):
    register_tap_configs(FAKE_KEY, formulations={(512, 7, 1): "direct"})
    chain.failing = {("direct", None)}
    out, cache = _call()
    assert out == "chunk128-out"
    # direct (table, auto) fails; the trial chain then skips the identical (direct, auto) attempt.
    assert chain.calls == [("direct", None), (128, None)]
    assert cache[("tap_path", 1, 166, 512, 7, 1)] == (128, None)
    assert any("failed" in w for w in chain.warnings)


def test_no_row_probes_widest_first_and_says_which_row_to_add(chain):
    chain.failing = {("direct", None)}
    out, _ = _call()
    assert out == "chunk128-out"
    assert chain.calls == [("direct", None), (128, None)]
    assert any("no table row" in w for w in chain.warnings)
    assert any("(512, 7, 1): 128," in w for w in chain.warnings)


def test_nothing_fits_ends_in_mac(chain):
    chain.failing = {("direct", None), (128, None), (64, None), (32, None)}
    out, cache = _call()
    assert out == "mac-out"
    assert [c[0] for c in chain.calls] == ["direct", 128, 64, 32, "mac"]
    assert cache[("tap_path", 1, 166, 512, 7, 1)] == ("mac", None)
    assert any("MAC fallback" in w for w in chain.warnings)


def test_inapplicable_row_is_ignored(chain):
    register_tap_configs(FAKE_KEY, formulations={(96, 7, 1): 128})  # 128 does not divide 96
    out, _ = _call(C=96)
    assert out == "direct-out"
    assert chain.calls == [("direct", None)]
    assert any("does not apply" in w for w in chain.warnings)


def test_prepared_weight_is_keyed_on_geometry_and_slicing(chain, monkeypatch):
    """The prepared weight depends on the parallelization conv1d picked: two lengths or two slice configs must
    never share one (that was the 15 s garbage-audio bug)."""
    keys = []

    def conv(x_BTC, weight, *, C, slice_config, cache, wkey, **kwargs):
        keys.append(wkey)
        return "direct-out"

    monkeypatch.setattr(audio_ops, "_depthwise_tap_conv1d", conv)
    register_tap_configs(FAKE_KEY, formulations={(32, 7, 1): "direct"}, slices={(32, 7, 1): (160, 2)})
    _, cache = _call(C=32, T_pad=166)
    _call(cache, C=32, T_pad=326)
    assert len(set(keys)) == 2
    assert keys[0][-3:] == (1, 166, ("DRAM_WIDTH", 2)) and keys[1][-3:] == (1, 326, ("DRAM_WIDTH", 4))


# ---------------------------------------------------------------- device: the table against reality


def _inputs(C, K, T_pad, mesh_device):
    torch.manual_seed(0)
    x = torch.randn(1, T_pad, C, dtype=torch.float32)
    x_dev = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)
    taps = [float(t) for t in torch.randn(K)]
    return x, x_dev, taps


def _reference(x, taps, stride):
    K, C = len(taps), x.shape[-1]
    weight = torch.tensor(taps, dtype=torch.float32).view(1, 1, K).expand(C, 1, K).contiguous()
    return torch.nn.functional.conv1d(x.transpose(1, 2), weight, stride=stride, groups=C).transpose(1, 2)


def _run_and_check(mesh_device, C, K, stride, T_pad, cache):
    x, x_dev, taps = _inputs(C, K, T_pad, mesh_device)
    out = depthwise_tap_filter(x_dev, taps, stride, mesh_device=mesh_device, dtype=ttnn.float32, cache=cache)
    actual = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()
    expected = _reference(x, taps, stride)
    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4), "numerically wrong"
    return cache[("tap_path", 1, T_pad, C, K, stride)]


@pytest.fixture
def quiet_warnings(monkeypatch):
    warnings = []
    monkeypatch.setattr(audio_ops.logger, "warning", lambda msg: warnings.append(msg))
    audio_ops._TAP_WARNED.clear()
    return warnings


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", SINGLE_DEVICE_PARAMS, indirect=True)
def test_every_table_row_runs_as_tabled(mesh_device, quiet_warnings):
    """Each formulation row for this device class runs with its explicit slice count, no fallback, right answer --
    at a swept length and at a different one (the derived count)."""
    key = tap_device_key(mesh_device)
    rows = tfc._FORMULATIONS.get(key)
    if not rows:
        pytest.skip(f"no tap-filter table for {key}; run tools/sweep_tap_filter_configs.py to create one")
    lengths = {(C, K, s): T_pad for (C, K, s, T_pad) in SHAPES}
    for (C, K, stride), formulation in rows.items():
        T_ref = lengths.get((C, K, stride), 8 * K)
        for T_pad in (T_ref, 2 * T_ref + 3):
            plan = _run_and_check(mesh_device, C, K, stride, T_pad, cache={})
            assert (
                plan[0] == formulation
            ), f"{(C, K, stride)} at T_pad={T_pad}: ran {plan[0]!r}, table says {formulation!r}"
            channels = C if formulation == "direct" else formulation
            if (channels, K, stride) in tfc._SLICES.get(key, {}):
                assert plan[1] is not None, f"{(C, K, stride)} at T_pad={T_pad}: fell back to auto slicing"
    assert quiet_warnings == [], f"table rows must run silently, got: {quiet_warnings[:3]}"


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", SINGLE_DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize(("C", "K", "stride", "T_pad"), SHAPES)
def test_production_shapes_are_correct(mesh_device, C, K, stride, T_pad):
    """Every MiniMax-H3 shape decodes right whichever path the chain lands on."""
    plan = _run_and_check(mesh_device, C, K, stride, T_pad, cache={})
    assert plan[0] in applicable_formulations(C) + ["mac"]


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", SINGLE_DEVICE_PARAMS, indirect=True)
def test_unknown_shape_probes_and_is_correct(mesh_device, quiet_warnings):
    """A channel count no table has (96) goes through the trial chain and still comes out right."""
    plan = _run_and_check(mesh_device, 96, 7, 1, 166, cache={})
    assert plan[0] in applicable_formulations(96) + ["mac"]
    assert any("no table row" in w for w in quiet_warnings)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", SINGLE_DEVICE_PARAMS, indirect=True)
def test_stale_row_falls_back_on_device(mesh_device, quiet_warnings):
    """Force a row that cannot fit (full C=512 at K=7: the C*K activation block never fits L1) and check the chain
    recovers to a working formulation with one warning and a correct result."""
    key = tap_device_key(mesh_device)
    saved_f = dict(tfc._FORMULATIONS.get(key, {}))
    saved_s = dict(tfc._SLICES.get(key, {}))
    try:
        tfc.clear_tap_configs(key)
        register_tap_configs(key, formulations={(512, 7, 1): "direct"})
        plan = _run_and_check(mesh_device, 512, 7, 1, 166, cache={})
        assert plan[0] != "direct"
        assert any("failed" in w for w in quiet_warnings)
    finally:
        tfc.clear_tap_configs(key)
        register_tap_configs(key, formulations=saved_f or None, slices=saved_s or None)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", SINGLE_DEVICE_PARAMS, indirect=True)
def test_short_then_long_clip_shares_a_cache_correctly(mesh_device):
    """The served order that produced garbage 15 s audio: a short clip prepares the weights, a longer clip reuses
    the cache. Both must be right."""
    cache: dict = {}
    _run_and_check(mesh_device, 512, 7, 1, 40, cache)
    _run_and_check(mesh_device, 512, 7, 1, 166, cache)
    _run_and_check(mesh_device, 512, 7, 1, 40, cache)
