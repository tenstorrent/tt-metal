# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The ceiling divides by what is ON THE CHIP, measured, not by a rule about what might be.

Every byte figure the roofline had was a PREDICTION of the served width, and the served width is a
per-model decision the checkpoint does not record:

    params x 1.0 B/param   voxtral (bf16) published at 141.8 tok/s/u against a true 54.7 -- 2.6x, in
                           the number the stop gate decides "done" against
    checkpoint bytes       gemma-3's 24.37 GB bf16 file implies a 21.0 ceiling for a model that
                           MEASURES 30.8 -- a ceiling it has already passed, so not a ceiling
    profile op bytes       89 GB for one gemma-3 token (it sums a whole window: prefill plus many
                           decode steps) -> 5.8 tok/s/u

No single prediction can be right for both models, because gemma-3 is served narrower than its
checkpoint and voxtral is not. So the width is OBSERVED instead: weight_census walks the built model
after setup(), when the loader has already decided each tensor's dtype, and sums numel x that dtype.

perf_target.active_bytes has accepted exactly this shape (`weight_tensors: [{numel, dtype}]`) since
it was written. Nothing ever filled it in, which is why every caller fell through to a prediction.

  r1  the census: mixed precision, shared tensors, unknown dtypes
  r2  the ceiling prefers it, and both models come out right
  r3  an incomplete census is refused, not used as a lower bound
"""
from __future__ import annotations

import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))

from agent.perf_target import compute_target  # noqa: E402
from agent.weight_census import bytes_per_elem, census, dtype_name, marker  # noqa: E402

_ENV = {"dram_bw_gbps": 512.0}


class _DT:
    def __init__(self, n):
        self.name = n

    def __str__(self):
        return "DataType." + self.name


class _T:
    """A DEVICE-resident tensor. storage_type() is what tells the census this is on the chip -- a
    torch tensor carries .shape and .dtype identically, and counting those reported voxtral at
    29.96 GB against a real device footprint of ~11.3, at a width no device tensor has."""

    def __init__(self, shape, dt):
        self.shape, self.dtype = shape, dt

    def storage_type(self):
        return "DEVICE"


# ---------------------------------------------------------------- r1 THE CENSUS


def test_r1_the_block_formats_carry_their_shared_exponent():
    """bfloat8_b and bfloat4_b are BLOCK formats: a shared exponent per 16-element tile adds 1/16 of
    a byte to every element. Calling them 1 and 0.5 understates a bf8 model's bytes by 6%."""
    assert bytes_per_elem("bfloat8_b") == 1.0625
    assert bytes_per_elem("bfloat4_b") == 0.5625
    assert bytes_per_elem("bfloat16") == 2.0


def test_r1_an_unknown_dtype_contributes_nothing_and_is_counted():
    """0.0, never a default. A dtype guessed here becomes a byte width nobody chose."""
    assert bytes_per_elem("mystery_format") == 0.0
    c = census(_T((16, 16), _DT("MYSTERY_FORMAT")))
    assert c["unknown_dtype_tensors"] == 1
    assert c["complete"] is False
    assert c["weight_bytes"] == 0


def test_r1_mixed_precision_is_summed_as_it_actually_is():
    """A model serving attention at bf8 and MLP at bf4 is not one blended width."""

    class L:
        def __init__(self):
            self.attn = _T((3840, 15360), _DT("BFLOAT8_B"))
            self.mlp = _T((3840, 15360), _DT("BFLOAT4_B"))

    c = census([L(), L()])
    want = 2 * (3840 * 15360 * 1.0625 + 3840 * 15360 * 0.5625)
    assert abs(c["weight_bytes"] - want) < 1, (c["weight_bytes"], want)
    # `complete` means "usable as the model's byte count", which now also requires a checkpoint to
    # tell a weight from a runtime tensor -- there is none here. What this test is about is the
    # dtype summation, and that is what it asserts.
    assert c["unknown_dtype_tensors"] == 0


def test_r1_a_shared_tensor_is_counted_once():
    """Tied embeddings are one tensor referenced twice. Counting it twice inflates the total exactly
    as much as missing it deflates it."""
    shared = _T((4096, 4096), _DT("BFLOAT8_B"))

    class M:
        def __init__(self):
            self.emb = shared
            self.head = shared

    c = census(M())
    assert len(c["weight_tensors"]) == 1
    assert abs(c["weight_bytes"] - 4096 * 4096 * 1.0625) < 1


def test_r1_the_marker_states_enough_to_judge_it():
    c = census(_T((32, 32), _DT("BFLOAT16")), scope="pipeline")
    m = marker(c)
    # complete=0 without a checkpoint: the census cannot classify weights from scratch, and says so.
    assert m.startswith("TRACE_WEIGHT_BYTES=") and "scope=pipeline" in m and "complete=" in m


def test_r1_dtype_names_come_from_ttnn_or_torch_alike():
    assert dtype_name(_DT("BFLOAT8_B")) == "bfloat8_b"
    assert dtype_name("torch.bfloat16") == "bfloat16"
    assert dtype_name(None) == ""


# ---------------------------------------------------------------- r2 THE CEILING


_GEMMA = {"total_params": 11180446320, "weight_bytes": 24374793024, "dominant_dtype": "bfloat16"}
_VOXTRAL = {"total_params": 4700000000, "weight_bytes": int(9.36e9), "dominant_dtype": "bfloat16"}


def _rate(facts, **extra):
    return compute_target(dict(facts, **extra), _ENV).theoretical_rate


def test_r2_the_census_outranks_the_prediction_for_both_models():
    """One mechanism, both models right -- which is what none of the predictions could manage."""
    # voxtral: served AT its checkpoint precision -> 9.36 GB -> 54.7, not the 141.8 params x 1.0 gave
    vox = _rate(_VOXTRAL, device_weight_bytes=int(9.36e9), device_census_complete=True)
    assert abs(vox - 54.7) < 0.5, vox
    # gemma-3: served NARROWER than its checkpoint -> 11.9 GB -> 43.0, and 30.8 sits under it
    gem = _rate(_GEMMA, device_weight_bytes=int(11.9e9), device_census_complete=True)
    assert abs(gem - 43.0) < 0.5, gem
    assert gem > 30.8, "a ceiling the model has already passed is not a ceiling"


def test_r2_the_checkpoint_would_have_broken_gemma():
    """Why the census is needed at all, stated as the arithmetic: gemma-3 measures 30.8, and its
    checkpoint implies 21.0. Kept as a test so the next person proposing 'just use weight_bytes'
    sees the counterexample rather than rediscovering it on a run."""
    assert abs(512e9 / _GEMMA["weight_bytes"] - 21.0) < 0.5
    assert 512e9 / _GEMMA["weight_bytes"] < 30.8


# ---------------------------------------------------------------- r3 REFUSAL


def test_r3_an_incomplete_census_is_refused_not_used_as_a_lower_bound():
    """Too FEW bytes reads as too HIGH a ceiling -- the direction that ends a run early believing it
    is at the wall. An incomplete census falls back to the existing rule instead."""
    partial = _rate(_GEMMA, device_weight_bytes=int(5e9), device_census_complete=False)
    assert abs(partial - _rate(_GEMMA)) < 0.01, "an incomplete census was used"
    assert partial < 102.4, "5 GB would have published a 102 tok/s/u ceiling"


def test_r3_a_zero_census_changes_nothing():
    assert abs(_rate(_GEMMA, device_weight_bytes=0, device_census_complete=True) - _rate(_GEMMA)) < 0.01


# --- the WIDTH is measured, which is what the placeholder was standing in for -------------------


def test_the_measured_width_replaces_the_placeholder():
    """params x 1.0 B/param was a placeholder for a number nobody could get. voxtral is served bf16
    -- 2 bytes per parameter -- so it published 149.3 tok/s/u against a true 74.7. gemma-3 is served
    at a MIX (bf16 + bf4 + bf8, measured: 0.8057) and the placeholder happened to land within 24%,
    which is why this survived: on one model it looked nearly right."""
    E = {"dram_bw_gbps": 512.0}
    vox = {"total_params": 3429000000}
    # THE PLACEHOLDER IS GONE ENTIRELY, so this no longer publishes 149.3 even before the census: with
    # no width stated anywhere the byte model assumes bf16, which is the honest default for a
    # checkpoint that says nothing, and gives the same 74.7 the census confirms below.
    assert abs(compute_target(vox, E).theoretical_rate - 74.7) < 1.0
    vox_m = dict(vox, bytes_per_param=2.0, device_census_complete=True)
    assert abs(compute_target(vox_m, E).theoretical_rate - 74.7) < 1.0


def test_a_fractional_width_is_not_truncated_to_the_placeholder():
    """_scalar coerces with type(default), so an int default turns 1.0625 into 1 -- silently
    restoring the placeholder. gemma-3's measured bf8 width vanished exactly that way while
    voxtral's 2.0 survived, so the fix appeared to work on one model and not the other. A width is
    fractional BY NATURE: bf8 is 1.0625 and bf4 0.5625, because a 16-element tile shares an
    exponent."""
    E = {"dram_bw_gbps": 512.0}
    f = {"total_params": 11180446320, "bytes_per_param": 1.0625, "device_census_complete": True}
    got = compute_target(f, E).theoretical_rate
    assert abs(got - 43.1) < 0.5, got
    assert abs(got - 45.8) > 1.0, "1.0625 was truncated to 1 -- the placeholder is back"


def test_an_incomplete_census_is_refused_without_reviving_the_placeholder():
    """An incomplete census is still refused -- too few bytes reads as too HIGH a ceiling, which ends
    a run early believing it is at the wall. What it falls back TO has changed: not params x 1.0
    (45.8), but the byte model's own bf16 default, which is a width rather than a stand-in for one."""
    E = {"dram_bw_gbps": 512.0}
    f = {"total_params": 11180446320, "bytes_per_param": 0.5, "device_census_complete": False}
    got = compute_target(f, E).theoretical_rate
    assert abs(got - 22.9) < 0.5, got
    assert abs(got - 45.8) > 1.0, "params x 1.0 is back"
    assert abs(got - 91.6) > 1.0, "the refused 0.5 width was used anyway"


def test_the_width_survives_the_marker_round_trip():
    """The census runs in the workload process and the ceiling in another, so the marker is the only
    place the width crosses. It was emitted and read at both ends, and parsed at neither."""
    line = (
        "TRACE_WEIGHT_BYTES=15485398560 scope=pipeline tensors=757 unknown_dtype=0 "
        "complete=1 bytes_per_param=0.8057 dtypes=bfloat16:400,bfloat4_b:257"
    )
    assert float(line.split("bytes_per_param=", 1)[1].split()[0]) == 0.8057
    src = (_PA / "cc_optimize" / "perf_mcp.py").read_text()
    assert 'line.split("bytes_per_param=", 1)' in src, "the marker's width is emitted but never parsed"


def test_the_mix_is_reported_so_the_average_can_be_checked():
    """An average nobody can decompose is a number to be trusted rather than read. gemma-3's
    0.8057 is only meaningful beside bfloat16:400, bfloat4_b:257, bfloat8_b:96."""
    from agent.weight_census import census, marker

    class _DT:
        def __init__(self, n):
            self.name = n

    class _T:
        def __init__(self, sh, dt):
            self.shape, self.dtype = sh, dt

        def storage_type(self):
            return "DEVICE"

    class _M:
        def __init__(self):
            self.a = [_T((64, 64), _DT("BFLOAT8_B")) for _ in range(3)]
            self.b = [_T((64, 64), _DT("BFLOAT4_B")) for _ in range(1)]

    m = marker(census(_M()))
    assert "bytes_per_param=" in m and "bfloat8_b:3" in m and "bfloat4_b:1" in m


# --- resident on the CHIP, not merely shaped like a tensor --------------------------------------


class _HostT:
    """A torch-style HOST tensor: identical duck type, no device."""

    def __init__(self, shape, dt):
        self.shape, self.dtype = shape, dt


def test_a_host_copy_is_not_counted_as_device_bytes():
    """THE VOXTRAL CASE. Its pipeline loads weights with dtype=torch.float32 and keeps that copy
    alive on the HOST while the device holds bf16. `anything with .shape and .dtype is a tensor`
    counted both, reporting 29.96 GB for a model whose device footprint is ~11.3 GB and a width of
    2.9076 B/param that no device tensor has.

    The fp32 half is real waste -- 18.7 GB of host RAM held for nothing -- but the chip never
    streams it, so it cannot slow a token down and must not enter the ceiling."""
    from agent.weight_census import census

    class _M:
        def __init__(self):
            self.dev = [_T((4096, 4096), _DT("BFLOAT16")) for _ in range(10)]
            self.hf = [_HostT((4096, 4096), "torch.float32") for _ in range(10)]

    c = census(_M())
    assert len(c["weight_tensors"]) == 10, "host copies were counted"
    assert abs(c["bytes_per_param"] - 2.0) < 1e-9, c["bytes_per_param"]
    assert abs(c["weight_bytes"] - 10 * 4096 * 4096 * 2) < 1


def test_residency_is_asked_of_the_tensor_not_inferred_from_dtype():
    """fp32 is legal on device and bf16 is legal on the host, so dtype says nothing about where a
    tensor lives. Inferring from it would be the same class of mistake one level along."""
    from agent.weight_census import census

    class _M:
        def __init__(self):
            self.on_dev_fp32 = _T((1024, 1024), _DT("FLOAT32"))
            self.on_host_bf16 = _HostT((1024, 1024), "torch.bfloat16")

    c = census(_M())
    assert [t["dtype"] for t in c["weight_tensors"]] == ["float32"], c["weight_tensors"]


def test_a_tensor_whose_device_probe_raises_is_treated_as_host():
    """Failing closed: an object that defines the name and raises has not shown it is on device."""
    from agent.weight_census import census

    class _Angry:
        shape, dtype = (8, 8), "torch.float32"

        def storage_type(self):
            raise RuntimeError("no device")

    assert census(_Angry())["weight_tensors"] == []
