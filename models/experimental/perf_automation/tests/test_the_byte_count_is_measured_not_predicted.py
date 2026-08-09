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
    def __init__(self, shape, dt):
        self.shape, self.dtype = shape, dt


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
    assert c["complete"] is True


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
    assert m.startswith("TRACE_WEIGHT_BYTES=") and "scope=pipeline" in m and "complete=1" in m


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
