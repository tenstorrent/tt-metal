# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Does a macro-scheduled SFPSWAP honour SFPU index tracking? (Blackhole)

THE GATING QUESTION
-------------------
The measured topk_xl merge/rebuild SFPLOADMACRO win is FUSED-only, and the one
shipping consumer (``ttnn.experimental.topk_large_indices``) runs merge and
rebuild UNFUSED: values in LREG0..3, indices riding along in LREG4..7 through
SFPU index-tracking mode (``LaneConfig.ENABLE_DEST_INDEX``, the ``0x4`` that
``_topk_xl_init_<K, false>`` writes). An unfused port is possible only if a
SFPSWAP scheduled into an SFPLOADMACRO Simple slot still performs the
argmin/argmax companion swap ``LReg[4+VC] <-> LReg[4+VD]`` that a software
SFPSWAP performs (SFPSWAP.md:58-70; ttsim tensix.cpp SFPSWAP model agrees).

DESIGN: PAIRED ARMS, ONE HARNESS — DIFFERENTIAL FIRST
------------------------------------------------------
All arms run the identical Dst geometry (a fragment of the unfused merge:
value run A/B in Dst tile 0, index run A/B in Dst tile 1) on identical
stimuli; only the MATH body differs. THE PRIMARY EVIDENCE IS DIFFERENTIAL
(macro arm == software arm, byte-for-byte on both packed tiles): for the
port's correctness, macro == software is sufficient — the absolute semantics
of SFPSWAP are the shipping baseline whichever way they fall.

WHAT THE FIRST SILICON RUN TAUGHT (2026-08-16), AND HOW THIS FILE ADAPTED
-------------------------------------------------------------------------
1. All four differential macro arms PASSED — macro == software held on every
   word, including the signed lanes.
2. The software reference itself disagreed with the ISA-doc golden on lanes
   that were simultaneously (a) all-negative pairs and (b) k % 4 == 3 —
   because the original stimuli negated exactly every 4th lane, sign
   semantics and lane position were aliased and the failure could not be
   attributed. BOTH the BH ISA doc (SFPSWAP.md functional model) and ttsim
   (``sign_mag32_total_order``, whose comment says it matches tested BH)
   predict a swap on those lanes, so silicon disagreed with both models —
   or a lane-patterned effect (e.g. stale CC lane-enable state, which
   ``_llk_math_eltwise_unary_sfpu_init_once_`` does NOT reset) suppressed
   those lanes.
   THIS REVISION DE-ALIASES THE TWO: the strict-golden window (pair A0/B0)
   is now ALL-POSITIVE with a k % 3 swap pattern (positives matched the doc
   model on silicon), the kernel now issues an explicit ``TTI_SFPENCC(0,0,0,0)``
   CC reset, and all signed/mixed pairs moved to a CHARACTERIZATION window
   (pair A1/B1) that is differential-asserted and consistency-asserted but
   positionally only REPORTED, classified against candidate semantic models.
3. The 0x80-bit mutation produced garbage (0xBF2CC4C7 broadcast), not the
   modeled self-compare no-op: SFPSWAP(VC == VD) is a same-register 2-cycle
   read-modify-write with no architectural contract. The mutation is now
   "zero the whole Sequence word" — the documented degeneration of
   SFPLOADMACRO into a plain SFPLOAD — whose expected output is exactly the
   identity (loads + write-back stores), which IS modelable.

FRESHNESS. Each arm's kernel writes a per-run RUNTIME nonce (carried in the
RELU_CONFIG runtime slot) into its diag tile; the driver asserts the echo, so
a cached/stale result buffer cannot masquerade as a fresh execution. Results
ARE memoized in-process across tests of one consumer session (that is why the
session is fast) — the nonce proves the one real execution per arm was fresh.

Arms (see sources/topk_unfused_macro_probe_test.cpp for the exact bodies):
  0 SW_BOTH          software reference, both pairs
  1 MACRO_SINGLE     one macro swap                      == arm0 <=> YES
  2 MACRO_DUAL_2A    two macro swaps, 2 slots apart      == arm0 <=> fused rule holds
  3 MACRO_DUAL_3A    two macro swaps, 3 slots apart      isolates separation
  4 MACRO_FULL       swap + value store on the macro     == arm0 <=> full trick
  5 MACRO_MUTATE     Sequence words zeroed               MUST equal identity, differ from arm0
  6 IDX_STORE_MACRO  index load+deferred store on macro  == arm7 <=> idx rides
  7 SW_SINGLE        software reference for arm 6
"""

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from conftest import blackhole_only
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, DestSync, Tilize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    NUM_FACES,
    RELU_CONFIG,
    TILIZE,
    TemplateParameter,
    generate_input_dim,
)

TILE_DATUMS = 1024
RES_TILES = 3  # values, indices, diag
DIAG_TILE = 2

# Dst geometry — MUST match sources/topk_unfused_macro_probe_test.cpp.
# One SFPU load at dest-unit offset u covers 32 lanes = datums [u*16, u*16+32).
# (Confirmed by the first silicon run: golden-swapped positive lanes at datums
# 8..15 matched, which is only possible under this contiguous mapping.)
LANES = 32
WIN_A0 = 0 * 16  # STRICT window: value/index run A, first vector  (offset 0)
WIN_A1 = 4 * 16  # CHARACTERIZATION window: run A, second vector   (offset 4)
WIN_B0 = 16 * 16  # STRICT window: run B, first vector             (offset 16)
WIN_B1 = 20 * 16  # CHARACTERIZATION window: run B, second vector  (offset 20)

STRICT_PAIR = (WIN_A0, WIN_B0)  # all-positive; positional golden asserted
CHAR_PAIR = (WIN_A1, WIN_B1)  # signed/mixed; reported + consistency-asserted

SENTINEL_START = 0xC0DEBA5E
SENTINEL_END = 0xC0DEE0D1

REPORT_DIR = Path(
    os.environ.get("TOPK_PROBE_REPORT_DIR", "/tmp/topk_unfused_macro_probe")
)


@dataclass
class PROBE_ARM(TemplateParameter):
    """Emits ``#define PROBE_ARM <n>`` — selects the MATH body under test."""

    probe_arm: int = 0

    def convert_to_cpp(self) -> str:
        return f"#define PROBE_ARM {self.probe_arm}"


# ---------------------------------------------------------------------------
# Stimuli.
#
# STRICT window (A0/B0): all-positive, distinct, no ties — the regime silicon
# demonstrably matches the ISA-doc model in. Swap on k % 3 == 0 (11 of 32
# lanes), deliberately NOT aliased to any mod-4/mod-8 lane structure so a
# lane-patterned hardware effect shows up as a strict-window failure with a
# recognisable histogram instead of masquerading as sign semantics.
#
# CHARACTERIZATION window (A1/B1):
#   k  0..15  both-negative pairs   (doc model: swap iff |b| > |a|)
#   k 16..23  mixed-sign pairs      (doc model: always swap; b < 0 < a)
#             k 16..19 with |b| < |a|, k 20..23 with |b| > |a| — separates a
#             raw-unsigned compare from a magnitude compare.
#   k 24..31  positive controls     (doc model: swap iff k odd) — detects
#             lane-position effects inside this window independent of sign.
# ---------------------------------------------------------------------------


def _pair_values(k: int, base: int, char: bool):
    """(a, b) for lane k of a window. a -> run A (VD), b -> run B (VC)."""
    if not char:
        a = float(base + 3 * k)
        b = a - 1.0 if (k % 3 == 0) else a + 1.0
        return a, b
    if k < 16:  # both negative
        mag_a = float(400 + 3 * k)
        mag_b = mag_a + 1.0 if (k % 3 == 0) else mag_a - 1.0
        return -mag_a, -mag_b
    if k < 24:  # mixed sign, a positive
        a = float(500 + 3 * k)
        b = -(a - 10.0) if (k < 20) else -(a + 10.0)
        return a, b
    a = float(700 + 3 * k)  # positive controls
    b = a - 1.0 if (k % 2 == 1) else a + 1.0
    return a, b


def _value_tile() -> torch.Tensor:
    vals = [float(2000 + i) for i in range(TILE_DATUMS)]  # distinct filler
    for k in range(LANES):
        a, b = _pair_values(k, base=2, char=False)
        vals[WIN_A0 + k] = a
        vals[WIN_B0 + k] = b
        a, b = _pair_values(k, base=0, char=True)
        vals[WIN_A1 + k] = a
        vals[WIN_B1 + k] = b
    return torch.tensor(vals, dtype=torch.float32)


def _index_tile() -> torch.Tensor:
    # 0x41000000 | position: unique normal floats whose low mantissa bits carry
    # the position — raw-word arithmetic never leaves the normal-float range.
    words = torch.arange(TILE_DATUMS, dtype=torch.int32) + 0x41000000
    return words.view(torch.float32)


def _input_words():
    """(val_words, idx_words) as lists of u32, positionally."""
    v = _value_tile().view(torch.int32).tolist()
    i = _index_tile().view(torch.int32).tolist()
    return [x & 0xFFFFFFFF for x in v], [x & 0xFFFFFFFF for x in i]


def _sign_mag_is_smaller(c: int, d: int) -> bool:
    """SFPSWAP's total order per SFPSWAP.md AND ttsim sign_mag32_total_order:
    a sign-bit word maps to (word ^ 0x7FFFFFFF) as int32."""

    def remap(u):
        u &= 0xFFFFFFFF
        if u & 0x80000000:
            u ^= 0x7FFFFFFF
        return u - (1 << 32) if u >= (1 << 31) else u

    return remap(c) < remap(d)


# Candidate swap-decision models for the characterization report. b = VC,
# a = VD, ascending Mod1=1 (doc: VD gets min). No ties by stimuli design.
def _swap_doc(a_w, b_w):
    return _sign_mag_is_smaller(b_w, a_w)


def _swap_noswap_bothneg(a_w, b_w):
    if (a_w & 0x80000000) and (b_w & 0x80000000):
        return False
    return _swap_doc(a_w, b_w)


def _swap_noswap_anyneg(a_w, b_w):
    if (a_w & 0x80000000) or (b_w & 0x80000000):
        return False
    return _swap_doc(a_w, b_w)


def _swap_raw_unsigned(a_w, b_w):
    return b_w < a_w  # raw u32 compare, no sign remap


def _swap_magnitude(a_w, b_w):
    return (b_w & 0x7FFFFFFF) < (a_w & 0x7FFFFFFF)


CANDIDATE_MODELS = {
    "doc_ttsim_sign_mag": _swap_doc,
    "noswap_when_both_negative": _swap_noswap_bothneg,
    "noswap_when_any_negative": _swap_noswap_anyneg,
    "raw_unsigned_compare": _swap_raw_unsigned,
    "magnitude_only_compare": _swap_magnitude,
}


def _golden_strict(pairs):
    """Expected (val_words, idx_words) after the ascending compare-exchange
    under the doc/ttsim model, applied to the given pairs only. Positions
    outside the given pairs' windows are the identity."""
    val, idx = _input_words()
    for a_base, b_base in pairs:
        for k in range(LANES):
            if _swap_doc(val[a_base + k], val[b_base + k]):
                p, q = a_base + k, b_base + k
                val[p], val[q] = val[q], val[p]
                idx[p], idx[q] = idx[q], idx[p]
    return val, idx


def _expected_strict_swaps() -> int:
    val, _ = _input_words()
    a_base, b_base = STRICT_PAIR
    return sum(1 for k in range(LANES) if _swap_doc(val[a_base + k], val[b_base + k]))


# ---------------------------------------------------------------------------
# Build / run / cache
# ---------------------------------------------------------------------------

_RESULTS: dict[int, tuple[list[int], list[int]]] = {}


def _config(arm: int, nonce: int) -> TestConfig:
    src_A = torch.cat([_value_tile(), _index_tile()])
    src_B = torch.zeros(TILE_DATUMS, dtype=torch.float32)
    formats = InputOutputFormat(DataFormat.Float32, DataFormat.Float32)
    return TestConfig(
        "sources/topk_unfused_macro_probe_test.cpp",
        formats,
        templates=[
            generate_input_dim([32, 32], [32, 32]),
            TILIZE(Tilize.No),
            DEST_SYNC(DestSync.Half),
            PROBE_ARM(probe_arm=arm),
        ],
        runtimes=[
            # Freshness nonce riding the RELU_CONFIG runtime slot (the packer
            # never consumes it in this kernel; PACK echoes it into diag[2]).
            # Runtime args do not perturb the build variant, so the nonce can
            # differ per run without recompiling.
            RELU_CONFIG(nonce),
            NUM_FACES(num_faces=4),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            DataFormat.Float32,
            src_B,
            DataFormat.Float32,
            DataFormat.Float32,
            tile_count_A=2,
            tile_count_B=1,
            tile_count_res=RES_TILES,
        ),
        dest_acc=DestAccumulation.Yes,  # 32-bit Dst words
        unpack_to_dest=True,  # raw bit-exact FP32 copy into Dst
    )


def _run_arm(arm: int):
    """(val_words[1024], idx_words[1024]) as u32 lists. Memoized per session;
    the single real execution is nonce-verified fresh."""
    if arm in _RESULTS:
        return _RESULTS[arm]
    nonce = random.getrandbits(30)
    result = _config(arm, nonce).run().result
    words = result.view(torch.int32).tolist()
    words = [w & 0xFFFFFFFF for w in words]
    assert len(words) >= RES_TILES * TILE_DATUMS, "short result buffer"
    diag = words[DIAG_TILE * TILE_DATUMS :]
    assert diag[0] == SENTINEL_START and diag[12] == SENTINEL_END, (
        f"arm {arm}: kernel did not run to completion "
        f"(diag[0]=0x{diag[0]:08X}, diag[12]=0x{diag[12]:08X})"
    )
    assert diag[1] == arm, f"stale build: diag arm {diag[1]} != requested {arm}"
    assert diag[2] == nonce, (
        f"STALE RESULT: diag nonce 0x{diag[2]:08X} != this run's 0x{nonce:08X} "
        f"— the consumer served a cached/previous execution, not a fresh one"
    )
    out = (words[0:TILE_DATUMS], words[TILE_DATUMS : 2 * TILE_DATUMS])
    _RESULTS[arm] = out
    return out


def _prepare(*arms: int):
    """Build every needed variant before running any (the --compile-producer
    pattern the rebuild-macro mutation test uses)."""
    for arm in arms:
        _config(arm, nonce=0).prepare()


def _diff(got, want, tag_got, tag_want, positions=None):
    gv, gi = got
    wv, wi = want
    pos = range(TILE_DATUMS) if positions is None else positions
    mism = [
        ("val", p, f"0x{wv[p]:08X}", f"0x{gv[p]:08X}") for p in pos if gv[p] != wv[p]
    ] + [("idx", p, f"0x{wi[p]:08X}", f"0x{gi[p]:08X}") for p in pos if gi[p] != wi[p]]
    info = {
        "got": tag_got,
        "want": tag_want,
        "num_mismatches": len(mism),
        "mismatch_lane_mod4_histogram": _mod_hist(mism, 4),
        "mismatch_lane_mod8_histogram": _mod_hist(mism, 8),
        "first_mismatches": mism[:16],
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"probe_{tag_got}_vs_{tag_want}.json").write_text(
        json.dumps(info, indent=2)
    )
    return mism, info


def _mod_hist(mism, m):
    h = {i: 0 for i in range(m)}
    for _tag, p, *_rest in mism:
        h[p % m] += 1
    return h


def _characterize(got):
    """Classify the characterization pair's device behavior lane by lane.

    Returns (report_dict, inconsistent_lanes) where inconsistent_lanes are
    lanes whose INDEX movement did not follow their VALUE movement — an
    index-tracking desynchronisation, which is a hard failure regardless of
    which swap-decision semantics silicon implements.
    """
    val_in, idx_in = _input_words()
    gv, gi = got
    a_base, b_base = CHAR_PAIR
    lanes = []
    inconsistent = []
    model_miss: dict[str, list[int]] = {name: [] for name in CANDIDATE_MODELS}
    for k in range(LANES):
        p, q = a_base + k, b_base + k
        a_w, b_w = val_in[p], val_in[q]
        if gv[p] == a_w and gv[q] == b_w:
            v_state = "no_swap"
        elif gv[p] == b_w and gv[q] == a_w:
            v_state = "swap"
        else:
            v_state = "corrupt"
        if gi[p] == idx_in[p] and gi[q] == idx_in[q]:
            i_state = "no_swap"
        elif gi[p] == idx_in[q] and gi[q] == idx_in[p]:
            i_state = "swap"
        else:
            i_state = "corrupt"
        if v_state != i_state:
            inconsistent.append(k)
        for name, model in CANDIDATE_MODELS.items():
            if v_state == "corrupt" or (v_state == "swap") != model(a_w, b_w):
                model_miss[name].append(k)
        lanes.append(
            {
                "k": k,
                "class": (
                    "both_negative"
                    if k < 16
                    else ("mixed_sign" if k < 24 else "positive_control")
                ),
                "a": f"0x{a_w:08X}",
                "b": f"0x{b_w:08X}",
                "value": v_state,
                "index": i_state,
                "doc_model_says_swap": _swap_doc(a_w, b_w),
            }
        )
    fitting = [name for name, miss in model_miss.items() if not miss]
    report = {
        "verdict_models_fitting_all_lanes": fitting,
        "model_mismatch_lanes": model_miss,
        "index_follows_value_everywhere": not inconsistent,
        "lanes": lanes,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "characterization_signed_pairs.json").write_text(
        json.dumps(report, indent=2)
    )
    return report, inconsistent


# ---------------------------------------------------------------------------
# Tests. Each is independently runnable; results are cached across them within
# one consumer session (freshness proven by the per-run nonce).
# ---------------------------------------------------------------------------


@blackhole_only
def test_probe_harness_software_reference():
    """Arm 0 (software SFPSWAP under index tracking — the shipping unfused
    primitive) must match the positional doc-model golden on the ALL-POSITIVE
    strict window and leave every filler word untouched. The signed
    characterization window is NOT positionally asserted (silicon's negative-
    pair semantics are under characterization — see the JSON report); it IS
    asserted for value/index consistency: whatever silicon decided per lane,
    the index must have moved with the value, or shipping index tracking
    itself is desynchronised and the unfused op is in trouble far beyond any
    macro port. If THIS test fails, every other verdict in the file is void."""
    _prepare(0)
    got = _run_arm(0)
    want = _golden_strict([STRICT_PAIR])
    char_positions = set(range(WIN_A1, WIN_A1 + LANES)) | set(
        range(WIN_B1, WIN_B1 + LANES)
    )
    strict_positions = [p for p in range(TILE_DATUMS) if p not in char_positions]
    n_swap = _expected_strict_swaps()
    assert 0 < n_swap < LANES, "stimuli must mix swapping and non-swapping lanes"
    mism, info = _diff(got, want, "arm0", "golden_strict", positions=strict_positions)
    assert not mism, (
        f"software reference disagrees with the doc-model golden on the "
        f"ALL-POSITIVE strict window / fillers ({info['num_mismatches']} words). "
        f"Positives matched this model on the first silicon run, so suspect the "
        f"harness (mapping, CC state) — check the lane-mod histograms in the "
        f"report: mod4={info['mismatch_lane_mod4_histogram']}. "
        f"First: {info['first_mismatches'][:6]}"
    )
    report, inconsistent = _characterize(got)
    assert not inconsistent, (
        f"INDEX TRACKING DESYNC in the shipping software primitive: lanes "
        f"{inconsistent} of the signed characterization pair moved values "
        f"without their indices (or vice versa). This breaks the unfused "
        f"topk_xl contract itself — escalate independently of the macro port."
    )
    # The signed-pair semantics verdict is informational here (silicon vs
    # candidate models); the port only needs macro == software.
    print(
        "\nsigned-pair characterization: models fitting all lanes = "
        f"{report['verdict_models_fitting_all_lanes']} "
        f"(full detail: {REPORT_DIR / 'characterization_signed_pairs.json'})"
    )


@blackhole_only
def test_probe_macro_swap_honours_index_tracking():
    """THE yes/no arm. Arm 1 (one macro-scheduled SFPSWAP) must equal arm 0
    byte-for-byte on both tiles — signed lanes included, since the comparison
    is device-vs-device and needs no semantic model. Failure classification
    in the message."""
    _prepare(0, 1)
    got, ref = _run_arm(1), _run_arm(0)
    mism, info = _diff(got, ref, "arm1", "arm0")
    if mism:
        # Classify: did the VALUE half track arm 0 while the INDEX half
        # stayed at the input? That means the scheduled swap ran but the
        # companion (index-tracking) write was suppressed under the macro.
        val_ok = all(g == r for g, r in zip(got[0], ref[0]))
        idx_in = _input_words()[1]
        idx_stale = all(g == i for g, i in zip(got[1], idx_in))
        verdict = (
            "swap executed but companion index swap SUPPRESSED under the macro "
            "-- index tracking is NOT honoured; unfused macro port must keep "
            "swaps in software (see ceiling arithmetic in the port plan)"
            if (val_ok and idx_stale)
            else "macro-scheduled swap misbehaved beyond the companion write"
        )
        pytest.fail(
            f"arm1 != arm0 ({info['num_mismatches']} words): {verdict}. "
            f"First: {info['first_mismatches'][:6]}"
        )


@blackhole_only
def test_probe_macro_dual_two_slots_apart():
    """Arm 2: the fused merge's interleave rule (macros 2 issue slots apart,
    2 drain SFPNOPs) under index tracking. A FAILURE here while
    test_probe_macro_dual_three_slots_apart passes means tracking extends the
    Simple-unit occupancy and the unfused port needs a 3-slot interleave —
    a scheduling change, not a dead end."""
    _prepare(0, 2)
    mism, info = _diff(_run_arm(2), _run_arm(0), "arm2", "arm0")
    assert not mism, (
        f"2-slot macro separation corrupts under index tracking "
        f"({info['num_mismatches']} words) -- check arm3 before concluding "
        f"anything. First: {info['first_mismatches'][:6]}"
    )


@blackhole_only
def test_probe_macro_dual_three_slots_apart():
    """Arm 3: same two macro swaps with one extra slot between them. If arm 1
    passes, this must pass — its failure with arm 1 green would mean the
    hazard is inter-macro rather than occupancy, and the port needs per-body
    single-macro scheduling."""
    _prepare(0, 3)
    mism, info = _diff(_run_arm(3), _run_arm(0), "arm3", "arm0")
    assert not mism, (
        f"3-slot macro separation still corrupts ({info['num_mismatches']} "
        f"words). First: {info['first_mismatches'][:6]}"
    )


@blackhole_only
def test_probe_macro_full_trick_value_store():
    """Arm 4: value stores ride the macros' Store slots (delay 2). Passing
    means the merge's full trick transfers to the unfused value half — the
    macro store reads macroVD AFTER the swap's write, tracking active."""
    _prepare(0, 4)
    mism, info = _diff(_run_arm(4), _run_arm(0), "arm4", "arm0")
    assert not mism, (
        f"macro value store under index tracking is wrong "
        f"({info['num_mismatches']} words) -- if only the two macroVD value "
        f"windows differ, the store fired before the swap's write (timing); "
        f"stores must stay software. First: {info['first_mismatches'][:6]}"
    )


@blackhole_only
def test_probe_mutation_control():
    """Arm 5 zeroes both swap macros' Sequence words: each SFPLOADMACRO
    degenerates into a plain SFPLOAD (the documented "schedule nothing"
    failure mode that timing cannot see), no compare-exchange runs anywhere,
    and the body's write-back stores make the expected output exactly the RAW
    INPUT. Assert both halves: arm5 == identity (the degeneration is
    modelable) and arm5 != arm0 (the differential is sensitive to a missing
    swap). If arm5 matched arm 0, every green macro arm above would prove
    nothing.

    (History: the first mutation cleared only the Simple byte's 0x80 bit,
    producing SFPSWAP(VC == VD) — a same-register two-cycle read-modify-write
    with no architectural contract. Silicon emitted a garbage broadcast
    (0xBF2CC4C7), so that mutation could not serve as a control.)"""
    _prepare(0, 5)
    got, ref = _run_arm(5), _run_arm(0)
    val_in, idx_in = _input_words()
    mism_id, info_id = _diff(got, (val_in, idx_in), "arm5", "identity")
    assert not mism_id, (
        f"schedule-nothing macro did not degenerate to a plain load "
        f"({info_id['num_mismatches']} words differ from the raw input) -- "
        f"the mutation is not modelable and cannot serve as a control. "
        f"First: {info_id['first_mismatches'][:6]}"
    )
    mism_ref, _ = _diff(got, ref, "arm5", "arm0")
    assert mism_ref, (
        f"MUTATION NOT DETECTED: arm5 == arm0 with {_expected_strict_swaps()} "
        "strict-window lanes expected to swap. The differential is blind to "
        "the scheduled swap; every PASS in this file is void."
    )


@blackhole_only
def test_probe_index_load_and_deferred_store_on_macro():
    """Arm 6 vs arm 7: the run-B index (LREG6) is LOADED by an SFPLOADMACRO
    (macroVD in LREG4..7 — outside TEN-2932's allowed instruction list) and
    its Dst word is written ONLY by that macro's store, deferred (delay 6)
    past the companion swap. PASS means unfused index stores can ride macros —
    the TEN-2932 mechanism is exactly the one the port wants. Failure modes
    are classified in the message. (This pair operates on the all-positive
    strict window, so the doc-model golden pin on arm 7 is valid.)"""
    _prepare(6, 7)
    got, ref = _run_arm(6), _run_arm(7)
    mism, info = _diff(got, ref, "arm6", "arm7")
    if mism:
        val_in, idx_in = _input_words()
        b0 = range(WIN_B0, WIN_B0 + LANES)
        only_b0_idx = all(t == "idx" and p in b0 for (t, p, *_rest) in mism)
        pre_swap = all(got[1][p] == idx_in[p] for p in b0)
        if only_b0_idx and pre_swap:
            verdict = (
                "deferred macro store emitted the PRE-swap index: the store "
                "latched its datum too early or the companion write landed "
                "after delay 6 -- retry with a larger delay before declaring "
                "the mechanism dead"
            )
        elif only_b0_idx:
            verdict = (
                "macro load/store on an index register is corrupt -- "
                "consistent with TEN-2932 (SFPLOADMACRO is not in the "
                "SFPLOAD/SFPLOADI/SFPSWAP/SFPTRANSP allowed list); index "
                "loads/stores must stay software"
            )
        else:
            verdict = (
                "corruption beyond the macro-carried index window -- the "
                "index-register macro perturbs unrelated state; do not use"
            )
        pytest.fail(
            f"arm6 != arm7 ({info['num_mismatches']} words): {verdict}. "
            f"First: {info['first_mismatches'][:6]}"
        )
    # Also pin arm 7 itself to the single-pair doc-model golden — valid
    # because the (A0, B0) pair is all-positive by design.
    want = _golden_strict([STRICT_PAIR])
    mism7, info7 = _diff(ref, want, "arm7", "golden_single")
    assert not mism7, (
        f"single-pair software reference disagrees with golden "
        f"({info7['num_mismatches']} words). First: {info7['first_mismatches'][:6]}"
    )
