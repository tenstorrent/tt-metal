# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Does a macro-scheduled SFPSWAP honour SFPU index tracking? (Blackhole)

THE GATING QUESTION — ANSWERED YES ON SILICON (2026-08-16)
----------------------------------------------------------
The topk_xl merge/rebuild SFPLOADMACRO win was fused-only; the shipping
consumer (``ttnn.experimental.topk_large_indices``) runs merge/rebuild UNFUSED
with indices riding LREG4..7 through SFPU index-tracking mode
(``LaneConfig.ENABLE_DEST_INDEX``). This probe established, differentially
(macro arm == software arm, byte-for-byte, mutation-sensitivity proven,
per-run nonce freshness): a macro-scheduled SFPSWAP performs the companion
swap; the fused 2-slot interleave + 2-drain rule transfers; the macroVD store
rides at delay 2; an index-register macro load with a deferred store is clean
(TEN-2932 does not bite the load path). The unfused port in
``ckernel_sfpu_topk_xl.h`` is built on exactly the arm-1/2/4 pattern.

THE DST LANE MAPPING, LEARNED THE HARD WAY (this file's golden, v3)
-------------------------------------------------------------------
An SFPU load/store at dest-unit offset ``u`` covers the 32 lanes

    datum = (u >> 2) * 64  +  16 * r  +  2 * c  +  ((u >> 1) & 1)
            for r in 0..3, c in 0..7

i.e. FOUR rows x the EVEN columns; offset +2 addresses the ODD columns of the
same rows. This is what the shipping header's ``set_dst_write_addr_offset``
comment ("switch between the even and odd columns ... offsets +0 and +2")
describes, and it is why every merge/rebuild runs its loops twice, at +0 and
+2. Two earlier revisions of this golden assumed contiguous
``[u*16, u*16+32)`` windows and mis-attributed the resulting mismatches:
  v1: negation was aliased onto k%4==3, so the odd-position (never-loaded)
      datums looked like a negative-pair semantics difference;
  v2: all-positive stimuli exposed the truth — mismatches landed on exactly
      the ODD datum positions of the golden-swap lanes (positions 3,9,15,
      21,27 and partners, mod-8 histogram all-odd), values bit-equal to the
      INPUT: the device had simply never touched them.
The SFPSWAP semantics (doc SFPSWAP.md == ttsim sign_mag32_total_order) were
never wrong. The signed-pair characterization below is expected to fit the
doc model now that the mapping is right; if it does not, THAT is a real
doc/silicon divergence worth reporting upstream.

FRESHNESS. Each arm echoes a per-run RUNTIME nonce (RELU_CONFIG slot) into
diag[2]; a cached result cannot echo this run's value. Results are memoized
in-process across tests of one consumer session (hence the fast session); the
nonce proves the one real execution per arm was fresh.

Arms (see sources/topk_unfused_macro_probe_test.cpp):
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
LANES = 32

SENTINEL_START = 0xC0DEBA5E
SENTINEL_END = 0xC0DEE0D1

REPORT_DIR = Path(
    os.environ.get("TOPK_PROBE_REPORT_DIR", "/tmp/topk_unfused_macro_probe")
)


def _lanes(u: int) -> list[int]:
    """Datum positions covered by an SFPU load/store at dest-unit offset u.

    Four rows x the even columns, plus the odd-column select from offset
    bit 1 (unused by this probe — all offsets here are multiples of 4).
    Lane enumeration order is (r, c) row-major; only CONSISTENCY across
    loads matters (pairing and swaps are lanewise), not the absolute order.
    """
    base = (u >> 2) * 64 + ((u >> 1) & 1)
    return [base + 16 * r + 2 * c for r in range(4) for c in range(8)]


# Dst geometry — MUST match sources/topk_unfused_macro_probe_test.cpp.
LANES_A0 = _lanes(0)  # STRICT window: value/index run A  (dest offset 0)
LANES_A1 = _lanes(4)  # CHARACTERIZATION window: run A    (dest offset 4)
LANES_B0 = _lanes(16)  # STRICT window: run B             (dest offset 16)
LANES_B1 = _lanes(20)  # CHARACTERIZATION window: run B   (dest offset 20)

STRICT_PAIR = (LANES_A0, LANES_B0)  # all-positive
CHAR_PAIR = (LANES_A1, LANES_B1)  # signed/mixed, model-classified


@dataclass
class PROBE_ARM(TemplateParameter):
    """Emits ``#define PROBE_ARM <n>`` — selects the MATH body under test."""

    probe_arm: int = 0

    def convert_to_cpp(self) -> str:
        return f"#define PROBE_ARM {self.probe_arm}"


# ---------------------------------------------------------------------------
# Stimuli.
#
# STRICT window (A0/B0): all-positive, distinct, no ties. Swap on k % 3 == 0
# (11 of 32 lanes) — not aliased to any mod-2/4/8 lane structure, so a
# lane-patterned effect shows up with a recognisable positional histogram
# instead of masquerading as semantics.
#
# CHARACTERIZATION window (A1/B1):
#   k  0..15  both-negative pairs   (doc model: swap iff |b| > |a|)
#   k 16..23  mixed-sign pairs      (doc model: always swap; b < 0 < a)
#             k 16..19 with |b| < |a|, k 20..23 with |b| > |a|
#   k 24..31  positive controls     (doc model: swap iff k odd)
# ---------------------------------------------------------------------------


def _pair_values(k: int, char: bool):
    """(a, b) for lane k of a window pair. a -> run A (VD), b -> run B (VC)."""
    if not char:
        a = float(2 + 3 * k)
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
        a, b = _pair_values(k, char=False)
        vals[LANES_A0[k]] = a
        vals[LANES_B0[k]] = b
        a, b = _pair_values(k, char=True)
        vals[LANES_A1[k]] = a
        vals[LANES_B1[k]] = b
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


def _swap_raw_unsigned(a_w, b_w):
    return b_w < a_w  # raw u32 compare, no sign remap


def _swap_magnitude(a_w, b_w):
    return (b_w & 0x7FFFFFFF) < (a_w & 0x7FFFFFFF)


CANDIDATE_MODELS = {
    "doc_ttsim_sign_mag": _swap_doc,
    "noswap_when_both_negative": _swap_noswap_bothneg,
    "raw_unsigned_compare": _swap_raw_unsigned,
    "magnitude_only_compare": _swap_magnitude,
}


def _golden(pairs):
    """Expected (val_words, idx_words) after the ascending compare-exchange
    under the doc/ttsim model, applied to the given (lanesA, lanesB) pairs.
    All other positions are the identity."""
    val, idx = _input_words()
    for lanes_a, lanes_b in pairs:
        for k in range(LANES):
            p, q = lanes_a[k], lanes_b[k]
            if _swap_doc(val[p], val[q]):
                val[p], val[q] = val[q], val[p]
                idx[p], idx[q] = idx[q], idx[p]
    return val, idx


def _expected_strict_swaps() -> int:
    val, _ = _input_words()
    lanes_a, lanes_b = STRICT_PAIR
    return sum(1 for k in range(LANES) if _swap_doc(val[lanes_a[k]], val[lanes_b[k]]))


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
            # Runtime args do not perturb the build variant.
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
    index-tracking desynchronisation, a hard failure regardless of which
    swap-decision semantics silicon implements.
    """
    val_in, idx_in = _input_words()
    gv, gi = got
    lanes_a, lanes_b = CHAR_PAIR
    lanes = []
    inconsistent = []
    model_miss: dict[str, list[int]] = {name: [] for name in CANDIDATE_MODELS}
    for k in range(LANES):
        p, q = lanes_a[k], lanes_b[k]
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
# Tests. Results are cached across them within one consumer session
# (freshness proven by the per-run nonce).
# ---------------------------------------------------------------------------


@blackhole_only
def test_probe_harness_software_reference():
    """Arm 0 (software SFPSWAP under index tracking — the shipping unfused
    primitive) must match the doc-model positional golden EVERYWHERE, both
    pairs, under the interleaved even-column lane mapping. The signed
    characterization pair is additionally classified against the candidate
    semantic models (JSON report) — with the mapping fixed, the doc/ttsim
    model is expected to fit; anything else is a genuine silicon-vs-doc
    divergence to report upstream. If THIS test fails, every other verdict
    in the file is void."""
    _prepare(0)
    got = _run_arm(0)
    n_swap = _expected_strict_swaps()
    assert 0 < n_swap < LANES, "stimuli must mix swapping and non-swapping lanes"
    want = _golden([STRICT_PAIR, CHAR_PAIR])
    mism, info = _diff(got, want, "arm0", "golden_both")
    report, inconsistent = _characterize(got)
    assert not inconsistent, (
        f"INDEX TRACKING DESYNC in the shipping software primitive: lanes "
        f"{inconsistent} of the signed characterization pair moved values "
        f"without their indices (or vice versa). This breaks the unfused "
        f"topk_xl contract itself — escalate independently of the macro port."
    )
    assert not mism, (
        f"software reference disagrees with the doc-model golden "
        f"({info['num_mismatches']} words). Check the positional histograms "
        f"(mod4={info['mismatch_lane_mod4_histogram']}, "
        f"mod8={info['mismatch_lane_mod8_histogram']}): all-odd positions "
        f"means the lane mapping regressed; signed-lane-only mismatches mean "
        f"a real semantics divergence — see the characterization verdict "
        f"{report['verdict_models_fitting_all_lanes']} in "
        f"{REPORT_DIR / 'characterization_signed_pairs.json'}. "
        f"First: {info['first_mismatches'][:6]}"
    )


@blackhole_only
def test_probe_macro_swap_honours_index_tracking():
    """THE yes/no arm. Arm 1 (one macro-scheduled SFPSWAP) must equal arm 0
    byte-for-byte on both tiles — a device-vs-device comparison needing no
    semantic model. Failure classification in the message."""
    _prepare(0, 1)
    got, ref = _run_arm(1), _run_arm(0)
    mism, info = _diff(got, ref, "arm1", "arm0")
    if mism:
        val_ok = all(g == r for g, r in zip(got[0], ref[0]))
        idx_in = _input_words()[1]
        idx_stale = all(g == i for g, i in zip(got[1], idx_in))
        verdict = (
            "swap executed but companion index swap SUPPRESSED under the macro "
            "-- index tracking is NOT honoured; unfused macro port must keep "
            "swaps in software"
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
    2 drain SFPNOPs) under index tracking. A FAILURE here while the 3-slot
    arm passes means tracking extends the Simple-unit occupancy and the
    unfused port needs a 3-slot interleave — a scheduling change, not a
    dead end."""
    _prepare(0, 2)
    mism, info = _diff(_run_arm(2), _run_arm(0), "arm2", "arm0")
    assert not mism, (
        f"2-slot macro separation corrupts under index tracking "
        f"({info['num_mismatches']} words) -- check arm3 before concluding "
        f"anything. First: {info['first_mismatches'][:6]}"
    )


@blackhole_only
def test_probe_macro_dual_three_slots_apart():
    """Arm 3: same two macro swaps, one extra slot between them. If arm 1
    passes, this must pass — failure with arm 1 green would mean the hazard
    is inter-macro rather than occupancy."""
    _prepare(0, 3)
    mism, info = _diff(_run_arm(3), _run_arm(0), "arm3", "arm0")
    assert not mism, (
        f"3-slot macro separation still corrupts ({info['num_mismatches']} "
        f"words). First: {info['first_mismatches'][:6]}"
    )


@blackhole_only
def test_probe_macro_full_trick_value_store():
    """Arm 4: value stores ride the macros' Store slots (delay 2). Passing
    means the merge's full trick transfers to the unfused value half."""
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
    failure mode that timing cannot see), no compare-exchange runs, and the
    body's write-back stores make the expected output exactly the RAW INPUT.
    Assert both halves: arm5 == identity (the degeneration is modelable) and
    arm5 != arm0 (the differential is sensitive to a missing swap).

    (History: a 0x80-bit mutation produced SFPSWAP(VC == VD) — a
    same-register two-cycle read-modify-write with no architectural
    contract; silicon emitted a garbage broadcast (0xBF2CC4C7), so that
    mutation could not serve as a control.)"""
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
    past the companion swap. PASS means index loads/stores can ride macros.
    (The pair is the all-positive strict window, so the doc-model golden pin
    on arm 7 is valid.)"""
    _prepare(6, 7)
    got, ref = _run_arm(6), _run_arm(7)
    mism, info = _diff(got, ref, "arm6", "arm7")
    if mism:
        val_in, idx_in = _input_words()
        b0 = set(LANES_B0)
        only_b0_idx = all(t == "idx" and p in b0 for (t, p, *_rest) in mism)
        pre_swap = all(got[1][p] == idx_in[p] for p in LANES_B0)
        if only_b0_idx and pre_swap:
            verdict = (
                "deferred macro store emitted the PRE-swap index: retry with "
                "a larger delay before declaring the mechanism dead"
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
    # Also pin arm 7 itself to the single-pair doc-model golden.
    want = _golden([STRICT_PAIR])
    mism7, info7 = _diff(ref, want, "arm7", "golden_single")
    assert not mism7, (
        f"single-pair software reference disagrees with golden "
        f"({info7['num_mismatches']} words). First: {info7['first_mismatches'][:6]}"
    )
