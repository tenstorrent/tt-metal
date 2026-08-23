# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""X6 FPU face-transpose arsenal: SIM battery (lane FV, 2026-08-22).

Runs sources/sfpu_facetranspose_probe.cpp on the pinned simulator and
compares EVERY lane against helpers/facetranspose_oracle.py (transcribed
from the tt-isa-documentation MOVD2B/MOVB2A/MOVB2D/MOVA2D/TRNSPSRCB
functional models -- the WormholeB0 pages carrying the Blackhole arms;
BlackholeA0 has no pages for this family, so the pinned sim is the BH
oracle and any doc-vs-sim divergence is a reportable finding).

The tensor<->(vector row, lane) mapping is calibrated empirically
(identity/rowtag/lanetag -- the lane FB method); the (vector row, lane)
<-> Dst32b (row, column) map is the SFPLOAD.md doc model

    Dst row = 4*(vrow//2) + lane//8,  column = (lane%8)*2 + vrow%2

whose validity the full-face checks prove implicitly: a 16x16 transpose
verified pointwise over random data on all 256 positions cannot pass
under a wrong map.

Ambiguity policy: MOVB2D's masking arm keys on the implied SrcBFmt of a
dummy-validated bank (NonContractualBehavior on BH).  The Dst-row
roundtrip and hi-stage modes adjudicate the sim's arm empirically among
the oracle's candidates and print the winner as an arsenal FACT; the
END-TO-END transpose is candidate-independent (host theorem).

Run: pytest -q --run-simulator test_crosslane_facetranspose.py
"""

from dataclasses import dataclass

import pytest
import torch
from helpers import crosslane_oracle as co
from helpers import facetranspose_oracle as fo
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import TemplateParameter

M32 = 0xFFFFFFFF
ELEMS = 1024
ROWS = 16
LANES = 32

SRCB_FMT_FACT = {"winner": None}


@dataclass
class PROBE_MODE(TemplateParameter):
    probe_mode: int

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t PROBE_MODE = {self.probe_mode}u;"


def run_probe(mode, input_vec):
    formats = InputOutputFormat(DataFormat.UInt32, DataFormat.UInt32)
    src = torch.tensor(input_vec, dtype=torch.int64)
    config = TestConfig(
        "sources/sfpu_facetranspose_probe.cpp",
        formats,
        templates=[PROBE_MODE(mode)],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            src,
            formats.input_format,
            torch.zeros_like(src),
            formats.input_format,
            formats.output_format,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=4,
        ),
        dest_acc=DestAccumulation.Yes,
        unpack_to_dest=True,
        disable_format_inference=True,
        compile_time_formats=True,
    )
    res = config.run().result
    assert len(res) == 4 * ELEMS, f"expected 4 tiles back, got {len(res)}"
    return [int(v) & M32 for v in res]


# ---------------------------------------------------------------------------
# calibration (lane FB pattern, module-scoped)
# ---------------------------------------------------------------------------


class Cal:
    T = None
    m1inv = None


@pytest.fixture(scope="module")
def cal():
    if Cal.T is not None:
        return Cal
    ramp = list(range(ELEMS))
    rowtag = run_probe(1, ramp)
    lanetag = run_probe(2, ramp)
    ident = run_probe(0, ramp)

    row_positions = {}
    for pos, v in enumerate(rowtag):
        if 0x00A00000 <= v < 0x00A00000 + ROWS:
            row_positions.setdefault(v - 0x00A00000, []).append(pos)
    assert sorted(row_positions.keys()) == list(range(ROWS))
    for i, positions in row_positions.items():
        assert len(positions) == LANES, f"row {i}: {len(positions)} tagged"

    T = {}
    for i, positions in row_positions.items():
        tags = [lanetag[p] for p in positions]
        assert sorted(tags) == [2 * l for l in range(LANES)], (
            f"row {i}: vConstTileId tags != 2*lane (finding!)"
        )
        for k, p in enumerate(positions):
            T[(i, tags[k] // 2)] = p

    m1inv = {}
    for i in range(ROWS):
        for l in range(LANES):
            v = ident[T[(i, l)]]
            assert 0 <= v < ELEMS
            m1inv[(i, l)] = v
    assert len(set(m1inv.values())) == ROWS * LANES

    Cal.T, Cal.m1inv = T, m1inv
    return Cal


def build_input(cal, rows):
    vec = [0] * ELEMS
    for i, lane_vals in rows.items():
        for l in range(LANES):
            vec[cal.m1inv[(i, l)]] = lane_vals[l] & M32
    return vec


def read_rows(cal, out, indices):
    return {i: [out[cal.T[(i, l)]] for l in range(LANES)] for i in indices}


def check_rows(mode, got, want, note=""):
    bad = []
    for i in sorted(want.keys()):
        for l in range(LANES):
            w = want[i][l]
            if w is None:
                continue
            if got[i][l] != (w & M32):
                bad.append((i, l, got[i][l], w & M32))
    if bad:
        print(f"ORACLE MISMATCH mode={mode} {note}: {len(bad)} lanes")
        for i, l, g, w in bad[:16]:
            print(f"  row={i} lane={l} got={g:08x} want={w:08x}")
        for i in sorted(want.keys()):
            print(f"  got  row{i}: " + " ".join(f"{v:08x}" for v in got[i]))
            print(
                f"  want row{i}: "
                + " ".join(
                    "????????" if v is None else f"{v & M32:08x}" for v in want[i]
                )
            )
    assert not bad, f"mode {mode} {note}: sim disagrees with oracle"


# ---------------------------------------------------------------------------
# The doc-model (vrow, lane) <-> Dst32b (row, col) map (SFPLOAD.md).
# ---------------------------------------------------------------------------


def dst_pos(vrow, lane):
    return (4 * (vrow // 2) + lane // 8, (lane % 8) * 2 + vrow % 2)


def rows_to_face(rows, vrow_base=0):
    """rows: dict vrow -> [32 lane values] covering one 16-Dst-row face
    (vrows vrow_base..vrow_base+7) -> 16x16 Dst face array."""
    face = [[0] * 16 for _ in range(16)]
    base_row = 4 * (vrow_base // 2)
    for i in range(8):
        for l in range(LANES):
            r, c = dst_pos(vrow_base + i, l)
            face[r - base_row][c] = rows[vrow_base + i][l] & M32
    return face


def face_to_rows(face, vrow_base=0):
    """16x16 Dst face -> dict vrow -> [32 lane values]."""
    out = {}
    for i in range(8):
        vals = []
        for l in range(LANES):
            r, c = dst_pos(vrow_base + i, l)
            vals.append(face[r - 4 * (vrow_base // 2)][c] & M32)
        out[vrow_base + i] = vals
    return out


def stimuli(nrows, tag):
    sent = {i: [s ^ (i << 16) for s in co.lane_id_sentinels(tag)] for i in range(nrows)}
    varied = {i: co.varied_stimulus(i, seed=tag + 100) for i in range(nrows)}
    return [("sentinel", sent), ("varied", varied)]


def adversarial_rows(nrows, seed):
    """Varied values with planted zero-low-byte lo16 halves and
    all-zero-exponent hi16 halves -- the flush-arm victims."""
    rows = {i: co.varied_stimulus(i, seed=seed) for i in range(nrows)}
    for i in rows:
        vals = list(rows[i])
        for l in range(0, LANES, 3):
            vals[l] = (vals[l] & 0xFFFF0000) | ((vals[l] & 0xFF00) & 0xFFFF)
        for l in range(1, LANES, 5):
            vals[l] = vals[l] & 0x0000FFFF  # hi16 zero: TF32 image low byte 0
        rows[i] = [v & M32 for v in vals]
    return rows


# ---------------------------------------------------------------------------
# host battery (no sim)
# ---------------------------------------------------------------------------


def test_host_bitexact_theorem():
    import random

    rng = random.Random(0xFACE)
    for trial in range(8):
        face = [[rng.getrandbits(32) for _ in range(16)] for _ in range(16)]
        ok, fmt, got, want = fo.theorem_bitexact_transpose(face)
        assert ok, f"trial {trial}: composition != transpose under fmt={fmt}"
    # Adversarial patterns too (zero-low-byte shuffle images).
    face = [[((r * 16 + c) << 16) | ((c * 0x100) & 0xFFFF) for c in range(16)]
            for r in range(16)]
    ok, fmt, _, _ = fo.theorem_bitexact_transpose(face)
    assert ok, f"adversarial: composition != transpose under fmt={fmt}"
    # The contract edge's negative half: FP16-class SrcBFmt corrupts.
    assert fo.theorem_fp16_srcbfmt_corrupts(face), (
        "fp16 SrcBFmt unexpectedly exact -- the contract edge note is stale"
    )


def test_host_flush_necessity():
    face = [[((r * 16 + c) << 16) | (c * 0x100) for c in range(16)]
            for r in range(16)]
    victims = fo.flush_victims(face)
    assert victims, (
        "zero-flag twin has no victims on the adversarial face -- the "
        "contract-necessity probe would be vacuous"
    )


# ---------------------------------------------------------------------------
# sim battery
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stim_name_idx", [0, 1], ids=["sentinel", "varied"])
def test_dstrow_roundtrip_and_fmt_fact(cal, stim_name_idx):
    name, rows = stimuli(8, tag=61)[stim_name_idx]
    vec = build_input(cal, rows)
    out = run_probe(3, vec)
    got = read_rows(cal, out, range(8))
    cands = {}
    for fmt in ("tf32", "fp16", "other"):
        want = {}
        for j in range(8):
            want[j] = []
            for l in range(LANES):
                v = rows[j][l] & M32
                sb = fo.shuffle_tf32((v >> 13) & fo.M19)
                if fmt == "fp16":
                    sb &= 0x7FF1F
                elif fmt != "tf32":
                    sb &= 0x7F8FF
                val16 = fo.remove_low_mantissa(sb)
                lm = (sb >> 8) & 7
                want[j].append(((val16 << 16) | (lm << 13)) & M32)
        cands[fmt] = want
    matches = [
        fmt
        for fmt, want in cands.items()
        if all(got[j][l] == want[j][l] for j in range(8) for l in range(LANES))
    ]
    # 'tf32' and 'other' coincide unless bits 15..13 differ; collapse dups.
    assert matches, f"dstrow roundtrip matches NO SrcBFmt candidate ({name})"
    winner = matches[0] if len(matches) == 1 else "+".join(sorted(matches))
    if SRCB_FMT_FACT["winner"] is None:
        SRCB_FMT_FACT["winner"] = winner
        print(f"SRCB-FMT-FACT: implied SrcBFmt masking arm = {winner}")
    else:
        assert SRCB_FMT_FACT["winner"] == winner, "SrcBFmt fact inconsistent"


@pytest.mark.parametrize("mode,face_idx", [(4, 0), (5, 1)],
                         ids=["face0", "face1"])
@pytest.mark.parametrize("stim_name_idx", [0, 1], ids=["sentinel", "varied"])
def test_face_transpose(cal, mode, face_idx, stim_name_idx):
    name, rows = stimuli(16, tag=62 + mode)[stim_name_idx]
    vec = build_input(cal, rows)
    out = run_probe(mode, vec)
    got = read_rows(cal, out, range(16))
    vb = 8 * face_idx
    face = rows_to_face({i: rows[i] for i in range(vb, vb + 8)}, vrow_base=vb)
    tface = [[face[j][i] & M32 for j in range(16)] for i in range(16)]
    want = dict(rows)  # untouched face keeps its input
    want.update(face_to_rows(tface, vrow_base=vb))
    check_rows(mode, got, want, f"{name} face{face_idx}")


@pytest.mark.parametrize("stim_name_idx", [0, 1], ids=["sentinel", "varied"])
def test_face_transpose_batch2(cal, stim_name_idx):
    name, rows = stimuli(16, tag=70)[stim_name_idx]
    vec = build_input(cal, rows)
    out = run_probe(6, vec)
    got = read_rows(cal, out, range(16))
    want = {}
    for fidx in (0, 1):
        vb = 8 * fidx
        face = rows_to_face({i: rows[i] for i in range(vb, vb + 8)}, vrow_base=vb)
        tface = [[face[j][i] & M32 for j in range(16)] for i in range(16)]
        want.update(face_to_rows(tface, vrow_base=vb))
    check_rows(6, got, want, f"{name} batch2")


def test_face_transpose_adversarial(cal):
    rows = adversarial_rows(16, seed=1234)
    vec = build_input(cal, rows)
    out = run_probe(4, vec)
    got = read_rows(cal, out, range(16))
    face = rows_to_face({i: rows[i] for i in range(8)}, vrow_base=0)
    tface = [[face[j][i] & M32 for j in range(16)] for i in range(16)]
    want = dict(rows)
    want.update(face_to_rows(tface, vrow_base=0))
    check_rows(4, got, want, "adversarial (zero-flag protected)")


@pytest.mark.parametrize("stim_name_idx", [0, 1], ids=["sentinel", "varied"])
def test_hi_stage_truncation(cal, stim_name_idx):
    name, rows = stimuli(8, tag=77)[stim_name_idx]
    vec = build_input(cal, rows)
    out = run_probe(7, vec)
    got = read_rows(cal, out, range(8))
    face = rows_to_face({i: rows[i] for i in range(8)}, vrow_base=0)
    matches = []
    per_fmt = {}
    for fmt in ("tf32", "fp16", "other"):
        hface = fo.face_transpose_32b_hi_stage(face, srcb_fmt=fmt)
        want = face_to_rows(hface, vrow_base=0)
        per_fmt[fmt] = want
        if all(
            got[j][l] == (want[j][l] & M32)
            for j in range(8)
            for l in range(LANES)
        ):
            matches.append(fmt)
    if not matches:
        check_rows(7, got, per_fmt["other"], f"{name} (shown vs 'other')")
    winner = matches[0] if len(matches) == 1 else "+".join(sorted(matches))
    print(f"HI-STAGE SRCB-FMT-FACT: {winner}")


def test_zeroflag_twin_flush(cal):
    rows = adversarial_rows(8, seed=4321)
    vec = build_input(cal, rows)
    out = run_probe(8, vec)
    got = read_rows(cal, out, range(8))
    face = rows_to_face({i: rows[i] for i in range(8)}, vrow_base=0)
    victims = fo.flush_victims(face)
    assert victims, "adversarial face produced no flush victims (vacuous)"
    fmt = "other"
    if SRCB_FMT_FACT["winner"] and "tf32" in SRCB_FMT_FACT["winner"] and \
       "+" not in SRCB_FMT_FACT["winner"]:
        fmt = "tf32"
    fface = fo.face_transpose_32b(face, srcb_fmt=fmt, zero_flag_disabled=False)
    want = face_to_rows(fface, vrow_base=0)
    check_rows(8, got, want, "zero-flag OFF twin (flush arm live)")
    print(
        f"ZERO-FLAG-FACT: {len(victims)} of 256 face positions corrupted "
        "without the cfg block's zero-flag arm -- the surface contract is "
        "load-bearing"
    )
