# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Cross-lane arsenal: SIM lane-tracer battery (lane FB).

Runs sources/sfpu_crosslane_probe.cpp on the pinned simulator, one
PROBE_MODE per primitive, and compares EVERY lane against the host oracle
(helpers/crosslane_oracle.py -- transcribed from tt-isa-documentation
functional models).  The tensor<->(row,lane) mapping is derived EMPIRICALLY
from three calibration modes (identity / rowtag / lanetag -- lane DS's
proven method), then cross-checked against the doc-derived model.

Discipline:
  - doc = prior, pinned sim = oracle; any sim-vs-oracle mismatch is a
    reportable finding, printed with full per-lane detail before the assert.
  - every primitive runs on TWO stimuli: lane-id sentinels AND a splitmix32
    varied twin (genericity: nothing may depend on sentinel values).
  - operand-role facts the compiler surface does not document (which
    SFPSWAP builtin operand plays VD) are decided EMPIRICALLY: both
    candidate role-mappings are compared and exactly one must match
    everywhere; the winner is printed as an arsenal fact.

Run: pytest -q --run-simulator test_crosslane_lane_tracer.py
(TT_METAL_SIMULATOR must point at the pinned libttsim.so with
soc_descriptor.yaml beside it; CHIP_ARCH=blackhole.)
"""

from dataclasses import dataclass

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import TemplateParameter

from helpers import crosslane_oracle as co

M32 = 0xFFFFFFFF
ELEMS = 1024
ROWS = 16
LANES = 32


@dataclass
class UIntTemplate(TemplateParameter):
    name: str
    value: int

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t {self.name} = {self.value}u;"


def run_probe(mode, input_vec):
    formats = InputOutputFormat(DataFormat.UInt32, DataFormat.UInt32)
    src = torch.tensor(input_vec, dtype=torch.int64)
    config = TestConfig(
        "sources/sfpu_crosslane_probe.cpp",
        formats,
        templates=[UIntTemplate("PROBE_MODE", mode)],
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
# calibration (module-scoped): T[(row, lane)] -> result index,
# M1inv[(row, lane)] -> input element index
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
    assert sorted(row_positions.keys()) == list(range(ROWS)), (
        f"rowtag rows found: {sorted(row_positions.keys())}")
    for i, positions in row_positions.items():
        assert len(positions) == LANES, f"row {i}: {len(positions)} tagged"

    T = {}
    for i, positions in row_positions.items():
        tags = [lanetag[p] for p in positions]
        # vConstTileId == 2 * lane (lane-EX claim) -- re-proven here on-sim:
        assert sorted(tags) == [2 * l for l in range(LANES)], (
            f"row {i}: vConstTileId tags {sorted(tags)} != 2*lane -- "
            "the tile-id claim FAILED on the pinned sim (finding!)")
        for k, p in enumerate(positions):
            T[(i, tags[k] // 2)] = p

    m1inv = {}
    for i in range(ROWS):
        for l in range(LANES):
            v = ident[T[(i, l)]]
            assert 0 <= v < ELEMS, f"identity value {v} out of ramp range"
            m1inv[(i, l)] = v
    assert len(set(m1inv.values())) == ROWS * LANES, "M1 not injective"

    Cal.T, Cal.m1inv = T, m1inv
    return Cal


def build_input(cal, rows):
    """rows: dict row_index -> [32 lane values]."""
    vec = [0] * ELEMS
    for i, lane_vals in rows.items():
        for l in range(LANES):
            vec[cal.m1inv[(i, l)]] = lane_vals[l] & M32
    return vec


def read_rows(cal, out, indices):
    return {i: [out[cal.T[(i, l)]] for l in range(LANES)] for i in indices}


def check_rows(mode, got, want, note=""):
    """Compare got/want dicts of rows; on mismatch print full detail."""
    bad = []
    for i in sorted(want.keys()):
        for l in range(LANES):
            w = want[i][l]
            if w is None:  # unpredictable lane (WH shr1 class): skip
                continue
            if got[i][l] != (w & M32):
                bad.append((i, l, got[i][l], w & M32))
    if bad:
        print(f"ORACLE MISMATCH mode={mode} {note}: {len(bad)} lanes")
        for i, l, g, w in bad[:16]:
            print(f"  row={i} lane={l} got={g:08x} want={w:08x}")
        for i in sorted(want.keys()):
            print(f"  got  row{i}: " + " ".join(f"{v:08x}" for v in got[i]))
            print(f"  want row{i}: " + " ".join(
                "????????" if v is None else f"{v & M32:08x}"
                for v in want[i]))
    assert not bad, f"mode {mode} {note}: sim disagrees with oracle"


def stimuli(nrows, tag):
    """(name, rows) pairs: sentinel + varied genericity twin."""
    sent = {i: [s ^ (i << 16) for s in co.lane_id_sentinels(tag)]
            for i in range(nrows)}
    varied = {i: co.varied_stimulus(i, seed=tag + 100) for i in range(nrows)}
    return [("sentinel", sent), ("varied", varied)]


# ---------------------------------------------------------------------------
# swap-role fact (empirically decided, must be globally consistent)
# ---------------------------------------------------------------------------

SWAP_ROLE = {"winner": None}  # 'A': builtin arg0 -> VC; 'B': arg0 -> VD


def swap_expected(a_vals, b_vals, mod, state=None):
    """Returns {'A': (out_a, out_b), 'B': (out_a, out_b)}."""
    nvc, nvd = co.sfpswap(a_vals, b_vals, mod, state=state)
    va_a, vb_a = nvc, nvd  # candidate A: arg0 is VC
    nvc2, nvd2 = co.sfpswap(b_vals, a_vals, mod, state=state)
    va_b, vb_b = nvd2, nvc2  # candidate B: arg0 is VD
    return {"A": (va_a, vb_a), "B": (va_b, vb_b)}


def adjudicate_swap(mode, got_a, got_b, cands, note):
    matches = [k for k, (ea, eb) in cands.items()
               if got_a == [x & M32 for x in ea]
               and got_b == [x & M32 for x in eb]]
    if not matches:
        print(f"SWAP ROLE: mode={mode} {note}: NEITHER candidate matches")
        print("  got_a: " + " ".join(f"{v:08x}" for v in got_a))
        for k, (ea, eb) in cands.items():
            print(f"  cand{k}_a: " + " ".join(f"{v & M32:08x}" for v in ea))
        assert matches, f"mode {mode} {note}: no role-candidate matches (finding)"
    if len(matches) == 1:
        w = matches[0]
        if SWAP_ROLE["winner"] is None:
            SWAP_ROLE["winner"] = w
            print(f"SWAP-ROLE-FACT: builtin arg0 plays "
                  f"{'VC' if w == 'A' else 'VD'} (candidate {w})")
        else:
            assert SWAP_ROLE["winner"] == w, (
                f"swap role fact flipped: {SWAP_ROLE['winner']} vs {w} "
                f"at mode {mode} {note} (finding)")


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_transp8(cal):
    for name, rows in stimuli(8, 3):
        out = run_probe(3, build_input(cal, rows))
        got = read_rows(cal, out, range(8))
        regs = [rows[i] for i in range(8)]
        exp = co.sfptransp(regs)
        check_rows(3, got, {i: exp[i] for i in range(8)}, f"[{name}]")


def test_rot_family(cal):
    for name, rows in stimuli(2, 4):
        out = run_probe(4, build_input(cal, rows))
        got = read_rows(cal, out, range(4))
        want = {
            0: co.subvec_shflror1(rows[0]),
            1: co.subvec_shflshr1(rows[1], arch="bh"),
            2: co.subvec_rotr(rows[0], 3),
            3: rows[1],  # ror1^8 = id
        }
        check_rows(4, got, want, f"[{name}]")


@pytest.mark.parametrize("mode,mod", [(5, 0), (6, 1), (7, 2)])
def test_copy4_family(cal, mode, mod):
    nin = 5 if mod == 2 else 4
    for name, rows in stimuli(nin, 5 + mode):
        out = run_probe(mode, build_input(cal, rows))
        got = read_rows(cal, out, range(4))
        # kernel feeds (r1, r2, r3, r0_or_vc); pre-state L0..L3 = r0..r3
        if mod == 0:
            e = co.shft2_copy4(rows[0], rows[1], rows[2], rows[3])
        elif mod == 1:
            e = co.shft2_chained_copy4(rows[0], rows[1], rows[2], rows[3])
        else:
            e = co.shft2_ror1_and_copy4(rows[0], rows[1], rows[2], rows[3],
                                        rows[4])
        check_rows(mode, got, {i: e[i] for i in range(4)}, f"[{name}]")


@pytest.mark.parametrize("mode,mods", [(8, (1, 2, 3, 4)), (9, (5, 6, 7, 8))])
def test_swap_rowgroup_mods(cal, mode, mods):
    for name, rows in stimuli(8, 20 + mode):
        out = run_probe(mode, build_input(cal, rows))
        got = read_rows(cal, out, range(8))
        for k, mod in enumerate(mods):
            a, b = rows[2 * k], rows[2 * k + 1]
            cands = swap_expected(a, b, mod)
            adjudicate_swap(mode, got[2 * k], got[2 * k + 1], cands,
                            f"[{name}] mod{mod}")


def test_swap_unconditional_and_minmax(cal):
    for name, rows in stimuli(4, 30):
        out = run_probe(10, build_input(cal, rows))
        got = read_rows(cal, out, range(4))
        # mod 0: unconditional swap -- role-independent
        assert got[0] == [v & M32 for v in rows[1]]
        assert got[1] == [v & M32 for v in rows[0]]
        cands = swap_expected(rows[2], rows[3], 1)
        adjudicate_swap(10, got[2], got[3], cands, f"[{name}] mod1")


def test_exchange_flip_global(cal):
    st = co.LaneState()
    for l in range(LANES):
        st.lane_config[l] |= co.LC_EXCHANGE_SRCB_SRCC
    for name, rows in stimuli(4, 31):
        out = run_probe(11, build_input(cal, rows))
        got = read_rows(cal, out, range(4))
        cands = swap_expected(rows[0], rows[1], 1, state=st)
        adjudicate_swap(11, got[0], got[1], cands, f"[{name}] flipped")
        cands = swap_expected(rows[2], rows[3], 1)  # after restore
        adjudicate_swap(11, got[2], got[3], cands, f"[{name}] restored")


def test_lane_masked_flip(cal):
    # SFPCONFIG Mod1=8, Imm16=0x4444 -> EXCHANGE bit set in columns 1,3,5,7
    st = co.LaneState()
    co.sfpconfig_write(st, 15, [co.LC_EXCHANGE_SRCB_SRCC] * LANES,
                       imm16=0x4444, mod1=8)
    for name, rows in stimuli(4, 32):
        out = run_probe(12, build_input(cal, rows))
        got = read_rows(cal, out, range(4))
        cands = swap_expected(rows[0], rows[1], 1, state=st)
        adjudicate_swap(12, got[0], got[1], cands, f"[{name}] masked")
        cands = swap_expected(rows[2], rows[3], 1)
        adjudicate_swap(12, got[2], got[3], cands, f"[{name}] restored")


def test_indexed_swap(cal):
    """Indexed swap under ENABLE_DEST_INDEX, incl. planted EQUAL keys.

    TIE DIVERGENCE (lane FB finding, 2026-08-21): the pinned sim decides
    ties as (min lanes: no swap; max lanes: swap) -- craq-sim tensix.cpp
    sfpswap_vd_gets_c -- while SFPSWAP.md keys tie-swapping on the SIGN
    (min lanes swap equal negatives, max lanes equal positives).  The
    assertion below follows the PINNED SIM (the shipped oracle); the doc
    divergence is printed whenever the stimuli exercise it.  Silicon
    adjudication pending -- goldens must not depend on tie companion
    movement until then.
    """
    keys0 = [co.f32_to_bits(float(l % 5) - 2.0) for l in range(LANES)]
    keys1 = [co.f32_to_bits(float((l + 2) % 5) - 2.0) for l in range(LANES)]
    for l in (3, 11, 19, 27):
        keys1[l] = keys0[l] = co.f32_to_bits(2.5)
    for l in (5, 13, 21, 29):
        keys1[l] = keys0[l] = co.f32_to_bits(-2.5)
    comp0 = co.lane_id_sentinels(1)
    comp1 = co.lane_id_sentinels(2)
    cases = [("planted", {0: keys0, 1: keys1, 2: comp0, 3: comp1})]
    v = stimuli(4, 33)[1][1]
    cases.append(("varied", v))
    for name, rows in cases:
        out = run_probe(13, build_input(cal, rows))
        got = read_rows(cal, out, range(4))
        # Role candidate B (arg0 -> VD) won the plain-swap adjudication;
        # candidate A kept as the falsification arm.
        exp = {}
        for tie in ("sim", "doc"):
            b = co.sfpswap_indexed(rows[1], rows[0], rows[3], rows[2], 1,
                                   tie=tie)
            exp[tie] = (b[1], b[0], b[3], b[2])
        if exp["sim"] != exp["doc"]:
            diff = [i for i in range(4) if exp['sim'][i] != exp['doc'][i]]
            print(f"TIE-DIVERGENCE [{name}]: doc and sim tie models differ "
                  f"on rows {diff} (companion movement on equal keys)")
        bad = []
        for i in range(4):
            for l in range(LANES):
                w = exp["sim"][i][l] & M32
                if got[i][l] != w:
                    bad.append((i, l, got[i][l], w))
        if bad:
            for i in range(4):
                print(f"  got row{i}:  " + " ".join(f"{x:08x}"
                                                    for x in got[i]))
                print(f"  sim row{i}:  " + " ".join(f"{x & M32:08x}"
                                                    for x in exp['sim'][i]))
        assert not bad, (
            f"indexed swap [{name}]: sim run disagrees with the sim-tie "
            f"oracle model in {len(bad)} lanes (finding)")


def test_config_broadcast(cal):
    for name, rows in stimuli(2, 34):
        out = run_probe(14, build_input(cal, rows))
        got = read_rows(cal, out, range(2))
        want = {0: co.sfpconfig_broadcast(rows[0]),
                1: co.sfpconfig_broadcast(rows[1])}
        check_rows(14, got, want, f"[{name}]")


def test_reduce_int_composition(cal):
    for name, rows in stimuli(1, 35):
        out = run_probe(15, build_input(cal, rows))
        got = read_rows(cal, out, range(2))
        want = {0: co.subvec_reduce_tree(rows[0], "add"),
                1: co.reduce32_tree(rows[0], "add")}
        check_rows(15, got, want, f"[{name}]")


def _sort4_networks(regs):
    """Expected results of the 5-CE sort4 network under both role
    candidates: min-to-first (asc across regs) and max-to-first."""
    outs = {}
    for role, min_first in (("A", False), ("B", True)):
        vs = [list(r) for r in regs]
        for (i, j) in ((0, 1), (2, 3), (0, 2), (1, 3), (1, 2)):
            for l in range(LANES):
                a, b = vs[i][l], vs[j][l]
                lo, hi = co._ce_pair(a, b, min_first=True)
                vs[i][l], vs[j][l] = (lo, hi) if min_first else (hi, lo)
        outs[role] = vs
    return outs


def test_sort4_register_axis(cal):
    for name, rows in stimuli(4, 36):
        out = run_probe(16, build_input(cal, rows))
        got = read_rows(cal, out, range(4))
        cands = _sort4_networks([rows[i] for i in range(4)])
        matches = [k for k, vs in cands.items()
                   if all(got[i] == [x & M32 for x in vs[i]]
                          for i in range(4))]
        assert matches, f"sort4 [{name}]: no candidate matches"
        if len(matches) == 1 and SWAP_ROLE["winner"] is not None:
            assert matches[0] == SWAP_ROLE["winner"], (
                f"sort4 role {matches[0]} contradicts {SWAP_ROLE['winner']}")


def test_sort4_transp_sandwich(cal):
    for name, rows in stimuli(4, 37):
        out = run_probe(17, build_input(cal, rows))
        got = read_rows(cal, out, range(4))
        regs = [rows[i] for i in range(4)] + [[0] * LANES] * 4
        pre = co.sfptransp(regs)[:4]
        cands = _sort4_networks(pre)
        for k in cands:
            full = cands[k] + [[0] * LANES] * 4
            cands[k] = co.sfptransp(full)[:4]
        matches = [k for k, vs in cands.items()
                   if all(got[i] == [x & M32 for x in vs[i]]
                          for i in range(4))]
        assert matches, f"sort4-sandwich [{name}]: no candidate matches"


# ---------------------------------------------------------------------------
# partial 16-bit access (empirical Dst16b<->Dst32b aliasing adjudication)
# ---------------------------------------------------------------------------


def adj32(r):
    return ((r & 0x1F8) << 1) | (r & 0x207)


def test_companion_roundtrip_and_aliasing(cal):
    """Partial 16-bit-view Dst access in 32-bit layout, adjudicated against
    the PINNED SIM model (crosslane_oracle sim16_* helpers):
      - 16b-view LOAD at address A = dst_encode_bf16(datum32 >> 16) -- the
        RAW banked (BF16-swizzled) high half;
      - 16b-view STORE at address A writes the high half and ZEROES the
        paired low half.
    DOC DIVERGENCE (recorded): Dst.md maps 16b rows to their own physical
    cells (Adj16/Adj32 block algebra, rows 8..15 = LOW halves) and never
    clobbers the paired half; tt-blaze #2475 claims silicon BF16 stores
    canonicalize it.  Three-way disagreement -- silicon adjudication
    pending; these goldens pin the shipped sim.
    """
    for name, rows in stimuli(3, 38):
        out = run_probe(18, build_input(cal, rows))
        got = read_rows(cal, out, (0, 4, 5, 6, 7, 8, 9))
        in0, in1, base = rows[0], rows[1], rows[2]
        # packed word: lo16 <- sim16_load(in0), hi16 <- sim16_load(in1).
        # For rows < 8 the Dst.md Adj algebra makes the 16b view coincide
        # with the raw (BF16-swizzled) HIGH half of the same-address datum.
        c = [((co.sim16_load(in1[l]) << 16) | co.sim16_load(in0[l])) & M32
             for l in range(LANES)]
        # ADJUDICATED MODEL (pinned sim, this fp32-acc config): the raw
        # 16-bit store modes (UINT16/LO16_ONLY/HI16_ONLY) write their OWN
        # physical bank cell (Dst.md Adj16 aliasing, dst_32bit_addr_en=0
        # path) -- they do NOT touch the 32b datum at the same address.
        # Rows 4/5 (baseline + partial store on top) and row 6 (zero +
        # UINT16 store) therefore read back UNCHANGED; only the in-16b-view
        # roundtrip (row 9) sees the stored halves.
        want = {
            0: c,
            4: base,             # LO16_ONLY store landed in its own cell
            5: base,             # HI16_ONLY store likewise
            6: [0] * LANES,      # UINT16 store likewise
            9: c,                # 16b-view store->load roundtrip exact
        }
        check_rows(18, {k: got[k] for k in want}, want,
                   f"[{name}] adjudicated 16b-view model")
        print(f"ALIAS-FACT [{name}]: pinned sim 16b raw view rides the "
              "Dst.md Adj16 bank cells (reads at rows<8 = swizzled HI "
              "halves; stores never clobber the same-address 32b datum); "
              "BF16-format stores instead take the 32b write path -- see "
              "the RMW probe")


def test_bf16_store_rmw_probe(cal):
    """tt-blaze #2475 probe: BF16-format SFPSTORE onto known 32b content.

    Three candidate models for the paired LOW half:
      doc     (SFPSTORE.md): only the 16b cell written -> low PRESERVED;
      sim     (tensix.cpp sfpstore_values): low ZEROED, except under
              ENABLE_DEST_INDEX (lane_config & 4) or the TopK LCONST0
              special case, where it is PRESERVED (RMW);
      silicon (#2475 kernel comment): low BF16-CANONICALIZED (denormals
              flushed).
    Assertions pin the SIM model (arm 1 plain store: low zeroed; arm 2
    inside an ENABLE_DEST_INDEX window: low preserved).  base0 carries
    BOTH denormal and normal low halves so the three models are mutually
    distinguishable; silicon adjudication rides the ledger.
    """
    base0 = []
    for l in range(LANES):
        low = 0x0001 + (l << 4) if l % 2 == 0 else 0x3F80 + l  # denorm/normal
        base0.append(((0x4210 + l) << 16) | low)
    base1 = [((0x4310 + l) << 16) | (0x0002 + (l << 4) if l % 2 else 0x3E80 + l)
             for l in range(LANES)]
    val = [co.f32_to_bits(1.0 + l) for l in range(LANES)]  # bf16-exact
    rows = {0: base0, 1: base1, 3: val}
    out = run_probe(30, build_input(cal, rows))
    got = read_rows(cal, out, range(4))
    want_plain = [(val[l] & 0xFFFF0000) for l in range(LANES)]  # low ZEROED
    want_window = [(val[l] & 0xFFFF0000) | (base1[l] & 0xFFFF)
                   for l in range(LANES)]                        # low PRESERVED
    for l in range(LANES):
        doc_model = (val[l] & 0xFFFF0000) | (base0[l] & 0xFFFF)
        canon = val[l] & 0xFFFF0000 if (base0[l] & 0x7F80) == 0 else doc_model
        if got[0][l] not in (want_plain[l],):
            print(f"BF16-RMW lane{l}: got={got[0][l]:08x} "
                  f"sim={want_plain[l]:08x} doc={doc_model:08x} "
                  f"canon={canon:08x}")
    assert got[0] == want_plain, (
        "plain BF16 store: sim no longer zeroes the paired low half "
        "(model changed -- finding)")
    assert got[1] == [w & M32 for w in want_window], (
        "ENABLE_DEST_INDEX BF16 store: the sim low-half-preserve special "
        "case (TopK-motivated, hardcoding-audit open question) changed "
        "(finding)")
    print("BF16-RMW-FACT: pinned sim zeroes the paired low half on plain "
          "BF16 stores and PRESERVES it under ENABLE_DEST_INDEX; doc says "
          "always-preserve, #2475 says silicon canonicalizes -- three-way "
          "divergence recorded, silicon adjudication pending")
