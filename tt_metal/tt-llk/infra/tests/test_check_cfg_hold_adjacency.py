#!/usr/bin/env python3
"""Tests for the config-write-under-a-held-instruction checker.

Each test pins one shape the detector is designed to separate, so a future edit to `holds()`,
`READER`, the adjacency rule or the window cannot silently stop detecting.
Run: python3 -m pytest tt_metal/tt-llk/infra/tests/ -q --noconftest
"""

import os
import subprocess
import sys
import textwrap

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "check_cfg_hold_adjacency.py")

INDUCER = "    TTI_MOVD2B(p_movd2b::MOV_1_ROW, 0, ADDR_MOD_0, 0, 0);"
VICTIM = "    TTI_MOVA2D(0, 0, ADDR_MOD_0, 0, 0);"
CFG = "    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_RMW>(fmt);"
NOP = "    TTI_NOP;"
GUARD = "    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH | p_stall::WAIT_SFPU);"
# A victim inside the held class of a move into Dest: it READS Dest.
DEST_READER = "    TTI_MOVD2B(p_movd2b::MOV_1_ROW, 0, ADDR_MOD_0, 0, 0);"


@pytest.fixture
def wh(tmp_path):
    """The checker is Wormhole-only and selects on the path, so the file must live in that tree."""
    d = tmp_path / "tt_llk_wormhole_b0"
    d.mkdir()
    return d


def run(d, name, body):
    p = d / name
    p.write_text(textwrap.dedent(body))
    r = subprocess.run([sys.executable, SCRIPT, str(p)], capture_output=True, text=True)
    return r


def test_adjacent_inducer_reader_write_is_flagged(wh):
    r = run(wh, "x.h", f"{INDUCER}\n{VICTIM}\n{CFG}\n")
    assert r.returncode == 1, r.stdout
    assert "held at issue" in r.stdout


def test_stallwait_guard_suppresses_the_finding(wh):
    r = run(wh, "x.h", f"{INDUCER}\n{VICTIM}\n{GUARD}\n{CFG}\n")
    assert r.returncode == 0, r.stdout


def test_a_guard_split_over_several_lines_is_still_a_guard(wh):
    """The tree writes multi-line STALLWAIT calls; matching per raw line reads them as absent."""
    r = run(
        wh,
        "x.h",
        f"""{INDUCER}
{VICTIM}
    TTI_STALLWAIT(
        p_stall::STALL_CFG,
        p_stall::MATH | p_stall::WAIT_SFPU);
{CFG}
""",
    )
    assert r.returncode == 0, f"a wrapped guard must not read as unguarded:\n{r.stdout}"


def test_trailing_comment_does_not_glue_two_instructions(wh):
    """A `// note` after the `;` must still end the statement, or adjacency shifts by one."""
    r = run(wh, "x.h", f"{INDUCER}  // set up\n{VICTIM}  // victim\n{CFG}\n")
    assert r.returncode == 1, r.stdout


def test_no_inducer_before_the_reader_is_not_flagged(wh):
    """The hold catches only the instruction presented next; with nothing holding it there is none."""
    r = run(wh, "x.h", f"{VICTIM}\n{CFG}\n")
    assert r.returncode == 0, r.stdout


def test_inducer_separated_from_the_reader_is_not_flagged(wh):
    """Anything issued in between absorbs the hold, so the reader is no longer the one held."""
    r = run(wh, "x.h", f"{INDUCER}\n{NOP}\n{VICTIM}\n{CFG}\n")
    assert r.returncode == 0, r.stdout


@pytest.mark.parametrize("fillers,flagged", [(0, True), (2, True), (3, False)])
def test_window_matches_the_measured_dose_response(wh, fillers, flagged):
    """Measured: 1 and 2 fillers between reader and write still corrupt, 3 and 4 do not."""
    body = f"{INDUCER}\n{VICTIM}\n" + f"{NOP}\n" * fillers + f"{CFG}\n"
    r = run(wh, "x.h", body)
    assert (r.returncode == 1) is flagged, f"{fillers} filler(s):\n{r.stdout}"


def test_address_shaping_config_is_not_flagged(wh):
    """ADDR_MOD_* is out of scope structurally, not by a timing margin.

    An instruction carries its formed addresses forward as values when it is accepted out of the
    issue stage, so a write issued after it has nothing left to change about them -- unlike the
    numeric control, which it does not carry and which is fetched live later. Measured on n150: an
    address-mod section write in the earliest slot after a held reader leaves the result clean,
    with a clean no-hold control and a failing liveness arm.
    """
    r = run(
        wh,
        "x.h",
        f"{INDUCER}\n{VICTIM}\n    TTI_SETC16(ADDR_MOD_0_SEC0_Base_ADDR32, 0);\n",
    )
    assert r.returncode == 0, r.stdout


def test_non_wormhole_tree_is_skipped(tmp_path):
    """Blackhole captures these fields at accept; pointing the check there would be a false positive."""
    d = tmp_path / "tt_llk_blackhole"
    d.mkdir()
    r = run(d, "x.h", f"{INDUCER}\n{VICTIM}\n{CFG}\n")
    assert r.returncode == 0, r.stdout


def test_elwadd_is_not_held_by_a_move_into_dest(wh):
    """The post-move hold's class excludes ELWADD/ELWSUB; asserting a hold hardware does not impose
    would flag a write that lands after nothing was ever held."""
    move_into_dest = "    TTI_MOVB2D(p_movb2d::MOV_1_ROW, 0, ADDR_MOD_0, 0, 0);"
    r = run(
        wh, "x.h", f"{move_into_dest}\n    TTI_ELWADD(0, 0, ADDR_MOD_0, 0);\n{CFG}\n"
    )
    assert r.returncode == 0, f"ELWADD is exempt from the move hold:\n{r.stdout}"


@pytest.mark.parametrize(
    "inducer,victim,distance,flagged",
    [
        # A move into Dest holds for 3, so a write 3 slots after the reader is still inside.
        (
            "    TTI_MOVB2D(p_movb2d::MOV_1_ROW, 0, ADDR_MOD_0, 0, 0);",
            DEST_READER,
            3,
            True,
        ),
        # A move into SrcA holds for 1, so the same distance is outside its window.
        ("    TTI_MOVD2A(0, 0, ADDR_MOD_0, 0, 0);", DEST_READER, 3, False),
        # ... but immediately after, it is inside.
        ("    TTI_MOVD2A(0, 0, ADDR_MOD_0, 0, 0);", DEST_READER, 1, True),
    ],
)
def test_window_is_per_inducer_not_one_number(wh, inducer, victim, distance, flagged):
    """Hold lengths differ by inducer. One shared window over-states the short ones."""
    body = f"{inducer}\n{victim}\n" + f"{NOP}\n" * (distance - 1) + f"{CFG}\n"
    r = run(wh, "x.h", body)
    assert (
        r.returncode == 1
    ) is flagged, f"{inducer.strip()} at distance {distance}:\n{r.stdout}"
