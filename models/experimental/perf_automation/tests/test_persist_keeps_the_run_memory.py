"""`optimize --persist` keeps the record of what has been tried; the default keeps /tmp.

WHAT IS AT RISK. tmpstate.state_dir() resolves to

    PERF_MCP_STATE_DIR or tempfile.gettempdir()

and nothing in the tool sets that variable -- run.py only FORWARDS it to the MCP server when an
operator has exported it, and says so: "Unset on both sides they agree via gettempdir(), which is why
this is latent rather than broken." So by default these live in /tmp:

    cc_kernlog_<model>_<task>.json.cumulative   which knobs and rungs have been tried
    perf_measurements_<model>_<task>.jsonl      the ledger and its byte anchors
    perf_mcp_full_pipeline_baseline_1cq_*.json  the bar the ratchet defends

A reboot erases all of it. The next run then re-tries every knob it had already proved useless --
precisely the behaviour the rung-closure enforcement exists to prevent, reintroduced by where the
evidence is stored rather than by any logic.

WHY NOT JUST MOVE IT. /tmp is the right home for the WORKTREE: a disposable sandbox that self-cleans,
whose only durable output is committed to the run's branch, and whose build is worth hours but is
regenerable. Moving that to $HOME trades a self-cleaning directory for a disk-space chore. The MEMORY
is the part worth keeping, and it is small.

Hence an opt-in flag rather than a new default: the one-off run keeps /tmp's hygiene, and a run you
expect to repeat says so. Both sides are then explicit, which a silent default never is.

NOT PERSISTED, deliberately: the worktree and the build. Everything valuable in a worktree is on the
run's branch -- run 39's wins survived its worktree being destroyed by a reboot, because they were
committed. What was lost was compile time, which is a separate problem with a separate answer
(a compiler cache), not a reason to keep a sandbox forever.
"""

import re
import sys
from pathlib import Path

import pytest

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))

_PLANNER = _PA.parent.parent.parent / "scripts" / "tt_hw_planner"
CLI = _PLANNER / "cli.py"
OPTC = _PLANNER / "commands" / "optimize.py"


# ---------------------------------------------------------------- the flag exists and is opt-in


def test_the_flag_is_declared():
    src = CLI.read_text()
    assert '"--persist"' in src, "optimize has no --persist flag"
    i = src.index('"--persist"')
    assert 'action="store_true"' in src[i : i + 200], "it must be a switch, not a value"


def test_it_is_off_by_default():
    """The default must stay /tmp: a one-off run should not leave state behind, and self-cleaning is
    the property that makes /tmp the right place for a sandbox."""
    src = CLI.read_text()
    i = src.index('"--persist"')
    win = src[i : i + 400]
    assert "default=True" not in win and 'action="store_false"' not in win


def test_the_help_says_what_survives_and_what_does_not():
    """An operator choosing between /tmp and $HOME needs to know the worktree is NOT kept -- otherwise
    the flag reads as "keep everything" and the disk fills."""
    src = CLI.read_text()
    i = src.index('"--persist"')
    win = src[i : i + 900].lower()
    assert "reboot" in win
    assert "worktree" in win and "/tmp" in win


# ---------------------------------------------------------------- it redirects the memory, both halves


def _persist_block() -> str:
    """The whole --persist branch, bounded by the NEXT branch rather than by a character count.

    This sliced a fixed 1200-character window, so any comment added inside the branch pushed the
    code being asserted out of view and three tests failed on an unrelated edit. That is the fourth
    character-window assertion in this suite to break that way; anchor on structure.
    """
    src = OPTC.read_text()
    i = src.index('if getattr(args, "persist", False):')
    j = src.index('if getattr(args, "fresh", False):', i)
    return src[i:j]


def test_it_redirects_the_state_dir():
    assert "PERF_MCP_STATE_DIR" in _persist_block()


def test_the_ledger_follows_the_state_dir():
    """measurements.py resolves the ledger RELATIVE to the state dir. Setting one without the other
    splits them, and the report then finds no anchors -- the defect measurements.py:213 documents,
    where every production run wrote anchors to /tmp while its other state went elsewhere."""
    assert "PERF_MCP_LEDGER_DIR" in _persist_block()


def test_an_operator_export_still_wins():
    """setdefault, not assignment: someone who exported PERF_MCP_STATE_DIR by hand -- as this box did
    for run 39 -- must not have it silently overridden."""
    blk = _persist_block()
    assert "setdefault" in blk, "the flag overwrites an explicit operator setting"
    assert 'os.environ["PERF_MCP_STATE_DIR"] =' not in blk


def test_the_path_is_keyed_per_model():
    """One shared directory would let two models read each other's attempt history, which is how a
    closed rung on one model silently closes it on another."""
    blk = _persist_block()
    assert "_slug" in blk and ".state" in blk


def test_the_directory_is_created():
    """A state dir that does not exist is indistinguishable from an empty one, and the run would fall
    back to /tmp semantics without saying so."""
    assert "mkdir(parents=True, exist_ok=True)" in _persist_block()


def test_it_says_where_the_memory_went():
    """Silent redirection is how you end up with two histories and no idea which one a run used."""
    assert "print(" in _persist_block()


# ---------------------------------------------------------------- the slug is safe as a path


@pytest.mark.parametrize(
    "name,expect",
    [
        ("gemma3", "gemma3"),
        ("Qwen/Qwen2-VL-7B", "Qwen_Qwen2-VL-7B"),
        ("a b c", "a_b_c"),
        ("../../etc", ".._.._etc"),
        ("", "model"),
        ("///", "model"),
    ],
)
def test_the_model_slug_cannot_escape_the_directory(name, expect):
    """It becomes a path component. A raw model id contains slashes, and `../` would place the state
    dir outside ~/.perf_mcp entirely."""
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", (name or "model")).strip("_") or "model"
    assert slug == expect, slug
    assert "/" not in slug


def test_the_slug_matches_the_implementation():
    """Pins the regex itself: a test computing its own slug proves nothing if the code uses another."""
    blk = _persist_block()
    assert 'r"[^A-Za-z0-9_.-]+"' in blk, blk[:400]


# ---------------------------------------------------------------- the default is genuinely unchanged


def test_nothing_is_set_when_the_flag_is_absent():
    """The whole point of opt-in. The redirect must sit INSIDE the flag's branch, not beside it."""
    src = OPTC.read_text()
    i = src.index('getattr(args, "persist"')
    j = src.index("result = run_cc(", i)
    body = src[i:j]
    assert "PERF_MCP_STATE_DIR" in body, "the redirect is not inside the --persist branch"
    before = src[:i]
    assert 'os.environ["PERF_MCP_STATE_DIR"]' not in before, "something sets the state dir unconditionally"


def test_the_engine_still_forwards_it_to_the_mcp_server():
    """perf_mcp is a SEPARATE PROCESS. run.py forwards the variable so both sides agree on one
    directory; without that the orchestrator and the server read different state and the report
    silently finds nothing."""
    run_src = (_PA / "cc_optimize" / "run.py").read_text()
    i = run_src.index("PERF_MCP_STATE_DIR")
    assert "PERF_MCP_LEDGER_DIR" in run_src[i - 200 : i + 300]
