# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""emit-e2e used to tell its builder nothing about the box it was targeting, bound
nothing it spawned, and reaped only the direct child on timeout. One run showed what
that costs: the builder derived a depth cap from "~12 GB of Wormhole DRAM per chip"
on a 32 GB Blackhole box, called it "measured not assumed", then ran a script to
re-measure the ceiling it had invented — which grew until the kernel OOM-killed it,
failing the enclosing scope and destroying a run whose work had already succeeded.

These pin the three fixes and, as importantly, that a run WITHOUT a box behaves
exactly as it did before."""
import os

from scripts.tt_hw_planner.commands.emit_e2e import (
    _agent_mem_cap_bytes,
    _build_agent_prompt,
    _hardware_prompt_block,
    _mem_cap_preexec,
    _resolve_box,
)
from scripts.tt_hw_planner.hardware import HARDWARE, find_box


# --- the board spec reaches the builder ---------------------------------------


def test_resolve_box_returns_registered_box():
    box = _resolve_box("QB2")
    assert box is not None
    assert box.name == "QB2"
    assert box is find_box("QB2")


def test_resolve_box_declines_unknown_or_absent_instead_of_raising():
    """`find_box` raises KeyError; emit-e2e must degrade to "no hardware block"
    rather than aborting a run over a flag typo."""
    assert _resolve_box(None) is None
    assert _resolve_box("") is None
    assert _resolve_box("p300c") is None, "a board series is not a box name"
    assert _resolve_box("qb2") is None, "find_box is case-sensitive; must not raise"


def test_hardware_block_states_the_registered_per_chip_memory():
    box = find_box("QB2")
    block = _hardware_prompt_block(box)
    assert f"{box.hbm_per_chip_gb:.0f} GB" in block
    assert f"{box.total_hbm_gb:.0f} GB" in block
    assert str(box.chips) in block
    assert box.arch in block


def test_hardware_block_reuses_the_overhead_model_not_raw_dram():
    """The usable figure must come from the box's own overhead decomposition, so it
    tracks calibration instead of being re-derived in the prompt."""
    box = find_box("QB2")
    block = _hardware_prompt_block(box)
    assert f"{box.usable_per_chip_gb(1):.1f} GB" in block
    assert f"{box.usable_per_chip_gb(2):.1f} GB" in block


def test_hardware_block_is_empty_without_a_box():
    assert _hardware_prompt_block(None) == ""


def test_prompt_is_byte_identical_without_a_box():
    """Non-breaking guarantee: a run that passes no --box gets the previous prompt."""
    base = _build_agent_prompt(model_id="m/x", demo_dir="/tmp/x", pcc=0.95)
    noted = _build_agent_prompt(model_id="m/x", demo_dir="/tmp/x", pcc=0.95, hardware_note="")
    assert base == noted
    assert "HARDWARE —" not in base


def test_hardware_block_lands_in_the_builder_prompt():
    block = _hardware_prompt_block(find_box("QB2"))
    prompt = _build_agent_prompt(model_id="m/x", demo_dir="/tmp/x", pcc=0.9, hardware_note=block)
    assert "HARDWARE — QB2" in prompt
    assert "do NOT" in prompt


def _executable_source(fn) -> str:
    """Source of `fn` with docstrings and comments stripped. The prose is allowed to
    name the boards that motivated the fix; the executable code is not."""
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body:
            first = body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                if isinstance(first.value.value, str):
                    node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)  # unparse drops comments


def test_no_board_figures_are_hardcoded_in_the_block():
    """Constraint 3: every number comes from the registered box, so the block is
    correct for any board in the table — not just the one that exposed the bug."""
    src = _executable_source(_hardware_prompt_block)
    for forbidden in ("QB2", "Blackhole", "Wormhole", "32.0", "12.0"):
        assert forbidden not in src, f"hardware block must not hardcode {forbidden!r}"
    # and it genuinely renders per-box, not one fixed string
    rendered = {b.name: _hardware_prompt_block(b) for b in HARDWARE}
    assert len(set(rendered.values())) > 1


# --- the host-memory ceiling ---------------------------------------------------


def test_mem_cap_is_a_fraction_of_physical_ram():
    cap = _agent_mem_cap_bytes()
    total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    assert 0 < cap < total, "a cap must leave headroom for everything else on the box"
    assert cap == int(total * 0.7)


def test_mem_cap_is_disablable(monkeypatch):
    """An escape hatch matters: a legitimately huge model on a quiet box should not
    be capped into failure."""
    monkeypatch.setenv("TT_HW_PLANNER_AGENT_MEM_FRACTION", "0")
    assert _agent_mem_cap_bytes() == 0


def test_mem_cap_fraction_is_tunable(monkeypatch):
    total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    monkeypatch.setenv("TT_HW_PLANNER_AGENT_MEM_FRACTION", "0.25")
    assert _agent_mem_cap_bytes() == int(total * 0.25)


def test_mem_cap_survives_a_garbage_fraction(monkeypatch):
    monkeypatch.setenv("TT_HW_PLANNER_AGENT_MEM_FRACTION", "not-a-number")
    total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    assert _agent_mem_cap_bytes() == int(total * 0.7)


def test_mem_cap_clamps_at_all_of_ram(monkeypatch):
    total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    monkeypatch.setenv("TT_HW_PLANNER_AGENT_MEM_FRACTION", "5")
    assert _agent_mem_cap_bytes() == total


def test_preexec_targets_rlimit_data_not_address_space():
    """RLIMIT_AS would fire on healthy runs: the process that died mapped 463 GB of
    address space while resident at 199 GB, because device memory is mapped in."""
    src = _executable_source(_mem_cap_preexec)
    assert "RLIMIT_DATA" in src
    assert "RLIMIT_AS" not in src


def _apply_in_forked_child(limit: int) -> int:
    """Apply the limit in a forked child and return its exit status.

    Never in-process: an earlier version of this test applied a 1-byte limit to the
    test runner itself, which then died at the next allocation needing `brk()` — in
    an unrelated place, with no summary. A cap test must not cap the capper."""
    pid = os.fork()
    if pid == 0:  # pragma: no cover -- child
        try:
            _mem_cap_preexec(limit)()
            os._exit(0)
        except BaseException:
            os._exit(1)
    _, status = os.waitpid(pid, 0)
    return os.WEXITSTATUS(status)


def test_preexec_is_callable():
    assert callable(_mem_cap_preexec(1 << 40))


def test_preexec_never_raises_on_a_sane_limit():
    assert _apply_in_forked_child(1 << 40) == 0


def test_preexec_never_raises_on_an_absurd_limit():
    assert _apply_in_forked_child(1) == 0


def test_preexec_preserves_the_hard_limit():
    """Lowering the hard limit is irreversible without privileges, so the cap must
    only move the soft limit — otherwise a capped process can never recover, and a
    caller that applied it to itself would be permanently crippled."""
    import resource

    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:  # pragma: no cover -- child
        try:
            os.close(r)
            before = resource.getrlimit(resource.RLIMIT_DATA)
            _mem_cap_preexec(1 << 30)()
            after = resource.getrlimit(resource.RLIMIT_DATA)
            os.write(w, f"{before[1]},{after[0]},{after[1]}".encode())
            os._exit(0)
        except BaseException:
            os._exit(1)
    os.close(w)
    payload = os.read(r, 128).decode()
    os.close(r)
    _, status = os.waitpid(pid, 0)
    assert os.WEXITSTATUS(status) == 0
    hard_before, soft_after, hard_after = (int(x) for x in payload.split(","))
    assert soft_after == 1 << 30, "the soft limit must be the cap"
    assert hard_after == hard_before, "the hard limit must be left untouched"


# --- the spawn + timeout contract ---------------------------------------------


def _run_agent_source() -> str:
    import inspect

    from scripts.tt_hw_planner.commands import emit_e2e

    return inspect.getsource(emit_e2e._run_agent)


def test_agent_spawn_uses_its_own_session():
    """Required for the tree kill: `_kill_process_tree` signals the process GROUP,
    so without a new session it would signal the planner itself."""
    assert "start_new_session=True" in _run_agent_source()


def test_agent_spawn_applies_the_cap():
    src = _run_agent_source()
    assert "preexec_fn=" in src
    assert "_mem_cap_preexec" in src


def test_timeout_reaps_descendants_not_just_the_child():
    """The agent writes and runs its own scripts; `proc.kill()` left those holding
    host memory and the devices. Checked on executable lines only — the comment
    explaining why it is NOT used legitimately names it."""
    import re

    src = _run_agent_source()
    assert "_kill_process_tree" in src
    calls = [ln for ln in src.splitlines() if re.match(r"\s*proc\.kill\(\)", ln)]
    assert not calls, f"bare child kill still present: {calls}"


def test_tree_kill_helper_is_the_shared_one():
    """Constraint 2: reuse cli's /proc-walking killer rather than a second copy."""
    from scripts.tt_hw_planner import cli

    assert hasattr(cli, "_kill_process_tree")
    assert "from ..cli import _kill_process_tree" in _run_agent_source()
