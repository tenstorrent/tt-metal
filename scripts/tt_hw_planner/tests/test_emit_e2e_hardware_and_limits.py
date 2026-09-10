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
from types import SimpleNamespace

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


def _chips_in_use_as_emit_e2e_derives_it(mesh_arg):
    """Reproduce emit-e2e's own derivation, so these tests exercise the wiring rather
    than a number the test chose."""
    from scripts.tt_hw_planner.commands.emit_e2e import _parsed_mesh_chips

    return _parsed_mesh_chips(mesh_arg) or 0


def test_mesh_parser_separates_a_real_mesh_from_a_typo():
    """`_mesh_chip_count` must keep its 1-chip floor (the topology guard depends on
    it), while the new parser reports None so the hardware block can say "unknown"
    instead of asserting a single-chip run the operator never asked for."""
    from scripts.tt_hw_planner.commands.emit_e2e import _mesh_chip_count, _parsed_mesh_chips

    # unchanged contract, as pinned by test_emit_e2e_parallelism
    assert _mesh_chip_count(None) == 1
    assert _mesh_chip_count("") == 1
    assert _mesh_chip_count("2x2") == 4
    assert _mesh_chip_count("1x8") == 8
    assert _mesh_chip_count("garbage") == 1

    # the new distinction: only a cleanly-declared mesh yields a count
    assert _parsed_mesh_chips("2x2") == 4
    assert _parsed_mesh_chips("2,2") == 4
    assert _parsed_mesh_chips("1x1") == 1
    assert _parsed_mesh_chips("2X2") == 4, "case is not what makes a mesh invalid"
    assert _parsed_mesh_chips(" 2x2 ") == 4, "surrounding whitespace is fine"
    # `.lower()` turns a typed X into the separator, so `2xX` produces an EMPTY token
    # rather than a non-numeric one — which is why empties must be rejected.
    for bad in (None, "", "garbage", "2xX", "-1x4", "0x4", "x", "2x", "2x2x"):
        assert _parsed_mesh_chips(bad) is None, f"{bad!r} should be unparseable"


def test_lenient_count_is_byte_identical_to_the_old_contract():
    """Constraint 1: the topology guard and `optimize` both consume
    `_mesh_chip_count`, so its every observable value must be unchanged — including
    the odd ones the strict parser now rejects."""
    from scripts.tt_hw_planner.commands.emit_e2e import _mesh_chip_count

    expected = {
        None: 1,
        "": 1,
        "garbage": 1,
        "abc": 1,
        "x": 1,
        "2x2": 4,
        "2,2": 4,
        "2X2": 4,
        " 2x2 ": 4,
        "1x8": 8,
        "4x2": 8,
        "8x1": 8,
        "32x1": 32,
        "1x1": 1,
        "1x1x1": 1,
        "4": 4,
        "7": 7,
        "2xX": 2,  # lenient: the empty token is skipped
        "2x": 2,
        "2x2x": 4,
        "0x4": 1,  # clamped by the floor
        "-1x4": 1,
    }
    for value, want in expected.items():
        assert _mesh_chip_count(value) == want, f"{value!r}: contract changed"


def test_typo_mesh_does_not_claim_a_single_chip_run():
    """The point of the split: a mistyped mesh must budget against the whole box
    rather than silently asserting a 1-chip run."""
    box = find_box("GalaxyBH")
    block = _hardware_prompt_block(box, chips_in_use=_chips_in_use_as_emit_e2e_derives_it("nonsens"))
    assert f"{box.chips} (all of the box)" in block
    assert "opens 1 of the box's" not in block


def test_mem_cap_uses_the_shared_host_ram_reader():
    """Constraint 2: one place answers "how much RAM does this host have", so a cap
    and the concurrency scaler cannot disagree about the box."""
    import inspect

    from scripts.tt_hw_planner._cli_helpers.adaptive_scheduler import host_total_ram_bytes
    from scripts.tt_hw_planner.commands.emit_e2e import _agent_mem_cap_bytes

    src = inspect.getsource(_agent_mem_cap_bytes)
    assert "host_total_ram_bytes" in src
    assert "SC_PHYS_PAGES" not in src, "must not read the machine a second way"
    assert host_total_ram_bytes() > 0


def test_shared_host_ram_reader_survives_missing_psutil(monkeypatch):
    """psutil is imported lazily throughout this module because it may be absent; the
    fallback must still produce a real number rather than 0."""
    import builtins

    from scripts.tt_hw_planner._cli_helpers import adaptive_scheduler as A

    real_import = builtins.__import__

    def _no_psutil(name, *a, **k):
        if name == "psutil":
            raise ImportError("psutil unavailable")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_psutil)
    assert A.host_total_ram_bytes() > 0


def test_single_chip_mesh_does_not_advertise_the_whole_box():
    """The regression that mattered: `plan_parallelism` returns None for a 1-chip
    mesh, so deriving the count from the ParallelConfig fell back to the whole box.
    On a 32-chip box that advertised 32x the addressable memory."""
    for box in HARDWARE:
        if box.chips <= 1:
            continue
        used = _chips_in_use_as_emit_e2e_derives_it("1x1")
        assert used == 1
        block = _hardware_prompt_block(box, chips_in_use=used)
        assert f"opens 1 of the box's {box.chips}" in block, f"{box.name}: 1-chip mesh not scoped"
        assert f"{box.hbm_per_chip_gb:.0f} GB (1 x {box.hbm_per_chip_gb:.0f} GB)" in block
        assert "NOT the box total" in block


def test_no_mesh_given_still_means_whole_box():
    """An absent --mesh is genuinely unknown, and must stay distinguishable from an
    explicit single-chip mesh."""
    for box in HARDWARE:
        assert _chips_in_use_as_emit_e2e_derives_it(None) == 0
        block = _hardware_prompt_block(box, chips_in_use=0)
        assert f"{box.chips} (all of the box)" in block


def test_every_box_and_mesh_through_emit_e2e_derivation():
    """End-to-end over the real wiring: every box, every canonical mesh it declares,
    with the chip count derived the way emit-e2e derives it from --mesh."""
    import re

    for box in HARDWARE:
        for rows, cols in box.mesh_shapes:
            for mesh_arg in (f"{rows}x{cols}", f"{rows},{cols}"):
                used = _chips_in_use_as_emit_e2e_derives_it(mesh_arg)
                assert used == rows * cols, f"{box.name} {mesh_arg}: chip count misderived"
                block = _hardware_prompt_block(box, chips_in_use=used)
                assert f"DRAM per chip: {box.hbm_per_chip_gb:.0f} GB" in block
                if used == box.chips:
                    assert f"available to this run: {box.total_hbm_gb:.0f} GB" in block
                    assert "NOT the box total" not in block
                else:
                    m = re.search(r"available to THIS RUN: (\d+) GB", block)
                    assert m, f"{box.name} {mesh_arg}: not run-scoped"
                    assert float(m.group(1)) == used * box.hbm_per_chip_gb


def _drive_phase_a(monkeypatch, tmp_path, *, box_name, mesh, demo=None, expect_rc=1):
    """Run the REAL `_emit_e2e_phase_a` and return the prompt it actually built.

    Asserting on source text only proves the expression is written down, not that it
    reaches the builder. This drives the true call site instead. The builder is
    stubbed to exit non-zero with no `tt/` present, which is the documented path that
    returns before PHASE 3 — so no agent is spawned and no device is touched."""
    from scripts.tt_hw_planner.commands import emit_e2e as E
    from scripts.tt_hw_planner.parallelism import ParallelConfig

    demo = demo if demo is not None else (tmp_path / "demo")
    (demo / "_stubs").mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)  # `generated/<log>` is written relative to cwd

    captured = {}

    def _fake_agent(*, prompt, **_kw):
        captured["prompt"] = prompt
        return 1, ""  # rc!=0 with no tt/ dir -> returns before PHASE 3

    monkeypatch.setattr(E, "_run_agent", _fake_agent)
    # Keep this offline: both of these probe the model over the network otherwise.
    monkeypatch.setattr(E, "_required_heads_block", lambda *_a, **_k: "")
    chips = E._mesh_chip_count(mesh) if mesh else 1
    monkeypatch.setattr(
        E,
        "_planned_parallelism",
        lambda *_a, **_k: (ParallelConfig(tp=chips, dp=1) if chips > 1 else None),
    )

    args = SimpleNamespace(
        model_id="vendor/some-model",
        output=str(demo),
        pcc_target=0.95,
        box=box_name,
        mesh=mesh,
        batch=1,
        all_tasks=False,
        max_grade_rounds=0,
        model="opus",
        agent_bin="claude",
        agent_timeout_s=60,
    )
    rc = E._emit_e2e_phase_a(args)
    assert rc == expect_rc, f"phase A returned rc={rc}, expected {expect_rc}"
    if expect_rc == 1:
        # rc=1 is the stubbed builder's early return, which proves PHASE 3 never ran.
        assert "prompt" in captured, "the builder was never invoked"
    return captured.get("prompt", "")


def test_real_phase_a_prompt_carries_each_box_and_mesh(monkeypatch, tmp_path):
    """The end-to-end link: for every box and every canonical mesh, drive the actual
    phase-A entry point and read the figures out of the prompt it handed the
    builder."""
    import re

    for box in HARDWARE:
        for rows, cols in box.mesh_shapes:
            used = rows * cols
            prompt = _drive_phase_a(monkeypatch, tmp_path, box_name=box.name, mesh=f"{rows}x{cols}")
            assert f"HARDWARE — {box.name} ({box.chips}x {box.arch})" in prompt
            assert f"DRAM per chip: {box.hbm_per_chip_gb:.0f} GB" in prompt
            if used == box.chips:
                assert f"available to this run: {box.total_hbm_gb:.0f} GB" in prompt
            else:
                m = re.search(r"available to THIS RUN: (\d+) GB", prompt)
                assert m, f"{box.name} {rows}x{cols}: prompt not run-scoped"
                assert float(m.group(1)) == used * box.hbm_per_chip_gb, f"{box.name} {rows}x{cols}"


def test_real_phase_a_single_chip_mesh_is_scoped(monkeypatch, tmp_path):
    """The exact regression, through the real entry point: a 1-chip mesh on a
    multi-chip box must not advertise the box total."""
    for box in HARDWARE:
        if box.chips <= 1:
            continue
        prompt = _drive_phase_a(monkeypatch, tmp_path, box_name=box.name, mesh="1x1")
        assert f"opens 1 of the box's {box.chips}" in prompt
        assert f"THIS RUN: {box.hbm_per_chip_gb:.0f} GB" in prompt
        assert f"available to this run: {box.total_hbm_gb:.0f} GB" not in prompt


def test_real_phase_a_without_box_has_no_hardware_block(monkeypatch, tmp_path):
    """Non-breaking guarantee, proven through the real call site rather than by
    calling the helper directly."""
    prompt = _drive_phase_a(monkeypatch, tmp_path, box_name=None, mesh="2x2")
    assert "HARDWARE —" not in prompt
    assert "DRAM per chip" not in prompt


def test_guard_and_block_agree_on_the_same_run(monkeypatch, tmp_path):
    """The guard and the hardware block must describe the SAME run. Proven through
    behaviour rather than by grepping the source: a manifest that matches the mesh
    passes the guard (no rc=2 abort), and the prompt that follows is scoped to that
    same mesh rather than to the whole box."""
    import json
    import re

    box = find_box("T3K")  # 8 chips, so a 4-chip mesh is a genuine partial
    demo = tmp_path / "demo"
    (demo / "_stubs").mkdir(parents=True, exist_ok=True)
    (demo / "parallelism_manifest.json").write_text(json.dumps({"chips": 4, "tp": 4, "dp": 1, "mesh": [1, 4]}))

    prompt = _drive_phase_a(monkeypatch, tmp_path, box_name=box.name, mesh="1x4", demo=demo)
    # rc==1 from the stubbed builder (asserted inside the helper) proves the guard did
    # not abort with rc=2 — the manifest and the mesh agreed.
    m = re.search(r"available to THIS RUN: (\d+) GB", prompt)
    assert m, "guard passed but the block was not run-scoped"
    assert float(m.group(1)) == 4 * box.hbm_per_chip_gb
    assert f"opens 4 of the box's {box.chips}" in prompt


def test_guard_still_aborts_when_the_mesh_contradicts_the_manifest(monkeypatch, tmp_path):
    """The new wiring must not have weakened the existing guard: a mesh that does not
    match the graduated topology still aborts before any builder runs."""
    import json

    demo = tmp_path / "demo"
    (demo / "_stubs").mkdir(parents=True, exist_ok=True)
    (demo / "parallelism_manifest.json").write_text(json.dumps({"chips": 8, "tp": 8, "dp": 1, "mesh": [1, 8]}))

    # rc=2 is the guard's abort, asserted directly rather than inferred.
    prompt = _drive_phase_a(monkeypatch, tmp_path, box_name="T3K", mesh="1x4", demo=demo, expect_rc=2)
    assert prompt == "", "the builder must not be invoked when the guard aborts"


def test_partial_mesh_budgets_against_the_chips_actually_opened():
    """A box can be run on fewer chips than it holds. Handing the builder the BOX
    total would give it an aggregate budget larger than it can address — the same
    wrong-arithmetic class this block exists to prevent. Checked across every box
    and every canonical mesh that box declares."""
    import re

    for box in HARDWARE:
        for rows, cols in box.mesh_shapes:
            used = rows * cols
            if used >= box.chips:
                continue
            block = _hardware_prompt_block(box, chips_in_use=used)
            expected = used * box.hbm_per_chip_gb
            m = re.search(r"Aggregate DRAM available to THIS RUN: (\d+) GB", block)
            assert m, f"{box.name} mesh {rows}x{cols}: no run-scoped aggregate"
            assert float(m.group(1)) == expected, f"{box.name} mesh {rows}x{cols}: wrong run aggregate"
            assert f"opens {used} of the box's {box.chips}" in block
            assert "NOT the box total" in block, "the box total must be explicitly disclaimed"


def test_full_mesh_reports_the_box_total_without_a_disclaimer():
    """When the run uses the whole box there is nothing to disambiguate, so the
    prompt stays simple."""
    for box in HARDWARE:
        block = _hardware_prompt_block(box, chips_in_use=box.chips)
        assert f"{box.chips} (all of the box)" in block
        assert "NOT the box total" not in block
        assert f"Aggregate DRAM available to this run: {box.total_hbm_gb:.0f} GB" in block


def test_absent_or_bogus_chip_count_falls_back_to_the_whole_box():
    """chips_in_use=0 is the "unknown" case (no mesh given); a count larger than the
    box, or negative, must not produce a bigger budget than the hardware has."""
    box = find_box("T3K")
    for bogus in (0, -1, box.chips + 1, 9999):
        block = _hardware_prompt_block(box, chips_in_use=bogus)
        assert f"{box.chips} (all of the box)" in block, f"chips_in_use={bogus} must clamp to the box"
        assert f"{box.total_hbm_gb:.0f} GB" in block


def test_hardware_and_placement_blocks_agree_on_chip_count():
    """The two blocks land in the same prompt; if one says 8 chips and the other
    says a 4-chip mesh, the builder has to guess which governs."""
    import re

    from scripts.tt_hw_planner.commands.emit_e2e import _parallelism_prompt_block
    from scripts.tt_hw_planner.parallelism import ParallelConfig

    box = find_box("T3K")
    pc = ParallelConfig(tp=4, dp=1)
    prompt = _build_agent_prompt(
        model_id="m/x",
        demo_dir="/tmp/x",
        pcc=0.95,
        hardware_note=_hardware_prompt_block(box, chips_in_use=pc.chips),
        parallel_note=_parallelism_prompt_block(pc),
    )
    assert f"{pc.chips}-CHIP MESH" in prompt
    assert f"opens {pc.chips} of the box's {box.chips}" in prompt
    run_totals = re.findall(r"Aggregate DRAM available to THIS RUN: (\d+) GB", prompt)
    assert run_totals == [str(int(pc.chips * box.hbm_per_chip_gb))]


def test_every_registered_box_renders_its_own_figures():
    """Correctness for ALL boxes, not just the one that exposed the bug: read the
    numbers back out of the rendered block and compare to the box they came from.
    Several boxes share a DRAM figure, so identity is checked numerically per box
    rather than by asserting other boxes' values are absent."""
    import re

    for box in HARDWARE:
        block = _hardware_prompt_block(box)
        m = re.search(r"DRAM per chip: (\d+) GB", block)
        assert m, f"{box.name}: no DRAM line rendered"
        assert float(m.group(1)) == box.hbm_per_chip_gb, f"{box.name}: wrong per-chip DRAM"
        t = re.search(r"Aggregate DRAM available to this run: (\d+) GB", block)
        assert t, f"{box.name}: no aggregate line rendered"
        assert float(t.group(1)) == box.total_hbm_gb, f"{box.name}: wrong total DRAM"

        u = re.search(r"overhead: ([\d.]+) GB\s+at TP=1, ([\d.]+) GB", block)
        assert u, f"{box.name}: no usable line rendered"
        assert float(u.group(1)) == round(box.usable_per_chip_gb(1), 1), f"{box.name}: wrong TP=1 usable"
        assert float(u.group(2)) == round(box.usable_per_chip_gb(2), 1), f"{box.name}: wrong TP>1 usable"

        assert f"— {box.name} (" in block, f"{box.name}: box not named in the header"
        assert f"{box.chips}x {box.arch}" in block, f"{box.name}: chips/arch not in the header"


def test_emit_e2e_accepts_exactly_the_boxes_auto_up_accepts():
    """The flag must span the same hardware table as auto-up, so a box that can be
    brought up can also be handed to emit-e2e. Read the accepted set out of
    argparse's own rejection message — no device work, no parser refactor."""
    import re
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.tt_hw_planner", "emit-e2e", "m/x", "--box", "definitely-not-a-box"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 2, f"argparse should reject an unknown box; got rc={proc.returncode}"
    m = re.search(r"choose from ([^)]+)\)", proc.stderr)
    assert m, f"no choice list in stderr: {proc.stderr[-400:]}"
    offered = {tok.strip().strip("'\"") for tok in m.group(1).split(",")}
    assert offered == {b.name for b in HARDWARE}, f"offered {sorted(offered)}"


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


def _host_total() -> int:
    """The same reader the cap uses, so the two cannot disagree about the box."""
    from scripts.tt_hw_planner._cli_helpers.adaptive_scheduler import host_total_ram_bytes

    return host_total_ram_bytes()


def test_mem_cap_is_a_fraction_of_physical_ram():
    cap = _agent_mem_cap_bytes()
    total = _host_total()
    assert 0 < cap < total, "a cap must leave headroom for everything else on the box"
    assert cap == int(total * 0.7)


def test_mem_cap_is_disablable(monkeypatch):
    """An escape hatch matters: a legitimately huge model on a quiet box should not
    be capped into failure."""
    monkeypatch.setenv("TT_HW_PLANNER_AGENT_MEM_FRACTION", "0")
    assert _agent_mem_cap_bytes() == 0


def test_mem_cap_fraction_is_tunable(monkeypatch):
    total = _host_total()
    monkeypatch.setenv("TT_HW_PLANNER_AGENT_MEM_FRACTION", "0.25")
    assert _agent_mem_cap_bytes() == int(total * 0.25)


def test_mem_cap_survives_a_garbage_fraction(monkeypatch):
    monkeypatch.setenv("TT_HW_PLANNER_AGENT_MEM_FRACTION", "not-a-number")
    total = _host_total()
    assert _agent_mem_cap_bytes() == int(total * 0.7)


def test_mem_cap_clamps_at_all_of_ram(monkeypatch):
    total = _host_total()
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
