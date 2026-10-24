"""Pin: bring-up must take its TP x DP split from select_parallelism.

Four places used to work out parallelism independently, and the one that
launches the demo asked none of them:

  * the fit table enumerates pure TP = chip count;
  * select_parallelism picks the largest TP that DIVIDES the chips and has no
    kernel blockers, filling the rest with DP;
  * bring-up's `prepare` inferred TP from the mesh's COLUMN count;
  * the demo splits the mesh by a `--data_parallel` flag that nothing passed,
    defaulting to one group over every chip.

For a model whose head counts divide 4 but not 8 (Qwen2.5-VL: 28 attention /
4 KV heads) that combination reported blockers at TP=8 and refused to emit a
command -- and when a mesh shape was chosen to imply TP=4, the demo still put
all eight chips in one TP group because the split never reached it.
"""

from __future__ import annotations

import inspect

from scripts.tt_hw_planner import bringup as bringup_mod
from scripts.tt_hw_planner.kernel_constraints import KernelReport, KernelFinding, Severity
from scripts.tt_hw_planner.parallelism import select_parallelism


def _report_blocking_only(blocked_tp: int, grid=(1, 2, 4, 8)) -> KernelReport:
    """A report where exactly `blocked_tp` carries a BLOCKER."""
    report = KernelReport(tp_grid=list(grid))
    for tp in grid:
        report.findings_by_tp[tp] = (
            [
                KernelFinding(
                    op="op",
                    field="num_attention_heads",
                    value=28,
                    constraint=f"not divisible by TP({tp})",
                    passes=False,
                    severity=Severity.BLOCKER,
                )
            ]
            if tp == blocked_tp
            else []
        )
    return report


def test_bringup_asks_the_selector_instead_of_reading_the_mesh_columns():
    src = inspect.getsource(bringup_mod)
    assert "select_parallelism" in src, "bring-up must route its split through select_parallelism"
    assert "chosen_tp = max(1, int(best.mesh_shape[1]))" not in src or "_pcfg" in src, (
        "bring-up must not infer TP from the mesh column count alone — that "
        "number knows nothing about the model's head counts"
    )


def test_selector_avoids_the_blocked_degree_and_fills_the_rest_with_dp():
    """The whole point: 8 chips, TP=8 blocked -> TP=4 x DP=2, not a refusal."""
    pcfg = select_parallelism(8, _report_blocking_only(8))
    assert pcfg.tp == 4, f"expected the largest viable TP, got {pcfg.tp}"
    assert pcfg.dp == 2, f"expected the remaining chips as DP replicas, got {pcfg.dp}"
    assert pcfg.tp * pcfg.dp == 8, "the split must use every chip"


def test_the_split_reaches_the_demo_command():
    """A split the demo never receives is a split that does not happen."""
    inv = bringup_mod._build_tt_transformers_invocation(
        hf_model="org/model",
        mesh_device="LABEL",
        accuracy=False,
        batch=1,
        max_seq_len=128,
        max_generated_tokens=8,
        trace=False,
        paged_attention=False,
        instruct=True,
        data_parallel=2,
    )
    argv = inv.argv()
    assert "--data_parallel" in argv, "the emitted command must carry the split"
    assert argv[argv.index("--data_parallel") + 1] == "2"


def test_a_single_group_adds_no_flag():
    """Every other model keeps the command it had before."""
    inv = bringup_mod._build_tt_transformers_invocation(
        hf_model="org/model",
        mesh_device="LABEL",
        accuracy=False,
        batch=1,
        max_seq_len=128,
        max_generated_tokens=8,
        trace=False,
        paged_attention=False,
        instruct=True,
    )
    assert "--data_parallel" not in inv.argv(), "dp=1 is the demo's default; do not pass it"
