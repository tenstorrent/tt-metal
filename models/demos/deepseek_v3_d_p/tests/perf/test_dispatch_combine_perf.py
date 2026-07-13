# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Device performance tests for DeepSeek V3 MoE dispatch and combine operations.

Replays the 4 hottest (layer, col) pairs from the real longbook_qa_eng_25600
prefill capture on LB 8x1 — one worker spawn per (layer, col, topology) runs
TtDispatchModule → production layout transform (squeeze → TILE+bfp8 → unsqueeze)
→ TtCombineModule(init_zeros=True) end-to-end on device. Tracy captures
DispatchDeviceOperation, the layout op(s), and CombineDeviceOperation in one CSV;
the perf wrapper asserts dispatch and combine independently so a regression
localizes to the responsible kernel.

The 256-experts indexing space, top-k=8, experts_per_chip=8 explicit override are
needed because the captures are Galaxy-global IDs in [0, 256); the loader
(`load_captured_routing`) remaps them to [0, 64) ∪ {255} so the LB single-col
combine kernel (first_expert_id=0) interprets them correctly, then slices the
gate outputs to [0:1] for LB's single dispatch group.
"""

import pytest

from models.demos.deepseek_v3_d_p.utils.perf_utils import run_model_device_perf_test_per_op

# Top 4 absolute hottest (layer, col) pairs from LONGBOOK_QA_ENG_25600.
# Each token picks 8 of 256 experts; "in-col share" = fraction of those picks
# landing in the column's 64 experts (uniform random would be 25%).
_REAL_INDICES_PICKS: list[tuple[int, int]] = [
    # (layer, col)
    (27, 2),  # 43.2% in-col share — hottest in the corpus
    (38, 0),  # 41.2%
    (50, 0),  # 39.9%
    (28, 1),  # 39.5%
]
_REAL_INDICES_TOPOS = [("linear", 2), ("ring", 2)]

# Per-(topo, nlinks, layer, col) baselines in nanoseconds. Dispatch and combine
# are developed separately, so each is asserted against its own baseline —
# a regression localizes to the responsible kernel.
_DISPATCH_REAL_INDICES_EXPECTED_NS: dict[tuple[str, int, int, int], int] = {
    # (topo, nlinks, layer, col): expected_ns. Re-centered to the midpoint of the observed
    # min/max across 16 main-branch CI runs (2026-06-13..18) spanning 5 LB runners
    # (f01cs01/02/08, f04cs03/04), against LONGBOOK_QA_ENG_25600/expert_routing.safetensors.
    # Percent comment = dispatch-group in-col share for that layer/col.
    ("linear", 2, 27, 2): 12_129_621,  # 43.2%
    ("linear", 2, 38, 0): 7_158_222,  # 41.2%
    ("linear", 2, 50, 0): 8_484_394,  # 39.9%
    ("linear", 2, 28, 1): 11_039_234,  # 39.5%
    ("ring", 2, 27, 2): 7_216_744,
    ("ring", 2, 38, 0): 5_130_980,
    ("ring", 2, 50, 0): 4_848_732,
    ("ring", 2, 28, 1): 5_595_338,
}
_COMBINE_REAL_INDICES_EXPECTED_NS: dict[tuple[str, int, int, int], int] = {
    # Re-baselined to the 2026-06-24 measurement after a combine-kernel speedup.
    # 2026-06-25: the two linear-8-2link entries below were set from a single optimistic
    # 06-24 run; real CI measures higher (l27-col2 8.49 ms, l28-col1 9.75 ms), so they are
    # bumped to the observed values (margin 0.03). Unrelated to the routed-expert FFN change
    # (it does not touch combine); recheck if a real combine regression is suspected.
    ("linear", 2, 27, 2): 8_490_000,
    ("linear", 2, 38, 0): 7_139_837,
    ("linear", 2, 50, 0): 7_341_465,
    ("linear", 2, 28, 1): 9_750_000,
    ("ring", 2, 27, 2): 8_294_634,
    ("ring", 2, 38, 0): 5_308_695,
    ("ring", 2, 50, 0): 5_711_088,
    ("ring", 2, 28, 1): 7_746_036,
}


def _perf_param_per_op(
    op,
    worker_file,
    worker_test,
    topo,
    nlinks,
    expected_per_op: dict,
    margin: float = 0.03,
    captured_layer: int | None = None,
    captured_col: int | None = None,
    worker_filter_extras: str | None = "",
    worker_dir: str = "models/demos/deepseek_v3_d_p/tests/perf",
):
    """Build one pytest.param tuple for a per-op perf test.

    Each entry spawns one worker subprocess that runs dispatch+combine end-to-end
    on device. The result tuple carries an `expected_per_op` dict
    (op_code_substring → expected_ns) so dispatch and combine are asserted
    independently via `run_model_device_perf_test_per_op`.

    `worker_dir` is the path to the directory containing `worker_file`; defaults
    to the pcc test dir. Override to `tests/perf` for workers that only run the
    perf path and shouldn't be collected by the PCC pipeline.
    """
    worker_id = f"{topo}-8-{nlinks}link"
    model_name = f"deepseek_v3_{op}_{topo}_8_{nlinks}link"
    use_captured = captured_layer is not None and captured_col is not None
    parametrize_id = "perf_real_indices" if use_captured else "perf_no_pcc"
    k_filter = f"{parametrize_id} and {worker_id}"
    if worker_filter_extras:
        k_filter += f" and {worker_filter_extras}"
    if use_captured:
        model_name += f"_real_l{captured_layer:02d}_col{captured_col}"
        extra_env = {"TT_DS_CAPTURED_LAYER": str(captured_layer), "TT_DS_CAPTURED_COL": str(captured_col)}
    else:
        extra_env = {}
    command = f"pytest {worker_dir}/{worker_file}::{worker_test} -k '{k_filter}'"
    return (
        command,
        expected_per_op,
        f"deepseek_v3_{op}",
        model_name,
        margin,
        f"{topo}-8-{nlinks}link",
        extra_env,
    )


_DISPATCH_COMBINE_PERF_PARAMS = [
    _perf_param_per_op(
        "dispatch_combine",
        "test_prefill_dispatch_combine.py",
        "test_ttnn_dispatch_combine",
        topo,
        nlinks,
        expected_per_op={
            "DispatchDeviceOperation": _DISPATCH_REAL_INDICES_EXPECTED_NS[(topo, nlinks, layer, col)],
            "CombineDeviceOperation": _COMBINE_REAL_INDICES_EXPECTED_NS[(topo, nlinks, layer, col)],
        },
        margin=0.045 if topo == "ring" else 0.03,
        captured_layer=layer,
        captured_col=col,
        worker_dir="models/demos/deepseek_v3_d_p/tests/perf",
    )
    for topo, nlinks in _REAL_INDICES_TOPOS
    for layer, col in _REAL_INDICES_PICKS
]


def _ids_for(params):
    ids = []
    for p in params:
        mn = p[3]
        mn = mn.removeprefix("deepseek_v3_dispatch_combine_")
        ids.append(mn.replace("_", "-"))
    return ids


_PARAMS_HEADER = "command, expected_per_op, subdir, model_name, margin, comments, extra_env"


@pytest.mark.parametrize(
    _PARAMS_HEADER,
    _DISPATCH_COMBINE_PERF_PARAMS,
    ids=_ids_for(_DISPATCH_COMBINE_PERF_PARAMS),
)
@pytest.mark.models_device_performance_bare_metal
def test_device_perf_dispatch_combine(
    command,
    expected_per_op,
    subdir,
    model_name,
    margin,
    comments,
    extra_env,
):
    run_model_device_perf_test_per_op(
        command=command,
        expected_per_op=expected_per_op,
        subdir=subdir,
        model_name=model_name,
        margin=margin,
        comments=comments,
        extra_env=extra_env,
    )
