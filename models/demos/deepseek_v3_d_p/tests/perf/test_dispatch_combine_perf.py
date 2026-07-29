# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Device performance tests for DeepSeek/Kimi MoE dispatch and combine operations.

Replays the hottest (layer, col) pairs of a real prefill capture on LB 8x1 — one
worker spawn per (layer, col, topology) runs TtDispatchModule → production layout
transform (squeeze → TILE+bfp8 → unsqueeze) → TtCombineModule(init_zeros=True)
end-to-end on device. Tracy captures DispatchDeviceOperation, the layout op(s),
and CombineDeviceOperation in one CSV; the perf wrapper asserts dispatch and
combine independently so a regression localizes to the responsible kernel.

The captures hold Galaxy-global expert IDs; the loader (`load_captured_routing`)
shifts a column's own experts down to [0, experts_per_col) and sends the rest to
sentinel 255, so the LB single-col combine kernel (first_expert_id=0) interprets
them correctly, then slices the gate outputs to [0:1] for LB's single dispatch
group. The per-model kernel config lives in the worker's parametrize entries.
"""

import pytest

from models.demos.deepseek_v3_d_p.utils.perf_utils import run_model_device_perf_test_per_op

_REAL_INDICES_TOPOS = [("linear", 2), ("ring", 2)]

# Picks come from the production 11x5120 chunked-prefill run (code_debug), for both models we ship.
# One chunk per case: 5120 tokens over the 8-chip dispatch group => seq_len_per_chip 640, hence the
# worker's perf_captured_*_chunk entries.
#
# TT_DS_USE_CAPTURED_INDICES (read by the worker, inherited from the environment) is the
# expert_routing.safetensors holding BOTH models' cases, keyed by model. The test is parametrized
# per model and the worker slices that model's tensors out of the file.
#
# A pick contributes a test only once its baselines are filled in, so this is inert by default: the
# 75W box the captures came from cannot produce valid perf numbers, and nothing is asserted against
# a guessed baseline.

_DS_CHUNK_PICKS = [(19, 2), (29, 0), (42, 3), (24, 1)]  # 37.2 / 37.0 / 36.9 / 36.7 % in-col share
_KIMI_CHUNK_PICKS = [(45, 2), (48, 0), (44, 1), (50, 1)]  # 38.6 / 38.1 / 37.4 / 37.1 %
_DISPATCH_DS_CHUNK_EXPECTED_NS: dict[tuple[str, int, int, int], int] = {}
_COMBINE_DS_CHUNK_EXPECTED_NS: dict[tuple[str, int, int, int], int] = {}
_DISPATCH_KIMI_CHUNK_EXPECTED_NS: dict[tuple[str, int, int, int], int] = {}
_COMBINE_KIMI_CHUNK_EXPECTED_NS: dict[tuple[str, int, int, int], int] = {}

# model -> (picks, dispatch baselines, combine baselines). The model name is also the worker's
# parametrize id (perf_captured_<model>_chunk) and its namespace in the capture file.
_MODELS = {
    "dsv3": (_DS_CHUNK_PICKS, _DISPATCH_DS_CHUNK_EXPECTED_NS, _COMBINE_DS_CHUNK_EXPECTED_NS),
    "kimi26": (_KIMI_CHUNK_PICKS, _DISPATCH_KIMI_CHUNK_EXPECTED_NS, _COMBINE_KIMI_CHUNK_EXPECTED_NS),
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
    model: str = "",
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
    parametrize_id = f"perf_captured_{model}_chunk" if use_captured else "perf_no_pcc"
    k_filter = f"{parametrize_id} and {worker_id}"
    if worker_filter_extras:
        k_filter += f" and {worker_filter_extras}"
    if use_captured:
        model_name += f"_{model}_l{captured_layer:02d}_col{captured_col}"
        # TT_DS_USE_CAPTURED_INDICES is not forwarded: the worker reads the same var and inherits it.
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
            "DispatchDeviceOperation": dispatch_ns[(topo, nlinks, layer, col)],
            "CombineDeviceOperation": combine_ns[(topo, nlinks, layer, col)],
        },
        margin=0.045 if topo == "ring" else 0.03,
        captured_layer=layer,
        captured_col=col,
        model=model,
        worker_dir="models/demos/deepseek_v3_d_p/tests/perf",
    )
    for model, (picks, dispatch_ns, combine_ns) in _MODELS.items()
    for topo, nlinks in _REAL_INDICES_TOPOS
    for layer, col in picks
    if (topo, nlinks, layer, col) in dispatch_ns and (topo, nlinks, layer, col) in combine_ns
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
    is_ci_env,
):
    # Skip the Kimi perf test in CI; only DSv3 is gated for now.
    if is_ci_env and "kimi26" in model_name:
        pytest.skip("Only perf test for DeepSeek V3 in CI")
    run_model_device_perf_test_per_op(
        command=command,
        expected_per_op=expected_per_op,
        subdir=subdir,
        model_name=model_name,
        margin=margin,
        comments=comments,
        extra_env=extra_env,
    )
