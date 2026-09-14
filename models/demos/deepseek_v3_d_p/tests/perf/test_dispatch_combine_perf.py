# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Device performance tests for DeepSeekv3/KimiK2.6/GLM5.2 MoE dispatch and combine operations.

Runs test_prefill_dispatch_combine.py::test_ttnn_dispatch_combine[perf_captured_<model>_chunk]
(<model> in {dsv3, kimi26, glm52}) on the existing LB 8x1 proxy migrated to Fabric2D TorusY. It replays the hottest
(layer, col) pairs from a real prefill capture. The operation sequence is the same as the
production workflow.
Tests that were ran in order to capture the routing data:

The captures hold Galaxy-global expert IDs; the loader (`load_captured_routing`)
shifts a column's own experts down to [0, experts_per_col) and sends the rest to
sentinel (num_routed_experts-1), then runs one Galaxy column on the LoudBox.
"""

import pytest

from models.demos.deepseek_v3_d_p.utils.perf_utils import run_model_device_perf_test_per_op

# Picks come from the production 11x5120 chunked-prefill run (code_debug), for DSv3 and KimiK2.6.
# One chunk per case: 5120 tokens over the 8-chip dispatch group => seq_len_per_chip 640.

# Column send volume (in-col picks × token bytes) drives device time; ranked by
# analyze_routing_send.py. Each model gets its 2 hottest columns + 2 at uniform.
_DS_CHUNK_PICKS = [
    (19, 2),  # 37.2%
    (29, 0),  # 37.0%
    (4, 2),  # 25.0%
    (56, 3),  # 25.0%
]
_KIMI_CHUNK_PICKS = [
    (45, 2),  # 38.6%
    (48, 0),  # 38.1%
    (30, 3),  # 25.0%
    (32, 1),  # 25.0%
]
_GLM52_CHUNK_PICKS = [
    (3, 0),  # 47.3%
    (8, 0),  # 38.5%
    (14, 2),  # 25.0%
    (10, 0),  # 25.0%
]
# Key is (layer, col). The retained values are the old ring-profile measurements and are
# migration starting points for Fabric2D TorusY on the same LoudBox proxy.
_DISPATCH_DS_CHUNK_EXPECTED_NS: dict[tuple[int, int], int] = {
    (19, 2): 872_934,
    (29, 0): 837_074,
    (4, 2): 552_136,
    (56, 3): 634_804,
}
_COMBINE_DS_CHUNK_EXPECTED_NS: dict[tuple[int, int], int] = {
    (19, 2): 1_433_990,
    (29, 0): 1_372_652,
    (4, 2): 749_534,
    (56, 3): 905_037,
}
_DISPATCH_KIMI_CHUNK_EXPECTED_NS: dict[tuple[int, int], int] = {
    (45, 2): 878_836,
    (48, 0): 806_776,
    (30, 3): 553_826,
    (32, 1): 555_157,
}
_COMBINE_KIMI_CHUNK_EXPECTED_NS: dict[tuple[int, int], int] = {
    (45, 2): 1_267_431,
    (48, 0): 1_291_803,
    (30, 3): 870_001,
    (32, 1): 848_981,
}

_DISPATCH_GLM52_CHUNK_EXPECTED_NS: dict[tuple[int, int], int] = {
    (3, 0): 1_247_679,
    (8, 0): 815_378,
    (14, 2): 493_043,
    (10, 0): 473_592,
}
_COMBINE_GLM52_CHUNK_EXPECTED_NS: dict[tuple[int, int], int] = {
    (3, 0): 1_576_451,
    (8, 0): 1_218_739,
    (14, 2): 692_124,
    (10, 0): 806_988,
}

# model -> (picks, dispatch baselines, combine baselines).
_MODELS = {
    "dsv3": (_DS_CHUNK_PICKS, _DISPATCH_DS_CHUNK_EXPECTED_NS, _COMBINE_DS_CHUNK_EXPECTED_NS),
    # https://github.com/tenstorrent/tt-metal/issues/54972
    "kimi26": (_KIMI_CHUNK_PICKS, _DISPATCH_KIMI_CHUNK_EXPECTED_NS, _COMBINE_KIMI_CHUNK_EXPECTED_NS),
    "glm52": (_GLM52_CHUNK_PICKS, _DISPATCH_GLM52_CHUNK_EXPECTED_NS, _COMBINE_GLM52_CHUNK_EXPECTED_NS),
}


def _perf_param_per_op(
    op,
    worker_file,
    worker_test,
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
    worker_id = "fabric2d-torus-y-8x1-2link"
    model_name = f"deepseek_v3_{op}_torus_y_8x1_2link"
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
    command = f"pytest {worker_dir}/{worker_file}::{worker_test} -k '{k_filter}' --wrapper-invocation"
    return (
        command,
        expected_per_op,
        f"deepseek_v3_{op}",
        model_name,
        margin,
        "torus-y-8x1-2link",
        extra_env,
    )


_DISPATCH_COMBINE_PERF_PARAMS = [
    _perf_param_per_op(
        "dispatch_combine",
        "test_prefill_dispatch_combine.py",
        "test_ttnn_dispatch_combine",
        expected_per_op={
            "DispatchDeviceOperation": dispatch_ns[(layer, col)],
            "CombineDeviceOperation": combine_ns[(layer, col)],
        },
        margin=0.045,
        captured_layer=layer,
        captured_col=col,
        model=model,
        worker_dir="models/demos/deepseek_v3_d_p/tests/perf",
    )
    for model, (picks, dispatch_ns, combine_ns) in _MODELS.items()
    for layer, col in picks
    if (layer, col) in dispatch_ns and (layer, col) in combine_ns
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
