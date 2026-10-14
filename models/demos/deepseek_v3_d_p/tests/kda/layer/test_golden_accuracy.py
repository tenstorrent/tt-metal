# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Measure the device KDA layer against a prepared FP64 golden of real Kimi-K3 ``layers.0`` inputs.

The golden comes from ``tests/kda/golden_run.py``, prepared on CPU outside the device lock; point
``KDA_GOLDEN_RUN`` at its directory, and ``KDA_TRACE_DIR`` at the vLLM trace when its tokens came from one.
This test only loads the golden, so it never computes a reference while holding the device. The device runs consecutive full prefill calls, carrying its own state from token 0,
and the result is reported as per-token, per-position and per-head-state NRMSE against the golden
(``KDA_GOLDEN_ACCURACY=`` line). It fails only on non-finite output; the budget gate belongs to the CI test.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import kimi_k3_kda_config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric_1d_device_params
from models.demos.deepseek_v3_d_p.tests.kda.accuracy_metric import position_profile, slice_nrmse
from models.demos.deepseek_v3_d_p.tests.kda.checkpoint_utils import kda_state_dict_sha256, load_kda_layer_state_dict
from models.demos.deepseek_v3_d_p.tests.kda.golden_run import MANIFEST, load_window, token_ids_sha256
from models.demos.deepseek_v3_d_p.tests.kda.real_input import (
    FIRST_LAYER_INPUT,
    FirstLayerInput,
    gutenberg_token_ids,
    trace_token_ids,
    trace_token_source,
)
from models.demos.deepseek_v3_d_p.tests.kda.utils import (
    KimiK3TestCase,
    make_kimi_k3_device_case,
    mla_row_permutation,
    reconstruct_sp_tp_tensor,
    reconstruct_state_at_sp_rank,
    to_sp_input,
)
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import make_actual_start

_CALL = 5120  # tokens per prefill call, a multiple of the golden's 1024-token segments


@pytest.mark.parametrize(
    "mesh_device,tensor_parallel_axis,device_params",
    [pytest.param((2, 4), 1, fabric_1d_device_params(), id="SP2xTP4-fabric-1d")],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("calls", [1, 4])
def test_kimi_k3_layer0_against_fp64_golden(
    mesh_device: ttnn.MeshDevice, tensor_parallel_axis: int, device_params: dict, calls: int
) -> None:
    run_dir = os.environ.get("KDA_GOLDEN_RUN")
    checkpoint = os.environ.get("KIMI_K3_CKPT")
    if not run_dir or not checkpoint:
        pytest.skip("set KDA_GOLDEN_RUN (a prepared golden_run directory) and KIMI_K3_CKPT")
    run_dir, checkpoint = Path(run_dir), Path(checkpoint)
    manifest = json.loads((run_dir / MANIFEST).read_text(encoding="utf-8"))
    source = manifest["identity"]["token_source"]
    if source == "gutenberg pg1342":
        token_ids = gutenberg_token_ids(Path.home() / ".cache" / "tt-linear-layers-shared" / "text_corpus")
    else:
        trace = os.environ.get("KDA_TRACE_DIR")
        if not trace or trace_token_source(Path(trace)) != source:
            pytest.skip(f"golden tokens come from {source!r}: point KDA_TRACE_DIR at that trace directory")
        token_ids = trace_token_ids(Path(trace))
    token_ids = token_ids[: manifest["tokens"]]
    assert token_ids_sha256(token_ids) == manifest["token_ids_sha256"], "golden used other tokens"

    config = kimi_k3_kda_config()
    weights = load_kda_layer_state_dict(checkpoint, 0, config)
    assert kda_state_dict_sha256(weights) == manifest["identity"]["weights_sha256"], "golden used other weights"
    assert manifest["identity"]["input"] == FIRST_LAYER_INPUT, f"golden input is {manifest['identity']['input']!r}"
    length = calls * _CALL
    golden_output, _ = load_window(run_dir, 0, length)  # raises if the golden is too short
    golden_state = torch.load(run_dir / f"segment_{length // manifest['segment'] - 1:06d}.pt", weights_only=True)
    golden_recurrent = golden_state["state"]["recurrent"][0]

    build = FirstLayerInput(checkpoint)
    sp_axis = 1 - tensor_parallel_axis
    sp_size = tuple(mesh_device.shape)[sp_axis]
    local_rows = _CALL // sp_size
    permutation = mla_row_permutation(0, sp_size, local_rows)
    first = KimiK3TestCase(config, weights, build(token_ids[:_CALL]), manifest["identity"]["weights_sha256"])
    layer, hidden_tt = make_kimi_k3_device_case(
        mesh_device, first, tensor_parallel_axis=tensor_parallel_axis, cache_weights=False
    )
    ttnn.deallocate(hidden_tt)
    actual_start_tt = make_actual_start(layer.device, 0)
    actual_end_tt = make_actual_start(layer.device, _CALL)

    state = layer.allocate_state(batch_size=1)
    outputs = []
    for call in range(calls):
        hidden = build(token_ids[call * _CALL : (call + 1) * _CALL])
        hidden_tt = to_sp_input(hidden[:, permutation, :], mesh_device, sp_axis)
        with ttnn.manage_config("throw_exception_on_fallback", True):
            output_tt, state = layer.forward(hidden_tt, state, actual_start_tt, actual_end_tt)
        rotated = reconstruct_sp_tp_tensor(output_tt, mesh_device, sp_axis, tensor_parallel_axis, tp_dim=2, sp_dim=1)
        natural = torch.empty_like(rotated)
        natural[:, permutation, :] = rotated
        outputs.append(natural[0])
        ttnn.deallocate(hidden_tt)
        ttnn.deallocate(output_tt)
    output = torch.cat(outputs)
    recurrent = reconstruct_state_at_sp_rank(state.recurrent, mesh_device, sp_axis, tensor_parallel_axis, sp_size - 1)[
        0
    ]

    assert torch.isfinite(output).all(), "device KDA output has non-finite values"
    per_token = slice_nrmse(output, golden_output, keep_dims=(0,))
    per_head_state = slice_nrmse(recurrent, golden_recurrent, keep_dims=(0,))
    result = {
        "implementation": "TT main (T1)",
        "layout": f"SP{sp_size}xTP{tuple(mesh_device.shape)[tensor_parallel_axis]}",
        "golden": str(run_dir),
        "golden_identity": manifest["identity"],
        "tokens": length,
        "calls": calls,
        "output_nrmse": float(slice_nrmse(output, golden_output, keep_dims=())),
        "output_token_nrmse_median": float(per_token.median()),
        "output_token_nrmse_max": float(per_token.max()),
        "output_token_nrmse_argmax": int(per_token.argmax()),
        "output_nrmse_per_1024": [float(x) for x in position_profile(output, golden_output, token_dim=0, bucket=1024)],
        "state_nrmse_per_head_median": float(per_head_state.median()),
        "state_nrmse_per_head_max": float(per_head_state.max()),
        "state_nrmse_per_head_argmax": int(per_head_state.argmax()),
    }
    print("KDA_GOLDEN_ACCURACY=" + json.dumps(result, sort_keys=True))
