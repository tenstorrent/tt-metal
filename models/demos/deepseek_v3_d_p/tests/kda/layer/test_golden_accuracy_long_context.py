# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Long-context accuracy of the device KDA layer against the FP64 golden of real Kimi-K3 ``layers.0`` inputs.

The device runs consecutive full prefill calls from token 0, carrying its own state, over real code tokens from a
vLLM trace; the golden (``tests/kda/long_context_golden.py``) was prepared on CPU outside the device lock, and this
test only loads it. Error is reported and gated as NRMSE against the golden: per call (with PCC), over the whole
window, per 1,024-token window (drift with context), per token (the golden-wide floor from its manifest), and
for the final recurrent state per head (one drifting head is not averaged away).

Needs ``KIMI_K3_CKPT`` (layers.0 and embedding shards), ``KDA_LONG_CONTEXT_GOLDEN`` (a prepared golden) and
``KDA_TRACE_DIR`` (the vLLM trace whose token ids the golden used); skips without them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.common.prefill.runners.runner_utils import load_trace_token_ids, resolve_trace_dir
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import kimi_k3_kda_config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric_1d_device_params
from models.demos.deepseek_v3_d_p.tests.kda.checkpoint_utils import kda_state_dict_sha256, load_kda_layer_state_dict
from models.demos.deepseek_v3_d_p.tests.kda.first_layer_input import FIRST_LAYER_INPUT, FirstLayerInput
from models.demos.deepseek_v3_d_p.tests.kda.long_context_golden import MANIFEST, load_window, token_ids_sha256
from models.demos.deepseek_v3_d_p.tests.kda.utils import (
    KimiK3TestCase,
    make_kimi_k3_device_case,
    mla_row_permutation,
    reconstruct_sp_tp_tensor,
    reconstruct_state_at_sp_rank,
    to_sp_input,
)
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import (
    TokenErrorAccumulator,
    assert_accurate,
    make_actual_start,
    nrmse,
)

# The 1M case runs ~7 min on SP2xTP4 plus setup; pytest.ini's 300 s default would cut it short.
pytestmark = [run_for_blackhole(), pytest.mark.timeout(900)]

_CALL = PREFILL_CHUNK_TOKENS  # tokens per prefill call, as in production; a multiple of the golden's segments
_WINDOW = 1024
_PCC = 0.999
# PCC is the structural sanity gate; NRMSE carries precision. The worst head's state drifts with context on
# main (PCC 0.9975 and NRMSE 0.16 after 1M tokens), so the state's per-head sanity bound is looser than the output's.
_STATE_PCC = 0.99

# NRMSE thresholds per context length (call, output, window, token, state head).
_THRESHOLDS: dict[int, dict[str, float]] = {
    5120: {"call": 0.015, "output": 0.015, "window": 0.02, "token": 0.03, "state_head": 0.06},
    20480: {"call": 0.02, "output": 0.02, "window": 0.02, "token": 0.03, "state_head": 0.15},
    1044480: {"call": 0.025, "output": 0.02, "window": 0.03, "token": 0.14, "state_head": 0.2},
}


@pytest.mark.parametrize(
    "mesh_device,tensor_parallel_axis,device_params",
    [pytest.param((2, 4), 1, fabric_1d_device_params(), id="SP2xTP4-fabric-1d")],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("tokens", [5120, 20480, 1044480], ids=["5k", "20k", "1M"])
def test_kimi_k3_layer0_long_context_accuracy(
    mesh_device: ttnn.MeshDevice, tensor_parallel_axis: int, device_params: dict, tokens: int
) -> None:
    run_dir, checkpoint, trace = (
        os.environ.get(name) for name in ("KDA_LONG_CONTEXT_GOLDEN", "KIMI_K3_CKPT", "KDA_TRACE_DIR")
    )
    if not (run_dir and checkpoint and trace):
        pytest.skip("set KDA_LONG_CONTEXT_GOLDEN (a prepared golden), KIMI_K3_CKPT and KDA_TRACE_DIR")
    run_dir, checkpoint, trace = Path(run_dir), Path(checkpoint), resolve_trace_dir(trace)
    manifest = json.loads((run_dir / MANIFEST).read_text(encoding="utf-8"))
    if manifest["segments"][-1]["end"] < tokens or "token_floor" not in manifest:
        pytest.skip(f"the golden in {run_dir} is incomplete or shorter than {tokens} tokens")
    thresholds = _THRESHOLDS[tokens]

    config = kimi_k3_kda_config()
    weights = load_kda_layer_state_dict(checkpoint, 0, config)
    identity = manifest["identity"]
    assert kda_state_dict_sha256(weights) == identity["weights_sha256"], "the golden used other weights"
    assert identity["input"] == FIRST_LAYER_INPUT, f"the golden's input is {identity['input']!r}"
    token_ids = load_trace_token_ids(trace)[: manifest["tokens"]]
    assert token_ids_sha256(token_ids) == manifest["token_ids_sha256"], "the golden used other tokens"

    build = FirstLayerInput(checkpoint)
    assert build.weights_sha256() == identity["input_weights_sha256"], "the golden's input used other embeddings"
    sp_axis = 1 - tensor_parallel_axis
    sp_size = tuple(mesh_device.shape)[sp_axis]
    permutation = mla_row_permutation(0, sp_size, _CALL // sp_size)
    first = KimiK3TestCase(config, weights, build(token_ids[:_CALL]), identity["weights_sha256"])
    layer, hidden_tt = make_kimi_k3_device_case(
        mesh_device, first, tensor_parallel_axis=tensor_parallel_axis, cache_weights=False
    )
    ttnn.deallocate(hidden_tt)
    actual_start_tt = make_actual_start(layer.device, 0)
    actual_end_tt = make_actual_start(layer.device, _CALL)

    state = layer.allocate_state(batch_size=1)
    errors = TokenErrorAccumulator()
    for start in range(0, tokens, _CALL):
        hidden_tt = to_sp_input(build(token_ids[start : start + _CALL])[:, permutation, :], mesh_device, sp_axis)
        with ttnn.manage_config("throw_exception_on_fallback", True):
            output_tt, state = layer.forward(hidden_tt, state, actual_start_tt, actual_end_tt)
        rotated = reconstruct_sp_tp_tensor(output_tt, mesh_device, sp_axis, tensor_parallel_axis, tp_dim=2, sp_dim=1)
        ttnn.deallocate(hidden_tt)
        ttnn.deallocate(output_tt)
        output = torch.empty_like(rotated)
        output[:, permutation, :] = rotated
        golden, _ = load_window(run_dir, start, start + _CALL)
        output = output[0].double()
        assert_accurate(
            golden,
            output,
            name=f"layers.0 output tokens [{start}, {start + _CALL})",
            pcc_threshold=_PCC,
            nrmse_threshold=thresholds["call"],
        )
        errors.update(golden, output)

    windows = errors.window_nrmse(_WINDOW)
    per_token = errors.token_nrmse(manifest["token_floor"])
    golden_state = torch.load(run_dir / f"segment_{tokens // manifest['segment'] - 1:06d}.pt", weights_only=True)
    recurrent = reconstruct_state_at_sp_rank(state.recurrent, mesh_device, sp_axis, tensor_parallel_axis, sp_size - 1)
    result = {
        "tokens": tokens,
        "output_nrmse": errors.nrmse(),
        "call_nrmse_max": float(errors.window_nrmse(_CALL).max()),
        "window_nrmse_max": float(windows.max()),
        "window_nrmse_last": float(windows[-1]),
        "token_nrmse_median": float(per_token.median()),
        "token_nrmse_max": float(per_token.max()),
        "token_nrmse_argmax": int(per_token.argmax()),
        "state_head_nrmse_max": float(nrmse(golden_state["state"]["recurrent"][0], recurrent[0].double(), (0,)).max()),
    }
    print("KDA_LONG_CONTEXT_ACCURACY=" + json.dumps(result, sort_keys=True))
    assert_accurate(
        golden_state["state"]["recurrent"][0],
        recurrent[0].double(),
        name=f"layers.0 recurrent state after {tokens} tokens",
        pcc_threshold=_STATE_PCC,
        nrmse_threshold=thresholds["state_head"],
        nrmse_keep_dims=(0,),
        pcc_keep_dims=(0,),
    )
    assert result["output_nrmse"] <= thresholds["output"], result
    assert result["window_nrmse_max"] <= thresholds["window"], result
    assert result["token_nrmse_max"] <= thresholds["token"], result
