# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Long-context accuracy of the device KDA layer against the FP64 golden of real Kimi-K3 ``layers.0`` inputs.

The device runs consecutive full prefill calls from token 0, carrying its own state, over the real code tokens the
golden holds. The golden (``tests/kda/long_context_golden.py``, ``K3_LAYER0_GOLDEN`` in the long-context golden
registry, ``tests/accuracy/golden_registry.py``) was prepared on CPU outside the device lock, and this test only
loads it; a missing or incomplete golden fails the test. Error is scored with the shared windowed core
(``tests/accuracy/windowed.py``) and gated as NRMSE against the golden: per call (with PCC), over the whole run,
per 1,024-token window (drift with context), per token (the golden-wide floor from its manifest), and for the final
recurrent state per head (one drifting head is not averaged away).

Needs ``KIMI_K3_CKPT`` (layers.0 and embedding shards, checked against the golden's digests); skips without it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import kimi_k3_kda_config
from models.demos.deepseek_v3_d_p.tests.accuracy.windowed import Bar, RowSeries, check, nrmse, row_nrmse, row_sums
from models.demos.deepseek_v3_d_p.tests.kda.checkpoint_utils import kda_state_dict_sha256, load_kda_layer_state_dict
from models.demos.deepseek_v3_d_p.tests.kda.first_layer_input import FirstLayerInput
from models.demos.deepseek_v3_d_p.tests.kda.kda_test_utils import assert_accurate, make_actual_start
from models.demos.deepseek_v3_d_p.tests.kda.kda_test_utils import nrmse as slice_nrmse
from models.demos.deepseek_v3_d_p.tests.kda.long_context_golden import (
    K3_LAYER0_GOLDEN,
    load_window,
    open_k3_layer0_golden,
)
from models.demos.deepseek_v3_d_p.tests.kda.utils import (
    KimiK3TestCase,
    kda_device_params,
    make_kimi_k3_device_case,
    mla_row_permutation,
    reconstruct_sp_tp_tensor,
    reconstruct_state_at_sp_rank,
    to_sp_input,
)
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS

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
    [pytest.param((2, 4), 1, kda_device_params((2, 4)), id="SP2xTP4-fabric-2d")],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("tokens", [5120, 20480, 1044480], ids=["5k", "20k", "1M"])
def test_kimi_k3_layer0_long_context_accuracy(
    mesh_device: ttnn.MeshDevice,
    tensor_parallel_axis: int,
    device_params: dict,
    tokens: int,
    kimi_k3_checkpoint_dir: Path,
) -> None:
    registered = open_k3_layer0_golden()
    manifest, run_dir, checkpoint = registered.manifest, registered.path, kimi_k3_checkpoint_dir
    assert tokens <= registered.rows, f"golden {K3_LAYER0_GOLDEN} has {registered.rows} rows, fewer than {tokens}"
    thresholds = _THRESHOLDS[tokens]
    floor = manifest["token_floor"]  # read before device time: a golden without it fails here, not after the run

    config = kimi_k3_kda_config()
    weights = load_kda_layer_state_dict(checkpoint, 0, config)
    identity = registered.identity
    assert kda_state_dict_sha256(weights) == identity["weights_sha256"], "the golden used other weights"
    token_ids = registered.token_ids()

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
    series = RowSeries()
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
        series.append(start, row_sums(golden, output))

    whole = series.windows(
        tokens, {"output_nrmse": nrmse, "token_nrmse_max": lambda s: float(row_nrmse(s, floor).max())}
    )
    windows = series.windows(_WINDOW, {"window_nrmse": nrmse})
    per_token = row_nrmse(series.sums(), floor)
    golden_state = torch.load(run_dir / f"segment_{tokens // manifest['segment'] - 1:06d}.pt", weights_only=True)
    recurrent = reconstruct_state_at_sp_rank(state.recurrent, mesh_device, sp_axis, tensor_parallel_axis, sp_size - 1)
    result = {
        "golden_id": K3_LAYER0_GOLDEN,
        "tokens": tokens,
        "output_nrmse": whole[0]["output_nrmse"],
        "call_nrmse_max": max(w["call_nrmse"] for w in series.windows(_CALL, {"call_nrmse": nrmse})),
        "window_nrmse_max": max(w["window_nrmse"] for w in windows),
        "window_nrmse_last": windows[-1]["window_nrmse"],
        "token_nrmse_median": float(per_token.median()),
        "token_nrmse_max": whole[0]["token_nrmse_max"],
        "token_nrmse_argmax": int(per_token.argmax()),
        "state_head_nrmse_max": float(
            slice_nrmse(golden_state["state"]["recurrent"][0], recurrent[0].double(), (0,)).max()
        ),
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
    failures = check(
        whole, [Bar("output_nrmse", "max", thresholds["output"]), Bar("token_nrmse_max", "max", thresholds["token"])]
    ) + check(windows, [Bar("window_nrmse", "max", thresholds["window"])])
    assert not failures, failures
