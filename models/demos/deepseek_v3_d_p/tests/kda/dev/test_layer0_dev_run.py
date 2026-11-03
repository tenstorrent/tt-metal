# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Development run of the device Kimi-K3 ``layers.0`` KDA layer over the 1M-token FP64 golden, with perf.

Not a gate: it measures and records. First it times one traced 5,120-token call at start 0 (real weights and
tokens; not the ``test_layer_perf`` gate, which times a start inside a rank); then it runs consecutive full prefill
calls from token 0 carrying its own state, exactly as ``test_golden_accuracy_long_context.py`` does, and after every
call writes the call's per-token output sums and per-head recurrent-state sums against the golden to the run
directory (format in ``layer0_run_analysis.py``) and prints a timestamped progress line. An interrupted run keeps
every finished call. Analyze with ``python layer0_run_analysis.py report <run dir>``.

``run.json`` records the commit, not the working tree: run committed code. It also records the host, its power
limit, the fabric and the golden's token floor (when the golden has one or the run covers it all; otherwise the
analysis computes it from the golden).

Two fabrics on the 8x4 mesh: ``SP8xTP4-torus-xy`` is production; ``SP8xTP4-fabric2d`` runs on a Galaxy whose wrap
links are not cabled. Compare runs only on the same fabric (``run.json`` records it).

Environment:
    KDA_DEV_RUN_DIR          run directory to create (required; the test skips without it)
    KDA_LONG_CONTEXT_GOLDEN  the FP64 golden directory
    KIMI_K3_CKPT             checkpoint with layers.0 and the embedding
    KDA_TRACE_DIR            the vLLM trace whose token ids the golden used
    KDA_DEV_TOKENS           tokens to run (default: the whole golden; a multiple of 5,120)
    KDA_DEV_PROGRAM_CONFIG   JSON overrides of the program config, e.g.
                             {"recurrence.scan_math_fidelity": "HiFi3", "input_projection_math_fidelity": "HiFi4"}
"""

from __future__ import annotations

import dataclasses
import json
import os
import socket
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.common.prefill.runners.runner_utils import load_trace_token_ids, resolve_trace_dir
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import kimi_k3_kda_config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import (
    fabric2d_device_params,
    torus_xy_device_params,
    tp_axis_is_wrapped,
)
from models.demos.deepseek_v3_d_p.tests.kda.checkpoint_utils import kda_state_dict_sha256, load_kda_layer_state_dict
from models.demos.deepseek_v3_d_p.tests.kda.dev.layer0_run_analysis import (
    RunWriter,
    error_sums,
    log,
    sums_token_floor,
    tiers,
)
from models.demos.deepseek_v3_d_p.tests.kda.first_layer_input import FIRST_LAYER_INPUT, FirstLayerInput
from models.demos.deepseek_v3_d_p.tests.kda.long_context_golden import (
    MANIFEST,
    load_window,
    state_after,
    token_ids_sha256,
)
from models.demos.deepseek_v3_d_p.tests.kda.perf.test_layer_perf import _L1_SMALL_SIZE, _trace_wall_samples_ms
from models.demos.deepseek_v3_d_p.tests.kda.utils import (
    KimiK3TestCase,
    make_kimi_k3_device_case,
    mla_row_permutation,
    reconstruct_sp_tp_tensor,
    reconstruct_state_at_sp_rank,
    to_sp_input,
)
from models.demos.deepseek_v3_d_p.tt.kda.config import kimi_k3_program_config
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS
from models.demos.deepseek_v3_d_p.utils.smbus_telemetry import get_tdp_limit_max, is_high_power
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import make_actual_start

pytestmark = [run_for_blackhole(), pytest.mark.timeout(3600)]

_CALL = PREFILL_CHUNK_TOKENS
_PERF_REPETITIONS = 10


def _override(config, overrides: dict[str, str]):
    """Apply ``{"a.b": value}`` overrides to a frozen dataclass tree; values name ttnn enum members."""
    for key, value in overrides.items():
        *path, field = key.split(".")
        parents = [config]
        for name in path:
            parents.append(getattr(parents[-1], name))
        current = getattr(parents[-1], field)
        if isinstance(current, ttnn.MathFidelity):
            value = getattr(ttnn.MathFidelity, value)
        elif isinstance(current, ttnn.DataType):
            value = getattr(ttnn, value)
        node = dataclasses.replace(parents[-1], **{field: value})
        for parent, name in zip(reversed(parents[:-1]), reversed(path)):
            node = dataclasses.replace(parent, **{name: node})
        config = node
    return config


@pytest.mark.parametrize(
    "mesh_device,tensor_parallel_axis,device_params",
    [
        pytest.param(
            (8, 4),
            1,
            torus_xy_device_params(l1_small_size=_L1_SMALL_SIZE),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="SP8xTP4-torus-xy",
        ),
        pytest.param(
            (8, 4),
            1,
            fabric2d_device_params(l1_small_size=_L1_SMALL_SIZE),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="SP8xTP4-fabric2d",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_kimi_k3_layer0_dev_run(mesh_device: ttnn.MeshDevice, tensor_parallel_axis: int, device_params: dict) -> None:
    env = {
        name: os.environ.get(name)
        for name in ("KDA_DEV_RUN_DIR", "KDA_LONG_CONTEXT_GOLDEN", "KIMI_K3_CKPT", "KDA_TRACE_DIR")
    }
    if not all(env.values()):
        pytest.skip("set KDA_DEV_RUN_DIR, KDA_LONG_CONTEXT_GOLDEN, KIMI_K3_CKPT and KDA_TRACE_DIR")
    run_dir = Path(env["KDA_DEV_RUN_DIR"])
    golden_dir = Path(env["KDA_LONG_CONTEXT_GOLDEN"])
    checkpoint = Path(env["KIMI_K3_CKPT"])
    manifest = json.loads((golden_dir / MANIFEST).read_text(encoding="utf-8"))
    available = manifest["segments"][-1]["end"]
    tokens = int(os.environ.get("KDA_DEV_TOKENS", available))
    assert tokens % _CALL == 0 and _CALL % manifest["segment"] == 0, f"{tokens} tokens in {_CALL}-token calls"
    assert tokens <= available, f"{tokens} tokens, but the golden in {golden_dir} has {available}"
    overrides = json.loads(os.environ.get("KDA_DEV_PROGRAM_CONFIG", "{}"))
    run_dir.mkdir(parents=True, exist_ok=False)

    # Identity: the device must see exactly the golden's weights, tokens and input.
    config = kimi_k3_kda_config()
    weights = load_kda_layer_state_dict(checkpoint, 0, config)
    identity = manifest["identity"]
    assert kda_state_dict_sha256(weights) == identity["weights_sha256"], "the golden used other weights"
    assert identity["input"] == FIRST_LAYER_INPUT, f"the golden's input is {identity['input']!r}"
    token_ids = load_trace_token_ids(resolve_trace_dir(env["KDA_TRACE_DIR"]))[: manifest["tokens"]]
    assert token_ids_sha256(token_ids) == manifest["token_ids_sha256"], "the golden used other tokens"
    build = FirstLayerInput(checkpoint)
    if "input_weights_sha256" in identity:
        assert build.weights_sha256() == identity["input_weights_sha256"], "the golden's input used other embeddings"

    sp_axis = 1 - tensor_parallel_axis
    sp_size = tuple(mesh_device.shape)[sp_axis]
    program_config = _override(
        kimi_k3_program_config(
            active_seq_len_local=_CALL // sp_size, tp_ccl_topology=per_axis_topology()[tensor_parallel_axis]
        ),
        overrides,
    )
    permutation = mla_row_permutation(0, sp_size, _CALL // sp_size)
    first = KimiK3TestCase(config, weights, build(token_ids[:_CALL]), identity["weights_sha256"])
    layer, first_hidden_tt = make_kimi_k3_device_case(
        mesh_device,
        first,
        tensor_parallel_axis=tensor_parallel_axis,
        program_config=program_config,
        cache_weights=False,
    )
    meta = {
        "kind": "device-run",
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=Path(__file__).parent
        ).stdout.strip(),
        "host": socket.gethostname(),
        "tdp_limit_max_w": get_tdp_limit_max(),
        "high_power": is_high_power(),
        "mesh": list(mesh_device.shape),
        "layout": f"SP{sp_size}xTP{tuple(mesh_device.shape)[tensor_parallel_axis]}",
        "fabric_config": ttnn.get_fabric_config().name,
        "tp_axis_wrapped": tp_axis_is_wrapped(mesh_device),
        "call_tokens": _CALL,
        "tokens": tokens,
        "program_config_overrides": overrides,
        "program_config": repr(program_config),
        "golden": str(golden_dir),
        "golden_identity": identity,
        "token_floor": manifest.get("token_floor"),
        "trace": env["KDA_TRACE_DIR"],
    }
    (run_dir / "run.json").write_text(json.dumps(meta, indent=1))

    # Perf first, on a fresh layer: median traced-replay wall of one 5,120-token call at start 0.
    samples_ms, _ = _trace_wall_samples_ms(
        mesh_device, layer, first_hidden_tt, _PERF_REPETITIONS, actual_start=0, actual_end=_CALL
    )
    median_ms = statistics.median(samples_ms)
    meta["perf"] = {
        "method": f"median of {len(samples_ms)} samples x {_PERF_REPETITIONS} traced replays of one "
        f"{_CALL}-token call at start 0",
        "samples_ms": samples_ms,
        "latency_ms": median_ms,
        "tokens_per_s": _CALL / (median_ms / 1e3),
    }
    (run_dir / "run.json").write_text(json.dumps(meta, indent=1))
    log(f"perf: {median_ms:.3f} ms per {_CALL}-token call, {meta['perf']['tokens_per_s']:,.0f} tokens/s")
    ttnn.deallocate(first_hidden_tt)

    calls = tokens // _CALL
    writer = RunWriter(run_dir, tokens, calls, config.num_heads)
    actual_start_tt = make_actual_start(layer.device, 0)
    actual_end_tt = make_actual_start(layer.device, _CALL)
    state = layer.allocate_state(batch_size=1)

    def golden_call(index: int):
        start = index * _CALL
        rows, _ = load_window(golden_dir, start, start + _CALL)
        return rows, state_after(golden_dir, start + _CALL).recurrent[0]

    began = time.perf_counter()
    with ThreadPoolExecutor(max_workers=1) as prefetch:
        upcoming = prefetch.submit(golden_call, 0)
        for index in range(calls):
            step = time.perf_counter()
            start = index * _CALL
            hidden_tt = to_sp_input(build(token_ids[start : start + _CALL])[:, permutation, :], mesh_device, sp_axis)
            with ttnn.manage_config("throw_exception_on_fallback", True):
                output_tt, state = layer.forward(hidden_tt, state, actual_start_tt, actual_end_tt)
            rotated = reconstruct_sp_tp_tensor(
                output_tt, mesh_device, sp_axis, tensor_parallel_axis, tp_dim=2, sp_dim=1
            )
            ttnn.deallocate(hidden_tt)
            ttnn.deallocate(output_tt)
            output = torch.empty_like(rotated)
            output[:, permutation, :] = rotated
            recurrent = reconstruct_state_at_sp_rank(
                state.recurrent, mesh_device, sp_axis, tensor_parallel_axis, sp_size - 1
            )[0]
            golden_rows, golden_state = upcoming.result()
            if index + 1 < calls:
                upcoming = prefetch.submit(golden_call, index + 1)
            token_sums = error_sums(golden_rows, output[0])
            state_sums = error_sums(golden_state, recurrent, dim=(-2, -1))
            writer.write_call(start, start + _CALL, token_sums, state_sums)
            call = tiers(token_sums.numpy(), state_sums.numpy()[None], [start + _CALL], floor=0.0)
            worst = call["tier1"]["state_head_final"]
            elapsed = time.perf_counter() - began
            log(
                f"call {index + 1}/{calls} [{start},{start + _CALL}) {time.perf_counter() - step:.2f} s, "
                f"ETA {elapsed / (index + 1) * (calls - index - 1):.0f} s | out {call['tier0']['output_nrmse']:.4%} "
                f"gain {call['tier2']['output_gain']:.4f} | state {call['tier0']['state_final_nrmse']:.3%} | "
                f"worst head h{worst['worst_at']} {worst['worst']:.3%}"
            )
    if meta["token_floor"] is None and tokens == available:  # the run's golden sums cover the whole golden
        meta["token_floor"] = sums_token_floor(writer.token_sums)
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    meta["run_seconds"] = time.perf_counter() - began
    (run_dir / "run.json").write_text(json.dumps(meta, indent=1))
    (run_dir / "DONE").write_text(f"{calls} calls in {meta['run_seconds']:.1f} s\n")
    log(f"done: {calls} calls in {meta['run_seconds']:.1f} s -> {run_dir}")
