# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The current GDN implementation on the GDN-on-KDA cases, gated against the same FP32 reference (g1b.5.10).

The current layer (qwen36 ``TPGatedDeltaNet`` -> ``ttnn.transformer.chunk_gated_delta_rule``, ``Qwen36ModelArgs``
defaults, op dispatch by its cost model, carried-state chunk-outer prefill) runs the registered SP1 1x4 cases of
``tests/gdn/cases.py`` with their weights, inputs and chained CPU references (``reference/gdn``), so its accuracy is
measured on exactly the cells test_gdn_accuracy.py measures for ttGDN, with the same metrics and §6.3 gates
(``device_utils.chunk_gate_rows``). Per case: the chunks run eagerly twice (bit-exact repeat, R10); a schedule without
a ragged tail then runs under one trace captured at chunk 0 and every replay must equal the eager pass bit for bit
(R11). A ragged last chunk uses the layer's masked ``valid_len`` path (eager only: it uploads a host mask).

Layer I/O differs from ttGDN: the input is hidden-sharded across the TP ranks (the fused all-gather + matmul gathers
it) and the output is TP-sharded on hidden; the reference compares the gathered tensors.
Mesh: the LoudBox 2x4 fixture with the layer on its 1x4 row-0 submesh (a standalone 1x4 mesh fails fabric init).
"""

from __future__ import annotations

import json
import os

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.tests.gdn_baseline.cases import model_dir, per_device_conv_columns
from models.demos.blackhole.qwen36.tests.test_factory import shard_to_device, tp_composer
from models.demos.blackhole.qwen36.tt.gdn.tp import TPGatedDeltaNet, load_gdn_weights_tp
from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE, Qwen36ModelArgs
from models.demos.deepseek_v3_d_p.reference.gdn import GDNReferenceState
from models.demos.deepseek_v3_d_p.tests.gdn.cases import (
    CURRENT_IMPLEMENTATION_LAYOUTS,
    CURRENT_IMPLEMENTATION_MODELS,
    GDNTestCase,
    build_gdn_case,
    registered_gdn_case,
)
from models.demos.deepseek_v3_d_p.tests.gdn.device_utils import PCC_THRESHOLD, chunk_gate_rows
from models.demos.deepseek_v3_d_p.tests.gdn.reference_cache import cpu_references
from models.tt_transformers.tt.ccl import TT_CCL
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import accuracy_metrics

pytestmark = [run_for_blackhole(), pytest.mark.timeout(1800)]

_CELLS = (("synthetic", "randn"), ("real", "text"))
_SCHEDULES = ("chained3", "ragged")
_CHECKPOINT_PREFIX = "linear_attn."


def _layer(mesh: ttnn.MeshDevice, case: GDNTestCase) -> tuple[Qwen36ModelArgs, TPGatedDeltaNet]:
    os.environ["HF_MODEL"] = str(model_dir(case.spec.model))
    args = Qwen36ModelArgs(mesh, max_batch_size=1, max_seq_len=case.spec.chunk_tokens * case.num_chunks)
    state_dict = {_CHECKPOINT_PREFIX + name: tensor for name, tensor in case.weights.load_state_dict().items()}
    layer = TPGatedDeltaNet(mesh, args, load_gdn_weights_tp(mesh, state_dict, args), TT_CCL(mesh))
    layer._stable_state = True
    layer.reset_state()
    return args, layer


def _snapshot(mesh: ttnn.MeshDevice, layer: TPGatedDeltaNet, output: ttnn.Tensor, valid: int) -> dict:
    hidden = layer.args.dim
    return {
        "output": ttnn.to_torch(output, mesh_composer=tp_composer(mesh)).reshape(-1, hidden)[:valid].clone(),
        "recurrent_sp0": ttnn.to_torch(layer.rec_state, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=1))[0].clone(),
        "convolution_sp0": ttnn.to_torch(layer.conv_carry, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1))[
            0
        ].clone(),
    }


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "l1_small_size": GDN_CONV1D_L1_SMALL_SIZE,
            "trace_region_size": 1 << 28,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [pytest.param((2, 4), id="1x4of2x4")], indirect=True)
@pytest.mark.parametrize(
    "model,layout,weights,inputs,schedule",
    [
        pytest.param(model, layout, weights, inputs, schedule, id=f"{model}-{layout}-{weights}-{inputs}-{schedule}")
        for model in CURRENT_IMPLEMENTATION_MODELS
        for layout in CURRENT_IMPLEMENTATION_LAYOUTS
        for weights, inputs in _CELLS
        for schedule in _SCHEDULES
    ],
)
@torch.no_grad()
def test_current_gdn_accuracy(
    mesh_device: ttnn.MeshDevice,
    device_params: dict,
    model: str,
    layout: str,
    weights: str,
    inputs: str,
    schedule: str,
) -> None:
    spec = registered_gdn_case(model, layout, schedule, weights, inputs)
    mesh = mesh_device.create_submesh(ttnn.MeshShape(*spec.mesh_shape), offset=ttnn.MeshCoordinate(0, 0))
    case = build_gdn_case(spec)
    references = cpu_references(case)
    args, layer = _layer(mesh, case)
    tokens = spec.chunk_tokens
    buffer = shard_to_device(mesh, case.chunk_hidden(0)[None], dim=-1)

    def load(chunk: int) -> None:
        source = shard_to_device(mesh, case.chunk_hidden(chunk)[None], dim=-1)
        ttnn.copy(source, buffer)
        ttnn.deallocate(source)

    def eager_pass() -> list[dict]:
        layer.reset_state_inplace()
        snapshots = []
        for chunk, valid in enumerate(spec.chunk_valid_tokens):
            load(chunk)
            output = layer.forward_prefill(
                buffer, chunk_size=args.gdn_chunk_size, valid_len=None if valid == tokens else valid
            )
            snapshots.append(_snapshot(mesh, layer, output, valid))
            ttnn.deallocate(output)
        return snapshots

    first = eager_pass()
    second = eager_pass()
    traced = None
    if spec.chunk_valid_tokens[-1] == tokens:
        layer.reset_state_inplace()
        load(0)
        ttnn.synchronize_device(mesh)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        output = layer.forward_prefill(buffer, chunk_size=args.gdn_chunk_size)
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        traced = []
        try:
            for chunk, valid in enumerate(spec.chunk_valid_tokens):
                load(chunk)
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                traced.append(_snapshot(mesh, layer, output, valid))
        finally:
            ttnn.release_trace(mesh, trace)
            ttnn.deallocate(output)
    ttnn.deallocate(buffer)

    tensor_parallel_size = mesh.get_num_devices()
    failures, rows = [], []
    for chunk, (chunk_snapshot, reference) in enumerate(zip(first, references, strict=True)):
        # The layer keeps its conv carry in per-rank [q_r | k_r | v_r] column order.
        state = GDNReferenceState(
            conv=per_device_conv_columns(reference.state.conv, case.config, tensor_parallel_size),
            recurrent=reference.state.recurrent,
        )
        chunk_rows, chunk_failures = chunk_gate_rows(chunk, chunk_snapshot, reference.output, state)
        rows.extend(chunk_rows)
        failures.extend(chunk_failures)
    sequence = accuracy_metrics(
        torch.cat([reference.output.bfloat16() for reference in references]),
        torch.cat([chunk_snapshot["output"] for chunk_snapshot in first]),
    )
    rows.append({"chunk": "all", "tensor": "output", **sequence})
    if sequence["pcc"] < PCC_THRESHOLD:
        failures.append(f"chained output PCC {sequence['pcc']:.6f} < {PCC_THRESHOLD}")

    def mismatches(a_pass: list[dict], b_pass: list[dict]) -> list[str]:
        return [
            f"chunk {chunk} {name}"
            for chunk, (a, b) in enumerate(zip(a_pass, b_pass, strict=True))
            for name in a
            if not torch.equal(a[name], b[name])
        ]

    repeat_mismatches = mismatches(first, second)
    trace_mismatches = mismatches(first, traced) if traced is not None else None
    if repeat_mismatches:
        failures.append(f"repeated eager pass is not bit-identical: {repeat_mismatches}")
    if trace_mismatches:
        failures.append(f"trace replay differs from eager execution: {trace_mismatches}")
    for row in rows:
        print(f"CURRENT_GDN_ACCURACY_ROW {spec.name} " + json.dumps(row, sort_keys=True))
    recurrent_rows = [row for row in rows if row["tensor"] == "recurrent"]
    output_rows = [row for row in rows if row["tensor"] == "output"]
    print(
        "CURRENT_GDN_ACCURACY="
        + json.dumps(
            {
                "case": spec.name,
                "implementation": "qwen36 TPGatedDeltaNet / ttnn.transformer.chunk_gated_delta_rule",
                "gdn_program_config": repr(args.gdn_program_config),
                "model": model,
                "layout": layout,
                "schedule": schedule,
                "chunk_valid_tokens": list(spec.chunk_valid_tokens),
                "weights": weights,
                "inputs": inputs,
                "min_pcc": min(row["pcc"] for row in rows),
                "min_output_pcc": min(row["pcc"] for row in output_rows),
                "min_recurrent_pcc": min(row["pcc"] for row in recurrent_rows),
                "min_convolution_pcc": min(row["pcc"] for row in rows if row["tensor"] == "convolution"),
                "max_output_rel_rmse": max(row["rel_rmse"] for row in output_rows),
                "max_output_rel_linf": max(row["rel_linf"] for row in output_rows),
                "max_recurrent_rel_rmse": max(row["rel_rmse"] for row in recurrent_rows),
                "output_norm_ratio_range": [
                    min(row["norm_ratio"] for row in rows if "norm_ratio" in row),
                    max(row["norm_ratio"] for row in rows if "norm_ratio" in row),
                ],
                "worst_head_state_rel_rmse": max(row["worst_head_rel_rmse"] for row in recurrent_rows),
                "repeat_bit_identical": not repeat_mismatches,
                "trace_equals_eager": None if trace_mismatches is None else not trace_mismatches,
                "passed": not failures,
            },
            sort_keys=True,
        )
    )
    assert not failures, "\n".join(failures)
