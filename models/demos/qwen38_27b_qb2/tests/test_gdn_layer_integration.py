# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real-weight GDN layer integration, FP32 state reference, and native timing control."""

import gc
import os
import statistics
import time
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig

import ttnn
from models.demos.qwen38_27b_qb2.demo.galaxy_serving import model_source_hashes
from models.demos.qwen38_27b_qb2.tests.test_gdn_model_adapter import tensor_digest
from models.demos.qwen38_27b_qb2.tests.test_gdn_step_candidate import accuracy, reference
from models.demos.qwen38_27b_qb2.tests.test_long_context_attention import save
from models.demos.qwen38_27b_qb2.tt.decoder_tp import Qwen38TPDecoder
from models.demos.qwen38_27b_qb2.tt.gdn_step import op
from models.demos.qwen38_27b_qb2.tt.generator import configure_fabric
from models.demos.qwen38_27b_qb2.tt.model import Checkpoint, checkpoint_path
from models.demos.qwen38_27b_qb2.tt.precision import decoder_policy, load_precision


def host_ranks(tensor):
    values = [ttnn.to_torch(rank).float() for rank in ttnn.get_device_tensors(tensor)]
    assert len(values) == 4
    return values


def prepare_reference(inputs, batch):
    prepared = []
    for q, k, v, g, beta in zip(*(host_ranks(tensor) for tensor in inputs)):
        vectors = []
        for raw, scale in ((q, 128**-0.5), (k, 1.0)):
            raw = raw[:, 0].reshape(batch, 4, 128)
            norm = raw * torch.rsqrt(raw.square().sum(-1, keepdim=True) + 1e-6) * scale
            vectors.append(norm.repeat_interleave(3, dim=1).reshape(batch * 12, 128))
        gates = torch.zeros(batch * 12, 8)
        gates[:, 0] = g[:, 0].flatten().exp()
        gates[:, 1] = beta[:, 0].flatten()
        prepared.append((*vectors, v[:, 0].reshape(batch * 12, 128), gates))
    return prepared


def capture(mesh, call):
    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        output = call()
    except BaseException:
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        ttnn.release_trace(mesh, trace)
        raise
    ttnn.end_trace_capture(mesh, trace, cq_id=0)
    return trace, output


def timing(mesh, trace):
    samples = []
    for _ in range(5):
        ttnn.synchronize_device(mesh)
        tick = time.perf_counter()
        for _ in range(30):
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        samples.append((time.perf_counter() - tick) * 1e6 / 30)
    return dict(samples_us=samples, median_us=statistics.median(samples))


def run_case(layer, mesh, batch):
    state = layer.allocate_state(batch_size=batch)
    rng = torch.Generator().manual_seed(20261007 + batch)
    shape = [1, 1, batch, 5120] if batch > 1 else [1, 1, 5120]
    x = ttnn.from_torch(
        torch.randn(shape, generator=rng).bfloat16(),
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    layer.policy["decode_recurrence"] = "single_step"
    original = layer._delta_recurrence
    original_step = op.step
    raw_inputs = []
    kernel_inputs = []
    kernel_normalizes = False
    kernel_qk_repeat = 1

    def observe(q, k, v, g, beta, current, *, decode=False):
        raw_inputs[:] = [q, k, v, g, beta]
        return original(q, k, v, g, beta, current, decode=decode)

    def observe_step(q, k, v, gates, current, output, **kwargs):
        nonlocal kernel_normalizes, kernel_qk_repeat
        kernel_inputs[:] = [q, k, v, gates]
        kernel_normalizes = kwargs.get("normalize_qk", False)
        kernel_qk_repeat = kwargs.get("qk_head_repeat", 1)
        return original_step(q, k, v, gates, current, output, **kwargs)

    layer._delta_recurrence = observe
    op.step = observe_step
    try:
        # The real convolution becomes stationary after four identical inputs.
        # Readback and CPU reference preparation occur outside trace capture.
        for _ in range(5):
            layer._delta(x, state, decode=True)
        prepared = prepare_reference(raw_inputs, batch)
        input_hashes = [tensor_digest(value) for tensor in raw_inputs for value in host_ranks(tensor)]
        device_prepared = list(zip(*(host_ranks(tensor) for tensor in kernel_inputs)))
        if kernel_qk_repeat != 1:
            device_prepared = [
                (q.repeat_interleave(kernel_qk_repeat, dim=0), k.repeat_interleave(kernel_qk_repeat, dim=0), v, gates)
                for q, k, v, gates in device_prepared
            ]
        if kernel_normalizes:
            device_prepared = [
                (
                    q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * 128**-0.5,
                    k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6),
                    v,
                    gates,
                )
                for q, k, v, gates in device_prepared
            ]
    finally:
        layer._delta_recurrence = original
        op.step = original_step
    reset_state = ttnn.clone(state.recurrent)
    reset_conv = ttnn.clone(state.conv)
    expected = [value.reshape(batch * 12, 128, 128) for value in host_ranks(reset_state)]
    expected_kernel = [value.clone() for value in expected]
    preparation_checks = [
        {name: accuracy(actual, expected) for name, actual, expected in zip(("q", "k", "v", "gates"), actuals, golds)}
        for actuals, golds in zip(device_prepared, prepared)
    ]
    scratch_address = layer.gdn_decode_workspace.output(batch).buffer_address()
    state_address = state.recurrent.buffer_address()
    trace, output = capture(mesh, lambda: layer._delta(x, state, decode=True))
    try:
        ttnn.copy(reset_state, state.recurrent)
        ttnn.copy(reset_conv, state.conv)
        expected_output = []
        expected_kernel_output = []
        for _ in range(64):
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            updated = [reference(s, *values) for s, values in zip(expected, prepared)]
            expected, expected_output = map(list, zip(*updated))
            updated_kernel = [reference(s, *values) for s, values in zip(expected_kernel, device_prepared)]
            expected_kernel, expected_kernel_output = map(list, zip(*updated_kernel))
        checks = []
        kernel_checks = []
        actual_states = host_ranks(state.recurrent)
        actual_outputs = host_ranks(layer.gdn_decode_workspace.output(batch))
        projected_outputs = host_ranks(output)
        state_hashes = [tensor_digest(value) for value in actual_states]
        output_hashes = [tensor_digest(value) for value in actual_outputs]
        projected_hashes = [tensor_digest(value) for value in projected_outputs]
        for actual_state, actual_output, gold_state, gold_output in zip(
            actual_states,
            actual_outputs,
            expected,
            expected_output,
        ):
            checks.append(
                dict(
                    state=accuracy(actual_state.reshape(batch * 12, 128, 128), gold_state),
                    output=accuracy(actual_output, gold_output),
                )
            )
        for actual_state, actual_output, gold_state, gold_output in zip(
            actual_states, actual_outputs, expected_kernel, expected_kernel_output
        ):
            kernel_checks.append(
                dict(
                    state=accuracy(actual_state.reshape(batch * 12, 128, 128), gold_state),
                    output=accuracy(actual_output, gold_output),
                )
            )
        passed = all(row[part]["passed"] for row in checks for part in ("state", "output"))
        if not passed:
            # Freeze only the worst head of each rank for bounded offline
            # diagnosis. Keep both references: actual kernel operands isolate
            # recurrence arithmetic, while the raw-input reference remains
            # the unchanged model-integration acceptance gate.
            failures = []
            for rank, (actual, gold) in enumerate(zip(actual_outputs, expected_output)):
                relative = (actual - gold).square().mean(-1).sqrt() / gold.square().mean(-1).sqrt().clamp_min(1e-12)
                head = int(relative.argmax())
                failures.append(
                    dict(
                        rank=rank,
                        head=head,
                        relative_rms=float(relative[head]),
                        prepared=[v[head].clone() for v in device_prepared[rank]],
                        reference_prepared=[v[head].clone() for v in prepared[rank]],
                        actual_state=actual_states[rank].reshape(batch * 12, 128, 128)[head].clone(),
                        reference_state=expected[rank][head].clone(),
                        kernel_reference_state=expected_kernel[rank][head].clone(),
                        actual_output=actual[head].clone(),
                        reference_output=gold[head].clone(),
                        kernel_reference_output=expected_kernel_output[rank][head].clone(),
                    )
                )
            torch.save(failures, Path(os.environ["QWEN_GDN_LAYER_RECEIPT"]).with_name(f"batch-{batch}-worst-heads.pt"))
        assert all(torch.isfinite(value).all() for value in projected_outputs), "Nonfinite projected layer output"
        assert state.recurrent.buffer_address() == state_address
        assert layer.gdn_decode_workspace.output(batch).buffer_address() == scratch_address
        candidate = timing(mesh, trace)
    finally:
        ttnn.release_trace(mesh, trace)
    # Restore identical history/state before the native timing control. Its
    # drift has a separate diagnostic; it is not the candidate's golden reference.
    layer.policy["decode_recurrence"] = "native"
    ttnn.copy(reset_state, state.recurrent)
    ttnn.copy(reset_conv, state.conv)
    for _ in range(2):
        layer._delta(x, state, decode=True)
    trace, native_output = capture(mesh, lambda: layer._delta(x, state, decode=True))
    try:
        native = timing(mesh, trace)
        assert all(torch.isfinite(value).all() for value in host_ranks(native_output))
    finally:
        ttnn.release_trace(mesh, trace)
        layer.policy["decode_recurrence"] = "single_step"
    return dict(
        batch=batch,
        passed=passed,
        input_sha256=input_hashes,
        state_sha256_per_rank=state_hashes,
        output_sha256_per_rank=output_hashes,
        projected_output_sha256_per_rank=projected_hashes,
        checks=checks,
        preparation_checks=preparation_checks,
        recurrence_only_checks=kernel_checks,
        candidate=candidate,
        native=native,
        native_over_candidate=native["median_us"] / candidate["median_us"],
    )


@pytest.mark.skipif(os.getenv("QWEN_GDN_LAYER_INTEGRATION") != "1", reason="explicit allocated-Galaxy test")
def test_gdn_layer_integration():
    path = Path(os.environ["QWEN_GDN_LAYER_RECEIPT"])
    assert not path.exists(), "Use a new result directory"
    torch.set_num_threads(8)
    source = Path(__file__).resolve().parents[1]
    checkpoint = checkpoint_path()
    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True).text_config
    precision = load_precision(source / "config/precision_single_step_gdn.json")
    report = dict(
        state="opening",
        passed=False,
        checkpoint=str(checkpoint),
        precision=precision,
        source_sha256=model_source_hashes(source),
        cases=[],
        scope="Layer 0 GDN projection, convolution, gates, recurrence, output norm and TP output projection; MLP excluded",
        reference="FP32 recurrence on actual per-rank convolution outputs and gates, 64 traced updates with stationary input",
        measurement="Five warm samples of 30 trace replays; native control is timing only",
    )
    save(path, report)
    configure_fabric(topology=ttnn.Topology.Linear)
    parent = ttnn.open_mesh_device(ttnn.MeshShape(8, 4), trace_region_size=200000000)
    mesh = None
    try:
        mesh = parent.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
        report["device_ids"] = list(mesh.get_device_ids())
        layer = Qwen38TPDecoder.from_state_dict(
            Checkpoint(checkpoint).layer(0),
            hf_config=config,
            layer_idx=0,
            mesh_device=mesh,
            policy={**decoder_policy(precision, 0), "ring": False, "compact_decode_residual": True},
        )
        assert layer.kind == "linear_attention"
        batches = [int(value) for value in os.getenv("QWEN_GDN_LAYER_BATCHES", "1,8,16,32,64").split(",")]
        report["requested_batches"] = batches
        for batch in batches:
            report.update(state="running", active_batch=batch)
            save(path, report)
            report["cases"].append(run_case(layer, mesh, batch))
            save(path, report)
            gc.collect()
        report.update(state="completed", passed=all(case["passed"] for case in report["cases"]))
        assert report["passed"], "At least one batch failed the original FP32 integration accuracy gate; see layer.json"
    except BaseException as error:
        report.update(state="failed", error=dict(type=type(error).__name__, message=str(error)[:4000]))
        raise
    finally:
        try:
            if mesh is not None:
                ttnn.close_mesh_device(mesh)
        finally:
            try:
                ttnn.close_mesh_device(parent)
            finally:
                save(path, report)
