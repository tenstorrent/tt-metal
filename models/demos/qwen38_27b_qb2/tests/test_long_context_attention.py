# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in TP4 paged-attention sweep, with full-context CPU numerical checks.

Synthetic Q/K/V uses Qwen's local TP4 dimensions and deployed dtypes. Every
candidate uses the same aligned, randomly mapped cache. No model policy changes
are made. Traced wall cost includes dispatch and is not pure device-kernel time.
"""

import gc
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.qwen38_27b_qb2.tests.attention_tuning import (
    CASES,
    CHUNKS,
    accuracy,
    geometry,
    reference,
    select_candidate,
)
from models.demos.qwen38_27b_qb2.tt.generator import configure_fabric


def save(path, report):
    report["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


def measure(mesh, invoke):
    for _ in range(2):
        warm = invoke()
        ttnn.synchronize_device(mesh)
        del warm
    trace = None
    capture_open = False
    try:
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        capture_open = True
        output = invoke()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        capture_open = False
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
        samples = []
        for _ in range(5):
            ttnn.synchronize_device(mesh)
            tick = time.perf_counter()
            for _ in range(100):
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            samples.append((time.perf_counter() - tick) * 1e6 / 100)
        # Readback is outside all timing windows. Each physical rank must pass.
        actuals = [ttnn.to_torch(part).float()[:, :, :6, :] for part in ttnn.get_device_tensors(output)]
        assert len(actuals) == 4
        return samples, actuals
    finally:
        if trace is not None:
            try:
                if capture_open:
                    ttnn.end_trace_capture(mesh, trace, cq_id=0)
            finally:
                ttnn.release_trace(mesh, trace)


def run_case(
    mesh,
    case,
    report,
    path,
    *,
    chunks=CHUNKS,
    precision="native",
    require_native_accuracy=True,
    max_cores_per_head_batch=16,
    core_placement=None,
):
    if precision not in (
        "native",
        "hifi4_fp32",
        "hifi4_fp32_accurate_exp",
        "hifi4_fp32_full_tile",
        "hifi4_fp32_full_tile_accurate_exp",
        "model_accurate",
    ):
        raise ValueError(f"Unsupported precision diagnostic {precision}")
    if max_cores_per_head_batch not in (16, 32):
        raise ValueError("Diagnostic core budget must be 16 or 32")
    if core_placement is not None and precision != "hifi4_fp32_full_tile_accurate_exp":
        raise ValueError("Placement experiments require unchanged full-tile accurate attention precision")
    batch, capacity = case["batch"], case["aligned_capacity"]
    case["seed"] = 20261006 + case["input_tokens"] + batch
    rng = torch.Generator().manual_seed(case["seed"])
    pages = capacity // 32
    table = torch.randperm(batch * pages, generator=rng, dtype=torch.int32).reshape(batch, pages)
    query = torch.randn((1, batch, 6, 256), generator=rng).bfloat16()
    case.update(state="uploading", page_table_sha256=hashlib.sha256(table.numpy().tobytes()).hexdigest(), candidates=[])
    save(path, report)

    def upload(tensor, dtype, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            tensor,
            dtype=dtype,
            layout=layout,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    # Six BF16 Q heads select a half tile. The pinned shared exp helper forces
    # approximate exp for that partial-face path, even with exp_approx_mode=False.
    # Preserve the same six live Q rows and RNG stream while forcing a full tile.
    full_tile = "full_tile" in precision
    device_query = (
        torch.cat([query, torch.zeros((1, batch, 26, 256), dtype=query.dtype)], dim=2) if full_tile else query
    )
    case["device_query_heads"] = device_query.shape[2]
    case["live_query_heads"] = 6
    q = upload(device_query, ttnn.bfloat16)
    tt_table = upload(table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    positions = upload(torch.tensor(case["positions"], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    key_host = torch.randn((batch * pages, 1, 32, 256), generator=rng).bfloat16()
    key = upload(key_host, ttnn.bfloat8_b)
    del key_host
    value_host = torch.randn((batch * pages, 1, 32, 256), generator=rng).bfloat16()
    for user, position in enumerate(case["positions"]):
        for virtual_page in range((position + 1) // 32, pages):
            begin = max(0, position + 1 - virtual_page * 32)
            value_host[int(table[user, virtual_page]), 0, begin:, :] = 32
    value = upload(value_host, ttnn.bfloat8_b)
    del value_host
    case["state"] = "cpu_reference"
    save(path, report)
    # Replication gives every rank identical inputs. Use the quantized device
    # data for the reference, avoiding a hidden BF16-to-BFP8 reference mismatch.
    key_quantized = ttnn.to_torch(ttnn.get_device_tensors(key)[0])
    value_quantized = ttnn.to_torch(ttnn.get_device_tensors(value)[0])
    expected = reference(query, key_quantized, value_quantized, table, case["positions"])
    del key_quantized, value_quantized
    grid = mesh.compute_with_storage_grid_size()
    case["worker_grid"] = [grid.x, grid.y]
    placement_options = {}
    output_memory = None
    if core_placement is not None:
        from models.demos.qwen38_27b_qb2.tests.attention_placement import placement

        selected = placement(core_placement, batch, (grid.x, grid.y))
        case["placement"] = selected
        grid = ttnn.CoreCoord(*selected["grid"])
        max_cores_per_head_batch = selected["max_cores_per_head_batch"]
        if selected["explicit_subgrid"]:

            def core_set(points):
                return ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in points]
                )

            points = selected["logical_cores"]
            placement_options["sub_core_grids"] = core_set(points)
            # The native factory requires sharded Q or output on explicit grids.
            # Keep Q identical and give each reducer its own 32x256 output shard.
            output_memory = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(core_set(points[:batch]), [32, 256], ttnn.ShardOrientation.ROW_MAJOR),
            )
    case.update(state="measuring", grid=[grid.x, grid.y])
    save(path, report)

    def operation(chunk):
        if precision == "model_accurate":
            from models.demos.qwen38_27b_qb2.tt.decode_attention import paged_decode

            return paged_decode(
                q,
                key,
                value,
                positions=positions,
                page_table=tt_table,
                policy={"decode_attention": "accurate_full_tile", "sdpa_short_grid": [8, 2]},
            )
        config = {}
        if precision != "native":
            # This pinned runtime exposes the shared BH/WH config under the
            # Wormhole name; the newer Blackhole alias is not exported by ttnn.
            config["compute_kernel_config"] = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )
        result = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            key,
            value,
            cur_pos_tensor=positions,
            page_table_tensor=tt_table,
            scale=256**-0.5,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=[grid.x, grid.y],
                q_chunk_size=32,
                k_chunk_size=chunk,
                exp_approx_mode=not precision.endswith("accurate_exp"),
                max_cores_per_head_batch=max_cores_per_head_batch,
                **placement_options,
            ),
            **({"memory_config": output_memory} if output_memory is not None else {}),
            **config,
        )
        # Include output conversion in every candidate's timed call, preserving
        # the current decoder's DRAM output boundary rather than hiding its cost.
        return ttnn.to_memory_config(result, ttnn.DRAM_MEMORY_CONFIG) if output_memory is not None else result

    # Measure the native-selected chunk first, then revisit it to expose drift.
    case["precision_mode"] = precision
    case["max_cores_per_head_batch"] = max_cores_per_head_batch
    for chunk in [case["native_chunk"], *[k for k in chunks if k != case["native_chunk"]]]:
        case["active_chunk"] = chunk
        save(path, report)
        samples, actuals = measure(mesh, lambda: operation(chunk))
        checks = [accuracy(actual, expected) for actual in actuals]
        candidate = dict(
            chunk=chunk,
            traced_call_us=samples,
            median_traced_call_us=statistics.median(samples),
            accuracy_per_rank=checks,
            accuracy_passed=all(check["passed"] for check in checks),
        )
        case["candidates"].append(candidate)
        save(path, report)
        print("ATTENTION_CANDIDATE", case["input_tokens"], batch, json.dumps(candidate), flush=True)
        if chunk == case["native_chunk"]:
            if require_native_accuracy:
                assert candidate["accuracy_passed"], "Native chunk failed full-context numerical reference"
    samples, actuals = measure(mesh, lambda: operation(case["native_chunk"]))
    repeat_checks = [accuracy(actual, expected) for actual in actuals]
    repeat_passed = all(check["passed"] for check in repeat_checks)
    baseline_passed = case["candidates"][0]["accuracy_passed"] and repeat_passed
    if require_native_accuracy:
        assert repeat_passed, "Repeated baseline correctness changed"
    case.update(
        state="completed" if baseline_passed else "numerical_failure",
        passed=baseline_passed,
        baseline_repeat_us=samples,
        baseline_repeat_accuracy_per_rank=repeat_checks,
        passing_chunks=[candidate["chunk"] for candidate in case["candidates"] if candidate["accuracy_passed"]],
        selection=select_candidate(case["candidates"], case["native_chunk"], samples) if baseline_passed else None,
    )
    case.pop("active_chunk", None)
    save(path, report)


@pytest.mark.skipif(os.getenv("QWEN_LONG_CONTEXT_ATTENTION") != "1", reason="explicit allocated-Galaxy tuning")
def test_long_context_attention():
    path = Path(os.environ["QWEN_ATTENTION_RECEIPT"])
    assert not path.exists(), "Use a new result directory"
    torch.set_num_threads(8)
    report = dict(
        state="opening",
        passed=False,
        promoted_to_model=False,
        scope="Synthetic TP4 attention kernel experiment, not full-model performance or quality",
        precision=dict(query="bfloat16", kv="bfloat8_b", reference="float32 on quantized device inputs"),
        local_heads=dict(query=6, kv=1, head_dim=256),
        measurement="Five warm samples of 100 trace replays; wall cost includes dispatch, excludes CPU/readback",
        cache_contract="Full causal attention, shuffled physical pages, mapped aligned tail with sentinel future values",
        torch_version=torch.__version__,
        source_sha256={
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("test_long_context_attention.py", "attention_tuning.py")
        },
        cases=[geometry(length, batch) for length, batch in CASES],
    )
    save(path, report)
    configure_fabric(topology=ttnn.Topology.Linear)
    parent = ttnn.open_mesh_device(ttnn.MeshShape(8, 4), trace_region_size=200000000)
    try:
        mesh = parent.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
        report["device_ids"] = list(mesh.get_device_ids())
        for case in report["cases"]:
            run_case(mesh, case, report, path)
            ttnn.synchronize_device(mesh)
            gc.collect()
        report.update(state="completed", passed=True)
    except BaseException as error:
        report.update(state="failed", error=dict(type=type(error).__name__, message=str(error)[:2000]))
        raise
    finally:
        try:
            save(path, report)
        finally:
            ttnn.close_mesh_device(parent)
