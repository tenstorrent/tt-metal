# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reproduce the prefill control error and isolate numerical configuration effects."""

import gc
import hashlib
import os
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.qwen38_27b_qb2.demo.galaxy_serving import model_source_hashes
from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import save
from models.demos.qwen38_27b_qb2.tests.attention_tuning import accuracy
from models.demos.qwen38_27b_qb2.tests.prefill_attention_diagnostic import (
    CASES,
    VARIANTS,
    configuration,
    selected_reference,
    validate_report,
)
from models.demos.qwen38_27b_qb2.tests.test_prefill_attention_batch import digest
from models.demos.qwen38_27b_qb2.tt.generator import configure_fabric
from models.demos.qwen38_27b_qb2.tt.prefill_attention import fill_and_attend


def run_case(mesh, batch, start, length, report, path):
    # Match the failed baseline's exact generator and allocation order.
    seed = 20261010 + batch + start + length
    rng = torch.Generator().manual_seed(seed)
    pages = (start + length + 31) // 32 + 1
    table = (
        torch.randperm((batch + 2) * pages, generator=rng, dtype=torch.int32)
        .reshape(batch + 2, pages)[1 : batch + 1]
        .contiguous()
    )
    row = dict(batch=batch, start_pos=start, chunk_tokens=length, seed=seed, arms=[])
    report["cases"].append(row)
    save(path, report)

    def upload(host, dtype, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            host,
            dtype=dtype,
            layout=layout,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    def hashes(tensor):
        ranks = ttnn.get_device_tensors(tensor)
        assert len(ranks) == 4
        return [digest(ttnn.to_torch(part)) for part in ranks]

    query = torch.randn(batch, 6, length, 256, generator=rng).bfloat16()
    q = upload(query, ttnn.bfloat16)
    tt_table = upload(table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    updates = [upload(torch.randn(batch, 1, length, 256, generator=rng).bfloat16(), ttnn.bfloat16) for _ in range(2)]
    caches, expected_caches = [], []
    for update in updates:
        cache = upload(torch.randn((batch + 2) * pages, 1, 32, 256, generator=rng).bfloat16(), ttnn.bfloat8_b)
        expected = ttnn.to_torch(ttnn.get_device_tensors(cache)[0]).clone()
        quantized = ttnn.typecast(update, ttnn.bfloat8_b)
        host_update = ttnn.to_torch(ttnn.get_device_tensors(quantized)[0])
        ttnn.deallocate(quantized)
        host_update = torch.nn.functional.pad(host_update, (0, 0, 0, (-length) % 32))
        for user in range(batch):
            for chunk_page in range((length + 31) // 32):
                physical = table[user, start // 32 + chunk_page]
                expected[physical] = host_update[user, :, chunk_page * 32 : (chunk_page + 1) * 32]
        caches.append(cache)
        expected_caches.append(expected)
    expected_hashes = [digest(host) for host in expected_caches]
    selected = sorted({0, length // 2, length - 1})
    expected_output, independent_output = selected_reference(query, expected_caches, table, start, selected)
    row.update(
        page_table_sha256=digest(table),
        query_sha256=digest(query),
        expected_cache_sha256=expected_hashes,
        reference_query_rows=selected,
        reference_output_sha256=digest(expected_output),
        independent_reference_sha256=digest(independent_output),
        independent_reference_checked=True,
    )
    query_hashes = hashes(q)
    row["query_sha256_per_rank"] = query_hashes
    del expected_caches, expected, host_update, query
    for name in VARIANTS:
        row["active_variant"] = name
        save(path, report)
        choice = configuration(name)
        chunk = 128 if start % 128 == 0 and length >= 128 else 32
        extra = {} if choice["exp_approx_mode"] is None else {"exp_approx_mode": choice["exp_approx_mode"]}
        program = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=mesh.compute_with_storage_grid_size(),
            q_chunk_size=chunk,
            k_chunk_size=chunk,
            **extra,
        )
        kernel = choice["compute_kernel"]
        compute = (
            None
            if kernel is None
            else ttnn.WormholeComputeKernelConfig(
                **dict(kernel, math_fidelity=getattr(ttnn.MathFidelity, kernel["math_fidelity"]))
            )
        )
        tick = time.perf_counter()
        output = fill_and_attend(
            q,
            *updates,
            *caches,
            tt_table,
            start,
            page_size=32,
            scale=256**-0.5,
            program_config=program,
            compute_kernel_config=compute,
        )
        ttnn.synchronize_device(mesh)
        elapsed = time.perf_counter() - tick
        output_hashes, checks, by_row = [], [], []
        for part in ttnn.get_device_tensors(output):
            actual = ttnn.to_torch(part)
            assert torch.isfinite(actual).all()
            output_hashes.append(digest(actual))
            checks.append(
                accuracy(actual[:, :, selected, :].reshape(1, batch, -1), expected_output.reshape(1, batch, -1))
            )
            by_row.append(
                [
                    accuracy(
                        actual[:, :, index, :].reshape(1, batch, -1), expected_output[:, :, i, :].reshape(1, batch, -1)
                    )
                    for i, index in enumerate(selected)
                ]
            )
        ttnn.deallocate(output)
        cache_hashes = [hashes(cache) for cache in caches]
        cache_unchanged = cache_hashes == [[h] * 4 for h in expected_hashes]
        query_unchanged = hashes(q) == query_hashes
        assert cache_unchanged and query_unchanged, "Precision diagnostic changed cache contents or query"
        row["arms"].append(
            dict(
                name=name,
                configuration=choice,
                output_sha256_per_rank=output_hashes,
                accuracy_per_rank=checks,
                accuracy_by_query_row_per_rank=by_row,
                cache_sha256_per_rank=cache_hashes,
                cache_unchanged=cache_unchanged,
                query_unchanged=query_unchanged,
                wall_seconds_including_compilation=elapsed,
                performance_measurement=False,
            )
        )
        save(path, report)
        print("PREFILL_NUMERICS", batch, start, length, name, checks[0], flush=True)
    assert row["arms"][0]["output_sha256_per_rank"] == row["arms"][-1]["output_sha256_per_rank"]
    for tensor in [q, tt_table, *updates, *caches]:
        ttnn.deallocate(tensor)
    ttnn.synchronize_device(mesh)
    gc.collect()


@pytest.mark.skipif(os.getenv("QWEN_PREFILL_DIAGNOSTIC") != "1", reason="explicit allocated Galaxy diagnostic")
def test_prefill_attention_diagnostic():
    assert not any(os.getenv(k) for k in ("TT_METAL_SIMULATOR", "TT_METAL_SLOW_DISPATCH_MODE"))
    path = Path(os.environ["QWEN_PREFILL_DIAGNOSTIC_RECEIPT"])
    assert not path.exists()
    torch.set_num_threads(8)
    root = Path(__file__).resolve().parents[1]
    native = Path(os.environ["TT_METAL_HOME"])
    source_names = (
        "ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa.cpp",
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp",
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_streaming.hpp",
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp",
    )
    report = dict(
        state="opening",
        cleanup_completed=False,
        cases=[],
        source_sha256=model_source_hashes(root),
        native_source_sha256={n: hashlib.sha256((native / n).read_bytes()).hexdigest() for n in source_names},
        precision=dict(query="bfloat16", kv_cache="bfloat8_b", reference="FP64 plus independent FP32 Torch SDPA"),
        scope="Numerical attribution only; unchanged gates, no speed comparison, no model/serving promotion",
    )
    save(path, report)
    parent = mesh = None
    try:
        configure_fabric(topology=ttnn.Topology.Linear)
        parent = ttnn.open_mesh_device(ttnn.MeshShape(8, 4), trace_region_size=200000000)
        mesh = parent.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
        report["device_ids"] = list(mesh.get_device_ids())
        for case in CASES:
            report.update(state="running", active_case=case)
            save(path, report)
            run_case(mesh, *case, report, path)
        report.update(state="completed")
    except BaseException as error:
        report.update(state="failed", error=type(error).__name__, detail=str(error)[:3000])
        raise
    finally:
        try:
            try:
                if mesh is not None:
                    ttnn.close_mesh_device(mesh)
            finally:
                if parent is not None:
                    ttnn.close_mesh_device(parent)
            report["cleanup_completed"] = parent is not None
            if report["state"] == "completed":
                report["validation"] = validate_report(report)
        finally:
            save(path, report)
