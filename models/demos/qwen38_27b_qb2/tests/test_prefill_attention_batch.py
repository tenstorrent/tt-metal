# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Physical TP4 control/batched/control prefill boundary with exact cache checks."""

import gc
import hashlib
import os
import statistics
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.qwen38_27b_qb2.tests.attention_tuning import accuracy
from models.demos.qwen38_27b_qb2.tests.prefill_attention_batch import CASES
from models.demos.qwen38_27b_qb2.tests.test_long_context_attention import save
from models.demos.qwen38_27b_qb2.tt.generator import configure_fabric
from models.demos.qwen38_27b_qb2.tt.prefill_attention import fill_and_attend, make_batch_indices


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def run_case(mesh, indices, batch, start, length, report, path):
    rng = torch.Generator().manual_seed(20261010 + batch + start + length)
    pages = (start + length + 31) // 32 + 1
    table = (
        torch.randperm((batch + 2) * pages, generator=rng, dtype=torch.int32)
        .reshape(batch + 2, pages)[1 : batch + 1]
        .contiguous()
    )
    row = dict(
        batch=batch,
        start_pos=start,
        chunk_tokens=length,
        total_context=start + length,
        page_table_sha256=digest(table),
        state="uploading",
        arms=[],
        live_query_heads=6,
        kv_heads=1,
        head_dim=256,
    )
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
    row["expected_cache_sha256"] = expected_hashes
    # Dense reference for selected first/middle/last query rows covers each user,
    # every head and the whole prefix while bounding host attention workspace.
    selected = sorted({0, length // 2, length - 1})
    expected_output = []
    for user in range(batch):
        key, value = [host[table[user]].permute(1, 0, 2, 3).flatten(1, 2).float() for host in expected_caches]
        scores = query[user, :, selected, :].float() @ key.transpose(-1, -2) * 256**-0.5
        allowed = torch.arange(pages * 32)[None, :] <= start + torch.tensor(selected)[:, None]
        expected_output.append(torch.softmax(scores.masked_fill(~allowed, float("-inf")), dim=-1) @ value)
    expected_output = torch.stack(expected_output)
    row.update(reference_query_rows=selected, reference_output_sha256=digest(expected_output))
    del expected_caches, expected, host_update, query
    gc.collect()
    chunk = 128 if start % 128 == 0 and length >= 128 else 32
    config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=mesh.compute_with_storage_grid_size(), q_chunk_size=chunk, k_chunk_size=chunk
    )

    def invoke(batched):
        return fill_and_attend(
            q,
            *updates,
            *caches,
            tt_table,
            start,
            page_size=32,
            scale=256**-0.5,
            program_config=config,
            batch_indices=indices[batch] if batched else None,
        )

    for name, batched in (("before", False), ("batched", True), ("after", False)):
        row.update(state="measuring", active_arm=name)
        save(path, report)
        for _ in range(2):
            output = invoke(batched)
            ttnn.synchronize_device(mesh)
            ttnn.deallocate(output)
        samples = []
        for _ in range(5):
            tick = time.perf_counter()
            output = invoke(batched)
            ttnn.synchronize_device(mesh)
            samples.append((time.perf_counter() - tick) * 1000)
            ttnn.deallocate(output)
        output = invoke(batched)
        ttnn.synchronize_device(mesh)
        hashes, checks = [], []
        for device_tensor in ttnn.get_device_tensors(output):
            actual = ttnn.to_torch(device_tensor)
            hashes.append(digest(actual))
            checks.append(
                accuracy(actual[:, :, selected, :].reshape(1, batch, -1).float(), expected_output.reshape(1, batch, -1))
            )
            del actual
        ttnn.deallocate(output)
        cache_hashes = []
        for cache, expected_hash in zip(caches, expected_hashes):
            actual_hashes = [digest(ttnn.to_torch(part)) for part in ttnn.get_device_tensors(cache)]
            assert (
                len(actual_hashes) == 4 and actual_hashes == [expected_hash] * 4
            ), "Cache write changed another request or prefix"
            cache_hashes.append(actual_hashes)
        assert len(checks) == 4 and all(
            check["passed"] for check in checks
        ), "Prefill attention failed selected-row dense reference"
        row["arms"].append(
            dict(
                name=name,
                wall_ms=samples,
                median_ms=statistics.median(samples),
                output_sha256_per_rank=hashes,
                accuracy_per_rank=checks,
                cache_sha256_per_rank=cache_hashes,
            )
        )
        save(path, report)
    before, candidate, after = row["arms"]
    assert (
        before["output_sha256_per_rank"] == candidate["output_sha256_per_rank"] == after["output_sha256_per_rank"]
    ), "Batched attention output differs"
    drift = abs(after["median_ms"] / before["median_ms"] - 1)
    row.update(
        state="completed",
        outputs_bit_identical=True,
        control_drift_fraction=drift,
        timing_qualified=drift <= 0.03,
        speedup=statistics.median([before["median_ms"], after["median_ms"]]) / candidate["median_ms"]
        if drift <= 0.03
        else None,
    )
    save(path, report)
    print("PREFILL_BATCH_CASE", batch, start, length, row["speedup"], flush=True)
    for tensor in [q, tt_table, *updates, *caches]:
        ttnn.deallocate(tensor)
    ttnn.synchronize_device(mesh)
    gc.collect()


@pytest.mark.skipif(
    os.getenv("QWEN_PREFILL_ATTENTION_BATCH_TEST") != "1", reason="explicit allocated-Galaxy boundary test"
)
def test_prefill_attention_batch():
    path = Path(os.environ["QWEN_PREFILL_ATTENTION_BATCH_RECEIPT"])
    assert not path.exists(), "Preserve each attempt"
    source = Path(__file__).resolve().parents[1]
    report = dict(
        state="opening",
        passed=False,
        cleanup_completed=False,
        cases=[],
        scope="One TP4 replica, prefill cache-fill and attention boundary; no projection, model or eval gain claimed",
        source_sha256={
            str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (source / "tt").rglob("*")
            if p.suffix in (".py", ".cpp", ".h", ".hpp")
        },
        precision=dict(query="bfloat16", kv_cache="bfloat8_b", native_sdpa_defaults=True),
        timing="Two eager warmups then five synchronized calls per arm; includes host dispatch, not trace replay",
    )
    save(path, report)
    torch.set_num_threads(8)
    configure_fabric(topology=ttnn.Topology.Linear)
    parent = ttnn.open_mesh_device(ttnn.MeshShape(8, 4), trace_region_size=200000000)
    mesh = None
    try:
        mesh = parent.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
        report["device_ids"] = list(mesh.get_device_ids())
        assert len(report["device_ids"]) == 4
        indices = make_batch_indices(mesh)
        for case in CASES:
            run_case(mesh, indices, *case, report, path)
        report.update(state="completed", passed=True, promoted_to_serving=False)
    except BaseException as error:
        report.update(state="failed", error=dict(type=type(error).__name__, message=str(error)[:3000]))
        raise
    finally:
        try:
            if mesh is not None:
                ttnn.close_mesh_device(mesh)
        finally:
            ttnn.close_mesh_device(parent)
        report["cleanup_completed"] = True
        save(path, report)
