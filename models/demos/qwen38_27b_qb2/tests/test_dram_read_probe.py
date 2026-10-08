# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Physical read calibration; UINT32 payload matches BFP8 tile byte geometry."""

import gc
import hashlib
import os
import statistics
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.qwen38_27b_qb2.tests.test_long_context_attention import save
from models.demos.qwen38_27b_qb2.tt import dram_read_probe as probe
from models.demos.qwen38_27b_qb2.tt.generator import configure_fabric


def payload(pages, salt):
    result = torch.empty((pages, probe.WORDS), dtype=torch.int32)
    words = torch.arange(probe.WORDS, dtype=torch.int64) * 0x45D9F3B
    for start in range(0, pages, 4096):
        end = min(start + 4096, pages)
        ids = torch.arange(start, end, dtype=torch.int64).reshape(-1, 1)
        result[start:end] = ((ids * 0x1F123BB5 + words + salt) & 0x7FFFFFFF).to(torch.int32)
    return result


def upload(mesh, host):
    return ttnn.from_torch(
        host,
        device=mesh,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def check_receipt(receipt, pages, variant, salt):
    work = probe.assignments(pages, variant["mode"], variant["placement"])
    expected = torch.tensor([probe.expected_markers(row, variant["packet_pages"], salt) for row in work])
    ranks = ttnn.get_device_tensors(receipt)
    assert len(ranks) == 4
    for rank in ranks:
        actual = ttnn.to_torch(rank).to(torch.int64) & 0xFFFFFFFF
        assert torch.equal(
            actual, expected
        ), "Read stream has missing, reordered, corrupt or overwritten packet markers"
    return hashlib.sha256(expected.numpy().tobytes()).hexdigest()


def buffers(mesh, host, variant):
    work = probe.assignments(host.shape[0], variant["mode"], variant["placement"])
    return (
        upload(mesh, host),
        ttnn.empty(
            tuple(host.shape),
            device=mesh,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
        upload(mesh, torch.zeros(len(work), 8, dtype=torch.int32)),
    )


def validate_bytes(mesh, variant):
    # Keep two independent allocations alive simultaneously to check runtime
    # rebinding on descriptor-cache hits. Odd bank length exercises every tail.
    salts = (12345, 891011)
    hosts = [payload(8 * 257, salt) for salt in salts]
    tensors = [buffers(mesh, host, variant) for host in hosts]
    assert tensors[0][0].buffer_address() != tensors[1][0].buffer_address()
    hashes = []
    for values, host, salt in zip(tensors, hosts, salts):
        source, copied, receipt = values
        addresses = [t.buffer_address() for t in values]
        probe.read(*values, **variant, copy_payload=True)
        check_receipt(receipt, host.shape[0], variant, salt)
        for tensor in (source, copied):
            ranks = ttnn.get_device_tensors(tensor)
            assert len(ranks) == 4
            assert all(torch.equal(ttnn.to_torch(rank).to(torch.int32), host) for rank in ranks)
        # Same reader under fast-consumer pressure, preserving complete packet
        # markers but omitting the validation-mode DRAM copy from timing.
        probe.read(*values, **variant, copy_payload=False)
        check_receipt(receipt, host.shape[0], variant, salt)
        assert addresses == [t.buffer_address() for t in values]
        hashes.append(hashlib.sha256(host.numpy().tobytes()).hexdigest())
    return dict(
        variant=variant,
        full_payload_pages=2056,
        independent_allocations=2,
        full_bytes_equal_all_four_ranks=True,
        input_sha256=hashes,
    )


def timed_case(mesh, source, copied, pages, variant, salt):
    work = probe.assignments(pages, variant["mode"], variant["placement"])
    receipt = upload(mesh, torch.zeros(len(work), 8, dtype=torch.int32))
    values = (source, copied, receipt)
    addresses = [t.buffer_address() for t in values]

    def invoke():
        probe.read(*values, **variant)

    invoke()
    marker_hash = check_receipt(receipt, pages, variant, salt)
    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        invoke()
    except BaseException:
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        ttnn.release_trace(mesh, trace)
        raise
    ttnn.end_trace_capture(mesh, trace, cq_id=0)
    samples = []
    try:
        for _ in range(5):
            ttnn.synchronize_device(mesh)
            start = time.perf_counter()
            for _ in range(20):
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            elapsed = (time.perf_counter() - start) * 1e6 / 20
            assert 0 < elapsed < 1e7
            samples.append(elapsed)
            assert check_receipt(receipt, pages, variant, salt) == marker_hash
    finally:
        ttnn.release_trace(mesh, trace)
    assert addresses == [t.buffer_address() for t in values]
    median = statistics.median(samples)
    return dict(
        variant=variant,
        pages=pages,
        unique_read_bytes_per_chip=pages * probe.PAGE_BYTES,
        traced_call_us=samples,
        median_traced_call_us=median,
        useful_read_gb_s_per_chip=pages * probe.PAGE_BYTES / median / 1000,
        packet_markers_passed_all_four_ranks=True,
        marker_sha256=marker_hash,
        includes_trace_dispatch_and_consumer_marker_work=True,
        model_speedup_claim=False,
    )


@pytest.mark.skipif(os.getenv("QWEN_DRAM_READ_PROBE") != "1", reason="explicit allocated-Galaxy diagnostic")
def test_dram_read_probe():
    path = Path(os.environ["QWEN_DRAM_READ_RECEIPT"])
    assert not path.exists(), "Preserve each attempt"
    torch.set_num_threads(8)
    sources = [Path(__file__), Path(probe.__file__), *Path(probe.__file__).parent.glob("dram_read_probe_*.cpp")]
    report = dict(
        state="opening",
        passed=False,
        cleanup_completed=False,
        promoted_to_model=False,
        scope="Raw UINT32 1088-byte page read calibration; not BFP8 attention or counter bandwidth",
        full_byte_validation_scope="2056 pages, two inputs/all four ranks, every variant",
        timed_integrity_scope="first/last word of every packet plus ordered aggregate markers on all ranks",
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        validation=[],
        cases=[],
        comparisons=[],
    )
    save(path, report)
    configure_fabric(topology=ttnn.Topology.Linear)
    parent = ttnn.open_mesh_device(ttnn.MeshShape(8, 4), trace_region_size=200000000)
    mesh = None
    try:
        mesh = parent.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
        report["device_ids"] = list(mesh.get_device_ids())
        assert len(report["device_ids"]) == 4
        cases = probe.variants()
        for variant in cases:
            report.update(state="byte_validation", active_variant=variant)
            save(path, report)
            report["validation"].append(validate_bytes(mesh, variant))
            save(path, report)
            gc.collect()
        # K+V bytes for one attention layer/chip. The last volume covers both
        # 128K/B16 and 256K/B8; this does not duplicate a model-context result.
        for pages, geometries in (
            (262144, [[32768, 16], [16384, 32]]),
            (524288, [[32768, 32]]),
            (1048576, [[131072, 16], [262144, 8]]),
        ):
            host = payload(pages, 314159)
            source, copied, initial_receipt = buffers(mesh, host, cases[0])
            del initial_receipt, host
            group = []
            for variant in (*cases, cases[0]):
                report.update(state="timing", active_pages=pages, active_variant=variant)
                save(path, report)
                row = timed_case(mesh, source, copied, pages, variant, 314159)
                row["equivalent_kv_geometries_isl_batch"] = geometries
                group.append(row)
                report["cases"].append(row)
                print("DRAM_READ_PROBE", row, flush=True)
                save(path, report)
            before, after = (group[i]["median_traced_call_us"] for i in (0, -1))
            drift = abs(after / before - 1)
            report["comparisons"].append(
                dict(
                    pages=pages,
                    control_drift=drift,
                    timing_comparison_qualified=drift <= 0.03,
                    best_raw_read_gb_s=max(row["useful_read_gb_s_per_chip"] for row in group),
                    model_promotion=False,
                )
            )
            save(path, report)
            del source, copied
            gc.collect()
        report.update(state="completed", passed=True)
    except BaseException as error:
        report.update(state="failed", passed=False, error=dict(type=type(error).__name__, message=str(error)[:3000]))
        raise
    finally:
        try:
            try:
                if mesh is not None:
                    ttnn.close_mesh_device(mesh)
            finally:
                ttnn.close_mesh_device(parent)
            report["cleanup_completed"] = True
        except BaseException as error:
            report.update(state="failed", passed=False, cleanup_error=str(error)[:2000])
            raise
        finally:
            save(path, report)
