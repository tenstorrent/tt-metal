# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Physical DRAM-to-worker delivery, credit/backpressure and trace-replay test."""

import gc
import hashlib
import os
import statistics
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.qwen38_27b_qb2.tests.test_dram_read_probe import buffers, check_receipt, payload, upload
from models.demos.qwen38_27b_qb2.tests.test_long_context_attention import save
from models.demos.qwen38_27b_qb2.tt import dram_delivery_probe as delivery
from models.demos.qwen38_27b_qb2.tt import dram_read_probe as raw
from models.demos.qwen38_27b_qb2.tt.generator import configure_fabric


def validate(mesh, variant):
    # Both allocations remain alive: different addresses, salts and packet-tail
    # lengths exercise runtime rebinding instead of one cached lucky address.
    hosts = [payload(pages, salt) for pages, salt in ((8, 12345), (2056, 891011))]
    allocations = [buffers(mesh, host, variant) for host in hosts]
    assert allocations[0][0].buffer_address() != allocations[1][0].buffer_address()
    checks = []
    for values, host, salt in zip(allocations, hosts, (12345, 891011)):
        source, copied, receipt = values
        for delay in (0, 4096):
            options = {**variant, "consumer_delay": delay}
            delivery.read(*values, **options, copy_payload=True)
            check_receipt(receipt, host.shape[0], variant, salt)
            for tensor in (source, copied):
                ranks = ttnn.get_device_tensors(tensor)
                assert len(ranks) == 4
                assert all(torch.equal(ttnn.to_torch(rank).to(torch.int32), host) for rank in ranks)
            # Repeat through trace dispatch to catch stale semaphore/CB state.
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            try:
                delivery.read(*values, **options, copy_payload=True)
            finally:
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
            try:
                for _ in range(3):
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    check_receipt(receipt, host.shape[0], variant, salt)
                    assert all(
                        torch.equal(ttnn.to_torch(rank).to(torch.int32), host)
                        for rank in ttnn.get_device_tensors(copied)
                    )
            finally:
                ttnn.release_trace(mesh, trace)
            checks.append(dict(pages=host.shape[0], delay=delay, trace_replays=3, all_four_ranks_bytes_equal=True))
    return dict(variant=variant, checks=checks, full_bytes_equal_all_four_ranks=True)


def measure(mesh, source, copied, pages, variant, *, remote):
    receipt = upload(mesh, torch.zeros(raw.BANKS, 8, dtype=torch.int32))
    values = (source, copied, receipt)
    read = delivery.read if remote else raw.read

    def invoke():
        read(*values, **variant)

    invoke()
    marker = check_receipt(receipt, pages, variant, 314159)
    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        invoke()
    finally:
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
            assert check_receipt(receipt, pages, variant, 314159) == marker
    finally:
        ttnn.release_trace(mesh, trace)
    median = statistics.median(samples)
    return dict(
        variant=variant,
        remote_delivery=remote,
        pages=pages,
        traced_call_us=samples,
        median_traced_call_us=median,
        unique_read_bytes_per_chip=pages * raw.PAGE_BYTES,
        useful_read_gb_s_per_chip=pages * raw.PAGE_BYTES / median / 1000,
        packet_markers_passed_all_four_ranks=True,
        marker_sha256=marker,
        includes_dispatch_and_consumer_work=True,
        includes_attention_math=False,
        model_speedup_claim=False,
    )


@pytest.mark.skipif(os.getenv("QWEN_DRAM_DELIVERY_PROBE") != "1", reason="explicit allocated-Galaxy diagnostic")
def test_dram_delivery_probe():
    path = Path(os.environ["QWEN_DRAM_DELIVERY_RECEIPT"])
    assert not path.exists(), "Preserve each attempt"
    torch.set_num_threads(8)
    directory = Path(delivery.__file__).parent
    files = [
        Path(__file__),
        Path(delivery.__file__),
        Path(raw.__file__),
        Path(__file__).with_name("test_dram_read_probe.py"),
    ]
    files += list(directory.glob("dram_*_probe_*.cpp"))
    report = dict(
        state="opening",
        passed=False,
        cleanup_completed=False,
        promoted_to_model=False,
        scope="Eight bank readers plus eight remote consumers per chip; not production attention or KV paging",
        timed_integrity_scope="First/last word of every packet plus ordered markers, all four ranks",
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
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
        variants = delivery.variants()
        if os.getenv("QWEN_DRAM_DELIVERY_EXTENDED") == "1":
            # Reuse the already verified mover while sweeping ring size, packet
            # aggregation and deliberate receiver backpressure together.
            variants = [
                dict(
                    mode="bank_bulk",
                    placement="pinned",
                    consumer_placement=placement,
                    packet_pages=pages,
                    depth=depth,
                    consumer_delay=delay,
                )
                for placement in ("near", "center", "opposite")
                for pages in (4, 8, 15)
                for depth in (2, 4, 8)
                for delay in (0, 4096)
            ]
        for variant in variants:
            report.update(state="byte_validation", active_variant=variant)
            save(path, report)
            report["validation"].append(validate(mesh, variant))
            save(path, report)
            gc.collect()
        control = dict(mode="bank_bulk", placement="pinned", packet_pages=8, depth=4)
        for pages in (262144, 524288, 1048576):
            host = payload(pages, 314159)
            source, copied, receipt = buffers(mesh, host, control)
            del host, receipt
            group = []
            for remote, variant in [(False, control), *[(True, v) for v in variants], (False, control)]:
                report.update(state="timing", active_pages=pages, active_variant=variant, remote_delivery=remote)
                save(path, report)
                row = measure(mesh, source, copied, pages, variant, remote=remote)
                report["cases"].append(row)
                group.append(row)
                print("DRAM_DELIVERY_PROBE", row, flush=True)
                save(path, report)
            before, after = (group[i]["median_traced_call_us"] for i in (0, -1))
            drift = abs(after / before - 1)
            report["comparisons"].append(
                dict(
                    pages=pages,
                    control_drift=drift,
                    timing_comparison_qualified=drift <= 0.03,
                    raw_control_gb_s=statistics.mean(group[i]["useful_read_gb_s_per_chip"] for i in (0, -1)),
                    best_delivery_gb_s=max(row["useful_read_gb_s_per_chip"] for row in group if row["remote_delivery"]),
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
