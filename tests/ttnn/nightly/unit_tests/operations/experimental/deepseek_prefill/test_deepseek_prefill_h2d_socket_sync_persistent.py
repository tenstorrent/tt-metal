# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Persistent-destination / trace-capture coverage for inbound_socket_service_sync
(tenstorrent/tt-metal#52451, #52456).

`tokens_out` / `metadata_out` let the CALLER own the op's destinations so nothing is
allocated per call -- the precondition for capturing the op in a ttnn trace, since a
trace records destination addresses once and re-patches nothing on replay.

Single-device (1x1) on purpose: same fixture shape as
test_deepseek_prefill_h2d_socket_sync.py, so this runs in the existing nightly job.
"""

import gc
import struct

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole, skip_for_slow_dispatch

# Same gates as the sibling test_deepseek_prefill_h2d_socket_sync.py: the service claims a service core
# (Blackhole-only) and drains a host->device socket (host IOMMU), and trace capture is fast-dispatch only.
# requires_host_iommu also routes this file to the viommu nightly leg and OUT of the four non-IOMMU legs,
# which run -m "not requires_host_iommu".
pytestmark = [
    pytest.mark.requires_host_iommu,
    skip_for_slow_dispatch(),
    pytest.mark.skipif(
        not is_blackhole(),
        reason="H2DStreamService requires Blackhole (service-core claims); see service_core_manager.cpp",
    ),
]

_DTYPE_TORCH = torch.int32
_DTYPE_TTNN = ttnn.uint32
_DTYPE_SIZE = 4
_MD_BYTES = 12  # 3 x uint32: [slot_id, actual_start, actual_end]
_ISL = 640
_TRACE_REGION = 16 * 1024 * 1024


def _build_service(mesh_device):
    shape_list = [mesh_device.shape[0], 1, _ISL]
    return (
        ttnn.H2DStreamService(
            mesh_device=mesh_device,
            global_spec=ttnn.TensorSpec(
                shape=ttnn.Shape(shape_list),
                dtype=_DTYPE_TTNN,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                buffer_type=ttnn.BufferType.DRAM,
            ),
            max_socket_page_size_bytes=_ISL * _DTYPE_SIZE,
            worker_cores=ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0)),
            metadata_size_bytes=_MD_BYTES,
        ),
        shape_list,
    )


def _alloc_u32(mesh_device, shape):
    return ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec(
            shape=ttnn.Shape(shape),
            dtype=_DTYPE_TTNN,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            buffer_type=ttnn.BufferType.DRAM,
        ),
        mesh_device,
    )


def _push(service, shape_list, seed, slot, end):
    torch.manual_seed(seed)
    src = torch.randint(0, 2**31, shape_list, dtype=_DTYPE_TORCH)
    service.forward_to_tensor_bytes(src.contiguous().numpy(), metadata=struct.pack("<III", slot, 0, end))
    return src


def _tokens(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).view(-1).to(torch.int64)


def _words(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).flatten().to(torch.int64).tolist()


# ---------------------------------------------------------------------------
# 1. Each destination combination is honoured, and each is its own program.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mesh_device", [1], indirect=True)  # 1x1 single-device mesh
def test_persistent_destinations_all_modes(mesh_device):
    service, shape_list = _build_service(mesh_device)
    tokens_out = ttnn.allocate_tensor_on_device(service.get_per_shard_spec(), mesh_device)
    metadata_out = _alloc_u32(mesh_device, [1, 1, 1, _MD_BYTES // 4])
    tok_addr, md_addr = tokens_out.buffer_address(), metadata_out.buffer_address()

    modes = [
        ("eager", {}),
        ("tokens_only", {"tokens_out": tokens_out}),
        ("metadata_only", {"metadata_out": metadata_out}),
        ("full", {"tokens_out": tokens_out, "metadata_out": metadata_out}),
    ]
    entries = []
    for i, (name, kwargs) in enumerate(modes):
        src = _push(service, shape_list, i, 40 + i, _ISL)
        pre = mesh_device.num_program_cache_entries()
        tok, md = ttnn.experimental.deepseek_prefill.inbound_socket_service_sync(
            service, metadata_size_bytes=_MD_BYTES, **kwargs
        )
        entries.append(mesh_device.num_program_cache_entries())
        assert torch.equal(_tokens(tok), src.view(-1).to(torch.int64)), f"{name}: token mismatch"
        assert _words(md)[:3] == [40 + i, 0, _ISL], f"{name}: metadata mismatch"
        if "tokens_out" in kwargs:
            assert tok.buffer_address() == tok_addr, f"{name}: op ignored tokens_out"
        if "metadata_out" in kwargs:
            assert md.buffer_address() == md_addr, f"{name}: op ignored metadata_out"
        logger.info(f"{name}: correct (cache entries {pre} -> {entries[-1]})")

    assert entries[-1] == 4, f"expected one program per destination mode, got {entries[-1]} ({entries})"
    service.barrier()
    del service


# ---------------------------------------------------------------------------
# 2. Spec mismatch on a caller-supplied destination is rejected, not silently wrong.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_persistent_destination_spec_is_validated(mesh_device, expect_error):
    service, shape_list = _build_service(mesh_device)
    wrong = _alloc_u32(mesh_device, [1, 1, _ISL // 2])  # half the pages
    _push(service, shape_list, 7, 7, _ISL)
    with expect_error(RuntimeError, "tokens_out"):
        ttnn.experimental.deepseek_prefill.inbound_socket_service_sync(
            service, metadata_size_bytes=_MD_BYTES, tokens_out=wrong
        )
    logger.info("spec mismatch rejected")
    # Drain the pending push so teardown is clean.
    ttnn.experimental.deepseek_prefill.inbound_socket_service_sync(service, metadata_size_bytes=_MD_BYTES)
    service.barrier()
    del service
    # The rejection above leaves an exception whose traceback pins THIS frame; that
    # frame/traceback cycle only the cyclic collector can break, so without this the service
    # outlives the mesh_device fixture and its destructor logs TT_FATAL backtraces (device
    # already closed) into a passing run's log. Collect while the device is still open.
    gc.collect()


# ---------------------------------------------------------------------------
# 3. THE POINT: capture the op in a trace and replay it. Also covers the
#    on-device metadata scatter (ttnn.slice into pre-allocated 1-element
#    tensors) that lets a consumer read the record from inside the SAME trace.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("device_params", [{"trace_region_size": _TRACE_REGION}], indirect=True)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_inbound_replays_from_inside_a_trace(mesh_device):
    service, shape_list = _build_service(mesh_device)
    tokens_out = ttnn.allocate_tensor_on_device(service.get_per_shard_spec(), mesh_device)
    record = _alloc_u32(mesh_device, [1, 1, 1, _MD_BYTES // 4])
    slot_t = _alloc_u32(mesh_device, [1, 1, 1, 1])
    end_t = _alloc_u32(mesh_device, [1, 1, 1, 1])
    addrs = tuple(t.buffer_address() for t in (tokens_out, record, slot_t, end_t))

    # `svc=service` captures the service by VALUE so this stays callable independently of the teardown
    # `del service` below (the repo convention for releasing the service core).
    def drain_and_scatter(svc=service):
        ttnn.experimental.deepseek_prefill.inbound_socket_service_sync(
            svc, metadata_size_bytes=_MD_BYTES, tokens_out=tokens_out, metadata_out=record
        )
        ttnn.slice(record, [0, 0, 0, 0], [1, 1, 1, 1], output_tensor=slot_t)
        ttnn.slice(record, [0, 0, 0, 2], [1, 1, 1, 3], output_tensor=end_t)

    # Warm outside the capture so the capture only records.
    src = _push(service, shape_list, 100, 500, 321)
    drain_and_scatter()
    ttnn.synchronize_device(mesh_device)
    assert torch.equal(_tokens(tokens_out), src.view(-1).to(torch.int64)), "warmup token mismatch"
    assert (_words(slot_t)[0], _words(end_t)[0]) == (500, 321), "warmup scatter mismatch"

    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    drain_and_scatter()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    assert (
        tuple(t.buffer_address() for t in (tokens_out, record, slot_t, end_t)) == addrs
    ), "capture moved a caller-supplied destination"
    logger.info(f"captured inbound drain + metadata scatter as one trace (id={tid})")

    for i in range(3):
        slot, end = 600 + i, 111 + 7 * i
        src = _push(service, shape_list, 200 + i, slot, end)
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        assert torch.equal(_tokens(tokens_out), src.view(-1).to(torch.int64)), f"replay {i}: token mismatch"
        got = (_words(slot_t)[0], _words(end_t)[0])
        assert got == (slot, end), f"replay {i}: scattered scalars stale -- got {got}, want {(slot, end)}"
        assert tokens_out.buffer_address() == addrs[0], f"replay {i}: destination moved"
        logger.info(f"replay {i}: tokens + scattered scalars {got} both fresh")

    ttnn.release_trace(mesh_device, tid)
    service.barrier()
    del service
