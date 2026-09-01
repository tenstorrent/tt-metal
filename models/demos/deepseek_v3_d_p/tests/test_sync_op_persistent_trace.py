# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""
Acceptance tests for the persistent-destination / trace-aware socket sync ops
(tenstorrent/tt-metal#52451 inbound, #52456 outbound).

`inbound_socket_service_sync` takes `tokens_out` / `metadata_out` so the caller
owns the destinations and the op allocates nothing -- the precondition for
capturing the call in a ttnn trace, which records each kernel's runtime args once
and re-patches no addresses on replay.  These tests exercise those arguments
directly and demonstrate the actual goal: a trace that CONTAINS the op and
replays it correctly against fresh socket pushes.
"""

import gc
import struct

import pytest
import torch
from loguru import logger

import ttnn

_MD_BYTES = 12  # 3 x uint32: [slot_id, actual_start, actual_end]
_TRACE_REGION = 16 * 1024 * 1024

_MESH_8x4 = pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4")


def _params(**extra):
    dp = {"fabric_config": ttnn.FabricConfig.FABRIC_2D}
    dp.update(extra)
    return pytest.param((8, 4), dp, marks=_MESH_8x4, id="mesh-8x4")


def _make_service(mesh_device, isl_per_chip, metadata_size_bytes=_MD_BYTES):
    sp_factor = mesh_device.shape[0]
    global_spec = ttnn.TensorSpec(
        shape=ttnn.Shape([sp_factor, 1, isl_per_chip]),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        buffer_type=ttnn.BufferType.DRAM,
    )
    mapper = ttnn.create_mesh_mapper(
        mesh_device,
        ttnn.MeshMapperConfig(placements=[ttnn.PlacementShard(0), ttnn.PlacementReplicate()]),
    )
    return ttnn.H2DStreamService(
        mesh_device=mesh_device,
        global_spec=global_spec,
        max_socket_page_size_bytes=isl_per_chip * 4,
        mapper=mapper,
        worker_cores=ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0)),
        metadata_size_bytes=metadata_size_bytes,
    )


def _push(service, sp_factor, isl_per_chip, vocab, slot):
    """Push one random transfer; return the torch tokens that were sent."""
    tokens = torch.randint(0, vocab, (sp_factor, 1, isl_per_chip), dtype=torch.int64)
    service.forward_to_tensor_bytes(
        tokens.to(torch.int32).contiguous().numpy(),
        metadata=struct.pack("<III", slot, 0, isl_per_chip),
    )
    return tokens


def _check_tokens(tt_tokens, expected, tag):
    """Bit-exact per-shard compare. Shard k of the mesh row-major order holds row k // ncols."""
    shards = ttnn.get_device_tensors(tt_tokens)
    sp_factor = expected.shape[0]
    ncols = len(shards) // sp_factor
    for k, shard in enumerate(shards):
        got = ttnn.to_torch(shard).flatten().to(torch.int64)
        want = expected[k // ncols].flatten().to(torch.int64)
        assert torch.equal(got, want), f"{tag}: shard {k} token mismatch (first diff at {(got != want).nonzero()[:3]})"


def _check_meta(tt_meta, slot, isl_per_chip, tag):
    vals = ttnn.to_torch(ttnn.get_device_tensors(tt_meta)[0]).flatten().to(torch.int64)[:3].tolist()
    assert vals == [slot, 0, isl_per_chip], f"{tag}: metadata {vals} != {[slot, 0, isl_per_chip]}"


# ---------------------------------------------------------------------------
# 1. The new API actually works and actually stops allocating.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mesh_device, device_params", [_params()], indirect=["mesh_device", "device_params"])
def test_h2d_persistent_destinations(mesh_device):
    sp_factor = mesh_device.shape[0]
    isl_per_chip = 512
    vocab = 4096
    service = _make_service(mesh_device, isl_per_chip)

    tokens_out = ttnn.allocate_tensor_on_device(service.get_per_shard_spec(), mesh_device)
    metadata_out = ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec(
            shape=ttnn.Shape([1, 1, 1, _MD_BYTES // 4]),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            buffer_type=ttnn.BufferType.DRAM,
        ),
        mesh_device,
    )
    tok_addr, md_addr = tokens_out.buffer_address(), metadata_out.buffer_address()
    logger.info(f"persistent tokens_out @ 0x{tok_addr:x}, metadata_out @ 0x{md_addr:x}")

    deltas = []
    for i in range(4):
        sent = _push(service, sp_factor, isl_per_chip, vocab, i)
        pre = mesh_device.num_program_cache_entries()
        tt_tokens, tt_meta = ttnn.experimental.deepseek_prefill.inbound_socket_service_sync(
            service, metadata_size_bytes=_MD_BYTES, tokens_out=tokens_out, metadata_out=metadata_out
        )
        deltas.append(mesh_device.num_program_cache_entries() - pre)

        # The op must hand back the caller's tensors, unmoved.
        assert tt_tokens.buffer_address() == tok_addr, f"iter {i}: tokens moved to 0x{tt_tokens.buffer_address():x}"
        assert tt_meta.buffer_address() == md_addr, f"iter {i}: metadata moved to 0x{tt_meta.buffer_address():x}"
        assert tokens_out.buffer_address() == tok_addr, f"iter {i}: caller's tokens_out was reallocated"
        _check_tokens(tt_tokens, sent, f"iter {i}")
        _check_meta(tt_meta, i, isl_per_chip, f"iter {i}")

    assert deltas[0] >= 1 and all(d == 0 for d in deltas[1:]), f"program cache misbehaved: {deltas}"
    logger.info(f"PASS persistent destinations: stable addresses, cache deltas {deltas}")
    service.barrier()
    del service


# ---------------------------------------------------------------------------
# 2. The program cache must not share an entry between eager / half / full
#    persistent modes.  `nullopt` optionals are SKIPPED by
#    extract_tensor_buffers_into, so the buffer-slot indices that
#    resolve_bindings burns into the cached entry differ per mode:
#       eager        [backing, tok_fresh, md_fresh]                 tok=1 md=2
#       md-only      [backing, md_out, tok_fresh, md_out]           tok=2 md=1  <-- SWAPPED
#       full         [backing, tok_out, md_out, tok_out, md_out]    tok=1 md=2
#    A shared cache entry across modes would patch the tokens base address with
#    the metadata buffer's address, i.e. write the whole token page into a
#    12-byte buffer.  md-only is exactly what prefill_runner.py uses.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mesh_device, device_params", [_params()], indirect=["mesh_device", "device_params"])
def test_h2d_program_cache_mode_isolation(mesh_device):
    sp_factor = mesh_device.shape[0]
    isl_per_chip = 512
    vocab = 4096
    service = _make_service(mesh_device, isl_per_chip)

    tokens_out = ttnn.allocate_tensor_on_device(service.get_per_shard_spec(), mesh_device)
    md_spec = ttnn.TensorSpec(
        shape=ttnn.Shape([1, 1, 1, _MD_BYTES // 4]),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        buffer_type=ttnn.BufferType.DRAM,
    )
    metadata_out = ttnn.allocate_tensor_on_device(md_spec, mesh_device)

    modes = {
        "eager": {},
        "md_only": {"metadata_out": metadata_out},
        "tok_only": {"tokens_out": tokens_out},
        "full": {"tokens_out": tokens_out, "metadata_out": metadata_out},
    }

    # Probe which modes this build accepts (the inbound branch rejects the half-persistent ones).
    # Each accepted mode must claim its OWN program-cache entry: nullopt tensor_args are
    # skipped when the framework collects buffers, so the tokens/metadata buffer slots
    # differ per mode and a shared entry would cross-wire the two base addresses.
    # A rejected mode raises in validate_* BEFORE any device program runs, so the pushed
    # transfer is left un-drained and un-acked -- and the service cannot be flushed
    # (barrier() waits on worker acks that will never come).  So push lazily: only when
    # the previous attempt actually consumed one.  Order matters -- "eager" is accepted by
    # every build, so it is placed last to drain whatever a trailing rejection left pending.
    accepted = {}
    entries_before_probe = mesh_device.num_program_cache_entries()
    pending = False
    probe_order = [m for m in modes if m != "eager"] + ["eager"]
    for name in probe_order:
        if not pending:
            _push(service, sp_factor, isl_per_chip, vocab, 0)
            pending = True
        try:
            ttnn.experimental.deepseek_prefill.inbound_socket_service_sync(
                service, metadata_size_bytes=_MD_BYTES, **modes[name]
            )
            accepted[name] = True
            pending = False
        except Exception as e:  # noqa: BLE001
            accepted[name] = False
            logger.info(f"mode {name!r} REJECTED: {str(e).splitlines()[0][:200]}")
    assert not pending, "probe left an un-drained transfer"
    distinct = mesh_device.num_program_cache_entries() - entries_before_probe
    live = [m for m in modes if accepted[m]]
    logger.info(f"accepted modes: {accepted}; distinct program-cache entries created: {distinct}")
    assert distinct == len(live), (
        f"expected one program-cache entry per accepted mode ({len(live)}), got {distinct} -- "
        "modes are sharing a cached entry, which cross-wires the tokens/metadata base addresses"
    )

    entries = {}
    slot = 0
    # Two passes: first occurrence of a mode may compile, second must cache-hit,
    # and in both passes the DATA must be right for every mode.
    for pass_idx in range(2):
        for name in live:
            slot += 1
            sent = _push(service, sp_factor, isl_per_chip, vocab, slot)
            pre = mesh_device.num_program_cache_entries()
            out = ttnn.experimental.deepseek_prefill.inbound_socket_service_sync(
                service, metadata_size_bytes=_MD_BYTES, **modes[name]
            )
            delta = mesh_device.num_program_cache_entries() - pre
            tag = f"pass{pass_idx}/{name}"
            _check_tokens(out[0], sent, tag)
            _check_meta(out[1], slot, isl_per_chip, tag)
            if pass_idx == 0:
                entries[name] = delta
            else:
                assert delta == 0, f"{tag}: recompiled on a repeat call (delta={delta})"
            logger.info(f"{tag}: OK (cache delta {delta})")

    logger.info(f"first-pass cache deltas per mode: {entries}")
    service.barrier()
    del service


# ---------------------------------------------------------------------------
# 3. THE POINT OF THE ISSUES: capture the op in a trace and replay it.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [_params(trace_region_size=_TRACE_REGION)],
    indirect=["mesh_device", "device_params"],
)
def test_h2d_inbound_in_trace(mesh_device):
    sp_factor = mesh_device.shape[0]
    isl_per_chip = 512
    vocab = 4096
    service = _make_service(mesh_device, isl_per_chip)

    tokens_out = ttnn.allocate_tensor_on_device(service.get_per_shard_spec(), mesh_device)
    metadata_out = ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec(
            shape=ttnn.Shape([1, 1, 1, _MD_BYTES // 4]),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            buffer_type=ttnn.BufferType.DRAM,
        ),
        mesh_device,
    )
    tok_addr, md_addr = tokens_out.buffer_address(), metadata_out.buffer_address()

    # `svc=service` captures the service by VALUE: these helpers must stay callable independently of
    # the teardown `del service` below (the repo convention for releasing the service core).
    def call(svc=service):
        return ttnn.experimental.deepseek_prefill.inbound_socket_service_sync(
            svc, metadata_size_bytes=_MD_BYTES, tokens_out=tokens_out, metadata_out=metadata_out
        )

    # Compile run (populates the program cache so capture only records).
    sent = _push(service, sp_factor, isl_per_chip, vocab, 900)
    out = call()
    _check_tokens(out[0], sent, "warmup")
    _check_meta(out[1], 900, isl_per_chip, "warmup")
    ttnn.synchronize_device(mesh_device)

    logger.info("capturing trace")
    pre = mesh_device.num_program_cache_entries()
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced = call()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    logger.info(
        f"trace captured (id={tid}); program cache delta during capture = "
        f"{mesh_device.num_program_cache_entries() - pre}"
    )

    assert traced[0].buffer_address() == tok_addr, "capture moved the tokens destination"
    assert traced[1].buffer_address() == md_addr, "capture moved the metadata destination"

    for i in range(4):
        slot = 1000 + i
        sent = _push(service, sp_factor, isl_per_chip, vocab, slot)
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        _check_tokens(tokens_out, sent, f"replay {i}")
        _check_meta(metadata_out, slot, isl_per_chip, f"replay {i}")
        assert tokens_out.buffer_address() == tok_addr
        logger.info(f"replay {i}: tokens + metadata correct")

    ttnn.release_trace(mesh_device, tid)
    logger.info("PASS: inbound_socket_service_sync replays correctly from inside a ttnn trace")
    service.barrier()
    del service


# ---------------------------------------------------------------------------
# 4. Negative control: the pre-change (eager, allocating) call inside a trace.
#    Diagnostic -- records WHICH way it fails so the fix can be justified.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [_params(trace_region_size=_TRACE_REGION)],
    indirect=["mesh_device", "device_params"],
)
def test_h2d_inbound_in_trace_without_persistent_dest_is_unsafe(mesh_device):
    sp_factor = mesh_device.shape[0]
    isl_per_chip = 512
    vocab = 4096
    service = _make_service(mesh_device, isl_per_chip)

    def call(svc=service):  # by-value capture; see the note on the same idiom above
        return ttnn.experimental.deepseek_prefill.inbound_socket_service_sync(svc, metadata_size_bytes=_MD_BYTES)

    sent = _push(service, sp_factor, isl_per_chip, vocab, 800)
    warm = call()
    _check_tokens(warm[0], sent, "eager warmup")
    warm_addr = warm[0].buffer_address()
    ttnn.synchronize_device(mesh_device)

    verdict = None
    try:
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        traced = call()
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.synchronize_device(mesh_device)
    except Exception as e:  # noqa: BLE001
        verdict = f"REJECTED at capture: {str(e).splitlines()[0][:300]}"
        logger.info(verdict)
        service.barrier()
        del service
        pytest.skip(verdict)

    captured_addr = traced[0].buffer_address()
    logger.info(f"eager capture succeeded; warmup tokens @0x{warm_addr:x}, captured-output tokens @0x{captured_addr:x}")

    # Free the captured output the way an eager caller would (it is a fresh tensor
    # per call), then replay: the trace still writes to the now-dangling address.
    ttnn.deallocate(traced[0])
    ttnn.deallocate(traced[1])
    fresh = ttnn.allocate_tensor_on_device(service.get_per_shard_spec(), mesh_device)
    logger.info(f"after deallocate, a fresh same-spec tensor landed @0x{fresh.buffer_address():x}")

    sent = _push(service, sp_factor, isl_per_chip, vocab, 801)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)

    clobbered = fresh.buffer_address() == captured_addr
    logger.info(
        f"VERDICT eager-in-trace: capture ALLOWED; replay writes to the capture-time address "
        f"0x{captured_addr:x} forever. Reallocation collides with it: {clobbered}"
    )
    ttnn.release_trace(mesh_device, tid)
    service.barrier()
    del service


# ---------------------------------------------------------------------------
# 5. Guard rails on the new parameters.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mesh_device, device_params", [_params()], indirect=["mesh_device", "device_params"])
def test_h2d_persistent_dest_validation(mesh_device, expect_error):
    sp_factor = mesh_device.shape[0]
    isl_per_chip = 512
    vocab = 4096
    service = _make_service(mesh_device, isl_per_chip)
    good_tokens = ttnn.allocate_tensor_on_device(service.get_per_shard_spec(), mesh_device)
    md_spec = ttnn.TensorSpec(
        shape=ttnn.Shape([1, 1, 1, _MD_BYTES // 4]),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        buffer_type=ttnn.BufferType.DRAM,
    )
    good_md = ttnn.allocate_tensor_on_device(md_spec, mesh_device)

    # A short tokens_out must be rejected: the kernel takes its page count from the
    # backing but addresses through tokens_out's accessor -> writes past the end.
    short = ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec(
            shape=ttnn.Shape([1, 1, isl_per_chip // 2]),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            buffer_type=ttnn.BufferType.DRAM,
        ),
        mesh_device,
    )
    _push(service, sp_factor, isl_per_chip, vocab, 0)
    with expect_error(RuntimeError, "tokens_out"):
        ttnn.experimental.deepseek_prefill.inbound_socket_service_sync(
            service, metadata_size_bytes=_MD_BYTES, tokens_out=short, metadata_out=good_md
        )
    logger.info("short tokens_out rejected")
    # drain the pushed transfer so the service is clean for the next case
    ttnn.experimental.deepseek_prefill.inbound_socket_service_sync(
        service, metadata_size_bytes=_MD_BYTES, tokens_out=good_tokens, metadata_out=good_md
    )

    # metadata_out with the metadata path compiled out must be rejected.
    svc0 = _make_service(mesh_device, isl_per_chip, metadata_size_bytes=0)
    tokens0 = ttnn.allocate_tensor_on_device(svc0.get_per_shard_spec(), mesh_device)
    tokens = torch.randint(0, vocab, (sp_factor, 1, isl_per_chip), dtype=torch.int64)
    svc0.forward_to_tensor_bytes(tokens.to(torch.int32).contiguous().numpy())
    with expect_error(RuntimeError, "metadata_out"):
        ttnn.experimental.deepseek_prefill.inbound_socket_service_sync(
            svc0, metadata_size_bytes=0, tokens_out=tokens0, metadata_out=good_md
        )
    logger.info("metadata_out without metadata_size_bytes rejected")
    ttnn.experimental.deepseek_prefill.inbound_socket_service_sync(svc0, metadata_size_bytes=0, tokens_out=tokens0)
    svc0.barrier()
    del svc0

    service.barrier()
    del service
    # Both rejections above leave an exception whose traceback pins THIS frame, and the
    # frame/traceback cycle only the cyclic collector can break -- so without this the two
    # services above outlive the mesh_device fixture and their destructors log ~66 TT_FATAL
    # backtraces (device already closed) into a passing run's log. Collect while the device
    # is still open, so `del` above is what actually frees them.
    gc.collect()


# ===========================================================================
# D2D (#52456): the same inbound op on the receiver overload, plus the
# outbound sender op, both driven entirely from caller-owned tensors and both
# captured in a trace.
# ===========================================================================

_D2D_TOKENS_PER_ROW = 64
_D2D_HIDDEN = 7168
_D2D_WORKERS = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))
_D2D_FIFO = 16384
_D2D_MD_WORDS = [7, 128, 256, 1]


def _shard_mapper(mesh):
    return ttnn.create_mesh_mapper(
        mesh, ttnn.MeshMapperConfig(placements=[ttnn.PlacementShard(2), ttnn.PlacementShard(3)])
    )


def _replicate_mapper(mesh):
    return ttnn.create_mesh_mapper(
        mesh, ttnn.MeshMapperConfig(placements=[ttnn.PlacementReplicate(), ttnn.PlacementReplicate()])
    )


def _host(torch_u32, mapper):
    return ttnn.from_torch(torch_u32, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mapper)


def _d2d_pair(mesh_device):
    rows, cols = mesh_device.shape[0], mesh_device.shape[1]
    half = rows // 2
    sender_mesh = mesh_device.create_submesh(ttnn.MeshShape(half, cols), offset=ttnn.MeshCoordinate(0, 0))
    receiver_mesh = mesh_device.create_submesh(ttnn.MeshShape(half, cols), offset=ttnn.MeshCoordinate(half, 0))
    sender_mesh.enable_program_cache()
    receiver_mesh.enable_program_cache()
    g_tokens = half * _D2D_TOKENS_PER_ROW
    global_spec = ttnn.TensorSpec(
        shape=ttnn.Shape([1, 1, g_tokens, _D2D_HIDDEN]),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        buffer_type=ttnn.BufferType.DRAM,
    )
    sender, receiver = ttnn.D2DStreamService.create_pair(
        sender_mesh=sender_mesh,
        receiver_mesh=receiver_mesh,
        global_spec=global_spec,
        mapper=_shard_mapper(sender_mesh),
        fifo_size_bytes=_D2D_FIFO,
        sender_worker_cores=_D2D_WORKERS,
        receiver_worker_cores=_D2D_WORKERS,
        socket_buffer_type=ttnn.BufferType.L1,
        metadata_size_bytes=len(_D2D_MD_WORDS) * 4,
        share_fabric_links=False,  # OWN mode: no host lease in the loop
    )
    return sender, receiver, sender_mesh, receiver_mesh, g_tokens


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [_params(trace_region_size=_TRACE_REGION)],
    indirect=["mesh_device", "device_params"],
)
def test_d2d_inbound_and_outbound_in_trace(mesh_device):
    sender, receiver, sender_mesh, receiver_mesh, g_tokens = _d2d_pair(mesh_device)
    md_bytes = len(_D2D_MD_WORDS) * 4
    numel = g_tokens * _D2D_HIDDEN

    try:
        # ---- caller-owned tensors, allocated once, refreshed in place ----
        s_map, r_map = _shard_mapper(sender_mesh), _replicate_mapper(sender_mesh)
        dev_in = ttnn.from_torch(
            torch.zeros(1, 1, g_tokens, _D2D_HIDDEN, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=sender_mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=s_map,
        )
        dev_md = ttnn.from_torch(
            torch.zeros(1, 1, 1, len(_D2D_MD_WORDS), dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=sender_mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=r_map,
        )
        recv_tokens = ttnn.allocate_tensor_on_device(receiver.get_per_shard_spec(), receiver_mesh)
        recv_md = ttnn.allocate_tensor_on_device(
            ttnn.TensorSpec(
                shape=ttnn.Shape([1, 1, 1, md_bytes // 4]),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                buffer_type=ttnn.BufferType.DRAM,
            ),
            receiver_mesh,
        )
        addrs = (
            dev_in.buffer_address(),
            dev_md.buffer_address(),
            recv_tokens.buffer_address(),
            recv_md.buffer_address(),
        )
        logger.info("persistent d2d tensors: send_in=0x%x send_md=0x%x recv_tok=0x%x recv_md=0x%x" % addrs)

        def refresh(it):
            torch_in = (torch.arange(numel, dtype=torch.int32) + it * 100_000).reshape(1, 1, g_tokens, _D2D_HIDDEN)
            md = torch.tensor([w + it for w in _D2D_MD_WORDS], dtype=torch.int32).reshape(1, 1, 1, -1)
            ttnn.copy_host_to_device_tensor(_host(torch_in, s_map), dev_in)
            ttnn.copy_host_to_device_tensor(_host(md, r_map), dev_md)
            return [w + it for w in _D2D_MD_WORDS]

        def send(snd=sender):  # by-value capture; the endpoints are del'd in teardown below
            return ttnn.experimental.deepseek_prefill.outbound_socket_service_sync(snd, dev_in, metadata=dev_md)

        def recv(rcv=receiver):
            return ttnn.experimental.deepseek_prefill.inbound_socket_service_sync(
                rcv, metadata_size_bytes=md_bytes, tokens_out=recv_tokens, metadata_out=recv_md
            )

        def check(want_md, tag):
            for k, (a, b) in enumerate(zip(ttnn.get_device_tensors(dev_in), ttnn.get_device_tensors(recv_tokens))):
                assert torch.equal(
                    ttnn.to_torch(a).to(torch.int32), ttnn.to_torch(b).to(torch.int32)
                ), f"{tag}: shard {k} mismatch"
            for dm in ttnn.get_device_tensors(recv_md):
                got = ttnn.to_torch(dm).flatten().to(torch.int64).tolist()
                assert got == want_md, f"{tag}: metadata {got} != {want_md}"

        # ---- compile run (both ops), un-traced ----
        want = refresh(0)
        send()
        recv()
        ttnn.synchronize_device(sender_mesh)
        ttnn.synchronize_device(receiver_mesh)
        check(want, "warmup")
        logger.info("d2d warmup OK (persistent send input + send metadata + recv destinations)")

        # ---- capture: sender trace and receiver trace ----
        s_pre, r_pre = sender_mesh.num_program_cache_entries(), receiver_mesh.num_program_cache_entries()
        s_tid = ttnn.begin_trace_capture(sender_mesh, cq_id=0)
        send()
        ttnn.end_trace_capture(sender_mesh, s_tid, cq_id=0)
        r_tid = ttnn.begin_trace_capture(receiver_mesh, cq_id=0)
        recv()
        ttnn.end_trace_capture(receiver_mesh, r_tid, cq_id=0)
        ttnn.synchronize_device(sender_mesh)
        ttnn.synchronize_device(receiver_mesh)
        logger.info(
            f"captured sender trace {s_tid} (cache delta {sender_mesh.num_program_cache_entries() - s_pre}) and "
            f"receiver trace {r_tid} (cache delta {receiver_mesh.num_program_cache_entries() - r_pre})"
        )

        # ---- replay ----
        for it in range(1, 4):
            want = refresh(it)  # host writes stay OUTSIDE the capture window
            ttnn.execute_trace(sender_mesh, s_tid, cq_id=0, blocking=False)
            ttnn.execute_trace(receiver_mesh, r_tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(sender_mesh)
            ttnn.synchronize_device(receiver_mesh)
            check(want, f"replay {it}")
            assert (
                dev_in.buffer_address(),
                dev_md.buffer_address(),
                recv_tokens.buffer_address(),
                recv_md.buffer_address(),
            ) == addrs
            logger.info(f"d2d replay {it}: activation + metadata delivered correctly")

        ttnn.release_trace(sender_mesh, s_tid)
        ttnn.release_trace(receiver_mesh, r_tid)
        logger.info("PASS: D2D outbound push AND inbound drain both replay from inside ttnn traces")
    finally:
        del receiver
        del sender


# ---------------------------------------------------------------------------
# 7. Swapping the persistent destination between calls must re-target the write.
#
#    This is the contract the issues actually ask for -- "created outside the op
#    and just sent to it as a parameter" -- and it is the one property a
#    same-buffer-every-call test cannot see.  It lives on the program-cache HIT
#    path: resolve_bindings maps each Buffer* binding to the FIRST slot in
#    tensor_buffers holding it, and with a persistent destination that first slot
#    is the INPUT-region tensor_args entry, not the output.  apply_resolved_bindings
#    then patches from current_buffers[that index] on every dispatch, so handing in
#    a different buffer of the same spec must follow.  If the binding instead
#    resolved to the output slot, or were skipped as an "in-place alias", the
#    cached entry would keep writing into the FIRST destination the op ever saw and
#    the second one would silently stay stale -- data corruption with no error.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mesh_device, device_params", [_params()], indirect=["mesh_device", "device_params"])
def test_h2d_persistent_destination_swap_is_retargeted(mesh_device):
    sp_factor = mesh_device.shape[0]
    isl_per_chip = 512
    vocab = 4096
    service = _make_service(mesh_device, isl_per_chip)

    md_spec = ttnn.TensorSpec(
        shape=ttnn.Shape([1, 1, 1, _MD_BYTES // 4]),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        buffer_type=ttnn.BufferType.DRAM,
    )
    # Two independent destination pairs, identical specs, distinct addresses -- so the
    # program-cache key (which hashes specs, not addresses) is the SAME for both and
    # they are forced to share one cached entry.  That is the point: a shared entry is
    # exactly where a stale binding index would corrupt.
    pairs = [
        (
            ttnn.allocate_tensor_on_device(service.get_per_shard_spec(), mesh_device),
            ttnn.allocate_tensor_on_device(md_spec, mesh_device),
        )
        for _ in range(2)
    ]
    addrs = [(t.buffer_address(), m.buffer_address()) for t, m in pairs]
    assert addrs[0] != addrs[1], "the two destination pairs landed on the same addresses; test is vacuous"
    logger.info(f"dest A tokens @ 0x{addrs[0][0]:x} md @ 0x{addrs[0][1]:x}")
    logger.info(f"dest B tokens @ 0x{addrs[1][0]:x} md @ 0x{addrs[1][1]:x}")

    last_sent = [None, None]  # what each pair should be holding right now
    deltas = []
    # Alternate A,B,A,B.  After the first call every dispatch is a cache hit, which is
    # the path under test.
    for i in range(4):
        which = i % 2
        tok_out, md_out = pairs[which]
        sent = _push(service, sp_factor, isl_per_chip, vocab, i)
        pre = mesh_device.num_program_cache_entries()
        tt_tokens, tt_meta = ttnn.experimental.deepseek_prefill.inbound_socket_service_sync(
            service, metadata_size_bytes=_MD_BYTES, tokens_out=tok_out, metadata_out=md_out
        )
        deltas.append(mesh_device.num_program_cache_entries() - pre)
        assert tt_tokens.buffer_address() == addrs[which][0], f"iter {i}: op returned a different tokens buffer"

        # a) the destination we handed in this call holds THIS chunk.
        _check_tokens(tt_tokens, sent, f"iter {i} dest {which}")
        _check_meta(tt_meta, i, isl_per_chip, f"iter {i} dest {which}")
        last_sent[which] = (sent, i)

        # b) the OTHER destination must be untouched -- still holding whatever it last
        #    received.  A stale binding shows up here: the write lands in the wrong pair.
        other = 1 - which
        if last_sent[other] is not None:
            prev_sent, prev_slot = last_sent[other]
            other_tok, other_md = pairs[other]
            _check_tokens(other_tok, prev_sent, f"iter {i}: dest {other} was clobbered")
            _check_meta(other_md, prev_slot, isl_per_chip, f"iter {i}: dest {other} metadata was clobbered")

    assert deltas[0] >= 1 and all(d == 0 for d in deltas[1:]), (
        f"expected one cached entry shared by both destinations, got deltas {deltas} -- "
        "if every swap compiled its own entry the cache-hit retarget path was never exercised"
    )
    logger.info(f"PASS destination swap: both pairs retargeted correctly on cache hits, deltas {deltas}")
    service.barrier()
    del service


# ---------------------------------------------------------------------------
# 8. Can the drain share ONE trace with the on-device metadata bridge?
#
#    The traced forward reads slot_id and actual_end as two SEPARATE 1-element
#    device tensors (tt_prefill_block.py -> zero_padded_kv_cache), while the H2D
#    drain produces ONE 3-word record.  #52464 bridges the two ON DEVICE inside
#    the capture, with ttnn.slice into pre-allocated 1-element destinations
#    (TtPrefillRuntime._metadata_from_msg); the host readback + three
#    copy_host_to_device_tensor writes it replaced could not be captured at all.
#
#    This is the op-level guard for that bridge: drain and scatter captured as ONE
#    trace, every destination caller-owned.  The scattered scalars must track the
#    record written earlier in the SAME replay -- if a replay ever reads the
#    PREVIOUS push's scalars, it fails here rather than as a silently mis-slotted
#    KV write inside the model.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [_params(trace_region_size=_TRACE_REGION)],
    indirect=["mesh_device", "device_params"],
)
def test_h2d_drain_plus_on_device_metadata_scatter_in_trace(mesh_device):
    sp_factor = mesh_device.shape[0]
    isl_per_chip = 512
    vocab = 4096
    service = _make_service(mesh_device, isl_per_chip)

    def _alloc(shape):
        return ttnn.allocate_tensor_on_device(
            ttnn.TensorSpec(
                shape=ttnn.Shape(shape),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                buffer_type=ttnn.BufferType.DRAM,
            ),
            mesh_device,
        )

    tokens_out = ttnn.allocate_tensor_on_device(service.get_per_shard_spec(), mesh_device)
    record = _alloc([1, 1, 1, _MD_BYTES // 4])  # the drain's 3-word destination
    m0 = _alloc([1, 1, 1, 1])  # slot_id      -> zero_padded_kv_cache arg 1
    m2 = _alloc([1, 1, 1, 1])  # valid_global -> zero_padded_kv_cache arg 2
    addrs = (tokens_out.buffer_address(), record.buffer_address(), m0.buffer_address(), m2.buffer_address())

    def push(slot, end, svc=service):  # by-value capture; see the note on the same idiom above
        tokens = torch.randint(0, vocab, (sp_factor, 1, isl_per_chip), dtype=torch.int64)
        svc.forward_to_tensor_bytes(
            tokens.to(torch.int32).contiguous().numpy(), metadata=struct.pack("<III", slot, 0, end)
        )
        return tokens

    def drain_and_scatter(svc=service):
        ttnn.experimental.deepseek_prefill.inbound_socket_service_sync(
            svc, metadata_size_bytes=_MD_BYTES, tokens_out=tokens_out, metadata_out=record
        )
        # The on-device bridge under test: record[0] -> m0, record[2] -> m2.
        ttnn.slice(record, [0, 0, 0, 0], [1, 1, 1, 1], output_tensor=m0)
        ttnn.slice(record, [0, 0, 0, 2], [1, 1, 1, 3], output_tensor=m2)

    def scalar(t):
        return int(ttnn.to_torch(ttnn.get_device_tensors(t)[0]).flatten()[0])

    # Warm/compile outside any capture.
    sent = push(700, 111)
    drain_and_scatter()
    ttnn.synchronize_device(mesh_device)
    _check_tokens(tokens_out, sent, "warmup")
    assert (scalar(m0), scalar(m2)) == (700, 111), f"warmup scatter wrong: {scalar(m0)},{scalar(m2)}"
    logger.info("warmup: on-device scatter reproduces record[0] and record[2]")

    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    drain_and_scatter()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    assert (
        tokens_out.buffer_address(),
        record.buffer_address(),
        m0.buffer_address(),
        m2.buffer_address(),
    ) == addrs, "capture moved a destination buffer"
    logger.info(f"captured drain+scatter as one trace (id={tid})")

    # The crux: each replay's scalars must be THIS push's, not the previous one's.
    for i in range(4):
        slot, end = 800 + i, 400 + 7 * i
        sent = push(slot, end)
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        _check_tokens(tokens_out, sent, f"replay {i}")
        got = (scalar(m0), scalar(m2))
        assert got == (slot, end), f"replay {i}: scatter is STALE/wrong -- got {got}, want {(slot, end)}"
        logger.info(f"replay {i}: tokens OK, scattered scalars = {got} (fresh)")

    ttnn.release_trace(mesh_device, tid)
    logger.info("PASS: H2D drain + on-device metadata scatter replay correctly from ONE trace")
    service.barrier()
    del service
