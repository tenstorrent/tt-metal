# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TP=4 helpers for NemotronH-30B on QB (4× Blackhole).

All component forwards accept/return CPU torch tensors and handle
device sharding internally via these helpers.

Topology: FABRIC_1D + Topology.Linear (4-chip linear chain on QB).
"""

import weakref

import torch

import ttnn

TP = 4
FABRIC = ttnn.FabricConfig.FABRIC_1D
TOPOLOGY = ttnn.Topology.Linear

_R = ttnn.ReplicateTensorToMesh
_S = ttnn.ShardTensorToMesh
_C = ttnn.ConcatMeshToTensor

# ---------------------------------------------------------------------------
# Device weight cache — avoids re-uploading the same weight on every forward.
#
# Key: (id(source_tensor), shard_dim_or_None, layout, dtype, id(mesh))
# Value: (weakref.ref(source_tensor), device_tensor)
#
# Correctness: the weakref tracks the SOURCE tensor's lifetime.
#   - Weights (held by WeightCache._shards): their weakref stays alive → cache hits.
#   - Activations (transient): freed after the forward → weakref() returns None
#     on the next call → stale entry evicted → re-upload. Safe even if Python
#     reuses the same id() for a new tensor after GC.
# ---------------------------------------------------------------------------
_DEVICE_CACHE: dict = {}

# Cache for derived (unsqueezed/converted/sliced) weight tensors.
# Key: (arbitrary_hashable_key, layout, dtype, id(mesh)).
# Use _rep_keyed() when the CPU source is re-derived each call (unsqueeze, cast,
# slice) so id(derived) is unstable — caller provides a stable key instead.
_DERIVED_CACHE: dict = {}

# DRAM defect guards: tensors kept alive to permanently occupy physically-defective
# DRAM pages on device 2 so the allocator never re-uses them.
_DRAM_DEFECT_GUARDS: list = []

# Per-request DRAM prefill block: allocated at switch_mode("prefill") and freed at
# the START of the NEXT switch_mode("prefill") call.  This blindly occupies the
# first N × sizeof(shape) bytes of the free list, ensuring V_bad (the defective
# page that enters the free list after each request's decode frees intermediates)
# is never available to prefill chunk allocations.
#
# Why "blind" blocking instead of write-then-readback detection:
#   H2D DMA writes to V_bad silently fail, but the PCIe write-combining buffer
#   caches the written value.  D2H DMA reads return the cached value, NOT the
#   actual DRAM cell value.  This makes readback-based detection unreliable
#   (probe sees 1.0s everywhere, no mismatch detected, V_bad is freed back to pool).
_DRAM_PREFILL_BLOCK: list = []
# Device-2 addresses from the PREVIOUS probe call, used to detect which probe
# page was freed and not recaptured by the next probe — that missing page is V_bad.
_PROBE_PREV_ADDRS_D2: set = set()
# Set to True once V_bad has been permanently guarded via _DRAM_DEFECT_GUARDS.
# probe_dram_defect_for_shape skips the V_bad scan after this is set.
_VBAD_GUARDED: bool = False


def clear_device_weight_cache() -> None:
    _DEVICE_CACHE.clear()
    _DERIVED_CACHE.clear()


def probe_dram_defect_for_shape(
    mesh_device: ttnn.MeshDevice,
    shape: list,
    dtype: ttnn.DataType = ttnn.bfloat16,
    layout: ttnn.Layout = ttnn.TILE_LAYOUT,
    n_probes: int = 0,
) -> int:
    """Blind-block n_probes consecutive DRAM allocations to protect against defective pages.

    Called at switch_mode("prefill") before each request.  Frees the previous call's
    block (so DRAM is not permanently wasted), then allocates n_probes tensors at the
    current free-list head and keeps them alive in _DRAM_PREFILL_BLOCK for the
    duration of the prefill.

    MUST use ttnn.from_torch (H2D DMA via PCIe), NOT ttnn.zeros:
        ttnn.zeros dispatches a fill kernel that does NOC writes to every allocated page.
        NOC writes to the defective page(s) on device 2 hang the device.
        H2D DMA writes to defective pages are silently discarded by the PCIe bus —
        no hang, and the Python-side DRAM allocator still marks the address as occupied.

    Why readback-based detection is NOT used:
        The PCIe write-combining buffer caches the written value, so D2H DMA reads
        return the cached value (not the actual corrupted DRAM cell value).  Readback
        always appears correct, making detection unreliable.

    Why n_probes=0 (default):
        The probe approach was investigated and found ineffective: V_bad is at
        device-2 address ~0x90a8cb80, above probe[-1]=0x907fc880 (n_probes=4200).
        The probe never covered V_bad.  The hang is caused by chunk-level B_c/C_c
        destinations landing at V_bad for request 2 due to the different free-list
        state after request 1 frees its intermediates.
        The probe also consumes 2.1 GB/device (n_probes=4200 × 512 KB each),
        causing OOM at ISL≥256K.  Setting n_probes=0 recovers that memory.
        The correct fix is a permanent guard in _DRAM_DEFECT_GUARDS that occupies
        V_bad's DRAM page (0x90a8cb80) so it is never handed to a chunk destination.

    Returns n_probes (all blocked this call).
    """
    import logging as _logging

    log = _logging.getLogger(__name__)
    global _DRAM_PREFILL_BLOCK, _PROBE_PREV_ADDRS_D2, _VBAD_GUARDED

    # Collect device-2 addresses of the existing block BEFORE freeing, so we can
    # compare with the new probe and identify pages that were freed but not recaptured
    # (those are candidates for V_bad).
    prev_addrs_d2 = set()
    if _DRAM_PREFILL_BLOCK:
        try:
            for t in _DRAM_PREFILL_BLOCK:
                dev_list = ttnn.get_device_tensors(t)
                if len(dev_list) > 2:
                    prev_addrs_d2.add(dev_list[2].buffer_address())
        except Exception as _exc:
            log.warning("probe_compare: prev dev-2 addr collection failed: %s", _exc)
            prev_addrs_d2.clear()

    # Free previous block first (round-trip: free → reallocate keeps DRAM usage flat).
    for t in _DRAM_PREFILL_BLOCK:
        t.deallocate(True)
    _DRAM_PREFILL_BLOCK.clear()

    numel = 1
    for s in shape:
        numel *= s
    zero_cpu = torch.zeros(numel, dtype=torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32).reshape(shape)
    replicate_mapper = _R(mesh_device)  # create once, reuse across iterations

    for _ in range(n_probes):
        _DRAM_PREFILL_BLOCK.append(
            ttnn.from_torch(
                zero_cpu,
                dtype=dtype,
                layout=layout,
                device=mesh_device,
                mesh_mapper=replicate_mapper,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )

    # Collect device-2 addresses of new probe for comparison and future diff.
    new_addrs_d2 = set()
    try:
        for t in _DRAM_PREFILL_BLOCK:
            dev_list = ttnn.get_device_tensors(t)
            if len(dev_list) > 2:
                new_addrs_d2.add(dev_list[2].buffer_address())
    except Exception as _exc:
        log.warning("probe_compare: new dev-2 addr collection failed: %s", _exc)
        new_addrs_d2.clear()

    # Diff: pages in prev probe but NOT in new probe → freed and not recaptured → V_bad candidates.
    if prev_addrs_d2 and new_addrs_d2:
        freed_not_recaptured = prev_addrs_d2 - new_addrs_d2
        newly_captured = new_addrs_d2 - prev_addrs_d2
        if freed_not_recaptured or newly_captured:
            log.info(
                "probe_compare dev2: freed_not_recaptured(%d)=%s  newly_captured(%d)=%s",
                len(freed_not_recaptured),
                sorted(hex(a) for a in freed_not_recaptured),
                len(newly_captured),
                sorted(hex(a) for a in newly_captured),
            )
        else:
            log.info("probe_compare dev2: identical (%d addrs unchanged)", len(new_addrs_d2))
    _PROBE_PREV_ADDRS_D2 = new_addrs_d2

    # V_bad permanent guard: scan probes for device-2 address range containing
    # V_bad (0x90a8cb80) and move that probe to _DRAM_DEFECT_GUARDS permanently.
    # This runs on EVERY switch_mode("prefill") call until V_bad is found.
    # V_bad only enters the free list after the first decode's intermediates are
    # freed; the scan no-ops harmlessly until that point.
    if not _VBAD_GUARDED and _DRAM_PREFILL_BLOCK:
        _V_BAD_D2 = 0x90A8CB80
        probe_bytes = numel * 2  # BF16
        vbad_probe = None
        try:
            for t in _DRAM_PREFILL_BLOCK:
                dev_list = ttnn.get_device_tensors(t)
                if len(dev_list) > 2:
                    d2_addr = dev_list[2].buffer_address()
                    if d2_addr <= _V_BAD_D2 < d2_addr + probe_bytes:
                        vbad_probe = t
                        break
        except Exception as _exc:
            log.warning("vbad_scan: addr collection failed: %s", _exc)
        if vbad_probe is not None:
            # Keep only the V_bad probe; free all others immediately.
            for t in _DRAM_PREFILL_BLOCK:
                if t is not vbad_probe:
                    t.deallocate(True)
            _DRAM_PREFILL_BLOCK.clear()
            _DRAM_DEFECT_GUARDS.append(vbad_probe)
            _VBAD_GUARDED = True
            try:
                d2_addr = ttnn.get_device_tensors(vbad_probe)[2].buffer_address()
                log.info(
                    "vbad_scan: V_bad permanently guarded " "d2=0x%08x probe_size=%dKB", d2_addr, probe_bytes // 1024
                )
            except Exception:
                log.info("vbad_scan: V_bad permanently guarded")

    size_mb = n_probes * (numel * 2) // (1024 * 1024)
    try:
        addr_first = _DRAM_PREFILL_BLOCK[0].buffer_address()
        addr_last = _DRAM_PREFILL_BLOCK[-1].buffer_address()
        log.info(
            "probe_dram_defect_for_shape: blocked %d pages (%d MB) " "addr [0x%08x … 0x%08x] (shape=%s)",
            n_probes,
            size_mb,
            addr_first,
            addr_last,
            shape,
        )
    except Exception:
        log.info(
            "probe_dram_defect_for_shape: blocked %d pages (%d MB) (shape=%s)",
            n_probes,
            size_mb,
            shape,
        )
    return n_probes


def open_device_tp4() -> ttnn.MeshDevice:
    """Open the 4-chip QB mesh with FABRIC_1D fabric."""
    ttnn.set_fabric_config(FABRIC)
    return ttnn.open_mesh_device(ttnn.MeshShape(1, TP), physical_device_ids=list(range(TP)))


def close_device_tp4(mesh: ttnn.MeshDevice) -> None:
    clear_device_weight_cache()
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


# ---------------------------------------------------------------------------
# from_torch helpers
# ---------------------------------------------------------------------------


def _upload(t, mesh, shard_dim, layout, dtype):
    """Upload tensor to mesh, caching by source-tensor lifetime.

    A cache entry is valid only while the source tensor is alive.  Weight
    tensors (held by WeightCache) live for the model's lifetime → always hit
    after the first call.  Activation tensors die after each forward → their
    weakref() returns None → stale entry evicted → safe re-upload.
    """
    key = (id(t), shard_dim, layout, dtype, id(mesh))
    if key in _DEVICE_CACHE:
        weak_src, dev_tensor = _DEVICE_CACHE[key]
        if weak_src() is not None:  # source still alive → valid hit
            return dev_tensor
        del _DEVICE_CACHE[key]  # source GC'd → evict stale entry

    mapper = _R(mesh) if shard_dim is None else _S(mesh, dim=shard_dim)
    dev_tensor = ttnn.from_torch(
        t.bfloat16() if dtype == ttnn.bfloat16 else t,
        dtype=dtype,
        layout=layout,
        device=mesh,
        mesh_mapper=mapper,
    )
    try:
        _DEVICE_CACHE[key] = (weakref.ref(t), dev_tensor)
    except TypeError:
        pass  # not all types support weakrefs; skip caching for this tensor
    return dev_tensor


def _rep(t, mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    """Load tensor replicated on all TP devices (cached after first upload)."""
    return _upload(t, mesh, None, layout, dtype)


def _rep_keyed(stable_key, derived_cpu, mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=None):
    """Replicate derived_cpu; cached under stable_key, NOT id(derived_cpu).

    Use when derived_cpu is re-computed each forward (e.g. t.unsqueeze(0),
    t.bfloat16().unsqueeze(0)) so its id() is unstable.  stable_key must be
    a hashable value that uniquely identifies this particular derived tensor
    for the lifetime of the model (e.g. (id(parent_weight), 'tag')).

    memory_config: override the destination memory config (default DRAM).
    The DRAM defect fallback inside this function retries in DRAM (not L1) to
    avoid persistent L1 tensors conflicting with tilize_with_val_padding and
    rms_norm static circular buffer regions.
    """
    mc = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
    key = (stable_key, layout, dtype, id(mesh), mc)
    if key in _DERIVED_CACHE:
        return _DERIVED_CACHE[key]

    cpu_data = derived_cpu.bfloat16() if dtype == ttnn.bfloat16 and derived_cpu.dtype != torch.bfloat16 else derived_cpu
    dev = ttnn.from_torch(cpu_data, dtype=dtype, layout=layout, device=mesh, mesh_mapper=_R(mesh), memory_config=mc)

    # Detect silent DRAM write failures (hardware defect at certain low DRAM pages).
    # Small tensors (≤32768 elements) are most vulnerable; large weight matrices land
    # at higher DRAM addresses that are typically fault-free.
    if mc == ttnn.DRAM_MEMORY_CONFIG and derived_cpu.numel() <= 32768:
        expected_norm = cpu_data.float().norm().item()
        if expected_norm > 1e-4:
            readback = ttnn.to_torch(dev, mesh_composer=_C(mesh, dim=0))
            if any(abs(readback[i].float().norm().item() - expected_norm) / expected_norm > 0.02 for i in range(TP)):
                # Corrupt — DRAM defect on device 2 at this allocation address.
                # Keep the defective tensor as a guard: freeing it returns the bad page to
                # the DRAM allocator, which could then assign it to an inference intermediate
                # and cause a NOC read timeout (device hang).
                #
                # Retry in DRAM at the NEXT available page: the guard permanently occupies
                # the defective page so the allocator advances past it.
                #
                # Do NOT fall back to L1: persistent L1 user tensors whose addresses fall
                # inside the static-CB region of subsequent tilize_with_val_padding or
                # rms_norm kernels (e.g. [0, 119808]) cause TT_THROW dispatch failures.
                _DRAM_DEFECT_GUARDS.append(dev)
                dev = ttnn.from_torch(
                    cpu_data,
                    dtype=dtype,
                    layout=layout,
                    device=mesh,
                    mesh_mapper=_R(mesh),
                    memory_config=mc,
                )

    _DERIVED_CACHE[key] = dev
    return dev


def _col(t, mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    """Column-parallel: shard weight along output-feature dim (dim=0), cached."""
    return _upload(t, mesh, 0, layout, dtype)


def _row(t, mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    """Row-parallel: shard weight along input-feature dim (dim=1), cached."""
    return _upload(t, mesh, 1, layout, dtype)


def _kv_gqa(t, mesh, n_q=32, n_kv=2, tp=TP, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    """GQA-aware KV weight shard for n_kv < TP.

    Each device i receives the single KV head its Q heads map to:
        kv_idx = (i * q_per_device * n_kv) // n_q
    Devices 0,1 → head 0; devices 2,3 → head 1 (for n_q=32, n_kv=2, TP=4).
    Result per device: [head_dim, hidden] (n_kv_per_device = 1).
    Cached in _DEVICE_CACHE with a weakref to the source so stale entries
    are evicted when the caller's weight tensor goes out of scope (e.g. in tests).
    """
    key = (id(t), "kv_gqa", layout, dtype, id(mesh))
    if key in _DEVICE_CACHE:
        weak_src, dev_tensor = _DEVICE_CACHE[key]
        if weak_src() is not None:
            return dev_tensor
        del _DEVICE_CACHE[key]

    head_dim = t.shape[0] // n_kv  # e.g. 256//2 = 128
    q_per_dev = n_q // tp
    chunks = [
        t[(i * q_per_dev * n_kv // n_q) * head_dim : ((i * q_per_dev * n_kv // n_q) + 1) * head_dim] for i in range(tp)
    ]
    t_sharded = torch.cat(chunks, dim=0)  # [tp * head_dim, hidden]: [head0, head0, head1, head1]
    cpu_data = t_sharded.bfloat16() if dtype == ttnn.bfloat16 else t_sharded
    dev = ttnn.from_torch(cpu_data, dtype=dtype, layout=layout, device=mesh, mesh_mapper=_S(mesh, dim=0))
    try:
        _DEVICE_CACHE[key] = (weakref.ref(t), dev)
    except TypeError:
        pass
    return dev


def _shard_act(t, mesh, dim, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    """Shard an activation tensor along `dim` — NOT cached (activations change)."""
    return ttnn.from_torch(
        t.bfloat16(),
        dtype=dtype,
        layout=layout,
        device=mesh,
        mesh_mapper=_S(mesh, dim=dim),
    )


# ---------------------------------------------------------------------------
# to_torch helpers
# ---------------------------------------------------------------------------


def _host_rep(t_tt, mesh, n):
    """Bring a replicated device tensor to host, returning first `n` rows (dim-0 slice).

    With MeshShape(1,4) and ReplicateTensorToMesh, ConcatMeshToTensor(dim=0)
    concatenates the 4 identical copies along dim=0 → [4n, ...].  Take [:n].
    """
    full = ttnn.to_torch(t_tt, mesh_composer=_C(mesh, dim=0))
    return full[:n].bfloat16()


def _host_sharded(t_tt, mesh, concat_dim):
    """Bring a column-parallel (sharded) device tensor to host by concatenating along `concat_dim`."""
    return ttnn.to_torch(t_tt, mesh_composer=_C(mesh, dim=concat_dim)).bfloat16()


# ---------------------------------------------------------------------------
# CCL wrappers
# ---------------------------------------------------------------------------


# Max sequence rows per all_reduce call.  all_reduce uses NOC write-with-ack
# semantics; if the output DRAM buffer lands on device-2's defective page the
# ack never returns → hang.  88 MB (16384 × 2688 × BF16) has been empirically
# safe; 176 MB (32768 rows) hangs.  Chunk along dim=-2 to stay under the limit.
_ALL_REDUCE_S_CHUNK = 16384


def all_reduce(t_tt):
    """Element-wise sum across all TP devices; result identical on all devices.

    Automatically chunks along dim=-2 (sequence) for large tensors to avoid
    NOC write-ack hang on device-2's defective DRAM page.
    """
    ndim = len(t_tt.shape)
    S = t_tt.shape[ndim - 2]
    if S <= _ALL_REDUCE_S_CHUNK:
        return ttnn.all_reduce(t_tt, topology=TOPOLOGY)
    H = t_tt.shape[ndim - 1]
    s0 = [0] * (ndim - 2)
    e0 = [t_tt.shape[i] for i in range(ndim - 2)]
    chunks = []
    for start in range(0, S, _ALL_REDUCE_S_CHUNK):
        end = min(start + _ALL_REDUCE_S_CHUNK, S)
        chunk = ttnn.slice(t_tt, s0 + [start, 0], e0 + [end, H])
        reduced = ttnn.all_reduce(chunk, topology=TOPOLOGY)
        chunk.deallocate(True)
        chunks.append(reduced)
    if len(chunks) == 1:
        return chunks[0]
    result = ttnn.concat(chunks, dim=ndim - 2)
    for c in chunks:
        c.deallocate(True)
    return result


def all_gather(t_tt, dim):
    """Gather sharded tensors from all TP devices along `dim`."""
    return ttnn.all_gather(t_tt, dim=dim, topology=TOPOLOGY)
