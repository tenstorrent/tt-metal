"""Host-side CB address plan for the megakernel program (spec §20, v12cb / MK_V12_CB).

Why: §20 turns the engine's program-internal L1 tensors into CBs on their own role's cores, so they leave no persistent
L1 footprint. Peers still address some of them remotely: STREAM writes the owners' pbuf, and the owners and the AR write
the STREAM K-row x (v_xk / v_xpk). So the host needs each CB's L1 address before the program exists.

``plan(cbs, base)`` replays tt-metal's ProgramImpl::allocate_circular_buffers (tt_metal/impl/program/program.cpp) over a
ProgramDescriptor's CB list, in list order:
  * one CircularBufferAllocator per distinct CoreRange, created in list order (globally allocated CBs create
    allocators too, but take no space);
  * address = align64(max(base, region end of the allocator whose range EQUALS each of the CB's ranges));
  * then every allocator intersecting one of the CB's ranges is marked [addr, addr + size), except an intersecting
    allocator with a different range that already ends above addr.
One descriptor has ONE address on all of its cores. ``base`` is the persistent-L1 high-water mark (the L1 unreserved
base, 112000 on BH today). The device check ``verify`` runs a probe program with the SAME CB list and compares every
core's get_write_ptr(id) with the plan, so a wrong base or a tt-metal allocator change fails loudly instead of
corrupting L1.
"""
import os

ALIGN = 64  # BH DRAM alignment (allocate_circular_buffers aligns every CB address to it)
CB_BASE = int(os.environ.get("MKA_L1_CB_BASE", "112000"))


def _rng(r):
    return (r.start.x, r.start.y, r.end.x, r.end.y)


def _inter(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


# Tensor-backed (globally allocated) descriptors: CBDescriptor.buffer is not exposed to Python, so track() wraps
# ttnn.cb_descriptor_from_sharded_tensor and records each such descriptor's key (core ranges, ids). Size is left out
# (callers shrink total_size afterwards, e.g. the AR's RED / OUT views). Within one program a (core, id) pair can only
# exist once, so the key is unique; a static descriptor re-created from a tracked one (the AR's RS slot CB) is
# unmarked with mark_static(). verify() catches any misclassification on device.
_GKEY = set()


def _key(d):
    return (tuple(_rng(r) for r in d.core_ranges.ranges()), tuple(f.buffer_index for f in d.format_descriptors))


def track(ttnn):
    """Install once (before any engine CB descriptor is built): record every tensor-backed CB descriptor."""
    f = ttnn.cb_descriptor_from_sharded_tensor
    if getattr(f, "_cbplan", False):
        return

    def w(*a, **k):
        d = f(*a, **k)
        _GKEY.add(_key(d))
        return d

    w._cbplan = True
    ttnn.cb_descriptor_from_sharded_tensor = w


def mark_static(d):
    _GKEY.discard(_key(d))
    return d


def _global(d):
    return _key(d) in _GKEY


def plan(cbs, base=CB_BASE):
    """-> list of (address or None for globally allocated, [ranges], size, [ids]) in list order."""
    allocs = []  # [range, regions [(start, end)]] in creation order
    for d in cbs:
        for r in map(_rng, d.core_ranges.ranges()):
            if not any(a[0] == r for a in allocs):
                allocs.append([r, []])
    out = []
    for d in cbs:
        rs = [_rng(r) for r in d.core_ranges.ranges()]
        ids = [f.buffer_index for f in d.format_descriptors]
        if _global(d):
            out.append((None, rs, d.total_size, ids))
            continue
        addr = base
        for r in rs:
            for a in allocs:
                if a[0] == r:
                    addr = max(addr, a[1][-1][1] if a[1] else 0)
                    break
        addr = -(-addr // ALIGN) * ALIGN
        for r in rs:
            for a in allocs:
                if not _inter(a[0], r):
                    continue
                end = a[1][-1][1] if a[1] else 0
                if a[0] != r and addr < end:
                    continue
                if not a[1]:
                    a[1].append((base, base))
                if addr < a[1][-1][1]:
                    raise RuntimeError(f"cbplan: CB {ids} at {addr} below region end {a[1][-1][1]} of allocator {a[0]}")
                if addr == a[1][-1][1]:
                    a[1][-1] = (a[1][-1][0], addr + d.total_size)
                else:
                    a[1].append((addr, addr + d.total_size))
        out.append((addr, rs, d.total_size, ids))
    return out


def address_of(p, cb_id, core=None):
    """Planned address of CB id ``cb_id`` (on logical ``core`` (x, y) if the id sits on several disjoint descriptors)."""
    hits = [e for e in p if cb_id in e[3] and (core is None or any(r[0] <= core[0] <= r[2] and r[1] <= core[1] <= r[3] for r in e[1]))]
    assert len(hits) == 1, f"cbplan: CB {cb_id} core {core}: {len(hits)} descriptors"
    assert hits[0][0] is not None, f"cbplan: CB {cb_id} is globally allocated"
    return hits[0][0]


def per_core(p):
    """{(x, y): {id: address}} for every non-global CB."""
    m = {}
    for addr, rs, _, ids in p:
        if addr is None:
            continue
        for r in rs:
            for x in range(r[0], r[2] + 1):
                for y in range(r[1], r[3] + 1):
                    for i in ids:
                        m.setdefault((x, y), {})[i] = addr
    return m


def end_per_core(p):
    """{(x, y): CB region end} (non-global CBs)."""
    m = {}
    for addr, rs, size, _ in p:
        if addr is None:
            continue
        for r in rs:
            for x in range(r[0], r[2] + 1):
                for y in range(r[1], r[3] + 1):
                    m[(x, y)] = max(m.get((x, y), 0), addr + size)
    return m


def verify(dev, desc, base=CB_BASE, tag=""):
    """Device check (MK_V12_CBCHK): a probe program with ``desc``'s CB list writes get_write_ptr(id) of every planned
    (non-global) CB on every core; any difference from plan(desc.cbs) raises. Runs on every device of ``dev``."""
    import torch
    import ttnn

    p = plan(list(desc.cbs), base)
    pc = per_core(p)
    cores = sorted(pc)
    W = 64
    rep = ttnn.ReplicateTensorToMesh(dev) if hasattr(dev, "get_num_devices") else None
    out = ttnn.from_torch(torch.zeros(len(cores), W, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                          device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG, **({"mesh_mapper": rep} if rep else {}))
    rt = ttnn.RuntimeArgs()
    for i, c in enumerate(cores):
        ids = sorted(pc[c])
        assert len(ids) <= W - 2
        rt[c[0]][c[1]] = [out.buffer_address(), i, len(ids)] + ids
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c in cores])
    k = ttnn.KernelDescriptor(kernel_source=os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", "mk_cbplan_probe.cpp"),
                              core_ranges=crs, compile_time_args=ttnn.TensorAccessorArgs(out).get_compile_time_args(), runtime_args=rt,
                              config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0))
    ttnn.generic_op([out, out], ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=list(desc.cbs)))   # generic_op needs >= 2 io tensors
    ttnn.synchronize_device(dev)
    dts = ttnn.get_device_tensors(out) if rep else [out]
    bad = []
    for d, dt in enumerate(dts):
        got = (ttnn.to_torch(dt).to(torch.int64) & 0xFFFFFFFF).reshape(len(cores), W)
        for i, c in enumerate(cores):
            ids = sorted(pc[c])
            if int(got[i, 0]) != 0x43424B4D:
                bad.append((d, c, "probe did not run"))
                continue
            for j, cid in enumerate(ids):
                if int(got[i, 1 + j]) != pc[c][cid]:
                    bad.append((d, c, cid, int(got[i, 1 + j]), pc[c][cid]))
    ttnn.deallocate(out)
    n = sum(len(v) for v in pc.values())
    if bad:
        raise RuntimeError(f"[cbplan]{tag} FAIL: {len(bad)} mismatches of {n * len(dts)} (core, CB) pairs, e.g. {bad[:6]}")
    print(f"[cbplan]{tag} PASS: {len(dts)} device(s) x {len(cores)} cores, {n} (core, CB) addresses match the plan (base {base})", flush=True)
    return p
