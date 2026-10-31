# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The reference fused head's per-chip programs (models/demos/moe_block/moe_block.py FusedHead on mstaletovic/moe-trim),
built on the mock Galaxy (no device, no execution), as JSON: for test_head_program_parity.py, which builds the C++ op's
programs on the same mock and compares them field by field. Run in the reference tree's environment:

    P=/data/mstaletovic/tt-metal-trimb; cd $P && TT_METAL_MOCK_CLUSTER_DESC_PATH=$P/tt_metal/third_party/umd/tests/\
cluster_descriptor_examples/blackhole_galaxy.yaml TT_METAL_HOME=$P LD_LIBRARY_PATH=$P/build_Release/lib \
    PYTHONPATH=<pyroot>:$P python <this file> --models glm53 --chunk 5120 [--out dir]

(The reference's default fabric tie rule - ties by the source's parity - is main's and the C++ op's.)

What is saved, per case (model, mesh 8 x 4 torus, chunk, accumulators): the plan (head_plan's layout), every chip's
kernels (source kind, the defines the host put in front of the source, core ranges, compile-time args, runtime args per
core, common runtime args, RISC / NoC or compute config), CBs (size, cores, formats, arena offset) and semaphores.
Addresses are symbolic: the case is built twice, the second time with L1 and DRAM taken first so every buffer moves;
an argument that differs between the two builds is an address, saved as [name, offset] of the buffer it falls in (the
head's arena, a persistent word tensor, a semaphore, a DRAM tensor). The reference programs' kernels are source
strings; the planner's substitutions (routing_stream.kernel_subs) and the route words' parameters are recorded so the
C++ op's RAP_* / RW_* defines can be checked against them."""

import argparse
import gc
import gzip
import json
import re
import sys
from pathlib import Path

import torch

import ttnn

# (the reference's programs are built and prepared but never run: no kernel compiles)
ttnn.bringup.prepare_program = lambda desc, io: desc
ttnn.bringup.run_prepared = lambda pp, io: None

MODELS = ("glm53", "k3", "k27", "glm53_flash", "dsv41_flash")
_DEF = re.compile(r"^#define (\w+)(?:\s+(.*))?$")


def _ranges(crs):
    return sorted([r.start.x, r.start.y, r.end.x, r.end.y] for r in crs.ranges())


def _defines(src):
    """The '#define NAME VALUE' lines of a source, in order (value None for a bare define)."""
    out = []
    for ln in src.splitlines():
        m = _DEF.match(ln.strip())
        if m:
            out.append([m.group(1), (m.group(2) or "").strip()])
    return out


def _config(cfg):
    if isinstance(cfg, ttnn.ComputeConfigDescriptor):
        return dict(
            type="compute",
            math_fidelity=str(cfg.math_fidelity).split(".")[-1],
            fp32_dest_acc_en=bool(cfg.fp32_dest_acc_en),
            dst_full_sync_en=bool(cfg.dst_full_sync_en),
            math_approx_mode=bool(cfg.math_approx_mode),
            unpack_to_dest_mode=[i for i, m in enumerate(cfg.unpack_to_dest_mode) if "Fp32" in str(m)],
        )
    return dict(type="dm", processor=str(cfg.processor).split(".")[-1], noc=str(cfg.noc).split(".")[-1])


def _rt(rt, crs):
    """The runtime args of every core of the kernel's ranges (a RuntimeArgs grows on indexing: walk the cores)."""
    out = {}
    for r in crs.ranges():
        for x in range(r.start.x, r.end.x + 1):
            for y in range(r.start.y, r.end.y + 1):
                a = list(rt[x][y])
                if a:
                    out[f"{x},{y}"] = a
    return out


def _fmt(f):
    """A CB format's data format name (None where this tree's binding cannot return it)."""
    try:
        return str(f.data_format).split(".")[-1]
    except TypeError:
        return None


def program_json(prog):
    ks = []
    for k in prog.kernels:
        ks.append(
            dict(
                source=k.kernel_source,
                core_ranges=_ranges(k.core_ranges),
                compile_time_args=list(k.compile_time_args),
                named_compile_time_args=[[n, v] for n, v in k.named_compile_time_args],
                defines=[[n, v] for n, v in k.defines],
                runtime_args=_rt(k.runtime_args, k.core_ranges),
                common_runtime_args=list(k.common_runtime_args),
                config=_config(k.config),
            )
        )
    cbs = []
    for cb in prog.cbs:
        cbs.append(
            dict(
                total_size=cb.total_size,
                core_ranges=_ranges(cb.core_ranges),
                formats=[[f.buffer_index, _fmt(f), f.page_size] for f in cb.format_descriptors],
                buffer=int(cb.buffer_address()) if cb.has_buffer() else None,
                address_offset=cb.address_offset,
            )
        )
    sems = [dict(id=s.id, core_ranges=_ranges(s.core_ranges), initial_value=s.initial_value) for s in prog.semaphores]
    return dict(kernels=ks, cbs=cbs, semaphores=sems)


def registry(fused):
    """[(name, address, bytes)] of every buffer the head's programs address (the second build moves them all)."""
    h = fused.head
    e = h.eng
    out = [("arena", e.base, e.arena_bytes)]
    from ttnn.bringup.combine_rf import persist_l1

    def t(name, ten, size=None):
        if ten is not None and ten.is_allocated():
            out.append((name, int(ten.buffer_address()), size or 1 << 20))

    def s(name, sem):
        out.append((name, persist_l1.address(sem), 16))

    t("mbox", getattr(h, "mbox", None), 1 << 16)
    t("ewords", getattr(e, "ewords", None), 1 << 12)
    for i, sem in enumerate(getattr(h, "arr_sems", [])):
        s(f"arr_sem{i}", sem)
    for i, sem in enumerate(h.xg.sems):
        s(f"xg_sem{i}", sem)
    if h.router is not None:
        r = h.router
        for i, sem in enumerate(r.sems):
            s(f"rt_sem{i}", sem)
        t("slot_words", r.slot_words, 1 << 12)
        t("sfree_words", r.sfree_words, 1 << 12)
    if h.routing is not None:
        ra = h.routing
        for k, sem in ra.sems.items():
            s(f"ra_{k}", sem)
        t("yhello", ra.yhello, 1 << 12)
        t("ra_slots", ra.slots, 1 << 30)
        t("ra_fslots", ra.fslots, 1 << 30)
        for k, v in ra.outs.items():
            t(f"ra_{k}", v, 1 << 30)
    return out


_L1_FREE = {}


def _pin_free_l1():
    """Both builds plan on the first one's free L1 (the second runs with the first's persistent words still in L1 and
    would otherwise plan another K step)."""
    from models.demos.moe_block.tt import se_engine3

    if "f" not in _L1_FREE:
        _L1_FREE["f"] = se_engine3.l1_free_bytes
    real = _L1_FREE["f"]

    def pinned(mesh):
        _L1_FREE.setdefault("v", real(mesh))
        return _L1_FREE["v"]

    se_engine3.l1_free_bytes = pinned


def build(mesh, name, chunk, A, perturb):
    from models.demos.moe_block.moe_block import MODELS as BM
    from models.demos.moe_block.moe_block import FusedHead, Geometry, MoEDtypes
    from models.demos.moe_block.tests.test_moe_block import random_weights

    model = {"k3": BM["kimi_k3"], "k27": BM["kimi_k27"]}.get(name) or BM[name]
    _pin_free_l1()
    if not perturb:
        _L1_FREE.pop("v", None)
    keep = []
    if perturb:  # (every later buffer moves: L1 on every core, DRAM)
        g = mesh.compute_with_storage_grid_size()
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])
        keep.append(
            ttnn.allocate_tensor_on_device(
                ttnn.Shape([g.x * g.y, 64]),
                ttnn.uint32,
                ttnn.ROW_MAJOR_LAYOUT,
                mesh,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(crs, [1, 64], ttnn.ShardOrientation.ROW_MAJOR),
                ),
            )
        )
        keep.append(
            ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, 4096, 4096]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
            )
        )
    g = Geometry.of(mesh, model, chunk, 2, False)
    dtypes = MoEDtypes()
    fit = FusedHead.plan(g, model, dtypes, mesh=mesh, accumulators=A)
    assert fit.ok, fit.why
    w = random_weights(model, 0)
    # (the planner's substitutions and the route words' parameters, as the head builds them)
    from models.demos.moe_block.tt import prep_stream, routing_stream

    subs, routes = {}, {}
    ks0 = routing_stream.RoutingStream.kernel_subs

    def ks(self, coord, ct0):
        d = ks0(self, coord, ct0)
        subs.setdefault(f"{coord[0]},{coord[1]}", []).append(dict(d))
        return d

    rs0 = prep_stream.route_src

    def rsrc(mesh_shape, me, **kw):
        routes[str(me)] = dict(
            mesh=list(mesh_shape), me=me, **{k: list(v) if isinstance(v, tuple) else v for k, v in kw.items()}
        )
        return rs0(mesh_shape, me, **kw)

    routing_stream.RoutingStream.kernel_subs = ks
    prep_stream.route_src = rsrc
    try:
        fused = FusedHead(mesh, g, model, w, [], deterministic=True, accumulators=A)
        H = model.emb_dim
        x = ttnn.from_torch(
            torch.zeros(1, 1, chunk, H),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(2, 3)),
        )
        # (FusedHead.__call__ without giving the arena back: the CBs bound to it are read below)
        ttnn.copy(x, fused.x)
        se = fused.head.run(fused.x, fused.w, x_full=fused.x_full)
        out = dict(fused.head.outputs)
    finally:
        routing_stream.RoutingStream.kernel_subs = ks0
        prep_stream.route_src = rs0
    h = fused.head
    reg = registry(fused)
    io = dict(x=fused.x, w=fused.w, y=se, **out)
    if h.router is not None:
        io.update(rw=h.rw[0], rb=h.rw[1], rix=h.rw[2])
    for k, v in io.items():
        reg.append((k, int(v.buffer_address()), 1 << 30))
    progs = {f"{c[0]},{c[1]}": program_json(p) for c, p, _ in h.state["progs"]}
    lay = fit.planned[0].lay
    meta = dict(
        model=name,
        chunk=chunk,
        accumulators=A,
        mesh=list(mesh.shape),
        layout=dict(
            GY=lay.GY,
            MT=lay.MT,
            NCH=lay.NCH,
            KB=lay.KB,
            on_engine=lay.on_engine,
            GX=lay.GX,
            compact=lay.compact,
            split=lay.split,
            xdram=lay.xdram,
        ),
        arena_bytes=h.eng.arena_bytes,
        l1_seen=_L1_FREE.get("v"),
        subs=subs,
        routes=routes,
    )
    fused.head.free_arena()
    for t in [se, x] + list(out.values()) + keep:
        ttnn.deallocate(t)
    del fused, h, io, se, out, w
    gc.collect()  # (the head's persistent L1 too: the second build must place its arena as the first did)
    return meta, progs, reg


def symbolize(a, b, reg_a, reg_b, where):
    """Two builds' values of one argument list -> the list with address positions as [name, offset]."""
    out = []
    for i, (va, vb) in enumerate(zip(a, b)):
        if va == vb:
            out.append(va)
            continue
        ha = [(n, va - base) for n, base, size in reg_a if base <= va < base + size]
        hb = [(n, vb - base) for n, base, size in reg_b if base <= vb < base + size]
        common = [h for h in ha if h in hb]
        if not common:
            raise AssertionError(f"{where}[{i}]: {va:#x} / {vb:#x} differ but are no known buffer's ({ha} / {hb})")
        # (the closest base: offsets inside the smallest containing buffer)
        out.append(list(min(common, key=lambda h: h[1])))
    assert len(a) == len(b), f"{where}: lengths {len(a)} / {len(b)}"
    return out


def merge(pa, pb, ra, rb):
    for chip, prog in pa.items():
        qb = pb[chip]
        for ki, (k, kb) in enumerate(zip(prog["kernels"], qb["kernels"])):
            where = f"chip {chip} kernel {ki}"
            assert set(k["runtime_args"]) == set(kb["runtime_args"]), where
            for core, args in k["runtime_args"].items():
                k["runtime_args"][core] = symbolize(args, kb["runtime_args"][core], ra, rb, f"{where} core {core}")
            k["common_runtime_args"] = symbolize(
                k["common_runtime_args"], kb["common_runtime_args"], ra, rb, where + " common"
            )
            k["compile_time_args"] = symbolize(k["compile_time_args"], kb["compile_time_args"], ra, rb, where + " ct")
            if k["source"] != kb["source"]:  # (an address in a define: SE_EW, RT_MSYNC, ...)
                da, db = _defines(k["source"]), _defines(kb["source"])
                for (na, va), (nb, vb) in zip(da, db):
                    if va != vb:
                        ia, ib = int(va.rstrip("u")), int(vb.rstrip("u"))
                        sym = symbolize([ia], [ib], ra, rb, f"{where} define {na}")[0]
                        k.setdefault("address_defines", {})[na] = sym
            k["source_defines"] = _defines(k["source"])
        for c, cb in zip(prog["cbs"], qb["cbs"]):
            if c["buffer"] is not None:
                c["buffer"] = symbolize(
                    [c["buffer"] + c["address_offset"]],
                    [cb["buffer"] + cb["address_offset"]],
                    ra,
                    rb,
                    f"chip {chip} cb",
                )[0]
    return pa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="glm53")
    ap.add_argument("--chunk", type=int, default=5120)
    ap.add_argument("--accumulators", type=int, default=12)
    ap.add_argument("--fabric", default="FABRIC_2D_TORUS_XY")
    ap.add_argument("--out", default=str(Path(__file__).with_name("head_programs")))
    a = ap.parse_args()
    ttnn.set_fabric_config(getattr(ttnn.FabricConfig, a.fabric))
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(8, 4))
    try:
        for name in a.models.split(","):
            meta, pa, ra = build(mesh, name, a.chunk, a.accumulators, False)
            meta_b, pb, rb = build(mesh, name, a.chunk, a.accumulators, True)
            assert meta_b["layout"] == meta["layout"], f"the moved build planned another layout: {meta_b['layout']}"
            assert meta_b["arena_bytes"] == meta["arena_bytes"]
            progs = merge(pa, pb, ra, rb)
            out = Path(a.out)
            out.mkdir(parents=True, exist_ok=True)
            f = out / f"{name}-8x4-{a.fabric.lower()}-c{a.chunk}-A{a.accumulators}.json.gz"
            with gzip.open(f, "wt") as fh:
                json.dump(dict(meta=meta, programs=progs), fh, separators=(",", ":"))
            print(f"wrote {f}", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    sys.exit(main())
