# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The C++ fused head's per-chip programs (ttnn.experimental.moe_block.fused_head_programs: built, not launched)
against the reference's (dump_head_programs.py: models/demos/moe_block/moe_block.py FusedHead on mstaletovic/moe-trim)
on the mock Galaxy: no device, no kernel compiles. Host only, but it needs the mock cluster and the reference's dumps:

    TT_METAL_MOCK_CLUSTER_DESC_PATH=$TT_METAL_HOME/tt_metal/third_party/umd/tests/cluster_descriptor_examples/\
blackhole_galaxy.yaml MOE_BLOCK_REFS=/data/mstaletovic/jobs/port/refs pytest <this file>

(skipped without either). Per chip and kernel: the kernel file vs the reference source it came from, core ranges,
compile-time args, the runtime args of every core and the common args (addresses as the buffer and offset they point
at: the reference's dump marks which arguments are addresses), the defines the host sets; the CBs (index, size, page,
cores, arena offset). Expected differences, checked as such: the addresses the reference baked into defines are common runtime args (one more common arg on the engine and router
kernels); the fabric connections' arguments are main's (compared up to their start); the kernel_lib multicast and
fabric semaphores take other ids; the routing planner's tie split is the config's rule (NorthEast: YT 0)."""

import gzip
import json
import os
import re
from pathlib import Path

import pytest
import torch

import ttnn

mb = ttnn.experimental.moe_block
REFS = os.environ.get("MOE_BLOCK_REFS")
MOCK = "blackhole_galaxy" in os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH", "")
DIR = Path(REFS) / "fused_head" / "programs" if REFS else None
CASES = sorted(p.name for p in DIR.glob("*.json.gz")) if DIR is not None and DIR.exists() else []

pytestmark = pytest.mark.skipif(not (MOCK and CASES), reason="needs the mock Galaxy and MOE_BLOCK_REFS program dumps")

# (E, K, H, I, shared I, shared activation, score, route scale, latent)
MODELS = {
    "glm53": (256, 8, 6144, 2048, 2048, "Silu", "Sigmoid", 2.5, None),
    "glm53_flash": (288, 8, 4096, 2048, 2048, ("ClampedSilu", 10.0), "Sigmoid", 2.5, None),
    "k27": (384, 8, 7168, 2048, 2048, "Silu", "Sigmoid", 2.827, None),
    "k3": (896, 16, 7168, 3072, 6144, ("Situ", 4.0, 25.0), "Sigmoid", 1.0, 3584),
    "dsv41_flash": (384, 6, 5120, 2304, 2304, ("ClampedSilu", 10.0), "SqrtSoftplus", 1.5, None),
}

# the reference's names of the persistent words / tensors, and the offsets of mine (name, offset) they map to
_SEM16 = {"arr_sem": "arr", "xg_sem": "xg"}
_RA_P = ["calls_p", "calls_p2", "go", "go2", "parts", "ex", "ex2", "rows", "rows2"]
_RA_S = ["calls_s", "calls_s2", "ready", "ready2"]


def _act(spec):
    if isinstance(spec, str):
        return mb.ExpertActivation(kind=getattr(mb.ExpertActivationKind, spec))
    kind = spec[0]
    if kind == "ClampedSilu":
        return mb.ExpertActivation(kind=mb.ExpertActivationKind.ClampedSilu, limit=spec[1])
    return mb.ExpertActivation(kind=mb.ExpertActivationKind.Situ, beta=spec[1], linear_beta=spec[2])


def _dims(name):
    E, K, H, I, Is, act, score, scale, latent = MODELS[name]
    router = mb.RouterSpec(score_func=getattr(mb.RouterScoreFunc, score), route_scale=scale)
    return mb.MoEModelDims(
        num_experts=E,
        top_k=K,
        emb_dim=H,
        routed_expert_hidden_dim=I,
        routed_expert_activation=_act(act),
        router=router,
        shared_expert_hidden_dim=Is,
        shared_expert_activation=_act(act),
        latent_dim=latent,
    )


def _cores(ranges):
    return sorted((x, y) for x0, y0, x1, y1 in ranges for x in range(x0, x1 + 1) for y in range(y0, y1 + 1))


def _crs_cores(crs):
    return sorted(
        (x, y) for r in crs.ranges() for x in range(r.start.x, r.end.x + 1) for y in range(r.start.y, r.end.y + 1)
    )


def _ref_name(name, off):
    """A reference buffer (name, offset) -> the C++ op's (name, offset)."""
    m = re.match(r"(arr_sem|xg_sem)(\d+)$", name)
    if m:
        return (_SEM16[m.group(1)], 16 * int(m.group(2)) + off)
    if name.startswith("rt_sem"):
        return ("rt", 16 * int(name[6:]) + off)
    if name in ("ra_c0", "ra_c1"):
        return ("ra_c", 16 * int(name[-1]) + off)
    if name.startswith("ra_") and name[3:] in _RA_P:
        return ("ra_p", 16 * _RA_P.index(name[3:]) + off)
    if name.startswith("ra_") and name[3:] in _RA_S:
        return ("ra_s", 16 * _RA_S.index(name[3:]) + off)
    if name == "sfree_words":
        return ("sfree", off)
    if name.startswith("ra_") and name[3:] in ("slots", "fslots", "counts", "regions", "token_index", "table", "plan"):
        return (name[3:], off)
    return (name, off)


class Registry:
    """The C++ op's buffers: address ranges by name."""

    def __init__(self, entries):
        self.entries = entries  # [(name, base, size)]

    def name_of(self, v):
        hits = [(n, v - b) for n, b, s in self.entries if b <= v < b + s]
        return min(hits, key=lambda h: h[1]) if hits else None


def _compare_args(ref, mine, reg, where, errors, limit=None):
    n = len(ref) if limit is None else min(limit, len(ref), len(mine))
    if limit is None and len(ref) != len(mine):
        errors.append(f"{where}: {len(mine)} args, the reference {len(ref)}")
        n = min(len(ref), len(mine))
    for i in range(n):
        r, m = ref[i], mine[i]
        if isinstance(r, list):
            want = _ref_name(*r)
            got = reg.name_of(m)
            if got != want:
                errors.append(f"{where}[{i}]: address {m:#x} = {got}, the reference's {r} -> {want}")
        elif r != m:
            errors.append(f"{where}[{i}]: {m}, the reference {r}")
        if len(errors) > 40:
            return


@pytest.fixture(scope="module")
def mesh():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D_TORUS_XY)
    m = ttnn.open_mesh_device(ttnn.MeshShape(8, 4))
    yield m
    ttnn.close_mesh_device(m)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.mark.parametrize("case", CASES)
def test_head_program_parity(mesh, case):
    with gzip.open(DIR / case, "rt") as f:
        ref = json.load(f)
    meta = ref["meta"]
    name, chunk, A = meta["model"], meta["chunk"], meta["accumulators"]
    dims = _dims(name)
    res = mb.MoEBlockResources(mesh, dims, mb.MoEBlockConfig(chunk_tokens=chunk, num_links=2))
    dtypes = mb.MoEDtypes()
    l1 = meta["l1_seen"]
    fit = mb.plan_fused_head(res, dtypes, A, l1)
    assert fit.ok, fit.why
    lay = fit.planned[0].lay
    got = dict(
        GY=lay.GY, MT=lay.MT, NCH=lay.NCH, KB=lay.KB, on_engine=lay.on_engine, GX=lay.GX, compact=lay.compact,
        split=lay.split, xdram=lay.xdram,
    )  # fmt: skip
    assert got == meta["layout"], f"planned {got}, the reference {meta['layout']}"

    E, K, H, I, Is, *_ = MODELS[name]
    lat = MODELS[name][-1]
    host = lambda *s: ttnn.from_torch(torch.zeros(*s), dtype=ttnn.float32)  # noqa: E731
    w = mb.prepare_fused_head_weights(
        host(Is, H), host(Is, H), host(H, Is), host(lat, H) if lat else None, host(E, H), host(E),
        resources=res, dtypes=dtypes, accumulators=A, l1_free=l1,
    )  # fmt: skip
    x = ttnn.from_torch(
        torch.zeros(1, 1, chunk, H),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(8, 4), dims=(2, 3)),
    )
    progs = mb.fused_head_programs(x, w, resources=res, dtypes=dtypes, accumulators=A, l1_free=l1)
    # (my buffers: the arena, the persistent words, the tensors)
    entries = [(n, a, size) for n, (a, size) in progs.addresses.items()]
    reg = Registry(entries)

    errors = []
    for (r, c), desc in progs.programs:
        rp = ref["programs"][f"{r},{c}"]
        mk, rk = list(desc.kernels), rp["kernels"]
        assert len(mk) == len(rk), f"chip ({r}, {c}): {len(mk)} kernels, the reference {len(rk)}"
        for i, (k, kr) in enumerate(zip(mk, rk)):
            where = f"chip ({r},{c}) kernel {i} {Path(k.kernel_source).name}"
            if _crs_cores(k.core_ranges) != _cores(kr["core_ranges"]):
                errors.append(f"{where}: cores differ")
            cfg, rcfg = k.config, kr["config"]
            if rcfg["type"] == "compute":
                mine_cfg = dict(
                    math_fidelity=str(cfg.math_fidelity).split(".")[-1],
                    fp32_dest_acc_en=bool(cfg.fp32_dest_acc_en),
                    dst_full_sync_en=bool(cfg.dst_full_sync_en),
                    math_approx_mode=bool(cfg.math_approx_mode),
                    unpack_to_dest_mode=[j for j, m in enumerate(cfg.unpack_to_dest_mode) if "Fp32" in str(m)],
                )
                if mine_cfg != {k_: v for k_, v in rcfg.items() if k_ != "type"}:
                    errors.append(f"{where}: compute config {mine_cfg}, the reference {rcfg}")
            else:
                noc = lambda n: 0 if str(n).split(".")[-1] in ("NOC_0", "RISCV_0_default") else 1  # noqa: E731
                if str(cfg.processor).split(".")[-1] != rcfg["processor"] or noc(cfg.noc) != noc(rcfg["noc"]):
                    errors.append(f"{where}: {cfg.processor} / {cfg.noc}, the reference {rcfg}")
            ct = list(k.compile_time_args)
            if ct != kr["compile_time_args"]:
                errors.append(f"{where}: compile-time args {ct}, the reference {kr['compile_time_args']}")
            fname = Path(k.kernel_source).name
            rt = {f"{x},{y}": list(k.runtime_args[x][y]) for x, y in _crs_cores(k.core_ranges)}
            for core, args in kr["runtime_args"].items():
                # (a fabric connection's args are main's: compared up to where they start)
                limit = progs.fabric_starts.get(f"{r},{c},{fname},{core}")
                _compare_args(args, rt.get(core, []), reg, f"{where} core {core}", errors, limit)
            common = list(k.common_runtime_args)
            extra = 1 if any(s in k.kernel_source for s in ("se_brisc", "se_ncrisc")) else 0
            if any(s in k.kernel_source for s in ("router_",)):
                extra = 2
            _compare_args(kr["common_runtime_args"], common[: len(common) - extra] if extra else common, reg,
                          f"{where} common", errors)  # fmt: skip
            # the host's defines: every one of mine is the reference's (values), the reference's addresses moved
            ref_defs = {}
            for n, v in kr["source_defines"]:  # (the host's come first; the kernel's own defaults follow)
                ref_defs.setdefault(n, v.split("//")[0].strip())
            subs = meta["subs"][f"{r},{c}"]
            sub = subs[0] if "router_ncrisc" in fname else (subs[1] if "router_brisc" in fname else None)
            for n, v in k.defines:
                if n.startswith("RAP_") and sub is not None and str(sub[n[4:]]) != v:
                    errors.append(f"{where}: define {n} = {v}, the reference's substitution {sub[n[4:]]}")
            for n, v in k.defines:
                if (
                    n.endswith("_OFF")
                    or n in ("RT_ENG_ON", "SE_GROW_OWN_CR", "SE_GROW_OWN_RT")
                    or n.startswith(("RAP_", "RW_"))
                ):
                    continue
                if n == "RAP_YT":
                    continue
                if n not in ref_defs:
                    errors.append(f"{where}: define {n} not in the reference")
                elif ref_defs[n].rstrip("u") != v.rstrip("u"):
                    errors.append(f"{where}: define {n} = {v}, the reference {ref_defs[n]}")
            if len(errors) > 40:
                break
        mine_cbs = sorted(
            (fd.buffer_index, cb.total_size, fd.page_size, tuple(_crs_cores(cb.core_ranges)), cb.address_offset)
            for cb in desc.cbs
            for fd in cb.format_descriptors
        )
        ref_cbs = sorted(
            (f[0], cb["total_size"], f[2], tuple(_cores(cb["core_ranges"])), cb["buffer"][1] if cb["buffer"] else 0)
            for cb in rp["cbs"]
            for f in cb["formats"]
        )
        if mine_cbs != ref_cbs:
            errors.append(f"chip ({r},{c}): CBs differ: {set(mine_cbs) ^ set(ref_cbs)}")
        if len(errors) > 40:
            break
    assert not errors, "\n".join(errors[:40])
