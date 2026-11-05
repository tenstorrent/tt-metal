# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""A W-lane prefix alias of an 8-lane tensor (same device buffer) made with ttnn.experimental.view or
ttnn.reshape(logical, padded). Per kind x W: address, write ops, host write, 3-op trace, dealloc safety.

Run: flock -w 7200 /home/ttuser/atupe/device.lock timeout 1800 python -m pytest \
  models/demos/blackhole/qwen38_flash_next/tests/test_lane_prefix_view_device.py -s -p no:cacheprovider -o addopts=
Env: MESH_DEVICE=P150x8 TT_VISIBLE_DEVICES=0,1,2,3,7,6,5,4 QWEN38_X8_ROUTE=4,0,1,2,3,7,6,5. Result JSON: LPV_OUT.
"""

import json
import os
import traceback

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.demo import text_demo as demo
from models.demos.blackhole.qwen38_flash_next.demo.text_demo import device_params  # noqa: F401
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE

VIEW_ERR = [None]
DEVICE_PARAMS = dict(demo.DEVICE_PARAMS, trace_region_size=64 * 1024 * 1024)
TP = MESH_SHAPE[1]
OUT = os.environ.get("LPV_OUT")
BF, F32 = ttnn.bfloat16, ttnn.float32
TILE, RM = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT

# name: (dtype, layout, global torch shape, shard dim or None, per-device parent shape, lane dim, lanes-per-unit)
KINDS = {
    "k1_tile_bf16_repl": dict(dtype=BF, layout=TILE, gshape=(1, 8, 32, 128), shard=None, lane_dim=1, unit=1),
    "k2_tile_fp32_headshard": dict(dtype=F32, layout=TILE, gshape=(8, 12 * TP, 128, 128), shard=1, lane_dim=0, unit=1),
    "k3_tile_fp32_1x1": dict(dtype=F32, layout=TILE, gshape=(1, 8, 1, 1), shard=None, lane_dim=1, unit=1),
    "k4_rm_bf16": dict(dtype=BF, layout=RM, gshape=(1, 1, 32, 640), shard=None, lane_dim=2, unit=4),
    "k5_tile_hidden": dict(dtype=BF, layout=TILE, gshape=(1, 4, 8, 640 * TP), shard=3, lane_dim=2, unit=1),
}


def mapper(mesh, spec):
    if spec["shard"] is None:
        return ttnn.ReplicateTensorToMesh(mesh)
    return ttnn.ShardTensorToMesh(mesh, dim=spec["shard"])


def make(mesh, spec, x, device=True):
    kw = dict(dtype=spec["dtype"], layout=spec["layout"], mesh_mapper=mapper(mesh, spec))
    if device:
        kw.update(device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.from_torch(x, **kw)


def rand(spec, seed, shape=None):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(shape or spec["gshape"], generator=g)
    return x.to(torch.bfloat16).float() if spec["dtype"] == BF else x


def read(t):
    """[devices, *local] as float32 (bit view via float32 for both dtypes)."""
    return torch.stack([ttnn.to_torch(d).float() for d in ttnn.get_device_tensors(t)])


def addrs(t):
    return [d.buffer_address() for d in ttnn.get_device_tensors(t)]


def lane_slice(x, spec, w):
    """first w lanes of a [devices, *local] read; lane dim of the local shape shifts by one."""
    d = spec["lane_dim"] + 1
    idx = [slice(None)] * x.dim()
    idx[d] = slice(0, w * spec["unit"])
    return x[tuple(idx)]


def lane_rest(x, spec, w):
    d = spec["lane_dim"] + 1
    idx = [slice(None)] * x.dim()
    idx[d] = slice(w * spec["unit"], None)
    return x[tuple(idx)]


def alias_shape(spec, w):
    s = list(spec["gshape"])
    if spec["shard"] is not None:
        s[spec["shard"]] //= TP
    s[spec["lane_dim"]] = w * spec["unit"]
    return s


def padded_of(spec, local):
    s = list(local)
    if spec["layout"] == TILE:
        s[-1] = -(-s[-1] // 32) * 32
        s[-2] = -(-s[-2] // 32) * 32
    return s


def mk_alias(parent, spec, w):
    """returns (alias, mechanism). raises with both errors."""
    local = alias_shape(spec, w)
    errs = []
    try:
        a = ttnn.experimental.view(parent, local)
        return a, "view"
    except Exception as e:  # noqa
        errs.append(f"view: {type(e).__name__}: {' '.join(str(e).split())[:260]}")
        VIEW_ERR[0] = errs[-1]
    try:
        a = ttnn.reshape(parent, local, padded_of(spec, local))
        return a, "reshape(logical,padded) [" + str(VIEW_ERR[0]) + "]"
    except Exception as e:  # noqa
        errs.append(f"reshape: {type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}")
    raise RuntimeError(" | ".join(errs))


def fresh_like_alias(mesh, spec, w, seed):
    """A fresh (non-aliased) W-lane tensor of the same kind, built from a torch tensor."""
    gs = list(spec["gshape"])
    gs[spec["lane_dim"]] = w * spec["unit"]
    return make(mesh, spec, rand(spec, seed, tuple(gs))), gs


def checks(mesh, kname, spec, w, R):
    def rec(check, fn):
        if only_e and not check.startswith(("e_", "a_")):
            return
        if not only_e and check.startswith("e_"):
            return
        try:
            ok, msg = fn()
            R.append((kname, w, check, "PASS" if ok else "FAIL", msg))
        except Exception as e:  # noqa
            tb = traceback.extract_tb(e.__traceback__)[-1]
            R.append(
                (
                    kname,
                    w,
                    check,
                    "FAIL",
                    f"{type(e).__name__}: {' '.join(str(e).split())[:300]} @{os.path.basename(tb.filename)}:{tb.lineno}",
                )
            )

    parent = make(mesh, spec, rand(spec, 1))
    pbase = read(parent)
    box = {}
    only_e = os.environ.get("LPV_E") in ("del", "dealloc")

    def a_addr():
        a, mech = mk_alias(parent, spec, w)
        box["alias"], box["mech"] = a, mech
        same = addrs(a) == addrs(parent)
        return same, f"mech={mech} parent={addrs(parent)[0]} alias={addrs(a)[0]}"

    rec("a_same_address", a_addr)
    if "alias" not in box:
        return
    alias = box["alias"]

    def untouched_ok(label, expect_lanes):
        after = read(parent)
        okw = torch.equal(lane_slice(after, spec, w), expect_lanes)
        oku = torch.equal(lane_rest(after, spec, w), lane_rest(pbase, spec, w))
        return okw and oku, f"{label}: first_W_ok={okw} rest_untouched={oku}"

    def restore():
        ttnn.copy_host_to_device_tensor(make(mesh, spec, rand(spec, 1), device=False), parent)

    # b1 matmul
    def b_matmul():
        if kname.startswith("k1"):
            wt = rand(dict(spec, gshape=(1, 1, 128, 128)), 7, (1, 1, 128, 128))
            wdev = ttnn.from_torch(wt, dtype=BF, layout=TILE, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        elif kname.startswith("k2"):
            wdev, _ = fresh_like_alias(mesh, spec, w, 8)
        elif kname.startswith("k5"):
            wt = rand(spec, 7, (1, 1, 640, 640))
            wdev = ttnn.from_torch(wt, dtype=BF, layout=TILE, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        else:
            return True, "n/a (no matmul for this shape)"
        # reference: same matmul to a fresh output; the lane-prefix operand is the alias (input side)
        src = ttnn.experimental.view(parent, alias_shape(spec, w)) if False else alias
        ref = ttnn.matmul(src, wdev, dtype=spec["dtype"]) if not kname.startswith("k2") else ttnn.matmul(src, wdev)
        refh = read(ref)
        # now output into the alias: matmul(src_fresh, w, optional_output_tensor=alias)
        srcf, _ = fresh_like_alias(mesh, spec, w, 9)
        ref2 = read(ttnn.matmul(srcf, wdev))
        ttnn.matmul(srcf, wdev, optional_output_tensor=alias)
        return untouched_ok("matmul(optional_output_tensor)", lane_slice(ref2, spec, w) if False else ref2)

    rec("b_matmul_optional_output", b_matmul)
    restore()

    def b_add():
        srcf, gs = fresh_like_alias(mesh, spec, w, 10)
        c, _ = fresh_like_alias(mesh, spec, w, 11)
        ref = read(ttnn.add(srcf, c))
        ttnn.add(srcf, c, output_tensor=alias)
        return untouched_ok("add(output_tensor)", ref)

    rec("b_add_output_tensor", b_add)
    restore()

    def b_copy():
        srcf, _ = fresh_like_alias(mesh, spec, w, 12)
        ref = read(srcf)
        ttnn.copy(srcf, alias)
        return untouched_ok("copy(src, alias)", ref)

    rec("b_copy_src_alias", b_copy)
    restore()

    def c_hostwrite():
        x2 = rand(spec, 99)
        ttnn.copy_host_to_device_tensor(make(mesh, spec, x2, device=False), parent)
        ref = read(make(mesh, spec, x2))
        got = read(alias)
        return torch.equal(got, lane_slice(ref, spec, w)), "alias read after parent host write"

    rec("c_host_write_parent_read_alias", c_hostwrite)
    restore()

    def op3(src, dst):
        if spec["layout"] == RM:  # binary add with a preallocated output is refused for row-major inputs
            ttnn.copy(src, dst)
        else:
            ttnn.add(src, src, output_tensor=dst)

    def d_trace():
        scratch_a, _ = fresh_like_alias(mesh, spec, w, 20)
        scratch_b, _ = fresh_like_alias(mesh, spec, w, 21)
        ttnn.copy(alias, scratch_a)  # warm up (compile) outside capture
        ttnn.copy(scratch_a, scratch_b)
        op3(scratch_b, alias)
        restore()
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        ttnn.copy(alias, scratch_a)
        ttnn.copy(scratch_a, scratch_b)
        op3(scratch_b, alias)
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        okall, msg = True, []
        for i, seed in enumerate((31, 32)):
            x = rand(spec, seed)
            ttnn.copy_host_to_device_tensor(make(mesh, spec, x, device=False), parent)
            ref = read(make(mesh, spec, x))
            xw = lane_slice(ref, spec, w)
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            ok1 = torch.equal(read(scratch_b), xw)
            after = read(parent)
            if spec["layout"] == RM:
                dbl = xw
            elif spec["dtype"] == BF:
                dbl = (xw * 2).to(torch.bfloat16).float()
            else:
                dbl = xw * 2
            ok2 = torch.equal(lane_slice(after, spec, w), dbl)
            ok3 = torch.equal(lane_rest(after, spec, w), lane_rest(ref, spec, w))
            okall &= ok1 and ok2 and ok3
            msg.append(f"replay{i}: scratch={ok1} parentW=2x:{ok2} rest={ok3}")
        ttnn.release_trace(mesh, tid)
        return okall, "; ".join(msg)

    rec("d_trace_3op_replay_x2", d_trace)
    restore()

    def e_dealloc():
        a2, _ = mk_alias(parent, spec, w)
        a_before = addrs(parent)
        if os.environ.get("LPV_E") == "del":
            del a2
            import gc

            gc.collect()
            how = "del alias + gc"
        else:
            ttnn.deallocate(a2)
            how = "ttnn.deallocate(alias)"
        ok1 = torch.equal(read(parent), pbase) and addrs(parent) == a_before
        z = make(mesh, spec, rand(spec, 5))
        ok3 = addrs(z) != a_before
        return ok1 and ok3, f"after {how}: parent intact={ok1}; new alloc elsewhere={ok3}"

    rec("e_dealloc_alias_keeps_parent", e_dealloc)


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_lane_prefix_view(mesh_device):
    mesh = mesh_device
    R = []
    for kname, spec in KINDS.items():
        if os.environ.get("LPV_KIND") and not kname.startswith(os.environ["LPV_KIND"]):
            continue
        for w in (2, 4):
            if os.environ.get("LPV_W") and int(os.environ["LPV_W"]) != w:
                continue
            try:
                checks(mesh, kname, spec, w, R)
            except Exception as e:  # noqa
                R.append((kname, w, "setup", "FAIL", f"{type(e).__name__}: {' '.join(str(e).split())[:300]}"))
    for r in R:
        print("LPV", *r, sep=" | ")
    if OUT:
        with open(OUT, "w") as f:
            json.dump(R, f, indent=1)
    bad = [r for r in R if r[3] != "PASS"]
    assert not bad, f"{len(bad)} failing checks"
