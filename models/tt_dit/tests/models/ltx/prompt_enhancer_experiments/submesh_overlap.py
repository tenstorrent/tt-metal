# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Can a second submesh overlap the full-shape submesh LTX carves, and can both be used safely?

Runs on the Galaxy 4x8 with the LTX distilled device params (ring fabric, 8 KB payload, 500 MB
trace region). ``full`` is the (4,8) submesh the pipeline uses; ``single`` is a (1,1) submesh at
(0,0) on the same parent, standing in for a device prompt enhancer. Each stage records its outcome
to ``results.json`` under OUT_DIR ($LTX_ENHANCER_EXP_DIR, default ~/ltx_enhancer_experiments) and never asserts, so a surprising answer still completes
the matrix.

  S1  creation: does create_submesh accept the overlap?
  S2  aliasing: do the two handles hand out the same DRAM address on the shared chip?
  S3  reservation: does a guard buffer on ``full`` keep ``single``'s allocations clear of ``full``'s?
  S4  trace hazard: after a trace is captured on ``full``, do ``single``'s allocations collide
      with the trace's baked addresses, in either direction?
  S5  control: the same post-capture allocation on ``full`` itself, to see whether the trace
      allocation tracker catches it (TT_METAL_TRACE_ALLOC_TRACKING=1) when it is one handle.
  S6  (opt-in, SUBMESH_EXP_CCL=1) all-gather on a (1,8) row submesh under LTX's fabric config.
"""

import json
import os
import time
import traceback

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.tests.models.ltx.ltx_mesh_params import LTX_DISTILLED_MESH_PARAMS_DL

# Outputs (videos, logs, JSON) stay out of the repo tree; LTX_ENHANCER_EXP_DIR relocates them.
OUT_DIR = os.path.join(
    os.environ.get("LTX_ENHANCER_EXP_DIR", os.path.expanduser("~/ltx_enhancer_experiments")), "submesh_overlap"
)
os.makedirs(OUT_DIR, exist_ok=True)
GALAXY_RING = [p for p in LTX_DISTILLED_MESH_PARAMS_DL if p.id == "4x8sp1tp0nl2_ring_is_fsdp0"]

MB = 1 << 20
DRAM_BANKS = 8  # Blackhole: 8 DRAM channels; buffer addresses are per-bank offsets
# (4096, 4096) bf16 = 32 MB per device: big enough that addresses are unambiguous, small enough to be cheap.
SHAPE = (1, 1, 4096, 4096)
BYTES = 2 * SHAPE[2] * SHAPE[3]


def _rep(handle, value: float):
    """Replicated DRAM tensor filled with ``value`` on every device of ``handle``."""
    return ttnn.from_torch(
        torch.full(SHAPE, value, dtype=torch.bfloat16),
        device=handle,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(handle),
    )


def _dev(tensor, idx: int) -> float:
    """Mean of the per-device shard ``idx`` (row-major mesh order; 0 is coordinate (0,0))."""
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[idx]).float().mean().item()


def _dram_buffers(handle):
    return sorted(
        (b.device_id, b.address, b.max_size_per_bank)
        for b in ttnn._ttnn.reports.get_buffers(handle)
        if b.buffer_type == ttnn.BufferType.DRAM
    )


def _free(*tensors):
    for t in tensors:
        try:
            ttnn.deallocate(t)
        except Exception:  # noqa: BLE001 — already freed or never allocated
            pass


@pytest.mark.parametrize(
    "mesh_device, sp_axis, tp_axis, num_links, device_params, topology, is_fsdp, dynamic_load",
    GALAXY_RING,
    indirect=["mesh_device", "device_params"],
)
def test_submesh_overlap(mesh_device, device_params, sp_axis, tp_axis, num_links, topology, is_fsdp, dynamic_load):
    parent = mesh_device
    results = {
        "parent_shape": list(parent.shape),
        "device_params": {k: str(v) for k, v in device_params.items()},
        "trace_alloc_tracking": os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING"),
        "stages": {},
    }

    def record(stage, **kw):
        results["stages"][stage] = kw
        logger.info(f"[{stage}] " + json.dumps(kw, default=str))

    full = parent.create_submesh(ttnn.MeshShape(*parent.shape))
    full_dev0 = full.get_device_ids()[0]

    # ---- S1: creation --------------------------------------------------------------------
    single = row = None
    s1 = {}
    for name, shape in (("single_1x1", (1, 1)), ("row_1x8", (1, 8))):
        try:
            sub = parent.create_submesh(ttnn.MeshShape(*shape), offset=ttnn.MeshCoordinate(0, 0))
            s1[name] = {
                "created": True,
                "device_ids": sub.get_device_ids(),
                "shares_full_dev0": full_dev0 in sub.get_device_ids(),
                "allocator_free_dram_mb": None,
            }
            if name == "single_1x1":
                single = sub
            else:
                row = sub
        except Exception as e:  # noqa: BLE001 — the outcome is the data
            s1[name] = {"created": False, "error": f"{type(e).__name__}: {e}"}
    s1["submeshes_on_parent"] = len(parent.get_submeshes())
    s1["dram_base_full"] = ttnn.get_allocator_base_address(full, ttnn.BufferType.DRAM)
    if single is not None:
        s1["dram_base_single"] = ttnn.get_allocator_base_address(single, ttnn.BufferType.DRAM)
    record("S1_creation", **s1)
    if single is None:
        _dump(results)
        return

    def _stage_aliasing():
        # ---- S2: aliasing ----------------------------------------------------------------------
        A = _rep(full, 1.0)
        parent.quiesce_devices()
        B = _rep(single, 2.0)
        parent.quiesce_devices()
        a_dev0, a_dev1, b_val = _dev(A, 0), _dev(A, 1), _dev(B, 0)
        record(
            "S2_aliasing",
            addr_full_A=A.buffer_address(),
            addr_single_B=B.buffer_address(),
            same_address=A.buffer_address() == B.buffer_address(),
            full_A_dev0_after_single_write=a_dev0,
            full_A_dev1_after_single_write=a_dev1,
            single_B=b_val,
            full_A_corrupted_on_shared_chip=abs(a_dev0 - 1.0) > 1e-3,
            full_handle_dram_buffers=len(_dram_buffers(full)),
            single_handle_dram_buffers=len(_dram_buffers(single)),
            single_handle_sees_full_buffers=any(
                d == full_dev0 and addr == A.buffer_address() for d, addr, _ in _dram_buffers(single)
            ),
        )
        _free(A, B)
        parent.quiesce_devices()

    def _stage_guard():
        # ---- S3: guard-buffer reservation ------------------------------------------------------
        guard_bytes = 8 * BYTES  # 256 MB per device reserved on ``full`` before anything else
        # Direct device allocation: from_torch stages through temporaries that fragment the allocator.
        G = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, 4096, 4096 * 8]), ttnn.bfloat16, ttnn.TILE_LAYOUT, full, ttnn.DRAM_MEMORY_CONFIG
        )
        A2 = _rep(full, 1.0)
        parent.quiesce_devices()
        B2 = _rep(single, 2.0)
        B3 = _rep(single, 3.0)
        parent.quiesce_devices()
        g_lo = G.buffer_address()
        g_per_bank = guard_bytes // DRAM_BANKS
        g_hi = g_lo + g_per_bank  # addresses are per-bank offsets, so the guard spans its per-bank footprint
        sibling_addrs = [B2.buffer_address(), B3.buffer_address()]
        sibling_high_water = max(sibling_addrs) + BYTES // DRAM_BANKS
        inside = lambda a: g_lo <= a < g_hi  # noqa: E731
        record(
            "S3_guard_reservation",
            guard_addr=g_lo,
            guard_bytes=guard_bytes,
            guard_per_bank_bytes=g_per_bank,
            guard_end=g_hi,
            sibling_high_water=sibling_high_water,
            sibling_fully_inside_guard=g_lo <= min(sibling_addrs) and sibling_high_water <= g_hi,
            sibling_reported_buffers=_dram_buffers(single),
            full_A2_addr=A2.buffer_address(),
            single_B2_addr=B2.buffer_address(),
            single_B3_addr=B3.buffer_address(),
            single_allocs_inside_guard=inside(B2.buffer_address()) and inside(B3.buffer_address()),
            full_A2_dev0=_dev(A2, 0),
            full_A2_intact=abs(_dev(A2, 0) - 1.0) < 1e-3,
            single_B2=_dev(B2, 0),
            single_B3=_dev(B3, 0),
            allocation_direction="bottom_up" if A2.buffer_address() > g_lo else "top_down",
        )
        _free(G, A2, B2, B3)
        parent.quiesce_devices()

    if os.environ.get("SUBMESH_EXP_GUARD_FIRST") == "1":
        _stage_guard()
        _stage_aliasing()
    else:
        _stage_aliasing()
        _stage_guard()

    # ---- S4: cross-handle trace hazard -----------------------------------------------------
    s4 = {}
    tid = None
    try:
        X = _rep(full, 1.0)
        # Eager pass compiles the kernels; capture needs them precompiled.
        T0 = ttnn.multiply(X, 2.0)
        Y0 = ttnn.add(T0, 1.0)
        _free(T0, Y0)
        parent.quiesce_devices()

        tid = ttnn.begin_trace_capture(full, cq_id=0)
        T = ttnn.multiply(X, 2.0)
        t_addr = T.buffer_address()
        Y = ttnn.add(T, 1.0)
        _free(T)  # intermediate freed inside capture, as every real model does
        ttnn.end_trace_capture(full, tid, cq_id=0)
        s4.update(full_X_addr=X.buffer_address(), full_T_baked_addr=t_addr, full_Y_addr=Y.buffer_address())

        parent.quiesce_devices()
        # First allocation on the sibling lands wherever its empty allocator starts; the second
        # is sized to reach the trace's freed intermediate slot if the first took X's slot.
        Z1 = _rep(single, 7.0)
        Z2 = _rep(single, 9.0)
        parent.quiesce_devices()
        s4.update(
            single_Z1_addr=Z1.buffer_address(),
            single_Z2_addr=Z2.buffer_address(),
            Z1_overlaps_full_X=Z1.buffer_address() == X.buffer_address(),
            Z2_overlaps_full_T=Z2.buffer_address() == t_addr,
            full_X_dev0_after_single_alloc=_dev(X, 0),
        )

        replay_error = None
        try:
            ttnn.execute_trace(full, tid, cq_id=0, blocking=True)
        except Exception as e:  # noqa: BLE001 — the outcome is the data
            replay_error = f"{type(e).__name__}: {e}"
        parent.quiesce_devices()
        s4.update(
            replay_error=replay_error,
            full_Y_dev0_after_replay=_dev(Y, 0),
            full_Y_dev1_after_replay=_dev(Y, 1),
            expected_Y=3.0,
            single_Z1_after_replay=_dev(Z1, 0),
            single_Z2_after_replay=_dev(Z2, 0),
            expected_Z1=7.0,
            expected_Z2=9.0,
        )
        s4["full_output_corrupted_by_sibling"] = abs(s4["full_Y_dev0_after_replay"] - 3.0) > 1e-3
        s4["sibling_corrupted_by_replay"] = (
            abs(s4["single_Z1_after_replay"] - 7.0) > 1e-3 or abs(s4["single_Z2_after_replay"] - 9.0) > 1e-3
        )
        _free(Z1, Z2, Y, X)
    except Exception:  # noqa: BLE001
        s4["exception"] = traceback.format_exc()
    finally:
        if tid is not None:
            try:
                ttnn.release_trace(full, tid)
            except Exception:  # noqa: BLE001
                pass
    record("S4_cross_handle_trace_hazard", **s4)
    parent.quiesce_devices()

    # ---- S5: same-handle control -----------------------------------------------------------
    s5 = {}
    tid = None
    try:
        X3 = _rep(full, 1.0)
        tid = ttnn.begin_trace_capture(full, cq_id=0)
        T3 = ttnn.multiply(X3, 2.0)
        t3_addr = T3.buffer_address()
        Y3 = ttnn.add(T3, 1.0)
        _free(T3)
        ttnn.end_trace_capture(full, tid, cq_id=0)
        E = _rep(full, 7.0)  # post-capture allocation on the SAME handle
        s5.update(
            full_T3_baked_addr=t3_addr, full_E_addr=E.buffer_address(), E_overlaps_T3=E.buffer_address() == t3_addr
        )
        replay_error = None
        try:
            ttnn.execute_trace(full, tid, cq_id=0, blocking=True)
        except Exception as e:  # noqa: BLE001
            replay_error = f"{type(e).__name__}: {str(e)[:300]}"
        parent.quiesce_devices()
        s5.update(replay_error=replay_error, tracker_caught_it=replay_error is not None)
        if replay_error is None:
            s5.update(full_E_after_replay=_dev(E, 0), expected_E=7.0, E_corrupted=abs(_dev(E, 0) - 7.0) > 1e-3)
        _free(E, Y3, X3)
    except Exception:  # noqa: BLE001
        s5["exception"] = traceback.format_exc()
    finally:
        if tid is not None:
            try:
                ttnn.release_trace(full, tid)
            except Exception:  # noqa: BLE001
                pass
    record("S5_same_handle_control", **s5)
    parent.quiesce_devices()

    # ---- S6: CCL on the row submesh (opt-in) -----------------------------------------------
    if os.environ.get("SUBMESH_EXP_CCL") == "1" and row is not None:
        s6 = {}
        try:
            src = torch.arange(8 * 32 * 32, dtype=torch.float32).reshape(1, 1, 32, 8 * 32).to(torch.bfloat16)
            t = ttnn.from_torch(
                src,
                device=row,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=ttnn.ShardTensorToMesh(row, dim=3),
            )
            t0 = time.time()
            g = ttnn.all_gather(t, dim=3, num_links=1, topology=ttnn.Topology.Ring)
            out = ttnn.to_torch(ttnn.get_device_tensors(g)[0])
            s6.update(ok=torch.equal(out, src), seconds=round(time.time() - t0, 3))
            _free(t, g)
        except Exception:  # noqa: BLE001
            s6["exception"] = traceback.format_exc()
        record("S6_row_all_gather", **s6)
        parent.quiesce_devices()
    else:
        record("S6_row_all_gather", skipped="set SUBMESH_EXP_CCL=1 to run")

    _dump(results)


def _dump(results):
    path = os.path.join(OUT_DIR, f"results{os.environ.get('SUBMESH_EXP_TAG', '')}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"results -> {path}")
