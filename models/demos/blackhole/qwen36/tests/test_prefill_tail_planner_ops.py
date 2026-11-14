# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device tests (8-chip TP ring, e.g. one Galaxy row with QWEN36_FABRIC_RING=1) of the mid-scan state snapshots of
ttnn.transformer.chunk_gated_delta_rule (QWEN36_PREFILL_INTRACE_SNAP), op level, with the qwen36 GDN prefill's TP8
shapes and call (flat raw q/k/v from the KDA conv, fp32 beta/g, carried fp32 state, the layer's constant tiles):

* o and final_state of a call with snapshot_index / snapshot_state are bit-identical to the same call without them
  (also at 128 / 256 rows, where the default dispatch takes the phased program and a snapshot call the fused one);
* snapshot slot i holds the final_state of the call over the first (snapshot_index[i] + 1) * 32 rows, bit for bit, and
  the slots past the index list are left untouched;
* the device time of the call with and without snapshots (logged).
And the conv-input row gather of kda_conv_prefill (snap_rows / snap_out) against the rows of the row-major input.

  source <8-chip env>; python -m pytest models/demos/blackhole/qwen36/tests/test_prefill_tail_planner_ops.py -s
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt.gdn.fused_chunk import build_fused_const_tiles

_ROUTER = tpc.fabric_router_config()
DEVICE_PARAMS = [
    {
        "l1_small_size": 24576,
        "trace_region_size": 1 << 26,
        "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
        **({"fabric_router_config": _ROUTER} if _ROUTER is not None else {}),
    }
]
HK, HV, D = 2, 6, 128  # qwen36 GDN per device at TP8: 16 / 8 key heads, 48 / 8 value heads, head dim 128
END = 0x7FFFFFFF
SENTINEL = 7.25


def _rep(mesh, t, dtype, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _dev0(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


def _inputs(rows, seed):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, rows, HK * D, generator=g)
    k = torch.randn(1, rows, HK * D, generator=g)
    v = torch.randn(1, rows, HV * D, generator=g) * 0.5
    beta = torch.sigmoid(torch.randn(1, rows, HV, generator=g))
    gl = -torch.nn.functional.softplus(torch.randn(1, rows, HV, generator=g)) * 0.2  # log decay < 0
    s0 = torch.randn(1, HV, D, D, generator=g) * 0.05
    return q, k, v, beta, gl, s0


def _call(mesh, consts, host, rows, snap=None):
    q, k, v, beta, gl, s0 = host
    t = [
        _rep(mesh, q[:, :rows], ttnn.bfloat16),
        _rep(mesh, k[:, :rows], ttnn.bfloat16),
        _rep(mesh, v[:, :rows], ttnn.bfloat16),
        _rep(mesh, gl[:, :rows], ttnn.float32),
        _rep(mesh, beta[:, :rows], ttnn.float32),
        _rep(mesh, s0, ttnn.float32),
    ]
    eye, tril, ones, masks = consts
    o, fs = ttnn.transformer.chunk_gated_delta_rule(
        t[0],
        t[1],
        t[2],
        t[3],
        t[4],
        scale=D**-0.5,
        initial_state=t[5],
        output_final_state=True,
        chunk_size=32,
        output_head_major=True,
        eye=eye,
        tril=tril,
        ones=ones,
        masks=masks,
        **({} if snap is None else {"snapshot_index": snap[0], "snapshot_state": snap[1]}),
    )
    out = (_dev0(o), _dev0(fs))
    for x in t + [o, fs]:
        ttnn.deallocate(x)
    return out


def _bits(t):
    return t.contiguous().view(torch.int32) if t.dtype == torch.float32 else t.contiguous().view(torch.int16)


@pytest.mark.parametrize(
    "rows,chunks",
    [(128, [0, 2]), (256, [3]), (2048, [3, 31, 63]), (8192, [3, 251, 254, 255])],
    ids=["r128", "r256", "r2048", "r8192"],
)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_chunk_gdn_snapshots_bit_identical(mesh_device, rows, chunks):
    mesh = mesh_device
    consts = build_fused_const_tiles(mesh)
    host = _inputs(rows, seed=rows)
    S = 4
    words = list(chunks) + [END] * (8 - len(chunks))
    idx = _rep(mesh, torch.tensor([words], dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    snap = _rep(mesh, torch.full((S, HV, D, D), SENTINEL), ttnn.float32)

    o_ref, fs_ref = _call(mesh, consts, host, rows)
    o, fs = _call(mesh, consts, host, rows, snap=(idx, snap))
    assert torch.equal(_bits(o), _bits(o_ref)), f"rows={rows}: o changed with snapshots on"
    assert torch.equal(_bits(fs), _bits(fs_ref)), f"rows={rows}: final_state changed with snapshots on"
    got = _dev0(snap)
    for i, c in enumerate(chunks):
        _, fs_c = _call(mesh, consts, host, (c + 1) * 32)
        d = (got[i : i + 1] - fs_c).abs().max().item()
        assert torch.equal(_bits(got[i : i + 1]), _bits(fs_c)), f"rows={rows}: slot {i} (chunk {c}) differs, max|d|={d}"
    for i in range(len(chunks), S):
        assert bool((got[i] == SENTINEL).all()), f"rows={rows}: unused slot {i} was written"
    # the other devices hold the same state (replicated inputs)
    for dt in ttnn.get_device_tensors(snap)[1:]:
        assert torch.equal(_bits(ttnn.to_torch(dt)), _bits(got))

    # same index page again with the list cut short: slot 0 rewritten, the rest keep the first call's values
    words2 = [chunks[0]] + [END] * 7
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(
            torch.tensor([words2], dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        ),
        idx,
    )
    _call(mesh, consts, host, rows, snap=(idx, snap))
    assert torch.equal(_bits(_dev0(snap)), _bits(got)), "re-staged index: slots changed unexpectedly"

    # Timing at every size: at 128 / 256 rows the plain call takes the phased program (prep + scan) and the snapshot
    # call the fused one (ttsim per chip: 47.9 / 66.5 us phased vs 42.2 / 62.3 us fused, device kernel time).
    if True:
        reps = 10
        dev = [_rep(mesh, x[:, :rows], dt) for x, dt in zip(host[:5], (ttnn.bfloat16,) * 3 + (ttnn.float32,) * 2)]
        s0 = _rep(mesh, host[5], ttnn.float32)
        eye, tril, ones, masks = consts

        def run(with_snap):
            kw = {"snapshot_index": idx, "snapshot_state": snap} if with_snap else {}
            o, fs = ttnn.transformer.chunk_gated_delta_rule(
                dev[0], dev[1], dev[2], dev[3], dev[4], scale=D**-0.5, initial_state=s0, output_final_state=True,
                chunk_size=32, output_head_major=True, eye=eye, tril=tril, ones=ones, masks=masks, **kw,
            )
            ttnn.deallocate(o)
            ttnn.deallocate(fs)

        times = {}
        for with_snap in (False, True, False, True):
            run(with_snap)
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            for _ in range(reps):
                run(with_snap)
            ttnn.synchronize_device(mesh)
            times.setdefault(with_snap, []).append(1e3 * (time.perf_counter() - t0) / reps)
        logger.info(
            f"[snapshot] chunk_gated_delta_rule rows={rows}: plain {min(times[False]):.3f} ms, "
            f"with {len(chunks)} snapshots {min(times[True]):.3f} ms (eager, incl. host dispatch)"
        )


@pytest.mark.parametrize("T", [128, 2048])
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_kda_conv_snapshot_rows(mesh_device, T):
    """kda_conv_prefill(snap_rows, snap_out): the gathered rows are the row-major conv input's rows, bit for bit, and
    q/k/v/new_state are unchanged by the gather."""
    from models.demos.blackhole.qwen36.tt.gdn.tp import kda_conv_prefill

    mesh = mesh_device
    C, kd, vd = 1280, 256, 768
    g = torch.Generator().manual_seed(T)
    qkv = torch.randn(1, T, C, generator=g).to(torch.bfloat16)
    taps = [_rep(mesh, torch.randn(1, 1, C, generator=g) * 0.3, ttnn.bfloat16) for _ in range(4)]
    hist = _rep(mesh, torch.zeros(1, 3, C), ttnn.bfloat16)
    start = _rep(mesh, torch.zeros(1, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    rows = [125, 126, 127, T - 3, T - 2, T - 1] + [0] * 10
    idx = _rep(mesh, torch.tensor([rows], dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    out = _rep(mesh, torch.zeros(1, 16, C), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    x = _rep(mesh, qkv, ttnn.bfloat16)
    ref = kda_conv_prefill(x, T, hist, taps, (kd, kd, vd), start)
    got = kda_conv_prefill(x, T, hist, taps, (kd, kd, vd), start, snap_rows=idx, snap_out=out)
    for a, b in zip(ref, got):
        assert torch.equal(_bits(_dev0(a)), _bits(_dev0(b))), "the gather changed the conv outputs"
    gathered = _dev0(out)[0]
    assert torch.equal(_bits(gathered), _bits(qkv[0, rows])), "gathered rows differ from the conv input rows"
    # new_state (the last three rows) equals the gather of rows T-3..T-1
    assert torch.equal(_bits(_dev0(got[3])[0, :3]), _bits(gathered[3:6]))
