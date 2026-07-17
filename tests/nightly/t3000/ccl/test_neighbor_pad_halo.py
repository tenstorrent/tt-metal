# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# PCC test for the standalone halo-only op ttnn.experimental.neighbor_pad_halo.
#
# The op emits ONLY the compact halo buffer [H-top | H-bot | W-left | W-right] (no interior copy,
# no conv). We reuse the standalone neighbor_pad_async 2D golden (a full per-device padded tensor),
# then slice out exactly the halo bands in the compact-buffer stick order and compare byte-for-byte
# (bf16 copy, no arithmetic).

import os
import time
import torch
import pytest
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc

from ttnn import ShardTensor2dMesh
from tests.nightly.t3000.ccl.test_neighbor_pad_async import compute_2d_pad_golden


def _trace_and_time(mesh_device, run_op, num_iters=30):
    """Trace-replay wall/iter = device latency (untraced wall is host-dispatch-bound)."""
    run_op()  # warmup + cold compile
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    run_op()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(num_iters):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    us = (time.perf_counter() - t0) * 1e6 / num_iters
    ttnn.release_trace(mesh_device, tid)
    return us


def compact_halo_reference(golden, outer, H_dev, W_dev, pH, pW):
    """Build the expected compact halo buffer [total_sticks, C] from a per-device padded golden.

    golden: per-device tensor [B, T, H_dev+2pH, W_dev+2pW, C]. Sections + stick order match the
    program factory: H sections are W_dev wide (interior W only); W sections are h_total tall
    (include the corner/H-pad rows). Order within each section is (t, row, col), t-major.
    """
    C = golden.shape[-1]
    h_total = H_dev + 2 * pH
    g = golden.reshape(outer, H_dev + 2 * pH, W_dev + 2 * pW, C)
    sticks = []
    # H-top: pH rows above the chunk, interior W columns
    for t in range(outer):
        for pr in range(pH):
            for w in range(W_dev):
                sticks.append(g[t, pr, pW + w, :])
    # H-bot: pH rows below the chunk, interior W columns
    for t in range(outer):
        for pr in range(pH):
            for w in range(W_dev):
                sticks.append(g[t, pH + H_dev + pr, pW + w, :])
    # W-left: pW columns left of the chunk, full h_total rows (incl. corners)
    for t in range(outer):
        for hp in range(h_total):
            for wc in range(pW):
                sticks.append(g[t, hp, wc, :])
    # W-right: pW columns right of the chunk, full h_total rows (incl. corners)
    for t in range(outer):
        for hp in range(h_total):
            for wc in range(pW):
                sticks.append(g[t, hp, pW + W_dev + wc, :])
    return torch.stack(sticks, dim=0)  # [total_sticks, C]


def run_neighbor_pad_halo_2d(mesh_device, input_shape, h_dim, w_dim, h_axis, w_axis, pH, pW, padding_mode, num_links):
    mesh_shape = tuple(mesh_device.shape)
    h_factor = mesh_shape[h_axis]
    w_factor = mesh_shape[w_axis]
    assert input_shape[h_dim] % h_factor == 0
    assert input_shape[w_dim] % w_factor == 0

    torch.manual_seed(42)
    input_tensor = torch.rand(input_shape).bfloat16()
    goldens = compute_2d_pad_golden(input_tensor, mesh_shape, h_dim, w_dim, h_axis, w_axis, pH, pW, padding_mode)

    outer = 1
    for d in range(h_dim):
        outer *= input_shape[d]
    H_dev = input_shape[h_dim] // h_factor
    W_dev = input_shape[w_dim] // w_factor
    C = input_shape[-1]
    h_total = H_dev + 2 * pH
    total_sticks = outer * 2 * pH * W_dev + outer * 2 * pW * h_total

    # Sub-device + semaphores
    grid = mesh_device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    sub = ttnn.SubDevice([crs])
    sub_id = ttnn.SubDeviceId(0)
    mgr = mesh_device.create_sub_device_manager([sub], 0)
    mesh_device.load_sub_device_manager(mgr)
    mesh_device.set_sub_device_stall_group([sub_id])

    h_sem = ttnn.create_global_semaphore(mesh_device, crs, 0)
    w_sem = ttnn.create_global_semaphore(mesh_device, crs, 0)
    barrier_sem = ttnn.create_global_semaphore(mesh_device, crs, 0)

    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    dims = [None, None]
    dims[h_axis] = h_dim
    dims[w_axis] = w_dim
    input_tensor_mesh = ttnn.from_torch(
        input_tensor,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=mem_config,
        mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=dims),
    )
    # Compact halo buffer: per-device [total_sticks, C], replicated (each device owns one).
    halo_buf = ttnn.from_torch(
        torch.zeros([total_sticks, C]).bfloat16(),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=mem_config,
    )

    out = ttnn.experimental.neighbor_pad_halo(
        input_tensor_mesh,
        halo_buf,
        np_padding_h=pH,
        np_padding_w=pW,
        np_cluster_axis=h_axis,
        np_num_links=num_links,
        np_topology=ttnn.Topology.Linear,
        h_neighbor_semaphore=h_sem,
        barrier_semaphore=barrier_sem,
        w_neighbor_semaphore=w_sem,
        np_pad_dim2=w_dim,
        np_pad2_left=pW,
        np_pad2_right=pW,
        np_pad2_cluster_axis=w_axis,
        np_pad2_num_links=num_links,
        padding_mode=padding_mode,
    )
    ttnn.synchronize_device(mesh_device, sub_device_ids=[sub_id])

    out_host = ttnn.from_device(out)
    dev_tensors = ttnn.get_device_tensors(out_host)
    all_pass = True
    for row in range(mesh_shape[0]):
        for col in range(mesh_shape[1]):
            device_idx = row * mesh_shape[1] + col
            dev = ttnn.to_torch(dev_tensors[device_idx])
            ref = compact_halo_reference(goldens[(row, col)], outer, H_dev, W_dev, pH, pW)
            assert dev.shape == ref.shape, f"dev({row},{col}) shape {dev.shape} != ref {ref.shape}"
            eq, msg = comp_equal(dev, ref)
            if not eq:
                _, pcc = comp_pcc(dev, ref, 0.0)
                all_pass = False
                # Per-section diagnostic: which of [H-top | H-bot | W-left | W-right] mismatches.
                h_sec = outer * pH * W_dev
                w_sec = outer * (H_dev + 2 * pH) * pW
                bounds = [
                    ("Htop", 0, h_sec),
                    ("Hbot", h_sec, 2 * h_sec),
                    ("Wleft", 2 * h_sec, 2 * h_sec + w_sec),
                    ("Wright", 2 * h_sec + w_sec, 2 * h_sec + 2 * w_sec),
                ]
                secmsg = []
                for name, a, b in bounds:
                    if b <= dev.shape[0]:
                        seq, _ = comp_equal(dev[a:b], ref[a:b])
                        secmsg.append(f"{name}={'OK' if seq else 'BAD'}")
                print(f"FAIL dev({row},{col}): {pcc} | sections: {' '.join(secmsg)}")
                if os.environ.get("NP_DIAG"):
                    import torch as _t

                    h_sec = outer * pH * W_dev
                    for nm, a, b in [("Htop", 0, h_sec), ("Hbot", h_sec, 2 * h_sec)]:
                        sub_d, sub_r = dev[a:b], ref[a:b]
                        bad = (~(sub_d == sub_r).all(dim=1)).nonzero().flatten().tolist()
                        if not bad:
                            continue
                        frames = sorted({i // (pH * W_dev) for i in bad})
                        cols = sorted({(i % (pH * W_dev)) % W_dev for i in bad})
                        print(
                            f"  {nm} bad_sticks={len(bad)}/{b-a} frames={frames[:8]}(n={len(frames)}/{outer}) "
                            f"cols={cols}(n={len(cols)}/{W_dev}) banks={sorted({c%8 for c in cols})}"
                        )
                    # W sections: rows are [frame][h_row (0..H_dev+2pH)][pw]; flag which h_rows are bad and
                    # whether they are corner rows (h_row < pH or >= pH+H_dev) — corners come from the H exchange.
                    hh = H_dev + 2 * pH
                    for nm, a, b in [
                        ("Wleft", 2 * h_sec, 2 * h_sec + w_sec),
                        ("Wright", 2 * h_sec + w_sec, 2 * h_sec + 2 * w_sec),
                    ]:
                        sub_d, sub_r = dev[a:b], ref[a:b]
                        bad = (~(sub_d == sub_r).all(dim=1)).nonzero().flatten().tolist()
                        if not bad:
                            continue
                        hrows = sorted({(i // pW) % hh for i in bad})
                        corners = [r for r in hrows if r < pH or r >= pH + H_dev]
                        wframes = sorted({i // (hh * pW) for i in bad})
                        print(
                            f"  {nm} bad={len(bad)}/{b-a} frames={wframes[:8]}(n={len(wframes)}/{outer}) "
                            f"h_rows={hrows}(corner_rows={corners})"
                        )
            else:
                print(f"PASS dev({row},{col})")

    mesh_device.reset_sub_device_stall_group()
    mesh_device.clear_loaded_sub_device_manager()
    assert all_pass, "compact halo mismatch"


@pytest.mark.timeout(300)
@pytest.mark.parametrize("mesh_device", [(4, 8)], ids=["4x8"], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize(
    "input_shape",
    [[1, 8, 144, 256, 128], [1, 1, 92, 160, 128], [1, 2, 92, 160, 128]],
    ids=["s3ish", "conv0ish", "conv0ish_t2"],
)
def test_neighbor_pad_halo_fold(mesh_device, device_params, input_shape):
    """Full fold (deletes halo_scatter): neighbor_pad_halo with padded_output writes the ENTIRE padded
    [outer,H+2pH,W+2pW,C] buffer itself — interior overlaps the exchange on free cores, border scatters
    after the W-readers signal compact_ready. The whole padded buffer must match neighbor_pad_async's
    full-pad output bit-exact, in ONE op (no separate halo_scatter)."""
    h_dim, w_dim, h_axis, w_axis, pH, pW, num_links = 2, 3, 0, 1, 1, 1, 2
    mesh_shape = tuple(mesh_device.shape)
    hf, wf = mesh_shape[h_axis], mesh_shape[w_axis]
    B, T, Hf, Wf, C = input_shape
    Hd, Wd = Hf // hf, Wf // wf
    torch.manual_seed(0)
    inp = torch.rand(input_shape).bfloat16()

    grid = mesh_device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    mgr = mesh_device.create_sub_device_manager([ttnn.SubDevice([crs])], 0)
    mesh_device.load_sub_device_manager(mgr)
    sub_id = ttnn.SubDeviceId(0)
    mesh_device.set_sub_device_stall_group([sub_id])
    h_sem = ttnn.create_global_semaphore(mesh_device, crs, 0)
    w_sem = ttnn.create_global_semaphore(mesh_device, crs, 0)
    barrier_sem = ttnn.create_global_semaphore(mesh_device, crs, 0)
    mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    dims = [None, None]
    dims[h_axis], dims[w_axis] = h_dim, w_dim
    inp_mesh = ttnn.from_torch(
        inp,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=mem,
        mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=dims),
    )
    out_shape = list(input_shape)
    out_shape[h_dim] += hf * 2 * pH
    out_shape[w_dim] += wf * 2 * pW
    persist = ttnn.from_torch(
        torch.zeros(out_shape).bfloat16(),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=mem,
        mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=dims),
    )
    logical_h = int(os.environ.get("FOLD_LOGICAL_H", "0"))
    full = ttnn.experimental.neighbor_pad_async(
        inp_mesh,
        [h_dim, w_dim],
        [pH, pW],
        [pH, pW],
        "zeros",
        [h_axis, w_axis],
        [h_sem, w_sem],
        [barrier_sem],
        num_links=[min(B * T, num_links), min(B * T * Hf, num_links)],
        memory_config=mem,
        topology=ttnn.Topology.Linear,
        persistent_output_buffer=persist,
        logical_h=logical_h,
    )
    ttnn.synchronize_device(mesh_device, sub_device_ids=[sub_id])
    # Compact halo scratch (op writes it internally) + padded output (op fills interior + border).
    h_total = Hd + 2 * pH
    total_sticks = T * 2 * pH * Wd + T * 2 * pW * h_total
    compact = ttnn.from_torch(
        torch.zeros(total_sticks, C).bfloat16(),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=mem,
    )
    padded = ttnn.from_torch(
        torch.zeros(out_shape).bfloat16(),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=mem,
        mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=dims),
    )
    # Single fused op: exchange + full padded write (interior overlap + border fold).
    ttnn.experimental.neighbor_pad_halo(
        inp_mesh,
        compact,
        np_padding_h=pH,
        np_padding_w=pW,
        np_cluster_axis=h_axis,
        np_num_links=min(B * T, num_links),
        np_topology=ttnn.Topology.Linear,
        h_neighbor_semaphore=h_sem,
        barrier_semaphore=barrier_sem,
        w_neighbor_semaphore=w_sem,
        np_pad_dim2=w_dim,
        np_pad2_left=pW,
        np_pad2_right=pW,
        np_pad2_cluster_axis=w_axis,
        np_pad2_num_links=min(B * T * Hf, num_links),
        padding_mode="zeros",
        padded_output=padded,
        logical_h=logical_h,
    )
    ttnn.synchronize_device(mesh_device, sub_device_ids=[sub_id])

    full_dev = ttnn.get_device_tensors(ttnn.from_device(full))
    pad_dev = ttnn.get_device_tensors(ttnn.from_device(padded))

    all_pass = True
    for i in range(mesh_shape[0] * mesh_shape[1]):
        f = ttnn.to_torch(full_dev[i]).reshape(T, Hd + 2 * pH, Wd + 2 * pW, C)
        p = ttnn.to_torch(pad_dev[i]).reshape(T, Hd + 2 * pH, Wd + 2 * pW, C)
        eq, msg = comp_equal(p, f)
        if not eq:
            all_pass = False
            _, pcc = comp_pcc(p, f, 0.0)
            # Per-section breakdown: interior vs H-border (top/bot rows) vs W-border (left/right cols).
            inte_f, inte_p = f[:, pH : pH + Hd, pW : pW + Wd, :], p[:, pH : pH + Hd, pW : pW + Wd, :]
            htop_f, htop_p = f[:, :pH, :, :], p[:, :pH, :, :]
            hbot_f, hbot_p = f[:, pH + Hd :, :, :], p[:, pH + Hd :, :, :]
            wl_f, wl_p = f[:, pH : pH + Hd, :pW, :], p[:, pH : pH + Hd, :pW, :]
            wr_f, wr_p = f[:, pH : pH + Hd, pW + Wd :, :], p[:, pH : pH + Hd, pW + Wd :, :]
            _, pi = comp_pcc(inte_p, inte_f, 0.0)
            _, pt = comp_pcc(htop_p, htop_f, 0.0)
            _, pb = comp_pcc(hbot_p, hbot_f, 0.0)
            _, pwl = comp_pcc(wl_p, wl_f, 0.0)
            _, pwr = comp_pcc(wr_p, wr_f, 0.0)
            print(
                f"FAIL dev {i}: fold pcc={pcc} | interior={pi} htop={pt} hbot={pb} wleft={pwl} wright={pwr}",
                flush=True,
            )
        else:
            print(f"PASS dev {i}", flush=True)
    mesh_device.reset_sub_device_stall_group()
    mesh_device.clear_loaded_sub_device_manager()
    assert all_pass, "folded padded buffer != full-pad output"


@pytest.mark.timeout(300)
@pytest.mark.parametrize("mesh_device", [(4, 8)], ids=["4x8"], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("input_shape", [[1, 8, 144, 256, 128]], ids=["s3ish"])
def test_neighbor_pad_halo_strided_input(mesh_device, device_params, input_shape):
    """Copy-free primitive: neighbor_pad_halo reading a PADDED input's INTERIOR (input_pad_h/w) must
    produce the same compact halo as reading the equivalent contiguous input (interior == unpadded).
    Runs on the default (mux) path — its recv-authority H->W barrier makes the corner two-hop race-free,
    so the FULL compact (incl. corners) is bit-exact. (The non-mux send-done barrier can race the corner
    under padded timing — a pre-existing non-mux weakness; the production decode uses mux.)"""
    h_dim, w_dim, h_axis, w_axis, pH, pW, num_links = 2, 3, 0, 1, 1, 1, 2
    mesh_shape = tuple(mesh_device.shape)
    hf, wf = mesh_shape[h_axis], mesh_shape[w_axis]
    B, T, Hf, Wf, C = input_shape
    Hd, Wd = Hf // hf, Wf // wf
    torch.manual_seed(0)
    inp = torch.rand(input_shape).bfloat16()

    grid = mesh_device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    mgr = mesh_device.create_sub_device_manager([ttnn.SubDevice([crs])], 0)
    mesh_device.load_sub_device_manager(mgr)
    sub_id = ttnn.SubDeviceId(0)
    mesh_device.set_sub_device_stall_group([sub_id])
    h_sem = ttnn.create_global_semaphore(mesh_device, crs, 0)
    w_sem = ttnn.create_global_semaphore(mesh_device, crs, 0)
    barrier_sem = ttnn.create_global_semaphore(mesh_device, crs, 0)
    mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    dims = [None, None]
    dims[h_axis], dims[w_axis] = h_dim, w_dim

    h_total = Hd + 2 * pH
    total_sticks = T * 2 * pH * Wd + T * 2 * pW * h_total
    h_sec_sticks = T * 2 * pH * Wd  # Htop|Hbot span at the front of the compact buffer

    def run(x_mesh, ipad_h, ipad_w):
        compact = ttnn.from_torch(
            torch.zeros(total_sticks, C).bfloat16(),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=mem,
        )
        ttnn.experimental.neighbor_pad_halo(
            x_mesh,
            compact,
            np_padding_h=pH,
            np_padding_w=pW,
            np_cluster_axis=h_axis,
            np_num_links=num_links,
            np_topology=ttnn.Topology.Linear,
            h_neighbor_semaphore=h_sem,
            barrier_semaphore=barrier_sem,
            w_neighbor_semaphore=w_sem,
            np_pad_dim2=w_dim,
            np_pad2_left=pW,
            np_pad2_right=pW,
            np_pad2_cluster_axis=w_axis,
            np_pad2_num_links=num_links,
            padding_mode="zeros",
            input_pad_h=ipad_h,
            input_pad_w=ipad_w,
        )
        ttnn.synchronize_device(mesh_device, sub_device_ids=[sub_id])
        return ttnn.get_device_tensors(ttnn.from_device(compact))

    # Reference: contiguous input.
    inp_mesh = ttnn.from_torch(
        inp,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=mem,
        mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=dims),
    )
    ref = run(inp_mesh, 0, 0)

    # Padded input: each device's [Hd,Wd] shard padded to [Hd+2,Wd+2] (interior == its shard, border 0),
    # tiled into a global [.,(Hd+2)*hf,(Wd+2)*wf,.] so the 2D shard hands each device its padded block.
    xp = torch.zeros(B, T, h_total * hf, (Wd + 2 * pW) * wf, C).bfloat16()
    for hi in range(hf):
        for wi in range(wf):
            dev = inp[:, :, hi * Hd : (hi + 1) * Hd, wi * Wd : (wi + 1) * Wd, :]
            xp[
                :,
                :,
                hi * h_total + pH : hi * h_total + pH + Hd,
                wi * (Wd + 2 * pW) + pW : wi * (Wd + 2 * pW) + pW + Wd,
                :,
            ] = dev
    xp_mesh = ttnn.from_torch(
        xp,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=mem,
        mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=dims),
    )
    test = run(xp_mesh, pH, pW)

    all_pass = True
    for i in range(mesh_shape[0] * mesh_shape[1]):
        r = ttnn.to_torch(ref[i])
        t = ttnn.to_torch(test[i])
        eq, _ = comp_equal(r, t)
        if not eq:
            all_pass = False
            _, pcc = comp_pcc(r, t, 0.0)
            print(f"FAIL dev {i}: full compact pcc={pcc}", flush=True)
        else:
            print(f"PASS dev {i}", flush=True)
    mesh_device.reset_sub_device_stall_group()
    mesh_device.clear_loaded_sub_device_manager()
    assert all_pass, "strided-input compact != contiguous-input compact"


@pytest.mark.timeout(180)
@pytest.mark.parametrize("mesh_device", [(2, 4)], ids=["2x4"], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("padding_mode", ["zeros", "replicate"])
@pytest.mark.parametrize(
    "input_shape",
    [
        [1, 2, 8, 16, 16],  # per-stick W path (wright_base not 8-aligned -> W_COALESCE off)
        [1, 4, 16, 32, 16],  # coalesce-eligible, 8-aligned rows + W_dev
        [1, 3, 16, 24, 16],  # coalesce, NON-8-aligned rows (w_outer=30) + W_dev (6): exercises the relaxed gate
        [1, 7, 32, 64, 32],  # mid-size, NON-8-aligned rows (63/link), W_dev=16 (s4-like ratio)
        [1, 7, 32, 48, 32],  # mid-size, NON-8-aligned rows + W_dev=12 (s3-like non-aligned W)
        [1, 64, 32, 64, 32],  # scale bracket: T=64 at small H/W (fast golden) to repro production-scale race
        [1, 32, 272, 480, 128],  # s4 geometry at T=32 (the perf shape) — large H/W, moderate T
    ],
    ids=["perstick", "coalesce", "coalesce_nonalign", "mid_wdev16", "mid_wdev12", "scale_T64", "s4_T32"],
)
def test_neighbor_pad_halo_2d(mesh_device, device_params, padding_mode, input_shape):
    # [B, T, H, W, C]; H sharded over axis 0 (2 dev), W over axis 1 (4 dev). k333 halo (pH=pW=1).
    run_neighbor_pad_halo_2d(
        mesh_device,
        input_shape=input_shape,
        h_dim=2,
        w_dim=3,
        h_axis=0,
        w_axis=1,
        pH=1,
        pW=1,
        padding_mode=padding_mode,
        num_links=1,
    )


def _device_fw_us(mesh_device, run_op):
    """Op-level device-FW duration (max core, max over devices), excluding the fixed per-program trace/
    dispatch launch floor that trace-wall includes. Requires TT_METAL_DEVICE_PROFILER=1 (+ mid-run dump,
    cpp post-process). Units are whatever the profiler reports; only used for halo-vs-async ratios."""
    run_op()
    ttnn.synchronize_device(mesh_device)
    ttnn.ReadDeviceProfiler(mesh_device)
    latest = ttnn.get_latest_programs_perf_data()
    best = 0.0
    for dev_id, programs in (latest or {}).items():
        for program in programs:
            for _name, res in program.program_analyses_results.items():
                best = max(best, float(res.duration))
    return best


def run_halo_vs_async_perf(mesh_device, input_shape, h_dim, w_dim, h_axis, w_axis, pH, pW, num_links):
    """Trace-timed device latency: halo-only op vs full-pad neighbor_pad_async on the same input."""
    mesh_shape = tuple(mesh_device.shape)
    h_factor, w_factor = mesh_shape[h_axis], mesh_shape[w_axis]
    torch.manual_seed(0)
    inp = torch.rand(input_shape).bfloat16()

    outer = 1
    for d in range(h_dim):
        outer *= input_shape[d]
    H_dev, W_dev, C = input_shape[h_dim] // h_factor, input_shape[w_dim] // w_factor, input_shape[-1]
    h_total = H_dev + 2 * pH
    total_sticks = outer * 2 * pH * W_dev + outer * 2 * pW * h_total

    grid = mesh_device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    mgr = mesh_device.create_sub_device_manager([ttnn.SubDevice([crs])], 0)
    mesh_device.load_sub_device_manager(mgr)
    sub_id = ttnn.SubDeviceId(0)
    mesh_device.set_sub_device_stall_group([sub_id])

    h_sem = ttnn.create_global_semaphore(mesh_device, crs, 0)
    w_sem = ttnn.create_global_semaphore(mesh_device, crs, 0)
    barrier_sem = ttnn.create_global_semaphore(mesh_device, crs, 0)

    mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    dims = [None, None]
    dims[h_axis], dims[w_axis] = h_dim, w_dim
    inp_mesh = ttnn.from_torch(
        inp,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=mem,
        mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=dims),
    )
    halo_buf = ttnn.from_torch(
        torch.zeros([total_sticks, C]).bfloat16(),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=mem,
    )
    # persistent full-pad output for the async op (needed for tracing)
    out_shape = list(input_shape)
    out_shape[h_dim] += h_factor * 2 * pH
    out_shape[w_dim] += w_factor * 2 * pW
    persist = ttnn.from_torch(
        torch.zeros(out_shape).bfloat16(),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=mem,
        mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=dims),
    )

    # Fold mode (NP_FOLD): the halo op writes the FULL padded output too, so it is apples-to-apples with
    # neighbor_pad_async (both produce the same [.,H+2pH,W+2pW,.] buffer). Default is compact (transport only).
    fold = bool(os.environ.get("NP_FOLD"))
    halo_padded = None
    if fold:
        halo_padded = ttnn.from_torch(
            torch.zeros(out_shape).bfloat16(),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=mem,
            mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=dims),
        )

    def run_halo():
        return ttnn.experimental.neighbor_pad_halo(
            inp_mesh,
            halo_buf,
            np_padding_h=pH,
            np_padding_w=pW,
            np_cluster_axis=h_axis,
            np_num_links=num_links,
            np_topology=ttnn.Topology.Linear,
            h_neighbor_semaphore=h_sem,
            barrier_semaphore=barrier_sem,
            w_neighbor_semaphore=w_sem,
            np_pad_dim2=w_dim,
            np_pad2_left=pW,
            np_pad2_right=pW,
            np_pad2_cluster_axis=w_axis,
            np_pad2_num_links=num_links,
            padding_mode="zeros",
            **({"padded_output": halo_padded} if fold else {}),
        )

    def run_async():
        return ttnn.experimental.neighbor_pad_async(
            inp_mesh,
            [h_dim, w_dim],
            [pH, pW],
            [pH, pW],
            "zeros",
            [h_axis, w_axis],
            [h_sem, w_sem],
            [barrier_sem],
            num_links=[num_links, num_links],
            memory_config=mem,
            topology=ttnn.Topology.Linear,
            persistent_output_buffer=persist,
        )

    if os.environ.get("NP_DEVTIME"):
        # Op-level device-FW ratio: isolates op work from the fixed per-program launch floor that trace-wall
        # includes (that floor is identical for both ops and overlaps when the op runs inside the decode trace).
        h_dev = _device_fw_us(mesh_device, run_halo)
        a_dev = _device_fw_us(mesh_device, run_async)
        print(f"\n=== DEVTIME shape={input_shape} outer={outer} 2x4 ===")
        print(f"  async device-fw: {a_dev:.1f}  halo device-fw: {h_dev:.1f}  ratio: {a_dev/max(h_dev,1e-9):.2f}x")
        mesh_device.reset_sub_device_stall_group()
        mesh_device.clear_loaded_sub_device_manager()
        return

    halo_us = _trace_and_time(mesh_device, run_halo)
    async_us = _trace_and_time(mesh_device, run_async)
    # Effective halo-transport bandwidth vs the ~50 GB/s aggregate 2-link fabric ceiling
    # (12.5 GB/s/link/dir x 2 links x 2 dir). Bytes = the essential halo crossing the fabric per device:
    # one H-edge (W_dev sticks) + one W-edge (H_dev sticks) per frame. The compact BUFFER is ~2x this and
    # also holds edge zero-fill that never leaves the chip, so buffer-size/time overstates bandwidth ~4x.
    transfer_bytes = outer * (H_dev + W_dev) * C * 2  # bf16, minimal halo transport per device
    gbps = transfer_bytes / (halo_us * 1e-6) / 1e9
    print(f"\n=== PERF (trace wall/iter, device latency) shape={input_shape} outer={outer} mesh={mesh_shape} ===")
    print(f"  neighbor_pad_async (full-pad):     {async_us:8.1f} us")
    print(f"  neighbor_pad_halo  ({'fold/full-pad' if fold else 'compact     '}): {halo_us:8.1f} us")
    print(f"  speedup: {async_us / halo_us:.2f}x")
    print(f"  halo transfer/device: {transfer_bytes/1e6:.2f} MB;  achieved: {gbps:.1f} GB/s ({gbps/50*100:.0f}% of 50)")

    mesh_device.reset_sub_device_stall_group()
    mesh_device.clear_loaded_sub_device_manager()


@pytest.mark.timeout(400)
@pytest.mark.parametrize("mesh_device", [(2, 4)], ids=["2x4"], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112 * 64}], indirect=True
)
@pytest.mark.parametrize("T", [8, 32, 96], ids=["T8", "T32", "T96"])
def test_neighbor_pad_halo_perf(mesh_device, device_params, T):
    # s4_out-like at varying T: outer=T, per-device H=136, W=120, C=128, k333 halo. Small T is
    # barrier-dominated; large T shifts the op toward bandwidth-bound (fixed barriers, scaling data).
    # num_links is HARDWARE-CAPPED at 2 on BH-LB: each inter-chip hop has exactly 2 ethernet channels
    # (num_links=4 => TT_FATAL "Requested link index 2 is out of bounds. 2 ethernet channels available").
    # The op already uses both, so "more fabric links" is not a lever here.
    run_halo_vs_async_perf(
        mesh_device, input_shape=[1, T, 272, 480, 128], h_dim=2, w_dim=3, h_axis=0, w_axis=1, pH=1, pW=1, num_links=2
    )


# Production LTX 1080p 2x4 decoder NP-bound layers, as full [B,T,H,W,C] (per-device = H/2, W/4). All k333
# (pH=pW=1). Verifies the mux speedup holds on the real deployed shapes, not just the synthetic sweep.
# All distinct LTX VAE-decoder NP inputs routed on the 2x4 mesh (kernel (3,3,3) => pH=pW=1). Collapsed
# from models/tt_dit/utils/conv3d.py LTX table (lines ~414-429): 10 conv sites -> 6 unique (T, H, W, C_in).
# H/W here are the FULL spatial dims (H = per-dev*2 on axis-0, W = per-dev*4 on axis-1).
_LTX_PROD_2x4 = [
    ([1, 21, 34, 60, 128], "s0_conv_in_C128"),  # per-dev  17x15,  C_in=128
    ([1, 21, 34, 60, 1024], "s0_res_up_C1024"),  # per-dev  17x15,  C_in=1024 (2048B page, largest)
    ([1, 39, 68, 120, 512], "s1_res_up_C512"),  # per-dev  34x30,  C_in=512
    ([1, 75, 136, 240, 512], "s2_res_C512"),  # per-dev  68x60,  C_in=512
    ([1, 147, 136, 240, 256], "s3_res_chg_C256"),  # per-dev  68x60,  C_in=256 (s3_res / s3_chg)
    ([1, 147, 272, 480, 128], "s4_res_out_C128"),  # per-dev 136x120, C_in=128 (s4_res / s4_out)
]

# 4x8 physical shapes (h_factor=4, w_factor=8), CAPTURED live from the all-standalone decode
# (prof_vae_ltx NP_CAPTURE_SHAPES, LTX_USE_FUSED=0). The latent 34x60 pads once to 36x64 at the decode
# input (34->36 div4=9, 60->64 div8=8) and that padding PROPAGATES through the x2 upsamples, so every
# stage's physical dims are the per-device shard x factor: 36x64 -> 72x128 -> 144x256 -> 288x512 (the
# masked pad region is carried, not re-trimmed). These differ from _LTX_PROD_2x4's logical dims. The
# trailing comment is the per-call count in one decode (relative op-time weight).
_LTX_PROD_4x8 = [
    ([1, 21, 36, 64, 128], "s0_conv_in_C128"),  # per-dev  9x8,   C_in=128    (x1)
    ([1, 21, 36, 64, 1024], "s0_res_up_C1024"),  # per-dev  9x8,   C_in=1024   (x5, 2048B page, largest)
    ([1, 39, 72, 128, 512], "s1_res_up_C512"),  # per-dev 18x16,  C_in=512    (x5)
    ([1, 75, 144, 256, 512], "s2_res_C512"),  # per-dev 36x32,  C_in=512    (x9)
    ([1, 147, 144, 256, 256], "s3_res_chg_C256"),  # per-dev 36x32,  C_in=256    (x13, s3_res / s3_chg)
    ([1, 147, 288, 512, 128], "s4_res_out_C128"),  # per-dev 72x64,  C_in=128    (x9, s4_res / s4_out)
]


@pytest.mark.timeout(600)
@pytest.mark.parametrize("mesh_device", [(4, 8)], ids=["4x8"], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112 * 64}], indirect=True
)
@pytest.mark.parametrize("input_shape, shape_id", _LTX_PROD_4x8, ids=[s[1] for s in _LTX_PROD_4x8])
def test_neighbor_pad_halo_prod_perf(mesh_device, device_params, input_shape, shape_id):
    run_halo_vs_async_perf(
        mesh_device, input_shape=input_shape, h_dim=2, w_dim=3, h_axis=0, w_axis=1, pH=1, pW=1, num_links=2
    )


# WAN 2.2 VAE decoder NP inputs, captured live from test_wan_decoder_production_blocking[bh_4x8_h0_w1-480p_t7]
# (NP_LOG_SHAPES). Captured shapes were PER-DEVICE (H_dev,W_dev); these are the GLOBAL shapes (H_dev*4, W_dev*8)
# the perf harness shards back to per-device on the 4x8 mesh. All k333 (pH=pW=1).
_WAN_PROD_4x8 = [
    ([1, 9, 60, 104, 32], "wan_15x13_C32"),  # per-dev 15x13, from [1,9,15,13,32]
    ([1, 9, 60, 104, 384], "wan_15x13_C384"),  # per-dev 15x13
    ([1, 15, 120, 208, 192], "wan_30x26_C192"),  # per-dev 30x26
    ([1, 15, 120, 208, 384], "wan_30x26_C384"),  # per-dev 30x26
    ([1, 27, 240, 416, 192], "wan_60x52_C192"),  # per-dev 60x52
    ([1, 27, 480, 832, 96], "wan_120x104_C96"),  # per-dev 120x104
]


@pytest.mark.timeout(600)
@pytest.mark.parametrize("mesh_device", [(4, 8)], ids=["4x8"], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112 * 64}], indirect=True
)
@pytest.mark.parametrize("input_shape, shape_id", _WAN_PROD_4x8, ids=[s[1] for s in _WAN_PROD_4x8])
def test_neighbor_pad_halo_wan_perf(mesh_device, device_params, input_shape, shape_id):
    run_halo_vs_async_perf(
        mesh_device, input_shape=input_shape, h_dim=2, w_dim=3, h_axis=0, w_axis=1, pH=1, pW=1, num_links=2
    )


# Same shapes, but with an 8 KB fabric packet payload (default is 4352 B). The factory sizes its coalesce
# to the fabric max payload, so this ships ~2x the sticks per packet — tests whether packet count (fabric
# forwarding overhead) is a bound on the mid/large shapes. FabricRouterConfig has no kwargs ctor; set the
# field after default construction.
def _fabric_router_config_8k():
    frc = ttnn.FabricRouterConfig()
    frc.max_packet_payload_size_bytes = int(os.environ.get("NP_FABRIC_PAYLOAD", "8192"))  # BH max 15232
    return frc


@pytest.mark.timeout(600)
@pytest.mark.parametrize("mesh_device", [(4, 8)], ids=["4x8"], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 90112 * 64,
            "fabric_router_config": _fabric_router_config_8k(),
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("input_shape, shape_id", _LTX_PROD_4x8, ids=[s[1] for s in _LTX_PROD_4x8])
def test_neighbor_pad_halo_prod_perf_8k(mesh_device, device_params, input_shape, shape_id):
    run_halo_vs_async_perf(
        mesh_device, input_shape=input_shape, h_dim=2, w_dim=3, h_axis=0, w_axis=1, pH=1, pW=1, num_links=2
    )


# Byte-exact PCC with the 8 KB fabric payload: the coalesce forms larger (up to 32-stick) bank packets, so
# this guards that the bigger-packet path is still exact.
@pytest.mark.timeout(300)
@pytest.mark.parametrize("mesh_device", [(4, 8)], ids=["4x8"], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "fabric_router_config": _fabric_router_config_8k()}],
    indirect=True,
)
@pytest.mark.parametrize("input_shape, shape_id", _LTX_PROD_4x8, ids=[s[1] for s in _LTX_PROD_4x8])
def test_neighbor_pad_halo_prod_pcc_8k(mesh_device, device_params, input_shape, shape_id):
    run_neighbor_pad_halo_2d(
        mesh_device,
        input_shape=input_shape,
        h_dim=2,
        w_dim=3,
        h_axis=0,
        w_axis=1,
        pH=1,
        pW=1,
        padding_mode="zeros",
        num_links=2,
    )


# Scaling to longer cluster axes than 2x4 (a 4x8 mesh needs 32 chips; on an 8-chip BH-LB the runnable proxy
# is 4x2 = H-axis length 4 with MIDDLE devices, the case never exercised at 2x4 where both H devices are
# edges). H div by 4, W div by 2. Validates the 1-hop neighbor exchange + startup barrier on a >2 axis.
@pytest.mark.timeout(300)
@pytest.mark.parametrize("mesh_device", [(4, 2)], ids=["4x2"], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("padding_mode", ["zeros", "replicate"])
@pytest.mark.parametrize("input_shape", [[1, 32, 136, 240, 128]], ids=["Hx4_mid"])
def test_neighbor_pad_halo_4x2(mesh_device, device_params, padding_mode, input_shape):
    run_neighbor_pad_halo_2d(
        mesh_device,
        input_shape=input_shape,
        h_dim=2,
        w_dim=3,
        h_axis=0,
        w_axis=1,
        pH=1,
        pW=1,
        padding_mode=padding_mode,
        num_links=2,
    )


# LTX-2.3 spatial latent upsampler (x2) on 2x4 (conv3d.py "spatial latent upsampler" block). Small spatial
# (per-dev down to 9x8) and one k=(1,3,3) site — still spatial pH=pW=1. Not covered by the decoder set above.
_LTX_UPSAMPLER_2x4 = [
    ([1, 21, 18, 32, 128], "ups_initial_C128"),  # per-dev 9x8
    ([1, 21, 18, 32, 1024], "ups_pre_res_C1024"),  # per-dev 9x8
    ([1, 19, 18, 32, 1024], "ups_ups_C1024_k133"),  # per-dev 9x8, k=(1,3,3) -> spatial pH=pW=1
    ([1, 21, 36, 64, 1024], "ups_post_res_C1024"),  # per-dev 18x16
]


@pytest.mark.timeout(400)
@pytest.mark.parametrize("mesh_device", [(2, 4)], ids=["2x4"], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("padding_mode", ["zeros", "replicate"])
@pytest.mark.parametrize("input_shape, shape_id", _LTX_UPSAMPLER_2x4, ids=[s[1] for s in _LTX_UPSAMPLER_2x4])
def test_neighbor_pad_halo_upsampler_pcc(mesh_device, device_params, padding_mode, input_shape, shape_id):
    run_neighbor_pad_halo_2d(
        mesh_device,
        input_shape=input_shape,
        h_dim=2,
        w_dim=3,
        h_axis=0,
        w_axis=1,
        pH=1,
        pW=1,
        padding_mode=padding_mode,
        num_links=2,
    )


@pytest.mark.timeout(600)
@pytest.mark.parametrize("mesh_device", [(4, 8)], ids=["4x8"], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("padding_mode", ["zeros", "replicate"])
@pytest.mark.parametrize("input_shape, shape_id", _LTX_PROD_4x8, ids=[s[1] for s in _LTX_PROD_4x8])
def test_neighbor_pad_halo_prod_pcc(mesh_device, device_params, padding_mode, input_shape, shape_id):
    # Byte-exact PCC of the compact halo buffer on the production LTX 4x8 shapes (the mux path for zeros).
    run_neighbor_pad_halo_2d(
        mesh_device,
        input_shape=input_shape,
        h_dim=2,
        w_dim=3,
        h_axis=0,
        w_axis=1,
        pH=1,
        pW=1,
        padding_mode=padding_mode,
        num_links=2,
    )


@pytest.mark.timeout(300)
@pytest.mark.parametrize("mesh_device", [(2, 4)], ids=["2x4"], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_neighbor_pad_halo_devfw(mesh_device, device_params):
    # Non-traced dispatch (device-FW profiled via tracy) to find the bound (H cores vs W cores).
    input_shape, h_dim, w_dim, h_axis, w_axis, pH, pW, num_links = [1, 8, 272, 480, 128], 2, 3, 0, 1, 1, 1, 2
    mesh_shape = tuple(mesh_device.shape)
    h_factor, w_factor = mesh_shape[h_axis], mesh_shape[w_axis]
    torch.manual_seed(0)
    inp = torch.rand(input_shape).bfloat16()
    outer = input_shape[0] * input_shape[1]
    H_dev, W_dev, C = input_shape[h_dim] // h_factor, input_shape[w_dim] // w_factor, input_shape[-1]
    total_sticks = outer * 2 * pH * W_dev + outer * 2 * pW * (H_dev + 2 * pH)
    grid = mesh_device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    mgr = mesh_device.create_sub_device_manager([ttnn.SubDevice([crs])], 0)
    mesh_device.load_sub_device_manager(mgr)
    sub_id = ttnn.SubDeviceId(0)
    mesh_device.set_sub_device_stall_group([sub_id])
    h_sem = ttnn.create_global_semaphore(mesh_device, crs, 0)
    w_sem = ttnn.create_global_semaphore(mesh_device, crs, 0)
    barrier_sem = ttnn.create_global_semaphore(mesh_device, crs, 0)
    mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    dims = [None, None]
    dims[h_axis], dims[w_axis] = h_dim, w_dim
    inp_mesh = ttnn.from_torch(
        inp,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=mem,
        mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=dims),
    )
    halo_buf = ttnn.from_torch(
        torch.zeros([total_sticks, C]).bfloat16(),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=mem,
    )
    for _ in range(6):
        ttnn.experimental.neighbor_pad_halo(
            inp_mesh,
            halo_buf,
            np_padding_h=pH,
            np_padding_w=pW,
            np_cluster_axis=h_axis,
            np_num_links=num_links,
            np_topology=ttnn.Topology.Linear,
            h_neighbor_semaphore=h_sem,
            barrier_semaphore=barrier_sem,
            w_neighbor_semaphore=w_sem,
            np_pad_dim2=w_dim,
            np_pad2_left=pW,
            np_pad2_right=pW,
            np_pad2_cluster_axis=w_axis,
            np_pad2_num_links=num_links,
            padding_mode="zeros",
        )
    ttnn.synchronize_device(mesh_device, sub_device_ids=[sub_id])
    mesh_device.reset_sub_device_stall_group()
    mesh_device.clear_loaded_sub_device_manager()


# Halo-aware conv3d correctness: the compact two-dispatch (neighbor_pad_halo -> conv3d halo mode) must
# equal the current full-pad two-dispatch (neighbor_pad_async -> conv3d) on device. Same weight/input;
# both compute conv on the neighbor-padded input, so a device-to-device match proves the halo read.
def run_conv3d_halo_vs_fullpad_2d(mesh_device, input_shape, C_out, kernel_size, pH, pW, num_links=2):
    h_dim, w_dim, h_axis, w_axis = 2, 3, 0, 1
    mesh_shape = tuple(mesh_device.shape)
    h_factor, w_factor = mesh_shape[h_axis], mesh_shape[w_axis]
    B, T, Hf, Wf, C = input_shape
    kT, kH, kW = kernel_size
    pT = kT // 2
    torch.manual_seed(0)
    inp = torch.rand(input_shape).bfloat16()
    H_dev, W_dev = Hf // h_factor, Wf // w_factor
    outer = B * T
    total_sticks = outer * 2 * pH * W_dev + outer * 2 * pW * (H_dev + 2 * pH)

    grid = mesh_device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    mgr = mesh_device.create_sub_device_manager([ttnn.SubDevice([crs])], 0)
    mesh_device.load_sub_device_manager(mgr)
    sub_id = ttnn.SubDeviceId(0)
    mesh_device.set_sub_device_stall_group([sub_id])
    h_sem = ttnn.create_global_semaphore(mesh_device, crs, 0)
    w_sem = ttnn.create_global_semaphore(mesh_device, crs, 0)
    barrier_sem = ttnn.create_global_semaphore(mesh_device, crs, 0)
    mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    dims = [None, None]
    dims[h_axis], dims[w_axis] = h_dim, w_dim
    inp_mesh = ttnn.from_torch(
        inp,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=mem,
        mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=dims),
    )

    w = torch.rand(C_out, C, kT, kH, kW).bfloat16()
    tt_w = ttnn.from_torch(w, dtype=ttnn.bfloat16, pad_value=0)
    config = ttnn.Conv3dConfig(
        weights_dtype=ttnn.bfloat16,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        C_in_block=32,
        compute_with_storage_grid_size=(grid.x, grid.y),
    )
    tt_w = ttnn.experimental.prepare_conv3d_weights(
        weight_tensor=tt_w, groups=1, C_in_block=32, alignment=32, device=mesh_device
    )
    tt_b = ttnn.from_torch(
        torch.rand(1, C_out).bfloat16(), device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, pad_value=0
    )
    kcfg = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    def do_conv(inp_t, padding, halo=None):
        return ttnn.experimental.conv3d(
            input_tensor=inp_t,
            weight_tensor=tt_w,
            device=mesh_device,
            bias_tensor=tt_b,
            config=config,
            dtype=ttnn.bfloat16,
            output_channels=C_out,
            kernel_size=kernel_size,
            stride=(1, 1, 1),
            padding=padding,
            dilation=(1, 1, 1),
            padding_mode="zeros",
            groups=1,
            compute_kernel_config=kcfg,
            halo_buffer=halo,
        )

    # Golden: full-pad (neighbor_pad_async) then conv with spatial pad 0.
    out_shape = list(input_shape)
    out_shape[h_dim] += h_factor * 2 * pH
    out_shape[w_dim] += w_factor * 2 * pW
    persist = ttnn.from_torch(
        torch.zeros(out_shape).bfloat16(),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=mem,
        mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=dims),
    )
    full = ttnn.experimental.neighbor_pad_async(
        inp_mesh,
        [h_dim, w_dim],
        [pH, pW],
        [pH, pW],
        "zeros",
        [h_axis, w_axis],
        [h_sem, w_sem],
        [barrier_sem],
        num_links=[num_links, num_links],
        memory_config=mem,
        topology=ttnn.Topology.Linear,
        persistent_output_buffer=persist,
    )
    out_gold = do_conv(full, (pT, 0, 0))
    ttnn.synchronize_device(mesh_device, sub_device_ids=[sub_id])

    # Halo: compact (neighbor_pad_halo) then conv with spatial pad pH/pW reading the halo.
    halo_buf = ttnn.from_torch(
        torch.zeros([total_sticks, C]).bfloat16(),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=mem,
    )
    ttnn.experimental.neighbor_pad_halo(
        inp_mesh,
        halo_buf,
        np_padding_h=pH,
        np_padding_w=pW,
        np_cluster_axis=h_axis,
        np_num_links=num_links,
        np_topology=ttnn.Topology.Linear,
        h_neighbor_semaphore=h_sem,
        barrier_semaphore=barrier_sem,
        w_neighbor_semaphore=w_sem,
        np_pad_dim2=w_dim,
        np_pad2_left=pW,
        np_pad2_right=pW,
        np_pad2_cluster_axis=w_axis,
        np_pad2_num_links=num_links,
        padding_mode="zeros",
    )
    out_halo = do_conv(inp_mesh, (pT, pH, pW), halo=halo_buf)
    ttnn.synchronize_device(mesh_device, sub_device_ids=[sub_id])

    if os.environ.get("NP_CONV_PERF"):
        # Decompose the e2e halo-vs-fullpad delta: halo_path = np_halo + conv_halo_read; fullpad_path =
        # np_async(persistent) + conv_plain. Isolates whether the halo-mode conv3d (spatial padding active
        # -> boundary blocks skip the coalesced gather) is the net-loss source on small 4x8 shards.
        def _np_async():
            return ttnn.experimental.neighbor_pad_async(
                inp_mesh,
                [h_dim, w_dim],
                [pH, pW],
                [pH, pW],
                "zeros",
                [h_axis, w_axis],
                [h_sem, w_sem],
                [barrier_sem],
                num_links=[num_links, num_links],
                memory_config=mem,
                topology=ttnn.Topology.Linear,
                persistent_output_buffer=persist,
            )

        def _np_halo():
            return ttnn.experimental.neighbor_pad_halo(
                inp_mesh,
                halo_buf,
                np_padding_h=pH,
                np_padding_w=pW,
                np_cluster_axis=h_axis,
                np_num_links=num_links,
                np_topology=ttnn.Topology.Linear,
                h_neighbor_semaphore=h_sem,
                barrier_semaphore=barrier_sem,
                w_neighbor_semaphore=w_sem,
                np_pad_dim2=w_dim,
                np_pad2_left=pW,
                np_pad2_right=pW,
                np_pad2_cluster_axis=w_axis,
                np_pad2_num_links=num_links,
                padding_mode="zeros",
            )

        a = _trace_and_time(mesh_device, _np_async)
        hh = _trace_and_time(mesh_device, _np_halo)
        cp = _trace_and_time(mesh_device, lambda: do_conv(full, (pT, 0, 0)))
        ch = _trace_and_time(mesh_device, lambda: do_conv(inp_mesh, (pT, pH, pW), halo=halo_buf))
        print(
            f"CONV-PERF shape={input_shape} Cout={C_out} k={kernel_size}: np_async={a:.1f} np_halo={hh:.1f} "
            f"conv_plain={cp:.1f} conv_halo_read={ch:.1f} | fullpad_path={a + cp:.1f} halo_path={hh + ch:.1f} "
            f"delta={hh + ch - (a + cp):+.1f} us",
            flush=True,
        )
        mesh_device.reset_sub_device_stall_group()
        mesh_device.clear_loaded_sub_device_manager()
        return

    gold_dev = ttnn.get_device_tensors(ttnn.from_device(out_gold))
    halo_dev = ttnn.get_device_tensors(ttnn.from_device(out_halo))
    all_pass = True
    for i in range(mesh_shape[0] * mesh_shape[1]):
        g = ttnn.to_torch(gold_dev[i])
        h = ttnn.to_torch(halo_dev[i])
        ok, pcc = comp_pcc(g, h, 0.999)
        if not ok:
            all_pass = False
            print(f"FAIL dev {i}: {pcc}  shapes g={tuple(g.shape)} h={tuple(h.shape)}")
        else:
            print(f"PASS dev {i}: {pcc}")
    mesh_device.reset_sub_device_stall_group()
    mesh_device.clear_loaded_sub_device_manager()
    assert all_pass, "halo conv3d != full-pad conv3d"


@pytest.mark.timeout(600)
@pytest.mark.parametrize("mesh_device", [(4, 8)], ids=["4x8"], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize(
    "input_shape, C_out, kernel_size",
    [
        ([1, 8, 36, 64, 1024], 1024, (3, 3, 3)),  # s0 padded: per-dev 9x8, C_in=C_out=1024
        ([1, 8, 72, 128, 512], 512, (3, 3, 3)),  # s1: per-dev 18x16
        ([1, 8, 144, 256, 512], 512, (3, 3, 3)),  # s2: per-dev 36x32
        ([1, 8, 144, 256, 256], 256, (3, 3, 3)),  # s3: per-dev 36x32
        ([1, 8, 288, 512, 128], 128, (3, 3, 3)),  # s4: per-dev 72x64
    ],
    ids=["s0_36x64_C1024", "s1_72x128_C512", "s2_144x256_C512", "s3_144x256_C256", "s4_288x512_C128"],
)
def test_conv3d_halo_vs_fullpad(mesh_device, device_params, input_shape, C_out, kernel_size):
    pH, pW = kernel_size[1] // 2, kernel_size[2] // 2
    run_conv3d_halo_vs_fullpad_2d(
        mesh_device, input_shape, C_out=C_out, kernel_size=kernel_size, pH=pH, pW=pW, num_links=2
    )


# Reuse test: the decode calls neighbor_pad_halo ~30x sharing the manager's ping-pong sems. A single-op
# test can't catch a missing on-device self-reset — this loops the op N times reusing ONE sem set (no
# ping-pong to mask it). If call 2+ hangs, a semaphore isn't reset between runs.
@pytest.mark.timeout(200)
@pytest.mark.parametrize("mesh_device", [(2, 4)], ids=["2x4"], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_neighbor_pad_halo_reuse(mesh_device, device_params):
    input_shape, h_dim, w_dim, h_axis, w_axis, pH, pW, num_links = [1, 8, 68, 120, 64], 2, 3, 0, 1, 1, 1, 2
    mesh_shape = tuple(mesh_device.shape)
    h_factor, w_factor = mesh_shape[h_axis], mesh_shape[w_axis]
    torch.manual_seed(0)
    inp = torch.rand(input_shape).bfloat16()
    outer = input_shape[0] * input_shape[1]
    H_dev, W_dev, C = input_shape[h_dim] // h_factor, input_shape[w_dim] // w_factor, input_shape[-1]
    total_sticks = outer * 2 * pH * W_dev + outer * 2 * pW * (H_dev + 2 * pH)
    grid = mesh_device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    mgr = mesh_device.create_sub_device_manager([ttnn.SubDevice([crs])], 0)
    mesh_device.load_sub_device_manager(mgr)
    sub_id = ttnn.SubDeviceId(0)
    mesh_device.set_sub_device_stall_group([sub_id])
    h_sem = ttnn.create_global_semaphore(mesh_device, crs, 0)
    w_sem = ttnn.create_global_semaphore(mesh_device, crs, 0)
    barrier_sem = ttnn.create_global_semaphore(mesh_device, crs, 0)
    mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    dims = [None, None]
    dims[h_axis], dims[w_axis] = h_dim, w_dim
    inp_mesh = ttnn.from_torch(
        inp,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=mem,
        mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=dims),
    )
    halo_buf = ttnn.from_torch(
        torch.zeros([total_sticks, C]).bfloat16(),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=mem,
    )
    for it in range(5):
        ttnn.experimental.neighbor_pad_halo(
            inp_mesh,
            halo_buf,
            np_padding_h=pH,
            np_padding_w=pW,
            np_cluster_axis=h_axis,
            np_num_links=num_links,
            np_topology=ttnn.Topology.Linear,
            h_neighbor_semaphore=h_sem,
            barrier_semaphore=barrier_sem,
            w_neighbor_semaphore=w_sem,
            np_pad_dim2=w_dim,
            np_pad2_left=pW,
            np_pad2_right=pW,
            np_pad2_cluster_axis=w_axis,
            np_pad2_num_links=num_links,
            padding_mode="zeros",
        )
        ttnn.synchronize_device(mesh_device, sub_device_ids=[sub_id])
        print(f"iter {it}: OK", flush=True)
    mesh_device.reset_sub_device_stall_group()
    mesh_device.clear_loaded_sub_device_manager()
