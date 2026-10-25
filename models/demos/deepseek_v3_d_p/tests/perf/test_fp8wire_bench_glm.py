# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""GLM-5.3 fp8 dispatch wire microbenchmark (fp8wire track) on the production 8x4 Blackhole Galaxy.

Times, as trace replays of FW_K back-to-back launches:
  bf16    dispatch_fabric2d on the TILE bf16 x (untilizer on row 1), sub-device rows 0-1 (production today)
  fp8     dispatch_fabric2d moving Q1's packed [e4m3 | fp32 scale per block] rows byte for byte, sub-device row(s)
          0..FW_FP8_ROWS-1
  q1      per_token_cast_to_fp8 of the TILE bf16 x (full grid)
  dq1t    per_token_cast_back of the fp8 dispatched buffer to bf8 TILE (valid rows only, full grid)
  dq1r    the same to bf16 ROW_MAJOR
and checks, per routing, DQ1(fp8 dispatch(Q1(x))) against the bf16 dispatch of x over every valid row (PCC, max abs
error) and against a host model of the quantization.

Env knobs: FW_ROUTINGS (uniform,golden:24:5,golden:70:5), FW_K (8), FW_R (5), FW_BLOCK (128), FW_FP8_ROWS (1),
FW_VARIANTS (bf16,fp8,q1,dq1t,dq1r), FW_CHECK (1).
"""

import glob
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.op_unit_tests.test_dispatch_fabric2d import _expert_dispatch_table
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import compute_constants, get_gate_outputs

H, G = 8, 4
SEQ = 640
EMB = 6144
E = 256
TOPK = 8
CAP_FACTOR = 2
NUM_LINKS = 2
GOLDEN = "/mnt/weka/model-cache/scratch/zai-org/GLM-5.3-Cache/golden_traces/vllm-glm53-indexer-kcache-55k/routing"


def _routing(spec):
    g = torch.Generator().manual_seed(1234)
    if spec == "uniform":
        idx = torch.stack([torch.randperm(E, generator=g)[:TOPK] for _ in range(H * SEQ)]).view(H, SEQ, TOPK)
    else:
        from safetensors.torch import load_file

        _, layer, chunk = spec.split(":")
        f = glob.glob(f"{GOLDEN}/expert_ids_layer_{int(layer)}/*.safetensors")[0]
        ids = list(load_file(f).values())[0].long()
        c = int(chunk)
        idx = ids[c * H * SEQ : (c + 1) * H * SEQ].view(H, SEQ, TOPK)
    return idx.to(torch.int64)


def _time_trace(mesh_device, fn, k, r, label):
    # Warm-up: a trace capture cannot compile a program that is not in the program cache yet.
    for o in fn():
        ttnn.deallocate(o)
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for _ in range(k):
        outs = fn()
        for o in outs:
            ttnn.deallocate(o)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    ttnn.synchronize_device(mesh_device)
    times = []
    for _ in range(r):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        times.append((time.perf_counter() - t0) * 1e6 / k)
    ttnn.release_trace(mesh_device, tid)
    times.sort()
    med = times[len(times) // 2]
    logger.warning(f"[fw] RESULT {label}: per_launch_us median={med:.1f} min={times[0]:.1f} max={times[-1]:.1f}")
    return med


def _split(mesh_device, rows):
    grid = mesh_device.compute_with_storage_grid_size()
    a = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, rows - 1))})
    b = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, rows), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    return mesh_device.create_sub_device_manager([ttnn.SubDevice([a]), ttnn.SubDevice([b])], 0)


def _host_fp8_roundtrip(x, block):
    """Host model of Q1 + DQ1: per-1x<block> scale = 2^ceil(log2(amax/448)), e4m3 RNE, back to float."""
    xf = x.float().reshape(*x.shape[:-1], -1, block)
    amax = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    q = (xf / scale).to(torch.float8_e4m3fn).float()
    return (q * scale).reshape(x.shape)


def _pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(fabric_payload_size=12352, l1_small_size=1536, trace_region_size=512 * 1024 * 1024),
            id="p12352",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(2400)
def test_fp8wire_bench(mesh_device, device_params):
    k, r = int(os.environ.get("FW_K", "8")), int(os.environ.get("FW_R", "5"))
    block = int(os.environ.get("FW_BLOCK", "128"))
    fp8_rows = int(os.environ.get("FW_FP8_ROWS", "1"))
    variants = set(os.environ.get("FW_VARIANTS", "bf16,fp8,q1,dq1t,dq1r").split(","))
    do_check = os.environ.get("FW_CHECK", "1") == "1"
    torch.manual_seed(42)
    # Post-norm-like hidden: unit variance with a few large channels, as the RMSNorm output has.
    x = torch.randn(H, SEQ, EMB)
    x[..., torch.randperm(EMB)[:32]] *= 20.0
    x = x.to(torch.bfloat16)
    experts_per_chip, _, max_buf, _ = compute_constants(SEQ, E, TOPK, mesh_device.get_num_devices(), H, CAP_FACTOR)
    t2 = _expert_dispatch_table(E, H, G)
    logger.warning(f"[fw] payload={ttnn.get_tt_fabric_max_payload_size_bytes()} epc={experts_per_chip} max_buf={max_buf}")

    def shard(t, dims, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
        return ttnn.from_torch(
            t,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims),
            layout=layout,
            device=mesh_device,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    composer = ttnn.create_mesh_composer(mesh_device, ttnn.MeshComposerConfig(dims=[0, 1]))
    x4 = x.unsqueeze(1).expand(H, G, SEQ, EMB).contiguous()
    f_x = shard(x4, (0, 1), ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    f_table = shard(t2.unsqueeze(1), (None, 0), ttnn.int32)
    # Local expert table per chip (row h, col g): global ids g*64 + h*8 + j, UINT32 (experts_per_chip,).
    gidx = torch.zeros(H, G, experts_per_chip, dtype=torch.int32)
    for h in range(H):
        for gi in range(G):
            gidx[h, gi] = torch.arange(experts_per_chip) + gi * (E // G) + h * experts_per_chip
    f_gidx = ttnn.squeeze(ttnn.squeeze(shard(gidx.unsqueeze(2), (0, 1), ttnn.uint32), 0), 0)

    def q1():
        q, s = ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(
            f_x,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            round_scale_to_power_of_two=True,
            round_to_nearest_even=True,
            pack_scales_in_row=True,
            scale_block_w=block,
        )
        return [q, s]

    q_out = q1()
    f_q = q_out[0]
    ttnn.deallocate(q_out[1])
    logger.warning(f"[fw] Q1 out shape={f_q.shape} dtype={f_q.dtype} layout={f_q.layout}")
    if "q1" in variants:
        _time_trace(mesh_device, q1, k, r, f"q1 block{block}")

    results = {}
    for spec in [v for v in os.environ.get("FW_ROUTINGS", "uniform,golden:24:5,golden:70:5").split(",") if v and v != "none"]:
        indices = _routing(spec)
        ind4 = indices.unsqueeze(0).expand(G, -1, -1, -1).contiguous()
        o2 = torch.zeros(G, H, E, dtype=torch.int32)
        c2 = torch.zeros(G, H, E, dtype=torch.int32)
        r2 = torch.zeros(G, H, E, dtype=torch.int32)
        for gi in range(G):
            o, c, rr, _ = get_gate_outputs(
                ind4[gi], H, E, experts_per_chip, SEQ, TOPK, expert_dispatch_table=t2[gi : gi + 1]
            )
            o2[gi], c2[gi], r2[gi] = o[0].to(torch.int32), c[0].to(torch.int32), rr[0].to(torch.int32)
        f_idx = shard(ind4.permute(1, 0, 2, 3).to(torch.int32).to(torch.int16), (0, 1), ttnn.uint16)
        f_offs = shard(o2, (None, 0), ttnn.int32)
        f_counts = shard(c2[:, 0:1, :], (None, 0), ttnn.int32)
        f_region = shard(r2[:, 0:1, :], (None, 0), ttnn.int32)
        # (1, E) UINT32 per device, as DQ1 takes them (TtMoe's are the routing setup's 2D tensors).
        f_counts_u = ttnn.squeeze(shard(c2[:, 0:1, :], (None, 0), ttnn.uint32), 0)
        f_region_u = ttnn.squeeze(shard(r2[:, 0:1, :], (None, 0), ttnn.uint32), 0)
        maxchip = max(
            int(c2[gi, 0, gi * 64 + h * 8 : gi * 64 + h * 8 + 8].sum()) for gi in range(G) for h in range(H)
        )
        logger.warning(f"[fw] routing={spec} max chip load={maxchip} pairs")

        def disp(inp, fp8):
            return ttnn.experimental.deepseek_prefill.dispatch_fabric2d(
                inp,
                f_idx,
                f_offs,
                f_table,
                f_counts,
                f_region,
                experts_per_chip=experts_per_chip,
                num_routed_experts=E,
                num_experts_per_tok=TOPK,
                metadata_len=3,
                max_dispatch_buffer_token_size=max_buf,
                seq_len_per_chip=SEQ,
                cluster_axis=0,
                num_links=NUM_LINKS,
                topology=ttnn.Topology.Ring,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                subdevice_id=ttnn.SubDeviceId(0),
                fp8_output=fp8,
            )

        def dq1(buf, tile):
            return ttnn.experimental.deepseek_prefill.per_token_cast_back(
                buf,
                None,
                output_dtype=ttnn.bfloat8_b if tile else ttnn.bfloat16,
                tile_output=tile,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                scales_in_row_tail=True,
                token_count_aware=True,
                expert_region_offsets=f_region_u,
                expert_token_counts=f_counts_u,
                global_expert_idx_table=f_gidx,
                experts_per_chip=experts_per_chip,
                scale_block_w=block,
            )

        # bf16 dispatch on rows 0-1 (untilizer on row 1), as production today.
        mgr = _split(mesh_device, 2)
        mesh_device.load_sub_device_manager(mgr)
        try:
            ref_buf, ref_meta = disp(f_x, False)
            ttnn.synchronize_device(mesh_device)
            if "bf16" in variants:
                results[f"bf16_{spec}"] = _time_trace(mesh_device, lambda: disp(f_x, False), k, r, f"bf16 {spec}")
        finally:
            mesh_device.clear_loaded_sub_device_manager()
            mesh_device.remove_sub_device_manager(mgr)

        mgr = _split(mesh_device, fp8_rows)
        mesh_device.load_sub_device_manager(mgr)
        try:
            fp8_buf, fp8_meta = disp(f_q, True)
            ttnn.synchronize_device(mesh_device)
            if "fp8" in variants:
                results[f"fp8_{spec}"] = _time_trace(
                    mesh_device, lambda: disp(f_q, True), k, r, f"fp8 rows{fp8_rows} {spec}"
                )
        finally:
            mesh_device.clear_loaded_sub_device_manager()
            mesh_device.remove_sub_device_manager(mgr)

        if "dq1t" in variants:
            results[f"dq1t_{spec}"] = _time_trace(mesh_device, lambda: [dq1(fp8_buf, True)], k, r, f"dq1 tile {spec}")
        if "dq1r" in variants:
            results[f"dq1r_{spec}"] = _time_trace(mesh_device, lambda: [dq1(fp8_buf, False)], k, r, f"dq1 rm {spec}")

        if do_check:
            dq_t = dq1(fp8_buf, True)
            dq_r = dq1(fp8_buf, False)
            ttnn.synchronize_device(mesh_device)
            dev = {
                name: ttnn.get_device_tensors(t)
                for name, t in (("ref", ref_buf), ("t", dq_t), ("r", dq_r), ("mr", ref_meta), ("mf", fp8_meta))
            }
            rows_ref, rows_t, rows_r, meta_ok = [], [], [], True
            # 4 chips (column 0, rows 0, 3, 5, 7 -- the host conversion of a whole buffer is slow).
            for h in (0, 3, 5, 7):
                for gi in (0,):
                    d = h * G + gi
                    ref_d = ttnn.to_torch(dev["ref"][d]).float().reshape(-1, EMB)
                    t_d = ttnn.to_torch(dev["t"][d]).float().reshape(-1, EMB)
                    r_d = ttnn.to_torch(dev["r"][d]).float().reshape(-1, EMB)
                    mr_d = ttnn.to_torch(dev["mr"][d]).reshape(-1, 3)
                    mf_d = ttnn.to_torch(dev["mf"][d]).reshape(-1, 3)
                    for j in range(experts_per_chip):
                        e = gi * (E // G) + h * experts_per_chip + j
                        st, n = int(r2[gi, 0, e]), int(c2[gi, 0, e])
                        if n == 0:
                            continue
                        rows_ref.append(ref_d[st : st + n])
                        rows_t.append(t_d[st : st + n])
                        rows_r.append(r_d[st : st + n])
                        meta_ok &= bool(torch.equal(mr_d[st : st + n], mf_d[st : st + n]))
            a = torch.cat(rows_ref)
            bt = torch.cat(rows_t)
            br = torch.cat(rows_r)
            host = _host_fp8_roundtrip(a, block)
            logger.warning(
                f"[fw] CHECK {spec}: rows={a.shape[0]} metadata_equal={meta_ok} "
                f"PCC(dq1 tile, bf16)={_pcc(bt, a):.6f} PCC(dq1 rm, bf16)={_pcc(br, a):.6f} "
                f"PCC(host fp8, bf16)={_pcc(host, a):.6f} PCC(dq1 rm, host fp8)={_pcc(br, host):.6f} "
                f"rm==host exact frac={(br == host).float().mean():.6f} "
                f"max|rm-host|={(br - host).abs().max():.4g} max|tile-rm|/max|rm|="
                f"{((bt - br).abs().max() / br.abs().max()):.4g}"
            )
            for t in (dq_t, dq_r):
                ttnn.deallocate(t)
        for t in (ref_buf, ref_meta, fp8_buf, fp8_meta, f_idx, f_offs, f_counts, f_region, f_counts_u, f_region_u):
            ttnn.deallocate(t)
    logger.warning(f"[fw] SUMMARY {results}")
