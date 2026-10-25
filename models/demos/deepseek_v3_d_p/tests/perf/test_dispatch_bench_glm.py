# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""GLM-5.3 prefill MoE dispatch microbenchmark on the production 8x4 Blackhole Galaxy (FABRIC_2D_TORUS_XY).

Production shapes: 640 tokens/chip, hidden 6144 bf16 TILE input, 256 experts top-8, 8 experts/chip, dispatch along
axis 0 (ring of 8), 4 EP groups on axis 1, num_links=2. Times trace replays of DB_K back-to-back launches on the
model's sub-device split (row 0 for `deepseek_prefill.dispatch`, rows 0-1 for dispatch_fabric2d), so the number
is device time per launch plus the few-us inter-op gap.

Env knobs (all optional):
  DB_PAYLOADS   comma list of fabric payload sizes, one mesh open each (default 6208 = production)
  DB_WORKERS    comma list of num_workers_per_sender for the direct dispatch (default 2 = production)
  DB_ROUTINGS   comma list of uniform (default) | dist<d> (ring distance d, same pair count) | golden:<layer>:<chunk>
  DB_FAB2D_ROUTINGS  routings for the dispatch_fabric2d timing (default uniform)
  DB_K, DB_R    launches per trace (8), replays (5)
  DB_CHECK      first (default) = validate the first direct variant against the torch reference, 1 = all, 0 = none
  DB_DIRECT     1 = run the direct dispatch variants (default 1)
  DB_FAB2D      1 = also time dispatch_fabric2d (payload >= 12352 only) and probe repeated trace replays
  DB_FAB2D_REPLAYS  replays for the fab2d probe (default 4)
  DB_FAB2D_FP8  1 = dispatch_fabric2d with fp8_output (FP8_E4M3 rows; needs the fp8-wire op), no torch check
  DB_FAB2D_CHECK  1 = torch-reference check of dispatch_fabric2d outputs (bf16 only)
  DB_AG         1 = time an axis-0 all_gather of x as a link-bandwidth reference
  DB_HEARTBEAT  path touched after every step (for an external hang watchdog)
"""

import glob
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.tt.moe.dispatch import TorchDispatchModule
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    ExpertMapping,
    compute_constants,
    get_dispatch_input_mesh_mapper,
    get_ep_mesh_composer,
    get_gate_outputs,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatchModule
from models.demos.deepseek_v3_d_p.tt.moe.validation_helpers import validate_dispatch_buffer, validate_dispatch_metadata

H, G = 8, 4
SEQ = 640
EMB = 6144
E = 256
TOPK = 8
CAP_FACTOR = 2
NUM_LINKS = 2
GOLDEN = "/mnt/weka/model-cache/scratch/zai-org/GLM-5.3-Cache/golden_traces/vllm-glm53-indexer-kcache-55k/routing"


def _env_list(name, default):
    return [int(v) for v in os.environ.get(name, default).split(",") if v]


def _beat(tag):
    p = os.environ.get("DB_HEARTBEAT")
    if p:
        with open(p, "a") as f:
            f.write(f"{time.time():.1f} {tag}\n")
    logger.info(f"[dbench] step {tag}")


def _routing(spec, table):
    """indices [H, SEQ, TOPK] for a routing spec: uniform | dist<d> | golden:<layer>:<chunk>.

    dist<d>: every token picks 2 distinct experts in each of the 4 EP groups, all on the chip d rows below its own
    (mod 8): the same 2 pairs per token per column as uniform routing on average, at a controlled ring distance
    (dist0 = all local, no fabric; dist1 = neighbours only, no router pass-through; dist4 = the opposite chip)."""
    g = torch.Generator().manual_seed(1234)
    if spec == "uniform":
        idx = torch.stack([torch.randperm(E, generator=g)[:TOPK] for _ in range(H * SEQ)]).view(H, SEQ, TOPK)
    elif spec.startswith("dist"):
        d = int(spec[4:])
        per_group = TOPK // G
        idx = torch.zeros(H, SEQ, TOPK, dtype=torch.int64)
        for h in range(H):
            tgt = (h + d) % H
            on = [[e for e in range(E) if int(table[gi][e]) == tgt] for gi in range(G)]
            for t in range(SEQ):
                picks = []
                for gi in range(G):
                    perm = torch.randperm(len(on[gi]), generator=g)[:per_group]
                    picks += [on[gi][int(i)] for i in perm]
                idx[h, t] = torch.tensor(picks)
    else:
        from safetensors.torch import load_file

        _, layer, chunk = spec.split(":")
        f = glob.glob(f"{GOLDEN}/expert_ids_layer_{int(layer)}/*.safetensors")[0]
        ids = list(load_file(f).values())[0].long()
        c = int(chunk)
        idx = ids[c * H * SEQ : (c + 1) * H * SEQ].view(H, SEQ, TOPK)
    return idx.to(torch.int32)


def _time_trace(mesh_device, fn, k, r, label):
    """Capture k back-to-back launches of fn in one trace, replay r times; return us per launch."""
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for _ in range(k):
        outs = fn()
        for o in outs:
            ttnn.deallocate(o)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)  # warm replay
    ttnn.synchronize_device(mesh_device)
    times = []
    for i in range(r):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        times.append((time.perf_counter() - t0) * 1e6 / k)
        _beat(f"{label} replay {i}")
    ttnn.release_trace(mesh_device, tid)
    times.sort()
    med = times[len(times) // 2]
    logger.info(f"[dbench] RESULT {label}: per_launch_us median={med:.1f} min={times[0]:.1f} max={times[-1]:.1f}")
    return med


def _sub_device_split(mesh_device, rows):
    grid = mesh_device.compute_with_storage_grid_size()
    disp = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, rows - 1))})
    rest = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, rows), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    mgr = mesh_device.create_sub_device_manager([ttnn.SubDevice([disp]), ttnn.SubDevice([rest])], 0)
    return mgr


_PAYLOAD_PARAMS = [
    pytest.param(
        (8, 4),
        torus_xy_device_params(fabric_payload_size=p, l1_small_size=1536, trace_region_size=512 * 1024 * 1024),
        id=f"p{p}",
    )
    for p in _env_list("DB_PAYLOADS", "6208")
]


@pytest.mark.parametrize("mesh_device, device_params", _PAYLOAD_PARAMS, indirect=True)
@pytest.mark.timeout(1500)
def test_dispatch_bench(mesh_device, device_params):
    payload = ttnn.get_tt_fabric_max_payload_size_bytes()
    k, r = int(os.environ.get("DB_K", "8")), int(os.environ.get("DB_R", "5"))
    torch.manual_seed(42)
    x = torch.randn(H, SEQ, EMB, dtype=torch.bfloat16)
    experts_per_chip, metadata_len, max_buf, max_per_expert = compute_constants(
        SEQ, E, TOPK, mesh_device.get_num_devices(), H, CAP_FACTOR
    )
    table = ExpertMapping.create_dispatch_table(num_routed_experts=E, dispatch_group_size=H, num_dispatch_groups=G)
    mapper = get_dispatch_input_mesh_mapper(mesh_device, 0)
    tt_x = ttnn.from_torch(x, mesh_mapper=mapper, layout=ttnn.TILE_LAYOUT, device=mesh_device, dtype=ttnn.bfloat16)
    tt_w = ttnn.from_torch(
        torch.zeros(H, SEQ, TOPK, dtype=torch.bfloat16),
        mesh_mapper=mapper,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
    )
    tt_table = TtDispatchModule.shard_expert_dispatch_table(mesh_device, table, 0)
    logger.info(f"[dbench] payload={payload} epc={experts_per_chip} max_buf={max_buf} grid={mesh_device.compute_with_storage_grid_size()}")
    results = {}
    check_mode = os.environ.get("DB_CHECK", "first")
    checked = [False]

    def guarded(label, fn):
        try:
            results[label] = fn()
        except Exception as ex:  # one variant failing must not hide the others
            logger.error(f"[dbench] FAIL {label}: {type(ex).__name__}: {str(ex).splitlines()[0] if str(ex) else ''}")
            results[label] = None

    routings = [v for v in os.environ.get("DB_ROUTINGS", "uniform").split(",") if v]
    if os.environ.get("DB_DIRECT", "1") == "1":
        mgr = _sub_device_split(mesh_device, 1)
        mesh_device.load_sub_device_manager(mgr)
        try:
            for spec in routings:
                indices = _routing(spec, table)
                offs, counts, region, _ = get_gate_outputs(
                    indices, H, E, experts_per_chip, SEQ, TOPK, expert_dispatch_table=table
                )
                col_pairs = [int((table[gi][indices.long()] >= 0).sum()) for gi in range(G)]
                logger.info(f"[dbench] routing={spec} pairs per column={col_pairs}")
                tt_idx = ttnn.from_torch(
                    indices.to(torch.int16),
                    mesh_mapper=mapper,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=mesh_device,
                    dtype=ttnn.uint16,
                )
                tt_offs = TtDispatchModule.shard_expert_offsets(mesh_device, offs)
                for w in _env_list("DB_WORKERS", "2"):

                    def one(w=w, spec=spec, indices=indices, offs=offs, counts=counts, region=region, tt_idx=tt_idx, tt_offs=tt_offs):
                        mod = TtDispatchModule(
                            mesh_device=mesh_device,
                            dispatch_group_size=H,
                            experts_per_chip=experts_per_chip,
                            num_routed_experts=E,
                            num_experts_per_tok=TOPK,
                            metadata_len=metadata_len,
                            max_dispatch_buffer_token_size=max_buf,
                            seq_len_per_chip=SEQ,
                            emb_dim=EMB,
                            cluster_axis=0,
                            num_links=NUM_LINKS,
                            topology=ttnn.Topology.Ring,
                            subdevice_id=ttnn.SubDeviceId(0),
                            num_workers_per_sender=w,
                        )

                        def run():
                            return mod(tt_x, tt_w, tt_idx, tt_offs, tt_table)

                        out_buf, out_meta = run()
                        ttnn.synchronize_device(mesh_device)
                        _beat(f"direct w{w} {spec} eager")
                        if check_mode == "1" or (check_mode == "first" and not checked[0]):
                            checked[0] = True
                            ref = TorchDispatchModule(
                                dispatch_group_size=H,
                                experts_per_chip=experts_per_chip,
                                num_routed_experts=E,
                                num_experts_per_tok=TOPK,
                                metadata_len=metadata_len,
                                max_dispatched_tokens_per_expert=max_per_expert,
                                max_dispatch_buffer_token_size=max_buf,
                                seq_len_per_chip=SEQ,
                                emb_dim=EMB,
                                num_dispatch_groups=G,
                                expert_dispatch_table=table,
                            )
                            ref_buf, ref_meta = ref(x, torch.zeros(H, SEQ, TOPK), indices, offs)
                            comp = get_ep_mesh_composer(mesh_device)
                            got_buf = ttnn.to_torch(out_buf, mesh_composer=comp, dtype=torch.float32)
                            got_meta = ttnn.to_torch(out_meta, mesh_composer=comp)
                            rb = validate_dispatch_buffer(
                                ref_buf, got_buf, region, counts, table, G, H, experts_per_chip, verbose=False
                            )
                            rm = validate_dispatch_metadata(
                                ref_meta, got_meta, region, counts, table, G, H, experts_per_chip, verbose=False
                            )
                            logger.info(f"[dbench] CHECK direct w{w} {spec}: buffer={rb.passed} metadata={rm.passed}")
                            assert rb.passed and rm.passed, f"direct w{w} {spec} output mismatch"
                        ttnn.deallocate(out_buf)
                        ttnn.deallocate(out_meta)
                        return _time_trace(mesh_device, run, k, r, f"direct w{w} payload{payload} {spec}")

                    guarded(f"direct_w{w}_{spec}", one)
        finally:
            mesh_device.clear_loaded_sub_device_manager()
            mesh_device.remove_sub_device_manager(mgr)

    if os.environ.get("DB_AG", "0") == "1":

        def ag_one():
            def ag():
                return [ttnn.all_gather(tt_x, dim=1, cluster_axis=0)]

            o = ag()
            ttnn.synchronize_device(mesh_device)
            ttnn.deallocate(o[0])
            us = _time_trace(mesh_device, ag, k, r, f"all_gather axis0 x payload{payload}")
            logger.info(
                f"[dbench] all_gather axis0: {7 * SEQ * EMB * 2 / 1e6:.2f} MB rx/chip -> "
                f"{7 * SEQ * EMB * 2 / us / 1e3:.1f} GB/s rx per chip"
            )
            return us

        guarded("ag_axis0", ag_one)

    fab2d_min_payload = 6400 if os.environ.get("DB_FAB2D_FP8", "0") == "1" else 12352  # token (+scales) + 64 B tail
    if os.environ.get("DB_FAB2D", "0") == "1" and payload >= fab2d_min_payload:
        from models.demos.deepseek_v3_d_p.tests.op_unit_tests.test_dispatch_fabric2d import _expert_dispatch_table

        t2 = _expert_dispatch_table(E, H, G)

        def shard(t, dims, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
            return ttnn.from_torch(
                t,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims),
                layout=layout,
                device=mesh_device,
                dtype=dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        x4 = x.unsqueeze(1).expand(H, G, SEQ, EMB).contiguous()
        f_x = shard(x4, (0, 1), ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        f_table = shard(t2.unsqueeze(1), (None, 0), ttnn.int32)
        mgr2 = _sub_device_split(mesh_device, 2)
        mesh_device.load_sub_device_manager(mgr2)
        try:
            for spec in [v for v in os.environ.get("DB_FAB2D_ROUTINGS", "uniform").split(",") if v]:
                indices = _routing(spec, table)
                ind4 = indices.long().unsqueeze(0).expand(G, -1, -1, -1).contiguous()
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

                fp8 = os.environ.get("DB_FAB2D_FP8", "0") == "1"
                fp8_kw = {"fp8_output": True} if fp8 else {}

                def fab_one(spec=spec, f_idx=f_idx, f_offs=f_offs, f_counts=f_counts, f_region=f_region, fp8=fp8, fp8_kw=fp8_kw):
                    def run2():
                        return ttnn.experimental.deepseek_prefill.dispatch_fabric2d(
                            f_x,
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
                            **fp8_kw,
                        )

                    outs = run2()
                    ttnn.synchronize_device(mesh_device)
                    _beat(f"fab2d {spec} eager")
                    if os.environ.get("DB_FAB2D_CHECK", "0") == "1" and not fp8:
                        offs, counts, region, _ = get_gate_outputs(
                            indices, H, E, experts_per_chip, SEQ, TOPK, expert_dispatch_table=table
                        )
                        ref = TorchDispatchModule(
                            dispatch_group_size=H,
                            experts_per_chip=experts_per_chip,
                            num_routed_experts=E,
                            num_experts_per_tok=TOPK,
                            metadata_len=3,
                            max_dispatched_tokens_per_expert=max_per_expert,
                            max_dispatch_buffer_token_size=max_buf,
                            seq_len_per_chip=SEQ,
                            emb_dim=EMB,
                            num_dispatch_groups=G,
                            expert_dispatch_table=table,
                        )
                        ref_buf, ref_meta = ref(x, torch.zeros(H, SEQ, TOPK), indices, offs)
                        comp = get_ep_mesh_composer(mesh_device)
                        got_buf = ttnn.to_torch(outs[0], mesh_composer=comp, dtype=torch.float32)
                        got_meta = ttnn.to_torch(outs[1], mesh_composer=comp)
                        rb = validate_dispatch_buffer(
                            ref_buf, got_buf, region, counts, table, G, H, experts_per_chip, verbose=False
                        )
                        rm = validate_dispatch_metadata(
                            ref_meta, got_meta, region, counts, table, G, H, experts_per_chip, verbose=False
                        )
                        logger.info(f"[dbench] CHECK fab2d {spec}: buffer={rb.passed} metadata={rm.passed}")
                        assert rb.passed and rm.passed, f"fab2d {spec} output mismatch"
                    for o in outs:
                        ttnn.deallocate(o)
                    _beat(f"fab2d {spec} eager")
                    if os.environ.get("DB_PROFILE", "0") == "1":
                        # Device-profiler mode: a few synced eager launches, then dump zones (no trace).
                        ttnn.ReadDeviceProfiler(mesh_device)
                        for _ in range(3):
                            for o in run2():
                                ttnn.deallocate(o)
                            ttnn.synchronize_device(mesh_device)
                        ttnn.ReadDeviceProfiler(mesh_device)
                        _beat(f"fab2d {spec} profiled")
                        return 0.0
                    return _time_trace(
                        mesh_device, run2, k, int(os.environ.get("DB_FAB2D_REPLAYS", "4")), f"fab2d{'-fp8' if fp8 else ''} payload{payload} {spec}"
                    )

                guarded(f"fab2d_{spec}", fab_one)
        finally:
            mesh_device.clear_loaded_sub_device_manager()
            mesh_device.remove_sub_device_manager(mgr2)

    if os.environ.get("DB_ROUTEOPS", "0") == "1":
        # The MoE routing chain at production shapes: moe_grouped_topk -> masked_bincount -> offset_cumsum
        # (with its internal axis-0 all-gather), as TtMoEGatePrefill / TtMoERoutingSetup call them.
        spec = os.environ.get("DB_ROUTEOPS_ROUTING", "uniform")
        indices = _routing(spec, table)
        rows = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(0, None))
        tt_idx_tile = ttnn.from_torch(
            indices.reshape(H * SEQ, TOPK).to(torch.int16),
            mesh_mapper=rows,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            dtype=ttnn.uint16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        tt_mask = ttnn.from_torch(
            table,
            device=mesh_device,
            dtype=ttnn.int32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(None, 0)),
        )
        hist_holder = []

        def bincount():
            h = ttnn.experimental.deepseek_prefill.masked_bincount(tt_idx_tile, tt_mask, E, TOPK)
            return [h]

        def cumsum():
            if not hist_holder:
                hist_holder.append(ttnn.experimental.deepseek_prefill.masked_bincount(tt_idx_tile, tt_mask, E, TOPK))
            return list(
                ttnn.experimental.deepseek_prefill.offset_cumsum(
                    hist_holder[0],
                    cluster_axis=0,
                    num_links=NUM_LINKS,
                    experts_per_chip=experts_per_chip,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            )

        logits = ttnn.from_torch(
            torch.randn(H * SEQ, E, dtype=torch.float32),
            mesh_mapper=rows,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            dtype=ttnn.float32,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        bias = ttnn.from_torch(
            (torch.randn(1, E) * 0.01).repeat(SEQ, 1).to(torch.bfloat16),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )

        def topk():
            return list(
                ttnn.experimental.deepseek_prefill.moe_grouped_topk(
                    logits,
                    bias,
                    n_groups=1,
                    summed_experts_per_group=1,
                    topk_groups=1,
                    n_activated_experts=TOPK,
                    route_scale=2.5,
                    stable_sort=True,
                    epsilon=1e-20,
                    score_func="sigmoid",
                    weights_layout=ttnn.ROW_MAJOR_LAYOUT,
                )
            )

        if os.environ.get("DB_ROUTEOPS_CHECK", "1") == "1":
            h = bincount()[0]
            got = (
                ttnn.to_torch(ttnn.unsqueeze_to_4D(h), mesh_composer=get_ep_mesh_composer(mesh_device))
                .squeeze(2)[..., :E]
                .to(torch.int64)
            )
            ref = torch.zeros(G, H, E, dtype=torch.int64)
            for gi in range(G):
                for hi in range(H):
                    e = indices[hi].reshape(-1).long()
                    e = e[table[gi][e] >= 0]
                    ref[gi, hi] = torch.bincount(e, minlength=E)[:E]
            ok = torch.equal(got, ref)
            logger.info(f"[dbench] CHECK masked_bincount {spec}: exact={ok}")
            assert ok, "masked_bincount histogram mismatch"
            ttnn.deallocate(h)

            # offset_cumsum against the unit test's reference, per device (row r of group c holds hist ref[c, r]).
            def ref_cumsum(hist):
                cum = torch.cumsum(hist, 0)
                tot = cum[-1:]
                al = (tot.reshape(-1, experts_per_chip) + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE * ttnn.TILE_SIZE
                reg = (torch.cumsum(al, -1) - al).reshape(1, E)
                return cum - hist + reg, tot, reg

            outs = cumsum()
            per_dev = [ttnn.get_device_tensors(o) for o in outs]
            cok = True
            for dev_idx in range(H * G):
                r, c = divmod(dev_idx, G)
                ref_rows, ref_tot, ref_reg = ref_cumsum(ref[c])
                refs4 = (ref_rows[r : r + 1], ref_tot, ref_reg, ref_rows)
                for k, want in enumerate(refs4):
                    have = ttnn.to_torch(per_dev[k][dev_idx]).to(torch.int64)
                    have = have.reshape(-1, have.shape[-1])[:, :E]
                    if not torch.equal(have, want):
                        cok = False
                        logger.info(f"[dbench] CHECK offset_cumsum mismatch dev {dev_idx} output {k}")
            logger.info(f"[dbench] CHECK offset_cumsum {spec}: exact={cok}")
            for o in outs:
                ttnn.deallocate(o)
            assert cok, "offset_cumsum mismatch"

        for name, fn in [("grouped_topk", topk), ("masked_bincount", bincount), ("offset_cumsum", cumsum)]:

            def one(name=name, fn=fn):
                outs = fn()
                ttnn.synchronize_device(mesh_device)
                for o in outs:
                    ttnn.deallocate(o)
                _beat(f"{name} eager")
                if os.environ.get("DB_PROFILE", "0") == "1":
                    ttnn.ReadDeviceProfiler(mesh_device)
                    for _ in range(3):
                        for o in fn():
                            ttnn.deallocate(o)
                        ttnn.synchronize_device(mesh_device)
                    ttnn.ReadDeviceProfiler(mesh_device)
                    return 0.0
                return _time_trace(mesh_device, fn, k, r, f"{name} {spec}")

            guarded(name, one)

    logger.info(f"[dbench] SUMMARY payload={payload}: {results}")
