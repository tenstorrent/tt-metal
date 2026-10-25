# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""R0: hybrid_routed_expert_moe + combine_fabric2d (the overlapped production op) at experts_per_chip 8 vs 16.

Redundant experts (notes/moe-replication.md) give every chip 8 + R local slots. This measures, on the 8x4 Galaxy at
GLM-5.3 shapes (emb 6144, hidden 2048, top-8, 640 tokens/chip, threshold 320), what a slot costs: the 8-slot control,
16 slots with only 8 active (idle-slot overhead), 16 active, and mixes, each traced and replayed, against the RE cost
model 48 + sum_fused(65.9 + 0.166 n) + sum_unified(26 + 68.1 ceil(n/256)) of the hottest chip (+ combine tail).

Synthetic routing: every dispatch group replays the same per-group counts (2 of a token's 8 picks per group); counts are
given per chip slot. Weight values do not matter for timing: 8 distinct slot weight sets, reused for slots 8-15.
Lines start with "HRE16 ". Env: HRE16_CASES (comma list, default all), HRE16_ITERS (5).
"""

import os
import time
import traceback

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import GLM53Config
from models.demos.deepseek_v3_d_p.reference.tt.moe.dispatch import TorchDispatchModule
from models.demos.deepseek_v3_d_p.tests.op_unit_tests.test_hybrid_routed_expert_combine import (
    _capture,
    _device_params,
    _indices_from_counts,
    _int_tensor,
    _replay_us,
)
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    ExpertMapping,
    compute_constants,
    extract_mesh_config,
    get_ep_mesh_mapper,
    get_expert_token_counts_mesh_mapper,
    get_gate_outputs,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert, routed_expert_weight_address_table

SEQ = 640
CAP = 8
GROUP_PAIRS = 8 * SEQ * 2  # tokens in a ring x 2 of the 8 picks per group = 10240
ITERS = int(os.environ.get("HRE16_ITERS", "5"))


def log(*a):
    print("HRE16", *a, flush=True)


def per_chip(slots, pattern):
    """Per-group counts (8 chips x slots, chip-major) from one chip's slot pattern; every chip the same."""
    assert len(pattern) == slots and sum(pattern) == GROUP_PAIRS // 8, (slots, sum(pattern))
    return [c for _ in range(8) for c in pattern]


# name -> (slots, one chip's per-slot counts). Per chip 1280 pairs in every case (same combine volume).
CASES = {
    "s8-bal160": (8, [160] * 8),
    "s16-half160-lo": (16, [160] * 8 + [0] * 8),
    "s16-half160-hi": (16, [0] * 8 + [160] * 8),
    "s16-all80": (16, [80] * 16),
    "s16-mix10": (16, [160] * 6 + [80] * 4 + [0] * 6),
    "s8-uni640": (8, [640, 160, 160, 160, 160, 0, 0, 0]),
    "s16-uni-split": (16, [320, 320, 160, 160, 160, 160] + [0] * 10),
    "s8-bal160-again": (8, [160] * 8),
}


def model_us(pattern):
    t = 48.0
    for n in pattern:
        if 0 < n <= 320:
            t += 65.9 + 0.166 * n
        elif n > 320:
            t += 26.0 + 68.1 * -(-n // 256)
    return t


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (8, 4),
            _device_params(ttnn.FabricConfig.FABRIC_2D_TORUS_XY, "glm-53"),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="glm-8x4",
        )
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.skipif(not is_blackhole(), reason="Blackhole only")
@pytest.mark.timeout(0)
def test_hre_slots_r0(mesh_device, device_params):
    want = os.environ.get("HRE16_CASES", "all")
    names = [n for n in CASES if want == "all" or n in want.split(",")]
    cfg = GLM53Config
    emb, hidden, topk = cfg.EMB_SIZE, cfg.MOE_INTERMEDIATE_SIZE, cfg.NUM_EXPERTS_PER_TOKEN
    threshold = cfg.ROUTED_EXPERT_HYBRID_TOKEN_THRESHOLD
    mc = extract_mesh_config(mesh_device)
    sp_axis, G, C = mc.sp_axis, mc.num_dispatch_groups, mc.dispatch_group_size
    ndev = mesh_device.get_num_devices()
    num_links = 2
    log(f"mesh {tuple(mesh_device.shape)} groups {G} ring {C} threshold {threshold} iters {ITERS}")

    # One 8-slot weight set (8 distinct slots); 16-slot cases reuse it for slots 8..15.
    t0 = time.time()
    w = {
        "gate_proj": torch.randn(hidden, emb) * 0.02,
        "up_proj": torch.randn(hidden, emb) * 0.02,
        "down_proj": torch.randn(emb, hidden) * 0.02,
    }
    idx8 = ExpertMapping.create_global_expert_idx_table(
        experts_per_chip=8, dispatch_group_size=C, num_dispatch_groups=G
    )
    ep_mapper = get_ep_mesh_mapper(mesh_device)
    tt_idx8 = ttnn.squeeze(ttnn.squeeze(_int_tensor(idx8, mesh_device, ep_mapper, dtype=ttnn.uint32), 0), 0)
    base = TtRoutedExpert(
        mesh_device=mesh_device,
        experts_per_chip=8,
        global_expert_idx_table=tt_idx8,
        emb_dim=emb,
        hidden_dim=hidden,
        max_tokens=C * SEQ,
        torch_weights=[w] * (ndev * 8),
        activation=ttnn.RoutedExpertActivation.Silu,
    )
    log(f"weights built in {time.time() - t0:.0f} s")

    grid = mesh_device.compute_with_storage_grid_size()
    all_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    channels = ttnn.experimental.deepseek_prefill.hybrid_routed_expert_moe_channel_semaphore_count(
        mesh_device, combine_axis=sp_axis
    )
    sems = [ttnn.create_global_semaphore(mesh_device, all_cores, 0) for _ in range(3 + channels)]
    ttnn.synchronize_device(mesh_device)

    results = {}
    for name in names:
        slots, pattern = CASES[name]
        try:
            n_exp = ndev * slots
            experts_per_chip, metadata_len, max_buf, max_per_expert = compute_constants(SEQ, n_exp, topk, ndev, C, CAP)
            assert experts_per_chip == slots
            group_counts = per_chip(slots, pattern)
            gi = _indices_from_counts(group_counts, C, SEQ, topk // G)
            indices = torch.cat([gi + g * (8 * slots) for g in range(G)], dim=-1)
            table = ExpertMapping.create_dispatch_table(
                num_routed_experts=n_exp, dispatch_group_size=C, num_dispatch_groups=G
            )
            offsets, counts, region, _ = get_gate_outputs(
                indices, C, n_exp, slots, SEQ, topk, expert_dispatch_table=table
            )
            # Metadata from the host dispatch at a 32-wide payload (it does not depend on emb); the device buffer
            # itself is zeros at the real width.
            small = 32
            _, meta = TorchDispatchModule(
                dispatch_group_size=C,
                experts_per_chip=slots,
                num_routed_experts=n_exp,
                num_experts_per_tok=topk,
                metadata_len=metadata_len,
                max_dispatched_tokens_per_expert=max_per_expert,
                max_dispatch_buffer_token_size=max_buf,
                seq_len_per_chip=SEQ,
                emb_dim=small,
                num_dispatch_groups=G,
                expert_dispatch_table=table,
            )(torch.zeros(C, SEQ, small, dtype=torch.bfloat16), torch.ones(C, SEQ, topk), indices, offsets)
            per_dev_shape = [1, 1] + list(meta.shape[2:-1]) + [emb]
            try:
                tt_x = ttnn.zeros(
                    ttnn.Shape(per_dev_shape),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            except Exception as e:  # fall back to a host upload of the full buffer (slow but known-good)
                log(f"ttnn.zeros failed ({type(e).__name__}); uploading zeros from host")
                tt_x = ttnn.from_torch(
                    torch.zeros(G, C, per_dev_shape[2], emb, dtype=torch.bfloat16),
                    mesh_mapper=ep_mapper,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=mesh_device,
                    dtype=ttnn.bfloat16,
                )
            assert list(tt_x.shape) == per_dev_shape, (list(tt_x.shape), per_dev_shape)
            tt_meta = _int_tensor(meta, mesh_device, ep_mapper)
            cm = get_expert_token_counts_mesh_mapper(mesh_device)
            tt_counts = ttnn.squeeze(_int_tensor(counts, mesh_device, cm, dtype=ttnn.uint32), 0)
            tt_region = ttnn.squeeze(_int_tensor(region, mesh_device, cm, dtype=ttnn.uint32), 0)
            rep_ring = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(None, 0))
            tt_offsets = _int_tensor(offsets, mesh_device, rep_ring, dtype=ttnn.uint32)
            idx = ExpertMapping.create_global_expert_idx_table(
                experts_per_chip=slots, dispatch_group_size=C, num_dispatch_groups=G
            )
            tt_idx = ttnn.squeeze(ttnn.squeeze(_int_tensor(idx, mesh_device, ep_mapper, dtype=ttnn.uint32), 0), 0)
            tt_idx_full = _int_tensor(idx, mesh_device, rep_ring, dtype=ttnn.uint32)
            reps = slots // 8
            gates, ups, downs = base.gate_projs * reps, base.up_projs * reps, base.down_projs * reps
            wtable = routed_expert_weight_address_table(mesh_device, gates, ups, downs)

            def run():
                return ttnn.experimental.deepseek_prefill.hybrid_routed_expert_moe(
                    tt_x,
                    tt_region,
                    tt_counts,
                    tt_idx,
                    gates,
                    ups,
                    downs,
                    max_dispatched_tokens_per_expert=max_per_expert,
                    hybrid_token_threshold=threshold,
                    compute_kernel_config=base.compute_kernel_config,
                    activation=ttnn.RoutedExpertActivation.Silu,
                    weight_address_table=wtable,
                    dispatched_metadata=tt_meta,
                    expert_offsets=tt_offsets,
                    replicated_global_expert_idx_table=tt_idx_full,
                    combine_axis=sp_axis,
                    combine_num_links=num_links,
                    num_experts_per_tok=topk,
                    seq_len_per_chip=SEQ,
                    fwd_arrived_semaphore=sems[0],
                    final_arrived_semaphore=sems[1],
                    expert_go_semaphore=sems[2],
                    channel_semaphores=sems[3:],
                )

            t1 = time.time()
            out = run()
            ttnn.synchronize_device(mesh_device)
            ttnn.deallocate(out)
            eager_s = time.time() - t1
            trace, out = _capture(mesh_device, run)
            try:
                us = _replay_us(mesh_device, trace, ITERS)
            finally:
                ttnn.release_trace(mesh_device, trace)
                ttnn.deallocate(out)
            m = model_us(pattern)
            results[name] = us
            active = sum(1 for n in pattern if n)
            log(
                f"CASE {name}: slots={slots} active={active} counts/chip={pattern} -> overlap {us:.1f} us | "
                f"model RE {m:.0f} us (+ tail) | eager first call {eager_s:.1f} s"
            )
            for t in (tt_x, tt_meta, tt_counts, tt_region, tt_offsets, tt_idx, tt_idx_full, wtable):
                ttnn.deallocate(t)
        except Exception as e:
            log(f"FAIL {name}: {type(e).__name__}: {str(e)[:600]}")
            traceback.print_exc()
            if "s16" not in name:
                raise
    if "s8-bal160" in results:
        b = results["s8-bal160"]
        for name, us in results.items():
            slots, pattern = CASES[name]
            log(
                f"SUMMARY {name}: {us:.1f} us, vs s8-bal160 {us - b:+.1f} us; model delta {model_us(pattern) - model_us(CASES['s8-bal160'][1]):+.0f}"
            )
    assert "s8-bal160" in results, "control case failed"
