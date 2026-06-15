# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""A10 step 1: standalone sparse-MoE round-trip on the native 1x8 mesh with mistral4-style dims, vs a
torch dense top-k reference. Validates the FULL dispatch -> real local experts (SwiGLU) -> combine ->
weight+sum pipeline BEFORE refactoring the model — PCC 0.9998. KEY: experts shard 16/device across all 8
(ShardTensorToMesh dim0, expert_mapping e->e//16); TOKENS are SHARDED 1/device (NOT replicated — that was
the original 1x8 _forward_sparse bug); all_to_all_dispatch/combine cluster_axis=1 spans all 8 on (1,8).
(cluster_axis=None HANGS+degrades the device; a 2x4 mesh with cluster_axis=1 row-splits experts.)
hidden=4096, experts=128, k=4, reduced interm for speed."""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

MESH = (1, 8)  # cluster_axis=None hangs on this BH build; (1,8)+cluster_axis=1 spans all 8 for dispatch
DEVICES = 8
EXPERTS = 128
K = 4
HIDDEN = 4096
INTERM = 1024  # reduced (real mistral4 interm is larger) — this validates the pipeline, not the weights
BATCH = 8  # global tokens (sharded 1/device across all 8 for dispatch)
AXIS = 1  # (1,8) mesh: axis-1 spans all 8 devices, so dispatch reaches every expert (probe PASSED here).
# all_to_all_combine REQUIRES input_shape[0] == experts/num_devices: experts sharded across ALL 8 (16/dev).
PER = EXPERTS // DEVICES  # 16 local experts/device


@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000}], indirect=True
)
def test_m4_moe_roundtrip_2x4(mesh_device, reset_seeds):
    torch.manual_seed(0)
    x = torch.randn(BATCH, HIDDEN) * 0.1
    gup_w = torch.randn(EXPERTS, HIDDEN, 2 * INTERM) * 0.02  # [E,H,2I]
    down_w = torch.randn(EXPERTS, INTERM, HIDDEN) * 0.02  # [E,I,H]
    gate_logits = torch.randn(BATCH, EXPERTS)
    tw, ti = torch.topk(torch.softmax(gate_logits, -1), K, dim=-1)  # [B,K]
    tw = tw / tw.sum(-1, keepdim=True)

    # torch dense top-k reference
    ref = torch.zeros(BATCH, HIDDEN)
    for b in range(BATCH):
        for j in range(K):
            e = ti[b, j].item()
            gu = x[b] @ gup_w[e]
            h = torch.nn.functional.silu(gu[:INTERM]) * gu[INTERM:]
            ref[b] += tw[b, j] * (h @ down_w[e])

    # --- device CCL plumbing (mirrors test_moe_ccl_end_to_end) ---
    grid = mesh_device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    sdm = mesh_device.create_sub_device_manager([ttnn.SubDevice([crs])], 0)
    mesh_device.load_sub_device_manager(sdm)
    mesh_device.set_sub_device_stall_group([ttnn.SubDeviceId(0)])

    shard0 = ttnn.ShardTensorToMesh(mesh_device, dim=0)  # tokens: shard batch (1/device) across all 8
    repl = ttnn.ReplicateTensorToMesh(mesh_device)
    tt_x = ttnn.from_torch(
        x.reshape(BATCH, 1, 1, HIDDEN),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=shard0,
    )
    tt_idx = ttnn.from_torch(
        ti.reshape(BATCH, 1, 1, K),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint16,
        mesh_mapper=shard0,
    )
    expert_map = torch.eye(DEVICES, dtype=torch.int32).repeat_interleave(PER, dim=0).reshape(1, 1, EXPERTS, DEVICES)
    tt_map = ttnn.from_torch(
        expert_map, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint16, mesh_mapper=repl
    )

    disp, meta = ttnn.all_to_all_dispatch(
        tt_x, tt_idx, tt_map, num_links=1, cluster_axis=AXIS, subdevice_id=ttnn.SubDeviceId(0)
    )  # disp [devices, batch, 1, hidden] sharded dim0; meta [devices, batch, 1, k]

    # real local experts: broadcast each device's dispatched tokens over its PER local experts
    d = ttnn.repeat(disp, ttnn.Shape([PER, 1, 1, 1]))  # [E(/dev), batch, 1, hidden] sharded dim0
    bsz = d.shape[1]
    d = ttnn.to_layout(ttnn.reshape(d, (PER, bsz, HIDDEN)), ttnn.TILE_LAYOUT)
    # experts (dim0) sharded 16/device across all 8 (op requires experts/num_devices); row-major device order
    exp_map = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    gup_sh = ttnn.from_torch(
        gup_w, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=exp_map
    )
    down_sh = ttnn.from_torch(
        down_w, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=exp_map
    )
    gu = ttnn.matmul(d, gup_sh)  # [PER, bsz, 2I]
    h = ttnn.mul(
        ttnn.silu(ttnn.slice(gu, [0, 0, 0], [PER, bsz, INTERM])), ttnn.slice(gu, [0, 0, INTERM], [PER, bsz, 2 * INTERM])
    )
    eo = ttnn.matmul(h, down_sh)  # [PER, bsz, hidden]
    eo = ttnn.to_layout(ttnn.reshape(eo, (PER, bsz, 1, HIDDEN)), ttnn.ROW_MAJOR_LAYOUT)

    comb = ttnn.all_to_all_combine(
        eo, meta, tt_map, num_links=1, cluster_axis=AXIS, subdevice_id=ttnn.SubDeviceId(0)
    )  # [k, batch(/dev), 1, hidden]
    comb_t = ttnn.to_torch(comb, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1)).float()
    logger.info(
        f"A10 2x4 round-trip: disp {tuple(disp.shape)} meta {tuple(meta.shape)} comb {tuple(comb.shape)} -> host {tuple(comb_t.shape)}"
    )

    # comb_t [k, devices*batch_per_dev, 1, hidden] = [4, 16, 1, 4096]: 16 = 8 unique tokens (4 EP cols x
    # 2/col) x 2 DP-row replicas. Take DP row 0 (first BATCH entries, devices 0-3 -> tokens 0..7 in order).
    comb_bk = comb_t[:, :BATCH, 0, :]  # [K, BATCH, hidden]

    # DIAGNOSTIC: for token 0, identify which torch expert each device combine-slice matches (un-weighted),
    # to localize the residual (k-order vs token-order vs placement).
    with torch.no_grad():
        b0_exp_out = []  # token-0's 4 experts' raw outputs in ti order
        for j in range(K):
            e = ti[0, j].item()
            gu0 = x[0] @ gup_w[e]
            h0 = torch.nn.functional.silu(gu0[:INTERM]) * gu0[INTERM:]
            b0_exp_out.append(h0 @ down_w[e])

    def _cos(a, b):
        return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()

    for j in range(K):
        dev_slice = comb_bk[j, 0]  # [hidden]
        sims = [round(_cos(b0_exp_out[m], dev_slice), 3) for m in range(K)]
        logger.info(f"A10 diag token0 comb-slice {j} (expert ti[0,{j}]={ti[0,j].item()}): cos vs ti-experts {sims}")

    out = (comb_bk * tw.t().reshape(K, BATCH, 1)).sum(0)  # [BATCH, hidden]
    passing, msg = comp_pcc(ref, out, 0.98)
    logger.info(f"A10 2x4 sparse-MoE round-trip vs torch dense top-{K} PCC: {msg}")
    assert passing, f"2x4 sparse-MoE round-trip PCC below 0.98: {msg}"
