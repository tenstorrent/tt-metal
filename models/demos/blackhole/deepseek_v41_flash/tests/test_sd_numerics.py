"""(sd overlap numerics) stage-by-stage bit comparison of the shared expert of shared_big (auto configs, fused w01) and of SDOverlap.shared (split weights, explicit configs, sub-device)."""
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.moe_overlap import SDOverlap, mm_cfg
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_layer import shared_big
from models.demos.blackhole.deepseek_v41_flash.tt.shared_expert_v2 import DSV41SharedExpertV2

D = 5120


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 100_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@torch.no_grad()
def test_sd_numerics(mesh_device):
    md = mesh_device
    M = int(os.environ.get("DSV41_SDN_M", "512"))
    g_ = torch.Generator().manual_seed(5)
    w0, w1 = [(torch.randn(1, 1, D, 2304, generator=g_) * 0.02) for _ in range(2)]
    w2 = torch.randn(1, 1, 2304, D, generator=g_) * 0.02
    sh = DSV41SharedExpertV2(md, w0, w1, w2)
    ov = SDOverlap.get(md)
    ov.split_weights(sh)
    x = torch.randn(1, 1, M, D, generator=g_).to(torch.bfloat16)
    h = ttnn.from_torch(
        x,
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )
    dev = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()
    eq = lambda n, a, b: print(
        f"SDN {n}: equal {bool(torch.equal(a, b))} maxdiff {float((a - b).abs().max()):.3e} nneq {int((a != b).sum())}/{a.numel()}",
        flush=True,
    )
    kt, nt, mt = D // 32, sh.inter // 32, M // 32
    from models.demos.blackhole.deepseek_v41_flash.tt import pf_tune

    ckc = pf_tune.shared_ckc(sh, md)
    gu = ttnn.linear(
        h,
        sh.w01,
        dtype=sh.mid_dtype,
        compute_kernel_config=ckc,
        core_grid=ttnn.CoreGrid(y=8, x=8),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    gu_t = dev(gu)
    n = sh.inter
    g_ref, u_ref = gu_t[..., :n], gu_t[..., n:]
    # B: explicit config, split weights, full default grid 8x8 (no sub-device)
    for name, grid in (("8x8", (8, 8)), ("sgrid", ov.s_grid), ("8x4", (8, 4)), ("12x9", (12, 9))):
        pc = mm_cfg(grid, mt, kt, nt)
        try:
            gB = ttnn.matmul(
                h,
                sh.wg,
                program_config=pc,
                compute_kernel_config=ckc,
                dtype=sh.mid_dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            eq(f"gate split-weight explicit {name} (in0_block_w {pc.in0_block_w}) vs fused auto", dev(gB), g_ref)
        except Exception as e:  # noqa
            print("SDN", name, "failed", str(e)[:100])
    # same weights split but auto config
    gC = ttnn.linear(
        h,
        sh.wg,
        dtype=sh.mid_dtype,
        compute_kernel_config=ckc,
        core_grid=ttnn.CoreGrid(y=8, x=8),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    eq("gate split-weight AUTO 8x8 vs fused auto", dev(gC), g_ref)
    # fused weight with explicit config? (n = 2*inter)
    pc = mm_cfg((8, 8), mt, kt, 2 * nt)
    gD = ttnn.matmul(
        h,
        sh.w01,
        program_config=pc,
        compute_kernel_config=ckc,
        dtype=sh.mid_dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    eq("fused w01 explicit 8x8 vs fused auto", dev(gD), gu_t)
    # sub-device
    md.load_sub_device_manager(ov.mgr)
    pc = mm_cfg(ov.s_grid, mt, kt, nt)
    gE = ttnn.matmul(
        h,
        sh.wg,
        program_config=pc,
        compute_kernel_config=ckc,
        dtype=sh.mid_dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        sub_device_id=ov.s_id,
    )
    uE = ttnn.matmul(
        h,
        sh.wu,
        program_config=pc,
        compute_kernel_config=ckc,
        dtype=sh.mid_dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        sub_device_id=ov.s_id,
    )
    actE = ttnn.multiply(
        gE,
        uE,
        input_tensor_a_activations=sh.act_a,
        input_tensor_b_activations=sh.act_b,
        dtype=sh.mid_dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        sub_core_grids=ov.s_cores,
    )
    ttnn.synchronize_device(md)
    md.clear_loaded_sub_device_manager()
    eq("gate subdevice vs fused auto", dev(gE), g_ref)
    eq("up subdevice vs fused auto", dev(uE), u_ref)
    gv, uv = ttnn.from_torch(
        g_ref.to(torch.bfloat16),
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    ), ttnn.from_torch(
        u_ref.to(torch.bfloat16),
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )
    act_ref = ttnn.multiply(
        gv,
        uv,
        input_tensor_a_activations=sh.act_a,
        input_tensor_b_activations=sh.act_b,
        dtype=sh.mid_dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    md.load_sub_device_manager(ov.mgr)
    act_sc = ttnn.multiply(
        gv,
        uv,
        input_tensor_a_activations=sh.act_a,
        input_tensor_b_activations=sh.act_b,
        dtype=sh.mid_dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        sub_core_grids=ov.s_cores,
    )
    ttnn.synchronize_device(md)
    md.clear_loaded_sub_device_manager()
    eq("multiply sub_core_grids vs default (same inputs)", dev(act_sc), dev(act_ref))
    eq("act (sd g,u,sub_core_grids multiply) vs act_ref", dev(actE), dev(act_ref))
    # down projection
    out_ref = ttnn.linear(
        act_ref,
        sh.w2,
        dtype=ttnn.float32,
        compute_kernel_config=ckc,
        core_grid=ttnn.CoreGrid(y=8, x=8),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    outs = {}
    for name, grid in (("8x8", (8, 8)), ("sgrid", ov.s_grid), ("12x9", (12, 9))):
        pc = mm_cfg(grid, mt, nt, kt)
        try:
            o = ttnn.matmul(
                act_ref,
                sh.w2,
                program_config=pc,
                compute_kernel_config=ckc,
                dtype=ttnn.float32,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            eq(f"down explicit {name} (in0_block_w {pc.in0_block_w}) vs auto", dev(o), dev(out_ref))
        except Exception as e:  # noqa
            print("SDN down", name, "failed", str(e)[:100])
    full = shared_big(sh, h)
    md.load_sub_device_manager(ov.mgr)
    ovo = ov.shared(sh, h)
    ttnn.synchronize_device(md)
    md.clear_loaded_sub_device_manager()
    eq("END-TO-END ov.shared vs shared_big", dev(ovo), dev(full))
    # stage e2e on default grid: act of shared_big recomputed
    gu2 = ttnn.linear(
        h,
        sh.w01,
        dtype=sh.mid_dtype,
        compute_kernel_config=ckc,
        core_grid=ttnn.CoreGrid(y=8, x=8),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    g2, u2 = gu2[:, :, :, : sh.inter], gu2[:, :, :, sh.inter :]
    act2 = ttnn.multiply(
        g2,
        u2,
        input_tensor_a_activations=sh.act_a,
        input_tensor_b_activations=sh.act_b,
        dtype=sh.mid_dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    eq("act via slices of gu vs act_ref (bf16 roundtrip inputs)", dev(act2), dev(act_ref))
    eq("act via slices of gu vs actE", dev(act2), dev(actE))
    o2 = ttnn.linear(
        act2,
        sh.w2,
        dtype=ttnn.float32,
        compute_kernel_config=ckc,
        core_grid=ttnn.CoreGrid(y=8, x=8),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    eq("down(act2) vs shared_big", dev(o2), dev(full))
    eq("down(act_ref) vs shared_big", dev(out_ref), dev(full))
    for bw in (1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24):
        for nm, kk, nn, w_, a_, ref_ in (
            ("gate", kt, nt, sh.wg, h, g_ref),
            ("down", nt, kt, sh.w2, act_ref, dev(full)),
        ):
            if kk % bw:
                continue
            pc = mm_cfg((8, 8), mt, kk, nn, bw=bw)
            o = ttnn.matmul(
                a_,
                w_,
                program_config=pc,
                compute_kernel_config=ckc,
                dtype=sh.mid_dtype if nm == "gate" else ttnn.float32,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            eq(f"SWEEP {nm} bw={bw}", dev(o), ref_)
