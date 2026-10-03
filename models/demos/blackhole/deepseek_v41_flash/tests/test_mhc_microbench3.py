import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference.ref_layer import pcc
from models.demos.blackhole.deepseek_v41_flash.tests.test_mhc_microbench import traced_ms

D, T = 5120, 4


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
def test_mb3(mesh_device):
    md = mesh_device
    rep = ttnn.ReplicateTensorToMesh(md)
    up = lambda t, dt=ttnn.float32: ttnn.from_torch(
        t, device=md, dtype=dt, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep
    )
    ckc = ttnn.init_device_compute_kernel_config(
        md.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    for S in (32,):
        Ek = 4 * D // S
        xt_h = torch.randn(T, 1, 4, D)
        fn_h = torch.randn(4 * D, 24) * 0.02
        xt = up(xt_h)
        fnr = up(fn_h.reshape(S, 1, Ek, 24))

        def mixes_fast():
            xp = ttnn.reshape(xt, [T, 1, S, Ek])  # split packed columns
            xr = ttnn.permute(xp, (2, 1, 0, 3))  # [S,1,T,Ek]
            mx = ttnn.matmul(xr, fnr, compute_kernel_config=ckc)  # [S,1,T,24]
            ss = ttnn.sum(ttnn.multiply(xr, xr), dim=-1, keepdim=True)  # [S,1,T,1]
            return ttnn.sum(ttnn.concat([mx, ss], dim=-1), dim=0, keepdim=True)  # [1,1,T,25]

        r = mixes_fast()
        rh = ttnn.to_torch(ttnn.get_device_tensors(r)[0]).float().reshape(T, -1)
        xpk = xt_h.reshape(T, 4 * D)
        print(f"MB3 S={S} pcc mx", pcc(rh[:, :24], xpk @ fn_h), "pcc ss", pcc(rh[:, 24], (xpk**2).sum(-1)))
        print(f"MB3 S={S} mixes_fast {traced_ms(md, mixes_fast):.3f}")
        print(
            f"MB3 S={S} reshape+permute only {traced_ms(md, lambda: ttnn.permute(ttnn.reshape(xt, [T,1,S,Ek]), (2,1,0,3))):.3f}"
        )

        xr = ttnn.permute(ttnn.reshape(xt, [T, 1, S, Ek]), (2, 1, 0, 3))
        mx = ttnn.matmul(xr, fnr, compute_kernel_config=ckc)
        x2 = ttnn.multiply(xr, xr)
        ss = ttnn.sum(x2, dim=-1, keepdim=True)
        cc = ttnn.concat([mx, ss], dim=-1)
        for name, f in {
            "matmul": lambda: ttnn.matmul(xr, fnr, compute_kernel_config=ckc),
            "x*x": lambda: ttnn.multiply(xr, xr),
            "sum last": lambda: ttnn.sum(x2, dim=-1, keepdim=True),
            "concat": lambda: ttnn.concat([mx, ss], dim=-1),
            "sum dim0": lambda: ttnn.sum(cc, dim=0, keepdim=True),
            "sum dim0 mx": lambda: ttnn.sum(mx, dim=0, keepdim=True),
        }.items():
            print(f"MB3 stage {name:12s} {traced_ms(md, f):.3f}")
        lofi = ttnn.init_device_compute_kernel_config(
            md.arch(), math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        pc = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 4),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=1,
        )
        xb, fb = ttnn.typecast(xr, ttnn.bfloat16), ttnn.typecast(fnr, ttnn.bfloat16)
        tests = {
            "lofi": lambda: ttnn.matmul(xr, fnr, compute_kernel_config=lofi),
            "core_grid 4x8": lambda: ttnn.matmul(xr, fnr, compute_kernel_config=ckc, core_grid=ttnn.CoreGrid(y=4, x=8)),
            "reuse prog cfg": lambda: ttnn.matmul(xr, fnr, compute_kernel_config=ckc, program_config=pc),
            "bf16": lambda: ttnn.matmul(xb, fb, compute_kernel_config=ckc),
            "bf16 out bf16 lofi": lambda: ttnn.matmul(xb, fb, compute_kernel_config=lofi, dtype=ttnn.bfloat16),
            "bmm via mul+sum (bcast)": lambda: ttnn.sum(
                ttnn.multiply(ttnn.reshape(xr, [S, 1, T, 1, Ek]) if False else xr, xr), dim=-1
            ),
        }
        for name, f in tests.items():
            try:
                print(f"MB3 var {name:26s} {traced_ms(md, f):.3f}")
            except Exception as e:
                print(f"MB3 var {name:26s} FAIL {str(e)[:100]}")
