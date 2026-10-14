# #58817 legacy port: outputs of the legacy bmm kernel (TT_MM_LEGACY_1D routes 1D configs to the legacy builder) saved as
# raw bits, to compare the port against the PR head and main: the all_gather_matmul_async per-device cases, then the
# kernel's other paths (partials reload without L1 accumulation, fused bias, fused activation, in0 and in1 tile
# transposes, mcast_in1, batches, a Float32 output from a 16-bit DEST, the LoFi row MOP gates, the 8-tile sub blocks).
import os as _os
import sys as _sys

if not (_os.environ.get("HWLOCK_HELD") or _os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH")):
    _sys.exit("not under hwlock")
_os.environ["TT_MM_LEGACY_1D"] = "1"
import pytest
import torch
import ttnn

from test_leg_prof import CASES as AGMM, agmm_setup

BF, B8, F32 = ttnn.bfloat16, ttnn.bfloat8_b, ttnn.float32
FID = {"LoFi": ttnn.MathFidelity.LoFi, "HiFi2": ttnn.MathFidelity.HiFi2, "HiFi4": ttnn.MathFidelity.HiFi4}
OUT = _os.environ.get("BITID_OUT")
# (name, B, M, K, N, in0 dt, in1 dt, out dt, fidelity, fp32 dest, l1 acc, mcast_in0, in0_block_w, sub h, sub w,
#  per_core_M, per_core_N, grid (x, y), extra) with extra in {"", "bias", "gelu", "ta", "tb"}
GEN = [
    ("in0_hifi2_2x2_reload", 1, 256, 1024, 512, BF, B8, BF, "HiFi2", False, False, True, 2, 2, 2, 8, 2, (8, 1), ""),
    ("in0_hifi2_1x1_reload", 1, 64, 1024, 256, BF, B8, BF, "HiFi2", True, False, True, 4, 1, 1, 2, 1, (8, 1), ""),
    ("in0_hifi4_2x1_l1acc", 1, 128, 2048, 256, BF, BF, BF, "HiFi4", True, True, True, 4, 2, 1, 4, 1, (8, 1), ""),
    ("in0_hifi4_1x2_single_block", 1, 32, 256, 512, BF, BF, BF, "HiFi4", True, True, True, 8, 1, 2, 1, 2, (8, 1), ""),
    ("in0_lofi_1x6_w6", 1, 64, 1024, 1536, BF, B8, BF, "LoFi", False, True, True, 4, 1, 6, 2, 6, (8, 1), ""),
    ("in0_lofi_2x4_kt", 1, 128, 1024, 1024, BF, B8, BF, "LoFi", False, True, True, 4, 2, 4, 4, 4, (8, 1), ""),
    ("in0_lofi_2x2_tile_mop", 1, 128, 1024, 512, BF, B8, BF, "LoFi", False, True, True, 4, 2, 2, 4, 2, (8, 1), ""),
    ("in0_hifi2_f32out_16bit_dest", 1, 64, 1024, 512, BF, BF, F32, "HiFi2", False, True, True, 4, 2, 2, 2, 2, (8, 1), ""),
    ("in0_hifi2_f32out_fp32_dest", 1, 64, 1024, 512, BF, BF, F32, "HiFi2", True, True, True, 4, 2, 2, 2, 2, (8, 1), ""),
    ("in0_hifi2_bias", 1, 64, 1024, 512, BF, B8, BF, "HiFi2", True, True, True, 4, 1, 2, 2, 2, (8, 1), "bias"),
    ("in0_hifi2_bias_reload", 1, 64, 1024, 512, BF, B8, BF, "HiFi2", False, False, True, 4, 2, 2, 2, 2, (8, 1), "bias"),
    ("in0_hifi2_gelu", 1, 64, 1024, 512, BF, B8, BF, "HiFi2", True, True, True, 4, 1, 2, 2, 2, (8, 1), "gelu"),
    ("in0_hifi2_transpose_a", 1, 64, 1024, 512, BF, BF, BF, "HiFi2", True, True, True, 4, 1, 2, 2, 2, (8, 1), "ta"),
    ("in0_hifi2_transpose_b", 1, 64, 1024, 512, BF, BF, BF, "HiFi2", True, True, True, 4, 1, 2, 2, 2, (8, 1), "tb"),
    ("in1_hifi2_2x2", 1, 1024, 512, 64, BF, B8, BF, "HiFi2", True, True, False, 4, 2, 2, 4, 2, (8, 1), ""),
    ("in1_hifi4_4x2_reload", 1, 1024, 512, 64, BF, BF, BF, "HiFi4", False, False, False, 4, 4, 2, 4, 2, (8, 1), ""),
    ("in0_hifi2_batch3", 3, 64, 512, 512, BF, B8, BF, "HiFi2", True, True, True, 4, 1, 2, 2, 2, (8, 1), ""),
]


@pytest.fixture(scope="module")
def device():
    dev = ttnn.CreateDevice(device_id=0, l1_small_size=32768)
    ttnn.SetDefaultDevice(dev)
    yield dev
    ttnn.close_device(dev)


def save(name, out):
    if OUT:
        t = ttnn.to_torch(out).contiguous()
        torch.save(t.view(torch.int16) if t.dtype == torch.bfloat16 else t.view(torch.int32), _os.path.join(OUT, f"leg_{name}.pt"))


@pytest.mark.parametrize("case", AGMM, ids=[c[0] for c in AGMM])
def test_leg_bits_agmm(device, case):
    name, dim, wdt, fid, approx = case
    a, b, pc, ckc, out_mc = agmm_setup(device, dim, wdt, fid, approx, seed=1)
    out = ttnn.matmul(a, b, program_config=pc, compute_kernel_config=ckc, memory_config=out_mc, dtype=BF)
    save(name, out)


@pytest.mark.parametrize("case", GEN, ids=[c[0] for c in GEN])
def test_leg_bits_gen(device, case):
    name, bt, m, k, n, d0, d1, do, fid, fp32, l1acc, mc0, ibw, sh, sw, pm, pn, grid, extra = case
    torch.manual_seed(2)
    D = ttnn.DRAM_MEMORY_CONFIG
    ta, tb = extra == "ta", extra == "tb"
    a_shape = (bt, 1, k, m) if ta else (bt, 1, m, k)
    b_shape = (1, 1, n, k) if tb else (1, 1, k, n)
    a = ttnn.from_torch(torch.randn(a_shape), dtype=d0, layout=ttnn.TILE_LAYOUT, device=device, memory_config=D)
    b = ttnn.from_torch(torch.randn(b_shape) * 0.05, dtype=d1, layout=ttnn.TILE_LAYOUT, device=device, memory_config=D)
    pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=ibw,
        out_subblock_h=sh,
        out_subblock_w=sw,
        per_core_M=pm,
        per_core_N=pn,
        fuse_batch=bt == 1,
        fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU) if extra == "gelu" else None,
        mcast_in0=mc0,
    )
    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=FID[fid], math_approx_mode=False, fp32_dest_acc_en=fp32, packer_l1_acc=l1acc
    )
    kw = dict(program_config=pc, compute_kernel_config=ckc, memory_config=D, dtype=do, transpose_a=ta, transpose_b=tb)
    if extra == "bias":
        bias = ttnn.from_torch(torch.randn(1, 1, 32, n), dtype=BF, layout=ttnn.TILE_LAYOUT, device=device, memory_config=D)
        out = ttnn.linear(a, b, bias=bias, **kw)
    else:
        out = ttnn.matmul(a, b, **kw)
    save(name, out)
