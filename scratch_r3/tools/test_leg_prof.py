# #58817 legacy port: the per-device matmul of all_gather_matmul_async in tt_transformers and attention_1d decode on a
# LoudBox (8 x P150, Ring), on one device. TT_MM_LEGACY_1D routes the 1D config to the legacy builder, which builds the
# same compute kernel and arguments as the CCL op's matmul helper (the fused-op signaler only changes the readers).
# Each repetition runs MM_CALLS launches on the same inputs; mm_prof_plugin sums them per repetition.
import os as _os
import sys as _sys

if not (_os.environ.get("HWLOCK_HELD") or _os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH")):
    _sys.exit("not under hwlock")
_os.environ["TT_MM_LEGACY_1D"] = "1"
import pytest
import torch
import ttnn

N_CALLS = int(_os.environ.get("MM_CALLS", "16"))
BF, B8 = ttnn.bfloat16, ttnn.bfloat8_b
FID = {"HiFi2": ttnn.MathFidelity.HiFi2, "HiFi4": ttnn.MathFidelity.HiFi4}
NUM_DEVICES = 8
# (name, model dim, wo dtype, fidelity, math approx): models/tt_transformers/tt/model_config.py:2081 (program config),
# :928 (compute_kernel_config_hifi2, the call's config), :940 (hifi4, attention_1d's LI_O_DECODE for Qwen accuracy)
CASES = [
    ("agmm_d2048_bfp8_hifi2", 2048, B8, "HiFi2", True),  # Llama 3.2 1B
    ("agmm_d4096_bfp8_hifi2", 4096, B8, "HiFi2", True),  # Llama 3.1 8B, Mistral 7B
    ("agmm_d8192_bfp8_hifi2", 8192, B8, "HiFi2", True),  # Llama 3.3 70B
    ("agmm_d8192_bf16_hifi2", 8192, BF, "HiFi2", True),  # Qwen2.5 72B (accuracy, WO bf16)
    ("agmm_d8192_bf16_hifi4", 8192, BF, "HiFi4", False),  # Qwen2.5 72B through attention_1d (LI_O_DECODE HiFi4)
]


def out_subblock_w(per_core_n):  # models/tt_transformers/tt/common.py:672 at out_subblock_h 1
    w = 4
    while w > 1 and per_core_n % w:
        w -= 1
    return w


def agmm_setup(device, dim, wdt, fid, approx, seed=0):
    torch.manual_seed(seed)
    k, n = dim, dim // NUM_DEVICES
    in0_mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(NUM_DEVICES - 1, 0))}),
            [32, dim // NUM_DEVICES],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    # without a persistent buffer the CCL op skips the matmul validate and takes the matmul's own output spec: width
    # sharded on the 8 compute cores (matmul_device_operation.cpp, the 1D branch of compute_output_specs)
    out_mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
            [32, n // 8],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    a = ttnn.from_torch(torch.randn(1, 1, 32, k), dtype=BF, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in0_mc)
    b = ttnn.from_torch(
        torch.randn(1, 1, k, n) * 0.05, dtype=wdt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    per_core_n = n // 32 // 8
    pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 1),
        in0_block_w=dim // 32 // 8,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w(per_core_n),
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )
    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=FID[fid], math_approx_mode=approx, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    return a, b, pc, ckc, out_mc


@pytest.fixture(scope="module")
def device():
    dev = ttnn.CreateDevice(device_id=0, l1_small_size=32768)
    ttnn.SetDefaultDevice(dev)
    yield dev
    ttnn.close_device(dev)


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_leg_agmm(device, case):
    name, dim, wdt, fid, approx = case
    a, b, pc, ckc, out_mc = agmm_setup(device, dim, wdt, fid, approx)
    for _ in range(N_CALLS):
        out = ttnn.matmul(a, b, program_config=pc, compute_kernel_config=ckc, memory_config=out_mc, dtype=BF)
        out.deallocate()
    ttnn.synchronize_device(device)
    a.deallocate()
    b.deallocate()
