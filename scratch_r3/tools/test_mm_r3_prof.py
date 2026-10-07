# Round 3 matmul: ttnn.matmul on production-like shapes under the device profiler (one launch per repetition), to compare main
# against the branch: prefill shapes whose sub blocks hold eight tiles (row MOP) and decode shapes with one-row and one-tile sub blocks
# (unpack k loop), bf16 and bfp8 inputs, LoFi to HiFi4, fp32 DEST on and off.
import os as _os
import sys as _sys

if not (_os.environ.get("HWLOCK_HELD") or _os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH")):
    _sys.exit("not under hwlock")
import pytest
import torch
import ttnn

CASES = [
    # (name, M, K, N, in0 dtype, in1 dtype, fidelity, fp32 dest)
    ("prefill_1k_2k_2k_bf16_hifi2", 1024, 2048, 2048, "bf16", "bf16", "HiFi2", False),
    ("prefill_2k_2k_8k_bfp8_lofi", 2048, 2048, 8192, "bf16", "bfp8", "LoFi", False),
    ("prefill_1k_1k_1k_bf16_hifi4", 1024, 1024, 1024, "bf16", "bf16", "HiFi4", False),
    ("prefill_512_4k_4k_bfp8_hifi2", 512, 4096, 4096, "bf16", "bfp8", "HiFi2", False),
    ("prefill_1k_1k_1k_bf16_hifi4_fp32", 1024, 1024, 1024, "bf16", "bf16", "HiFi4", True),
    ("prefill_2k_4k_1k_bf16_lofi", 2048, 4096, 1024, "bf16", "bf16", "LoFi", False),
    ("decode_32_4k_4k_bfp8_lofi", 32, 4096, 4096, "bf16", "bfp8", "LoFi", False),
    ("decode_32_2k_8k_bf16_hifi2", 32, 2048, 8192, "bf16", "bf16", "HiFi2", False),
    ("decode_32_8k_1k_bfp8_lofi", 32, 8192, 1024, "bf16", "bfp8", "LoFi", False),
    ("decode_32_4k_14k_bfp8_hifi2", 32, 4096, 14336, "bf16", "bfp8", "HiFi2", False),
    ("decode_32_4k_1k_bf16_hifi2_fp32", 32, 4096, 1024, "bf16", "bf16", "HiFi2", True),
    ("decode_32_14k_1k_bfp8_lofi", 32, 14336, 1024, "bf16", "bfp8", "LoFi", False),
    ("decode_32_2k_2k_bf16_lofi", 32, 2048, 2048, "bf16", "bf16", "LoFi", False),
    ("small_32_1k_64_bf16_lofi", 32, 1024, 64, "bf16", "bf16", "LoFi", False),
    ("small_64_2k_96_bf16_lofi_fp32", 64, 2048, 96, "bf16", "bf16", "LoFi", True),
]
DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}


@pytest.fixture(scope="module")
def device():
    from tests.tests_common.cache_entries_counter import CacheEntriesCounter

    dev = ttnn.CreateDevice(device_id=0, l1_small_size=32768)
    ttnn.SetDefaultDevice(dev)
    dev.cache_entries_counter = CacheEntriesCounter(dev)
    yield dev
    ttnn.close_device(dev)
FID = {"LoFi": ttnn.MathFidelity.LoFi, "HiFi2": ttnn.MathFidelity.HiFi2, "HiFi4": ttnn.MathFidelity.HiFi4}


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_mm_prof(device, case):
    name, m, k, n, d0, d1, fid, fp32 = case
    torch.manual_seed(0)
    a = ttnn.from_torch(torch.randn(1, 1, m, k) * 0.1, dtype=DT[d0], layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(torch.randn(1, 1, k, n) * 0.1, dtype=DT[d1], layout=ttnn.TILE_LAYOUT, device=device)
    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=FID[fid], math_approx_mode=False, fp32_dest_acc_en=fp32, packer_l1_acc=True
    )
    out = ttnn.matmul(a, b, compute_kernel_config=ckc)
    ttnn.synchronize_device(device)
    out.deallocate()
    a.deallocate()
    b.deallocate()


# Other ops whose unpack goes through the metal matmul init (the CFGSHIFTMASK address advance for every format).
SDPA_CASES = [
    # (name, b, nh, nkv, s, d, dtype, q chunk, k chunk, fidelity)
    ("sdpa_prefill_8h_2k_d128_bf16_hifi2", 1, 8, 8, 2048, 128, "bf16", 128, 128, "HiFi2"),
    ("sdpa_prefill_32h_8kv_1k_d128_bfp8_hifi2", 1, 32, 8, 1024, 128, "bfp8", 256, 256, "HiFi2"),
    ("sdpa_prefill_16h_4k_d64_bf16_hifi4", 1, 16, 16, 4096, 64, "bf16", 128, 256, "HiFi4"),
]


@pytest.mark.parametrize("case", SDPA_CASES, ids=[c[0] for c in SDPA_CASES])
def test_sdpa_prof(device, case):
    name, b, nh, nkv, s, d, dt, qc, kc, fid = case
    torch.manual_seed(0)
    q = ttnn.from_torch(torch.randn(b, nh, s, d), dtype=DT[dt], layout=ttnn.TILE_LAYOUT, device=device)
    k = ttnn.from_torch(torch.randn(b, nkv, s, d), dtype=DT[dt], layout=ttnn.TILE_LAYOUT, device=device)
    v = ttnn.from_torch(torch.randn(b, nkv, s, d), dtype=DT[dt], layout=ttnn.TILE_LAYOUT, device=device)
    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=qc,
        k_chunk_size=kc,
        exp_approx_mode=True,
    )
    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=FID[fid], math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
    )
    out = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True, program_config=pc, compute_kernel_config=ckc)
    ttnn.synchronize_device(device)
    out.deallocate()
    q.deallocate()
    k.deallocate()
    v.deallocate()


# Production program configs of Blackhole models: the DeepSeek V3 prefill MLA wkv_b1 matmuls (models/demos/deepseek_v3_d_p,
# sub blocks of eight tiles at in0_block_w 2 and 4, the row MOP), and a decode matmul with one-tile sub blocks at
# in0_block_w 8 (the unpack k loop).
def _cfg(name):
    grid = ttnn.CoreCoord(11, 10)
    if name == "mla_wkv_b1_1d":
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=grid, in0_block_w=4, out_subblock_h=1, out_subblock_w=8,
            per_core_M=1, per_core_N=16, fuse_batch=False, mcast_in0=False,
        )
    if name == "mla_wkv_b1_reuse":
        return ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=grid, in0_block_w=2, out_subblock_h=2, out_subblock_w=4, per_core_M=4, per_core_N=16,
        )
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid, in0_block_w=8, out_subblock_h=1, out_subblock_w=1,
        per_core_M=1, per_core_N=1, fuse_batch=True, mcast_in0=True,
    )


CFG_CASES = [
    # (name, in0 shape, in1 shape, in1 dtype, fidelity)
    ("mla_wkv_b1_1d", (1, 32, 640, 128), (1, 32, 128, 512), "bfp8", "HiFi2"),
    ("mla_wkv_b1_reuse", (1, 16, 640, 128), (1, 16, 128, 512), "bfp8", "HiFi2"),
    ("decode_1x1_kw8", (1, 1, 32, 8192), (1, 1, 8192, 1024), "bfp8", "LoFi"),
]


@pytest.mark.parametrize("case", CFG_CASES, ids=[c[0] for c in CFG_CASES])
def test_mm_cfg_prof(device, case):
    name, s0, s1, d1, fid = case
    torch.manual_seed(0)
    a = ttnn.from_torch(torch.randn(s0) * 0.1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(torch.randn(s1) * 0.1, dtype=DT[d1], layout=ttnn.TILE_LAYOUT, device=device)
    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=FID[fid], math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    out = ttnn.matmul(a, b, program_config=_cfg(name), compute_kernel_config=ckc)
    ttnn.synchronize_device(device)
    out.deallocate()
    a.deallocate()
    b.deallocate()
