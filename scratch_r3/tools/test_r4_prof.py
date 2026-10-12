# Round 3 matmul, third review of #58714 (2026-10-08): the explicit Blackhole model configs whose rows hold 6 or 7 tiles,
# which the candidate's gate newly takes, as single-device slices under the device profiler: sentence_bert's FF2, QKV and
# pre-softmax (models/demos/blackhole/sentence_bert/ttnn/common.py; batch 8 x 384 tokens, bfp8, no compute config, so
# LoFi) and the prefill MLA 1x6 and 1x7 configs of mla_config.py at their per-device shapes (HiFi2, 16-bit DEST, bf16
# activations, bfp8 weights; DeepSeek V3 128 heads, Kimi K2.6 64, Kimi K3 96, GLM-5.3 64 with q_lora 2048, TP 4).
import os as _os
import sys as _sys

if not (_os.environ.get("HWLOCK_HELD") or _os.path.isfile(_os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH", ""))):
    _sys.exit("not under hwlock")
import pytest
import torch
import ttnn

BF, B8 = ttnn.bfloat16, ttnn.bfloat8_b


@pytest.fixture(scope="module")
def device():
    dev = ttnn.CreateDevice(device_id=0, l1_small_size=32768)
    ttnn.SetDefaultDevice(dev)
    yield dev
    ttnn.close_device(dev)


def _save(name, out):
    if _os.environ.get("V12_OUT"):
        t = ttnn.to_torch(out).contiguous()
        torch.save(t.view(torch.int16) if t.dtype == torch.bfloat16 else t, _os.path.join(_os.environ["V12_OUT"], f"r4_{name}.pt"))


def _pc2d(grid, ibw, sw, pm, pn):
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid, in0_block_w=ibw, out_subblock_h=1, out_subblock_w=sw, per_core_M=pm,
        per_core_N=pn, transpose_mcast=False, fused_activation=None,
    )


def _block_sharded(shape, grid):
    return ttnn.create_sharded_memory_config(
        shape, core_grid=ttnn.CoreGrid(y=grid[1], x=grid[0]), strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )


# (name, M, K, N): sentence_bert's FF2 and QKV with its common.py configs
SBERT_LIN = [("sbert_ff2", 3072, 3072, 768), ("sbert_qkv", 3072, 768, 2304)]


@pytest.mark.parametrize("case", SBERT_LIN, ids=[c[0] for c in SBERT_LIN])
def test_r4_sbert_linear(device, case):
    name, m, k, n = case
    torch.manual_seed(0)
    grid = (6, 8)
    a = ttnn.from_torch(torch.randn(1, 1, m, k) * 0.1, dtype=B8, layout=ttnn.TILE_LAYOUT, device=device,
                        memory_config=_block_sharded((1, 1, m, k), grid))
    w = ttnn.from_torch(torch.randn(1, 1, k, n) * 0.05, dtype=B8, layout=ttnn.TILE_LAYOUT, device=device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    b = ttnn.from_torch(torch.randn(1, 1, 1, n) * 0.05, dtype=B8, layout=ttnn.TILE_LAYOUT, device=device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = ttnn.linear(a, w, bias=b, memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
                      program_config=_pc2d(grid, 4, 6, 12, 12), dtype=B8)
    ttnn.synchronize_device(device)
    _save(name, out)
    for t in (out, a, w, b):
        t.deallocate()


def test_r4_sbert_pre_softmax(device):
    torch.manual_seed(0)
    q = ttnn.from_torch(torch.randn(8, 12, 384, 64) * 0.5, dtype=B8, layout=ttnn.TILE_LAYOUT, device=device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    kt = ttnn.from_torch(torch.randn(8, 12, 64, 384) * 0.5, dtype=B8, layout=ttnn.TILE_LAYOUT, device=device,
                         memory_config=ttnn.DRAM_MEMORY_CONFIG)
    pc = ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=(6, 8), in0_block_w=2, out_subblock_h=1, out_subblock_w=6, per_core_M=24,
        per_core_N=12,
    )
    out = ttnn.matmul(q, kt, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=B8, program_config=pc)
    ttnn.synchronize_device(device)
    _save("sbert_pre_softmax", out)
    for t in (out, q, kt):
        t.deallocate()


# (name, M rows, K, N, in0_block_w, sub w, per_core_M, per_core_N): the 1x6 and 1x7 entries of MLA_MATMUL_CONFIG
MLA = [
    ("glm_q_a_640", 640, 1536, 2048, 8, 6, 2, 6),
    ("kimi3_q_b_640", 640, 1536, 4608, 8, 7, 2, 14),
    ("glm_q_b_640", 640, 2048, 4096, 8, 6, 2, 12),
    ("dsv3_q_b_4096", 4096, 1536, 6144, 4, 6, 13, 18),
    ("dsv3_q_b_3200", 3200, 1536, 6144, 4, 6, 10, 18),
    ("kimi26_o_640", 640, 2048, 7168, 8, 7, 2, 21),
    ("kimi3_o_640", 640, 3072, 7168, 8, 7, 2, 21),
    ("glm_o_640", 640, 4096, 6144, 8, 6, 2, 18),
    ("dsv3_o_4096", 4096, 4096, 7168, 8, 7, 13, 21),
    ("dsv3_o_3200", 3200, 4096, 7168, 8, 7, 10, 21),
    ("glm_idx_wq_b_640", 640, 2048, 4096, 8, 6, 2, 12),
]


@pytest.mark.parametrize("case", MLA, ids=[c[0] for c in MLA])
def test_r4_mla(device, case):
    name, m, k, n, ibw, sw, pm, pn = case
    torch.manual_seed(0)
    D = ttnn.DRAM_MEMORY_CONFIG
    a = ttnn.from_torch(torch.randn(1, 1, m, k) * 0.1, dtype=BF, layout=ttnn.TILE_LAYOUT, device=device, memory_config=D)
    w = ttnn.from_torch(torch.randn(1, 1, k, n) * 0.02, dtype=B8, layout=ttnn.TILE_LAYOUT, device=device, memory_config=D)
    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    out = ttnn.linear(a, w, memory_config=D, program_config=_pc2d((11, 10), ibw, sw, pm, pn), dtype=BF,
                      compute_kernel_config=ckc)
    ttnn.synchronize_device(device)
    _save(name, out)
    for t in (out, a, w):
        t.deallocate()
