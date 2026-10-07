# Round 3 matmul, second review of #58716 (2026-10-07): Blackhole production callers that set dst_full_sync_en, each run
# with full sync (as the model does) and with half sync, everything else identical, under the device profiler:
# - Llama 3.1 8B on QuietBox 2, decode SDPA per device (models/demos/llama31_8b_qb2/tt/decoder.py:242-256, 306-310, 700-709):
#   paged_scaled_dot_product_attention_decode, 8 q heads, 2 kv heads, head dim 128, bfp8 paged cache of 128-token pages,
#   q chunk 32, k chunk 256, HiFi4, fp32 DEST; batch 1 (grid 8x4) and 32 (the device grid), position 1279 and 4095.
# - Llama 3.3 70B on Galaxy, prefill per device at 2048 tokens (models/demos/llama3_70b_galaxy/tt): FF1/FF3 with
#   w1_w3_prg_config (grid 7x7, in0_block_w 4, 1x8 sub blocks, LoFi bf16 DEST, bfp4 weights), and the QKV matmul (auto
#   config, HiFi2 fp32 DEST, bfp8 weights).
# - DiffusionGemma's expert matmuls (concat_moe.py:100-107, the opt-in full sync config): gate/up and down per device.
# SYNC_OUT saves raw outputs for a bit comparison of the two modes.
import os as _os
import sys as _sys

if not (_os.environ.get("HWLOCK_HELD") or _os.path.isfile(_os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH", ""))):
    _sys.exit("not under hwlock")
import os

import pytest
import torch
import ttnn

NH, NKV, D, BLOCK, PT_WIDTH = 8, 2, 128, 128, 1024
_SDPA = {}


@pytest.fixture(scope="module")
def device():
    dev = ttnn.CreateDevice(device_id=0, l1_small_size=32768)
    ttnn.SetDefaultDevice(dev)
    yield dev
    ttnn.close_device(dev)


def _save(name, t):
    if os.environ.get("SYNC_OUT"):
        torch.save(t.contiguous().view(torch.int16 if t.dtype == torch.bfloat16 else torch.int32), os.path.join(os.environ["SYNC_OUT"], f"{name}.pt"))


def _sdpa_inputs(device, batch, cur_pos):
    key = (batch, cur_pos)
    if key not in _SDPA:
        torch.manual_seed(0)
        grid = device.compute_with_storage_grid_size()
        used = -(-(cur_pos + 1) // BLOCK)
        used += used % 2
        pages = 1 + batch * used
        table = torch.zeros(batch, PT_WIDTH, dtype=torch.int32)
        table[:, :used] = torch.arange(1, pages, dtype=torch.int32).reshape(batch, used)
        q = torch.randn(1, batch, NH, D)
        k, v = torch.randn(pages, NKV, BLOCK, D), torch.randn(pages, NKV, BLOCK, D)
        q_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(ttnn.num_cores_to_corerangeset(batch, grid, True), [32, D], ttnn.ShardOrientation.ROW_MAJOR),
        )
        tq = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=q_mem)
        tk, tv = (
            ttnn.from_torch(t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for t in (k, v)
        )
        rm = dict(dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tt_table = ttnn.from_torch(table, **rm)
        tt_pos = ttnn.from_torch(torch.full((batch,), cur_pos, dtype=torch.int32), **rm)
        _SDPA.clear()
        _SDPA[key] = (tq, tk, tv, tt_table, tt_pos, ttnn.CoreCoord(8, 4) if batch <= 8 else grid)
    return _SDPA[key]


SDPA_CASES = [(b, p, s) for b, p in ((1, 1279), (1, 4095), (32, 1279), (32, 4095)) for s in ("full", "half")]


@pytest.mark.parametrize("case", SDPA_CASES, ids=[f"sdpa_dec_b{b}_p{p}_{s}" for b, p, s in SDPA_CASES])
def test_sync_sdpa_decode(device, case):
    batch, cur_pos, sync = case
    tq, tk, tv, tt_table, tt_pos, grid = _sdpa_inputs(device, batch, cur_pos)
    pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid, q_chunk_size=32, k_chunk_size=256, exp_approx_mode=False)
    cfg = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True,
        packer_l1_acc=False, dst_full_sync_en=(sync == "full"),
    )
    out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        tq, tk, tv, cur_pos_tensor=tt_pos, page_table_tensor=tt_table, program_config=pc, compute_kernel_config=cfg,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn.synchronize_device(device)
    _save(f"sdpa_dec_b{batch}_p{cur_pos}_{sync}", ttnn.to_torch(out))
    out.deallocate()


def _ff13_pc():
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(7, 7), in0_block_w=4, out_subblock_h=1, out_subblock_w=8, out_block_h=10,
        out_block_w=16, per_core_M=10, per_core_N=16, transpose_mcast=False, fused_activation=None, fuse_batch=False,
    )


# (name, in0 shape, in1 shape, in0 dtype, in1 dtype, out dtype, fidelity, approx, fp32 DEST, program config)
MM_CASES = [
    ("g70b_ff13_2k", (1, 2, 1024, 2048), (1, 1, 2048, 3584), ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.bfloat8_b, "LoFi", False, False, _ff13_pc),
    ("g70b_qkv_2k", (1, 1, 2048, 2048), (1, 1, 2048, 1280), ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat16, "HiFi2", True, True, None),
    ("dg_gate_up", (1, 1, 256, 2816), (1, 1, 2816, 24576), ttnn.bfloat16, ttnn.bfloat16, ttnn.bfloat16, "HiFi4", False, True, None),
    ("dg_down", (1, 1, 256, 24576), (1, 1, 24576, 2816), ttnn.bfloat16, ttnn.bfloat16, ttnn.bfloat16, "HiFi4", False, True, None),
]
FID = {"LoFi": ttnn.MathFidelity.LoFi, "HiFi2": ttnn.MathFidelity.HiFi2, "HiFi4": ttnn.MathFidelity.HiFi4}
_MM = {}
MM_IDS = [(c, s) for c in MM_CASES for s in ("full", "half")]


@pytest.mark.parametrize("case", MM_IDS, ids=[f"{c[0]}_{s}" for c, s in MM_IDS])
def test_sync_mm(device, case):
    (name, s0, s1, d0, d1, do, fid, approx, fp32, pc), sync = case
    if name not in _MM:
        _MM.clear()
        torch.manual_seed(0)
        _MM[name] = tuple(
            ttnn.from_torch(torch.randn(s) * sc, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for s, dt, sc in ((s0, d0, 0.1), (s1, d1, 0.02))
        )
    a, b = _MM[name]
    cfg = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=FID[fid], math_approx_mode=approx, fp32_dest_acc_en=fp32,
        packer_l1_acc=not name.startswith("dg_"), dst_full_sync_en=(sync == "full"),
    )
    kw = {"program_config": pc()} if pc else {}
    out = ttnn.matmul(a, b, compute_kernel_config=cfg, dtype=do, memory_config=ttnn.DRAM_MEMORY_CONFIG, **kw)
    ttnn.synchronize_device(device)
    _save(f"{name}_{sync}", ttnn.to_torch(out))
    out.deallocate()
