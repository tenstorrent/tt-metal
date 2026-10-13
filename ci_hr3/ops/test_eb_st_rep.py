# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Stacking #58818 on #58816 (the block pack switch): binary_ng one-op replicas of the Blackhole models' sharded binary ops
(copied from round 3's test_eb_r11.py: Gemma-4 31B, Llama 3.1-8B, Qwen3.8-27B decode residual adds, gemma4_d_p prefill
residual adds, Qwen3-32B Galaxy adds) and SDXL-shaped block sections on 8x10 block shards; the last output of each is
hashed by eb_bits_plugin. Not for merge."""
import os as _os_guard, sys as _sys_guard

if not (_os_guard.environ.get("HWLOCK_HELD") or _os_guard.environ.get("GITHUB_ACTIONS")):
    _sys_guard.exit("not under hwlock")
import pytest
import torch
import ttnn


def _qb2_mc(device, cfg):
    if cfg.startswith("dp_"):
        # gemma4_d_p/tt/rms_norm.py _block_sharded_memory_config: chunk 8192 on 8x4, 256 rows of 5376 per chip, block
        # shards of 64x448 on 12x4 Galaxy cores (28 tiles per core); a P150 has 11 columns, so 8x4 cores of the same shard
        return ttnn.create_sharded_memory_config(
            shape=(64, 448),
            core_grid=ttnn.CoreGrid(x=8, y=4),
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    if cfg.startswith("gemma"):
        # gemma4_31b_qb2/tt/decoder.py _mem(5376, 28): width shards of 32x192 on a 7x4 grid
        return ttnn.create_sharded_memory_config(
            (32, 192),
            ttnn.CoreGrid(x=7, y=4),
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    if cfg == "llama_p150":
        # tt_transformers Llama 3.1-8B decode on a P150: width shards of 32x128 on 8x4 cores (4 tiles per core)
        return ttnn.create_sharded_memory_config(
            (32, 128), ttnn.CoreGrid(x=8, y=4), ttnn.ShardStrategy.WIDTH, ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True
        )
    if cfg == "qwen_qb2_40":
        # qwen38_27b_qb2/tt/decoder_tp.py residual_cores=40 (the TP decoder's override): 32x128 on a 10x4 grid
        crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 3))})
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, ttnn.ShardSpec(crs, [32, 128], ttnn.ShardOrientation.ROW_MAJOR)
        )
    if cfg.startswith("qwen32_glx"):
        # llama3_70b_galaxy qwen_model_config.py DECODE_RESIDUAL_MEMCFG: width shards of 32x128 on cores (1,0)-(2,4) (10 cores)
        crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(2, 4))})
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, ttnn.ShardSpec(crs, [32, 128], ttnn.ShardOrientation.ROW_MAJOR)
        )
    if cfg == "llama_qb2":
        # llama31_8b_qb2/tt/decoder.py _width_memcfg(1024, 8): 32x128 on the first 8 cores, row-wise
        grid = device.compute_with_storage_grid_size()
        crs = ttnn.num_cores_to_corerangeset(8, grid, True)
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(crs, [32, 128], ttnn.ShardOrientation.ROW_MAJOR),
        )
    # qwen38_27b_qb2/tt/decoder.py _residual_memory, residual_cores 80: 32x64 on a 10x8 grid
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 7))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, ttnn.ShardSpec(crs, [32, 64], ttnn.ShardOrientation.ROW_MAJOR)
    )


QB2_WIDTH = {"gemma_post": 5376, "gemma_add": 5376, "llama_qb2": 1024, "qwen_qb2": 5120, "dp_post": 3584, "dp_add": 3584, "llama_p150": 4096, "qwen_qb2_40": 5120}
QB2_ROWS = {"dp_post": 256, "dp_add": 256}


@pytest.mark.parametrize("cfg", list(QB2_WIDTH))
def test_qb2_add(device, cfg):
    # gemma_post: decoder.py:676-681 (MUL_UNARY_SFPU by the layer scalar after the add); gemma_add: decoder.py:668;
    # llama_qb2: llama31_8b_qb2 decoder.py:578-582, 597-601; qwen_qb2: qwen38_27b_qb2 decoder.py:548-550, 588-590;
    # dp_post / dp_add: gemma4_d_p tt/layer.py:152-157 (MUL_UNARY_SFPU after) and :139, prefill chunk 8192 on 8x4
    mc = _qb2_mc(device, cfg)
    torch.manual_seed(1)
    mk = lambda: ttnn.from_torch(
        torch.rand(1, 1, QB2_ROWS.get(cfg, 32), QB2_WIDTH[cfg]) - 0.5,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=mc,
    )
    a, b = mk(), mk()
    kw = dict(memory_config=mc, dtype=ttnn.bfloat16)
    if cfg in ("gemma_post", "dp_post"):
        kw["activations"] = [ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, 0.6875)]
    for _ in range(7):
        ttnn.deallocate(ttnn.add(a, b, **kw))
    o = ttnn.add(a, b, **kw)
    ttnn.to_torch(o)
    ttnn.deallocate(o)
    ttnn.deallocate(a)
    ttnn.deallocate(b)


QWEN32_GLX = {"qwen32_glx": ttnn.bfloat16, "qwen32_glx_pf": ttnn.bfloat8_b, "qwen32_glx_bb": ttnn.bfloat8_b}


@pytest.mark.parametrize("cfg", list(QWEN32_GLX))
def test_qwen32_glx_add(device, cfg):
    # llama3_70b_galaxy/tt/llama_decoder.py:181, 213, 232 (Qwen3-32B on a Blackhole Galaxy, one chip): bfp8_b ff_out or attn_out
    # plus the bf16 residual into bf16 (dtype=res_dtype), or into bfp8_b on the prefetcher path (QWEN_BH_PREFETCHER=1, :206)
    mc = _qb2_mc(device, cfg)
    torch.manual_seed(1)
    a = ttnn.from_torch(torch.rand(1, 1, 32, 1280) - 0.5, dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mc)
    # qwen32_glx_bb: the prefetcher path's :181 and :232 (CI's ci-token-matching run), bfp8_b + bfp8_b into bfp8_b
    b_dt = ttnn.bfloat8_b if cfg == "qwen32_glx_bb" else ttnn.bfloat16
    b = ttnn.from_torch(torch.rand(1, 1, 32, 1280) - 0.5, dtype=b_dt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mc)
    for _ in range(7):
        ttnn.deallocate(ttnn.add(a, b, memory_config=mc, dtype=QWEN32_GLX[cfg]))
    o = ttnn.add(a, b, memory_config=mc, dtype=QWEN32_GLX[cfg])
    ttnn.to_torch(o)
    ttnn.deallocate(o)
    ttnn.deallocate(a)
    ttnn.deallocate(b)


SDXL_SHAPES = {
    "resadd_16384x320": ((1, 1, 16384, 320), (8, 10), "add"),
    "resadd_4096x640": ((1, 1, 4096, 640), (8, 10), "add"),
    "tadd_1024x1280": ((1, 1, 1024, 1280), (8, 10), "add"),
    "geglu_4096x2560": ((1, 1, 4096, 2560), (8, 10), "mul"),
    "geglu_1024x5120": ((1, 1, 1024, 5120), (8, 10), "mul"),
    "geglu_1024x6144": ((1, 1, 1024, 6144), (8, 8), "mul"),
    "geglu_256x6144": ((1, 1, 256, 6144), (8, 8), "mul"),
}


@pytest.mark.parametrize("case", list(SDXL_SHAPES))
def test_sdxl_shape(device, case):
    shape, (gy, gx), op = SDXL_SHAPES[case]
    mc = ttnn.create_sharded_memory_config(shape, core_grid=ttnn.CoreGrid(y=gy, x=gx), strategy=ttnn.ShardStrategy.BLOCK)
    torch.manual_seed(1)
    a = ttnn.from_torch(torch.rand(shape) - 0.5, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mc)
    b = ttnn.from_torch(torch.rand(shape) - 0.5, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mc)
    f = (lambda: ttnn.multiply(a, b, fast_and_approximate_mode=True, memory_config=mc)) if op == "mul" else (lambda: ttnn.add(a, b, memory_config=mc))
    for _ in range(3):
        ttnn.deallocate(f())
    o = f()
    ttnn.to_torch(o)
    ttnn.deallocate(o)
    ttnn.deallocate(a)
    ttnn.deallocate(b)
