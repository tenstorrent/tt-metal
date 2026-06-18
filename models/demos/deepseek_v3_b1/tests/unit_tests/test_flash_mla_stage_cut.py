# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
#43563 / tt-blaze#842 INTERMEDIATE-stage harness for the flash-decode SDPA kernel.

This severs `compute_sdpa_chunk` (in unified_kernels/flash_mla.hpp) at a configurable STAGE
via the kernel macro `SDPA_STAGE_CUT_43563` and reads back, PER OUTPUT CORE, the chosen
intermediate so it can be compared against a torch golden in the 2-chunks/core, no-mask regime
where the multichunk bug lives.

How the cut works (see flash_mla.hpp), all reading DEST tiles the SDPA kernel itself wrote so
the packer addressing matches (env SDPA_STAGE selects the matching torch golden):
  - SDPA_STAGE_CUT_43563 == 2 -> Stage B: running row-max over the core's chunks (col 0).
  - SDPA_STAGE_CUT_43563 == 3 -> Stage C: exp((score-max)*scale) of the LAST chunk (mm1 post-exp).
  - SDPA_STAGE_CUT_43563 == 4 -> Stage D: P@V partial accumulator over the core's chunks.
  - SDPA_STAGE_CUT_43563 == 5 -> Stage E: running row-SUM over the core's chunks (col 1). [suspect]
  - SDPA_STAGE_CUT_43563 == 1 -> Stage A placeholder: the raw UNSCALED QK^T is overwritten in
    place by exp inside the same chunk, so it is NOT observable post-loop with the kernel's native
    DEST addressing. Slot 1 falls through to the post-exp mm1 (== Stage C content); a true raw-score
    capture would need in-chunk SDPA-addressed surgery and is intentionally left unimplemented.
  - On output cores the cross-core tree reduction is bypassed, so each output core's OWN per-chunk
    intermediate lands in cb_out_final unmerged and host-readable.

Regime selection (pos=2047, k_chunk_size=128):
  valid_seq_len=2048 -> 16 k-chunks; num_cores_per_head=8 -> each core gets exactly 2 chunks.
  (2047+1) % 128 == 0  -> NO mask. This is the bug regime.
  Output core (core_num_in_reduce==0) processes chunks {0, 8}:
    chunk 0 -> keys [0:128], chunk 8 -> keys [1024:1152].

Output tensor is [1, 1, 64, 512] height-sharded onto the 8 S1 output cores; output core b
(== batch b) holds heads [b*8 : b*8+8]. So output_torch[0, 0, b*8:(b+1)*8, :] is core b's
intermediate.

YOU MUST set the kernel macro before running. To run Stage A:
  edit flash_mla.hpp:  #define SDPA_STAGE_CUT_43563 1   (and NOP_DRAIN_43563 0 for bug active)
  then:
  TT_METAL_HOME=... PYTHONPATH=... ARCH_NAME=blackhole TT_METAL_SLOW_DISPATCH_MODE=1 \
    ./python_env/bin/pytest <this file>::test_flash_mla_stage_cut -p no:cacheprovider -s
The test asserts nothing fatal; it logs per-core PCC + exactness so the cut stage is reported.
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, is_blackhole, is_watcher_enabled
from models.demos.deepseek_v3_b1.micro_ops.flash_mla.op import FlashMLADecode

# Which stage the kernel is currently cut at. This MUST match the SDPA_STAGE_CUT_43563 value
# compiled into flash_mla.hpp (1=A, 2=B). Set env SDPA_STAGE=A|B to match what you compiled.
STAGE_DEFAULT = os.environ.get("SDPA_STAGE", "A")


def _bf8_roundtrip(t: torch.Tensor) -> torch.Tensor:
    """Quantize a torch tensor through bfloat8_b and back, to mimic the KV-cache dtype the
    kernel actually reads. Done by a ttnn round-trip on host."""
    tt = ttnn.from_torch(t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    return ttnn.to_torch(tt).to(torch.float32)


def pytest_addoption_stage():
    pass


@pytest.mark.parametrize("stage", [STAGE_DEFAULT])
@pytest.mark.parametrize("decode_position", [2047])  # -> 16 chunks / 8 cores = 2 chunks/core, no mask
@pytest.mark.parametrize("k_chunk_size", [128])
@pytest.mark.parametrize("max_seq_len", [32 * 1024])
@pytest.mark.parametrize("batch_size", [1])
def test_flash_mla_stage_cut(device, stage, decode_position, k_chunk_size, max_seq_len, batch_size):
    if is_blackhole() and is_watcher_enabled():
        pytest.skip("Skipping on Blackhole with watcher enabled, see issue #37631")

    torch.manual_seed(0)

    num_heads = 64
    num_q_heads_per_core = 8
    kv_lora_rank = 512
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim  # 192
    kvpe_dim = kv_lora_rank + qk_rope_head_dim  # 576
    scale = qk_head_dim**-0.5

    num_cores_per_head = 8  # 8 S blocks
    k_chunk_t = k_chunk_size  # keys per chunk

    logger.info(
        f"[STAGE CUT {stage}] pos={decode_position} k_chunk_size={k_chunk_size} -> "
        f"{(decode_position + 1 + k_chunk_size - 1) // k_chunk_size} chunks, "
        f"{num_cores_per_head} cores/head"
    )

    tiny_tile = ttnn.Tile((num_q_heads_per_core, 32))

    s1_cores, _ = FlashMLADecode.ProgramConfig.grid.BLOCKS[0]
    q_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in s1_cores])
    q_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(q_core_grid, (num_q_heads_per_core, kvpe_dim), ttnn.ShardOrientation.ROW_MAJOR),
    )
    out_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(q_core_grid, (num_q_heads_per_core, kv_lora_rank), ttnn.ShardOrientation.ROW_MAJOR),
    )

    q_shape = (1, batch_size, num_heads, kvpe_dim)
    torch_q = torch.randn(q_shape, dtype=torch.bfloat16)
    tt_q = ttnn.from_torch(
        torch_q,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=q_mem_config,
        tile=tiny_tile,
    )

    program_config = FlashMLADecode.ProgramConfig(k_chunk_size=k_chunk_size, exp_approx_mode=False)

    cache_shape = (batch_size, 1, max_seq_len, kvpe_dim)
    torch_cache = torch.randn(cache_shape, dtype=torch.bfloat16)

    grid = program_config.grid
    kv_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=[1, 1, program_config.k_chunk_size, kvpe_dim],
        grid=grid.optimal_dram_grid(),
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    kv_mem_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=kv_nd_shard_spec)
    tt_cache = ttnn.from_torch(
        torch_cache,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=kv_mem_config,
    )

    grid_size = device.compute_with_storage_grid_size()
    position_ids = torch.ones(batch_size, dtype=torch.int32) * decode_position
    position_replicated = position_ids.repeat(grid_size.x * grid_size.y, 1)
    pos_core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1))]
    )
    pos_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(pos_core_grid, (1, 1), ttnn.ShardOrientation.ROW_MAJOR),
    )
    tt_position_ids = ttnn.from_torch(
        position_replicated,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=pos_mem_config,
    )

    out_shape = (1, batch_size, num_heads, kv_lora_rank)
    tt_out = ttnn.from_torch(
        torch.zeros(out_shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=out_mem_config,
        tile=tiny_tile,
    )

    compute_kernel_config = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    # ---------------------------------------------------------------------------------------------
    # Run the (severed) kernel twice to also check iteration parity of the intermediate.
    # ---------------------------------------------------------------------------------------------
    outs = []
    for it in range(2):
        attn_out = FlashMLADecode.op(
            q_tensor=tt_q,
            kv_cache_tensor=tt_cache,
            head_dim_v=kv_lora_rank,
            cur_pos_tensor=tt_position_ids,
            output_tensor=tt_out,
            scale=scale,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )
        outs.append(ttnn.to_torch(attn_out).to(torch.float32).clone())
    kernel_out = outs[0]  # [1, 1, 64, 512]
    iter_max_diff = (outs[0] - outs[1]).abs().max().item()

    # ---------------------------------------------------------------------------------------------
    # Build the per-core golden. Output core b (== batch b) processes chunks {0, 8}.
    # ---------------------------------------------------------------------------------------------
    # KV the kernel actually reads is bfloat8_b. Quantize the cache accordingly.
    cache_bf8 = _bf8_roundtrip(torch_cache.to(torch.float32))  # [1,1,max_seq,576]
    q_f = torch_q.to(torch.float32)[0, 0]  # [64, 576]

    # chunk index -> key range
    def chunk_keys(chunk_idx):
        s = chunk_idx * k_chunk_t
        return cache_bf8[0, 0, s : s + k_chunk_t, :]  # [128, 576]

    core0_chunks = [0, num_cores_per_head]  # {0, 8}
    logger.info(
        f"Output core chunks = {core0_chunks} -> key ranges "
        f"{[(c*k_chunk_t, c*k_chunk_t+k_chunk_t) for c in core0_chunks]}"
    )

    num_output_cores = num_heads // num_q_heads_per_core  # 8
    per_core_pcc = []
    per_core_exact = []
    for b in range(num_output_cores):
        q_b = q_f[b * num_q_heads_per_core : (b + 1) * num_q_heads_per_core]  # [8, 576]

        # The kernel keeps mm1 (QK^T) UNSCALED in DEST; scale is folded into the later exp/max/sum.
        scores = {c: (q_b @ chunk_keys(c).transpose(0, 1)) for c in core0_chunks}  # unscaled, [8,128]
        stacked = torch.cat([scores[c] for c in core0_chunks], dim=1)  # [8, 256] unscaled
        # Running max over the core's chunks (UNSCALED, as stored in DEST).
        run_max = stacked.max(dim=1, keepdim=True).values  # [8, 1]
        last = core0_chunks[-1]

        heads = slice(b * num_q_heads_per_core, (b + 1) * num_q_heads_per_core)
        if stage == "B":
            # reduce_max -> col 0 of an 8x32 tile; other cols are not the max. Compare col 0.
            golden = run_max
            kern = kernel_out[0, 0, heads, :1]
        elif stage in ("A", "C"):
            # Stage C (and the A placeholder): exp((score - run_max) * scale) of the LAST chunk,
            # i.e. the post-exp probabilities of mm1 before P@V consumes them. [8, 128]
            golden = torch.exp((scores[last] - run_max) * scale)
            kern = kernel_out[0, 0, heads, :128]
        elif stage == "D":
            # Stage D: P@V partial = sum_c exp((score_c - run_max)*scale) @ V_c, over the core's
            # chunks (UNnormalized, no /sum). V = first kv_lora_rank cols of the chunk's keys. [8,512]
            acc = torch.zeros(num_q_heads_per_core, kv_lora_rank, dtype=torch.float32)
            for c in core0_chunks:
                p = torch.exp((scores[c] - run_max) * scale)  # [8, 128]
                v = chunk_keys(c)[:, :kv_lora_rank]  # [128, 512]
                acc = acc + p @ v
            golden = acc
            kern = kernel_out[0, 0, heads, :kv_lora_rank]
        elif stage == "E":
            # Stage E: running row-SUM = sum over the core's chunks of exp((score - run_max)*scale).
            # Stored in col 1 of the SAME 8x32 tile as max -> compare col 1. *** bug suspect ***
            golden = torch.exp((stacked - run_max) * scale).sum(dim=1, keepdim=True)  # [8, 1]
            kern = kernel_out[0, 0, heads, 1:2]
        else:
            raise ValueError(f"Unknown/unwired stage {stage!r}")

        passing, msg = comp_pcc(golden, kern, 0.99)
        exact = torch.equal(golden.to(torch.bfloat16).to(torch.float32), kern)
        max_abs = (golden - kern).abs().max().item()
        per_core_pcc.append(msg)
        per_core_exact.append(exact)
        logger.info(
            f"  core b={b}: PCC={msg} | max|Δ|={max_abs:.6e} | exact(bf16)={exact} | "
            f"golden[0,:4]={golden.flatten()[:4].tolist()} kern[0,:4]={kern.flatten()[:4].tolist()}"
        )

    logger.info(f"[STAGE CUT {stage}] iter0-vs-iter1 kernel max|Δ| = {iter_max_diff:.6e}")
    logger.info(f"[STAGE CUT {stage}] per-core PCC: {per_core_pcc}")
    logger.info(f"[STAGE CUT {stage}] per-core exact(bf16): {per_core_exact}")


# =================================================================================================
# #43563 NON-INVASIVE SUM TAP harness.
#
# Unlike the stage-cut test above, this leaves the FULL pipeline control flow UNCHANGED and runs the
# real buggy path (no forced is_final, no last_chunk=false, no disabled tree reduction). It only adds
# a side buffer (sum_tap_tensor, sharded identically to the output) into which each OUTPUT core packs
# its running max/sum DEST tile AFTER its per-core chunk loop but BEFORE the cross-core tree reduction
# consumes it (see flash_mla.hpp::SDPA_SUM_TAP_43563). We read that side buffer back and compare each
# output core's pre-merge running sum (col 1 of the max/sum tile) to the torch golden per-core running
# sum, AND we report the final tree-reduced output PCC vs golden in the SAME run.
#
# To run you MUST first set, in unified_kernels/flash_mla.hpp:
#     #define SDPA_SUM_TAP_43563 1     (and SDPA_STAGE_CUT_43563 0)
# and choose NOP_DRAIN_43563 0 (bug active) or 1 (stopgap on).
# =================================================================================================
@pytest.mark.parametrize("decode_position", [2047])  # -> 16 chunks / 8 cores = 2 chunks/core, no mask
@pytest.mark.parametrize("k_chunk_size", [128])
@pytest.mark.parametrize("max_seq_len", [32 * 1024])
@pytest.mark.parametrize("batch_size", [1])
def test_flash_mla_sum_tap(device, decode_position, k_chunk_size, max_seq_len, batch_size):
    if is_blackhole() and is_watcher_enabled():
        pytest.skip("Skipping on Blackhole with watcher enabled, see issue #37631")

    torch.manual_seed(0)

    num_heads = 64
    num_q_heads_per_core = 8
    kv_lora_rank = 512
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim  # 192
    kvpe_dim = kv_lora_rank + qk_rope_head_dim  # 576
    scale = qk_head_dim**-0.5

    num_cores_per_head = 8
    k_chunk_t = k_chunk_size

    tiny_tile = ttnn.Tile((num_q_heads_per_core, 32))

    s1_cores, _ = FlashMLADecode.ProgramConfig.grid.BLOCKS[0]
    q_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in s1_cores])
    q_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(q_core_grid, (num_q_heads_per_core, kvpe_dim), ttnn.ShardOrientation.ROW_MAJOR),
    )
    out_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(q_core_grid, (num_q_heads_per_core, kv_lora_rank), ttnn.ShardOrientation.ROW_MAJOR),
    )

    q_shape = (1, batch_size, num_heads, kvpe_dim)
    torch_q = torch.randn(q_shape, dtype=torch.bfloat16)
    tt_q = ttnn.from_torch(
        torch_q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=q_mem_config, tile=tiny_tile
    )

    program_config = FlashMLADecode.ProgramConfig(k_chunk_size=k_chunk_size, exp_approx_mode=False)

    cache_shape = (batch_size, 1, max_seq_len, kvpe_dim)
    torch_cache = torch.randn(cache_shape, dtype=torch.bfloat16)

    grid = program_config.grid
    kv_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=[1, 1, program_config.k_chunk_size, kvpe_dim],
        grid=grid.optimal_dram_grid(),
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    kv_mem_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=kv_nd_shard_spec)
    tt_cache = ttnn.from_torch(
        torch_cache, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=kv_mem_config
    )

    grid_size = device.compute_with_storage_grid_size()
    position_ids = torch.ones(batch_size, dtype=torch.int32) * decode_position
    position_replicated = position_ids.repeat(grid_size.x * grid_size.y, 1)
    pos_core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1))]
    )
    pos_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(pos_core_grid, (1, 1), ttnn.ShardOrientation.ROW_MAJOR),
    )
    tt_position_ids = ttnn.from_torch(
        position_replicated, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=pos_mem_config
    )

    out_shape = (1, batch_size, num_heads, kv_lora_rank)
    tt_out = ttnn.from_torch(
        torch.zeros(out_shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=out_mem_config,
        tile=tiny_tile,
    )
    # Side tap buffer: sharded identically to the output (8 output cores, [8, kv_lora_rank]).
    tt_sum_tap = ttnn.from_torch(
        torch.zeros(out_shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=out_mem_config,
        tile=tiny_tile,
    )

    compute_kernel_config = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    # Run twice to ALSO measure iteration-to-iteration nondeterminism (a known signature of #43563).
    sum_tap_iters = []
    out_iters = []
    for _it in range(2):
        attn_out = FlashMLADecode.op(
            q_tensor=tt_q,
            kv_cache_tensor=tt_cache,
            head_dim_v=kv_lora_rank,
            cur_pos_tensor=tt_position_ids,
            output_tensor=tt_out,
            scale=scale,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            sum_tap_tensor=tt_sum_tap,
        )
        out_iters.append(ttnn.to_torch(attn_out).to(torch.float32).clone())
        sum_tap_iters.append(ttnn.to_torch(tt_sum_tap).to(torch.float32).clone())
    kernel_out = out_iters[0]  # [1, 1, 64, 512] final tree-reduced
    sum_tap = sum_tap_iters[0]  # [1, 1, 64, 512]; col1 of each head shard = running sum
    sumtap_iter_diff = (sum_tap_iters[0][..., 1:2] - sum_tap_iters[1][..., 1:2]).abs().max().item()
    out_iter_diff = (out_iters[0] - out_iters[1]).abs().max().item()
    logger.info(
        f"[SUM TAP] iter0-vs-iter1: running-sum max|Δ|={sumtap_iter_diff:.6e} | "
        f"final-output max|Δ|={out_iter_diff:.6e}"
    )

    # KV the kernel actually reads is bfloat8_b. Quantize the cache accordingly.
    cache_bf8 = _bf8_roundtrip(torch_cache.to(torch.float32))  # [1,1,max_seq,576]
    q_f = torch_q.to(torch.float32)[0, 0]  # [64, 576]

    def chunk_keys(chunk_idx):
        s = chunk_idx * k_chunk_t
        return cache_bf8[0, 0, s : s + k_chunk_t, :]  # [128, 576]

    core0_chunks = [0, num_cores_per_head]  # {0, 8} -> this output core's assigned chunks

    num_output_cores = num_heads // num_q_heads_per_core  # 8

    # ---- Per-core running-sum tap vs golden (full pipeline, pre-merge) -------------------------
    sum_pcc = []
    sum_max_abs = []
    for b in range(num_output_cores):
        q_b = q_f[b * num_q_heads_per_core : (b + 1) * num_q_heads_per_core]  # [8, 576]
        scores = {c: (q_b @ chunk_keys(c).transpose(0, 1)) for c in core0_chunks}  # unscaled, [8,128]
        stacked = torch.cat([scores[c] for c in core0_chunks], dim=1)  # [8, 256] unscaled
        run_max = stacked.max(dim=1, keepdim=True).values  # [8, 1]
        # running sum = sum over the core's chunks of exp((score - run_max) * scale)
        golden_sum = torch.exp((stacked - run_max) * scale).sum(dim=1, keepdim=True)  # [8, 1]
        heads = slice(b * num_q_heads_per_core, (b + 1) * num_q_heads_per_core)
        kern_sum = sum_tap[0, 0, heads, 1:2]  # col 1 of the max/sum tile
        passing, msg = comp_pcc(golden_sum, kern_sum, 0.99)
        max_abs = (golden_sum - kern_sum).abs().max().item()
        max_rel = ((golden_sum - kern_sum).abs() / golden_sum.abs().clamp_min(1e-6)).max().item()
        sum_pcc.append(float(msg))
        sum_max_abs.append(max_abs)
        logger.info(
            f"  [SUM TAP] core b={b}: PCC={msg} | max|Δ|={max_abs:.6e} | max relΔ={max_rel:.4e} | "
            f"golden={golden_sum.flatten()[:4].tolist()} kern={kern_sum.flatten()[:4].tolist()}"
        )

    # ---- Final tree-reduced output vs golden (full flash attention) ----------------------------
    out_pcc = []
    for b in range(num_output_cores):
        q_b = q_f[b * num_q_heads_per_core : (b + 1) * num_q_heads_per_core]  # [8, 576]
        # Full attention over ALL 16 chunks (all keys [0:2048]).
        all_keys = cache_bf8[0, 0, 0 : 16 * k_chunk_t, :]  # [2048, 576]
        s_full = (q_b @ all_keys.transpose(0, 1)) * scale  # [8, 2048]
        p = torch.softmax(s_full, dim=1)  # [8, 2048]
        v = all_keys[:, :kv_lora_rank]  # [2048, 512]
        golden_out = p @ v  # [8, 512]
        heads = slice(b * num_q_heads_per_core, (b + 1) * num_q_heads_per_core)
        kern_o = kernel_out[0, 0, heads, :]  # [8, 512]
        passing, msg = comp_pcc(golden_out, kern_o, 0.99)
        max_rel = ((golden_out - kern_o).abs() / golden_out.abs().clamp_min(1e-6)).max().item()
        out_pcc.append(float(msg))
        logger.info(f"  [FINAL OUT] core b={b}: PCC={msg} | max relΔ={max_rel:.4e}")

    logger.info(f"[SUM TAP] per-core running-sum PCC vs golden: {sum_pcc}")
    logger.info(f"[SUM TAP] per-core running-sum max|Δ|: {sum_max_abs}")
    logger.info(f"[FINAL OUT] per-core final-output PCC vs golden: {out_pcc}")
