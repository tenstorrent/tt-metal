# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone unit test for the C++ op ``ttnn.experimental.deepseek.moe.generalized_moe_gate``.

Exercises the device op directly (via ``GeneralizedMoeGateOp.op``) against the PyTorch
``golden`` reference, so the op can be validated in isolation without running the full
``MoEGate`` module. Modeled on
``models/demos/deepseek_v3_b1/tests/unit_tests/test_deepseek_moe_gate.py``.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3.tt.generalized_moe_gate.op import GeneralizedMoeGateOp


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("output_softmax", [False, True])
@pytest.mark.parametrize("topk", [8, 6, 4])
@pytest.mark.parametrize("enable_sigmoid", [True, False])
@pytest.mark.parametrize("seed", [42, 201, 512])
def test_generalized_moe_gate(device, batch_size, enable_sigmoid, seed, topk, output_softmax):
    """Test the generalized MoE gate C++ op on a 32x32 tile against the golden reference (top-`topk`,
    linear-normalize or softmax-over-selected)."""

    # Tensor dimensions — full 32x32 tile, logical 32x32 per shard.
    input_shape = (batch_size, 8, 32)
    reshaped_input_shape = (batch_size, 16, 16)
    input_shard_shape = (32, 32)
    input_tile = ttnn.Tile(input_shard_shape)
    output_shape = (batch_size, 1, 16)
    output_shard_shape = (32, 32)
    output_tile = ttnn.Tile(output_shard_shape)

    logger.info(f"Testing generalized MoE gate with input shape {input_shape}")

    # Create input PyTorch tensor with random values.
    torch.manual_seed(seed)
    torch_input = (2 * torch.rand(input_shape, dtype=torch.bfloat16)) - 1
    if not enable_sigmoid:
        torch_input = torch.sigmoid(torch_input)
    torch_bias = (2 * torch.rand(input_shape, dtype=torch.bfloat16)) - 1
    eps = 1e-20
    scaling_factor = 2.5

    # Reference output. (Only the golden indices are used — scores are validated tie-robustly below
    # against the device's OWN selection, not the golden's scores, so the golden scores are unused here.)
    _, top8_indices = GeneralizedMoeGateOp.golden(
        torch_input, torch_bias, eps, scaling_factor, enable_sigmoid, topk, output_softmax
    )

    grid = device.compute_with_storage_grid_size()
    core_grid = ttnn.num_cores_to_corerangeset(
        batch_size,
        ttnn.CoreCoord(grid.x, grid.y),
        row_wise=True,
    )
    input_shard_spec = ttnn.ShardSpec(
        core_grid,
        input_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec)

    output_shard_spec = ttnn.ShardSpec(
        core_grid,
        output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    # Input values — sharded on a single core per batch.
    reshaped_input = torch.reshape(torch_input, reshaped_input_shape)
    ttnn_input = ttnn.from_torch(
        reshaped_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
        tile=input_tile,
    )

    # Bias is transposed before upload (the kernel expects the transposed layout).
    reshaped_bias = torch.transpose(torch.reshape(torch_bias, reshaped_input_shape), -2, -1)
    ttnn_bias = ttnn.from_torch(
        reshaped_bias,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
        tile=input_tile,
    )

    # Transposed routing indices: 0..255 laid out as (16,16) then transposed.
    torch_input_indices = torch.arange(reshaped_input_shape[1] * reshaped_input_shape[2], dtype=torch.int32)
    torch_input_indices = torch_input_indices.unsqueeze(0).expand(reshaped_input_shape[0], -1)
    torch_input_indices = torch_input_indices.reshape(reshaped_input_shape)
    torch_input_indices = torch.transpose(torch_input_indices, -2, -1).to(torch.uint16)
    ttnn_input_indices = ttnn.from_torch(
        torch_input_indices,
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
        tile=input_tile,
    )

    # Preallocated output buffers (filled in place by the op).
    ttnn_output = ttnn.from_torch(
        torch.zeros(output_shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_config,
        tile=output_tile,
    )
    ttnn_output_indices = ttnn.from_torch(
        torch.zeros(output_shape, dtype=torch.uint16),
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_config,
        tile=output_tile,
    )

    logger.info("Running generalized MoE gate operation...")
    ttnn_result, ttnn_result_indices = GeneralizedMoeGateOp.op(
        ttnn_input,
        ttnn_bias,
        ttnn_output,
        ttnn_input_indices,
        ttnn_output_indices,
        eps,
        scaling_factor,
        enable_sigmoid,
        topk,
        output_softmax,
    )

    # Convert back to torch and keep the top-`topk` slots (ranks 0..topk-1 sit in the first topk cols;
    # the dropped ranks topk..7 are zeroed by the kernel).
    output_torch = ttnn.to_torch(ttnn_result)[:, 0, :topk]
    output_indices_torch = ttnn.to_torch(ttnn_result_indices)[:, 0, :topk]

    # The op does not guarantee a stable order across ties, so sort both by index
    # before comparing (same approach as the reference unit test).
    sorted_output_indices_torch, i = torch.sort(output_indices_torch, dim=-1)
    sorted_output_torch = torch.gather(output_torch, dim=-1, index=i)

    top8_indices = torch.sort(top8_indices, dim=-1).values

    # bf16 produces many equal bias-corrected values, so the exact top-8 *indices* are ambiguous at
    # the rank-8 cutoff (genuine ties — e.g. two experts with identical bf16 bias fight for the last
    # slot, and torch.topk vs the device break it differently). A strict index match is the wrong
    # check. Validate tie-robustly:
    #   (1) the device's selected experts form a VALID top-8 by the bias-corrected ranking key
    #       (same sorted key multiset as the golden), and
    #   (2) the normalized scores are self-consistent with the device's own selection.
    ranking = torch.sigmoid(torch_input) if enable_sigmoid else torch_input
    bias_key = (ranking + torch_bias).reshape(batch_size, -1).float()
    raw_scores = ranking.reshape(batch_size, -1).float()
    dev_idx = sorted_output_indices_torch.long()
    gold_idx = top8_indices.long()

    logger.info(f"dev_idx=\n{dev_idx}\ngold_idx=\n{gold_idx}")
    assert dev_idx.min() >= 0 and dev_idx.max() < 256, f"device produced out-of-range expert id:\n{dev_idx}"

    dev_key = torch.gather(bias_key, dim=-1, index=dev_idx).sort(dim=-1).values
    gold_key = torch.gather(bias_key, dim=-1, index=gold_idx).sort(dim=-1).values
    assert torch.allclose(dev_key, gold_key, atol=1e-2), (
        f"Device selection is not a valid top-8 by bias key.\n dev_idx={dev_idx}\n gold_idx={gold_idx}"
        f"\n dev_key={dev_key}\n gold_key={gold_key}"
    )

    dev_sel = torch.gather(raw_scores, dim=-1, index=dev_idx)
    # Consistency check vs the device's OWN selection: softmax-over-selected when output_softmax, else linear.
    weights = torch.exp(dev_sel) if output_softmax else dev_sel
    expected_norm = weights / (weights.sum(dim=-1, keepdim=True) + eps) * scaling_factor
    assert torch.allclose(
        sorted_output_torch.float(), expected_norm, atol=1e-2, rtol=1e-4
    ), "Normalized scores are not consistent with the device's own top-8 selection"


@pytest.mark.parametrize("batch_size", [1])
def test_dump_sum_top2_layout(device, batch_size):
    """DEBUG PROBE: dump the per-group top-8 DEST layout right after sum_top2.

    Requires the kernel built with ``GMG_DUMP_AFTER_SUM_TOP2`` enabled (see
    generalized_moe_gate_kernel.cpp). With that macro on, the op stops after sum_top2 and
    returns the *bias region* (ranking keys) in the score output and the *indices region*
    (expert ids) in the index output, both as a full 16x16 face.

    Inputs are rigged so the layout is readable:
      - input = 0.5 everywhere (scores irrelevant to layout)
      - bias[g, j] = j  -> each group's top-8 are its experts j in {24..31}
      - indices = global arange(256) (so each cell's value = global expert id; group = id // 32)

    Run it, then read the printed grids: in the INDEX grid, find where the ids of each
    group (id // 32 == g) land -> that's group g's top-8 column slot / lane layout.
    """
    n_group, group_size = 8, 32  # 256 experts
    reshaped = (batch_size, 16, 16)
    shard_shape = (32, 32)
    tile = ttnn.Tile(shard_shape)

    torch_input = torch.full((batch_size, n_group, group_size), 0.5, dtype=torch.bfloat16)
    # bias[g, j] = j + g  -> within every group the top-8 is still j in {24..31}, but the
    # per-group top-2 sum (= 61 + 2g) is distinct, so sort_top4 deterministically keeps
    # groups {4,5,6,7}. That makes the AFTER_STEP1 dump readable (you can tell which group
    # is which from the expert ids), while AFTER_SUM_TOP2 still shows all 8 groups.
    j = torch.arange(group_size, dtype=torch.bfloat16).view(1, 1, group_size)
    g = torch.arange(n_group, dtype=torch.bfloat16).view(1, n_group, 1)
    torch_bias = (j + g).expand(batch_size, n_group, group_size).contiguous()

    grid = device.compute_with_storage_grid_size()
    core_grid = ttnn.num_cores_to_corerangeset(batch_size, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
    mem_cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
    )

    def upload(t, dtype):
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_cfg, tile=tile)

    ttnn_input = upload(torch.reshape(torch_input, reshaped), ttnn.bfloat16)
    ttnn_bias = upload(torch.transpose(torch.reshape(torch_bias, reshaped), -2, -1), ttnn.bfloat16)

    idx = torch.arange(n_group * group_size, dtype=torch.int32).view(1, n_group, group_size)
    idx = idx.expand(batch_size, n_group, group_size).reshape(reshaped)
    idx = torch.transpose(idx, -2, -1).to(torch.uint16)
    ttnn_idx = upload(idx, ttnn.uint16)

    # FULL 32x32 output (logical) so the dump shows all 4 faces of the tile, incl. dst rows 16-31
    # (step1 reads offsets 16/28 = the bottom half, which a 16x16 face dump never showed).
    out_shaped = (batch_size, 32, 32)
    ttnn_out = upload(torch.zeros(out_shaped, dtype=torch.bfloat16), ttnn.bfloat16)
    ttnn_out_idx = upload(torch.zeros(out_shaped, dtype=torch.uint16), ttnn.uint16)

    GeneralizedMoeGateOp.op(ttnn_input, ttnn_bias, ttnn_out, ttnn_idx, ttnn_out_idx, 1e-20, 1.0, False)

    keys = ttnn.to_torch(ttnn_out)[0].float()
    ids = ttnn.to_torch(ttnn_out_idx)[0].to(torch.int32)

    torch.set_printoptions(profile="full")
    torch.set_printoptions(linewidth=200, precision=1, sci_mode=False)
    logger.info(f"\n=== bias region (ranking keys), 16x16 ===\n{keys}")
    logger.info(f"\n=== indices region (expert ids), 16x16 ===\n{ids}")
    logger.info(f"\n=== group id (= expert_id // 32), 16x16 ===\n{ids // group_size}")


@pytest.mark.parametrize("diag_block", [0, 1])  # must equal GMG_DIAG_BLOCK in the kernel
@pytest.mark.parametrize("enable_sigmoid", [True])
@pytest.mark.parametrize("seed", [42])
def test_generalized_moe_gate_512_per_block(device, diag_block, enable_sigmoid, seed):
    """512-expert bring-up (A1): kernel runs the full per-256 pipeline on each of NUM_BLOCKS blocks
    (one input tile per block) and, with ``GMG_DIAG_BLOCK`` set, outputs ONLY that block's top-8.
    This validates that block in isolation vs its own 256-golden (block-LOCAL indices 0-255; the
    +b*256 offset and cross-block combine come in A2). Keep GMG_DIAG_BLOCK == diag_block.

    Input layout (slice / option 1): per token 512 experts = 2 blocks of 256, each reshaped to (16,16)
    -> face0 of its own 32x32 tile; logits/bias sharded as NUM_BLOCKS tiles/core; indices = one tile.
    """
    num_experts = 512
    num_blocks = num_experts // 256
    batch_size = 1
    eps, scaling_factor = 1e-20, 2.5
    tile = ttnn.Tile((32, 32))

    torch.manual_seed(seed)
    torch_input = (2 * torch.rand((batch_size, num_experts), dtype=torch.bfloat16)) - 1
    if not enable_sigmoid:
        torch_input = torch.sigmoid(torch_input)
    torch_bias = (2 * torch.rand((batch_size, num_experts), dtype=torch.bfloat16)) - 1

    # golden: top-8 of block `diag_block`'s 256 experts (block-LOCAL indices 0-255).
    lo, hi = diag_block * 256, (diag_block + 1) * 256
    scores_blk = torch.sigmoid(torch_input[:, lo:hi]) if enable_sigmoid else torch_input[:, lo:hi]
    bias_key = (scores_blk + torch_bias[:, lo:hi]).float()
    _, gold_local = torch.topk(bias_key, 8, dim=-1, sorted=True)

    logits_blocks = torch_input.reshape(batch_size, num_blocks, 16, 16)
    # bias uploaded transposed within each (16,16) block (kernel expects the transposed layout).
    bias_blocks = torch.transpose(torch_bias.reshape(batch_size, num_blocks, 16, 16), -2, -1).contiguous()

    grid = device.compute_with_storage_grid_size()
    core_grid = ttnn.num_cores_to_corerangeset(batch_size, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)

    def mem(shard):
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(core_grid, shard, ttnn.ShardOrientation.ROW_MAJOR),
        )

    multi, one = (num_blocks * 32, 32), (32, 32)
    ttnn_input = ttnn.from_torch(
        logits_blocks, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem(multi), tile=tile
    )
    ttnn_bias = ttnn.from_torch(
        bias_blocks, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem(multi), tile=tile
    )
    idx = torch.transpose(torch.arange(256, dtype=torch.int32).reshape(batch_size, 16, 16), -2, -1).to(torch.uint16)
    ttnn_input_indices = ttnn.from_torch(
        idx, dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem(one), tile=tile
    )
    out_shape = (batch_size, 1, 16)
    ttnn_output = ttnn.from_torch(
        torch.zeros(out_shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem(one),
        tile=tile,
    )
    ttnn_output_indices = ttnn.from_torch(
        torch.zeros(out_shape, dtype=torch.uint16),
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem(one),
        tile=tile,
    )

    _, res_idx = GeneralizedMoeGateOp.op(
        ttnn_input, ttnn_bias, ttnn_output, ttnn_input_indices, ttnn_output_indices, eps, scaling_factor, enable_sigmoid
    )

    dev_idx = ttnn.to_torch(res_idx)[:, 0, :8].to(torch.int32)
    logger.info(f"block {diag_block}: dev_idx(local)={dev_idx}  gold_local={gold_local}")
    dev_key = torch.gather(bias_key, -1, dev_idx).sort(-1).values
    gold_key = torch.gather(bias_key, -1, gold_local).sort(-1).values
    assert torch.allclose(dev_key, gold_key, atol=1e-2), (
        f"block {diag_block} not a valid top-8.\n dev_idx={dev_idx}\n gold={gold_local}\n"
        f" dev_key={dev_key}\n gold_key={gold_key}"
    )


@pytest.mark.parametrize("output_softmax", [False, True])
@pytest.mark.parametrize("topk", [8, 6, 4])
@pytest.mark.parametrize("enable_sigmoid", [True, False])
@pytest.mark.parametrize("seed", [42, 201, 512])
def test_generalized_moe_gate_512_global(device, enable_sigmoid, seed, topk, output_softmax):
    """512-expert true GLOBAL top-8 (A2 combine). Each of the 2 blocks produces a re-mergeable top-8
    RUN (idx made global via +b*256), stashed to L1; the combine places run0 at {0,2} and run1 at
    {4,6} and finalizes -> the global top-8 over all 512 experts (indices 0-511). GMG_DIAG_BLOCK must
    be UNSET in the kernel. Input layout = slice (each 256-block -> face0 of its own 32x32 tile)."""
    num_experts = 512
    num_blocks = num_experts // 256
    batch_size = 1
    eps, scaling_factor = 1e-20, 2.5
    tile = ttnn.Tile((32, 32))

    torch.manual_seed(seed)
    torch_input = (2 * torch.rand((batch_size, num_experts), dtype=torch.bfloat16)) - 1
    if not enable_sigmoid:
        torch_input = torch.sigmoid(torch_input)
    torch_bias = (2 * torch.rand((batch_size, num_experts), dtype=torch.bfloat16)) - 1

    # Golden: flatten (batch, 512) -> true global top-`topk` (indices 0-511), normalized scores.
    gold_scores, gold_idx = GeneralizedMoeGateOp.golden(
        torch_input, torch_bias, eps, scaling_factor, enable_sigmoid, topk, output_softmax
    )
    scores_all = (torch.sigmoid(torch_input) if enable_sigmoid else torch_input).float()
    bias_key = scores_all + torch_bias.float()  # bias-corrected ranking key, (batch, 512)

    logits_blocks = torch_input.reshape(batch_size, num_blocks, 16, 16)
    # bias uploaded transposed within each (16,16) block (kernel expects the transposed layout).
    bias_blocks = torch.transpose(torch_bias.reshape(batch_size, num_blocks, 16, 16), -2, -1).contiguous()

    grid = device.compute_with_storage_grid_size()
    core_grid = ttnn.num_cores_to_corerangeset(batch_size, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)

    def mem(shard):
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(core_grid, shard, ttnn.ShardOrientation.ROW_MAJOR),
        )

    multi, one = (num_blocks * 32, 32), (32, 32)
    ttnn_input = ttnn.from_torch(
        logits_blocks, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem(multi), tile=tile
    )
    ttnn_bias = ttnn.from_torch(
        bias_blocks, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem(multi), tile=tile
    )
    # input_indices: one tile per block, holding that block's GLOBAL expert ids (block b = arange + b*256),
    # transposed per block (kernel expects the transposed layout). The pipeline tracks global ids directly.
    ar = torch.arange(256, dtype=torch.int32).reshape(1, 1, 16, 16)
    offs = (torch.arange(num_blocks, dtype=torch.int32) * 256).reshape(1, num_blocks, 1, 1)
    idx_blocks = torch.transpose(ar + offs, -2, -1).contiguous().to(torch.uint16)  # (1, num_blocks, 16, 16)
    ttnn_input_indices = ttnn.from_torch(
        idx_blocks, dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem(multi), tile=tile
    )
    out_shape = (batch_size, 1, 16)
    ttnn_output = ttnn.from_torch(
        torch.zeros(out_shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem(one),
        tile=tile,
    )
    ttnn_output_indices = ttnn.from_torch(
        torch.zeros(out_shape, dtype=torch.uint16),
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem(one),
        tile=tile,
    )

    res_scores, res_idx = GeneralizedMoeGateOp.op(
        ttnn_input,
        ttnn_bias,
        ttnn_output,
        ttnn_input_indices,
        ttnn_output_indices,
        eps,
        scaling_factor,
        enable_sigmoid,
        topk,
        output_softmax,
    )

    dev_idx = ttnn.to_torch(res_idx)[:, 0, :topk].to(torch.int64)
    dev_scores = ttnn.to_torch(res_scores)[:, 0, :topk].float()
    logger.info(f"512 global (topk={topk}): dev_idx={dev_idx}  gold_idx={gold_idx}")

    # Indices: dev must be a valid GLOBAL top-`topk` (tie-robust: compare the gathered bias-keys, sorted).
    dev_key = torch.gather(bias_key, -1, dev_idx).sort(-1).values
    gold_key = torch.gather(bias_key, -1, gold_idx.to(torch.int64)).sort(-1).values
    assert torch.allclose(dev_key, gold_key, atol=1e-2), (
        f"512 global not a valid top-{topk}.\n dev_idx={dev_idx}\n gold_idx={gold_idx}\n"
        f" dev_key={dev_key}\n gold_key={gold_key}"
    )

    # Scores: normalized top-8 scores match (sorted, tie-robust).
    assert torch.allclose(
        dev_scores.sort(-1).values, gold_scores.float().sort(-1).values, atol=2e-2
    ), f"512 normalized scores mismatch.\n dev={dev_scores}\n gold={gold_scores}"


def test_dump_stash_run(device):
    """DEBUG PROBE: with GMG_TEST_STASH set, the 256 path produces a run, packs it to L1 run_cb,
    unpacks it back into the indices region, and packs that RAW (no relocate/normalize/step2). This
    reads the full 16x16 idx face to see whether the run's expert ids survived the pack/unpack stash
    (and where they landed). Compare the printed grid to the golden top-8."""
    batch_size = 1
    reshaped = (batch_size, 16, 16)
    shard = (32, 32)
    tile = ttnn.Tile(shard)

    torch.manual_seed(42)
    torch_input = (2 * torch.rand((batch_size, 256), dtype=torch.bfloat16)) - 1
    torch_bias = (2 * torch.rand((batch_size, 256), dtype=torch.bfloat16)) - 1
    scores = torch.sigmoid(torch_input).float()
    key = scores + torch_bias.float()
    gk, gold = torch.topk(key, 8, dim=-1, sorted=True)
    logger.info(f"256 golden top8 (sorted): {sorted(gold[0].tolist())}")
    logger.info(f"256 golden top8 by DESC key (rank0..7): {gold[0].tolist()}")
    logger.info(f"  their keys (rank0..7): {[round(v, 4) for v in gk[0].tolist()]}")

    grid = device.compute_with_storage_grid_size()
    core_grid = ttnn.num_cores_to_corerangeset(batch_size, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)

    def mem(s):
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(core_grid, s, ttnn.ShardOrientation.ROW_MAJOR),
        )

    def up(t, dt):
        return ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem(shard), tile=tile)

    ttnn_input = up(torch.reshape(torch_input, reshaped), ttnn.bfloat16)
    ttnn_bias = up(torch.transpose(torch.reshape(torch_bias, reshaped), -2, -1).contiguous(), ttnn.bfloat16)
    idx = torch.transpose(torch.arange(256, dtype=torch.int32).reshape(batch_size, 16, 16), -2, -1).to(torch.uint16)
    ttnn_idx = up(idx, ttnn.uint16)
    out32 = (batch_size, 32, 32)
    ttnn_out = up(torch.zeros(out32, dtype=torch.bfloat16), ttnn.bfloat16)
    ttnn_out_idx = up(torch.zeros(out32, dtype=torch.uint16), ttnn.uint16)

    GeneralizedMoeGateOp.op(ttnn_input, ttnn_bias, ttnn_out, ttnn_idx, ttnn_out_idx, 1e-20, 2.5, True)

    ids = ttnn.to_torch(ttnn_out_idx)[0].to(torch.int32)
    scs = ttnn.to_torch(ttnn_out)[0].float()
    torch.set_printoptions(linewidth=260)
    # SCORE-CHECK: did transpose_wh-unpack recover the bf16 score field?
    logger.info(f"scores region (after transpose_wh-unpack) 16x16:\n{scs[:16, :16]}")
    logger.info(
        f"scores region all-zero? {float(scs[:16, :16].abs().max()) <= 1e-6}  (max |.|={float(scs[:16,:16].abs().max()):.4f})"
    )
    nz_cols = [c for c in range(16) if float(scs[:16, c].abs().max()) > 1e-6]
    logger.info(f"scores nonzero columns: {nz_cols}")

    # LOCALIZE: the kernel produced block0's run, step2'd it to standard, packed to L1, then transpose_wh-
    # unpacked it back. This dumps the idx region RIGHT AFTER that unpack (no relocate/normalize). If the
    # L1 round-trip + transpose_wh-convert works, block0's golden top-8 expert ids should appear at math
    # cols {0,2}. Report the grid + which golden ids are present and in which columns.
    dump = ids[:16, :16]
    logger.info(f"idx region after L1 round-trip + transpose_wh-unpack (rows 0-15, cols 0-15):\n{dump}")
    goldset = sorted(gold[0].tolist())
    cols_with_gold = {}
    for c in range(16):
        present = sorted(int(v) for v in dump[:, c].tolist() if int(v) in set(goldset))
        if present:
            cols_with_gold[c] = present
    logger.info(f"block0 golden top8: {goldset}")
    logger.info(f"golden ids found per column: {cols_with_gold}")
    allpresent = sorted(int(v) for v in dump.flatten().tolist() if int(v) in set(goldset))
    logger.info(f"golden ids present anywhere in the 16x16: {allpresent}")


def test_dump_combine_run(device):
    """DEBUG: 512 combine in DUMP mode (kernel packs the PLACED idx region (tile1) -> output_indices and the
    bias region (tile2) -> output_cb, merge SKIPPED). Uses a 32x32 output so the FULL 16x16 face is readable.
    Shows both placed runs: block1 at rows {0,2}, block0 at rows {4,6}. Compare to each block's local top-8."""
    num_experts, num_blocks, batch_size = 512, 2, 1
    tile = ttnn.Tile((32, 32))
    torch.manual_seed(42)
    torch_input = (2 * torch.rand((batch_size, num_experts), dtype=torch.bfloat16)) - 1
    torch_bias = (2 * torch.rand((batch_size, num_experts), dtype=torch.bfloat16)) - 1
    key = torch.sigmoid(torch_input).float() + torch_bias.float()
    for b in range(num_blocks):
        _, gl = torch.topk(key[:, b * 256 : (b + 1) * 256], 8, dim=-1, sorted=True)
        logger.info(f"block {b} local top8 (global ids): {sorted((gl[0] + b * 256).tolist())}")
    _, gg = torch.topk(key, 8, dim=-1, sorted=True)
    logger.info(f"512 global top8: {sorted(gg[0].tolist())}")

    logits_blocks = torch_input.reshape(batch_size, num_blocks, 16, 16)
    bias_blocks = torch.transpose(torch_bias.reshape(batch_size, num_blocks, 16, 16), -2, -1).contiguous()
    grid = device.compute_with_storage_grid_size()
    core_grid = ttnn.num_cores_to_corerangeset(batch_size, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)

    def mem(shard):
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(core_grid, shard, ttnn.ShardOrientation.ROW_MAJOR),
        )

    multi = (num_blocks * 32, 32)
    ttnn_input = ttnn.from_torch(
        logits_blocks, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem(multi), tile=tile
    )
    ttnn_bias = ttnn.from_torch(
        bias_blocks, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem(multi), tile=tile
    )
    ar = torch.arange(256, dtype=torch.int32).reshape(1, 1, 16, 16)
    offs = (torch.arange(num_blocks, dtype=torch.int32) * 256).reshape(1, num_blocks, 1, 1)
    idx_blocks = torch.transpose(ar + offs, -2, -1).contiguous().to(torch.uint16)
    ttnn_input_indices = ttnn.from_torch(
        idx_blocks, dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem(multi), tile=tile
    )
    out32 = (batch_size, 32, 32)
    ttnn_output = ttnn.from_torch(
        torch.zeros(out32, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem((32, 32)),
        tile=tile,
    )
    ttnn_output_indices = ttnn.from_torch(
        torch.zeros(out32, dtype=torch.uint16),
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem((32, 32)),
        tile=tile,
    )

    GeneralizedMoeGateOp.op(
        ttnn_input, ttnn_bias, ttnn_output, ttnn_input_indices, ttnn_output_indices, 1e-20, 2.5, True
    )

    ids = ttnn.to_torch(ttnn_output_indices)[0].to(torch.int32)
    scr = ttnn.to_torch(ttnn_output)[0].float()  # bias region (dump mode: tile2 -> output_cb)
    torch.set_printoptions(linewidth=260)
    logger.info(f"PLACED idx region full 16x16 (block1 rows 0,2; block0 rows 4,6):\n{ids[:16, :16]}")
    logger.info(f"PLACED bias region full 16x16 (sort key):\n{scr[:16, :16]}")


if __name__ == "__main__":
    pytest.main([__file__])
