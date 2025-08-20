import pytest
import torch
import ttnn
from loguru import logger
from models.utility_functions import comp_pcc


@pytest.mark.timeout(30)
def test_split_last_dim_two_chunks_tiled_program_cache_override_output1_addr_bug(device):
    torch.manual_seed(0)

    # Choose a shape that triggers the tiled two-chunk split fast-path
    # Conditions (from SplitOperation):
    # - split sizes == 2 on last dim
    # - TILE layout
    # - input_shape[-2]/32 >= 2 and input_shape[-1]/32 >= 2
    # - fits_in_core_grid across first two dims
    shape = (1, 1, 64, 64)

    # 1) First run compiles and seeds the cache
    logger.debug("Executing first run")
    a1 = torch.randn(shape).bfloat16()
    tt_a1 = ttnn.Tensor(a1, ttnn.bfloat16).to(ttnn.TILE_LAYOUT)
    tt_a1 = tt_a1.to(device, ttnn.DRAM_MEMORY_CONFIG)

    num_cache_start = device.num_program_cache_entries()
    logger.debug(f"Number of program cache entries (start): {num_cache_start}")

    logger.debug("Launching split op for first run")
    out1_parts = ttnn.split(tt_a1, 2, 3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    num_cache_end = device.num_program_cache_entries()
    assert num_cache_end == num_cache_start + 1, "Expected one new program cache entry on first run"
    logger.debug("Finished first run")

    # Validate correctness vs. golden
    out1_0 = out1_parts[0].cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    out1_1 = out1_parts[1].cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    golden1_0, golden1_1 = torch.tensor_split(a1.float(), 2, dim=-1)
    ok0, pcc0 = comp_pcc(out1_0, golden1_0)
    ok1, pcc1 = comp_pcc(out1_1, golden1_1)
    logger.debug(f"First run PCC: part0 ok={ok0}, pcc={pcc0}; part1 ok={ok1}, pcc={pcc1}")
    assert ok0 and ok1, f"First run PCC failed: part0 {pcc0}, part1 {pcc1}"

    # 2) Second run hits cache and triggers override path
    logger.debug("Executing second run (cache-hit expected)")
    a2 = torch.randn(shape).bfloat16()
    tt_a2 = ttnn.Tensor(a2, ttnn.bfloat16).to(ttnn.TILE_LAYOUT)
    tt_a2 = tt_a2.to(device, ttnn.DRAM_MEMORY_CONFIG)

    logger.debug("Launching split op for second run (cache-hit expected)")
    out2_parts = ttnn.split(tt_a2, 2, 3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    logger.debug("Finished second run")

    # Expect a failure on cache-hit due to wrong output1 base address override
    out2_0 = out2_parts[0].cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    out2_1 = out2_parts[1].cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    golden2_0, golden2_1 = torch.tensor_split(a2.float(), 2, dim=-1)
    ok0, pcc0 = comp_pcc(out2_0, golden2_0)
    ok1, pcc1 = comp_pcc(out2_1, golden2_1)
    logger.debug(f"Second run PCC: part0 ok={ok0}, pcc={pcc0}; part1 ok={ok1}, pcc={pcc1}")

    # Let this assertion FAIL on cache-hit path due to the override bug
    assert ok0 and ok1, "PCC mismatch on cache-hit path (expected failure if output1 addr is not overridden correctly)"
