import pytest
import torch
import ttnn
from loguru import logger
from models.utility_functions import comp_pcc


@pytest.mark.timeout(30)
def test_moreh_group_norm_program_cache_optional_outputs_override(device):
    """
    Exposes a program-cache override bug in moreh_group_norm where optional outputs (mean/rstd)
    are unconditionally dereferenced in override_runtime_arguments on cache-hit.

    Location:
      - ttnn/cpp/ttnn/operations/moreh/moreh_group_norm/device/moreh_group_norm_program_factory.cpp
        lines 317-319 dereference optional outputs without has_value() checks.

    Expected failure mode:
      - Crash or hang on the second run (cache-hit) when mean/rstd are not requested.
    """

    torch.manual_seed(0)

    # Shapes and attributes (hashed properties remain identical across runs)
    N, C, H, W = 2, 4, 32, 32
    num_groups = 2
    eps = 1e-5

    # 1) First run - compile and seed program cache (do not request mean/rstd)
    logger.debug("Executing first run")
    x1 = torch.randn((N, C, H, W)).bfloat16()

    tt_x1 = ttnn.from_torch(x1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    num_cache_start = device.num_program_cache_entries()
    logger.debug(f"Number of program cache entries (start): {num_cache_start}")
    logger.debug("Launching OP for first run")
    out1, mean1, rstd1 = ttnn.operations.moreh.group_norm(
        tt_x1,
        num_groups,
        eps,
        gamma=None,
        beta=None,
        are_required_outputs=(True, False, False),
        mean=None,
        rstd=None,
    )
    num_cache_end = device.num_program_cache_entries()
    assert num_cache_end == num_cache_start + 1, "Expected one new program cache entry on first run"
    logger.debug(f"Number of program cache entries (end of first run): {num_cache_end}")

    # Validate correctness against torch for the first run
    out1_host = ttnn.to_torch(out1)
    golden1 = torch.nn.functional.group_norm(x1.float(), num_groups=num_groups, eps=eps).to(dtype=torch.bfloat16)
    ok, pcc = comp_pcc(out1_host, golden1)
    logger.debug(f"First run PCC: ok={ok}, pcc={pcc}")
    assert ok, f"First run PCC failed: {pcc}"

    # 2) Second run - allocate new buffers (addresses differ) but keep hashed properties identical
    logger.debug("Executing second run")
    x2 = torch.randn((N, C, H, W)).bfloat16()
    tt_x2 = ttnn.from_torch(x2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    logger.debug("Launching OP for second run (cache-hit expected)")
    # Expected behavior: cache hit triggers override path; bug causes deref of missing mean/rstd
    out2, mean2, rstd2 = ttnn.operations.moreh.group_norm(
        tt_x2,
        num_groups,
        eps,
        gamma=None,
        beta=None,
        are_required_outputs=(True, False, False),
        mean=None,
        rstd=None,
    )

    # If it reaches here without crash, also compare PCC (not strictly required to expose the bug)
    out2_host = ttnn.to_torch(out2)
    golden2 = torch.nn.functional.group_norm(x2.float(), num_groups=num_groups, eps=eps).to(dtype=torch.bfloat16)
    ok2, pcc2 = comp_pcc(out2_host, golden2)
    logger.debug(f"Second run PCC: ok={ok2}, pcc={pcc2}")
    assert ok2, f"Second run PCC failed: {pcc2}"
