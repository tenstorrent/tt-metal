import pytest, torch, ttnn
from loguru import logger


@pytest.mark.timeout(30)
def test_moreh_sgd_program_cache_override_rtargs_optional_momentum_out(device):
    torch.manual_seed(0)

    # Use small shape to keep runtime short
    shape = [32, 32]
    lr = 0.1
    momentum = 0.0  # No momentum → no momentum_buffer_out
    dampening = 0.0
    weight_decay = 0.0
    nesterov = False
    momentum_initialized = False

    # 1) First run compiles and seeds the cache
    logger.debug("Executing first run")
    a1 = torch.randn(shape).bfloat16()
    g1 = torch.randn(shape).bfloat16()

    tt_param_in1 = ttnn.from_torch(a1, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_grad1 = ttnn.from_torch(g1, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_param_out1 = ttnn.from_torch(a1, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    num_cache_start = device.num_program_cache_entries()
    logger.debug(f"Number of program cache entries: {num_cache_start}")
    logger.debug("Launching OP for first run")
    # momentum==0 → momentum buffers are None
    tt_param_out1, tt_momentum_out1 = ttnn.operations.moreh.sgd(
        tt_param_in1,
        tt_grad1,
        None,  # momentum_buffer_in
        tt_param_out1,  # param_out
        None,  # momentum_buffer_out
        lr,
        momentum,
        dampening,
        weight_decay,
        nesterov,
        momentum_initialized=momentum_initialized,
    )

    num_cache_end = device.num_program_cache_entries()
    logger.debug(f"Number of program cache entries after first run: {num_cache_end}")
    assert num_cache_end == num_cache_start + 1, "Expected one new program cache entry on first run"

    # 2) Second run hits cache and triggers override path
    logger.debug("Executing second run (cache-hit expected)")
    a2 = torch.randn(shape).bfloat16()
    g2 = torch.randn(shape).bfloat16()
    tt_param_in2 = ttnn.from_torch(a2, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_grad2 = ttnn.from_torch(g2, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_param_out2 = ttnn.from_torch(a2, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    logger.debug("Launching OP for second run (should reuse cached program and override rtargs)")
    # Expect failure on cache-hit due to override dereferencing optional momentum out unconditionally
    tt_param_out2, tt_momentum_out2 = ttnn.operations.moreh.sgd(
        tt_param_in2,
        tt_grad2,
        None,
        tt_param_out2,
        None,
        lr,
        momentum,
        dampening,
        weight_decay,
        nesterov,
        momentum_initialized=momentum_initialized,
    )
