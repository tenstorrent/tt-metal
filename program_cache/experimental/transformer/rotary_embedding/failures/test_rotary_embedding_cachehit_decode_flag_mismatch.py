import pytest
import torch
import ttnn
from loguru import logger


@pytest.mark.timeout(30)
def test_rotary_embedding_program_cache_decode_flag_mismatch(device):
    """
    This test targets under-keyed program hashing for experimental/transformer/rotary_embedding:

    - Suspected issue: custom program hash omits `token_idx` and `compute_kernel_config`.
      File: ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/rotary_embedding_device_operation.cpp (compute_program_hash)
      The program factory conditionally compiles different kernels/CBs based on `token_idx` (DECODE_MODE define),
      but `token_idx` is not included in the custom hash, causing cache entry reuse across prefill vs decode.

    - Failure expectation: second run reuses the prefill-compiled program for a decode invocation and hangs during kernel execution.
      We let the test fail via timeout on the cache-hit path.
    """

    torch.manual_seed(0)

    # Shapes chosen so that seq_len == 1 to keep the hashed seq_len identical across runs
    # Input tensor logical/padded dims: [B, H, seq_len, head_dim]
    B = 1
    H = 1
    seq_len = 1  # important: same in both runs so hash (which includes seq_len) stays the same
    head_dim = 128  # multiple of 2*TILE_WIDTH (>= 2*64) requirement

    # Host tensors
    a1 = torch.randn(B, H, seq_len, head_dim).bfloat16()
    cos_cache = torch.randn(1, 1, max(16, seq_len), head_dim).bfloat16()
    sin_cache = torch.randn(1, 1, max(16, seq_len), head_dim).bfloat16()

    # Device tensors, interleaved, TILE layout
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    tt_a1 = ttnn.Tensor(a1, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device, mem_config)
    tt_cos = ttnn.Tensor(cos_cache, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device, mem_config)
    tt_sin = ttnn.Tensor(sin_cache, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device, mem_config)

    # 1) First run (prefill): token_index=None builds and seeds the cache
    logger.debug("Executing first run (prefill, token_index=None)")
    num_cache_start = device.num_program_cache_entries()
    out1 = ttnn.experimental.rotary_embedding(tt_a1, tt_cos, tt_sin, token_index=None, memory_config=mem_config)
    num_cache_end = device.num_program_cache_entries()
    assert num_cache_end == num_cache_start + 1, "Expected one new program cache entry on first run"

    # 2) Second run (decode): same shapes/mem_config but now with token_index set
    # Reallocate inputs to force different buffer base addresses while keeping hash-equal properties
    a2 = torch.randn(B, H, seq_len, head_dim).bfloat16()
    tt_a2 = ttnn.Tensor(a2, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device, mem_config)

    logger.debug("Executing second run (decode, token_index set) - cache hit expected")
    # Expectation: this reuses the prefill program (same hash) and hangs due to DECODE_MODE mismatch
    _ = ttnn.experimental.rotary_embedding(tt_a2, tt_cos, tt_sin, token_index=5, memory_config=mem_config)
