import pytest
import torch
import ttnn
from loguru import logger
from models.utility_functions import comp_pcc


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x: torch.Tensor, cos_cached: torch.Tensor, sin_cached: torch.Tensor, token_idx=None):
    seq_len = x.shape[-2]
    if token_idx is None:
        cos = cos_cached[:, :, :seq_len, ...]
        sin = sin_cached[:, :, :seq_len, ...]
    else:
        cos = cos_cached[:, :, token_idx : token_idx + 1, ...]
        sin = sin_cached[:, :, token_idx : token_idx + 1, ...]
    return (x * cos) + (rotate_half(x) * sin)


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

    # Use TILE-aligned shapes to avoid layout conversion errors while keeping
    # all hash-included dimensions identical across runs.
    # Input tensor logical/padded dims: [B, H, seq_len, head_dim]
    # Important: prefill hashes seq_len from [-2]; decode hashes seq_len from [0] (batch).
    # Decode requires batch==1. Set B == seq_len == 1 to equalize the hashed seq_len across runs.
    B = 1
    H = 1
    seq_len = 1
    head_dim = 128  # multiple of tile width (32)

    # Cache must also be TILE-aligned in height (use 2048 like reference tests)
    cache_size = 2048

    # Host tensors
    a1 = torch.randn(B, H, seq_len, head_dim).bfloat16()
    cos_cache = torch.randn(1, 1, cache_size, head_dim).bfloat16()
    sin_cache = torch.randn(1, 1, cache_size, head_dim).bfloat16()

    # Keep inputs in ROW_MAJOR and move to the provided device; auto-format will tilize as needed.
    # Important: create fresh device tensors per run to keep the hashed tensor layout and metadata identical across runs.
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    tt_a1 = ttnn.Tensor(a1, ttnn.bfloat16).to(device, mem_config)
    tt_cos = ttnn.Tensor(cos_cache, ttnn.bfloat16).to(device, mem_config)
    tt_sin = ttnn.Tensor(sin_cache, ttnn.bfloat16).to(device, mem_config)

    # 1) First run (prefill): token_index=None builds and seeds the cache
    logger.debug("Executing first run (prefill, token_index=None)")
    num_cache_start = device.num_program_cache_entries()
    out1 = ttnn.experimental.rotary_embedding(tt_a1, tt_cos, tt_sin, memory_config=mem_config)
    num_cache_end = device.num_program_cache_entries()
    logger.debug(
        f"Prefill compiled {num_cache_end - num_cache_start} new program(s); proceeding to decode step for cache validation."
    )

    # 2) Second run (decode): same shapes/mem_config but now with token_index set
    # Recreate device tensors to ensure the hashed input tensor metadata (layout, etc.) matches the first-run state.
    a2 = torch.randn(B, H, seq_len, head_dim).bfloat16()
    tt_a2 = ttnn.Tensor(a2, ttnn.bfloat16).to(device, mem_config)
    tt_cos2 = ttnn.Tensor(cos_cache, ttnn.bfloat16).to(device, mem_config)
    tt_sin2 = ttnn.Tensor(sin_cache, ttnn.bfloat16).to(device, mem_config)
    logger.debug("Executing second run (decode, token_index set) - should be a distinct program if hash is correct")
    num_cache_before_second = device.num_program_cache_entries()
    token_idx = 5
    out2_tt = ttnn.experimental.rotary_embedding(tt_a2, tt_cos2, tt_sin2, token_idx, memory_config=mem_config)
    num_cache_after_second = device.num_program_cache_entries()
    delta = num_cache_after_second - num_cache_before_second
    # Expectation: switching from prefill to decode must create a distinct program.
    # If the program hash omits token_idx/DECODE_MODE, we would see delta == 0 (cache hit), which is a bug.
    assert delta >= 1, "Expected at least one new program when switching from prefill to decode (token index provided)."

    # Additionally validate correctness of decode output against PyTorch reference
    got = out2_tt.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    ref = apply_rotary_pos_emb(a2.float(), cos_cache.float(), sin_cache.float(), token_idx)
    p, o = comp_pcc(ref, got)
    logger.info(o)
    assert p
