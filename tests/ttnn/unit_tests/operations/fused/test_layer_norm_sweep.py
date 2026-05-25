# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
LayerNorm shape-sweep harness.

Covers common real-model hidden dimensions (BERT, GPT-2, Llama, ViT) across a
range of leading-dim sizes, plus a few non-tile-aligned hidden widths. Cheap
per op (no matmul), so good for stress-testing the harness/xdist scaling
without the FLOP cost of conv or SDPA.

Filter examples:
  pytest -k "llama"
  pytest -k "bert or vit"
  pytest -m "not slow"

See scripts/run_layernorm_sweep.sh for the recommended invocation.
"""

import pytest
import torch
import ttnn

from tests.ttnn.unit_tests.operations.fused.test_layer_norm import (
    PAD_VALUE,
    assert_output_accuracy,
    create_recip_tensor,
)


# ---------------------------------------------------------------------------
# Layernorm presets
# h = leading dim (~batch * seq), w = hidden dim
# ---------------------------------------------------------------------------
LAYERNORM_CASES = [
    # --- BERT family ---
    pytest.param(32, 768, id="bert_base_h32"),
    pytest.param(128, 768, id="bert_base_h128"),
    pytest.param(512, 768, id="bert_base_h512"),
    pytest.param(128, 1024, id="bert_large_h128"),
    pytest.param(1024, 1024, id="bert_large_h1024"),
    # --- GPT-2 family ---
    pytest.param(128, 1280, id="gpt2_large_h128"),
    pytest.param(128, 1600, id="gpt2_xl_h128"),
    # --- Llama family ---
    pytest.param(32, 4096, id="llama7b_h32"),
    pytest.param(2048, 4096, id="llama7b_h2048"),
    pytest.param(32, 5120, id="llama13b_h32"),
    pytest.param(2048, 5120, id="llama13b_h2048"),
    pytest.param(32, 8192, id="llama65b_h32"),
    pytest.param(1024, 8192, id="llama65b_h1024"),
    # --- Vision (ViT) ---
    pytest.param(196, 768, id="vit_base_h196"),
    pytest.param(196, 1024, id="vit_large_h196"),
    # --- Non-tile-aligned hidden dim (padding path) ---
    pytest.param(128, 127, id="non_aligned_w127"),
    pytest.param(128, 519, id="non_aligned_w519"),
    pytest.param(128, 1023, id="non_aligned_w1023"),
    # --- Stress ---
    pytest.param(4096, 4096, id="stress_llama7b_4kx4k"),
    pytest.param(8192, 8192, id="stress_8kx8k", marks=pytest.mark.slow),
]


# ---------------------------------------------------------------------------
# LayerNorm sweep (with weight + bias — the common case in models)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16, torch.float32],
    ids=["bf16", "fp32"],
)
@pytest.mark.parametrize("use_welford", [True, False], ids=["welford", "naive"])
@pytest.mark.parametrize("h, w", LAYERNORM_CASES)
@pytest.mark.timeout(600)
def test_layer_norm_sweep(device, h, w, use_welford, dtype):
    torch.manual_seed(0)

    torch_x = torch.rand((h, w), dtype=dtype)
    torch_weight = torch.rand((w,), dtype=dtype)
    torch_bias = torch.rand((w,), dtype=dtype)
    torch_out = torch.nn.functional.layer_norm(torch_x, normalized_shape=[w], weight=torch_weight, bias=torch_bias)

    x = ttnn.from_torch(torch_x, layout=ttnn.TILE_LAYOUT, device=device)
    x = ttnn.fill_implicit_tile_padding(x, PAD_VALUE)
    weight = ttnn.from_torch(torch_weight, layout=ttnn.TILE_LAYOUT, device=device)
    bias = ttnn.from_torch(torch_bias, layout=ttnn.TILE_LAYOUT, device=device)

    program_config = ttnn.LayerNormDefaultProgramConfig(use_welford=use_welford)
    recip_tensor = create_recip_tensor(device, w, use_welford)

    tt_out = ttnn.layer_norm(
        x,
        weight=weight,
        bias=bias,
        program_config=program_config,
        recip_tensor=recip_tensor,
    )
    tt_out = ttnn.to_torch(ttnn.from_device(tt_out))

    assert_output_accuracy(torch_out, tt_out)
