# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""``decode_sample_candidates`` against a host top-k: values (times ``inv_temp``) and indices exactly, ties by lower
index, then the fill. Shapes: the TP4 ``lm_head`` shard (32320), a partial last tile, few tiles.

    pytest models/experimental/deepseek_v41_flash/tests/decode/test_sample_candidates.py
"""

import pytest
import torch

import ttnn

FILL = -1.0e9


def _reference(logits: torch.Tensor, k: int, inv_temp: float, offset: int):
    vals = logits.reshape(-1).bfloat16()
    order = sorted(range(vals.numel()), key=lambda i: (-float(vals[i]), i))[:k]
    top = vals[order]
    scaled = (top.float() * torch.tensor(inv_temp).bfloat16().float()).bfloat16()
    want_vals = torch.cat([scaled.float(), torch.full((k,), FILL).bfloat16().float()])
    want_idx = torch.tensor([i + offset for i in order] + [0] * k, dtype=torch.int64)
    return want_vals, want_idx


@pytest.mark.parametrize("v", (32320, 1000, 64))
@pytest.mark.parametrize("k", (32, 16))
@pytest.mark.parametrize("inv_temp", (1.0, 1.4375))
@pytest.mark.parametrize("ties", (False, True))
def test_sample_candidates(device, v, k, inv_temp, ties):
    torch.manual_seed(v + k)
    logits = torch.randn(1, 1, 1, v) * 4
    if ties:
        # Coarse values: many equal logits, including at the top.
        logits = torch.round(logits)
    logits = logits.bfloat16().float()
    offset = 3 * 32320
    tt_logits = ttnn.from_torch(
        logits, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_temp = ttnn.from_torch(
        torch.full((1, 1, 1, 1), inv_temp), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    tt_offset = ttnn.from_torch(
        torch.full((1, 1, 1, 1), offset, dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    want_vals, want_idx = _reference(logits, k, inv_temp, offset)
    # Twice: the second run is from the program cache.
    for _ in range(2):
        vals, idx = ttnn.experimental.deepseek.decode_sample_candidates(
            tt_logits, tt_temp, tt_offset, k=k, fill_value=FILL
        )
        assert vals.layout == ttnn.TILE_LAYOUT and vals.dtype == ttnn.bfloat16 and list(vals.shape) == [1, 1, 1, 2 * k]
        assert idx.layout == ttnn.ROW_MAJOR_LAYOUT and idx.dtype == ttnn.int32 and list(idx.shape) == [1, 1, 1, 2 * k]
        got_vals = ttnn.to_torch(vals).float().reshape(-1)
        got_idx = ttnn.to_torch(idx).to(torch.int64).reshape(-1)
        assert torch.equal(got_idx, want_idx), (got_idx[:k], want_idx[:k])
        assert torch.equal(got_vals, want_vals), (got_vals[:k], want_vals[:k])
