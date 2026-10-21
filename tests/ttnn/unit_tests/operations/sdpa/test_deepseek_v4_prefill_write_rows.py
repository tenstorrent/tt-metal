# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.transformer.deepseek_v4_prefill_write_rows: an in-place row scatter at device-held row indices."""

import pytest
import torch
import ttnn


def _dram(device, x: torch.Tensor, dtype) -> ttnn.Tensor:
    return ttnn.from_torch(
        x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _run(device, n: int, rows_out: int, width: int, index: torch.Tensor) -> None:
    torch.manual_seed(n * 7919 + rows_out + width)
    src = torch.randn(1, 1, n, width).bfloat16()
    dst = torch.randn(1, 1, rows_out, width).bfloat16()
    want = dst.clone()
    for i, r in enumerate(index.tolist()):
        if r < rows_out:
            want[0, 0, r] = src[0, 0, i]
    dst_tt = _dram(device, dst, ttnn.bfloat16)
    out = ttnn.transformer.deepseek_v4_prefill_write_rows(
        _dram(device, src, ttnn.bfloat16), dst_tt, _dram(device, index.to(torch.int32), ttnn.uint32)
    )
    assert out.buffer_address() == dst_tt.buffer_address()
    got = ttnn.to_torch(dst_tt)
    assert torch.equal(got, want)


@pytest.mark.parametrize("width", [128, 512])
@pytest.mark.parametrize("n, rows_out", [(512, 4096), (1024, 1024), (8, 64), (100, 300)])
def test_deepseek_v4_prefill_write_rows(device, width, n, rows_out):
    _run(device, n, rows_out, width, torch.randperm(rows_out)[:n] if n < rows_out else torch.randperm(n))


@pytest.mark.parametrize("width", [128, 512])
def test_deepseek_v4_prefill_write_rows_contiguous_and_skipped(device, width):
    n, rows_out = 512, 2048
    # A chunk's rows: contiguous at an offset, the first few redirected past the end (skipped).
    index = 1536 + torch.arange(n)
    index[:16] = rows_out
    _run(device, n, rows_out, width, index)


def test_deepseek_v4_prefill_write_rows_program_cache(device):
    """A different output size and new indices reuse the program (both ride on runtime args)."""
    width, n = 512, 256
    _run(device, n, 1024, width, 256 + torch.arange(n))
    entries = device.num_program_cache_entries()
    _run(device, n, 4096, width, torch.randperm(4096)[:n])
    assert device.num_program_cache_entries() == entries
