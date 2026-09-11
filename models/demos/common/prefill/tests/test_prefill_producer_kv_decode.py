# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from models.demos.common.prefill.runners.prefill_producer import _decode_kv_chunk


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.bfloat16], ids=["fp8_e4m3", "bfloat16"])
def test_decode_row_major_kv_chunk(dtype):
    head_dim = 576
    values = torch.linspace(-2, 2, 32 * head_dim, dtype=torch.float32).reshape(32, head_dim).to(dtype)
    raw = values.view(torch.uint8).numpy().tobytes()

    actual = _decode_kv_chunk(raw, head_dim)

    assert torch.equal(actual, values.float())


def test_decode_row_major_fp8_kv_chunk_with_page_padding():
    head_dim = 33
    row_size_bytes = 64
    values = (
        torch.arange(32 * head_dim, dtype=torch.float32).reshape(32, head_dim).remainder(31).to(torch.float8_e4m3fn)
    )
    padded = torch.full((32, row_size_bytes), 0xA5, dtype=torch.uint8)
    padded[:, :head_dim] = values.view(torch.uint8)

    actual = _decode_kv_chunk(padded.numpy().tobytes(), head_dim)

    assert torch.equal(actual, values.float())


def test_decode_packed_scaled_fp8_kv_chunk_with_page_padding():
    latent = torch.arange(32 * 512, dtype=torch.float32).reshape(32, 512).remainder(31).sub(15).to(torch.float8_e4m3fn)
    scales = torch.tensor([0.25, 0.5, 1.0, 2.0], dtype=torch.float32).repeat(32, 1)
    rope = torch.arange(32 * 64, dtype=torch.float32).reshape(32, 64).to(torch.bfloat16)
    rows = torch.full((32, 672), 0xA5, dtype=torch.uint8)
    rows[:, :512] = latent.view(torch.uint8)
    rows[:, 512:528] = scales.view(torch.uint8)
    rows[:, 528:656] = rope.view(torch.uint8)

    actual = _decode_kv_chunk(rows.numpy().tobytes(), head_dim=576)
    expected = torch.cat((latent.float() * scales.repeat_interleave(128, dim=-1), rope.float()), dim=-1)

    assert torch.equal(actual, expected)


def test_decode_unknown_kv_chunk_rejected(expect_error):
    with expect_error(ValueError, "unsupported"):
        _decode_kv_chunk(bytes(17), head_dim=576)
