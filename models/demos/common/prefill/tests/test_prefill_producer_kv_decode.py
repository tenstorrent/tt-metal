# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from models.demos.common.prefill.runners.prefill_producer import (
    CHUNK_SIZE,
    ProducerConfig,
    _decode_kv_chunk,
    _resolve_slot_prompts,
)


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


def test_synthetic_tokens_do_not_require_golden_trace(monkeypatch, expect_error):
    monkeypatch.setenv("PREFILL_PRODUCER_SYNTHETIC_TOKENS", "1")
    monkeypatch.setenv("PREFILL_TRACE_DIR", "/path/does/not/exist")
    cfg = ProducerConfig(
        num_users=2,
        chunks_min=1,
        chunks_max=2,
        max_requests=1,
        duration_s=1,
        p_gap=0,
        p_burst=0,
        gap_ms=(0, 0),
        mid_chunk_end_prob=0,
        seed=1,
        verify=False,
        pcc_threshold=0.9,
    )

    slot_traces, slot_lengths, pools = _resolve_slot_prompts(cfg)

    assert slot_lengths is None
    assert slot_traces == {0: "<synthetic>", 1: "<synthetic>"}
    assert pools == {"<synthetic>": [1] * (2 * CHUNK_SIZE)}

    cfg.verify = True
    with expect_error(ValueError, "cannot be used"):
        _resolve_slot_prompts(cfg)
