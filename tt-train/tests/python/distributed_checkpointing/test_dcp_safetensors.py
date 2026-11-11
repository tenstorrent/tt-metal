# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The per-device files are real safetensors files: fixed offsets up front, 4 KiB-aligned data, readable by the
``safetensors`` package."""

import json
import os

import numpy as np
import pytest
import torch

from ttml.distributed_checkpointing._safetensors import (
    ALIGNMENT,
    build_header,
    create_file,
    pread_all,
    pwrite_all,
    read_header,
    read_metadata,
)


def test_header_fixes_offsets_and_aligns_the_data_section(expect_error):
    header, offsets, data_bytes = build_header(
        [("model/a", "BF16", (1, 1, 32, 64), 4096), ("model/b", "F32", (1, 1, 8, 8), 256)],
        metadata={"ttml.format": "ttml-dcp/1", "ttml.writer": {"world_rank": 0, "coord": [0, 1]}},
    )
    assert offsets == {"model/a": (0, 4096), "model/b": (4096, 4352)} and data_bytes == 4352
    assert len(header) % ALIGNMENT == 0
    with expect_error(ValueError, "needs"):
        build_header([("x", "BF16", (1, 1, 32, 64), 100)])
    with expect_error(ValueError, "twice"):
        build_header([("x", "BF16", (1, 1, 1, 2), 4), ("x", "BF16", (1, 1, 1, 2), 4)])


def test_file_written_at_fixed_offsets_opens_with_the_safetensors_package(tmp_path):
    pytest.importorskip("safetensors")
    from safetensors import safe_open

    a = torch.randn(1, 1, 32, 64).to(torch.bfloat16)
    b = torch.randn(1, 1, 8, 8)
    header, offsets, data_bytes = build_header(
        [("model/a", "BF16", a.shape, a.numel() * 2), ("model/b", "F32", b.shape, b.numel() * 4)],
        metadata={"ttml.format": "ttml-dcp/1"},
    )
    path = str(tmp_path / "w000_m0_d0-0.safetensors")
    data_start = create_file(path, header, data_bytes)
    assert data_start % ALIGNMENT == 0 and os.path.getsize(path) == data_start + data_bytes

    fd = os.open(path, os.O_RDWR)
    try:
        # Written out of order, as per-device writer threads would.
        pwrite_all(fd, b.contiguous().numpy(), data_start + offsets["model/b"][0])
        pwrite_all(fd, a.contiguous().view(torch.int16).numpy(), data_start + offsets["model/a"][0])
        back = np.frombuffer(pread_all(fd, b.numel() * 4, data_start + offsets["model/b"][0]), dtype=np.float32)
        np.testing.assert_array_equal(back.reshape(b.shape), b.numpy())
    finally:
        os.close(fd)

    parsed, start = read_header(path)
    assert start == data_start and parsed["model/a"]["dtype"] == "BF16" and parsed["model/a"]["shape"] == [1, 1, 32, 64]
    assert read_metadata(parsed) == {"ttml.format": "ttml-dcp/1"}

    with safe_open(path, framework="pt") as f:
        assert set(f.keys()) == {"model/a", "model/b"}
        torch.testing.assert_close(f.get_tensor("model/a"), a, rtol=0, atol=0)
        torch.testing.assert_close(f.get_tensor("model/b"), b, rtol=0, atol=0)
        assert f.metadata() == {"ttml.format": "ttml-dcp/1"}


def test_metadata_reads_back_exactly_what_was_written(expect_error):
    metadata = {"format": "ttml-dcp/1", "a": "1", "b": "true", "c": "null", "writer": {"coord": [0, 1]}, "l": [1, 2]}
    header, _, _ = build_header([("x", "BF16", (1, 1, 1, 2), 4)], metadata)
    parsed = json.loads(header[8:])
    assert parsed["__metadata__"]["a"] == "1" and parsed["__metadata__"]["format"] == "ttml-dcp/1", "stored as is"
    assert read_metadata(parsed) == metadata
    with expect_error(ValueError, "reads back as JSON"):
        build_header([], {"a": "[not a list]"})
    with expect_error(ValueError, "must be a str, dict or list"):
        build_header([], {"steps": 3})
    assert read_metadata({"__metadata__": {"foreign": "{not json"}}) == {"foreign": "{not json"}
