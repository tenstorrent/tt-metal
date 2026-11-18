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
    read_file_header,
)


def test_header_fixes_offsets_and_aligns_the_data_section(expect_error):
    header, offsets, data_bytes = build_header(
        [("model/a", "BF16", (1, 1, 32, 64), 4096), ("model/b", "F32", (1, 1, 8, 8), 256)],
        metadata={"ttml.format": "ttml-dcp/1"},
    )
    assert offsets == {"model/a": (0, 4096), "model/b": (4096, 4352)} and data_bytes == 4352
    assert len(header) % ALIGNMENT == 0
    with expect_error(ValueError, "needs"):
        build_header([("x", "BF16", (1, 1, 32, 64), 100)])
    with expect_error(ValueError, "twice"):
        build_header([("x", "BF16", (1, 1, 1, 2), 4), ("x", "BF16", (1, 1, 1, 2), 4)])
    with expect_error(ValueError, "unsupported dtype 'F8'"):
        build_header([("x", "F8", (1, 1, 1, 2), 2)])


def test_short_and_truncated_files_are_refused(tmp_path, expect_error):
    header, _, data_bytes = build_header([("x", "F32", (1, 1, 1, 4), 16)])
    path = str(tmp_path / "w000_m0_d0.safetensors")
    with open(path, "wb") as f:
        f.write(header[:5])
    with expect_error(ValueError, "too short for a header"):
        read_file_header(path)
    with open(path, "wb") as f:
        f.write(header[:-3])
    with expect_error(ValueError, "truncated inside its header"):
        read_file_header(path)
    data_start = create_file(path, header, data_bytes)
    fd = os.open(path, os.O_RDONLY)
    try:
        assert len(pread_all(fd, data_bytes, data_start)) == data_bytes
        with expect_error(EOFError, f"read past the end of the file at offset {data_start + data_bytes}"):
            pread_all(fd, data_bytes + 4, data_start)
    finally:
        os.close(fd)


def test_file_opens_with_safetensors(tmp_path):
    """A file written at fixed offsets, out of order, opens with the safetensors package."""
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

    parsed, start = read_file_header(path)
    assert start == data_start and parsed["model/a"]["dtype"] == "BF16" and parsed["model/a"]["shape"] == [1, 1, 32, 64]
    assert parsed["__metadata__"] == {"ttml.format": "ttml-dcp/1"}

    with safe_open(path, framework="pt") as f:
        assert set(f.keys()) == {"model/a", "model/b"}
        torch.testing.assert_close(f.get_tensor("model/a"), a, rtol=0, atol=0)
        torch.testing.assert_close(f.get_tensor("model/b"), b, rtol=0, atol=0)
        assert f.metadata() == {"ttml.format": "ttml-dcp/1"}


def test_metadata_is_stored_as_given_and_must_be_strings(expect_error):
    metadata = {"ttml.format": "ttml-dcp/1", "note": "[1, 2]"}
    header, _, _ = build_header([("x", "BF16", (1, 1, 1, 2), 4)], metadata)
    assert json.loads(header[8:])["__metadata__"] == metadata
    with expect_error(ValueError, "metadata 'steps' must be a string, not int"):
        build_header([], {"steps": 3})
