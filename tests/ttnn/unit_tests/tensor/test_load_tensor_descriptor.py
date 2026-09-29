# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Descriptor-bound loads read the verified file and retain ordinary path behavior."""

import os

import pytest
import torch

import ttnn


def _dump(path, offset=0):
    values = torch.arange(64, dtype=torch.float32).reshape(2, 32) + offset
    ttnn.dump_tensor(path, ttnn.from_torch(values, dtype=ttnn.float32))
    return values


def test_descriptor_load_and_ordinary_path_control(tmp_path):
    path = tmp_path / "weights.tensorbin"
    expected = _dump(path)
    with path.open("rb") as verified:
        actual = ttnn.load_tensor(f"/proc/self/fd/{verified.fileno()}")
        torch.testing.assert_close(ttnn.to_torch(actual), expected, rtol=0, atol=0)
        assert not verified.closed
    torch.testing.assert_close(ttnn.to_torch(ttnn.load_tensor(path)), expected, rtol=0, atol=0)


def test_descriptor_load_to_device(tmp_path, device):
    path = tmp_path / "weights.tensorbin"
    expected = _dump(path)
    with path.open("rb") as verified:
        actual = ttnn.load_tensor(f"/proc/self/fd/{verified.fileno()}", device=device)
        torch.testing.assert_close(ttnn.to_torch(actual), expected, rtol=0, atol=0)
        assert not verified.closed


def test_descriptor_ignores_path_replacement(tmp_path):
    path = tmp_path / "weights.tensorbin"
    expected = _dump(path)
    with path.open("rb") as verified:
        path.rename(tmp_path / "verified.tensorbin")
        _dump(path, offset=1000)
        torch.testing.assert_close(
            ttnn.to_torch(ttnn.load_tensor(f"/proc/self/fd/{verified.fileno()}")), expected, rtol=0, atol=0
        )


@pytest.mark.parametrize("suffix", ["/", "/.", "/weights.tensorbin"])
def test_descriptor_alias_rejected(tmp_path, suffix, expect_error):
    path = tmp_path / "weights.tensorbin"
    _dump(path)
    with path.open("rb") as verified, expect_error(RuntimeError, message="canonical"):
        ttnn.load_tensor(f"/proc/self/fd/{verified.fileno()}{suffix}")


def test_descriptor_symlink_alias_rejected(tmp_path, expect_error):
    path = tmp_path / "weights.tensorbin"
    _dump(path)
    with path.open("rb") as verified:
        alias = tmp_path / "alias.tensorbin"
        alias.symlink_to(f"/proc/self/fd/{verified.fileno()}")
        with expect_error(RuntimeError, message="alias"):
            ttnn.load_tensor(alias)


def test_closed_descriptor_rejected(tmp_path, expect_error):
    path = tmp_path / "weights.tensorbin"
    _dump(path)
    with path.open("rb") as verified:
        fd = verified.fileno()
    with expect_error(RuntimeError, message="open descriptor"):
        ttnn.load_tensor(f"/proc/self/fd/{fd}")


def test_non_file_descriptor_rejected(expect_error):
    reader, writer = os.pipe()
    try:
        with expect_error(RuntimeError, message="regular file"):
            ttnn.load_tensor(f"/proc/self/fd/{reader}")
    finally:
        os.close(reader)
        os.close(writer)


@pytest.mark.parametrize("kind", ["directory", "fifo"])
@pytest.mark.timeout(5)
def test_non_regular_path_rejected_without_blocking(tmp_path, expect_error, kind):
    path = tmp_path / "invalid.tensorbin"
    if kind == "directory":
        path.mkdir()
    else:
        os.mkfifo(path)
    with expect_error(RuntimeError, message="regular file"):
        ttnn.load_tensor(path)


@pytest.mark.parametrize("relative", [False, True])
def test_ordinary_path_preserves_symlink_parent_traversal(tmp_path, monkeypatch, relative):
    local = tmp_path / "local"
    actual = tmp_path / "actual"
    local.mkdir()
    (actual / "child").mkdir(parents=True)
    local.joinpath("link").symlink_to(actual / "child", target_is_directory=True)
    _dump(local / "weights.tensorbin", offset=1000)
    expected = _dump(actual / "weights.tensorbin", offset=2000)
    requested = local / "link" / ".." / "weights.tensorbin"
    if relative:
        monkeypatch.chdir(tmp_path)
        requested = requested.relative_to(tmp_path)
    torch.testing.assert_close(ttnn.to_torch(ttnn.load_tensor(requested)), expected, rtol=0, atol=0)


def test_descriptor_alias_through_symlink_parent_is_rejected(tmp_path, expect_error):
    local = tmp_path / "local"
    actual = tmp_path / "actual"
    local.mkdir()
    (actual / "child").mkdir(parents=True)
    local.joinpath("link").symlink_to(actual / "child", target_is_directory=True)
    original = tmp_path / "original.tensorbin"
    _dump(original)
    _dump(local / "alias.tensorbin", offset=1000)
    with original.open("rb") as verified:
        actual.joinpath("alias.tensorbin").symlink_to(f"/proc/self/fd/{verified.fileno()}")
        with expect_error(RuntimeError, message="alias"):
            ttnn.load_tensor(local / "link" / ".." / "alias.tensorbin")
