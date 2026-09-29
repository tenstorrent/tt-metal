# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Descriptor loads retain an open inode and preserve ordinary pathname behavior."""

import fcntl
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


@pytest.mark.parametrize("replacement", ["rename", "replace", "unlink"])
def test_descriptor_retains_original_inode(tmp_path, replacement):
    path = tmp_path / "weights.tensorbin"
    expected = _dump(path)
    staged = tmp_path / "staged.tensorbin"
    _dump(staged, offset=1000)
    with path.open("rb") as original:
        if replacement == "rename":
            path.rename(tmp_path / "original.tensorbin")
            staged.rename(path)
        elif replacement == "replace":
            os.replace(staged, path)
        else:
            path.unlink()
        loaded = ttnn.load_tensor(f"/proc/self/fd/{original.fileno()}")
    # Native mmap ownership must outlive both caller and wrapper descriptors.
    torch.testing.assert_close(ttnn.to_torch(loaded), expected, rtol=0, atol=0)


def test_descriptor_owns_duplicate_during_native_read(tmp_path, monkeypatch, expect_error):
    path = tmp_path / "weights.tensorbin"
    path.write_bytes(b"original")
    staged = tmp_path / "staged.tensorbin"
    staged.write_bytes(b"replacement")
    observed = []

    def native_read(owned_path, device):
        owned_fd = int(owned_path.rsplit("/", 1)[1])
        observed.append(owned_fd)
        assert owned_fd != original.fileno()
        assert fcntl.fcntl(owned_fd, fcntl.F_GETFD) & fcntl.FD_CLOEXEC
        os.replace(staged, path)
        assert os.pread(owned_fd, 8, 0) == b"original"
        return sentinel

    sentinel = object()
    monkeypatch.setattr(ttnn._ttnn.tensor, "load_tensor_flatbuffer", native_read)
    with path.open("rb") as original:
        assert ttnn.operations.core.load_tensor(f"/proc/self/fd/{original.fileno()}") is sentinel
        assert os.pread(original.fileno(), 8, 0) == b"original"
        with expect_error(OSError):
            os.fstat(observed[0])


def test_descriptor_duplicate_closes_on_native_error(tmp_path, monkeypatch, expect_error):
    path = tmp_path / "weights.tensorbin"
    path.write_bytes(b"invalid tensor")
    observed = []

    def native_read(owned_path, device):
        observed.append(int(owned_path.rsplit("/", 1)[1]))
        raise RuntimeError("invalid tensor")

    monkeypatch.setattr(ttnn._ttnn.tensor, "load_tensor_flatbuffer", native_read)
    with path.open("rb") as original:
        with expect_error(RuntimeError, message="invalid tensor"):
            ttnn.load_tensor(f"/proc/self/fd/{original.fileno()}")
        os.fstat(original.fileno())
        with expect_error(OSError):
            os.fstat(observed[0])


@pytest.mark.parametrize("suffix", ["/", "/.", "/weights.tensorbin"])
def test_descriptor_alias_rejected(tmp_path, suffix, expect_error):
    path = tmp_path / "weights.tensorbin"
    _dump(path)
    with path.open("rb") as verified, expect_error(RuntimeError, message="canonical"):
        ttnn.load_tensor(f"/proc/self/fd/{verified.fileno()}{suffix}")


def test_ordinary_symlink_to_descriptor_retains_existing_behavior(tmp_path):
    path = tmp_path / "weights.tensorbin"
    expected = _dump(path)
    with path.open("rb") as original:
        alias = tmp_path / "alias.tensorbin"
        alias.symlink_to(f"/proc/self/fd/{original.fileno()}")
        torch.testing.assert_close(ttnn.to_torch(ttnn.load_tensor(alias)), expected, rtol=0, atol=0)


def test_closed_descriptor_rejected(tmp_path, expect_error):
    path = tmp_path / "weights.tensorbin"
    _dump(path)
    with path.open("rb") as verified:
        fd = verified.fileno()
    with expect_error(RuntimeError, message="open descriptor"):
        ttnn.load_tensor(f"/proc/self/fd/{fd}")


def test_non_file_descriptor_rejected(expect_error):
    reader, writer = os.pipe()
    if reader == 0:
        positive_reader = fcntl.fcntl(reader, fcntl.F_DUPFD_CLOEXEC, 3)
        os.close(reader)
        reader = positive_reader
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
    with expect_error(RuntimeError, message="not a file"):
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
