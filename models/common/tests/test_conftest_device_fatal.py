# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

from models.common.tests.conftest import _is_device_fatal_error


class PcieHangError(Exception):
    pass


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("Read 0xffffffff over PCIe ID 13: the board should be reset."),
        PcieHangError("Read 0xffffffff over PCIe ID 0: the board should be reset."),
        RuntimeError("UMD_THROW PcieHangError"),
    ],
)
def test_device_fatal_hang_signatures(exc):
    assert _is_device_fatal_error(exc)


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("No TT devices detected on this system"),
        RuntimeError("Unable to query TT devices on this system"),
        RuntimeError("mesh_shape is required"),
        ValueError("unsupported mesh"),
    ],
)
def test_benign_device_errors_are_not_fatal(exc):
    assert not _is_device_fatal_error(exc)
