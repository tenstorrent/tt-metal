# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn


def test_open_device():
    """Simple unit test to test device open/close APIs"""
    device = ttnn.open_device(device_id=0)
    ttnn.close_device(device)


def test_manage_device():
    """Simple unit test to test device context manager APIs"""
    with ttnn.manage_device(0) as device:
        pass
