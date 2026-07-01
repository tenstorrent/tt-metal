# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Shared pytest fixtures for the HunyuanImage-3.0 PCC tests.

import sys

import pytest

ROOT = "/home/iguser/ign-tt/tt-metal"
HUNYUAN = "/home/iguser/ign-tt/hunyan_instruct"
for _p in (ROOT, HUNYUAN):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@pytest.fixture(scope="session")
def device():
    """One TTNN device shared across the test session (open is expensive)."""
    import ttnn

    # l1_small_size is required by conv2d (halo/sliding-window scratch buffers);
    # harmless for the conv-free tests that share this fixture.
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    yield dev
    ttnn.close_device(dev)
