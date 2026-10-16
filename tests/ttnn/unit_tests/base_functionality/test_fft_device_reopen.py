# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Lives outside tests/ttnn/unit_tests/operations/experimental/fft/ on purpose: that directory's conftest gives every test
# an open `device` fixture, and this test opens (and closes) the device itself.

import pytest
import torch

import ttnn


@pytest.mark.parametrize("N", [256, 4096])  # Stockham, two-pass (twiddle plan caches)
def test_fft_after_device_reopen(N):
    """The FFT plan caches are per device: a device opened after another was closed must get its own tables."""
    torch_in = torch.randn(N, dtype=torch.float32)
    expected = torch.fft.fft(torch_in)
    try:
        for _ in range(3):
            device = ttnn.open_device(device_id=0)
            try:
                tt_in = ttnn.from_torch(torch_in, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
                tt_re, tt_im = ttnn.experimental.fft(tt_in)
                got = torch.complex(
                    ttnn.to_torch(tt_re).reshape(-1).to(torch.float32),
                    ttnn.to_torch(tt_im).reshape(-1).to(torch.float32),
                )
                rel_err = (torch.linalg.norm(got - expected) / torch.linalg.norm(expected)).item()
                assert rel_err < 1e-3
            finally:
                ttnn.close_device(device)
    finally:
        # The plan caches own device tensors; release them so they do not outlive the last device.
        ttnn._ttnn.operations.experimental.clear_fft_device_caches()
