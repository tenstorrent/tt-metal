# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The upsampler only folds depth-to-space into conv3d for channel
blockings the conv3d writer supports. No device."""

from types import SimpleNamespace

import pytest

from ...models.vae import vae_ltx


def _conv(out_channels, c_out_block):
    return SimpleNamespace(out_channels=out_channels, conv_config=SimpleNamespace(C_out_block=c_out_block))


@pytest.mark.parametrize(
    ("out_channels", "c_out_block", "stride", "expected"),
    [
        (4096, 0, (2, 2, 2), True),  # one block holds all 8 groups of 512
        (4096, 256, (2, 2, 2), True),  # a group spans two blocks
        (4096, 1024, (2, 2, 2), True),  # two groups per block
        (1024, 0, (1, 2, 2), True),  # spatial only
        (512, 0, (2, 2, 2), True),  # 8 groups of 64 in one block
        (256, 0, (2, 2, 2), True),  # 8 groups of 32 in one block
        (384, 0, (2, 2, 2), False),  # group 48 is not tile aligned
        (1024, 384, (2, 2, 2), False),  # C_out_block does not divide the channels
        (3072, 256, (2, 2, 2), False),  # C_out_block 256 and group 384 do not nest
        (2048, 0, (4, 2, 2), False),  # 16 groups in one block, the writer takes at most 8
    ],
)
def test_can_fold_depth_to_space(out_channels, c_out_block, stride, expected):
    conv = _conv(out_channels, c_out_block)
    assert vae_ltx.LTXCausalConv3d.can_fold_depth_to_space(conv, stride) is expected
