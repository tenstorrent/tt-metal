# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for VAE TILE layout contracts (conv1→snake2→conv2)."""

from __future__ import annotations

from models.experimental.ace_step_v1_5.ttnn_impl.vae.conv1d import _conv1_wants_tile_output


def test_conv1_return_sharded_falls_back_to_tile_output():
    assert _conv1_wants_tile_output(return_tile=False, return_sharded=True, use_sharded=False) is True
    assert _conv1_wants_tile_output(return_tile=True, return_sharded=False, use_sharded=False) is True
    assert _conv1_wants_tile_output(return_tile=False, return_sharded=False, use_sharded=False) is False
    # HEIGHT_SHARDED path returns before the tile-output branch.
    assert _conv1_wants_tile_output(return_tile=False, return_sharded=True, use_sharded=True) is False
