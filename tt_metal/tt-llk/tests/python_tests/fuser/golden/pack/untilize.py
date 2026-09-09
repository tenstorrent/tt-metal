# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from .common import append_tile


def untilize_golden(call, state, pack_node, operation, config):
    for tile_call in call.tiles:
        append_tile(tile_call.dest, tile_call.out, state, pack_node, config)
