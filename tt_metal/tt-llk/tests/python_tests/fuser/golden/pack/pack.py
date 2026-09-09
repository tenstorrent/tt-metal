# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from .common import append_tile


def pack_golden(call, state, pack_node, operation, config):
    append_tile(call.dest, call.out, state, pack_node, config)
