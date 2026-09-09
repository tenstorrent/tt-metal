# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from helpers.golden_generators import PackGolden
from helpers.llk_params import PackerReluType


def append_tile(src, out, state, pack_node, config):
    tile = state.dest.get(src)
    if pack_node.pack_relu != PackerReluType.NoRelu:
        data_format = config.sentinel.golden_pack_src
        key = (id(pack_node), data_format)
        if key not in state.relu_configs:
            state.relu_configs[key] = PackGolden.generate_relu_config(
                pack_node.pack_relu, pack_node.relu_threshold, data_format
            )
        relu_config = state.relu_configs[key]
        tile = PackGolden.apply_relu(tile, relu_config, data_format)
    state.output.setdefault(out, []).append(tile)
