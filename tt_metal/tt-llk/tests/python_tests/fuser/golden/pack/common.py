# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from helpers.golden_generators import PackGolden
from helpers.llk_params import L1Accumulation, PackerReluType, ReduceDimension


def prepare_tile(src, state, pack_node, operation, config):
    tile = state.dest.get(src)
    reduce_dim = operation.reduce_dim
    if reduce_dim in (ReduceDimension.Row, ReduceDimension.Scalar):
        tile[:, 1:] = 0
    if reduce_dim in (ReduceDimension.Column, ReduceDimension.Scalar):
        tile[1:, :] = 0
    if pack_node.pack_relu != PackerReluType.NoRelu:
        data_format = config.sentinel.golden_pack_src
        key = (id(pack_node), data_format)
        if key not in state.relu_configs:
            state.relu_configs[key] = PackGolden.generate_relu_config(
                pack_node.pack_relu, pack_node.relu_threshold, data_format
            )
        relu_config = state.relu_configs[key]
        tile = PackGolden.apply_relu(tile, relu_config, data_format)
    return tile


def append_tile(src, out, state, pack_node, operation, config):
    tile = prepare_tile(src, state, pack_node, operation, config)
    shape = pack_node.output.tile_shape
    packed = (
        tile.reshape(
            shape.num_faces_r_dim,
            shape.face_r_dim,
            shape.num_faces_c_dim,
            shape.face_c_dim,
        )
        .permute(0, 2, 1, 3)
        .flatten()
    )
    offset = out * shape.total_tile_size()
    if pack_node.pack_l1_accumulation == L1Accumulation.Yes:
        state.output.setdefault(offset, []).append(packed)
    else:
        state.output[offset] = [packed]
