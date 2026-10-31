# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Functional golden ids of the pack conversions of the Blackhole LLK matrix (llk_pack_space.llk_accepts) that test_pack does not run: the per-tile pack and the default
block pack, plain and with the zero ReLU, against the torch golden of the output format."""

import pytest
from conftest import skip_for_wormhole
from helpers.llk_params import DestSync, PackerReluType
from helpers.param_config import parametrize
from llk_pack_space import hw_undefined_reason, llkx_accs, llkx_formats
from test_pack import test_pack as run_pack


@skip_for_wormhole
@parametrize(
    formats=llkx_formats(),
    dest_acc=llkx_accs,
    input_dimensions=[[32, 32], [64, 64]],
    relu_type=[PackerReluType.NoRelu, PackerReluType.ZeroRelu],
    dest_sync=[DestSync.Half],
    dest_index=[0],
    pack_block=[False, 2],
)
def test_pack_llkx(formats, dest_acc, input_dimensions, relu_type, dest_sync, dest_index, pack_block):
    reason = hw_undefined_reason(formats, dest_acc)
    try:
        run_pack(formats, dest_acc, input_dimensions, relu_type, dest_sync, dest_index, pack_block=pack_block, allow_int32_mix=True)
    except AssertionError:
        if reason is None:
            raise
        pytest.xfail(reason)
