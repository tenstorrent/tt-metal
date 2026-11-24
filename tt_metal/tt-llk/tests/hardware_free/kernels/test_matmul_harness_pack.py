# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The matmul perf kernels time production's block pack: their pack thread calls the compute API's pack_block through pack_api_shim.h, not a copy of llk_pack_block."""

import pathlib

TESTS = pathlib.Path(__file__).resolve().parents[2]


def test_matmul_harness_packs_through_production():
    shim = (TESTS / "helpers/include/pack_api_shim.h").read_text()
    assert (
        "llk_pack_block<ntiles, is_fp32_dest_acc_en, false, matmul_kt, arm_form>(start_tile_index, output);"
        in shim
    ), "pack_api_shim::pack_block no longer calls llk_pack_block"
    for name in ("matmul_test.cpp", "math_matmul_test.cpp"):
        src = (TESTS / "sources" / name).read_text()
        assert '#include "pack_api_shim.h"' in src, name
        assert (
            "pack_api_shim::pack_block<ntiles, is_fp32_dest_acc_en, matmul_pack_kt, MATMUL_PACK_ARM>("
            in src
        ), f"{name} no longer packs a constant-count block through production's pack_block"
        assert (
            "_llk_pack_block" not in src
        ), f"{name} calls a block pack of its own instead of production's"
