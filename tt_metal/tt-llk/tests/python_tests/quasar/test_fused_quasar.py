# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import yaml
from conftest import skip_for_blackhole, skip_for_coverage, skip_for_wormhole
from fuser.config_parser import FUSER_CONFIG_DIR, FuserConfigSchema

yaml_files = sorted(FUSER_CONFIG_DIR.glob("*.yaml"))
yaml_files += sorted((FUSER_CONFIG_DIR / "quasar").glob("*.yaml"))
test_names = [str(f.relative_to(FUSER_CONFIG_DIR).with_suffix("")) for f in yaml_files]


@skip_for_blackhole
@skip_for_wormhole
@skip_for_coverage
@pytest.mark.parametrize("test_name", test_names, ids=test_names)
def test_fuser(
    test_name,
    regenerate_cpp,
):
    config = FuserConfigSchema.load(test_name)
    config.global_config.regenerate_cpp = regenerate_cpp
    config.run_regular_test()


@skip_for_blackhole
@skip_for_wormhole
@skip_for_coverage
@pytest.mark.parametrize(
    "tile_dims", [(1, 32), (16, 32), (32, 16)], ids=["1x32", "16x32", "32x16"]
)
def test_fuser_eltwise_partial_tiles(tile_dims, regenerate_cpp):
    with (FUSER_CONFIG_DIR / "fpu_elwadd.yaml").open() as config_file:
        config_dict = yaml.safe_load(config_file)

    tile_rows, tile_cols = tile_dims
    for operand in config_dict["operands"]:
        operand["tile_dims"] = tile_dims
        operand["dims"] = (4 * tile_rows, 4 * tile_cols)
    # Multiple tiles per block and multiple blocks exercise both destination and L1 offsets.
    config_dict["operations"][0]["block_size"] = (2 * tile_rows, 2 * tile_cols)
    config_dict["loop_factor"] = 1

    schema = FuserConfigSchema.model_validate(config_dict)
    config = schema.to_fuser_config(f"fpu_elwadd_{tile_rows}x{tile_cols}")
    config.global_config.regenerate_cpp = regenerate_cpp
    config.run_regular_test()
