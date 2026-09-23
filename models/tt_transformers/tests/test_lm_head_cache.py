# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from models.tt_transformers.tt.lm_head import _lm_head_cache_file_name


def test_galaxy_2d_lm_head_uses_distinct_cache_key():
    cache_root = Path("weights")
    common = {
        "dummy_weights": False,
        "num_splits": 1,
        "shard": 0,
        "width": 131072,
        "mode": 0,
        "mesh_shape": (8, 4),
    }

    flat = _lm_head_cache_file_name(cache_root, galaxy_2d=False, **common)
    galaxy_2d = _lm_head_cache_file_name(cache_root, galaxy_2d=True, **common)

    assert flat.name == "output_lm_head_1_split_shard_0_131072_mode_0"
    assert galaxy_2d.name == "output_lm_head_1_split_shard_0_131072_mode_0_galaxy_2d_8x4_v1"
    assert flat != galaxy_2d


def test_dummy_lm_head_does_not_use_weight_cache():
    assert (
        _lm_head_cache_file_name(
            Path("weights"),
            dummy_weights=True,
            num_splits=1,
            shard=0,
            width=131072,
            mode=0,
            galaxy_2d=True,
            mesh_shape=(8, 4),
        )
        is None
    )
