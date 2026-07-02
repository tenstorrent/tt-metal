# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn

from models.experimental.hunyuan_image_3_0.tt.cache import (
    cache_dir_is_set,
    transformer_cache_dir,
)


def test_transformer_cache_dir_key(monkeypatch):
    monkeypatch.setenv("TT_DIT_CACHE_DIR", "/tmp/TT_DIT_CACHE")
    path = transformer_cache_dir(
        model_name="hunyuan-image-3.0",
        mesh_shape=(2, 2),
        tp_axis=1,
        tp_factor=2,
        sp_axis=0,
        sp_factor=2,
        weight_dtype=ttnn.bfloat8_b,
        num_layers=32,
        bf16_layers={0, 1, 2, 3, 28, 29, 30, 31},
    )
    assert path is not None
    assert path.parts[-1] == "SP2a0_TP2a1_mesh2x2_L32_BFLOAT8_B_bf16_0-3_28-31"
    assert path.parent.name == "transformer"
    assert path.parent.parent.name == "hunyuan-image-3.0"


def test_cache_disabled_without_env(monkeypatch):
    monkeypatch.delenv("TT_DIT_CACHE_DIR", raising=False)
    assert not cache_dir_is_set()
    assert (
        transformer_cache_dir(
            model_name="hunyuan-image-3.0",
            mesh_shape=(2, 2),
            tp_axis=1,
            tp_factor=2,
            sp_axis=0,
            sp_factor=2,
            weight_dtype=ttnn.bfloat8_b,
            num_layers=4,
            bf16_layers=None,
        )
        is None
    )
