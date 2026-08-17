# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only regression coverage for the AttnRes tensorbin cache identities.

A stem that silently absorbs placement or dtype is the failure this file exists to catch:
two geometries would either collide on one file or miss each other's, and both read as a
cache that simply works. Nothing here touches a device.
"""

from types import SimpleNamespace

import ttnn
from models.demos.deepseek_v3_d_p.tests.attn_res.checkpoint_utils import attn_res_tensor_cache_path
from models.demos.deepseek_v3_d_p.tt.attn_res.weights import (
    AttnResWeights,
    _cache_artifact_names,
    _cache_stem,
    _serialized_path,
)
from models.demos.deepseek_v3_d_p.tt.runners.adapters.kimi_k3 import KimiK3Adapter

LAYERS = 4

# The resolver reads a shape and nothing else, so no device has to be opened for it.
MESH_2X4 = SimpleNamespace(shape=(2, 4))


def test_cache_stem_uses_only_the_caller_owned_namespace():
    assert _cache_stem("attn_res", "layers.7.mlp") == "attn_res.layers.7.mlp"
    assert _cache_stem("block_3.attn_res", "output") == "block_3.attn_res.output"


def test_serialized_path_separates_dtypes_and_names_the_layout():
    stem = "attn_res.output"
    assert _serialized_path(stem, ttnn.bfloat16).name == "attn_res.output_dtype_BFLOAT16_layout_TILE.tensorbin"
    assert _serialized_path(stem, ttnn.bfloat16) != _serialized_path(stem, ttnn.float32)


def test_artifact_names_cover_every_query_once():
    names = _cache_artifact_names(LAYERS)
    assert len(names) == len(set(names)) == 2 * LAYERS + 1
    assert names[-1] == "output"


def test_incomplete_cache_is_not_reported_complete(tmp_path):
    """An interrupted build leaves every file but one, which must not read as a hit."""
    names = _cache_artifact_names(LAYERS)
    for name in names[:-1]:
        _serialized_path(tmp_path / _cache_stem("attn_res", name), ttnn.bfloat16).touch()

    assert not AttnResWeights.check_cache_complete(tmp_path, "attn_res", num_layers=LAYERS)
    _serialized_path(tmp_path / _cache_stem("attn_res", names[-1]), ttnn.bfloat16).touch()
    assert AttnResWeights.check_cache_complete(tmp_path, "attn_res", num_layers=LAYERS)
    # Same files, other dtype: a hit here would hand the op weights it did not ask for.
    assert not AttnResWeights.check_cache_complete(tmp_path, "attn_res", num_layers=LAYERS, dtype=ttnn.float32)


def test_cache_root_takes_the_env_var_over_a_checkpoint(monkeypatch, tmp_path):
    """The published cache has to win, or a box holding both would rebuild what it shipped."""
    monkeypatch.delenv(KimiK3Adapter.ttnn_cache_env, raising=False)
    assert attn_res_tensor_cache_path(MESH_2X4) is None
    assert attn_res_tensor_cache_path(MESH_2X4, 1, tmp_path) == tmp_path / "ttnn_cache" / "sp2_tp4"

    monkeypatch.setenv(KimiK3Adapter.ttnn_cache_env, str(tmp_path / "published"))
    assert attn_res_tensor_cache_path(MESH_2X4, 1, tmp_path) == tmp_path / "published" / "sp2_tp4"
    # Same weights, other placement: distinct roots, because the shards differ.
    assert attn_res_tensor_cache_path(MESH_2X4, 0).name == "sp4_tp2"


def test_walk_order_skips_layer_zeros_pre_read():
    """186 reads out of 187 queries — the one arithmetic the whole schedule rests on."""
    pre = tuple(f"pre{idx}" for idx in range(LAYERS))
    post = tuple(f"post{idx}" for idx in range(LAYERS))
    weights = AttnResWeights(pre=pre, post=post, output="out", tensor_parallel_size=4, tensor_parallel_axis=1)

    order = weights.walk_order()
    assert len(order) == 2 * LAYERS
    assert order[0] == "post0" and order[-1] == "out"
    assert "pre0" not in order
