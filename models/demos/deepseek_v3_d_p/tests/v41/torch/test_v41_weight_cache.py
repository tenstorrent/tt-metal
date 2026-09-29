# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""CPU tests of the device-weight cache key (tests/v41/weight_cache.py, bead 8y7.18.1): what changes the key and
what must not."""

from pathlib import Path

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import oracle as orc
from models.demos.deepseek_v3_d_p.tests.v41 import weight_cache as wc
from models.demos.deepseek_v3_d_p.tests.v41.small_config import small_spec

LAYERS = (2, 3, 20, 21, 24)
MESH = (2, 4)


def _real(path="/snapshots/a"):
    return orc.real_spec(LAYERS, 2048, candidate_topk_blocks=96, checkpoint=Path(path))


def _synthetic():
    return orc.real_spec(LAYERS, 2048, candidate_topk_blocks=96)


def test_forward_source_edit_keeps_the_key(monkeypatch):
    before = [wc.weight_cache_dir(s, MESH) for s in (_real(), _synthetic(), small_spec(LAYERS, 512))]
    # the old key hashed the forward-pass sources through _reference_digest; the new one must not read them
    monkeypatch.setattr(orc, "_reference_digest", lambda synthetic=True: "edited")
    assert [wc.weight_cache_dir(s, MESH) for s in (_real(), _synthetic(), small_spec(LAYERS, 512))] == before
    forward_modules = {"model", "engram"}
    assert not any(f.__module__.rsplit(".", 1)[-1] in forward_modules for f in wc.CONVERSION_CODE)


def test_synthetic_init_change_changes_the_synthetic_key_only(monkeypatch, tmp_path):
    real, synthetic = wc.weight_cache_dir(_real(), MESH), wc.weight_cache_dir(_synthetic(), MESH)
    edited = tmp_path / "testing.py"
    edited.write_bytes(wc.SYNTHETIC_INIT_SOURCE.read_bytes() + b"\n# edited\n")
    monkeypatch.setattr(wc, "SYNTHETIC_INIT_SOURCE", edited)
    assert wc.weight_cache_dir(_real(), MESH) == real
    assert wc.weight_cache_dir(_synthetic(), MESH) != synthetic


def test_conversion_dtype_and_mesh_change_the_key(monkeypatch):
    spec = _real()
    base = wc.weight_cache_dir(spec, MESH)
    assert wc.weight_cache_dir(spec, MESH, ttnn.bfloat4_b) != base
    assert wc.weight_cache_dir(spec, (4, 2)) != base
    assert base.name.endswith("-mesh2x4") and wc.weight_cache_dir(spec, (4, 2)).name.endswith("-mesh4x2")
    monkeypatch.setattr(wc, "CONVERSION_CODE", wc.CONVERSION_CODE[1:])  # a conversion-code change
    assert wc.weight_cache_dir(spec, MESH) != base


def test_checkpoint_identity_is_the_pinned_revision_not_the_path():
    assert wc.weight_cache_dir(_real("/snapshots/a"), MESH) == wc.weight_cache_dir(_real("/other/place"), MESH)
    assert wc.source_key(_real()).endswith("@" + wc.checkpoint_weights.CHECKPOINT_REVISION)


def test_schedules_of_the_same_weights_share_a_directory():
    # per-layer files: the block stack and the transformer stack of the same weights hit one directory
    stack = orc.real_spec(LAYERS, 2048, candidate_topk_blocks=96, checkpoint=Path("/s"))
    transformer = orc.real_spec((0, 2, 3, 20, 21, 24), 2048, candidate_topk_blocks=96, checkpoint=Path("/s"))
    assert wc.weight_cache_dir(stack, MESH) == wc.weight_cache_dir(transformer, MESH)
    assert wc.weight_cache_dir(small_spec(LAYERS, 512), MESH) == wc.weight_cache_dir(small_spec((0, 2, 3), 256), MESH)
    assert wc.weight_cache_dir(small_spec(LAYERS, 512), MESH) != wc.weight_cache_dir(_synthetic(), MESH)  # dims


def test_positional_specs_are_rejected(expect_error):
    with expect_error(AssertionError, "V4.1 role"):
        wc.weight_cache_dir(orc.small_spec(41), MESH)
