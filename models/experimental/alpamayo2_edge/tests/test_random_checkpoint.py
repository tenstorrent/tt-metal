# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""`make_random_checkpoint`: layout, seeding, init scale and golden-key separation of the random weights."""

import zlib
from dataclasses import replace

import pytest
import torch

from models.experimental.alpamayo2_edge.reference import golden_cache
from models.experimental.alpamayo2_edge.reference.config import load_config
from models.experimental.alpamayo2_edge.reference.weights import load_weights, make_random_checkpoint

# Small tensor groups, so the layout check stays cheap.
PREFIXES = ("vlm.model.projector.", "expert.action_in_proj.", "expert.action_out_proj.")


@pytest.fixture(scope="module")
def random_dir(model_dir, tmp_path_factory):
    return make_random_checkpoint(model_dir, tmp_path_factory.mktemp("random"), seed=0)


@pytest.mark.parametrize("prefix", PREFIXES)
def test_random_weights_match_checkpoint_layout(model_dir, random_dir, prefix):
    real = load_weights(model_dir, prefix)
    fake = load_weights(random_dir, prefix)
    assert sorted(fake) == sorted(real)
    for name, tensor in real.items():
        assert fake[name].dtype == tensor.dtype, name
        assert fake[name].shape == tensor.shape, name
        assert not torch.equal(fake[name], tensor), name


def test_random_weights_are_seeded(model_dir, random_dir, tmp_path):
    weights = load_weights(random_dir, "expert.action_out_proj.")
    # Pinned, so a seed that varies per process (Python's salted hash()) fails here instead of silently missing every
    # random golden in the cache.
    checksums = {name: zlib.crc32(t.view(torch.int16).numpy().tobytes()) for name, t in weights.items()}
    assert checksums == {"bias": 1232939531, "weight": 1842313128}

    other = make_random_checkpoint(model_dir, tmp_path, seed=1)
    for name, tensor in load_weights(other, "expert.action_out_proj.").items():
        assert not torch.equal(weights[name], tensor), name


def test_random_weights_init_scale(random_dir):
    weights = load_weights(random_dir, "vlm.model.projector.")
    fc1 = weights["linear_fc1.weight"].float()
    assert fc1.std().item() == pytest.approx(fc1.shape[1] ** -0.5, rel=0.02)
    assert weights["norm.weight"].float().mean().item() == pytest.approx(1.0, abs=0.02)
    assert weights["linear_fc1.bias"].float().std().item() == pytest.approx(0.02, rel=0.05)


def test_random_checkpoint_golden_key(cfg, model_dir, random_dir, tmp_path):
    """Same config as the checkpoint, a key of its own, and the same key in every session for the same seed."""
    assert replace(load_config(random_dir), model_dir=model_dir) == cfg
    key = golden_cache._checkpoint_hash(random_dir)
    assert key != golden_cache._checkpoint_hash(model_dir)
    assert golden_cache._checkpoint_hash(make_random_checkpoint(model_dir, tmp_path / "again", seed=0)) == key
    assert golden_cache._checkpoint_hash(make_random_checkpoint(model_dir, tmp_path / "seed1", seed=1)) != key
