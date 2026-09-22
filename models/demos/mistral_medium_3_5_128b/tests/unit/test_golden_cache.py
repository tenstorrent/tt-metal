# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""D1 — the package-local golden cache obeys its two rules.

Host-only; no device, no hardware. The cache is only safe if (a) every output-affecting field is in
the key and (b) a miss is loud. Both are checked here rather than assumed, because a cache that
quietly serves a stale reference turns every downstream PCC number into fiction.
"""

import json
from dataclasses import replace

import pytest
import torch

from models.demos.mistral_medium_3_5_128b.reference.golden import load_golden, save_golden
from models.demos.mistral_medium_3_5_128b.reference.model_config import host_reduced_config
from models.demos.mistral_medium_3_5_128b.reference.regenerate import (
    compute_decoder_layer,
    decoder_layer_key,
    model_key,
    regenerate_all,
)


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Point the cache at a temp dir so tests never read or clobber the real one."""
    monkeypatch.setenv("MISTRAL_GOLDEN_CACHE", str(tmp_path))
    return tmp_path


@pytest.fixture(scope="module")
def cfg():
    return host_reduced_config()


def test_round_trip(cfg):
    key = decoder_layer_key(cfg)
    tensors = compute_decoder_layer(cfg)
    save_golden(key, tensors)
    loaded = load_golden(key)
    assert set(loaded) == set(tensors)
    for name, t in tensors.items():
        torch.testing.assert_close(loaded[name], t, rtol=0, atol=0)


def test_miss_raises_and_names_the_fix(cfg):
    with pytest.raises(FileNotFoundError) as exc:  # allow-pytest.raises: host-side
        load_golden(decoder_layer_key(cfg))
    assert "--regenerate" in str(exc.value), "a miss must tell the operator how to fix it"


@pytest.mark.parametrize(
    "field,value",
    [
        ("hidden_size", 1024),
        ("num_attention_heads", 16),
        ("num_key_value_heads", 4),
        ("head_dim", 128),
        ("intermediate_size", 2048),
        ("rms_norm_eps", 1e-6),
        ("rope_theta", 500000.0),
        ("rope_factor", 32.0),
        ("rope_beta_fast", 32.0),
        ("rope_beta_slow", 2.0),
        ("rope_original_max_position_embeddings", 8192),
        ("max_position_embeddings", 131072),
    ],
)
def test_every_output_affecting_config_field_moves_the_digest(cfg, field, value):
    """Change anything the numbers depend on and the key must change with it."""
    base = decoder_layer_key(cfg)
    perturbed = decoder_layer_key(cfg.reduced(**{field: value}))
    assert perturbed.digest() != base.digest(), f"{field} is missing from the cache key"


@pytest.mark.parametrize("field,value", [("weight_source", "random:7"), ("input_source", "randn:7"), ("n_tokens", 512)])
def test_source_fields_move_the_digest(cfg, field, value):
    base = decoder_layer_key(cfg)
    assert replace(base, **{field: value}).digest() != base.digest()


def test_digest_is_stable_across_processes(cfg):
    """Two constructions of the same key hash identically — the digest is content-derived, not
    dependent on dict ordering or object identity."""
    assert decoder_layer_key(cfg).digest() == decoder_layer_key(host_reduced_config()).digest()


def test_kind_separates_the_two_entries(cfg):
    """``decoder_layer`` and ``model`` describe the same config but hold different payloads.

    They share every shape field, so without ``kind`` in the key the whole-model entry and the
    single-layer entry would collide on one filename.
    """
    assert model_key(cfg).digest() != decoder_layer_key(cfg).digest()
    assert model_key(cfg).kind == "model" and decoder_layer_key(cfg).kind == "decoder_layer"


def test_model_key_tracks_depth(cfg):
    """The whole-model entry's numbers depend on the layer count, so the key must carry it.

    ``decoder_layer_key`` pins ``num_layers=1`` because one layer is all it computes; ``model_key``
    takes it from the config, and a 2-layer stack cached under an 8-layer key would be a stale
    reference that still loads.
    """
    assert model_key(cfg.reduced(num_hidden_layers=cfg.num_hidden_layers + 1)).digest() != model_key(cfg).digest()
    assert decoder_layer_key(cfg.reduced(num_hidden_layers=99)).digest() == decoder_layer_key(cfg).digest()


def test_regenerate_writes_every_entry(isolated_cache, cfg):
    """Both entries exist after a regeneration, under the filenames their keys name.

    This is the check that catches the realistic regression: someone adds a third cache kind and
    forgets to list it in ``regenerate_all``, so its only writer is a test and the operator command
    silently leaves it missing.
    """
    written = regenerate_all()
    assert {p.name for p in written} == {decoder_layer_key(cfg).filename(), model_key(cfg).filename()}
    for path in written:
        assert path.parent == isolated_cache, f"{path} escaped MISTRAL_GOLDEN_CACHE"
        assert path.exists()
    for key in (decoder_layer_key(cfg), model_key(cfg)):
        load_golden(key)  # raises if the key the writer used does not match the key a reader builds


def test_stored_key_is_verified_on_load(cfg, isolated_cache):
    """A file whose embedded key does not match the requested one is rejected, so a digest
    collision cannot silently return the wrong reference."""
    from safetensors.torch import save_file

    key = decoder_layer_key(cfg)
    wrong = replace(key, weight_source="random:999")
    save_file(
        {"output": torch.zeros(4)},
        str(isolated_cache / key.filename()),
        metadata={"golden_cache_key": json.dumps({**wrong.__dict__}, sort_keys=True)},
    )
    with pytest.raises(ValueError, match="collision"):  # allow-pytest.raises: host-side
        load_golden(key)
