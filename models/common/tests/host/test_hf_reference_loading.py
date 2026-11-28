# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no device, no network): a reference that fits host RAM loads whole, exactly as before;
one that does not loads only the repeated blocks a test needs, with the checkpoint's own weights."""
import pytest
import torch
import transformers

from models.common import hf_reference_loading as h

CLASSES = ("AutoModelForCausalLM", "AutoModel")
LAYERS = 6


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    cfg = transformers.LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=LAYERS,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(cfg).to(torch.bfloat16)
    d = tmp_path_factory.mktemp("tiny_llama")
    model.save_pretrained(d)
    return str(d), model


@pytest.fixture(autouse=True)
def _fresh_cache():
    h._skeleton_cached.cache_clear()
    yield
    h._skeleton_cached.cache_clear()


def _ram(monkeypatch, nbytes):
    monkeypatch.setattr(h, "host_ram_available_bytes", lambda: nbytes)


def test_skeleton_has_no_weight_memory(tiny_model_dir):
    path, _ = tiny_model_dir
    sk = h.build_hf_skeleton(path, CLASSES)
    assert all(p.device.type == "meta" for p in sk.parameters())


def test_a_model_that_fits_loads_whole_exactly_as_before(tiny_model_dir, monkeypatch):
    path, _ = tiny_model_dir
    _ram(monkeypatch, 10**12)
    model, err, overrides = h.load_hf_reference(path, CLASSES, ["model.layers.2"])
    assert err is None and overrides == {}
    assert len(model.model.layers) == LAYERS


def test_a_model_that_does_not_fit_loads_only_the_needed_blocks_with_real_weights(tiny_model_dir, monkeypatch):
    path, full = tiny_model_dir
    _ram(monkeypatch, 1)  # nothing fits
    model, err, overrides = h.load_hf_reference(path, CLASSES, ["model.layers.2"])
    assert err is None
    assert overrides == {"num_hidden_layers": 3}, overrides
    assert len(model.model.layers) == 3
    for name, ref in full.model.layers[2].state_dict().items():
        assert torch.equal(model.model.layers[2].state_dict()[name], ref), name


def test_a_path_outside_the_repeated_blocks_keeps_one_block(tiny_model_dir, monkeypatch):
    path, _ = tiny_model_dir
    _ram(monkeypatch, 1)
    assert h.truncation_overrides(path, CLASSES, ["model.embed_tokens"]) == {"num_hidden_layers": 1}


def test_the_deepest_path_wins(tiny_model_dir, monkeypatch):
    path, _ = tiny_model_dir
    assert h.truncation_overrides(path, CLASSES, ["model.layers.1", "model.layers.4", "model.layers"]) == {
        "num_hidden_layers": 5
    }


def test_no_paths_means_whole_model_even_when_it_does_not_fit(tiny_model_dir, monkeypatch):
    path, _ = tiny_model_dir
    _ram(monkeypatch, 1)
    model, err, overrides = h.load_hf_reference(path, CLASSES)
    assert err is None and overrides == {} and len(model.model.layers) == LAYERS


def test_whole_model_refusal_is_up_front_and_says_why(tiny_model_dir, monkeypatch, expect_error):
    path, _ = tiny_model_dir
    _ram(monkeypatch, 1)
    with expect_error(MemoryError, "whole-model CPU reference"):
        h.require_whole_model_fits(path, CLASSES)
    _ram(monkeypatch, 10**12)
    h.require_whole_model_fits(path, CLASSES)


def test_unknown_ram_or_size_keeps_the_previous_behavior(tiny_model_dir, monkeypatch):
    path, _ = tiny_model_dir
    _ram(monkeypatch, 0)
    assert h.fits_in_host_ram(path, CLASSES)
    assert h.fits_in_host_ram("no/such-model-xyz", CLASSES)


def test_the_ram_fraction_is_config(tiny_model_dir, monkeypatch):
    path, _ = tiny_model_dir
    need = h.loaded_weight_bytes(path, CLASSES)
    _ram(monkeypatch, need * 2)
    monkeypatch.setenv(h.RAM_FRACTION_ENV, "0.4")
    assert not h.fits_in_host_ram(path, CLASSES)
    monkeypatch.setenv(h.RAM_FRACTION_ENV, "0.6")
    assert h.fits_in_host_ram(path, CLASSES)
