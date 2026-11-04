# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Lightweight (CPU-only, no model download) regression tests for the
multimodal HF-key-remapping pipeline in load_checkpoints.py.
"""

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

import models.ttt_compat.tt.load_checkpoints as load_checkpoints
from models.ttt_compat.tt.load_checkpoints import (
    convert_hf_to_meta_mllama,
    load_hf_state_dict_filtered,
    load_hf_state_dict_for_layers,
    map_hf_to_meta_keys_mllama,
    split_hf_keys,
    standardize_hf_keys_multimodal,
)


def _make_mllama_config(num_hidden_layers=4, cross_attention_layers=None):
    """Build a minimal config object accepted by map_hf_to_meta_keys_mllama."""
    if cross_attention_layers is None:
        cross_attention_layers = [1]
    return SimpleNamespace(
        text_config=SimpleNamespace(
            num_hidden_layers=num_hidden_layers,
            cross_attention_layers=cross_attention_layers,
        )
    )


def _make_sample_mllama_state_dict():
    """Return a minimal HF-format state_dict that covers the projector keys
    plus the embed_tokens and lm_head keys required by map_hf_to_meta_keys_mllama."""
    t = torch.zeros(1)
    return {
        "model.multi_modal_projector.weight": t,
        "model.multi_modal_projector.bias": t,
        "model.vision_model.layernorm_pre.weight": t,
        "model.vision_model.layernorm_pre.bias": t,
        # Both must be present so standardize_hf_keys (called inside
        # standardize_hf_keys_multimodal) doesn't delete embed_tokens.
        "lm_head.weight": torch.zeros(16, 4),
        "model.embed_tokens.weight": torch.zeros(16, 4),
    }


class TestMllamaProjectorKeyRemap:
    """Ensure model.multi_modal_projector.* keys survive the two-stage
    multimodal pipeline and land as vision_model.vision_projection.*."""

    def test_projector_keys_after_full_pipeline(self):
        state_dict = _make_sample_mllama_state_dict()
        config = _make_mllama_config()

        state_dict = standardize_hf_keys_multimodal(state_dict)
        state_dict = split_hf_keys(state_dict)
        state_dict = map_hf_to_meta_keys_mllama(state_dict, config)

        assert "vision_model.vision_projection.weight" in state_dict
        assert "vision_model.vision_projection.bias" in state_dict
        assert not any("multi_modal_projector" in k for k in state_dict)

    def test_projector_keys_via_convert_hf_to_meta_mllama(self):
        """Standardize_hf_keys_multimodal() -> convert_hf_to_meta_mllama().
        Asserts model.multi_modal_projector.weight ends up as
        vision_model.vision_projection.weight."""
        state_dict = _make_sample_mllama_state_dict()
        config = _make_mllama_config()
        head_dim = 64

        state_dict = standardize_hf_keys_multimodal(state_dict)
        state_dict = convert_hf_to_meta_mllama(state_dict, head_dim, config)

        assert "vision_model.vision_projection.weight" in state_dict
        assert "vision_model.vision_projection.bias" in state_dict
        assert not any("multi_modal_projector" in k for k in state_dict)

    def test_projector_keys_without_standardize(self):
        """map_hf_to_meta_keys_mllama should also work when called directly
        with the original model.-prefixed keys (backward compat)."""
        t = torch.zeros(1)
        state_dict = {
            "model.multi_modal_projector.weight": t,
            "model.multi_modal_projector.bias": t,
            "model.embed_tokens.weight": torch.zeros(16, 4),
        }
        config = _make_mllama_config()

        state_dict = split_hf_keys(state_dict)
        state_dict = map_hf_to_meta_keys_mllama(state_dict, config)

        assert "vision_model.vision_projection.weight" in state_dict
        assert "vision_model.vision_projection.bias" in state_dict


_SHARD_A = "model-00001-of-00002.safetensors"
_SHARD_B = "model-00002-of-00002.safetensors"


def _layer_keys(i):
    return {
        f"model.layers.{i}.input_layernorm.weight": torch.full((2,), float(i)),
        f"model.layers.{i}.mlp.up_proj.weight": torch.full((2, 2), float(i)),
    }


def _write_fake_hf_checkpoint(ckpt_dir, sharded):
    """Write a tiny HF-layout text checkpoint: layers 0, 1 and 10, embeddings and the final norm.

    Layer 10 pins the numeric layer comparison (a prefix match on "model.layers.1" would leak it).
    In the sharded layout shard A holds the embeddings and layers 0 and 1, shard B holds layer 10
    and the final norm, so a one-layer read must not open shard B for the layers but does for the norm.
    """
    tensors = {
        "model.embed_tokens.weight": torch.arange(8.0).reshape(4, 2),
        "model.norm.weight": torch.ones(2),
    }
    for i in (0, 1, 10):
        tensors.update(_layer_keys(i))

    if not sharded:
        save_file(tensors, str(ckpt_dir / "model.safetensors"))
        return tensors

    shard_a = {
        k: v for k, v in tensors.items() if k.startswith(("model.embed_tokens.", "model.layers.0.", "model.layers.1."))
    }
    shard_b = {k: v for k, v in tensors.items() if k not in shard_a}
    assert set(shard_b) == {"model.norm.weight", *_layer_keys(10)}
    save_file(shard_a, str(ckpt_dir / _SHARD_A))
    save_file(shard_b, str(ckpt_dir / _SHARD_B))
    weight_map = {k: _SHARD_A for k in shard_a}
    weight_map.update({k: _SHARD_B for k in shard_b})
    (ckpt_dir / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    return tensors


def _assert_same_tensors(actual, expected_all, expected_keys):
    assert set(actual) == set(expected_keys)
    for key in expected_keys:
        assert torch.equal(actual[key], expected_all[key]), key


class TestLoadHfStateDictForLayers:
    """load_hf_state_dict_for_layers keeps decoder layers [0, n_layers) plus every non-layer weight."""

    @pytest.mark.parametrize("sharded", [True, False], ids=["sharded", "single_file"])
    def test_one_layer_keeps_layer_zero_and_non_layer_weights(self, tmp_path, sharded):
        tensors = _write_fake_hf_checkpoint(tmp_path, sharded)

        state_dict = load_hf_state_dict_for_layers(tmp_path, n_layers=1, local_files_only=True)

        expected = {"model.embed_tokens.weight", "model.norm.weight", *_layer_keys(0)}
        _assert_same_tensors(state_dict, tensors, expected)

    @pytest.mark.parametrize("sharded", [True, False], ids=["sharded", "single_file"])
    def test_layer_index_is_compared_numerically(self, tmp_path, sharded):
        tensors = _write_fake_hf_checkpoint(tmp_path, sharded)

        state_dict = load_hf_state_dict_for_layers(tmp_path, n_layers=2, local_files_only=True)

        expected = {"model.embed_tokens.weight", "model.norm.weight", *_layer_keys(0), *_layer_keys(1)}
        _assert_same_tensors(state_dict, tensors, expected)
        assert not any(k.startswith("model.layers.10.") for k in state_dict)

    def test_n_layers_at_or_above_checkpoint_returns_everything(self, tmp_path):
        tensors = _write_fake_hf_checkpoint(tmp_path, sharded=True)

        state_dict = load_hf_state_dict_for_layers(tmp_path, n_layers=11, local_files_only=True)

        _assert_same_tensors(state_dict, tensors, tensors.keys())

    def test_sharded_read_opens_only_the_shards_that_hold_kept_keys(self, tmp_path, monkeypatch):
        _write_fake_hf_checkpoint(tmp_path, sharded=True)
        opened = []
        real_safe_open = load_checkpoints.safetensors_safe_open

        def recording_safe_open(path, *args, **kwargs):
            opened.append(path.rsplit("/", 1)[-1])
            return real_safe_open(path, *args, **kwargs)

        monkeypatch.setattr(load_checkpoints, "safetensors_safe_open", recording_safe_open)

        # Layer 0 only: shard B is still needed for the final norm.
        load_hf_state_dict_for_layers(tmp_path, n_layers=1, local_files_only=True)
        assert sorted(opened) == [_SHARD_A, _SHARD_B]

        # A prefix filter whose keys all live in shard A must not touch shard B.
        opened.clear()
        load_hf_state_dict_filtered(tmp_path, ["model.layers.0.input_layernorm."], local_files_only=True)
        assert opened == [_SHARD_A]


class TestLoadHfStateDictFiltered:
    @pytest.mark.parametrize("sharded", [True, False], ids=["sharded", "single_file"])
    def test_returns_exactly_the_prefixed_keys(self, tmp_path, sharded):
        tensors = _write_fake_hf_checkpoint(tmp_path, sharded)

        state_dict = load_hf_state_dict_filtered(
            tmp_path, ["model.layers.0.input_layernorm.", "model.norm."], local_files_only=True
        )

        _assert_same_tensors(state_dict, tensors, {"model.layers.0.input_layernorm.weight", "model.norm.weight"})

    def test_empty_prefix_list_reads_nothing(self, tmp_path):
        _write_fake_hf_checkpoint(tmp_path, sharded=True)

        assert load_hf_state_dict_filtered(tmp_path, [], local_files_only=True) == {}


class TestCheckpointIndexContainment:
    """A weight_map entry must not pull a shard from outside the checkpoint directory (#59616)."""

    @staticmethod
    def _point_shard_b_at(ckpt_dir, target):
        index_path = ckpt_dir / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        index["weight_map"] = {k: (target if v == _SHARD_B else v) for k, v in index["weight_map"].items()}
        index_path.write_text(json.dumps(index))

    @pytest.fixture(params=["relative", "absolute"])
    def escaping_ckpt(self, request, tmp_path):
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()
        _write_fake_hf_checkpoint(ckpt_dir, sharded=True)
        outside = tmp_path / "outside.safetensors"
        (ckpt_dir / _SHARD_B).rename(outside)
        self._point_shard_b_at(ckpt_dir, "../outside.safetensors" if request.param == "relative" else str(outside))
        return ckpt_dir

    @pytest.mark.parametrize(
        "load",
        [
            lambda d: load_checkpoints.load_hf_state_dict(d),
            lambda d: load_hf_state_dict_filtered(d, ["model.norm."], local_files_only=True),
            lambda d: load_hf_state_dict_for_layers(d, n_layers=1, local_files_only=True),
            lambda d: load_checkpoints.check_checkpoint_index(d),
        ],
        ids=["load_hf_state_dict", "filtered", "for_layers", "check_checkpoint_index"],
    )
    def test_escaping_shard_is_rejected(self, escaping_ckpt, load, expect_error):
        with expect_error(ValueError, "outside the checkpoint directory"):
            load(escaping_ckpt)

    @pytest.mark.parametrize("method", ["load_state_dict", "reference_transformer", "reference_vision_transformer"])
    def test_model_args_rejects_index_before_from_pretrained(self, escaping_ckpt, method, expect_error):
        # transformers' from_pretrained follows the escaping weight_map entry itself, and load_state_dict
        # swallows a ValueError from the per-layer read into a from_pretrained fallback. The index check has
        # to run first and propagate, so from_pretrained must never be reached here.
        pytest.importorskip("ttnn")
        from models.ttt_compat.tt.model_config import ModelArgs

        class _FromPretrainedMustNotRun:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                raise AssertionError("from_pretrained reached with an escaping checkpoint index")

        # Bypass __init__ (it needs a mesh device); set only what these load paths read. n_layers below
        # full_model_n_layers sends load_state_dict through the per-layer read and its fallback.
        args = ModelArgs.__new__(ModelArgs)
        args.CKPT_DIR = str(escaping_ckpt)
        args.dummy_weights = False
        args.n_layers = 1
        args.full_model_n_layers = 11
        args.cache_hf_flag = False
        args.cached_hf_model = None
        args.trust_remote_code_hf = False
        args.is_multimodal = False
        args.force_text_only = False
        args.get_hf_model_cls = lambda: _FromPretrainedMustNotRun

        with expect_error(ValueError, "outside the checkpoint directory"):
            getattr(args, method)()

    def test_hf_cache_snapshot_symlinks_still_load(self, tmp_path):
        # HF cache snapshot files are symlinks into ../../blobs; containment must not resolve them.
        snapshot = tmp_path / "models--org--repo" / "snapshots" / "rev"
        blobs = tmp_path / "models--org--repo" / "blobs"
        build = tmp_path / "build"
        build.mkdir()
        tensors = _write_fake_hf_checkpoint(build, sharded=True)
        blobs.mkdir(parents=True)
        snapshot.mkdir(parents=True)
        for f in build.iterdir():
            f.rename(blobs / f.name)
            (snapshot / f.name).symlink_to(f"../../blobs/{f.name}")

        load_checkpoints.check_checkpoint_index(snapshot)
        state_dict = load_hf_state_dict_for_layers(snapshot, n_layers=11, local_files_only=True)

        _assert_same_tensors(state_dict, tensors, tensors.keys())
