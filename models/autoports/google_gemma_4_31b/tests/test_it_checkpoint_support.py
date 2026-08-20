# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Instruction-tuned checkpoint support. Host-only: no device, no weight load."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from transformers import AutoConfig

from models.autoports.google_gemma_4_31b.tt.functional_decoder import _validate_target_config
from models.autoports.google_gemma_4_31b.tt.model import (
    HF_MODEL_ID,
    SUPPORTED_HF_MODEL_IDS,
    _resolve_checkpoint,
    resolve_eos_token_ids,
)

LOCAL = Path("/mnt/models/blaze/google")


def _skip_without(repo: str) -> Path:
    path = LOCAL / repo
    if not (path / "config.json").exists():
        pytest.skip(f"{repo} not available on this host")
    return path


def test_both_checkpoint_ids_are_supported():
    assert SUPPORTED_HF_MODEL_IDS == (HF_MODEL_ID, f"{HF_MODEL_ID}-it")


def test_local_paths_resolve_directly(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    assert _resolve_checkpoint(tmp_path) == tmp_path.resolve()


def test_the_it_id_resolves_from_its_own_cache_directory(tmp_path, monkeypatch):
    """Previously only the base id fell back to the HF cache; the -it id raised."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    for repo in SUPPORTED_HF_MODEL_IDS:
        cache = tmp_path / ".cache/huggingface/hub" / ("models--" + repo.replace("/", "--")) / "snapshots" / "abc123"
        cache.mkdir(parents=True)
        (cache / "config.json").write_text("{}", encoding="utf-8")
        assert _resolve_checkpoint(repo) == cache.resolve()


def test_an_unrelated_id_still_raises(expect_error):
    with expect_error(FileNotFoundError, "checkpoint is not local"):
        _resolve_checkpoint("google/gemma-4-12B-it")


@pytest.mark.parametrize("repo, expected", [("gemma-4-31B", {1}), ("gemma-4-31B-it", {1, 106, 50})])
def test_eos_ids_come_from_the_checkpoint_not_just_the_tokenizer(repo, expected):
    """-it declares [1, 106, 50]; tokenizer.eos_token_id alone reports only <eos>=1,
    so a single-value comparison would miss end-of-turn."""
    checkpoint = _skip_without(repo)
    declared = json.loads((checkpoint / "generation_config.json").read_text())["eos_token_id"]
    assert resolve_eos_token_ids(SimpleNamespace(eos_token_id=1), checkpoint) == expected
    assert set(declared if isinstance(declared, list) else [declared]) <= expected


@pytest.mark.parametrize("repo", ["gemma-4-31B", "gemma-4-31B-it"])
def test_the_decoder_contract_accepts_both_checkpoints(repo):
    """Both configs must satisfy the same shape contract, for a sliding and a full layer."""
    config = AutoConfig.from_pretrained(_skip_without(repo), trust_remote_code=True)
    kinds = {_validate_target_config(config, idx).layer_kind for idx in (0, 5)}
    assert kinds == {"sliding_attention", "full_attention"}


def _keys_and_shapes(checkpoint: Path, prefixes: tuple[str, ...]) -> dict[str, tuple[int, ...]]:
    """Weight names and shapes, read from safetensors metadata without materialising tensors."""
    from safetensors import safe_open

    out: dict[str, tuple[int, ...]] = {}
    for shard in sorted(checkpoint.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith(prefixes):
                    out[key] = tuple(handle.get_slice(key).get_shape())
    return out


def test_the_two_checkpoints_expose_identical_weight_names_and_shapes():
    """The implementation's weight mapping is checkpoint-independent only if the
    tensor inventory matches. Covers a sliding layer, a full-attention layer, the
    embedding and the final norm."""
    base = _skip_without("gemma-4-31B")
    it = _skip_without("gemma-4-31B-it")
    prefixes = (
        "model.language_model.layers.0.",
        "model.language_model.layers.5.",
        "model.language_model.embed_tokens.",
        "model.language_model.norm.",
    )
    base_map = _keys_and_shapes(base, prefixes)
    it_map = _keys_and_shapes(it, prefixes)
    assert base_map, "no matching weights found in the base checkpoint"
    assert set(base_map) == set(
        it_map
    ), f"only in base: {sorted(set(base_map) - set(it_map))}; only in -it: {sorted(set(it_map) - set(base_map))}"
    mismatched = {k: (base_map[k], it_map[k]) for k in base_map if base_map[k] != it_map[k]}
    assert not mismatched, f"shape mismatches: {mismatched}"
