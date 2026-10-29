# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import MethodType, SimpleNamespace

from models.tt_transformers.tt import model_config
from models.tt_transformers.tt.lm_head import _lm_head_cache_file_name
from models.tt_transformers.tt.model_config import ModelArgs


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


def _args_stub(cache_dir, *, max_columns, prefetcher=None, num_devices=2, padded_vocab=166912):
    """Host-only stand-in for the ModelArgs attributes the LM head cache check reads."""
    stub = SimpleNamespace(
        dummy_weights=False,
        is_galaxy=False,
        prefetcher=prefetcher,
        num_devices=num_devices,
        cluster_shape=(1, num_devices),
        padded_vocab_size=padded_vocab,
        max_columns_per_device_lm_head=max_columns,
        weight_cache_path=lambda dtype: Path(cache_dir),
        _weight_cache_identity=lambda components=None: {},
    )
    for name in ("lm_head_split_sizes", "lm_head_cache_files_present", "weight_cache_is_complete"):
        setattr(stub, name, MethodType(getattr(ModelArgs, name), stub))
    return stub


def _touch_lm_head_files(stub, cache_dir):
    modes, vocab_shards = stub.lm_head_split_sizes()
    for mode, sizes in enumerate(modes):
        for i, split in enumerate(sizes):
            stem = _lm_head_cache_file_name(
                Path(cache_dir),
                dummy_weights=False,
                num_splits=len(sizes),
                shard=i,
                width=split * vocab_shards,
                mode=mode,
                galaxy_2d=False,
                mesh_shape=stub.cluster_shape,
            )
            (stem.parent / (stem.name + "_dtype_BFLOAT8_B.tensorbin")).touch()


def test_lm_head_split_sizes_match_lm_head():
    # 166912 / 2 = 83456 columns per device; 16032-wide splits -> 5 full + 3296 remaining, as in LMHead
    stub = _args_stub("unused", max_columns=16032)
    modes, vocab_shards = stub.lm_head_split_sizes()
    assert vocab_shards == 2
    assert modes == [[16032] * 5 + [3296]]
    # with a prefetcher the ring-matmul mode rounds the split width up to a tile multiple
    with_prefetcher = _args_stub("unused", max_columns=16030, prefetcher=object())
    modes, _ = with_prefetcher.lm_head_split_sizes()
    assert modes[0] == [16030] * 5 + [3306]
    assert modes[1] == [16032] * 5 + [3296]


def test_complete_marker_without_lm_head_files_cold_loads(tmp_path, monkeypatch):
    # The marker says "complete", but the LM head tensorbins for the current split width are missing
    # (e.g. the split width changed since the cache was written): report incomplete so the HF
    # state_dict is loaded instead of converting a dataless placeholder into the cache.
    monkeypatch.setattr(model_config, "_weight_cache_is_complete", lambda path, **identity: True)
    stub = _args_stub(tmp_path, max_columns=16032)
    assert not stub.lm_head_cache_files_present("bfp8")
    assert stub.weight_cache_is_complete("bfp8") is False

    _touch_lm_head_files(stub, tmp_path)
    assert stub.lm_head_cache_files_present("bfp8")
    assert stub.weight_cache_is_complete("bfp8") is True

    # Files written for another split width do not count.
    other = _args_stub(tmp_path, max_columns=15872)
    assert not other.lm_head_cache_files_present("bfp8")
    assert other.weight_cache_is_complete("bfp8") is False


def test_incomplete_marker_is_reported_regardless_of_lm_head_files(tmp_path, monkeypatch):
    monkeypatch.setattr(model_config, "_weight_cache_is_complete", lambda path, **identity: False)
    stub = _args_stub(tmp_path, max_columns=16032)
    _touch_lm_head_files(stub, tmp_path)
    assert stub.weight_cache_is_complete("bfp8") is False
