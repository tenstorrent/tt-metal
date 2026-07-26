# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.indexer import TtIndexer


class _Tensor:
    def __init__(self, name, shape):
        self.name = name
        self.shape = tuple(shape)


def _install_fake_tensor_ops(monkeypatch):
    calls = SimpleNamespace(score=[], topk=[], deallocate=[])

    def score(*args, **kwargs):
        calls.score.append(kwargs)
        return _Tensor("scores", (1, 1, args[0].shape[2], kwargs["kv_len"] - kwargs["kv_start"]))

    def topk(values, **kwargs):
        calls.topk.append(kwargs)
        shape = (*values.shape[:-1], kwargs["k"])
        return _Tensor("values", shape), _Tensor("indices", shape)

    def concat(tensors, dim, **_):
        shape = list(tensors[0].shape)
        shape[dim] = sum(t.shape[dim] for t in tensors)
        return _Tensor("concat", shape)

    monkeypatch.setattr(ttnn.experimental, "indexer_score_dsa_paged", score)
    monkeypatch.setattr(ttnn.experimental, "topk_large_values_indices", topk)
    monkeypatch.setattr(ttnn, "concat", concat)
    monkeypatch.setattr(ttnn, "deallocate", calls.deallocate.append)
    return calls


def _indexer(sp_factor, tp_factor):
    return SimpleNamespace(
        index_args=SimpleNamespace(index_topk=2048),
        sp_factor=sp_factor,
        tp_factor=tp_factor,
        sp_axis=0,
    )


@pytest.mark.parametrize(
    "sp_factor,tp_factor,q_rows,local_bundle_tokens",
    [(4, 1, 1280, 1280), (8, 4, 160, 640)],
)
def test_paged_indexer_reuses_score_and_topk(monkeypatch, sp_factor, tp_factor, q_rows, local_bundle_tokens):
    """Each query shard reads paged K remotely; no SP gather is orchestrated."""

    calls = _install_fake_tensor_ops(monkeypatch)
    indexer = _indexer(sp_factor, tp_factor)
    result = TtIndexer._paged_remote_topk(
        indexer,
        _Tensor("q", (1, 32, q_rows, 128)),
        _Tensor("k_pool", (84, 1, local_bundle_tokens, 128)),
        _Tensor("table", (4, 205)),
        _Tensor("weights", (1, 32, q_rows, 1)),
        layer_idx=7,
        num_layers=21,
        end_pos=11003,
        chunk_start_idx=10240,
        paged_sp_axis=0,
        query_seq_axis=None if tp_factor == 1 else 1,
        cache_slot=3,
        program_config=ttnn.IndexerScoreProgramConfig(q_chunk_size=64, k_chunk_size=224, head_group_size=0),
    )

    assert len(calls.score) == 3
    assert [(c["kv_start"], c["kv_len"]) for c in calls.score] == [
        (0, 5120),
        (5120, 10240),
        (10240, 15360),
    ]
    assert all(c["chunk_start_idx"] == 10240 for c in calls.score)
    expected_axes = [0] if tp_factor == 1 else [0, 1]
    assert all(c["paged_sp_axis"] == 0 and c["seq_shard_axes"] == expected_axes for c in calls.score)
    assert all(c["cache_slot"] == 3 and c["layer_idx"] == 7 and c["num_layers"] == 21 for c in calls.score)

    bundle_topk = [c for c in calls.topk if "index_offset" in c]
    assert [c["index_offset"] for c in bundle_topk] == [0, 5120, 10240]
    assert all(c["valid_length"] == 5120 and c["k"] == 2048 for c in bundle_topk)
    assert result.shape == (1, 1, q_rows, 2048)


def test_sp4_paged_indexer_short_prefix_keeps_fixed_width(monkeypatch):
    """A short prefix visits one fixed-width bundle; causality yields sentinels."""

    calls = _install_fake_tensor_ops(monkeypatch)
    indexer = _indexer(4, 1)
    result = TtIndexer._paged_remote_topk(
        indexer,
        _Tensor("q", (1, 32, 1280, 128)),
        _Tensor("k_pool", (84, 1, 1280, 128)),
        _Tensor("table", (4, 205)),
        _Tensor("weights", (1, 32, 1280, 1)),
        layer_idx=0,
        num_layers=21,
        end_pos=37,
        chunk_start_idx=0,
        paged_sp_axis=0,
        query_seq_axis=None,
        cache_slot=0,
        program_config=ttnn.IndexerScoreProgramConfig(q_chunk_size=64, k_chunk_size=224, head_group_size=0),
    )

    assert len(calls.score) == 1
    assert (calls.score[0]["kv_start"], calls.score[0]["kv_len"]) == (0, 5120)
    assert calls.topk == [{"k": 2048, "valid_length": 5120, "index_offset": 0}]
    assert result.shape == (1, 1, 1280, 2048)
