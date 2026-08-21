# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Locks the gating predicate for the indexer top-k TP-regather skip (`skip_tp_regather`).

Why this test exists: the skip is *value-preserving* (gather-then-partition is an identity over
the TP axis), so the full-model PCC/correctness matrix keeps passing even if the optimization is
accidentally disabled — the perf win would regress with nothing failing. `ttMLA._needs_head_to_seq_reshard`
is the single decision that gates the skip (and is passed straight into `TtIndexer(skip_tp_regather=...)`),
so pinning its truth table here catches a silent regression cheaply and without a device.

The rule: the skip fires exactly when the per-chip head shard is too thin for `sparse_sdpa`
(needs `H/tp >= 32` and `H/tp % 32 == 0`) — i.e. when `_sparse_mla` will transpose heads->seq and
re-split the indices over TP anyway. tp=1 and fat head shards never fire.
"""

from types import SimpleNamespace

import pytest

from models.demos.deepseek_v3_d_p.tt.mla import indexer as indexer_module
from models.demos.deepseek_v3_d_p.tt.mla import ttMLA
from models.demos.deepseek_v3_d_p.tt.mla.indexer import TtIndexer


class _MlaStub:
    """Minimal stand-in exposing only the two attributes the predicate reads, so the *real*
    `ttMLA._needs_head_to_seq_reshard` getter is exercised without constructing a device model."""

    def __init__(self, num_heads: int, tp_factor: int):
        self.num_heads = num_heads
        self.tp_factor = tp_factor


def _needs_reshard(num_heads: int, tp_factor: int) -> bool:
    return ttMLA._needs_head_to_seq_reshard.fget(_MlaStub(num_heads, tp_factor))


@pytest.mark.parametrize(
    "num_heads, tp_factor, expected_skip, note",
    [
        # Thin head shard -> skip fires (the models this PR targets).
        (64, 4, True, "GLM-5.1/5.2 & DeepSeek-V4: 64h/tp4 -> 16 < 32"),
        (64, 8, True, "GLM at tp=8: 64h/tp8 -> 8 < 32"),
        (96, 4, True, "96h/tp4 -> 24 < 32"),
        # H/tp >= 32 but not a multiple of 32 -> still too thin for sparse_sdpa -> skip fires.
        (160, 4, True, "160h/tp4 -> 40, 40 % 32 != 0"),
        # Fat head shard -> skip does NOT fire; the gathered S/sp contract is genuinely needed.
        (128, 4, False, "DeepSeek-V3.2: 128h/tp4 -> 32 (exactly, divisible)"),
        (128, 2, False, "128h/tp2 -> 64"),
        (64, 2, False, "64h/tp2 -> 32 (exactly, divisible)"),
        # tp=1: no TP axis to gather/re-split -> never fires, regardless of head count.
        (64, 1, False, "tp=1: no tensor-parallel axis"),
        (16, 1, False, "tp=1 with a thin model"),
    ],
)
def test_needs_head_to_seq_reshard_truth_table(num_heads, tp_factor, expected_skip, note):
    assert _needs_reshard(num_heads, tp_factor) is expected_skip, note


def test_skip_boundary_at_32_heads_per_chip():
    """The threshold is exactly 32 heads/chip: 31 fires, 32 does not (guards an off-by-one that
    would either disable the skip for GLM or wrongly enable it for a fat-head model)."""
    assert _needs_reshard(124, 4) is True  # 124h/tp4 -> 31 per chip -> too thin -> fires
    assert _needs_reshard(128, 4) is False  # 128h/tp4 -> 32 per chip -> fat enough -> no skip


def _skip_injected(num_heads, tp_factor, config_opt_in):
    # Mirrors the injection expression in mla.py (KEEP IN SYNC): the reshard predicate AND an
    # explicit per-model config opt-in (config.indexer_skip_tp_regather). Head-count scoping is
    # insufficient: Kimi-K2.6 is also 64-head and with the skip active its padded chunked
    # no-trace path returned PCC ~ 0 on the 8x4 blaze pipeline.
    return _needs_reshard(num_heads, tp_factor) and config_opt_in


@pytest.mark.parametrize(
    "num_heads, tp_factor, opt_in, expected_injected, note",
    [
        (64, 4, True, True, "GLM-5.x at tp=4: opted in + predicate fires"),
        (64, 8, True, True, "GLM-5.x at tp=8"),
        (64, 8, False, False, "Kimi-K2.6 (64h!) at tp=8: predicate fires, NOT opted in"),
        (96, 8, False, False, "Kimi-K3 96h: not opted in"),
        (128, 4, True, False, "hypothetical opt-in on a fat shard: predicate is False"),
    ],
)
def test_skip_injection_scope(num_heads, tp_factor, opt_in, expected_injected, note):
    assert _skip_injected(num_heads, tp_factor, opt_in) is expected_injected, note


def test_glm_configs_opt_in_and_others_do_not():
    """The runtime namespaces GLM builders produce carry the opt-in; Kimi-K2.6's does not
    (getattr default False in mla.py keeps every non-opted model on today's gather)."""
    from models.demos.deepseek_v3_d_p.reference.glm_5_1_config import glm_hf_config
    from models.demos.deepseek_v3_d_p.reference.glm_5_2_config import glm_5_2_hf_config

    assert getattr(glm_hf_config(), "indexer_skip_tp_regather", False) is True
    assert getattr(glm_5_2_hf_config(), "indexer_skip_tp_regather", False) is True
    try:
        from models.demos.deepseek_v3_d_p.reference.kimi_k2_6_config import kimi_k2_6_hf_config

        assert getattr(kimi_k2_6_hf_config(), "indexer_skip_tp_regather", False) is False
    except ImportError:
        pass  # builder name differs; the getattr default in mla.py is the guarantee


class _FakeTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)


class _FakeCCL:
    def get_indexer_ring_k_buffer(self, *, local_k, sp_axis):
        return _FakeTensor(local_k.shape)

    def get_and_cycle_ag_semaphore_handles(self, *, cluster_axis):
        return object()


def _run_direct_indexer_forward(monkeypatch, *, skip_tp_regather):
    """Execute the real TtIndexer.forward control flow with shape-only TTNN operators."""
    seq_len = 128  # S/sp: the local sequence slab supplied to one SP rank.
    tp_factor = 4
    index_n_heads = 32
    index_head_dim = 128
    calls = {"tp_all_gather": 0, "to_layout": 0}
    q_weight = object()
    weights_proj = object()

    indexer = object.__new__(TtIndexer)
    indexer.index_args = SimpleNamespace(index_n_heads=index_n_heads, index_head_dim=index_head_dim)
    indexer._is_index_compact = False
    indexer.sp_factor = 2
    indexer.tp_factor = tp_factor
    indexer.tp_axis = 1
    indexer.sp_axis = 0
    indexer._index_cache_layers = 1
    indexer._idx_wq_b = q_weight
    indexer._idx_wproj = weights_proj
    indexer.default_compute_kernel_config = None
    indexer.tt_ccl = _FakeCCL()
    indexer.sp_ccl_topology = None
    indexer.ccl_num_links = 1
    indexer.index_topk_capacity = 32
    indexer.skip_tp_regather = skip_tp_regather
    indexer.write_k = lambda *args, **kwargs: None
    indexer._bc_rope_pe = lambda tensor, *args, **kwargs: tensor
    indexer._tp_all_reduce_via_gather = lambda tensor: tensor

    def tp_all_gather(tensor, dim):
        calls["tp_all_gather"] += 1
        gathered_shape = list(tensor.shape)
        gathered_shape[dim] *= tp_factor
        return _FakeTensor(gathered_shape)

    indexer._tp_all_gather = tp_all_gather

    def linear(tensor, weight, **kwargs):
        output_width = index_n_heads * index_head_dim if weight is q_weight else index_n_heads
        return _FakeTensor((1, 1, tensor.shape[2], output_width))

    def create_qkv_heads(tensor, **kwargs):
        return _FakeTensor((1, index_n_heads, tensor.shape[2], index_head_dim)), None, None

    def mesh_partition(tensor, dim, cluster_axis):
        partitioned_shape = list(tensor.shape)
        partitioned_shape[dim] //= tp_factor
        return _FakeTensor(partitioned_shape)

    def permute(tensor, order):
        return _FakeTensor(tuple(tensor.shape[dim] for dim in order))

    def ring_indexer_score(q, *args, **kwargs):
        return _FakeTensor((1, 1, q.shape[2], kwargs["kv_len"]))

    def topk_large_indices(logits, *, k, valid_length):
        return _FakeTensor((1, 1, logits.shape[2], k))

    def to_layout(tensor, layout):
        calls["to_layout"] += 1
        return _FakeTensor(tensor.shape)

    monkeypatch.setattr(indexer_module.ttnn, "linear", linear)
    monkeypatch.setattr(indexer_module.ttnn.experimental, "nlp_create_qkv_heads", create_qkv_heads)
    monkeypatch.setattr(indexer_module.ttnn, "multiply", lambda tensor, scalar: tensor)
    monkeypatch.setattr(indexer_module.ttnn, "permute", permute)
    monkeypatch.setattr(indexer_module.ttnn, "mesh_partition", mesh_partition)
    monkeypatch.setattr(
        indexer_module.ttnn,
        "IndexerScoreProgramConfig",
        lambda **kwargs: SimpleNamespace(**kwargs),
        raising=False,
    )
    monkeypatch.setattr(indexer_module.ttnn.experimental, "ring_indexer_score_dsa", ring_indexer_score, raising=False)
    monkeypatch.setattr(indexer_module.ttnn.experimental, "topk_large_indices", topk_large_indices, raising=False)
    monkeypatch.setattr(indexer_module.ttnn, "to_layout", to_layout)
    monkeypatch.setattr(indexer_module.ttnn, "deallocate", lambda tensor: None)

    hidden_states = _FakeTensor((1, 1, seq_len, 7168))
    qr = _FakeTensor((1, 1, seq_len, 1536))
    index_kv_cache = _FakeTensor((1, 1, 1024, index_head_dim))
    output = indexer.forward(
        hidden_states,
        qr,
        seq_len,
        rope_tensors={},
        index_kv_cache=index_kv_cache,
    )
    return output, calls, seq_len, tp_factor


def test_thin_head_forward_skips_tp_all_gather(monkeypatch):
    output, calls, _, _ = _run_direct_indexer_forward(monkeypatch, skip_tp_regather=True)

    assert calls["tp_all_gather"] == 0
    assert calls["to_layout"] == 0
    assert output.shape[-1] == 32


def test_thin_head_forward_returns_tp_local_sequence_rows(monkeypatch):
    output, _, seq_len, tp_factor = _run_direct_indexer_forward(monkeypatch, skip_tp_regather=True)

    assert output.shape[2] == seq_len // tp_factor  # S/(sp*tp), since seq_len is S/sp.


def test_fat_head_forward_retains_gathered_sequence_contract(monkeypatch):
    assert _needs_reshard(128, 4) is False  # Fat-head negative: 32 heads/chip needs the gathered contract.
    output, calls, seq_len, _ = _run_direct_indexer_forward(monkeypatch, skip_tp_regather=False)

    assert calls["tp_all_gather"] == 1
    assert calls["to_layout"] == 2
    assert output.shape[2] == seq_len  # S/sp after regather.
