# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only: a traced packed forward runs empty chunk positions in a scratch slot past the real users, so the GLM-5.3
adapter must allocate that slot in both its caches, and only then; the engine still addresses num_users slots."""

from types import SimpleNamespace

import pytest

from models.demos.common.prefill.adapter import PrefillRunParams
from models.demos.deepseek_v3_d_p.tt.runners.adapters.glm_5_3 import GLM53Adapter
from models.demos.deepseek_v3_d_p.tt.runners.kv_caches import kv_cache_slots
from models.demos.deepseek_v3_d_p.tt.tt_prefill_runtime import TtPrefillRuntimeConfig
from models.demos.deepseek_v3_d_p.utils import kv_cache_utils

NUM_USERS = 54


def _params(forward_chunks, use_trace):
    return PrefillRunParams(
        mesh_shape=(8, 4),
        num_layers=4,
        first_layer_idx=0,
        is_first_rank=True,
        is_last_rank=True,
        max_seq_len=4 * 5120,
        chunk_size=5120,
        num_users=NUM_USERS,
        capacity_factor=8,
        num_links=2,
        gate_mode_name="DEVICE_FP32",
        kv_only_last_layer=False,
        weight_cache_path=None,
        use_trace=use_trace,
        forward_chunks=forward_chunks,
    )


CASES = [(1, False, NUM_USERS), (1, True, NUM_USERS), (2, False, NUM_USERS), (2, True, NUM_USERS + 1)]


@pytest.mark.parametrize("forward_chunks, use_trace, slots", CASES)
def test_glm_caches_hold_the_runtime_slots(monkeypatch, forward_chunks, use_trace, slots):
    """Both the KVPE and the indexer-key cache get the runtime's cache_num_users: one extra slot exactly when a traced
    forward packs more than one chunk."""
    seen = {}
    monkeypatch.setattr(kv_cache_utils, "init_mla_kv_cache", lambda **kw: seen.setdefault("kvpe", kw["num_users"]))
    monkeypatch.setattr(kv_cache_utils, "init_kvpe_cache", lambda **kw: seen.setdefault("index", kw["num_users"]))
    hf_config = SimpleNamespace(index_head_dim=128, indexer_types=["full", "full", "shared", "full"])
    GLM53Adapter().allocate_kv_cache(mesh_device=None, hf_config=hf_config, params=_params(forward_chunks, use_trace))
    assert seen == {"kvpe": slots, "index": slots}
    runtime_slots = TtPrefillRuntimeConfig(
        num_layers=4,
        num_users=NUM_USERS,
        chunk_size=5120,
        max_seq_len=4 * 5120,
        forward_tokens_budget=forward_chunks * 5120,
        use_trace=use_trace,
    ).cache_num_users
    assert runtime_slots == slots == kv_cache_slots(NUM_USERS, forward_chunks=forward_chunks, use_trace=use_trace)


def test_an_adapter_that_does_not_pack_keeps_num_users(monkeypatch):
    """Without supports_packed_forward the runner runs a forward chunk by chunk, so no scratch slot."""
    adapter = GLM53Adapter()
    monkeypatch.setattr(adapter, "supports_packed_forward", False)
    assert adapter.kv_cache_slots(_params(2, True)) == NUM_USERS
