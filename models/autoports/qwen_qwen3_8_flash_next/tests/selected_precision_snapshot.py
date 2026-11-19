# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Inspect the selected default and public cache plan outside measured windows."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path


def snapshot(generator, model_dir):
    from models.autoports.qwen_qwen3_8_flash_next.tt.cache_budget import KNOWN_STATIC_BYTES, plan_shared_kv_cache
    from models.autoports.qwen_qwen3_8_flash_next.tt.precision import static_weight_delta

    path = Path(model_dir) / "doc/datatype_sweep/selected_precision_config.json"
    selected_bytes = path.read_bytes()
    selected = json.loads(selected_bytes)
    assert generator.model.precision == selected
    assert generator.capacity == generator.model.config.max_position_embeddings == selected["max_context"] == 262144
    assert generator.max_batch_size == 1
    delta = static_weight_delta(selected)
    plan = plan_shared_kv_cache(
        262144, 1, known_static_bytes=KNOWN_STATIC_BYTES + delta, kv_cache_dtype=selected["kv_cache_dtype"]
    )
    public_tokens = generator.get_max_tokens_all_users(max_model_len=262144, max_num_seqs=1)
    assert public_tokens == plan.plugin_max_tokens_all_users()
    cache = generator.cache_spec()
    assert cache["context"] == 262144 and cache["slots"] == 1 and cache["page_size"] == 32
    attention = [state.attention for state in generator.caches[0].layers if state.attention is not None]
    assert len(attention) == 12
    for state in attention:
        assert state.k_cache is not state.v_cache
        for tensor in (state.k_cache, state.v_cache):
            assert list(tensor.shape) == [8192, 1, 32, 256]
            assert str(tensor.dtype).lower().replace("datatype.", "") == selected["kv_cache_dtype"]
    return {
        "selected_path": str(path.resolve()),
        "selected_sha256": hashlib.sha256(selected_bytes).hexdigest(),
        "policy_equals_runtime": True,
        "static_weight_delta_bytes_per_rank": delta,
        "public_plan": asdict(plan),
        "public_max_tokens_all_users": public_tokens,
        "public_budget_semantics": "request context plus null page; worker separately adds output page",
        "cache_spec": cache,
        "distinct_kv_pairs": len(attention),
        "passed": True,
    }
