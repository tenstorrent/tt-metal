# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only coverage for the chunked-prefill trace plumbing the standalone runner depends on:
resolve_trace_dir (descend into the vllm run-hash subdir), load_trace_token_ids, and the
format-aware golden loader (_load_golden_kv_post) that reassembles both the DeepSeek single-file
golden and the Kimi vllm row-sharded layout into one [seq, 576] tensor.

Device-level chunked correctness (full transformer, both variants) is covered by the standalone
runner's KV-cache PCC; this only guards the trace-format handling so a layout change is caught in CI."""

from pathlib import Path

import pytest

from models.demos.common.prefill.adapter import get_adapter
from models.demos.common.prefill.runners.runner_utils import load_trace_token_ids, resolve_trace_dir
from models.demos.deepseek_v3_d_p.tt.runners.prefill_kv_validation import _load_golden_kv_post
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS

KVPE_DIM = 576  # kv_lora_rank (512) + qk_rope_head_dim (64)


# BLOCKED ON a row-sharded K2.7 golden trace. This module tests the trace LOADER, not a model, and
# the two Kimi goldens are different shapes on disk: K2.6's prefill_trace_default nests metadata.json
# in a run-hash subdir and splits each layer's KV into 14 row shards, while K2.7's is flat with one
# shard per layer. So K2.6 is the only fixture that reaches resolve_trace_dir's subdir-descent branch
# and the multi-shard torch.cat path in _load_golden_kv_post -- dropping it would delete that coverage,
# not migrate it. Concretely: resolve_trace_dir returns early when metadata.json is at the top level
# (runner_utils.py:167), so K2.7 and deepseek_v3_d_p never reach the descent branch below it, and
# K2.6 is the only staged Kimi golden that does. Every other trace in the cache is single-shard too.
# To retire this: re-record a K2.7 golden at chunk_rows < seq_len (K2.6's used 4096, giving 14
# shards), stage it, then repoint test_golden_row_shard_concat_is_contiguous and drop kimi_k2_6.
def _trace_or_skip(variant_name):
    trace = get_adapter(variant_name).prefill_trace_default
    if not Path(trace).exists():
        pytest.skip(f"golden trace not staged: {trace}")
    return resolve_trace_dir(trace)


@pytest.mark.parametrize("variant_name", ["deepseek_v3_d_p", "kimi_k2_6", "kimi_k2_7"])
def test_resolve_trace_dir_has_metadata(variant_name):
    """resolve_trace_dir lands on a dir with metadata.json (descending the vllm hash subdir for Kimi)."""
    trace = _trace_or_skip(variant_name)
    assert (trace / "metadata.json").exists(), trace


@pytest.mark.parametrize("variant_name", ["deepseek_v3_d_p", "kimi_k2_6", "kimi_k2_7"])
def test_load_trace_token_ids(variant_name):
    """token_ids load, truncate to the requested length, and span >= one production chunk."""
    trace = _trace_or_skip(variant_name)
    assert len(load_trace_token_ids(trace, PREFILL_CHUNK_TOKENS)) == PREFILL_CHUNK_TOKENS
    assert len(load_trace_token_ids(trace)) >= PREFILL_CHUNK_TOKENS


@pytest.mark.parametrize("variant_name", ["deepseek_v3_d_p", "kimi_k2_6", "kimi_k2_7"])
@pytest.mark.parametrize("layer", [0, 60])
def test_golden_kv_post_shape(variant_name, layer):
    """Both formats reassemble to [total_len, 576], finite — DS single-file and Kimi row-shards alike."""
    import torch

    trace = _trace_or_skip(variant_name)
    total_len = 10240
    g = _load_golden_kv_post(trace, layer, total_len)
    assert g.shape == (total_len, KVPE_DIM), g.shape
    assert g.dtype == torch.float32
    assert torch.isfinite(g).all()


def test_golden_row_shard_concat_is_contiguous():
    """Kimi row-shards concatenate in start-row order (no gaps/overlaps): a longer slice strictly
    extends the shorter one row-for-row."""
    import torch

    trace = _trace_or_skip("kimi_k2_6")  # row-sharded (14/layer); K2.7's golden is a single shard
    short = _load_golden_kv_post(trace, 0, 4096)
    long = _load_golden_kv_post(trace, 0, 8192)
    assert torch.equal(short, long[:4096])
