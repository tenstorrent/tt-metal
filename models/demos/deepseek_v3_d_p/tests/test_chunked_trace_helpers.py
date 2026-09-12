# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only coverage for the chunked-prefill trace plumbing the standalone runner depends on:
resolve_trace_dir (descend into the vllm run-hash subdir), load_trace_token_ids, the format-aware
golden loader (_load_golden_kv_post) that reassembles both the DeepSeek single-file golden and the
Kimi vllm row-sharded layout into one [seq, 576] tensor, and the position window a golden implies.

Device-level chunked correctness (full transformer, both variants) is covered by the standalone
runner's KV-cache PCC; this only guards the trace-format handling so a layout change is caught in CI."""

from pathlib import Path

import pytest

from models.demos.common.prefill.adapter import get_adapter
from models.demos.common.prefill.runners.runner_utils import (
    load_trace_golden_span,
    load_trace_token_ids,
    resolve_trace_dir,
)
from models.demos.deepseek_v3_d_p.tt.runners.prefill_kv_validation import _load_golden_kv_post
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS

KVPE_DIM = 576  # kv_lora_rank (512) + qk_rope_head_dim (64)


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

    trace = _trace_or_skip("kimi_k2_6")
    short = _load_golden_kv_post(trace, 0, 4096)
    long = _load_golden_kv_post(trace, 0, 8192)
    assert torch.equal(short, long[:4096])


def _windowed_trace(tmp_path: Path, start: int, end: int) -> Path:
    """A minimal trace whose streams cover prompt positions [start, end): rows stored from 0, one
    kv_cache shard, and the capture_rows that say where those rows actually sit."""
    import json

    import torch
    from safetensors.torch import save_file

    rows = end - start
    trace = tmp_path / "windowed"
    (trace / "kv_cache" / "layer_0").mkdir(parents=True)
    with open(trace / "metadata.json", "w") as f:
        json.dump({"token_ids": list(range(rows)), "n_prompt_tokens": rows, "capture_rows": [start, end]}, f)
    payload = torch.arange(rows, dtype=torch.float32).unsqueeze(1).expand(rows, KVPE_DIM).contiguous()
    save_file(
        {"kv_post_transform_layer_0": payload},
        trace / "kv_cache" / "layer_0" / f"rows_{0:08d}_{rows:08d}.safetensors",
    )
    return trace


@pytest.mark.parametrize("variant_name", ["deepseek_v3_d_p", "kimi_k2_6", "kimi_k2_7"])
def test_golden_span_of_prefix_trace(variant_name):
    """A trace without capture_rows spans [0, n_prompt_tokens): row index is prompt position."""
    trace = _trace_or_skip(variant_name)
    assert load_trace_golden_span(trace) == (0, len(load_trace_token_ids(trace)))


def test_golden_span_of_windowed_trace(tmp_path):
    """capture_rows makes the span absolute, so the loader's rows and the run's positions differ."""
    trace = _windowed_trace(tmp_path, 1043456, 1048576)
    assert load_trace_golden_span(trace) == (1043456, 1048576)


def test_golden_windowed_rows_are_stored_from_zero(tmp_path):
    """The window's first position is the golden's row 0, not row 1043456."""
    import torch

    trace = _windowed_trace(tmp_path, 1043456, 1048576)
    g = _load_golden_kv_post(trace, 0, 5120)
    assert g.shape == (5120, KVPE_DIM), g.shape
    assert torch.equal(g[:, 0], torch.arange(5120, dtype=torch.float32))


@pytest.mark.parametrize("variant_name", ["deepseek_v3_d_p", "kimi_k2_7"])
def test_golden_row_start_slices_the_tail(variant_name):
    """row_start skips rows without shifting them: the tail load matches the full load's tail."""
    import torch

    trace = _trace_or_skip(variant_name)
    total_len, row_start = 10240, 5120
    tail = _load_golden_kv_post(trace, 0, total_len, row_start)
    full = _load_golden_kv_post(trace, 0, total_len)
    assert tail.shape == (total_len - row_start, KVPE_DIM), tail.shape
    assert torch.equal(tail, full[row_start:])
