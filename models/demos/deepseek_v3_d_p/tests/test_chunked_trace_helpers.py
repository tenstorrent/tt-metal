# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only coverage for the chunked-prefill trace plumbing the standalone runner depends on:
resolve_trace_dir (descend into the vllm run-hash subdir), load_trace_token_ids, and the
format-aware golden loaders (_load_golden_kv_post, _load_golden_index_k) that reassemble both the
DeepSeek single-file golden and the Kimi vllm row-sharded layout into one [seq, dim] tensor, and
the windowed-golden path (load_trace_golden_span + offset reads) a last-chunk trace is scored on.

Device-level chunked correctness (full transformer, both variants) is covered by the standalone
runner's KV-cache PCC; this only guards the trace-format handling so a layout change is caught in CI."""

import json
from pathlib import Path

import pytest

from models.demos.common.prefill.adapter import get_adapter
from models.demos.common.prefill.runners.runner_utils import (
    load_trace_golden_span,
    load_trace_token_ids,
    resolve_trace_dir,
)
from models.demos.deepseek_v3_d_p.tt.runners.prefill_kv_validation import _load_golden_index_k, _load_golden_kv_post
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS

KVPE_DIM = 576  # kv_lora_rank (512) + qk_rope_head_dim (64)


def _trace_or_skip(variant_name):
    trace = get_adapter(variant_name).prefill_trace_default
    if not Path(trace).exists():
        pytest.skip(f"golden trace not staged: {trace}")
    return resolve_trace_dir(trace)


@pytest.mark.parametrize("variant_name", ["deepseek_v3_d_p", "kimi_k2_7"])
def test_resolve_trace_dir_has_metadata(variant_name):
    """resolve_trace_dir lands on a dir with metadata.json (descending the vllm hash subdir for Kimi)."""
    trace = _trace_or_skip(variant_name)
    assert (trace / "metadata.json").exists(), trace


@pytest.mark.parametrize("variant_name", ["deepseek_v3_d_p", "kimi_k2_7"])
def test_load_trace_token_ids(variant_name):
    """token_ids load, truncate to the requested length, and span >= one production chunk."""
    trace = _trace_or_skip(variant_name)
    assert len(load_trace_token_ids(trace, PREFILL_CHUNK_TOKENS)) == PREFILL_CHUNK_TOKENS
    assert len(load_trace_token_ids(trace)) >= PREFILL_CHUNK_TOKENS


@pytest.mark.parametrize("variant_name", ["deepseek_v3_d_p", "kimi_k2_7"])
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

    trace = _trace_or_skip("kimi_k2_7")
    short = _load_golden_kv_post(trace, 0, 4096)
    long = _load_golden_kv_post(trace, 0, 8192)
    assert torch.equal(short, long[:4096])


def _write_windowed_trace(root: Path, *, capture_rows, rows_per_shard=64, total_rows=256, dim=8):
    """A miniature vllm-layout trace whose kv_cache and dsa streams hold only `capture_rows`."""
    import torch
    from safetensors.torch import save_file

    (root / "kv_cache" / "layer_0").mkdir(parents=True)
    (root / "dsa" / "indexer_k_layer_0").mkdir(parents=True)
    for first in range(0, total_rows, rows_per_shard):
        last = first + rows_per_shard
        name = f"rows_{first:08d}_{last:08d}.safetensors"
        rows = torch.arange(first, last, dtype=torch.float32).unsqueeze(1).expand(rows_per_shard, dim).contiguous()
        save_file({"kv_post_transform_layer_0": rows}, str(root / "kv_cache" / "layer_0" / name))
        save_file({"indexer_k_layer_0": rows}, str(root / "dsa" / "indexer_k_layer_0" / name))
    (root / "metadata.json").write_text(
        json.dumps({"token_ids": list(range(total_rows)), "capture_rows": list(capture_rows)})
    )
    return root


def test_golden_span_defaults_to_prefix(tmp_path):
    """No capture_rows -> the streams start at position 0 and row index is the position."""
    (tmp_path / "metadata.json").write_text(json.dumps({"token_ids": list(range(4096))}))
    assert load_trace_golden_span(tmp_path) == (0, 4096)


def test_golden_span_reads_capture_rows(tmp_path):
    """capture_rows is the absolute prompt window the stored rows stand for."""
    trace = _write_windowed_trace(tmp_path, capture_rows=(250880, 256000))
    assert load_trace_golden_span(trace) == (250880, 256000)


@pytest.mark.parametrize("load", [_load_golden_kv_post, _load_golden_index_k])
def test_sharded_offset_read_crosses_shards(tmp_path, load):
    """A `start` offset into the row-sharded layout returns those rows, stitched across shards."""
    import torch

    trace = _write_windowed_trace(tmp_path, capture_rows=(0, 256))
    g = load(trace, 0, 96, start=100)
    assert g.shape == (96, 8), g.shape
    assert torch.equal(g[:, 0], torch.arange(100, 196, dtype=torch.float32))


def test_sharded_offset_read_past_the_end_raises(tmp_path, expect_error):
    """Asking for rows the trace does not hold fails instead of returning a short tensor."""
    trace = _write_windowed_trace(tmp_path, capture_rows=(0, 256))
    with expect_error(FileNotFoundError, "no shard covering rows"):
        _load_golden_kv_post(trace, 0, 32, start=1024)
