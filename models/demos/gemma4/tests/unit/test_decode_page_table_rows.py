# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Decode-side per-layer page-table row handling under vLLM hybrid kv-cache groups.

The plugin pads per-layer tables to max_num_seqs rows while each decode trace
binds the persistent buffers of its own bucket, so the refresh must slice to
the step's batch."""

import pytest
import torch

# generator_vllm imports vllm at module level; the unit tier has no vllm.
pytest.importorskip("vllm")

from models.demos.gemma4.tt.generator_vllm import Gemma4ForCausalLM  # noqa: E402

_slice = Gemma4ForCausalLM._slice_page_tables_rows


def test_slice_keeps_bucket_rows_and_shared_identity():
    sliding = torch.arange(32 * 8, dtype=torch.int32).reshape(32, 8)
    full = sliding + 1000
    tables = [sliding, sliding, full, sliding]
    out = _slice(tables, 4)
    assert all(tuple(t.shape) == (4, 8) for t in out)
    assert torch.equal(out[0], sliding[:4]) and torch.equal(out[2], full[:4])
    assert out[0] is out[1] is out[3] and out[2] is not out[0]


def test_slice_is_noop_when_tables_fit_or_rows_unknown():
    t = torch.zeros(4, 8, dtype=torch.int32)
    assert _slice([t], 4)[0] is t
    assert _slice([t], 32)[0] is t
    assert _slice([t], None)[0] is t
    assert _slice(None, 4) is None


def test_slice_passes_non_tensor_entries_through():
    t = torch.zeros(32, 8, dtype=torch.int32)
    out = _slice([None, t, "device-tensor-stand-in"], 2)
    assert out[0] is None and out[2] == "device-tensor-stand-in" and tuple(out[1].shape) == (2, 8)
