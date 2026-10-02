# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Decode-side per-layer page-table row handling under vLLM hybrid kv-cache groups.

The plugin pads per-layer tables to max_num_seqs rows while each decode trace
binds the persistent buffers of its own bucket, so the refresh must slice to
the step's batch. The async-ahead merge must keep host tokens for the rows
whose state slot was prefilled since the last decode, mapped through
``slot_remap``."""

import torch

from models.demos.gemma4.tt.async_decode import prefilled_rows
from models.demos.gemma4.tt.generator_vllm import Gemma4ForCausalLM

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


def test_prefilled_rows_identity_remap():
    assert prefilled_rows({1}, None, 4) == {1}
    assert prefilled_rows({0, 1}, None, 4) == {0, 1}
    assert prefilled_rows(set(), None, 4) == set()
    assert prefilled_rows(None, None, 4) == set()


def test_prefilled_rows_follow_slot_remap():
    # Request A (slot 0) was decoding; B was prefilled into slot 1 and now
    # decodes at row 0, A at row 1: only row 0 is fresh.
    remap = torch.tensor([1, 0, 2, 3])
    assert prefilled_rows({1}, remap, 4) == {0}
    assert prefilled_rows({0}, remap, 4) == {1}


def test_prefilled_rows_rebase_global_slots_per_rank():
    # Rank 1 of a DP=2 run with host_b=4 owns global slots 4..7.
    remap = torch.tensor([1, 0, 2, 3])
    assert prefilled_rows({5}, remap, 4, slot_offset=4) == {0}
    assert prefilled_rows({1}, remap, 4, slot_offset=4) == set()
