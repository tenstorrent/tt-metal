# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The adaptive gate must count REQUESTS, not the padded decode wire width.

The plugin pads a decode batch up to a wire bucket and pads the added rows'
positions with -1 ("Pad positions with -1 to indicate no position",
model_runner._prepare_model_inputs). A model that reads ``tokens.shape[0]``
therefore sees the BUCKET, not the number of scheduled requests.

That is what killed a BH Galaxy DP=4 server at 121k ISL: a solo request whose
bucket padded above 1 looked batched, so the model served plain baseline at
width 1 while the scheduler -- counting scheduled requests, and correct --
had reserved a full block, and the commit died on

    ValueError: Model output width violates output_tokens_per_step: 1 != 64

A P150x8 server pads solo decodes to 1, which is why the same prompt passed
there. Host-only: pure arithmetic over the input tensors.
"""

import pytest
import torch

# The gemma4 vLLM generator imports vllm at module scope (through
# tt_transformers.generator_vllm), so COLLECTING this file fails outright on a
# runner without vLLM installed -- which is the tt-metal unit-test job. Skip
# before the import rather than inside the tests: the failure is at import.
pytest.importorskip("vllm")
from models.demos.gemma4.tt.generator_vllm import Gemma4DFlashForCausalLM as DF

real_batch = DF._spec_real_batch


def _toks(n):
    return torch.zeros(n, 1, dtype=torch.int32)


def test_solo_request_padded_to_a_bucket_is_still_solo():
    """The Galaxy case: one request, wire width 32."""
    start = torch.cat([torch.tensor([121004]), torch.full((31,), -1)]).to(torch.int32)
    assert real_batch(_toks(32), start) == 1


def test_unpadded_solo_is_solo():
    assert real_batch(_toks(1), torch.tensor([512], dtype=torch.int32)) == 1


@pytest.mark.parametrize("live", [2, 5, 31, 32])
def test_partially_filled_batch_counts_live_rows(live):
    start = torch.cat([torch.arange(live), torch.full((32 - live,), -1)]).to(torch.int32)
    assert real_batch(_toks(32), start) == live


def test_full_batch_has_no_padding():
    assert real_batch(_toks(32), torch.arange(32, dtype=torch.int32)) == 32


def test_position_zero_is_a_real_row_not_padding():
    """-1 is the pad sentinel; 0 is a legitimate position and must not be
    mistaken for one."""
    start = torch.cat([torch.tensor([0, 0]), torch.full((30,), -1)]).to(torch.int32)
    assert real_batch(_toks(32), start) == 2


def test_two_dimensional_positions_are_flattened():
    start = torch.cat([torch.tensor([7]), torch.full((7,), -1)]).to(torch.int32).reshape(8, 1)
    assert real_batch(_toks(8), start) == 1


def test_missing_positions_fall_back_to_the_wire_width():
    """Nothing to count: keep the old behaviour rather than guessing solo."""
    assert real_batch(_toks(4), None) == 4


def test_all_rows_padded_never_reports_zero():
    """A zero would make the gate divide-by-zero downstream; the floor is 1."""
    assert real_batch(_toks(8), torch.full((8,), -1, dtype=torch.int32)) == 1
