# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Construction-equivalence test for ``LMHead.update`` (real weights, HF auth).

Round-trip: generate -> snapshot dram_sharded chunks to a torch ``(V, H)``
tensor -> overwrite with a constant (must change output) -> restore via update
-> generate must match. Also regresses the on-device snapshot pad/transpose/slice
pipeline against the constructor's host-side path.
"""

from __future__ import annotations

import pytest
import torch

from _completer_utils import as_update_input, open_completer, to_torch_2d

PROMPT = "Explain a tensor in a paragraph."
MAX_NEW_TOKENS = 32
TEMPERATURE = 0.0  # greedy decoding -> deterministic, byte-comparable
OVERWRITE_VALUE = 0.0


@pytest.fixture(scope="module")
def completer():
    with open_completer(dummy_weights=False) as c:
        yield c


def _snapshot_lm_head_hf(lm_head):
    """Invert ``__init__`` (concat chunks -> slice padding -> transpose) to read
    ``output_weights_dram_sharded`` back into a torch ``(vocab_size, hidden_size)``
    tensor. The column-concat works for any ``num_devices``.
    """
    chunks_2d = [to_torch_2d(chunk) for chunk in lm_head.output_weights_dram_sharded]
    permuted_padded = torch.cat(chunks_2d, dim=-1)  # (dim, padded_vocab_size)
    permuted = permuted_padded[:, : lm_head.vocab_size]  # (dim, vocab_size)
    return permuted.transpose(0, 1).contiguous().to(torch.bfloat16)  # (vocab_size, dim)


def _generate(completer, prompt_ids):
    return completer.generate([prompt_ids], max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE)[0]


def test_lm_head_update_round_trip(completer):
    """Snapshot -> overwrite -> restore must reproduce the original tokens."""
    model = completer.models[0]
    lm_head = model.lm_head
    V = lm_head.vocab_size
    H = lm_head.args.dim
    prompt_ids = completer.tokenizer.encode(PROMPT, add_special_tokens=True)

    tokens_A = _generate(completer, prompt_ids)
    snap_hf = _snapshot_lm_head_hf(lm_head)

    overwrite_hf = torch.full((V, H), float(OVERWRITE_VALUE), dtype=torch.bfloat16)
    lm_head.update(weight=as_update_input(overwrite_hf, model.mesh_device))

    tokens_broken = _generate(completer, prompt_ids)
    assert tokens_broken != tokens_A, (
        f"overwriting LMHead with constant {OVERWRITE_VALUE} did not change generation; "
        "the overwrite step was a no-op, so the rest of the test is meaningless"
    )

    lm_head.update(weight=as_update_input(snap_hf, model.mesh_device))
    tokens_B = _generate(completer, prompt_ids)
    assert tokens_B == tokens_A, (
        "LMHead.update did not reproduce __init__-equivalent state: " f"tokens_A={tokens_A}, tokens_B={tokens_B}"
    )
