# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Construction-equivalence pytest test for ``LMHead.update``.

Real Llama-3.2-1B-Instruct weights (HF auth required). Round-trip on
the model's single ``LMHead``:

    1.  Generate greedily with the real, ``__init__``-loaded weights -> ``tokens_A``.
    2.  Snapshot the dram_sharded chunks back to a torch ``(V, H)`` tensor,
        i.e. the exact shape of ``state_dict["output.weight"]``.
    3.  Overwrite with a constant via ``LMHead.update``.
    4.  Generate again -> ``tokens_broken`` (sanity: must differ from
        ``tokens_A``, otherwise the overwrite was a no-op).
    5.  Restore via ``LMHead.update(snapshot)``.
    6.  Generate again -> ``tokens_B``.
    7.  Assert ``tokens_A == tokens_B`` byte-for-byte.

The snapshot path also doubles as a regression check on
``_permute_and_pad_output_weights`` / ``_build_dram_sharded_output_weights`` --
if those transformations are inconsistent with the inverse we apply
here, the restored model will not match the original.
"""

from __future__ import annotations

import pytest
import torch

from _completer_utils import build_completer, teardown_completer

PROMPT = "Explain a tensor in a paragraph."
MAX_NEW_TOKENS = 32
TEMPERATURE = 0.0  # greedy decoding -> deterministic, byte-comparable
OVERWRITE_VALUE = 0.0


@pytest.fixture(scope="module")
def completer():
    c = build_completer(dummy_weights=False)
    try:
        yield c
    finally:
        teardown_completer(c)


def _snapshot_lm_head(lm_head):
    """Read ``output_weights_dram_sharded`` back into a torch ``(vocab_size, dim)`` tensor.

    Inverse of ``__init__``:
        chunks -> concat along vocab axis -> slice off padding -> permute (dim, V) -> (V, dim)

    For a multi-device mesh, ``ttnn.to_torch`` of a
    ``ShardTensorToMesh(dim=-1)`` tensor returns the concatenated form
    across the mesh, so the column-concat below works regardless of
    ``num_devices``.
    """
    import ttnn

    chunk_torches = [ttnn.to_torch(chunk) for chunk in lm_head.output_weights_dram_sharded]
    permuted_padded = torch.cat(chunk_torches, dim=-1)  # (dim, padded_vocab_size)
    permuted = permuted_padded[:, : lm_head.vocab_size]  # (dim, vocab_size)
    return permuted.permute(1, 0).contiguous().to(torch.bfloat16)  # (vocab_size, dim)


def _generate(completer, prompt_ids):
    return completer.generate([prompt_ids], max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE)[0]


def test_lm_head_update_round_trip(completer):
    """Snapshot -> overwrite -> restore must reproduce the original tokens."""
    lm_head = completer.model.lm_head
    V, H = lm_head.vocab_size, lm_head.args.dim
    prompt_ids = completer.tokenizer.encode(PROMPT, add_special_tokens=True)

    tokens_A = _generate(completer, prompt_ids)
    snap = _snapshot_lm_head(lm_head)

    lm_head.update(torch.full((V, H), float(OVERWRITE_VALUE), dtype=torch.bfloat16))
    tokens_broken = _generate(completer, prompt_ids)
    assert tokens_broken != tokens_A, (
        f"overwriting LMHead with constant {OVERWRITE_VALUE} did not change generation; "
        "the overwrite step was a no-op, so the rest of the test is meaningless"
    )

    lm_head.update(snap)
    tokens_B = _generate(completer, prompt_ids)
    assert tokens_B == tokens_A, (
        "LMHead.update did not reproduce __init__-equivalent state: " f"tokens_A={tokens_A}, tokens_B={tokens_B}"
    )
