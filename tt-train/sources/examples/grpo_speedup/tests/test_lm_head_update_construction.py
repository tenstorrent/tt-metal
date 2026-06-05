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
    5.  Restore via ``LMHead.update(weight=snapshot)``.
    6.  Generate again -> ``tokens_B``.
    7.  Assert ``tokens_A == tokens_B`` byte-for-byte.

The snapshot path also doubles as a regression check on
``LMHead._update_output_weights_dram_sharded``'s on-device
``pad + transpose + slice`` pipeline -- if it disagrees with the
constructor's host-side ``_permute_and_pad_output_weights`` +
``_build_dram_sharded_output_weights``, the restored model won't match
the original.
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
    """Read ``output_weights_dram_sharded`` back into a torch
    ``(vocab_size, hidden_size)`` tensor -- the exact HF shape of
    ``state_dict["lm_head.weight"]`` (or ``state_dict["output.weight"]``
    in the Meta/internal naming).

    Inverse of ``__init__``:
        chunks -> concat along vocab axis -> slice off padding
        -> transpose to (V, dim)

    For a multi-device mesh, ``ttnn.to_torch`` of a
    ``ShardTensorToMesh(dim=-1)`` tensor returns the concatenated form
    across the mesh, so the column-concat below works regardless of
    ``num_devices``.
    """
    chunks_2d = [to_torch_2d(chunk) for chunk in lm_head.output_weights_dram_sharded]
    permuted_padded = torch.cat(chunks_2d, dim=-1)  # (dim, padded_vocab_size)
    permuted = permuted_padded[:, : lm_head.vocab_size]  # (dim, vocab_size)
    return permuted.transpose(0, 1).contiguous().to(torch.bfloat16)  # (vocab_size, dim)


def _generate(completer, prompt_ids):
    return completer.generate([prompt_ids], max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE)[0]


def test_lm_head_update_round_trip(completer):
    """Snapshot -> overwrite -> restore must reproduce the original tokens."""
    lm_head = completer.model.lm_head
    V = lm_head.vocab_size
    H = lm_head.args.dim
    prompt_ids = completer.tokenizer.encode(PROMPT, add_special_tokens=True)

    tokens_A = _generate(completer, prompt_ids)
    snap_hf = _snapshot_lm_head_hf(lm_head)

    overwrite_hf = torch.full((V, H), float(OVERWRITE_VALUE), dtype=torch.bfloat16)
    lm_head.update(weight=as_update_input(overwrite_hf, completer.mesh_device))

    tokens_broken = _generate(completer, prompt_ids)
    assert tokens_broken != tokens_A, (
        f"overwriting LMHead with constant {OVERWRITE_VALUE} did not change generation; "
        "the overwrite step was a no-op, so the rest of the test is meaningless"
    )

    lm_head.update(weight=as_update_input(snap_hf, completer.mesh_device))
    tokens_B = _generate(completer, prompt_ids)
    assert tokens_B == tokens_A, (
        "LMHead.update did not reproduce __init__-equivalent state: " f"tokens_A={tokens_A}, tokens_B={tokens_B}"
    )
