# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Uniform-distribution pytest test for ``LMHead.update``.

After zeroing every chunk of ``LMHead.output_weights_dram_sharded`` we
have ``logits = x @ 0 = 0``. Softmax over zero logits is uniform across
the entire vocab -- every token has the exact same probability.

Two assertions on the same zeroed state (sharing one ``completer``
fixture to avoid two heavy completer builds):

1. ``LMHead.forward`` produces elementwise-zero logits.
2. Greedy generation under those uniform logits collapses to a single
   repeated token (the deterministic tie-broken argmax of zero).

Uses ``dummy_weights=True`` -- no HF auth required.
"""

from __future__ import annotations

import pytest
import torch

from _completer_utils import as_update_input, open_completer

PROMPT = "Explain a tensor in a paragraph."
MAX_NEW_TOKENS = 8
TEMPERATURE = 0.0  # greedy -> uniform softmax collapses to a fixed argmax tie-break


@pytest.fixture(scope="module")
def completer():
    """Module-scoped: build once, zero the LM head once, run both tests."""
    with open_completer(dummy_weights=True) as c:
        V = c.model.lm_head.vocab_size
        H = c.model.lm_head.args.dim
        weight_hf = torch.zeros(V, H, dtype=torch.bfloat16)
        c.model.lm_head.update(weight=as_update_input(weight_hf, c.mesh_device))
        yield c


def _build_random_hidden_state(completer):
    """Construct a synthetic ``(1, 1, 32, dim)`` hidden state already in
    the DRAM-sharded layout that ``LMHead.forward`` expects in prefill mode.

    32 rows matches a single decode-mode tile and is the shape
    ``_apply_norm_and_lm_head`` itself documents
    (``models/tt_transformers/tt/model.py``). We mirror the resharding
    step that function does after the final RMSNorm: read the
    LM-head-input memory config from ``model_args`` and, if it's
    sharded, call ``interleaved_to_sharded`` before handing the tensor
    back. Without that, the program config inside ``LMHead.forward``'s
    ``ttnn.linear`` dereferences a shard spec the interleaved DRAM
    tensor doesn't carry, throwing ``RuntimeError: bad optional access``.
    """
    import ttnn

    from models.tt_transformers.tt.common import Mode

    dim = completer.model_args.dim
    x = ttnn.from_torch(
        torch.randn(1, 1, 32, dim, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=completer.mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(completer.mesh_device),
    )

    lm_head_input_mem_cfg = completer.model_args.get_lm_head_input_mem_config(Mode.PREFILL, None)
    if lm_head_input_mem_cfg.is_sharded():
        x = ttnn.interleaved_to_sharded(x, lm_head_input_mem_cfg)

    return x


def test_lm_head_logits_zero_when_weights_zero(completer):
    """Zero weights -> zero logits elementwise."""
    import ttnn

    logits = ttnn.to_torch(completer.model.lm_head.forward(_build_random_hidden_state(completer)))
    assert torch.equal(logits, torch.zeros_like(logits)), (
        "LMHead logits != 0 after zeroing weights: "
        f"max|logits|={float(logits.abs().max()):.6g}, "
        f"mean|logits|={float(logits.abs().mean()):.6g}"
    )


def test_lm_head_greedy_collapses_to_single_token(completer):
    """Uniform softmax + greedy decoding -> all generated tokens identical."""
    prompt_ids = completer.tokenizer.encode(PROMPT, add_special_tokens=True)
    tokens = completer.generate(
        [prompt_ids],
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
    )[0]
    assert (
        len(set(tokens)) <= 1
    ), f"greedy decoding under uniform logits did not collapse to a single token: tokens={tokens}"
