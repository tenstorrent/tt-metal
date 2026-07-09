# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Zero-weights tests for ``LMHead.update`` (dummy weights).

Zeroing the LM head gives ``logits = 0`` -> uniform softmax. Checks (1) forward
logits are elementwise zero and (2) greedy decoding collapses to one token.
"""

from __future__ import annotations

import pytest
import torch

from _completer_utils import as_update_input, open_completer

PROMPT = "Explain a tensor in a paragraph."
MAX_NEW_TOKENS = 8
TEMPERATURE = 0.0  # greedy


@pytest.fixture(scope="module")
def completer():
    """Module-scoped: build once, zero the LM head once, run both tests."""
    with open_completer(dummy_weights=True) as c:
        model = c.models[0]
        V = model.lm_head.vocab_size
        H = model.lm_head.args.dim
        weight_hf = torch.zeros(V, H, dtype=torch.bfloat16)
        model.lm_head.update(weight=as_update_input(weight_hf, model.mesh_device))
        yield c


def _build_random_hidden_state(completer):
    """Construct a synthetic ``(1, 1, 32, dim)`` hidden state in the layout
    ``LMHead.forward`` expects in prefill mode.

    Must reshard to the LM-head-input memory config when it's sharded, else
    ``ttnn.linear`` throws ``RuntimeError: bad optional access`` on the missing
    shard spec.
    """
    import ttnn

    from models.tt_transformers.tt.common import Mode

    model = completer.models[0]
    dim = model.args.dim
    x = ttnn.from_torch(
        torch.randn(1, 1, 32, dim, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=model.mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(model.mesh_device),
    )

    lm_head_input_mem_cfg = model.args.get_lm_head_input_mem_config(Mode.PREFILL, None)
    if lm_head_input_mem_cfg.is_sharded():
        x = ttnn.interleaved_to_sharded(x, lm_head_input_mem_cfg)

    return x


def test_lm_head_logits_zero_when_weights_zero(completer):
    """Zero weights -> zero logits elementwise."""
    import ttnn

    model = completer.models[0]
    logits = ttnn.to_torch(model.lm_head.forward(_build_random_hidden_state(completer)))
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
