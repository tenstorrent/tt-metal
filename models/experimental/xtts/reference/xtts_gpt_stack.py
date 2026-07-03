# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Reference (pure-PyTorch) implementation of the full XTTS-v2 GPT decoder stack.

XTTS-v2's autoregressive core (``gpt.gpt``) is a HuggingFace GPT-2 transformer:
30 identical ``GPT2Block`` decoder layers followed by a final LayerNorm
(``ln_f``). This module stacks the single-block reference
(:mod:`models.experimental.xtts.reference.xtts_gpt_block`) 30 times and applies
``ln_f`` — i.e. the decoder output *before* the text/mel heads.

Weights come straight from the upstream checkpoint at
https://huggingface.co/coqui/XTTS-v2 (``model.pth``); see ``xtts_gpt_block`` for
how the tensor state dict is extracted without depending on ``coqui-tts``.
"""

from torch import nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

from models.experimental.xtts.reference.xtts_gpt_block import (
    HIDDEN_SIZE,
    LAYER_NORM_EPS,
    NUM_LAYERS,
    build_causal_mask,
    build_gpt2_config,
)


class XttsReferenceGptStack(nn.Module):
    """The 30-layer XTTS GPT decoder: ``GPT2Block`` x N followed by ``ln_f``.

    ``forward`` takes hidden states ``[batch, seq, hidden]`` (already embedded,
    as the block-level ports operate on hidden states, not token ids) and
    returns the final layer-normed hidden states of the same shape.
    """

    def __init__(self, config, num_layers=NUM_LAYERS):
        super().__init__()
        self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(num_layers)])
        self.ln_f = nn.LayerNorm(HIDDEN_SIZE, eps=LAYER_NORM_EPS)

    def forward(self, hidden_states):
        # Causal mask — the XTTS GPT is an autoregressive decoder. Built once and
        # shared across all 30 blocks.
        mask = build_causal_mask(hidden_states.shape[1], hidden_states.dtype)
        for block in self.h:
            out = block(hidden_states, attention_mask=mask)
            hidden_states = out[0] if isinstance(out, tuple) else out
        return self.ln_f(hidden_states)


def reference_gpt_stack(state_dict, num_layers=NUM_LAYERS):
    """Build the full XTTS GPT decoder stack with real weights, in eval mode.

    Args:
        state_dict: full checkpoint state dict from
            :func:`models.experimental.xtts.reference.xtts_gpt_block.load_xtts_state_dict`.
        num_layers: number of repeating blocks to stack (30 for XTTS-v2).
    """
    config = build_gpt2_config()
    module = XttsReferenceGptStack(config, num_layers=num_layers)

    for i in range(num_layers):
        prefix = f"gpt.gpt.h.{i}."
        block_state = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
        # GPT2Attention registers non-persistent causal-mask buffers that are
        # absent from the checkpoint and from state_dict(); every real parameter
        # must still be present in the slice we loaded.
        _, unexpected = module.h[i].load_state_dict(block_state, strict=False)
        truly_missing = set(module.h[i].state_dict().keys()) - set(block_state.keys())
        assert not unexpected, f"unexpected keys in block {i}: {unexpected}"
        assert not truly_missing, f"missing keys in block {i}: {sorted(truly_missing)}"

    module.ln_f.load_state_dict({"weight": state_dict["gpt.gpt.ln_f.weight"], "bias": state_dict["gpt.gpt.ln_f.bias"]})

    module.eval()
    return module
