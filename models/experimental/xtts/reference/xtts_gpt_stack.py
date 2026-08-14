# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

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
    def __init__(self, config, num_layers=NUM_LAYERS):
        """Build GPT-2 blocks and final LayerNorm."""
        super().__init__()
        self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(num_layers)])
        self.ln_f = nn.LayerNorm(HIDDEN_SIZE, eps=LAYER_NORM_EPS)

    def forward(self, hidden_states):
        """Run causal GPT blocks then apply final LayerNorm."""
        mask = build_causal_mask(hidden_states.shape[1], hidden_states.dtype)
        for block in self.h:
            out = block(hidden_states, attention_mask=mask)
            hidden_states = out[0] if isinstance(out, tuple) else out
        return self.ln_f(hidden_states)


def reference_gpt_stack(state_dict, num_layers=NUM_LAYERS):
    """Load GPT stack block and ln_f weights from a checkpoint."""
    config = build_gpt2_config()
    module = XttsReferenceGptStack(config, num_layers=num_layers)

    for i in range(num_layers):
        prefix = f"gpt.gpt.h.{i}."
        block_state = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
        # GPT2Attention non-persistent causal-mask buffers are absent from the checkpoint.
        _, unexpected = module.h[i].load_state_dict(block_state, strict=False)
        truly_missing = set(module.h[i].state_dict().keys()) - set(block_state.keys())
        assert not unexpected, f"unexpected keys in block {i}: {unexpected}"
        assert not truly_missing, f"missing keys in block {i}: {sorted(truly_missing)}"

    module.ln_f.load_state_dict({"weight": state_dict["gpt.gpt.ln_f.weight"], "bias": state_dict["gpt.gpt.ln_f.bias"]})

    module.eval()
    return module
