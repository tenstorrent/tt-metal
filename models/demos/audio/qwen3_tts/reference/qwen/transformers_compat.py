# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Two transformers APIs the vendored talker needs and 5.12.1 no longer provides as-is.

`qwen-tts` 0.1.1 pins transformers 4.57.3. Import this module before
`reference/qwen/talker.py` and that file stays a faithful copy, with the compatibility work
kept outside it.

**RoPE "default".** 4.57.3's `ROPE_INIT_FUNCTIONS` carried a `"default"` entry; 5.12.1 does
not (`dynamic`, `linear`, `llama3`, `longrope`, `proportional`, `yarn` remain) and moved
that path into a config mixin. The talker's `rope_scaling` names `"default"`, so its rotary
embedding cannot build without it. `_compute_default_rope_parameters` below is copied
verbatim from 4.57.3, the version the checkpoint was released against, minus its docstring.
Taking the original rather than rewriting the formula keeps the rotation the model was
trained with, including how it derives `dim` from `head_dim` and `partial_rotary_factor`.

**The mask builders.** `create_causal_mask` and `create_sliding_window_causal_mask` both
changed signature: `input_embeds` became `inputs_embeds`, and `cache_position` is gone, with
5.12.1 deriving the offset from `past_key_values` and `position_ids`. The wrapper below
renames the first and drops the second. Dropping it is sound for the prefill these references
run, where there is no cache and positions are explicit; anything relying on a populated
cache needs checking against upstream before it is trusted.
"""

import torch
import transformers.masking_utils as masking_utils
import transformers.utils.generic as generic_utils
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

REGISTERED_KEY = "default"


def _compute_default_rope_parameters(config=None, device=None, seq_len=None, layer_type=None):
    """Inverse frequencies for the original RoPE, from transformers 4.57.3.

    `layer_type` is accepted and ignored: 5.12.1 passes it to every registry entry, and
    4.57.3 had no such parameter.
    """
    base = config.rope_theta
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)

    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, attention_factor


def _mask_builder_4573(build_mask):
    """Accept a 4.57.3 call and forward it to the 5.12.1 function."""

    def wrapper(*args, **kwargs):
        if "input_embeds" in kwargs:
            kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
        kwargs.pop("cache_position", None)
        return build_mask(*args, **kwargs)

    wrapper.__wrapped__ = build_mask
    return wrapper


def _check_model_inputs_4573(check_model_inputs):
    """Accept a 4.57.3 factory call and return the 5.12.1 decorator.

    4.57.3 spelled it `check_model_inputs(tie_last_hidden_states=True)`, a factory that
    returned the decorator. 5.12.1 spells it `check_model_inputs(func)`, the decorator
    itself. The vendored codec writes `@check_model_inputs()`, which under 5.12.1 raises
    for a missing argument. `tie_last_hidden_states` has no counterpart in 5.12.1 and is
    dropped; the cross-check against the genuine package is what says that costs nothing.
    """

    def wrapper(*args, **kwargs):
        if args and callable(args[0]):
            return check_model_inputs(args[0])
        return check_model_inputs

    wrapper.__wrapped__ = check_model_inputs
    return wrapper


def register():
    """Put the 4.57.3 behaviour back, leaving anything already adapted alone."""
    ROPE_INIT_FUNCTIONS.setdefault(REGISTERED_KEY, _compute_default_rope_parameters)
    for name in ("create_causal_mask", "create_sliding_window_causal_mask"):
        builder = getattr(masking_utils, name)
        if not hasattr(builder, "__wrapped__"):
            setattr(masking_utils, name, _mask_builder_4573(builder))
    if not hasattr(generic_utils.check_model_inputs, "__wrapped__"):
        generic_utils.check_model_inputs = _check_model_inputs_4573(generic_utils.check_model_inputs)


register()
