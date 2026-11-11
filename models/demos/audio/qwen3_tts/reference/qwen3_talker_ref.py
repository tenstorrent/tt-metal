# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CPU reference for the Qwen3-TTS talker, the 28-layer decoder.

Wraps the vendored upstream stack (`reference/qwen/talker.py`) without touching its forward
pass: intermediates come from forward hooks, so the oracle runs exactly the code the model
ships.

Block boundary for the TTNN port:

    embeddings [1, T, 2048] --[28 decoder layers + final norm]--> hidden [1, 2048]

Embeddings rather than token ids, because the talker sums two streams. A text table of
151936 entries and a codec table of 3072 both project to width 2048, and the model adds
them at every position, padding whichever stream has nothing to say. Building that prompt
is a separate job; this block starts once it exists.

Shape, from `talker_config` and identical on the Base and CustomVoice checkpoints:

    28 layers, hidden 2048, 16 query heads over 8 key/value heads, head_dim 128,
    intermediate 6144, SiLU, RMS eps 1e-6, RoPE theta 1e6,
    MRoPE sections [24, 20, 20], interleaved.

Two details that separate this from a stock decoder. Attention carries Qwen3's QK-norm, an
RMSNorm over each head's 128 channels applied to queries and keys before the rotation. And
the rotation is multi-axis: the 64 rotary pairs split [24, 20, 20] across three position
axes rather than all advancing together.
"""

import functools

import torch

from models.demos.audio.qwen3_tts import weights
from models.demos.audio.qwen3_tts.reference.qwen import transformers_compat  # noqa: F401  (registers rope "default")
from models.demos.audio.qwen3_tts.reference.qwen.talker import Qwen3TTSTalkerConfig, Qwen3TTSTalkerModel

# Hooked module -> the name the PCC test compares against. `layers.N` is filled per layer.
FIXED_INTERMEDIATES = {"norm": "norm"}


class TalkerReference:
    """The upstream decoder stack, loaded from the checkpoint and frozen in eval mode."""

    def __init__(self, config=None, state=None, dtype=torch.float32, num_layers=None):
        cfg = dict(config or weights.talker_config())
        if num_layers is not None:  # a thin stack keeps component tests cheap
            cfg["num_hidden_layers"] = num_layers
        # transformers 4.57.3 created `pad_token_id` on every config; 5.12.1 only sets what
        # it is given, and the vendored stack reads the attribute. Supplied here so the
        # oracle stays a faithful copy.
        cfg.setdefault("pad_token_id", None)
        self.config = Qwen3TTSTalkerConfig(**cfg)
        self.model = Qwen3TTSTalkerModel(self.config)

        if state is None:
            state = weights.load_talker_state(dtype=dtype)
            if num_layers is not None:
                keep = lambda name: not name.startswith("layers.") or int(name.split(".")[1]) < num_layers
                state = {name: tensor for name, tensor in state.items() if keep(name)}
        self.model.load_state_dict(state, strict=True)
        self.model.to(dtype).eval()

    @property
    def num_layers(self):
        return self.config.num_hidden_layers

    @torch.inference_mode()
    def __call__(self, embeddings, position_ids=None, return_intermediates=False):
        """embeddings [1, T, 2048] -> hidden states [1, T, 2048].

        `position_ids` is [3, 1, T], one row per MRoPE axis. Passing None stacks the same
        ascending positions on all three, which is what a single-axis prompt reduces to.
        """
        if position_ids is None:
            position_ids = default_position_ids(embeddings.shape[1])

        if not return_intermediates:
            return self.model(inputs_embeds=embeddings, position_ids=position_ids).last_hidden_state

        captured = {}
        handles = []

        def capture(name):
            def hook(_module, _args, output):
                tensor = output[0] if isinstance(output, tuple) else output
                captured[name] = tensor.detach().clone()

            return hook

        modules = dict(self.model.named_modules())
        for index in range(self.num_layers):
            handles.append(modules[f"layers.{index}"].register_forward_hook(capture(f"layers.{index}")))
        for module_name, key in FIXED_INTERMEDIATES.items():
            handles.append(modules[module_name].register_forward_hook(capture(key)))
        try:
            hidden = self.model(inputs_embeds=embeddings, position_ids=position_ids).last_hidden_state
        finally:
            for handle in handles:
                handle.remove()
        return hidden, captured


def default_position_ids(length, batch=1):
    """[3, batch, length], the same ascending positions on every MRoPE axis."""
    positions = torch.arange(length, dtype=torch.long).reshape(1, 1, length)
    return positions.expand(3, batch, length).contiguous()


@functools.lru_cache(maxsize=2)
def reference_model(dtype=torch.float32, num_layers=None):
    """One loaded reference per (dtype, depth); the checkpoint read is the expensive part."""
    return TalkerReference(dtype=dtype, num_layers=num_layers)
