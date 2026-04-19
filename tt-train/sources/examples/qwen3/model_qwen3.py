# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Qwen3 HF finetuning wrapper.

Provides ``Qwen3ForCausalLM`` — a thin wrapper around ``ttml.models.qwen3``
building blocks arranged in an HF-compatible parameter tree for weight loading.

For training from scratch via ``train_nanogpt.py``, use
``ttml.models.qwen3.Qwen3`` directly.
"""

import torch
from tqdm import tqdm

import ttml
from ttml.modules import AbstractModuleBase, ModuleList, Parameter

# Re-export shared components so existing callers (model_qwen3_distributed,
# model_factory, etc.) continue to work with ``from model_qwen3 import ...``
from ttml.models.qwen3 import (  # noqa: F401
    Qwen3Config,
    Qwen3RMSNorm,
    Qwen3RopeScalingConfig,
    create_qwen3_config_from_hf,
)
from ttml.models.qwen3.autograd_ops import ConcatLastDim, RMSNormFunction  # noqa: F401
from ttml.models.qwen3.transformer import Qwen3Block
from ttml.models import memory_efficient_runner

from utils.tensor_utils import (
    torch_to_ttml,
    make_weight,
)
from utils.param_utils import (  # noqa: F401 — re-exported for callers
    unpermute_proj_rows,
    unpermute_norm_weights,
    build_weight_mapping_single,
)


def linear(x, weight, bias=None):
    return ttml.ops.linear.linear(x, weight, bias)


# =====================================================================
# Qwen3ForCausalLM  (HF-compatible parameter tree)
# =====================================================================


class _Qwen3Backbone(AbstractModuleBase):
    """Backbone that mirrors the HF ``Qwen3Model`` parameter tree:
    embed_tokens, layers[i], norm."""

    def __init__(self, config: Qwen3Config, rope_params) -> None:
        super().__init__()
        vocab_tiled = ((config.vocab_size + 31) // 32) * 32
        self.embed_tokens = Parameter(make_weight((1, 1, vocab_tiled, config.hidden_size)))
        self.layers = ModuleList([Qwen3Block(config, i, rope_params) for i in range(config.num_hidden_layers)])
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Qwen3ForCausalLM(AbstractModuleBase):
    """HF-compatible CausalLM using shared ttml building blocks.

    Parameter tree matches HuggingFace naming so that
    ``build_weight_mapping_single`` / ``load_weights_from_hf`` work unchanged.
    """

    def __init__(
        self,
        config: Qwen3Config,
        tie_word_embeddings: bool = False,
        track_memory: int = 0,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.create_name("Qwen3ForCausalLM")
        self.config = config
        self.tie_word_embeddings = tie_word_embeddings
        self.track_memory = track_memory
        self.use_checkpoint = use_checkpoint

        rope_scaling_params = ttml.ops.rope.RopeScalingParams()
        rs = config.rope_scaling
        if rs.scaling_factor != 0.0 and rs.original_context_length != 0:
            rope_scaling_params.scaling_factor = rs.scaling_factor
            rope_scaling_params.high_freq_factor = rs.high_freq_factor
            rope_scaling_params.low_freq_factor = rs.low_freq_factor
            rope_scaling_params.original_context_length = rs.original_context_length

        rope_params = ttml.ops.rope.build_rope_params(
            config.max_position_embeddings,
            config.head_dim,
            config.rope_theta,
            rope_scaling_params,
        )

        self.model = _Qwen3Backbone(config, rope_params)

        if tie_word_embeddings:
            self.lm_head_weight = None
        else:
            vocab_tiled = ((config.vocab_size + 31) // 32) * 32
            self.lm_head_weight = Parameter(make_weight((1, 1, vocab_tiled, config.hidden_size)))

    def _snapshot(self, x, fwd_label: str, bwd_label: str):
        if not self.track_memory:
            return x
        from utils.memory import memory_snapshot

        return memory_snapshot(x, fwd_label, bwd_label)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, **kwargs):
        hidden_states = ttml.ops.embedding.embedding(input_ids, self.model.embed_tokens.tensor)
        hidden_states = self._snapshot(hidden_states, "AFTER_EMBEDDING_FWD", "AFTER_EMBEDDING_BWD")

        position_offset = 0
        if past_key_values is not None:
            position_offset = past_key_values.get_seq_length()

        for i, layer in enumerate(self.model.layers):
            if self.use_checkpoint:
                hidden_states = memory_efficient_runner(
                    layer,
                    hidden_states,
                    attention_mask,
                    past_key_values,
                    position_offset,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask,
                    past_key_values,
                    position_offset,
                )
            if self.track_memory and (i + 1) % self.track_memory == 0:
                hidden_states = self._snapshot(
                    hidden_states,
                    f"AFTER_LAYER_{i}_FWD",
                    f"AFTER_LAYER_{i}_BWD",
                )

        hidden_states = self.model.norm(hidden_states)
        hidden_states = self._snapshot(hidden_states, "AFTER_NORM_FWD", "AFTER_NORM_BWD")

        if self.tie_word_embeddings:
            logits = linear(hidden_states, self.model.embed_tokens.tensor, None)
        else:
            logits = linear(hidden_states, self.lm_head_weight.tensor, None)

        logits = self._snapshot(logits, "AFTER_LM_HEAD_FWD", "AFTER_LM_HEAD_BWD")
        return logits


# =====================================================================
# Weight loading from HuggingFace
# =====================================================================


def load_weights_from_hf(
    ttml_model: Qwen3ForCausalLM,
    hf_state_dict: dict,
    config: Qwen3Config,
    tie_word_embeddings: bool = False,
    verbose: bool = False,
) -> None:
    """Load HF weights into single-device ttml model."""
    ttml_params = ttml_model.parameters()

    if verbose:
        print("\n  TTML parameter names:")
        for name in sorted(ttml_params.keys()):
            shape = ttml_params[name].shape()
            print(f"    {name}: {list(shape)}")

    any_key = next(iter(ttml_params))
    root_prefix = any_key.split("/")[0]

    mapping, transforms = build_weight_mapping_single(config, root_prefix, tie_word_embeddings)

    ttml_shapes = {name: list(ttml_params[name].shape()) for name in ttml_params}

    def _prepare_and_transfer(hf_name, ttml_name):
        """CPU prep + host-side conversion + device transfer (pipelined)."""
        if hf_name not in hf_state_dict:
            return None
        if ttml_name not in ttml_shapes:
            return None

        weight = hf_state_dict[hf_name].float()

        if hf_name in transforms:
            tr = transforms[hf_name]
            if tr[0] == "unpermute_proj":
                weight = unpermute_proj_rows(weight, num_heads=tr[1])
            elif tr[0] == "unpermute_norm":
                weight = unpermute_norm_weights(weight)

        ttml_shape = ttml_shapes[ttml_name]

        if weight.dim() == 2:
            rows, cols = weight.shape
            tgt_rows, tgt_cols = ttml_shape[2], ttml_shape[3]
            if rows != tgt_rows or cols != tgt_cols:
                padded = torch.zeros(tgt_rows, tgt_cols, dtype=weight.dtype)
                padded[: min(rows, tgt_rows), : min(cols, tgt_cols)] = weight[
                    : min(rows, tgt_rows), : min(cols, tgt_cols)
                ]
                weight = padded
            weight = weight.unsqueeze(0).unsqueeze(0)
        elif weight.dim() == 1:
            dim = weight.shape[0]
            tgt_dim = ttml_shape[-1]
            if dim != tgt_dim:
                padded = torch.zeros(tgt_dim, dtype=weight.dtype)
                padded[: min(dim, tgt_dim)] = weight[: min(dim, tgt_dim)]
                weight = padded
            weight = weight.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        return torch_to_ttml(weight)

    from concurrent.futures import ThreadPoolExecutor

    items = list(mapping.items())
    loaded = 0
    skipped = []

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            (hf_name, ttml_name, pool.submit(_prepare_and_transfer, hf_name, ttml_name)) for hf_name, ttml_name in items
        ]

        for hf_name, ttml_name, future in tqdm(
            futures,
            total=len(items),
            desc="  Loading weights",
            unit="w",
        ):
            new_tensor = future.result()
            if new_tensor is None:
                if ttml_name not in ttml_shapes:
                    print(f"  WARNING: ttml param '{ttml_name}' not found for HF '{hf_name}'")
                skipped.append(hf_name)
                continue
            ttml_params[ttml_name].assign(new_tensor)
            loaded += 1

    print(f"\n  Weight loading: {loaded} loaded, {len(skipped)} skipped")
    if skipped:
        print(f"  Skipped: {skipped}")
