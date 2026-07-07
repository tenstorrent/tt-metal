# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""SmolLM3 text-encoder wrapper for the Bria FIBO pipeline.

Tokenizes a single prompt, runs it through the (sp1-validated) on-device SmolLM3 encoder, and
returns host tensors matching the diffusers ``pipeline_bria_fibo.py`` contract: ``prompt_embeds =
cat(hidden_states[-1], hidden_states[-2])`` plus the full list of hidden states (used by
``build_text_encoder_layers`` to feed the transformer's per-block caption injection).

No CFG concatenation happens here -- the pipeline calls ``encode_prompt`` once per branch
(positive / negative) and concatenates itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from transformers import AutoConfig, AutoTokenizer, SmolLM3ForCausalLM

import ttnn

from ...encoders.smollm3.config import SmolLM3Config
from ...encoders.smollm3.model_smollm3 import SmolLM3TextEncoder
from ...utils import tensor as tt_tensor

if TYPE_CHECKING:
    from ...parallel.config import EncoderParallelConfig
    from ...parallel.manager import CCLManager

# diffusers pipeline_bria_fibo.py `get_prompt_embeds`: empty prompts ("") are special-cased to a
# single begin-of-text token, since the tokenizer itself does not add a BOS and would otherwise
# tokenize "" to a 0-length sequence.
BOT_TOKEN_ID = 128000


def build_text_encoder_layers(all_hidden_states: list, num_blocks: int) -> list:
    """Stretch/trim SmolLM3's hidden-state list to the transformer's per-block count.

    Mirrors diffusers ``pipeline_bria_fibo.py`` (~L613-621): if there are fewer hidden states
    than transformer blocks, the last hidden state is repeated to fill the remainder; if there
    are more, the earliest ones are dropped (right-trim, keeping the deepest layers).
    """
    layers = list(all_hidden_states)
    n = len(layers)
    if n >= num_blocks:
        return layers[n - num_blocks :]
    return layers + [layers[-1]] * (num_blocks - n)


class SmolLM3TextEncoderWrapper:
    """Tokenizer + on-device SmolLM3 text encoder.

    ``encode_prompt(prompt)`` tokenizes at the prompt's true length (no fixed max-length
    padding) and returns ``(prompt_embeds[1, T, 4096], all_hidden_states)`` as host tensors.
    """

    def __init__(
        self,
        checkpoint: str,
        *,
        device: ttnn.MeshDevice,
        ccl_manager: "CCLManager | None",
        parallel_config: "EncoderParallelConfig",
        use_torch: bool = False,
    ) -> None:
        self._device = device
        self._use_torch = use_torch

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, subfolder="tokenizer")

        if use_torch:
            self._torch_encoder = SmolLM3ForCausalLM.from_pretrained(checkpoint, subfolder="text_encoder").eval()
            self._encoder = None
        else:
            hf_config = AutoConfig.from_pretrained(checkpoint, subfolder="text_encoder")
            config = SmolLM3Config.from_hf_config(hf_config)
            self._encoder = SmolLM3TextEncoder(
                config, device=device, parallel_config=parallel_config, ccl_manager=ccl_manager
            )
            state_dict = SmolLM3ForCausalLM.from_pretrained(checkpoint, subfolder="text_encoder").model.state_dict()
            self._encoder.load_torch_state_dict(state_dict)
            self._torch_encoder = None

    def _tokenize(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize at true length (no fixed-length padding); special-case empty prompts."""
        if prompt == "":
            input_ids = torch.full((1, 1), BOT_TOKEN_ID, dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            return input_ids, attention_mask

        tokenized = self.tokenizer(
            [prompt],
            padding="longest",
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        return tokenized.input_ids, tokenized.attention_mask

    @torch.no_grad()
    def encode_prompt(self, prompt: str) -> tuple[torch.Tensor, list[torch.Tensor]]:
        input_ids, attention_mask = self._tokenize(prompt)
        seq_len = input_ids.shape[1]

        if self._use_torch:
            output = self._torch_encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            all_hidden_states = [h.detach() for h in output.hidden_states]
            prompt_embeds = torch.cat([all_hidden_states[-1], all_hidden_states[-2]], dim=-1)
            return prompt_embeds, all_hidden_states

        # The tokenized attention_mask is all-ones (no right-padding beyond the true token count),
        # so an explicit attention bias is redundant here: tile-pad to a valid ttnn shape
        # ourselves and run with attention_mask=None (is_causal SDPA) -- verified numerically
        # equivalent to routing the same all-ones mask through SmolLM3TextEncoder's own internal
        # tile-padding path, since causal masking guarantees the padded tail never influences the
        # real (leading) positions' outputs.
        padded_len = -(-seq_len // 32) * 32
        padded_ids = torch.nn.functional.pad(input_ids, (0, padded_len - seq_len), value=0)
        cos, sin = self._encoder.create_rope_tensors(1, padded_len)

        tt_ids = tt_tensor.from_torch(padded_ids, device=self._device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        tt_cos = tt_tensor.from_torch(cos, device=self._device)
        tt_sin = tt_tensor.from_torch(sin, device=self._device)

        prompt_embeds, all_hidden_states = self._encoder.encode(
            tt_ids, attention_mask=None, pos_embeds=(tt_cos, tt_sin)
        )

        host_prompt_embeds = tt_tensor.to_torch(prompt_embeds)[:, :seq_len, :]
        host_hidden_states = [tt_tensor.to_torch(h)[:, :seq_len, :] for h in all_hidden_states]
        return host_prompt_embeds, host_hidden_states
