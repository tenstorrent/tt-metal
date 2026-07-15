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
from ...utils.tracing import Tracer

if TYPE_CHECKING:
    from ...parallel.config import EncoderParallelConfig
    from ...parallel.manager import CCLManager

# diffusers pipeline_bria_fibo.py `get_prompt_embeds`: empty prompts ("") are special-cased to a
# single begin-of-text token, since the tokenizer itself does not add a BOS and would otherwise
# tokenize "" to a 0-length sequence.
BOT_TOKEN_ID = 128000


def pick_bucket(seq_len: int, buckets, sp_factor: int) -> int:
    """Smallest bucket >= seq_len; validate each bucket is divisible by ``sp_factor * 32``.

    A fixed bucket gives the encoder a stable padded shape (needed for clean sequence-parallel
    sharding and program-cache reuse). Raises ``ValueError`` if a bucket is not shardable or if the
    prompt is longer than every bucket (the caller should then add a larger bucket).
    """
    for b in sorted(buckets):
        if b % (sp_factor * 32) != 0:
            raise ValueError(f"bucket {b} not divisible by sp_factor*32 = {sp_factor * 32}")
    for b in sorted(buckets):
        if b >= seq_len:
            return b
    raise ValueError(f"prompt seq_len {seq_len} exceeds all buckets {sorted(buckets)}; add a larger bucket")


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
        pad_buckets=(1024,),
        use_trace: bool = True,
        use_torch: bool = False,
    ) -> None:
        self._device = device
        self._use_torch = use_torch
        self._pad_buckets = tuple(pad_buckets)
        sp = parallel_config.sequence_parallel
        self._sp_axis = sp.mesh_axis if (sp is not None and sp.factor > 1) else None
        self._sp_factor = sp.factor if (sp is not None) else 1
        # Trace the device forward (one trace per padding bucket): the forward is host-dispatch-bound,
        # and the fixed bucket gives it a static shape. Off for the torch path.
        self._use_trace = use_trace and not use_torch
        self._tracers: dict[int, Tracer] = {}

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
            max_length=3000,  # matches diffusers pipeline_bria_fibo.py default max_sequence_length
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

        tt_ids, tt_cos, tt_sin, bucket = self._prep_inputs(input_ids, seq_len)

        # The device forward is host-dispatch-bound; trace it (per bucket) and replay. First call for a
        # bucket captures (prep_run compiles + warms CCL/bias caches outside the captured region), later
        # calls copy the new inputs into the trace buffers and replay. Pos+neg share the 1024 trace.
        if self._use_trace:
            tracer = self._tracers.get(bucket)
            if tracer is None:
                tracer = Tracer(self._forward, device=self._device, prep_run=True, clone_prep_inputs=False)
                self._tracers[bucket] = tracer
            outputs = tracer(tt_ids, tt_cos, tt_sin)
        else:
            outputs = self._forward(tt_ids, tt_cos, tt_sin)
        prompt_embeds, all_hidden_states = outputs[0], list(outputs[1:])

        # Under SP the outputs are sequence-sharded; gather them back over the sp axis on readback.
        gather = (
            dict(mesh_axes=[None, self._sp_axis, None], composer_device=self._device) if self._sp_factor > 1 else {}
        )
        host_prompt_embeds = tt_tensor.to_torch(prompt_embeds, **gather)[:, :seq_len, :]
        host_hidden_states = [tt_tensor.to_torch(h, **gather)[:, :seq_len, :] for h in all_hidden_states]
        return host_prompt_embeds, host_hidden_states

    def _prep_inputs(self, input_ids: torch.Tensor, seq_len: int) -> tuple:
        """Host prep: pad to a fixed bucket, build RoPE, move to device (sharded on the SP axis).

        A fixed bucket gives a stable shape (SP sharding + trace/program-cache reuse). The tokenized
        attention_mask is all-ones, so the encoder runs with attention_mask=None: at sp=1 that is the
        is_causal SDPA path (the padded tail never influences leading tokens under causal masking); at
        sp>1 the encoder builds/caches a per-shard rectangular causal bias internally.
        """
        bucket = pick_bucket(seq_len, self._pad_buckets, self._sp_factor)
        padded_ids = torch.nn.functional.pad(input_ids, (0, bucket - seq_len), value=0)
        cos, sin = self._encoder.create_rope_tensors(1, bucket)
        if self._sp_factor > 1:
            tt_ids = tt_tensor.from_torch(
                padded_ids,
                device=self._device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_axes=[None, self._sp_axis],
            )
            tt_cos = tt_tensor.from_torch(cos, device=self._device, mesh_axes=[None, None, self._sp_axis, None])
            tt_sin = tt_tensor.from_torch(sin, device=self._device, mesh_axes=[None, None, self._sp_axis, None])
        else:
            tt_ids = tt_tensor.from_torch(
                padded_ids, device=self._device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            tt_cos = tt_tensor.from_torch(cos, device=self._device)
            tt_sin = tt_tensor.from_torch(sin, device=self._device)
        return tt_ids, tt_cos, tt_sin, bucket

    def _forward(self, tt_ids: ttnn.Tensor, tt_cos: ttnn.Tensor, tt_sin: ttnn.Tensor) -> tuple:
        """Device forward (the traced unit): flat tuple ``(prompt_embeds, *all_hidden_states)``."""
        prompt_embeds, all_hidden_states = self._encoder.encode(
            tt_ids, attention_mask=None, pos_embeds=(tt_cos, tt_sin)
        )
        return (prompt_embeds, *all_hidden_states)
