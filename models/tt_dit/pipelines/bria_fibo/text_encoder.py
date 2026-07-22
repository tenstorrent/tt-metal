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

from typing import TYPE_CHECKING, Any

import torch
from transformers import AutoTokenizer

import ttnn

from ...encoders.smollm3.model_smollm3 import SmolLm3Checkpoint
from ...utils import tensor as tt_tensor

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
    ) -> None:
        self._device = device
        self._pad_buckets = tuple(pad_buckets)
        sp = parallel_config.sequence_parallel
        self._sp_axis = sp.mesh_axis if (sp is not None and sp.factor > 1) else None
        self._sp_factor = sp.factor if (sp is not None) else 1

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, subfolder="tokenizer")
        self._encoder = SmolLm3Checkpoint(checkpoint).build(
            device=device, parallel_config=parallel_config, ccl_manager=ccl_manager
        )

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
        input_ids, _attention_mask = self._tokenize(prompt)
        seq_len = input_ids.shape[1]

        tt_ids, tt_cos, tt_sin = self._prep_inputs(input_ids, seq_len)
        stacked = self._forward(tt_ids, tt_cos, tt_sin)

        # ONE readback, fast path: read only the SP-axis shards from one index of the (TP-replicated)
        # other axis and concat on host. tt_tensor.to_torch's mesh composer pulls all mesh devices --
        # including the redundant TP replicas -- and is ~17x slower for this tensor (measured ~10.5 s
        # vs ~0.6 s), which dominated the encode. See _read_seq_sharded.
        stacked_host = self._read_seq_sharded(stacked)[:, :seq_len, :]  # [N, seq_len, hidden]
        host_hidden_states = [stacked_host[i : i + 1] for i in range(stacked_host.shape[0])]
        # prompt_embeds = cat(last, second-last) -- diffusers pipeline_bria_fibo.py contract.
        host_prompt_embeds = torch.cat([host_hidden_states[-1], host_hidden_states[-2]], dim=-1)
        return host_prompt_embeds, host_hidden_states

    def _prep_inputs(self, input_ids: torch.Tensor, seq_len: int) -> tuple:
        """Host prep: pad to a fixed bucket, build RoPE, move to device (sharded on the SP axis)."""
        bucket = pick_bucket(seq_len, self._pad_buckets, self._sp_factor)
        padded_ids = torch.nn.functional.pad(input_ids, (0, bucket - seq_len), value=0)
        cos, sin = self._encoder.create_rope_tensors(1, bucket)
        tt_ids = tt_tensor.from_torch(
            padded_ids,
            device=self._device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_axes=[None, self._sp_axis],
        )
        tt_cos = tt_tensor.from_torch(cos, device=self._device, mesh_axes=[None, None, self._sp_axis, None])
        tt_sin = tt_tensor.from_torch(sin, device=self._device, mesh_axes=[None, None, self._sp_axis, None])
        return tt_ids, tt_cos, tt_sin

    def _forward(self, tt_ids: ttnn.Tensor, tt_cos: ttnn.Tensor, tt_sin: ttnn.Tensor) -> ttnn.Tensor:
        """Device forward: the N hidden states stacked on dim 0 into ONE tensor.

        Stacking device-side turns the readback into a single ``to_torch`` (one mesh-gather over the SP
        axis) instead of one per hidden state (~N x sp_factor tiny transfers), which is what dominated
        the readback. ``prompt_embeds`` is derived on host from the last two hidden states, so it needs
        no separate readback.
        """
        all_hidden_states = self._encoder.forward(tt_ids, pos_embeds=(tt_cos, tt_sin))
        return ttnn.concat(all_hidden_states, dim=0)  # [N, seq_local, hidden]; seq still sharded on the SP axis

    def _read_seq_sharded(self, x: ttnn.Tensor) -> torch.Tensor:
        """Read a seq-dim-1-sharded (over the SP axis), TP-replicated tensor to host, fast.

        ``tt_tensor.to_torch``'s mesh composer pulls from every device -- including the redundant TP
        replicas along the non-SP axis -- and is ~17x slower for this tensor. Instead read only the SP
        shards from index 0 of the (replicated) other axis via ``get_device_tensors`` and concat the seq
        dim on host. Device order is row-major over mesh coords (axis0-major).
        """
        rows, cols = tuple[Any, ...](self._device.shape)
        shards = ttnn.get_device_tensors(x)
        # SP shards live along self._sp_axis at index 0 of the other (replicated) axis.
        idxs = list(range(cols)) if self._sp_axis == 1 else [r * cols for r in range(rows)]
        return torch.cat([ttnn.to_torch(shards[i]) for i in idxs], dim=1)  # concat seq (tensor dim 1)
