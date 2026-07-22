# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""KREA-2 text encoder wrapper for tt_dit.

Thin wrapper around :class:`Qwen3VlTokenizerEncoderPair` that reproduces the KREA-2
reference `Krea2Pipeline.get_text_hidden_states` tokenization + tap logic exactly:

  * chat-template prefix / suffix with MID-PADDING (``[prefix | prompt | PAD | suffix]``),
  * cumulative-valid-token position ids (padding does not consume a position),
  * ``output_hidden_states`` tap of the 12 KREA-2 select layers,
  * stack the taps along a new axis -> ``(B, text_seq, num_text_layers, text_hidden_dim)``,
  * slice off the ``prompt_template_encode_start_idx = 34`` system-prefix tokens.

Reference: diffusers_main/src/diffusers/pipelines/krea2/pipeline_krea2.py
           ``Krea2Pipeline.get_text_hidden_states`` (lines 214-261).

The heavy lifting (real-weight tt encoder) lives in the encoder_pair; here we only build
the exact `input_ids` / `attention_mask` / `position_ids` and drive `Qwen3VlTextEncoder`
directly so we control the (mid-padding) position ids the KREA-2 reference requires.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from loguru import logger

import ttnn
from models.tt_dit.encoders.qwen3vl.encoder_pair import Qwen3VlTokenizerEncoderPair
from models.tt_dit.parallel.config import EncoderParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils import tensor

if TYPE_CHECKING:
    from collections.abc import Sequence

# ---- KREA-2 reference constants (pipeline_krea2.py lines 189, 206-212) ----------------
# Indices into the encoder hidden_states tuple (0 = embedding output) to stack per token.
KREA2_SELECT_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
PROMPT_TEMPLATE_ENCODE_PREFIX = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n"
)
PROMPT_TEMPLATE_ENCODE_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
PROMPT_TEMPLATE_ENCODE_START_IDX = 34
PROMPT_TEMPLATE_ENCODE_NUM_SUFFIX_TOKENS = 5


class TextEncoder:
    """KREA-2 text conditioning encoder.

    Produces the stacked-hidden-state text features and the boolean attention mask that
    the KREA-2 transformer's text-fusion stage consumes.
    """

    def __init__(
        self,
        *,
        checkpoint_name: str,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
        use_torch: bool = False,
        is_fsdp: bool = True,
        select_layers: Sequence[int] = KREA2_SELECT_LAYERS,
    ) -> None:
        self._device = device
        self._use_torch = use_torch
        self._select_layers = tuple(select_layers)
        self._pair = Qwen3VlTokenizerEncoderPair(
            checkpoint_name,
            tokenizer_subfolder="tokenizer",
            encoder_subfolder="text_encoder",
            device=device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            use_torch=use_torch,
            is_fsdp=is_fsdp,
            select_layers=self._select_layers,
        )

    @property
    def tokenizer(self):
        return self._pair.tokenizer

    def _build_tokens(
        self, prompt: str | list[str], max_sequence_length: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reproduce reference mid-padding tokenization.

        Returns (input_ids, attention_mask_bool, position_ids[3, B, S]).
        Mirrors pipeline_krea2.py get_text_hidden_states lines 226-249.
        """
        tokenizer = self._pair.tokenizer
        prefix_idx = PROMPT_TEMPLATE_ENCODE_START_IDX
        prompt = [prompt] if isinstance(prompt, str) else list(prompt)
        text = [PROMPT_TEMPLATE_ENCODE_PREFIX + e for e in prompt]

        text_tokens = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_sequence_length + prefix_idx - PROMPT_TEMPLATE_ENCODE_NUM_SUFFIX_TOKENS,
            return_tensors="pt",
        )
        suffix_tokens = tokenizer(
            [PROMPT_TEMPLATE_ENCODE_SUFFIX] * len(text),
            return_tensors="pt",
        )

        input_ids = torch.cat([text_tokens.input_ids, suffix_tokens.input_ids], dim=1)
        attention_mask = torch.cat([text_tokens.attention_mask, suffix_tokens.attention_mask], dim=1).bool()

        # Cumulative-valid-token positions, broadcast over the 3 mRoPE axes (T/H/W equal for text).
        position_ids = (attention_mask.long().cumsum(dim=-1) - 1).clamp(min=0)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        return input_ids, attention_mask, position_ids

    @torch.no_grad()
    def get_text_hidden_states(
        self,
        prompt: str | list[str],
        max_sequence_length: int = 512,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """KREA-2 `get_text_hidden_states`.

        Returns:
            hidden_states: (B, text_seq, num_text_layers, text_hidden_dim) torch tensor on host.
            attention_mask: (B, text_seq) bool torch tensor on host.

        text_seq == max_sequence_length (== max_sequence_length + prefix_idx - prefix_idx).
        """
        prefix_idx = PROMPT_TEMPLATE_ENCODE_START_IDX
        input_ids, attention_mask, position_ids = self._build_tokens(prompt, max_sequence_length)

        if self._use_torch:
            # torch reference encoder (transformers Qwen3VLModel.language_model)
            encoder = self._pair.encoder
            dev = getattr(encoder, "device", "cpu")
            outputs = encoder(
                input_ids=input_ids.to(dev),
                attention_mask=attention_mask.to(dev),
                position_ids=position_ids.to(dev),
                output_hidden_states=True,
            )
            taps = [outputs.hidden_states[i].to("cpu").float() for i in self._select_layers]
        else:
            taps = self._encode_device(input_ids, attention_mask, position_ids)

        # Stack the tapped layers along a new axis-2: (B, seq, num_text_layers, dim).
        hidden_states = torch.stack(taps, dim=2)

        # Slice off the system-prefix tokens (reference lines 259-260).
        hidden_states = hidden_states[:, prefix_idx:]
        attention_mask = attention_mask[:, prefix_idx:]
        return hidden_states, attention_mask

    def _encode_device(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, position_ids: torch.Tensor
    ) -> list[torch.Tensor]:
        """Run the tt Qwen3-VL encoder with the KREA-2 mid-padding position ids.

        We drive `Qwen3VlTextEncoder.forward` directly (rather than the pair's `encode`,
        which recomputes position ids from the mask) so the suffix tokens receive the
        correct mRoPE phase.
        """
        encoder = self._pair.encoder
        batch_size, seq_len = input_ids.shape

        # Build cos/sin from the explicit KREA-2 position ids. create_rope_tensors derives
        # positions from the mask internally, so we recompute cos/sin here using the exact
        # mid-padding position_ids to keep suffix-token phases correct.
        cos, sin = _rope_from_position_ids(
            position_ids,
            head_dim=encoder._head_dim,
            rope_theta=encoder._rope_theta,
            mrope_section=encoder._mrope_section,
            mrope_interleaved=encoder._mrope_interleaved,
        )

        tt_tokens = tensor.from_torch(input_ids, device=self._device, dtype=ttnn.uint32)
        tt_mask = tensor.from_torch(attention_mask.to(torch.float32), device=self._device)
        tt_cos = tensor.from_torch(cos, device=self._device)
        tt_sin = tensor.from_torch(sin, device=self._device)

        tt_hidden = encoder.forward(
            tt_tokens,
            attention_mask=tt_mask,
            pos_embeds=(tt_cos, tt_sin),
            select_layers=self._select_layers,
        )
        logger.info("krea2 text encoder produced {} tapped hidden states", len(tt_hidden))
        return [tensor.to_torch(h).float() for h in tt_hidden]

    # ---- dynamic-load hooks (mirror qwenimage TextEncoder) ----------------------------
    def encoder_loaded(self) -> bool:
        return getattr(self._pair, "encoder_loaded", lambda: True)()

    def reload_encoder_weights(self) -> None:
        if hasattr(self._pair, "reload_encoder_weights"):
            self._pair.reload_encoder_weights()

    def deallocate_encoder_weights(self) -> None:
        if hasattr(self._pair, "deallocate_encoder_weights"):
            self._pair.deallocate_encoder_weights()


def _rope_from_position_ids(
    position_ids: torch.Tensor,
    *,
    head_dim: int,
    rope_theta: float,
    mrope_section: Sequence[int],
    mrope_interleaved: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Qwen3-VL (interleaved) mRoPE cos/sin from explicit position ids.

    position_ids: (3, batch, seq). Returns cos, sin each (batch, 1, seq, head_dim).

    This mirrors `create_rope_tensors` (model_qwen3vl.py lines 553-612) but consumes
    caller-supplied position ids instead of deriving them from the attention mask, so the
    KREA-2 mid-padding layout is honoured.
    """
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
    batch_size = position_ids.shape[1]
    inv_freq_expanded = inv_freq[None, None, :, None].float().expand(3, batch_size, -1, 1)
    position_ids_expanded = position_ids[:, :, None, :].float()
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)  # (3, B, S, head_dim//2)

    if mrope_interleaved:
        freqs_t = freqs[0].clone()
        for dim, offset in enumerate((1, 2), start=1):  # height, width
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        freqs = freqs_t
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(1)
        sin = emb.sin().unsqueeze(1)
        return cos, sin

    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    s = list(mrope_section) * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(s, dim=-1))], dim=-1)
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(s, dim=-1))], dim=-1)
    return cos.unsqueeze(1), sin.unsqueeze(1)
