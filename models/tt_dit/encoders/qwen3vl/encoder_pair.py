# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Tokenizer + encoder pair for the Qwen3-VL text tower (KREA-2 text encoder).

Mirrors models/tt_dit/encoders/qwen25vl/encoder_pair.py. Kept minimal but importable:
it wires the HF `Qwen3VLModel` reference and the tt `Qwen3VlTextEncoder`, reading the
nested `text_config` and `rope_parameters` layout used by transformers 5.x.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from loguru import logger
from transformers import AutoTokenizer, Qwen3VLModel

import ttnn

from ...encoders.qwen3vl.model_qwen3vl import Qwen3VlTextEncoder
from ...parallel.config import EncoderParallelConfig
from ...parallel.manager import CCLManager
from ...utils import tensor

if TYPE_CHECKING:
    from collections.abc import Sequence


def _text_config(model: Qwen3VLModel):
    cfg = model.config
    return getattr(cfg, "text_config", cfg)


def build_tt_encoder(
    text_config,
    *,
    device: ttnn.MeshDevice,
    parallel_config: EncoderParallelConfig | None = None,
    ccl_manager: CCLManager | None = None,
    is_fsdp: bool = False,
    num_hidden_layers: int | None = None,
) -> Qwen3VlTextEncoder:
    rope_params = getattr(text_config, "rope_parameters", None) or getattr(text_config, "rope_scaling", None) or {}
    rope_theta = getattr(text_config, "rope_theta", None) or rope_params.get("rope_theta")
    head_dim = getattr(text_config, "head_dim", None) or (text_config.hidden_size // text_config.num_attention_heads)
    mrope_interleaved = bool(rope_params.get("mrope_interleaved", False))

    return Qwen3VlTextEncoder(
        vocab_size=text_config.vocab_size,
        hidden_size=text_config.hidden_size,
        intermediate_size=text_config.intermediate_size,
        hidden_act=text_config.hidden_act,
        num_hidden_layers=num_hidden_layers if num_hidden_layers is not None else text_config.num_hidden_layers,
        num_attention_heads=text_config.num_attention_heads,
        num_key_value_heads=text_config.num_key_value_heads,
        head_dim=head_dim,
        rms_norm_eps=text_config.rms_norm_eps,
        rope_theta=rope_theta,
        mrope_section=rope_params["mrope_section"],
        mrope_interleaved=mrope_interleaved,
        device=device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
        is_fsdp=is_fsdp,
    )


class Qwen3VlTokenizerEncoderPair:
    def __init__(
        self,
        checkpoint: str,
        *,
        tokenizer_subfolder: str | None = None,
        encoder_subfolder: str | None = None,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None = None,
        parallel_config: EncoderParallelConfig | None = None,
        use_torch: bool = False,
        is_fsdp: bool = False,
        select_layers: Sequence[int] | None = None,
    ) -> None:
        self._device = device
        self._ccl_manager = ccl_manager
        self._parallel_config = parallel_config
        self._checkpoint = checkpoint
        self._encoder_subfolder = encoder_subfolder
        self._use_torch = use_torch
        self._is_fsdp = is_fsdp
        self._select_layers = select_layers

        tok_kwargs = {"subfolder": tokenizer_subfolder} if tokenizer_subfolder is not None else {}
        self._tokenizer = AutoTokenizer.from_pretrained(checkpoint, **tok_kwargs)
        self._encoder = self._load_encoder(checkpoint, encoder_subfolder, use_torch=use_torch)

    def _load_encoder(self, checkpoint: str, subfolder: str | None, *, use_torch: bool):
        enc_kwargs = {"subfolder": subfolder} if subfolder is not None else {}
        torch_model = Qwen3VLModel.from_pretrained(checkpoint, **enc_kwargs)

        if use_torch:
            return torch_model

        text_config = _text_config(torch_model)
        model = build_tt_encoder(
            text_config,
            device=self._device,
            parallel_config=self._parallel_config,
            ccl_manager=self._ccl_manager,
            is_fsdp=self._is_fsdp,
        )

        # Qwen3VLModel exposes the text tower as `.language_model`.
        torch_text_model = torch_model.language_model
        state_dict = torch_text_model.state_dict()
        del torch_model
        del torch_text_model
        model.load_torch_state_dict(state_dict)
        return model

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def encoder(self):
        return self._encoder

    def encode(
        self,
        prompts: Sequence[str],
        *,
        sequence_length: int,
        select_layers: Sequence[int] | None = None,
    ) -> list[torch.Tensor]:
        tokenizer_out = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            max_length=sequence_length,
            truncation=True,
        )
        tokens = tokenizer_out.input_ids
        attention_mask = tokenizer_out.attention_mask

        if self._use_torch:
            tokens = tokens.to(device=self._encoder.device)
            with torch.no_grad():
                out = self._encoder.forward(tokens, attention_mask=attention_mask, output_hidden_states=True)
            hs = out.hidden_states
            sel = select_layers if select_layers is not None else self._select_layers
            if sel is not None:
                return [hs[i].to("cpu") for i in sel]
            return [h.to("cpu") for h in hs]

        cos, sin = self._encoder.create_rope_tensors(tokens.shape[0], tokens.shape[1], attention_mask)
        tt_tokens = tensor.from_torch(tokens, device=self._device, dtype=ttnn.uint32)
        tt_mask = tensor.from_torch(attention_mask, device=self._device)
        tt_cos = tensor.from_torch(cos, device=self._device)
        tt_sin = tensor.from_torch(sin, device=self._device)

        sel = select_layers if select_layers is not None else self._select_layers
        tt_hidden = self._encoder.forward(
            tt_tokens, attention_mask=tt_mask, pos_embeds=(tt_cos, tt_sin), select_layers=sel
        )
        logger.info("qwen3vl text encoder produced {} hidden states", len(tt_hidden))
        return [tensor.to_torch(h) for h in tt_hidden]
