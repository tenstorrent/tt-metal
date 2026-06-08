# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU fp32 reference LM runner — matches HuggingFace hidden states in generate()."""

from typing import Optional, Tuple

import torch
import ttnn


class ReferenceLMRunner:
    """Mirror of TTVibeVoiceLM using VibeVoiceForConditionalGenerationInference.forward()."""

    def __init__(self, ref_inference, mesh_device):
        self.ref = ref_inference
        self.embed = ref_inference.model.get_input_embeddings()
        self.device = mesh_device
        self.past_pos = None
        self.past_neg = None

    def build_prefill_embeds(
        self,
        input_ids: torch.Tensor,
        speech_input_mask: Optional[torch.Tensor],
        prefill_speech_embeds: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Return [1, S, H] float32 on CPU (reference token + speech scatter)."""
        with torch.no_grad():
            embeds = self.embed(input_ids).to(torch.float32)
        if speech_input_mask is not None and prefill_speech_embeds is not None:
            mask = speech_input_mask[0].cpu().bool()
            n = int(mask.sum().item())
            embeds[0, mask[: embeds.shape[1]]] = prefill_speech_embeds[:n].to(torch.float32)
        return embeds

    def prefill(self, inputs_embeds: torch.Tensor) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        with torch.no_grad():
            out = self.ref(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )
        self.past_pos = out.past_key_values
        return self._to_logits_tt(out.logits), self._to_hidden_tt(out.last_hidden_state)

    def step_embeds(self, embeds: torch.Tensor) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        with torch.no_grad():
            out = self.ref(
                inputs_embeds=embeds,
                past_key_values=self.past_pos,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )
        self.past_pos = out.past_key_values
        return self._to_logits_tt(out.logits), self._to_hidden_tt(out.last_hidden_state)

    def step_token(self, token_id: int) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        ids = torch.tensor([[token_id]], dtype=torch.long)
        embeds = self.embed(ids).to(torch.float32)
        return self.step_embeds(embeds)

    def reset_neg(self, speech_start_id: int) -> ttnn.Tensor:
        ids = torch.tensor([[speech_start_id]], dtype=torch.long)
        with torch.no_grad():
            out = self.ref(
                inputs_embeds=self.embed(ids).to(torch.float32),
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )
        self.past_neg = out.past_key_values
        return self._to_hidden_tt(out.last_hidden_state)

    def neg_step_token(self, token_id: int) -> ttnn.Tensor:
        ids = torch.tensor([[token_id]], dtype=torch.long)
        with torch.no_grad():
            out = self.ref(
                inputs_embeds=self.embed(ids).to(torch.float32),
                past_key_values=self.past_neg,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )
        self.past_neg = out.past_key_values
        return self._to_hidden_tt(out.last_hidden_state)

    def _to_hidden_tt(self, hidden: torch.Tensor) -> ttnn.Tensor:
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(1)
        return ttnn.as_tensor(
            hidden.unsqueeze(1).to(torch.float32),
            device=self.device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _to_logits_tt(self, logits: torch.Tensor) -> ttnn.Tensor:
        return ttnn.as_tensor(
            logits.unsqueeze(1).to(torch.bfloat16),
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
