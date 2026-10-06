# SPDX-License-Identifier: Apache-2.0
"""Independent PyTorch/HF full-model reference for the pinned Kolibri checkpoint.

Extends the provider-validated Stage1 PyTorch decoder; never imports TTNN.
Registration is explicit in readiness_entry, not an environment-wide monkeypatch.
"""
import torch
from torch import nn
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..tt.checkpoint import MODEL_ID, REVISION, SNAPSHOT, load_weights
from .reference import Kolibri1Config, ReferenceDecoder, norm


class Kolibri1ForCausalLM(PreTrainedModel):
    config_class = Kolibri1Config

    def __init__(self, config):
        super().__init__(config)
        self.embedding = nn.Parameter(
            load_weights("model.embed_tokens.")["model.embed_tokens.weight"], requires_grad=False
        )
        self.final_norm = nn.Parameter(load_weights("model.norm.")["model.norm.weight"], requires_grad=False)
        self.lm_head = nn.Parameter(load_weights("lm_head.")["lm_head.weight"], requires_grad=False)
        self.layers = []
        for idx in range(config.num_hidden_layers):
            prefix = f"model.layers.{idx}."
            self.layers.append(
                ReferenceDecoder({k.removeprefix(prefix): v for k, v in load_weights(prefix).items()}, idx)
            )
            print(f"HF_LOADED_LAYER {idx}", flush=True)
        self.position = 0

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        if str(pretrained_model_name_or_path) not in (MODEL_ID, str(SNAPSHOT)):
            raise ValueError("Reference is pinned to the Kolibri checkpoint")
        if kwargs.get("revision", REVISION) != REVISION:
            raise ValueError("Wrong Kolibri revision")
        return cls(Kolibri1Config.from_pretrained(SNAPSHOT, local_files_only=True))

    @torch.inference_mode()
    def forward(self, input_ids, *, past_key_values=None, use_cache=False, **kwargs):
        if past_key_values is None:
            for layer in self.layers:
                layer.reset()
            self.position = 0
        hidden = self.embedding[input_ids]
        for layer in self.layers:
            hidden = layer(hidden, start=self.position)
        self.position += input_ids.shape[1]
        hidden = norm(hidden, self.final_norm)
        logits = torch.nn.functional.linear(hidden.float(), self.lm_head.float())
        return CausalLMOutputWithPast(logits=logits, past_key_values=(self.position,) if use_cache else None)

    @torch.inference_mode()
    def generate(self, input_ids, *, max_new_tokens, **kwargs):
        result = input_ids.clone()
        current, state = input_ids, None
        eos = kwargs.get("eos_token_id", self.config.eos_token_id)
        eos = [eos] if isinstance(eos, int) else list(eos or [])
        for step in range(max_new_tokens):
            output = self(current, past_key_values=state, use_cache=True)
            current = output.logits[:, -1].argmax(-1, keepdim=True)
            result = torch.cat([result, current], 1)
            state = output.past_key_values
            print(f"HF_TOKEN {step} {current.flatten().tolist()}", flush=True)
            if step + 1 >= kwargs.get("min_new_tokens", 0) and all(int(token) in eos for token in current.flatten()):
                break
        return result


AutoModelForCausalLM.register(Kolibri1Config, Kolibri1ForCausalLM, exist_ok=True)
