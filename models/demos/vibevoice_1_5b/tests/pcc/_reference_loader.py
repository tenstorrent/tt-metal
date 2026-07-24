# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Reference-model loader for microsoft/VibeVoice-1.5B.

This checkpoint's `config.json` declares `model_type: "vibevoice"`, which is NOT a
transformers-native architecture (`AutoConfig.from_pretrained` raises "does not recognize
this architecture" even on current transformers). The model only loads through the
architecture's own PyPI package (`vibevoice`, published by the microsoft/VibeVoice repo),
which vendors `VibeVoiceConfig` / `VibeVoiceForConditionalGeneration` and self-registers
them into transformers' Auto* mappings on import.

Two version-skew issues show up against a current transformers install and are worked
around here (narrowly, only around the `vibevoice` package's own import/init code):

1. Newer transformers ships native config classes for a couple of vibevoice *submodules*
   (e.g. `vibevoice_acoustic_tokenizer`) under the same class names the `vibevoice` package
   defines for itself. Its own `AutoModel.register(...)` calls then raise
   `ValueError: ... already used by a Transformers model.` because transformers' lazy
   auto-mapping matches on class *name*, not identity. This does not affect correctness of
   the classes actually used below (`VibeVoiceForConditionalGeneration`, `VibeVoiceConfig`),
   so registration is patched to tolerate re-registration during this import only.
2. `VibeVoiceForConditionalGeneration.tie_weights(self)` (vendored by the package) doesn't
   accept the `recompute_mapping` kwarg that current transformers' `PreTrainedModel.init_weights`
   now passes through; it's wrapped to swallow unknown kwargs.
3. `transformers.PreTrainedModel.from_pretrained` now constructs the module on the `meta`
   device by default before materializing weights. VibeVoice's own `__init__` eagerly builds
   a `DPMSolverMultistepScheduler` (diffusers) whose buffers get real values at construction
   time, which then fails to move off the meta device ("Cannot copy out of meta tensor").
   To avoid this, the model is constructed directly under a real `cpu` device context and the
   safetensors shards are loaded into it by hand via the checkpoint's weight index, instead of
   going through `from_pretrained`.
"""

import json

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers.models.auto.auto_factory import _LazyAutoMapping
from transformers.models.auto.configuration_auto import CONFIG_MAPPING, AutoConfig


def _register_tolerantly():
    """Let the vibevoice package re-register config/model classes that collide (by name)
    with newer transformers-native mappings, without touching those native mappings."""
    original_register = _LazyAutoMapping.register

    def _patched(self, key, value, exist_ok=False):
        return original_register(self, key, value, exist_ok=True)

    _LazyAutoMapping.register = _patched
    try:
        import vibevoice.modular.modeling_vibevoice as modeling_vibevoice
    finally:
        _LazyAutoMapping.register = original_register
    return modeling_vibevoice


def load_reference_model(model_id: str):
    """Return an nn.Module (in eval mode) equivalent to the HF reference for this model,
    loaded from whatever real format the repo actually ships."""
    modeling_vibevoice = _register_tolerantly()
    from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig

    CONFIG_MAPPING.register("vibevoice", VibeVoiceConfig, exist_ok=True)

    model_cls = modeling_vibevoice.VibeVoiceForConditionalGeneration
    original_tie_weights = model_cls.tie_weights

    def _tie_weights(self, *args, **kwargs):
        return original_tie_weights(self)

    model_cls.tie_weights = _tie_weights

    config = AutoConfig.from_pretrained(model_id)
    with torch.device("cpu"):
        model = model_cls(config)

    index_path = hf_hub_download(model_id, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_index = json.load(f)
    shard_files = sorted(set(weight_index["weight_map"].values()))

    state_dict = {}
    for shard_file in shard_files:
        shard_path = hf_hub_download(model_id, shard_file)
        state_dict.update(load_file(shard_path))
    model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model


if __name__ == "__main__":
    reference_model = load_reference_model("microsoft/VibeVoice-1.5B")
    print(type(reference_model))

    vocab_size = reference_model.config.decoder_config.vocab_size
    semantic_vae_dim = reference_model.config.semantic_vae_dim
    model_dtype = reference_model.get_input_embeddings().weight.dtype
    input_ids = torch.randint(0, vocab_size, (1, 8))
    speech_semantic_tensors = torch.zeros(1, 8, semantic_vae_dim, dtype=model_dtype)

    with torch.no_grad():
        output = reference_model(
            input_ids=input_ids,
            speech_semantic_tensors=speech_semantic_tensors,
        )
    print(type(output), output.logits.shape)
