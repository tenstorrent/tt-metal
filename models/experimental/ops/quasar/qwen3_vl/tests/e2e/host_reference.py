# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Truncated HF Qwen3-VL reference: runs prefill plus teacher-forced decode and keeps per-stage goldens."""
from dataclasses import dataclass, field

import torch
from transformers import AutoConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

from models.experimental.ops.quasar.qwen3_vl.tests.e2e.config import HF_MODEL_ID
from models.experimental.ops.quasar.qwen3_vl.tt.quasar_config import truncate_hf_config


@dataclass
class Goldens:
    tensors: dict = field(default_factory=dict)
    teacher_tokens: list = field(default_factory=list)
    prefill_len: int = 0
    num_patches: int = 0


def load_hf_model(vision_layers, text_layers, deepstack_at):
    config = truncate_hf_config(AutoConfig.from_pretrained(HF_MODEL_ID), vision_layers, text_layers, deepstack_at)
    model = Qwen3VLForConditionalGeneration.from_pretrained(HF_MODEL_ID, config=config, torch_dtype=torch.float32)
    return model.eval()


def _first(out):
    return out[0] if isinstance(out, (tuple, list)) else out


@torch.no_grad()
def run_reference(model, inputs, decode_steps):
    g = Goldens(prefill_len=int(inputs["input_ids"].shape[1]), num_patches=int(inputs["image_grid_thw"][0].prod()))
    visual, lm = model.model.visual, model.model.language_model
    hooks = []

    def keep(name, fn=lambda t: t):
        def hook(_mod, _inp, out):
            g.tensors[name] = fn(_first(out)).detach().float().clone()

        return hook

    flat = lambda t: t.reshape(-1, t.shape[-1])
    for i, blk in enumerate(visual.blocks):
        hooks.append(blk.register_forward_hook(keep(f"vision.block{i}", flat)))
        hooks.append(blk.attn.register_forward_hook(keep(f"vision.block{i}.attn", flat)))
        hooks.append(blk.mlp.register_forward_hook(keep(f"vision.block{i}.mlp", flat)))
    taps = [i for i in visual.deepstack_visual_indexes if i < len(visual.blocks)]
    for j, _ in enumerate(taps):
        hooks.append(visual.deepstack_merger_list[j].register_forward_hook(keep(f"vision.deepstack{j}")))
    hooks.append(visual.merger.register_forward_hook(keep("vision.merger")))
    for i, layer in enumerate(lm.layers):
        hooks.append(layer.register_forward_hook(keep(f"text.layer{i}", flat)))
        hooks.append(layer.self_attn.register_forward_hook(keep(f"text.layer{i}.attn", flat)))
        hooks.append(layer.mlp.register_forward_hook(keep(f"text.layer{i}.mlp", flat)))
    hooks.append(lm.norm.register_forward_hook(keep("text.norm", lambda t: t.reshape(-1, t.shape[-1])[-1])))
    try:
        out = model(**inputs)
    finally:
        for h in hooks:
            h.remove()
    g.tensors["text.logits.prefill"] = out.logits[0, -1].float().clone()

    ids = inputs["input_ids"]
    mm_types = inputs["mm_token_type_ids"]
    tok = int(out.logits[0, -1].argmax())
    for k in range(decode_steps):
        g.teacher_tokens.append(tok)
        ids = torch.cat([ids, torch.tensor([[tok]])], dim=1)
        mm_types = torch.cat([mm_types, torch.zeros((1, 1), dtype=mm_types.dtype)], dim=1)  # 0 = text token
        step = model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            mm_token_type_ids=mm_types,
            pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"],
        )
        g.tensors[f"text.logits.decode{k}"] = step.logits[0, -1].float().clone()
        tok = int(step.logits[0, -1].argmax())
    return g
