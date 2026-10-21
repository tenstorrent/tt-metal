# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""HF float32 references for the full-model tests: the real checkpoint, optionally reduced to a subset of layers.

``hf_causal_lm(layer_indices)`` builds ``Gemma4UnifiedForCausalLM`` with only the listed decoder layers (their real
weights, renumbered 0..n-1 with their own layer types), the real embedding, final norm and tied LM head with the
final-logit soft-cap, so the reduced TTNN model (``Gemma4Model(layer_indices=...)``) has an exact counterpart.
"""

import copy

import torch

from models.autoports.google_gemma_4_12b_it.tt.model import Checkpoint, text_config


def hf_causal_lm(layer_indices=None, dtype=torch.float32):
    from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedForCausalLM

    base = text_config()
    layers = list(range(base.num_hidden_layers)) if layer_indices is None else list(layer_indices)
    cfg = copy.deepcopy(base)
    cfg.num_hidden_layers = len(layers)
    cfg.layer_types = [base.layer_types[i] for i in layers]
    with torch.device("meta"):
        model = Gemma4UnifiedForCausalLM(cfg)
    ckpt = Checkpoint()
    sd = {
        "model.embed_tokens.weight": ckpt.tensor("embed_tokens.weight"),
        "model.norm.weight": ckpt.tensor("norm.weight"),
    }
    for j, i in enumerate(layers):
        for k, v in ckpt.layer(i).items():
            sd[f"model.layers.{j}.{k}"] = v
    sd["lm_head.weight"] = sd["model.embed_tokens.weight"]
    model.load_state_dict({k: v.to(dtype) for k, v in sd.items()}, assign=True, strict=False)
    model.tie_weights()
    # non-persistent buffers (rotary inv_freq, embed scale) are not in the state dict: rebuild them on CPU
    rope = type(model.model.rotary_emb)(cfg)
    model.model.rotary_emb = rope
    emb = model.model.embed_tokens
    emb.register_buffer("embed_scale", torch.tensor(emb.scalar_embed_scale), persistent=False)
    return model.eval()


@torch.no_grad()
def hf_streamed_logits(sequences, positions, parts=4, dtype=torch.bfloat16):
    """Full 48-layer HF logits at ``positions[k]`` (an int or a list) of each ``sequences[k]``, streaming the layers in
    ``parts`` groups so the host holds about 1 / ``parts`` of the model (each group is ``hf_causal_lm`` over its real
    layers; the last layer's output, captured before the final norm, is the next group's ``inputs_embeds``, which skips
    the embedding scale as a hidden state must; the last group applies the final norm, LM head and soft-cap).
    Bitwise equal to the unsplit model (2-layer check, doc/datatype_sweep work log 12)."""
    import gc

    n = text_config().num_hidden_layers
    groups = [list(range(n))[g * n // parts : (g + 1) * n // parts] for g in range(parts)]
    hidden = [None] * len(sequences)
    out = [None] * len(sequences)
    for gi, layers in enumerate(groups):
        model = hf_causal_lm(layers, dtype)
        cap = {}
        hook = model.model.layers[-1].register_forward_hook(
            lambda _m, _i, o: cap.__setitem__("h", o[0] if isinstance(o, tuple) else o)
        )
        for k, ids in enumerate(sequences):
            logits = model(torch.tensor([ids])).logits if gi == 0 else model(inputs_embeds=hidden[k]).logits
            hidden[k] = cap["h"]
            if gi == len(groups) - 1:
                out[k] = logits[0, positions[k]].float()
            del logits
        hook.remove()
        del model
        gc.collect()
    return out


@torch.no_grad()
def hf_logits(model, ids):
    """Logits ``[S, vocab]`` of a prompt (HF forward with the model's own masks)."""
    return model(torch.tensor([ids])).logits[0].float()


@torch.no_grad()
def hf_greedy(model, ids, n):
    out = model.generate(torch.tensor([ids]), max_new_tokens=n, do_sample=False, num_beams=1)
    return out[0, len(ids) :].tolist()
