# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Reference gate, part 1: the model's standalone CPU reference matches the HF implementation (fp32 both sides).

    python -m models.demos.common.bringup.reference.check_hf --spec S --seq 512 [--num-layers N]

Records pcc_hidden_L{i:02d} per layer, pcc_logits, top1_match_frac, text_next_token_acc.
``--num-layers`` (or spec ``hf.parity_layers``) truncates both models to their first N layers, for checkpoints that
do not fit in host memory in fp32. Hooks the model may provide: ``hf_model(spec, num_layers)``, ``hf_layers(model)``.
"""

from __future__ import annotations

import argparse
import gc
import os

import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.reference.golden import hf_path, load_spec, text_tokens


def load_hf(spec, num_layers: int | None):
    hooks = spec.hooks()
    if hasattr(hooks, "hf_model"):
        return hooks.hf_model(spec, num_layers)
    from transformers import AutoConfig, AutoModelForCausalLM

    path = hf_path(spec)
    trust = bool(spec.get("hf.trust_remote_code"))
    cfg = AutoConfig.from_pretrained(path, trust_remote_code=trust)
    if num_layers:
        cfg.num_hidden_layers = num_layers
    return AutoModelForCausalLM.from_pretrained(
        path, config=cfg, dtype=torch.float32, attn_implementation="eager", trust_remote_code=trust
    ).eval()


def hf_layers(spec, model):
    hooks = spec.hooks()
    if hasattr(hooks, "hf_layers"):
        return hooks.hf_layers(model)
    return model.model.layers


def _hidden(o) -> torch.Tensor:
    t = o[0] if isinstance(o, tuple) else o
    return t.reshape(-1, t.shape[-1]).detach().clone()


def run_hf(spec, tokens, num_layers):
    model = load_hf(spec, num_layers)
    outs = {}
    hooks = [
        layer.register_forward_hook(lambda m, i, o, idx=idx: outs.__setitem__(idx, _hidden(o)))
        for idx, layer in enumerate(hf_layers(spec, model))
    ]
    with torch.no_grad():
        logits = model(tokens[None], use_cache=False).logits[0].float()
    for h in hooks:
        h.remove()
    del model
    gc.collect()
    return outs, logits


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--num-layers", type=int, default=None)
    a = ap.parse_args(argv)
    torch.set_num_threads(os.cpu_count())
    spec = load_spec(a.spec)
    n = a.num_layers or spec.get("hf.parity_layers")
    tokens = text_tokens(spec, a.seq)

    hf_out, hf_logits = run_hf(spec, tokens, n)

    layers = list(range(n or spec.num_layers))
    ref = spec.hooks().reference(spec, layers=layers, dtype=torch.float32)
    rec_out = {}

    def rec(name, t):
        if name.startswith("L") and name.endswith(".out"):
            rec_out[int(name[1:].split(".")[0])] = t.detach().clone()

    state = ref.new_state(a.seq)
    _, logits = ref.forward_chunk(tokens, 0, state, rec, logits_last_n=a.seq)
    logits = logits.float()

    worst = 1.0
    for i in layers:
        p = metrics.pcc(rec_out[i], hf_out[i])
        worst = min(worst, p)
        metrics.record(f"pcc_hidden_L{i:02d}", p)
        print(f"L{i:02d} pcc={p:.7f} maxabs={(rec_out[i] - hf_out[i]).abs().max():.3e}")
    p_log = metrics.pcc(logits, hf_logits)
    top1 = (logits.argmax(-1) == hf_logits.argmax(-1)).float().mean().item()
    nxt = (logits[:-1].argmax(-1) == tokens[1:]).float().mean().item()
    print(f"logits pcc={p_log:.7f} top1_match={top1:.4f} worst_layer={worst:.7f} next-token acc on text={nxt:.3f}")
    metrics.record("pcc_logits", p_log)
    metrics.record("top1_match_frac", top1)
    metrics.record("text_next_token_acc", nxt)
    metrics.record("parity_layers", len(layers))


if __name__ == "__main__":
    main()
