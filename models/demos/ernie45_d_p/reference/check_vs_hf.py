# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Gate P1.2: standalone reference == HF transformers Ernie4_5_MoeForCausalLM (fp32 both sides)."""

import argparse
import gc
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
from models.demos.ernie45_d_p.bringup import metrics  # noqa: E402
from models.demos.ernie45_d_p.reference.ernie_ref import (  # noqa: E402
    ErnieReference,
    load_book_tokens,
    pcc,
    resolve_model_path,
)

TASK = os.environ.get("ERNIE_BRINGUP_TASK", "P1.2")


def run_hf(path, tokens):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, attn_implementation="eager")
    model.eval()
    outs = {}
    hooks = [
        layer.register_forward_hook(
            lambda m, i, o, idx=idx: outs.__setitem__(idx, (o[0] if isinstance(o, tuple) else o)[0].clone())
        )
        for idx, layer in enumerate(model.model.layers)
    ]
    with torch.no_grad():
        logits = model(tokens[None], use_cache=False).logits[0]
    for h in hooks:
        h.remove()
    del model
    gc.collect()
    return outs, logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=512)
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count())
    path = resolve_model_path()
    from transformers import AutoTokenizer

    tokens = load_book_tokens(AutoTokenizer.from_pretrained(path), a.seq)

    hf_layers, hf_logits = run_hf(path, tokens)

    ref = ErnieReference(path, dtype=torch.float32)
    rec_out = {}

    def rec(name, t):
        if name.endswith(".out"):
            rec_out[int(name[1:].split(".")[0])] = t.clone()

    _, _, _ = None, None, None
    cache = ref.new_cache(a.seq)
    _, logits = ref.forward_chunk(tokens, 0, cache, rec, logits_last_n=a.seq)

    worst = 1.0
    for i in sorted(hf_layers):
        p = pcc(rec_out[i], hf_layers[i])
        worst = min(worst, p)
        metrics.record(TASK, f"pcc_hidden_L{i:02d}", p)
        print(f"L{i:02d} pcc={p:.7f} maxabs={(rec_out[i]-hf_layers[i]).abs().max():.3e}")
    p_log = pcc(logits, hf_logits)
    top1 = (logits.argmax(-1) == hf_logits.argmax(-1)).float().mean().item()
    print(f"logits pcc={p_log:.7f} top1_match={top1:.4f} worst_layer={worst:.7f}")
    metrics.record(TASK, "pcc_logits", p_log)
    metrics.record(TASK, "top1_match_frac", top1)
    # Sanity that the model actually predicts the book: next-token accuracy vs the text itself.
    nxt = (logits[:-1].argmax(-1) == tokens[1:]).float().mean().item()
    print(f"next-token accuracy on book text: {nxt:.3f}")
    metrics.record(TASK, "book_next_token_acc", nxt)


if __name__ == "__main__":
    main()
