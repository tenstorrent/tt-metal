# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Natural-prompt HF chain dump + reference logits for the logits-level accuracy check.

The ``L45_S128`` dump (``hf_layer_reference.py --random-ids 45 --seq-len 128 --layers 0-44 --num-decode 2
--stream``) feeds RANDOM token ids: its 45-layer residual chain is depth-chaotic and its logits are of no
interest. This script makes the same files (``prefill_input.pt``, ``layer{i}_prefill.pt`` / ``_decode{j}.pt``,
``layer{i}_weight_stats.json``; the same ``run_layers(stream=True)`` code path, bf16 text layers, strict-fp32
groups) for a NATURAL prompt: a short instruction + a paragraph of ordinary English through the chat template
(``[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>...<|assistant|><think></think>``, the convention of
``tests/test_mesh_long_context.py::build_needle_ids``), trimmed at a word boundary to exactly ``--seq-len``
tokens. It then adds the HF logits of every position (``logits.pt`` ``[S, vocab]`` fp32: the ``_ttft`` host
head on the last layer's ``x_out``, ``reference/logits_check.py::host_head``), ``top5.pt`` (HF top-5 / top-1
ids per position, plus the top-1 of the model's own bf16 head path) and ``prompt_meta.json``.

    python models/demos/glm53_flash/reference/dump_natural_prompt.py --out bringup/references/glm53_flash/L45_NAT_S128
    python models/demos/glm53_flash/reference/dump_natural_prompt.py --prompt-only          # print the prompt / ids
    python models/demos/glm53_flash/reference/dump_natural_prompt.py --logits-only --out ...  # (re)write the logits files
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkpoint import default_snapshot_dir  # noqa: E402
from hf_layer_reference import load_config, parse_layers, run_layers  # noqa: E402
from logits_check import LOGITS_FILE, META_FILE, TOPK_FILE, collapse_streams, host_head, load_head_weights  # noqa: E402

INSTRUCTION = "Summarize the paragraph below in two sentences."
PARAGRAPH = (
    "Every morning before the shops open, the baker on Mill Lane carries a tray of warm loaves to the bench outside "
    "his door and leaves them for the people who walk to the early train. He started the habit during a cold winter "
    "when the station cafe closed, and he has kept it up for nearly six years. Regulars leave coins in a tin by the "
    "door, though he has never asked for payment, and on most days the tin holds more than the bread is worth. The "
    "money is given to the local school. The baker says he does not think of it as charity; he simply likes to see "
    "the street busy before dawn, and the bread would otherwise go stale by noon."
)  # 128 chat-templated tokens end exactly at "...local school." (sentence boundary); the rest is spare for longer S


def chat_ids(tok, user_text: str) -> list[int]:
    """Chat-templated ids of one user turn + the closed think block (greedy continuation = the answer)."""
    ids = tok.apply_chat_template([{"role": "user", "content": user_text}], add_generation_prompt=True, tokenize=True)
    if hasattr(ids, "input_ids"):  # transformers >= 5 returns a BatchEncoding
        ids = ids["input_ids"]
    return [int(t) for t in ids] + tok.encode("</think>", add_special_tokens=False)


def build_prompt(
    tok, n_tokens: int, instruction: str = INSTRUCTION, paragraph: str = PARAGRAPH
) -> tuple[list[int], str]:
    """``instruction + paragraph`` trimmed at a word boundary (the cut sentence closed with a period) to exactly
    ``n_tokens`` chat-templated tokens."""
    words = paragraph.split()
    sentence_ends = [k for k in range(1, len(words) + 1) if words[k - 1].endswith(".")]
    word_cuts = [k for k in range(len(words), 0, -1) if k not in sentence_ends]
    nearest = None
    for k in sentence_ends[::-1] + word_cuts:  # whole sentences first, then any word boundary
        body = " ".join(words[:k]).rstrip(",;:")
        if not body.endswith("."):
            body += "."
        text = f"{instruction}\n\n{body}"
        ids = chat_ids(tok, text)
        if len(ids) == n_tokens:
            return ids, text
        if nearest is None or abs(len(ids) - n_tokens) < abs(nearest[0] - n_tokens):
            nearest = (len(ids), text)
    raise ValueError(f"no cut of the paragraph gives exactly {n_tokens} tokens (nearest {nearest[0]}); edit PARAGRAPH")


@torch.no_grad()
def write_logits(
    out_dir: Path, last_layer: int, eps: float, snapshot: Path, tok, ids: list[int], text: str, extra: dict
):
    """``logits.pt`` / ``top5.pt`` / ``prompt_meta.json`` from the dump's ``layer{last_layer}_prefill.pt``."""
    t0 = time.time()
    x_out = torch.load(out_dir / f"layer{last_layer}_prefill.pt")["x_out"][0]  # [S, 4, H] bf16
    norm_w, lm_w = load_head_weights(snapshot)
    logits = host_head(x_out, norm_w, lm_w, eps)  # [S, V] fp32: the _ttft head
    torch.save(logits, out_dir / LOGITS_FILE)
    top = logits.topk(5, dim=-1)
    # the model's own bf16 head path (hc_head mean -> RMSNorm -> lm_head, bf16 as generate_streaming): its top-1
    h = collapse_streams(x_out).to(torch.bfloat16)
    hf32 = h.float()
    hn = (hf32 * torch.rsqrt(hf32.pow(2).mean(-1, keepdim=True) + eps)).to(torch.bfloat16) * norm_w.to(torch.bfloat16)
    bf16_top1 = torch.cat(
        [
            torch.nn.functional.linear(hn[s0 : s0 + 32], lm_w.to(torch.bfloat16)).float().argmax(-1)
            for s0 in range(0, hn.shape[0], 32)
        ]
    )
    torch.save(
        {
            "top5_ids": top.indices,
            "top5_logits": top.values,
            "top1_ids": top.indices[:, 0],
            "top1_ids_bf16_head": bf16_top1,
            "prompt_ids": torch.tensor(ids),
        },
        out_dir / TOPK_FILE,
    )
    S = len(ids)
    meta = {
        "prompt_text": text,
        "prompt_ids": ids,
        "prompt_tokens": [tok.decode([i]) for i in ids],
        "n_tokens": S,
        "logits_layer": last_layer,
        "logits_head": "fp32: mean(streams) -> rmsnorm(eps, checkpoint norm weight) -> lm_head (tests/test_mesh_model.py::_ttft)",
        "rms_norm_eps": eps,
        "top1_fp32_vs_bf16_head_agreement": float((top.indices[:, 0] == bf16_top1).float().mean()),
        "next_token_top5": top.indices[-1].tolist(),
        "next_token_top5_text": [tok.decode([i]) for i in top.indices[-1].tolist()],
        "greedy_next_token_per_position_text": [tok.decode([i]) for i in top.indices[:, 0].tolist()],
        "logits_seconds": round(time.time() - t0, 1),
    }
    meta.update(extra)
    (out_dir / META_FILE).write_text(json.dumps(meta, indent=1))
    print(
        f"logits {tuple(logits.shape)} written ({meta['logits_seconds']} s); fp32 head vs bf16 head top-1 agreement "
        f"{meta['top1_fp32_vs_bf16_head_agreement']:.3f}; next token top-5 {meta['next_token_top5_text']}",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, default=None)
    ap.add_argument("--layers", type=str, default="0-44")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--num-decode", type=int, default=2, help="as the L45_S128 dump (random follow-on ids)")
    ap.add_argument("--out", type=Path, default=Path("bringup/references/glm53_flash/L45_NAT_S128"))
    ap.add_argument("--prompt-only", action="store_true", help="print the prompt / ids and exit")
    ap.add_argument("--logits-only", action="store_true", help="only (re)write logits.pt / top5.pt / prompt_meta.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    snapshot = args.snapshot or default_snapshot_dir()
    _, text_cfg = load_config(snapshot)
    tok = AutoTokenizer.from_pretrained(str(snapshot))
    layers = parse_layers(args.layers)
    assert layers[0] == 0 and layers == list(range(layers[0], layers[-1] + 1)), "layers must be 0..N"
    ids, text = build_prompt(tok, args.seq_len)
    print(f"snapshot {snapshot}\nprompt ({len(ids)} tokens): {text!r}\nids {ids}", flush=True)
    if args.prompt_only:
        return
    extra = {
        "command": " ".join(sys.argv),
        "layers": [layers[0], layers[-1]],
        "num_decode": args.num_decode,
        "prompt_convention": "apply_chat_template(add_generation_prompt=True) + '</think>' (tests/test_mesh_long_context.py::build_needle_ids)",
    }
    if not args.logits_only:
        t0 = time.time()
        torch.manual_seed(0)
        run_layers(snapshot, layers, torch.tensor([ids]), args.num_decode, args.out, text_cfg, stream=True, perturb=0.0)
        extra["chain_seconds"] = round(time.time() - t0, 1)
        print(f"chain of {len(layers)} layers done in {extra['chain_seconds']} s", flush=True)
    write_logits(args.out, layers[-1], float(text_cfg.rms_norm_eps), snapshot, tok, ids, text, extra)


if __name__ == "__main__":
    main()
