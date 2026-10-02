# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Layer-streamed HF readiness reference for Laguna (no full-model load).

Laguna-S-2.1 is 235 GB in bf16, more than this host can hold next to other jobs, so the shared
``models.common.readiness_check.generate`` (which loads the whole model) cannot produce its reference.
This tool runs the exact HF ``LagunaDecoderLayer`` code one layer at a time over one fixed token
sequence: load layer i from the checkpoint, run it on the hidden states of all positions, free it,
continue. Peak host memory is one layer (~30 GB for an S MoE layer in fp32), not the model.

The sequence is the AIME24 chat prompt (rendered with this checkpoint's chat template) followed by a
fixed continuation. The readiness metric is teacher-forced: ``topk_tokens[i]`` is HF's top-K
prediction for position ``P + i`` given the forced tokens before it, so the continuation does not need
to be this model's own greedy output. By default it is the Laguna-XS-2.1 reference continuation,
re-tokenized with this checkpoint's tokenizer.

Optional ``--router-dump`` saves, for every MoE layer, the router input (the post-attention RMSNorm
output for all positions) plus HF's top-k expert ids and weights, so the device router can be checked
on real hidden states (``tests/test_router_precision.py``).

Usage (CPU only; no device):
  python -m models.demos.laguna.tests.gen_streamed_reference \
    --output tests/reference_outputs/readiness_aime24_chat_s.refpt --dtype fp32 --router-dump <dir>
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import torch

from models.demos.laguna.tests import laguna_reference as R
from models.demos.laguna.tests import laguna_weights as W
from models.demos.laguna.tt.model_spec import MODEL_ID
from models.common.readiness_check.schema import Reference, ReferenceEntry, load_reference, save_reference

TESTS_DIR = Path(__file__).resolve().parent
XS_REFERENCE = TESTS_DIR / "reference_outputs" / "readiness_aime24_chat.refpt"
XS_MODEL_ID = "poolside/Laguna-XS-2.1"
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def chat_template_tokens(tokenizer, prompt_text):
    out = tokenizer.apply_chat_template([{"role": "user", "content": prompt_text}], add_generation_prompt=True, tokenize=True)
    if isinstance(out, dict) or hasattr(out, "get"):
        out = out["input_ids"]
    if len(out) and isinstance(out[0], (list, tuple)):
        out = out[0]
    return [int(t) for t in out]


def load_top_level(keys):
    """fp32 copies of the given top-level checkpoint tensors (embed, final norm, lm_head)."""
    from safetensors import safe_open

    snap_dir, weight_map = W._index()
    out = {}
    for shard in sorted({weight_map[k] for k in keys}):
        with safe_open(W._resolve_shard(snap_dir, shard), "pt") as f:
            for k in keys:
                if weight_map[k] == shard:
                    out[k] = f.get_tensor(k).to(torch.float32)
    return out


def build_sequence(tokenizer, gen_len, source_reference):
    """Return (prompt_text, prompt_ids, continuation_ids)."""
    from transformers import AutoTokenizer

    ref = load_reference(source_reference)
    entry = ref.entries[0]
    prompt_ids = chat_template_tokens(tokenizer, entry.prompt_text)
    source_tokenizer = AutoTokenizer.from_pretrained(ref.hf_model_id, trust_remote_code=True)
    continuation_text = source_tokenizer.decode(entry.generated_tokens[0].tolist(), skip_special_tokens=False)
    continuation_ids = tokenizer.encode(continuation_text, add_special_tokens=False)[:gen_len]
    if len(continuation_ids) < gen_len:
        raise ValueError(f"continuation has only {len(continuation_ids)} tokens; requested {gen_len}")
    return entry.prompt_text, prompt_ids, continuation_ids


def rms_norm(x, weight, eps):
    variance = x.pow(2).mean(-1, keepdim=True)
    return weight * (x * torch.rsqrt(variance + eps))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--gen-len", type=int, default=100)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--dtype", choices=tuple(DTYPES), default="fp32")
    ap.add_argument("--source-reference", type=Path, default=XS_REFERENCE)
    ap.add_argument("--router-dump", type=Path, default=None, help="directory for per-MoE-layer router inputs")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = torch default)")
    ap.add_argument("--save-logits", type=Path, default=None, help="also save the [G, vocab] fp32 logits (accuracy gates)")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    dtype = DTYPES[args.dtype]

    from transformers import AutoTokenizer

    config = R.build_config()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    prompt_text, prompt_ids, cont_ids = build_sequence(tokenizer, args.gen_len, args.source_reference)
    seq_ids = prompt_ids + cont_ids
    P, G = len(prompt_ids), len(cont_ids)
    print(f"{MODEL_ID}: prompt {P} tokens + continuation {G} tokens, dtype {args.dtype}", flush=True)

    top = load_top_level(["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"])
    hidden = top["model.embed_tokens.weight"][torch.tensor(seq_ids)].unsqueeze(0).to(dtype)  # [1, S, H]

    if args.router_dump is not None:
        args.router_dump.mkdir(parents=True, exist_ok=True)
    num_layers = config.num_hidden_layers
    for layer_idx in range(num_layers):
        t0 = time.time()
        raw = W.load_layer_tensors(layer_idx)
        state = W.to_hf_layer_state_dict(raw, config, layer_idx)
        del raw
        ctx = R.make_context(config, layer_idx, state_dict=state, dtype=dtype)
        del state
        captured = {}
        hook = None
        if args.router_dump is not None and hasattr(ctx.layer.mlp, "gate"):

            def _capture(module, inputs):
                captured["x"] = inputs[0].detach().reshape(-1, config.hidden_size).to(torch.float32).clone()

            hook = ctx.layer.mlp.register_forward_pre_hook(_capture)
        hidden, _ = R.reference_forward(ctx, hidden)
        if hook is not None:
            hook.remove()
            x = captured["x"]
            gate = ctx.layer.mlp.gate
            # HF's own router on the captured input, in fp32 (the reference selection).
            gate32 = type(gate)(config)
            gate32.load_state_dict({k: v.to(torch.float32) for k, v in gate.state_dict().items()})
            _, weights, experts = gate32(x)
            torch.save(
                {
                    "layer": layer_idx,
                    "x": x,
                    "hf_experts": experts.to(torch.int32),
                    "hf_weights": weights.to(torch.float32),
                    "gate_weight": gate.weight.detach().to(torch.float32).clone(),
                    "e_score_correction_bias": gate.e_score_correction_bias.detach().to(torch.float32).clone(),
                    "dtype": args.dtype,
                },
                args.router_dump / f"router_L{layer_idx:02d}.pt",
            )
        del ctx
        gc.collect()
        print(
            f"layer {layer_idx:2d}/{num_layers} {config.layer_types[layer_idx]:17s} "
            f"rms={hidden.float().pow(2).mean().sqrt():.4f} {time.time() - t0:.1f}s",
            flush=True,
        )

    h = hidden[0].to(torch.float32)
    h = rms_norm(h, top["model.norm.weight"], config.rms_norm_eps)
    logits = h[P - 1 : P + G - 1] @ top["lm_head.weight"].t()  # predictions for positions P .. P+G-1
    topk = torch.topk(logits, args.top_k, dim=-1).indices.to(torch.int32)
    if args.save_logits is not None:
        args.save_logits.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"logits": logits.to(torch.float32).contiguous(), "prompt_len": P, "gen_len": G,
                    "model": MODEL_ID, "dtype": args.dtype}, args.save_logits)
        print(f"Logits saved to: {args.save_logits} {tuple(logits.shape)}", flush=True)
    agree = (topk[:, 0] == torch.tensor(cont_ids, dtype=torch.int32)).float().mean().item()
    print(f"HF top-1 equals the forced continuation token at {agree:.3f} of positions", flush=True)

    eos = tokenizer.eos_token_id
    eos = eos[0] if isinstance(eos, (list, tuple)) else eos
    reference = Reference(
        k=args.top_k,
        hf_model_id=MODEL_ID,
        entries=[
            ReferenceEntry(
                prompt_text=prompt_text,
                prompt_tokens=torch.tensor([prompt_ids], dtype=torch.long),
                generated_tokens=torch.tensor([cont_ids], dtype=torch.long),
                topk_tokens=topk,
                tf_prompt_len=P,
            )
        ],
        token_ids_meta={
            "bos_id": tokenizer.bos_token_id,
            "eos_id": int(eos),
            "pad_id": tokenizer.pad_token_id,
        },
    )
    path = save_reference(reference, args.output)
    meta = {"model": MODEL_ID, "dtype": args.dtype, "prompt_len": P, "gen_len": G, "hf_top1_eq_forced": agree}
    Path(str(path) + ".json").write_text(json.dumps(meta, indent=2))
    print(f"Reference saved to: {path}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
