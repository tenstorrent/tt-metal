#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Compare TT decode dumps (TT_DECODE_HF_REF=1) to Hugging Face Gemma3 on CPU.

1) On device, run text_demo / pytest with trace off, e.g.:
     export TT_DECODE_HF_REF=1
     export TT_DECODE_HF_REF_OUT=/tmp/gemma_hf_ref
     # optional: TT_DECODE_HF_REF_STEPS=1,2,3  TT_DECODE_HF_REF_LAYERS=0  TT_DECODE_HF_REF_LM_HEAD=1

2) On CPU (from tt-metal repo root):
     python models/demos/multimodal/gemma3/scripts/compare_decode_hf_ref.py \\
       --dump-dir /tmp/gemma_hf_ref \\
       --model google/gemma-3-4b-it \\
       --prompt \"Your exact user prompt string (same as demo input)\"

Use the same --instruct / --no-instruct as the demo (default: instruct on).
"""
import argparse
import glob
import os
import re
import sys

import torch


def _tt_metal_root() -> str:
    # scripts -> gemma3 -> multimodal -> demos -> models -> tt-metal
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(here)))))


def _meta_step(path: str) -> int:
    m = re.search(r"meta_step(\d+)\.pt$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def _report(name: str, tt: torch.Tensor, hf: torch.Tensor) -> None:
    a = tt.detach().float().reshape(-1)
    b = hf.detach().float().reshape(-1)
    if a.numel() != b.numel():
        print(f"  {name}: size mismatch tt={a.numel()} hf={b.numel()}")
        return
    diff = a - b
    rmse = diff.pow(2).mean().sqrt().item()
    denom = a.norm() * b.norm() + 1e-12
    cos = (a @ b / denom).item()
    print(f"  {name}: rmse={rmse:.6f}  cos_sim={cos:.6f}  max_abs={diff.abs().max().item():.4f}")


def main() -> None:
    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump-dir", required=True, help="TT_DECODE_HF_REF_OUT directory")
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument(
        "--prompt",
        required=True,
        help="Raw user prompt (same string as text_demo input; chat template applied if --instruct)",
    )
    parser.add_argument("--instruct", action="store_true", default=True)
    parser.add_argument("--no-instruct", dest="instruct", action="store_false")
    args = parser.parse_args()

    root = os.path.realpath(_tt_metal_root())
    sys.path.insert(0, root)

    from models.tt_transformers.tt.common import encode_prompt_hf

    dump_dir = os.path.abspath(args.dump_dir)
    meta_files = sorted(glob.glob(os.path.join(dump_dir, "meta_step*.pt")), key=_meta_step)
    if not meta_files:
        print(f"No meta_step*.pt under {dump_dir}. Run TT with TT_DECODE_HF_REF=1 first.", file=sys.stderr)
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=os.getenv("CI") == "true")
    if args.instruct:
        prompt_ids = encode_prompt_hf(tokenizer, args.prompt)
    else:
        prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    ids = torch.tensor([prompt_ids], dtype=torch.long)
    decoding_pos_len = ids.shape[1]

    print(f"Prompt token len (decoding_pos expected for step 1): {decoding_pos_len}")

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        local_files_only=os.getenv("CI") == "true",
    )
    model.eval()

    l0_attn = model.language_model.layers[0].self_attn
    captured: dict = {}

    def pre_o_hook(_m, inputs):
        captured["pre_o_proj"] = inputs[0].detach().float().cpu().clone()

    hook_handle = l0_attn.o_proj.register_forward_pre_hook(pre_o_hook)

    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
        past = out.past_key_values

    try:
        for mf in meta_files:
            step = _meta_step(mf)
            meta = torch.load(mf, map_location="cpu")
            tok = meta["tokens"].view(-1)[0].long()
            pos = meta["start_pos"].view(-1)[0].long()
            print(f"\n=== decode step {step} (meta) token={tok.item()} pos={pos.item()} ===")
            if step == 1 and pos.item() != decoding_pos_len:
                print(
                    f"  NOTE: pos {pos.item()} != prompt len {decoding_pos_len} — prompt mismatch vs TT run?",
                    file=sys.stderr,
                )

            captured.clear()
            with torch.no_grad():
                out = model(
                    input_ids=tok.view(1, 1),
                    past_key_values=past,
                    cache_position=pos.view(1),
                    use_cache=True,
                )
            past = out.past_key_values

            if "pre_o_proj" not in captured:
                print("  ERROR: hook did not capture pre_o_proj")
                continue

            hf_pre = captured["pre_o_proj"].squeeze(0).squeeze(0)

            tt_pre_path = os.path.join(dump_dir, f"tt_step_{step:04d}_layer0_attn_post_sdpa_concat.pt")
            if os.path.isfile(tt_pre_path):
                tt_blob = torch.load(tt_pre_path, map_location="cpu")
                tt_pre = tt_blob["tensor_b0"].float()
                _report("attn_post_sdpa_concat (L0 pre o_proj)", tt_pre, hf_pre)
            else:
                print(f"  (skip) no {tt_pre_path}")

            tt_wo_path = os.path.join(dump_dir, f"tt_step_{step:04d}_layer0_attn_out_post_wo.pt")
            if os.path.isfile(tt_wo_path):
                with torch.no_grad():
                    hf_post = l0_attn.o_proj(hf_pre.view(1, 1, -1).to(model.dtype)).float().view(-1)
                tt_post = torch.load(tt_wo_path, map_location="cpu")["tensor_b0"].float()
                _report("attn_out_post_wo (L0)", tt_post, hf_post)

            tt_lm_path = os.path.join(dump_dir, f"tt_step_{step:04d}_layerNA_lm_head_out.pt")
            if os.path.isfile(tt_lm_path):
                hf_logits = out.logits[0, -1, :].float().cpu()
                tt_logits = torch.load(tt_lm_path, map_location="cpu")["tensor_b0"].float()
                _report("lm_head_out (last logits)", tt_logits, hf_logits)
    finally:
        hook_handle.remove()


if __name__ == "__main__":
    main()
