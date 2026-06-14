# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Run the HF (torch CPU) LocateAnything-3B reference and dump golden tensors.

Outputs a golden .pt with: inputs, projected vision embeds, prefill logits,
last hidden state, and the AR/hybrid generated token sequences + decoded text.
These goldens drive PCC accuracy checks for the tt-nn device port.

Usage:
  python run_reference.py [--image PATH] [--query "person</c>car"] [--out golden.pt]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import la_inputs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None)
    ap.add_argument("--query", default="person</c>car")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "golden.pt"))
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument(
        "--in-token-limit", type=int, default=4096, help="Cap vision tokens for fast bring-up (HF default 25600)."
    )
    ap.add_argument("--gen", action="store_true", help="Also run generate() (slow AR).")
    ap.add_argument(
        "--vision-fp32",
        dest="vision_fp32",
        action="store_true",
        default=True,
        help="Compute vision goldens (vit_*) in fp32 high precision (default on).",
    )
    ap.add_argument(
        "--no-vision-fp32",
        dest="vision_fp32",
        action="store_false",
        help="Keep vision goldens in the model's native bf16 dtype.",
    )
    args = ap.parse_args()

    torch.manual_seed(0)
    mp = la_inputs.find_model_path()
    sys.path.insert(0, mp)
    print(f"[ref] model path: {mp}", flush=True)

    from transformers import AutoConfig, AutoModel, AutoTokenizer

    config = AutoConfig.from_pretrained(mp, trust_remote_code=True)
    # Force sdpa everywhere (magi/flash unavailable here).
    config._attn_implementation = "sdpa"
    config.text_config._attn_implementation = "sdpa"
    config.vision_config._attn_implementation = "sdpa"

    tokenizer = AutoTokenizer.from_pretrained(mp, trust_remote_code=True)
    print("[ref] loading model (bf16, sdpa) ...", flush=True)
    model = AutoModel.from_pretrained(mp, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16).eval()

    img, img_path = la_inputs.load_test_image(args.image)
    print(f"[ref] image: {img_path} size={img.size}", flush=True)
    bundle = la_inputs.build_inputs(tokenizer, img, args.query, in_token_limit=args.in_token_limit)
    input_ids = bundle["input_ids"]
    grid = torch.from_numpy(bundle["image_grid_hws"]).to(torch.int32)
    pixel_values = bundle["pixel_values"].to(torch.bfloat16)
    print(
        f"[ref] grid_hw={bundle['grid_hw']} n_img_tokens={bundle['n_img_tokens']} " f"seq_len={input_ids.shape[1]}",
        flush=True,
    )

    # --- Hooks to capture vision intermediate tensors for incremental PCC debugging ---
    vit_inter = {}

    def _patch_embed_hook(module, inp, out):
        vit_inter["patch_embed"] = out.detach().float().cpu()

    def _encoder_hook(module, inp, out):
        vit_inter["encoder_out"] = out.detach().float().cpu()

    h1 = model.vision_model.patch_embed.register_forward_hook(_patch_embed_hook)
    h2 = model.vision_model.encoder.register_forward_hook(_encoder_hook)

    # Per-image rope freqs_cis (complex) -> dump real/imag for the device cos/sin build.
    rope_freqs_cis = model.vision_model.encoder.rope_2d.get_freqs_cis(grid)  # [L, head_dim/2] complex64
    vit_inter["rope_cos"] = rope_freqs_cis.real.detach().float().cpu()  # [L, head_dim/2]
    vit_inter["rope_sin"] = rope_freqs_cis.imag.detach().float().cpu()  # [L, head_dim/2]

    # PRECISION-FIRST vision golden: the bf16 vision tower accumulates rounding chaos over
    # 27 layers (a perfect fp32 impl only scores ~0.995 PCC vs a bf16 vision golden). To give
    # the tt-nn bf16+HiFi4 port a fair high-precision target, compute the VISION goldens in
    # fp32. The LLM prefill golden below stays in the model's native (bf16) dtype.
    if args.vision_fp32:
        print("[ref] computing vision goldens in fp32 (high-precision target)", flush=True)
        model.vision_model.float()
        model.mlp1.float()
        vision_pixel_values = bundle["pixel_values"].to(torch.float32)
    else:
        vision_pixel_values = pixel_values

    with torch.no_grad():
        # 1) vision tower -> list per image -> cat -> mlp1 projector
        vit_list = model.extract_feature(vision_pixel_values, grid)
        vit_raw = torch.cat(vit_list, dim=0)  # [N, 4608]
        vit_proj = model.mlp1(vit_raw)  # [N, 2048]
        print(f"[ref] vit_raw={tuple(vit_raw.shape)} vit_proj={tuple(vit_proj.shape)}", flush=True)

    h1.remove()
    h2.remove()

    if args.vision_fp32:
        # restore bf16 for the LLM prefill path so its golden is unchanged.
        model.vision_model.to(torch.bfloat16)
        model.mlp1.to(torch.bfloat16)
    print(
        f"[ref] patch_embed={tuple(vit_inter['patch_embed'].shape)} "
        f"encoder_out={tuple(vit_inter['encoder_out'].shape)} "
        f"rope_cos={tuple(vit_inter['rope_cos'].shape)}",
        flush=True,
    )

    # The saved vit_proj golden stays fp32; the LLM prefill needs bf16 to match its embed buffer.
    llm_dtype = model.language_model.get_input_embeddings().weight.dtype
    vit_proj_for_llm = vit_proj.to(llm_dtype) if args.vision_fp32 else vit_proj

    with torch.no_grad():
        # 2) LLM prefill with visual features injected (isolates LLM; matches generate() iter 1)
        out = model.language_model(
            input_ids=input_ids,
            visual_features=vit_proj_for_llm.unsqueeze(0),
            image_token_index=la_inputs.IMAGE_TOKEN_INDEX,
            attention_mask=bundle["attention_mask"],
            use_cache=False,
            output_hidden_states=True,
        )
        prefill_logits = out.logits.float()  # [1,S,V]
        last_hidden = out.hidden_states[-1].float()
        print(
            f"[ref] prefill_logits={tuple(prefill_logits.shape)} " f"argmax_last={int(prefill_logits[0,-1].argmax())}",
            flush=True,
        )

    golden = {
        "model_path": mp,
        "query": args.query,
        "image_path": img_path,
        "in_token_limit": args.in_token_limit,
        "input_ids": input_ids,
        "attention_mask": bundle["attention_mask"],
        "image_grid_hws": bundle["image_grid_hws"],
        "grid_hw": bundle["grid_hw"],
        "n_img_tokens": bundle["n_img_tokens"],
        "pixel_values": bundle["pixel_values"],  # fp32 patches
        "vit_raw": vit_raw.float(),
        "vit_proj": vit_proj.float(),
        # vision intermediate goldens (for incremental tt-nn PCC debugging)
        "vit_patch_embed": vit_inter["patch_embed"],  # [L, hidden] post patch_embed + pos_emb
        "vit_encoder_out": vit_inter["encoder_out"],  # [L, hidden] post 27 blocks + final_layernorm
        "vit_rope_cos": vit_inter["rope_cos"],  # [L, head_dim/2] = Re(freqs_cis)
        "vit_rope_sin": vit_inter["rope_sin"],  # [L, head_dim/2] = Im(freqs_cis)
        "prefill_logits": prefill_logits,
        "last_hidden": last_hidden,
        "image_token_index": la_inputs.IMAGE_TOKEN_INDEX,
    }

    if args.gen:
        print("[ref] generate() slow AR ...", flush=True)
        with torch.no_grad():
            resp = model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=bundle["attention_mask"],
                image_grid_hws=grid,
                tokenizer=tokenizer,
                use_cache=True,
                max_new_tokens=args.max_new_tokens,
                generation_mode="slow",
                temperature=0,
            )
        print(f"[ref] slow response: {resp}", flush=True)
        golden["response_slow"] = resp

    torch.save(golden, args.out)
    print(f"[ref] saved golden -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
