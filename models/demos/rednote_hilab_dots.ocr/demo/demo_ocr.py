# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""End-to-end OCR demo for rednote-hilab/dots.ocr (image -> text) on Tenstorrent.

Loads a document image, runs the host-side DotsVL preprocessing (Qwen2-VL image
patchify + chat-prompt construction), then runs the TTNN OCR pipeline
(:class:`tt.ocr_model.TtOcrModel`) and decodes the generated tokens to text.

The demo can ALSO run the HuggingFace reference path -- the real
``DotsOCRForCausalLM`` (loaded via ``transformers.AutoModelForCausalLM`` with
``trust_remote_code=True``) -- for a side-by-side parity comparison. Pass
``--reference`` to print the HF output too.

Usage::

    python demo_ocr.py --image demo/demo_image1.jpg --prompt "Read the text." \
        --lm-layers 2 --vision-layers 2 --max-new-tokens 16 --reference

Notes:
- The HF preprocessing (resize/patchify into pixel_values + image_grid_thw, and
  the tokenizer chat-template prompt with the <|imgpad|> (151665) expansion) and
  the vision->text masked_scatter glue are the documented host-resident ops
  (use_case.hybrid_notes). Everything heavy (vision trunk + Qwen2 LM trunk) runs
  on device via the composed TTNN component modules.
- Full config is 28 LM layers + 42 vision layers. That full-depth run is heavy;
  the demo defaults to a representative reduced depth (--lm-layers / --vision-layers)
  so the AR loop + scatter + decode wiring can be exercised quickly, and the HF
  reference is loaded at the SAME reduced depth for an apples-to-apples compare.
"""
import argparse
import importlib.util
import os

import torch

import ttnn

_HERE = os.path.dirname(os.path.abspath(__file__))
_TT_DIR = os.path.normpath(os.path.join(_HERE, "..", "tt"))

CHECKPOINT_PATH = os.environ.get(
    "DOTS_OCR_CHECKPOINT",
    "/local/ttuser/.cache/huggingface/hub/models--rednote-hilab--dots.ocr/snapshots/"
    "c0111ce6bc07803dbc267932ffef0ae3a51dc951",
)
IMAGE_TOKEN_ID = 151665  # <|imgpad|>
EOS_TOKEN_IDS = (151643, 151673)  # generation_config.eos_token_id


def _load_by_path(name, filename, directory):
    spec = importlib.util.spec_from_file_location(name, os.path.join(directory, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def host_preprocess(image_path: str, prompt: str, checkpoint_path: str = CHECKPOINT_PATH):
    """DotsVL host preprocessing: image -> (input_ids, pixel_values, grid_thw, tokenizer).

    Uses the Qwen2-VL image processor (patchify) + the tokenizer to build the
    chat prompt with the <|imgpad|> (151665) image-token expansion sized to the
    merged patch count. This mirrors what ``DotsVLProcessor`` does (its base
    Qwen2_5_VLProcessor wiring is version-fragile, so the image processor and
    tokenizer are driven directly -- functionally identical preprocessing).
    """
    from PIL import Image
    from transformers import AutoImageProcessor, AutoTokenizer

    img = Image.open(image_path).convert("RGB")
    img_proc = AutoImageProcessor.from_pretrained(checkpoint_path, use_fast=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    enc = img_proc(images=[img], return_tensors="pt")
    pixel_values = enc["pixel_values"]
    grid_thw = enc["image_grid_thw"]  # [num_images, 3]
    t, h, w = grid_thw[0].tolist()
    merge = 2
    n_img_tokens = (t * h * w) // (merge * merge)

    img_block = "<|vision_start|>" + "<|imgpad|>" * n_img_tokens + "<|vision_end|>"
    full_prompt = f"<|im_start|>user\n{img_block}{prompt}<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tokenizer(full_prompt, return_tensors="pt")["input_ids"]
    assert int((input_ids == IMAGE_TOKEN_ID).sum()) == n_img_tokens
    return input_ids, pixel_values, grid_thw, tokenizer


def run_ttnn(device, input_ids, pixel_values, grid_thw, lm_layers, vision_layers, max_new_tokens):
    """Run the TTNN OCR pipeline and return (generated_ids, text)."""
    loader = _load_by_path("dots_demo_loader", "weight_loader.py", _TT_DIR)
    ocr_model_mod = _load_by_path("dots_demo_ocr_model", "ocr_model.py", _TT_DIR)

    lm_sd = loader.load_language_model_weights(CHECKPOINT_PATH, num_layers=lm_layers)
    vis_sd = loader.load_vision_tower_weights(CHECKPOINT_PATH, num_layers=vision_layers)

    model = ocr_model_mod.TtOcrModel(
        device=device,
        lm_state_dict=lm_sd,
        vision_state_dict=vis_sd,
        grid_thw=grid_thw,
        lm_num_layers=lm_layers,
        vision_num_layers=vision_layers,
    )
    gen = model.generate(
        input_ids,
        pixel_values.to(torch.float32),
        max_new_tokens=max_new_tokens,
        eos_token_ids=EOS_TOKEN_IDS,
    )
    return gen


def run_hf_reference(input_ids, pixel_values, grid_thw, lm_layers, vision_layers, max_new_tokens):
    """Run the HF reference DotsOCRForCausalLM at the SAME reduced depth.

    Loads the real ``DotsOCRForCausalLM`` (trust_remote_code) with the LM/vision
    layer counts truncated to match the TTNN run, then greedy-decodes. Returns
    the list of generated token ids (the new tokens only).
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(CHECKPOINT_PATH, trust_remote_code=True)
    cfg.num_hidden_layers = lm_layers
    if isinstance(cfg.vision_config, dict):
        cfg.vision_config["num_hidden_layers"] = vision_layers
    else:
        cfg.vision_config.num_hidden_layers = vision_layers

    # The vision tower's forward casts pixel_values to bf16 by default
    # (DotsVisionTransformer.forward(..., bf16=True)); load the whole model in
    # bf16 so the Conv2d patchify weights match the bf16 pixel cast. bf16 is the
    # production inference dtype and is consistent with the TTNN bf16 path.
    model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT_PATH, config=cfg, trust_remote_code=True, dtype=torch.bfloat16
    )
    model.eval()

    ids = input_ids.clone()
    generated = []
    eos = set(EOS_TOKEN_IDS)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(
                input_ids=ids,
                pixel_values=pixel_values.to(torch.bfloat16),
                image_grid_thw=grid_thw,
                use_cache=False,
            )
            logits = out.logits if hasattr(out, "logits") else out[0]
            next_id = int(torch.argmax(logits[0, -1]).item())
            generated.append(next_id)
            ids = torch.cat([ids, torch.tensor([[next_id]], dtype=ids.dtype)], dim=1)
            if next_id in eos:
                break
    return generated


def main():
    ap = argparse.ArgumentParser(description="dots.ocr OCR demo (image -> text)")
    ap.add_argument("--image", default=os.path.join(_HERE, "demo_image1.jpg"))
    ap.add_argument("--prompt", default="Read the text in the image.")
    ap.add_argument("--lm-layers", type=int, default=2)
    ap.add_argument("--vision-layers", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--reference", action="store_true", help="also run the HF DotsOCRForCausalLM reference")
    args = ap.parse_args()

    input_ids, pixel_values, grid_thw, tokenizer = host_preprocess(args.image, args.prompt)
    print(f"input image: {args.image} | grid_thw={grid_thw.tolist()} | seq_len={input_ids.shape[1]}")

    device = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        tt_ids = run_ttnn(
            device, input_ids, pixel_values, grid_thw, args.lm_layers, args.vision_layers, args.max_new_tokens
        )
    finally:
        ttnn.close_device(device)
    tt_text = tokenizer.decode(tt_ids, skip_special_tokens=True)
    print(f"[TTNN]      tokens={tt_ids}")
    print(f"[TTNN]      text={tt_text!r}")

    if args.reference:
        hf_ids = run_hf_reference(
            input_ids, pixel_values, grid_thw, args.lm_layers, args.vision_layers, args.max_new_tokens
        )
        hf_text = tokenizer.decode(hf_ids, skip_special_tokens=True)
        print(f"[HF ref]    tokens={hf_ids}")
        print(f"[HF ref]    text={hf_text!r}")
        match = sum(1 for a, b in zip(tt_ids, hf_ids) if a == b)
        denom = max(len(tt_ids), len(hf_ids), 1)
        print(f"[parity]    token accuracy vs HF = {match/denom:.4f}")


if __name__ == "__main__":
    main()
