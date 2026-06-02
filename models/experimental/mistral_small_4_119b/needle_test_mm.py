# SPDX-License-Identifier: Apache-2.0
"""Long-context MULTIMODAL quality check (image-needle-in-a-text-haystack).

Puts the image at the START, a long text filler (haystack) after it, then asks a
question ABOUT THE IMAGE at the end. Validates that image-token information survives
chunked prefill across many chunk boundaries — the vision+language long-context path
that the text-only needle test doesn't cover.

  MESH_DEVICE=P150x8 NEEDLE_TOKENS=16000 MISTRAL4_WEIGHT_CACHE_DIR=... \
    python models/experimental/mistral_small_4_119b/needle_test_mm.py --image Battle.jpg
"""
import argparse
import os

import torch
import ttnn
from loguru import logger

from models.experimental.mistral_small_4_119b.constants import HF_MODEL_ID
from models.experimental.mistral_small_4_119b.demo_multimodal import (
    _build_chat_inputs,
    _open_mesh_device,
    _state_dict_prefixes,
    generate,
)
from models.experimental.mistral_small_4_119b.tt.mistral3_for_conditional_generation_unified import (
    TtMistral3ForConditionalGenerationUnified,
)
from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict_filtered

TARGET_TEXT_TOKENS = int(os.environ.get("NEEDLE_TOKENS", "16000"))
# The image is a Pokémon battle (Pikachu vs Pidgey) — retrieval keywords.
KEYWORDS = [k.lower() for k in os.environ.get("NEEDLE_KEYWORDS", "pikachu,pidgey,pokémon,pokemon,battle").split(",")]


def _long_prompt(tokenizer) -> str:
    filler = "".join(
        f"Log entry {i}: routine system check {i} completed, status nominal, value {i * 7 % 1000}.\n"
        for i in range(1, 40000)
    )
    ids = tokenizer(filler, return_tensors="pt").input_ids[0]
    if ids.shape[0] > TARGET_TEXT_TOKENS:
        filler = tokenizer.decode(ids[:TARGET_TEXT_TOKENS], skip_special_tokens=True)
    return (
        "Below are some irrelevant logs, then a question about the IMAGE shown above.\n\n"
        + filler
        + "\n\nNow ignore the logs entirely and look only at the image. "
        "What two creatures are shown, and what is happening? Name them."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="Battle.jpg")
    ap.add_argument("--image-max-side", type=int, default=512)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--n-text-layers", type=int, default=36)
    ap.add_argument("--n-vision-layers", type=int, default=24)
    args = ap.parse_args()

    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.mistral4.modeling_mistral4 import Mistral4RotaryEmbedding

    cfg = AutoConfig.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
    text_cfg = cfg.text_config
    image_token_id = int(getattr(cfg, "image_token_index", 10))
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)

    prompt = _long_prompt(tokenizer)
    pixel_values, input_ids, _ = _build_chat_inputs(args.image, prompt, args.image_max_side)
    seq_len = input_ids.shape[-1]
    n_img = int((input_ids[0] == image_token_id).sum().item())
    logger.info(f"MM needle: {seq_len} tokens ({n_img} image + {seq_len - n_img} text), image={args.image}")

    state_dict = load_hf_state_dict_filtered(
        HF_MODEL_ID, _state_dict_prefixes(args.n_text_layers, args.n_vision_layers)
    )
    mesh = _open_mesh_device()
    try:
        max_seq_len = seq_len + args.max_new_tokens + 16
        model = TtMistral3ForConditionalGenerationUnified(
            mesh_device=mesh,
            state_dict=state_dict,
            text_config=text_cfg,
            image_token_id=image_token_id,
            num_text_layers=args.n_text_layers,
            num_vision_layers=args.n_vision_layers,
            max_seq_len=max_seq_len,
            vision_dtype=ttnn.bfloat8_b,
        )
        with torch.no_grad():
            out = generate(
                model=model,
                tokenizer=tokenizer,
                rotary_cls=Mistral4RotaryEmbedding,
                text_config=text_cfg,
                pixel_values=pixel_values,
                input_ids=input_ids,
                prompt="<image> + long filler + question about the image",
                image_token_id=image_token_id,
                max_new_tokens=args.max_new_tokens,
            )
        hit = [k for k in KEYWORDS if k in out.lower()]
        logger.info("=" * 60)
        logger.info(f"MM needle @ {seq_len} tok — answer mentions image content: {hit if hit else 'NONE'}")
        logger.info(f"{'✅ PASS — image retrieved after long text' if hit else '❌ FAIL — image content lost'}")
        logger.info("=" * 60)
    finally:
        ttnn.close_mesh_device(mesh)
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
