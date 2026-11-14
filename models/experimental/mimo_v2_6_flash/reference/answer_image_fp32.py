# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""What the fp32 reference answers to an image question: the vLLM processor, reference/vision.py and all 48 layers of
reference/model.py on the host, prompt rendered with the checkpoint's chat template (non-thinking), as the server
renders an OpenAI request. Prints the top next tokens per prompt. Use it to tell a TT serving answer that differs
from another provider's apart from what the checkpoint itself says (~25 s per layer with 16 threads).

    python -m models.experimental.mimo_v2_6_flash.reference.answer_image_fp32 \\
        --image solid:220,20,20:256x256 --image circle:220,20,20:448x448 --question "What color is this image? One word."

An ``--image`` is a file path, ``solid:R,G,B:WxH`` or ``circle:R,G,B:WxH`` (a disc on white, as the conformance
suite draws); ``--text-first`` puts the question before the image, as tests/evals/mm_requests.py does.
"""

import argparse
import time

import torch
from loguru import logger
from PIL import Image, ImageDraw

from models.experimental.mimo_v2_6_flash.reference import model as ref
from models.experimental.mimo_v2_6_flash.reference.checkpoint import MiMoCheckpoint, default_model_path
from models.experimental.mimo_v2_6_flash.reference.config import MiMoConfig
from models.experimental.mimo_v2_6_flash.reference.vision import MiMoVisionReference

IMAGE_TOKEN = 151655


def make_image(spec):
    kind, _, rest = spec.partition(":")
    if kind not in ("solid", "circle"):
        return Image.open(spec).convert("RGB")
    color, size = rest.split(":")
    color = tuple(int(c) for c in color.split(","))
    w, h = (int(v) for v in size.split("x"))
    if kind == "solid":
        return Image.new("RGB", (w, h), color)
    img = Image.new("RGB", (w, h), (255, 255, 255))
    ImageDraw.Draw(img).ellipse([w // 5, h // 5, 4 * w // 5, 4 * h // 5], fill=color)
    return img


def main():
    from transformers import AutoConfig, AutoTokenizer
    from vllm.transformers_utils.processors.mimo_v2_omni import MiMoOmniProcessor

    from models.experimental.mimo_v2_6_flash.tt import multimodal as mm

    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default=default_model_path())
    p.add_argument("--image", action="append", required=True)
    p.add_argument("--question", required=True)
    p.add_argument("--text-first", action="store_true")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--top", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=None, help="fewer layers: a smoke run only")
    a = p.parse_args()
    torch.set_num_threads(a.threads)

    tok = AutoTokenizer.from_pretrained(a.model_path, trust_remote_code=True)
    proc = MiMoOmniProcessor.from_hf_config(tok, AutoConfig.from_pretrained(a.model_path, trust_remote_code=True))
    cfg = MiMoConfig.from_model_path(a.model_path)
    ckpt = MiMoCheckpoint(a.model_path, cfg)
    vision = MiMoVisionReference.from_model_path(a.model_path)
    embedding = ckpt.embedding()
    prompts = []
    for spec in a.image:
        parts = [{"type": "text", "text": a.question}, {"type": "image"}]
        messages = [{"role": "user", "content": parts if a.text_first else parts[::-1]}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        out = proc(text=text, images=[make_image(spec)], return_tensors="pt")
        ids = out["input_ids"][0]
        ((offset, length),) = mm.placeholder_rows(ids, IMAGE_TOKEN)
        with torch.no_grad():
            rows = vision(out["pixel_values"].float(), out["image_grid_thw"].reshape(-1, 3))
        embeds = embedding[ids].to(torch.bfloat16)
        embeds[offset : offset + length] = rows.to(torch.bfloat16)  # as the device receives them
        logger.info(f"{spec}: {ids.numel()} prompt tokens, image rows [{offset}, {offset + length})")
        prompts.append((spec, ids, embeds.float()))
    del embedding

    hidden = [e for _, _, e in prompts]
    for layer_idx in range(a.num_layers or cfg.num_hidden_layers):
        t0 = time.time()
        layer = ref.ReferenceDecoderLayer(cfg, layer_idx, ckpt=ckpt)
        with torch.no_grad():
            for i, (_, ids, _) in enumerate(prompts):
                hidden[i] = layer(hidden[i], torch.arange(ids.numel()))
        del layer
        logger.info(f"layer {layer_idx} in {time.time() - t0:.1f}s")
    norm, lm_head = ckpt.final_norm(), ckpt.lm_head().float()
    with torch.no_grad():
        for i, (spec, _, _) in enumerate(prompts):
            top = (ref.rms_norm(hidden[i][-1:], norm, cfg.layernorm_epsilon) @ lm_head.T)[-1].topk(a.top)
            print(f"{spec}: " + ", ".join(f"{tok.decode([int(t)])!r} {float(v):.2f}" for v, t in zip(*top)), flush=True)


if __name__ == "__main__":
    main()
