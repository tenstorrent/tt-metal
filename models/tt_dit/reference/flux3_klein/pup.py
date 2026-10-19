# SPDX-FileCopyrightText: © 2026 Black Forest Labs, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from https://github.com/black-forest-labs/flux-3-image-klein (src/flux3/pup.py) at e53f4ea,
# licensed under Apache-2.0. Modified by Tenstorrent as the reference of the prompt upsampler port.

"""Local prompt upsampling on vLLM, one joint model. No reference -> t2i prompt, with references -> i2i prompt;
with boxes, their layout variants.

  python -m flux3.pup --prompt "..." [--image a.png ...] [--boxes boxes.json] --out caption.txt
"""

import argparse
import io
import json
import math
import os
from pathlib import Path

from PIL import Image, ImageOps

from . import bbox

WEIGHTS = Path(os.environ.get("WEIGHTS_DIR", "weights"))
MODEL = str(WEIGHTS / "pup")
PROMPTS = Path(__file__).with_name("pup_prompts.json")  # fallback; the weights ship them as system_prompts.json
STOP_TOKENS = ("<|im_end|>", "<|endoftext|>")
MAX_REFERENCES = 10

ENGINE = {"max_model_len": 8192, "max_num_seqs": 16}
SAMPLING = {"temperature": 0.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "max_tokens": 3584}
PRESENCE_PENALTY = {"t2i": 1.5, "i2i": 0.0}
I2I_SCAFFOLD = {
    False: (
        "Reference images follow in order (the first image, the second image, ...).\n"
        "User request: {instruction}\n"
        "Output the caption."
    ),
    True: (
        "Reference images follow in attachment order: ref_image_0, ref_image_1, ...\n"
        "User editing instruction: {instruction}\n"
        "Decide the edit, imagine the target image, and output the annotation record."
    ),
}
REFERENCE_MULTIPLE = 32
MAX_ASPECT_RATIO = 200
MAX_PIXELS_BY_REFS = {1: 1024 * 1024, 2: 724 * 724, **{n: 512 * 512 for n in range(3, MAX_REFERENCES + 1)}}
JPEG_QUALITY = 92


def reference_size(width: int, height: int, n_refs: int) -> tuple[int, int]:
    if max(width, height) / min(width, height) > MAX_ASPECT_RATIO:
        raise ValueError(f"aspect ratio of {width}x{height} exceeds {MAX_ASPECT_RATIO}:1")
    floor = lambda v: max(REFERENCE_MULTIPLE, int(math.floor(v / REFERENCE_MULTIPLE)) * REFERENCE_MULTIPLE)
    cap = MAX_PIXELS_BY_REFS[n_refs]
    w = max(REFERENCE_MULTIPLE, round(width / REFERENCE_MULTIPLE) * REFERENCE_MULTIPLE)
    h = max(REFERENCE_MULTIPLE, round(height / REFERENCE_MULTIPLE) * REFERENCE_MULTIPLE)
    if w * h > cap:
        beta = math.sqrt(width * height / cap)
        w, h = floor(width / beta), floor(height / beta)
    if w * h > cap:
        if w == REFERENCE_MULTIPLE:
            h = floor(cap / w)
        else:
            w = floor(cap / h)
    return w, h


def policy_reference(image: Image.Image, n_refs: int) -> Image.Image:
    """Reference preprocessing of the upsampler: RGB over white, LANCZOS to the size budget, JPEG q92."""
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        image = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba)
    image = image.convert("RGB")
    size = reference_size(image.width, image.height, n_refs)
    if image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")


class Upsampler:
    """Loads the model once. ``memory`` is vLLM's share of the GPU, the rest is left to the DiT."""

    def __init__(self, model: str = MODEL, memory: float = 0.35):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")  # DeepGEMM JIT-compiles: would need nvcc
        from transformers import AutoProcessor
        from vllm import LLM

        self.llm = LLM(
            model=model,
            dtype="bfloat16",
            tensor_parallel_size=1,
            enable_prefix_caching=True,
            generation_config="vllm",
            limit_mm_per_prompt={"image": MAX_REFERENCES},
            attention_config={"backend": "TRITON_ATTN"},  # no JIT: FlashInfer would need ninja + nvcc
            gpu_memory_utilization=memory,
            **ENGINE,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.processor = AutoProcessor.from_pretrained(model, local_files_only=True)
        shipped = Path(model) / "system_prompts.json"
        self.prompts = json.loads((shipped if shipped.exists() else PROMPTS).read_text())

    def __call__(self, prompt: str, images: list[Image.Image], boxes: list[dict] | None = None) -> str:
        from vllm import SamplingParams

        if len(images) > MAX_REFERENCES:
            raise ValueError(f"the upsampler takes at most {MAX_REFERENCES} references")
        task = "i2i" if images else "t2i"
        system_key, prompt, locked = bbox.plan(prompt, boxes, len(images))
        params = SamplingParams(
            **SAMPLING,
            presence_penalty=PRESENCE_PENALTY[task],
            stop_token_ids=[self.tokenizer.convert_tokens_to_ids(t) for t in STOP_TOKENS],
        )
        if images:
            references = [policy_reference(image, len(images)) for image in images]
            scaffold = I2I_SCAFFOLD[locked is not None].format(instruction=prompt)
            content = [{"type": "image"} for _ in references] + [{"type": "text", "text": scaffold}]
            messages = [{"role": "system", "content": self.prompts[system_key]}, {"role": "user", "content": content}]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            request = {"prompt": text, "multi_modal_data": {"image": references}}
        else:
            messages = [{"role": "system", "content": self.prompts[system_key]}, {"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            request = {"prompt_token_ids": self.tokenizer.encode(text, add_special_tokens=False)}
        caption = self.llm.generate([request], params, use_tqdm=False)[0].outputs[0].text.strip()
        if not caption:
            raise RuntimeError("empty generation")
        return bbox.reconcile(caption, locked) if locked else caption


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--image", action="append", default=[], help=f"reference image; repeat for several (max {MAX_REFERENCES})"
    )
    parser.add_argument("--boxes", help="JSON list of boxes, see README")
    parser.add_argument("--out", help="write the caption here (stdout carries vLLM logs)")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--memory", type=float, default=0.85, help="vLLM's share of the GPU")
    args = parser.parse_args()
    boxes = json.loads(Path(args.boxes).read_text()) if args.boxes else None
    caption = Upsampler(args.model, args.memory)(args.prompt, [Image.open(p) for p in args.image], boxes)
    if args.out:
        Path(args.out).write_text(caption + "\n")
    print(f"\n{caption}", flush=True)


if __name__ == "__main__":
    main()
