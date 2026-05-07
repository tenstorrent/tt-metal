#!/usr/bin/env python3
"""
Multimodal (image + text) inference demo for Mistral-Small-4-119B-2603.

Use `--reference-torch` to load the vendored PyTorch reference in
`reference/model.py` (Mistral3ForConditionalGeneration with optional @instrument).

Designed for bifrost server:
  - GPU: RTX 3080, 10 GB VRAM
  - RAM: ~64 GB

Quantization options (--quant flag):
  --quant 4   INT4/NF4   ~60 GB   ← DEFAULT  (10 GB GPU + 50 GB CPU)
  --quant 8   INT8       ~119 GB  (10 GB GPU + ~60 GB CPU + disk offload)
  --quant none BF16/FP16 ~238 GB  ← WARNING: will NOT fit on bifrost RAM

Weights live under --model-dir after download; reruns load from disk (no weight re-download).
Optional: MISTRAL_LOAD_LOCAL_ONLY=1 to forbid Hugging Face Hub calls during load.

Usage:
    python demo_multimodal.py --model-dir ./models/mistral_small_4 --image /path/to/image.jpg
    python demo_multimodal.py --model-dir ./models/mistral_small_4 --image /path/to/image.jpg --quant 8
    python demo_multimodal.py --model-dir ./models/mistral_small_4 --image-url https://...
    python demo_multimodal.py --model-dir ./models/mistral_small_4
        # default: bundled images/test1.png (no network; avoids blocked sample URLs)
"""

import argparse
import gc
import io
import sys
import time
from pathlib import Path

import requests
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, TextStreamer

from models.demos.mistral_small_4_119B.bnb_dispatch import (
    accelerate_max_memory,
    build_bnb_config_and_labels,
    cli_quant_label,
    cuda_available,
    resolve_effective_quant,
)
from models.demos.mistral_small_4_119B.config_patches import (
    patch_model_config_for_quantization,
    patch_tokenizer_config,
    restore_fp8_quantization_config_if_missing,
    sync_generation_config_with_text_config,
)
from models.demos.mistral_small_4_119B.env_check import ensure_transformers_has_mistral4
from models.demos.mistral_small_4_119B.mistral_moe_fp8_materialize import materialize_mistral_moe_fp8_experts_from_disk
from models.demos.mistral_small_4_119B.moe_grouped_mm_patch import apply_moe_grouped_mm_fp8_fallback_fix
from models.demos.mistral_small_4_119B.paths import DEFAULT_MODEL_DIR, from_pretrained_local_kw, resolve_model_dir
from models.demos.mistral_small_4_119B.processor_layout import ensure_preprocessor_config_json

apply_moe_grouped_mm_fp8_fallback_fix()

# Shipped with the demo so ``python demo_multimodal.py`` works offline (no 403 from hotlinking).
_BUNDLED_SAMPLE_IMAGE = Path(__file__).resolve().parent / "images" / "test_2.png"
_DEFAULT_QUESTION = "Explain the image in about 5 lines."


# Memory footprint per mode
_QUANT_INFO = {
    "4": "INT4 NF4 on GPU; **without CUDA** → auto BF16 on CPU (likely OOM for 119B)",
    "8": "INT8 on GPU; **without CUDA** → auto BF16 on CPU (likely OOM for 119B)",
    "none": "BF16 on GPU with offload, or BF16 on CPU if no CUDA (huge RAM)",
}


def load_model(model_dir: str, quant: str, reference_torch: bool = False):
    assert quant in _QUANT_INFO, f"--quant must be one of: {list(_QUANT_INFO)}"

    ensure_transformers_has_mistral4()

    model_dir = str(resolve_model_dir(model_dir))

    if cuda_available():
        props = torch.cuda.get_device_properties(0)
        print(
            f"  [cuda] device=0  name={props.name}  "
            f"total_mem={props.total_memory / (1024**3):.1f} GiB  "
            f"multi_processor_count={props.multi_processor_count}"
        )
    else:
        print("  [cuda] No CUDA GPU visible — INT4/INT8 will fall back to BF16 on CPU (see [quantize] lines below).")

    _local_kw = from_pretrained_local_kw()
    if _local_kw:
        print("  [cache] MISTRAL_LOAD_LOCAL_ONLY — load from disk only (no Hub); needs a complete snapshot.")

    patch_tokenizer_config(model_dir)
    ensure_preprocessor_config_json(model_dir)

    if reference_torch:
        from models.demos.mistral_small_4_119B.reference.model import Mistral3ForConditionalGeneration

        model_cls = Mistral3ForConditionalGeneration
        print("  [reference-torch] Using vendored Mistral3ForConditionalGeneration (see reference/model.py)")
    else:
        model_cls = AutoModelForImageTextToText

    print(f"Loading processor from {model_dir} ...")
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True, **_local_kw)

    print(f"Loading multimodal model  (--quant {quant})  {_QUANT_INFO[quant]}")
    print("  Weights are read from the local --model-dir (no re-download once snapshot is complete).")

    effective_quant, quant_fallback_msg = resolve_effective_quant(quant)
    if quant_fallback_msg:
        print(quant_fallback_msg)

    bnb_cfg, _, label_effective = build_bnb_config_and_labels(effective_quant)
    print(f"  [quantize] CLI request : {cli_quant_label(quant)}")
    print(f"  [quantize] Effective : --quant {effective_quant} → {label_effective}")

    patch_model_config_for_quantization(model_dir, strip_fp8_for_bnb=bnb_cfg is not None)
    if bnb_cfg is None:
        restore_fp8_quantization_config_if_missing(
            model_dir,
            local_files_only=_local_kw.get("local_files_only", False),
        )

    load_torch_dtype = torch.float32 if (bnb_cfg is None and not cuda_available()) else torch.bfloat16

    if bnb_cfg is None:
        print("  WARNING: --quant none needs ~238 GiB RAM/VRAM for this checkpoint.")
        if cuda_available():
            model = model_cls.from_pretrained(
                model_dir,
                dtype=load_torch_dtype,
                device_map="auto",
                max_memory=accelerate_max_memory(),
                offload_folder="./offload",
                offload_state_dict=True,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                **_local_kw,
            )
        else:
            print("  [device] float32 on CPU only (no GPU); expect OOM unless you have hundreds of GiB RAM.")
            model = model_cls.from_pretrained(
                model_dir,
                dtype=load_torch_dtype,
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                **_local_kw,
            )
    else:
        print("  First INT8/INT4 load can take a long time while bitsandbytes prepares weights.")
        model = model_cls.from_pretrained(
            model_dir,
            quantization_config=bnb_cfg,
            device_map="auto",
            max_memory=accelerate_max_memory(),
            offload_folder="./offload",
            offload_state_dict=True,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            **_local_kw,
        )

    model.eval()
    try:
        if materialize_mistral_moe_fp8_experts_from_disk(model, model_dir):
            print("  [patched] MoE routed experts dequantized from safetensors (fixes CPU garbage logits).")
    except Exception as exc:
        print(f"  [warn] MoE FP8 expert materialize skipped: {exc}")
    sync_generation_config_with_text_config(model)
    print("Model loaded.\n")
    return processor, model


def load_image(path_or_url: str) -> Image.Image:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        resp = requests.get(path_or_url, timeout=15)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    return Image.open(path_or_url).convert("RGB")


def run_inference(processor, model, image: Image.Image, question: str, max_new_tokens: int = 256):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(images=image, text=text, return_tensors="pt")

    first_device = next(model.parameters()).device
    inputs = {k: v.to(first_device) if hasattr(v, "to") else v for k, v in inputs.items()}

    streamer = TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)

    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}")
    print("Response:")

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            streamer=streamer,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0

    n_new = output_ids.shape[-1] - inputs["input_ids"].shape[-1]
    print(f"\n[Generated {n_new} tokens in {elapsed:.1f}s — {n_new/elapsed:.1f} tok/s]")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help=f"Path to downloaded weights (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument("--image", default=None, help="Local image file path")
    parser.add_argument("--image-url", default=None, help="Image URL")
    parser.add_argument(
        "--question",
        default=_DEFAULT_QUESTION,
        help="Question to ask about the image",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--quant",
        choices=["4", "8", "none"],
        default="4",
        help="Quantization: 4=INT4/NF4 (~60 GB, default), 8=INT8 (~119 GB), none=BF16 (~238 GB)",
    )
    parser.add_argument(
        "--reference-torch",
        action="store_true",
        help="Load Mistral3ForConditionalGeneration from reference/model.py instead of AutoModel",
    )
    args = parser.parse_args()

    processor, model = load_model(
        model_dir=args.model_dir,
        quant=args.quant,
        reference_torch=args.reference_torch,
    )

    if args.image:
        img_source = args.image
    elif args.image_url:
        img_source = args.image_url
    else:
        if not _BUNDLED_SAMPLE_IMAGE.is_file():
            print(
                f"Error: default sample image not found: {_BUNDLED_SAMPLE_IMAGE}\n"
                "  Pass --image PATH or --image-url URL.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        img_source = str(_BUNDLED_SAMPLE_IMAGE)
        print(f"No image specified. Using bundled sample: {img_source}")

    image = load_image(img_source)
    print(f"Image loaded: {image.size} {image.mode}")

    run_inference(processor, model, image, args.question, args.max_new_tokens)


if __name__ == "__main__":
    main()
