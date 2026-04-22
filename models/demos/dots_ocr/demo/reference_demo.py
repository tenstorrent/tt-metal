# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Dots OCR reference demo (HF / torch only).

Purpose
-------
Provide a simple reference-only CLI that:
- accepts an image *or* a document (PDF),
- runs HF `rednote-hilab/dots.mocr`,
- prints extracted text,
- saves the actual image fed to the model.

This is intentionally device-free (no TTNN).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from models.demos.dots_ocr.reference.hf_utils import HFLoadSpec, get_hf_model_id
from models.demos.dots_ocr.reference.model import DotsOCRReference


def _load_image_from_path(path: Path, *, pdf_page_index: int = 0) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(str(path))

    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
        return Image.open(path).convert("RGB")

    if path.suffix.lower() == ".pdf":
        try:
            import fitz  # type: ignore
        except Exception as e:
            raise RuntimeError("PDF input requires PyMuPDF. Install with `pip install pymupdf` and retry.") from e

        doc = fitz.open(str(path))
        if len(doc) == 0:
            raise ValueError(f"PDF has no pages: {path}")
        page_index = int(pdf_page_index)
        if page_index < 0 or page_index >= len(doc):
            raise ValueError(f"pdf_page_index out of range: {page_index} (pages={len(doc)})")
        page = doc.load_page(page_index)
        pix = page.get_pixmap()  # default resolution; adjust via Matrix() if needed
        mode = "RGBA" if pix.alpha else "RGB"
        img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
        return img.convert("RGB")

    raise ValueError(f"Unsupported input type: {path.suffix} (expected image or .pdf)")


def main() -> None:
    p = argparse.ArgumentParser(description="Dots OCR reference demo (HF)")
    p.add_argument("--input", required=True, help="Path to image or PDF document")
    p.add_argument(
        "--prompt",
        default="OCR: transcribe the text in the image exactly.",
        help="OCR prompt",
    )
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--hf-model", type=str, default=None, help="Override HF model id (default: HF_MODEL or dots.mocr)")
    p.add_argument("--out-dir", type=str, default="dots_ocr_reference_out", help="Where to write outputs")
    p.add_argument("--pdf-page", type=int, default=0, help="PDF only: 0-based page index to render")
    p.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Passed to generate(); helps avoid repeated phrases (set 1.0 to disable).",
    )
    p.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=3,
        help="Passed to generate(); prevents repeating 4-grams (set 0 to disable).",
    )
    p.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling (off by default for deterministic OCR).",
    )
    p.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (used only with --do-sample).")
    p.add_argument("--num-beams", type=int, default=1, help="Beam search width (1 = greedy).")
    p.add_argument(
        "--dtype",
        type=str,
        default="fp32",
        choices=["fp32", "bf16"],
        help="Model dtype. On CPU, fp32 is usually more stable for OCR.",
    )
    p.add_argument(
        "--use-slow-processor",
        action="store_true",
        help="Force the slow processor (`use_fast=False`). Can improve OCR tokenization/preprocessing stability.",
    )
    p.add_argument(
        "--dump-input-text",
        action="store_true",
        help="Print decoded input prompt tokens (helps debug prompt/template issues).",
    )
    p.add_argument(
        "--ocr-preset",
        type=str,
        default="en",
        choices=["en", "zh"],
        help="Override --prompt with a known-good OCR prompt style (English or Chinese).",
    )
    p.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Do not apply the processor/tokenizer chat template; pass prompt string directly.",
    )
    args = p.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_id = args.hf_model or get_hf_model_id()
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    ref = DotsOCRReference(
        HFLoadSpec(
            model_id=model_id,
            dtype=dtype,
            use_fast_processor=False if args.use_slow_processor else True,
        )
    )

    image = _load_image_from_path(in_path, pdf_page_index=args.pdf_page)
    fed_image_path = out_dir / f"{in_path.stem}.fed_to_model.png"
    image.save(fed_image_path)

    prompt = args.prompt
    if args.ocr_preset == "en":
        prompt = "OCR: transcribe the text in the image exactly. Output only the transcription."
    elif args.ocr_preset == "zh":
        prompt = "请识别图片中的文字，逐字输出，不要解释，不要重复题目。"

    if args.no_chat_template:
        # Bypass `preprocess_image_and_prompt`'s chat-template behavior by directly calling the processor.
        raw = ref.processor(images=image, text=prompt, return_tensors="pt")
        # Construct canonical wrapper (same shape as `preprocess_image_and_prompt`).
        from models.demos.dots_ocr.reference.model import DotsOCRInputs

        inputs = DotsOCRInputs(
            input_ids=raw["input_ids"],
            attention_mask=raw.get("attention_mask", torch.ones_like(raw["input_ids"])),
            pixel_values=raw.get("pixel_values"),
            image_grid_thw=raw.get("image_grid_thw"),
        )
    else:
        inputs = ref.preprocess_image_and_prompt(image, prompt)

    # Quick diagnostic: if this is 0, the prompt/image placeholder token wasn't inserted and OCR will be wrong.
    image_tok = ref.image_token_id
    n_img_tokens = int((inputs.input_ids == image_tok).sum().item())
    print(f"[debug] image_token_id={image_tok} image_tokens_in_prompt={n_img_tokens}")
    if args.dump_input_text:
        try:
            decoded_in = ref.processor.batch_decode(inputs.input_ids, skip_special_tokens=False)[0]
            print("[debug] decoded input_ids (incl specials):")
            print(decoded_in)
        except Exception as e:
            print(f"[debug] could not decode input_ids: {e!r}")

    # Make vision input dtype match the model dtype (Conv2d requires input/bias dtypes to match).
    if inputs.pixel_values is not None:
        print(f"[debug] demo selected dtype={dtype} pixel_values(before)={inputs.pixel_values.dtype}")
        inputs = type(inputs)(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pixel_values=inputs.pixel_values.to(dtype=dtype),
            image_grid_thw=inputs.image_grid_thw,
        )
        print(f"[debug] pixel_values(after)={inputs.pixel_values.dtype}")

    gen_kwargs = {
        "do_sample": bool(args.do_sample),
        "temperature": float(args.temperature) if args.do_sample else None,
        "repetition_penalty": float(args.repetition_penalty)
        if args.repetition_penalty and args.repetition_penalty != 1.0
        else None,
        "no_repeat_ngram_size": int(args.no_repeat_ngram_size)
        if args.no_repeat_ngram_size and args.no_repeat_ngram_size > 0
        else None,
        "num_beams": int(args.num_beams) if args.num_beams and args.num_beams > 1 else None,
    }
    # Avoid HF warning and keep generation consistent.
    gen_kwargs["pad_token_id"] = int(
        getattr(ref.tokenizer, "eos_token_id", None) or getattr(ref.model.config, "eos_token_id", 0)
    )
    out = ref.forward(inputs, max_new_tokens=int(args.max_new_tokens), **gen_kwargs)
    text = ref.decode_generated_suffix(out["generated_ids"], inputs.input_ids)

    print("\n=== OUTPUT ===")
    print(text)
    (out_dir / f"{in_path.stem}.txt").write_text(text, encoding="utf-8")

    print("\n---")
    print(f"Saved image: {fed_image_path}")
    print(f"Saved text:  {out_dir / f'{in_path.stem}.txt'}")


if __name__ == "__main__":
    main()
