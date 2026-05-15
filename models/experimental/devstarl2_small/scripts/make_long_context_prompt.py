# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Build chat messages whose tokenized length (after the same template as the demos) is near a target
(e.g. 256K) for long-context / perf experiments.

Usage (repo root, venv active)::

    python models/experimental/devstarl2_small/scripts/make_long_context_prompt.py \\
        --target-tokens 256000 \\
        --mode multimodal \\
        --image models/experimental/devstarl2_small/reference/sample.jpeg \\
        --vision-square-pixels 1540 \\
        --output models/experimental/devstarl2_small/reference/messages_256k.json

    # Then run the multimodal demo with the generated messages:
    python models/experimental/devstarl2_small/demo/demo_model_loading_prompt.py \\
        --backend tt \\
        --messages-json models/experimental/devstarl2_small/reference/messages_256k.json \\
        --image models/experimental/devstarl2_small/reference/sample.jpeg \\
        --vision-square-pixels 1540 \\
        --max-new-tokens 1

Text-only (no vision tokens)::

    python models/experimental/devstarl2_small/scripts/make_long_context_prompt.py \\
        --target-tokens 256000 --mode text-only \\
        --output models/experimental/devstarl2_small/reference/messages_256k_text.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor

import ttnn

from models.experimental.devstarl2_small.devstral_utils import (
    devstral_estimate_max_prompt_tokens_dense_kv,
    devstral_model_args_for_kv_estimate,
    open_devstral_demo_mesh,
)

_DEFAULT_MODEL_ID = "mistralai/Devstral-Small-2-24B-Instruct-2512"
_SCRIPT_DIR = Path(__file__).resolve().parent
_REF_DIR = _SCRIPT_DIR.parent / "reference"

# Short, stable filler (tokenized once; repeated to reach target length).
_FILLER_LINE = (
    "Context padding for long-window benchmarking on Tenstorrent Devstral demos. "
    "Please ignore this repeated section when answering; only respond to the final instruction. "
)

_TAIL_INSTRUCTION = "\n\n--- End of padding. Final instruction: Summarize the above in one short sentence."


def _prepare_image(image_path: Path, vision_max_edge: int, vision_square_pixels: int | None) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    if vision_square_pixels is not None and vision_square_pixels > 0:
        return image.resize((vision_square_pixels, vision_square_pixels), Image.Resampling.LANCZOS)
    if vision_max_edge > 0:
        w, h = image.size
        if max(w, h) > vision_max_edge:
            out = image.copy()
            out.thumbnail((vision_max_edge, vision_max_edge), Image.Resampling.LANCZOS)
            return out
    return image


def _count_tokens_multimodal(
    processor: AutoProcessor,
    messages: list,
    image: Image.Image | None,
) -> int:
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    if image is not None:
        batch = processor(text=prompt, images=image, return_tensors="pt")
    else:
        batch = processor(text=prompt, return_tensors="pt")
    return int(batch["input_ids"].shape[1])


def _count_tokens_text_only(processor: AutoProcessor, messages: list) -> int:
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    batch = processor(text=prompt, return_tensors="pt")
    return int(batch["input_ids"].shape[1])


def _base_messages(mode: str, tail: str) -> list:
    if mode == "multimodal":
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": tail},
                ],
            }
        ]
    if mode == "text-only":
        return [{"role": "user", "content": [{"type": "text", "text": tail}]}]
    raise ValueError(f"Unknown mode {mode!r}")


def _messages_with_filler(mode: str, n_repeat: int, tail: str) -> list:
    body = (_FILLER_LINE * n_repeat) + tail
    return _base_messages(mode, body)


def _resolve_target_tokens(target: str, model_id: str, mesh_width: int) -> int:
    if target.lower() not in ("max", "auto"):
        return int(target)
    mesh_device = open_devstral_demo_mesh(max(1, min(mesh_width, ttnn.get_num_devices())))
    try:
        model_args = devstral_model_args_for_kv_estimate(mesh_device, model_id=model_id)
        return devstral_estimate_max_prompt_tokens_dense_kv(model_args, mesh_device)
    finally:
        ttnn.close_mesh_device(mesh_device)


def _find_repeat_count(
    processor: AutoProcessor,
    *,
    mode: str,
    target: int,
    image: Image.Image | None,
    tail: str,
    tolerance: int,
) -> tuple[int, int]:
    """Binary-search repeat count so token length is in [target - tol, target + tol]."""
    lo, hi = 0, 2
    count_fn = (
        (lambda m: _count_tokens_multimodal(processor, m, image))
        if mode == "multimodal"
        else (lambda m: _count_tokens_text_only(processor, m))
    )

    def tok(n: int) -> int:
        return count_fn(_messages_with_filler(mode, n, tail))

    while tok(hi) < target - tolerance:
        hi *= 2
        if hi > 50_000_000:
            raise RuntimeError("Could not bracket target token count; check target / mode.")

    best_n, best_t = 0, tok(0)
    while lo <= hi:
        mid = (lo + hi) // 2
        t = tok(mid)
        if abs(t - target) < abs(best_t - target):
            best_n, best_t = mid, t
        if t < target:
            lo = mid + 1
        else:
            hi = mid - 1

    if best_t < target - tolerance:
        raise RuntimeError(
            f"Best effort {best_t:,} tokens (repeats={best_n:,}) is below target {target:,} "
            f"(tolerance {tolerance}). Try text-only or a smaller image."
        )
    return best_n, best_t


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate long-context chat messages for Devstral demos.")
    parser.add_argument("--model-id", default=_DEFAULT_MODEL_ID)
    parser.add_argument(
        "--target-tokens",
        default="256000",
        help="Token target (int) or 'max'/'auto' for TT dense-KV estimate on this mesh (see find_max_tt_context_tokens.py).",
    )
    parser.add_argument("--mesh-width", type=int, default=1, help="Used with --target-tokens max.")
    parser.add_argument(
        "--mode",
        choices=("multimodal", "text-only"),
        default="multimodal",
        help="multimodal: image + text (same as demo_model_loading_prompt). text-only: no vision tokens.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=_REF_DIR / "sample.jpeg",
        help="Image for multimodal token counting (and noted in output JSON metadata).",
    )
    parser.add_argument("--vision-max-edge", type=int, default=0)
    parser.add_argument("--vision-square-pixels", type=int, default=None)
    parser.add_argument(
        "--tolerance",
        type=int,
        default=512,
        help="Accept token count within target ± tolerance.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_REF_DIR / "messages_256k.json",
        help="Write messages JSON (OpenAI-style list for apply_chat_template).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats only; do not write output file.",
    )
    args = parser.parse_args()

    target_tokens = _resolve_target_tokens(args.target_tokens, args.model_id, args.mesh_width)
    if str(args.target_tokens).lower() in ("max", "auto"):
        print(f"Resolved --target-tokens max → {target_tokens:,} (TT dense-KV estimate).")

    if not args.image.is_file() and args.mode == "multimodal":
        print(f"Image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading processor {args.model_id} …")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

    pil_image: Image.Image | None = None
    if args.mode == "multimodal":
        pil_image = _prepare_image(args.image, args.vision_max_edge, args.vision_square_pixels)
        base_t = _count_tokens_multimodal(processor, _messages_with_filler(args.mode, 0, _TAIL_INSTRUCTION), pil_image)
        print(f"Base multimodal prompt (0 filler repeats): {base_t:,} tokens")
    else:
        base_t = _count_tokens_text_only(processor, _messages_with_filler(args.mode, 0, _TAIL_INSTRUCTION))
        print(f"Base text-only prompt (0 filler repeats): {base_t:,} tokens")

    if base_t >= target_tokens:
        print(
            f"Target {target_tokens:,} is not above base {base_t:,}; "
            "use a smaller image or lower --vision-square-pixels.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Searching filler repeats for ~{target_tokens:,} tokens (±{args.tolerance}) …")
    n_repeat, final_t = _find_repeat_count(
        processor,
        mode=args.mode,
        target=target_tokens,
        image=pil_image,
        tail=_TAIL_INSTRUCTION,
        tolerance=args.tolerance,
    )
    messages = _messages_with_filler(args.mode, n_repeat, _TAIL_INSTRUCTION)
    text_chars = len(messages[0]["content"][-1]["text"])

    print(
        f"Result: {final_t:,} tokens  (target {target_tokens:,}, "
        f"filler repeats {n_repeat:,}, text chars {text_chars:,})"
    )
    if final_t >= 32_768:
        print(
            "\nWARNING: TT demos allocate dense K/V DRAM per decoder layer at this token length. "
            "On a single Blackhole chip with the full Devstral checkpoint, ~256K context typically "
            "OOMs during init_kv_cache (~500+ MiB per layer). Use --backend hf for HF/GPU, a shorter "
            "--target-tokens for TT perf tests, or multi-device / future paged-KV paths.\n"
        )

    if args.dry_run:
        return

    payload = {
        "metadata": {
            "model_id": args.model_id,
            "mode": args.mode,
            "target_tokens": target_tokens,
            "measured_tokens": final_t,
            "filler_repeats": n_repeat,
            "image": str(args.image) if args.mode == "multimodal" else None,
            "vision_square_pixels": args.vision_square_pixels,
            "vision_max_edge": args.vision_max_edge,
        },
        "messages": messages,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    print(f"Wrote {args.output} ({args.output.stat().st_size / (1024 * 1024):.2f} MiB)")


if __name__ == "__main__":
    main()
