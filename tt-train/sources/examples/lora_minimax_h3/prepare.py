# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Dataset preparation for the MiniMax-H3 LoRA: canvas sizing, crop, trigger insertion and tokenization.

    python prepare.py --pairs-dir data/pixelart-150 --cache cache/pixelart-150 --trigger pxlrt

Writes per image `N`: `pixels/N.npy` (uint8 HxWx3 canvas) and `tokens/N.npy` (int32 ids), plus `manifest.json`
and `meta.json`. The VAE and text-encoder stages consume this cache and add their own files next to it.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger("minimax_h3.prepare")

MODEL_ID = "MiniMaxAI/MiniMax-H3"
MODEL_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
VAE_SPATIAL_COMPRESSION = 16
PATCH_SIZE = (1, 2, 2)
# The tt_dit pipeline rejects longer prompts (MINIMAX_H3_MAX_TEXT_TOKENS).
MAX_TEXT_TOKENS = 3008

# Stable-Diffusion prompt boilerplate that says nothing about the picture; stripped from plain captions.
_CAPTION_JUNK = re.compile(
    r"\b(trending on artstation|artstation|cgsociety|deviantart|behance|unreal engine( 5)?|octane render|"
    r"8 ?k|4 ?k|uhd|hdr|highly detailed|ultra detailed|intricate details?|sharp focus|masterpiece|"
    r"award[- ]winning|game concept art|concept art|digital painting|digital art|smooth|elegant|hyperrealism|hyper ?realistic|"
    r"photorealistic|realistic|illustration|by [a-z][a-z .'-]+?(?=,|$))\b",
    re.IGNORECASE,
)


def clean_caption(text: str) -> str:
    text = _CAPTION_JUNK.sub("", text)
    text = re.sub(r"\s*,\s*(,\s*)+", ", ", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r"\s+", " ", text).strip(" ,.")
    return text


def apply_trigger(caption: str, trigger: str | None) -> str:
    """Insert the trigger after `[Shot 1] ` of a brief, else prepend it."""
    if not trigger:
        return caption
    marker = "[Shot 1] "
    if marker in caption:
        return caption.replace(marker, f"{marker}{trigger}, ", 1)
    return f"{trigger}, {caption}"


def training_canvas(width: int, height: int, short_edge: int, multiple: int = 32) -> tuple[int, int]:
    """(H, W): short edge -> `short_edge`, area capped at `short_edge^2 * 1.75`, both multiples of `multiple`."""
    ratio = width / height
    if ratio >= 1:
        w, h = short_edge * ratio, float(short_edge)
    else:
        w, h = float(short_edge), short_edge / ratio
    cap = short_edge * short_edge * 1.75
    if w * h > cap:
        s = (cap / (w * h)) ** 0.5
        w, h = w * s, h * s
    return max(multiple, round(h / multiple) * multiple), max(multiple, round(w / multiple) * multiple)


def fit_image(image: Image.Image, height: int, width: int, resample) -> Image.Image:
    """Cover-resize then centre-crop. NEAREST keeps pixel-art blocks crisp."""
    image = image.convert("RGB")
    scale = max(width / image.width, height / image.height)
    size = (max(width, round(image.width * scale)), max(height, round(image.height * scale)))
    image = image.resize(size, resample)
    left, top = (size[0] - width) // 2, (size[1] - height) // 2
    return image.crop((left, top, left + width, top + height))


def read_pairs_dir(folder: Path, captions: str) -> list[tuple[Path, str]]:
    """`images/N.png` with `<captions>/N.txt`; falls back to `prompts/N.txt`."""
    pairs, fallbacks = [], []
    for png in sorted((folder / "images").glob("*.png"), key=lambda q: int(q.stem) if q.stem.isdigit() else q.stem):
        txt = folder / captions / f"{png.stem}.txt"
        if not txt.exists():
            txt = folder / "prompts" / f"{png.stem}.txt"
            if captions != "prompts" and txt.exists():
                fallbacks.append(png.stem)
        if txt.exists() and txt.read_text(encoding="utf-8").strip():
            pairs.append((png, txt.read_text(encoding="utf-8").strip()))
    if fallbacks:
        log.warning(
            "%d of %d images have no %s/N.txt and use prompts/N.txt instead, mixing caption formats (e.g. %s)",
            len(fallbacks),
            len(pairs),
            captions,
            ", ".join(fallbacks[:5]),
        )
    return pairs


def load_tokenizer(spec: str | None, model: str, revision: str):
    from tokenizers import Tokenizer

    if spec:
        path = Path(spec)
        if path.is_dir():
            path = path / "tokenizer.json"
    else:
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(model, "tokenizer/tokenizer.json", revision=revision))
    return Tokenizer.from_file(str(path)), str(path)


def tokenize(tokenizer, caption: str, max_tokens: int) -> np.ndarray:
    """H3's t2va presentation: the verbatim prompt, no chat template, no special tokens."""
    ids = tokenizer.encode(caption, add_special_tokens=False).ids
    if len(ids) > max_tokens:
        raise SystemExit(f"caption is {len(ids)} tokens, over --max-tokens {max_tokens}: {caption[:80]!r}")
    return np.asarray(ids, dtype=np.int32)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs-dir", required=True, help="folder with images/N.png + <captions>/N.txt")
    p.add_argument("--cache", required=True)
    p.add_argument("--captions", default="ir-prompts", help="caption subfolder (ir-prompts briefs are kept verbatim)")
    p.add_argument("--limit", type=int, default=0, help="use only the first N pairs (after a fixed shuffle)")
    p.add_argument("--short-edge", type=int, default=768)
    p.add_argument("--resample", choices=["nearest", "lanczos"], default="nearest")
    p.add_argument("--trigger", default="pxlrt", help="'' trains without one")
    p.add_argument(
        "--clean-captions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="strip SD boilerplate from plain captions (never from ir-prompts)",
    )
    p.add_argument("--max-tokens", type=int, default=MAX_TEXT_TOKENS)
    p.add_argument("--tokenizer", default=None, help="tokenizer.json or its folder; default: download from --model")
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--revision", default=MODEL_REVISION)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    pairs = read_pairs_dir(Path(args.pairs_dir), args.captions)
    if not pairs:
        raise SystemExit(f"no image/caption pairs under {args.pairs_dir}")
    clean = args.clean_captions and args.captions != "ir-prompts"
    random.Random(0).shuffle(pairs)
    if args.limit:
        pairs = pairs[: args.limit]
    log.info("%d image/caption pairs", len(pairs))

    tokenizer, tokenizer_path = load_tokenizer(args.tokenizer, args.model, args.revision)
    resample = Image.Resampling.NEAREST if args.resample == "nearest" else Image.Resampling.LANCZOS

    cache = Path(args.cache)
    (cache / "pixels").mkdir(parents=True, exist_ok=True)
    (cache / "tokens").mkdir(exist_ok=True)
    manifest_path = cache / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    (cache / "meta.json").write_text(
        json.dumps(
            {
                "trigger": args.trigger or None,
                "short_edge": args.short_edge,
                "resample": args.resample,
                "captions": args.captions,
                "max_tokens": args.max_tokens,
                "model": args.model,
                "revision": args.revision,
                "tokenizer": tokenizer_path,
                "source": str(args.pairs_dir),
            },
            indent=1,
        )
    )

    t0 = time.perf_counter()
    for i, (png, raw_caption) in enumerate(pairs):
        key = png.stem
        caption = apply_trigger(clean_caption(raw_caption) if clean else raw_caption, args.trigger)
        if (
            key in manifest
            and manifest[key]["caption"] == caption
            and (cache / "pixels" / f"{key}.npy").exists()
            and (cache / "tokens" / f"{key}.npy").exists()
        ):
            continue
        # encode_text.py skips existing embeds, which would still hold the old caption.
        (cache / "embeds" / f"{key}.npy").unlink(missing_ok=True)
        image = Image.open(png)
        height, width = training_canvas(image.width, image.height, args.short_edge)
        image = fit_image(image, height, width, resample)
        tokens = tokenize(tokenizer, caption, args.max_tokens)

        np.save(cache / "pixels" / f"{key}.npy", np.asarray(image, dtype=np.uint8))
        np.save(cache / "tokens" / f"{key}.npy", tokens)
        manifest[key] = {
            "image": str(png),
            "caption": caption,
            "height": height,
            "width": width,
            "latent_height": height // VAE_SPATIAL_COMPRESSION,
            "latent_width": width // VAE_SPATIAL_COMPRESSION,
            "video_rows": (height // VAE_SPATIAL_COMPRESSION // PATCH_SIZE[1])
            * (width // VAE_SPATIAL_COMPRESSION // PATCH_SIZE[2]),
            "num_tokens": int(tokens.shape[0]),
        }
        if (i + 1) % 25 == 0 or i + 1 == len(pairs):
            manifest_path.write_text(json.dumps(manifest, indent=1))
            log.info(
                "[%d/%d] %.0f s | %dx%d %d tok | %s",
                i + 1,
                len(pairs),
                time.perf_counter() - t0,
                height,
                width,
                tokens.shape[0],
                caption[:80],
            )
    manifest_path.write_text(json.dumps(manifest, indent=1))
    sizes = {}
    for entry in manifest.values():
        k = f"{entry['height']}x{entry['width']}"
        sizes[k] = sizes.get(k, 0) + 1
    log.info("cache ready: %d samples in %s | canvases %s", len(manifest), cache, sizes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
