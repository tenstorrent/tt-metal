# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Deterministic synthetic document images for the dots.ocr e2e gate.

Renders short text lines with DejaVuSans onto white canvases sized to
small patch grids (patch 14, merge 2), writes PNGs plus
``ocr_samples.json`` mapping each image to its ground-truth text. Re-run
any time; output is bit-stable for a fixed Pillow/font install.
"""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

# (name, text, image WxH) — sizes are multiples of 28 (patch 14 x merge 2)
# so the Qwen2VL preprocessor keeps them unresized. Texts use in-vocabulary
# words: OOV brand words sit on subword knife-edges where bf16-HF and the
# fp32 weights themselves disagree (HF-fp32 reads rendered "Tenstorrent" as
# "Tenstorment" too), which gates tokenizer tie-breaking, not OCR quality.
# Five samples / 31 words keep one residual near-tie flip inside the
# "HF + 0.05" corpus-WER tolerance per the generation skill.
SAMPLES = [
    ("invoice_total", "Invoice total: 1,250 dollars", (560, 84)),
    ("pangram", "The quick brown fox jumps over the lazy dog", (728, 84)),
    ("serial_no", "Serial number 7 4 2 9", (448, 84)),
    ("meeting", "Meeting starts at 10:30 in room 4", (560, 84)),
    ("keys", "Please return the keys before Friday", (616, 84)),
]


def main():
    font = ImageFont.truetype(FONT_PATH, 30)
    manifest = []
    for name, text, (w, h) in SAMPLES:
        img = Image.new("RGB", (w, h), "white")
        draw = ImageDraw.Draw(img)
        draw.text((16, h // 2), text, fill="black", font=font, anchor="lm")
        path = HERE / f"{name}.png"
        img.save(path)
        manifest.append({"image": f"{name}.png", "ref": text})
    (HERE / "ocr_samples.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(manifest)} samples to {HERE}")


if __name__ == "__main__":
    main()
