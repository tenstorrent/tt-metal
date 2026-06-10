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
# so the Qwen2VL preprocessor keeps them unresized.
SAMPLES = [
    ("invoice_total", "Invoice total: 1,250 dollars", (560, 84)),
    ("hello_doc", "Hello world from Tenstorrent", (560, 84)),
    ("serial_no", "Serial number 7 4 2 9", (448, 84)),
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
