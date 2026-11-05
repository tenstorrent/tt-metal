# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from models.demos.blackhole.pp_ocrv6.common import DET_L1_SMALL_SIZE
from models.demos.blackhole.pp_ocrv6.demo.pipeline import PPOCRv6Pipeline, TorchBackend, TtnnBackend

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def plate(lines, size=(900, 420), rotate_deg=0.0, font_px=88):
    """Trailer-side style crop: dark stencil text on a light panel, as YOLO would hand to OCR."""
    img = Image.new("RGB", size, (200, 202, 196))
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT, font_px)
    y = 60
    for text in lines:
        d.text((60, y), text, fill=(25, 25, 25), font=font)
        y += int(font_px * 1.6)
    bgr = np.ascontiguousarray(np.array(img)[:, :, ::-1])
    if rotate_deg:
        m = cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), rotate_deg, 1.0)
        bgr = cv2.warpAffine(bgr, m, size, borderMode=cv2.BORDER_REPLICATE)
    return bgr


CASES = {
    "one_line": (plate(["TRLU 482913"], size=(900, 220)), ["TRLU 482913"]),
    "two_lines": (plate(["MSCU 7731045", "53 FT 0921"]), ["MSCU 7731045", "53 FT 0921"]),
    "rotated_6deg": (plate(["JBHU 238811"], size=(900, 260), rotate_deg=6), ["JBHU 238811"]),
}


@pytest.fixture(scope="module")
def torch_pipeline():
    return PPOCRv6Pipeline(TorchBackend())


@pytest.mark.parametrize("device_params", [{"l1_small_size": DET_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize("case", list(CASES))
def test_pipeline(device, torch_pipeline, case):
    bgr, expected = CASES[case]
    ttnn_pipeline = PPOCRv6Pipeline(TtnnBackend(device), torch_pipeline.charset)

    ttnn_pipeline(bgr)  # warm-up: compiles kernels for this shape
    ref = torch_pipeline(bgr)
    out = ttnn_pipeline(bgr)

    ref_texts = [line.text for line in ref.lines]
    tt_texts = [line.text for line in out.lines]
    print(f"\n[{case}] expected {expected}")
    print(f"  torch  {ref_texts}  {[round(line.confidence, 3) for line in ref.lines]}")
    print(f"  ttnn   {tt_texts}  {[round(line.confidence, 3) for line in out.lines]}")
    print("  ttnn timings ms: " + ", ".join(f"{k} {v:.2f}" for k, v in out.timings_ms.items()))
    print("  torch timings ms: " + ", ".join(f"{k} {v:.2f}" for k, v in ref.timings_ms.items()))

    assert tt_texts == ref_texts
    assert tt_texts == expected
