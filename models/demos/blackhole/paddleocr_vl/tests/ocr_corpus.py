# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Deterministic synthetic OCR corpus for PaddleOCR-VL bring-up.

The bring-up gates compare TT output against a HuggingFace CPU reference, so the
corpus has to be **reproducible** rather than realistic: the same twenty images
must come out byte-identical on any machine, or a golden recorded today stops
meaning anything tomorrow. Everything here is therefore rendered from a fixed
seed with bundled fonts, and no image is committed to the repo.

Two properties are deliberate:

*Bucket coverage.* The vision tower compiles one program set per padded patch
count, so the corpus spans every bucket the model can produce. ``smart_resize``
snaps both dimensions to a multiple of 28 (patch 14 x spatial merge 2) and clamps
the token count to [130, 1280], which is [520, 5120] patches. Sizes below are
chosen so the set lands in all four of {1024, 2048, 4096, 6144}.

*Known ground truth.* Each entry carries the exact string it renders, so the same
corpus measures absolute CER, not just TT-vs-HF agreement. That matters when a
port regression and a model limitation would otherwise look alike.

Degradations (rotation, noise, blur, JPEG, contrast) are applied from the seeded
RNG. They exist to move the pixel statistics off "clean synthetic render", which
is where a bf16 numerical difference is most likely to change a character.
"""

from __future__ import annotations

import io
import math
import os
import random
from dataclasses import dataclass, field

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

SEED = 20260914

_FONT_DIRS = (
    "/usr/share/fonts/truetype/dejavu",
    "/usr/share/fonts/truetype/liberation",
)

# Rendered at a multiple of 28 so smart_resize is a no-op and the patch count is
# exactly (h/14)*(w/14). Anything that needs clamping is listed as "large".
BUCKET_SIZES = {
    1024: (448, 448),  # 16x16 merged tokens
    2048: (896, 448),  # 32x16
    4096: (896, 896),  # 32x32
    6144: (1400, 1120),  # clamps to 1280 tokens -> 5120 patches
}


def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    for d in _FONT_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    raise FileNotFoundError(f"font {name!r} not found in {_FONT_DIRS}")


@dataclass
class Sample:
    """One corpus image: how to draw it, and what it says."""

    name: str
    bucket: int
    kind: str
    lines: list[str]
    font: str = "DejaVuSans.ttf"
    font_size: int = 22
    align: str = "left"
    degrade: tuple[str, ...] = ()
    columns: int = 1
    extra: dict = field(default_factory=dict)

    @property
    def ground_truth(self) -> str:
        return "\n".join(self.lines)


# ---------------------------------------------------------------------------
# Corpus. Twenty samples, five per bucket, spread across document kinds and
# degradations. Text is ASCII-only: the goal is to isolate port fidelity, and a
# CJK tokenizer difference would confound that.
# ---------------------------------------------------------------------------

CORPUS: list[Sample] = [
    # ---- bucket 1024: small, sparse, high contrast --------------------------
    Sample(
        "sign_exit",
        1024,
        "sign",
        ["EMERGENCY EXIT", "KEEP CLEAR AT ALL TIMES"],
        font="DejaVuSans-Bold.ttf",
        font_size=38,
        align="center",
    ),
    Sample(
        "sign_parking",
        1024,
        "sign",
        ["NO PARKING", "MON - FRI", "8 AM TO 6 PM", "TOW AWAY ZONE"],
        font="LiberationSans-Bold.ttf",
        font_size=34,
        align="center",
        degrade=("rotate",),
    ),
    Sample(
        "label_shipping",
        1024,
        "label",
        ["TRACKING 1Z 999 AA1 0123 4567", "SHIP DATE 14 SEP 2026", "WEIGHT 2.4 LB", "ZONE 05"],
        font="DejaVuSansMono.ttf",
        font_size=20,
    ),
    Sample(
        "note_short",
        1024,
        "paragraph",
        ["Meeting moved to 3pm.", "Room 412, west wing.", "Bring the Q3 figures."],
        font="DejaVuSerif.ttf",
        font_size=26,
        degrade=("noise",),
    ),
    Sample(
        "price_tag",
        1024,
        "label",
        ["SALE", "$ 24.99", "was $ 39.99", "SKU 883-2041"],
        font="LiberationSans-Bold.ttf",
        font_size=32,
        align="center",
        degrade=("contrast",),
    ),
    # ---- bucket 2048: narrow receipts and lists -----------------------------
    Sample(
        "receipt_grocery",
        2048,
        "receipt",
        [
            "NORTHGATE MARKET",
            "1420 CEDAR AVENUE",
            "--------------------------",
            "MILK 2L            3.49",
            "SOURDOUGH LOAF     4.25",
            "EGGS DOZEN         5.10",
            "COFFEE BEANS 1KG  18.00",
            "OLIVE OIL 500ML    9.75",
            "--------------------------",
            "SUBTOTAL          40.59",
            "TAX 8.25%          3.35",
            "TOTAL             43.94",
            "VISA ****1234",
            "14 SEP 2026 18:42",
        ],
        font="DejaVuSansMono.ttf",
        font_size=18,
    ),
    Sample(
        "receipt_cafe",
        2048,
        "receipt",
        [
            "THE CORNER CAFE",
            "ORDER 0471",
            "==========================",
            "2x FLAT WHITE      8.00",
            "1x ALMOND CROISSANT 4.50",
            "1x ORANGE JUICE    3.75",
            "==========================",
            "TOTAL             16.25",
            "CASH              20.00",
            "CHANGE             3.75",
            "THANK YOU",
        ],
        font="LiberationMono-Regular.ttf",
        font_size=19,
        degrade=("jpeg",),
    ),
    Sample(
        "list_checklist",
        2048,
        "list",
        [
            "PREFLIGHT CHECKLIST",
            "1. Verify hugepages allocated",
            "2. Confirm device is idle",
            "3. Export MESH_DEVICE",
            "4. Warm the vision buckets",
            "5. Capture prefill trace",
            "6. Run the smoke request",
            "7. Check logs for recompiles",
        ],
        font_size=20,
    ),
    Sample(
        "code_snippet",
        2048,
        "code",
        [
            "def smart_resize(h, w, factor=28):",
            "    h_bar = round(h / factor) * factor",
            "    w_bar = round(w / factor) * factor",
            "    if h_bar * w_bar > MAX_PIXELS:",
            "        beta = sqrt((h * w) / MAX_PIXELS)",
            "        h_bar = floor(h / beta / factor)",
            "        w_bar = floor(w / beta / factor)",
            "    return h_bar, w_bar",
        ],
        font="DejaVuSansMono.ttf",
        font_size=17,
    ),
    Sample(
        "form_intake",
        2048,
        "form",
        [
            "PATIENT INTAKE",
            "NAME: Dana Okafor",
            "DOB: 1987-03-22",
            "MRN: 4471902",
            "PHONE: (555) 018-2245",
            "ALLERGIES: penicillin",
            "VISIT: 2026-09-14",
            "PROVIDER: R. Lindqvist",
        ],
        font_size=21,
        degrade=("rotate", "noise"),
    ),
    # ---- bucket 4096: dense single and multi column -------------------------
    Sample(
        "para_dense",
        4096,
        "paragraph",
        [
            "Optical character recognition converts an image of text",
            "into machine readable characters. Early systems relied on",
            "template matching and worked only for a single typeface.",
            "Modern approaches treat the task as sequence prediction,",
            "encoding the page with a vision transformer and decoding",
            "the text autoregressively. The encoder must preserve fine",
            "stroke detail, so the input is tiled at native resolution",
            "rather than squashed to a fixed square. Accuracy is then",
            "reported as character error rate, the edit distance to the",
            "reference divided by the reference length.",
        ],
        font="DejaVuSerif.ttf",
        font_size=21,
    ),
    Sample(
        "table_quarterly",
        4096,
        "table",
        [
            "QUARTERLY RESULTS",
            "Region      Q1      Q2      Q3",
            "North     1204    1388    1502",
            "South      890     945    1011",
            "East      1567    1602    1744",
            "West       733     802     868",
            "Central    412     455     498",
            "TOTAL     4806    5192    5623",
        ],
        font="DejaVuSansMono.ttf",
        font_size=22,
    ),
    Sample(
        "two_column",
        4096,
        "paragraph",
        [
            "The tower emits one token per patch, then a merger",
            "folds each two by two block into a single embedding.",
            "That reduces the sequence the decoder sees by four,",
            "which matters because the decoder is small and the",
            "page can be large. Padding to a fixed bucket keeps",
            "the compiled program set finite. Without it every",
            "new page shape would trigger a fresh compile, and a",
            "compile while a trace is parked corrupts the trace.",
        ],
        columns=2,
        font_size=19,
        degrade=("blur",),
    ),
    Sample(
        "invoice_full",
        4096,
        "form",
        [
            "INVOICE 2026-0914",
            "BILL TO: Harbour Logistics",
            "DUE: 2026-10-14",
            "",
            "DESCRIPTION            QTY    AMOUNT",
            "Freight forwarding       3    1440.00",
            "Customs handling         1     275.50",
            "Warehouse storage       12     960.00",
            "Insurance premium        1     188.25",
            "",
            "SUBTOTAL                      2863.75",
            "VAT 20%                        572.75",
            "AMOUNT DUE                    3436.50",
        ],
        font="LiberationSans-Regular.ttf",
        font_size=20,
    ),
    Sample(
        "mixed_headings",
        4096,
        "paragraph",
        [
            "SECTION 3 - DEPLOYMENT",
            "3.1 Prerequisites",
            "The host must expose at least one accelerator device",
            "and sixteen gigabyte pages. Verify both before start.",
            "3.2 Launching",
            "Export the mesh identifier, then invoke the server with",
            "a single sequence slot. Multimodal batching is not yet",
            "supported and will be rejected at request admission.",
            "3.3 Verification",
            "Issue one request per shape and confirm no recompiles.",
        ],
        font_size=20,
        degrade=("contrast", "noise"),
    ),
    # ---- bucket 6144: large pages that clamp to the token cap ---------------
    Sample(
        "page_article",
        6144,
        "paragraph",
        [
            "A COMPACT MODEL FOR DOCUMENT PARSING",
            "",
            "Document parsing has converged on a two stage design. A",
            "layout model first proposes regions and their reading",
            "order. A recognition model then transcribes each region",
            "in isolation. Splitting the problem this way lets each",
            "component stay small, because neither has to reason about",
            "the whole page at once.",
            "",
            "The recognition stage is where accuracy is won or lost.",
            "It sees a crop that may be a paragraph, a table cell, a",
            "formula, or a stamp, and must emit structured text for",
            "each. A prompt prefix selects the behaviour, so a single",
            "set of weights covers every element type.",
            "",
            "Resolution handling is the second lever. Fixed square",
            "inputs destroy the aspect ratio of a text line, which is",
            "exactly the signal a recognizer needs. Native resolution",
            "tiling preserves it at the cost of a variable sequence",
            "length, which the runtime must then absorb.",
        ],
        font="DejaVuSerif.ttf",
        font_size=24,
    ),
    Sample(
        "page_manual",
        6144,
        "list",
        [
            "OPERATING MANUAL - SECTION 7",
            "",
            "7.1 BEFORE EACH RUN",
            "  a. Confirm the enclosure is closed and latched.",
            "  b. Check coolant level is between MIN and MAX.",
            "  c. Verify the emergency stop is not engaged.",
            "",
            "7.2 STARTING THE UNIT",
            "  a. Turn the main isolator to ON.",
            "  b. Wait for the status lamp to show steady green.",
            "  c. Press and hold START for two seconds.",
            "",
            "7.3 SHUTDOWN",
            "  a. Press STOP and wait for the spindle to halt.",
            "  b. Turn the main isolator to OFF.",
            "  c. Record the run hours in the logbook.",
            "",
            "WARNING: Do not bypass the interlock under any",
            "circumstances. Doing so voids the warranty and may",
            "cause serious injury.",
        ],
        font_size=23,
    ),
    Sample(
        "page_table_large",
        6144,
        "table",
        [
            "INVENTORY REPORT - WAREHOUSE 4",
            "",
            "PART      DESCRIPTION          ON HAND   RESERVED   FREE",
            "A-1002    Bearing 30mm             418        120    298",
            "A-1145    Bearing 45mm             207         64    143",
            "B-2210    Drive belt 900            95         30     65",
            "B-2255    Drive belt 1200          142         48     94",
            "C-3081    Hydraulic seal           630        210    420",
            "C-3099    Hydraulic hose 2m         77         12     65",
            "D-4410    Control relay 24V        355         95    260",
            "D-4488    Control relay 48V        188         40    148",
            "E-5500    Filter cartridge         912        305    607",
            "",
            "TOTAL LINES: 9",
            "GENERATED: 2026-09-14 20:15",
        ],
        font="DejaVuSansMono.ttf",
        font_size=20,
        degrade=("jpeg",),
    ),
    Sample(
        "page_skewed",
        6144,
        "paragraph",
        [
            "FIELD NOTES",
            "",
            "Collected samples at three sites along the ridge.",
            "Site one showed heavy erosion on the north face,",
            "with exposed root systems and loose scree below.",
            "Site two was stable but showed early signs of the",
            "same pattern near the drainage channel.",
            "",
            "Site three was inaccessible due to standing water.",
            "Recommend returning after the dry season and re-",
            "surveying all three with the updated equipment.",
            "",
            "Weather: overcast, 14C, light wind from the west.",
            "Observer: M. Trevino",
        ],
        font="LiberationSerif-Regular.ttf",
        font_size=24,
        degrade=("rotate", "blur", "noise"),
    ),
    Sample(
        "page_statement",
        6144,
        "table",
        [
            "ACCOUNT STATEMENT",
            "Period: 01 AUG 2026 to 31 AUG 2026",
            "",
            "DATE        DESCRIPTION              DEBIT    CREDIT",
            "03 AUG      Opening balance                  2140.00",
            "05 AUG      Direct debit utilities  142.60",
            "09 AUG      Salary                          3200.00",
            "12 AUG      Card purchase            88.45",
            "15 AUG      Transfer to savings     500.00",
            "19 AUG      Card purchase           231.10",
            "24 AUG      Insurance premium       176.00",
            "28 AUG      Card purchase            64.75",
            "31 AUG      Interest                            4.12",
            "",
            "CLOSING BALANCE                            4141.22",
        ],
        font="LiberationMono-Regular.ttf",
        font_size=21,
        degrade=("contrast",),
    ),
]


def _fit_font(sample: Sample, size) -> ImageFont.FreeTypeFont:
    """Pick the largest font size that fills the canvas without overflowing it.

    Rendering every sample at its nominal point size leaves the big buckets
    mostly white, which spends the 1280-token budget on blank tiles and shrinks
    the glyphs the recognizer actually has to read. Fitting to the canvas keeps
    text density roughly constant across buckets, so a bucket comparison is
    about sequence length rather than font scale.
    """
    w, h = size
    margin = max(18, int(w * 0.05))
    n = max(1, len(sample.lines))

    # Height-driven first guess: n lines plus one of slack inside the margins.
    avail_h = h - 2 * margin
    size_from_h = max(10, int((avail_h / (n + 1)) / 1.55))

    # Width cap: the longest line must fit the column it is drawn into.
    col_w = (w - 3 * margin) // 2 if sample.columns == 2 else w - 2 * margin
    longest = max(sample.lines, key=len) if sample.lines else ""

    chosen = min(size_from_h, int(sample.font_size * 2.2))
    while chosen > 10:
        f = _font(sample.font, chosen)
        if ImageDraw.Draw(Image.new("L", (1, 1))).textlength(longest, font=f) <= col_w:
            return f
        chosen -= 1
    return _font(sample.font, 10)


def _draw_columns(draw, sample: Sample, size, font, rng):
    w, h = size
    margin = max(18, int(w * 0.05))
    line_h = int(font.size * 1.55)

    if sample.columns == 1:
        y = margin
        for line in sample.lines:
            if sample.align == "center":
                tw = draw.textlength(line, font=font)
                x = (w - tw) / 2
            else:
                x = margin
            draw.text((x, y), line, fill=0, font=font)
            y += line_h
        return

    # Two columns: split the lines down the middle, balanced by count.
    half = math.ceil(len(sample.lines) / 2)
    col_w = (w - 3 * margin) // 2
    for ci, chunk in enumerate((sample.lines[:half], sample.lines[half:])):
        x0 = margin + ci * (col_w + margin)
        y = margin
        for line in chunk:
            draw.text((x0, y), line, fill=0, font=font)
            y += line_h


def _degrade(img: Image.Image, sample: Sample, rng: random.Random) -> Image.Image:
    for step in sample.degrade:
        if step == "rotate":
            angle = rng.uniform(-3.5, 3.5)
            img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=255, expand=False)
        elif step == "blur":
            img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.4, 0.9)))
        elif step == "noise":
            px = img.load()
            w, h = img.size
            for _ in range(int(w * h * 0.02)):
                x, y = rng.randrange(w), rng.randrange(h)
                v = px[x, y]
                px[x, y] = max(0, min(255, v + rng.randint(-55, 55)))
        elif step == "contrast":
            img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.55, 0.75))
            img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.95, 1.12))
        elif step == "jpeg":
            buf = io.BytesIO()
            img.convert("L").save(buf, format="JPEG", quality=rng.randint(40, 62))
            buf.seek(0)
            img = Image.open(buf).convert("L")
        else:
            raise ValueError(f"unknown degradation {step!r}")
    return img


def render(sample: Sample) -> Image.Image:
    """Render one sample deterministically. Same input, same pixels, always."""
    rng = random.Random(f"{SEED}:{sample.name}")
    size = BUCKET_SIZES[sample.bucket]
    img = Image.new("L", size, color=255)
    draw = ImageDraw.Draw(img)
    font = _fit_font(sample, size)
    _draw_columns(draw, sample, size, font, rng)
    img = _degrade(img, sample, rng)
    return img.convert("RGB")


def build_corpus(out_dir: str) -> list[dict]:
    """Render every sample to ``out_dir``; return one record per image."""
    os.makedirs(out_dir, exist_ok=True)
    records = []
    for s in CORPUS:
        img = render(s)
        path = os.path.join(out_dir, f"{s.name}.png")
        img.save(path, format="PNG", optimize=False)
        records.append(
            {
                "name": s.name,
                "path": path,
                "bucket": s.bucket,
                "kind": s.kind,
                "size": list(img.size),
                "ground_truth": s.ground_truth,
            }
        )
    return records


if __name__ == "__main__":
    import json
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "demo", "sample_images")
    recs = build_corpus(os.path.abspath(out))
    by_bucket = {}
    for r in recs:
        by_bucket.setdefault(r["bucket"], []).append(r["name"])
    print(json.dumps({"count": len(recs), "out_dir": os.path.abspath(out), "by_bucket": by_bucket}, indent=2))
