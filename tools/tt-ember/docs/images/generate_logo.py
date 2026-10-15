#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Generate the tt-ember logo assets.

A flat, geometric flame drawn with 90 and 45 degree edges only, in the
Tenstorrent brand style (single accent color, chamfered/octagonal silhouette,
transparent negative space). The wordmark uses DejaVu Sans Bold as a stand-in
for the TT brand typeface.

Run with Pillow installed:  python3 generate_logo.py
Writes, next to this file:
    tt_ember_logo.png   full lockup (icon + wordmark), transparent
    tt_ember_icon.png   icon only, transparent
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageChops

OUT_DIR = Path(__file__).resolve().parent
FONT    = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# ---------- palette / scale ----------
SS       = 3                      # supersample factor
EMBER    = (238, 78, 34, 255)     # flame orange  (#EE4E22)
GRAY     = (128, 130, 133, 255)   # wordmark gray (#808285)
ICON_W, ICON_H = 1000, 1200       # icon design box

# ---------- flame geometry (edges are vertical, horizontal, or 45 deg only) ----------
def mirror_half(half):
    """Right-side points (x>=500, top->bottom) -> full closed symmetric polygon."""
    return half + [(1000 - x, y) for (x, y) in reversed(half)]

OUTER_HALF = [
    (500, 260),   # apex
    (620, 380),   # 45 down-right (tip flare)
    (620, 470),   # vertical
    (760, 610),   # 45 widen -> half-width 260
    (760, 820),   # vertical (fat mid wall)
    (680, 900),   # 45 pull in
    (680, 940),   # vertical
    (600, 1020),  # 45 chamfer to base
    (500, 1020),  # horizontal to base center
]
INNER_HALF = [
    (500, 430), (600, 530), (600, 580), (690, 670), (690, 810),
    (620, 880), (620, 915), (555, 980), (500, 980),
]
FLAME       = mirror_half(OUTER_HALF)
INNER_FLAME = mirror_half(INNER_HALF)
CORE        = [(500, 740), (560, 800), (500, 860), (440, 800)]  # ember spark diamond

def sc(poly):
    return [(x * SS, y * SS) for (x, y) in poly]

def render_icon():
    W, H = ICON_W * SS, ICON_H * SS
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)
    dr.polygon(sc(FLAME), fill=EMBER)
    # punch the inner flame out to transparent (true negative space)
    hole = Image.new("L", (W, H), 0)
    ImageDraw.Draw(hole).polygon(sc(INNER_FLAME), fill=255)
    img.putalpha(ImageChops.subtract(img.getchannel("A"), hole))
    dr.polygon(sc(CORE), fill=EMBER)  # ember spark floats in the hole
    return img

# ---------- wordmark:  superscript TT + EMBER + superscript TM ----------
def render_wordmark(target_w):
    big = ImageFont.truetype(FONT, 300)
    sml = ImageFont.truetype(FONT, 120)
    track = 40
    tmp = Image.new("RGBA", (4000, 700), (0, 0, 0, 0))
    d = ImageDraw.Draw(tmp)
    tt_w = d.textlength("TT", font=sml)
    word_w = sum(d.textlength(c, font=big) + track for c in "EMBER") - track
    tm_w = d.textlength("TM", font=sml)
    gap = 30
    x0 = (4000 - (tt_w + gap + word_w + 10 + tm_w)) / 2
    d.text((x0, 250), "TT", font=sml, fill=GRAY, anchor="lm")
    x = x0 + tt_w + gap
    for c in "EMBER":
        d.text((x, 360), c, font=big, fill=GRAY, anchor="lm")
        x += d.textlength(c, font=big) + track
    d.text((x - track + 10, 250), "TM", font=sml, fill=GRAY, anchor="lm")
    tmp = tmp.crop(tmp.getbbox())
    return tmp.resize((target_w, max(1, round(tmp.height * target_w / tmp.width))), Image.LANCZOS)

def build_lockup():
    icon = render_icon()
    icon_w = 1500
    icon = icon.resize((icon_w, round(icon.height * icon_w / icon.width)), Image.LANCZOS)
    word = render_wordmark(round(icon_w * 0.95))
    pad, gap = 120, 120
    CW = max(icon.width, word.width) + pad * 2
    CH = icon.height + gap + word.height + pad * 2
    canvas = Image.new("RGBA", (CW, CH), (0, 0, 0, 0))
    canvas.alpha_composite(icon, ((CW - icon.width) // 2, pad))
    canvas.alpha_composite(word, ((CW - word.width) // 2, pad + icon.height + gap))
    return canvas

if __name__ == "__main__":
    build_lockup().save(OUT_DIR / "tt_ember_logo.png")
    render_icon().save(OUT_DIR / "tt_ember_icon.png")
    print("wrote tt_ember_logo.png and tt_ember_icon.png to", OUT_DIR)
