"""Six-checks subset, computed not eyeballed. The skill's node validator is not
available on this host, so the two gates that matter for a one-hue two-shade
before/after pair are computed here: OKLab dE between the shades (identity), and
WCAG contrast of each against the chart surface (legibility)."""
import math

def srgb_to_lin(c):
    c /= 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def oklab(h):
    r, g, b = (srgb_to_lin(v) for v in hex_rgb(h))
    l = 0.4122214708*r + 0.5363325363*g + 0.0514459929*b
    m = 0.2119034982*r + 0.6806995451*g + 0.1073969566*b
    s = 0.0883024619*r + 0.2817188376*g + 0.6299787005*b
    l_, m_, s_ = l ** (1/3), m ** (1/3), s ** (1/3)
    return (0.2104542553*l_ + 0.7936177850*m_ - 0.0040720468*s_,
            1.9779984951*l_ - 2.4285922050*m_ + 0.4505937099*s_,
            0.0259040371*l_ + 0.7827717662*m_ - 0.8086757660*s_)

def de(h1, h2):
    a, b = oklab(h1), oklab(h2)
    return 100 * math.dist(a, b)

def lum(h):
    r, g, b = (srgb_to_lin(v) for v in hex_rgb(h))
    return 0.2126*r + 0.7152*g + 0.0722*b

def contrast(fg, bg):
    a, b = sorted((lum(fg), lum(bg)))
    return (b + 0.05) / (a + 0.05)

SURFACE = "#fcfcfb"
MAIN, NEW = "#86b6ef", "#1c5cab"      # blue ramp steps 250 and 550
print(f"shade pair dE (OKLab x100)      {de(MAIN, NEW):5.1f}   need >= 15 for a normal-vision series pair")
for name, h in (("main  #86b6ef", MAIN), ("retuned #1c5cab", NEW),
                ("exact ink #52514e", "#52514e"), ("grid #e1e0d9", "#e1e0d9")):
    print(f"{name:<22} contrast vs surface {contrast(h, SURFACE):5.2f}:1")
print(f"lightness L: main {oklab(MAIN)[0]:.3f}  retuned {oklab(NEW)[0]:.3f}  "
      f"separation {abs(oklab(MAIN)[0]-oklab(NEW)[0]):.3f}")
