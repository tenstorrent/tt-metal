#!/usr/bin/env python3
"""Stress + synthetic accuracy of image input. (1) image after ~450 tokens of text (straddles the 512-token model chunk), (2) image at the very end of a long prompt,
(3) 16 concurrent image requests with 200-token answers, (4) accuracy: N random images (4-digit number + 2 coloured shapes), ask the number and the colours."""
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageDraw

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from client_img import (  # noqa (runs the smoke checks of client_img on import: keep them short)
    b64,
    chat,
    font,
    img,
    txt,
)

COLORS = {
    "red": (220, 30, 30),
    "blue": (30, 60, 220),
    "green": (20, 160, 40),
    "yellow": (240, 210, 20),
    "purple": (140, 40, 180),
    "orange": (250, 130, 10),
}


def sample(rng, size=(640, 480)):
    num = rng.randint(1000, 9999)
    cs = rng.sample(list(COLORS), 2)
    im = Image.new("RGB", size, (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.text((30, 20), str(num), fill=(0, 0, 0), font=font(110))
    d.rectangle((60, 200, 260, 420), fill=COLORS[cs[0]])
    d.ellipse((340, 200, 580, 420), fill=COLORS[cs[1]])
    return im, num, cs


def main():
    long_text = "The quick brown fox jumps over the lazy dog. " * 45  # ~450 tokens
    rng = random.Random(1)
    im, num, cs = sample(rng)
    print("== image after ~450 tokens of text (straddles the model chunk)")
    a, t, u = chat(
        [txt(long_text + " Now look at this image."), img(im), txt("What 4-digit number is written in it?")], 24
    )
    print(a, "| expected", num, "| prompt tokens", u["prompt_tokens"])
    print("== image at the end of a long prompt")
    a, t, u = chat([txt(long_text * 3 + "Describe the shapes in the picture: "), img(im)], 40)
    print(a, "| expected colors", cs, "| prompt tokens", u["prompt_tokens"])
    print("== 16 concurrent image requests, 200 tokens each")
    t0 = time.time()

    def one(i):
        im, num, cs = sample(random.Random(100 + i))
        a, t, u = chat([img(im), txt("Describe this image in detail.")], 200)
        return (str(num) in a), t, u["completion_tokens"]

    with ThreadPoolExecutor(16) as ex:
        rs = list(ex.map(one, range(16)))
    print(
        f"all done in {time.time()-t0:.1f}s; number mentioned in {sum(r[0] for r in rs)}/16; completion tokens {[r[2] for r in rs]}"
    )
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 48
    print(f"== accuracy over {N} random images")
    ok_num = ok_col = 0

    def acc(i):
        rng = random.Random(1000 + i)
        im, num, cs = sample(rng)
        a, _, _ = chat(
            [
                img(im),
                txt(
                    "Answer in this exact format: NUMBER=<4-digit number>; LEFT=<color of the left shape>; RIGHT=<color of the right shape>"
                ),
            ],
            80,
        )
        m = re.search(r"NUMBER\D*(\d{4}).*LEFT\W*(\w+).*RIGHT\W*(\w+)", a, re.S | re.I)
        return a, num, cs, m

    with ThreadPoolExecutor(8) as ex:
        rs = list(ex.map(acc, range(N)))
    bad = []
    for a, num, cs, m in rs:
        n_ok = bool(m) and int(m.group(1)) == num
        c_ok = bool(m) and m.group(2).lower() == cs[0] and m.group(3).lower() == cs[1]
        ok_num += n_ok
        ok_col += c_ok
        if not (n_ok and c_ok) and len(bad) < 5:
            bad.append((a, num, cs))
    print(f"number exact: {ok_num}/{N}; both colours: {ok_col}/{N}")
    for b in bad:
        print("  miss:", repr(b[0]), b[1:])


if __name__ == "__main__":
    main()
