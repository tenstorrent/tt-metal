#!/usr/bin/env python3
"""Image-request checks against a running DSV4.1 server (OpenAI chat API): text only, one image, two images, a concurrent mix, streaming. Prints answers + latency."""
import base64
import io
import json
import threading
import time
import urllib.request

from PIL import Image, ImageDraw, ImageFont

URL = "http://127.0.0.1:8000"


def font(sz):
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def b64(im):
    b = io.BytesIO()
    im.save(b, format="PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def card(text, shape, color, bg=(255, 255, 255), size=(512, 384)):
    im = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(im)
    d.text((24, 20), text, fill=(0, 0, 0), font=font(72))
    box = (60, 160, 260, 360)
    (d.rectangle if shape == "square" else d.ellipse)(box, fill=color)
    return im


def chat(content, max_tokens=64, stream=False, model="deepseek-ai/DeepSeek-V4.1-Flash"):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": stream,
    }
    req = urllib.request.Request(
        URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        if not stream:
            j = json.load(r)
            return j["choices"][0]["message"]["content"], time.time() - t0, j["usage"]
        out, first = [], None
        for line in r:
            line = line.decode().strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                d = json.loads(line[6:])["choices"][0]["delta"].get("content")
                if d:
                    first = first or time.time() - t0
                    out.append(d)
        return "".join(out), time.time() - t0, {"ttft": first}


def img(im):
    return {"type": "image_url", "image_url": {"url": b64(im)}}


def txt(t):
    return {"type": "text", "text": t}


def main():
    red_sq = card("CAT 42", "square", (220, 30, 30))
    blue_ci = card("DOG 7", "circle", (30, 60, 220))
    print("== text only")
    print(chat("What is 7 times 8? Answer with just the number."))
    print("== one image")
    print(chat([img(red_sq), txt("What text is written in this image, and what color is the shape?")], 48))
    print("== two images")
    print(
        chat(
            [
                txt("Image 1:"),
                img(red_sq),
                txt("Image 2:"),
                img(blue_ci),
                txt("Which image has a circle, and what number is written on it?"),
            ],
            48,
        )
    )
    print("== concurrent mix (4 image + 4 text)")
    res = {}

    def run(i, content):
        res[i] = chat(content, 40)

    ts = []
    for i in range(8):
        c = (
            [img(red_sq if i % 2 else blue_ci), txt("What color is the shape?")]
            if i < 4
            else f"Say the number {i} in words."
        )
        ts.append(threading.Thread(target=run, args=(i, c)))
        ts[-1].start()
    for t in ts:
        t.join()
    for i in sorted(res):
        print(i, repr(res[i][0]), f"{res[i][1]:.1f}s")
    print("== streaming")
    print(chat([img(blue_ci), txt("Describe the image in one sentence.")], 40, stream=True))


if __name__ == "__main__":
    main()
