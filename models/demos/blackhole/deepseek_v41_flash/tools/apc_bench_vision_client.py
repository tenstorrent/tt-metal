#!/usr/bin/env python3
"""Vision guard check against a DSV41_VISION=1 server with/without APC: text prompts that share the 8-shot header (they use the cache) mixed with image prompts that share the SAME header text before the
image (they must not): sequential + concurrent, greedy; outputs saved for an on/off comparison. usage: apc_vision_client.py out.json"""
import json
import sys
import threading

sys.path.insert(0, "/mnt/tt-data/ssinghal/dsv4-logs/vis")
sys.path.insert(0, "/mnt/tt-data/ssinghal/dsv4-logs/apc")
import apc_client as AC
import client_img as CI


def chat(content, mt=64):
    return CI.chat(content, mt)[0]


HEADER = AC.HEADER
red = CI.card("CAT 42", "square", (220, 30, 30))
blue = CI.card("DOG 7", "circle", (30, 60, 220))
green = CI.card("OWL 9", "square", (20, 160, 40))


def t(q):
    return [CI.txt(HEADER + f"Q: {q}\nA:")]


def i(im, q):
    return [CI.txt(HEADER), CI.img(im), CI.txt(f"Q: {q}\nA:")]


Q1 = "What is 12 times 12?"
QI = "What text is written in the image and what colour is the shape?"
res = {}
seq = [
    ("t1", t(Q1)),
    ("t1b", t(Q1)),
    ("t2", t("What is 15 plus 27?")),
    ("i_red", i(red, QI)),
    ("i_red_b", i(red, QI)),
    ("i_blue", i(blue, QI)),
    ("i_green", i(green, QI)),
    ("t1c", t(Q1)),
    ("i_red_c", i(red, QI)),
    ("t3", t("What is 9 times 7?")),
]
for name, c in seq:
    res[name] = chat(c, 48)
    print(name, repr(res[name][:100]), flush=True)
conc = {}


def run(k, c):
    conc[k] = chat(c, 48)


jobs = [
    (f"c{k}", i([red, blue, green][k % 3], QI)) if k < 4 else (f"c{k}", t(f"What is {k} plus {k}?")) for k in range(8)
]
ts = [threading.Thread(target=run, args=j) for j in jobs]
[x.start() for x in ts]
[x.join() for x in ts]
res.update(conc)
print("conc", {k: v[:60] for k, v in conc.items()}, flush=True)
json.dump(res, open(sys.argv[1], "w"))
print("DONE")
