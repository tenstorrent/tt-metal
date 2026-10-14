# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""HF greedy reference tokens for adapter_cases.json (needs transformers==4.57.1, CPU).

    python make_hf_adapter_reference.py <out.json> <cat.jpg> <video_dir> [max_new_tokens=32]

Set MOLMO2_LONG_VIDEO=/path/video.mp4 to also add a "long-video" case (card-default 384-frame video,
S~34k). Set MOLMO2_ONLY_LONG=1 to run only that case and merge it into an existing <out.json>.
"""
import json
import os
import sys
from pathlib import Path

import torch
import transformers
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

assert transformers.__version__ == "4.57.1", transformers.__version__
out_path, cat, vdir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
max_new = int(sys.argv[4]) if len(sys.argv) > 4 else 32
rev = "e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b"
snap = Path(snapshot_download("allenai/Molmo2-8B", revision=rev))
images = {"logo": Image.open(snap / "molmo_2_logo_RGB.png").convert("RGB"), "cat": Image.open(cat).convert("RGB")}
proc = AutoProcessor.from_pretrained("allenai/Molmo2-8B", revision=rev, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    "allenai/Molmo2-8B", revision=rev, trust_remote_code=True, dtype=torch.bfloat16, device_map="cpu"
).eval()
res = json.load(open(out_path)) if os.path.exists(out_path) else {}
cases = [] if os.environ.get("MOLMO2_ONLY_LONG") else json.load(open(Path(__file__).parent / "adapter_cases.json"))
if os.environ.get("MOLMO2_LONG_VIDEO"):
    cases.append(
        {
            "name": "long-video",
            "content": [
                {"type": "video", "video": os.environ["MOLMO2_LONG_VIDEO"]},
                {"type": "text", "text": "Describe what is happening in this video."},
            ],
        }
    )
for case in cases:
    imgs = [images[c["image"]] for c in case["content"] if c["type"] == "image"]
    vids = [str(vdir / c["video"]) for c in case["content"] if c["type"] == "video"]
    content = [
        {"type": "image", "image": "x"}
        if c["type"] == "image"
        else {"type": "video", "video": "v"}
        if c["type"] == "video"
        else c
        for c in case["content"]
    ]
    text = proc.apply_chat_template(
        [[{"role": "user", "content": content}]], tokenize=False, add_generation_prompt=True
    )
    kw = {"text": text, "return_tensors": "pt", "padding": True}
    if imgs:
        kw["images"] = imgs
    if vids:
        kw["videos"] = vids[0]
    inp = proc(**kw)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new, do_sample=False)
    toks = out[0, inp["input_ids"].shape[1] :].tolist()
    res[case["name"]] = {
        "S": int(inp["input_ids"].shape[1]),
        "tokens": toks,
        "text": proc.tokenizer.decode(toks, skip_special_tokens=True),
    }
    print(case["name"], res[case["name"]]["S"], repr(res[case["name"]]["text"][:100]), flush=True)
json.dump(res, open(out_path, "w"), indent=1)
