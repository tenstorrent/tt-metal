# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
For each of the 3 genuine failing test cases (idx 26, 56, 71), run the TTNN
model and show the logit values for every answer choice.

Identifies whether the failure is a clear TTNN error or a narrow margin case.
"""
import json
from pathlib import Path

import torch

VERIF = Path("/home/ttuser/ssinghal/PR-fix/tt-metal/models/demos/molmo2/verification")
VIDEO_DIR = VERIF / "videos"
WEIGHT_CACHE = Path("/tmp/molmo2_weight_cache")


# Token IDs: A=32, B=33, C=34, D=35
ANSWER_TOKS = {"A": 32, "B": 33, "C": 34, "D": 35}

FAILING = [
    {"idx": 22, "expected": "B", "ttnn_got": "C"},
    {"idx": 26, "expected": "B", "ttnn_got": "A"},
    {"idx": 71, "expected": "A", "ttnn_got": "B"},
]

# ---- Load test metadata ----
with open(VERIF / "test.jsonl") as f:
    all_tests = [json.loads(l) for l in f]

# ---- Load TTNN model ----
import ttnn
from models.demos.molmo2.tt.hf_loading import load_molmo2_processor, load_molmo2_state_dict
from models.demos.molmo2.tt.model import TtMolmo2Model
from models.demos.molmo2.tt.model_config import Molmo2Config
from models.tt_transformers.tt.ccl import TT_CCL

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 8))
cfg = Molmo2Config(mesh_device=mesh)
ccl = TT_CCL(mesh)
WEIGHT_CACHE.mkdir(parents=True, exist_ok=True)

print("Loading weights...")
sd = load_molmo2_state_dict(dtype=torch.bfloat16)

model = TtMolmo2Model(
    mesh_device=mesh, tt_ccl=ccl, state_dict=sd, weight_cache_path=WEIGHT_CACHE, dtype=ttnn.bfloat16, configuration=cfg
)
del sd

processor = load_molmo2_processor()
print("Model ready\n")

# ---- Run each failing test ----
for entry in FAILING:
    idx = entry["idx"]
    expected = entry["expected"]
    got = entry["ttnn_got"]
    exp_tok = ANSWER_TOKS[expected]
    got_tok = ANSWER_TOKS[got]

    test = all_tests[idx]
    content = test["messages"][0]["content"]
    prompt = next(i["text"] for i in content if i.get("type") == "text")
    url = next(i["video_url"]["url"] for i in content if i.get("type") == "video_url")
    vpath = VIDEO_DIR / url.split("/")[-1]

    print(f"{'='*60}")
    print(f"Test idx={idx}  expected='{expected}' ({exp_tok})  ttnn='{got}' ({got_tok})")
    print(f"Prompt: {prompt[:100]!r}")

    conversation = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "video"}]}]
    formatted = processor.apply_chat_template([conversation], tokenize=False, add_generation_prompt=True)
    if isinstance(formatted, list):
        formatted = formatted[0]
    inputs = processor(text=formatted, videos=str(vpath), return_tensors="pt", padding=True)

    input_ids = inputs["input_ids"]
    token_type_ids = inputs.get("token_type_ids")
    pv = inputs.get("pixel_values_videos")
    pool_idx = inputs.get("video_token_pooling")
    if pv is not None:
        pv = pv.float().unsqueeze(0)
        pool_idx = pool_idx.unsqueeze(0)

    model.reset_kv_cache(user_id=0)
    logits = model.forward_prefill(
        input_ids=input_ids[:1],
        pixel_values=pv[:1] if pv is not None else None,
        pooled_patches_idx=pool_idx[:1] if pool_idx is not None else None,
        token_type_ids=token_type_ids[:1] if token_type_ids is not None else None,
        user_id=0,
    )
    logits_1d = logits[0, 0, :].float()  # [vocab]

    print(f"  S={input_ids.shape[1]}")
    print(f"  Answer logits:")
    for letter, tid in ANSWER_TOKS.items():
        marker = " ← expected" if letter == expected else (" ← TTNN predicted" if letter == got else "")
        print(f"    {letter} (tok {tid}): {logits_1d[tid]:8.4f}{marker}")

    margin = logits_1d[exp_tok] - logits_1d[got_tok]
    print(f"  Margin (expected - predicted): {margin:+.4f}  {'NARROW (<0.5)' if abs(margin)<0.5 else 'WIDE'}")
    top3 = torch.topk(logits_1d, 3)
    print(f"  Top-3 predicted: {[(int(i.item()), f'{v.item():.4f}') for i, v in zip(top3.indices, top3.values)]}")
    print()

ttnn.close_mesh_device(mesh)
