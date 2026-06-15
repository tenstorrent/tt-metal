# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Torch-CPU MTP correctness oracle for NVIDIA LocateAnything-3B.

Finds an (image, query) that yields REAL coordinate boxes (not <box>None</box>),
then runs the HF (torch CPU) LocateAnything-3B `generate()` in `slow` (pure AR),
`hybrid` (MTP+AR fallback, the model's default eval mode), and `fast` (MTP-only)
modes on the SAME inputs and reports whether the decoded box strings MATCH.

LocateAnything is "evaluated in Hybrid Mode by default" (README). Hybrid is the
AR-faithful MTP path; that is the correctness target the device port reproduces.

Also dumps everything the device test needs (matching the bench/golden pipeline):
  input_ids, attention_mask, pixel_values, image_grid_hws, n_img_tokens,
  vit_proj (fp32 projector output), prefill_logits, plus the box strings and the
  HF token_ids dict + block_size + the captured representative MTP 4D mask.

Saved to reference/mtp_oracle.pt. NEVER overwrites golden.pt.

Usage:
  ./python_env/bin/python models/experimental/locate_anything/reference/mtp_oracle.py \
      [--in-token-limit 1024]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import la_inputs  # noqa: E402

ORACLE_PATH = os.path.join(os.path.dirname(__file__), "mtp_oracle.pt")

# (image asset, query) candidates likely to yield real boxes; first that produces
# >=1 coordinate box AND matching slow/hybrid is chosen.
CANDIDATES = [
    ("coco_lvis.png", "person</c>car"),
    ("dense_object_detection.png", "person"),
    ("coco_lvis.png", "person"),
    ("referring.png", "person"),
    ("pointing.png", "person"),
    ("teaser.jpg", "person</c>car"),
]


def _gen(model, tokenizer, bundle, mode, in_token_limit, n_future=6, verbose=False):
    input_ids = bundle["input_ids"]
    attention_mask = bundle["attention_mask"]
    pixel_values = bundle["pixel_values"].to(torch.bfloat16)
    grid = torch.from_numpy(bundle["image_grid_hws"]).to(torch.int32)
    kwargs = dict(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_grid_hws=grid,
        tokenizer=tokenizer,
        use_cache=True,
        max_new_tokens=64,
        n_future_tokens=n_future,
        generation_mode=mode,
        temperature=0,
    )
    if verbose:
        kwargs["verbose"] = True
    with torch.no_grad():
        resp = model.generate(**kwargs)
    if isinstance(resp, tuple):
        return resp[0], (resp[2] if len(resp) > 2 else None)
    return resp, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-token-limit", type=int, default=1024)
    args = ap.parse_args()

    torch.manual_seed(0)
    mp = la_inputs.find_model_path()
    sys.path.insert(0, mp)
    print(f"[oracle] model path: {mp}", flush=True)

    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from PIL import Image

    config = AutoConfig.from_pretrained(mp, trust_remote_code=True)
    config._attn_implementation = "sdpa"
    config.text_config._attn_implementation = "sdpa"
    config.vision_config._attn_implementation = "sdpa"

    tokenizer = AutoTokenizer.from_pretrained(mp, trust_remote_code=True)
    print("[oracle] loading model (bf16, sdpa) ...", flush=True)
    model = AutoModel.from_pretrained(mp, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16).eval()
    block_size = getattr(model.config.text_config, "block_size", 6)
    n_future = 6
    print(f"[oracle] block_size={block_size}", flush=True)

    chosen = None
    for asset, query in CANDIDATES:
        path = os.path.join(mp, "assets", asset)
        if not os.path.exists(path):
            continue
        img = Image.open(path).convert("RGB")
        bundle = la_inputs.build_inputs(tokenizer, img, query, in_token_limit=args.in_token_limit)
        box_slow, _ = _gen(model, tokenizer, bundle, "slow", args.in_token_limit, n_future)
        n_real_boxes = box_slow.count("<box><")  # coordinate boxes start "<box><"
        print(f"[oracle] try {asset!r} q={query!r}: slow={box_slow!r} real_boxes={n_real_boxes}", flush=True)
        if n_real_boxes >= 1:
            chosen = (asset, query, img, bundle, box_slow)
            break

    if chosen is None:
        print("[oracle] WARNING: no candidate produced real coordinate boxes; using first candidate.", flush=True)
        asset, query = CANDIDATES[0]
        img = Image.open(os.path.join(mp, "assets", asset)).convert("RGB")
        bundle = la_inputs.build_inputs(tokenizer, img, query, in_token_limit=args.in_token_limit)
        box_slow, _ = _gen(model, tokenizer, bundle, "slow", args.in_token_limit, n_future)
        chosen = (asset, query, img, bundle, box_slow)

    asset, query, img, bundle, box_slow = chosen
    print(f"\n[oracle] CHOSEN image={asset!r} query={query!r}", flush=True)
    print(f"[oracle] SLOW (AR)    : {box_slow!r}", flush=True)

    box_hybrid, info_h = _gen(model, tokenizer, bundle, "hybrid", args.in_token_limit, n_future, verbose=True)
    print(f"[oracle] HYBRID (MTP) : {box_hybrid!r}", flush=True)
    if info_h:
        print(info_h, flush=True)
    box_fast, info_f = _gen(model, tokenizer, bundle, "fast", args.in_token_limit, n_future, verbose=True)
    print(f"[oracle] FAST (MTP)   : {box_fast!r}", flush=True)
    if info_f:
        print(info_f, flush=True)

    match_hybrid = box_slow.strip() == box_hybrid.strip()
    match_fast = box_slow.strip() == box_fast.strip()
    print(f"\n[oracle] HYBRID boxes == AR boxes ? {match_hybrid}", flush=True)
    print(f"[oracle] FAST   boxes == AR boxes ? {match_fast}", flush=True)

    # ---- Capture a representative MTP 4D mask + positions from a hybrid run ----
    captured = {}
    lm = model.language_model.model
    layer0_attn = lm.layers[0].self_attn
    orig = layer0_attn.forward

    def spy(hidden_states, attention_mask=None, position_ids=None, **kw):
        if hidden_states.shape[1] == n_future:  # the MTP window forward
            captured["mask"] = None if attention_mask is None else attention_mask.detach().cpu().clone()
            captured["position_ids"] = None if position_ids is None else position_ids.detach().cpu().clone()
            captured["q_len"] = hidden_states.shape[1]
        return orig(hidden_states, attention_mask=attention_mask, position_ids=position_ids, **kw)

    layer0_attn.forward = spy
    _gen(model, tokenizer, bundle, "hybrid", args.in_token_limit, n_future)
    layer0_attn.forward = orig

    # ---- Build the same prefill goldens the bench/device path consumes ----
    vit_list = (
        model.extract_feature(
            bundle["pixel_values"].to(torch.float32), torch.from_numpy(bundle["image_grid_hws"]).to(torch.int32)
        )
        if False
        else None
    )  # vit in fp32 below
    # vision goldens in fp32 (matches run_reference precision-first policy)
    model.vision_model.float()
    model.mlp1.float()
    with torch.no_grad():
        vit_list = model.extract_feature(
            bundle["pixel_values"].to(torch.float32), torch.from_numpy(bundle["image_grid_hws"]).to(torch.int32)
        )
        vit_raw = torch.cat(vit_list, dim=0)
        vit_proj = model.mlp1(vit_raw).float()
    model.vision_model.to(torch.bfloat16)
    model.mlp1.to(torch.bfloat16)

    llm_dtype = model.language_model.get_input_embeddings().weight.dtype
    with torch.no_grad():
        out = model.language_model(
            input_ids=bundle["input_ids"],
            visual_features=vit_proj.to(llm_dtype).unsqueeze(0),
            image_token_index=la_inputs.IMAGE_TOKEN_INDEX,
            attention_mask=bundle["attention_mask"],
            use_cache=False,
            output_hidden_states=False,
        )
        prefill_logits = out.logits.float()

    oracle = {
        "image_asset": asset,
        "query": query,
        "in_token_limit": args.in_token_limit,
        "input_ids": bundle["input_ids"],
        "attention_mask": bundle["attention_mask"],
        "pixel_values": bundle["pixel_values"],
        "image_grid_hws": bundle["image_grid_hws"],
        "grid_hw": bundle["grid_hw"],
        "n_img_tokens": bundle["n_img_tokens"],
        "image_token_index": la_inputs.IMAGE_TOKEN_INDEX,
        "vit_proj": vit_proj,
        "prefill_logits": prefill_logits,
        "box_slow": box_slow,
        "box_hybrid": box_hybrid,
        "box_fast": box_fast,
        "match_hybrid": match_hybrid,
        "match_fast": match_fast,
        "n_future": n_future,
        "block_size": block_size,
        "token_ids": dict(model.token_ids) if hasattr(model, "token_ids") else None,
        "rep_mtp_mask": captured.get("mask"),
        "rep_mtp_position_ids": captured.get("position_ids"),
        "rep_mtp_q_len": captured.get("q_len"),
    }
    torch.save(oracle, ORACLE_PATH)
    print(f"[oracle] token_ids: {oracle['token_ids']}", flush=True)
    print(f"[oracle] saved -> {ORACLE_PATH}", flush=True)

    if captured.get("mask") is not None:
        m = captured["mask"]
        nf = n_future
        print(f"[oracle] rep MTP mask shape={tuple(m.shape)} q_len={captured.get('q_len')}", flush=True)
        sub = m[0, 0, -nf:, -(nf + 2) :]
        print("[oracle] mask window [-nf:, -(nf+2):] (1=attend,0=block):", flush=True)
        print((sub == 0).int().tolist(), flush=True)
        print(f"[oracle] position_ids last {nf+2}: {captured['position_ids'][0, -(nf+2):].tolist()}", flush=True)


if __name__ == "__main__":
    main()
