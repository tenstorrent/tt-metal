# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Correct bsz=1 hybrid/fast MTP decode loop for LocateAnything-3B (torch CPU).

The model ships TWO MTP loops:
  * ``modeling_locateanything.py:generate()`` — its SDPA KV-cache truncation
    (``kv[:, :, :generated.shape[1], :]``) keeps the WRONG cache rows after a
    multi-token window forward, so multi-step MTP degenerates (boxes repeat).
  * ``batch_utils/engine_hybrid.py`` (the production runtime) — re-implements the
    loop and unpacks KV as ``old_real_kv + uncached_real_token_kv`` (dropping the
    duplicate+mask window K/V). This is correct but hard-imports cv2/lmdb/decord.

This module reproduces the *correct* engine loop for bsz=1 in pure torch (no
cv2), so it is BOTH the correctness oracle (MTP boxes vs slow-AR boxes) AND the
exact blueprint the tt-nn device port (`tt/mtp.py`) follows.

Per MTP step (steady state, kv already holds `cur_len` real tokens):
  window ids       = [last_real_tok, mask, mask, mask, mask, mask]   (len 6)
  window positions = [cur_len-1,     cur_len, cur_len+1, ..., cur_len+4]
  attention        = window attends to all real kv [0:cur_len] EXCEPT the blocked
                     column (cur_len-1, i.e. kv_len-block_size-1) and is fully
                     bidirectional within the 6-token window.
  readout          = the 6 logits -> sample_tokens/handle_pattern -> box tokens
  KV update        = write the 6 window K/V, then keep only the K/V of accepted
                     real tokens (drop the duplicate+mask rows).

Usage:
  ./python_env/bin/python models/experimental/locate_anything/reference/mtp_cpu_loop.py \
      [--image coco_lvis.png] [--query "person</c>car"] [--in-token-limit 1024] \
      [--mode hybrid|fast]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import la_inputs  # noqa: E402


def build_mtp_mask(cached_len, uncached_len, n_future, dtype=torch.bfloat16):
    """Additive [1,1,q_len, kv_len] mask for one MTP-window forward.

    q_len = uncached_len + n_future, fed ids = [uncached_real..., dup_last, mask*5].
    kv_len = cached_len + q_len.  Key layout:
      [0:cached_len)                 cached real-token K/V
      [cached_len:cached_len+uncached_len)  this step's leading real tokens
      [.. : kv_len)                  the n_future window K/V (dup_last + masks)
    Rows:
      * the first ``uncached_len`` rows are CAUSAL (recomputed real tokens).
      * the last ``n_future`` rows are the window: attend to all real keys EXCEPT
        the blocked column ``window_start_k - 1`` and are bidirectional within the
        window. Reproduces update_causal_mask_for_one_gen_window_2d==build_magi_ranges.
    """
    q_len = uncached_len + n_future
    kv_len = cached_len + q_len
    neg = torch.finfo(dtype).min
    mask = torch.full((1, 1, q_len, kv_len), neg, dtype=dtype)
    window_start_k = kv_len - n_future
    blocked_k = window_start_k - 1
    # causal rows for the uncached real tokens (global positions cached_len..)
    for i in range(uncached_len):
        gpos = cached_len + i  # this row's global kv index too
        mask[0, 0, i, : gpos + 1] = 0.0
    # window rows
    for r in range(uncached_len, q_len):
        mask[0, 0, r, :window_start_k] = 0.0  # all real keys
        mask[0, 0, r, window_start_k:kv_len] = 0.0  # window bidirectional
        if 0 <= blocked_k < kv_len:
            mask[0, 0, r, blocked_k] = neg
    return mask


def mtp_loop(model, tokenizer, bundle, mode="hybrid", n_future=6, max_new_tokens=64, verbose=True, capture=None):
    """Pure-torch correct hybrid/fast MTP loop, bsz=1. Returns (text, stats).

    If ``capture`` is a list, every MTP-window forward appends a dict with the
    exact window inputs and ALL q_len readout logits, so a device port can replay
    byte-identical windows and compute an end-to-end logit PCC:
      {"win_ids", "win_pos", "uncached_len", "cached_len", "logits" [q_len, vocab]}
    """
    from generate_utils import sample_tokens, handle_pattern

    tids = model.token_ids
    mask_tok = tids["default_mask_token_id"]
    im_end = tids["im_end_token_id"]
    box_end = tids["box_end_token_id"]
    coord_lo, coord_hi = tids["coord_start_token_id"], tids["coord_end_token_id"]
    none_id = tids["none_token_id"]
    img_tok = model.config.image_token_index

    input_ids = bundle["input_ids"]
    pixel_values = bundle["pixel_values"].to(model.language_model.dtype)
    grid = torch.from_numpy(bundle["image_grid_hws"]).to(torch.int32)

    with torch.no_grad():
        vit = model.extract_feature(pixel_values, grid)
        vit = torch.cat(vit, dim=0)
        vit = model.mlp1(vit).unsqueeze(0)  # [1,N,hidden]

    prompt_len = input_ids.shape[1]
    full_ids = input_ids[0].tolist()
    gen_ids = []
    cur_mode = "mtp" if mode in ("fast", "hybrid") else "ar"
    forward_steps = 0
    first_step_logits = None  # the first MTP window's 6 readout logits (for device PCC)

    def _gen_pad():
        return torch.tensor([full_ids], dtype=torch.long)

    # ---- iteration 1: prefill the whole prompt to build the real-token KV ----
    with torch.no_grad():
        out = model.language_model(
            input_ids=input_ids,
            visual_features=vit,
            image_token_index=img_tok,
            attention_mask=torch.ones_like(input_ids),
            position_ids=torch.arange(prompt_len).unsqueeze(0),
            use_cache=True,
        )
    forward_steps += 1
    past = out.past_key_values  # real-token KV, len == prompt_len
    cached_len = prompt_len  # tokens whose K/V are committed in `past`

    # KV is committed LAZILY: accepted tokens become the leading "uncached" real
    # tokens of the NEXT window forward (engine_hybrid semantics). After each
    # forward we keep old real KV + the uncached real-token KV, dropping the
    # duplicate-last + mask window K/V.
    while len(full_ids) < prompt_len + max_new_tokens:
        cur_len = len(full_ids)
        uncached_len = cur_len - cached_len

        if cur_mode == "mtp":
            uncached = full_ids[cached_len:]
            win_ids = torch.tensor([uncached + [full_ids[-1]] + [mask_tok] * (n_future - 1)], dtype=torch.long)
            win_pos = torch.tensor(
                [
                    list(range(cached_len, cur_len))  # uncached real positions
                    + [cur_len - 1]  # duplicated last token (its own position)
                    + [cur_len + j for j in range(n_future - 1)]  # masks
                ],
                dtype=torch.long,
            )
            # Pass a 2D key-valid mask; the stock Qwen2Model.forward then builds its
            # own 4D window mask via update_causal_mask_for_one_gen_window_2d (the last
            # input id is a mask token, so the MTP branch activates). This is exactly
            # what generate() does and is verified equivalent to build_mtp_mask.
            kv_len = cached_len + win_ids.shape[1]
            attn2d = torch.ones((1, kv_len), dtype=torch.long)
            with torch.no_grad():
                out = model.language_model(
                    input_ids=win_ids,
                    attention_mask=attn2d,
                    position_ids=win_pos,
                    past_key_values=past,
                    use_cache=True,
                )
            forward_steps += 1
            logits6 = out.logits[:, -n_future:, :]
            if first_step_logits is None:
                first_step_logits = logits6[0].detach().float().clone()  # [n_future, vocab]
            if capture is not None:
                capture.append(
                    {
                        "win_ids": win_ids[0].tolist(),
                        "win_pos": win_pos[0].tolist(),
                        "uncached_len": uncached_len,
                        "cached_len": cached_len,
                        "logits": out.logits[0].detach().float().clone(),  # [q_len, vocab]
                    }
                )
            _, _, x0, box_avg = sample_tokens(logits6, _gen_pad(), tids, keep_k=5, generation_mode=mode)
            nt = x0[0] if bool((box_avg[0] == 0).all()) else box_avg[0]
            op = handle_pattern(nt, tids, mode)
            toks = [int(t) for t in op["tokens"]]

            # Commit KV: keep [0:cur_len] real K/V (old cached + this step's uncached
            # leading real tokens); drop the dup-last + mask window K/V.
            past = tuple((k[:, :, :cur_len, :], v[:, :, :cur_len, :]) for k, v in out.past_key_values)
            cached_len = cur_len
            for t in toks:
                gen_ids.append(t)
                full_ids.append(t)
            if op["type"] == "im_end":
                break
            if mode == "hybrid" and op["type"] == "error_box":
                cur_mode = "ar"
        else:  # AR step (hybrid fallback / slow)
            uncached = full_ids[cached_len:]
            ar_ids = torch.tensor([uncached], dtype=torch.long)
            ar_pos = torch.arange(cached_len, cur_len).unsqueeze(0)
            with torch.no_grad():
                out = model.language_model(
                    input_ids=ar_ids,
                    attention_mask=torch.ones((1, cur_len), dtype=torch.long),
                    position_ids=ar_pos,
                    past_key_values=past,
                    use_cache=True,
                )
            forward_steps += 1
            past = out.past_key_values
            cached_len = cur_len
            _, _, x0, _ = sample_tokens(out.logits[:, -1:, :], _gen_pad(), tids, generation_mode=mode)
            tv = int(x0[0, 0].item())
            gen_ids.append(tv)
            full_ids.append(tv)
            if tv == im_end:
                break
            if mode == "hybrid" and tv == box_end:
                cur_mode = "mtp"

    text = tokenizer.decode(torch.tensor(gen_ids, dtype=torch.long), skip_special_tokens=False)
    stats = {
        "forward_steps": forward_steps,
        "num_tokens": len(gen_ids),
        "num_boxes": text.count("<box>"),
        "first_step_logits": first_step_logits,
    }
    return text, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="coco_lvis.png")
    ap.add_argument("--query", default="person</c>car")
    ap.add_argument("--in-token-limit", type=int, default=1024)
    ap.add_argument("--mode", default="hybrid", choices=["hybrid", "fast"])
    args = ap.parse_args()

    torch.manual_seed(0)
    mp = la_inputs.find_model_path()
    sys.path.insert(0, mp)
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from PIL import Image

    cfg = AutoConfig.from_pretrained(mp, trust_remote_code=True)
    cfg._attn_implementation = "sdpa"
    cfg.text_config._attn_implementation = "sdpa"
    cfg.vision_config._attn_implementation = "sdpa"
    tok = AutoTokenizer.from_pretrained(mp, trust_remote_code=True)
    model = AutoModel.from_pretrained(mp, config=cfg, trust_remote_code=True, torch_dtype=torch.bfloat16).eval()

    img = Image.open(os.path.join(mp, "assets", args.image)).convert("RGB")
    bundle = la_inputs.build_inputs(tok, img, args.query, in_token_limit=args.in_token_limit)

    # slow AR via stock generate
    with torch.no_grad():
        resp_slow = model.generate(
            pixel_values=bundle["pixel_values"].to(torch.bfloat16),
            input_ids=bundle["input_ids"],
            attention_mask=bundle["attention_mask"],
            image_grid_hws=torch.from_numpy(bundle["image_grid_hws"]).to(torch.int32),
            tokenizer=tok,
            use_cache=True,
            max_new_tokens=64,
            generation_mode="slow",
            temperature=0,
        )
    box_slow = resp_slow[0] if isinstance(resp_slow, (list, tuple)) else resp_slow
    print(f"[loop] SLOW (AR)        : {box_slow!r}", flush=True)

    text, stats = mtp_loop(model, tok, bundle, mode=args.mode, max_new_tokens=64)
    print(f"[loop] {args.mode.upper():7s} (our loop): {text!r}", flush=True)
    print(f"[loop] stats: {stats}", flush=True)
    print(f"[loop] MATCH(slow==loop) ? {box_slow.strip() == text.strip()}", flush=True)


if __name__ == "__main__":
    main()
