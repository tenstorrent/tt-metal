# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# The loop is diffusers' MiniMaxMusic3AutoregressiveStep (encoders.py, Apache-2.0, Copyright 2026 The MiniMax Team
# and The HuggingFace Team) with the depth-decoder helpers imported from it; only bookkeeping was added.
"""CPU fp32 reference control for the end-token behaviour (runs in the diffusers venv, no device):

    source ~/mm3-bringup/common.sh && \
    $MM3_REF_PY $MM3_MODEL_DIR/scripts/end_token_control_cpu.py --max-frames 2500 --seed 7

Reports at which frame the reference emits the end token for the same short lyric the device probe uses and the
end token's per-frame rank / sampling probability.
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import MiniMaxMusic3RVQDepthDecoder
from diffusers.modular_pipelines.minimax_music3 import encoders as E
from transformers import Qwen2Tokenizer, Qwen3ForCausalLM

CAPTION = "Genre: acoustic pop. BPM: 96. Key: C major. A short intimate vocal phrase over one guitar."
LYRICS = "[verse]\nMorning light through the pine"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    dtype = getattr(torch, args.dtype)
    weights = Path(os.environ["MM3_WEIGHTS"])
    out_path = Path(
        args.out or Path(__file__).resolve().parents[1] / "generated" / f"end_token_control_cpu_{args.seed}.json"
    )
    out_path.parent.mkdir(exist_ok=True)

    t0 = time.time()
    tok = Qwen2Tokenizer.from_pretrained(str(weights / "tokenizer"))
    lm = Qwen3ForCausalLM.from_pretrained(str(weights / "language_model"), dtype=dtype).eval()
    dd = MiniMaxMusic3RVQDepthDecoder.from_pretrained(str(weights / "rvq_depth_decoder"), torch_dtype=dtype).eval()
    print(f"[ctl] models loaded in {time.time() - t0:.0f}s", flush=True)

    text = (
        f"{E._IM_START}{E._CAPTION_START}{E._clean_caption(CAPTION)}{E._CAPTION_END}"
        f"{E._LYRICS_START}{E._normalize_lyrics(LYRICS)}{E._LYRICS_END}{E._IM_END}{E._AUDIO_START}"
    )
    ids = tok(text, return_tensors="pt")["input_ids"]
    unc = ids.clone()
    unc[:, 1:-2] = E._AUDIO_CFG_TOKEN_ID
    text_ids = torch.cat((ids, unc), dim=0)
    generator = torch.Generator("cpu").manual_seed(args.seed)

    vocab_mask = torch.ones(lm.config.vocab_size, dtype=torch.bool)
    vocab_mask[E._AUDIO_CODE_OFFSET : E._AUDIO_CODE_OFFSET + E._SEMANTIC_VOCAB_SIZE] = False
    vocab_mask[E._AUDIO_END_TOKEN_ID] = False

    output = lm.model(inputs_embeds=lm.model.embed_tokens(text_ids), use_cache=True)
    past = output.past_key_values
    last_hidden = output.last_hidden_state[:, -1]
    stats, codes, stopped_by = [], [], "max_frames"
    t1 = time.time()
    for frame_index in range(args.max_frames + 1):
        logits = lm.lm_head(last_hidden).float().masked_fill(vocab_mask, -float("inf"))
        conditional, unconditional = logits[0:1], logits[1:2]
        guided = unconditional + (conditional - unconditional) * E._AR_CFG_SCALE
        threshold = torch.topk(conditional, E._AR_CFG_TOP_K, dim=-1).values[..., -1, None]
        guided = guided.masked_fill(conditional < threshold, -float("inf"))
        guided = guided.masked_fill(vocab_mask.unsqueeze(0), -float("inf"))
        # end-token diagnostics (same definitions as ARGenerator.end_token_stats)
        values = torch.nan_to_num(guided, nan=-1e9, posinf=1e9, neginf=-1e9)
        thr = torch.topk(values, E._AR_SAMPLING_TOP_K, dim=-1).values[..., -1, None]
        probs = F.softmax(values.masked_fill(values < thr, -float("inf")), dim=-1)[0]
        c = conditional[0]
        stats.append(
            {
                "rank_conditional": int((c > c[E._AUDIO_END_TOKEN_ID]).sum()),
                "prob_sampling": float(probs[E._AUDIO_END_TOKEN_ID]),
                "logit_gap_conditional": float(c.max() - c[E._AUDIO_END_TOKEN_ID]),
            }
        )
        sampled = E._sample_top_k(guided, generator)
        if int(sampled.item()) == E._AUDIO_END_TOKEN_ID:
            stopped_by = "end_token"
            break
        semantic_code = sampled - E._AUDIO_CODE_OFFSET
        frame_codes, _ = E._generate_depth_codes(lm, dd, last_hidden, semantic_code.repeat(2), generator)
        if frame_index > 0:
            codes.append(frame_codes[0].tolist())
            if len(codes) >= args.max_frames:
                break
        feedback = E._embed_audio_frame(lm, dd, frame_codes)
        output = lm.model(inputs_embeds=feedback, past_key_values=past, use_cache=True)
        past = output.past_key_values
        last_hidden = output.last_hidden_state[:, -1]
        if frame_index % 50 == 0:
            s = stats[-1]
            print(
                f"[ctl] frame {frame_index}: {(time.time() - t1) / (frame_index + 1):.2f} s/frame, end rank {s['rank_conditional']}, "
                f"prob {s['prob_sampling']:.4f}, gap {s['logit_gap_conditional']:.2f}",
                flush=True,
            )
            json.dump({"partial": True, "frames": len(codes), "end_token_stats": stats}, open(out_path, "w"))
    result = {
        "caption": CAPTION,
        "lyrics": LYRICS,
        "seed": args.seed,
        "dtype": args.dtype,
        "max_frames": args.max_frames,
        "stopped_by": stopped_by,
        "frames": len(codes),
        "wall_seconds": time.time() - t1,
        "end_token_stats": stats,
        "codes": codes,
    }
    json.dump(result, open(out_path, "w"))
    print(
        f"[ctl] stopped_by={stopped_by} after {len(codes)} frames ({len(codes) / 25:.1f} s) in {time.time() - t1:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
