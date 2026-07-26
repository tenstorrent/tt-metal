#!/usr/bin/env python3
"""Does the TT-specific collapse correlate with the prompt crossing the sliding window?

DG's production denoise path is maskless all-attend (denoise_forward.denoise_hidden_forward:
"The production path is maskless all-attend"). HF instead evicts committed positions from the
sliding cache once prompt_len > sliding_window - 1 = 1023, on the 15 sliding_attention layers.

So for prompt_len <= 1023 the two geometries agree exactly, and above it TT lets half its layers
attend to positions HF has dropped. If that is the defect, the 60 TT-specific collapses should sit
above the threshold and the 66 clean questions below it.
"""
import json
import os
import sys

CKPT = "/home/zni/dg_models/diffusiongemma-26B-A4B-it"
PROMPTS = "/tmp/gpqa198"
SUMMARY = "/tmp/dg_gpqa198_stoparm/hifi2_summary.txt"
SLIDING_WINDOW = 1024


def main() -> int:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(CKPT, local_files_only=True)

    tt = {}
    for line in open(SUMMARY, encoding="utf-8"):
        parts = line.split()
        if len(parts) < 3 or not parts[0].startswith("q"):
            continue
        tt[int(parts[0][1:])] = {
            "guard": int(parts[1].split("=")[1]),
            "blocks": int(parts[2].split("=")[1]),
        }

    rows = []
    for index, t in sorted(tt.items()):
        path = f"{PROMPTS}/q{index:03d}.txt"
        if not os.path.exists(path):
            continue
        prompt = open(path, encoding="utf-8").read()
        ids = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True, enable_thinking=True
        )
        n = len(ids["input_ids"]) if hasattr(ids, "keys") else len(ids)
        rows.append({"q": index, "prompt_len": n, "collapsed": bool(t["guard"]), "blocks": t["blocks"]})

    collapsed = [r for r in rows if r["collapsed"]]
    clean = [r for r in rows if not r["collapsed"]]
    print(f"{len(rows)} questions with a TT result: {len(collapsed)} collapsed, {len(clean)} clean\n")

    def stats(label, group):
        if not group:
            print(f"{label:>12}: none")
            return
        lens = sorted(r["prompt_len"] for r in group)
        over = sum(1 for n in lens if n > SLIDING_WINDOW - 1)
        print(
            f"{label:>12}: n={len(lens):>3}  min={lens[0]:>5}  median={lens[len(lens)//2]:>5}  "
            f"max={lens[-1]:>5}  over_1023={over:>3} ({100*over/len(lens):.0f}%)"
        )

    stats("collapsed", collapsed)
    stats("clean", clean)

    print(f"\nblock-0 collapses (prompt_len is the whole committed prefix there):")
    for r in sorted((r for r in collapsed if r["blocks"] == 0), key=lambda r: r["prompt_len"]):
        print(
            f"  q{r['q']:03d} prompt_len={r['prompt_len']:>5} "
            f"{'OVER window' if r['prompt_len'] > SLIDING_WINDOW - 1 else 'within window'}"
        )

    print(f"\nwhich block would first cross prompt_len>1023 (canvas=256 committed per block):")
    for label, group in (("collapsed", collapsed), ("clean", clean)):
        crossing = [max(0, -(-(1024 - r["prompt_len"]) // 256)) for r in group]
        if crossing:
            print(f"  {label:>10}: median first-crossing block = {sorted(crossing)[len(crossing)//2]}")

    json.dump(rows, open("/tmp/promptlen_vs_collapse.json", "w"))
    print("\n-> /tmp/promptlen_vs_collapse.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
