# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
HF GROUND-TRUTH golden for the M3 real-weights run: load the REAL `minimax_m3_vl` checkpoint
(`MiniMaxM3SparseForConditionalGeneration`, the checkpoint's own architecture) on CPU (mmap, bf16,
low_cpu_mem_usage so RAM stays bounded) and compare its greedy output to our TTNN bf4 galaxy run.

Compares:
  • FIRST token (argmax + top-5) for all 8 prompts — HF vs our recorded TTNN tokens.
  • Oracle prompt ("The capital of France is") — 6-token greedy, HF vs TTNN (token-by-token).
The bar (bf4 vs bf16): first-token argmax match + top-5 overlap; later tokens may diverge (expected).
This is COHERENCE+GROUND-TRUTH; no PCC (would need our logit vectors dumped from a separate run).

Standalone (no `test_` prefix → pytest skips it). CPU only, no ttnn. Run in the throwaway uv env:
  export HF_MODEL=/data/vmelnykov/MiniMax-M3-ref
  uv run --no-project --with "transformers==5.14.1" --with "torch" --python 3.10 \
      python models/demos/minimax_m3/tests/golden_hf_first_token.py
"""

import os
import sys
import time

import torch

ORACLE = "The capital of France is"
PROMPTS = [
    ORACLE,
    "Once upon a time",
    "The meaning of life is",
    "Water boils at",
    "The opposite of hot is",
    "Two plus two equals",
    "The sun rises in the",
    "Roses are red, violets are",
]
# Our TTNN bf4 galaxy results (galaxy_generate_m3.py, 6-token greedy, recorded 2026-06-23).
TTNN = {
    0: [200059, 758, 3100, 355, 11752, 258],
    1: [200059, 758, 3100, 760, 4882, 258],
    2: [200059, 758, 3100, 355, 11752, 894],
    3: [200059, 758, 3100, 1127, 1367, 494],
    4: [200059, 758, 3100, 355, 11752, 258],
    5: [200059, 758, 3100, 355, 11752, 258],
    6: [200059, 758, 3100, 760, 41362, 292],
    7: [200059, 758, 3100, 355, 8061, 258],
}
GEN = 3  # oracle multi-token depth (each step is a slow offloaded CPU forward)


def main():
    from transformers import AutoTokenizer
    from transformers.models.minimax_m3_vl import MiniMaxM3SparseForConditionalGeneration

    hf = os.environ["HF_MODEL"]
    t0 = time.time()
    print(f"[golden] loading tokenizer + model from {hf} (CPU, mmap, bf16) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
    # 869GB bf16 model > 566GB RAM -> CANNOT fit on CPU. Use accelerate device_map disk-offload:
    # keep ~500GB resident, spill the rest to LOCAL ext4 (/tmp is local, 2.6TB free; NOT the NFS
    # weights mount). Forward streams the offloaded layers from local disk (~slow but correct).
    offload = os.environ.get("OFFLOAD_DIR", "/tmp/m3_offload")
    os.makedirs(offload, exist_ok=True)
    model = MiniMaxM3SparseForConditionalGeneration.from_pretrained(
        hf,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        max_memory={"cpu": "500GiB"},
        offload_folder=offload,
        attn_implementation="eager",
    )
    model.eval()
    print(f"[golden] model loaded in {time.time()-t0:.0f}s", flush=True)

    def last_logits(ids):
        with torch.no_grad():
            out = model(input_ids=torch.tensor([ids], dtype=torch.long), use_cache=False)
        return out.logits[0, -1].float()

    def chat_ids(p):
        out = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=True)
        if isinstance(out, str):  # this tokenizer's template ignores tokenize=True -> encode explicitly
            out = tok(out, add_special_tokens=False)["input_ids"]
        if hasattr(out, "input_ids"):
            out = out.input_ids
        if out and isinstance(out[0], (list, tuple)):  # batched -> first row
            out = out[0]
        return [int(t) for t in out]

    # --- ORACLE first (so the key result lands early even if CPU is slow) ---
    print(f"\n[golden] ===== ORACLE multi-token (HF greedy vs TTNN bf4) =====", flush=True)
    ids = chat_ids(ORACLE)
    cur, hf_gen = list(ids), []
    for g in range(GEN):
        s = time.time()
        nxt = int(last_logits(cur).argmax())
        hf_gen.append(nxt)
        cur.append(nxt)
        print(f"[golden]   oracle step {g+1}/{GEN}: id={nxt} {tok.decode([nxt])!r} ({time.time()-s:.0f}s)", flush=True)
    nmatch = sum(1 for a, b in zip(hf_gen, TTNN[0]) if a == b)
    print(f"[golden] HF   oracle ids={hf_gen} -> {tok.decode(hf_gen)!r}", flush=True)
    print(f"[golden] TTNN oracle ids={TTNN[0]} -> {tok.decode(TTNN[0])!r}", flush=True)
    print(f"[golden] >>> ORACLE token match: {nmatch}/{GEN}", flush=True)

    # --- first-token + top-5 for all 8 ---
    print(f"\n[golden] ===== FIRST-TOKEN: HF top-1/top-5 vs TTNN (all 8) =====", flush=True)
    hits = 0
    for r, p in enumerate(PROMPTS):
        lg = last_logits(chat_ids(p))
        top5 = torch.topk(lg, 5).indices.tolist()
        hf1, tt1 = top5[0], TTNN[r][0]
        verdict = "MATCH" if hf1 == tt1 else ("in-top5" if tt1 in top5 else "MISS")
        hits += hf1 == tt1
        print(
            f"[golden] p{r} {p!r}: HF top1={hf1} {tok.decode([hf1])!r} | "
            f"TTNN={tt1} {tok.decode([tt1])!r} -> {verdict}  (top5={top5})",
            flush=True,
        )
    print(f"\n[golden] >>> FIRST-TOKEN argmax match: {hits}/8", flush=True)
    print(f"[golden] DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
