#!/usr/bin/env python3
"""
Minimal repro: ttnn.experimental.topk_large_indices drops indices when the input
has VALUE TIES (equal bf16 values) embedded among distinct values.

Shape: 1 row, n == k == 512 (the smallest LLK window), single device.
  * single window  -> topk_xl_merge never runs (the cross-window merge is NOT the cause);
  * k == n         -> the correct output is a permutation of 0..511, every index
                      exactly once, so any repeat / missing index is unambiguous.

Root cause is a tie-handling defect in the LLK top-k INDEX carry, not data movement
and not the window merge. Evidence, all on one device, one row, one window:
  * BUG  case: torch.randn bf16 -> 394/512 distinct values (118 collide; bf16 has only
               8 mantissa bits), and exactly the tied-value positions go missing while
               the window-base index 0 is duplicated.
  * CTRL case: 512 STRICTLY-DISTINCT bf16 values in RANDOM order (zero ties) -> perfect.
               So random ordering of distinct values sorts fine; only ties break it.
  (all-equal / two-valued inputs also pass: with <=2 distinct values the sort does no
   real compare-exchange against distinct neighbours, so the buggy path is never hit.)

Run:  python repro_topk_minimal.py     (exits non-zero on the bug)
"""
import hashlib
import warnings
from collections import Counter

import torch

import ttnn

N = K = 512  # smallest LLK window; k == n => output must be a permutation of 0..N-1
SEED = 0

# Canonical environment + input fingerprint this repro was authored against. torch CPU
# randn(seed) is reproducible within a torch version but can drift across major versions,
# so we pin both and WARN (not fail) if either differs — a non-repro on a mismatched
# input is inconclusive, not a fix.
EXPECTED_TORCH = "2.11.0+cpu"
EXPECTED_INPUT_MD5 = "46abe5e344985b969f88c7784f9ce03f"

dev = ttnn.open_device(device_id=0)


def topk_indices(x):
    t = ttnn.from_torch(x.reshape(1, 1, 1, N), device=dev, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)
    return ttnn.to_torch(ttnn.experimental.topk_large_indices(t, k=K)).long()[0, 0, 0]


def summarize(tag, x):
    vals = x.float()
    n_distinct = vals.unique().numel()
    counts = Counter(vals.tolist())
    idx = topk_indices(x)
    n_unique = idx.unique().numel()
    missing = sorted(set(range(N)) - set(idx.tolist()))
    missing_tied = [m for m in missing if counts[vals[m].item()] > 1]
    print(f"[{tag}] distinct_values={n_distinct}/{N}  ->  topk unique_indices={n_unique}/{K}, missing={len(missing)}")
    if missing:
        dup = {v: c for v, c in Counter(idx.tolist()).items() if c > 1}
        print(f"        duplicated indices: {dup}")
        print(f"        missing indices at TIED-value positions: {len(missing_tied)}/{len(missing)}")
    return n_unique


# ---- BUG case: random bf16 (ties scattered among distinct values) ----
torch.manual_seed(SEED)
x = torch.randn(N, dtype=torch.bfloat16)

bits = x.view(torch.uint16)
input_md5 = hashlib.md5(bits.numpy().tobytes()).hexdigest()
print(f"SEED={SEED}  torch={torch.__version__}  input bf16-bits md5: {input_md5}")
print(f"FIRST8 bf16-hex: {[f'0x{v:04x}' for v in bits[:8].tolist()]}")
print(f"LAST8  bf16-hex: {[f'0x{v:04x}' for v in bits[-8:].tolist()]}")
if torch.__version__ != EXPECTED_TORCH:
    warnings.warn(f"torch {torch.__version__} != authored {EXPECTED_TORCH}; RNG may differ", stacklevel=2)
if input_md5 != EXPECTED_INPUT_MD5:
    warnings.warn(
        f"input md5 {input_md5} != authored {EXPECTED_INPUT_MD5}: this is NOT the canonical "
        f"input (likely a torch RNG difference). A clean result here is inconclusive — pin the "
        f"input from FIRST8/LAST8 bf16-hex above to reproduce exactly.",
        stacklevel=2,
    )
bug_unique = summarize("BUG  randn        ", x)

# ---- CTRL case: 512 strictly-distinct bf16 values, randomly shuffled (zero ties) ----
distinct = torch.tensor([float(2**b) * (1.0 + j / 128.0) for b in range(4) for j in range(128)], dtype=torch.bfloat16)
assert distinct.float().unique().numel() == N, "control input is not strictly distinct"
torch.manual_seed(SEED)
ctrl_unique = summarize("CTRL distinct+shuf", distinct[torch.randperm(N)])

ttnn.close_device(dev)

assert ctrl_unique == K, "control regressed: strictly-distinct input should never duplicate"
assert bug_unique == K, f"topk_large_indices dropped {K - bug_unique} indices to duplicates on tied input"
print("OK: all indices distinct")
