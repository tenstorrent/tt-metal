# Bug: `topk_large_indices` drops indices on tied bf16 values

**Op:** `ttnn.experimental.topk_large_indices` (branch `pjosipovic/topk_xl`)
**Severity:** correctness — silently drops genuine top-k entries (replaced by duplicates).
**Repro:** `models/demos/deepseek_v32/context/repro_topk_minimal.py` (single Blackhole device; asserts, exit 1).

## Contract vs. actual
- **Expected:** for each row, `k` **distinct** indices = the indices of the top-k values.
- **Actual:** when the input row contains **value ties** (equal bf16 values), some indices
  are emitted multiple times and the same number of genuine top-k indices go missing. The
  duplicated index is the window-base index (0, or 2048 for a second window) — i.e. a tied
  element's index gets collapsed onto the base instead of being preserved. The duplicates
  are real finite-valued indices, **not** the `-1` / `0xFFFFFFFF` pad sentinel.

## Root cause: tie handling in the LLK index carry (NOT the window merge)
The defect is in how a tied element's **index** is carried through the compare-exchange
network during the sort (`_topk_xl_local_sort_` / `separate_indices_row_major` path), when
tied values must be ranked against **distinct** neighbours. It is **not** the cross-window
merge and **not** data movement. Minimal triggering setup: **1 row, n == k == 512, one
window** — so `topk_xl_merge` never runs, and because `k == n` the output must be a
permutation of `0..511`, making any missing/duplicate index unambiguous.

Evidence (all single device, 1 row, 1 window, n = k = 512):

| input | distinct values | result |
|---|---|---|
| `torch.randn` bf16 | 394 / 512 (118 collide — bf16 has 8 mantissa bits) | **506/512 unique; the 6 missing are exactly the tied-value positions; index 0 ×7** |
| 512 strictly-distinct bf16 values, **random order** | 512 / 512 | 512/512 unique — perfect |
| all-equal / two-valued | 1–2 | perfect |

- The strictly-distinct + shuffle control is decisive: random ordering of distinct values
  sorts correctly, so it is **not** an ordering / sort-network bug — only ties break it.
- `ttnn.add(t, 0.0)` round-trips the same input bit-exact, so generic data movement is fine.
- The all-equal / two-valued cases pass because with ≤2 distinct values the sort does no
  real compare-exchange against distinct neighbours, so the buggy path is never exercised —
  "has ties" alone is not sufficient; ties **embedded among many distinct values** are.

## Card count is a red herring
Reproduces on a **single Blackhole chip**, single row, single window — it is not
multi-device specific. A single-chip run only looks clean when it happens to use
distinct-valued input, a clean n/tie geometry, or the test is skipped by the
`skipif(not is_blackhole())` guard.

## Why the branch's own tests missed it
- `_make_bf16_exact_input` builds **strictly distinct** values → never ties → never triggers.
- The new `test_topk_large_indices_random_bfloat16_ties_return_distinct_indices` (n=4096,
  k=2048) does trigger it and currently **fails** — so the owner is aware; the fix has not
  landed (the dedup kernels `compute.cpp` / `ckernel_sfpu_topk_xl.h` are unchanged).

## Impact on DeepSeek-V3.2 DSA
`tt/ops.py::topk_indices` wraps this op for indexer top-k (k=2048). Real indexer logits are
bf16 and tie heavily, so dropped/duplicated indices feed `sparse_mla`, where a dup'd latent
is double-counted in the softmax while a genuinely-selected key is dropped. Sentinel
`test_topk_indices_match[k2048]` is intentionally left RED until the index carry preserves
tied elements' indices.
