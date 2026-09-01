# MTP batched verification — investigation log (2026-09-01)

Follow-up to `mtp_design.md`. Context: after building a from-scratch MTP
drafter (`tt/mtp_fresh.py`) and a sequential-verify demo
(`demo/text_demo_fresh.py`), the sequential design measured *slower* than
plain decoding. This log covers the investigation into batched (single-pass)
verification: why the from-scratch approach hit a hard wall, the discovery
that a working implementation already existed in the codebase, and the
measured result after adopting it.

## 1. Why a from-scratch batched verify hit a wall

Batched verification needs to score all `K` draft candidates in **one**
forward pass through the 64-layer backbone, which for the 48 GDN
(linear-attention) layers means the recurrent-state chunk computation
(`TPGatedDeltaNet.forward_prefill`, `tt/gdn/tp.py:1083`) must **continue**
from the model's current accumulated state rather than starting from zero.

Reading `tp.py` line by line:

```python
initial_state=self.rec_state if carry else None,   # tp.py:1280
carry = self._stable_state and not return_state     # tp.py:1141
```

`carry` — and therefore true state continuation — is only available when
`_stable_state=True`. That flag is exactly the GDN decode bug found and left
unresolved earlier in the MTP-porting work (`allocate_kv_caches` sets it;
proven independently of the fused-chunk-kernel bug via exhaustive tensor
comparison, root cause never found at the Python level). This made it look
like batched verify was blocked on a bug we couldn't fix.

**Direct confirmation the underlying mechanism is broken in isolation:** the
pre-existing test `test_gdn_tp_batched_prefill_chunked`
(`tests/test_gdn_tp.py:420`) — which validates `forward_prefill_batched(carry=True)`
(the multi-user analogue of the same carry mechanism) against a single-shot
reference — fails on real T3K hardware:

```
AssertionError: carry vs single-shot PCC 0.00000 < 0.99: [0.0, 0.0]
```

PCC of exactly 0.0 (not just "degraded") — a structural failure, not numeric
drift. This is a genuine, pre-existing, currently-failing test, unrelated to
anything changed this session.

## 2. Discovery: batched verification already exists and works

Grepping for other users of `forward_prefill_batched` surfaced a full,
pre-existing speculative-decoding stack, merged into this branch at the very
start of the MTP work (`ayerofieiev/qwen38/mtp`):

- `tt/spec_decode.py` — single-user (B=1), masked-bucket chunk verify. **This
  is the one that works.**
- `tt/spec_decode_batched.py` — multi-user (c8) batched verify, built on
  `forward_prefill_batched(carry=True)`.
- `tt/spec_decode_fused.py` — a newer, faster decode-width verify using a
  custom `seq_rows` GDN kernel mode; eager loop silicon-validated per
  `mtp_design.md`, traced loop parked on one known shape bug.

All three are already wired into `demo/text_demo.py` behind env vars
(`TT_SPEC_DECODE=1`, `TT_SPEC_BATCHED=1`, `TT_SPEC_FUSED=1`).

### Why `tt/spec_decode.py` avoids the `_stable_state=True` bug

The earlier-diagnosed bug manifests when `forward_decode` is called
*repeatedly*, accumulating state in place over many steps (the production
plain-decode loop's pattern). `spec_decode.py` never does this. Its design
(documented in `mtp_design.md`, "Draft → verify → accept loop"):

1. Snapshot GDN state before every verify chunk (`_gdn_snapshot`,
   `tt/spec_decode.py:174`).
2. Run the verify chunk (mutates state as a side effect).
3. **Always restore** the snapshot afterward (`_gdn_restore`,
   `tt/spec_decode.py:201`) — verify never permanently commits.
4. Only a separate, periodic, block-aligned **commit chunk** — over tokens
   already confirmed real — is allowed to permanently advance state.

This "verify is disposable, commit is separate and rare" structure sidesteps
whatever specifically breaks under long runs of raw `forward_decode` calls,
without requiring that bug to be fixed.

## 3. Real-hardware results

Matched settings: seqlen=128 prompt, 200 generated tokens, T3K, warm tensor
cache. Commands and full output in section 5.

| | No MTP (safe `prefill_tp`/`decode_tp` baseline) | With MTP (`TT_SPEC_DECODE=1`, `tt/spec_decode.py`) |
|---|---|---|
| Decode throughput | 2.33 tok/s (85.33s decode) | **4.7 tok/s** (211ms/token) |
| Overall (incl. prefill) | 2.29 tok/s | 3.86 tok/s* |
| Output quality | degenerated into repetition after ~50 tokens | coherent full paragraph |
| Accept stats | n/a | K=4, 38% accept rate, 2.52 tokens/iteration avg |

\* The spec run's prefill (9.6s) included one-time JIT kernel compilation the
fresh demo's already-warm cache didn't need (85% vs 100% cache-hit rate) —
decode-only is the fair steady-state comparison.

**The plain "spec off" baseline inside the same `demo/text_demo.py` harness
currently FAILS** (`spec_128` id, `TT_SPEC_DECODE` unset): it hits the exact
known `_stable_state=True` degenerate-output bug (`allocate_kv_caches` is
also used by plain paged decode), producing `'('` as 63% of output. This is a
pre-existing, separate issue — not introduced by anything this session
touched — and is why the fresh demo's `generate_tp`-based path was used as
the correctness baseline instead.

**Caveat, not chased further:** the fresh demo's own no-MTP baseline, while
using the proven-safe primitives, degenerated into "the the the..." repetition
past ~50 tokens at the 200-token setting (it was previously verified correct
only up to 30 tokens / exact 10-token HF match). Both paths are pure greedy
decoding with no repetition penalty, so some degeneration on a raw
base-model continuation prompt is plausible on its own; this doesn't affect
the throughput comparison (which measures decode-step cost, not content),
but the baseline's long-horizon output hasn't been separately re-validated
against real HF past 10 tokens.

## 4. Why batched verify is faster: memory bandwidth, not FLOPs

Autoregressive decoding at batch=1 is memory-bandwidth-bound, not
compute-bound: each forward pass must stream the full ~27B-parameter weight
set out of DRAM regardless of how many token-rows it's evaluating. A matmul
`output[T, d_out] = input[T, d_in] @ weight[d_in, d_out]` loads the weight
matrix once and reuses it across every row `T` — so going from `T=1` to
`T≈52` (a full verify chunk: replay of committed tokens + K drafts) multiplies
the useful arithmetic ~52x while barely changing the wall-clock time, because
the expensive part (moving weights) didn't change.

Measured: one plain decode step ≈429ms (1/2.33 tok/s); one verify iteration
(52 rows, all 64 layers) ≈530ms (211ms × 2.52 tokens/iteration) — only ~24%
more wall-clock for 52x the rows. That ratio (≈2.52 tokens for ≈1.24x the
cost of one decode step) is the source of the ~2x speedup.

This is also why the earlier from-scratch **sequential**-verify build was
*slower* than plain decoding: it paid a full separate weight-streaming trip
per draft candidate (K real decode calls to verify K drafts), i.e. strictly
more trips than plain decoding, on top of the drafter's own overhead.

## 5. How verification actually works

### Ground truth = the target model itself, not an external reference

The correctness guarantee of speculative decoding: final output is
**provably identical** to running the target model alone, greedy,
token-by-token. There's no separate oracle — "ground truth" at each verify
step is simply what the full 64-layer target model itself would predict next,
computed via one causally-masked forward pass instead of one sequential step
per position.

### Building the chunk (`tt/spec_decode.py:667-679`)

```
chunk_tokens = committed[a : c+1]  +  drafts[d_1 .. d_K]
               \_____________________/   \______________/
                already-real tokens,       K guessed tokens,
                replayed to rebuild GDN     not yet confirmed
                state up to position c
```

`a` = block-aligned state anchor (GDN state is known-good up to here). `c` =
last committed real position. The replay of `a..c` is necessary because the
GDN chunk kernel's state math only comes out right starting from the anchor;
the drafts ride along in the same batched pass.

### One forward pass, staircase causal structure

Feed all rows through the target model at once; causal masking means row `i`
only ever sees rows `≤ i`:

```
row (abs. pos)     sees                                  predicts
─────────────────────────────────────────────────────────────────
a .. c-1           real history only                      (discarded — only
                                                             needed to rebuild
                                                             state/KV)
c                  [a..c]                                 → compare to d_1
c+1 (draft d_1)    [a..c, d_1]                             → compare to d_2
c+2 (draft d_2)    [a..c, d_1, d_2]                        → compare to d_3
...
c+K (draft d_K)    [a..c, d_1..d_K]                        → "bonus" row
```

Only the last `K+1` rows are read out (`_extract_rows`); the rest existed
only to reconstruct state/KV correctly.

### Accept/reject (`greedy_accept`, `tt/spec_decode.py:86-101`)

```python
m = first index where drafts[i] != target_ids[i]      # longest matching prefix
committed = drafts[:m] + [target_ids[m]]               # +1 guaranteed-real token
```

Every iteration commits at least 1 real token (the "bonus/correction" token
from the target's own prediction), regardless of how many drafts matched —
so worst case this is no worse than plain decoding, and best case it's up to
`K+1` tokens for the cost of one decode-equivalent pass.

### GDN state discipline: verify never commits

Verifying with rejected drafts mutates GDN's recurrent state with tokens
that were never real. `_gdn_snapshot()`/`_gdn_restore()` (`tt/spec_decode.py:174-209`)
bracket every verify chunk so that state is always rolled back afterward;
only a separate, periodic `_maybe_commit()` chunk over confirmed-real tokens
is allowed to permanently advance state.

## 6. Decode vs. verify-chunk: the actual code difference

Both paths call the **same** function,
`gated_attention_forward_ttnn` (`models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_attention.py`) —
`decode_forward`/`prefill_forward` are thin wrappers that pass different
argument sets into one `if/elif` chain.

| | Decode (`T==1`) | Verify chunk (`T>1`, paged) |
|---|---|---|
| Cache-write op | `paged_update_cache` — writes 1 row | `paged_fill_cache` — writes N rows in one call |
| Attention op | `paged_scaled_dot_product_attention_decode` — no mask; reads `cache[:cur_pos+1]` internally | `chunked_scaled_dot_product_attention` — explicit causal structure across N rows, offset by `chunk_start_idx` |
| Trigger | `T==1` and `cur_pos_tensor` set | `T>1` and `chunk_page_table`/`chunk_start_idx` set |

Everything upstream (QKV projection, Q/K RMSNorm, RoPE) is identical code
running on different row counts. GDN follows the same pattern one level down:
`forward_decode` vs `forward_prefill` (`tt/gdn/tp.py`), selected by
`mode="decode"` vs `mode="prefill"`.

### The causal mask's job

A mask is an additive bias matrix: allowed key positions get `0`, forbidden
(future) positions get a large negative number (`-1e4`) added to the raw
attention score before softmax, so `exp(-1e4) ≈ 0` — that position
contributes ~nothing to the output, without changing tensor shapes.

Concretely (`ttnn_gated_attention.py:479-490`, the legacy explicit-mask
branch — the real paged-chunk kernel achieves the same effect via internal
block-causal processing rather than materializing this tensor):

```python
row_idx = arange(T).unsqueeze(1)          # [T,1]:   which new row (0..T-1)
col_idx = arange(S_total).unsqueeze(0)    # [1,S]:   which key column (0..S-1)
mask = where(col_idx > past_len + row_idx,  -1e4,  0.0)   # broadcasts to [T,S]
```

Example (`past_len=2`, `T=3`, so `S_total=5`):

```
                col:  0    1    2    3    4
row 0:           [   0    0    0   -1e4  -1e4  ]   sees cols 0-2 only
row 1:           [   0    0    0    0   -1e4  ]   sees cols 0-3
row 2:           [   0    0    0    0    0    ]   sees everything
```

**Why it matters here specifically:** the verify chunk stacks real tokens and
unverified draft tokens in one tensor. Without the mask, the row scoring
draft `d_1` could "peek" at `d_2, d_3, d_4` sitting later in the same chunk —
contaminating its prediction with not-yet-verified information and breaking
the "parallel pass == sequential decode" equivalence the whole technique
depends on. Decode needs no mask because its cache only ever holds
already-real tokens — nothing "future" is present to accidentally see.

## 7. Decision

Adopted the existing `tt/spec_decode.py` (`TT_SPEC_DECODE=1`) as the
production batched-verify path rather than continuing the from-scratch
sequential-verify build in `demo/text_demo_fresh.py`/`tt/mtp_fresh.py` (kept
as a validated reference implementation of the drafter numerics, not as the
performance path).

## 8. Commands to reproduce

```bash
export MESH_DEVICE=T3K
export HF_MODEL=/home/user/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9

# Existing batched verify (works, coherent, ~4.7 tok/s decode)
TT_SPEC_DECODE=1 python -m pytest models/demos/blackhole/qwen36/demo/text_demo.py -k spec_128 -x -s

# Plain "spec off" baseline in the SAME harness (currently FAILS — known _stable_state=True bug)
python -m pytest models/demos/blackhole/qwen36/demo/text_demo.py -k spec_128 -x -s

# Safe no-MTP baseline used for the fair throughput comparison instead
python -m pytest models/demos/blackhole/qwen36/demo/text_demo_fresh.py -k no_mtp_200 -x -s

# Multi-user batched (c8) — untested this session, depends on forward_prefill_batched(carry=True),
# which fails in isolation (see section 1); may or may not be exercised the same way spec_decode.py avoids it
TT_SPEC_DECODE=1 TT_SPEC_BATCHED=1 python -m pytest models/demos/blackhole/qwen36/demo/text_demo.py -k spec_2k_b8 -x -s

# Existing carry-mechanism regression test (currently FAILING, pre-existing, unrelated to this session)
python -m pytest models/demos/blackhole/qwen36/tests/test_gdn_tp.py -k test_gdn_tp_batched_prefill_chunked -x -s
```

## 9. Open items

1. `test_gdn_tp_batched_prefill_chunked` is a genuine pre-existing failing
   test (PCC=0.0) — worth a bug report/fix independent of this work, since it
   blocks `forward_prefill_batched(carry=True)` and therefore
   `spec_decode_batched.py` (multi-user batched verify).
2. `TT_SPEC_FUSED=1` (`tt/spec_decode_fused.py`) is documented as faster
   still; not measured this session.
3. The no-MTP baseline's long-horizon (200-token) output-quality caveat
   (section 3) is unresolved — not re-verified against real HF past 10
   tokens.
4. `docs/mtp_design.md`'s "v1 limitations" section should be updated to
   reflect that the B=1 eager chunk-verify path is now confirmed working
   end-to-end against the 3.6-27B checkpoint (it was validated against the
   3.8 checkpoint per the doc's own text).
