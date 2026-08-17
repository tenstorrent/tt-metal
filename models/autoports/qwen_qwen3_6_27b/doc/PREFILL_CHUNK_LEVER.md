# The prefill chunk size is the first thing to test against a 3.8 s TTFT

Operator analysis, 2026-08-17. Default behaviour is unchanged; this adds a
measurement hook and records the arithmetic that motivates it.

## Why prefill is the target

Warm serving TTFT is **~3,784 ms for a 128-token prompt** (stage 10, and warm
because the TT plugin compiles prefill during startup — see the correction in
`TTFT_MEASUREMENT_DEFECT.md`). A decode step costs ~56 ms. So prefilling 128
tokens costs about what **68 decode steps** cost: ~30 ms per prompt token against
56 ms per generated token.

That ratio is the anomaly. Prefill processes the whole prompt in batched passes
and should be *much* cheaper per token than decode, which processes one token at a
time. Here it is within a factor of two.

## Where the sequential depth goes

48 of this model's 64 layers are gated-delta (linear attention). Their prefill
path chunks the sequence at `LINEAR_PREFILL_CHUNK_SIZE = 32` and scans each chunk
with a Hillis-Steele affine scan. From the implementation's own docstring:

> "the recurrent update is `R' = A R + B` ... Affine transforms compose
> associatively, so a Hillis-Steele scan produces every token state in
> log2(chunk) batched matmuls instead of submitting one decode graph per token."

So the sequential scan depth for a sequence of length `S` is

```
steps(S, chunk) = ceil(S / chunk) * log2(chunk)
```

which is **decreasing in chunk**:

| chunk | steps at S=128 | steps at S=512 | host uploads at S=512 |
|---:|---:|---:|---:|
| **32 (current)** | **20** | **80** | **80** |
| 64 | 12 | 48 | 40 |
| 128 | 7 | 28 | 20 |
| 256 | 4 | 16 | 10 |

Chunking smaller does not buy parallelism — it *adds* sequential chunk
boundaries, because chunks must run in order to carry the recurrent state. The
scan inside a chunk is the parallel part; the chunk loop is the serial part. At
chunk 32 and S=512, the 48 gated-delta layers execute 80 serial scan steps each.

The host-upload term compounds it. `generator.prefill_forward` builds and uploads
**five** tensors per chunk — one sequence mask and four conv-state lane selectors:

```
for start in range(0, physical_len, LINEAR_PREFILL_CHUNK_SIZE):
    sequence_mask_tt.append(self._upload(mask, dtype=ttnn.bfloat16))
    for lane in range(4):
        chunk_selectors.append(self._upload(selector, dtype=ttnn.bfloat16))
```

so uploads scale as `5*ceil(S/chunk)` — 20 at S=128, but **5,120 at S=32,768** and
**40,960 at the advertised 262,144**. The comment at `generator.py:211`
acknowledges this `O(ceil(S/LINEAR_PREFILL_CHUNK_SIZE))` term without bounding its
cost.

## Why this is a one-constant experiment

Everything inside the chunk derives from the chunk's actual extent —
`sequence = hidden_states.shape[2]`, and the scan loop is `while distance <
sequence:` — so the scan adapts to any chunk size with no other change. The
constraints are:

- multiple of the 32-element tile;
- `model.py` ties the streaming prefill quantum to `lcm(page_size, chunk)`, so
  that quantum moves too;
- memory: the scan materialises `[groups, chunk, ...]` intermediates, so footprint
  grows **linearly** with chunk. This is the real cost and the reason 32 is not
  obviously wrong — it is the trade that has to be measured.

`functional_decoder.py` now reads `QWEN36_LINEAR_PREFILL_CHUNK_SIZE`, **defaulting
to 32 so shipped behaviour is byte-identical**, and rejects values that are not
multiples of 32.

## The measurement

`/tmp/qwen_chunk_sweep.sh` runs chunk ∈ {32, 64, 128, 256} at S ∈ {128, 512} from
one weight load per chunk, reporting cold and warm TTFT per length.

Reads:

- **Warm TTFT falls with chunk** → prefill is scan-depth and/or upload bound, and
  the fix is a constant plus a memory check. Best case at S=512 is a 5× reduction
  in serial scan steps.
- **Warm TTFT is flat** → prefill is bound by per-token matmul work or by the
  full-attention layers, the chunk is not the lever, and the next suspects are the
  16 full-attention layers' prefill path and the projection matmul configs. A flat
  result is a genuine outcome, not a failed experiment.
- **Larger chunks OOM** → records the memory ceiling, which is itself the answer
  to "why 32", and the intermediate values still bound the available gain.

## What is not claimed

No TTFT improvement is claimed yet. The arithmetic above is exact — it follows
from the scan's own documented structure — but arithmetic on step counts is not a
latency prediction: each step's cost depends on the tensor shapes, and larger
chunks make each step wider. Wider-and-fewer is usually better on this hardware,
which is why it is worth measuring, not why it is certain.
