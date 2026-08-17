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

---

## Addendum: the tradeoff is op-count against FLOPs, so expect an optimum

Added the same day, after counting the scan body. The section above says
sequential steps decrease with chunk size, which is true, but it is only half the
trade and "bigger is better" does not follow. The full picture:

### Exact per-step cost

The scan loop body is exactly **five device ops**, two of them matmuls:

```
while distance < sequence:
    previous_transform = ttnn.concat([identity[:, :distance], transform[:, :-distance]], dim=1)
    previous_bias      = ttnn.concat([zero[:, :distance],     bias[:, :-distance]],      dim=1)
    old_transform = transform
    transform = ttnn.matmul(old_transform, previous_transform)
    bias      = ttnn.add(ttnn.matmul(old_transform, previous_bias), bias)
    distance *= 2
```

Each step's tensors are shaped `[groups, chunk, d, d]`, so **per-step work is
proportional to the chunk**, while the number of steps is `log2(chunk)`.

### Two quantities move in opposite directions

For a sequence `S` with chunk `C`, per gated-delta layer:

| quantity | formula | C=32 | C=128 | C=256 |
|---|---|---:|---:|---:|
| scan **dispatches** | `5 * (S/C) * log2(C)` | 100 | 35 | 20 |
| scan **FLOPs** | `∝ S * log2(C)` | 5·S | 7·S | 8·S |

(at S=128; dispatch counts exclude the ~35 setup/teardown ops per chunk.)

So raising the chunk **reduces dispatches** by up to 5× and **increases total
arithmetic** by up to 1.6×. This is inherent to a Hillis-Steele scan, which is
work-inefficient by design: `O(C log C)` work for `O(log C)` depth, against
`O(C)` work and `O(C)` depth for a sequential scan.

### Which term dominates today

Measured, at S=128: ~240 device ops per linear layer × 48 layers ≈ **11,500
dispatches**, against a warm TTFT of ~3,784 ms — about **0.3 ms per op**, which is
a dispatch-scale cost, not a compute-scale one for tensors this small.

Independent check from the other direction: prefill of a 128-token prompt is
roughly `2 * 27e9 * 128 ≈ 6.9 TFLOP`, delivered in 3.784 s, i.e. **~1.8 TFLOP/s
aggregate across four Blackhole devices**. That is one to two orders of magnitude
below what this mesh can sustain, so prefill today is nowhere near arithmetic
limited.

**Therefore the dispatch term should dominate and larger chunks should win — up to
the point where the 1.6× extra arithmetic and the linear growth in intermediate
footprint catch up.** That is a prediction with a predicted *shape*: warm TTFT
should fall from 32 to 64 to 128, then flatten or reverse. If it falls
monotonically through 256, the sweep should be extended.

### A further lever, not attempted here

If arithmetic does become the limit, the work-inefficiency is itself removable: a
Blelloch (work-efficient) scan does `O(C)` work with `2*log2(C)` depth, trading a
doubling of depth for the removal of the `log2(C)` work factor. That is a real
kernel-level change rather than a constant, so it is only worth considering if the
chunk sweep shows the FLOP term binding.

---

## Two secondary dispatch levers in the same scan, chunk-size independent

Recorded while counting the scan body. Neither is attempted; both reduce
dispatches without touching the chunk, so they compose with whatever the sweep
concludes.

### 1. The two shift-concats are 40% of the scan's ops

Of the five ops per scan step, two are pure data movement:

```
previous_transform = ttnn.concat([identity[:, :distance], transform[:, :-distance]], dim=1)
previous_bias      = ttnn.concat([zero[:, :distance],     bias[:, :-distance]],      dim=1)
```

Both implement "shift right by `distance`, fill with the scan identity". Each
allocates a fresh full-size tensor per step. If the same effect can be had from a
pre-padded buffer plus a slice — allocate `transform` and `bias` with `chunk`
leading padding already holding identity/zero, then read the shifted view — the
step drops from **5 ops to 3**, a 40% dispatch reduction on the term that the
measurements above suggest dominates.

Caveat that has to be checked before believing it: whether a slice at a non-tile
offset produces a view or forces a copy on this stack. If it forces a copy, this
buys nothing and the concat is already the cheap way to do it.

### 2. `identity` and `zero` are rebuilt per chunk, per layer

```
identity = ttnn.repeat(self.weights["linear_identity"], ttnn.Shape([groups, sequence, 1, 1]))
zero = ttnn.multiply(identity, 0.0)
```

These are **constants** for a given `(groups, chunk)` shape, but they are
materialised inside `_linear_attention_prefill_chunk`, which runs once per chunk
per layer. At S=128 with chunk 32 that is 4 chunks × 48 gated-delta layers = **192
rebuilds of each**, or 384 large tensor allocate/deallocate pairs per prefill, for
tensors whose contents never change.

Hoisting them to a per-shape cache on the model is a small, contained change. The
direct op saving is modest (two ops per chunk), but the allocator churn it removes
is not obviously modest, and `zero` in particular is an entire tensor of zeros
produced by a multiply.

Both of these are worth measuring only after the chunk sweep, because the chunk
result determines how many scan steps exist to optimise in the first place.

---

## Root cause: the gated-delta prefill path was never optimised

Added 2026-08-17 from the committed functional-decoder profile plus the optimized
decoder's own source. This supersedes the framing above — the chunk size matters,
but it is a symptom of something larger.

### The optimized decoder delegates its linear prefill to the correctness-first code

`optimized_decoder.py:_linear_attention_prefill_chunk` does not implement an
optimised scan. It expands the recurrent state to FP32, calls the functional
implementation, and compresses the result back:

```python
state_dtype = self.policy.linear_recurrent_state_dtype
if state_dtype == ttnn.float32:
    return FunctionalDecoder._linear_attention_prefill_chunk(self, hidden_states)
...
output = FunctionalDecoder._linear_attention_prefill_chunk(self, hidden_states)
```

Its own docstring is explicit: *"run the proven implementation **unchanged**"*.

Decode was optimised thoroughly — packed projections, fused CCL, traced replay,
DRAM-sharded matmuls. MLP and generic projections got optimised prefill paths
(`_prefill_linear`, `_mlp_prefill`). But the **gated-delta token mixer's prefill —
48 of 64 layers — is still the stage-01 correctness-first implementation** with a
dtype wrapper around it.

### What that implementation costs, measured

From `doc/functional_decoder/tracy/linear_attention_prefill_s5/prefill_perf_report.csv`,
**one** linear-attention layer, **one** chunk:

| op family | n | % of ops | device µs | gap µs | % of wall |
|---|---:|---:|---:|---:|---:|
| BinaryNg | 78 | 19.5% | 454 | **992** | 12.5% |
| Matmul | 43 | 10.8% | 6264 | 188 | 55.8% |
| Unary | 36 | 9.0% | 111 | **550** | 5.7% |
| ReshapeView | 35 | 8.8% | 281 | 154 | 3.8% |
| UntilizeWithUnpadding | 34 | 8.5% | 482 | 114 | 5.2% |
| Transpose | 30 | 7.5% | 92 | 133 | 1.9% |
| Permute | 30 | 7.5% | 334 | 45 | 3.3% |
| Slice | 25 | 6.3% | 291 | 17 | 2.7% |
| TilizeWithValPadding | 25 | 6.3% | 309 | 136 | 3.8% |
| Concat, FillPad, Reduce, Copy, LayerNorm | 63 | 15.8% | 483 | 125 | 5.3% |
| **total** | **399** | | **9101** | **2452** | |

Three things stand out:

1. **43 of 399 ops (11%) are matmuls.** The other **356 (89%) are layout and
   elementwise glue** — 59 tile/untile conversions, 120 pure data-movement ops
   (reshape/transpose/permute/slice), 114 elementwise ops.
2. **For the small ops, dispatch costs more than the work.** BinaryNg: 992 µs gap
   against 454 µs device. Unary: 550 µs gap against 111 µs device, a **5× ratio**.
   Overall gap is 21% of wall time in this layer.
3. **All 399 are per chunk.** `_linear_attention_prefill` slices
   `hidden_states[:, :, start:stop, :]` and calls the chunk function, so the
   projections, norms, layout conversions and elementwise work all repeat for every
   32-token slice — not just the scan.

### Why this explains ~3.8 s, and why the chunk is a big lever

At S=128 with chunk 32 there are 4 chunks per linear layer:

```
48 linear layers x 4 chunks x ~399 ops  ~=  76,600 ops
16 full-attention layers x ~73 ops      ~=   1,170 ops
                                        ~=  78,000 device ops
```

At a warm TTFT of 3,784 ms that is **~48 µs per op**, which is dispatch-scale.
The 48 gated-delta layers account for ~98% of the op count.

So raising the chunk does far more than shorten the scan: it divides *the entire
per-chunk op count* by `S/C`. Going 32 → 128 at S=128 collapses 4 chunks into 1 —
roughly a **4× reduction in device ops** — while making each tensor 4× wider, which
is the direction this hardware prefers. The scan's `log2(C)` work growth applies
only to the 43 matmuls, and only the handful that are scan matmuls.

### Caveats, because this is extrapolation from one small profile

- The profile is **S=5, one chunk, functional decoder**. S=5 pads to a single
  32-tile, so a chunk-32 slice should cost similarly, but that is an assumption.
- The optimized decoder wraps rather than replaces this path, so its op count
  should be close, plus two typecasts per chunk. Not confirmed by a profile.
- The 78,000-op figure is arithmetic on those assumptions, not a measurement. The
  confirming measurement is an optimized-path prefill profile at S=128, which does
  not exist on this branch.

### What this reprioritises

The chunk sweep already queued now tests the cheap version of this (one constant,
no code change). But the larger finding is that **prefill optimisation was never
done for the dominant layer type**, and the 89%-glue op mix says the ceiling is
much higher than a constant can reach: fusing the elementwise chains, removing the
tile/untile round trips at the conv and recurrent boundaries, and hoisting the
per-chunk constants are all still available.

That also reframes the stage-08 precision work. It ranked candidates on a metric
that moves the 11% of ops that are matmuls, while 89% of the ops — and all of the
dispatch overhead — were untouched by any precision choice.
