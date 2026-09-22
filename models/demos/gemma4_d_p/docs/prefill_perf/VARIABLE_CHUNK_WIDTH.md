# Gemma4 prefill: per-request chunk width — investigated, measured, not pursued

> **The PoC code was dropped from this branch on 2026-09-21.**
> `tt/chunk_buckets.py`, `tt/variable_chunk_prefill.py`, `tests/test_variable_chunk_prefill.py`
> and `tests/unit/test_chunk_buckets.py` no longer exist here, and the `model.py` / `common.py`
> hooks went with them. The conclusion below is why: it is a no-ship, and the multi-width build
> carried a live correctness bug (PCC 0.869 at a width it shares with a single-width build)
> that produced three retracted perf claims before it was identified. This writeup is kept as
> the record; the code is recoverable from the pre-rewrite history if anyone needs it.
> **File paths referenced further down therefore point at files that are no longer present.**

Measured on a BH Galaxy, mesh 8x4 (CP8/TP4). Gemma4-31B: 60 layers = 50 sliding + 10 full
attention, sliding window 1024.

Follow-on to [`../Gemma4PrefillChunkSize/`](../Gemma4PrefillChunkSize/README.md), which measured
that no single prefill chunk width is good at both ends of the ISL range. This report asked whether
choosing the width *per request* is worth building. **The answer is no**, and the rest of this
document is why.

---

## TL;DR — why this is not being pursued

**It buys 1.21x, and almost everything else it appears to buy is available for free.**

Measured, three buckets {4096, 8192, 32768}, every prompt run at every width, against the shipping
fixed 8192: **1.19x** on the workload, **1.00x–1.39x** per request. But the short-prompt half of
that — 1.39x at 4,096 tokens — comes entirely from avoiding padding, and is **fully available by
deploying a single fixed width of 4096**, which the prior report already identified as the best
single setting. Once you subtract what a one-line config change gives you, the genuine incremental
value of per-request width is **1.21x at 256k context, and nothing else.**

Against that 1.21x:

| | |
|---|---|
| **It is not correct today** | A model built with several widths computes a different answer at a width it shares with a single-width build: chunk 8192 at **PCC 0.869** (`max_abs_diff` 29.9), one layer, identical KV geometry. That is a bug in this implementation (§4), unfixed |
| **Two silent-corruption modes** | The ring KV cache is block-cyclic with period `C`. Changing width mid-request, or a decode worker reading with the wrong period, corrupts the prefix and **raises nothing** |
| **Scheduling gets worse, invisibly** | The quantum becomes non-uniform: 176 ms for a 4096-wide chunk vs **928 ms** for a 32768-wide one. Under continuous batching a short request behind a wide chunk eats up to ~930 ms of head-of-line blocking. **Every number in this report is single-request on an exclusive mesh and cannot see this** |
| **It leaks into disaggregation** | Width becomes per-slot state that must cross the prefill/decode process boundary in slot metadata |
| **The policy is not a tiering** | With three buckets the optimal width **oscillates 11 times** over [4k, 256k], flipping between 8192 and 32768 at every padding remainder. Two users with similar prompt lengths get different widths and different TTFT (§3) |
| **Memory per bucket** | One captured 60-layer trace per width (900 MB of trace region for three) plus a chunk-major RoPE table per width (~67 MB/device, and never read on the traced path) |

**Recommendation: deploy a single fixed width chosen from the ISL distribution** — 4096 if short
prompts dominate, 8192 as the balanced default, 16384/32768 if long contexts dominate. Revisit
per-request width only if the ISL distribution is strongly bimodal *and* 1.21x at long context is
worth a per-slot invariant with silent failure modes.

### What is worth keeping from this

* The measured cost model `T(P,C) = N·a(C) + slope(C)·N(N-1)/2` predicted four newly measured
  chunk-8192 points to **0.7–1.3%**. It is a reliable way to choose a fixed width for a given ISL
  distribution without running the hardware (§1, §3).
* **Padding is bit-exact** (PCC 1.0): a prompt served in a chunk wider than it fills gets exactly
  the answer a fitting chunk would. Useful independently — it means chunk-size changes never need
  to worry about the padded tail (§5).
* **`ring_joint` sliding SDPA is width-neutral** against torch at every width on both SP4 and SP8,
  including geometries with no in-tree coverage. Worth landing those cases as tests (§4).
* Two methodological lessons that cost real time here: **a bucket set can only be judged against
  widths it does not contain** (the first version of this report reported 1.42x against a pair's own
  two members, while that pair actually loses 0.69x to shipping 8192 mid-range), and **two
  configurations of one model must be compared in separate processes** — comparing them inside one
  build is what produced three retracted claims.

**Retracted (2026-09-17):** an earlier version claimed Gemma4 prefill is inherently not
chunk-width invariant, that this predated the work, and that it invalidated the published
8192 → 32768 throughput recommendation. **All three are withdrawn.** They rested on cross-width
comparisons made inside a single multi-width build — inside the bug in §4. `ring_joint` matching
torch at every width was the signal that the op had been correct all along.

---

## 1. Why per-request width, and why it is *not* "small chunks for small prompts"

`a(C)`, the device time for one chunk, is **sublinear** in `C` up to 16384 — one 8192-token chunk
costs 242.7 ms where four 2048-token chunks cost 534 ms for the same tokens. A wider chunk is
always better *for tokens that fill it*. The entire case for a narrow bucket is the `ceil`:

> A 4096-token prompt in a 32768-wide chunk pays for 32768 tokens of attention and MLP.

So the win is **padding waste**, not a preference for small chunks. Measured: 175.8 ms in a 4096
chunk against 928.6 ms in a 32768 chunk, for the same 4096 tokens — **5.28x**.

Cost model (fitted in the prior report, reproduced here to <5%):

    T(P, C) = N * a(C) + slope(C) * N*(N-1)/2,   N = ceil(P / C)

`a(C)` turns *superlinear* at 32768 (intra-chunk causal self-attention is O(C^2)), which is what
bounds the useful width and is why the wide bucket is 32768 rather than the context length.

## 2. The admission policy is a sawtooth, not a threshold

This surprised us and is the one design consequence worth carrying forward. With buckets
{4096, 32768}, `select_chunk_size` changes its mind **four** times over [4k, 256k]:

| prompt >= | picks |
|---:|---|
| 4,096 | 4096 |
| 24,576 | 32768 |
| **36,864** | **4096** |
| 45,056 | 32768 |

Padding waste **recurs at every wide-chunk boundary**: a prompt just past 32768 tokens pays for a
second, nearly-empty wide chunk. Measured at 36,864 tokens: **1693 ms** at chunk 4096 against
**1985 ms** at chunk 32768 — the narrow bucket wins again, 8k tokens *after* it first lost.

**Any policy written as `narrow if prompt < T else wide` is wrong in that band.** Pinned by
`test_selection_is_sawtooth_not_a_single_threshold`. It gets worse with more buckets — the
three-bucket policy oscillates **eleven** times (§3).

## 3. Measured, three buckets, against the shipping config

`test_variable_chunk_prefill_beats_either_fixed_width` with `GEMMA4_CHUNK_BUCKETS=4096,8192,32768`.
Every prompt run at **all three** widths, so nothing here is projected.

| prompt | chunk 4096 | **chunk 8192 (shipping)** | chunk 32768 | picked | **picked vs fixed 8192** |
|---:|---:|---:|---:|---:|---:|
| 4,096 | **175.8 ms** | 244.5 ms | 928.4 ms | 4096 | **1.39x** |
| 16,384 | 722.3 ms | **501.1 ms** | 930.0 ms | 8192 | 1.00x |
| 36,864 | 1692.6 ms | **1343.8 ms** | 1984.4 ms | 8192 | 1.00x |
| 262,144 | 17292.3 ms | 13890.0 ms | **11447.9 ms** | 32768 | **1.21x** |

* Selector picked the measured-cheapest width **4 of 4**.
* Workload total: fixed 8192 **15.98 s** -> variable **13.47 s** = **1.19x**; per-request geometric
  mean **1.14x**, range 1.00x-1.39x.
* vs fixed 4096: 1.48x total. vs fixed 32768: 1.14x total (1.95x geomean) — but nobody would deploy
  a fixed 32768, so that column flatters the result and should be ignored.

**A two-bucket {4096, 32768} set LOSES to shipping 8192 in the middle** — 0.69x at 16,384 and 0.79x
at 36,864 — because 8192 pays half the per-chunk floors of 4096 and wastes none of 32768's padding.
An earlier version of this report reported 1.42x/1.68x for that pair against *its own two members*
and called it a win. That was the wrong baseline: **a bucket set can only be judged against widths
it does not contain.** Three buckets are the minimum that never loses to shipping.

The cost model held up: it predicted the four newly measured chunk-8192 points to **0.7-1.3%**
(244.5 vs 242.7, 501.1 vs 497.4, 1343.8 vs 1333.3, 13890.0 vs 13710.0).

### The admission policy oscillates 11 times

With three buckets, `select_chunk_size` changes its answer **eleven** times over [4k, 256k]:

```
4096->4096  8192->8192  28672->32768  36864->8192  61440->32768  69632->8192
86016->32768  102400->8192  110592->32768  135168->8192  143360->32768
```

It flips between 8192 and 32768 at every 32768-boundary remainder, because a prompt just past a
multiple of 32768 pays for a second, nearly-empty wide chunk. Operationally this matters: the
bucket abstraction is **not** a clean "short -> narrow, long -> wide" tiering, two users with
similar prompt lengths get different widths and different TTFT, and the policy cannot be summarised
to a capacity planner in one sentence.

## 4. The bug: multi-width construction changes the answer

Same prompt, one decoder layer, `max_seq_len` 65536, and `ring_cache_capacity` = 65536 for **both**
builds (it is `max(max_seq_len, 2*max(C))`, which saturates at 65536 either way) — so the KV cache
geometry is byte-identical and cannot explain the difference. The only variable is the width tuple
the model was constructed with:

| build | widths | chunk-8192 layer-1 output std |
|---|---|---:|
| single-width (the shipping config) | `(8192,)` | **0.883347** |
| multi-width | `(4096, 8192, 32768)` | **0.928255** |

`bitwise_identical = False`, `max_abs_diff = 29.9`, **PCC = 0.869**.

A lead worth starting from — the single-width 8192 result matches the multi-build's **4096** result,
not its 8192 result:

```
single-build @8192  = 0.883347   ~=   multi-build @4096  = 0.883675
                                      multi-build @8192  = 0.928255
                                      multi-build @32768 = 0.928538
```

That pattern suggests per-width resources being bound by position-in-tuple, or by the model's
"default" width (`prefill_chunk_sizes[-1]`), rather than by the width actually requested. The two
candidates are `VariableChunkPrefill.capture()`, which captures N traces sequentially and may bake
overlapping intermediate addresses, and the per-width RoPE table construction. Not yet tested —
this is the next thing to do.

**What this invalidates from the earlier investigation.** Every cross-width number previously
reported here was measured inside one multi-width build, so all of it is suspect: the pair table
(4096-vs-8192 PCC 0.794, etc.), and the PCC-vs-depth curve (0.871 at 1 layer rising to 0.969 at 16,
collapsing to 0.576 at 60). The replay-determinism control (PCC 1.0000) and the padding-invariance
result (PCC 1.0) were both measured *within* one build and remain valid as far as they go, but they
do not establish cross-build correctness.

### `ring_joint` itself is width-neutral — measured, and still useful

`ring_joint` sliding SDPA was checked against **torch** at every width involved, on both ring
sizes, using the in-tree harnesses. All pass, and the PCCs are flat in the width:

| ring | geometry | vs torch |
|---|---|---:|
| SP4 | local slab 1024 (= chunk 8192 at CP8) | 0.99972 / 0.99973 |
| SP4 | local slab 2048 | 0.99972 / 0.99973 |
| SP4 | local slab 4096 (= chunk 32768 at CP8) | 0.99973 / 0.99973 |
| **SP8, linear fabric** (what the model runs) | **global chunk 8192** | **0.99961** |
| **SP8, linear fabric** | **global chunk 32768** | **0.99963** |

Note the SP4 cases alone would not have settled it — the model is CP8 — so the SP8 pair is the
one that matters. There is no in-tree coverage at slab 4096 / global 32768; those rows were run
for this report.

**The op is width-neutral.** With the retraction above this is no longer evidence of a model-level
defect — it is what rules the kernel out as a suspect for the construction bug in §4. If a deeper
bisect is ever wanted, it would go inside one `sliding_attention` layer: capture the
post-RoPE Q handed to `ring_joint` and the tensor it returns, at both widths, and find which is the
first to disagree. If Q already differs it is RoPE or the projections; if only the SDPA output
differs it is the arguments the model passes (program config, halo sizing, `logical_n`, the
persistent buffer). An eager (untraced) attempt at this segfaulted while cloning intermediates and
was not pursued — it needs a gentler capture than `ttnn.clone` on a tensor the layer later
deallocates.

---

## 5. What is validated

`test_variable_chunk_prefill_is_correct_within_a_width` — the three properties bucketing rests on:

| property | result |
|---|---|
| **Padding invariance.** 4096 real tokens in a 32768 chunk vs the same tokens in a chunk they fill, at that width | **PCC 1.0000, every row, bit-exact** |
| **Replay determinism.** Two traces sharing one KV cache, one metadata pair, one set of ring-gather buffers, replayed in either order | **PCC 1.0000** |
| **Negative control.** Width changed *mid-request* must corrupt the prefix | fires as required (PCC 0.78) |

Padding invariance is the important one and it is exact: admitting a short prompt to a wider
bucket is **wasteful, never wrong**. It also means the perf comparison above is honest — the padded
runs compute the same answer they would have computed anyway.

The negative control matters because without it the padding check would be vacuous. The ring KV
cache is block-cyclic with period `C` (local row `chunk*L + j` on rank `r` holds global token
`chunk*C + r*L + j`, `L = C/cp`) and `ring_joint` reconstructs a cached row's global position from
the **current** chunk's width. Changing width mid-request reads rank `r`'s row 0 as position
`r*C_new/cp` instead of `r*C_old/cp`. Nothing raises; the answer is just wrong. Hence: **width is
chosen at admission and pinned for the request's lifetime.** Different requests at different widths
are fine — each owns its cache slot.

One caveat on the negative control, and it is a consequence of §4: the mixed-width run differs
from the all-narrow reference for two reasons at once — the layout corruption the control is for,
and the fact that the prefix chunk ran at a different width at all. They cannot be separated while
width is not neutral. It establishes that changing width mid-request is unsafe; it does not
measure how unsafe.

`test_prefill_is_chunk_width_invariant` asserts the property we *want*, marked
`xfail(strict=True)`. It reproduces independently at **PCC 0.873** (chunk 4096 vs 32768, one
layer). An XPASS means the op was fixed; drop the xfail and raise the depth.

---

## 6. What was built

| file | what |
|---|---|
| `tt/chunk_buckets.py` | the cost model, `select_chunk_size`, `bucket_switch_points`, geometry validation. Host-only, no device |
| `tt/variable_chunk_prefill.py` | `VariableChunkPrefill`: one captured trace and one set of pinned staging tensors per width; `prefill()` pads, chunks and replays |
| `tt/model.py` | `prefill_chunk_size` accepts one width or several; RoPE tables become per-width; the width is recovered per call from the CP-local slab rather than read from a stored default |
| `tt/common.py` | validates every configured width before the 62 GB weight load |

The model change is small because `prefill_chunk_size` was already a constructor argument rather
than a module constant (unlike the Mistral4 sibling, where it is `PREFILL_CHUNK_TOKENS`). Four
touch points: normalize the widths, size the KV cache for the widest, build one RoPE table per
width, and key the RoPE lookup on the width the call is actually running at.

Already multi-width and needing no change: the TT_CCL ring-gather scratch buffers (cached by shape
signature, and both widths want the same shapes), `ChunkMetadata`/`set_ring_metadata`, and the
gather extent, which `ring_joint` derives on-device per dispatch from `kv_actual_isl`.

### Known waste

The chunk-major 4D RoPE tables are built per width (~67 MB/device each) and are **never read on
the traced path** — the traced path gathers RoPE by absolute position from the replicated 2D
tables, which are width-independent. They are built for the eager path's benefit. Worth making
lazy before this ships.

### Not done (out of PoC scope)

Phase 2 from the seed doc: the migration/decode boundary. `iter_cache_chunk_locations` already
takes `chunk_size` as a parameter, but the width has to reach the decode worker in the slot
metadata or it will read a block-cyclic cache with the wrong period.

---

## 7. Reproducing

```bash
source /data/kmabee/gemma4_runs/env.sh
cd $TT_METAL_HOME
D=models/demos/gemma4_d_p/tests/test_variable_chunk_prefill.py

# host-only: cost model, sawtooth, geometry validation (~2 s, no device)
./python_env/bin/python3 -m pytest models/demos/gemma4_d_p/tests/unit/test_chunk_buckets.py -q

# the perf claim: every prompt at all three widths (~5.5 min)
./python_env/bin/python3 -m pytest "$D" -k beats_either_fixed_width -sv

# padding invariance / replay determinism / negative control (~3 min)
./python_env/bin/python3 -m pytest "$D" -k correct_within_a_width -sv

# the single-vs-multi-width control -- TWO processes, ~2 min each. This is the one that
# retracted three claims; a one-process version cannot see the defect.
GEMMA4_WIDTHS=8192 GEMMA4_SAVE_PT=/tmp/ref8192.pt \
  ./python_env/bin/python3 -m pytest "$D" -k multi_width_build -sv
GEMMA4_WIDTHS=4096,8192,32768 GEMMA4_REFERENCE_PT=/tmp/ref8192.pt \
  ./python_env/bin/python3 -m pytest "$D" -k multi_width_build -sv -rxX   # xfail(strict) today

# other bucket sets
GEMMA4_CHUNK_BUCKETS=8192,32768 ./python_env/bin/python3 -m pytest "$D" -k beats -sv
```

Run logs from this session: `/data/kmabee/gemma4_runs/varchunk/`.
