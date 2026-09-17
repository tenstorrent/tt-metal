# Gemma4 prefill: per-request chunk width, and the correctness blocker it exposed

Measured on a BH Galaxy, mesh 8x4 (CP8/TP4), branch `kmabee/gemma4-swa-multihop-halo`.
Gemma4-31B: 60 layers = 50 sliding + 10 full attention, sliding window 1024.

Follow-on to [`../Gemma4PrefillChunkSize/`](../Gemma4PrefillChunkSize/README.md), which measured
that no single prefill chunk width is good at both ends of the ISL range.

---

## TL;DR

**Two things, and the second one gates the first.**

1. **Per-request chunk width works and is worth it.** One model, one KV cache, two captured
   traces; the width is chosen at admission from the prompt length. The selector picked the
   measured-cheapest width in **4 of 4** prompt lengths, for **1.42x** over pinning 4096 and a
   **1.68x per-request geometric mean (up to 5.28x)** over pinning 32768.

2. **The implementation has a bug: a model built with several widths computes the wrong answer
   even at a width it shares with a single-width build.** Same prompt, one layer, identical
   `ring_cache` geometry — chunk 8192 from a `(8192,)` build and from a `(4096, 8192, 32768)`
   build differ at **PCC 0.869**, `max_abs_diff` 29.9. So per-request bucketing as implemented here
   is **not functionally correct**, and the fault is mine, not the model's.

**Retracted (2026-09-17):** an earlier version of this report claimed Gemma4 prefill is inherently
not chunk-width invariant, that this predated the work, and that it invalidated the published
8192 -> 32768 throughput recommendation. **All three claims are withdrawn.** They rested on
cross-width comparisons made *inside a single multi-width build* — that is, inside the bug above.
The single-vs-multi-width control that would have caught it was identified as missing but not run
until later. `ring_joint` matching torch at every width (§4) is consistent with the op having been
correct all along.

The mechanism in (1) is sound and separately validated (padding is bit-exact, replay is exact,
the negative control fires). The blocker is underneath it, in the attention op.

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
`test_selection_is_sawtooth_not_a_single_threshold`.

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

# the perf claim: every prompt at both widths (~3 min)
./python_env/bin/python3 -m pytest "$D" -k beats_either_fixed_width -sv

# padding / determinism / negative control, and the xfail width-invariance gate (~5 min)
./python_env/bin/python3 -m pytest "$D" -k "correct_within_a_width or width_invariant" -sv -rxX

# other bucket pairs
GEMMA4_CHUNK_BUCKETS=8192,32768 ./python_env/bin/python3 -m pytest "$D" -k beats -sv
```

Run logs from this session: `/data/kmabee/gemma4_runs/varchunk/`.

---

## 8. Should this ship? No.

Recorded because the investigation reached a clear answer, against the thing it was built to do.

**What it is worth, measured:** 1.19x on the workload above against the shipping 8192, per-request
1.00x-1.39x. And the short-prompt half of that (1.39x at 4,096) is **fully available from a single
fixed width of 4096**, with no new machinery at all — the prior report already found 4096 the best
single setting. So the genuine *incremental* value of per-request width is only the long-context
term, **1.21x at 256k**.

**What it costs:**

| cost | detail |
|---|---|
| **Correctness** | broken today (§4), and the layout has two silent-corruption modes: width changed mid-request, and decode reading with the wrong block-cyclic period. Neither raises |
| **Scheduling** | the quantum becomes non-uniform — 176 ms for a 4096 chunk vs 928 ms for a 32768 one. Under continuous batching a short request behind a wide chunk eats up to ~930 ms of head-of-line blocking. **Every number in this report is single-request on an exclusive mesh and cannot see this** |
| **Disagg** | per-slot width must cross the process boundary in slot metadata; `iter_cache_chunk_locations` is already width-parametric, but a wrong period at the decode worker is silent corruption |
| **Memory** | one captured 60-layer trace per bucket (900 MB of trace region for three), plus one chunk-major RoPE table per width (~67 MB/device, and **never read on the traced path**) |
| **Predictability** | the 11-switch policy above |

**Recommendation.** Deploy a single fixed width chosen from the ISL distribution — 4096 if short
prompts dominate, 8192 as the balanced default, 16384/32768 if long contexts dominate. Revisit
per-request width only if the ISL distribution is strongly bimodal *and* the 1.21x long-context term
is worth a per-slot invariant with silent failure modes. The code here is kept as a measured answer
to "what would it buy", not as a candidate for merge.
