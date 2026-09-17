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

2. **Gemma4 CP prefill is not chunk-width invariant, and that predates this work.** The same
   prompt through **one sliding layer** over an **empty** KV history gives **PCC 0.871** between
   chunk 8192 and chunk 32768. One layer cannot amplify anything. Until that is fixed, changing
   chunk width changes the model's output — which blocks per-request bucketing **and equally
   invalidates the already-published 8192 -> 32768 throughput recommendation**, since nothing
   had checked whether that change is numerically neutral.

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

## 3. Measured: the selector beats pinning either width

`test_variable_chunk_prefill_beats_either_fixed_width`, one model with both widths captured,
every prompt run at **both** widths so the comparison is measured rather than modelled.

| prompt | chunk 4096 | chunk 32768 | picked | modelled | picked vs best measured |
|---:|---:|---:|---:|---:|---:|
| 4,096 | **175.8 ms** | 928.6 ms | 4096 | 174.2 | 1.000x |
| 16,384 | **722.5 ms** | 930.0 ms | 4096 | 714.8 | 1.000x |
| 36,864 | **1693.1 ms** | 1984.9 ms | 4096 | 1675.7 | 1.000x |
| 262,144 | 17295.9 ms | **11439.2 ms** | 32768 | 10960.0 | 1.000x |

* **4 of 4 correct picks**, and the model predicts each measurement to **0.9-4.2%**.
* vs pinning **4096**: 19.89 s -> 14.03 s = **1.42x** on the workload.
* vs pinning **32768**: 15.28 s -> 14.03 s = 1.09x on the workload, but **1.68x per-request
  geometric mean**, range 1.00x-**5.28x**. The workload total understates it because the single
  256k prompt dominates the sum; per-request is what a short request actually experiences.

Cost of the second bucket: one extra captured trace (~2.4 s capture, trace region 600 MB for two)
and one extra chunk-major RoPE table (~67 MB/device, and **unused on the traced path** — see §6).
No extra weights and no extra KV cache.

---

## 4. The blocker: prefill is not chunk-width invariant

Found while building the correctness gate for the above. **It is not caused by variable chunking**
— it reproduces between any two fixed widths.

Same prompt, same tokens, two chunk widths, comparing the output hidden states:

| pair | row 1 PCC | worst row |
|---|---:|---:|
| 4096 vs 8192 | 0.794 | **-0.269** |
| 4096 vs 32768 | 0.840 | -0.177 |
| 8192 vs 32768 | 0.986 | 0.189 |
| **4096 vs itself, replayed last** | **1.0000** | **1.0000** |

The control is exact, so this is not nondeterminism and not cross-trace contamination.

### It is a first-order difference, not 60-layer amplification

The obvious benign explanation is that a different SDPA blocking gives slightly different
arithmetic that compounds over 60 layers. **It does not.** Truncating the same loaded model to N
layers (chunk 8192 vs 32768, prompt 8192, whole-tensor PCC):

| layers | 1 | 2 | 4 | 8 | 16 | 30 | 60 |
|---|---:|---:|---:|---:|---:|---:|---:|
| PCC | **0.871** | 0.922 | 0.924 | 0.965 | 0.969 | 0.939 | 0.576 |

**0.871 at one layer**, and the curve *improves* to 16 layers before collapsing. Amplification
would decay monotonically from ~1.0. Layer 0 is `sliding_attention`, and at chunk 0 the KV history
is empty — so a single sliding-window ring attention over a fresh cache already depends on the
chunk width.

### What this costs

* **Per-request bucketing cannot ship** until width is neutral: two requests with the same prompt
  would get different answers depending on how busy the server was.
* **The published 8192 -> 32768 recommendation (+24.3% tok/s) silently changes the output.** That
  is the same defect, and it was never checked: the only surviving CP prefill test asserts
  finiteness and non-degeneracy, never numerics.
* **Which width is correct is unknown.** There is no absolute reference on this path — the CPU
  reference was deleted, and the model here has no final norm or LM head, so there are no logits.

### Next step, and it is cheap

The op-level suite `tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py` **does** check against
torch. Run its sliding cases at the two Gemma geometries — `chunk_size_local` 1024 (= chunk 8192
at CP8) and 4096 (= chunk 32768 at CP8) — on the SP8 linear-fabric config the model actually uses.
Whichever fails against torch is the wrong one. If both pass at op level, the difference is in how
the model drives the op (program config, halo sizing, `logical_n`) rather than in the op.

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
