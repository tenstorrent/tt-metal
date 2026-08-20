# Mistral Small 4 119B — prefill performance log

Running record of measured numbers, so each new thing we try can be compared against the last.
**Append, don't overwrite.** Every results block names the machine and the **commit it was measured
at** — that is not the commit that adds the block, so state it explicitly.

Model: `mistralai/Mistral-Small-4-119B-2603`, 36 layers, MLA (`kv_lora_rank=256`,
`qk_rope_head_dim=64` → 320-wide KVPE), MoE 128 routed / top-4 / 1 shared.

Machines: **12 kW** — the box prefill will actually run on, so this is the column that decides things.
**8 kW** (`bh-glx-110-a10u08`) — the dev box; useful as a second data point, not as a target.

---

## 1. The decision table — 12 kW, single-rank vs PP=4

| window | single-rank (production) | PP=4 × (8,1) | ratio | verdict |
|---|---|---|---|---|
| 5,120 | **36,312** | 34,004 | 0.94x | single-rank wins |
| 25,600 | 32,821 | **41,664** | 1.27x | PP wins — but see caveat below |
| 102,400 | **23,546** | 23,437 steady / 20,592 total | 1.00x / 0.87x | single-rank wins |
| 261,120 | 15,022 | **16,377** steady / 15,697 total | 1.09x / 1.05x | PP wins, modestly |

tok/s. Single-rank measured at `c34e372b47d`; PP=4 at `9f44e5fe988` (identical content to
`b5c4de3551a` on `kmabee/mistral4-prefill-full`), both 2026-08-20.

**Two caveats shrink PP's case further, and both cut the same way:**

1. **`PP_HANDOFF=none` on every PP row — no activations actually cross a stage boundary.** These are
   upper bounds. The only handoff we have measured, `host`, costs 42 MB/hop and ~1121 ms/iteration,
   i.e. catastrophic. A real device-to-device handoff cost has to come off margins that are already
   ≤9% at three of four windows.
2. **The 5,120 and 25,600 rows are not chunked.** They use `concurrent_throughput` with
   `PP_WINDOW` = the whole window, so each stage does one single-shot forward. Production always
   chunks. Against the single-shot single-rank peak (~33,552 near a 25k window) the 25,600 win is
   ~1.24x, not 1.27x-over-chunked. Only the 102,400 and 261,120 rows (`PP_WINDOW=5120`) are
   production-shaped — and those read **1.00x and 1.09x steady, before handoff cost**.

---

## 2. 8 kW vs 12 kW — and why PP's advantage keeps shrinking

### Single-rank scales with power

| window | 8 kW | 12 kW | 12/8 |
|---|---|---|---|
| 5,120 | 32,611 | 36,312 | 1.11x |
| 25,600 | 27,090 | 32,821 | 1.21x |
| 102,400 | 19,810 | 23,546 | 1.19x |
| 261,120 | 10,888 | 15,022 | **1.38x** |

### PP=4 mostly doesn't

| window | 8 kW | 12 kW | 12/8 |
|---|---|---|---|
| 5,120 | 34,059 | 34,004 | **1.00x** |
| 25,600 | 41,376 | 41,664 | **1.01x** |
| 102,400 steady | 23,234 | 23,437 | **1.01x** |
| 102,400 total | 19,520 | 20,592 | 1.05x |
| 261,120 steady | 12,434 | 16,377 | 1.32x |
| 261,120 total | 12,346 | 15,697 | 1.27x |

**This is the most informative result in the doc.** At three of four windows PP=4 gains *nothing*
from 50% more board power (1.00x / 1.01x / 1.01x) while single-rank gains 11-21%. PP=4 × (8,1) is
therefore **not power-bound** at short and mid windows — it is limited by something extra power does
not relieve (per-stage serialisation, host dispatch, or the 8-chip SP ring). Single-rank at TP=4 is
collective- and bandwidth-hungry and absorbs the headroom.

That single fact explains the whole "PP's advantage shrinks as power goes up" pattern, and it means
the gap will keep closing on any future higher-power part. Only at 261,120 does PP scale with power
(1.32x steady) — the one window where it still wins.

### Head-to-head ratios side by side

| window | 8 kW PP/single | 12 kW PP/single |
|---|---|---|
| 5,120 | 1.04x | **0.94x** |
| 25,600 | 1.53x | 1.27x |
| 102,400 (steady) | 1.17x | **1.00x** |
| 261,120 (steady) | 1.14x | 1.09x |

Every window got worse for PP. Two crossed from win to loss/wash.

---

## 3. Recommendations — next steps

1. **Keep single-rank SP=8 × TP=4 chunked as the production config.** On current 12 kW evidence PP=4
   wins one of four windows outright, and the two production-shaped windows read 1.00x and 1.09x
   *before* any handoff cost. That does not justify productising a second parallelism scheme.
2. **Before spending more on PP, measure a real device-to-device handoff.** It is the cheapest
   decisive experiment we have: if it costs more than ~5%, it erases three of four windows and the
   question is closed. `handoff=none` numbers cannot settle anything.
3. **Explain PP's flat power scaling** (1.00x, 8→12 kW). Highest-information measurement after #2.
   If the limiter is host dispatch or stage serialisation, PP's ceiling moves and it is worth
   revisiting; if it is the 8-chip SP ring, PP is structurally done at these windows.
4. **Attack MoE Dispatch + Combine — 33.8% of device time**, the largest single category, larger than
   matmul or attention. It is SP-axis token routing, so `PP=4 × (8,1)` does not touch it at all. This
   is the biggest lever available to the *production* config, which is the one that ships.
5. **Close the ~13.7% host/dispatch overhead that survives trace replay** (100.1 ms device busy vs
   116.0 ms wall clock).
6. **Re-run the 25,600 row chunked** (`PP_CONTEXT=25600 PP_WINDOW=5120`) so the one window PP clearly
   wins is measured on the production path rather than single-shot.
7. **Re-measure accuracy (§5) and the op breakdown (§6) on the 12 kW box** — both have only ever been
   run on 8 kW.
8. **Fix the `analyze_ops_perf.py` CCL regex** before anyone trusts its numbers (see §6).

---

## 4. Full results and how to reproduce

### 4.1 Single-rank — SP=8 × TP=4, chunked (the production path)

`prefill_producer.py:534` computes `target_chunks = ceil(real_len / CHUNK_SIZE)` unconditionally with
`CHUNK_SIZE=5120`; there is no single-shot branch in the runner. Numbers measured any other way are
microbenchmarks. Measured at `c34e372b47d`; 12 kW iter0/iter1 agreed within 0.8% on every row.

| window | chunks | 8 kW | 12 kW | 8 kW wall clock |
|---|---|---|---|---|
| 5,120 | 1 | 32,611 | 36,312 | 2 min 22 s |
| 25,600 | 5 | 27,090 | 32,821 | 6 min 22 s |
| 102,400 | 20 | 19,810 | 23,546 | 5 min 10 s |
| 261,120 | 51 | 10,888 | 15,022 | 3 min 37 s |

Most of the wall clock is model build + weight load, not measurement — the 261,120 forward is 24 s.

```bash
T=models/demos/deepseek_v3_d_p/tests/test_prefill_transformer_chunked.py
ID="$T::test_mistral4_prefill_transformer_chunked_no_pcc[blackhole-mistral4-mesh-8x4-L36-chunks51-two_iters-traced]"
PREFILL_NOPCC_SEQ_CACHE=261120 pytest "$ID" -q -rs -s
```

Swap `chunks51`/`261120` for `chunks01`/`5120`, `chunks05`/`25600`, `chunks20`/`102400`.
`-s` is required or the timing table is swallowed. `traced` is required — the `notrace` row reports
~0.67 s/chunk flat and is measuring host dispatch, not the device.

### 4.2 PP=4 × (8,1) — four 9-layer stages, TP=1, concurrent

Four stages of 9 layers, each SP/CP=8 × TP=1 on 8 chips, tiling the 32-chip galaxy. Motivation: every
collective in this model's dense path is on the TP axis, so TP=1 deletes all of it.

Measured at `9f44e5fe988`, `PP_HANDOFF=none`, traced. The 8 kW figures are within 1.4% of those
measured on the original pre-rebase branch, so the PP integration is faithful.

| window | 8 kW | 12 kW | 12 kW verbatim |
|---|---|---|---|
| 5,120 | 34,059 | 34,004 min / 33,282 med | `min_ms=150.6 med_ms=153.8` |
| 25,600 | 41,376 | 41,664 min / 39,639 med | `min_ms=614.4 med_ms=645.8` |
| 102,400 | 23,234 steady / 19,520 total | 23,437 steady / 20,592 total | `total_s=4.97 med_ms=218.5` |
| 261,120 | 12,434 steady / 12,346 total | 16,377 steady / 15,697 total | `total_s=16.63 med_ms=312.6` |

**Read `total` vs `steady` carefully.** `total` is the whole request including pipeline fill/drain;
`steady` is the steady-state median, i.e. the server case with back-to-back requests. At 102,400 the
total is negative on both boxes — fill/drain eats the entire benefit for a single request.

Wall clock: 5,120 → 1 min 49 s (8 kW) / 4 min 26 s (12 kW); 25,600 → 2 min 59 s / 3 min 12 s;
102,400 → 1 min 59 s / 1 min 49 s; 261,120 → 1 min 57 s / 2 min 13 s.

```bash
export TT_MISTRAL4_PREFILL_TTNN_CACHE=/data/kmabee/mistral4_caches/ttnn_cache_pp   # NOT the 8x4 cache
export PP_HANDOFF=none
T=models/demos/deepseek_v3_d_p/tests/test_prefill_pipeline_concurrent.py

PP_WINDOW=5120  PP_ITERS=12 pytest "$T::test_mistral4_pp4_concurrent_throughput[blackhole-mistral4-mesh-8x4-8x1]" -q -s
PP_WINDOW=25600 PP_ITERS=12 pytest "$T::test_mistral4_pp4_concurrent_throughput[blackhole-mistral4-mesh-8x4-8x1]" -q -s
PP_CONTEXT=102400 PP_WINDOW=5120 pytest "$T::test_mistral4_pp4_concurrent_longctx[blackhole-mistral4-mesh-8x4-8x1]" -q -s
PP_CONTEXT=261120 PP_WINDOW=5120 pytest "$T::test_mistral4_pp4_concurrent_longctx[blackhole-mistral4-mesh-8x4-8x1]" -q -s

# correctness (uses the 8x4 cache, not the pp one)
TT_MISTRAL4_PREFILL_TTNN_CACHE=/data/kmabee/mistral4_caches/ttnn_cache_8x4 \
  pytest models/demos/deepseek_v3_d_p/tests/test_prefill_pipeline_stages.py -q -s
```

The kernel cache (`~/.cache/tt-metal-cache`, ~27 GB) is local per machine and **survives
`tt-smi -r`** — a run straight after a reset still completed in 1:49. A first run on a machine that
has never built these kernels adds a few minutes, no more. If a run takes dramatically longer,
suspect contention, not compilation.

---

## 5. Accuracy

Measured on 8 kW only.

| check | value | threshold |
|---|---|---|
| block, `pcc-prompt_5k` — output | 0.990894 | 0.95 |
| block — KVPE KV part | 0.999896 | 0.999 |
| block — KVPE PE part | 0.999899 | 0.999 |
| chunked per-layer, `chunks03` | passes | raw ≥ 0.88 **or** nPCC ≥ 0.90 |
| PP stages vs single-rank (8 kW) | same token (2), p=0.7147 | exact match |
| PP stages vs single-rank (12 kW) | same token (2), p=0.7147 | exact match |

Runtimes: block PCC ~1 min 25 s, chunked PCC ~4 min 17 s, PP stages 2 min 41 s (8 kW) / 2 min 1 s
(12 kW).

**Late layers are expected to look bad on raw PCC.** Layers 32-35 read raw 0.28 / 0.32 / 0.35 / 0.65
because a few channels carry massive activations (attention sink) and raw PCC is dominated by them.
Their nPCC is 0.9657 / 0.9673 / 0.9692 / 0.9913. Judge depth on nPCC.

---

## 6. Where the time goes (device op breakdown, one PP stage, window 5,120, 8 kW)

Device busy 100.1 ms/forward vs 116.0 ms traced wall clock → ~13.7% host/dispatch overhead survives
even under trace replay.

| op | % device time |
|---|---|
| MatmulDeviceOperation | 21.6 |
| MoE Combine | 20.0 |
| UnifiedRoutedExpertFfn | 15.4 |
| RingJointSDPA (attention) | 14.7 |
| MoE Dispatch | 13.8 |
| AllGather (true CCL) | 4.5 |
| LayerNorm pre/post-gather | 3.4 |

**MoE Dispatch + Combine together are 33.8%** — see recommendation #4.

Caveat: `analyze_ops_perf.py` reports CCL as 8.0% rather than 4.5%, because its regex string-matches
op *names* and `LayerNormPostAllGatherDeviceOperation` contains "AllGather". Fix before trusting it.

---

## 7. Measured and rejected — do not re-run

### PP=4 × (4,2)

| window | (4,2) | (8,1) | (4,2) vs single-rank |
|---|---|---|---|
| 5,120 | 24,416 | 34,059 | **0.93x** |
| 25,600 | 17,483 | 41,376 | **0.52x** |

`(4,2)` is not merely worse than `(8,1)` — it is worse than not using PP at all. It keeps 2-way TP
collectives *and* halves the sequence split (SP 8 → 4), so each rank carries twice the tokens through
a shorter ring.

### Others

- **Single-shot throughput.** The runner always chunks; single-shot dies on L1 at 102,400
  (`circular buffers grow to 1721216 B beyond max L1 size of 1572864 B`) and peaks at ~33.5k around a
  25k window. Interesting as a curve, irrelevant as a production number.
- **Eager (untraced).** ~0.67 s/chunk flat — measuring host dispatch.
- **`PP_HANDOFF=host`.** 42 MB/hop, ~1121 ms/iteration. Shows what a naive host hand-off costs; not a
  candidate. A *device* hand-off is still unmeasured and is recommendation #2.

---

## 8. Environment requirements

```bash
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
export LD_LIBRARY_PATH=$PWD/build_Release/lib:$LD_LIBRARY_PATH   # or import ttnn fails on _ttnncpp.so
export MISTRAL4_HF_MODEL=/data/kmabee/models/Mistral-Small-4-119B-2603
export TT_MISTRAL4_PREFILL_TTNN_CACHE=/data/kmabee/mistral4_caches/ttnn_cache_8x4  # or ttnn_cache_pp for PP
export TT_MISTRAL4_PREFILL_HOST_REF_CACHE=/data/kmabee/mistral4_caches/ref_cache
export PREFILL_TRACE_DIR=/data/kmabee/mistral4_golden_traces/mistral4_15360_36L_fp32rope  # PCC rows only
```

Omitting `MISTRAL4_HF_MODEL` makes the test try to download the model. Omitting
`TT_MISTRAL4_PREFILL_TTNN_CACHE` rebuilds a 65 GB weight cache.

`/data/kmabee` is NFS **shared between the 8 kW and 12 kW boxes** — model weights, ttnn caches and
this checkout are the same files on both. `~/.cache/tt-metal-cache` is *not*; it is local per machine.

A failure at *fixture setup*, before any model code, is usually contention. Two signatures seen:

- `Sysmem mapped at unexpected NOC address` — left by a `kill -9` on a process holding 32 devices.
  One run spent **34 min** retrying before reporting it.
- `TopologyMapper auto-discovery: Downgrading to mesh shape 4x4 (16 total nodes) for 32 physical
  chips` → `Requested more devices (32) than available (16)`. Seen on 12 kW right after another
  user's job exited.

Both are fixed by `tt-smi -r`, and both are worth reaching for immediately rather than debugging.
Check `/dev/shm` for `tt_h2d_<pid>_*` whose pid is still alive — `fuser` will not show another user's
process.
