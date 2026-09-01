# Qwen3-TTS — Wormhole performance notes

Working notes on device-performance optimisation of the Qwen3-TTS decode path on Wormhole
(N150 = 1 chip, N300 = 2 chips at TP=2). Records what was changed, what it measured, what
was tried and rejected, and what is still on the table.

The [README](README.md) covers the Blackhole P150 path; this file is Wormhole-specific.

---

## 1. Where the time actually goes

The obvious assumption is that the 28-layer Talker (hidden 2048) dominates and the 5-layer
CodePredictor (hidden 1024) is a rounding error. It is the other way round.

One autoregressive audio frame runs the CodePredictor **15 times** — a 2-token prefill plus
13 residual decodes, one per codec group — against a single Talker decode step. That is
**75 CP layer evaluations per frame versus 28 Talker layers**.

Traced device time for one AR frame (single wormhole chip, Metal trace, `test_qwen3_tts_trace_perf.py`):

| window | device time | share |
|---|---|---|
| CodePredictor residual decodes (13x) | 42,044 us | 56 % |
| Talker decode | 30,497 us | 40 % |
| CodePredictor prefill (seq=2) | 2,987 us | 4 % |
| **full AR frame** | **75,528 us** | |

**Rule of thumb: compare per-*frame* cost (layer time x invocations per frame), never
per-layer time.** The CodePredictor is the first thing to optimise.

A second observation: in the CP decode layer, roughly 60 % of the time was *not* matmul. It
was layout churn and collectives — ops that move or reshape data rather than compute with it.

---

## 2. What was changed

### 2.1 CodePredictor N300 fast path — gated

`models/demos/qwen3_tts/tt/code_predictor.py`, gated on `mesh_utils.is_n300(device)`
(a 2-chip wormhole mesh; N150, T3K and Blackhole keep the generic path). Set
`QWEN3_TTS_CP_N300_OPT=0` to A/B at runtime.

The Talker already had sharded equivalents of all of this in `attention.py` /
`decoder_layer.py`; the CodePredictor had been written with generic ops. This ports them.

| change | before | after |
|---|---|---|
| Input + post-attention RMSNorm | 25 us each, **1 core** | 10 / 9 us, 32 / 16 cores |
| `nlp_create_qkv_heads` | 31 us, **1 core** | 2 us, 8 cores |
| `nlp_concat_heads` | 12 us, **1 core** | 0.5 us |
| TP=2 all-reduce (x2 per layer) | 107 us (2 CCL ops each) | 77 us (1 CCL op each) |

**Why the norms landed on one core:** a decode token is `[1, 1, 32, 1024]` — a single tile
row. The default RMSNorm parallelises over *rows*, so one row-block means one core while the
other 63 idle. Width-sharding splits along the hidden dim instead.

The post-attention norm emits directly in the gate/up matmul's `in0` layout, so the MLP's own
`to_memory_config` disappears.

**Sharded `nlp_create_qkv_heads` needs a permuted weight.** The sharded kernel reads a
KV-group-interleaved fused QKV (`[q..q, k, v]` per KV group), so a second copy of the QKV
weight with that row permutation is built at init (`lw["wqkv_kvgi"]`); the plain `[Q|K|V]`
copy is freed on this path.

**The 2-chip all-reduce** (`mesh_utils.tp_all_reduce_2chip`): `ttnn.all_reduce` lowers to
reduce_scatter + all_gather. On N300 both are dominated by fixed fabric setup rather than
payload — a 1-tile activation pays ~51 us to reduce 64 KB. With only two chips you can
all-gather the two partial sums and add the halves locally: one CCL op instead of two.
Same all-reduce, same maths, **same tensor parallelism** — only the lowering changes.

### 2.2 Decode-mode RoPE — NOT gated

`models/demos/qwen3_tts/tt/rope.py::apply_rope_qk`, used by both `attention.py` (Talker) and
`code_predictor.py` (CodePredictor).

`rotary_embedding_llama` with `is_decode_mode=False` **loops once per head**. Measured at
head_dim=128, seq padded to one tile:

| n_heads | 1 | 4 | 8 | 16 |
|---|---|---|---|---|
| prefill mode | 12.9 | 18.4 | 26.1 | **41.3 us** |

That is ~9 us fixed + **~2 us per head**, and it is unaffected by memory config (DRAM vs L1
cos/sin/trans) or math fidelity — both were tested. `is_decode_mode=True` rotates every head
inside a single tile: **3.4 us**.

Getting there costs a `ttnn.transpose(x, 1, 2)` — `[1, n_heads, 1, hd]` -> `[1, 1, n_heads, hd]`
— at 2 us, and one back. Transpose reads and writes the sharded layout directly, so no extra
reshard is needed on either side.

**Gate is `seq == 1`, not the device and not the `mode` argument.** Shape is the honest test:
the CP's "prefill" is only 2 tokens, so it *is* a prefill call, but it has two distinct
positions and must keep the prefill kernel. There is also a fallback for `n_heads > 32`, which
cannot pack into one tile (nothing in this model hits it; the helper is shared).

Constraints worth knowing:

- Decode mode requires **all** of Q/K, cos, sin and trans_mat to be `HEIGHT_SHARDED`.
  Interleaved and width-sharded are rejected outright.
- `get_rot_transformation_mat` **ignores its `dhead` argument** and always returns one 32x32
  tile, so prefill and decode share the same matrix — only the memory config differs.
  `get_decode_transformation_mat` builds it at module init so it predates any trace capture.
- cos/sin are reshared once and reused for both Q and K.

---

## 3. Measured results

### 3.1 Accuracy — no change

The two RoPE kernels are **bit-identical**, not merely close: `max|prefill - decode| == 0`
at every head count either model uses.

`tests/test_qwen3_tts_pcc.py` (real HF weights), before and after all changes:

| block | baseline | after | |
|---|---|---|---|
| `mlp_decode` | 0.999692 | 0.999692 | unchanged |
| `attention_decode` | 0.999790 | 0.999790 | unchanged |
| `cp_step` | 0.999835 | 0.999835 | unchanged |
| `talker_chain` | 0.972521 | 0.972521 | unchanged |

Identical to six decimals. The CP N300 path is bit-exact in decode and within 1.9 bf16 ULP in
prefill (the sharded RMSNorm reduces in a different order).

> **Coverage caveat.** Only `attention_decode` exercises the decode RoPE path. `cp_step` is CP
> prefill at seq=2, and `talker_chain` — despite its "seq_len=1" docstring — pads to 32 rows
> and runs `mode="prefill"`, so both use the prefill kernel. This was confirmed by
> instrumenting `rotary_embedding_llama` and counting which branch each test took. That gap is
> why `test_qwen3_tts_rope_decode.py` exists.

### 3.2 Op-level — the reliable numbers

Per-op device times were stable across all captures.

| RoPE per layer | before | after |
|---|---|---|
| Talker prefill seq=32/64/128 | 26 + 19 us | 26 + 19 us (unchanged, as intended) |
| CP prefill seq=2 | 27 + 19 us | 27 + 19 us (unchanged, 2 positions) |
| **Talker decode, N300** (8 heads / 4 KV) | 26 + 18 us | **3 + 3 us** |
| **CP decode, N300** (8 / 4) | 27 + 19 us | **4 + 4 us** |
| **Talker decode, N150** (16 / 8) | 41 + 26 us | **3 + 3 us** |
| **CP decode, N150** (16 / 8) | 43 + 27 us | **4 + 4 us** |

Counting the whole block (two cos/sin reshards + a transpose either side of each rotary op):
Talker decode 44 -> 16 us and CP decode 46 -> 18 us per layer on N300; ~67 -> ~17 us on N150.

Note the core counts in the profile: the prefill kernel spreads one token across **64 cores**
and still takes 26 us, because it walks heads serially. The decode kernel uses **1 core** and
takes 3 us.

### 3.3 Block windows

N150 (no CCL, so windows are clean — single captures):

| decode layer | before | after | |
|---|---|---|---|
| Talker | 762 us | **711 us** | -6.7 % |
| CodePredictor | 530 us | **477 us** | -10.0 % |

N300 (medians of 3 captures):

| block | baseline | after | |
|---|---|---|---|
| CP decode | 567 us | **385 us** | **-32 %** |
| CP prefill seq=2 | 517 us | **424 us** | **-18 %** |
| Talker decode | 659 us | 654 us | within noise — see below |

### 3.4 End-to-end

Traced AR decode frame, single chip: **75,528 -> 70,784 us (-6.3 %)** from the RoPE change
alone (the CP N300 sharding does not apply on one chip). CP decodes -8.3 %, Talker decode
-4.1 %, CP prefill unchanged.

---

## 4. Measurement methodology — read before comparing reports

**N300 CCL timings swing ~2x run to run.** The same `ttnn.all_gather` of a 64 KB payload
measured 34 us in one Tracy capture and 65-71 us in the next three. Baseline windows for an
identical CP decode layer ranged 484-582 us.

Consequences:

- A **single capture per block is not comparable** to another single capture. In this work an
  *untouched* code path (`talker_layer_prefill_64`, seq > 1, unaffected by either change)
  moved **+10.7 %** between two single captures. Always take medians of 3+.
- The Talker decode window on N300 is dominated by its two (still unmodified) all-reduces, so
  a real 28 us/layer RoPE saving is invisible there. **Judge that change at op level, or on
  N150 where there is no CCL.**
- N150 windows are clean; `-n 1` is acceptable there.

Regenerate the block report with:

```bash
source python_env/bin/activate
./qwen3_tts_block_report.sh                                # N300, 3 runs/block, ~16 min
./qwen3_tts_block_report.sh -m N150 -n 1 -o n150_blocks.txt
./qwen3_tts_block_report.sh -b qwen3_tts_n300_blocks.txt   # add a vs-baseline column
./qwen3_tts_block_report.sh -A                             # re-assemble, no device time
```

Manually, per block:

```bash
export TT_METAL_HOME=$(pwd) PYTHONPATH="$(pwd)" ARCH_NAME=wormhole_b0 MESH_DEVICE=N300
python -m tracy -p -v -r -m pytest -s -q \
  models/demos/qwen3_tts/tests/test_qwen3_tts_profile_single_layer.py -k test_cp_layer_decode
CSV=$(ls -t generated/profiler/reports/*/ops_perf_results_*.csv | head -1)
tt-perf-report --start-signpost start --end-signpost stop "$CSV"
```

Use the **full** test name — `-k talker_layer_prefill` matches all three buckets and puts
three windows in one capture — and one `-k` per Tracy run, since the CSV is picked by
newest timestamp.

---

## 5. Tried and rejected — do not re-propose without new evidence

| idea | measured | verdict |
|---|---|---|
| `all_gather` on dim 0 for the 2-chip all-reduce | 78 us | Tile padding / a size-1 outer dim pushes it onto the composite all-broadcast fallback. Gather on the **last** dim: 34 us. |
| `num_links=2` on that all_gather | 69 us vs 34 us on auto | Payload far too small to amortise a second link's setup. |
| Moving RoPE cos/sin/trans_mat from DRAM to L1 | 40.9 vs 41.4 us | Irrelevant — the cost is the per-head loop, not memory. |
| Lowering RoPE math fidelity | 41.4 (LoFi) vs 43.4 (HiFi4) | Same. Not a fidelity problem. |
| `nlp_create_qkv_heads_decode` instead of the sharded prefill-style split | 13.3 us vs 2 us | Much worse here. Its value is feeding a full decode-layout attention pipeline, which this model does not use. |
| Running the CodePredictor at TP=1 (replicated) on N300 to delete all CCLs | est. matmul growth +103 us vs CCL saving -107 us | Net ~zero for a large, risky change. |
| DRAM-sharding the CP QKV matmul | ~2 us | N is padded 2048 -> 2304, so it needs an S2I + slice that eats the gain. |

---

## 6. What to do next — ranked

### 6.1 Port the 2-chip all-reduce to the Talker  *(largest remaining N300 win)*

The Talker still calls `tp_all_reduce` (i.e. `ttnn.all_reduce`). In its decode layer that is
**~197 us — about 30 % of the window**, and it is the reason Talker decode numbers are so
noisy. `mesh_utils.tp_all_reduce_2chip` already exists and is trace-safe; the CP measured
107 -> 77 us per layer from it.

Call sites — note they span two files, and `mlp.py` has four:

- `tt/attention.py:1063` (after `o_proj`)
- `tt/mlp.py:336`, `:351`, `:391` (after `down_proj`, one per path)

Expect roughly -60 us/layer x 28 layers. Gate on `is_n300(device)` as the CP does — the 2-chip
form is only correct for exactly two chips — and verify with medians of 3+ captures.

`mlp.py` is shared, so check whether any non-Talker caller reaches those lines before
switching them wholesale.

### 6.2 Reduce the remaining CCL cost

Even in the 1-CCL form the all-gather is 34-70 us for 64 KB, on **1 core**. This is pure
fabric latency, not bandwidth. Worth trying:

- `ttnn.experimental.all_gather_async` with persistent output buffers and pre-allocated
  semaphores (avoids per-call setup).
- `use_l1_small_for_semaphores=True`.
- Investigating the run-to-run variance itself — if it is fabric arbitration, pinning
  `sub_core_grids` may stabilise it.

### 6.3 CP matmul fidelity: HiFi4 -> HiFi2

The CodePredictor hardcodes `MathFidelity.HiFi4` for all matmuls (`self.kcfg`) while the
Talker uses LoFi, and `tt-perf-report` explicitly advises HiFi2. That is ~119 us of matmul per
CP decode layer, likely 15-25 % recoverable.

**Not attempted** because it is a genuine accuracy change, unlike everything above. Gate it,
then run `test_qwen3_tts_pcc.py` and listen to the demo output. Note `sdpa_kcfg` is separate
and should stay high-fidelity: the code documents that QK-norm amplifies K by ~68x and q.k
dot products can overflow bf16, which is why the SDPA chain runs in fp32.

### 6.4 The CP's manual fp32 SDPA chain

~37 us/layer of `repeat_interleave` (which lowers to untilize + concat + tilize) plus five
typecasts, to expand KV heads for GQA and move between bf16 and fp32.

`ttnn.transformer.scaled_dot_product_attention_decode` handles GQA natively without
`repeat_interleave` and supports fp32 accumulation via `fp32_dest_acc_en`. Bigger refactor,
real overflow risk — read the comment above the SDPA chain before starting.

### 6.5 Talker prefill buckets 64 and 128

`attention.py` has `use_dram_shard_qkv = seq_len <= 32`, with a TODO: buckets 64 and 128 need
their own per-`m` shard configs to engage the DRAM-sharded QKV and the sharded
`nlp_create_qkv_heads`. At seq=64 the profile shows `nlp_create_qkv_heads` at 25 us on 2 cores
— the same single-core-ish problem already fixed elsewhere.

Prefill runs once per utterance, so this matters for time-to-first-audio, not steady state.

### 6.6 Lower value

- **CP `o_proj` DRAM-sharded** — ~5 us/layer net after the S2I + slice for the 1024 -> 1152 pad.
- **Hoist the cos/sin reshard out of the layer.** `apply_rope_qk` reshards cos/sin per layer
  (2 us); they are identical across all layers of a forward pass, so they could be resharded
  once by the caller. ~2 us x 5 CP layers x 15 CP passes = ~150 us/frame. Needs care to stay
  trace-safe.

---

## 7. Files

| file | change |
|---|---|
| `tt/rope.py` | `apply_rope_qk`, `get_decode_transformation_mat`, `_rope_decode_memcfg` |
| `tt/attention.py` | Talker uses `apply_rope_qk`; builds `_decode_trans_mat` at init |
| `tt/code_predictor.py` | N300 fast path (`_n300_cp_opt`) + `apply_rope_qk` |
| `tt/mesh_utils.py` | `is_n300`, `tp_all_reduce_2chip` |
| `tests/test_qwen3_tts_rope_decode.py` | RoPE bit-exactness + routing guard |
| `tests/test_qwen3_tts_cp_n300_opt.py` | CP fast path A/B + Metal-trace replay guard |
| `qwen3_tts_block_report.sh` (repo root) | regenerates the block report |

### Validation run before merging

```bash
source python_env/bin/activate
export TT_METAL_HOME=$(pwd) PYTHONPATH="$(pwd)" ARCH_NAME=wormhole_b0

pytest -s models/demos/qwen3_tts/tests/test_qwen3_tts_pcc.py            # accuracy, real weights
pytest -s models/demos/qwen3_tts/tests/test_qwen3_tts_rope_decode.py    # RoPE bit-exact + routing
pytest -s models/demos/qwen3_tts/tests/test_qwen3_tts_cp_n300_opt.py    # CP A/B + trace (opens its own 1x2 mesh)
pytest    models/demos/qwen3_tts/tests/test_qwen3_tts_trace_perf.py     # full model under Metal trace
MESH_DEVICE=N150 pytest models/demos/qwen3_tts/tests/test_qwen3_tts_profile_single_layer.py
MESH_DEVICE=N300 pytest models/demos/qwen3_tts/tests/test_qwen3_tts_profile_single_layer.py
```

Every one of these passed with the changes described above.
