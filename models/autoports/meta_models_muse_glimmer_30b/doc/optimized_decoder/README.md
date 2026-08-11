# Optimized decoder — meta-models/Muse-Glimmer-30B (text decoder)

Stage 02 of the repo-local TTNN autoport pipeline: the same Muse-Glimmer **text** decoder
layer as stage 01, with the per-device performance work that stage deliberately deferred.
Same model math, same public contract, same 0.995 PCC bar, on one Blackhole chip.

* implementation: [`../../tt/optimized_decoder.py`](../../tt/optimized_decoder.py)
* stage-01 baseline: [`../functional_decoder/README.md`](../functional_decoder/README.md)
* tests: [`../../tests/test_optimized_decoder.py`](../../tests/test_optimized_decoder.py),
  [`../../tests/test_optimized_decoder_perf.py`](../../tests/test_optimized_decoder_perf.py)
* measured numbers: [`evidence_tables.md`](evidence_tables.md) (generated from the raw artifacts)
* chronology, bugs, exact commands: [`work_log.md`](work_log.md)
* context contract: [`../context_contract.json`](../context_contract.json)

## Headline

One decoder layer, one Blackhole p300c chip, warmed, unprofiled wall clock:

| measurement | functional (stage 01) | optimized | speedup |
|---|---|---|---|
| warmed prefill, 4096 tokens, sliding_rope | 58.79 ms | **25.20 ms** | 2.33x |
| warmed prefill, 4096 tokens, full_nope | 59.12 ms | **24.53 ms** | 2.41x |
| warmed prefill, 8192 tokens, sliding_rope | 97.65 ms | **53.44 ms** | 1.83x |
| warmed prefill, 8192 tokens, full_nope | 97.55 ms | **52.65 ms** | 1.85x |
| traced warmed decode, batch 1, context 4096, sliding_rope | 3.242 ms | **1.063 ms** | 3.05x |
| traced warmed decode, batch 1, context 4096, full_nope | 3.223 ms | **1.045 ms** | 3.08x |
| traced warmed decode, batch 32, context 4096, sliding_rope | 3.398 ms (9.4 k tok/s) | **1.160 ms (27.6 k tok/s)** | 2.93x |
| traced warmed decode, batch 32, context 4096, full_nope | 3.565 ms (9.0 k tok/s) | **1.278 ms (25.0 k tok/s)** | 2.79x |

Both stages measured with the same harness on the same device; the table in
[`evidence_tables.md`](evidence_tables.md) is generated from the two `perf_summary.json`
artifacts so it cannot drift.

## What changed

Five things, in descending order of measured effect. The candidate table in
[`evidence_tables.md`](evidence_tables.md) has the isolated cost of each.

1. **A named per-tensor-group precision policy** instead of one global BF16.
   `bfp4_all`: BF16 activations and norms, **BFP4 weights at LoFi** for the attention
   projections *and* the SwiGLU projections, **BFP8 paged KV cache**, HiFi4 for the norms
   and SDPA. Decode 2.202 ms → 1.064 ms against the same optimized layout.
   Math fidelity alone is worth 32% (BFP8+HiFi2 1.845 ms → BFP8+LoFi 1.254 ms), which is
   why fidelity is swept as a knob of its own and not inferred from the dtype.
2. **DRAM-sharded decode matmuls.** Weights width-sharded over the 8 Blackhole DRAM banks,
   activations and outputs L1 width-sharded on the matching core grid, explicit
   `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` per role with an L1-derived
   `in0_block_w`. 1.402 ms → 1.065 ms against the functional stage's default-program-config
   interleaved family at the same dtype.
3. **An L1 width-sharded decode residual stream** carried through both pre-norms, both
   post-norms, both residual adds, the attention-gate multiply and every projection, with
   sharded `LayerNormShardedMultiCoreProgramConfig` norms. The functional stage's four
   hidden-size RMSNorms ran on **one core at 138 µs each** — 552 µs of a 3.24 ms decode
   step. They now run on 16 cores at ~9 µs each: 37 µs total. The two weight-less QK
   RMSNorms over `head_dim` are *not* sharded — they run on 1 core at ~4 µs each, 8 µs of
   the 45 µs total norm time, because they consume the height-sharded per-(user, head)
   layout `nlp_create_qkv_heads_decode` produces and `ttnn.rms_norm` has no sharded program
   config for it. Named in *Limitations*.
4. **Explicit 2D prefill program configs** over a large Blackhole grid, with the sequence
   folded into 512-row blocks so the per-core output block fits L1. The functional stage's
   prefill matmuls were all marked `SLOW` with `in0_block_w=1` and ran at 37-50% of the
   HiFi2 FLOP peak. They now run at an L1-budget-derived `in0_block_w` — 26 tiles for the
   QKV, attention-gate and SwiGLU-down projections, 16 for the output projection, 8 for the
   19968-wide SwiGLU gate/up where the budget stops it — with output subblocks up to `1x8`.

   <!-- generated:prefill-bounds -->
Measured across every prefill matmul row in the four committed reports: **53.7-77.7%** of the LoFi FLOP peak. Rows marked `SLOW` in at least one artifact: `4096 x 6656`, `6656 x 19968`; every other shape is `FLOP`-bound.
<!-- /generated:prefill-bounds -->

   `out_subblock_h` was swept too (1 / 2 / 4 → 52.86 / 53.14 / 53.34 ms at 8192 tokens);
   `h=1` stays. One SwiGLU row is additionally `SLOW` on the sliding path at 8192 tokens for
   a reason that is not a program-config choice — see *Anomalies*.
5. **Fused elementwise**: SiLU folded into the SwiGLU multiply and sigmoid folded into the
   attention-gate multiply, removing two full-size unary passes per layer per phase
   (3.2% of functional prefill device time on its own).

Everything that decides *correctness* is unchanged from stage 01: chunked prefill, the
sliding-window overlap trim, the paged fill/update contract, the SDPA chunk-size and
page-table-capacity guards (each of which exists because a real device bug was measured),
RoPE/NoPE dispatch, the `qk_scale_factor` placement, and support for any logical sequence
length. `optimized_decoder.py` imports those constants and guards from
`functional_decoder.py` rather than restating them.

## Public contract

Identical to the functional layer, with two documented differences:

```python
OptimizedDecoder.from_state_dict(
    state_dict, *, hf_config, layer_idx, mesh_device,
    precision="bfp4_all",           # name in POLICIES, or a PrecisionPolicy
    state_dict_prefix=None, block_size=64, prefill_chunk_size=8192,
    rope_max_seq_len=None, ...)     # plus the swept geometry knobs

decoder.allocate_kv_cache(*, batch_size, max_seq_len, num_blocks=None)
decoder.prefill_forward(hidden_states, *, kv_cache, page_table, user_ids=None,
                        seq_len=None, start_pos=0)
decoder.decode_forward(hidden_states, *, kv_cache, page_table, current_pos, rope_idxs=None)
decoder.forward(*args, mode="prefill"|"decode", **kwargs)
decoder.config_summary()                    # the selected policy and program configs
decoder.decode_residual_memory_config       # the layer-to-layer decode activation contract
```

1. **`decode_forward` returns the hidden state L1 width-sharded**, in
   `decoder.decode_residual_memory_config` (16 cores, 416-wide shards), and accepts either
   an interleaved or an already-sharded input. That is the optimized layer-to-layer
   contract: stacking layers needs no reshard. `ttnn.to_torch` reads it back unchanged, so
   every functional test still applies —
   `test_no_runtime_host_fallback` asserts both the sharding and the round trip.
2. **`precision`** selects the named policy. Prefill still takes and returns
   `[batch, 1, seq_len, hidden_size]` DRAM-interleaved tensors, because prefill activations
   are large and belong in DRAM.

The invariants stage 01 guarantees are unchanged and re-tested here: a single
`prefill_forward` call handles **any** logical `seq_len` in `[1, 131072]` with no
`seq_len % N == 0` requirement on the public path; `current_pos`/`rope_idxs` are plain
device tensors so decode is trace-safe; after `from_state_dict` both paths are pure device
paths; continued prefill (`start_pos > 0`) is supported on `full_attention` layers only.

## Precision policy

| tensor group | dtype | math fidelity |
|---|---|---|
| activations, residual stream | BF16 | — |
| norm weights | BF16 | HiFi4, `fp32_dest_acc_en` |
| QKV projection, attention gate projection, output projection | **BFP4_B** | **LoFi** |
| SwiGLU gate / up projections | **BFP4_B** | **LoFi** |
| SwiGLU down projection | **BFP4_B** | **LoFi** |
| paged K/V cache | **BFP8_B** | — |
| SDPA / FlashDecode | BF16 Q, BFP8 cache | HiFi4, `fp32_dest_acc_en` |

Decode matmuls use `packer_l1_acc=True, fp32_dest_acc_en=False`: turning packer L1
accumulation off costs 26% and FP32 accumulation costs 2% with no PCC benefit at BFP4.
Math fidelity is swept as a knob of its own, per projection group and at the shipped BFP4
weights, not inferred from the dtype: HiFi2 on the attention projections costs 12%
(1.195 ms) and HiFi2 on the SwiGLU projections costs 58% (1.675 ms) against LoFi
everywhere.

The dominant-matmul table in [`evidence_tables.md`](evidence_tables.md) is read out of the
committed `tt-perf-report` rows and checked against this table by
[`../../scripts/render_optimized_evidence.py`](../../scripts/render_optimized_evidence.py),
which exits non-zero on a mismatch. A policy object is intent; the profiler row is the
evidence.

### Real weights versus synthetic weights

`bfp4_all` is selected on **real checkpoint weights and real activations**, where it measures
prefill PCC 0.9985 / 0.9989 and decode 0.9970 / 0.9984 (sliding_rope / full_nope) — above
the 0.995 bar with margin. On **synthetic random weights at the real per-tensor scales** the
same code measures 0.960-0.969, because BFP4's shared-exponent block quantisation is far
lossier on i.i.d. Gaussian blocks than on the checkpoint's own strongly correlated ones. That is
the `$optimize` skill's OPT-012 case: a weights discrepancy, not an implementation defect.

The suite is built so this cannot be waved through:

* every structural test (page tables, block sizes, non-aligned lengths, ragged slots,
  batching, window enforcement, continued prefill, short prefills) runs the **same code
  path** at `bfp8_all_lofi` and asserts the full 0.995 bar on synthetic weights, so a
  page-table, chunking or layout bug still fails the suite at full strength;
* acceptance for the shipped BFP4 policy comes from real-weight tests that reproduce the
  disputed contracts — `test_real_weights_prefill_decode`,
  `test_real_weights_non_aligned_and_traced` (non-aligned length, paged prefill→decode
  transition, 3-step traced replay), `test_real_weights_batched` (four users sharing a
  prefill and a decode step) and `test_stress_repeated_prefill_decode` — all at 0.995;
* `test_synthetic_weight_precision_discrepancy` records the gap itself, for every policy,
  into the PCC artifact. Nothing is marked xfail.

Every candidate latency comes from one harness (`scripts/sweep_optimized_decoder.py`), which
also PCC-gates each candidate in the same loop — a wall-clock sweep without a correctness
gate is what nearly shipped a NaN-producing prefill config (see *TTNN findings*). Six repeats
of the shipped configuration in one process measured 1.0632, 1.0792, 1.0776, 1.0775, 1.0806,
1.0764 ms: **1.6% peak-to-peak**, with the first repeat consistently fastest, i.e. there is a
within-process warm-up drift larger than several of the differences the table reports. So a
single measurement cannot separate a 1-2% effect from drift, and the two selections that fall
in that band were re-measured **paired** (A/B/A/B) instead:

| paired comparison | pairs | effect |
|---|---|---|
| SwiGLU working cores 52 vs 26 (decode) | 1.0638/1.0886, 1.0811/1.0953, 1.0780/1.0951 | 26 slower by 1.42%, 1.31%, 1.59% — **consistent, 52 kept** |
| prefill `in0_block_w` cap 26 vs 8 | 53.334/54.360, 53.385/54.294 | cap 8 slower by 1.92%, 1.70% — **consistent, 26 kept** |

Differences below ~1.7% that were *not* re-measured paired are reported as ties, not wins:
the packed QKV+gate result (1.0651 vs 1.0636), the KV-cache dtype result, the SDPA-output
memory-config result, the RoPE-pad result, the `wo`/`mlp_down` `in0_block_w` neighbours and
the prefill `out_subblock_h` result. The differences that decided the stage — dtype and
fidelity policy, matmul family, the small-core SwiGLU geometries, `in0_block_w` at 1-2,
`packer_l1_acc`, prefill `grid_y` and prefill fold cutoff — are all 12-150%, far outside it.

The conservative alternative is one constructor argument away:
`precision="bfp8_attn_bfp4_mlp"` keeps BFP8 attention weights and costs 3.3% of decode
(1.0987 ms vs 1.0639 ms); `bfp8_all_lofi` costs 18% (1.2540 ms). Both are in
[`perf/candidates.json`](perf/candidates.json).

## Decode topology

Per token, batch 1 (the op sequence a `tt-perf-report` window shows):

```
x (L1 width-sharded, 16 cores)
  -> sharded RMSNorm ------------------------------------- 9 µs
  -> QKV matmul   [32 x 6656 x 4608]  DRAM-sharded ------- 57 µs
  -> gate matmul  [32 x 6656 x 4096]  DRAM-sharded ------- 52 µs
  -> nlp_create_qkv_heads_decode (L1) -------------------- 13 µs
  -> QK RMSNorm x2, scale, RoPE gather + rotate ---------- 26 µs
  -> paged_update_cache x2 (BF16 in, BFP8 cache) ---------- 7 µs
  -> paged SDPA decode (8x4 grid, k_chunk 64) ------------ 37 µs
  -> nlp_concat_heads_decode -> gate multiply (sigmoid fused)
  -> out matmul   [32 x 4096 x 6656]  DRAM-sharded ------- 53 µs
  -> reshard to residual grid, sharded RMSNorm, residual add
  -> sharded RMSNorm -> reshard to the SwiGLU working shard (52 cores)
  -> gate matmul  [32 x 6656 x 19968] DRAM-sharded ------ 237 µs
  -> up matmul    [32 x 6656 x 19968] DRAM-sharded ------ 236 µs
  -> multiply with SiLU fused
  -> down matmul  [32 x 19968 x 6656] DRAM-sharded ------ 230 µs
  -> reshard to residual grid, sharded RMSNorm, residual add
```

Device-time share: matmuls 82.4%, elementwise 4.2%, norms 4.3%, SDPA 3.5%, everything
else 5.5%. The four `Reshard` ops per layer — the SwiGLU working-shard boundary in and
out, and the two matmul outputs that have to land back on the residual grid before their
norm — cost 7.1 µs combined, 0.7%.

The SwiGLU projections use a **phase-specific working shard** of 52 cores rather than the
residual grid's 16. `gcd(hidden_tiles 208, qkv_tiles 144) = 16` pins the attention
projections to 16 cores, but `gcd(208, 624) = 208` lets the SwiGLU projections use far
more — and at 16 cores the gate/up matmul needs `per_core_N = 39`, whose triple-buffered
weight circular buffer alone is 2080640 B against a 1572864 B L1, so the op does not fit at
all. Resharding in and out costs two ~425 KB L1→L1 moves and unlocks the family:
16 cores 1.506 ms, 26 cores 1.080 ms, **52 cores 1.064 ms**, 104 cores 1.116 ms.

## Performance accounting

| workload | roofline | device | end-to-end | device/roofline | host term |
|---|---|---|---|---|---|
| decode b1 ctx 4096 sliding_rope | 0.536 ms (DRAM) | 1.050 ms | 1.063 ms | 1.96x | 0.014 ms (1.3%) |
| decode b1 ctx 4096 full_nope | 0.536 ms (DRAM) | 1.034 ms | 1.045 ms | 1.93x | 0.010 ms (1.0%) |
| prefill 8192 sliding_rope | 13.035 ms (110-core LoFi) | 49.909 ms | 53.436 ms | 3.83x | 3.527 ms (6.6%) |
| prefill 8192 full_nope | 13.035 ms (110-core LoFi) | 47.697 ms | 52.655 ms | 3.66x | 4.957 ms (9.4%) |

Generated into [`perf/accounting.json`](perf/accounting.json) by
[`../../scripts/render_optimized_accounting.py`](../../scripts/render_optimized_accounting.py)
so none of it is hand-typed. The decode roofline is 272.2 MB of BFP4 weights plus 2.2 MB of
BFP8 KV cache at 512 GB/s; the prefill roofline is 7.93 TFLOP at 5.5296 TFLOP/s per core
across 110 cores. Both constants are `tt_perf_report`'s own
`ArchitectureSpec("blackhole")` values - the same ones its `DRAM %` and `FLOPs %` columns
normalise against - so the roofline and the committed report percentages cannot disagree.
`dram_bandwidth_gb_s` is chip-wide, not scaled by the cores an op used.

Named limitations, in the order they bound the result:

* **Decode sits at ~1.96x the DRAM roofline because the DRAM-sharded matmul picks its own
  worker set — 12 cores on this p300c.** Two different counts matter here and are easy to
  conflate: the **DRAM bank count is 8** (`mesh_device.dram_grid_size().x`, the grid the
  weights are width-sharded over, and the value `_prefill_grid_x` must match), while the
  **DRAM-reader worker set is 12 cores**, chosen by the op's own
  `get_optimal_dram_bank_to_reader_assignment` and reported in the `Cores` column of every
  dominant decode row. The roofline itself is chip-wide, not worker-relative: 512 GB/s is
  `tt_perf_report`'s `ArchitectureSpec("blackhole").dram_bandwidth_gb_s`, the same constant
  its `DRAM %` column divides by, and 5.5296 TFLOP/s per core is its LoFi
  `tflops_per_core`. The committed rows show those 12 workers simultaneously at ~53% of the
  chip DRAM bandwidth *and* ~53% of their own 12-core LoFi FLOP peak, so neither is
  saturated and the gap is read/compute overlap on a 12-core worker set.
  This was not accepted on the untuned default: an explicitly configured
  `MatmulMultiCoreReuseMultiCast1DProgramConfig` candidate — L1 width-sharded `mcast_in0`,
  DRAM-interleaved weights, per-role `in0_block_w`, same BFP4/LoFi policy — was built
  (`decode_matmul="mcast1d"`) and measured at **1.231 ms**, 16% slower than the
  DRAM-sharded 1.064 ms, and the no-config interleaved baseline at 1.402 ms. 16 cores is
  the largest legal rectangle that family can use for every role at once, because
  `num_cores` must divide both the tiled K and the tiled N and `gcd(208, 144) = 16` while
  26/52/104 have no rectangle inside an 11x10 grid. So more compute cores than 12 *is*
  expressible and it loses; going further needs an op-level change, not a config. This
  remains the single largest decode opportunity.
* **Decode end-to-end is 1.3% above device time**, so the traced decode loop carries no
  material host term. There is nothing left to remove there.
* **Prefill matmuls run on 64 of 110 cores.** `grid_x` is pinned to the 8 DRAM banks
  because a 2D multicast matmul with DRAM width-sharded weights returns NaN for any other
  width (below), and `grid_y` above 8 measured 48% slower. The reachable 64-core LoFi
  roofline is 22.4 ms against a measured matmul
  time of 32.59 ms sliding /
  31.75 ms full
  (69% /
  71% of it);
  both are generated into `perf/accounting.json` as `roofline_64_core_ms` and
  `matmul_device_ms`. The 13.0 ms figure in the table above is the unreachable 110-core
  roofline.
* **Prefill end-to-end is 6.6-9.4% above device time**, down from ~20% in the functional stage.
  What remains is per-chunk host dispatch. Prefill is not traced because the chunk count
  depends on the prompt length.
* **Non-matmul prefill time is 34% of the window**: SDPA 13.1%, elementwise 9.3%, norms
  8.5%. Prefill activations are DRAM interleaved by design, so the norms are not sharded.

## Anomalies

**One of the two identical SwiGLU projections is sporadically slower, on the sliding path at
8192 tokens only.** The figures below are regenerated from the committed CSV by
[`../../scripts/render_optimized_evidence.py`](../../scripts/render_optimized_evidence.py) —
an earlier hand-typed version of this paragraph drifted from the artifact it cited, so it is
no longer typed.

<!-- generated:anomaly -->
In `tracy/sliding_rope/prefill_8192_perf_report.csv` the 10 `6656 x 19968` rows (5 iterations x 2, identical
shape, cores, dtype, fidelity, `in0_block_w` and output subblock) measure

```
8940 / 10047    8940 / 9747    8942 / 8942    8935 / 9776    8938 / 9774     (microseconds)
```

6 of the 10 sit at 8935-8942; **4 carry the `SLOW` marker** at
9747-10047, i.e. 9.0-12.4% slower. 1 of the 5 iterations is
symmetric, so it is intermittent rather than positional. The excess over the stable
baseline is 717 microseconds per iteration, **1.44% of the window**.

Controls, all from the committed artifacts: the same op at 4096 on the same path 4468-4473 us over 10 rows, 0 SLOW; the same op at 8192 on the full-attention path 8938-8946 us over 10 rows, 0 SLOW; the down projection on the same path 7921-7933 us over 5 rows, 0 SLOW. So it is neither a shape, a
program-config nor a dtype effect.
<!-- /generated:anomaly -->

What is specific to sliding_rope at 8192 is DRAM occupancy history: that path runs the
*non-chunked* windowed SDPA over the whole 8192-token slice, so Q `[1, 32, 8192, 128]` plus
K/V plus the attention output are allocated and freed immediately before the SwiGLU block,
whereas the full-attention path streams K/V out of the paged cache and never holds them. The
two SwiGLU projections then allocate 327 MB outputs back to back. The reading is DRAM
allocator/refresh state, not compute — which is why it is intermittent and why the down
projection, which runs after both large intermediates are freed, never shows it.

It is left in place rather than worked around: both candidate mitigations lose. Packing gate
and up into one 39936-wide matmul measured 56.6 ms against 53.2 ms, and sub-blocking the
SwiGLU over the sequence would add a concat per chunk. It is inside the headline number, not
excluded from it.

## TTNN findings

Two silent-wrong-answer cases were found and worked around; both deserve upstream
validation checks. Standalone repro:
[`../../scripts/repro_prefill_matmul_grid_x9.py`](../../scripts/repro_prefill_matmul_grid_x9.py),
output committed at [`logs/repro_prefill_matmul_grid_x9.log`](logs/repro_prefill_matmul_grid_x9.log).

1. **`MatmulMultiCoreReuseMultiCastProgramConfig` returns NaN when `in1` is DRAM
   width-sharded and `per_core_N` is not exactly the in1 DRAM shard width in tiles**, i.e.
   unless `grid_x == num_dram_banks`. With a DRAM-interleaved `in1` every grid width is
   correct. The QKV projection's 144 tiled columns divide by 9 exactly, so a 9-wide grid
   looks like a legal way to use 72 cores instead of 64 — and returns NaN. Nothing is
   raised. It survived a whole-layer wall-clock sweep looking ~5% *faster* than the legal
   grid before a PCC test caught it (whole-layer prefill PCC 0.765, paged K cache entirely
   NaN, at every sequence length).
2. **The same op accepts `per_core_N * grid_x > N_tiles`** (a 10-wide grid on 144 tiles
   gives `per_core_N = 15`, covering 150) rather than rejecting it.

The layer pins `grid_x` to the DRAM bank count in `_prefill_grid_x`, which makes both
unreachable.

The functional stage's four TTNN findings still stand and their guards are carried over
verbatim: `chunked_scaled_dot_product_attention` cannot take a Python `scale`; the chunked
prefill SDPA and the paged decode SDPA both round their K extent up and read the page table
past its last entry; `scaled_dot_product_attention` *hangs* when `q_chunk_size` does not
divide the Q length; the decode SDPA needs one core per (user, KV head) and silently folds
heads together otherwise.

## Tests

<!-- generated:counts -->
**105** correctness tests, **285** PCC records, **86** measured candidates in 13 sections, **56** tests under watcher.
<!-- /generated:counts -->

All passing ([`logs/full_suite.log`](logs/full_suite.log)):

```bash
python -m pytest models/autoports/meta_models_muse_glimmer_30b/tests/test_optimized_decoder.py -q
```

Same coverage and the same test names as the functional suite, so the two can be diffed
test-for-test: paged prefill+decode PCC at 9 sequence lengths x 2 layer kinds, page block
sizes 32/64/128, batch 4 and 32, ragged slots and ragged decode positions, sub-tile prompts
(1/7/31), batched multi-chunk prefill from a shared pool, the continued-prefill contract,
sliding-window enforcement, bit-identical determinism, trace capture+replay PCC measured
from the replayed output, the SDPA chunk/page-table capacity guards, the runtime
host-fallback tripwire, the AST audit, real-weight prefill/decode, real-weight non-aligned
+ traced coverage, repeated-run stress, and the synthetic-vs-real precision diagnostic.

Two tests were added after the first stage review: `test_full_context_prefill_and_decode`
(the 131072-token contract on this stage's own prefill path) and `test_real_weights_batched`
(batch 4 on real weights at the shipped policy, which OPT-012 lists among the conditions a
synthetic-only precision failure must be re-checked under).

`test_optimized_beats_functional_topology` asserts from `config_summary()` that the shipped
configuration really is DRAM-sharded, L1-width-sharded, BFP4 and BFP8-cached — a cheap
guard against a future edit quietly reverting the stage.

## Runtime fallback audit

Unchanged from stage 01 and re-run against this layer: `test_no_runtime_host_fallback` runs
one prefill and one decode inside a `torch.overrides.TorchFunctionMode` that raises on any
torch op, with `ttnn.from_torch` / `ttnn.to_torch` / `ttnn.as_tensor` replaced by
tripwires; `test_source_has_no_runtime_torch` walks the module AST and fails if `torch` or
a ttnn host-transfer entry point appears outside `from_state_dict` and `allocate_kv_cache`.
No `to_torch`/`from_torch`, no `tilize`/`untilize` and no host fallback appears in the
committed decode or prefill perf reports. The tilize took work to remove and is worth
recording: the batch-1 decode window originally carried 2
`TilizeWithValPaddingDeviceOperation` per token (11.2 µs, 1.07% of the step), inherited
unchanged from the functional stage. It came from the `ttnn.embedding` ->
`rotary_embedding_hf` boundary in `_rope_decode_mats`, where the decode cos/sin tables must
be ROW_MAJOR for the gather and TILE for the rotary kernel and a 1-row gather has 31 rows to
pad. `decode_rope_pad_to_tile` (on by default) gathers a whole tile row of positions
instead, and the op disappears from the report: the RoPE metadata path is now
Embeddings 5.2 + Transpose 8.3 + Slice 3.2 + Pad 1.0 = 17.7 µs per token against the previous
Embeddings + Tilize + Transpose = 17.8 µs, i.e. the same total with the tilize gone. It never
appeared at batch 32, where 32 users fill the tile row.

One `Typecast` pair remains in prefill (2 per chunk, 29.8 µs of a 49.7 ms window, 0.06%): the
BF16 K/V fill tensors cast to the BFP8 cache dtype, which `paged_fill_cache` requires.

The other layout ops in the measured decode path are the four
`Reshard`s at the SwiGLU working-shard boundary and the `InterleavedToSharded` /
`ShardedToInterleaved` pairs the head-creation and RoPE helpers require at their API
boundaries; all are named in the op-family table with their cost. A `sliding_rope` decode
window carries three `Slice` ops and one `Pad` per token — the head-concat batch trim plus two
`Slice` and one `Pad` from the tile-padded RoPE gather; a `full_nope` window carries one
`Slice` and no `Pad`, because a NoPE layer has no gather at all. None is a fallback.

## Watcher

`TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 TT_METAL_WATCHER_DISABLE_ETH=1` over 56 of
the 105 tests, both layer kinds, including the real-weight and stress tests: **56 passed,
zero watcher-detected errors**, minimum stack headroom 1312 bytes free. See
[`watcher/WATCHER_SUMMARY.md`](watcher/WATCHER_SUMMARY.md). Watcher and profiler runs are
kept separate, as `$tt-device-usage` requires.

## Capability and context contract

No reduction, and the full 131072-token contract is measured on **this** path at **both**
the structural BFP8 policy and the shipped BFP4 one, rather than inherited from stage 01:
`test_full_context_prefill_and_decode` prefills 131072 and 131071 tokens and decodes at
position 131071, on both layer kinds. At `bfp8_all_lofi` it clears the full 0.995 bar
(prefill 0.99757-0.99831, K-cache 0.99970, decode 0.99851). At the shipped `bfp4_all` it
measures 0.9496-0.9624 on *synthetic* weights, which is the same synthetic-BFP4 artifact as
everywhere else and is asserted as a length-*shape* check rather than an absolute bar — see
below. The reference is driven with a query-block filter that keeps the first, a middle and
the last block of each prompt (24576 of 131072 positions in 3 blocks); the shorter lengths
keep full every-position coverage in `test_paged_prefill_decode_pcc`, and each PCC record
carries its exact coverage string.

**Does BFP4 lose accuracy as the context grows?** It is the one question a per-layer PCC bar
cannot answer, because a `full_attention` layer attends over the whole paged prefix, so error
in the cached K/V compounds with the number of keys in the softmax while a sliding layer never
sees more than its 2048-token window. Measured, last query block of a `full_nope` prefill
([`pcc/length_dependence.json`](pcc/length_dependence.json)):

| weights | policy | 1000 | 8192 | 32768 / 12345 | drift |
|---|---|---|---|---|---|
| synthetic | `bfp8_all_lofi` | 0.99814 | 0.99795 | 0.99760 | -0.0005 |
| synthetic | `bfp8_attn_bfp4_mlp` | 0.99148 | 0.99131 | 0.99096 | -0.0005 |
| synthetic | **`bfp4_all`** | 0.95764 | 0.95553 | 0.94978 | -0.0079 |
| real | `bfp8_all_lofi` | 0.99992 | 0.99992 | 0.99980 | -0.0001 |
| real | `bfp8_attn_bfp4_mlp` | 0.99925 | 0.99926 | 0.99915 | -0.0001 |
| real | **`bfp4_all`** | 0.99887 | 0.99886 | 0.99867 | -0.0002 |

(synthetic to 32768, real to 12345 — the largest real-weight length whose host reference is
affordable.)

So the effect is real on synthetic weights, it is caused by BFP4 **attention** weights
specifically — `bfp8_attn_bfp4_mlp` is flat, and it differs from `bfp4_all` only there — and
on **real** weights it is 40x smaller and 25x inside the bar. `test_real_weights_length_independence`
gates it: both lengths at 0.995 absolute and the long length within 0.005 of the short one,
for the shipped policy and for the conservative alternative. This is the same real-versus-
synthetic discipline the precision policy itself was selected under, applied to the one axis
a fixed-length PCC test cannot see.

Batch is still capped at 55 users by the decode SDPA's
one-core-per-(user, KV head) requirement on the 110-core grid; 32 is tested.

## `$optimize` checklist

Every item of the skill's *Evidence To Leave* checklist, with the artifact behind it. The
items marked *n/a* are checked against this model rather than asserted: `text_config` has no
expert/MoE keys and `model.safetensors.index.json` contains zero `*expert*` tensors (dense
SwiGLU, so no routed active-expert path); `lm_head.weight` is a model-level tensor and is not
in `reference/hf_reference.py::LAYER_PARAM_NAMES` (so no LM head or sampling inside this
layer); and the mesh is `(1, 1)`, so no collective appears in any committed report.

| item | evidence |
|---|---|
| Decoder path fully traced, no host fallbacks | `test_traced_decode_pcc`, `test_no_runtime_host_fallback`, `test_source_has_no_runtime_torch`; the op lists in `tracy/` |
| Decode activations width-sharded in L1 across norm / attention / residual / MLP / output boundaries | `decode_residual_memory_config`; `Input 0 Memory = L1_WIDTH_SHARDED` on every dominant decode row in `evidence_tables.md` |
| Prefill activations DRAM interleaved, 2D program configs for the large matmuls | `_prefill_linear`; the `in0:dram_interleaved` op-family rows and the 2D `b={16} x 512 x …` matmul rows |
| Operation-topology audit recorded | `work_log.md` §1 — current op sequence with µs, candidate replacements, dtype constraints, action, evidence |
| Multi-device topology candidate families | *n/a*, `(1, 1)` mesh; stage 03 owns it |
| Lower-movement residual candidates measured without an old-contract restore | *n/a*, no collectives |
| Best-candidate comparison against the strongest baseline | `perf/candidates.json` — 13 sections, against the stage-01 baseline and each other; candidates measured after the second review are PCC-gated in the same loop, and the two selections inside the 1.6% drift band were re-measured paired |
| Final default reproduced the selected candidate | `perf/perf_summary.json` unprofiled rows are the headline numbers, measured on the shipped defaults |
| Dtype/fidelity policy verified in the measured rows, not the policy object | the dominant-matmul table in `evidence_tables.md`, machine-checked by `render_optimized_evidence.py` (non-zero exit on mismatch) |
| SDPA and other optimized composite ops used | `scaled_dot_product_attention`, `chunked_scaled_dot_product_attention`, `paged_scaled_dot_product_attention_decode`, `paged_fill_cache`, `paged_update_cache`, `nlp_create_qkv_heads*`, `nlp_concat_heads*`, `rotary_embedding_hf` |
| Fused/packed same-input projections measured, kept only if they win | Q/K/V packed; QKV+gate and SwiGLU gate/up packing both built and measured, both lose or tie — `perf/candidates.json` §"Same-input projection packing" |
| Explicit `memory_config`, `program_config`, `compute_kernel_config` on the important ops | `config_summary()`, recorded into every `perf_summary.json` record |
| Per-role program-config sweep for the dominant matmuls (core grid, `in0_block_w`, output subblock h and w, memory configs, compute kernel) | `perf/candidates.json` §"in0_block_w per matmul role", §"SwiGLU working-shard core count", §"Decode residual shard grid", §"Prefill 2D matmul geometry", §"Review follow-ups", §"Second-review follow-ups" (`out_subblock_h` 1/2/4), §"Paired re-measurements" |
| Compute fidelity swept as a performance knob per dominant projection group, in both phases | `work_log.md` §3 (whole policy) and §6.7 (per group at BFP4: attention +12%, SwiGLU gate/up +39%, SwiGLU down +20%, prefill attention +7.6%, prefill SwiGLU +35%) |
| Attention weight dtype/fidelity swept separately from MLP | `pcc/policy_sweep.json` and `perf/candidates.json` §"Precision / fidelity policy" — `bfp8_attn_bfp4_*` isolate the two groups |
| BFP4/LoFi MLP trial before lower-priority prefill advice | `work_log.md` §3, done first; prefill program-config work is §5 |
| Shard specs and core grids divide the tensor dimensions cleanly | `work_log.md` §2 and §4 — every grid is derived from `gcd` of the tiled dimensions, and an illegal one is an explicit `ValueError` |
| DRAM-sharded decode matmuls | `DRAM Sharded = True` on every dominant decode row |
| Collective topology minimized / fused matmul-CCL / persistent CCL buffers | *n/a*, no collectives |
| MoE routed active-expert path with `ttnn.sparse_matmul` | *n/a*, dense SwiGLU |
| LM head, sampling, logits movement, token feedback | *n/a*, model-level; the softcap and `output_multiplier` are applied by `MuseGlimmerForConditionalGeneration.forward`, not by the decoder layer |
| Reduced-precision experiments on real weights and real activations | `pcc/policy_sweep.json` — six policies x two weight sources x two layer kinds; plus `pcc/length_dependence.json` for the context-length axis |
| Performance accounting reconciled (roofline, device, end-to-end from the same run) | `perf/accounting.json`, generated by `render_optimized_accounting.py` |
| Batch capability preserved | batch 1 is the optimized target; batch 4 and 32 correctness at BFP8, batch 4 at the shipped BFP4 policy on real weights |
| Functional checks still pass against the optimized path | `logs/full_suite.log` (count in the *Tests* section, generated) |
| PCC at the functional bar for every layer kind | `pcc/pcc_results.json`; the gating minimum and the per-test bars are tabulated in `evidence_tables.md` |
| Paged KV cache and warmed trace replay still correct | `test_paged_prefill_decode_pcc`, `test_page_block_sizes`, `test_traced_decode_pcc`, `test_ragged_slots_and_current_positions` |
| Runtime fallback audit clean | *Runtime fallback audit* above, with the two on-device layout conversions named and costed |
| Stress / repeated-run coverage | `test_stress_repeated_prefill_decode` — three cycles of prefill + 8 decode steps on real weights, outputs bit-identical across cycles |
| Warmed prefill and traced warmed decode before/after | the headline table, generated from the two stages' `perf_summary.json` |
| `tt-perf-report` output with advice, and its conclusions | `tracy/<kind>/*_perf_report.summary.txt`; disposition table in `work_log.md` §7 |
| Watcher clean, separate from the profiler | `watcher/WATCHER_SUMMARY.md` |

## Limitations and follow-ups

* **Single device by design.** Stage 02 optimizes per-device performance on a `(1, 1)`
  mesh; the target `(1, 4)` P300x2 mesh is stage 03 (`$multichip`). Nothing in the layer
  hard-codes one device: weights are uploaded with `ReplicateTensorToMesh` and no CCL is
  needed yet. The DRAM-sharded weight memory configs and the residual/working shard grids
  are all derived from the live device's DRAM and compute grids, so they follow the mesh.
* **The 12-core DRAM-sharded matmul worker set** is the largest remaining decode
  opportunity (see *Performance accounting*). It is an op-level limitation.
* **Prefill is not traced.** The number of chunks depends on the prompt length, so a single
  captured trace would not cover the contract. The residual ~7% host term is per-chunk
  dispatch.
* **Continued prefill on sliding layers** is still unavailable — the same op-contract
  blocker as stage 01, unchanged by this stage.
* **The two QK RMSNorms are the one unsharded norm pair**, 1 core and ~4 µs each (8 µs of the
  45 µs total norm time, 0.8% of the decode step). They consume the height-sharded
  per-(user, head) layout `nlp_create_qkv_heads_decode` produces, and `ttnn.rms_norm` has no
  sharded program config for that layout, so the layer converts to L1 interleaved and back at
  that boundary. Fixing it needs either a sharded norm that accepts a height-sharded
  head-major tensor or a fused QK-norm in the head-creation op.
* **Real-weight coverage at the shipped BFP4 policy is batch 1 and batch 4, block size 64.**
  Batch 32 and block sizes 32/128 are covered at `bfp8_all_lofi` (the structural policy, same
  code path) rather than at BFP4. Those paths are precision-independent — they change page-table
  addressing and shard grids, not arithmetic — but the combination is not measured at BFP4.
* **`$datatype-sweep` still owns the final accuracy/performance frontier.** This stage
  swept six named policies and selected on real-weight PCC plus traced decode latency; it
  did not explore the full Pareto front, per-layer exceptions, or activation dtype below
  BF16.
