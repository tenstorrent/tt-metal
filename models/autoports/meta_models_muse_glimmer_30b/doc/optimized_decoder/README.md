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
| warmed prefill, 4096 tokens, sliding_rope | 58.79 ms | **25.76 ms** | 2.28x |
| warmed prefill, 4096 tokens, full_nope | 59.12 ms | **25.09 ms** | 2.36x |
| warmed prefill, 8192 tokens, sliding_rope | 97.65 ms | **54.81 ms** | 1.78x |
| warmed prefill, 8192 tokens, full_nope | 97.55 ms | **53.54 ms** | 1.82x |
| traced warmed decode, batch 1, context 4096, sliding_rope | 3.242 ms | **1.064 ms** | 3.05x |
| traced warmed decode, batch 1, context 4096, full_nope | 3.223 ms | **1.045 ms** | 3.08x |
| traced warmed decode, batch 32, context 4096, sliding_rope | 3.398 ms (9.4 k tok/s) | **1.161 ms (27.6 k tok/s)** | 2.93x |
| traced warmed decode, batch 32, context 4096, full_nope | 3.565 ms (9.0 k tok/s) | **1.272 ms (25.2 k tok/s)** | 2.80x |

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
   step. They now run on 16 cores at ~9 µs each: 37 µs total.
4. **Explicit 2D prefill program configs** over a large Blackhole grid, with the sequence
   folded into 512-row blocks so the per-core output block fits L1. The functional stage's
   prefill matmuls were all marked `SLOW` with `in0_block_w=1` and ran at 37-50% of the
   HiFi2 FLOP peak; they now run at `in0_block_w=8`, output subblocks up to `1x8`, and
   59-71% of the LoFi peak — and the two dominant SwiGLU projections are no longer marked
   `SLOW` at all, they are `FLOP`-bound at 68.8% and 71.4%.
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

The dominant-matmul table in [`evidence_tables.md`](evidence_tables.md) is read out of the
committed `tt-perf-report` rows and checked against this table by
[`../../scripts/render_optimized_evidence.py`](../../scripts/render_optimized_evidence.py),
which exits non-zero on a mismatch. A policy object is intent; the profiler row is the
evidence.

### Real weights versus synthetic weights

`bfp4_all` is selected on **real checkpoint weights**, where it measures prefill PCC
0.9986 / 0.9990 and decode 0.9970 / 0.9984 (sliding_rope / full_nope) — above the 0.995
bar with margin. On **synthetic random weights at the real per-tensor scales** the same
code measures 0.961-0.970, because BFP4's shared-exponent block quantisation is far lossier
on i.i.d. Gaussian blocks than on the checkpoint's own strongly correlated ones. That is
the `$optimize` skill's OPT-012 case: a weights discrepancy, not an implementation defect.

The suite is built so this cannot be waved through:

* every structural test (page tables, block sizes, non-aligned lengths, ragged slots,
  batching, window enforcement, continued prefill, short prefills) runs the **same code
  path** at `bfp8_all_lofi` and asserts the full 0.995 bar on synthetic weights, so a
  page-table, chunking or layout bug still fails the suite at full strength;
* acceptance for the shipped BFP4 policy comes from real-weight tests that reproduce the
  disputed contracts — `test_real_weights_prefill_decode`,
  `test_real_weights_non_aligned_and_traced` (non-aligned length, paged prefill→decode
  transition, 3-step traced replay) and `test_stress_repeated_prefill_decode` — all at
  0.995;
* `test_synthetic_weight_precision_discrepancy` records the gap itself, for every policy,
  into the PCC artifact. Nothing is marked xfail.

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

Device-time share: matmuls 82.4%, elementwise 4.2%, SDPA 3.6%, norms 4.3%, everything
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
| decode b1 ctx 4096 sliding_rope | 0.536 ms (DRAM) | 1.050 ms | 1.064 ms | 1.96x | 0.014 ms (1.3%) |
| decode b1 ctx 4096 full_nope | 0.536 ms (DRAM) | 1.035 ms | 1.045 ms | 1.93x | 0.010 ms (1.0%) |
| prefill 8192 sliding_rope | 13.04 ms (110-core LoFi) | 51.04 ms | 54.81 ms | 3.92x | 3.77 ms (6.9%) |
| prefill 8192 full_nope | 13.04 ms (110-core LoFi) | 48.91 ms | 53.54 ms | 3.75x | 4.63 ms (8.7%) |

Full derivation in [`perf/accounting.json`](perf/accounting.json). The decode roofline is
272.2 MB of BFP4 weights plus 2.2 MB of BFP8 KV cache at the 512 GB/s peak the committed
`tt-perf-report` DRAM percentages imply.

Named limitations, in the order they bound the result:

* **Decode sits at 1.96x the DRAM roofline because the DRAM-sharded matmul fixes its
  worker set to one core per DRAM bank — 12 on this p300c.** The committed rows show
  those 12 workers simultaneously at ~53% of their DRAM share *and* ~53% of their LoFi FLOP
  share, so neither is saturated and the gap is read/compute overlap on a 12-core worker
  set, not a missing knob. A wider worker set is not expressible: the op derives its
  workers itself, and the 104-core interleaved alternative measured 32% slower at the same
  dtype. This is the single largest remaining decode opportunity and it needs an op-level
  change, not a config.
* **Decode end-to-end is 1.3% above device time**, so the traced decode loop carries no
  material host term. There is nothing left to remove there.
* **Prefill matmuls run on 64 of 110 cores.** `grid_x` is pinned to the 8 DRAM banks
  because a 2D multicast matmul with DRAM width-sharded weights returns NaN for any other
  width (below), and `grid_y` above 8 measured 48% slower. Against the reachable 64-core
  LoFi peak the prefill matmul roofline is 22.4 ms and the measured matmul time is 33.4 ms
  (67% of it); the 13.0 ms figure in the table is the unreachable 110-core roofline.
* **Prefill end-to-end is 7-9% above device time**, down from ~20% in the functional stage.
  What remains is per-chunk host dispatch. Prefill is not traced because the chunk count
  depends on the prompt length.
* **Non-matmul prefill time is 34% of the window**: SDPA 13.1%, elementwise 9.3%, norms
  8.5%. Prefill activations are DRAM interleaved by design, so the norms are not sharded.

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

97 tests, all passing ([`logs/full_suite.log`](logs/full_suite.log)):

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

`test_optimized_beats_functional_topology` asserts from `config_summary()` that the shipped
configuration really is DRAM-sharded, L1-width-sharded, BFP4 and BFP8-cached — a cheap
guard against a future edit quietly reverting the stage.

## Runtime fallback audit

Unchanged from stage 01 and re-run against this layer: `test_no_runtime_host_fallback` runs
one prefill and one decode inside a `torch.overrides.TorchFunctionMode` that raises on any
torch op, with `ttnn.from_torch` / `ttnn.to_torch` / `ttnn.as_tensor` replaced by
tripwires; `test_source_has_no_runtime_torch` walks the module AST and fails if `torch` or
a ttnn host-transfer entry point appears outside `from_state_dict` and `allocate_kv_cache`.
No `tilize`/`untilize`, `to_torch`/`from_torch` or host fallback appears in the committed
decode or prefill perf reports. The only layout ops in the measured decode path are the two
`Reshard`s at the SwiGLU working-shard boundary and the `InterleavedToSharded` /
`ShardedToInterleaved` pairs the head-creation and RoPE helpers require at their API
boundaries; both are named in the op-family table with their cost. The one `Slice` in the
decode window is the head-concat batch trim, not a fallback.

## Watcher

`TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 TT_METAL_WATCHER_DISABLE_ETH=1` over 52 of
the 97 tests, both layer kinds, including the real-weight and stress tests: **52 passed,
zero watcher-detected errors**, minimum stack headroom 1312 bytes free. See
[`watcher/WATCHER_SUMMARY.md`](watcher/WATCHER_SUMMARY.md). Watcher and profiler runs are
kept separate, as `$tt-device-usage` requires.

## Capability and context contract

No reduction. [`../context_contract.json`](../context_contract.json) is updated for the
BFP8 KV cache, which **halves** the cache footprint: 512 B per token per layer instead of
1024 B, i.e. 64 MiB instead of 128 MiB at the full 131072-token context for batch 1, and
2 GiB instead of 4 GiB at batch 32. `current_supported_context` stays at the full
HF-advertised 131072. Batch is still capped at 55 users by the decode SDPA's
one-core-per-(user, KV head) requirement on the 110-core grid; 32 is tested.

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
* **`$datatype-sweep` still owns the final accuracy/performance frontier.** This stage
  swept six named policies and selected on real-weight PCC plus traced decode latency; it
  did not explore the full Pareto front, per-layer exceptions, or activation dtype below
  BF16.
