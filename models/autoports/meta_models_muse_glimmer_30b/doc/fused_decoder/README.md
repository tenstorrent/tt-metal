# Fused decoder — `meta-models/Muse-Glimmer-30B`

A numerically-equivalent, faster op graph for the
[functional decoder](../functional_decoder/README.md): same public contract,
same paged prefill/decode semantics, same 131072-token capability, fewer and
larger device ops.

| item | value |
| --- | --- |
| implementation | `models/autoports/meta_models_muse_glimmer_30b/tt/fused_decoder.py` |
| tests | `models/autoports/meta_models_muse_glimmer_30b/tests/test_fused_decoder.py` |
| baseline | `models/autoports/meta_models_muse_glimmer_30b/tt/functional_decoder.py` |
| candidate rewrites that lost | `doc/fused_decoder/bench/variants.py`, `logs/variant_sweep.log`, `logs/op_merge_probes.log` |
| device | 1 x Blackhole (`ttnn.MeshShape(1, 1)`, 11x10 compute grid) |
| dtypes | BF16 weights / activations / KV cache, TILE layout, DRAM interleaved — **unchanged** |
| acceptance bar | PCC >= 0.995 vs HF (the functional stage's bar), plus fused-vs-unfused >= 0.995 **and** no accuracy regression vs HF (strict in prefill, 2e-4 BF16 band in decode) |


## Result

Same model, same dtypes, same contract — measured with the Tracy device
profiler and `tt-perf-report --csv` (`Device Time`, microseconds), warmed and
signposted, in runs separate from the watcher run:

| kind | window | device ops / iter | device time / iter | speedup |
| --- | --- | --- | --- | --- |
| sliding | prefill, 8192 tokens (1 chunk) | 42 -> **24** | 101.23 -> **49.32 ms** | **2.05x** |
| full | prefill, 8192 tokens (1 chunk) | 24 -> **22** | 99.38 -> **47.98 ms** | **2.07x** |
| sliding | prefill, 16384 tokens (2 chunks) | 95 -> **61** | 214.37 -> **104.79 ms** | **2.05x** |
| full | prefill, 16384 tokens (2 chunks) | 51 -> **47** | 221.58 -> **111.04 ms** | **2.00x** |
| sliding | traced decode @ 2048 | 64 -> **44** | 3.163 -> **2.710 ms/token** | **1.17x** |
| sliding | traced decode @ 131071 | 64 -> **44** | 3.160 -> **2.710 ms/token** | **1.17x** |
| full | traced decode @ 2048 | 32 -> **34** | 3.080 -> **2.687 ms/token** | **1.15x** |
| full | traced decode @ 131071 | 32 -> **34** | 3.575 -> **3.179 ms/token** | **1.13x** |

And it is *more accurate* than the graph it replaces. All six head-to-head
prefill accuracy controls improve (+3.3e-4 to +1.2e-3); five of six decode
controls improve and one drifts by -4.6e-5. The worst HF-vs-TTNN check in the
whole suite goes 0.997422 (functional) -> 0.998152 (fused).

Two things drive that, and the stage separates them rather than claiming it all
for the topology:

* `minimal_matmul` is both faster **and** closer to an FP32 reference than
  `ttnn.linear` at the *same* math fidelity — a genuine like-for-like kernel
  win, worth +6e-4 to +8e-4 on the prefill controls at 4097 and 12345 tokens.
* every RMSNorm runs on a **higher-fidelity compute-kernel config** than the
  op's default, which is the one place this stage changes precision rather than
  topology (limitation 3). `logs/norm_fidelity_control.log` is the same graph
  with the norms on the op default: the 100-token prefill controls fall to
  -8e-6 and +0.0 there, so that uplift is worth +3.3e-4 to +3.8e-4 wherever no matmul
  kernel changes.

The `full` decode op count goes *up* by two: the sharded-residual rewrite adds
reshards that the NoPE layer's missing RoPE ops no longer offset. Fewer ops was
never the goal — the 0.39 ms/token it buys is.


## What changed

`FusedDecoder` subclasses `FunctionalDecoder`, so everything the fusing stage did
*not* touch — the paged prefill/decode contract, the internal 8192-token prefill
chunking, the sliding-window tail hand-off, the `qk_scale_factor` fold into the
SDPA scale, the centered-RMSNorm `1 + w` fold, the page-table row slicing, the
`ttnn.slice` aliasing guards — is literally the same code.

### Dedicated fused ops (highest priority)

1. **RoPE.** `slice, slice, neg, concat, mul, mul, add` per tensor (x Q and K,
   plus two `tilize`s for the cos/sin slice) collapses to one
   `ttnn.experimental.rotary_embedding_hf` per tensor. That op implements the
   HuggingFace `rotate_half` convention with `cat(freqs, freqs)` cos/sin — the
   exact math `MuseGlimmerTextAttention` spells out — so it needs no weight or
   table permutation. (`rotary_embedding_llama` and `rotary_embedding_llama_fused_qk`
   are the Meta odd/even-interleaved convention and would need both.)
2. **Dense projections** use `ttnn.experimental.minimal_matmul` at prefill row
   counts and stay on `ttnn.linear` at decode row counts. Same op, same
   compute-kernel config, different kernel: at the 8192-token internal prefill
   chunk it is 1.11-2.36x faster on every projection in this layer, and 1.5x
   *slower* at 32 rows. The threshold is 3072 rows, chosen so this *dispatch
   choice* is never slower than the functional baseline at any row count
   (limitation 7) — the layer as a whole also carries the norm-fidelity uplift,
   which costs 0.11-0.16 % of prefill at every length (limitation 3).
   This is the single biggest prefill win.
3. **Decode RMSNorm** runs the sharded multi-core `ttnn.rms_norm` program config
   on a `4x2` core grid instead of landing on one core. This is the single
   biggest decode win: the four hidden-size norms go 109.9-110.0 -> 12.3-13.3 us
   each (per-instance means over the committed capture's eight replays). 13/26/52/104
   -core *non-rectangular* grids are legal for this op and were built and
   measured too; all are slower (limitation 6).

### Graph rewrites

4. The **decode residual stream stays width-sharded in L1** across the whole
   layer, so the two residual adds are sharded element-wise ops and the four
   hidden-size norms consume and produce that layout directly.
5. **Prefill RoPE tables are stored pre-tilized**, and at `start_pos == 0` the
   persistent table is handed to the op as-is (`rotary_embedding_hf` only needs
   `cos_seq_len >= seq_len`), removing two `tilize`s and two `slice`s per chunk.
6. **Decode RoPE tables** are gathered straight into the height-sharded
   `[1, batch, 1, head_dim]` layout the decode-mode op reads, replacing the four
   `ttnn.repeat` broadcasts (and their untilize/tilize round trips) the
   functional layer needed to line cos/sin up with a plain `ttnn.mul`.
7. **Decode Q** stays height-sharded from `nlp_create_qkv_heads_decode` through
   RoPE into the SDPA kernel instead of round-tripping through DRAM, and the
   decode QKV projection writes **directly to L1** (removing the
   `CopyDeviceOperation` the tt-metal #16667 workaround used to cost).

### Op merging

8. `silu(gate) * up` -> `ttnn.mul(gate, up, input_tensor_a_activations=[SILU])`.
9. `heads * sigmoid(gate_proj(h))` ->
   `ttnn.mul(heads, gate, input_tensor_b_activations=[SIGMOID])`.
   The matmul's *pack-time* activation was tried first on **both** dense
   kernels and is worse on each — see "Rejected" below.
10. Prefill SDPA `q_chunk == k_chunk` moves 128 -> **256** at *both* call sites
    — the in-memory op (fastest at 7 of 9 swept lengths on `full`, 6 of 9 on
    `sliding`, and at the 8192-token internal chunk) and the paged
    `chunked_scaled_dot_product_attention` a `full` layer uses for every chunk
    after the first, where it is worth **1.59x** (36.204 -> 22.831 ms at
    `chunk_start_idx=8192`) and is the dominant op of any long-context prefill.
    See limitation 8.
11. Explicit `MinimalMatmulConfig` blocking on the two projection shapes that
    want one — `o_proj` -> `M16 K4 N8` (full 8192-row chunk only), MLP gate/up
    -> `M8 K4 N16` (every height). +2.80 % and +2.92 % of device kernel time on
    the op that is 65 % of prefill; `wqkv` and `mlp_down` keep the op's own
    default because they are faster on it. This one had to be decided on
    profiler device time: the gaps are 1-3 % and a host-side A/B of the shipped
    default against *itself* reports -0.5 % to -10.8 %. The other three shapes
    were swept the same way and keep the op's default.

## Correctness

`pytest models/autoports/meta_models_muse_glimmer_30b/tests/test_fused_decoder.py`

-> **94 passed** in 473 s (`test_results.xml`, `logs/full_test_run.log`).
`logs/pcc_summary.txt` holds **214 asserted PCC checks** — 202 HF-vs-TTNN (worst
**0.998152**, bar 0.995) and 12 fused-vs-unfused equivalence (worst 0.996797,
bar 0.995) — plus 12 accuracy controls comparing both graphs to the same HF
reference.

The functional decoder's worst HF-vs-TTNN check was 0.997422; the fused
decoder's is 0.998152, i.e. the accuracy floor moved *up*.

### The fused graph is asserted, not assumed

`test_fused_graph_uses_fused_ops` traces the TTNN ops a real prefill and a real
decode dispatch and asserts:

* `rotary_embedding_hf` **is** dispatched (on `sliding` layers) and is **not**
  dispatched on `full` (NoPE) layers;
* `minimal_matmul` **is** dispatched in prefill and is **not** in decode;
* the sharded-residual reshards **are** in decode and are **not** in prefill;
* `silu`, `sigmoid`, `repeat`, `to_layout`, `tilize`, `untilize`, `typecast`,
  `from_torch`, `to_torch` and `as_tensor` are dispatched **zero** times in
  either mode;
* and — the positive form — exactly the expected number of `ttnn.mul` calls
  carry an input activation (one SiLU + one sigmoid per prefill chunk, one of
  each in decode). Asserting only that `silu`/`sigmoid` are *absent* would also
  pass if the activations had been dropped altogether.

Without it, a silent regression to the functional graph would still pass every
PCC test in the file. It runs at `seq_len in {2048, 3000, 12345}`, so the multi-chunk
prefill paths (sliced RoPE tables, sliding tail carry, paged chunked SDPA) are
inside the trace, and the three lengths straddle the `_dense` crossover so the
dispatch rule itself is asserted, not assumed.

### Fused vs unfused equivalence

`test_fused_vs_functional_equivalence` builds both decoders from the same
weights, feeds them the same tensors and compares **on device** at
`seq_len in {100, 4097, 12345}` for both kinds, prefill and decode (100 is below
the `_dense` row threshold and 4097/12345 above it, so both dense-kernel
branches are compared). It asserts
two things:

1. fused vs *unfused* at the stage acceptance bar (0.995) — an HF-only
   comparison could hide a rewrite that is subtly worse, since both graphs are
   held to 0.995 against HF;
2. **no accuracy regression** against the HF reference: *strictly* in prefill,
   and within a documented BF16 re-association band in decode.

The second is the one with teeth. The two TTNN graphs differ from *each other*
by about the unfused graph's own BF16 error (~0.998), so demanding
"fused == unfused" would be demanding that the fused graph reproduce the
baseline's error. What it must not do is drift *away* from the reference:

| seq_len | kind | mode | unfused vs HF | fused vs HF | delta |
| --- | --- | --- | --- | --- | --- |
| 100 | sliding | prefill | 0.999102 | 0.999471 | +0.000369 |
| 100 | full | prefill | 0.999076 | 0.999409 | +0.000334 |
| 4097 | sliding | prefill | 0.998519 | 0.999511 | +0.000992 |
| 4097 | full | prefill | 0.998327 | 0.999465 | +0.001138 |
| 12345 | sliding | prefill | 0.998480 | 0.999498 | +0.001018 |
| 12345 | full | prefill | 0.998233 | 0.999423 | +0.001190 |
| 100 | sliding | decode | 0.999049 | 0.999457 | +0.000408 |
| 100 | full | decode | 0.998999 | 0.999302 | +0.000302 |
| 4097 | sliding | decode | 0.998843 | 0.998797 | **-0.000046** |
| 4097 | full | decode | 0.998342 | 0.998827 | +0.000485 |
| 12345 | sliding | decode | 0.997622 | 0.998957 | +0.001335 |
| 12345 | full | decode | 0.998569 | 0.998770 | +0.000201 |

**All six prefill comparisons improve**, by +3.3e-4 to +1.2e-3, so the prefill
tolerance is zero — that is what the evidence supports, and any future drift
should be re-examined rather than absorbed. Note the two regimes the table
covers: at 4097 and 12345 tokens prefill also *changes matmul kernel* (rows >=
3072) to a more accurate one at the same math fidelity, while at 100 tokens it
runs the baseline's kernel and improves purely from the RoPE, norm and
activation rewrites.

One of the six decode comparisons drifts, by -4.6e-5. Decode never changes
matmul kernel (a step is 32 rows), so apart from the norm-fidelity uplift
(limitation 3, which is *free* in decode) its rewrites only re-associate BF16
rounding and can land either side of the baseline's; the decode tolerance is
**2e-4**, which bounds that observed drift by about **4.3x**. It was 5e-4 —
a 1.15x margin — while the prefill per-head QK norms were still on the op's
default config: those norms write the Q and K that prefill stores in the paged
cache, so giving them the same uplift as every other norm made the decode a
step reads from measurably more accurate, and the worst drift fell 10x. The
suite's *worst* HF-vs-TTNN check moves up either way, 0.997422 (functional) ->
0.998152 (fused).

### Coverage inherited from the functional stage

Every correctness dimension the functional stage established is re-run against
the fused implementation, not assumed:

| dimension | fused test |
| --- | --- |
| 9 prefill lengths incl. non-aligned (1, 100, 2049, 4097, 8193, 12345) | `test_prefill_pcc` |
| decode past prompts of 100 / 2048 / 3000, 4 steps each | `test_decode_pcc` |
| FP32 HF control (not just BF16-vs-BF16) | `test_prefill_decode_pcc_vs_fp32_reference` |
| `max_seq_len == chunk == seq_len` RoPE-table aliasing regression | `test_prefill_seq_len_equals_max_and_chunk` |
| caller-chunked continuation prefill, 3 splits incl. sub-window tails | `test_continuation_prefill_pcc` |
| sliding continuation without its window raises | `test_continuation_prefill_requires_sliding_tail` |
| batch 4 / 13 / 32, ragged prompts straddling the window, ragged decode positions | `test_batched_prefill_decode_pcc` |
| multi-chunk prefill into a non-zero cache slot | `test_multi_chunk_prefill_nonzero_user` |
| full 131072 context, prefill (last + interior rows) and decode at 131071 | `test_full_context_{prefill_tail,decode}_pcc` |
| released bf16 checkpoint | `test_real_weights_prefill_decode_pcc` |
| bit-identical repeated prefill and decode | `test_determinism_repeated_inputs` |
| no torch / host round-trip in a measured pass | `test_no_host_fallback_in_forward` |
| traced decode PCC from the *replay*, and one trace across 3 positions | `test_traced_decode_pcc`, `test_traced_decode_advances_positions` |
| 64-step traced-decode soak | `test_repeated_run_stress` |
| multi-chunk prefill at a `max_seq_len` whose page count is awkward (12416 = 194 blocks) | `test_multi_chunk_prefill_page_table_bound` |
| both prefill SDPA chunk sizes, and the two constraints that shrink the paged one | `test_prefill_sdpa_chunk_sizes` |
| the decode sharded-norm config for every allowed batch | `test_decode_norm_config_shapes` |

`batch=13` is still the interesting one: 13 is prime and larger than the 11-wide
grid, so it has no `batch`-core rectangle and takes the decode head-concat's
shape-agnostic fallback — and it is also a batch that is neither 1 nor
tile-aligned for the new sharded RoPE gather and width-sharded norms.

### The eight lowest HF-vs-TTNN PCCs

| PCC | check |
| --- | --- |
| 0.998152 | fused decode[full] prompt=3000 pos=3002 |
| 0.998213 | fused decode[full] prompt=2048 pos=2051 |
| 0.998233 | fused traced decode replay[sliding] pos=1025 |
| 0.998264 | fused decode[sliding] prompt=2048 pos=2048 |
| 0.998281 | fused traced decode replay[sliding] pos=1024 |
| 0.998333 | fused stress decode[full] step=32 pos=1056 |
| 0.998362 | fused decode[full] prompt=2048 pos=2049 |
| 0.998371 | fused decode[full] prompt=3000 pos=3003 |

All eight are decode, where the fused graph's accuracy gain is smallest: decode
keeps `ttnn.linear` (a decode step is 32 rows, below the `minimal_matmul`
crossover), so only the RoPE, norm and activation rewrites apply there. Across
the whole suite decode spans 0.998152-0.999693 and prefill — which switches
matmul kernel at and above 3072 rows — spans 0.998729-0.999786 (that floor is
`fused prefill[full] batch=32 user=28 seq_len=3036` in `logs/pcc_summary.txt`).

## Performance

Warmed, signposted windows, profiled with Tracy in **separate runs from the
watcher run**. Column: `Device Time` from the `tt-perf-report --csv` output, in
microseconds. Decode windows use **8 trace replays**
(`MG_PERF_DECODE_ITERS=8`), matching the functional stage exactly so the
before/after numbers compare like with like — the functional stage established
that 32 replays overflow the profiler's DRAM marker buffer and silently
under-count.

Integrity check on all ten committed captures — the eight fused windows plus the
two multi-chunk functional baselines (must be 0 everywhere, and every op-code
count in the filtered CSV must divide by the replay count):

```bash
grep -c "markers were dropped" doc/fused_decoder/logs/tracy_*.log   # all 0
```

### Before / after

| kind | mode | context | ops/iter | device time / iter | incl. op-to-op gaps | vs functional |
| --- | --- | --- | --- | --- | --- | --- |
| sliding | prefill, 8192 tokens (1 chunk), batch 1 | — | 42 -> 24 | 101.23 -> **49.32 ms** | 101.26 -> 49.34 ms | **2.05x** |
| full | prefill, 8192 tokens (1 chunk), batch 1 | — | 24 -> 22 | 99.38 -> **47.98 ms** | 99.39 -> 47.99 ms | **2.07x** |
| sliding | prefill, 16384 tokens (2 chunks), batch 1 | — | 95 -> 61 | 214.37 -> **104.79 ms** | 216.59 -> 106.62 ms | **2.05x** |
| full | prefill, 16384 tokens (2 chunks), batch 1 | — | 51 -> 47 | 221.58 -> **111.04 ms** | 221.61 -> 111.07 ms | **2.00x** |
| sliding | traced decode, batch 1 | 2048 | 64 -> 44 | 3.163 -> **2.710 ms/token** | 3.226 -> 2.760 | **1.17x** |
| sliding | traced decode, batch 1 | 131071 | 64 -> 44 | 3.160 -> **2.710 ms/token** | 3.223 -> 2.759 | **1.17x** |
| full | traced decode, batch 1 | 2048 | 32 -> 34 | 3.080 -> **2.687 ms/token** | 3.114 -> 2.722 | **1.15x** |
| full | traced decode, batch 1 | 131071 | 32 -> 34 | 3.575 -> **3.179 ms/token** | 3.608 -> 3.214 | **1.13x** |

At 8192 tokens per layer that is 166.1 k tok/s of layer prefill throughput for
`sliding` (was 80.9 k) and 170.8 k for `full` (was 82.4 k).

The 16384-token rows are the *multi-chunk* regime — the one a long prompt
actually runs, and the only one in which a `full` layer touches the paged
`chunked_scaled_dot_product_attention` at all. The `full` kind's two SDPA calls in that window go
45,771 -> 27,701 us — both are retuned (the in-memory chunk-0 call and the paged
one at `chunk_start_idx=8192`), and the paged call is the one the functional
layer left at 128. Both 16384 captures use the same signposted window and replay
count as the 8192 ones, and the baseline captures are committed alongside them
as `prefill_16384_baseline_*`.

### Where the prefill time goes now (sliding, 8192 tokens)

| share | op | x | device time | note |
| --- | --- | --- | --- | --- |
| 65.0 % | `MinimalMatmulDeviceOperation` | 6 | 32,067 us | 228.5-255.5 TFLOPs (was 95.7-214.1 on `ttnn.linear`) |
| 15.4 % | `SDPAOperation` | 1 | 7,600 us | `q_chunk == k_chunk == 256` (was 11,838 us at 128) |
| 8.6 % | `BinaryNgDeviceOperation` | 4 | 4,240 us | 2 residual adds + gating mul + SwiGLU mul (sigmoid/SiLU folded in) |
| 7.9 % | `LayerNormDeviceOperation` | 6 | 3,890 us | 110 cores, DRAM-bandwidth bound (and the one fidelity uplift, limitation 3) |
| 1.6 % | `RotaryEmbeddingHfDeviceOperation` | 2 | 780 us | was 2,355 us of primitive ops + 2 tilizes |
| 1.5 % | heads split/concat, paged fill, page-table slice | 5 | 742 us | |

In the 8192 window the six `MinimalMatmul` rows run at 228.5-255.5 TFLOPs on the
`sliding` kind and 228.6-255.5 on `full` (in the 16384 window, twelve rows,
228.3-255.5 and 228.9-255.7); the functional graph's six matmuls were 95.7-214.1
and 93.7-214.3, and `tt-perf-report` no longer marks any matmul row `SLOW`. Prefill is now genuinely compute-bound on the matmuls, and the next
lever on them is precision, not topology. The blocking itself has been taken as
far as this op allows: `MinimalMatmulConfig` was swept over the whole legal
`K_block` range (4 up to the full K-tile count, divisors and non-divisors) and
the `M_block`/`N_block` neighbourhood of every winner, on **device kernel time**.
All five projection shapes were swept. Two take a config — `o_proj` ->
`M16 K4 N8` at the full 8192-row chunk (+2.80 %) and the MLP gate/up ->
`M8 K4 N16` at every height (+2.92 % at 8192, +1.49 % at 6144, +0.92 % at 4096)
— and `wqkv`, the attention gate and `mlp_down` are fastest on the op's own
`M=K=N=8` default (best candidate -2.6 %, -0.0 % and -0.1 % respectively; the
attention gate's best candidate *is* the default, re-measuring to -0.01 %). Everything larger is a hard L1
stop: `K_block >= 20`, and every `M16 x N16` / `M16 K8` / `N24` variant, fail
with *"Statically allocated circular buffers on core range [0-0 - 10-9] grow to
1684352-8893312 B which is beyond max L1 size of 1572864 B"*
(`program.cpp:1722`). All 28 explicit 2D
`MatmulMultiCoreReuseMultiCastProgramConfig` grids hit the same budget
(`logs/prefill_matmul_probe.log`, `logs/prefill_matmul_kblock_*`).

### Where the decode time goes now (per token, sliding @ 2048 / full @ 131071)

| share | op | x | device time | note |
| --- | --- | --- | --- | --- |
| 93.2 % / 79.4 % | `MatmulDeviceOperation` | 6 | 2,526 us | weight-bandwidth bound at 383 GB/s |
| 2.2 % / 1.9 % | `LayerNormDeviceOperation` | 6 | 59 us | **was 447 us** — 4 hidden-size norms on 8 cores + 2 tiny QK norms |
| 1.4 % / 16.6 % | `SdpaDecodeDeviceOperation` | 1 | 37 / 529 us | the only op that scales with context |
| 0.9 % / 0.8 % | `BinaryNgDeviceOperation` | 4 | 24 us | |
| 2.4 % / 1.3 % | everything else | 27 / 17 | 64 / 42 us | head split/concat, RoPE gather+apply, paged update, reshards |

One decode step streams
`(6656*4608 + 6656*4096 + 4096*6656 + 3*6656*19968) * 2 B = 967,835,648 B`
of BF16 weights. At the 383 GB/s the matmuls actually achieve that is 2.526 ms,
i.e. **93 % of the fused decode step is the BF16 weight-streaming roofline**.
Everything else in the layer now costs 0.18 ms. Further decode gains have to
come from moving fewer bytes (weight dtype) or from a higher-efficiency matmul
config — both of which are the optimized-decoder stage's job, not fusing's.

### Artifacts

Per kind, under `tracy/<kind>/`:

* `prefill_ops.csv`, `decode_ops.csv.gz`, `decode_131071_ops.csv.gz` — raw Tracy
  ops CSVs copied from `generated/profiler/reports/<ts>/` (the decode ones are
  gzipped to stay under the repo's 500 KB file-size hook; `gunzip -k` them to
  re-run `tt-perf-report` on the raw capture)
* `*_perf_report.txt` — human-readable tt-perf-report tables
* `*_perf_report.csv` — filtered CSV for the signposted window
* `*_perf_report.console.log` — provenance for the `--csv` invocation
* `*_perf_report_stacked.{csv,png}` — tt-perf-report 1.2.8 stacked breakdown

The rejected matmul-activation variant's profile is kept for comparison in
`logs/rejected/prefill_perf_report_matmul_activation_{sliding,full}.txt`. That
capture predates the `minimal_matmul` rewrite (it shows the old
`MatmulDeviceOperation` graph at 96,052 us), which is why the same rejection was
re-tested on `minimal_matmul`'s own `fused_activation` in
`logs/prefill_matmul_probe.log`.

Under `tracy/probes/` are the four raw ops CSVs behind the two device-time
probes that are not part of the eight windows — the five
`MinimalMatmulConfig` rounds (`kblock_device{,2,3,4,5}_ops.csv`, rounds 1 and 4
gzipped for the file-size hook) and the decode sharded-output probe
(`decode_sharded_out_ops.csv`). Each probe prints a `GROUP <n> <label>` line per
run of consecutive device ops, and `bench/summarize_device_probe.py` slices the
CSV back apart with them, so every percentage quoted from those probes is
regenerable:

```bash
# .gz is read transparently: the two largest console logs and two ops CSVs are
# committed gzipped because the repo rejects files over 500 KB
python bench/summarize_device_probe.py logs/prefill_matmul_kblock_device2.log.gz \
    tracy/probes/kblock_device2_ops.csv                      # -> logs/..._summary.txt
python bench/summarize_device_probe.py logs/decode_sharded_out_probe.log \
    tracy/probes/decode_sharded_out_ops.csv \
    --op-code MatmulDeviceOperation,InterleavedToSharded
```

Driver: `bench/run_tracy.sh` (all eight fused windows and the two multi-chunk
functional baselines, one device job at a time, plus the dropped-marker
integrity check).

The **8192-token and decode baselines are the functional stage's own committed
captures** (`../functional_decoder/tracy/`), not re-taken here; only the two
16384 baselines are re-captured by this stage's chain, because the functional
test takes that length from an env var. The two agree: this stage's own
pre-`minimal_matmul` capture
(`logs/rejected/prefill_perf_report_matmul_activation_sliding.txt`) reproduces
the functional baseline's matmul rows to within 5 us (2,441 vs 2,440; 4,672 vs
4,667; 21,323 vs 21,322), and the freshly captured 16384 baselines give the same
2.05x / 2.00x ratios as the inherited 8192 ones.

## Rejected rewrites

Split the way the work log does: the first group was **implemented and measured
on device** (candidate classes in `bench/variants.py`, numbers in
`logs/variant_sweep.log`, min of 3 rounds with the per-round spread printed —
decode reproduces to +/- 0.001 ms/token, prefill to about +/- 2 %); the second
group is blocked by an **exact op contract**, not by a measurement. Full
reasoning in [`work_log.md`](work_log.md) section 4.

### Measured and rejected

| candidate | verdict (sliding: prefill / decode; shipped = 65.24 ms / 2.734 ms per token) |
| --- | --- |
| `ttnn.linear(..., activation="silu"/"sigmoid")` | **worse**: does not actually fuse on this build — a separate 2,128 us `UnaryDeviceOperation` still runs alongside the activation-carrying matmul — and the same shape measured in isolation goes 23.964 -> 26.461 ms |
| `minimal_matmul(..., fused_activation=SILU)` | the same idea on the kernel prefill actually uses: it *does* fuse, but costs 12.101 vs 10.283 ms on the MLP gate shape |
| shared-LHS packing of `wqkv` + attention gate | 65.78 / 2.738 — the decision rests on traced decode, which reproduces to +-0.001 ms/token and is a consistent loss on both kinds (2.738 vs 2.734 sliding, 2.709 vs 2.708 full); the prefill wall-clock A/B has a +-2 % round spread, so it only says "not faster". The slices cost what the dispatch saves, and decode matmuls are weight-bandwidth bound so packing moves no bytes |
| shared-LHS packing of the MLP gate/up | 66.78 / 2.757 — same reason, and the slices are on a 19968-wide tensor |
| `ttnn.swiglu` | 68.38 / 2.767 — a composite (2 slices + swish + multiply) |
| `minimal_matmul(fuse_swiglu=True)` | faster in prefill (24.682 vs 25.718 ms at 8192 rows) but 84 % **slower** in decode (2.593 vs 1.406 ms at 32 rows), so the layer would need both weight layouts: +531 MB per layer, +27 GB over 52 layers |
| `paged_fused_update_cache` + the V reshard this layout forces | 64.91 / 2.737 sliding and 64.93 / 2.705 full — +0.003 ms/token on one kind and -0.003 on the other, i.e. ~0.1 % and sign-flipping between the two layer kinds, so the reshard costs what the dispatch saves. A per-kind selection was considered and rejected: the fused op swaps two 3.58 us cache dispatches for one write plus a ~1.4 us reshard, so it removes no device work and the delta is dispatch overlap — not worth forking the cache write by layer kind (`logs/kv_update_ab.log`, 5 rounds x 256 iters). See the contract row below for why the reshard is unavoidable here |
| explicit 2D matmul program configs | 28 attempts (7 rectangles x 4 projections), all exceed the L1 circular-buffer budget |
| explicit `MinimalMatmulConfig` on `wqkv`, the attention gate and `mlp_down` | all three are fastest on the op's own default; the best candidate is 2.6 % worse on `wqkv`, 0.01 % on the attention gate (where the best candidate is the default itself) and 0.1 % on `mlp_down`, so those three pass no `config=` (the other two shapes do take one — see the prefill table) |
| `o_proj`'s `M16 K4 N8` on tail chunks | wins 2.80 % at the full 8192-row chunk but loses 6.3 % at 4096 and 6.0 % at 6144, so the entry carries a minimum row count instead of applying everywhere (the MLP entry was measured at all four heights — +2.61/+0.92/+1.49/+2.92 % at 3072/4096/6144/8192 — and needs no such gate) |
| writing the decode `o_proj` / `mlp_down` output straight into the sharded residual | `ttnn.linear` takes the width-sharded memory config but keeps its own 110-core grid, so the sharded norm rejects the result (`shard_spec_validation.cpp:46`); forcing the matmul onto the norm's 4x2 grid works but costs 222.1 us against the shipped pair's 149.4 (o_proj) and 1071.5 against 700.9 (mlp_down) — the shipped pair is 32.8 % / 34.6 % faster, and the sharded matmul alone is ~1.5x the matmul it replaces — both figures already including the ~2.5 us reshard the merge would have removed |
| non-rectangular 13/26/52/104-core sharded decode norm | legal and correct, 25.3-57.6 us vs 22.8 us at 8 cores |
| per-projection `_dense` thresholds instead of one | -0.17/-0.28 ms at 512/1024 rows but +0.21/+0.24 at 1536/2048, and it forfeits the never-slower-than-baseline property |
| SDPA chunk 320 / 384 / 512 | 384/512 overflow L1; 320 wins only in a two-point band around 4k and loses at 7 of the 9 swept lengths, including the 8192-token internal chunk |
| `minimal_matmul`'s own (more accurate) compute-kernel config | slower per 8192-token prefill by 2.3-2.5 ms in the in-graph probe and ~1.0 ms summed over the six dispatches in the isolated sweep; precision policy is the next stage's |

### Rejected on an exact op contract

| candidate | the contract that blocks it |
| --- | --- |
| `paged_fused_update_cache` **without** a reshard | needs K and V on disjoint cores; `nlp_create_qkv_heads_decode` emits both on Q's grid, and its `overlap_qk_coregrid=False` mode is *dropped* by the frontend for an interleaved input (`nlp_create_qkv_heads_decode.cpp:23`) and constrained by the device op to a width-sharded QKV with `head_dim % shard_width == 0` (`..._device_operation.cpp:56-72`) — measured: with this layer's L1-interleaved QKV the flag changes nothing (identical Q/K/V grids at batch 1/4/32) and the fused write is rejected at every batch, while with a WIDTH_SHARDED input and a shard width dividing `head_dim` K and V *do* come out disjoint. A width-sharded decode QKV is the DRAM-sharded matmul, i.e. the next stage, and would also bring `num_cores >= 2*num_users`, which is not binding here (the op already caps `num_users` at 32 and this grid has 110 cores) but is one more constraint the next stage inherits. `logs/kv_coregrid_probe.log` |
| `ttnn.rms_norm(residual_input_tensor=...)` | computes `norm(x + residual)`; Muse-Glimmer is post-norm, `residual + norm(x)`. The one add-then-norm site that does match is still a no-win because the sum is consumed again by the final residual add |
| `rotary_embedding_llama` / `_fused_qk` | Meta odd/even-interleaved convention; would need both the Q/K weight columns and the cos/sin tables permuted. `rotary_embedding_hf` is the same math with no permutation |
| `ttnn.transformer.concatenate_heads` | literally calls `ttnn::prim::nlp_concat_heads` and squeezes (`concatenate_heads.cpp:45-47`), i.e. the same op the layer already uses |
| `split_query_key_value_and_split_heads` | not GQA-capable; `nlp_create_qkv_heads` already in use |
| `ttnn.experimental.matmul_decode` | a dedicated decode matmul for the op class that is 93 % of the decode step, but it requires **both** operands `WIDTH_SHARDED` (`matmul_decode_device_operation.cpp:32-39`), i.e. the weights resident in L1. This layer streams 968 MB of BF16 weights per step from DRAM against 1.5 MB of L1 per core, so it is unreachable until the weights shrink — one line to re-check once they do |
| distributed norms (`fused_rms_minimal`, `rms_norm_pre/post_all_gather`, `dit_fused_distributed_rmsnorm`) | require a multi-device mesh and a global semaphore |
| `dit_rms_norm_unary_fused` | fuses `unary(rms_norm(x))`; no norm here is followed by a unary |

## Capability contract

Unchanged. `doc/context_contract.json` still records
`current_supported_context = 131072` and `capability_reduction: "none"`; the
fused decoder is tested at 131072 in both prefill and decode for both layer
kinds, at the non-aligned 130073, and at batch 4/13/32.

The only memory the fusing added is the pre-tilized prefill RoPE tables:
`2 (cos, sin) x 131072 x 128 x 2 B = 67 MB` per **sliding** layer, on top of the
67 MB row-major tables the functional layer already carried for the decode
gather (both layouts are needed — decode gathers per-user rows from a ROW_MAJOR
table with `ttnn.embedding`, prefill hands a TILE table to
`rotary_embedding_hf`). `full` (NoPE) layers carry neither. Against a 968 MB
per-layer weight footprint and a 32 GB part this changes no capacity limit, and
it is why `minimal_matmul(fuse_swiglu=True)` — which would have needed a second
**531 MB** copy of the gate/up weights — was rejected.

## Limitations and known issues

1. **Decode is at the BF16 weight-streaming roofline.** 93 % of the fused decode
   step is six matmuls streaming 968 MB of BF16 weights at 383 GB/s (75 % of
   peak). No graph rewrite can move that; the remaining levers are weight dtype
   and matmul config, i.e. the optimized-decoder stage.
2. **Prefill is compute-bound on `minimal_matmul`.** 65 % of prefill is six
   `MinimalMatmul` ops at 228.5-255.5 TFLOPs on `sliding` and 228.6-255.5 on
   `full` (the functional graph's six matmuls were 95.7-214.1 / 93.7-214.3). `tt-perf-report` no longer marks any matmul `SLOW`. The
   next lever is again precision.
3. **`minimal_matmul` is not bit-identical to `ttnn.linear`** — it accumulates
   differently (and, at equal math fidelity, better). All PCC evidence in this
   stage is against the HF reference and against the unfused graph; nothing here
   claims bit-equality with the functional decoder. Determinism *within* the
   fused implementation is asserted (`test_determinism_repeated_inputs`,
   `torch.equal` over three repeats of prefill and of decode).
   `minimal_matmul`'s *own* default compute-kernel config is more accurate again
   (PCC 0.999994 vs 0.999947 against FP32) but costs 2.3-2.5 ms on an
   8192-token prefill; the shipped config pins the same HiFi2 policy
   `ttnn.linear` uses, so the *matmul* before/after is topology only and that
   precision choice is left to the stage that owns it
   (`logs/dense_compute_kernel_probe.log`).

   **The RMSNorms are the one exception, and the only fidelity change in this
   stage.** (`rope_compute_kernel_config` also hands `rotary_embedding_hf` an
   explicit config, but it is inert: its `math_fidelity` and
   `fp32_dest_acc_en` equal that op's own default
   (`rotary_embedding_hf.cpp:46`), and the one field that differs,
   `math_approx_mode`, is unpacked by both of that op's program factories and
   never reaches the `ComputeDescriptor` — so it has no numeric or timing
   effect.) They run on `HiFi4 / math_approx_mode=False /
   fp32_dest_acc_en=True / packer_l1_acc=True`, where `ttnn.rms_norm`'s own
   default — what the functional layer used, since it passed no config — is
   `HiFi4 / approx=True / fp32_dest_acc_en=False / packer_l1_acc=False`
   (`rmsnorm.cpp:16-20`). Measured against a float64 reference
   (`bench/norm_fidelity_probe.py`): the prefill norm goes 978.27 -> 991.78 us
   with max relative error 6.5e-2 -> 4.2e-3, and the sharded decode norm is
   *faster* as well as more accurate, 15.53 -> 14.92 us and 1.0e-2 -> 4.8e-3.
   So it costs ~54 us across the four hidden-size norms (13.5 us each, measured)
   and of order 25 us across the two much smaller per-head QK norms — the
   prefill `LayerNorm` total moved ~3,868 -> 3,890 us when those two were
   included, which is at the edge of the run-to-run spread, and only the
   post-fix capture is committed — so **0.11-0.16 %** of a 49,318 us prefill
   window, nothing in decode, and buys a 15x smaller worst-case error on the
   op feeding every matmul. It reaches *every* `ttnn.rms_norm` dispatch, six per
   prefill and six per decode, which
   `test_every_norm_takes_the_uplifted_config` asserts by patching the op
   rather than by reading an attribute — the prefill per-head QK norms reach it
   through the inherited call path, and an earlier revision left exactly those
   two on the default while claiming otherwise.
   It is worth +3.3e-4 to +3.8e-4 of the accuracy gain wherever no matmul kernel changes
   (`logs/norm_fidelity_control.log`), and it improves *decode* indirectly by
   more than it improves prefill: the per-head QK norms write the Q and K that
   prefill stores in the paged cache, so a decode step reads a more accurate
   cache. Extending the uplift to those two took the worst decode accuracy
   control from -4.3e-4 to **-4.6e-5**, which is what let the decode tolerance
   tighten from 5e-4 to 2e-4. Pinned by
   `test_norm_compute_kernel_config_is_the_documented_uplift` and
   `test_every_norm_takes_the_uplifted_config`; called out here because "the
   fused graph is more accurate" would otherwise read as a pure topology
   result.
4. **The decode RoPE cos/sin gather costs ~19 us, 0.7 % of the step, in ops
   that are either invisible to the op trace or measured not worth merging.**
   The obvious peer merge — the two `ttnn.embedding` calls share one index
   tensor, so pack `cos` and `sin` into a single `[max_seq, 2*head_dim]` table
   and gather once — was built and measured: it produces **bit-identical**
   cos/sin (`torch.equal` on both) and is a **wash**, 14.08 vs 13.92 us,
   because the two width slices needed to split the halves apart again
   (5.23 us) cost more than the saved `embedding` + `transpose` (5.09 us:
   11.69 -> 6.60 us across those two ops).
   See `logs/decode_rope_gather_probe*` and
   `tracy/probes/decode_rope_gather_ops.csv`. Two
   `TilizeWithValPaddingDeviceOperation` (5.6 us each) live *inside*
   `ttnn.embedding`: they are its own ROW_MAJOR -> TILE conversion for a
   per-user positional gather, not a layout round trip the layer chose, and
   `models/common/modules/rope/rope_1d.py` has the same two.
   `test_fused_graph_uses_fused_ops` traps the *Python-level*
   `ttnn.tilize`/`to_layout`/`untilize` calls, which are zero — it cannot see
   inside another op's kernel, so this pair is disclosed rather than asserted.
   Two smaller decode items are recorded here rather than left silent: the
   `SliceDeviceOperation` that trims the head-concat output back to `batch`
   (1.99 us mean over the eight replays, 0.07 % of the step), and the fact that asking
   `nlp_create_qkv_heads_decode` for an interleaved output would remove two of
   the four QK-norm reshards (~1.1 us) — neither was pursued, both are under
   0.1 % of a decode step.
   The rest of the gather is two `EmbeddingsDeviceOperation` (1.1 us each), two
   `TransposeDeviceOperation` (2.1 us) and two
   `InterleavedToShardedDeviceOperation` (0.6 us each): the transpose is real data movement,
   not an identity permute — in TILE layout `[1,1,batch,d] -> [1,batch,1,d]`
   changes the tile grid — so it is not a missed permute/reshape rewrite.
   Separately, the two scale-less per-head QK norms still run on **one core**
   each (~3.75 us) plus four reshards: `ttnn.rms_norm` rejects height-sharded
   inputs (`layernorm_device_operation.cpp:166`), so they round-trip through L1
   interleaved. That is 0.35 % of the step and was left alone.
5. **The decode SDPA output is resharded rather than produced sharded.** Passing
   the head-concat shard config straight to
   `paged_scaled_dot_product_attention_decode` would remove one
   `InterleavedToShardedDeviceOperation` — the eight in a decode step span
   0.48-2.63 us — out of 2710, and
   only for batches that have a `batch`-core rectangle. Not taken.
6. **The sharded decode RMSNorm is capped at 8 cores by measurement, not by
   legality.** The core count must divide `6656/32 = 208 = 2^4 x 13` tiles, and
   13 has no rectangle on an 11-wide grid — but that is not a blocker: the
   sharded LayerNorm program factory explicitly accepts a *non-rectangular*
   `CoreRangeSet` that is a shard-order prefix of its bounding box
   (`layernorm_device_operation.cpp:185-215`), which always holds for a decode
   step. 13/26/52/104-core prefix grids were built with
   `ttnn.num_cores_to_corerangeset_in_subcoregrids` and measured: all legal, all
   correct (PCC 0.9999964), all **slower** — 25.3 / 28.0 / 37.7 / 57.6 us
   against 22.8 us at 8 cores, even at `subblock_w = 4`. A decode step is one
   tile-row, so past ~8 cores the per-core reduction and cross-core stats
   exchange cost more than the extra width parallelism buys.
   `logs/norm_shard_probe.log` has all 14 configurations.
7. **`minimal_matmul` is only used at or above 3072 rows.** `ttnn.linear`'s
   auto-selected program config is not monotone in M — on the MLP shapes it
   costs 2.85 ms at 2048 rows, 11.66 at 4096 and 8.98 at 6144 — so the crossover
   is a band, not a point, and it differs per projection: the widest
   projection, `mlp_gate_up` (two of the six dispatches), stays ahead of
   `minimal_matmul` all the way to 2048 rows, while the other four — including
   the other 19968-wide one, `mlp_down` — first cross over at 512 and then lose
   again at 1536-2048. The threshold is set at the first row count
   from which every measured point favours the fused kernel (per-chunk delta
   -0.82 ms at 3072, -20.40 at 4096, -4.38 at 6144, -43.51 at 8192), which also
   means **the fused layer is never slower than the functional baseline at any
   row count** — below 3072 it runs exactly the baseline's kernel. The cost is a
   0.17-0.28 ms per-chunk win at 512-1024 rows, which a per-projection threshold
   would capture but would give back as 0.21-0.24 ms at 1536-2048 rows while
   forfeiting the never-slower property (work log 3.1 B has the table). Both
   branches are PCC-tested (`test_prefill_pcc` covers 1/100/128/2048/2049 on the
   `ttnn.linear` branch and 4097/8192/8193/12345 on the `minimal_matmul` branch)
   and
   `test_fused_graph_uses_fused_ops` asserts which branch each prefill length
   takes. The dispatch cannot mis-fire in decode: a decode step is
   `ceil(batch/32)*32` rows and `nlp_create_qkv_heads_decode` needs one core per
   user; more to the point the op hard-caps `num_users` at 32
   (`..._device_operation.cpp:45-51`), so a decode step is always exactly one
   32-row tile.
   `logs/minimal_matmul_sweep.log` has all 5 x 11 measurements.

8. **The prefill SDPA chunk is one constant, seeded into both call sites.** The
   paged `chunked_scaled_dot_product_attention` a `full` layer uses for every
   chunk after the first can only *halve* it (that op additionally requires
   `chunk_start_idx % q_chunk_size == 0`), which is why 320 is unusable there
   at all — 8192 is not a multiple of it. For the in-memory op, 256 is fastest
   at 7 of the 9 swept lengths on the `full` kind and 6 of 9 on `sliding` (the
   ninth, 8224 sliding, goes to 320 by 0.1 %: 8.628 vs 8.636 ms), including the 8192-token internal prefill chunk (8.155
   vs 8.600 ms) and everything below 3008; 320 wins in a narrow band around 4k
   (4096: 2.426 vs 2.574 ms; 4128: 2.421 vs 2.588) and ties at 8224. A
   length-dependent rule was not taken: the band is two sample points wide, the
   win there is ~6 % of SDPA (under 1 % of prefill), and prefill is chunked at
   8192 so only a prompt whose *final* chunk lands in that band would see it.
   `logs/sdpa_chunk_sweep.log` has all 90 measurements.
9. **The `ttnn.embedding` cos/sin gather is measured against the committed
   perf report, not asserted by the op trace** — see limitation 4. Likewise the
   watcher run covers 18 of the 94 tests — both kinds' multi-chunk prefill,
   decode, continuation prefill, traced replay, batch 13 and batch 32, the
   non-zero cache slot, the awkward page-count prefill (the only case that
   exercises the *halved* paged-SDPA chunk), the graph audit, the norm-config
   shapes, the fused-vs-unfused comparison and the 64-step stress soak — but not
   the full-131072-context paths (a
   watcher-instrumented 131072-token prefill is minutes of runtime per case).
   `watcher/watcher.log.gz` is 20490 lines with 38 periodic dumps and zero hits
   for `Watcher detected`, `tripped`, `sanitize`, `TT_ASSERT`, `DEBUG_ASSERT`,
   `out of bounds`, `fault` or `Error`.

10. **One functional-layer tolerance was deliberately dropped.** The functional
   decode path trimmed the gathered cos/sin to `batch` when `rope_pos_ids` came
   back longer (`functional_decoder.py:1076-1078`), so it accepted a tile-padded
   position tensor. The fused path gathers straight into a one-shard-per-user
   layout derived from Q, so it takes the documented `[1, batch]` contract only —
   which is what `decode_position_tensors` builds and what every caller passes.
   A longer `rope_pos_ids` now fails in the shard/RoPE validation rather than
   being silently trimmed. Nothing in the suite or the public API regresses;
   recorded because it is a narrowing, not a no-op.
11. **Everything the functional stage disclosed still applies**, in particular
   the tt-metal sliding-window SDPA chunk bug (which is *why* the chunk sweep
   was restricted to `q_chunk == k_chunk`), the
   `chunked_scaled_dot_product_attention` `scale` nanobind bug, the `ttnn.slice`
   full-range aliasing hazard, the reduced full-context reference harness, and
   the untested batch-32-at-131072 combination. See
   [`../functional_decoder/README.md`](../functional_decoder/README.md).
12. **Multi-chunk prefill's own data movement is not optimised.** Chunking is
   inherited from the functional layer and this stage did not revisit it: in the
   `full` 16384-token window the two per-chunk input `ttnn.slice`s cost 478.6
   and 469.7 us and the final `ttnn.concat` 935.5 us, 1.70 % of that window
   (`sliding`'s equivalents are 491.5 / 482.5 / 950.5 us, plus a 374.8 us
   `q_filler` concat and a 301.2 us trim for the sliding-window tail, 2.48 %). Disclosed rather than fixed: removing it means changing
   the chunking contract itself, which the functional stage owns and every later
   stage inherits.
13. **`MINIMAL_MATMUL_BLOCKS`'s `o_proj` entry is gated at `min_rows = 8192`
   and measured only there.** `from_state_dict` accepts a `prefill_chunk_size`
   up to 16384, so a caller that raised the chunk would apply that config at a
   height it was never measured at (and it inverts by ~6 % below 8192). Not
   reachable at the shipped default chunk, which is 8192 and is what every test
   and capture uses.
14. **Scope.** BF16 everywhere, DRAM interleaved weights, no quantisation, no
   multi-device. Weight dtype/fidelity policy, DRAM-sharded decode matmuls and
   multi-chip are explicitly *not* this stage. The matmul path pins the
   baseline's fidelity exactly; the RMSNorms are the single knob this stage did
   turn, and limitation 3 records what it costs and buys.
