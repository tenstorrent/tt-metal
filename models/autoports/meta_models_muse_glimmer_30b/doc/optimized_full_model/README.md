# Optimized full model — `meta-models/Muse-Glimmer-30B`

An optimization pass over the [full model](../full_model/README.md), in place, on the
same four Blackhole dies. Same public contract, same 131072-token capability, the same
canonical split-sampling token-out path, the same carried-forward decoder precision /
KV-cache / CCL / residual policy — and **2.1 % faster token-out decode** from three
layout changes that move no arithmetic at all, plus **21 % faster TTFT** from an
opt-in traced prefill that is bit-identical to the eager path.

## Result

Batch 1, prompt 128 / generate 128, warmed, measured end to end from the host through
the public generator. `before` and `after` come from the **same script** on the same
host — `bench/evidence.py --stages perf` with and without `--baseline`, which reverts
exactly the three changes ([`logs/evidence_perf_before.log`](logs/evidence_perf_before.log),
[`logs/evidence_perf.log`](logs/evidence_perf.log)).

| | before | **after** | delta |
| --- | --- | --- | --- |
| **token-out decode** | 23.844 ms/token · 41.94 t/s/u | **23.298 ms/token · 42.92 t/s/u** | **-2.29 %** |
| **traced logits-only decode** | 23.164 ms/token · 43.17 t/s/u | **22.656 ms/token · 44.14 t/s/u** | **-2.19 %** |
| traced teacher-forcing decode | 36.88–37.99 t/s/u † | **37.07–38.15 t/s/u** † | overlapping ranges; no claim † |
| sampling trace | 0.632 ms | 0.632 ms | unchanged |
| **TTFT**, prompt 128, shipped default | 65.41 ms (min of 3) | 63.68 ms (min of 3) | inside the process spread; it is not device-bound, see below |
| **TTFT**, prompt 128, `prefill_trace=True` | — | **50.19 ms (min of 3)** | **-21.2 %** against the default arm |
| layer-stack lower bound | 23.239 ms/token | **22.858 ms/token** | −1.64 % |
| decode accuracy (teacher forcing) | top-1 0.990 · top-5 **1.000** · top-100 **1.000** | same | — |
| prefill accuracy | top-1 0.990 · top-5 **1.000** · top-100 **1.000** | same | — |
| qualitative suite, 6 chat prompts | — | **byte-identical to the full-model stage** | — |

† The teacher-forcing row is the **one** cross-process comparison in this table, and it
supports no improvement claim. Its `before` is the full-model stage's committed spread
over its own three runs (36.88 in `../full_model/evidence_accuracy.json`, 37.10–37.99 in
`../full_model/evidence_fp32_gate.json`); its `after` is this stage's three
(**37.07** in [`evidence_accuracy.json`](evidence_accuracy.json), 37.28 and 38.15 in
[`evidence_fp32_gate.json`](evidence_fp32_gate.json); all three are the `decode_t/s/u`
field, not the `e2e_t/s/u` one). **The ranges overlap**, so the
honest reading is "unchanged or slightly better, consistent with the two decode rows
above" — not +1.3 %, which an earlier version of this table claimed by quoting only the
top two of this stage's three measurements. The readiness runner is driven by the
accuracy stages rather than by `--baseline`, so a same-script teacher-forcing
before/after would have cost another 52-layer build for a number the token-out and
logits-only rows already establish exactly. Every other row in the table is
same-script.

Two things to read off that table. The decode figures moved by a hair over 2 % and the
**generated text did not move at all** — all six qualitative completions are
byte-for-byte what the full-model stage produced
([`qualitative/qualitative_tt_vs_full_model_stage.json`](qualitative/qualitative_tt_vs_full_model_stage.json)),
which is the strongest available statement that three layout changes changed only
layout. And **TTFT is not device-bound** — it is bound by host dispatch, which is the
largest single finding of this stage. The default arm does not move it; the opt-in
prefill trace this stage added does, by 21 %, with bit-identical logits. Both numbers,
the attribution and why the trace is opt-in are in
[Where TTFT actually goes](#where-ttft-actually-goes).

| item | value |
| --- | --- |
| model | `tt/model.py` (`_LMHead.forward`, `_embed`, `embed_decode`, `prefill_tokens_to_device`) |
| generator | `tt/generator.py` (`prefill_trace`, `_prefill_traced`, `_capture_prefill_trace`, `_kv_cache_signature`, `_release_prefill_traces`) |
| decoder | `tt/optimized_decoder.py` (`_OptimizedMLP.decode_forward`) |
| device | 4 x Blackhole, `ClusterType::P300_X2`, `ttnn.MeshShape(1, 4)`, `FABRIC_1D_RING`, `ttnn.Topology.Ring`, 2 links |
| context | **131072**, unreduced — [`../context_contract.json`](../context_contract.json) |
| tests | `tests/test_full_model.py`, **55** cases (46 inherited + 9 new), forward and reverse order |
| watcher | `WATCHER_CLEAN` on the shipped default (10 device cases) and on each opt-in prefill-trace case separately; 0 fatal messages, 0 tripped asserts — [`logs/check_watcher.log`](logs/check_watcher.log) |

## What ships

**Three decode-path changes and one opt-in prefill change.** The three decode changes are
each a layout change with no arithmetic in it, each measured on its own and cumulatively,
and each pinned by a test; they are on by default and they are what moves the decode rows
above. The fourth — a traced prefill, `GeneratorConfig.prefill_trace` — is off by default
and is what moves TTFT when a caller turns it on; it has its own section
([Tracing the prefill](#tracing-the-prefill-measured-and-shipped-as-an-opt-in)) because
the reason it is opt-in is the whole point of it.

| # | change | where | worth |
| --- | --- | --- | --- |
| 1 | the tanh softcap runs on the LM head matmul's own width-sharded L1 output instead of on DRAM-interleaved logits | `_LMHead.forward`, `LM_HEAD_SOFTCAP_IN_L1` | **−13.1 µs/step** of device time (36.85 → 23.79 µs for the pair) |
| 2 | the decode embedding all-gather writes the decoder's boundary layout directly | `_embed` / `embed_decode`, `EMBED_DECODE_GATHER_SHARDED` | **−2.0 µs/step**: one `interleaved_to_sharded` removed |
| 3 | the SwiGLU multiply's SFPU SiLU runs on 80 cores instead of 16, with three reshards to get there and back | `_OptimizedMLP.decode_forward`, `DECODE_SWIGLU_MUL_CORES` | **−7.4 µs/layer** = −383 µs/step over 52 layers (18.03 → 4.75 µs of multiply for 5.91 µs of reshard) |
| 4 | the prefill runs from a trace, keyed by padded prompt length. **Off by default** | `tt/generator.py`, `GeneratorConfig.prefill_trace` / `prefill_trace_max_entries` | **-21.2 % of TTFT** when a caller turns it on, with bit-identical logits; nothing on decode |

Cumulative, on the reduced two-layer build, traced logits-only decode, min of 3 rounds
x 64 replays, **one invocation** so the arms are like-for-like
([`logs/decode_ab_shipped.log`](logs/decode_ab_shipped.log)). Every arm is PCC
1.000000 against the baseline and picks the same token:

| arm | ms / 2 layers | delta |
| --- | --- | --- |
| the full-model stage's decode path | 1.5535 | — |
| \+ changes 1 and 2 (terminal only) | 1.5376 | −1.02 % |
| \+ change 3 (**shipped**) | **1.5246** | **−1.86 %** |

The two terminal changes are once per step, so their 0.0159 ms carries to the 52-layer
model unchanged; change 3 is per layer, so its 0.0130 ms becomes 26x that. Predicted
52-layer step: 23.164 − 0.016 − 0.338 = 22.81 ms. **Measured: 22.656 ms.** The
prediction is 0.7 % pessimistic, which is the right direction for a per-layer effect
extrapolated from two layers.

### 1. The softcap, on the shard the matmul already wrote

The head is `T * tanh(lm_head(h) * m / T)`, and `m/T` is folded into the weight at
setup, so the runtime tail is one matmul, one `tanh` and one scalar `mul`. The
full-model stage ran the matmul into width-sharded L1, then `sharded_to_interleaved`,
then both elementwise ops on a DRAM-interleaved `[1, 1, 32, 50688]` bf16 tensor. That
is 3.24 MB read and written twice over for two ops that touch each element once:

| row | before (DRAM interleaved) | after (width-sharded L1) |
| --- | --- | --- |
| `tanh` (`UnaryDeviceOperation`) | **17.71 µs** | **11.64 µs** |
| `* T` (`BinaryNgDeviceOperation`) | **19.14 µs** | **12.15 µs** |
| `sharded_to_interleaved` | 10.95 µs | 10.80 µs |
| pair total | 36.85 µs | **23.79 µs** |

`before` from [`../full_model/tracy/decode_perf_report.csv`](../full_model/tracy/decode_perf_report.csv)
ids 4283/4370/4196; `after` from [`tracy/decode_sliding_perf_report.csv`](tracy/decode_sliding_perf_report.csv)
ids 3140/3196/3197. The conversion still happens — the sampler's per-device index
arithmetic assumes DRAM-interleaved vocab shards, so it is a contract, not a choice —
but once, at the end, instead of before two full DRAM passes.

**The shard is padded, and that needed checking rather than assuming.**
`50688 / 52 = 975` columns per core is not a tile multiple, so `width_sharded_l1`
rounds each core to 992 and the shard set covers 51584 columns for 50688 real ones. Two
things make that safe for this pair specifically: `tanh` is bounded on every input, and
the scalar multiply keeps it bounded, so no padded lane can produce an inf or a NaN that
`sharded_to_interleaved` would then have to drop. It is not left as an argument:
`test_lm_head_softcap_runs_in_l1_and_matches_the_dram_form` runs the head both ways on
the same hidden state and asserts the two logit tensors are **bit-identical**
(`torch.equal`, not a PCC), that every value is finite, and that `|logits| <= T`.

### 2. The decode embedding gather, straight into the boundary layout

A decode step's embedding is one tile row. The fractured lookup gathers `32 x 1664` per
device back to `32 x 6656`, and the gather wrote DRAM interleaved and was immediately
followed by `interleaved_to_sharded` into the 16-core width-sharded L1 boundary spec —
the layout the norm, layer 0 and every layer boundary after it use.
`ttnn.experimental.all_gather_async` takes an output `memory_config`, so that
conversion can be the collective's own output layout. The
`InterleavedToShardedDeviceOperation` at 1.99 µs
([`../full_model/tracy/decode_perf_report.csv`](../full_model/tracy/decode_perf_report.csv)
id 4289) is gone from the new capture.

Prefill is untouched and deliberately so: it needs the interleaved form, and its gather
is the chunk-and-clone reproducibility path of `EMBED_GATHER_CHUNK_ROWS`
([full-model work log §15](../full_model/work_log.md)). The gate is the row count, as
it already was.

### 3. The SwiGLU multiply, and the op contract that forced a reshard

The largest non-matmul row in the decode layer was the SwiGLU multiply: **18.03 µs**,
against **1.88 µs** for a plain 6656-wide residual add on the *same* 16-core grid. The
difference is that the multiply carries the SFPU SiLU — the decoder stage measured
folding it into the gate matmul instead and it was 4.4 % slower, because the
DRAM-sharded matmul's `SFPU_ACTIVATION` runs on its 12 fixed worker cores — and an SFPU
transcendental costs time per *tile per core*, so 10 tiles on 16 cores is ten
serialised transcendentals per core.

The obvious lever is to give `mlp_gate`/`mlp_up` a wider **output** grid, so the
multiply inherits it for free. **That lever does not exist**, and the blocker is exact:

```
TT_FATAL: in DRAM sharded Matmul we don't have support for un-even sharding
currently. K: 208, per_core_K: 11.
```

The op requires `K_tiles % cores == 0`. `mlp_gate`/`mlp_up` have `K = 6656` = 208 tiles,
so their core count must divide 208: {8, 13, 16, 26, 52, 104}. `mlp_down` consumes
their output and has `K = 5120` = 160 tiles, so the same count must divide 160:
{8, 10, 16, 20, 32, 40, 80}. The intersection is **{8, 16}**, and 16 is already the
shipped value. Measured anyway ([`logs/decode_ab_swiglu.log`](logs/decode_ab_swiglu.log)):
`mlp8` is **+2.6 %**.

So the multiply is widened the other way round — reshard both operands onto a wider
grid for the multiply alone, and reshard the product back for `mlp_down`, whose
`in0_block_w=10` is derived from the gate/up grid and must keep it. Three reshards to
divide the SFPU work:

| gate/up mul grid | ms / 2 layers | delta | PCC vs 16 | same token |
| --- | --- | --- | --- | --- |
| 16 (no reshard, shipped by the full-model stage) | 1.5375 | — | — | — |
| 20 | 1.5408 | +0.21 % | 1.000000 | yes |
| 32 | 1.5357 | −0.12 % | 1.000000 | yes |
| 40 | 1.5306 | −0.45 % | 1.000000 | yes |
| **80** | **1.5248** | **−0.83 %** | **1.000000** | **yes** |
| `mlp8` (the matmul-grid family instead) | 1.5781 | +2.6 % | 0.999789 | yes |

Per-round spread is ±0.0005 ms, so the ordering is ~60x the noise. 80 is the largest
count that divides the 160-tile intermediate width. In the shipped profile the row
reads **4.75 µs of multiply for 5.91 µs of reshard** (2.20 + 1.60 + 2.10), against
18.03 µs before: **−7.4 µs per layer**, which is where ~383 µs of the 506 µs step delta
comes from.

`test_swiglu_multiply_runs_on_the_wide_grid_and_returns_the_narrow_one` pins that the
MLP's public output layout is still the boundary spec, so the inter-layer residual
contract the decoder stage asked this model to preserve is preserved literally: every
layer boundary is still the same 16-core `[32, 416]` width-sharded L1 fixed point, with
no conversion and no collective.

## Carried-forward decoder contract, unchanged

Read off the built model
([`evidence_accuracy.json`](evidence_accuracy.json)`:capacity`), not restated:

| item | value | changed? |
| --- | --- | --- |
| activation dtype | `BFLOAT16` | no |
| KV-cache dtype | `BFLOAT8_B` | no |
| weight dtypes | attention `BFLOAT8_B`, MLP `BFLOAT4_B` | no |
| LM-head dtype / fidelity / geometry | `BFLOAT4_B`, LoFi, `dram_sharded`, cores=52, `in0_block_w=2` | no |
| prefill collective | `async` (`reduce_scatter_minimal_async` + `all_gather_async`), BFP8 payload, 4 RS workers, AG barrier on | no |
| decode collective | `wrapper` `rs_ag`, activation-dtype payload, 1 RS worker | no |
| persistent CCL staging buffers | **off** — rejected by the decoder stage on a first-use correctness race; re-measured here and it is not a host-cost lever either | no |
| fractured prefill norm | on, gated at 256 rows | no |
| inter-layer decode residual | `WIDTH_SHARDED` L1, 16 cores, `[32, 416]` shards, replicated, at every layer boundary and into the terminal norm | no |
| `o_proj` decode geometry | 16 cores / `in0_block_w=2` (the decoder stage's shipped default; its 8-core/`in0_block_w=4` candidate stays declined) | no |
| decode SDPA | `max_cores_per_head_batch=32` | no |
| sampler | `models.common.sampling.SamplingGenerator`, traced, `tt_out_tok` into the decode token input, force-argmax **off** | no |
| sampler vocab masking | `sampler_invalid_vocab_mask_built: true`, tail width 704 | no |
| sampler topk geometry | 2 pieces x 32768, 64 candidates/device | no |

The dtype policy is verified **in the measured rows**, not in the JSON ($optimize
OPT-013). [`tracy/decode_sliding_perf_report.txt`](tracy/decode_sliding_perf_report.txt):

| row | shape | dtypes | fidelity |
| --- | --- | --- | --- |
| `wqkv` | 32 x 6656 x 1280 | `BF16 x BFP8 => BF16` | LoFi |
| `attn_gate` | 32 x 6656 x 1024 | `BF16 x BFP8 => BF16` | LoFi |
| `o_proj` | 32 x 1024 x 6656 | `BF16 x BFP8 => BF16` | LoFi |
| `mlp_gate` / `mlp_up` | 32 x 6656 x 5120 | `BF16 x BFP4 => BF16` | LoFi |
| `mlp_down` | 32 x 5120 x 6656 | `BF16 x BFP4 => BF16` | LoFi |
| LM head | 32 x 6656 x 50688 | `BF16 x BFP4 => BF16` | LoFi |
| KV-cache update | — | `BFP8, BF16 => BFP8` | — |
| decode SDPA | — | `BF16, BFP8 => BF16` | — |

No broad datatype frontier search was run here; `$datatype-sweep` owns Pareto
selection. The one precision question this stage did ask is the one $optimize insists
on when a full-model step is materially above its roofline — see
[Performance accounting](#performance-accounting) for why the answer was "the gap is
not precision".

## Performance accounting

Three numbers from the same configuration, reconciled
([`perf_summary.json`](perf_summary.json)):

| | value | source |
| --- | --- | --- |
| roofline | **8.829 ms/token** | 4,520,382,464 B/device ÷ 512 GB/s |
| device-time decode | **22.838 ms/token** | 39 x 431.48 µs + 13 x 409.16 µs + 691.07 µs |
| end-to-end token-out | **23.298 ms/token** | `evidence_perf.json`, min of 3 |
| end-to-end logits-only | **22.656 ms/token** | same run, decode trace alone |

The bytes are per device per token: 4,327,784,448 of layer weights (measured from the
built model, not a formula), 189,775,872 of BFP4 LM head, 106,496 of embedding rows and
~2.7 MB of BFP8 KV cache at the benchmark's context. The bandwidth is back-derived from
`tt-perf-report`'s own DRAM% columns rather than a data sheet: **512 GB/s per device**,
which is what every DRAM-classified row in
[`tracy/decode_sliding_perf_report.csv`](tracy/decode_sliding_perf_report.csv) implies —
the LM head at 279.38 GB/s / 54.57 %, `wqkv` at 394.85 / 77.12 %, `attn_gate` at
355.08 / 69.35 %, `o_proj` at 318.97 / 62.3 % and the three MLP rows at
267.64–269.58 / 52.27–52.65 %.
That is a **consistency** check, not an independent derivation: `tt-perf-report` computes
`DRAM %` as `DRAM / peak`, so `DRAM / DRAM% x 100` returns its assumed peak by
construction. What it establishes is that one peak is assumed for every row, and 512
GB/s is the number the tool's own model uses for this part.

**The roofline fraction is 37.9 %, and the reason is structural rather than a missing
knob.** Of the 431.48 µs sliding layer (the 1122.551 µs window minus the 691.07 µs
terminal term), **252.41 µs is the six DRAM-sharded projections** — `wqkv` 21.577,
`o_proj` 21.368, `attn_gate` 19.195 and MLP gate/up/down 63.208 / 63.393 / 63.665 — and
the remaining **179.07 µs is everything else**: 8.7 µs hidden-size RMSNorms and 4.6 µs
per-head norms, a 15.136 µs `SdpaDecode`, two reduce-scatter + all-gather pairs at
54.886 µs together, 3.7–3.9 µs paged cache updates, 2.7–2.8 µs rotary applications, head
create/concat, and the layout conversions. None of that moves DRAM bytes for the roofline
to count, and the projections themselves run at 52.27–77.12 % of peak. A model built from
many small ops sits lower on the roofline; the requirement is the explanation, and the
explanation is that **41.5 %** of the layer is latency-bound rather than bandwidth-bound
work. Every figure in this paragraph is a row or a group of the audit table below, which
is a partition of the cited CSV.

One row in the committed profile looks like it contradicts "no host gap" and does not.
It is restated from *this* capture, and the first version of this paragraph got both the
numbers and the mechanism wrong, so the corrected version is spelled out.

[`tracy/decode_sliding_perf_report.csv`](tracy/decode_sliding_perf_report.csv) shows
`EmbeddingsDeviceOperation` (id 3145) with 9.3 µs of device time and a **310.959 µs
op-to-op gap** — **21.6 %** of the window — and
[`tracy/decode_sliding_perf_report.txt`](tracy/decode_sliding_perf_report.txt) advises
*"Running with tracing could save 307 μs (20.8% of overall time)"* on a window that
**is** a `ttnn.execute_trace` replay, with a footer of `55 device ops … 1,123 μs
[device] 358 μs [gap]`.

It is **not** a first-op window boundary: the report is sorted by `HOST START TS`, and id
3145 is row **29 of 55**, preceded by 27 ops whose gaps are 0.47–1.47 µs and immediately
by `PlusOneDeviceOperation` (id 3143) at 0.645 µs. It is the **inter-replay boundary**
inside the signposted window — `run_tracy.sh` captures one replay, and the embedding is
the first op of the graph, so its "previous operation" is the last op of the *warm-up*
replay that ran before the signpost, across a profiler flush.

What bounds it is arithmetic rather than the mechanism: summed device time is
**22.838 ms** against a measured logits-only replay of **22.656 ms**, so there is no room
for a real 311 µs per-step bubble in the un-profiled run — and the steady-state loop's
token, position and synchronisation counters are all zero per token
([`evidence_accuracy.json`](evidence_accuracy.json)`:fallback_audit`). Every percentage
in this document comes from device-time sums, which exclude gap entirely.

**Device time is 0.8 % *above* the traced replay it is meant to explain** (22.838 ms
against 22.656 ms of logits-only replay), which is worth naming rather than hiding.
`tt-perf-report` merges a 4-device capture by taking the **max** per op, so a step
where different devices peak on different ops sums higher than any one device's
critical path, and profiler instrumentation adds to each row. The direction is the
useful part: there is no room left between device time and end-to-end for host work,
which is what the zero per-token refresh counters independently say.

`22.656 + 0.632 = 23.288` against a measured token-out of `23.298` — **the two traces
account for the whole step to within 10 µs**, and that 10 µs is the caller's token readback.
(The residual on the unrounded values in [`evidence_perf.json`](evidence_perf.json) is
**9.9 µs**, which is what the figure gate computes and rounds; it was 26.8 µs in the run this
figure was first taken from, and the difference is process-to-process variance in a ~10 µs
quantity, not a change in the path.)

### Against the layer-stack lower bound

The floor is this stage's own per-layer traced decode, re-measured because change 3
moved it, with the decoder stage's own harness at the decoder stage's own context
([`logs/layer_ab_after.log`](logs/layer_ab_after.log),
`layer_ab.py --candidates tp4,tp4b --decode-context 2048`).

**The two columns are not one measurement.** `logs/layer_ab_after.log` contains only
`tp4` and `tp4b`, which are both *after* arms — `tp4b` is the same-config repeat control,
and it reproduces `tp4` to 1e-4. A second invocation on a different day
([`logs/layer_ab_oproj.log`](logs/layer_ab_oproj.log), run for the `o_proj` candidate)
reproduces the same two arms at 0.4474 / 0.4475 sliding and 0.4166 / 0.4162 full, so the
floor is stable across processes as well as within one. The `before` column is the decoder stage's committed
figure from [`../optimized_multichip_decoder/README.md`](../optimized_multichip_decoder/README.md),
measured by the same script on the same harness in an earlier process. The delta is
corroborated independently by the reduced two-layer A/B (6.5 µs/layer there against
7.3 µs/layer here), so the direction and rough size are safe; a same-process before/after
would have cost a second 52-layer decoder A/B for a number the A/B already gives.

| | layers | ms/layer before | ms/layer after | ms after |
| --- | --- | --- | --- | --- |
| sliding | 39 | 0.4546 | **0.4473** | 17.445 |
| full attention | 13 | 0.4238 | **0.4164** | 5.413 |
| **layer-stack lower bound** | 52 | 23.239 | | **22.858** |

Both arms report identical PCC — prefill 0.993700, decode 0.993488 (sliding) and
0.992220 / 0.992188 (full) — so the 1.6 % is not bought with numerics.

* the model decode trace is **22.656 ms**, i.e. **-0.9 %** against the 22.858 ms
  context-2048 floor;
* token-out is **23.298 ms**, i.e. **+1.9 %** on that floor and **-3.7 %** on
  floor-plus-terminal (22.858 + 0.691 device terminal + 0.632 sampling = 24.181).

The gate is 10–15 % over floor-plus-terminal, and on those numbers there is no gap to
split. But round 5 was right that the comparison as stated could not *fail*: the floor was
measured at context 2048 while the benchmark decodes at 128–256 positions, and a floor
inflated by longer attention can only make the gate pass. So the floor was re-measured at
the benchmark's own context ([`logs/layer_ab_after_ctx256.log`](logs/layer_ab_after_ctx256.log),
`layer_ab.py --candidates tp4,tp4b --prefill-seq 128 --decode-context 256`):

| | ms/layer @2048 | ms/layer @256 | delta |
| --- | --- | --- | --- |
| sliding x39 | 0.4473 | **0.4390** | −1.9 % |
| full x13 | 0.4164 | **0.4077** | −2.1 % |
| **layer-stack floor** | 22.858 | **22.421** | −1.9 % |

Against that tighter comparator:

* the model decode trace is **22.656 ms**, i.e. **+1.05 %** over 22.421 ms — the 52 layers
  plus the whole terminal path cost 1 % more than the 52 layers alone cost when measured one
  at a time;
* token-out is **23.298 ms**, **+3.91 %** on it, of which the sampling trace is 632 µs
  (2.8 %);
* the goal's comparison is against floor **plus terminal work**: 22.421 + 0.691 device
  terminal + 0.632 sampling = **23.744 ms**, and token-out is **1.9 % below** that.

**It is a comparator, not a lower bound, and round 6 was right that calling it "not
conservative" overstated it.** The proof is in the document's own numbers: 22.421 + 0.691 =
**23.112 ms** for 52 bare layers plus the terminal device term, against a **22.656 ms**
measured 52-layer traced replay that also contains the embedding, the gather, the terminal
norm, the LM head, the softcap and two `plus_one`s. The real step beats the sum of its
separately-measured parts by **0.456 ms**, so the per-layer harness overprices the stack by
about **2 %** — it isolates one layer with its own warm-up, dispatch and drain rather than
measuring it inside a 52-layer pipeline. Measuring at context 256 rather than the window's
~192 mean adds a further +0.07 %, which is negligible but is conservatism in the same
direction.

So what the re-measurement bought is a **tighter and same-context** comparator, not a
falsifiable bound: 22.421 instead of 22.858 removes 1.9 % of unrelated slack, and the
`+1.05 %` figure is a real reading rather than the `−0.9 %` that made the old comparison
uninterpretable. Both remain far inside the 10–15 % bar, and the honest statement of *why*
there is no gap to close is the one that does not depend on the floor at all: the fallback
audit is zero per-token host work, the two traces account for the step to 9.9 µs, and the
terminal path is priced directly in the profile.

`tp4b`, the same-config repeat control, reproduces `tp4` to 1e-4 at this context too
(0.4390 / 0.4390 sliding, 0.4077 / 0.4078 full), and PCC is unchanged from the 2048 run —
prefill 0.993700 / 0.992220, decode 0.993488 / 0.992188 — so the shorter context is a
cheaper floor, not a different model.

## Where TTFT actually goes

TTFT at prompt 128 on the shipped default is **63.68 ms**, of which the 52-layer stack
is **60.3 ms** ([`ttft_breakdown_before.json`](ttft_breakdown_before.json) — the name is historical: `bench/ttft_breakdown.py` has no `--baseline` flag and this file is the **shipped default's** phase table, not a before-arm). The phase
table below is the **min of each phase across three rounds** of that probe, which is not
one run — the phases are timed with a device synchronisation around each, and the sum
(64.45 ms) is quoted against a 63.68 ms TTFT measured in a different process. It is a
budget, not an accounting identity; what it establishes is which phase dominates.

| phase | ms (min of 3) | |
| --- | --- | --- |
| token staging + page table | 0.65 | host tensors + H2D |
| embedding + gather + norm | 0.51 | |
| **52 decoder layers** | **60.28** | |
| terminal norm + LM head + softcap | 0.85 | |
| eager sampling + token readback | 2.16 | untraced, once per request |

The reduced-variant Tracy window prices one prefill layer at ~0.83 ms of device time at
128 rows ([`tracy/prefill_128_perf_report.csv`](tracy/prefill_128_perf_report.csv):
**2606.3 µs** of window over 2 layers plus the terminal path), so 52 of them is ~43 ms. The
missing ~17 ms is **host dispatch**, and three independent measurements say so:

1. **The device finishes when the host stops talking.** Issuing all 52 layers with no
   synchronisation takes **54.91 ms**; draining afterwards takes **55.08 ms**
   ([`prefill_host_probe.json`](prefill_host_probe.json)). 0.17 ms of device work
   outlives the last dispatch.
2. **89 % of the wall time is inside one Python frame**, `ttnn/decorators.py::FastOperation.__call__`
   — 0.057 s of 0.064 s over **4122 ttnn calls**, with the decorator's own bookkeeping
   (`_requires_slow_runtime`, `is_python_io_recording_enabled`) at ~0.5 µs of it
   ([`logs/prefill_cprofile_128.txt`](logs/prefill_cprofile_128.txt)). The cost is the
   C++ dispatch, not the wrapper.
3. **Per-op wall time inside that frame is 1.7–140 µs**, and the collectives are the top
   of that range ([`prefill_opcount.json`](prefill_opcount.json)):

| op | calls | ms | µs/call |
| --- | --- | --- | --- |
| `ttnn.experimental.reduce_scatter_minimal_async` | 104 | **14.60** | 140.3 |
| `ttnn.experimental.all_gather_async` | 105 | **6.33** | 60.3 |
| `ttnn.linear` | 312 | 5.97 | 19.1 |
| `ttnn.add` | 104 | 5.19 | 49.9 |
| `ttnn.rms_norm` | 313 | 5.17 | 16.5 |
| `ttnn.multiply` | 104 | 4.26 | 41.0 |
| `ttnn.interleaved_to_sharded` | 208 | 3.67 | 17.6 |
| `ttnn.deallocate` | **1957** | 3.27 | **1.67** |
| `ttnn.sharded_to_interleaved` | 208 | 2.26 | 10.9 |
| the other 12 op kinds | 707 | 7.86 | — |
| **sum of the per-op rows** | **4122** | **58.56** | |
| wall time for the same prefill | | **62.75** | |

The **4.19 ms** difference between the two totals is not unattributed device time: the
patched frame times the ttnn call only, so the model's own Python between calls — shape
reads, memory-config lookups, dict lookups, the `_prefill_chunk`/`_prefill_attention`
control flow — is the rest. `cProfile` (measurement 2) puts that at ~11 % of the window,
which is the same number from the other side.

Two useful facts fall out. `ttnn.deallocate` is 47 % of the calls and 5.6 % of the time,
so the obvious "there are too many deallocates" theory is wrong. And the **209
collective dispatches are 20.93 ms — 33 % of the wall time on 5 % of the calls.**

### The collectives, and the correction the stage review forced

The first version of this section measured the collectives standalone at a **BF16**
payload in a hot loop, got 57–70 µs/call, and concluded "nothing moves it". That was
wrong twice over: the model's prefill reduction payload is **BFP8**, not BF16 (verified
in [`tracy/prefill_128_perf_report.csv`](tracy/prefill_128_perf_report.csv):
`ReduceScatterMinimalAsyncDeviceOperation` input dtype `BFLOAT8_B`, plus 104
`ttnn.typecast` calls that produce it), and a hot loop of identical collectives is not
the regime the model runs them in. Re-measured like-for-like
([`ccl_host_probe_bfp8.json`](ccl_host_probe_bfp8.json),
[`ccl_host_probe_bfp8_loaded.json`](ccl_host_probe_bfp8_loaded.json),
[`ccl_host_probe_bf16.json`](ccl_host_probe_bf16.json) — the BF16 arm is re-run under
its own name so the superseded numbers are inspectable rather than deleted):

| arm, BFP8 payload | hot loop | with one prefill-sized matmul in front |
| --- | --- | --- |
| `ttnn.clone` (reference, same payload) | 9.11 | 24.73 |
| `ttnn.add` (reference, same payload) | 29.82 | 45.91 |
| `reduce_scatter_minimal_async`, allocating, **4 workers** (the shipped prefill setting) | 72.10 | **117.05** |
| `reduce_scatter_minimal_async`, **persistent buffers**, 4 workers | **62.12** | **96.86** |
| `reduce_scatter_minimal_async`, allocating, 1 worker | 58.65 | 61.45 |
| `all_gather_async`, allocating | 56.04 | 61.22 |
| `all_gather_async`, persistent output buffer | 56.18 | 69.09 |
| `ttnn.reduce_scatter` (wrapper), 4 workers | 58.88 | 91.42 |
| `ttnn.all_gather` (wrapper) | 56.31 | 58.40 |
| `ttnn.all_reduce` (wrapper, one call) | 118.04 | 128.26 |

And in the model itself, with the device **drained before each collective** so the
recorded time is dispatch without queue backpressure
([`prefill_opcount.json`](prefill_opcount.json)`:prefill_drained_collectives`):

| | pipelined | drained |
| --- | --- | --- |
| `reduce_scatter_minimal_async` | 140.3 µs/call | **114.6 µs/call** |
| `all_gather_async` | 60.3 µs/call | 64.2 µs/call |

Three conclusions, and the third replaces the retracted one:

* **the 2x gap between the isolated probe and the model is reproduced by putting one
  matmul in front of the call** — 117.05 µs against 72.10 in the hot loop, against
  114.6–140.3 in the model. So ~26 µs of the in-model figure is queue backpressure the
  drained pass removes, and the rest is the op's own dispatch in a realistic
  instruction stream. It is not an unexplained 6.8 ms;
* **`ttnn.all_reduce` costs two dispatches** (118.04 µs), so "one fused call instead of
  two" is not available, and the composite wrappers cost what the primitives they lower
  to cost;
* **persistent buffers *do* move it**, by 14 % in the hot loop (62.12 against 72.10) and
  17 % loaded (96.86 against 117.05), at the model's own BFP8 payload and 4-worker
  setting. That is ~2 ms of this prefill. It is **not adoptable**: the decoder stage
  rejected `ccl_persistent_buffers` on an intermittent *first-use* correctness race
  that moved between arms and between runs of the same arm
  ([`../optimized_multichip_decoder/README.md`](../optimized_multichip_decoder/README.md)),
  and an intermittently wrong first token is not something a 52-layer stack may ship for
  2 ms of TTFT. The earlier claim in this document that persistent buffers are "within
  noise" came from the BF16 hot-loop arm and is withdrawn.

### Tracing the prefill: measured, and shipped as an opt-in

With persistent buffers unavailable on correctness and no cheaper collective to switch
to, tracing is the remaining mechanism, so it was **captured and measured** rather than
costed ([`prefill_trace_probe.json`](prefill_trace_probe.json),
[`logs/prefill_trace_probe.log`](logs/prefill_trace_probe.log)), on the real 52-layer
build with the decode and sampling traces already captured in the same trace region:

| | value |
| --- | --- |
| eager prefill | 59.80 ms |
| **warmed traced replay** | **44.96 ms — 1.33x** |
| replay vs eager | **bit-identical** (`torch.equal`, `max_abs_diff = 0.0`, same argmax) |
| capture cost | 98.16 ms |
| DRAM retained per device, 128 rows | 3.3 MB |
| payback | **6.6 replays of the same bucket** |
| coexists with the decode + sampling traces | yes (`MeshTraceId(0)` decode, sampling captured, prefill captured after both) |

So it works, it is exact, and it fits. It **ships as
`GeneratorConfig.prefill_trace`**, with `prefill_trace_max_entries` bounding how many
padded-length buckets may hold one, and through the same evidence harness it gives:

| | shipped default | **`--prefill-trace`** |
| --- | --- | --- |
| **TTFT**, prompt 128 | 63.68 ms (min of 3; 67.74 / 63.68 / 67.43) | **50.19 ms (min of 3; 50.62 / 50.19 / 50.20)** |
| token-out decode | 23.298 ms/token | 23.328 ms/token |
| traced logits-only decode | 22.656 ms/token | 22.657 ms/token |
| prefill trace buckets captured | — | `[128]` |

**-21.2 % of TTFT**, with decode untouched to 0.13 %
([`evidence_perf_prefill_trace.json`](evidence_perf_prefill_trace.json),
[`logs/evidence_perf_prefill_trace.log`](logs/evidence_perf_prefill_trace.log)).

It is **off by default**, and that is a contract decision rather than caution. The graph
bakes in the padded row count, so one trace serves one 32-row bucket — all prompt
lengths in `(R-32, R]` share the same last-token tile-row slice, so the bucket is a real
equivalence class, but only 32 wide — and capture costs 98 ms against a ~15 ms
per-replay saving. That is a clear win for a caller whose prompt lengths repeat or are
bucketed and a one-time 98 ms cost for one whose lengths do not, and the generator
cannot tell which it is. A serving stage that buckets prompt lengths should turn it on
and raise `prefill_trace_max_entries` to its bucket count; DRAM per entry scales with
the padded row count, so the 3.3 MB at 128 rows is ~210 MB at 8192.

**Three eligibility conditions the advertisement above does not imply**, and round 7 was
right that a caller reading only the headline would not know them
(`tt/generator.py:_prefill_user`): the trace is used only when `prompt_len <=
config.prefill_chunk_size` (**8192**), only for `user_id == 0`, and only when
`return_all_logits` is false. Outside any of those the call falls back to the eager path —
**correctly**, and with no error, which is the right behaviour but means the 21 % win is
silently absent for a prompt above 8192, for every batch row but the first, and for the
teacher-forcing all-logits path. A serving stage that chunks long prompts or batches users
gets the flag's benefit on the short single-user requests only.

`test_prefill_trace_is_opt_in_and_matches_the_eager_path` pins the contract that a
caller turning it on depends on: the traced prompt returns exactly the tokens the eager
path returned, a second call on the same bucket replays instead of recapturing, a
non-tile-aligned prompt inside the bucket (120 tokens in the 128 bucket) is served by
it, a different bucket past the cache bound falls back to eager rather than evicting,
and `teardown()` releases everything. Binding an external KV cache releases every
prefill trace, because the trace bakes in the cache buffer addresses it writes.

### The default TTFT does not move, and its spread is why

TTFT is the one figure in this document that moves between processes. On the shipped
default it is 67.74 / 63.68 / 67.43 ms in this run; the baseline arm measured
70.69 / 67.68 / 65.41 ms in its own process. The full-model stage documented four passes of
*identical* code spanning 61.09–66.04 ms, an 8 % spread that no round-to-round variance
predicts, and attributed it to prefill being compiled, allocated and scheduled once per
process. The decode figures do not do this: 23.334 / 23.340 / 23.298 here,
23.854 / 23.844 / 23.884 in the baseline arm — a 0.2 % spread,
and the two arms do not overlap.
Read the default TTFT as ~61–70 ms in both arms and the decode numbers as exact. Nothing
this stage changed touches prefill except the terminal softcap, which is ~30 µs of a
65 ms window. The `--prefill-trace` arm's 50.19 ms is well outside that spread in the
other direction.

### One remaining prefill inefficiency, priced and left

The prefill terminal norm runs on **one core for 133.868 µs** and the embedding norm on
four cores for **133.979 µs** ([`tracy/prefill_128_perf_report.csv`](tracy/prefill_128_perf_report.csv)
ids **3886** and **3579**), against 8.8 µs for the same norm width-sharded in decode:
`ttnn.rms_norm` on a DRAM-interleaved input parallelises over tile *rows*, and both of
these run on a 32-row slice. That is 0.27 ms of a 65 ms TTFT (0.4 %), and it is left
alone deliberately: the fix is to route both through the sharded form, which changes
prefill *numerics* — the two norms sit on the accuracy gates' critical path — for 0.4 %
of a figure whose process-to-process spread is 8 %. Recorded as limitation 9 with the
measurement so a later stage can take it with the accuracy re-run it needs.
## Operation-topology audit

The measured decode path — the signposted sliding-layer window, one sliding layer plus the
terminal work — from
[`tracy/decode_sliding_perf_report.csv`](tracy/decode_sliding_perf_report.csv). "Action"
is what this stage did about it.

Round 3 of the stage review found this table's µs column irreconcilable with the CSV it
cites: three values were rounded up rather than summed, and two were silently the
*pre-change* values from the previous stage's capture. It is now a **partition**: the `ids`
column names every row of the CSV, each group is the exact sum of its rows, no row appears
twice, and all 14 groups together are all 55 rows and sum to the window's **1122.551 µs**.
`bench/check_reported_figures.py` re-derives each cell from the CSV and checks the
partition (every id used exactly once, group sums, column total). Where a change moved a
group, the pre-change value is named as such with its own id in the previous stage's
capture; nothing in this column is a pre-change value wearing a post-change label.

| op group | ids | device µs | candidate | action |
| --- | --- | --- | --- | --- |
| LM-head matmul, 32 x 6656 x 50688 BFP4 | 3139 | 603.798 | **2.6 %** of the 23.298 ms token-out step, and 40.8 % of this one-layer profiling window (the CSV's own `Total %` cell); DRAM-bound at 279.38 GB/s reading 190 MB of weights | kept: the geometry ladder (`dram_sharded`, cores=52, `in0_block_w=2`) is the full-model stage's measured winner over the legal values 1/2/4 at BFP4, and `in0_block_w=4` fails with an exact L1 blocker (*"Statically allocated circular buffers ... grow to 1821824 B which is beyond max L1 size of 1572864 B"*) |
| MLP gate/up/down, 3 x BFP4 | 3071, 3127, 3132 | 190.266 | `in0_block_w` already at the largest legal divisor (13/13/10); packed gate/up measured +5.5 % by the decoder stage | kept |
| `wqkv` + `attn_gate`, 2 x BFP8, same input | 3039, 3172 | 40.772 | OPT-001: two projections consuming the same post-norm activation could be one | **not taken, with a reason**: both rows are the *best* DRAM utilisation in the layer (77.12 % and 69.35 %), so packing cannot reduce the bytes, and the two outputs need different downstream layouts — QKV `sharded_to_interleaved` into head creation, the gate kept sharded until after SDPA — so a packed output needs an unshard plus two slices plus a reshard to split |
| `o_proj`, BFP8 | 3065 | 21.368 | OPT-011 narrower working shard (8 cores / `in0_block_w=4`) | **re-measured on this stage's shipped path** rather than re-declined on the decoder stage's note ([`logs/layer_ab_oproj.log`](logs/layer_ab_oproj.log), `layer_ab.py --candidates tp4,tp4b,oproj_c8_bw4`): **sliding 0.4467 against 0.4474 / 0.4475**, i.e. −0.17 %, and **full 0.4163 against 0.4166 / 0.4162**, i.e. inside the repeat control's own spread. HF-reference PCC is unchanged on prefill (0.993700 / 0.992220) and marginally *better* on decode (0.993503 vs 0.993488 sliding, 0.992196 vs 0.992188 full). Worth **−0.03 ms/token, 0.12 %** of the step. Declined, and the reason is now specific: adopting it changes the **decoder stage's** shipped geometry and its single-grid invariant, which three of that stage's tests assert and whose multichip-vs-single-chip gate (0.999183 against a 0.999 bar) this stage's harness does not run. 0.12 % is not worth moving another stage's default and gate from here; the measurement is committed so that stage can take it |
| SwiGLU multiply (SFPU SiLU) † | 3075 | 4.747 | widen the grid | **taken**: 80-core reshard. The pre-change row is **18.026** (`../full_model/tracy/decode_perf_report.csv` id 4187; its sibling layer, id 4240, is 17.941), so the multiply itself is −13.28 µs and change 3's three reshards cost 5.907 (ids 3073, 3129, 3076, in the layout-conversion row below) — **−7.4 µs/layer of device time**. The A/B measures **−6.4 µs/layer** of *step* time (1.5375 → 1.5248 ms per 2 layers, [`decode_ab_swiglu.json`](decode_ab_swiglu.json)); the step win is the smaller of the two because the step is not device-bound |
| softcap `tanh` + `* T` | 3140, 3196 | 23.786 | run on the matmul's shard | **taken**: pre-change **36.853** (`../full_model/tracy/decode_perf_report.csv` ids 4283 + 4370), so **−13.07 µs/step** of device time, at +126,976 B/bank of peak L1 |
| RMSNorm x 8 | 3137, 3147, 3153, 3180, 3203, 3211, 3233, 3245 | 61.628 | wider norm grid | not available: the norm must consume and produce the 16-core boundary spec, which is the preserved inter-layer residual contract. Six run on 22 cores (~8.7 µs each) and the two `q_norm`/`k_norm` on 32 (~4.7 µs) |
| reduce-scatter + all-gather x2 | 3231, 3232, 3243, 3244 | 54.886 | fewer/cheaper collectives | not available at this residual contract: `ttnn.all_reduce` costs two dispatches, the wrappers cost what the primitives cost, and the count is the preserved replicated-residual contract. Persistent buffers *are* worth 14–17 % of the reduce-scatter's **host** cost and are blocked on the decoder stage's correctness race |
| `SdpaDecode` | 3168 | 15.136 | — | kept: explicit program config, BFP8 cache, `max_cores_per_head_batch=32`, all from the decoder stage |
| embedding all-gather | 3201 | 16.498 | collective writes the boundary layout | **taken**: change 2 removes the `interleaved_to_sharded` that used to follow it — **1.992 µs**, `../full_model/tracy/decode_perf_report.csv` id 4289 — so this group is one row here and two there (15.838 + 1.992 = 17.830). The gather itself measures 0.66 µs *slower* writing the sharded layout — one row, well inside the ±0.5 µs these rows move between captures — so the honest reading of change 2 is **the removed conversion, −2.0 µs/step**, and the group as a whole is −1.33 µs. It is 0.008 % of the step either way; it is taken because it removes an op, not because the profile can resolve it |
| embeddings (token lookup + the two RoPE tables) | 3145, 3158, 3161 | 14.331 | — | kept: 1-core lookups on a 32-row slice, the same shape as the prefill norms in limitation 9 |
| attention glue (RoPE apply x2, qkv-head create, concat-heads, transpose x2, paged cache update x2, the attention-gate multiply, residual adds x2) | 3054, 3055, 3096, 3107, 3112, 3115, 3119, 3124, 3191, 3214, 3221 | 39.604 | — | kept: all of these are the decoder stage's optimized composite ops already |
| layout conversions (`i2s`/`s2i`/`reshard`) | 3073, 3076, 3095, 3099, 3100, 3105, 3108, 3116, 3118, 3129, 3193, 3197, 3207, 3212, 3224 | 33.886 | — | 5.907 µs of it is change 3's own three reshards (3073, 3129, 3076), which buy 13.28 µs of multiply; 13.283 µs of it is the LM head's own input reshard and output unshard (3193, 3197), i.e. terminal rather than layer work |
| `plus_one` x2 | 3143, 3254 | 1.845 | — | kept: this is the device-side position/RoPE advance |
| **window total** | all 55 | **1122.551** | | the 14 groups partition the CSV: every id once, no id twice |

† `tt-perf-report`'s `Cores` column does not report the grid an op ran on, in either direction: it reads **110 for every elementwise row** — including the softcap pair, which runs on the 52-core LM-head shard — and **12 for every DRAM-sharded matmul row** (3039, 3065, 3071, 3127, 3132, 3139), where the shipped geometry is `cores=52`. Round 5 caught the second half, which matters because the `SLOW` paragraph below says those rows "do expose core count". Neither number is wrong so much as answering a different question, and this table uses the column as evidence of a grid nowhere. That the SwiGLU multiply runs on 80 cores is pinned by the reshard rows either side of it (3073 and 3129 at 80 cores, 3076 back to 16), by `DECODE_SWIGLU_MUL_CORES`, and by `test_swiglu_multiply_runs_on_the_wide_grid_and_returns_the_narrow_one`. Round 4 of the stage review caught the earlier wording, which read as if the profile itself showed the grid. Two ids in the attention-glue row were also described as residual adds when 3119 (4.610 µs) is the attention-gate multiply; only 3124 and 3191 (1.880 / 1.873 µs) are the 6656-wide residual adds the document prices at 1.88 µs elsewhere. No value moves.

`tt-perf-report` marks five rows `SLOW` with *"No output subblock size found"* — ids
**3065** (`o_proj`), **3071** and **3127** (MLP gate/up), **3132** (MLP down) and **3139**
(the LM head, 40.8 % of this window). That is structural, not a finding:
`MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` has no output-subblock fields to
report. The geometry those rows *do* expose — `in0_block_w`, and the DRAM-sharded flag and dtypes (the `Cores` column does not, see the footnote above) — is at the
largest legal divisor for every one of them (13/13/10 for MLP gate/up/down, 2 for
`o_proj` and the LM head, both capped by their per-core K-tile count), and the two rows
where a larger value is arithmetically legal fail with exact L1 circular-buffer blockers
recorded by the decoder and full-model stages.

## What each trace bakes in, and who owns it

Round 4 of the stage review found the decode trace replaying against a KV cache the caller
had rebound away from. The general form of that bug is "a captured graph holds a device
address that someone else can invalidate", so this is the full inventory of captured graphs
in the shipped path and of the device state each one bakes in. There are three captures and
exactly one caller-owned input among them.

| trace | captured in | device state baked in | owner | guarded by |
| --- | --- | --- | --- | --- |
| decode | `_capture_decode_trace` | the 4 persistent decode inputs (`tokens`, `current_pos`, `rope_pos_ids`, `page_table`); **every layer's `k_cache`/`v_cache`** via `paged_update_cache` | inputs: the generator, allocated once in `_allocate_device_inputs` and never rebound. **Cache: the caller** | `_decode_trace_cache_sig`, compared on every `prefill_forward`/`decode_forward` that passes a cache |
| sampling | `SamplingGenerator.capture_trace` | the decode trace's `logits` output and `tt_out_tok`, which *is* `_device_inputs["tokens"]` | the generator, both of them | released together with the decode trace — it is validated by tensor identity against those logits, so it cannot outlive them |
| prefill (opt-in) | `_capture_prefill_trace` | its own persistent `tokens`/`page_table`/`logits`; **every layer's `k_cache`/`v_cache`** via `paged_fill_cache` | its own tensors: the generator. **Cache: the caller** | `_prefill_trace_cache_sig`, same comparison |

Two things follow, and both are why the fix is where it is. The **KV cache is the only
caller-owned device state any trace holds**, so one signature per trace over the cache
buffer addresses covers the whole hazard — there is no second class of rebindable input to
miss. And the sampling trace needs no signature of its own: everything it holds belongs to
the generator, and it is released whenever the decode trace it was captured over is.

`reset()` deliberately does *not* invalidate anything. `reset_kv_cache()` zeroes the paged
cache in place, so the buffer addresses are unchanged and every trace over them is still
correct — which is the property `test_reset_zeroes_the_cache_without_dropping_traces` pins.
`teardown()` and the rebind path now go through the same `_release_decode_trace()`, because
two copies of a trace lifecycle drifting apart is what produced the round-4 finding.

## Split sampling, preserved

The canonical contract is unchanged and re-asserted on the tensors in this stage's own
run ([`evidence_accuracy.json`](evidence_accuracy.json)`:split_sampling`,
[`logs/evidence_accuracy.log`](logs/evidence_accuracy.log)):

| claim | result |
| --- | --- |
| the sampler is traced, not eager | `MeshTraceId(3)` |
| it consumes the decode trace's logits tensor | `slot["input"] is generator._trace_logits` → **True** |
| `tt_out_tok` **is** the decode token input | **True** |
| the sampled token becomes the next input on device | 45116 → **25** → **1102** |
| position advances on device | 128 → **129** → **130** |
| nothing is staged between replays | token 0, position 0, page table 0 |
| greedy is the top-k op path | `k=1, p=0, 1/temp=1`, `force_argmax=False` |
| top-k/top-p uses the same path | different tokens, same traces |
| a sampled request does not corrupt greedy | identical to before |

### Greedy sampler choice, re-benchmarked for this mesh

Re-run on this stage's build, since the LM head now hands the sampler its logits
through a different sequence of ops ([`sampler_ab.json`](sampler_ab.json),
[`logs/sampler_ab.log`](logs/sampler_ab.log)). Every arm samples the same four tokens,
so this is latency only:

| arm | sampling trace ms | token-out ms (2 layers) |
| --- | --- | --- |
| **topk split to 2 x 32768 (shipped)** | **0.6323** | **2.140** |
| `max_top_k=8`, pad to power of 2 | 0.7942 | 2.274 |
| `max_top_k=8`, no pad | 0.7943 | 2.325 |
| no split: single-core topk over 50688 | 9.7295 | 11.270 |
| `max_top_k=32`, no pad, no split | 9.7294 | 11.274 |
| `max_top_k=32`, pad to pow2, no split | 13.0082 | 14.547 |

The shipped split is still the winner, and by 15x over the nearest non-split arm. The
sampling trace is **2.7 % of the token-out step**, so no sampler op dominates it: the
largest single row in the sampling window is `TopKDeviceOperation` at 284 µs over its
two multi-core pieces ([`tracy/sampling_perf_report.txt`](tracy/sampling_perf_report.txt)),
1.2 % of the step.

Force-argmax remains **off** and **unreachable** rather than merely undesirable: it
needs a full-vocab all-gather (`[1,1,32,202752]` bf16, 12.9 MB per step) through
`self.tt_ccl.get_and_cycle_ag_semaphore_handles(...)`, and this port constructs
`SamplingGenerator` with `tt_ccl=None` because a `TT_CCL` puts 36 more global
semaphores in the main L1 pool that the decode step has 7,296 B of headroom for. With
`tt_ccl=None` the arm does not error, it **hangs**, which is why it is not in the table
above. The greedy benchmark is therefore semantically greedy on both sides — `k=1`
through `ttnn.sampling` against the same `k=1` on slower topk geometries — not a
generic sampled path standing in for greedy.

## Accuracy

Against the full-model stage's AIME24 chat-template reference and its fp32 control,
both unchanged and both re-scored on this build
([`evidence_accuracy.json`](evidence_accuracy.json),
[`evidence_fp32_gate.json`](evidence_fp32_gate.json)):

| gate | reference | top-1 | top-5 | top-100 |
| --- | --- | --- | --- | --- |
| prefill (`run_prefill_check`) | bf16 | 0.990 | **1.000** | **1.000** |
| prefill (`run_prefill_check`) | fp32 control | 0.990 | **1.000** | **1.000** |
| decode (`run_teacher_forcing`) | bf16 | 0.990 | **1.000** | **1.000** |
| decode (`run_teacher_forcing`) | fp32 control | 0.990 | **1.000** | **1.000** |

**Both bars clear on both gates against both references: `top-5 >= 98 %` and
`top-100 = 100 %`.** The single non-top-1 position is the same one the full-model stage
records — `gen_index 64`, where TT picks the reference's **rank 1** with a 2.0-logit gap
to its own runner-up ([`evidence_misses.json`](evidence_misses.json)) — and there are
**zero** positions outside the reference's top 100. Identical to the full-model stage to
three decimals, which is what a layout-only change should produce.

Teacher forcing also reports its own decode rate, and this stage's three runs give
**37.07** (`evidence_accuracy.json`) **/ 37.28 / 38.15** (`evidence_fp32_gate.json`,
its two entries) **t/s/u** against the full-model stage's 36.88 / 37.10 / 37.99.
The ranges overlap, so no improvement is claimed from them; the token-out and
logits-only rows are the measurement. Its *first* entry reads TTFT 162–182 ms because
that entry pays decode-trace capture inside the window, which is why the headline TTFT
comes from the separately warmed perf measurement.

## Qualitative

The shared suite, chat-templated, 6 prompts x 128 tokens, HF control reused from the
full-model stage (same checkpoint, tokenizer, prompt set and parameters — the copied
[`qualitative/qualitative_prompt_format.json`](qualitative/qualitative_prompt_format.json)
is the proof) — [`logs/qualitative_tt.log`](logs/qualitative_tt.log),
[`logs/qualitative_compare.log`](logs/qualitative_compare.log):

| | worst | threshold |
| --- | --- | --- |
| TT adjacent duplication | **0.0** | 0.10 critical |
| TT trigram loop fraction | 0.117 (HF 0.117 on the same prompt) | — |
| TT non-ASCII fraction | 0.0016 (HF 0.0017) | — |

And the comparison that matters for an *optimization* stage: **all six TT completions
are byte-identical to the full-model stage's**
([`qualitative/qualitative_tt_vs_full_model_stage.json`](qualitative/qualitative_tt_vs_full_model_stage.json)).
Not "similar", not "same PCC" — the same 406/716/638/609/556/682 characters.

The runner-side degeneracy gate passes on freshly regenerated free-running output
([`logs/check_degenerate_output.log`](logs/check_degenerate_output.log)):
`No degenerate output detected`, adjacent duplication 0.0 on both the chat and raw
prompts.

## Runtime fallback audit

[`evidence_accuracy.json`](evidence_accuracy.json)`:fallback_audit`. Counters over 33
generated tokens, i.e. 32 decode steps:

| counter | total | per token |
| --- | --- | --- |
| trace replays | 32 | 1.0 |
| token refreshes | **1** | **0.0** |
| position / RoPE refreshes | **1** | **0.0** |
| page-table refreshes | **1** | 0.031 |
| synchronizations | **0** | **0.0** |
| readbacks | 33 | 1.0 (the caller asked for the tokens) |

| `device_position_advances` | **0** | see below |

One token/position stage for the post-prefill reseed, one page-table copy per request,
and one 32-uint32 readback per token because `generate()` returns tokens.

`device_position_advances: 0` sitting next to "positions advance on device" reads as a
contradiction and is not one — round 4 of the stage review asked, and the answer is worth
recording. The counter is incremented where the `ttnn.plus_one` is *built into the graph*,
which happens once at trace capture, and `reset_counters()` runs after capture and before
the measured window. Replays execute the op without re-entering the Python that counts it,
which is precisely the property `trace replays: 32` with `position refreshes: 1` is
asserting. The device-side advance itself is pinned by
`test_steady_state_decode_does_no_per_token_host_work` and by the two `PlusOneDeviceOperation`
rows (3143, 3254) inside the traced decode window, not by this counter. The token-out
path has no host argmax, no full-logits readback, and no untraced sampling. The three
host-logit boundaries that exist are all explicit and all outside the measured path:
`prefill_forward()` by contract, `generate(host_sampling=True)` as a compatibility
mode, and `decode_forward(sample_on_device=False)` for callers that sample themselves.

### Watcher

Two separate concerns, and the second is a finding rather than a formality.

**The shipped default path: `WATCHER_CLEAN`.** Ten device cases covering all three
decode changes, `TT_METAL_WATCHER=10`, no profiler in the same run
([`bench/run_watcher.sh`](bench/run_watcher.sh), [`logs/run_watcher.log`](logs/run_watcher.log)),
verdict re-derived from the committed log rather than asserted
([`logs/check_watcher.log`](logs/check_watcher.log)):

```
watcher/watcher.log.gz: 6991 lines
  dump boundaries: 12 (min 2)
  kernel id lines: 3169 (min 100)
  stack usage rows: 100 (min 10)
  device attach: 4 (min 1)
  device detach: 4 (min 1)
  fatal watcher messages: 0
WATCHER_CLEAN
```

All ten pass, zero tripped asserts. The process then dies with `SIGABRT` at mesh close on
*"Timed out while waiting for active ethernet core 29-25 to become active again"* — the
same watcher-only teardown fault the full-model stage recorded as its limitation 11,
unchanged by this stage and absent from every non-watcher run. The devices recover
(`tt_reset.py` → `RESET_DONE failures=0`).

**The watcher caught one real defect, and it was in the test rather than in the model.**
`test_prefill_trace_survives_rebinding_the_same_external_cache` originally freed the
cloned KV cache **before** releasing the trace that held its addresses — a use-after-free,
which on this fabric shows up not as a wrong number but as

```
Device 0 acteth core(x= 0,y= 9) virtual(x=29,y=25): subordinate_erisc detected invalid
NOC command buffer state before starting the next kernel (write-capable NOC packet tags
must be zero so implicit transaction ID users start with transaction ID 0).
Current kernel: tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp.
```

([`logs/watcher_bisect_rebind.log`](logs/watcher_bisect_rebind.log) before,
[`logs/watcher_bisect_rebind_fixed.log`](logs/watcher_bisect_rebind_fixed.log) after.
Draining the device before the release was tried first and did **not** move it; the
ordering did.)

**A second cause survives, and rounds 3 and 4 of the stage review are why it is now
measured rather than argued.** This section has carried three different statements of it,
and the first two were wrong in instructive ways.

Round 3 rejected the original claim — "releasing a prefill trace and then building and
running another model on the same mesh trips the assert" — because the four arms of
[`bench/prefill_trace_release_probe.py`](bench/prefill_trace_release_probe.py) supporting
it (**capture**, **release**, **recapture**, **clone-and-free the 104 KV-cache tensors**)
are all *negative* controls: none builds a second model after a release, so a clean result
from them localises nothing. That was right, and running the missing configurations
disproved the claim outright.

Round 4 then rejected the replacement claim — "the trigger needs the opt-in prefill-trace
cases **and** the larger preceding workload" — because every configuration that tripped had
twelve cases and every clean one had ten, so the attribution to the prefill-trace cases was
confounded with plain process length. That was also right. The missing control is twelve
device cases *without* those two, and it has now been run three times.

**Twenty-four watcher processes across five configurations** (25 including the
single-purpose `--arm rebuild` row below, which is not one of the five), each with a device
reset before it. The count is the `runs` column of the table, and round 7 caught that this
sentence had said twenty-eight — a figure that resolved to nothing, held in place by a gate
check that matched the *word*. The gate now sums the column instead. Rounds 3, 4 and 5 each attributed this to something the next round's missing
control refuted, so round 6's design adds the two arms that were still absent: a **work-matched**
twelve-case arm, and the **opt-in pair alone** repeated.

| configuration | 12 cases | builds an extra generator | clones/frees the 104 cache tensors | captures a **prefill** trace | runs | tripped |
| --- | --- | --- | --- | --- | --- | --- |
| the ten gated cases | | | | | 5 | **0** |
| the two opt-in `prefill_trace` cases **alone** | | ✓ | ✓ | ✓ | 4 | **0** |
| `--arm rebuild` (release a prefill trace, then build and run a second generator) | | ✓ | | ✓ | 1 | **0** |
| twelve: the ten + two other **sampling** cases (count-matched) | ✓ | | | | 6 | **2** |
| twelve: the ten + `decode_follows_the_cache…` + a sampling case (**work-matched**) | ✓ | ✓ | ✓ | | 3 | **0** |
| **twelve: the ten + both opt-in `prefill_trace` cases** | ✓ | ✓ | ✓ | ✓ | 6 | **6** |

The work-matched arm is the one round 6 asked for. Its extra case
(`test_decode_follows_the_cache_it_is_rebound_to_after_the_trace_is_captured`) does
everything the opt-in cases do — builds its own `reuse=False` generator, clones and frees all
104 KV-cache tensors, captures **and releases** a trace — except that the trace is the
*decode* trace, not a prefill one. It is **0 of 3**.

| contrast | rates | two-sided Fisher |
| --- | --- | --- |
| opt-in twelve vs **everything else pooled** † | 6/6 vs 2/18 | **p = 0.00021** |
| opt-in twelve vs ten | 6/6 vs 0/5 | p = 0.0022 |
| opt-in twelve vs the **work-matched** twelve | 6/6 vs 0/3 | **p = 0.0119** |
| opt-in twelve vs the pair **alone** | 6/6 vs 0/4 | p = 0.0048 |
| opt-in twelve vs the count-matched twelve | 6/6 vs 2/6 | p = 0.061 |
| work-matched twelve vs count-matched twelve | 0/3 vs 2/6 | p = 0.500 |

† The pool is **heterogeneous** — 2-case, 10-case and 12-case arms — and **both** of its
trips come from the single arm that is the design's own matched control, so pooling dilutes
33 % to 11 % and turns the matched contrast (p = 0.061) into p = 0.00021. It also omits the
one `--arm rebuild` run, which is clean; including it gives p = 0.000158, so the omission is
conservative but was unstated before round 7. **The contrast to weigh is the matched one at
p = 0.061**; the pooled figure is reported because it is what "against everything else we
tried" means, not because it is the stronger inference.

These are six post-hoc contrasts on one run set and carry **no multiplicity correction**;
Bonferroni at ×6 leaves the primary contrast at p = 0.0013 and the work-matched one at
p = 0.071. Fisher also assumes independence, and these are sequential runs on one physical
mesh of a device-state phenomenon — the resets between runs are the only thing standing in
for that assumption. Treat the ordering as solid and the exact p-values as indicative.

**What the five arms support: the opt-in pair takes the rate from a background of roughly
0–33 % to 100 %.** That is weaker than "an interaction, and neither half alone is
sufficient", which is what this section said before round 7 pointed out that the stage's own
control arm refutes it:

* the pair **alone** is 0 of 4 — so the prefill trace by itself has not reproduced it;
* **a preceding workload alone *is* sometimes sufficient**: the count-matched twelve — twelve
  cases, no prefill trace, no extra generator, no cache churn — is **2 of 6**. Any claim that
  the trip *requires* the prefill-trace cases is contradicted by those two runs;
* the two together are **6 of 6**, which is the only configuration that reproduces every time.

The work-matched arm's 0 of 3 is therefore **not** an exclusion of "process length, generator
churn or trace lifecycle in general" — the document said that and it was wrong. 0 of 3 against
the count-matched 2 of 6 is **p = 0.500**: the two twelve-case arms are statistically
indistinguishable, and a 0-of-3 arm cannot exclude what a 2-of-6 arm demonstrates. What the
work-matched arm does establish is narrower and still useful: whatever raises the rate to
100 % is **not** reproduced by building an extra generator, cloning and freeing the whole
cache, and capturing and releasing a *decode* trace, in three attempts.

So the directional finding — the opt-in pair is associated with a large rate increase over a
non-zero background — is solid. The mechanistic one is not available from this design, and the
background rate is the reason: with a ~33 % baseline in the matched arm, separating "raises
the rate" from "is required" needs far more runs than 6 per arm.

That is, with the controls it always lacked, close to the round-3 statement that round 4
retracted. The honest reading of the sequence is not that any one round was careless but
that four of the five earlier claims were **underpowered or uncontrolled in a way the next
round's single missing arm exposed**:

1. "release then build another model" — refuted by `--arm rebuild` and the pair alone;
2. "the opt-in cases plus a larger workload" — retracted as confounded with length;
3. "length, and composition is not separable" — the pooling was circular and the 3-vs-3
   table's minimum attainable p is 0.100, so it could not have separated anything;
4. round 5's "composition, not length" — the significant contrast was still confounded in
   both dimensions, and no arm held work constant;
5. **this one** — with a work-matched arm and a pair-alone arm, the interaction is what
   survives, and the two unexplained trips in the count-matched arm are the residual
   (0/3 work-matched against 2/6 count-matched is itself p = 0.500, i.e. those two trips are
   not attributable to anything this design isolates).

What none of it explains is *why* the combination matters when each half is clean, so it
remains **not root-caused**, and the scope is exactly "both opt-in cases in a process that
has already done substantial other device work".

**What "below the model" rests on.** Round 5 was right that the phrase had been used
without testing the in-model candidates: a test fixture that closed the mesh over live
traces, and a `teardown()` that deferred a 3.24 MB/device free to Python GC and drained
nothing on the shipped default. Both are fixed — one release path with an unconditional
drain, a fixture finalizer, `close_multichip_mesh` also dropping the model's semaphore
cache, and `MuseGlimmerModel.deallocate()` warning rather than silently freeing a cache
under a live trace — and both twelve-case arms were re-measured against the fix. **Neither
moved** (3/3 → 3/3 and 1/3 → 1/3). Those are two 3-vs-3 tables and they cannot *exclude*
anything on their own — round 6 was right about that too, and the earlier wording here
claimed they did. What they do show is that the model-side lifecycle is now deterministic
and the trip survives it, which is the most this stage can say: the remaining mechanism is
not in the release ordering, the drain, the fixture, or the semaphore cache, because all
four changed and the rate did not. What it *is* remains unidentified.

Two properties bound it. **All twelve tests pass first** — the assert fires 1–4 s after the
last `PASSED`, in the watcher's teardown poll, so nothing produces a wrong number and no
gate in this stage is affected. And **it damages the artifact**: the abort lands inside the
watcher's own dump, leaving `watcher.log` truncated with no detach lines, which
`check_watcher.py` correctly rejects as `WATCHER_LOG_NOT_A_REAL_RUN`. That truncation also
made it report `fatal watcher messages: 0` for a run that tripped, so
[`bench/run_watcher.sh`](bench/run_watcher.sh) now greps the console for the assert directly
and writes `check_watcher_console*.log`; one inference fewer.

What follows for the shipped code:

* the gated set is **ten cases**. With composition rather than length as the discriminating
  factor, that is no longer just "the largest size measured clean" — the two cases the gate
  excludes are the two the data implicates, and the ten it keeps are 0 of 5. The opt-in pair
  is watched **together in its own process** (0 of 1), so the trace-lifecycle code this stage
  ships is still under watcher, which is what `$optimize` asks for with async CCLs in play;
* the shipped default never captures a prefill trace, so **none of this is on its path** —
  which is why the gated ten and the 55-case suite are unaffected either way;
* the operational note for a serving stage is now specific rather than the withdrawn
  "reset between builds": **with `prefill_trace=True`, expect a watcher-enabled process that
  also runs a substantial other workload to abort at teardown.** Reset afterwards, never read
  a truncated watcher log as a clean one, and if a long-lived server needs watcher coverage,
  give the prefill-trace path its own process;
* `_invalidate_traces_if_cache_moved` **does not retire** prefill tracing, and that is
  independent of all of the above: the trip is at teardown, `teardown()` releases regardless,
  so retirement removed no exposure and only cost a serving caller its flag after one rebind;
* `--arm rebuild` stays in the probe. It is a clean arm, but it is the arm that disproved the
  first statement of this limitation, and it is one command for the next stage.

Artifact note: of the five clean ten-case runs, two keep their `watcher.log.gz` for
re-derivation ([`watcher/`](watcher), [`watcher_default10/`](watcher_default10)) and three
are console-verdict-only, as are the three length-control runs; `run_watcher.sh` clears its
output directory per run and only the tagged directories were kept.

The shipped default never captures a prefill trace, so none of this is on its path.

### The allocator's active-trace warning

Every device run in this stage and the previous one logs

```
Allocating device buffers is unsafe due to the existence of an active trace.
These buffers may be corrupted once a trace is executed.
```

once per thread (`tt_metal/impl/allocator/allocator.cpp`), and it is classified here
rather than left as noise. It is emitted whenever a buffer is allocated after any trace
has been captured, which this model does by construction: the sampler's persistent
tensors are allocated after the decode trace, and with `prefill_trace=True` the prefill
trace is captured before both. It is **inherited, not introduced** — it appears in the
shipped-default logs ([`logs/evidence_perf.log`](logs/evidence_perf.log),
[`logs/sampler_ab.log`](logs/sampler_ab.log)) exactly as it does in the full-model
stage's. What makes the allocations safe is that they are made *before* the traces that
read them are replayed and are never freed while a trace holds them — which is the
contract `_release_prefill_traces` now enforces explicitly, and which the watcher
verified: 0 tripped asserts across the shipped-default set and both opt-in cases, and a
real violation of it (a test freeing a cloned cache before releasing the trace holding
its addresses) was caught by the watcher rather than by a wrong number.

## Capability and batch

Unchanged, and re-derived from this stage's evidence rather than inherited:
[`../context_contract.json`](../context_contract.json) is rebuilt by
[`bench/refresh_context_contract.py`](bench/refresh_context_contract.py) from this
stage's `evidence_*.json` and validates at the full HF context:

```
Context contract OK: target=131072, supported=131072 (full HF context).
```

| per device | bytes | |
| --- | --- | --- |
| 52 layers of weights | 4,327,784,448 | 4.33 GB |
| embedding + LM head + terminal norms | 863,073,536 | 0.86 GB |
| shared RoPE tables | 134,217,728 | 0.13 GB |
| KV cache (52 layers, 131072 tokens) | 1,853,882,368 | 1.85 GB |
| **total long-lived** | **7,178,958,080** | **7.18 GB** of 31.46 GiB |

**The long-lived DRAM budget is identical to the full-model stage's to the byte** — none
of the three changes allocates anything that outlives a decode step. But an earlier
version of this section said "none of the three changes allocates anything", and the
stage review was right that that is false: `ttnn.tanh` and `ttnn.multiply` have no
in-place form here, so change 1 replaces two DRAM-interleaved transients with two
**width-sharded L1** ones, inside the captured decode trace. Measured rather than
argued ([`l1_highwater_probe.json`](l1_highwater_probe.json),
[`logs/l1_highwater_probe.log`](logs/l1_highwater_probe.log)):

| | peak L1 allocated per bank | free at that peak |
| --- | --- | --- |
| softcap in DRAM (before) | 90,112 B | 1,365,120 B |
| **softcap in L1 (shipped)** | **217,088 B** | **1,238,144 B** |
| delta | **+126,976 B** = 2 x 63,488 (32 rows x 992 padded columns x 2 B) | |

So change 1 costs **8.7 % of the 1,455,232 B L1 bank** at the terminal path's own peak.
The 1.24 MB per bank left free at that peak is measured on an **otherwise-idle device**
(the probe's baseline is 0 B allocated), so it is free-in-isolation rather than free in
the real traced step, which also holds the boundary residual, the sampler's tensors and
the CCL semaphores. The delta is the sound number; the headroom claim rests on the
co-residency evidence below rather than on that figure. The "7,296 B of headroom" figure this document
quotes in the force-argmax section is a different pool and a different moment — it is
what a `TT_CCL`'s 36 extra *global semaphores* would have to fit into, at the decoder's
tightest in-layer point, not what the terminal path has after 52 layers have freed their
intermediates. The two are not in conflict, and the empirical co-residency evidence is
that the traced step runs clean: 55 tests, the batch-32 mixed-length case, and a
watcher-clean run over all three changes.

Batch: batch 1 at 131072 is the primary target and is tested; batch 4 and batch 32 at
1024 with mixed per-user prompt lengths pass through the low-level API
(`test_batched_prefill_and_decode_with_mixed_lengths`, in this stage's suite run and in
the watcher run). Decode always runs 32 rows whatever the batch, and inactive rows carry
`current_pos = -1`.

## Prompt lengths

Preserved, and re-verified through the optimized generator: 1, 37, 127, 129, 2049,
4097, 8193, 12345 and **130073** tokens all prefill and decode through the public path
([`evidence_accuracy.json`](evidence_accuracy.json)`:prompt_shapes`,
[`evidence_perf.json`](evidence_perf.json)), none of them divisible by the tile, the
64-token page or the 8192-token prefill chunk. `test_prefill_accepts_any_logical_prompt_length`
covers nine lengths in the suite. There is no `seq_len % chunk == 0` assertion anywhere
in the public path.

## Tests

| run | result |
| --- | --- |
| `pytest tests/test_full_model.py` | **55 passed** — [`logs/full_test_run.log`](logs/full_test_run.log), [`test_results.xml`](test_results.xml) |
| the same 55 in reverse order, one process | **55 passed** — [`logs/full_test_run_reverse.log`](logs/full_test_run_reverse.log) |
| watcher, gated subset (10 shipped-default cases) | **10 passed**, `WATCHER_CLEAN` — [`logs/watcher_pytest.log`](logs/watcher_pytest.log), [`watcher/`](watcher), [`logs/check_watcher.log`](logs/check_watcher.log) |
| watcher, each opt-in prefill-trace case alone | **1 passed** each, `WATCHER_CLEAN` |
| watcher, both opt-in cases together in one process | **2 passed**, `WATCHER_CLEAN` — [`logs/watcher_pytest_prefill_trace_pair.log`](logs/watcher_pytest_prefill_trace_pair.log) |
| watcher, ten cases repeated (limitation 6's 0-of-5 arm) | **10 passed**, `WATCHER_CLEAN`, x4 more — [`watcher_default10/`](watcher_default10), `logs/check_watcher_console_10case_rep{a,b,c}.log` |
| watcher, twelve in one process — ten + both opt-in cases | **12 passed** every time, then a tripped assert at teardown, **6 of 6** across pre- and post-teardown-fix — `logs/watcher_pytest_12case_tripped{,_run2,_run3}.log`, `logs/watcher_pytest_postfix_optin{1,2,3}.log` |
| watcher, twelve in one process — ten + two *other* sampling cases (the length control) | **12 passed** every time, tripped **2 of 6** — `logs/watcher_pytest_12case_control{,2,3}.log`, `logs/watcher_pytest_postfix_ctrl{1,2,3}.log` |
| watcher, twelve in one process — ten + `decode_follows_the_cache…` + a sampling case (**work-matched**: extra generator, cache churn, a decode trace captured and released) | **12 passed** every time, tripped **0 of 3** — `logs/watcher_pytest_workmatched{1,2,3}.log` |
| watcher, the opt-in pair **alone** in one process | **2 passed** every time, tripped **0 of 4** — [`logs/watcher_pytest_prefill_trace_pair.log`](logs/watcher_pytest_prefill_trace_pair.log), `logs/watcher_pytest_pairalone{1,2,3}.log` |
| `test_the_live_trace_count_round_trips_over_both_trace_kinds` | the model's live-trace count balances across capture and release of both trace kinds, and clamps rather than going negative |
| the layer-stack floor at the benchmark's own context | sliding 0.4390, full 0.4077 ms/layer → **22.421 ms** — [`logs/layer_ab_after_ctx256.log`](logs/layer_ab_after_ctx256.log) |

Nine cases are new (three of them parametrizations of one test) and each pins one shipped change:

| test | pins |
| --- | --- |
| `test_lm_head_softcap_runs_in_l1_and_matches_the_dram_form` | change 1 is bit-identical to the DRAM order, finite, and bounded by `T` |
| `test_decode_embedding_gathers_straight_into_the_boundary_layout[7,0,202047]` | change 2 returns the boundary layout **and the same values as the interleaved gather**, over four repeats per token id |
| `test_swiglu_multiply_runs_on_the_wide_grid_and_returns_the_narrow_one` | change 3 leaves the MLP's boundary output layout unchanged |
| `test_prefill_trace_is_opt_in_and_matches_the_eager_path` | the opt-in prefill trace: same tokens, one bucket, non-aligned lengths inside it, eager fallback past the bound, released on teardown |
| `test_prefill_trace_survives_rebinding_the_same_external_cache` | the serving path: the **same** external cache every request keeps its trace; a cache bound to **different** buffers releases it, recaptures against the new ones and still answers correctly |
| `test_decode_follows_the_cache_it_is_rebound_to_after_the_trace_is_captured` | the **decode** trace bakes the same cache addresses, so a rebind to different buffers must release and recapture it. Against the pre-fix code it fails, and that failure is committed as a negative control ([`logs/decode_rebind_prefix_negative_control.log`](logs/decode_rebind_prefix_negative_control.log): `AssertionError: a moved cache must release the decode trace`, with the shipped source restored afterwards. It is a **partial** revert — the signature is still recorded, only the comparison and the release are removed — because a full revert to `5e6022db622` fails on a missing attribute instead of on the behaviour). Two of its assertions do different jobs: that one catches the missing release, and the last one — the **traced** decode off the rebound cache must agree with the **eager** decode off the same cache — catches a release that recaptured against the wrong buffers. The committed control trips the first, because that is the one the pre-fix code violates |
| `test_the_live_trace_count_round_trips_over_both_trace_kinds` | the model's live-trace count balances across capture and release of both kinds, and the clamp is exercised directly rather than asserted about |

## Limitations and known issues

1. **Batch-1 TTFT is host-dispatch bound; the fix ships but is opt-in.** ~17 ms of the
   ~64 ms default TTFT is host issue over 4122 ttnn dispatches, and the 209 collective
   dispatches are 33 % of it. `prefill_trace=True` removes 21 % of TTFT (63.68 →
   50.19 ms) with bit-identical logits, and is **off by default** because one trace
   serves one 32-row padded-length bucket and capture costs 98 ms against a ~15 ms
   per-replay saving — a win for a caller that repeats or buckets prompt lengths, a
   one-time cost for one that does not, and the generator cannot tell which it is. DRAM
   per bucket scales with the padded row count (3.3 MB at 128 rows, ~210 MB at 8192), so
   `prefill_trace_max_entries` bounds it. A vLLM stage that buckets prompt lengths
   should turn it on. Full evidence in
   [Where TTFT actually goes](#where-ttft-actually-goes).

2. **Persistent CCL staging buffers would be worth ~2 ms of TTFT and are unavailable.**
   At the model's own BFP8 payload and 4-worker prefill setting they cut the
   reduce-scatter's host cost by 14–17 % (62.12 against 72.10 µs/call in a hot loop,
   96.86 against 117.05 loaded). The decoder stage rejected them on an intermittent
   first-use correctness race that moved between arms and between runs of the same arm,
   which an intermittently wrong first token makes non-negotiable for a 52-layer stack.
   This stage's earlier claim that they are "within noise" was measured at a BF16 payload
   and is withdrawn. Worth re-testing when the op's first-use contract changes.
3. **The MLP gate/up output grid is pinned at 16 cores by an op contract.** The
   DRAM-sharded matmul needs `K_tiles % cores == 0` on both 208 and 160 tiles, whose
   divisor sets intersect at {8, 16}. This stage widens the SwiGLU multiply by reshard
   instead, which costs 5.91 µs to save 13.3 µs. If the op gains uneven sharding, the
   reshards become unnecessary overhead and should be deleted rather than inherited —
   they are one constant (`DECODE_SWIGLU_MUL_CORES`) and one branch.
4. **`tt-perf-report`'s merged device time is ~1 % above the traced replay.** The tool
   takes the max per op across the four devices, so the summed critical path is
   pessimistic. Every device-time figure here inherits that; the end-to-end figures do
   not.
5. **The decode window had to be profiled one layer kind at a time.** The 80-core
   SwiGLU multiply emits markers per core, and the two-layer window overflowed the
   profiler's DRAM marker buffer at `ITERS=1` (20 dropped-marker lines, preserved as
   [`logs/run_tracy_two_layer_overflow.log`](logs/run_tracy_two_layer_overflow.log)).
   One real layer of each kind in its own capture is the same coverage with half the
   markers, and the integrity check in [`bench/run_tracy.sh`](bench/run_tracy.sh) fails
   the run rather than printing.
6. **Both opt-in `prefill_trace` cases in a watcher process that has already done
   substantial other device work trip a fabric ERISC assert at teardown.**
   `subordinate_erisc detected invalid NOC command buffer state … fabric_erisc_router.cpp`,
   **after every test in the process has passed**, in the watcher's teardown poll.
   **6 of 6** for that configuration. **It is a rate increase over a non-zero background, not
   a requirement.** The length-matched control — twelve cases, *no* prefill trace, no extra
   generator, no cache churn — trips **2 of 6** on its own, so a long watcher process can abort
   at teardown with no prefill trace involved at all. The pair **by itself** is 0 of 4, and a
   **work-matched** twelve-case process (its own extra generator, all 104 cache tensors cloned
   and freed, a trace captured *and released* — but a **decode** trace) is 0 of 3. Pooled
   against everything else the opt-in arm is 6/6 vs 2/18 at Fisher **p = 0.00021**, but that
   pool is heterogeneous and both its trips are the matched control's; the contrast to weigh is
   the **case-count-matched** one at **p = 0.061**, and 0-of-3 against 2-of-6 is p = 0.500.
   See [Watcher](#watcher) for all five arms, six contrasts and the multiplicity caveat.
   *This is the sixth statement of this limitation and it is close to the third; the four in
   between were each refuted by one control the next round found missing, which is recorded
   in the Watcher section rather than smoothed over.* Consequences are bounded: **no test
   fails, no number is wrong**, and the shipped default never captures a prefill trace so
   never releases one. What breaks is the watcher *artifact* — the abort lands inside the
   watcher's own dump, truncating the log, which `check_watcher.py` correctly rejects as
   `WATCHER_LOG_NOT_A_REAL_RUN`. Probably the same fault as limitation 7 (same teardown,
   same acteth cores 29-25 and 25-25, watcher-only). **Not root-caused**: four in-model
   candidates were fixed without changing the rate, so it is below the model, but the
   mechanism is unidentified. The gated set is the ten cases and the opt-in pair is watched
   in its own process; a serving stage enabling `prefill_trace` under watcher alongside other
   work should expect the teardown abort, reset afterwards, and never read a truncated
   watcher log as a clean one. Giving the prefill-trace path its own process is **necessary
   but may not be sufficient** — the 2-of-6 length-matched control says a long watcher process
   can abort at teardown with no prefill trace anywhere in it.

7. **A watcher-enabled run still aborts at device close.** Inherited unchanged from the
   full-model stage (its limitation 11); watcher-only, recoverable with a reset, not
   root-caused, and it does not touch the shipped path.
8. **Change 1 raises the decode step's peak L1 by 126,976 B per bank** — 8.7 % of the
   bank, measured, leaving 1.24 MB free at that peak. It is the price of running the
   softcap on the shard instead of in DRAM, `ttnn.tanh`/`ttnn.multiply` having no
   in-place form here. Not a risk at the tested batch/context, but it is a real increase
   in a resource this model is documented as being tight on elsewhere, so it is named
   rather than folded into "allocates nothing".

9. **Two prefill norms run on 1 and 4 cores for ~134 µs each.** `ttnn.rms_norm` on a
   DRAM-interleaved input parallelises over tile rows, and the terminal norm and the
   embedding norm both run on a 32-row slice; the same norm width-sharded in decode is
   8.8 µs. That is 0.27 ms of a 65 ms TTFT, and it is left because the fix changes
   prefill numerics on the accuracy gates' critical path for 0.4 % of a figure whose
   process-to-process spread is 8 %. Priced in
   [Where TTFT actually goes](#where-ttft-actually-goes) so a later stage can take it
   with the accuracy re-run it needs.
10. **The full-model stage's own limitations are inherited unchanged**: the
   `max_batch_size`/`max_seq_len` DRAM trade, non-tile-aligned prompts writing unread
   zero K/V past the logical length, `TTPenalties`' unused ~45 MB, log-probs being
   unavailable on 4 devices, greedy reproducibility being guaranteed after `reset()`,
   the chat template embedding the current date, the embedding all-gather being
   mitigated rather than root-caused, the two default-off shared sampling knobs,
   `max_top_k > 32` being untested, and `num_gather_links` being derived as
   `max_top_k // 32` upstream. See [that document](../full_model/README.md) for each
   one; none of them is changed by this stage.

## `$optimize` / `$multichip` checklist, with where the evidence is

| item | status | evidence |
| --- | --- | --- |
| decode path fully traced, no host fallback | yes | fallback audit: 0 token/position/sync refreshes per token; two traces account for the step to 10 µs (9.9 µs, re-derived by the figure gate from `evidence_perf.json` rather than transcribed) |
| decode activations width-sharded in L1 across norm/attention/residual/MLP/output | yes | `tracy/decode_sliding_perf_report.csv`: every matmul `in0` is `L1_WIDTH_SHARDED`; the boundary is a fixed point at every layer |
| prefill activations DRAM interleaved, 2D matmul program configs for large prefill matmuls | yes | `tracy/prefill_128_perf_report.csv`; `mcast2d` specs in `multichip_decoder.py` |
| operation-topology audit recorded | yes | [above](#operation-topology-audit) |
| multi-device topology candidates measured as coherent families | yes | 12 collective arms in `ccl_host_probe.json` (implementation x worker count x persistence x wrapper/primitive/fused), all at the real payload |
| lower-movement residual candidates measured without an old-contract restore | inherited | the residual contract is preserved by the goal; the decoder stage owns the fractured-residual family (`fractured_decode_probe.py`) and this stage did not re-open it |
| best-candidate comparison against the strongest prior artifact | yes | `--baseline` reproduces the full-model stage's committed numbers to 0.14 % (23.844 vs 23.811 token-out, 23.164 vs 23.164 logits-only) before the changes are applied |
| final default reproduces the selected candidate | yes | predicted 22.81 from the reduced A/B, measured 22.656 on the all-layer default |
| dtype/fidelity policy verified in the measured rows | yes | [the row table above](#carried-forward-decoder-contract-unchanged); LM head is `BF16 x BFP4 => BF16` LoFi, MLP `BFP4`, attention `BFP8`, cache `BFP8` |
| SDPA / optimized composite ops used | yes | `SdpaDecodeDeviceOperation`, `nlp_create_qkv_heads_decode`, `nlp_concat_heads_decode`, `rotary_embedding_hf`, `paged_update_cache`, `paged_fill_cache` |
| repeated same-input projections packed or rejected with evidence | yes | QKV already packed; `attn_gate` rejected with DRAM% and layout evidence in the audit |
| `memory_config` / `program_config` / `compute_kernel_config` explicit on important ops | yes | `MULTICHIP_DECODE_MATMUL`, `LM_HEAD_*`, `LayerNormShardedMultiCoreProgramConfig`, `SDPAProgramConfig`, LoFi kernel config |
| dominant matmul program-config sweep, incl. larger legal `in0_block_w` | inherited + extended | decoder stage's table (13/13/10/2 = the largest legal divisor per role); this stage measured the only other legal gate/up core count (`mlp8`, +2.6 %) and recorded the exact `K_tiles % cores` blocker |
| decode compute fidelity swept as a perf knob | inherited | decoder stage: `fid_hifi2` arm measured and rejected; LoFi ships and appears in every row |
| attention weight dtype/fidelity swept separately from MLP | inherited | decoder stage: `attn_bfp4` / `attn_bf16` arms |
| BFP4/LoFi trials for MLP gate/up and down | inherited | MLP is BFP4 in the measured rows |
| shard specs / core grids divide cleanly into tiles | mostly | 16-core boundary divides 208 tiles as 13; the SwiGLU wide grid divides 160 as 2; the two documented exceptions are the 40-tile QKV output (pad-not-wrong, pinned by a decoder-stage test) and the LM head's 975-column shard (pinned by the new bit-identical test) |
| DRAM-sharded decode matmuls | yes | all six projections plus the LM head |
| collective topology minimized | yes | 2 all-reduces per layer is the preserved contract; the embedding gather now writes its consumer's layout; the token-out path has no logits gather at all |
| fused matmul-CCL used or rejected with an adapted attempt | inherited | decoder stage's `fused_ccl_probe.py` |
| repeated decode CCLs use persistent buffers, or the reason is recorded | recorded | refuted twice: correctness (decoder stage) and host cost (this stage, `ccl_host_probe.json`) |
| MoE-specific items | n/a | dense MLP model |
| LM head + sampling in the optimized token-out path, terminal costs profiled separately | yes | terminal priced at 691 µs and sampling at 632 µs in `tracy/`; padded vocab masked; split TopK on power-of-two pieces; no `ArgMaxDeviceOperation`, no full-vocab all-gather, no host argmax |
| LM head optimized for DRAM-sharded matmul | yes | `dram_sharded`, cores=52, `in0_block_w=2`, BFP4 |
| reduced precision/fidelity experiments appropriate to this stage | yes | see [Performance accounting](#performance-accounting); frontier deferred to `$datatype-sweep` |
| performance accounting reconciled, `perf_summary.json` written | yes | [`perf_summary.json`](perf_summary.json) |
| batch capability preserved, larger batch tested to 32 | yes | `test_batched_prefill_and_decode_with_mixed_lengths[4,32]` in both suite runs and the watcher run |
| watcher clean, separate from profiler | yes | `WATCHER_CLEAN` on the ten-case gated set and on both opt-in prefill-trace cases run together; Tracy ran after a reset, in its own process. The one configuration that trips is limitation 6, it trips after every test has passed, and it is recorded with its positive control rather than excluded |
| context contract preserved | yes | 131072, unreduced, rebuilt from this stage's evidence |
| `$qualitative-check` after the selected optimization, with a control | yes | byte-identical to the previous-stage control on all six prompts |
| every reported figure resolves to a committed artifact | yes | `bench/check_reported_figures.py`, which also compares every literal it resolves back to the README text (digit-bounded, so `1.0` no longer matches `21.05`) and asserts its check count, the assertion/binding split within it, and the set of artifacts it actually **opened** — recorded by its readers, not tested with `is_file()`, which round 4 pointed out is an existence check wearing a coverage label (it caught two: `work_log.md` was read past the recorder, and `watcher_probe_rebuild/watcher.log.gz` was listed but never opened). Round 3 was right that a frozen count is not coverage: the gate passed 328/328 with six wrong figures in sections it never read, and the sections it missed — the prefill-128 capture, the audit table's µs column, the teacher-forcing rates and `work_log.md` — are all bound now |

## How to reproduce

One 52-layer build is ~160 s of host weight packing, so each device stage runs in one
process over one build.

```bash
M=models/autoports/meta_models_muse_glimmer_30b
B=$M/doc/optimized_full_model/bench

# before / after performance, from the same script (--baseline reverts the three changes)
python $B/evidence.py --stages capacity,perf --baseline --out evidence_perf_before.json
python $B/evidence.py --stages capacity,perf,shapes --shape-lengths 130073 --out evidence_perf.json
python $B/evidence.py --stages capacity,perf --prefill-trace --out evidence_perf_prefill_trace.json

# accuracy, split-sampling contract, prompt shapes, fallback audit
python $B/evidence.py --stages capacity,prefill,teacher,sampling,shapes,fallback --out evidence_accuracy.json

# the fp32 control gate and the per-position miss detail
python $B/evidence.py --stages prefill,teacher,misses \
    --reference readiness_aime24_chat.refpt,readiness_aime24_chat_fp32.refpt --out evidence_fp32_gate.json

# free-running generation, both prompt formats
python $B/evidence.py --stages autoregress --out evidence_autoregress.json
python models/common/readiness_check/check_degenerate_output.py \
    --model-dir $M --missing-artifacts critical --scope autoregressive

# the shared qualitative suite (HF control reused from the full-model stage)
python $B/qualitative.py --arm tt --reuse-hf-control
python $B/qualitative.py --arm compare --reuse-hf-control

# the decode A/B: the three changes, separately and cumulatively
python $B/decode_ab.py --arms full_model_stage,terminal_only,base --rounds 3 --replays 64 --out decode_ab_shipped.json
python $B/decode_ab.py --arms base,swiglu20,swiglu32,swiglu40,swiglu80,mlp8 --out decode_ab_swiglu.json

# the layer-stack lower bound, re-measured with the decoder stage's own harness
python $M/doc/optimized_multichip_decoder/bench/layer_ab.py \
    --candidates tp4,tp4b --prefill-seq 128 --decode-context 2048

# where TTFT goes
python $B/ttft_breakdown.py --lengths 128,256,512,1024,2048 --rounds 3 \
    --max-seq-len 131072 --layer-scan --out ttft_breakdown_before.json
python $B/prefill_host_probe.py --length 128          # issue vs drain, and a cProfile
python $B/prefill_opcount.py --length 128             # host time by op name
python $B/ccl_host_probe.py --rows 128 --reps 40 --dtype bfloat8_b \
    --out ccl_host_probe_bfp8.json                    # the model's payload
python $B/ccl_host_probe.py --rows 128 --reps 40 --dtype bfloat8_b --loaded-queue \
    --out ccl_host_probe_bfp8_loaded.json             # + one matmul in front
python $B/ccl_host_probe.py --rows 128 --reps 40 --dtype bfloat16 \
    --out ccl_host_probe_bf16.json                    # the superseded arm
python $B/prefill_trace_probe.py --length 128 --replays 10 --with-decode-traces
python $B/l1_highwater_probe.py

# the greedy sampler benchmark, on this build
python $M/doc/full_model/bench/sampler_ab.py --rounds 3 --replays 32

# acceptance tests, forward and reverse
pytest $M/tests/test_full_model.py
# reverse order, from the junit ids the forward run just wrote.  (`--collect-only -q` is
# not usable for this here: the repo's conftest prints prose containing `::`, and there is
# no pytest-reverse plugin in this environment.)
python - <<'EOF' > /tmp/reverse_ids.txt
import xml.etree.ElementTree as ET
M = "models/autoports/meta_models_muse_glimmer_30b"
tree = ET.parse(f"{M}/doc/optimized_full_model/test_results.xml")
ids = [f"{M}/tests/test_full_model.py::{tc.get('name')}" for tc in tree.iter("testcase")]
print("\n".join(reversed(ids)))
EOF
pytest -q --no-header -p no:randomly -rA $(tr '\n' ' ' < /tmp/reverse_ids.txt)

# watcher (no profiler in the same run), then reset before profiling.  The shipped
# default set, then each opt-in prefill-trace case on its own -- see the Watcher section
# for why they are separate -- with a reset between every one.
bash $B/run_watcher.sh
python $M/doc/full_model/bench/tt_reset.py
W="TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=0 TT_METAL_WATCHER_NOINLINE=1"
for t in test_prefill_trace_is_opt_in_and_matches_the_eager_path \
         test_prefill_trace_survives_rebinding_the_same_external_cache; do
  env $W TT_METAL_LOGS_PATH=$M/doc/optimized_full_model/watcher_$t \
      pytest -q "$M/tests/test_full_model.py::$t" --no-header -p no:randomly
  python $M/doc/full_model/bench/tt_reset.py
done

# both opt-in cases in ONE process under watcher: clean, and the release-then-build-
# another-model sequence limitation 6 used to blame
env WATCHER_TAG=_prefill_trace_pair bash $B/run_watcher.sh \
    test_prefill_trace_is_opt_in_and_matches_the_eager_path \
    test_prefill_trace_survives_rebinding_the_same_external_cache
python $M/doc/full_model/bench/tt_reset.py

# limitation 6's two twelve-case arms.  Both pass all twelve tests; the opt-in arm then
# trips at teardown 3 times in 3, the length control 1 time in 3.  Expect a truncated
# watcher log and WATCHER_LOG_NOT_A_REAL_RUN from the verdict step when it trips -- that
# is the finding, and check_watcher_console<TAG>.log is what records it.
#   opt-in arm      -> WATCHER_TAG=_12case_tripped, _12case_tripped_run2, _12case_tripped_run3
#   length control  -> swap the last two names for
#                      test_top_k_top_p_runs_through_the_same_path_and_greedy_survives_it and
#                      test_host_sampling_agrees_with_the_device_sampler_on_the_same_logits,
#                      WATCHER_TAG=_12case_control, _12case_control2, _12case_control3
env WATCHER_TAG=_12case_tripped bash $B/run_watcher.sh \
    "test_prefill_is_reproducible[1024]" \
    test_split_sampling_feeds_the_sampled_token_back_on_device \
    test_steady_state_decode_does_no_per_token_host_work \
    test_topk_runs_through_the_multi_core_factory \
    test_device_sampling_keeps_each_batch_row_token_in_its_own_row \
    test_lm_head_softcap_runs_in_l1_and_matches_the_dram_form \
    "test_decode_embedding_gathers_straight_into_the_boundary_layout[7]" \
    "test_decode_embedding_gathers_straight_into_the_boundary_layout[202047]" \
    test_swiglu_multiply_runs_on_the_wide_grid_and_returns_the_narrow_one \
    "test_batched_prefill_and_decode_with_mixed_lengths[32]" \
    test_prefill_trace_is_opt_in_and_matches_the_eager_path \
    test_prefill_trace_survives_rebinding_the_same_external_cache
python $M/doc/full_model/bench/tt_reset.py

# limitation 6's two arms re-measured against round 5's fixed teardown -- the runs that
# excluded the in-model candidates and doubled each arm to six.  TEN is run_watcher.sh's own
# default CASES list, copied out so the two extra cases can be appended:
TEN=( "test_prefill_is_reproducible[1024]" test_split_sampling_feeds_the_sampled_token_back_on_device \
      test_steady_state_decode_does_no_per_token_host_work test_topk_runs_through_the_multi_core_factory \
      test_device_sampling_keeps_each_batch_row_token_in_its_own_row \
      test_lm_head_softcap_runs_in_l1_and_matches_the_dram_form \
      "test_decode_embedding_gathers_straight_into_the_boundary_layout[7]" \
      "test_decode_embedding_gathers_straight_into_the_boundary_layout[202047]" \
      test_swiglu_multiply_runs_on_the_wide_grid_and_returns_the_narrow_one \
      "test_batched_prefill_and_decode_with_mixed_lengths[32]" )
for i in 1 2 3; do
  python $M/doc/full_model/bench/tt_reset.py
  env WATCHER_TAG=_postfix_optin$i bash $B/run_watcher.sh "${TEN[@]}" \
      test_prefill_trace_is_opt_in_and_matches_the_eager_path \
      test_prefill_trace_survives_rebinding_the_same_external_cache
  python $M/doc/full_model/bench/tt_reset.py
  env WATCHER_TAG=_postfix_ctrl$i bash $B/run_watcher.sh "${TEN[@]}" \
      test_top_k_top_p_runs_through_the_same_path_and_greedy_survives_it \
      test_host_sampling_agrees_with_the_device_sampler_on_the_same_logits
done

# the layer-stack floor at the benchmark's own decode context, which is what makes the
# floor-plus-terminal comparison able to fail
python $M/doc/optimized_multichip_decoder/bench/layer_ab.py \
    --candidates tp4,tp4b --prefill-seq 128 --decode-context 256

# round 6's two arms: work-matched twelve (extra generator + cache churn + a decode trace,
# but no prefill trace) and the opt-in pair alone.  These are what separate "the prefill
# trace" from "a long process that builds generators".
for i in 1 2 3; do
  python $M/doc/full_model/bench/tt_reset.py
  env WATCHER_TAG=_workmatched$i bash $B/run_watcher.sh "${TEN[@]}" \
      test_decode_follows_the_cache_it_is_rebound_to_after_the_trace_is_captured \
      test_top_k_top_p_runs_through_the_same_path_and_greedy_survives_it
  python $M/doc/full_model/bench/tt_reset.py
  env WATCHER_TAG=_pairalone$i bash $B/run_watcher.sh \
      test_prefill_trace_is_opt_in_and_matches_the_eager_path \
      test_prefill_trace_survives_rebinding_the_same_external_cache
done

# and the ten gated cases repeated, which is the 0-of-5 arm
for tag in _default10 _10case_repa _10case_repb _10case_repc; do
  env WATCHER_TAG=$tag bash $B/run_watcher.sh
  python $M/doc/full_model/bench/tt_reset.py
done

# the trace-lifecycle probe: four negative-control arms plus the arm that disproved the
# first statement of limitation 6
for arm in capture release recapture clone_cache rebuild; do
  env $W TT_METAL_LOGS_PATH=$M/doc/optimized_full_model/watcher_probe_$arm \
      python $B/prefill_trace_release_probe.py --arm $arm
  python $M/doc/full_model/bench/tt_reset.py
done

# the o_proj OPT-011 candidate, re-measured on this stage's shipped path
python $M/doc/optimized_multichip_decoder/bench/layer_ab.py \
    --candidates tp4,tp4b,oproj_c8_bw4 --prefill-seq 128 --decode-context 2048

# reduced-variant profiles: one real layer of each kind, never the 52-layer stack
bash $B/run_tracy.sh

# the context contract, recomputed from this stage's evidence
python $B/refresh_context_contract.py
python .agents/scripts/check_context_contract.py --model-dir $M \
    --hf-model meta-models/Muse-Glimmer-30B --stage optimized-full-model --require-contract

# the gated perimeter of this file's figures, resolved against committed runs
python $B/check_reported_figures.py
```

## Artifacts

Implementation:

| path | what changed |
| --- | --- |
| [`../../tt/model.py`](../../tt/model.py) | `LM_HEAD_SOFTCAP_IN_L1`, `EMBED_DECODE_GATHER_SHARDED`, `_LMHead.forward`, `_all_gather_async(memory_config=)`, `_embed(memory_config=)`, `embed_decode` |
| [`../../tt/optimized_decoder.py`](../../tt/optimized_decoder.py) | `DECODE_SWIGLU_MUL_CORES`, `_OptimizedMLP.decode_forward` |
| [`../../tests/test_full_model.py`](../../tests/test_full_model.py) | seven new contract tests (nine cases with parametrization) |

Evidence:

| path | what it is |
| --- | --- |
| `evidence_perf_before.json` / `evidence_perf.json` | the before/after perf rows, same script |
| `evidence_accuracy.json` | accuracy, split-sampling contract, prompt shapes, fallback audit, capacity |
| `evidence_fp32_gate.json` / `evidence_misses.json` | the fp32 control and the per-position miss detail |
| `evidence_autoregress.json` | free-running generation, chat and raw |
| `decode_ab_shipped.json` / `decode_ab_swiglu.json` | the decode A/B arms |
| `sampler_ab.json` | the greedy sampler benchmark, re-run on this build |
| `ttft_breakdown_before.json` | TTFT by phase at five prompt lengths |
| `prefill_host_probe.json` / `prefill_opcount.json` | the host-dispatch attribution, including the drained-collective pass |
| `ccl_host_probe_bfp8.json` / `ccl_host_probe_bfp8_loaded.json` / `ccl_host_probe_bf16.json` | the collective arms, at the model's payload dtype and with a loaded queue |
| `prefill_trace_probe.json` | the traced-prefill measurement: 1.33x, bit-identical, capture cost, retained DRAM |
| `evidence_perf_prefill_trace.json` | TTFT with the opt-in prefill trace on |
| `l1_highwater_probe.json` | what change 1 costs the decode step's peak L1 |
| `prefill_trace_release_probe.py` arms | `logs/watcher_probe_*.log`: capture / release / recapture / clone-cache, each watcher-clean |
| `decode_ab.json` | the `mlpN` arms; their `error` field keeps only the last traceback line, so the exact `TT_FATAL` is in `logs/decode_ab.log` |
| `perf_summary.json` | roofline / device / end-to-end reconciliation |
| `test_results.xml` | the 55-case suite |
| `tracy/` | reduced-variant profiles: prefill 128, decode sliding, decode full, sampling |
| `qualitative/` | the shared suite, the HF control, the comparison, and the diff against the full-model stage |
| `watcher/`, `watcher_prefill_trace_{optin,rebind}/` | the three watcher logs the verdicts are re-derived from (each run's `generated/inspector` and kernel-name dumps are 6 MB of build metadata and are not kept) |
| `logs/` | every console log named above |
| `bench/check_reported_figures.py` | resolves the figures in this document against the artifacts above, and asserts its advertised check count, its assertion/binding split and the artifacts it opened; `FIGURES_OK` |

## Stage review

See [`work_log.md`](work_log.md) for the review rounds and their outcomes.
