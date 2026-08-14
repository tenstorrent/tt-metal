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
| **token-out decode** | 23.815 ms/token · 41.99 t/s/u | **23.315 ms/token · 42.89 t/s/u** | **-2.10 %** |
| **traced logits-only decode** | 23.164 ms/token · 43.17 t/s/u | **22.657 ms/token · 44.14 t/s/u** | **-2.19 %** |
| traced teacher-forcing decode | 36.88–37.99 t/s/u † | **37.07–38.15 t/s/u** † | overlapping ranges; no claim † |
| sampling trace | 0.632 ms | 0.632 ms | unchanged |
| **TTFT**, prompt 128, shipped default | 65.94 ms (min of 3) | 63.66 ms (min of 3) | inside the process spread; it is not device-bound, see below |
| **TTFT**, prompt 128, `prefill_trace=True` | — | **50.19 ms (min of 3)** | **-21.2 %** against the default arm |
| layer-stack lower bound | 23.239 ms/token | **22.858 ms/token** | −1.64 % |
| decode accuracy (teacher forcing) | top-1 0.990 · top-5 **1.000** · top-100 **1.000** | same | — |
| prefill accuracy | top-1 0.990 · top-5 **1.000** · top-100 **1.000** | same | — |
| qualitative suite, 6 chat prompts | — | **byte-identical to the full-model stage** | — |

† The teacher-forcing row is the **one** cross-process comparison in this table, and it
supports no improvement claim. Its `before` is the full-model stage's committed spread
over its own three runs (36.88 in `../full_model/evidence_accuracy.json`, 37.10–37.99 in
`../full_model/evidence_fp32_gate.json`); its `after` is this stage's three
(**37.28** in [`evidence_accuracy.json`](evidence_accuracy.json), 38.15 and 37.07 in
[`evidence_fp32_gate.json`](evidence_fp32_gate.json)). **The ranges overlap**, so the
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
| tests | `tests/test_full_model.py`, **53** cases (46 inherited + 7 new), forward and reverse order |
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
52-layer step: 23.164 − 0.016 − 0.338 = 22.81 ms. **Measured: 22.657 ms.** The
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
| end-to-end token-out | **23.315 ms/token** | `evidence_perf.json`, min of 3 |
| end-to-end logits-only | **22.657 ms/token** | same run, decode trace alone |

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
knob.** Of a 431 µs sliding layer, 255 µs is the six DRAM-sharded projections and
**176 µs is everything else**: two 8.7 µs hidden-size RMSNorms plus two 4.6 µs per-head
norms, a 15.1 µs `SdpaDecode`, a 15.1 µs reduce-scatter and a 12.5 µs all-gather, two
3.7 µs paged cache updates, two 2.8 µs rotary applications, head create/concat, and ~12
µs of layout conversions. None of that moves DRAM bytes for the roofline to count, and
the projections themselves run at 52.27–77.12 % of peak. A model built from many small ops
sits lower on the roofline; the requirement is the explanation, and the explanation is
that 41 % of the layer is latency-bound rather than bandwidth-bound work.

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
**22.838 ms** against a measured logits-only replay of **22.657 ms**, so there is no room
for a real 311 µs per-step bubble in the un-profiled run — and the steady-state loop's
token, position and synchronisation counters are all zero per token
([`evidence_accuracy.json`](evidence_accuracy.json)`:fallback_audit`). Every percentage
in this document comes from device-time sums, which exclude gap entirely.

**Device time is 0.8 % *above* the traced replay it is meant to explain** (22.838 ms
against 22.657 ms of logits-only replay), which is worth naming rather than hiding.
`tt-perf-report` merges a 4-device capture by taking the **max** per op, so a step
where different devices peak on different ops sums higher than any one device's
critical path, and profiler instrumentation adds to each row. The direction is the
useful part: there is no room left between device time and end-to-end for host work,
which is what the zero per-token refresh counters independently say.

`22.657 + 0.632 = 23.289` against a measured token-out of `23.315` — **the two traces
account for the whole step to within 27 µs**, and that 27 µs is the caller's token
readback.

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

* the model decode trace is **22.657 ms**, i.e. **-0.9 %** against a 22.858 ms floor;
* token-out is **23.315 ms**, i.e. **+2.0 %** on the floor and **-3.6 %** on
  floor-plus-terminal (22.858 + 0.691 device terminal + 0.632 sampling = 24.181).

The gate is 10–15 % over floor-plus-terminal. There is no gap to split: the terminal
path is priced directly at 691 µs in the profile, the sampling trace at 632 µs, and
their sum plus the floor already exceeds what the whole step measures. As in the
full-model stage, the floor is measured at context 2048 and the benchmark runs at
128–256 positions, so it is conservative and "the terminal path costs less than
nothing" is not a conclusion it can support — what the comparison establishes is that
the terminal path adds no *structural* cost, which the residual-contract fixed point
and the zero refresh counters say directly.

## Where TTFT actually goes

TTFT at prompt 128 on the shipped default is **63.66 ms**, of which the 52-layer stack
is **60.3 ms** ([`ttft_breakdown_before.json`](ttft_breakdown_before.json)). The phase
table below is the **min of each phase across three rounds** of that probe, which is not
one run — the phases are timed with a device synchronisation around each, and the sum
(64.45 ms) is quoted against a 63.66 ms TTFT measured in a different process. It is a
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
2608.4 µs of window over 2 layers plus the terminal path), so 52 of them is ~43 ms. The
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
| the other 12 op kinds | 655 | 8.14 | — |
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
| **TTFT**, prompt 128 | 63.66 ms (min of 3; 67.44 / 63.66 / 65.04) | **50.19 ms (min of 3; 50.62 / 50.19 / 50.20)** |
| token-out decode | 23.315 ms/token | 23.328 ms/token |
| traced logits-only decode | 22.657 ms/token | 22.657 ms/token |
| prefill trace buckets captured | — | `[128]` |

**-21.2 % of TTFT**, with decode untouched to 0.06 %
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

`test_prefill_trace_is_opt_in_and_matches_the_eager_path` pins the contract that a
caller turning it on depends on: the traced prompt returns exactly the tokens the eager
path returned, a second call on the same bucket replays instead of recapturing, a
non-tile-aligned prompt inside the bucket (120 tokens in the 128 bucket) is served by
it, a different bucket past the cache bound falls back to eager rather than evicting,
and `teardown()` releases everything. Binding an external KV cache releases every
prefill trace, because the trace bakes in the cache buffer addresses it writes.

### The default TTFT does not move, and its spread is why

TTFT is the one figure in this document that moves between processes. On the shipped
default it is 67.44 / 63.66 / 65.04 ms in this run; the baseline arm measured
67.97 / 69.64 / 65.94 ms in its own process. The full-model stage documented four passes of
*identical* code spanning 61.09–66.04 ms, an 8 % spread that no round-to-round variance
predicts, and attributed it to prefill being compiled, allocated and scheduled once per
process. The decode figures do not do this: 23.315 / 23.345 / 23.340 here,
23.850 / 23.815 / 23.824 in the baseline arm — a 0.2 % spread,
and the two arms do not overlap.
Read the default TTFT as ~61–70 ms in both arms and the decode numbers as exact. Nothing
this stage changed touches prefill except the terminal softcap, which is ~30 µs of a
65 ms window. The `--prefill-trace` arm's 50.19 ms is well outside that spread in the
other direction.

### One remaining prefill inefficiency, priced and left

The prefill terminal norm runs on **one core for 133.86 µs** and the embedding norm on
four cores for 134.65 µs ([`tracy/prefill_128_perf_report.csv`](tracy/prefill_128_perf_report.csv)
ids 3884 and 3581), against 8.8 µs for the same norm width-sharded in decode:
`ttnn.rms_norm` on a DRAM-interleaved input parallelises over tile *rows*, and both of
these run on a 32-row slice. That is 0.27 ms of a 65 ms TTFT (0.4 %), and it is left
alone deliberately: the fix is to route both through the sharded form, which changes
prefill *numerics* — the two norms sit on the accuracy gates' critical path — for 0.4 %
of a figure whose process-to-process spread is 8 %. Recorded as limitation 8 with the
measurement so a later stage can take it with the accuracy re-run it needs.
## Operation-topology audit

The measured decode path, one sliding layer plus the terminal work, from
[`tracy/decode_sliding_perf_report.csv`](tracy/decode_sliding_perf_report.csv). "Action"
is what this stage did about it.

| op group | device µs | candidate | action |
| --- | --- | --- | --- |
| LM-head matmul, 32 x 6656 x 50688 BFP4 | 603.8 | **2.6 %** of the 23.315 ms token-out step, and 40.8 % of this one-layer profiling window; DRAM-bound at 279.8 GB/s reading 190 MB of weights | kept: the geometry ladder (`dram_sharded`, cores=52, `in0_block_w=2`) is the full-model stage's measured winner over the legal values 1/2/4 at BFP4, and `in0_block_w=4` fails with an exact L1 blocker (*"Statically allocated circular buffers ... grow to 1821824 B which is beyond max L1 size of 1572864 B"*) |
| MLP gate/up/down, 3 x BFP4 | 190.2 | `in0_block_w` already at the largest legal divisor (13/13/10); packed gate/up measured +5.5 % by the decoder stage | kept |
| `wqkv` + `attn_gate`, 2 x BFP8, same input | 41.8 | OPT-001: two projections consuming the same post-norm activation could be one | **not taken, with a reason**: both rows are the *best* DRAM utilisation in the layer (77.12 % and 69.35 %), so packing cannot reduce the bytes, and the two outputs need different downstream layouts — QKV `sharded_to_interleaved` into head creation, the gate kept sharded until after SDPA — so a packed output needs an unshard plus two slices plus a reshard to split |
| `o_proj`, BFP8 | 21.4 | OPT-011 narrower working shard (8 cores / `in0_block_w=4`) | **re-measured on this stage's shipped path** rather than re-declined on the decoder stage's note ([`logs/layer_ab_oproj.log`](logs/layer_ab_oproj.log), `layer_ab.py --candidates tp4,tp4b,oproj_c8_bw4`): **sliding 0.4467 against 0.4474 / 0.4475**, i.e. −0.17 %, and **full 0.4163 against 0.4166 / 0.4162**, i.e. inside the repeat control's own spread. HF-reference PCC is unchanged on prefill (0.993700 / 0.992220) and marginally *better* on decode (0.993503 vs 0.993488 sliding, 0.992196 vs 0.992188 full). Worth **−0.03 ms/token, 0.12 %** of the step. Declined, and the reason is now specific: adopting it changes the **decoder stage's** shipped geometry and its single-grid invariant, which three of that stage's tests assert and whose multichip-vs-single-chip gate (0.999183 against a 0.999 bar) this stage's harness does not run. 0.12 % is not worth moving another stage's default and gate from here; the measurement is committed so that stage can take it |
| SwiGLU multiply (SFPU SiLU) | 18.0 | widen the grid | **taken**: 80-core reshard, −7.4 µs/layer |
| softcap `tanh` + `* T` | 36.9 | run on the matmul's shard | **taken**: −13.1 µs/step, at +126,976 B/bank of peak L1 |
| 4 x RMSNorm | 26.7 | wider norm grid | not available: the norm must consume and produce the 16-core boundary spec, which is the preserved inter-layer residual contract |
| reduce-scatter + all-gather x2 | 55.5 | fewer/cheaper collectives | not available at this residual contract: `ttnn.all_reduce` costs two dispatches, the wrappers cost what the primitives cost, and the count is the preserved replicated-residual contract. Persistent buffers *are* worth 14–17 % of the reduce-scatter's **host** cost and are blocked on the decoder stage's correctness race |
| `SdpaDecode` | 15.1 | — | kept: explicit program config, BFP8 cache, `max_cores_per_head_batch=32`, all from the decoder stage |
| embedding all-gather + `interleaved_to_sharded` | 18.5 | collective writes the boundary layout | **taken**: −2.0 µs/step |
| layout conversions (`i2s`/`s2i`/`reshard`) | ~24 | — | 5.91 µs of it is change 3's own reshards, which buy 13.3 µs |
| `plus_one` x2 | 1.9 | — | kept: this is the device-side position/RoPE advance |

`tt-perf-report` marks five rows `SLOW` with *"No output subblock size found"* — ids
**3065** (`o_proj`), **3071** and **3127** (MLP gate/up), **3132** (MLP down) and **3139**
(the LM head, 40.8 % of this window). That is structural, not a finding:
`MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` has no output-subblock fields to
report. The geometry those rows *do* expose — core count and `in0_block_w` — is at the
largest legal divisor for every one of them (13/13/10 for MLP gate/up/down, 2 for
`o_proj` and the LM head, both capped by their per-core K-tile count), and the two rows
where a larger value is arithmetically legal fail with exact L1 circular-buffer blockers
recorded by the decoder and full-model stages.

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
**37.28 / 38.15 / 37.07 t/s/u** against the full-model stage's 36.88 / 37.10 / 37.99.
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

One token/position stage for the post-prefill reseed, one page-table copy per request,
and one 32-uint32 readback per token because `generate()` returns tokens. The token-out
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

**The two opt-in prefill-trace cases are watched separately, and why is the finding.**
Each is `WATCHER_CLEAN` on its own — 0 tripped asserts, 0 fatal messages
([`watcher_prefill_trace_optin/`](watcher_prefill_trace_optin),
[`watcher_prefill_trace_rebind/`](watcher_prefill_trace_rebind),
[`logs/check_watcher.log`](logs/check_watcher.log)). What is **not** clean is running
them inside the module's shared-fixture suite: after a generator that captured prefill
traces is torn down and *another* model is built and run on the same mesh in the same
process, the watcher stops the device on

```
Device 0 acteth core(x= 0,y= 9) virtual(x=29,y=25): subordinate_erisc detected invalid
NOC command buffer state before starting the next kernel (write-capable NOC packet tags
must be zero so implicit transaction ID users start with transaction ID 0).
Current kernel: tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp.
```

[`bench/prefill_trace_release_probe.py`](bench/prefill_trace_release_probe.py) bisects it
into four arms, each a fresh process under watcher — **capture**, **release**,
**recapture**, and **clone the 104 KV-cache tensors and free them** — and every one is
clean ([`logs/watcher_probe_*.log`](logs)). So it is not the release, not the recapture
and not the cache churn on its own; it is the cross-generator sequence. Two things follow
and both are in the shipped code:

* the **shipped default never captures a prefill trace**, so it never releases one, which
  is why the ten-case default run above is clean;
* the opt-in path now **retires** prefill tracing after the single release a cache move
  forces, instead of recapturing (`_invalidate_prefill_traces_if_cache_moved`). That
  keeps a generator to at most one release and never a recapture after one, which is the
  smallest exposure that still serves the caller the flag is for — a serving adapter
  binds its cache once and reuses it, and then never releases at all.

The watcher also caught a real defect while this was being localised, which is worth
recording because it was in the *test*, not the model:
`test_prefill_trace_survives_rebinding_the_same_external_cache` originally freed the
cloned KV cache **before** releasing the trace that held its addresses — a use-after-free
that shows up as this same ERISC assert rather than as a wrong number
([`logs/watcher_bisect_rebind.log`](logs/watcher_bisect_rebind.log) before,
[`logs/watcher_bisect_rebind_fixed.log`](logs/watcher_bisect_rebind_fixed.log) after).

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
that the traced step runs clean: 53 tests, the batch-32 mixed-length case, and a
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
| `pytest tests/test_full_model.py` | **53 passed** — [`logs/full_test_run.log`](logs/full_test_run.log), [`test_results.xml`](test_results.xml) |
| the same 53 in reverse order, one process | **53 passed** — [`logs/full_test_run_reverse.log`](logs/full_test_run_reverse.log) |
| watcher, shipped-default subset (10 cases) | **10 passed**, `WATCHER_CLEAN` |
| watcher, each opt-in prefill-trace case alone | **1 passed** each, `WATCHER_CLEAN` |

Seven cases are new (three of them parametrizations of one test) and each pins one shipped change:

| test | pins |
| --- | --- |
| `test_lm_head_softcap_runs_in_l1_and_matches_the_dram_form` | change 1 is bit-identical to the DRAM order, finite, and bounded by `T` |
| `test_decode_embedding_gathers_straight_into_the_boundary_layout[7,0,202047]` | change 2 returns the boundary layout **and the same values as the interleaved gather**, over four repeats per token id |
| `test_swiglu_multiply_runs_on_the_wide_grid_and_returns_the_narrow_one` | change 3 leaves the MLP's boundary output layout unchanged |
| `test_prefill_trace_is_opt_in_and_matches_the_eager_path` | the opt-in prefill trace: same tokens, one bucket, non-aligned lengths inside it, eager fallback past the bound, released on teardown |
| `test_prefill_trace_survives_rebinding_the_same_external_cache` | the serving path: the **same** external cache every request keeps its trace; a cache bound to **different** buffers releases it, retires prefill tracing and still answers correctly |

## Limitations and known issues

1. **Batch-1 TTFT is host-dispatch bound; the fix ships but is opt-in.** ~17 ms of the
   ~64 ms default TTFT is host issue over 4122 ttnn dispatches, and the 209 collective
   dispatches are 33 % of it. `prefill_trace=True` removes 21 % of TTFT (63.66 →
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
6. **Releasing a prefill trace and then building another model on the same mesh trips a
   fabric ERISC watcher assert.** `subordinate_erisc detected invalid NOC command buffer
   state … fabric_erisc_router.cpp` on acteth core 29-25. Each part in isolation is
   watcher-clean — capture, release, recapture, and cloning/freeing the 104 KV-cache
   tensors, the four arms of `bench/prefill_trace_release_probe.py` — and so is each
   opt-in test on its own; it is the cross-generator sequence. The shipped default never
   captures a prefill trace so never releases one, and the opt-in path retires prefill
   tracing after the single release a cache move forces rather than recapturing. Not
   root-caused: it is below the model, in the fabric router's state across a trace
   release, and a Metal-side bisect is out of this stage's scope. A serving stage that
   rebuilds a model on a live mesh after releasing prefill traces should reset the
   devices between builds.

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
| decode path fully traced, no host fallback | yes | fallback audit: 0 token/position/sync refreshes per token; two traces account for the step to 38 µs |
| decode activations width-sharded in L1 across norm/attention/residual/MLP/output | yes | `tracy/decode_sliding_perf_report.csv`: every matmul `in0` is `L1_WIDTH_SHARDED`; the boundary is a fixed point at every layer |
| prefill activations DRAM interleaved, 2D matmul program configs for large prefill matmuls | yes | `tracy/prefill_128_perf_report.csv`; `mcast2d` specs in `multichip_decoder.py` |
| operation-topology audit recorded | yes | [above](#operation-topology-audit) |
| multi-device topology candidates measured as coherent families | yes | 12 collective arms in `ccl_host_probe.json` (implementation x worker count x persistence x wrapper/primitive/fused), all at the real payload |
| lower-movement residual candidates measured without an old-contract restore | inherited | the residual contract is preserved by the goal; the decoder stage owns the fractured-residual family (`fractured_decode_probe.py`) and this stage did not re-open it |
| best-candidate comparison against the strongest prior artifact | yes | `--baseline` reproduces the full-model stage's committed numbers to 0.02 % (23.815 vs 23.811 token-out, 23.164 vs 23.164 logits-only) before the changes are applied |
| final default reproduces the selected candidate | yes | predicted 22.81 from the reduced A/B, measured 22.657 on the all-layer default |
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
| watcher clean, separate from profiler | yes | `WATCHER_CLEAN`; Tracy ran after a reset, in its own process |
| context contract preserved | yes | 131072, unreduced, rebuilt from this stage's evidence |
| `$qualitative-check` after the selected optimization, with a control | yes | byte-identical to the previous-stage control on all six prompts |
| every reported figure resolves to a committed artifact | yes | `bench/check_reported_figures.py`, which also compares every literal it resolves back to the README text and asserts its own check count |

## How to reproduce

One 52-layer build is ~160 s of host weight packing, so each device stage runs in one
process over one build.

```bash
M=models/autoports/meta_models_muse_glimmer_30b
B=$M/doc/optimized_full_model/bench

# before / after performance, from the same script (--baseline reverts the three changes)
python $B/evidence.py --stages capacity,perf --baseline --out evidence_perf_before.json
python $B/evidence.py --stages capacity,perf,shapes,autoregress --shape-lengths 130073 --out evidence_perf.json
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
pytest -q $(pytest --collect-only -q $M/tests/test_full_model.py 2>/dev/null \
    | grep -o "<Function [^>]*>" | sed 's/<Function \(.*\)>/'"$M"'\/tests\/test_full_model.py::\1/' | tac | tr '\n' ' ')

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

# and the four-arm bisect of what does trip the fabric ERISC assert, one arm per process
for arm in capture release recapture clone_cache; do
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
| [`../../tests/test_full_model.py`](../../tests/test_full_model.py) | five new contract tests (seven cases with parametrization) |

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
| `test_results.xml` | the 53-case suite |
| `tracy/` | reduced-variant profiles: prefill 128, decode sliding, decode full, sampling |
| `qualitative/` | the shared suite, the HF control, the comparison, and the diff against the full-model stage |
| `watcher/`, `watcher_prefill_trace_{optin,rebind}/` | the three watcher logs the verdicts are re-derived from (each run's `generated/inspector` and kernel-name dumps are 6 MB of build metadata and are not kept) |
| `logs/` | every console log named above |
| `bench/check_reported_figures.py` | resolves the figures in this document against the artifacts above, and asserts its own advertised check count; `FIGURES_OK` |

## Stage review

See [`work_log.md`](work_log.md) for the review rounds and their outcomes.
