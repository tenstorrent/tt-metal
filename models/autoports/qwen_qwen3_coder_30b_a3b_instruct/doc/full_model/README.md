# Full model — Qwen3-Coder-30B-A3B-Instruct on 4 Blackhole dies

Stage 05. The stage-04 optimized multichip decoder layer, stacked 48 deep with
embeddings, final norm, a column-parallel LM head and on-device traced sampling,
behind `tt/model.py` and `tt/generator.py`.

## Headline

Workload **prompt 128 / generate 128 / batch 1**, 48 layers, 1x4 P300_X2,
`FABRIC_1D_RING`. Every cell is a row of `probes/perf_full_model.csv`, written by
`probes/perf_full_model.py`; the JSON beside it carries the min/max spread.

| | | |
|---|---|---|
| **TTFT** (prompt 128, warmed, prefill → sampled token in hand) | **126.70 ms** | `perf_full_model.csv` row 2 |
| cold TTFT, same call, first pass | 1115.16 ms | `perf_full_model.json` `ttft_cold_ms` |
| **traced decode, token-out** (model trace + sampler + on-device feedback) | **22.08 ms — 45.29 t/s/u** | row 4 |
| traced decode, token-out **+ per-token host readback** | 22.75 ms — 43.96 t/s/u | row 5 |
| traced decode, **logits only** (model trace alone, no sampling, no feedback) | 20.21 ms — **49.48 t/s/u** | row 3 |

**Cold TTFT moved a lot and it is not a model change.** It was 188.87 ms on the
earlier run and is 1115.16 ms here. That figure is the *first* pass through a
freshly opened mesh, so it is dominated by JIT kernel compilation and therefore
by whatever is already in `~/.cache/ttnn`; this run followed a watcher run, which
builds a different set of binaries. Warmed TTFT — the same call, second pass
onward, which is what a served request costs — moved by −0.71 ms. Read the cold
number as a compile-cache figure, not a model figure.

The three decode numbers are different measurements and are reported separately
because they answer different questions:

* **logits only** is the fair comparison to a decoder-stack or PERF.md-style
  figure, and to the layer-stack lower bound below. It is what teacher forcing
  would cost if the harness did not also sample.
* **token-out** is the serving steady state: the sampled token becomes the next
  decode input on device and the host does nothing but replay two traces.
* **token-out + readback** adds the readback a standalone generator needs to
  print a token. It costs **0.67 ms**.

`models.common.readiness_check.run_teacher_forcing` reports a *fourth* number,
**38.50 t/s/u**, and it is not any of these: that runner teacher-forces, so
every step also uploads a forced token from the host and reads the prediction
back. It is a token-out correctness gate that happens to print a rate, not a
decode benchmark.

### Against the layer-stack lower bound

| | ms/token |
|---|---|
| 48 × stage-04 traced decode layer at ctx128 (0.4286 ms, `../optimized_multichip_decoder/perf_decode.csv`) | 20.57 |
| **measured full-model logits-only decode** | **20.21** |
| measured token-out decode | 22.08 |

The stacked model is **at** its layer-stack lower bound — marginally under it,
because the 0.4286 ms/layer figure is a per-iteration host-visible cost for a
trace containing one layer, and 48 layers in one trace amortise that once rather
than 48 times. Everything the full model adds beyond the layers — embedding,
final norm, LM head, sampling, token feedback — costs **1.87 ms**, 8.5% of
token-out decode.

## Accuracy

AIME24, HF chat template, 100 tokens, top-100, generated fresh by this stage
(`../../readiness_aime24_chat.refpt`, 158 prompt tokens). Bar: top-5 ≥ 98%,
top-100 = 100%.

| | top-1 | top-5 | top-100 |
|---|---|---|---|
| **prefill** (`run_prefill_check.log`) | 0.980 (98/100) | **1.000 (100/100)** | **1.000 (100/100)** |
| **decode**, teacher-forced through traced generate (`run_teacher_forcing.log`) | 0.990 (99/100) | **1.000 (100/100)** | **1.000 (100/100)** |

Both bars are met with the maximum possible margin: every single one of the 100
reference positions is inside the model's top-5, on both paths.

## The generated text

`run_autoregressive.py`, 128 free-running greedy tokens each side, from a
59-token story prompt. Full texts in `../../readiness_autoregressive/`.

> **HF**: *" something peculiar: a tiny, glowing seed had fallen from the sky. As
> she picked it up, the seed began to shimmer and grow, sprouting into a
> magnificent tree that bloomed with flowers of pure light. …"*
>
> **TT**: *" a peculiar shimmer in the air, like heat waves rising from summer
> stones. As she approached, the shimmer grew stronger, and suddenly, a portal
> opened before her eyes, revealing a world of impossible beauty and wonder. …"*

Read, not just scored. Both are fluent, grammatical, on-prompt English
narrative in the register the prompt sets. The TT completion is **coherent over
the whole 128 tokens**, keeps the character's name, maintains tense and
viewpoint, and closes on a quoted line of dialogue. There is no repetition, no
single-token collapse, no wrong-language drift, and no point at which it stops
making sense.

It **diverges from HF at the very first generated token** — the common prefix is
zero tokens long — and only 4 of 128 tokens match, at indices 1, 42, 43 and 44
(`../../readiness_autoregressive/autoregressive_meta.json`). HF's token 0 is
`2494` (" something"), TT's is `264` (" a"), straight out of prefill. An earlier
revision of this section said "diverges at token 3", which was understated in
the flattering direction; the 4/128 count was right. That is the
expected behaviour of free-running greedy decode over a 30B MoE at bfloat4_b
expert weights: the router picks 8 of 128 experts on a fp32 argmax over logits
that differ in the fifth decimal, so a single flipped expert selection early on
sends the two continuations down different but equally valid branches. The
teacher-forced numbers above are the ones that measure agreement, and they say
100/100 within top-5. The free-running run is there to catch *feedback* bugs,
which teacher forcing cannot see by construction, and it does not show one.

`check_degenerate_output.py --scope autoregressive`, archived verbatim at
`check_degenerate_output.log`:

```
measured: tt free-running completion [.../readiness_autoregressive/autoregressive_meta.json]
    {'source': 'words', 'num_tokens': 107, 'adjacent_duplication': 0.0094,
     'trigram_loop_fraction': 0.028}
measured: hf/tt token agreement (informational) {'matching_tokens': 4, 'compared_tokens': 128}

No degenerate output detected.
```

This is the runner-side stage gate, and until stage-05 review it was *only* a
quotation: the strings above appeared nowhere in the tree except in this file,
and `run_autoregressive.log` carries no degeneracy lines. It has now been re-run
and archived. The figures were in fact correct — the re-run reproduces 107 /
0.0094 / 0.028 exactly — but a quoted gate with no artifact is not evidence, and
`probes/check_published_figures.py` now parses this log rather than trusting the
prose.

### The qualitative suite: six prompts, not one

Free-running evidence used to be a single story prompt, which cannot show
whether a *coder* model holds up on code, translation, factual recall or
short-form instruction following. `probes/qualitative_probe.py` runs the
repository's shared suite — `models/common/readiness_check/vllm_prompts.txt`,
6 prompts — through the real 48-layer model, **greedy and sampled** for each,
and archives both to `qualitative_check.log` and to
`../../readiness_qualitative/vllm_qualitative_outputs.json` in the schema
`check_degenerate_output.py --scope vllm` already scores. All twelve
completions are clean (same log).

Read, not just scored: the haiku scans as a haiku; the supervised/unsupervised
explanation is correct and well-formatted; the story continuation is coherent
and on-prompt; the three laws of thermodynamics are stated correctly with the
right ΔU = Q − W; the French translation is correct and idiomatic
(*"Bonjour, comment allez-vous aujourd'hui ?"*); and the Fibonacci answer is
runnable Python with accurate complexity comments.

The sampled leg is worth one note, stated exactly. At `top_k=20, top_p=0.9,
temperature=0.7` it is byte-identical to greedy on **five of the six prompts**
and differs only on the story continuation, where it takes a different branch
after "found" and stays coherent
(`../../readiness_qualitative/vllm_qualitative_outputs.json`, 103 words greedy
against 107 sampled per `check_degenerate_output.log`).

That is a weaker signal than it first looks and is worth not overselling: five
identical completions are consistent both with "these prompts are low-entropy
enough that a 0.7 temperature and a 0.9 nucleus leave the top-1 token dominant"
and with "the sampled leg quietly ran greedy". The one divergence rules the
second reading out, and the test suite closes it properly rather than by
inference —
`test_top_k_top_p_sampling_runs_through_a_traced_generate` counts sampler
dispatches and asserts every one on a stochastic run goes to `sample_split` and
**none** to force-argmax. The prompt suite is qualitative evidence; the
dispatch count is the proof.

The first 48-layer run ever made, on a code prompt, is in
`probes/smoke_probe_48layer.log`:

> `Here are a few different ways to write a Python function that reverses a string:` / `## Method 1: Using String Slicing (Most Pythonic)` / ` ``` `

## What is preserved from stage 04, and what is new

Nothing about the decoder layer's strategy changed. `test_runtime_fallback_audit_is_clean`
asserts the whole list every run:

| carried forward | value |
|---|---|
| attention | TP=4, 8 Q / 1 K / 1 V heads per die, `bfloat8_b` DRAM-sharded projections |
| experts | EP=4, 32 of 128 per die, `bfloat4_b` at LoFi, `in0_block_w` 16 / 12 |
| expert intermediates | L1 |
| router | replicated, top-k in **fp32 logit space** |
| both residual RMSNorms | replicated, width-sharded over 8 L1 cores, feeding the qkv projection with no reshard |
| KV cache | `bfloat16`, paged, block size 32, **1 local KV head per die** |
| collectives | 2 all-reduces per layer, `FABRIC_1D_RING` / `Topology.Ring`, **2 links prefill, 1 link decode**, caller-owned persistent buffers |
| **inter-layer residual** | replicated `[1, 1, B, 2048]` bf16 `TILE` `DRAM_MEMORY_CONFIG` in and out, **no collective, gather or reshard between layers** |

`decode_hidden` and `prefill_hidden` are a bare `for` loop over 48 layers with
the residual threaded straight through; `test_inter_layer_residual_contract_is_preserved`
asserts a layer's output shape, dtype, layout and memory config equal its
input's, at the full-model boundary.

The stage-04 suite was re-run unchanged on the stage-05 tree: **112 passed, 0
failed** (`pytest_stage04_regression.log.gz`), so the one edit to
`multichip_decoder.py` — an optional `rope=` parameter that defaults to the op
it already called — changed no stage-04 number.

New, and where each new boundary lives:

| piece | choice | why |
|---|---|---|
| `embed_tokens` | **replicated** bf16, DRAM | its output *is* the residual contract, so there is no collective at all. A hidden-sharded table would be 4× smaller but owes an all-gather per prefill and per token. 0.622 GB/die against 22.1 GB of measured headroom |
| `model.norm` | replicated; decode reuses the layer's width-sharded norm kernel and compute config, prefill the interleaved one | same kernel, same numerics as the residual norms |
| `lm_head` | **column-parallel over the vocabulary**, `bfloat8_b`, 37984 columns per die | 151936 = 4 × 37984 and 37984 = 32 × 1187, so the split is exact and the vocabulary needs **no padding** |
| decode rotary | `ttnn.experimental.rotary_embedding_hf(is_decode_mode=True)` over a device-gathered cos/sin pair | see below — this one is a correctness fix, not a perf choice |

### The rotary had to change, and it is bit-identical

The stage-04 layer calls `ttnn.experimental.rotary_embedding(q, cos, sin,
token_index)` with `token_index` a **Python int**. That is a wrong model at
stage 05: a captured decode trace bakes the position in, so every replayed token
would be rotated at the position the trace was captured at.

The replacement is `rotary_embedding_hf`, which takes cos/sin as **tensors**, so
the position becomes a device tensor gathered by `ttnn.embedding` and advanced
inside the trace by `ttnn.plus_one`. It is the *same HF `rotate_half`
convention*, so — unlike stage 04's rejected `rotary_embedding_llama` lever — it
needs no weight permutation, changes no KV-cache channel convention and leaves
prefill untouched.

Stage 04 wrote that **neither** of its rotary spellings could advance the
position inside a replayed trace, and **that half of the note was wrong**.
`rotary_embedding_llama`'s nanobind signature
(`rotary_embedding_llama_nanobind.cpp:38-44`) takes tensors only —
`input_tensor`, `cos_cache`, `sin_cache`, `trans_mat` — and no position
argument, and `models/tt_transformers/tt/rope.py:571,739` builds exactly that
form: it is trace-replayable. What was unreplayable was stage 04's *wiring* of
it, which hoisted the cos/sin gather onto the first eager call. The rejection
itself stands unchanged, on the reason it always really had — **channel
convention**: PCC 0.1933 against a prefill-primed KV cache against 0.99997
against a fresh one, plus a bfloat8_b requantisation `max|diff|` of 3.125e-01.
The corrected framing is carried back into
`../optimized_multichip_decoder/README.md`, its `work_log.md`,
`tt/multichip_decoder.py` and `probes/rope_hf_probe.py`.

`probes/rope_hf_probe.log`, at the shipped per-die decode shapes, for both 8 Q
heads and 1 K head, at batch 1 / 4 / 8 / 32 with **a distinct position per
user** — which is the feature a Python-int `token_index` cannot express at all —
over positions from 0 to 262143, the full advertised context:

| | measured |
|---|---|
| `max\|diff\|` against the shipped op, **all 34 cases** — 13 positions × 2 head counts, plus 4 batch sizes × 2 head counts with a distinct position per user | **0.000e+00** |
| PCC, every case | **1.0** |
| `rotary_embedding` (int position), trace slope | 5.43 µs |
| `rotary_embedding_hf` + the reshard it needs | **3.75 µs** |
| the per-step cos/sin device gather at `rope_cache_len` 8192, once per token for all 48 layers | 52.43 µs |
| the same gather at `rope_cache_len` 262144 | **806.68 µs** |

The batch-32 row carries 13 distinct positions with a maximum of 262143, and the
reference for it is the shipped op run **once per user** at that user's own
position — because a Python-int `token_index` cannot express a batch of users at
different decode positions at all. That is the capability the swap buys, and it
agrees bit-for-bit.

The last row is a real cost, and it is why `rope_cache_len` defaults to 8192
rather than to the context: `ttnn.embedding` indexes the whole table, so growing
the tables to the full 262144 makes the once-per-token gather 15× dearer —
0.81 ms, 3.7% of a 22.08 ms token-out decode. It is still once per *token*, not
per layer, and the tables only grow when a caller actually decodes that far
(`ensure_rope_capacity`).

## The terminal path, op by op

`doc/optimized_multichip_decoder/` has a `tt-perf-report` for its layer; this
directory had none, so the four things stage 05 actually adds — embedding, final
norm, column-parallel LM head, sampler — had no op-level evidence at all.
`probes/profile_full_model.py` is the **reduced** profiling variant (one layer
of each kind, which here means 2 layers) and `probes/window_full_model.py`
slices the last traced decode iteration out of the capture, checking the
boundary rather than eyeballing it: the window must hold exactly `2 × layers`
reduce-scatters and `2 × layers + 1` all-gathers per device — the layer's two
all-reduces, plus the sampler's own vocabulary gather.
`ops_perf_full_model_decode.csv.gz` is that window (664 rows, 4 devices) and
`tt_perf_report_full_model_decode.txt` is its report.

Read the **ranking**, not the absolute microseconds: `--sync-host-device`
inflates every collective, and a 2-layer window over-weights the terminal path
by construction, which is exactly why it is the right window for looking at it.

| op | share of the window | note |
|---|---|---|
| `AllGatherAsyncDeviceOperation` ×5 | 31.69% | 4 layer all-reduce gathers + **the sampler's full-vocabulary gather** |
| `ArgMaxDeviceOperation` ×1 | 29.01% | the sampler, over the gathered 151936 |
| `MatmulDeviceOperation` `32 x 2048 x 37984` | 7.2% of the window on its own | **the LM head**, 234 µs, DRAM-bound at **66.9%** of bandwidth, `HiFi2 BF16 x BFP8 => BF16` |
| `UntilizeDeviceOperation` ×3 | 2.84% | includes the sampler's pre-argmax untilize |
| `LayerNormDeviceOperation`, width-sharded ×5 / interleaved ×4 | 1.13% / 0.54% | the residual norms and `model.norm` |
| `EmbeddingsDeviceOperation` ×3 | 0.44% | the token lookup plus the two cos/sin gathers |
| `PlusOneDeviceOperation` ×2 | 0.04% | the on-device position advance, 1.24 µs for both |

Two things worth carrying forward. The report's own advice on the LM head is
*"try a DRAM-sharded program config
(`MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`)"* — the same lever
stage 02 already took for the attention projections, not yet taken for the
vocabulary matmul, and the clearest next perf item in this directory. And the
sampler's gather-plus-argmax pair is the single largest block in the window,
which is the same conclusion the 1.125 ms force-argmax measurement reaches from
the other direction — and the same op the watcher item below is about.

## Split sampling and the sampler comparison

The two common implementations were read and compared before anything was
written:

* `models/common/sampling/tt_sampling.py` (TTTv1 `TTSampling`) — **rejected**.
  It is the older module and `Sampling1D`'s own docstrings describe themselves
  as ports of it; it carries mutable sampling state on the module, where the
  generator needs `k`/`p`/`temp` to be per-call arguments so that a serving
  caller can change them without the module remembering. Nothing it offers is
  missing from the newer one.
* `models/common/modules/sampling/sampling_1d.py` (`Sampling1D`) — **chosen**.
  Declarative config, lazy buffers, no mutable sampling state, per-call
  `k`/`p`/`temp`/`seeds`, `tt_out_tok` on both of its strategies, and a 1D-mesh
  top-k strategy that matches this 1x4 ring. No custom sampler code was written.

Within `Sampling1D` there are two strategies and **both are used**, chosen per
request rather than per convenience. That choice was measured, not asserted, at the shipped
`[1, 1, 32, 37984]`-per-die logits:

| strategy | ms | token | artifact |
|---|---|---|---|
| split top-k, `pad_to_power_of_2=True` — *the naive configuration* | 11.006 | 16 | `probes/sampler_probe.log` |
| split top-k, `pad_to_power_of_2=False` | **6.151** | 16 | `probes/sampler_probe.log`; **6.155** re-measured inside the 48-layer model, `perf_full_model.json` |
| force-argmax (all-gather the vocabulary, `ttnn.argmax`) | **1.125** | 16 | `perf_full_model.json` |

The standalone probe and the in-model measurement agree to the third decimal on
the split path (6.151 against 6.155) — which is the check that the probe's
synthetic logits are measuring the same thing the model does.

All four rows of `probes/sampler_probe.log` report `matches_host_argmax: 31/32`,
and the one disagreeing slot is the *probe's* reference, not the sampler. The
probe builds `torch.randn(1, 1, 32, 151936) * 4.0` at fp32, takes `expected`
from an fp32 argmax, then uploads the tensor as bfloat16 — so the device is
ranking a coarser tensor than the reference is. At seed 0 that costs exactly one
slot: in slot 4 the fp32 maximum 17.513796 (index 107275) and the runner-up
17.449680 (index 68114) both round to bf16 `17.5`, an exact tie the device
breaks the other way. Reproducible in plain torch with no device — the fp32 and
bf16 argmaxes of that tensor differ in slot 4 and nowhere else, 31/32. The
figure would be 32/32 against a bf16 reference; it is left as measured, because
the in-model equality that actually matters is asserted separately and exactly
by `test_device_split_sampling_matches_host_argmax` on real logits.

Two fixes came out of that, and both were required rather than optional: at
11.006 ms the sampler was **34.6% of a 31.826 ms token-out decode**
(11.005861550802365 / 31.826252466998994 = 0.34581), which is exactly the case
the full-model contract says must be fixed at the LM-head/sampling contract
before anything else is tuned. Earlier revisions of this file and of
`work_log.md` rounded that to "36%", which is not what the two artifacts
divide to.

1. **`pad_to_power_of_2=False`.** `Sampling1D` describes the power-of-two pad as
   "a big device-perf win for non-power-of-2 vocab on the multi-device path".
   For a 37984-wide shard it is the opposite: the pad is to 65536, a 1.73×
   blow-up of the tensor `ttnn.topk` then scans. **1.79× faster without it**,
   same token.
2. **Greedy takes the force-argmax strategy.** It is 5.5× faster than the top-k
   path here and produces the same token. It is still `Sampling1D`, still on
   device, still traced, still writes the sampled token straight into
   `tt_out_tok`. The moment any slot asks for `top_k > 1` or `top_p > 0` the
   generator releases the traces and recaptures on the split path — the
   top-k/top-p route is a live code path, not a promise.

`test_force_argmax_matches_split_sampling` and
`test_device_split_sampling_matches_host_argmax` assert the equality this rests
on: device greedy, split greedy and a host argmax of the gathered logits all
return the same tokens.

Together the two fixes took token-out decode from **31.826 ms to 22.678 ms**,
1.40×. A third change landed later — the watcher workaround below, which also
made the greedy sampler faster — and token-out is now **22.079 ms**, 1.44×
against the same pre-fix figure.

**The pre-fix column is unarchived, and that is a real gap in this evidence.**
22.079 is a row of `probes/perf_full_model.csv` (22.678 was, before the watcher
workaround re-measurement). 31.826 is not: it was read from
a `perf_full_model.json` written by a code state that no longer exists — the
model configured with `pad_to_power_of_2=True` and greedy routed to the split
path — and that file was overwritten by the post-fix run, because
`probes/perf_full_model.py` rewrites its outputs in place. An earlier revision
of this section cited it as though it were on disk; it is not. So the **1.40×
headline rests on an unarchived measurement**, and it is quoted here only with
that label attached. What *is* archived and does support the direction of the
claim: `probes/sampler_probe.log`'s own 11.006 → 6.151 on the same synthetic
logits (1.79×), and `perf_full_model.csv`'s in-model 6.155 against 1.125
force-argmax. Re-establishing 31.826 would need `perf_full_model.py` to grow a
legacy-sampler leg and a fresh 48-layer run; that was not done here and is the
honest thing to say rather than to imply the number is backed.

**What did not work, with the number.** `max_top_k` below 32 is not a lever. The
split path all-gathers a `[1, 1, 32, max_top_k]` candidate block; 32 is exactly
one tile wide, and below it `ttnn.all_gather` logs *"Using slower composite
all_gather: gather dim 3 is padded from 16 to 32"*. At 16 that is merely worse
— **6.268 ms against 32's 6.151**, despite gathering half the candidates. At 8
the composite gather **did not return at all**: the leg ran for over 20 minutes
on the mesh before it was killed, and the mesh needed a `tt-smi -r` afterwards.

## Trace evidence

`test_split_sampling_feeds_its_own_token_back_on_device` proves the loop rather
than describing it. Over four consecutive replays it reads the persistent decode
input tensors back and asserts:

* the value in the sampling trace's output equals the value in the **persistent
  decode token input** after every replay — they are the same tensor, because
  `tt_out_tok` points at it;
* `current_pos` and the rotary position both start at the prompt length + 1
  after the first replay and each advance by **exactly one per replay**, in
  lockstep, with nothing written to them from the host;
* `token_host_copies` does not move across any of it;
* and the tokens observed on device are **exactly** what the public `generate`
  returns for the same prompt.

`test_steady_state_decode_does_no_host_work` asserts the stronger form: between
two steady-state tokens, of the thirteen counters the generator keeps, **only
`replays` moves**. No token, position, rotary, page-table or sampling-parameter
host copy, no synchronisation, no cache reset.

`test_unchanged_page_table_costs_no_host_copy` covers both page-table cases: an
unchanged table costs zero host copies, a changed one costs exactly one.

### The stochastic path is tested, not asserted

Saying the top-k/top-p route is "a live code path, not a promise" needs a test
that drives it, and stage 05 shipped without one — the test module contained no
occurrence of `top_k`, `top_p`, `temperature` or `set_sampling_params` at all.
Four tests now own it, and they matter specifically because the generator caches
trace ids **by sampling mode**, so a stale trace served across a flip would
sample with the wrong strategy and no accuracy gate on this stage would notice:

* `test_top_k_top_p_sampling_runs_through_a_traced_generate` drives
  `sample_split` through a captured trace at `top_k=8, top_p=0.9,
  temperature=0.8`, and counts dispatches — every sampler call on that run goes
  to the split path and **none** to force-argmax;
* `test_alternating_sampling_modes_recapture_the_traces` runs greedy →
  stochastic → greedy and asserts the release/capture counters move on each
  flip **and** that the second greedy run reproduces the first exactly;
* `test_set_sampling_params_releases_traces_only_on_a_mode_flip` asserts the
  converse — changing `k`/`p` *within* stochastic mode costs no recapture;
* `test_temperature_zero_is_spelled_as_greedy` covers the spelling a serving
  stack actually sends.

### Rotary capacity on the low-level API

`decode_forward` never sized the rotary tables and never bounded `start_pos`,
while the traced loop advances `rotary_position` with `ttnn.plus_one` and
nothing on device clamps it. A caller following `decode_forward`'s own docstring
past `rope_cache_len` (8192, against a 262144 contract) therefore got an
out-of-range `ttnn.embedding` gather and **silently wrong rotary** — no raise,
no wrong shape, just a plausible token rotated at the wrong position. It also
had the second half of the same bug: `ensure_rope_capacity` reallocates cos/sin,
which invalidates the tensor identities a captured trace holds, and
`prefill_forward`/`generate` release their traces before calling it where
`decode_forward` did not.

Both are closed, and the API is now correct to the advertised context rather
than merely loud: `decode_forward` takes `decode_horizon=`, grows the tables
once before any capture, and releases the traces if and only if the tables
actually moved. Without `decode_horizon` a replay that would step past the table
raises with a message naming the fix.
`test_decode_past_the_rope_cache_length_through_the_low_level_api` runs all
three legs against a 64-entry table — the raise, the declared-horizon run, and
the equality of that run with what `generate` produces — and
`test_eager_decode_grows_the_rope_tables_for_its_position` covers the eager
branch, which gathers cos/sin too and holds no trace to protect.

## Capacity

`probes/footprint_probe.py --context 262144` builds the **real** model — real
weights, real embedding, real LM head, the real paged KV cache at the full
advertised context, the real RoPE tables — captures both decode traces and runs
a token through it. Stage 03's footprint probe allocated the shapes and never
ran through them; this one does. `probes/footprint_262144.json`:

| | GB/die |
|---|---|
| weights + `embed_tokens` + `lm_head` + RoPE tables | 5.311 |
| paged KV cache, 262144 tokens, batch 1 | 6.443 |
| captured traces + persistent collective buffers | 0.006 |
| **total** | **11.759** |
| free | **22.119** |
| DRAM per die reported by the allocator | 33.879 |

The three rows are complete and exact: they sum to **11.759415296** in
`probes/footprint_262144.json`, bit-identical to `total_gb_per_die`, with no
residual at all. The 0.001 that appears if the table above is added up is
introduced by **display rounding — this table's own rounding to 3dp** (5.310778368 + 6.443011072 +
0.005625855999999985), not by the allocator and not by an omitted term. An
earlier revision attributed it to allocator rounding between three separate
reads; `probes/check_published_figures.py` now asserts the raw rows sum to the
raw total exactly, so that misattribution cannot come back.

One row is measured at less than the advertised context and it is worth being
exact about: the probe builds the model with the default `rope_cache_len` of
8192, so the "RoPE tables" inside the 5.311 GB row are the 8192-row pair
(0.004 GB/die), not a 262144-row pair. Growing them to the full context costs
0.134 GB/die — 8192 × 128 × 2 B × 2 tables against 262144 × 128 × 2 B × 2 — so a
run that actually decodes to 262143 holds **11.889 GB/die** and has 21.985 free.
The conclusion is unchanged and the arithmetic is stated rather than folded in.

**No capability reduction.** The advertised context is 262144 and the model
holds and runs it. `doc/context_contract.json` is updated with these numbers.

Batch: the primary target is batch 1 and every number above is batch 1.
`test_mixed_length_batch_prefill_and_decode` runs four users at lengths 7 / 33 /
64 / 129 through one prefill and one decode with disjoint physical pages. The
hard ceiling is **32**, unchanged from stage 03 and not ours:
`nlp_create_qkv_heads_decode_device_operation.cpp:51` asserts `num_users <= 32`.

## Prompt lengths

Prefill is **single-shot** — there is no internal chunking, so the generator
owns no chunk padding or cross-chunk masking. Each user is prefilled at exactly
its own logical length and the logits are sliced back to it.
`test_non_aligned_prompt_lengths` covers 1, 31, 33, 100, 127, 128, 129, 257 and
1000, and `test_return_all_logits_is_sliced_to_the_logical_length` covers 37.

A **one-token prompt** was a segfault, and the cause is worth recording because
it does not raise. `prefill_forward` retains the last prompt row with
`ttnn.slice` and then deallocates the chunk. At `prompt_len == 1` the requested
slice covers the whole tensor and `ttnn.slice` returns a *view* — as a different
Python object, so an `is` guard does not catch it — leaving the retained row
pointing at freed DRAM. `probes/prompt_len_1_repro.py` is the four-line
reproduction and `probes/prompt_len_1_repro.log` is it running both ways:

```
seq_len=1 row=0: piece is hidden -> False, same buffer address -> True
  ^^ deallocating `hidden` here leaves `piece` dangling; the next read segfaults
Signal: Segmentation fault (11)
--- --fixed ---
seq_len=1 row=0: piece is hidden -> False, same buffer address -> False
  read back ok, finite=True
```

## Runtime fallback audit

`Qwen3CoderModel.runtime_fallback_audit()` extends the stage-04 layer audit to
the wrapper, and `test_runtime_fallback_audit_is_clean` asserts every field.
Verbatim at 48 layers, batch 1 (`probes/smoke_probe_48layer.log`):

* layer: `dram_sharded_taken` True, per-die qkv `(2048, 1280)`, wo `(1024, 2048)`,
  `gate_up_in0_block_w` 16, `down_in0_block_w` 12, expert intermediates **L1**,
  local heads `(8, 1)`, local experts 32, `norm_shard_feeds_qkv_directly` True,
  `decode_ccl_buffers_persistent` True;
* wrapper: embedding `replicated_bf16_no_collective`; residual contract
  `replicated [1,1,B,2048] bf16 TILE DRAM, no inter-layer collective`; LM head
  column-parallel, local vocab 37984, `bfloat8_b`, **`vocab_padding` 0**;
  decode rope `rotary_embedding_hf(is_decode_mode=True)` with the position
  advanced by `ttnn.plus_one` **inside the trace**; KV cache `bfloat16`, paged,
  block 32; topology `Topology.Ring`; 2 links prefill, 1 decode;
* boundaries: **`host_logit_readback_on_token_out_path` False**,
  **`host_argmax_on_token_out_path` False**.

Cache ownership is explicit both ways. `prefill_forward` / `decode_forward` use
the caller's `kv_cache` and `page_table` verbatim
(`test_caller_owned_cache_is_used_verbatim`), and `generate` allocates and owns
its own. `reset()` zeroes the cache in place with `ttnn.fill(..., output_tensor=)`
so tensor identities and DRAM addresses survive for trace replay
(`test_reset_zeroes_the_cache`, `test_reset_makes_generation_reproducible`).

Host sampling is an explicit compatibility mode — `sampling_mode="host"` on
`generate`, `prefill_forward` and `decode_forward` — and it is never on the
measured path.

## Verification

```bash
source python_env/bin/activate
D=models/autoports/qwen_qwen3_coder_30b_a3b_instruct

# accuracy (each opens its own mesh; one device job at a time)
python -m models.common.readiness_check.generate \
  --hf-model Qwen/Qwen3-Coder-30B-A3B-Instruct --prompt-source aime24 \
  --chat-template --gen-len 100 --top-k 100 --output $D/readiness_aime24_chat.refpt

for RUNNER in run_prefill_check run_teacher_forcing; do
  python -m models.common.readiness_check.$RUNNER \
    --model-dir $D --reference $D/readiness_aime24_chat.refpt \
    --mesh-device P300X2 --fabric-config FABRIC_1D_RING --trace-region-size 300000000
done

python -m models.common.readiness_check.run_autoregressive \
  --model-dir $D --hf-model Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --mesh-device P300X2 --fabric-config FABRIC_1D_RING \
  --trace-region-size 300000000 --max-new-tokens 128
python models/common/readiness_check/check_degenerate_output.py \
  --hf-model Qwen/Qwen3-Coder-30B-A3B-Instruct --missing-artifacts critical \
  --scope autoregressive --model-dir $D \
  | tee $D/doc/full_model/check_degenerate_output.log

# the shared qualitative prompt suite -- six prompts, greedy and sampled each
python $D/doc/full_model/probes/qualitative_probe.py --layers 48 --gen-len 128 \
  2>&1 | tee $D/doc/full_model/qualitative_check.log
python models/common/readiness_check/check_degenerate_output.py \
  --hf-model Qwen/Qwen3-Coder-30B-A3B-Instruct --missing-artifacts critical \
  --scope vllm --model-dir $D \
  | tee -a $D/doc/full_model/check_degenerate_output.log

# full-model gates: reduced tier, then the all-layer tier
pytest $D/tests/test_full_model.py -q
QWEN3_FULL_MODEL_LAYERS=48 pytest $D/tests/test_full_model.py -q

# nothing in stage 04 moved
pytest $D/tests/ -q -m "not models_performance_bare_metal" --ignore=$D/tests/test_full_model.py

# performance and capacity
python $D/doc/full_model/probes/perf_full_model.py --layers 48 --prompt-len 128 --gen-len 128
python $D/doc/full_model/probes/footprint_probe.py --context 262144

# the probes behind every number above
python $D/doc/full_model/probes/rope_hf_probe.py        # bit-identical rotary swap
python $D/doc/full_model/probes/sampler_probe.py        # the sampler sweep
python $D/doc/full_model/probes/prompt_len_1_repro.py   # the one-token-prompt segfault
python $D/doc/full_model/probes/smoke_probe.py --layers 2 --tokens 8   # the debugging loop

# the op-level profile of the terminal path (never with the watcher)
python -m tracy -v -r -p --sync-host-device -o /tmp/prof_fm_dec \
  $D/doc/full_model/probes/profile_full_model.py decode
tt-perf-report /tmp/prof_fm_dec/reports/*/ops_perf_results_*.csv

# the watcher A/B behind the open item below (aborts by design, one leg per process)
bash $D/doc/full_model/probes/run_watcher_ab.sh > $D/doc/full_model/watcher_ab.log 2>&1

# watcher-clean, end to end -- never combined with profiling
TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 \
  pytest $D/tests/ -m "not models_performance_bare_metal" -q

# and the check that no figure in these documents has drifted from its artifact
python $D/doc/full_model/probes/check_published_figures.py
```

## An upstream `all_gather_async` assert: found, localized, worked around

**Stage 05 is watcher-clean.** Under
`TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1`, `pytest $D/tests/ -m "not
models_performance_bare_metal" -q` is **145 passed, zero tripped asserts**
(`pytest_watcher_clean.log.gz`; `watcher.log.gz` is that run's final dump, which
shows the end state but cannot carry the tally) — the whole tree, the stage-04
layer suite and the full-model module together. That is the bar every previous
stage met and this one now meets.

It did not start there, and the route is worth recording because the diagnosis
was wrong twice before it was right.

**Why a watcher assert is not a watcher problem.** A device `ASSERT` **compiles
out when the watcher is off**. "It does not reproduce without the watcher"
therefore means the invariant is *unchecked*, not that it holds. This one fired
inside a shipped, traced, every-token collective on the delivered greedy decode
path — so the passing non-watcher runs established only that the model produced
good-looking tokens while a device-side invariant was being violated. Silent
undefined behaviour, not absence of a bug, and not something to ship while an
upstream fix is pending.

Before the workaround, under the same watcher configuration, the full-model test
module aborted:

```
Device 0 worker core(x= 0,y= 0) virtual(x= 1,y= 2): BRISC tripped an assert on line 119.
Current kernel: ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/
                kernels/minimal_default_writer.cpp
Last waypoint:    K,CRBW,   W,   W,   W
```

`pytest_full_model_watcher_FAILS.log` is that run (`watcher.log.gz` is *not* —
it is the final dump of the later clean run; see the artifact table). That
log was **already localized** and an earlier revision of this section misread
it: it is a *two-layer* run, and lines 13/46/52 show it reaching
`test_split_sampling_feeds_its_own_token_back_on_device` 22 s after session
start and aborting 10 s later, inside that test. The claim that "a two-layer
build did not finish inside a ten-minute budget" was contradicted by the file it
cited.

### The A/B, and what it found

`probes/sampler_watcher_ab.py` (`Sampling1D` over synthetic
`[1, 1, 32, 37984]`-per-die logits, **no model at all**) and
`probes/ccl_watcher_ab.py` (raw `ttnn.experimental.all_gather_async`, no
`Sampling1D`), driven by `probes/run_watcher_ab.sh`. One leg per process — the
watcher aborts the process on the first trip. Full matrix in `watcher_ab.log`,
per-leg logs in `watcher_ab/`.

| leg | watcher |
|---|---|
| `argmax_nobarrier` — `Sampling1D` force-argmax **unmodified**, i.e. whatever branch it picks here | **TRIPPED** |
| `argmax_barrier` — `_argmax_all_gather` replaced by the Ring + barrier-semaphore spelling | **TRIPPED** |
| `Sampling1D` force-argmax, stopped after the gather / slice / untilize | **TRIPPED** at every stop |
| `Sampling1D` split top-k/top-p | clean |
| raw `all_gather_async`, sampler's shape and knobs, **Ring** | clean |
| raw `all_gather_async`, **Linear**, default `num_workers_per_link` | clean |
| raw `all_gather_async`, **Ring**, `num_workers_per_link=1` | clean |
| raw `all_gather_async`, **Linear**, `num_workers_per_link=1` | **TRIPPED** |
| the same, at the decoder layer's 512-wide shape | **TRIPPED** |

Three conclusions, all of which contradict what this file previously said:

1. **The barrier semaphore is not the cause and not the fix.** The A/B the
   previous revision asked for was run, and the barrier leg trips identically.
2. **The Ring branch of `_argmax_all_gather` is dead code on this mesh.**
   `default_topology(mesh)` returns `Topology.Linear` for a 1x4 Blackhole ring
   (the probe prints it), so the ring/no-barrier spelling with its
   "trace-capture issues seen with some barrier-based configurations" comment —
   the upstream workaround this README leaned on — is never reached here. The
   fallback runs instead, and `_get_argmax_all_gather_config` forces `Linear`
   for any mesh under 8 devices while the fallback call hardcodes
   `num_workers_per_link=1`.
3. **The minimal trigger is `topology=Topology::Linear` together with
   `num_workers_per_link=1`** on `ttnn.experimental.all_gather_async`. Either
   alone is clean; both together trip `minimal_default_writer.cpp` on the first
   call, at the sampler's 37984-wide shape *and* at the layer's 512-wide one, so
   it is a property of the op's parameters and not of this model's tensors.

That is why the decoder layer is clean and the sampler is not: the layer never
passes `num_workers_per_link`, so it takes the default.

**Where it belongs upstream.** Two separate reports, both against
`ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/`:

* the op bug — `all_gather_async` with `Topology::Linear` and
  `num_workers_per_link=1` trips a BRISC `ASSERT` in
  `minimal_default_writer.cpp` on a 4-device Blackhole 1D-ring fabric.
  `probes/ccl_watcher_ab.py --leg linear_workers1` is a ~20-line reproducer with
  no model and no `Sampling1D`;
* the caller bug — `models/common/modules/sampling/sampling_1d.py:294-346`
  steers every sub-T3K mesh into exactly that combination, and its Ring branch
  (written to dodge a different trace-capture problem) cannot be reached on
  those meshes at all.

### The local workaround

The matrix contains its own escape: **only that one pair trips**, and the reason
the decoder layer has been watcher-clean for four stages is precisely that it
never passes `num_workers_per_link` and lets the op default. So this model does
the same thing.

`tt/model.py`'s `_WatcherCleanSampling1D` is a subclass of `Sampling1D` whose
entire body is **one overridden method**, `_argmax_all_gather`, which calls the
same op with the same `dim`, the same semaphores, the same `Topology.Ring` the
layer uses, a barrier semaphore, and **no tuning knobs pinned at all**. `Sampling1D.from_config`
builds through `object.__new__(cls)` and `_bind_strategy` binds
`self._pre_argmax_gather = self._argmax_all_gather` by attribute lookup on the
instance, so the override is what gets bound. **No shared code is edited** —
`models/common/modules/sampling/sampling_1d.py` is untouched.

The spelling it uses is not a guess: `sampler_shape_default_knobs` in the matrix
above is that exact call at that exact shape, and it is clean. The
`argmax_shipped` leg re-runs the assertion against the class the model actually
instantiates, and it is clean too, at the same sampled token (36885) the
upstream spelling produces.

**It is not free — it is cheaper.** Re-measured end to end on the same workload
(prompt 128 / generate 128 / batch 1, 48 layers), rather than assumed:

| | before (`Sampling1D` as-is) | after (`_WatcherCleanSampling1D`) | delta |
|---|---|---|---|
| watcher | **TRIPPED** | **clean** | — |
| force-argmax sampler, eager, same logits | 1.859 ms | **1.125 ms** | **−0.734 ms, 1.65× faster** |
| **token-out decode** | 22.678 ms — 44.09 t/s/u | **22.079 ms — 45.29 t/s/u** | **−0.599 ms, +1.20 t/s/u** |
| token-out + readback | 23.347 ms | 22.748 ms | −0.598 ms |
| model trace (logits only) | 20.225 ms | 20.211 ms | −0.015 ms |
| split sampler, eager | 6.155 ms | 6.155 ms | −0.000 ms |
| sampled token, both strategies | 16 | 16 | unchanged |

The last two rows are the controls, and they are the reason the first three can
be attributed to the change: the split path is not touched by this override and
does not move, and the model trace contains no sampler and does not move either.
The whole −0.599 ms of token-out lands in the force-argmax sampler, which is
exactly where the one-line difference is.

So dropping the pinned `num_workers_per_link=1` and letting the op choose its
own worker count is **both** the watcher fix and a 1.65× speed-up of the greedy
sampler. That is not a surprise in hindsight — pinning one worker per link is a
throughput cap on a gather of 151936 bf16 columns — but it was measured, not
predicted, and the direction was not assumed before measuring.

**The upstream reports still stand and are still worth filing.** They are now
"found, reproduced and worked around locally" rather than anything gating this
stage:

* the **op bug** — `all_gather_async` with `Topology::Linear` and
  `num_workers_per_link=1` trips a BRISC `ASSERT` in
  `minimal_default_writer.cpp` on a 4-device Blackhole 1D-ring fabric.
  `probes/ccl_watcher_ab.py --leg linear_workers1` is a ~20-line reproducer with
  no model and no `Sampling1D`;
* the **caller bug** — `models/common/modules/sampling/sampling_1d.py:294-346`
  steers every sub-T3K mesh into exactly that combination, and its Ring branch
  (written to dodge a different trace-capture problem) is unreachable on the
  meshes it was written for. Fixing it upstream would fix every sub-T3K caller,
  not just this model; the subclass here only fixes this one, and should be
  deleted when the op is fixed.

## Named limitations

1. **Prefill does not chunk.** A prompt is one pass through all 48 layers. That
   is what makes non-aligned lengths free, but the activation peak grows with
   the prompt, and the longest prompt actually run end to end through 48 layers
   here is 1000 tokens (`test_non_aligned_prompt_lengths`). The single-layer
   probe in `../context_contract.json` reached 262144 in 192 s; the 48-layer
   stack at that length is allocated and held (the footprint above) but has not
   been prefilled. Chunked prefill is the obvious next lever and would also cut
   the cold-TTFT compile cost.
2. **`max_top_k` is pinned at 32** by the tile-aligned all-gather, as measured
   above. Requests may ask for any `top_k` in `[1, 32]`.
3. **Batch is capped at 32** by `nlp_create_qkv_heads_decode`, unchanged from
   stage 03.
4. **The all-layer test tier reloads the model three times**, once each for the
   batch-1, batch-4 and short-rotary fixtures, because the repository's
   `mesh_device` fixture is function-scoped and a module-scoped generator cannot
   depend on it. This module therefore opens its own mesh. It costs about three
   minutes per fixture and changes no result.
5. **The rotary tables are sized lazily, and the gather gets dearer as they
   grow.** `rope_cache_len` defaults to 8192 against a 262144-token contract.
   `ttnn.embedding` indexes the whole table, so at the full context the
   once-per-token cos/sin gather costs 806.68 µs against 52.43 at 8192
   (`probes/rope_hf_probe.log`) — 3.6% of a token-out decode. Growing the tables
   also reallocates them, which invalidates a captured trace's tensor
   identities. The low-level API handles both explicitly: pass `decode_horizon=`
   to `decode_forward` on the installing call and the tables are grown once, up
   front, before any trace is captured. Without it the tables are sized only for
   `start_pos` and a replay that would step past them **raises** rather than
   gathering out of range and rotating at a silently wrong position
   (`test_decode_past_the_rope_cache_length_through_the_low_level_api`). Sizing
   the tables for the full context unconditionally would be simpler and would
   cost 0.134 GB/die, but it would also make every short-context decode pay the
   806 µs gather, so the lazy default stays.
6. **The pre-fix sampler column is unarchived**, as recorded above: the 1.40×
   token-out headline rests on a 31.826 ms figure whose artifact was overwritten
   in place.
7. Everything stage 04 named carries over: `TopK` on one core in the router,
   dynamic `nnz` at 1.47× exact, the collectives at 16.7% of a decode layer,
   28.478 µs of single-core attention body, SDPA-decode's
   `max_cores_per_head_batch` workaround, and the watcher needing
   `TT_METAL_WATCHER_DISABLE_ETH=1`.

## Artifacts

| file | what it is |
|---|---|
| `work_log.md` | what happened while building this, including what broke |
| `run_prefill_check.log`, `run_teacher_forcing.log`, `run_autoregressive.log` | the three readiness runs quoted above |
| `pytest_full_model_2layer.log` | the reduced tier, 33 passed |
| `pytest_full_model_48layer.log` | the all-layer tier, 33 passed |
| `pytest_stage04_regression.log.gz` | the stage-04 suite on this tree, 112 passed |
| `pytest_stage04_watcher.log.gz` | the stage-04 layer suite **under the watcher**, 112 passed, no assert — the contrast that pointed at the sampler |
| `pytest_watcher_clean.log.gz` | the **whole tree under the watcher after the workaround**: 145 passed, zero tripped asserts. This is the evidence for that claim |
| `watcher.log.gz` | the watcher's **final dump only** from that same run, not the whole watcher log — it begins at t=557.5 s of a 565 s run. Kept because the dump is what shows the end-state of every core; it cannot be used to argue nothing tripped earlier, which is what `pytest_watcher_clean.log.gz` is for |
| `pytest_full_model_watcher_FAILS.log` | the pre-workaround watcher run, which **aborted** — kept as the evidence the assert was real and already localized to the sampling test |
| `probes/perf_full_model.csv`, `.json` | every performance figure at the top of this file |
| `probes/footprint_262144.json`, `.log` | the capacity table |
| `probes/rope_hf_probe.log` | the bit-identical rotary swap, all 34 cases, to position 262143 |
| `watcher_ab.log`, `watcher_ab/` | the watcher A/B matrix and its per-leg logs — the localization of the open item above |
| `qualitative_check.log`, `../../readiness_qualitative/` | the shared six-prompt suite, greedy and sampled, read and machine-scored |
| `check_degenerate_output.log` | the runner-side degeneracy gate, archived rather than quoted |
| `ops_perf_full_model_decode.csv.gz`, `tt_perf_report_full_model_decode.txt` | the op-level profile of the terminal path (embedding / final norm / LM head / sampler) at the reduced tier |
| `probes/sampler_probe.log` | the sampler sweep |
| `probes/prompt_len_1_repro.log` | the one-token-prompt segfault, before and after |
| `probes/smoke_probe_48layer.log` | the first 48-layer run, with its output text |
| `probes/check_published_figures.py` | re-derives all 137 figures in these documents from the artifacts they cite; host-only, under a second |
| `probes/run_watcher_ab.sh`, `probes/sampler_watcher_ab.py`, `probes/ccl_watcher_ab.py` | the watcher A/B, one leg per process |
| `probes/profile_full_model.py`, `probes/window_full_model.py` | the reduced profiling variant and the window that slices its last decode iteration |
| `probes/qualitative_probe.py` | the shared prompt suite through the real model |
| `probes/*.py` | every probe, runnable |
| `../../readiness_aime24_chat.refpt` | the reference this stage generated |
| `../../readiness_autoregressive/` | HF and TT completions and their metadata |
