# Muse-Glimmer-30B — vLLM serving integration

`meta-models/Muse-Glimmer-30B` serves through the shared TT vLLM path on a 4-die
Blackhole P300_X2 (mesh `1x4`, `FABRIC_1D_RING`), at the full advertised
131072-token context, with on-device split sampling and a traced decode step.

## Headline — primary single-user profile

Workload: **128-token prompt, 128 output tokens, 1 request, `--max-concurrency 1`,
greedy (`--temperature 0.0`), `ignore_eos`**. Server at `--max_model_len 131072`,
`--max_num_seqs 32`. Raw: `readiness_vllm/vllm_result.json`; normalized:
`readiness_vllm/vllm_benchmark.json`.

| metric | value |
|---|---|
| **TTFT** (median = p99, single request) | **72.68 ms** |
| **Decode t/s/u** (`1000 / mean_tpot_ms`) | **43.39 t/s/u** |
| TPOT mean / p99 | 23.049 ms / 23.049 ms |
| ITL p50 / p99 | 23.015 ms / 23.641 ms |
| E2E latency | 2999.9 ms |
| Aggregate output throughput | 42.66 tok/s |
| Completed | 1/1, 0 missing output tokens |

**Serving costs essentially nothing against the standalone decoder.** The
datatype-sweep stage measured the same 128/128/1 shape through the public
generator at **43.33 t/s/u** (23.078 ms/token); through vLLM it is **43.39 t/s/u**
(23.049 ms/token) — **100.1 %** of the standalone token-out rate, i.e. the serving
orchestration, sampling, token feedback, request handling and readback are inside
the measurement noise of the decoder itself. The teacher-forcing lower bound for
the same model is 37.43 t/s/u, which serving clears by 15.9 %.

TTFT is 72.68 ms against the standalone generator's 64.17 ms at the same prompt
length: +8.5 ms for HTTP, tokenization, scheduler admission and detokenization.
Serving prefill is eager by design (see *Trace policy*).

## Secondary — CI serving-burst profile (capacity, not headline)

Workload: **100-token prompts, 100 output tokens, 32 requests, no explicit
`--max-concurrency`** (the vLLM-nightly shape), greedy, `ignore_eos`. Raw:
`readiness_vllm/vllm_ci_serving_result.json`; normalized:
`readiness_vllm/vllm_ci_serving_benchmark.json`.

| metric | value |
|---|---|
| Aggregate output throughput | **713.95 tok/s** |
| TPOT mean / p99 | 23.102 ms / 23.186 ms |
| Decode t/s/u from mean TPOT | 43.29 t/s/u |
| ITL p50 / p99 | 23.053 ms / 30.350 ms |
| TTFT p50 / p99 | 2193.67 ms / 2194.71 ms |
| E2E latency p50 | 4480.4 ms |
| Completed | 32/32, 0 missing output tokens |

This is **not** the headline decode number: all 32 prompts are admitted as one
burst, so TTFT carries 32 queued prefills and TPOT sees burst admission. It is
here for vLLM-nightly parity and to show that 32 concurrent sequences serve. The
useful figure is the aggregate 713.95 tok/s — 16.5x the single-user rate at the
same per-user TPOT, which is what the shared 32-row decode batch buys.

## Status

| gate | result |
|---|---|
| Server launch, `--sampling-profile full` | **pass** — server ready in ~4 min |
| Plugin sampling suite | 62 passed, 10 failed, 1 skipped — see *Sampling* |
| Qualitative (prompt-correct chat arm) | **pass** — character-identical to the standalone model once the API-stripped `<\|message\|>` is accounted for |
| Degenerate-output check (`--scope vllm`) | **pass**, no degenerate output, both artifact sets |
| Determinism run-to-run | **pass** — identical |
| Determinism cross-batch-position | **pass** — 8 concurrent, all identical, equal to the single-request output |
| Determinism, logit level | **pass** — full 20-way distribution bitwise identical run-to-run and across 8 batch positions (160 candidates, delta 0.0) |
| Non-aligned prompt lengths | **pass** — 9/9 lengths |
| Served context vs `doc/context_contract.json` | **131072 = 131072**, no reduction |
| Fallback + process audit | **clean** — no degraded markers, no leftover processes |
| `--async-scheduling` overlap validation | **pass** — capability accepted, output byte-identical to the non-overlap arm |

## Serving configuration

```bash
python -m models.common.readiness_check.run_vllm_server \
  --model-dir models/autoports/meta_models_muse_glimmer_30b \
  --hf-model meta-models/Muse-Glimmer-30B \
  --mesh-device P300x2 \
  --max-num-seqs 32 \
  --max-model-len 131072 \
  --sampling-profile full \
  --server-timeout 2400 \
  --tt-config '{"trace_region_size": 400000000, "fabric_config": "FABRIC_1D_RING",
                "fabric_packet_payload_bytes": 8192, "l1_small_size": 6144,
                "trace_mode": "decode_only"}'
```

Wrapped as `doc/vllm_integration/bench/serve.sh` (`hold` / `full` / `smoke` /
`checks <stages>`).

**The evidence above was not produced by that single command, and it could not
have been.** `run_vllm_server` stops after a failing stage, and the sampling stage
exits non-zero here (10 failures: 3 correctness-class, all resolved, plus 7
reproducibility-only, which are classified rather than fixed — both groups are
itemised below), so a one-shot `--stages serve,sampling,qualitative,benchmark` invocation
would never reach the benchmarks. The sweep therefore holds one server open and
attaches each stage to it separately —
`doc/vllm_integration/bench/run_serving_evidence.sh`, log
`logs/serving_evidence.log` — which is also what keeps the 52-layer model to a
single ~4-minute load for the whole gate set. The command above is the exact
server configuration; the sweep script is how the stages were run against it.

| TT config key | value | why |
|---|---|---|
| `trace_region_size` | 400000000 | the model decode trace plus the sampler's, over 52 layers |
| `fabric_config` | `FABRIC_1D_RING` | the 1x4 ring the collectives were measured on |
| `fabric_packet_payload_bytes` | 8192 | the router payload the decode collective was tuned at; without it serving opens a different fabric from every earlier stage |
| `l1_small_size` | 6144 | holds the per-program CCL global semaphores. Carried from the decoder stage's measured ladder, which is a *margin* choice, not a pass/fail one: 32768 fails the first 256-row prefill and 8192 overruns the decode budget by 896 B, but 7168, 6144 and 4096 all pass. 6144 ships because it clears 24 distinct CCL programs with 1,152 B of margin against 7168's 128 B, while 4096 makes the region itself the constraint (`doc/context_contract.json` -> `device.l1_small_note`). This stage measured no new value; it carries the decoder stage's. |
| `trace_mode` | `decode_only` | decode is traced; prefill is eager on purpose |
| `sample_on_device_mode` | `all` | enforced by the runner |

Confirmed from the server log: `max_model_len=131072`, `max_num_seqs=32`,
`block_size=64`, `enable_prefix_caching=False`, chunked prefill disabled with
`max_num_batched_tokens` bumped 2048 -> 131072 so full-context prompts are admitted.

## Precision policy — the selected config, not a serving default

`build_generator` reads `doc/datatype_sweep/selected_precision_config.json` on
every build, so serving constructs the swept policy through the same code path the
full-model evidence came from. Read back off the *built serving model*
(`probe_full_fixed.json` -> `capability_report.precision_policy`):

```text
selected_config_id       c14-attn4-cclbfp8-kv8     <- the datatype-sweep selection
activation / residual    BFLOAT16
wqkv, o_proj             BFLOAT4_B                 <- BFP4 attention
mlp_gate, mlp_down       BFLOAT4_B
prefill CCL / decode CCL BFLOAT8_B / BFLOAT8_B
KV cache                 BFLOAT8_B
LM head                  BFLOAT4_B, dram_sharded, 52 cores
```

## Adapter

`tt/generator_vllm.py`, class `MuseGlimmerForConditionalGeneration`, registered in
`register_tt_models()` in the TT vLLM plugin — canonically
`vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py`, which in this workspace
is the standalone editable checkout `/home/ttuser/dev/vllm-tt-plugin/src/vllm_tt_plugin/platform.py`
under `MuseGlimmerForConditionalGeneration`, `MuseGlimmerForCausalLM` and their
`TT`-prefixed aliases.

It is a translation shim. Every call delegates to `tt/generator.py`:

| vLLM entry point | delegates to |
|---|---|
| `initialize_vllm_model` | `build_generator(...)` — the readiness factory |
| `allocate_kv_cache` | `ttnn.zeros` + `MuseGlimmerModel.adopt_external_kv_cache` |
| `prefill_forward` | `MuseGlimmerGenerator.prefill_forward(sample_on_device=...)` |
| `decode_forward` | `MuseGlimmerGenerator.decode_forward(read_from_device, refresh_inputs, ...)` |
| `read_decode_output` / `process_decode_output_host` | the generator's own async split |
| sampling state | `apply_prefill_sampling_state` / `apply_decode_sampling_state`, which drive `models.common.sampling`'s `apply_prefill_state` / `apply_decode_state` |

**Sampling.** There is one sampling path in this port and the adapter adds none.
Serving decode replays the full-model generator's canonical split: the model
decode trace emits vocab-sharded logits, and `SamplingGenerator`'s own trace
samples them with `tt_out_tok` pointing at the persistent decode token input, so
the sampled token never leaves the device. No host argmax, no full-logits
readback, no generic top-k greedy fallback, no Python readback/writeback
token-feedback loop on the measured path. `force_argmax` is `False`;
`sampler_topk_split_to_power_of_2` is on; the invalid-vocab mask is built.

**Host sampling is explicit, optional, and never the measured path.** It is
reached only when *vLLM itself* decides a batch cannot be sampled on device
(`min_p`, `logit_bias`, `bad_words`, `allowed_token_ids`, structured outputs, or
logprobs on a mesh whose device count is not 8 or 32).

What settles that it was not taken on the benchmarked run is *not* the server log.
The serving host-sampling route now announces itself
(`DEGRADED PATH serving_full_logits_readback`), but that marker postdates the
committed logs, and `serving_audit.json -> marker_provenance` says so rather than
letting a green result be misread. The evidence that does settle it is timing and
counters:

* **no step is an outlier.** `std_itl_ms` is 0.372 over the 127 intervals, with
  p50 23.015 and p99 23.641 ms — a 0.6 ms spread on a 23 ms step. The
  host-sampling route gathers the *padded* vocab across the mesh (202752 columns,
  ~12.9 MB per step, sliced to the real 202048 only after it reaches host) and
  reads it back, against a sampled token's 32 uint32; a step doing that could not
  sit inside a 0.372 ms standard deviation.
  (An earlier revision of this section argued instead that TTFT plus 127 x mean
  TPOT reproduces E2E exactly. It does — but *identically*, for any distribution,
  because vLLM defines `tpot = (latency - ttft) / (n - 1)`. That is a tautology,
  not a bound, and it is withdrawn.);
* `probe_full_fixed.json` drives the adapter through the plugin's exact call
  sequence and reports `trace_replays` equal to the step count with
  `synchronizations: 0` — every step a traced replay;
* the benchmark ran with `sample_on_device_mode: all` and the server log records
  `captured sampling trace`.

**Cache ownership.** vLLM allocates and owns the attention KV cache;
`adopt_external_kv_cache` binds it, frees the build-time pool and updates the
model/layer configs so bounds checks describe the pool that is bound. Measured on
the 52-layer serving build: 52 layers x 2 x `(16416, 1, 64, 128)` BFLOAT8_B =
**14.86 GB/device, 1,050,624 KV tokens across all users**, `page_block_size` 64,
`blocks_per_seq` 2048. (The server log's earlier `~14.83 GB/device` line is the
*budget request* for a 1,048,576-token pool; the pool actually allocated rounds up
to a whole `blocks_per_seq` multiple, 16416 blocks, hence the two figures. The
allocated one is what is reported here and everywhere below.) Sizing is a *pool* statement: `max_model_len` stays 131072
and one request may still take all of it. 32 request slots; `max_num_seqs > 32` is
refused, citing the `nlp_create_qkv_heads_decode` `num_users` cap.

**Attention type.** 52 identical `FullAttentionSpec`s with `sliding_window=None`,
so vLLM builds one KV group and the legacy single-page-table path. Deliberate, and
the same choice `models/tt_transformers/tt/generator_vllm.py` makes for Gemma3 and
GPT-OSS: this port's decode passes an *absolute* position to the paged ops, while
vLLM's `SlidingWindowSpec` zero-pads a sliding group's page table past
`sliding_window / block_size` entries, which would collapse later positions onto
physical block 0. The 39 sliding layers still attend only their window — the SDPA
op gets `sliding_window` on the read side — so semantics are unchanged; the
uniform spec costs memory, not correctness.

## Async decode, trace and the stale-input contract

`supports_async_decode = True` is claimed because the split is implemented and
measured, not because it is available:
`decode_forward(read_from_device=False)` returns device handles,
`read_decode_output(async_read=True)` enqueues the minimal deferred read and
returns its event, `process_decode_output_host` does host formatting only.

Under overlap vLLM may build step N+1 before token N has reached host scheduler
state, so those host inputs are stale by construction. The adapter therefore reads
**nothing** from host on a device-sampled step whose padded batch layout has not
changed: the sampler wrote the token into the persistent decode token input and
the decode trace advanced `current_pos` and the RoPE index with `plus_one`, each
exactly once per emitted token. Host state is restaged only when it is
authoritative — first decode after a prefill or warmup, `reset_batch`, a slot
remap, or a step whose predecessor sampled on host.

Measured on the 52-layer serving build over 16 multi-slot decode steps with three
concurrent rows (`probe_full_fixed.json`):

```text
trace_replays 16   token_refreshes 1   position_refreshes 1
page_table_refreshes 1   synchronizations 0   readbacks 16
sampling_param_refreshes 1   sampling_param_reuses 15
```

One refresh each for sixteen tokens — the single `reset_batch=True` step after the
prefill — and zero synchronizations. That is the stale-input contract measured
rather than asserted. `doc/vllm_integration/bench/adapter_probe.py` drives the
adapter through the plugin's exact call sequence and feeds deliberately wrong host
values on every steady step to prove they are ignored.

**Overlap validated end to end.** `supports_async_decode=True` is not just
declared: a separate server run with `--async-scheduling`
(`bench/run_async_overlap.sh`, artifacts in `doc/vllm_integration/async_overlap/`)
confirms the plugin **accepted** the capability — no `Disabling async scheduling`
in its log — and that under real overlap the served completions are
**byte-identical to the non-overlap arm on all 6 pinned prompts**, with
**0.0000** adjacent-token duplication and no control tokens leaked into the text
(`async_overlap/overlap_vs_non_overlap.json`). The degenerate-output check passes
over the overlap artifacts too. A stale token or position reaching an overlapped
step would show up here as doubled subwords or repeated control tokens; it does
not.

The overlap arm exercises greedy and sampled requests, not penalised ones, and
that is structural rather than an untested corner: vLLM's own
`TTAsyncDecodeController.can_use_steady_decode_fast_path` returns `False` when
`model_input.prompt_tokens` or `output_tokens` is set, and the runner populates
both for any batch with active penalties. A penalised request therefore never
takes the overlapped path in the first place.

**Trace policy.** Decode is traced (`trace_mode: decode_only`); the decode trace
and the sampler's trace are both captured during warmup, so the first request pays
neither.

Prefill is eager, and the reason is a capacity bound rather than an impossibility.
The prefill graph is shaped by the *padded row count*, so one trace serves one
32-row bucket — bucketed capture is expressible, and `warmup_model_prefill` already
enumerates eight buckets. The binding constraint is
`GeneratorConfig.prefill_trace_max_entries = 1` (confirmed on the built serving
model in `probe_full_fixed.json`): with one resident bucket, mixed serving prompt
lengths would evict and recapture continuously, and capture costs ~98 ms against a
~15 ms per-replay saving. The opt-out is inherited from the optimized-full-model
stage, which measured the win as **1.33x at 128 rows** (`doc/optimized_full_model/prefill_trace_probe.json`)
and **1.00x at 8192** (`doc/optimized_full_model/prefill_trace_probe_8192.json`). So the 72.68 ms TTFT above
carries a known, measured 1.33x that a serving deployment with bucketed prompt
lengths could claim by raising `prefill_trace_max_entries` to its bucket count —
that is optimized-vLLM's work, not a limitation of this adapter. `warmup_model_prefill` compiles padded lengths
`[32, 96, 128, 160, 256, 512, 1024, 8192]` so the shapes the checks and both
benchmark profiles use compile before serving starts.

## Non-aligned prompt lengths

Nine lengths, none divisible by the 32-row tile, the 64-token page or the
8192-token prefill chunk, sent as explicit token-id prompts through
`/v1/completions` so the length is exact. All succeeded
(`doc/vllm_integration/determinism_vllm.json`):

```text
1, 37, 127, 129, 1023, 2049, 4097, 8193, 12345   -> 9/9 ok, 8 output tokens each
```

Nothing caps or truncates the advertised context: the generator pads the ids with
the zero-embedding pad id, the layer stack masks the padded tail, and the logits
are sliced back to the logical last position.

## Determinism

`doc/vllm_integration/determinism_vllm.json`:

* **run-to-run** — the same prompt sent twice greedily returns identical token ids.
* **cross-batch-position** — 8 copies of the same prompt sent concurrently return
  one distinct output, equal to the single-request output. Each of the 32 decode
  rows indexes its own cache slot and page-table row, so this is the check that a
  row-indexing or page-table bug would fail.
* **logit-level** — `$vllm-integration` requires this once seeding tests fail, and
  four of them did. `logit_determinism.json` reads the model's own pre-penalty
  logprobs (any logprobs request routes to vLLM's host sampler on a 4-device mesh
  and returns `raw_logprobs`, so this is the model's output, not a sampler's) and
  compares the **full 20-candidate distribution at every position** — 160
  candidate logprobs, key-sorted so the comparison is order-free, not a top-1
  reduction. Result: **bitwise identical run-to-run and across 8 concurrent batch
  positions, max absolute delta 0.0 over all 160 candidates, candidate sets
  matching, one distinct distribution**. Tokens are an argmax and can hide a logit
  wobbling below the winning margin; these do not wobble at all. It also bounds
  the seeding failures from the other side: the distribution the sampler draws
  from is deterministic across runs and across batch rows, so what differs at
  batch 10/32 is the seed stream, not the model or the adapter.
* **standalone baseline** — serving reproduces the datatype-sweep standalone TT
  completions exactly. `determinism_vllm.json -> standalone_baseline` records
  `first_divergence: 2` for the run as executed, because that comparison
  re-encoded the *returned text* and the API strips `<|message|>`, shifting every
  id after it. The like-for-like comparison, recomputed offline from the two
  committed completion sets in
  `doc/vllm_integration/determinism_baseline_recheck.json`, is
  **character-identical over the full common prefix on all 6 prompts** (381-703
  chars each); the only difference is the 11-character control token. The script
  now compares stripped text rather than re-encoded ids.

## Qualitative verdict — pass

Prompt format: the tokenizer has a non-empty `chat_template`, so this checkpoint
is chat/instruct and the verdict is read from the **chat-rendered** arm. To keep
the arms comparable, the exact token ids the full-model stage recorded and posted
to the HF control are replayed to the server as token-id prompts, so the serving
arm, the previous TT stages and the HF control all ran the same input.
Artifacts under `doc/vllm_integration/qualitative/`: `qualitative_prompt_format.json`
(the prompt-format decision), `qualitative_prompts.json` (the pinned prompts and
token ids), `qualitative_tt_chat.json` (serving completions),
`qualitative_hf_chat.json` (the HF control, copied from the full-model stage),
`qualitative_comparison_chat.json` (the mechanical HF-vs-serving comparison, computed
by the full-model stage's own `compare()`),
`qualitative_vllm_vs_datatype_sweep_chat.json` (serving vs the previous stage's TT arm)
and `qualitative_stripped_divergence_chat.json` (both of those recomputed with the
API-stripped control token removed, which is what makes their divergence columns
readable — see below).

Both of those files report `identical: false` and `first_divergence` 1-2 on every
prompt, and **that headline number cannot be read directly.** `compare()` diffs raw
token ids, the API strips `<|message|>` (id 200023), and the standalone and HF arms
both contain it — so every served completion diverges at position 1-2 whatever it
says. `qualitative.py`'s docstring makes that same field the wrapper-bug tripwire
("divergence at token 0-2 is a wrapper bug, late divergence with both texts coherent
is ordinary numerics"), so it reads as tripped 6/6 for a reason that is not the
model. `doc/vllm_integration/bench/stripped_divergence.py` removes that one token
from both sides and recomputes it offline
(`qualitative/qualitative_stripped_divergence_chat.json`). The two pairs answer
differently, and only one of them is explained by the stripped token:

Per prompt, in `p0 p1 p2 p3 p4 p5` order:

| pair | raw `first_divergence` | with `<\|message\|>` removed |
|---|---|---|
| served vs datatype-sweep standalone | 2 2 2 2 2 2 | **none — identical over the full 127-token common prefix, 6/6** |
| served vs HF control | 2 1 2 2 2 2 | 12 1 33 27 43 31 |

So the stripped token fully accounts for the standalone comparison and **does not
account for the HF one** — serving is not token-identical to HF, and never was
expected to be. Five prompts diverge late with both texts coherent, which is the
ordinary-numerics reading the tripwire asks for. p1 is the exception at token 1, and
it is the channel token: served picks `to=user` (it answers directly) where HF picks
`to=self` (it thinks first). That divergence is **not serving-introduced** — the
datatype-sweep stage's own HF comparison, computed with `<|message|>` present on both
sides and no stripping involved, records `first_divergence_from_hf: 1` for p1 too —
and that stage investigated it there rather than leaving it asserted.
`doc/datatype_sweep/channel_margin_probe.json` scores that exact position under the
shipped `c14-attn4-cclbfp8-kv8` policy and finds the `=self`/`=user` choice decided by
**0.0625 logits** in prefill — one BFP4 quantization step — with the same position
already landing on `=user` by 0.125 in decode. A tie that narrow is what a channel
divergence at token 1 looks like; it is a property of the precision policy this stage
inherits, not of serving.

`compare()`'s remaining metrics on the serving arm are clean: worst adjacent
duplication 0.0 across the six prompts, non-ASCII ≤ 0.0018, and trigram-loop
fractions in the same band as the HF control's (TT 0.0472-0.1417, HF 0.0469-0.1172; they
match closely on p1/p3, TT runs higher on p2/p4/p5 and lower on p0, all far below
anything the checker treats as looping). `determinism_baseline_recheck.json` is a TT-vs-TT
recheck, so it corroborates the first row of the table and says nothing about the
second. The runner's own raw-completion arm is kept as labelled continuation stress
coverage in `readiness_vllm/vllm_qualitative_outputs.json`.

**Serving output is character-identical to the standalone model** on all six
prompts, with one systematic difference: the OpenAI API strips special tokens, so
the `<|message|>` control token present in the standalone text is absent from the
served text. Everything after it matches exactly — mechanically confirmed at token
level in `qualitative/qualitative_stripped_divergence_chat.json`, where removing that
one id leaves the served and standalone sequences with **no divergence at all** over
the full 127-token common prefix on all six prompts. Example (p0):

```text
serving  : ' to=selfWrite a haiku about machine learning.\n\nHaiku is 5-7-5 syllables. ...'
standalone: ' to=self<|message|>Write a haiku about machine learning.\n\nHaiku is 5-7-5 syllables. ...'
HF control: ' to=self<|message|>Write a haiku about machine learning.\n\nHaiku: 5-7-5 syllable structure. ...'
```

Judged against the required axes:

* **coherent** — yes. The checkpoint answers in an analysis-first style
  (`to=self` … "We need 5-7-5. Let's craft."), and the **HF control does the same on
  five of six prompts**, so this is the model's own behaviour, not a serving artifact.
  p1 is the exception in the other direction: serving picks the `to=user` channel and
  answers directly ("Sure! Here's a simple breakdown: ### **Supervised Learning** …")
  where HF picks `to=self`. Both are coherent and on topic; the divergence is
  pre-existing rather than serving-introduced, per the table above.
* **on topic** — yes, 6/6, including correct French for the translation prompt.
* **repetition** — the prompt is echoed before the answer on several prompts; the
  HF control echoes on the same prompts, for the same reason (it is part of the
  `to=self` analysis style, so it is absent from both arms exactly where the model
  goes straight to `to=user` — p1 in the serving arm). Mechanical degeneracy is
  absent: over all 36
  measurements across the three artifact sets (runner raw-completion arm, chat
  arm, overlap chat arm) the **worst adjacent duplication is 0.0286** against the
  0.10 critical threshold (`logs/degenerate_check_all.log`, exit 0, "No degenerate
  output detected"). One advisory `trigram_loop_fraction` of 0.5 belongs to a
  6-token completion, where three repeated tokens are half the sample; the checker
  treats trigram looping as advisory for exactly that reason.
* **gibberish** — none.
* **wrong-language drift** — none.
* **request contamination** — none; each completion answers its own prompt, and
  the cross-batch-position check above is the stronger form of the same test.

## Sampling suite — 62 passed, 10 failed, 1 skipped

`readiness_vllm/sampling_tests.log`, `--sampling-profile full`, `--tt-max-num-seqs 32`.

Per file, as counted from the log:

| file | passed | failed | skipped |
|---|---|---|---|
| `test_logprobs.py` | 20 | 0 | 1 |
| `test_seeding_and_variety.py` | 22 | 6 | 0 |
| `test_build_logprobs_from_topk.py` | 8 | 0 | 0 |
| `test_tt_penalties.py` | 4 | 2 | 0 |
| `test_host_only_params.py` | 4 | 1 | 0 |
| `test_config.py` | 3 | 0 | 0 |
| `test_structured_output_dp1.py` | 1 | 0 | 0 |
| `test_request_isolation.py` | 0 | **1** | 0 |

`test_request_isolation.py` holds exactly one test, `test_mixed_params_batch`, and
it is one of the seven reproducibility failures classified below — so isolation is
**not** a clean file, and the failing assertion is the seeded-reproducibility one,
not a cross-request-contamination one. Cross-request contamination is ruled out by
the cross-batch-position checks instead: 8 concurrent copies of a prompt return one
distinct completion and one distinct 160-candidate logit distribution.

The single **skip** is `test_logprobs.py::TestLogprobs::test_chat_logprobs_all_vocab`.
It skips because the request asks for `top_logprobs=-1` (all-vocab) and the server's
`max_logprobs` cap rejects it — the TT platform clamps `max_logprobs` to 20, since
the device computes top-32 logprobs and the OpenAI API limits to 20
(`vllm_tt_plugin/platform.py`). That is an expected framework limit, not a model
result, and it is the suite's only skip. Serving never crashed.

**Reproducibility-only failures (7)** — the class `$vllm-integration` allows to be
classified separately: `test_seeding`, `test_same_seeds_reproduce_across_batches`,
`test_uniform_seed_deterministic[10-0]`, `[10-1]`, `[32-0]`, `[32-1]`, and
`test_mixed_params_batch`. The supporting evidence that this is seed-stream
reproducibility at batch and not a correctness defect: **batch-1 seeding is
reproducible** — `test_specific_seed_reproducible[42/123/999/0]`,
`test_batch1_seed_reproducible[0/1]` and `test_uniform_seed_deterministic[1-0]`,
`[1-1]` all pass — and `test_different_seeds_produce_different_outputs`,
`test_top1_is_greedy` and `test_topk` pass. Correctness, logprobs, crash-free
serving and qualitative output all pass, which is the condition the skill attaches
to this classification.

**The three correctness-class failures are resolved — both by measurement, and
neither required a code change.** Full experiments in `AUTOFIX.md` round 2.

* `TestPresencePenalty::test_different_presence_penalties` /
  `::test_presence_penalty_mixed_batch` — **not a serving defect.** The
  discriminating fact is that `TestRepetitionPenalty` and `TestFrequencyPenalty`
  both pass, so the on-device penalty path runs and only *presence* is invisible.
  Presence is binary (one `-penalty`, clamped by the API to 2.0) where frequency
  scales with occurrence count. Measuring the pre-penalty logit margins on the
  test's own prompt (host path with `presence=0`, so the logprobs are raw) gives a
  **minimum gap of 3.0** between an already-seen winner and the best token that has
  *not* been seen — `sampling_failure_probe.json ->
  item1_presence_penalty.greedy_flip_margins.min_gap_winner_appeared_to_best_fresh`,
  over the 40 scored steps in that same object. A positive presence penalty
  subtracts only from already-seen tokens, so that is the gap it has to close, and
  3.0 is above the API's 2.0 clamp: *no legal presence penalty can flip that
  prompt's argmax*. The mirror figure in the same object,
  `min_gap_winner_fresh_to_best_already_seen` = 4.5, rules out the negative
  penalties the failing test also sweeps. (`presence_flip_probe.json` phase A
  measures the same prompt over a differently-scoped 45-step window and reports its
  own `min_margin` of 3.0; the numbers quoted here are the
  `sampling_failure_probe.json` ones.) Running a prompt whose measured margin is **0.5** through the
  **device** path (greedy, no logprobs anywhere, so it really is the traced split
  sampler) flips the output at **presence_penalty 0.475** and stays flipped for
  every larger value. Observed threshold matches the predicted margin to within
  one quantization step: the penalty reaches the logits.
  Note the obvious probe is a trap here — any logprobs request routes to host
  sampling on a 4-device mesh and vLLM returns `raw_logprobs`, computed *before*
  penalties, so a logprobs-based measurement shows 0.0 shift regardless.
* `TestHostOnlyParameters::test_allowed_token_ids` — **not a serving defect.**
  All five requests generated their full 10 tokens (`finish_reason: length`) and
  **every emitted id is inside its allowed set**, so the constraint works. Ids
  1-12 are byte-fallback tokens that each decode to U+FFFD; an incomplete UTF-8
  sequence is buffered by the detokenizer, so empty *text* is the correct result
  of forcing only those ids. The one request whose ids are printable
  (`[13,14,15]` = `!"#`) returns 10 characters through the same path
  (`sampling_failure_probe.json`). This also exercises the host-sampling
  compatibility mode, which `allowed_token_ids` forces.

## Artifacts

| what | path |
|---|---|
| primary benchmark, raw / normalized / log | `readiness_vllm/vllm_result.json`, `vllm_benchmark.json`, `vllm_benchmark.log` |
| CI serving-burst, raw / normalized / log | `readiness_vllm/vllm_ci_serving_result.json`, `vllm_ci_serving_benchmark.json`, `vllm_ci_serving_benchmark.log` |
| sampling suite | `readiness_vllm/sampling_tests.log` |
| server log (81 MB, not committed) | `readiness_vllm/server.log` |
| server log, committed excerpt | `doc/vllm_integration/logs/server_excerpt.log` |
| standalone-baseline recheck (offline) | `doc/vllm_integration/determinism_baseline_recheck.json` |
| qualitative, runner arm | `readiness_vllm/vllm_qualitative_outputs.json` |
| qualitative, prompt-correct chat arm | `doc/vllm_integration/qualitative/` |
| determinism + non-aligned lengths | `doc/vllm_integration/determinism_vllm.json` |
| logit-level determinism | `doc/vllm_integration/logit_determinism.json` |
| fallback + process audit | `doc/vllm_integration/serving_audit.json` |
| degenerate-output check log | `doc/vllm_integration/logs/degenerate_check_all.log` |
| KV pool ceiling measurement | `doc/vllm_integration/kv_budget_probe.json` |
| independent stage review | `doc/vllm_integration/stage_review.md` |
| adapter probe, 52-layer and reduced | `doc/vllm_integration/probe_full_fixed.json`, `probe_fixed.json` |
| async-scheduling overlap run | `doc/vllm_integration/async_overlap/` (server.log, qualitative/, overlap_vs_non_overlap.json) |
| presence-penalty / allowed-token-ids probes | `doc/vllm_integration/presence_flip_probe.json`, `sampling_failure_probe.json` |
| debugging record | `doc/vllm_integration/AUTODEBUG.md`, `AUTOFIX.md` |
| stage narrative | `doc/vllm_integration/work_log.md` |

## Limitations

1. **Seeded reproducibility at batch > 1** — see *Sampling*. Batch-1 seeding is
   reproducible; uniform seeds across a 10- or 32-row batch are not.
2. **Prefill is eager.** Traced prefill exists (`GeneratorConfig.prefill_trace`)
   but is keyed by padded prompt length and off by default, so TTFT carries host
   dispatch. It is worth 1.33x at 128 rows and 1.00x at 8192; a serving deployment
   with bucketed prompt lengths should turn it on.
3. **Prefix caching is off** and declared off. Nothing in this port implements or
   tests prefix reuse, and 39 of 52 layers are sliding-window.
4. **Uniform KV-cache spec.** Sliding-window layers are allocated full-attention
   sized blocks, which costs DRAM that a working `SlidingWindowSpec` would save.
   The blocker is vLLM's page-table zero-padding, described under *Attention type*.
5. **`-1` is the only decode position the shared paged kernels treat as inactive.**
   Any other negative is read as a huge unsigned index and hangs the mesh with no
   in-band error. This port now refuses such values in `positions_to_device`; the
   underlying kernel behaviour is shared and worth raising upstream.
