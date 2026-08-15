# vLLM integration — work log

Model: `meta-models/Muse-Glimmer-30B`
Autoport: `models/autoports/meta_models_muse_glimmer_30b`
Stage input: the completed datatype-sweep stage (`doc/datatype_sweep/`), whose
selected precision policy (`doc/datatype_sweep/selected_precision_config.json`)
is a build input to `build_generator` and therefore to serving.
Device: 4-die Blackhole P300_X2, mesh `(1, 4)`, `FABRIC_1D_RING`.

---

## 1. Environment

vLLM was not installed in this workspace's env
(`/home/ttuser/dev/muse-glimmer/muse-glimmer_pyenv`). Installed the Tenstorrent
fork's plugin and its pinned vLLM from the local checkout, using the plugin's own
script contents so the pins and overrides are the project's, not this stage's:

```bash
cd /home/ttuser/dev/vllm-tt-plugin
VLLM_TARGET_DEVICE=empty uv pip install --no-binary vllm \
    --override docs/vllm-overrides.txt vllm==0.24.0
uv pip uninstall torchaudio
uv pip install -e .
```

* `vllm 0.24.0+empty`, `vllm_tt_plugin` editable from
  `/home/ttuser/dev/vllm-tt-plugin`.
* `torch 2.11.0+cpu` and `transformers 5.15.0` are **unchanged** by the install.
  The evidence is the wheel's own metadata — it requires only
  `transformers>=5.5.3` and pins no torch, since the `empty` target builds no
  kernels — together with the versions read back from the env after installing
  (`python -c "import torch, transformers"` -> `2.11.0+cpu`, `5.15.0`), which are
  the versions every earlier stage used and the ones that supply the
  `muse_glimmer` config/model classes. (An earlier revision cited a pipfreeze
  before/after pair; those files were written outside the repo, are byte-identical
  to each other and are a freeze of the *system* interpreter rather than this env,
  so they evidenced nothing. Citation withdrawn.)
* Install log: `logs/install_vllm.log`.

## 2. What the adapter had to bridge

Read first: `tech_reports/LLMs/vLLM_integration.md`,
`models/common/readiness_check/contract_vllm.py`,
`models/tt_transformers/tt/generator_vllm.py`, and the plugin's own call sites
(`vllm_tt_plugin/model_runner.py`, `async_decode.py`, `worker.py`,
`platform.py`), which are the real contract.

Four genuine contract gaps between the readiness generator and serving:

1. **Cache sizing.** The standalone build sizes the paged pool at
   `max_batch_size x blocks_per_seq` — every user simultaneously at the full
   advertised context — and refuses anything smaller. vLLM owns one *shared*
   pool and hands out block ids from it; at this geometry the standalone rule is
   not satisfiable for a serving batch (32 x 2048 x 905,216 B = 59 GB against
   31.5 GiB). Added `MuseGlimmerModel.adopt_external_kv_cache()`, which
   validates everything that makes a cache interpretable by the paged ops (rank,
   local KV heads, block size, head dim, dtype), accepts any block count that
   holds at least one full-context sequence, frees the build-time pool, and
   updates the model/layer configs so `normalize_page_table`'s bounds check and
   `dram_report` describe the pool that is bound. `set_kv_cache()` keeps its
   strict same-shape contract for readiness callers.
2. **Async decode split.** `decode_forward(read_from_device=False)` now returns a
   device-resident carrier, `read_decode_output(async_read=True)` enqueues the
   minimal deferred read and returns its event, and
   `process_decode_output_host()` is host formatting only.
3. **Serving sampling state.** `apply_prefill_sampling_state()` and
   `apply_decode_sampling_state()` drive `models.common.sampling`'s own
   `apply_prefill_state` / `apply_decode_state` / seed-manager helpers — the same
   contract `models/tt_transformers/tt/generator.py` drives — so the adapter owns
   no sampling decision. `prefill_forward(sample_on_device=True)` returns sampled
   ids from the same untraced `_sample_eager` call `generate()` already used for
   a prompt's first token.
4. **Stale-input rule.** `decode_forward(refresh_inputs=False)` is the overlap
   contract: when the step samples on device and the padded batch layout has not
   changed, the host copies of `tokens`/`start_pos` may predate the previous
   step's token, so the device copies are authoritative and nothing is restaged.

Shared-infra changes:

* `models/common/readiness_check/run_vllm_server.py`: `--mesh-device` accepted
  only the four Wormhole presets, which made the runner unusable on a Blackhole
  P300_X2. It now accepts every preset the TT plugin knows
  (`vllm_tt_plugin.utils.dp_discovery._MESH_GRID_PRESETS`, which includes
  `P300x2` -> (1, 4)) plus an explicit `"(rows, cols)"` grid.
* `models/common/readiness_check/run_vllm_server.py`: the runner passed the TT
  plugin config as `--plugin-config`, which **vLLM 0.24.0 — the version the
  plugin's own `docs/install-vllm-tt.sh` pins — rejects outright**:
  `api_server.py: error: unrecognized arguments: --plugin-config`. The launcher
  dies before any TT code runs, which reads as a model failure. The plugin always
  *reads* the same place (`vllm_config.additional_config["tt"]`, see
  `vllm_tt_plugin/config.py::get_tt_config`); only the CLI spelling moved between
  vLLM versions. `_tt_config_flag()` now asks the installed `EngineArgs` which
  field exists and picks `--additional-config` or `--plugin-config` accordingly,
  so the runner works on either.
* `vllm-tt-plugin` `platform.py::register_tt_models()`: registered
  `MuseGlimmerForConditionalGeneration` (and the `TT`-prefixed alias, plus
  `...ForCausalLM` spellings) against
  `models.autoports.meta_models_muse_glimmer_30b.tt.generator_vllm`. The plain
  HF name is registered for the same reason Gemma4 is: the checkpoint has nested
  text/vision configs, so `hf_config != hf_text_config` and upstream's resolver
  would fall back to `TransformersMultiModalForCausalLM` in
  `ModelConfig.__post_init__`, before the plugin's `TT`-prefix rewrite runs.
* `vllm-tt-plugin` `worker.py`: optional `fabric_packet_payload_bytes` in the TT
  plugin config, so serving can open the fabric with the same 8192-byte router
  payload the model's collectives were measured with
  (`doc/context_contract.json` -> `device.fabric_packet_payload_note`). Without
  it the served fabric silently differs from the offline one.

## 3. Minimum-surface bring-up loop

Per `$vllm-integration`, the adapter was brought up on a **reduced serving
target** first: `MUSE_GLIMMER_VLLM_LAYER_INDICES=0,3` builds one sliding and one
full-attention layer with the same generator, adapter, registration, cache and
page-table shapes, terminal norm / LM head / sampling path and traces.

`doc/vllm_integration/bench/adapter_probe.py` drives the adapter through exactly
the plugin's call sequence without vLLM, so the parts a live server cannot show
directly are inspectable: which inputs are refreshed on which step, what a
steady-state token costs in counters, and whether the stale-input contract holds
when the host tensors are wrong on purpose.

### 3.1 Two real defects the reduced target caught immediately

* **`normalize_page_table(None)` named blocks that do not exist.** The default
  table gives every cache *slot* a private run of `blocks_per_seq` blocks; with a
  shared serving pool that is 32 x 2048 = 65536 block ids against a 4128-block
  pool, and the very first call — `_allocate_device_inputs`, before any request
  supplies a table — raised
  `page_table references block 65535 but the cache holds 4128 blocks`. Fixed by
  bounding the private runs by the pool as well as by the slot count; a
  standalone build sizes the pool so the bound never binds, so its behaviour is
  unchanged.
* **Warmup wrote 30 rows to one physical block.** The decode warmup drives all 32
  rows at position 0; with the default table and a shared pool most rows aliased
  the same block. `warmup_model_decode` now builds an explicit one-distinct-block-
  per-row table.

### 3.2 The multi-request hang

Recorded in `AUTODEBUG.md` with the full bisect matrix, `triage/tt-triage.txt`
and the per-arm logs and JSON. Summary: the reduced target completes three
batch-1 requests (128 / 37 / 4097 tokens, 16 traced decode steps each,
byte-identical across processes and across a device reset) and then hangs in the
three-concurrent-slot section, inside the decode trace at
`PagedUpdateCacheDeviceOperation` with the fabric routers holding unretired NOC
reads.

The **shipped 52-layer target hung in the same place** (`logs/probe_full.log`),
so it was not an inner-loop artifact. Handed to `$autofix` with a forked
subagent; its report is `AUTOFIX.md`.

### 3.3 Root cause: an illegal decode position, and where my own localisation was wrong

`$autofix` found that **the multi-slot decode section was never failing**, and
that every bisect matrix above was measuring the wrong region. The probe prints
nothing between the last `prompt_len=4097 -> [...]` line and the end of the run,
so "the log goes silent after the all-gather pairs" covered the three-slot decode
loop *and* the host-sampling compatibility step that follows it. Instrumenting
per step (`--verbose-steps`) showed all 16 multi-slot decode steps completing
with sane device inputs — `current_pos=[111, 145, 76, -1, ...]`,
`page_table[:4,:6]=[[74,75,76,77,78,0], [79,80,81,82,83,0], [84,85,86,87,88,0],
[0,0,0,0,0,0]]` — and the hang landing on the step after.

The cause is a value, not a race. The probe deliberately writes `-7` into its host
position tensor after each steady step, to prove the stale-input contract. Every
traced device-sampled step ignores it, which is exactly what the contract
promises. The final host-sampling step passes `sampling_params=None`, so
`sample_on_device=False`, so `refresh_inputs=True` **by contract** — host state is
authoritative there — and `-7` is staged to the device.
`writer_paged_fused_update_cache_interleaved_start_id.cpp:82` skips an inactive
row by comparing the index against `(uint32_t)-1` *exactly*, so `-7` becomes
`0xFFFFFFF9`, `virtual_block_id = update_idx / block_size = 67108863` reads past
the page-table circular buffer, and the op issues a NOC transaction to a garbage
address that never retires. That is precisely the triage signature: an op that
never completes plus fabric routers stuck in `transaction_flushed`.

It retro-explains everything: `--no-stale-inputs` passed because its positions
are always legal; `--decode-steps 2` still hung because the host-sampling step
runs regardless; all four `multi_slot_bisect.py` arms passed because that harness
never writes `-7` and has no host-sampling step; the 52-layer target hung
identically because it is the same probe and the same final step; and it was
deterministic across resets because it is a value bug.

Fix, in two parts:

* `MuseGlimmerModel.positions_to_device` — the single funnel for `current_pos` /
  `rope_pos_ids` — now refuses any position that is neither `-1` nor in
  `[0, max_seq_len)`, naming the offending rows and values and the device
  consequence. `-1` is the *only* legal negative because that is the only value
  the shared kernel's skip test matches. A caller bug now fails as a caller bug
  instead of wedging the mesh with no in-band error.
  `tests/test_full_model.py::test_out_of_range_start_pos_is_rejected_instead_of_hanging_the_mesh`
  pins it next to the existing `-1`-sentinel test.
* `bench/adapter_probe.py` — the host-sampling step is not covered by the
  stale-input rule (it restages from the caller by contract), so it is handed the
  real positions. `--keep-stale-for-host-sampling` retains the old behaviour to
  reproduce the original hang and prove the guard fires
  (`logs/probe_guard.log`: `RC=1` in ~60 s, no hang, no reset).

Nothing about the serving contract was weakened to get here: no drain was added,
the traced decode path and on-device split sampling are untouched,
`supports_async_decode` stays `True`, and a device-sampled step with an unchanged
layout still reads no host token/position state.

### 3.4 Reduced and full targets after the fix

`probe_fixed.json` (reduced) and `probe_full_fixed.json` (**the shipped 52-layer
model**, stale inputs, async read) both pass. The 52-layer run:

```text
status ok, build 174.8 s, warmup 2.25 s
decode_trace_captured true, sampling_trace_captured true
multi-slot (3 concurrent rows): rows_are_distinct true
  counters over 16 steps: trace_replays 16, token_refreshes 1,
  position_refreshes 1, page_table_refreshes 1, synchronizations 0, readbacks 16
  serving counters: sampling_param_refreshes 1, sampling_param_reuses 15
host-sampling compatibility step: logits [32, 1, 202048], finite
```

One token/position/page-table refresh for sixteen tokens is the whole
stale-input contract measured rather than asserted: the single refresh is the
`reset_batch=True` step after the prefill, and the other fifteen read nothing
from host. Single-slot tokens are byte-identical to the pre-fix logs, so the fix
moved nothing about what the model computes.

## 4. Plugin registration and capacity contract, verified off-device

```text
REG  ['MuseGlimmerForCausalLM', 'MuseGlimmerForConditionalGeneration',
      'TTMuseGlimmerForCausalLM', 'TTMuseGlimmerForConditionalGeneration']
ARCH models.autoports.meta_models_muse_glimmer_30b.tt.generator_vllm
       :MuseGlimmerForConditionalGeneration
caps {'supports_prefix_caching': False, 'supports_async_decode': True,
      'supports_sample_on_device': True}
get_max_tokens_all_users(max_model_len=131072, max_num_seqs=32) = 1048576
max_num_seqs=33 -> ValueError (decode batch ceiling 32, nlp_create_qkv_heads_decode)
get_kv_cache_spec -> 52 x FullAttentionSpec, sliding_window=None,
                     model.layers.0.self_attn .. model.layers.51.self_attn
```

All 52 specs are identical, so vLLM builds a **single** KV-cache group, leaves
`block_tables_per_layer` unset, and takes the legacy uniform
`allocate_kv_cache(shape, dtype, num_layers)` path. The adapter refuses a
per-layer submission rather than dropping it, so if a future plugin change starts
sending one it fails loudly instead of paging against the wrong pool.

`sliding_window=None` on every spec is deliberate: this port's decode passes an
*absolute* position to `paged_update_cache` / `paged_scaled_dot_product_attention_decode`,
while vLLM's `SlidingWindowSpec` zero-pads a sliding group's page table past
`sliding_window / block_size` entries, so positions beyond the window would
collapse onto physical block 0. The sliding layers still attend only their window
(the SDPA op gets `sliding_window` on the read side from `tt/multichip_decoder.py`),
so the model's semantics are unchanged; the uniform spec costs memory, not
correctness. `models/tt_transformers/tt/generator_vllm.py` makes the same choice
for Gemma3 and GPT-OSS, for the same documented reason.

## 4b. Two launch-time incompatibilities the off-device checks could not see

Both killed the launcher before any TT code ran, and both read as model failures
until the traceback is opened.

1. **`--plugin-config` vs `--additional-config`** — see section 2. Fixed by
   asking the installed `EngineArgs` which field exists.
2. **`This model does not support --runner generate`.** `ModelConfig.__post_init__`
   decides whether a checkpoint is generative by inspecting the class its registry
   resolves for the checkpoint's architecture, and that happens *before*
   `TTPlatform.check_and_update_config` prepends `TT` to the architecture list.
   Most TT models never notice: vLLM finds an upstream torch implementation for
   `LlamaForCausalLM` and friends and inspects *that*, while the plugin's prefix
   logic routes execution to the TT class. Muse-Glimmer has no upstream vLLM
   implementation, so the inspection lands on this port's adapter, which failed
   `is_vllm_model`. Fixed with the same protocol shim
   `models/demos/gemma4/tt/generator_vllm.py` carries for the same reason:
   `embed_input_ids`, `forward(input_ids, positions)` and `compute_logits` that
   raise (execution goes through `prefill_forward` / `decode_forward`), plus a
   `vllm_config` keyword on `__init__` that is accepted and ignored. Verified:

   ```text
   is_vllm_model: True   is_text_generation_model: True   supports_multimodal: False
   ```

   `supports_multimodal` staying False is the point of not declaring
   `SupportsMultiModal`: the checkpoint carries a vision tower, this port
   implements the text decoder only, and the request path must stay text-only.

## 5. Serving capacity

`get_max_tokens_all_users` returns `min(max_model_len x max_num_seqs, 1_048_576)`
tokens, which the TT worker turns into
`ceil((tokens + block_size x max_num_seqs) / block_size)` = **16416 blocks** at the
shipped configuration. That is a *pool* statement, not a context one:
`--max_model_len` stays at the advertised 131072 and one request may still take
all of it (2048 of the pool's blocks).

The number comes from the byte budget in `doc/context_contract.json`: 31.46
GiB/device, less 5.19 GB of weights, 0.134 GB of RoPE tables and the 0.4 GB trace
region, leaves ~27.7 GB; 16416 blocks x 905,216 B is 14.86 GB, which leaves ~12.9
GB for the prefill/decode working set and allocator headroom. Measured on device:
the 52-layer allocation and adoption succeed and report
`14.86 GB/device, 1050624 tokens across all users` (`logs/probe_full.log`).
At 64 tokens per block that is eight concurrent requests at the full advertised
context, or all 32 request slots at 32768 tokens each.
`MUSE_GLIMMER_VLLM_KV_TOKEN_BUDGET` overrides it for experiments.

`max_num_seqs > 32` is refused with a `ValueError` naming
`nlp_create_qkv_heads_decode_device_operation.cpp:45-51`, which hard-caps
`num_users` at 32; that is a device-op ceiling, not a choice this port makes.


## 6. The serving run

One server, one model load, every gate attached to it
(`bench/run_serving_evidence.sh`, log `logs/serving_evidence.log`). Exact command
in `README.md`; wrapper `bench/serve.sh`.

```text
launching server 10:34:45   server ready 10:38:36   (~4 min: weights, KV pool, warmup)
sampling_full        rc=1   62 passed, 10 failed, 1 skipped
qualitative_runner   rc=0
qualitative_chat     rc=0
determinism          rc=0
benchmark            rc=0   both profiles, 1/1 and 32/32 completed
shutdown + audit     rc=0   clean
```

Server-side configuration confirmed from `readiness_vllm/server.log`:
`Resolved architecture: MuseGlimmerForConditionalGeneration`, V1 engine v0.24.0,
`max_model_len=131072`, `max_num_seqs=32`, `block_size=64`,
`enable_prefix_caching=False`, `TTModelRunner: trace_mode=decode_only,
sample_on_device_mode=all, enable_model_warmup=True`, and chunked prefill disabled
with `max_num_batched_tokens` bumped 2048 -> 131072 (without that bump the
scheduler would refuse any prompt over 2048 tokens, so it is load-bearing for the
advertised context).

### 6.1 Performance, and the comparison that matters

Primary single-user 128/128/1: TTFT 72.68 ms, TPOT mean 23.049 ms ->
**43.39 t/s/u**, ITL p50/p99 23.015/23.641 ms, 42.66 tok/s aggregate, 1/1
complete, 0 missing tokens.

The datatype-sweep stage measured the same 128/128/1 shape through the public
generator at 43.33 t/s/u (23.078 ms/token). Serving is **100.1 %** of that, i.e.
there is no measurable vLLM-specific decode overhead left: request handling,
scheduling, sampling, token feedback and readback together cost less than the
run-to-run spread of the decoder. Teacher forcing (37.43 t/s/u) is the lower bound
the skill asks serving to clear; it clears it by 15.9 %.

That result is a direct consequence of the two decisions the stage argued for: the
steady decode step reads nothing from host (counters above), and the sampling
parameters are not restaged when they have not changed
(`sampling_param_reuses 15/16`).

CI serving-burst 100/100/32: 713.95 tok/s aggregate, TPOT mean 23.102 ms
(43.29 t/s/u), ITL p50/p99 23.053/30.350 ms, TTFT p50 2193.67 ms, 32/32 complete.
Not the headline: 32 prompts are admitted as one burst, so TTFT carries 32 queued
prefills. Its value is that 32 concurrent sequences serve at the same per-user
TPOT, for 16.5x the single-user aggregate.

### 6.2 Qualitative

Verdict **pass**; full reasoning and the side-by-side excerpts are in `README.md`.
The one systematic difference from the standalone arm is that the OpenAI API
strips special tokens, so `<|message|>` is absent from served text; everything
after it is character-identical to the datatype-sweep standalone completions on
all six prompts — recomputed at token level in
`qualitative/qualitative_stripped_divergence_chat.json`, where removing that one id
leaves *no divergence at all* against the standalone arm over the full 127-token
common prefix, 6/6. The HF control shows the same analysis-first style and the same
prompt echo on five of six prompts, so neither is a serving artifact; p1 is the
exception, where serving takes the `to=user` channel and HF takes `to=self`. That
one is pre-existing rather than serving-introduced — the datatype-sweep stage's own
HF comparison records `first_divergence_from_hf: 1` for p1 as well, computed with
`<|message|>` present on both sides and no stripping involved.

This also explains the `first_divergence: 2` recorded in
`determinism_vllm.json -> standalone_baseline`: that comparison re-encodes the
*returned text*, and with `<|message|>` stripped the re-encoded ids necessarily
diverge at position 2. It is an artifact of comparing text-round-trips, not a
model difference — the completions themselves match.

`check_degenerate_output.py --scope vllm` passes; over all 36 measurements across
the three artifact sets the worst adjacent duplication is **0.0286** against the
0.10 critical threshold (`logs/degenerate_check_all.log`, exit 0).

### 6.3 Fallback and process audit

`bench/audit_serving.py` -> `serving_audit.json`: **clean**. No degraded markers
in an 84 MB server log — no prefill-trace capture failure, no orphaned trace, no
reduced target served, no "Disabling async scheduling", no engine crash.

Two of the markers this section originally leaned on were not real, and section 7
records the repair: "no eager decode" and "no host sampling on a measured step"
were being tested against strings nothing emitted. Those paths now announce
themselves, the audit verifies its own markers are emittable, and it reports that
the committed logs predate the new markers — so for those two conditions the
settling evidence is the ITL distribution and the step-resolved counters, named in
the README, not this log. Confirmed events: KV cache adopted,
decode trace captured, sampling trace captured, chunked prefill and prefix caching
disabled as declared. No surviving `EngineCore` / `vllm.entrypoints` process and
nothing holding `/dev/tenstorrent/*` after shutdown.

### 6.4 Sampling suite triage

62 passed, 10 failed, 1 skipped. Seven failures are the reproducibility-only class
the skill allows to be classified separately, and the classification is evidenced
rather than asserted: batch-1 seeding *is* reproducible
(`test_specific_seed_reproducible[42/123/999/0]`, `test_batch1_seed_reproducible[0/1]`,
`test_uniform_seed_deterministic[1-0]`/`[1-1]` all pass) while the uniform-seed
tests fail only at batch 10 and 32; `test_logprobs.py` is 20 passed with 1
skipped and no failures (the skip is `test_chat_logprobs_all_vocab`, an expected
framework limit — see `README.md`); serving never crashed; qualitative output is
clean. Those are exactly the conditions the skill attaches to the classification.
Note `test_request_isolation.py` is *not* a clean file: it holds exactly one test
and that test is one of the seven reproducibility failures. Cross-request
contamination is ruled out by the cross-batch-position checks instead.

Three failures are correctness-class and were handed to `$autofix`:

* `TestPresencePenalty::test_different_presence_penalties` and
  `::test_presence_penalty_mixed_batch` — 8 concurrent requests sweeping
  presence_penalty -1.5..2.0 at temperature 0 produce one unique output. The
  discriminating evidence already in hand is that **`TestRepetitionPenalty` and
  `TestFrequencyPenalty` both pass**, so the on-device penalty path runs and only
  *presence* is invisible; presence is binary (<=2.0 of logit shift) where
  frequency scales with occurrence count.
* `TestHostOnlyParameters::test_allowed_token_ids` — request 0 returns empty text.
  Off-device, ids 1-12 in this tokenizer are byte-fallback tokens that each decode
  to U+FFFD, and an incomplete UTF-8 sequence is buffered by the detokenizer, so
  empty *text* is expected when only those ids are allowed; what needed proving is
  that the request generated its tokens rather than failing.

**Both are resolved by measurement, and neither needed a code change**
(`AUTOFIX.md` round 2):

* **presence penalty — not a defect.** The pre-penalty logit margin on the test's
  own prompt is 3.0 at minimum over the 40 scored steps, above the 2.0 clamp the API puts on
  `presence_penalty`, so no legal value can flip that argmax. On a prompt whose
  measured margin is 0.5, the **device** path (greedy, no logprobs) flips at
  `presence_penalty=0.475` and stays flipped — the observed threshold matches the
  predicted margin to within one quantization step. The trap worth recording: any
  logprobs request routes to host sampling on a 4-device mesh and vLLM returns
  `raw_logprobs` computed *before* penalties, so a logprobs-based probe measures a
  0.0 shift no matter what the device does. A first probe did exactly that and its
  `measured_relative_shift: 0.0` must not be read as evidence of a defect.
* **`allowed_token_ids` — not a defect.** All five requests generated their full 10
  tokens and every emitted id is inside its allowed set; ids 1-12 are byte-fallback
  tokens decoding to U+FFFD, which the detokenizer buffers, so empty *text* is
  correct. The request whose ids are printable returns 10 characters through the
  same path.

Neither changed code, so the committed `readiness_vllm/sampling_tests.log` remains
the stage's sampling evidence and its counts stand at 62/10/1.

### 6.5 The --async-scheduling overlap validation

`$vllm-integration` requires a focused overlap test under `--async-scheduling`
with `sample_on_device_mode=all`, because that is the configuration in which vLLM
may submit decode step N+1 before sampled token N has been applied to host
scheduler state. None of the runs above passed `--async-scheduling`, so the stage
owed this one.

`bench/run_async_overlap.sh` launches a server with
`--additional-server-args=--async-scheduling` and the identical TT config, writing
to `doc/vllm_integration/async_overlap/` so it cannot overwrite the committed
non-overlap evidence. It fixed one shared-infra bug on the way: `serve.sh` passed
the extra args as a separate token and argparse rejects a value starting with
`--` (`expected one argument`); the `--flag=value` form is used now.

```text
server ready 11:20:42
ASYNC_ACCEPTED: no 'Disabling async scheduling' in the server log
STEP async_qualitative rc=0
STEP async_degenerate  rc=0

overlap vs non-overlap, 6 pinned prompts:
  identical completions          6 / 6
  max adjacent token duplication 0.0000  (critical threshold 0.10)
  control tokens leaked to text  none
```

The plugin accepting the capability matters as much as the output: had it printed
`Disabling async scheduling`, the declaration in `model_capabilities` would have
been refused and the overlap path never exercised. Byte-identical output under
overlap is the direct evidence that the stale-input rule holds — a stale token or
position on an overlapped step is precisely what would produce doubled subwords or
repeated control tokens here.

## 7. Stage review round 1 — `more-work-needed`, and what changed

Full report: `stage_review.md`. The reviewer confirmed the load-bearing claims by
re-deriving them (43.39 vs 43.33 is like-for-like; the six completions are
character-identical after stripping one `<|message|>`; overlap vs non-overlap is
6/6 byte-identical; `max_model_len` 131072 matches the contract; the position
guard is at the right boundary and is not masking a serving bug) and returned
three required items plus several smaller ones. Each is answered below.

### P1 — a cited artifact contradicted the claim it backed

`presence_flip_probe.json` carried
`presence_penalty_reaches_device_logits: false` while `README.md` and `AUTOFIX.md`
cited that exact file as proof of the opposite. The *measurements* were never in
question — the ladder shows `[0.0 … 0.375]` unchanged and `[0.475 … 2.0]` changed
against a 0.5 margin — but the file had been written by an earlier revision of the
probe whose verdict rule differed from the one it printed, and it was missing the
`monotone_step` / `largest_penalty_with_no_change` fields the shipped script
emits. A stale verdict field under a live citation is exactly the kind of thing
that turns into a false claim later.

Fixed by making the rule one pure function, `phase_b_verdict(runs, margin)`, used
by both the live probe and a new `--recompute-verdict` mode that re-derives the
fields from a committed ladder offline and rewrites *only* the derived fields:
`False -> True`, `observed_flip_threshold 0.475`, `largest_penalty_with_no_change
0.375`, `monotone_step True`, `within_tolerance True` (tolerance 0.15). The
artifact now carries a `verdict_provenance` field naming the recompute and the
value it replaced, and the shipped script exits 0 on it.

### P2 — the fallback audit was checking for strings nothing emits

`audit_serving.py` grepped the server log for `"Untraced decode"` and
`"host_sampling=True"`. Neither is ever logged: both occur only in *docstrings* in
`tt/generator.py`. So `serving_audit.json`'s `clean: true` was vacuous for exactly
the two conditions that would invalidate the headline decode number, while the
README claimed "the audit confirms no measured decode step took it".

Fixed in both directions:

* the degraded paths now announce themselves — `_decode_step_eager` and
  `_host_argmax` each log once per generator with an unambiguous
  `DEGRADED PATH untraced_eager_decode` / `DEGRADED PATH host_argmax_fallback`
  marker. A path that silently degrades the measured decode rate has to be able to
  say so;
* the audit now **verifies its own markers**. `verify_markers_live()` checks every
  `degraded` marker is a string some source file can actually emit, and `clean` is
  false if any is dead. It immediately earned its keep: it caught a wrong repo-root
  depth in its own path arithmetic, and then caught that
  `"prefill tracing is disabled for this generator"` is assembled from two source
  literals and so can never be found as a contiguous source substring (narrowed to
  `"tracing is disabled for this generator"`);
* what a server log *cannot* settle is now recorded in the report itself, under
  `conditions_evidenced_elsewhere`, pointing at the step-resolved counters in
  `probe_full_fixed.json` and at the benchmark's ITL distribution — a step that
  gathered the full 202048-wide vocab could not hide inside ITL p50 23.015 /
  p99 23.641 ms.

Re-run over both server logs: `clean: true`, `all_degraded_markers_live: true`, no
degraded markers, no leftover processes.

### P2 — the KV pool budget was a paper number

`KV_CACHE_TOKEN_BUDGET` was derived from a byte budget with a ~12.9 GB reserve and
no measurement, no OOM and no larger attempt — "largest feasible value" asserted
rather than shown.

`bench/kv_budget_probe.py` now measures it. A rung counts as feasible only if, at
that pool size, the model allocates all 104 cache tensors, captures both traces,
runs a **full 8192-token prefill chunk** — the largest activation working set the
serving path ever builds — and replays traced decode on top of it. Ladder
descends, so the first success is the ceiling on the least-fragmented allocator:

```text
free DRAM after weights          27.10 GB/device
rung 28672 blocks (1,835,008 tok, 25.95 GB/dev)  FEASIBLE
  free after alloc  3.00 GB      free at prefill peak  2.99 GB
```

The top rung succeeded, so **28672 blocks / 1,835,008 tokens** is a proven
*lower bound* on the ceiling rather than the ceiling itself — 1.75x what ships,
and the true limit is somewhere above it. 16416
still ships, and the margin is now a stated engineering choice rather than an
accident: the ceiling leaves 3.0 GB at a *single-user* prefill peak, while a
serving process must also absorb allocator fragmentation across thousands of
requests and the 39 sliding layers' persistent prefill tails, and an OOM
mid-request is a much worse failure than a smaller pool. Raising it is a one-line
change and belongs to optimized-vLLM, which re-runs the serving evidence anyway;
doing it here would leave this stage's committed server log and benchmarks
describing a configuration they were not produced with. The constant's docstring
now cites the measurement.

### Smaller findings

* **Wrong number in the README.** "worst adjacent duplication 0.0121 across both
  artifact sets" was wrong — 0.0121 is the primary set, the overlap set is 0.0286.
  Corrected to 0.0286 over all 36 measurements across three sets, and the
  previously-uncommitted degenerate-check log for the primary set is now saved as
  `logs/degenerate_check_all.log` (exit 0). The one advisory
  `trigram_loop_fraction` of 0.5 is a 6-token completion and is called out as such.
* **`probe_guard.json` does not exist.** The guard fires and the probe exits 1
  *before* writing its JSON, by design, so the cited path can never exist. The
  citations now point at `logs/probe_guard.log`, which carries the exact
  `ValueError`.
* **The README's canonical one-liner would not reproduce the evidence.** The
  runner stops after a failing stage and the sampling stage exits non-zero here, so
  a one-shot invocation never reaches the benchmarks. The README now says so
  explicitly and points at `run_serving_evidence.sh`, which holds one server open
  and attaches each stage to it — also what keeps the 52-layer model to a single
  ~4-minute load.
* **Eager prefill's stated reason was the weaker one.** Bucketed capture *is*
  expressible — `warmup_model_prefill` already enumerates eight buckets. The real
  constraint is `prefill_trace_max_entries = 1`, which would thrash on mixed
  serving lengths. The README now states that, keeps the measured 1.33x-at-128-rows
  figure, and names raising it as optimized-vLLM's work.
* **The `4.5` margin figure was cited to the wrong file.** It is in
  `sampling_failure_probe.json -> item1_presence_penalty.greedy_flip_margins`, not
  in `presence_flip_probe.json` whose phase A records only the
  already-emitted-winner direction. Citation corrected, and it is the figure that
  rules out the *negative* presence penalties the failing test also sweeps.
* **`audit_serving.py`'s docstring claimed a scan it does not do.** Corrected.
* **Penalties under `--async-scheduling` are untested.** They are also
  unreachable: vLLM's `can_use_steady_decode_fast_path` returns `False` when
  `prompt_tokens` / `output_tokens` are set, and the runner populates both for any
  batch with active penalties, so a penalised request never takes the overlapped
  path. Recorded in the README rather than left as an open corner.
* **Shared-infra scope.** `_tt_config_flag()` preferred `--additional-config`
  whenever the field exists, which would have changed behaviour for every other
  model on a build exposing both. Reversed: `--plugin-config` wins when both are
  present, so the change is a strict repair for builds that dropped it (like the
  pinned 0.24.0) and a no-op everywhere else.

### 7.1 Logit-level determinism — the last review gap

`$vllm-integration` line 157 requires that, when determinism tests fail, the
*logits* be shown reproducible across runs and batch positions rather than only
the tokens. Four seeded tests fail here, so the clause is live, and the stage had
only token-level and text-level evidence.

`bench/logit_determinism.py` reads the model's own pre-penalty logprobs — any
logprobs request routes to vLLM's host sampler on this 4-device mesh and returns
`raw_logprobs`, which is precisely why they are a valid probe of the model's
output rather than of a sampler — and compares them run-to-run and across 8
concurrent requests occupying different rows of the 32-row decode batch.

**The first version of this probe was not good enough, and review round 2 was
right to say so.** It collapsed each position to `(argmax token, its logprob)`
before comparing. On a confident greedy step the top-1 logprob saturates at
`0.0` — in the committed run, three of eight positions were exactly `0.0` and four
more were below `1e-5` — so the comparison was nearly information-free at exactly
the point it had to discriminate, while the probe's own docstring said it existed
because "an argmax can hide a logit that wobbles below the winning margin". It
measured the thing it was written to avoid measuring.

It now compares the **full 20-candidate distribution at every position**,
key-sorted so the comparison is order-free, and reports the largest absolute
delta over every candidate rather than over eight top-1 values; a candidate set
that differs at all is treated as non-identical outright. That matters for the
inference being drawn, because a seeded sampler draws from the whole
distribution, not from the argmax.

```text
RUN_TO_RUN   bitwise_identical=True  candidates=160  candidate_sets_match=True  max_delta=0.0
CROSS_BATCH  distinct_distributions=1  all_identical=True  candidates=160  max_delta=0.0
```

Bitwise, not merely close, over all 160 candidate logprobs (8 positions x 20).
That bounds the seeded-reproducibility failures from the other side: the
distribution the sampler draws from is deterministic across runs and across batch
rows, so what differs at batch 10 and 32 is the seed stream, not the model or the
adapter.

## 8. Stage review rounds 2 and 3

Round 2 verified round 1's fixes as genuine and found three more; round 3 verified
those and found two documentation-accuracy items. All are recorded here because
two of the four rounds' findings were of the same species — *a document or
artifact claiming more than the thing it cited actually showed* — and that is the
failure mode this stage has had to work hardest against.

### Round 2

* **The logit-determinism probe measured the wrong quantity.** It collapsed each
  position to `(argmax token, its logprob)` before comparing, and on a confident
  greedy step the top-1 logprob saturates: three of eight positions were exactly
  `0.0` and four more below `1e-5`. So the probe was nearly information-free at
  precisely the point it had to discriminate — while its own docstring said it
  existed because "an argmax can hide a logit that wobbles below the winning
  margin". It now compares the **full 20-candidate distribution at every
  position**, key-sorted, reporting the largest delta over every candidate and
  treating a differing candidate set as non-identical outright. Re-run: 160
  candidates, bitwise identical run-to-run and across 8 concurrent batch
  positions, delta 0.0.
* **The audit still could not see serving host sampling.** The marker added in
  round 1 sat on `_host_argmax`, which the *standalone* `generate(host_sampling=True)`
  mode calls and the serving adapter never does; the serving route is the
  `gather_and_untilize_logits` branch of `decode_forward`, which emitted nothing.
  That branch now announces itself (`DEGRADED PATH serving_full_logits_readback`),
  the `host_argmax_fallback` marker's meaning was corrected to say it is expected
  absent on a serving log, `verify_markers_live` now collects only **non-docstring**
  string literals via `ast` (the previous version would have accepted a
  docstring-only marker — exactly the original bug), and a new `marker_provenance`
  section reports each log's mtime against the newest marker source and states
  that a log predating a marker is not evidence for it. Both committed logs do
  predate the markers, and the report says so.
  The README no longer claims the audit settles it. What settles it is timing and
  counters: E2E 2999.886 ms is fully accounted for by TTFT 72.679 ms plus 127
  intervals at the 23.0489 ms mean TPOT (2999.89 ms), leaving no unaccounted time
  for an outlier step, with ITL p50 23.015 / p99 23.641 ms bounding it from the
  other side; a full-vocab gather and readback of ~12.9 MB cannot hide in that.
* **work_log still carried the stale 0.0121** duplication figure. Corrected to
  0.0286 over 36 measurements.

### Round 3

* **The `l1_small_size` rationale was false, and cited to the artifact that
  disproves it.** The README and `serve.sh` said "both larger and smaller fail".
  `doc/context_contract.json -> device.l1_small_note` records the opposite: 32768
  and 8192 fail, but **7168, 6144 and 4096 all pass**. 6144 is a *margin* choice —
  24 distinct CCL programs with 1,152 B of headroom against 7168's 128 B, while
  4096 makes the region itself the constraint — carried from the decoder stage,
  which is where it was measured. This stage measured no new value and now says so.
* **The status table said "byte-identical to the standalone model"** while the body
  and `determinism_baseline_recheck.json` say character-identical *after* the
  API-stripped `<|message|>` — an 11-character difference on all six prompts. The
  verdict was right and the summary word was wrong; corrected.
* Smaller: the logit probe's docstring over-promised its standalone section (it is
  a sanity cross-check, since one arm records token ids and the other returns
  text); the padded-vs-real vocab widths (202752 gathered, 202048 after slicing)
  are now used consistently in the timing argument; the 1.00x-at-8192 prefill-trace
  figure is cited to `prefill_trace_probe_8192.json` rather than the 128-row file;
  and `sampling_failure_probe.json` now carries an in-artifact caveat saying its
  `measured_relative_shift: 0.0` is an artefact of logprobs routing to host
  sampling and must not be read as evidence that the device penalty is missing.

### Round 4, and the systemic fix

Round 4 found four more, all of the same species, two of them in text written to
fix round 3:

* **"All 17 logprobs tests pass … isolation … pass" was wrong twice over.**
  `test_logprobs.py` is 20 passed + 1 **skipped**, not 17 passed, and
  `test_request_isolation.py` holds exactly one test which **failed** — and which
  the README itself listed among the reproducibility failures nine lines later.
  "Isolation tests pass" is precisely the sentence a reader would use to conclude
  cross-request contamination is ruled out. Replaced with a per-file table
  generated from the log, plus an explicit statement that contamination is ruled
  out by the cross-batch-position checks instead.
* **The one skip was never named or classified.** It is
  `test_logprobs.py::TestLogprobs::test_chat_logprobs_all_vocab`, which asks for
  all-vocab logprobs (`top_logprobs=-1`) against a platform that clamps
  `max_logprobs` to 20 because the device computes top-32 and the OpenAI API
  allows 20. An expected framework limit, now named and classified as the skill
  requires.
* **The host-sampling timing argument I promoted in round 3 was a tautology.**
  "TTFT + 127 x mean TPOT reproduces E2E, so there is no unaccounted time" holds
  *identically for any distribution*, because vLLM defines
  `tpot = (latency - ttft) / (n - 1)`. Withdrawn, and replaced with the bound that
  is actually in the artifact: `std_itl_ms = 0.372` over the 127 intervals, p50
  23.015 / p99 23.641 — a ~12.9 MB full-vocab gather and readback cannot sit
  inside a 0.372 ms standard deviation.
* **Four dangling citations**, the worst of which was a *null* one: the work log
  cited a pipfreeze before/after pair as evidence that torch and transformers were
  unchanged, but those files live outside the repo, are byte-identical to each
  other, and are a freeze of the system interpreter containing no torch or
  transformers line at all. The claim is true; the evidence was empty. Replaced
  with the wheel metadata and the versions read back from the env. The install log
  is now imported into `logs/install_vllm.log` rather than cited outside the repo.

**The systemic fix.** Four consecutive rounds found this one defect class and
almost nothing else, twice in the repair itself, which is a process problem rather
than four unrelated slips. `bench/check_reported_figures.py` — the serving
counterpart of `doc/multichip_decoder/bench/check_reported_figures.py`, which
exists because the single-chip stage had the same experience — now re-derives
every mechanically-sourced number in the reports from the artifact it came from
and asserts that every `doc/`, `readiness_vllm/` or `logs/` path they cite
exists. It found the dangling install-log citation on its first run. It is
runnable as a gate:

```text
$ python doc/vllm_integration/bench/check_reported_figures.py --check
  [ok ] TTFT p50 (ms) / decode t/s/u / TPOT / ITL / throughput      (vllm_benchmark.json)
  [ok ] burst throughput and t/s/u, 32/32                            (vllm_ci_serving_benchmark.json)
  [ok ] sampling 62/10/1 and all eight per-file rows                 (sampling_tests.log)
  [ok ] the one skip is named
  [ok ] worst adjacent duplication 0.0286 over 36 measurements       (degenerate_check_all.log)
  [ok ] served max_model_len 131072, contract reduction "none"       (context_contract.json)
  [ok ] KV pool 16416 blocks; measured bound 28672 is a lower bound  (kv_budget_probe.json)
  [ok ] logit determinism 160 candidates, delta 0.0                  (logit_determinism.json)
  [ok ] no dangling citations
all reported figures and cited paths re-derived from the artifacts.
```

### Round 5

The checker held up under attack — the reviewer built a 27-case perturbation
harness on a scratch tree and confirmed 16 of 17 value perturbations produce a
correct `BAD` line, that no check reports `None`, and that no check compares a
number to itself. But it found two live instances the checker could not see, and
one escape hatch in the checker itself:

* **The corrected sentence was fixed in the README and not in the work log.**
  §6.4 still asserted "all 17 logprobs tests pass" — the literal round-4 defect,
  in the same file that, 300 lines later, records it as fixed. It was load-bearing:
  one of the four conditions cited to classify seven failures as
  reproducibility-only. Corrected, and the file now also states that
  `test_request_isolation.py` is *not* a clean file.
* **The README claimed the qualitative directory held an HF-vs-serving comparison
  that did not exist.** `qualitative_vllm.py --compare` writes the HF control copy
  and the two comparison JSONs, and it had never been run — the sweep runs only the
  generation arm. The coherence verdict leaned on "the HF control does the same"
  with the HF line quoted from no committed path. `--compare` needs no hardware
  (it returns before the `openai` import and reads committed JSON), so it was run:
  `qualitative_hf_chat.json`, `qualitative_comparison_chat.json` and
  `qualitative_vllm_vs_datatype_sweep_chat.json` now exist, and `compare()`'s
  verdict on the serving arm is clean — worst adjacent duplication 0.0 across the
  six prompts, non-ASCII <= 0.0018, and trigram-loop fractions in the control's
  band. (Round 7: this line said the fractions *matched* the control's, which is
  the round-6 defect asserted a second time in the work log after the README had
  been fixed — the same README-fixed/work-log-missed shape as round 5. Corrected
  here, and the checker now scans both documents for it.)
  Both comparison files report `first_divergence` 1-2 and `identical: false`, and
  the README now flags that beside the citation rather than leaving a reader to
  find two freshly-committed files that appear to contradict the verdict. (Round 6
  found the *explanation* attached here — "the API strips `<|message|>`" — to be
  true of the standalone comparison and false of the HF one. Corrected in §9.)
* **The checker had a silent-skip hatch.** Its per-file loop was guarded by
  `if row:`, so deleting a row from the README table — exactly the round-4
  omission — dropped a check and still passed. It now *requires* a row for every
  file present in the log; deleting the `test_request_isolation.py` row is now
  `rc=1`. Integer counts compare exactly rather than within 0.5 %, and both
  documents are scanned for the stale "all N logprobs tests pass" and
  "isolation tests pass" claims. That scan was scoped by section — stopping at the
  review-history heading — which round 6 showed to be a blind spot, not a rule.
  Rescoped in §9.

Also corrected: the presence-margin step count (40 scored steps, from
`greedy_flip_margins.steps`, not the 45 in a different field of a different file),
and the plugin registration path, which now names both the canonical
`vllm/plugins/...` location and the standalone editable checkout this workspace
actually uses.

## 9. Stage review round 6 — the qualitative section, and the checker's own scope

Round 6 verified round 5's first fix as genuine and found that the other two had
each left a real hole. Both are of the stage's recurring species, and one of them
was *created* by the round-5 fix — which is the sixth time that has happened, and
the reason the response below is another mechanical check rather than more prose.

### P1 — the explanation attached to the new comparison files was false

Round 5 committed `qualitative_comparison_chat.json` (serving vs HF) and
`qualitative_vllm_vs_datatype_sweep_chat.json` (serving vs the previous TT stage),
both reporting `identical: false` with `first_divergence` 1-2 on every prompt, and
explained both with one cause: the API strips `<|message|>`, so the re-encoded ids
necessarily diverge early. That explanation is **true of one file and false of the
other**, and the difference matters because `qualitative.py`'s own docstring makes
that field the wrapper-bug tripwire ("divergence at token 0-2 is a wrapper bug").

`bench/stripped_divergence.py` removes id 200023 from both sides offline and
recomputes (`qualitative/qualitative_stripped_divergence_chat.json`):

| pair | raw | stripped |
|---|---|---|
| served vs datatype-sweep standalone | 2 on all six | **no divergence, 6/6, over the full 127-token common prefix** |
| served vs HF control | 2 on five, 1 on p1 | 12 / **1** / 33 / 27 / 43 / 31 |

So the stripped token accounts for the standalone comparison completely — the
claim was in fact *stronger* than round 5 stated — and does not account for the HF
one at all. Serving is not token-identical to HF and never should have been
expected to be; five prompts diverge late with both texts coherent, which is the
ordinary-numerics reading. p1 diverges at token 1 on the channel token (`to=user`
served, `to=self` HF), and that is pre-existing: the datatype-sweep stage's own HF
comparison records `first_divergence_from_hf: 1` for p1, computed with
`<|message|>` present on both sides and no stripping involved.

(An earlier revision of this section offered the "each stripped divergence is the
standalone stage's number minus one" pattern as corroboration. Round 7 correctly
called that numerology: `<|message|>` occurs exactly once at index 2 in both arms,
so given the exactness result in the row above, the −1 offset cannot come out any
other way. It is a restatement, not evidence, and it is dropped.)

Round 5 also pointed at `determinism_baseline_recheck.json` as the "like-for-like"
resolution. That file is TT-vs-TT, so it corroborates the first row and is silent
on the second. Both statements are corrected in `README.md`, and the coherence and
repetition bullets no longer say the HF control "does the same" without noting that
p1 is the exception.

### P2 — the stale-claim scan was scoped by position, so it had a blind spot

`_before_review_history()` cut the work log at the first `## N. Stage review`
heading and scanned only what preceded it. Planting the round-4 sentence proved the
gap: caught in §6.4, **missed** in §7.1 (live technical content that happens to sit
under a review heading) and **missed** in a newly appended `## 9.` — that is, in
every section the stage had yet to write, including this one.

The scan is now scoped by *form* rather than position (`_unquoted()`): the full
document is scanned, minus fenced blocks, blockquotes, inline code spans and
quotations. A review section may **quote** a corrected defect — which is what those
sections are for, and what §7 and §8 already do — and may not assert one as the
document's own claim. All three plants are now `rc=1`; the legitimate quotations in
§7/§8 still pass.

### P2 — the presence-margin figure was cited to the wrong file

Round 5 corrected the step count from 45 to 40 without moving the citation, so the
README cited `presence_flip_probe.json` for a number that is in
`sampling_failure_probe.json`, and labelled it the "winner-to-already-seen" gap
when the quantity that bounds a *positive* presence penalty is the reverse:
`min_gap_winner_appeared_to_best_fresh` = 3.0. Both file and field are now named,
the 4.5 mirror figure that rules out negative penalties is named beside it, and the
coincidence that `presence_flip_probe.json`'s own 45-step `min_margin` is also 3.0
is spelled out so the two are not confused again.

### The checker covered none of this

Every round-6 finding sat in the qualitative section, and
`check_reported_figures.py` had no check there at all — its coverage stopped at the
sampling table. It now re-derives the stripped-divergence table both ways, the raw
divergence counts, the worst adjacent duplication and non-ASCII bound, and the
trigram-loop bands.
Verified non-vacuous: perturbing a divergence value, deleting the table row,
perturbing a trigram band, perturbing the non-ASCII bound, flipping the artifact's
standalone verdict and reinstating the round-6 wording each give `rc=1`, and the
restored tree gives `rc=0`. The checker caught one real error in this round's own
edits — the trigram bands were written rounded to three decimals.

## 10. Stage review round 7 — two defects in round 6's own text, and a guard that leaked

Round 7 confirmed all three round-6 fixes on their own terms (it re-derived the
stripped divergences from raw token ids, reproduced the three plants at `rc=1`, and
checked the corrected presence-margin citation) and then found the recurring defect
class a seventh time — twice inside the text round 6 had just written.

* **The trigram direction was wrong on p0**, in the sentence written to replace
  round 6's "matching the HF control's". TT is 0.0945 against HF's 0.1172 on p0, so
  TT is *lower* there; it runs higher on p2/p4/p5 only. The new checker verified the
  four band endpoints and so could not see a per-prompt direction claim at all. Both
  the sentence and the gap are fixed: the README now says higher on p2/p4/p5, lower
  on p0, and the checker derives both sets.
* **The round-6-corrected wording was still asserted in `work_log.md` §6**, with
  "HF" dropped — the README-fixed/work-log-missed shape from round 5, recurring.
  Fixed, and the guard now scans both documents and matches the claim rather than
  one phrasing.
* **`_unquoted()` leaked.** Deleting inline code spans was far too permissive:
  these documents write every identifier and most numbers in backticks, so
  "All `21` logprobs tests pass", "All 21 `logprobs` tests pass",
  "All 21 *logprobs* tests pass" and a sentence describing
  `test_request_isolation.py` as a clean file all returned `rc=0`. Worse, the guard
  could never fire on the one spelling every sentence here actually uses, because
  its pattern excluded the period in the filename. Backticks, asterisks and
  underscores are now stripped **as characters**, the isolation pattern no longer
  excludes periods, and whitespace is collapsed *before* quotations are removed —
  the documents are hard-wrapped, and the round-7 fix's own quoted example straddled
  a line break and was therefore not recognised as a quotation. All four evasions
  are now `rc=1`.
* **The logprobs guard was a phrase match.** "Every one of the 21 logprobs tests
  passes" walked through the old pattern, "all N logprobs tests pass". It is now anchored on the
  count: any count asserted to pass must equal the count the log records as
  passing — so that paraphrase, ``All `21` …`` and "The 17 logprobs tests all pass"
  all fail, while "All 20 logprobs tests pass; one more is skipped" and "There are
  21 logprobs tests in total" correctly do not. (Round 8 found the match window
  here still excluded periods and widened it; see §11.)

Round 7 also killed a **vacuous check** and a **bogus argument**, both of which were
this round's own work and both of which are now gone:

* The `raw divergence vs HF (count at 2, count at 1)` check scraped the two integers
  out of the prose cell "2 on five, 1 on p1" and compared them against a value and a
  count — so "2 on three" and "1 on p3" both still passed, and the words carrying the
  claim were never parsed. The table now lists one value per prompt in `p0..p5`
  order and every cell is compared elementwise, including the standalone row's
  127-token prefix length and its 6/6.
* The "each stripped divergence is the standalone stage's number minus one"
  corroboration was numerology. `<|message|>` occurs exactly once at index 2 in both
  arms, so given the exactness result the −1 offset cannot come out any other way.
  Dropped from both documents; p1's dismissal now rests on the datatype-sweep
  stage's `channel_margin_probe.json`, which scores that exact position under the
  shipped `c14-attn4-cclbfp8-kv8` policy and finds the `=self`/`=user` choice decided
  by 0.0625 logits — one BFP4 quantization step — with decode already on `=user` by
  0.125. The checker re-derives both margins and the policy.
* Round 7 also noted that `serving_introduced_early_hf_divergences` cannot flag a
  late corruption, so it is not independent coverage. It is kept as a narrow
  early-divergence classifier and is no longer leaned on: the per-prompt divergence
  lists and the exactness row are what carry that claim.

Checker after this round: **62 checks, `rc=0`**, with the qualitative section — where
every round-6 and round-7 finding sat — covered for the first time.

### The limit of the guard, stated plainly

`_unquoted()` matches phrasings, not meanings. A sufficiently different paraphrase of
a stale claim will still pass, and no regex fixes that. What it does guarantee is
that the *specific* claims seven review rounds have caught cannot silently return,
in either document, in any section, in or out of backticks — which is the actual
observed failure mode. It is a ratchet, not a proof.

## 11. Stage review round 8 — the same period-exclusion bug, one line up

Round 8 re-derived the round-7 fixes and confirmed them, including two things worth
recording because they were open questions rather than assertions: the 0.005 trigram
deadband hides nothing (every fraction reconstructs exactly as *n*/127 for TT and
*n*/128 for HF; p1 and p3 have **identical covered-token counts**, 6 and 9, so their
residual is a pure denominator artifact of the stripped control token, max 0.0011
against a real trigram of 0.0236), and the channel-margin dismissal is sound rather
than a second piece of numerology — `c00` decides the channel by 1.5/1.75 logits
toward `=self`, BFP4 attention collapses that to 0.0625 under `c01`/`c14`, and `c14`
decode is already on `=user` by 0.125, which is the arm that emits the divergent
token. It then found two more.

* **The logprobs guard still excluded periods** — `[^.]{0,30}`, one line above the
  isolation pattern where round 7 had fixed exactly that defect and written down
  why. So "All 21 logprobs tests in test_logprobs.py pass", the backticked variant,
  "The 21 logprobs tests (test_logprobs.py) all pass" and "All 21 logprobs tests,
  e.g. the chat ones, pass" all returned `rc=0` — every one of them a sentence that
  names the file, which is how these documents habitually write it. §10's claim that
  the guard was "anchored on the claim" was therefore false as written. Any
  `.`-excluding window in documents full of `foo.py` is a hole by construction, and
  both windows are now period-tolerant. The isolation pattern was also tied to one
  word order and missed "Every test in test_request_isolation.py passes"; it now
  matches "isolation … pass(es|ed)" in either order. All five evasions are `rc=1`,
  and the phrasings that must not fire still do not — including the corrected
  "All 20 logprobs tests in test_logprobs.py pass; one more is skipped".

  The widened guard immediately fired on this stage's own §10, on the sentence
  "…describing `test_request_isolation.py` as clean all passed". That sentence was
  reporting a round-7 evasion, but nothing in its form said so. It was reworded
  rather than exempted — which is the rule working as intended.

* **`README.md`'s summary line contradicted its own sampling section.** It called
  the sampling stage's ten failures "10 reproducibility/prompt-shape failures",
  folding all ten into the *waivable* class, while the section 370 lines below
  splits them 7 reproducibility-only + 3 correctness-class, as does §6.4 here. Only
  the reproducibility-only class may be classified rather than fixed, so that
  summary understated the suite at the point a reader meets it first. ("prompt-shape"
  described no failure at all — it appears nowhere else in the evidence root.)
  Corrected to state the split and that all three correctness-class failures were
  resolved.

  No check guarded the taxonomy, which is why a contradiction could sit 370 lines
  from its refutation. Three now do: the split must sum to the log's failure count,
  the correctness-class count must equal the number of failures matching the
  presence-penalty and allowed-token-ids names, and **every** failed test in the log
  must be named somewhere in the README. Verified `rc=1` on reinstating the round-8
  wording, on a split that no longer sums to ten, and on renaming one failed test
  out of the README.

Checker after this round: **66 checks, `rc=0`**.

## 12. Stage review round 9 — word order, "resolved", and a citation the old check could not see

Round 9 independently re-derived both of round 8's open questions and confirmed
them, which is worth recording since they were the two places the stage was
relying on an argument rather than a measurement. The trigram deadband hides
nothing: every fraction reconstructs exactly as *n*/127 for TT and *n*/128 for HF
with *n* a multiple of 3, p1 and p3 have **identical covered-token counts** (6 and
9), their residuals are 0.00037 and 0.00055, the largest same-count artifact is
0.0011 and the smallest real step is 0.0236 — so the 0.005 deadband sits in open
space. The channel-margin dismissal likewise: `c00` decides the channel by
1.5/1.75 logits toward `=self`, BFP4 attention collapses it to 0.0625 under both
`c01` and the shipped `c14`, and `c14` decode is already on `=user` by 0.125.

Three more findings, all of the tracked class:

* **§11 said the isolation guard matched "in either order". It did not** — the
  pattern was strictly "isolation" before the pass-word, so "All tests pass in
  test_request_isolation.py", "The suite passes every test in
  test_request_isolation.py", "No failures: passed 1/1 in
  test_request_isolation.py" and "Everything passed in the request-isolation file"
  all returned `rc=0`. That is the ninth consecutive instance, and the third time
  the offending sentence was a review paragraph overstating the guard it had just
  written.

  Round 9 also showed the converse, which matters just as much: the widened guards
  rejected *true* sentences — "Of the 21 logprobs tests, 20 pass and one is
  skipped", "test_request_isolation.py fails; every other file passed",
  "test_request_isolation.py is not a clean file: 0 passed, 1 failed". A guard that
  forces the documents to stop stating facts accurately is not a ratchet, it is a
  gag.

  Both directions came from the same design error: matching a phrase inside a
  character window. Rounds 7-9 defeated windows four ways (periods in filenames,
  reversed word order, longer wording, paraphrase) while windows simultaneously
  produced those false positives. **Both guards now scope to a sentence and test a
  property of it:**

  - a sentence saying logprobs tests pass must state the count that actually passes
    (20) — it may say whatever it likes about the other counts;
  - a sentence mentioning the isolation file *and* passing must also say it fails.

  Word-order-free and length-free by construction. All eight evasions from rounds
  7-9 are now `rc=1`, and all four true sentences are `rc=0`. (Round 10 found one
  more: `_sentences()` protected filename dots but not `e.g.`, so a claim split at
  the abbreviation escaped. Abbreviations are protected now, and that phrasing is
  `rc=1`.)

* **`README.md`'s summary said "all classified below and all resolved" of ten
  failures.** Round 8 had corrected this same sentence in the *understating*
  direction; the replacement overstated it in the other. Only the three
  correctness-class failures are resolved — the seven reproducibility-only ones are
  classified, not fixed, and are open in Limitations 1. Reworded so the two groups
  carry their own verbs, and a check now requires every "all resolved" in the README
  to be attached to the correctness-class group.

* **`AUTODEBUG.md` cited `probe_full.json`, which does not exist and could not** —
  that arm hung before reaching its JSON write. The citation check could not see it
  because it only resolved `doc/`, `readiness_vllm/` and `logs/` prefixes, while
  these documents cite most artifacts by bare name. It now resolves bare
  `*.json`/`*.log` citations too, outside fenced blocks (a command may legitimately
  name an output that does not exist yet). Two names here are *deliberately* absent
  — there is no `probe_guard.json` (that run exits 1 at the position guard) and no
  `probe_full.json` (that run hung), neither reaching its JSON write — so a citation is exempt when the negation **determines** the name
  ("no `x.json`", "`x.json` does not exist"). Scoping the exemption to the sentence
  was too loose, and so was a proximity window: "though the run never finished,
  `probe_full.json` records it" escaped both. Three such constructions are now
  `rc=1` while the two genuine declarations pass.

Two smaller items, both round 9's:

* **`_unquoted()`'s quote exemption was parity-dependent.** The README carries an
  odd number of unfenced double quotes — one of them inside `` `!"#` `` — so the
  same quoted sentence was exempted in `work_log.md` and scanned as an assertion in
  `README.md`. Pairing is now positional: a `"` opens only after a space or an
  opening bracket and closes only before a space or punctuation, with the markdown
  emphasis markers included on both sides.
* **The "every failed test is named in the README" check was vacuous for one
  name.** `test_seeding` was satisfied by being a prefix of
  `test_seeding_and_variety.py`, the filename a different check already requires.
  Now matched with `(?!\w)`; deleting the standalone mention is `rc=1`.

Checker after this round: **68 checks, `rc=0`**.

### What these guards do and do not do

They match sentence-scoped properties, not meanings. A sufficiently inventive
paraphrase will still pass, and the bare-citation exemption still trusts a
two-word negation next to a filename. What they guarantee is that the specific
claims nine review rounds have caught cannot silently return — in either document,
in any section, in or out of backticks, in either word order — while the true
statements those same claims are usually corrected *into* remain sayable. That is
a ratchet, not a proof, and the distinction is the reason each round's fix is
recorded here with the perturbation that would catch its regression.
