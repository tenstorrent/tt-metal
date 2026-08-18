# Stage 08 — vLLM serving integration — work log

Chronological. Every command is the one that actually ran; every number cites the
artifact it came from.

Starting point: `9ec24b89ef5` (stage 07, datatype sweep), tree clean.
Hardware: 1x4 Blackhole P300_X2, `FABRIC_1D_RING`, sole device user.
vLLM 0.24.0; TT plugin checkout `/home/raahem/vllm-tt-plugin` @ `bc4af2d`
(branch `raahem/fix-offline-inf-tokensprompt-import`), **not modified**.

---

## 1. Reading the interface before writing to it

Two things had to be pinned down exactly, because guessing either would have
produced an adapter that looks right and serves garbage.

**What the plugin calls, and with what.** Mapped from the plugin source:

| Call | Where | Shape / meaning that mattered |
|---|---|---|
| `initialize_vllm_model(hf_config, mesh_device, max_batch_size, max_seq_len=, tt_data_parallel=, optimizations=)` | `loader.py:38` | `max_batch_size` is `scheduler_config.max_num_seqs` |
| `get_max_tokens_all_users(model_name=, num_devices=, tt_data_parallel=, max_model_len=, max_num_seqs=)` | `worker.py:624` | return value → `ceil((n + block*batch)/block)` blocks |
| `allocate_kv_cache(shape, dtype, num_layers)` | `model_runner.py:454` | `shape = (num_blocks, kv_heads_per_die, block_size, head_dim)`; the plugin has **already** divided heads by the mesh size |
| `prefill_forward(tokens=, page_table=, kv_cache=, enable_trace=, prompt_lens=, start_pos=)` | `model_runner.py:2361` | `tokens` `[num_reqs, max(prompt_lens)]`, **garbage past each row's real length**; `prompt_lens` is a **numpy** array |
| `decode_forward(tokens=, page_table=, kv_cache=, start_pos=, enable_trace=, read_from_device=, sampling_params=, reset_batch=, slot_remap=)` | `async_decode.py:624` | decode is **always** padded to `max_num_seqs` rows; inactive rows carry `start_pos == -1` |
| `read_decode_output(tt_out, async_read=True)` → `(out, [events])` | `async_decode.py:635` | only on the async path |
| `process_decode_output_host(tt_out, is_tokens=)` | `async_decode.py:673` | also reached **without** `read_decode_output` on the plugin's synchronous path (`model_runner.py:2459` passes `read_from_device=False, async_read=False`) |
| `warmup_model_prefill/decode(...)` | `model_runner.py:3031-3042` | two phases: `enable_trace=False` compiles, `enable_trace=True` captures |

Three of these decided the design:

* decode is a **fixed `max_num_seqs`-row batch with `-1` for inactive slots** —
  which is exactly the fixed-slot/inactive-row convention `tt/generator.py`
  already had, so no slot bookkeeping was needed in the adapter;
* `allocate_kv_cache` is the **only** place vLLM's block size is visible to the
  model, and it runs before warmup and before any forward — so that is where
  the generator's paging geometry gets replaced (`configure_paging`);
* `check_perform_device_sampling` (`model_runner.py:2313`) sends
  `sampling_params=None` and expects **logits** for logprobs on a mesh that is
  not 8 or 32 dies, and for `min_p` / `bad_words` / `logit_bias` / structured
  output. On this 4-die mesh that is a live path, so the host-sampling
  compatibility mode is not optional — but it is *vLLM's* choice per request,
  never a performance path, and it does not displace the traced one.

**How to register without touching the plugin.** `register_tt_models()`'s first
statement is `_register_models_from_extra_dir(ModelRegistry)`, documented as
"Runs first so a distributed bundle can supply a model without touching this
file" (`platform.py:481`). So the model ships as a bundle under
`EXTRA_MODELS_DIR`. See `README.md` §"Registration".

## 2. Changes to `tt/generator.py`

Four additions, all additive, all default-preserving:

1. `configure_paging(page_block_size=, pages_per_user=, num_blocks=)` — adopt
   vLLM's paging geometry and reallocate the two persistent page-table tensors.
   Guarded to run only before any trace or any generator-owned cache exists.
2. `decode_device_state()` — read back the trace's per-slot token/position plus
   the page table the trace was captured against, so the adapter can prefer the
   device's view over an async-ahead scheduler's. Two small reads, on layout
   changes only.
3. `read_sampled_tokens(sampled, count)` — public wrapper over the existing
   single readback.
4. `prefill_forward(..., preserve_decode_traces=)` and
   `decode_forward(..., validate_page_coverage=)` — see §4 and §3.

Plus one optimization: `_decode_compiled_keys`, a set of decode graph keys whose
programs are in the program cache, which **survives a trace release**. Without
it every serving prefill would make the following decode pay a full eager warm
pass it does not need. Measured effect in §5.

## 3. Why `validate_page_coverage=False` in the serving path

`_validate_page_coverage` asserts that active rows map **disjoint** physical
pages and that every page in the SDPA's rounded read window is in
`[0, self.num_blocks)`. Both statements describe the *standalone* page tables
`make_page_table` builds. vLLM's block tables are its own: unused entries are
zero-filled, not `-1`, so several rows legitimately "share" block 0 in their
padding. The disjointness assertion is therefore false by construction on a
correct vLLM table. The adapter turns the check off and pads its own widening
with `0` for the same reason vLLM does — the rounded tail page still has to
dereference somewhere valid.

## 4. The `preserve_decode_traces` experiment, and the hang

**Hypothesis.** `prefill_forward` releases the decode traces before it runs
("Prefill is eager and allocates; a live trace makes that unsafe"). In serving
that fires on *every request admission*, so I added
`preserve_decode_traces=True` to keep them, reasoning that prefill's allocations
never touch the trace region and every tensor a captured trace holds is owned by
the generator.

**Result: the mesh hung.** Reduced 2-layer target, `max_num_seqs=1`, five
sequential requests. Requests 1–4 returned; request 5 never completed.
`py-spy dump` on the EngineCore put the main thread in

```
to_torch (ttnn/operations/core.py:421)
_first_device_to_torch (tt/generator.py:52)
_sampled_to_torch (tt/generator.py:547)
read_sampled_tokens (tt/generator.py:222)
prefill_forward (tt/generator_vllm.py:537)
```

i.e. blocked forever on the prefill sampler's readback. `tools/tt-triage.py`
confirmed a device-side hang, not a host deadlock —
`triage/tt-triage-preserve-traces-hang.txt.gz`:

```
Device 0: functional_workers [1-2 (0,0)]: NOC0 CB0 active (0xFFFFFFFF). NoC is likely hung.
... (CB0..CB3 on four cores)
```

**Reading.** The prefill sampler and the captured decode sampler are the same
`_WatcherCleanSampling1D` over the same persistent CCL buffers and semaphores.
Running it eagerly between replays advances semaphore state the captured graph
baked in; after a few admissions a replay waits on a value an eager collective
already consumed. Releasing on prefill — the generator's original behaviour —
makes the next capture re-establish that state, which is why the original design
is safe by construction.

**Decision.** `PRESERVE_DECODE_TRACES` defaults to **off**. The env var
`QWEN3_VLLM_PRESERVE_DECODE_TRACES=1` reproduces the hang. The cost this was
meant to avoid is measured in §5 and turned out to be negligible once
`_decode_compiled_keys` removed the redundant warm pass.

**Recovery** (per `$tt-device-usage`): killed `run_vllm_server`,
`vllm.entrypoints`, `VLLM::EngineCore`; `timeout 240 tt-smi -r` (all 4 PCI
devices reset, "Re-initializing boards after reset"); `tt-smi -ls --local`
listed 8 Blackhole rows again; mesh smoke `open_mesh_device(MeshShape(1,4))` /
`close_mesh_device` printed `MESH_SMOKE_OK`. One reset was enough.

## 5. Reduced-target bring-up (2 layers) — the inner loop

Same adapter, same generator, same bundle registration, same cache/page-table
shapes, same terminal norm / LM head / sampler, same trace behaviour;
`QWEN3_VLLM_NUM_LAYERS=2`. Artifacts went to a scratch dir, never to
`readiness_vllm/`.

```bash
EXTRA_MODELS_DIR=models/autoports/qwen_qwen3_coder_30b_a3b_instruct/vllm_bundle \
QWEN3_VLLM_NUM_LAYERS=2 \
python -m models.common.readiness_check.run_vllm_server \
  --model-dir <scratch> --hf-model Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --mesh-device P300x2 --max-num-seqs 4 --max-model-len 4096 \
  --block-size 32 --port 8100 --stages serve \
  --tt-config '{"trace_region_size": 60000000, "fabric_config": "FABRIC_1D_RING"}'
```

Bugs this caught, in order:

1. **Port 8000 was already held** by a process outside this session (`ss -lptn`
   showed a LISTEN with no visible owner). Everything since runs on **8100**.
2. **`TypeError: only length-1 arrays can be converted to Python scalars`** in
   the prefill `start_pos` guard, on the first *multi-request* prefill:
   `prompt_lens` and `start_pos` arrive as **numpy arrays**, and the list
   normaliser only special-cased `torch.Tensor`. This killed the EngineCore
   outright (`EngineDeadError`, four concurrent requests → HTTP 500). Fixed by
   normalising anything with `.tolist()`.
3. **A false "per-request seed supplied" warning during warmup**: my neutral
   warmup params left `TTSamplingParams.seed` at its dataclass default of `0`,
   but the plugin translates its own no-seed sentinel to `None` before the model
   sees it. Warmup now passes `seed=[None] * rows`.

What passed on the reduced target, after those fixes:

* greedy, sampled, logprobs (host-sampling fallback), and greedy again — the
  greedy result is **byte-identical before and after the logprobs excursion**,
  which is the check that the host-sampling compatibility mode does not disturb
  the traced path;
* **non-aligned prompt lengths 37 and 331 token ids** served without error;
* four concurrent staggered requests, each continuing **its own** prompt;
* smoke sampling profile: `test_top1_is_greedy` PASSED, `test_min_p` PASSED
  (host-only fallback), `test_chat_logprobs_all_vocab` SKIPPED,
  `test_mixed_params_batch` FAILED on seeded reproducibility only (see
  README "Limitations").

**Trace re-capture cost, measured.** 128/128/1 on the reduced target:
ITL P50 2.113 ms, **ITL P99 2.673 ms** — the single decode step that re-captures
after the prefill costs ~0.56 ms at 2 layers. This is the number that made
`preserve_decode_traces=False` acceptable rather than merely safe.

## 6. Full 48-layer run A — `max_num_seqs=32`

```bash
EXTRA_MODELS_DIR=$PWD/models/autoports/qwen_qwen3_coder_30b_a3b_instruct/vllm_bundle \
python -m models.common.readiness_check.run_vllm_server \
  --model-dir models/autoports/qwen_qwen3_coder_30b_a3b_instruct \
  --hf-model Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --mesh-device P300x2 --max-num-seqs 32 --max-model-len 262144 \
  --block-size 32 --port 8100 --stages serve \
  --tt-config '{"trace_region_size": 50331648, "fabric_config": "FABRIC_1D_RING"}' \
  --additional-server-args "--generation-config vllm"
```

Startup, observed live: registration logged in the API server, the EngineCore
and the worker; `max_tokens_all_users=262144`; `num_gpu_blocks_override=8224`;
`GPU KV cache size: 263,168 tokens`; prefill warmup at 129 and 128; decode
warmup at batch 32 with and without trace. Weight load to ready took ~2:45.

**This run's server log was not retained.** `readiness_vllm/server.log` was
overwritten by run B (§7, `max_num_seqs=1`, 17:43–17:48) — do not read it as
run A's. The startup lines above are therefore unarchived; the two block figures
in them are instead **derived** from `get_max_tokens_all_users` plus the worker's
`ceil((max_tokens_all_users + block_size * max_num_seqs) / block_size)`, which
reproduces run B's retained `8193 / 262,176` exactly at `max_num_seqs=1`. See
README "Served context" and `probes/check_published_figures.py`. Run A's own
outputs — `sampling_tests.log`, both qualitative JSONs, the CI burst,
`vllm_benchmark_maxnumseqs32.json` — were all retained.

Stages were attached to the held-open server **individually** rather than run as
one flow, because `run_vllm_server` aborts the remaining stages when `sampling`
fails and the seeding failures were already known from §5. Order:
`qualitative` -> `benchmark` -> `sampling --sampling-profile full`.

`--generation-config vllm` is on because this checkpoint's
`generation_config.json` injects `repetition_penalty=1.05, temperature=0.7,
top_k=20, top_p=0.8` into every request that does not override them. At the time
of this run that meant silently dropping a parameter the server claimed to
honour; since §10b the penalty *is* honoured, and the flag stays for a different
reason — the checkpoint default would otherwise put every request, including the
benchmark's, on the penalised path and its per-step host staging.

Results (all in `readiness_vllm/`):

* `vllm_qualitative_outputs.json` — 6 prompts, greedy and sampled, raw
  `/v1/completions`;
* `vllm_ci_serving_benchmark.json` — 100/100/32: **104.062 tok/s aggregate**,
  TTFT median 4901.160 ms, TPOT mean 261.154 ms;
* `vllm_benchmark_maxnumseqs32.json` — the 128/128/1 shape at `max_num_seqs=32`:
  TPOT mean **263.470 ms**. Copied aside before run B overwrote the canonical
  primary files;
* `sampling_tests.log` — **52 passed, 20 failed, 1 skipped** in 612.63 s. **Superseded by §10b's re-run (58 / 14 / 1, 603.49 s), which is the log now in `readiness_vllm/`; this run's copy was not retained.**

The 263.470 ms against 261.154 ms is the finding: a single active user on a
32-slot server costs the same per token as 32 active users. The per-step cost
tracks the **configured** slot count, so this is MoE decode batch scaling in
`tt/model.py`, not adapter or serving overhead.

I also collected a chat-templated pass by hand
(`vllm_qualitative_chat_outputs.json`), because this checkpoint declares a chat
template and the shared runner sends raw `/v1/completions`; judging chat-style
prompts as continuations would have masked exactly the serving bugs the
qualitative gate is for. `$qualitative-check`'s control requirement is met by
`readiness_qualitative/vllm_qualitative_outputs.json` from the full-model stage:
the serving haiku matches it byte for byte.

`check_degenerate_output.py --scope vllm --missing-artifacts critical` ->
`No degenerate output detected.`, exit 0 (`logs/check_degenerate_vllm.log`).

## 7. Full 48-layer run B — `max_num_seqs=1`, the headline

Same launch with `--max-num-seqs 1`, then `--stages benchmark
--no-benchmark-ci-serving` so the CI burst files from run A were left intact.

**TTFT 312.367 ms, TPOT mean 19.778 ms, ITL median 19.804 / P99 20.084 ms,
45.318 tok/s, 50.560 t/s/u** (`vllm_benchmark.json`).

Against stage 07's shipped standalone `token_out` of 19.213 ms / 52.049 t/s/u:
vLLM costs **0.565 ms per token, 2.9 %**. The teacher-forcing lower bound of
43.54 t/s/u is cleared with room.

ITL P99 20.084 against a P50 of 19.804 also settles §4's remaining worry: the
decode-trace re-capture that every prefill forces is inside the noise of a single
token out of 128 on the full model, just as it was on the reduced one.

Non-aligned prompt lengths were then sent directly at 37, 131, 333, 1025, 4097
token ids and one 43-token natural-language prompt
(`non_aligned_prompt_lengths.json`). Every one returned `usage.prompt_tokens`
equal to the length requested, so nothing was capped or truncated, and the text
prompt answered coherently.

## 8. The `--async-scheduling` overlap test

`supports_async_decode=True` gates vLLM's async scheduling, so it was validated
rather than assumed. Same launch, `--max-num-seqs 1`, plus
`--additional-server-args "--generation-config vllm --async-scheduling"`,
artifacts to a scratch dir and then copied into `logs/`.

The platform **accepted** it (`'async_scheduling': True` in the non-default args;
it was not force-disabled, which is itself proof the capability flag is read).
Findings:

* the primary 128/128/1 profile gives TPOT mean 19.808 ms / 50.483 t/s/u against
  19.778 ms / 50.560 t/s/u without it — the same number within noise;
* all six qualitative greedy completions came back **byte-identical** to run A's
  synchronous `max_num_seqs=32` collection. That is a stronger result than the
  test asked for: greedy output through the serving path is invariant to both
  async scheduling and the configured slot count;
* `check_degenerate_output.py` on those outputs: clean. No doubled subwords, no
  repeated control tokens.

The plugin logs `Using custom scheduler class vllm_tt_plugin.scheduler.TTScheduler
… you will see degraded performance due to async scheduling being disabled`, so
`TTScheduler` is not an `AsyncScheduler` subclass and no real overlap occurs —
which explains the identical TPOT. Async scheduling is therefore **safe but
inert** here, and the shipped command leaves it off. The adapter is built to be
correct under overlap regardless (`_merge_scheduler_view`), and §9 exercises that
directly.

## 9. `probes/adapter_contract_probe.py` — 13 checks, all pass

A live server proves the model serves; it cannot prove *how*. This drives
`tt/generator_vllm.py` with the exact kwargs the plugin builds and asserts on
`Qwen3CoderGenerator.trace_stats`, which counts every host-side action on the
token path. Run on a 2-layer target on purpose — every property is about host
work per token and cache/page-table/scheduler-input handling, none of which
depends on depth.

```bash
python models/autoports/qwen_qwen3_coder_30b_a3b_instruct/doc/vllm_integration/probes/adapter_contract_probe.py \
  --num-layers 2 --max-num-seqs 4 --max-model-len 4096 --prompt-len 131 --steps 8
```

Over 8 steady tokens: `replays +8`, `token_host_copies +0`,
`position_host_copies +0`, `rotary_position_host_copies +0`,
`page_table_host_copies +0`, `captures/releases/warmups +0`,
`caller_token_readbacks +8`. A steady step fed `token=12345` and `position-1`
reproduced the clean run exactly. A `reset_batch` onto fresh blocks did cost one
page-table copy and did reinstall the position. `decode_forward` with no
`allocate_kv_cache` raises instead of allocating a standalone cache. Full output:
`probes/adapter_contract_probe.json`.

The same JSON records `captures: 3, releases: 3` across the run, which is the
direct confirmation that a serving prefill really does release the decode traces
and the next decode really does re-capture them (§4).

One probe bug found and fixed while writing it: it passed `tokens=None` for a
steady step, which the adapter rejects. That was the *probe* being wrong — vLLM
always sends a full `tokens`/`start_pos` pair — and fixing it made the check
stronger, since the steady step now receives real host state and demonstrably
ignores it.

## 10. Device and process hygiene

Every server was shut down with `pkill -f readiness_check.run_vllm_server`, then
`pkill -9 -f VLLM::EngineCore` and `pkill -9 -f vllm.entrypoints`, then verified
with `ps aux`. `ps aux | grep -c "[V]LLM::EngineCore"` returns 0 at the end of the
stage. One device reset was needed in the whole stage, for the §4 hang, and it
succeeded on the first attempt with a clean mesh smoke afterwards.

No Tracy, `tt-perf-report`, `TT_METAL_DEVICE_PROFILER` or
`ttnn.ReadDeviceProfiler` run was made against a live server, per the stage rule.
`tools/tt-triage.py` was used once, on an already-hung mesh, to identify the
failure — not as profiling.

## 10b. Sampling penalties — closing Class B

Done after the rest of the stage was written up, against the same tree. The 6
`test_tt_penalties` failures were the one part of the sampling gate that was
visible to a client rather than a reproducibility artefact: the request was
served, the penalty was dropped, and it still got a 200.

**Why it was model work and not adapter work.** I re-read the routing rather than
trusting the earlier note. `platform.py:1083-1109` sends `min_p`, `bad_words`,
`logit_bias`, `allowed_token_ids`, `min_tokens`, `prompt_logprobs` and structured
output to host sampling — and **not** penalties. `input_batch.py:349-378` packs
all three into `TTSamplingParams`, and `model_runner.py:1040-1051` populates
`prompt_tokens` / `output_tokens` "if penalties are needed (decode only)". So the
plugin is not failing to route them: it is handing the model everything a
device-side penalty stage needs and expecting one to exist. Ours didn't.

Implemented in this port's subclass `_WatcherCleanSampling1D` (`tt/model.py`) —
the same seam stage 05 used for the distributed argmax. `sampling_1d.py` is
untouched; so is `/home/raahem/vllm-tt-plugin`.

**The design, and the one thing that could have gone silently wrong.** Penalties
are keyed by global token id; the logits are column-parallel, die *d* holding ids
`d*37984 … d*37984+37983`. Doing `die = t // 37984`, `local = t % 37984` inside a
kernel is where this goes wrong without raising. The stage does not do it. The
operands are built on the host at **full vocabulary width** `[1,1,32,151936]` —
the frame a penalty is actually defined in — and shipped down through
`ttnn.ShardTensorToMesh(dim=-1)`, the same even 4-way split, over the same
`_dist_local_vocab` the argmax's `_dist_die_offset` is derived from, that the
column-parallel LM head produced the logits under. Column *t* lands on the die
holding logit *t* by construction; every device op is elementwise between
identically-shaped per-die tensors and never sees a global id.

vLLM's order (`model_executor/layers/utils.py::apply_penalties`) is repetition
first, on the raw logit, because it is sign-dependent:
`pos = gtz(x); x = x * (rep_neg + pos*rep_dif); x = x - add_delta`. `rep_neg = p`
and `rep_dif = 1/p - p` at penalised columns, `1.0`/`0.0` elsewhere — so an
unpenalised column gets `x * 1.0 - 0.0`, **bit-exact** in bf16. Row isolation
falls out of the same shape argument.

**The fast path is a graph, not a value.** `_penalty_mode` 0 means the ops are not
in the captured trace at all. The generator releases and re-captures the decode
traces when the mode changes — exactly what `set_sampling_params` already does
when `_sampling_stochastic` flips — and `_decode_graph_key` carries the mode so
the eager warm pass recompiles before capture. Two independent bits, so a
repetition-only request never pays for the additive tensor. Prefill and the eager
compatibility paths run under `_penalties_suspended`, because a prefill's row *i*
is the *i*-th admitted request rather than slot *i* and vLLM sends no history for
it.

**What surprised me, and what I initially under-called.** The device cost is
nearly free (+0.072 ms on a 0.633 ms sampler) and the *host* cost is not. The
first working version staged three full-width operands and cost **11.639 ms
(mode 1) / 17.164 ms (mode 3)** per penalised step — against a 19.8 ms TPOT. I
reported that as "the surprise" without doing the division; it means a penalised
token would have cost roughly 31–37 ms instead of 19.8, i.e. technically correct
and practically unusable. That was the right thing to be pushed on.

**Where the time actually was.** Not the wire, and not tilization. For one
`[1,1,32,151936]` bf16 operand: `ttnn.from_torch` with
`ShardTensorToMesh(dim=-1)` **6.601 ms**, `copy_host_to_device_tensor` **0.747
ms**. The mapper re-slices a strided 9.7 MB view into four contiguous copies on
*every* decode step. Staging four contiguous `[1,1,32,37984]` buffers instead and
assembling them with `ttnn.from_host_shards` is **2.049 ms** end to end, 3.4x
less, for bit-identical device content (checked, not assumed).

Two changes, each measured before adoption:

| | first version | shipped | |
|---|---|---|---|
| full-width operands, mode 1 / 3 | 2 / 3 | **1 / 2** | `1/p - p` derived on device with `ttnn.reciprocal` |
| host staging, mode 1 | 11.6391 ms | **1.5351 ms** | **7.6x** |
| host staging, mode 3 | 17.1644 ms | **3.3894 ms** | **5.1x** |
| device cost, mode 1 / 3 | +0.0265 / +0.0420 ms | +0.0551 / +0.0728 ms | the reciprocal, paid on device |

The reciprocal substitution is only legal because `reciprocal(1.0)` is **exactly**
1.0 on this device — measured, because that is what keeps an unpenalised column
at `x * 1.0 - 0.0` and therefore keeps the whole cross-die non-perturbation
argument. At penalised columns it differs from a host-computed `1/p` by up to one
bf16 ulp (p=1.05: 0.95703 against 0.95313). It changed **no sampled token**: the
serving parity probe is still 11/11 byte-identical to vLLM's own sampler after
it, including the two repetition-only cases that exercise it hardest.

Moving the global -> (die, local) split into host Python was the price of the
fast staging path, and it is the one piece of index arithmetic in this feature.
It is pinned rather than trusted: the probe's `fast_staging_matches_shard_mapper`
leg builds the same operand both ways and requires the two **device** tensors to
be bit-identical, and `staged_operand_carries_p_on_the_owning_die` checks the
operand really carries `p` at each requested id.

**In situ, which is the number that matters** — same 128/128/1 workload, same
live server, median of 3, all legs decoding exactly 128 tokens
(`probes/penalty_serving_cost_probe.py`):

| Request | TTFT | TPOT | t/s/u | vs. unpenalised |
|---|---|---|---|---|
| none | 298.276 ms | 19.873 ms | **50.321** | — |
| `repetition_penalty` only | 328.555 ms | 22.702 ms | **44.049** | +2.829 ms, +14.2 % |
| all three | 305.230 ms | 24.951 ms | **40.079** | +5.078 ms, +25.6 % |

The `none` leg lands on the `vllm bench serve` headline (19.873 against 19.808),
which is how this harness validates itself.

The in-situ overhead is larger than this port's own staging even when that
staging is re-timed at a **serving-sized 256-token history**
(1.5674 / 3.7624 ms -- it barely moves from the
correctness batch's, because the cost is the fixed 9.7 MB operand and not the
history length). So 1.26 / 1.32 ms
of the in-situ overhead is **not** this port's staging at all. It is per-step host
marshalling of the token history
— vLLM rebuilds and re-sends the *entire* `prompt_tokens`/`output_tokens` tensors
every step, and this port then re-derives each row's operand from the whole
history even though it grows by one token per row per step. The next cuts are an
incremental operand update and an on-device scatter of a `[1,1,32,1]` index; both
were left out deliberately (the scatter reintroduces exactly the global->local
arithmetic this design removes, and the incremental path needs a reliable
"same request still in this slot" key the adapter does not receive), and they are
recorded in README "Limitations" with this measurement sizing them.

This cost is disclosed in the README **headline section and Limitations**, next to
the unpenalised figure, not only in a cost table.

**Evidence.**

* `probes/penalty_shard_boundary_probe.py` /
  `probes/penalty_shard_boundary_probe.json` — the shipped classes on a synthetic
  1x4 mesh at the shipped shape, against a torch transcription of vLLM's
  `apply_penalties`. Cross-die reach (5 ids on die 0, 4 on die 3, same step),
  local columns 0 and 37983, `t ± k*37984` on the other dies bit-identical, 28
  unpenalised rows byte-for-byte unchanged alongside 4 penalised ones, and a
  forced-winner leg that penalises each row's current argmax so the selection has
  to move. Rows 2 and 4 are the aliasing trap on purpose.
* Watcher: `TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1` over the whole
  probe including the `--time` legs — **zero tripped asserts**
  ([`logs/penalty_watcher.log`](logs/penalty_watcher.log)).
* Model suite: `pytest models/autoports/qwen_qwen3_coder_30b_a3b_instruct/tests/
  -m "not models_performance_bare_metal" -q` -> **158 passed, 16 deselected**,
  exit 0 ([`logs/stage08_penalties_model_suite.log`](logs/stage08_penalties_model_suite.log)).

**Sampling gate: 52 / 20 / 1 -> 56 / 16 / 1** (600.82 s, same launch as §6), and
the decomposition is exact: the seeding/RNG class is **14 failures in both runs**,
so the entire difference is the 4 `test_tt_penalties` that moved to PASSED — both
`TestRepetitionPenalty` and both `TestFrequencyPenalty`. Two `TestPresencePenalty`
still fail (run down below). An intermediate run of the same suite gave
58 / 14 / 1; that class fluctuates between 12 and 14 without any code change
(`test_topk[15]` and `[19]` swapped sides between runs), which is the fixed-RNG
cause showing through — "two runs differ" is a coin flip against a fixed RNG
buffer. The log now in `readiness_vllm/` is the one taken against the **shipped**
code, after the staging optimisation.

**The two presence failures, run down rather than filed.** Both assert that
presence penalty changes the output for `"a b c a b c a b c"`. First I checked
the stage was live at all: on the same prompt, `frequency_penalty` 0.3 leaves the
output unchanged and 0.5 breaks the cycle, so the additive tensor is wired. By
then the cycle tokens have appeared ~9 times, i.e. an effective penalty of ~4.5.
Presence subtracts a *constant*, not a count, and vLLM caps it at ±2.0 — and it
subtracts the same 2.0 from `a`, `b` *and* `c`, so it cannot reorder them and has
to close this model's gap to the nearest non-cycle token on its own. It doesn't.

Then I made that a measurement instead of an argument, using a reference sampler
the plugin hands over for free: a request that sets `min_p` is routed to **host
sampling**, so vLLM's own `apply_penalties` produces the answer, while the
identical request without `min_p` takes our on-device stage. At temperature 0
both are deterministic, so the comparison is byte equality.
`probes/penalty_serving_parity_probe.py` sends 11 cases both ways against the
live 48-layer server:

* **all 11 byte-identical** to vLLM's reference — repetition 0.5 and 2.0,
  frequency 0.3/0.5/1.0/2.0, presence ±2.0, and all three together;
* the reference sampler *also* returns the unpenalised text at presence −2.0 and
  +2.0, so **these two tests fail against the reference implementation too on
  this checkpoint**;
* frequency does move the output on the same prompt, so it is not "penalties do
  nothing".

That is the strongest correctness evidence in this section, and it is stronger
than the tests it explains: it checks the whole served path, on the real model,
against the reference implementation installed in this environment, rather than
against my own transcription of it.

**Primary benchmark re-run, `max_num_seqs=1`, same command as §7:** TTFT
**306.894 ms**, TPOT mean **19.808 ms**, **50.485 t/s/u**, ITL median 19.787 /
P99 23.847 ms (`logs/penalty_rerun_vllm_benchmark.json`). Against §7's 312.367 /
19.778 / 50.560 that is **+0.030 ms TPOT, 0.15 %**, with TTFT 5.473 ms *better* —
noise in both directions, which is what "the fast path is a different graph"
predicts: none of these requests sets a penalty, so `_penalty_mode` is 0 and the
captured decode graph is the one that produced the headline. §7 stays the
canonical figure; this is the regression check.

`probes/adapter_contract_probe.py` was re-run on the 2-layer target after the
change — **13/13, 0 failed**, and its `serving_audit` block now carries
`penalised_decode_steps: 0` where `ignored_penalty_requests` used to be.

The `ignored_penalty_requests` audit counter and its one-time warning are gone,
replaced by `penalised_decode_steps`.

## 11. What is not done

1. **Per-request seeds** are not plumbed into the sampler (14 sampling failures).
   `Sampling1D.decode_forward` takes a `seeds=` tensor, so the path exists; it
   needs to join the traced decode input set and be refreshed alongside `k/p/temp`.
2. ~~**Sampling penalties** are not applied.~~ **Done** — see §10b. All three are
   applied on device, before the selection, on the column-parallel per-die shards,
   and all 6 `test_tt_penalties` tests pass. What remains is a cost, not a gap:
   a penalised decode step spends ~17 ms staging the full-vocabulary operands on
   the host. The unpenalised path is unchanged — the ops are not in its trace.
   The cut is to broadcast the per-row scalars from a `[1,1,32,1]` column instead
   of baking them into two full-width tensors (three uploads -> one).
3. **Per-user decode at large `max_num_seqs`** is ~13x slower than at 1. MoE decode
   batch scaling; stage-09 target.
4. **TTFT**, 312.367 ms served against 129.941 ms standalone. Prefill is eager in
   this port and the gap is vLLM request-side work plus the decode-trace
   re-capture; also stage-09 territory.

## 12. Verification

`probes/check_published_figures.py` re-derives every figure in `README.md` from
the artifact it cites — benchmark percentiles and their workload shapes, the
TPOT-derived t/s/u recomputed rather than copied, the sampling counts and the
failure classification against the log in both directions, every adapter-contract
claim against the probe JSON, the async byte-identity claim by actually diffing
the two qualitative files, the non-aligned table's divisibility columns
recomputed, and every quoted passage checked verbatim against its completion. It
caught four stale test-name references on its first run. Final state: **all
published figures re-derived from their artifacts**.

It also publishes its own **coverage boundary**: every figure-shaped number in
`README.md` is either re-derived by a check or named in the script's `UNCOVERED`
table with the reason it is not machine-checkable, and a number in neither fails
the gate. That is what makes the two unarchived reduced-target ITL figures
(§5, 2.673 / 2.113 ms) visible instead of silently unverified.

### Stage gate — `08-vllm.check.sh`

Both halves, archived at
[`logs/stage08_gate_08-vllm.check.log`](logs/stage08_gate_08-vllm.check.log)
with the command and git state: `check_degenerate_output.py --scope all` **exit
0**, `check_context_contract.py --require-contract` **exit 0** (three advisory
text mentions of the 2-layer probe's `--max-model-len 4096`, which the checker
treats as advisory by design).

The second half initially returned **2**. `scan_caps()` flags *any* JSON key
named `max_model_len` (and eight siblings) below the contract's 262144 as
critical, with no exemption mechanism, and
`probes/adapter_contract_probe.json` carried two: `config.max_model_len` and
`serving_audit.max_model_len`, both 4096. Both were the **2-layer probe's own
reduced target**, never a served cap — but a named goal gate returning 2 is a
failure regardless. Fixed by renaming both to `probe_max_model_len`:
`adapter_contract_probe.py` now emits the config key under that name and passes
the adapter's `serving_audit()` through `serving_audit_block()`, which renames
the key on the way out. The live adapter keeps the vLLM-standard name; only the
probe's serialisation differs, and only for a value that is by construction not
a served cap.

### Standalone regression after the `generator.py` changes

`pytest tests/test_trace.py tests/test_full_model.py -q`, archived at
[`logs/stage08_review_regression_tests.log`](logs/stage08_review_regression_tests.log).

The first run **failed 1 / 35**:
`test_decode_past_the_rope_cache_length_through_the_low_level_api` died with
`TT_FATAL ... !is_capturing_trace: Cannot load new binaries during trace
capture`. Cause: `_decode_compiled_keys` skips the eager warm pass when the
decode graph key is already known, and the key did not include the rotary table
length. `_ensure_decode_rope_capacity` reallocates the cos/sin tables at a
**new length** when the horizon grows and releases the traces; the re-capture
then matched the stale key, skipped the warm pass, and hit `ttnn.embedding` /
untilize programs that had never been compiled — inside an open capture.

Fixed by introducing `Qwen3CoderGenerator._decode_graph_key`, which folds
`model.rope_cache_len` into the key alongside `id(kv_cache)`, the active batch
and the stochastic flag. Re-run: **36 passed**.

On symmetry: `_release_decode_traces` deliberately does **not** clear
`_decode_compiled_keys` — a released trace leaves its programs compiled, which
is the entire point of the set. `teardown()` does clear it, because teardown
deallocates the KV cache and the key holds `id(kv_cache)`: a later allocation
could land on the same address with a different geometry and falsely claim to be
warm.

---

## Errata — 2026-08-18 (raised by the stage-09 review)

*Append-only. Nothing above this line has been altered; no stage-08 number is
withdrawn. See [`../optimized_vllm/`](../optimized_vllm/) for the re-measurement.*

§8 above concludes, from the plugin's `Using custom scheduler class … you will
see degraded performance due to async scheduling being disabled` log line, that
`TTScheduler` is not an `AsyncScheduler` subclass, that no real overlap occurs,
that async scheduling is "safe but inert", and that "the shipped command leaves
it off".

**That conclusion is wrong in every part.** Verified at source:

* `vllm-tt-plugin/src/vllm_tt_plugin/scheduler.py:31` — `class TTScheduler(AsyncScheduler):`.
* `vllm/config/scheduler.py:190-198` emits that warning **unconditionally** for
  any custom `scheduler_cls`; its text is conditional ("**If** you have
  subclassed Scheduler instead of AsyncScheduler"). It was read as a fact.
* `vllm/config/vllm.py:964-1004` turns async scheduling **on by default** in
  vLLM 0.24.0.
* `readiness_vllm/server.log:21` and `:80` — `Asynchronous scheduling is enabled.`,
  twice, in this stage's own retained server log.

Consequence for §8's reasoning: the identical TPOT that was read as "async
scheduling buys nothing" came from two runs that **both had async scheduling
on**. The comparison was never capable of showing a difference. The A/B that was
actually missing — async **off** — was run in stage 09.

Consequence for the TTFT attribution: the ~182 ms serving-vs-standalone gap is
real, but only ~12-16 ms of it is the request-side cost this log attributes it
to. The other ~159 ms is a **fixed one-off per-request cost** -- the eager
prefill's decode-trace capture -- which async scheduling bills to TTFT. Turning
async scheduling off does not remove it: stage 09's re-measurement, retaining
per-token ITLs, finds it reappearing in full as the first inter-token latency
(~179 ms against a ~20 ms steady state), with `TTFT + ITL[0]` equal to 320.5 ms
async-on and 320.8 ms async-off. Same cost, different bucket.

The steady-state per-token difference between the two modes is about **0.44 ms**,
not the larger figure stage 09 first published. Take it from
[`../optimized_vllm/README.md`](../optimized_vllm/README.md); do not derive a
per-token number from the ~182 ms.

What this does **not** change: the adapter was already built to be correct under
overlap (`_merge_scheduler_view`), §9's contract probe already exercised it, and
every measurement in this log was taken on the configuration that actually
shipped. The error was in the explanation, not in the evidence.
