# Serving the instruction-tuned checkpoint from the same autoport

Date: 2026-08-20 UTC. Host: `qb2-120-p04t03` (BH QuietBox 2, 2x p300c, 4 chips).

Why: `google/gemma-4-31B-it` already has a tt-inference-server entry, a nightly
lane and an `EVAL_CONFIGS` entry with real reference scores, all of which the base
checkpoint lacks. Running the autoport against `-it` weights therefore reuses the
platform's existing workflow instead of adding one. This document records what
that costs, what it does not buy, and what broke.

## How the checkpoints differ

Read from the two local checkpoints, not from documentation:

| | base `gemma-4-31B` | `gemma-4-31B-it` |
| --- | --- | --- |
| `architectures` | `Gemma4ForConditionalGeneration` | same |
| layers / hidden / heads / head_dim | 60 / 5376 / 32-16 / 256 | same |
| intermediate / vocab / sliding window | 21504 / 262144 / 1024 | same |
| `max_position_embeddings` | 262144 | same |
| weight shards | 2 | 2 |
| `chat_template.jinja` | **absent** | present, 18,683 B |
| `eos_token_id` | `1` | **`[1, 106, 50]`** |

Architecturally identical, so the model code needs no change. `tokenizer_config.json`
declares no `chat_template` in either; `-it` carries the standalone
`chat_template.jinja` that transformers reads since v4.43, which is why `-it`
serves the chat endpoint and the base checkpoint cannot.

## Implementation changes required

tt-metal branch `mvasiljevic/gemma4-31b-it-weights`, one commit per concern on top
of the benchmarks-doc commit.

### 1. Checkpoint resolution was hardwired to the base repo

`_resolve_checkpoint` (`tt/model.py`) fell back to the HF cache only for the
hardcoded `HF_MODEL_ID`, raising `checkpoint is not local` for anything else. It
now derives the cache directory from the id, for the ids in
`SUPPORTED_HF_MODEL_IDS`. A local path still short-circuits.

### 2. Multi-token EOS -- a real bug, not just an -it inconvenience

The decode loop compared `outputs[-1] == self.tokenizer.eos_token_id`, a single
value. The tokenizer reports only `<eos>` = 1, while `-it`'s
`generation_config.json` declares `[1, 106, 50]`, so end-of-turn (106) would never
have stopped generation. `resolve_eos_token_ids` unions the tokenizer value with
the checkpoint's generation config, and the loop tests membership.

### 3. The context contract required equality, not a ceiling

`generator_vllm.py` raised unless `max_model_len` **exactly equalled** the contract
value, so any spec entry had to be rewritten around the model. The hybrid KV pool
in `get_max_tokens_all_users` derives from `max_model_len`, so a shorter context
computes correctly -- the equality check was policy, not arithmetic. Relaxed to
`<=`, which is what lets the `-it` entry inherit the platform's `max_context:
49152` instead of overriding it. Above the contract still raises.

### 4. The qualitative harness refused any checkpoint with a chat template

Found by running it locally, in 45 seconds:

```text
RuntimeError: google/gemma-4-31B unexpectedly acquired a chat template;
update prompt rendering
```

`tests/run_full_model_qualitative.py` asserted `not tokenizer.chat_template`.
Downgraded to a printed note: the harness compares TT output against an HF
reference over the *same token ids*, so a raw completion continuation is a valid
probe for either checkpoint, and refusing outright made `-it` untestable.

### 5. The qualitative harness compared TT(base) against HF(-it)

Found by the same local run, after fix 4 let it get further. `--hf-model` was
passed to the HF **reference** model only; all three `build_generator` call sites
(`--benchmark-only`, `--aligned-ab-only`, and the main path) used the default id,
so the TT side loaded `HF_MODEL_ID` -- the base checkpoint -- while the reference
used `-it`.

On this host it surfaced as a clean crash, because the base checkpoint is not in
the HF cache here:

```text
FileNotFoundError: cached google/gemma-4-31B checkpoint not found under
  ~/.cache/huggingface/hub/models--google--gemma-4-31B/snapshots
```

**On a host where the base checkpoint is cached this would have been silent**: the
harness would have compared TT running base weights against an HF reference
running instruction-tuned weights and reported a PCC failure that looked like a
model bug. That makes it a latent trap independent of `-it` support, not just an
`-it` inconvenience. Fixed at all three call sites by passing
`model_id_or_path=args.hf_model`.

Both 4 and 5 were found locally, in minutes, by running the harness rather than
reasoning about it. Neither would have appeared in the CI run, which serves
through vLLM and never touches this script.

## Local verification on device: -it works

Measured on this host, greedy, same story prompt, 16-24 new tokens:

| Weights | Prompting | HF reference | TT |
| --- | --- | --- | --- |
| base | completion (correct) | coherent | coherent |
| `-it` | completion (**wrong**) | degenerate: repeats the prompt tail | degenerate: `"something" x11` then CJK tokens |
| `-it` | chat (correct) | not run | **coherent** |

The `-it` result with correct rendering:

```text
"something she had never seen before: a tiny, shimmering door carved
 directly into the gnarled root of the oak."
```

Three conclusions:

- **The base-tuned datatype policy carries over qualitatively.** BFP8 attention,
  BFP4 MLP and LoFi fidelities were selected against base weights; `-it` produces
  coherent, context-appropriate text under them. This does not replace a PCC or
  top-1 measurement, which needs a reference run.
- **The middle row was a prompting error, not a defect.** An instruction-tuned
  checkpoint fed raw completions degenerates on any implementation, which is why
  the HF reference -- containing no Tenstorrent code -- degenerated identically.
  The base-weight control row is what proves the harness, mesh, port and policy
  were fine all along.
- **The three implementation changes work end to end on device**: checkpoint
  resolution located `-it`, multi-token EOS is wired, and the relaxed context
  assert blocked nothing.

### A correction, and a method note

Finding 4's first fix was wrong. The assertion said "unexpectedly acquired a chat
template; **update prompt rendering**", and it was downgraded to a note while raw
completions were still rendered -- with the claim that "a completion continuation
is a valid probe for either checkpoint". The degenerate `-it` row disproves that.
The correct fix applies the template when the tokenizer declares one, and records
`prompt_mode`, `chat_template_present` and `rendering_method` from the tokenizer
rather than hardcoding the completion path.

On method: the coherence question was first attacked with the full qualitative
harness, which loads a 31B HF reference into ~62 GB of RAM and decodes on CPU
purely to have something to compare against. Two ten-minute loads went by before
the cheaper experiment was written: a TT-only script that renders the prompt,
calls `generator.generate(...)` and decodes, at 13 GB and no CPU decode. Where the
question is "is the output coherent", the base-weight run is already the reference
and no HF model is needed. Reach for the cheapest experiment that discriminates.

## What was deliberately not changed

The datatype policy. Stage 08 selected BFP8 attention, BFP4 MLP and LoFi
fidelities against **base** weights, validated at 0.92 top-1 / 1.00 top-5. A
quantisation policy validated on one weight distribution does not transfer for
free, and BFP4 MLP is the aggressive part. So **accuracy on `-it` is unmeasured**,
and every PCC, precision-sweep and eval figure in these docs describes the base
checkpoint.

## The tt-inference-server change

Branch `mvasiljevic/gemma4-31b-it-autoport`, **one commit** on top of
`vvukoman/add-8-models-to-release-flow` (`60f80c4b`): 3 files, +50/-1.

The `-it` entry mirrors the existing `tt_transformers` entry rather than inventing
a parallel configuration, so the autoport runs the same lane, the same eval config
and the same serving parameters. Inherited unchanged: `max_concurrency: 1`,
`max_context: 49152`, `MESH_DEVICE`, `GEMMA4_PAGE_BLOCK_SIZE`,
`GEMMA4_MAX_TOKENS_ALL_USERS`, `fabric_config`, `trace_region_size`, `status`,
`has_builtin_warmup`, `VLLM_ALLOW_LONG_MAX_MODEL_LEN`.

Five deviations, each required:

| Deviation | Why |
| --- | --- |
| `impl: gemma4_31b_autoport` | the point of the entry |
| `default_impl: false` | the `tt_transformers` entry owns the default; two defaults raise `Multiple default impls`. Select with `--impl gemma4-31b-autoport` |
| `EXTRA_MODELS_DIR` + `GEMMA4_31B_AUTOPORT_DIR` | without them the arch resolves to `models/demos/gemma4` and the run reports a different implementation |
| `sample_on_device_mode: all` | `decode_only` routes prefill to host sampling, which the adapter gates behind a logprob-compatibility env var rather than supporting as a serving mode |
| no `vllm_args` | that entry's tool-call and reasoning parsers killed its own release run 32132213868: `Gemma4ToolParser.__init__() takes 2 positional arguments but 3 were given` |

Plus `llm_module/runner.py`: `wait_healthy_timeout_s` 1200 -> 2400 s, overridable
with `TT_WAIT_HEALTHY_TIMEOUT_S`. Required, not cosmetic: a first start on a node
pays a ~7m46s HF download plus ~17m weight conversion inside that window, and
1200 s fits only a second run (see `agentic_bringup_ci_dispatch.md`).

Rejected alternative: adding only `EXTRA_MODELS_DIR` to the existing entry, which
would be one line. Its `max_context: 49152` was survivable after change 3, but
`sample_on_device_mode: decode_only` and the parser args are not, so overriding
those amounts to writing this entry anyway -- and it would alter the path the
stock model uses.

## The docker image cannot be reused for this

Verified from the `docker run` command in a run log. The bind mounts are
tt-inference-server only:

```text
--volume volume_id_gemma4_31b_autoport-gemma-4-31B:/home/container_app_user/cache_root
--mount .../tt-inference-server/reference_config  -> /app/reference_config
--mount .../tt-inference-server/utils             -> /app/utils
--mount .../tt-inference-server/tests             -> /app/tests
--mount .../tt-inference-server/vllm-tt-metal/src -> /app/src
```

**tt-metal is baked into the image** at `/home/container_app_user/tt-metal`, cloned
at the SHA in the image tag. So spec, eval-config and `reference_config` changes
are free, and any tt-metal change costs a ~60 min rebuild. This is the opposite of
the eval-config case, where `-f docker-image=...` skips `resolve-shas` and the
build entirely (`on-dispatch.yml`: both are gated on `inputs.docker-image == ''`,
and `run-tests` uses `always()` so skipped builds do not block it).

## Local verification

`tests/test_it_checkpoint_support.py` -- 9 tests, host-only, no device, 2.6 s.
Three of them use the real checkpoints rather than fixtures:

- EOS ids resolve to `{1}` for base and `{1, 106, 50}` for `-it`, read from the
  actual `generation_config.json` files
- the decoder contract accepts both configs, for a sliding and a full-attention
  layer
- **the two checkpoints expose identical weight names and shapes**, read from
  safetensors metadata without materialising tensors -- this is what makes "no
  model change needed" evidence rather than assertion

Plus resolution behaviour: both ids supported, local paths short-circuit, each id
resolves from its own cache directory, an unrelated id still raises. Full
host-runnable suite: 198 passed. tt-inference-server spec tests: 151 passed.

## CI

Run [32355285037](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/32355285037),
`--workflow release` (evals and benchmarks), dispatched with
`impl-of-model=gemma4-31b-autoport`.

| Repo | Resolved commit |
| --- | --- |
| tt-metal | `7a3c6e16109e00168c5bb3c046eeb87e3b3f96ae` |
| tt-inference-server | `14c3a27af1dd15d25fe5078164852bbd4f508394` |
| vllm-tt-plugin (`main`) | `fc07f3dd77be8cbd4f7abd46811be178a81fe252` |

`determine-server-type` resolved `gemma-4-31B-it` with impl `gemma4-31b-autoport`,
the first time a non-default impl has been selected on this path. Note the plugin
SHA moved from `bd150c7e` (the green base-model run) to `fc07f3dd`, so this is a
two-variable run on that axis as well.

## CI result: run 32355285037, `--workflow release`

`completed/success`, acceptance `PASS`. Report id
`id_gemma4-31b-autoport_gemma-4-31B-it_p300x2_2026-08-20_11-30-57`,
`model_impl: gemma4-31b-autoport`, `tt_metal_commit 7a3c6e16109`,
`vllm_commit fc07f3d`.

Confirmed from the log: `Registered TT model TTGemma4ForConditionalGeneration ->
models.autoports.google_gemma_4_31b.tt.generator_vllm:Gemma4ForCausalLM`, so the
autoport served, not `models/demos/gemma4`. `GPU KV cache size: 49,152`, the
inherited `max_context` -- which only worked because of the relaxed context
assert. `Waiting for inference server ... (timeout 2400s)`, so the raised health
budget was in force and the server came up.

### What the green PASS hides

| Item | Reality |
| --- | --- |
| `r1_gpqa_diamond` | **90** against an H100 `-it` reference of 80, ratio 1.125, PASS. First graded accuracy number for this autoport |
| `terminal_bench_2` | **0** (`n_trials: 0, pass_at_1: 0.0`), FAIL |
| `swe_bench_verified` | **0**, FAIL |
| Benchmark point 1, the only one with targets | **FAIL**: TTFT 581.8 ms against a `functional` target of 460 ms (ratio 1.265), and 6.324x the `complete` target of 92 ms |
| Throughput checks on that point | PASS: 28.0 t/s/u against a 3.70 target |
| Other 7 sweep points | NA, no targets exist for them |
| Spec tests | NA, no Gemma suites |

Acceptance is `PASS` only because `status: EXPERIMENTAL` waives the failures:
`Evals: PASS (1/3 passed, 2 waived)`, `Benchmarks: PASS (0/8 passed, 1 waived,
7 NA)`. That status is inherited from the `tt_transformers` entry, so it is
platform behaviour rather than anything this branch introduced -- the same fact
`local_release_flow_p300x2.md` records for the base model.

### Attribution of each failure

- **The two agentic zeros are ours.** The log carries 45 instances of `tool choice
  requires --enable-auto-tool-choice and --tool-call-parser to be set` with
  `BadRequestError`: the agentic harness sent tool-calling requests and the server
  rejected every one, because this entry omits those `vllm_args`.

  But the alternative is not better. Carrying them over is what killed the
  `tt_transformers` entry's own release run 32132213868
  (`Gemma4ToolParser.__init__() takes 2 positional arguments but 3 were given`).
  So on this stack **neither choice passes those two tasks**: omitting the args
  yields two soft zeros, including them yields a hard crash. The root cause is the
  upstream parser bug, and the deviation only chose which failure mode to take.
  If those blocks matter, fix the parser first.

- **The TTFT target miss is the implementation's**, and it echoes the base-model
  finding: 581.8 ms here, 592.99 ms for the base model on the chat endpoint, and
  100.91 ms locally on `/v1/completions`. The ~480 ms gap tracks the chat path
  rather than the checkpoint. For `-it` the chat endpoint is the *correct* one, so
  this is not something the driver fix would remove -- it is real overhead on the
  path this model must use, and it is what pushes TTFT past the 460 ms target.
  Unexplained; candidates are per-request template rendering and trace capture
  inside the measured window.

- **The misattributed reference is upstream.** `Published Score Ref` reads
  `https://huggingface.co/Qwen/Qwen3.6-27B` for all three Gemma tasks, inherited
  from TTI's eval config. Already recorded as a correction in
  `problems_and_fixes_log.md`; still wrong upstream.

### Does the run deserve to be green?

By the platform's rules, yes: every criterion is either passed or waived. On the
evidence, a reader should take from it that the autoport **serves `-it` correctly
and scores 90 on GPQA diamond against an H100 reference of 80**, and should not
take from it that latency targets were met or that the agentic tasks work. A
`PASS` on an `EXPERIMENTAL` model asserts that the workflow executed, which is
exactly what the base-model documentation already warns.

## Open risks

- ~~**Accuracy on `-it` is unmeasured.**~~ Partly resolved: `r1_gpqa_diamond`
  scored 90 against an H100 reference of 80, and local generation is coherent. No
  PCC or top-1 number exists yet, so numerical equivalence is still unmeasured.
- ~~**The eval config's tasks may not fit.**~~ Confirmed: `r1_gpqa_diamond` is
  the usable block; the two agentic tasks scored 0 because tool-calling requests
  were rejected. See the attribution section above -- the upstream parser bug means
  neither including nor omitting the args makes them pass.
- ~~**`trace_region_size` inherited at 200000000**~~ -- it held; trace capture
  succeeded and the sweep completed, so the 268435456 override the base entry
  carried is not required on this path.
- **TTFT is ~5x the completions-endpoint figure and misses the functional
  target.** Open, and the most substantive unexplained result.
