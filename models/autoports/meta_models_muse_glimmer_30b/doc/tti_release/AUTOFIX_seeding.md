# AutoFix Report — seeded reproducibility at batch 32 (`test_non_uniform_seeding`)

**Verdict: root cause identified and fixed**, in shared code
(`models/common/sampling/tt_sampling.py`), with a one-call reordering. The
tt-inference-server chat-completions parameter-conformance file now passes
**22/22** where it previously failed 2, and the shared plugin sampling suite's
seeding/isolation files now pass **29/29** where the two previous stages
recorded 6–7 failures and documented them as a known limitation.

## Starting Evidence

* Original failing check:
  `llm_module/test_vllm_chat_completions.py::test_non_uniform_seeding` in
  `/home/ttuser/dev/muse-glimmer/tti-release/muse-glimmer-30b/tt-inference-server`,
  run against the release server from
  `doc/tti_release/bench/serve_release.sh`. Recorded failure in
  `doc/tti_release/logs/tti_release_20260816T014529Z.log`:
  `AssertionError: Determinism Failed for seed=0. Expected 1 unique output, found 2.`
* No fresh `$autodebug` pass was run: the caller had already reproduced and
  characterised the failure (15/16 or 14/16 agreement at 32-way, uniform at
  2/4/8-way, always the same character offset and the same alternative
  continuation), and `doc/vllm_integration/README.md` (*Limitations* 1,
  *Sampling suite*), `doc/vllm_integration/AUTOFIX.md` and
  `doc/optimized_vllm/README.md` (*Sampling suite*) already recorded the prior
  stages' position: the class was **classified, never diagnosed** — no
  hypothesis about it had been tested, so there was nothing to avoid repeating.
* Evidence artifacts for everything below: `doc/tti_release/seeding/`.

## Hypothesis Experiments

### H1 — the divergence is in the *seed stream*: a request that joins the running batch later has its seed counter aligned to the wrong position

The design the caller pointed at: `SeedManager` derives the per-token device
seed as `hash(request_seed, counter)` and
`align_seed_counters_to_positions()` re-ties `counter` to the absolute decode
position so vLLM moving a request between slots cannot reset its stream.

**Experiment A (localisation, no restart).** Run the 32-way test twice more,
once with `logprobs` requested. On this 4-die mesh the TT plugin routes any
logprobs request to **host** sampling, so vLLM's own per-request RNG picks the
token from the same device logits. `doc/tti_release/seeding/seed_probe_host.py`:

```
--- A device sampling (logprobs=False)
    seed=0: n=16 distinct=2      (13 + 3)
--- B host sampling (logprobs=True)
    seed=0: n=16 distinct=1
--- B host sampling (logprobs=True) trial2
    seed=0: n=16 distinct=1
```

**Result:** the per-row logits agree across the 32-row batch (a host sampler
fed them is uniform); the defect is inside the device sampling path.

**Experiment B (direct measurement).** Env-gated instrumentation in the
autoport generator logged, per decode step, every active slot's position, its
`SeedManager` request seed, its counter and the resulting device seed, plus the
per-slot sampled token. Server restarted with `MUSE_GLIMMER_SEED_TRACE` set;
four 32-way trials. Artifacts `seeding/seedtrace_trial1.jsonl`,
`seeding/seedtrace_trial4.jsonl`.

```
step 3 pos [65]*32  cnt [67]*32   step 4 pos [66]*32 cnt [68]*32 ...
first token divergence at tok_step 12
dev_seed(zero-seed slots) distinct: {776604}      <- ONE value, all 16 rows
```

Every active row is handed the same absolute position every step, so all 16
`seed=0` rows are pushed **one** device seed, and one of them still sampled a
different token (42526 vs 2991).

**Verdict: REFUTED.** The seed values, the counters and
`align_seed_counters_to_positions` are all correct here; nothing in
`SeedManager`, `reset_seed*`, `apply_slot_remap` or the adapter's
`start_pos` handling contributes. Same instrumentation refutes the caller's
prior suspicion that a later-admitted request gets the wrong offset: no row
ever had a position different from the rest.

*(One incidental observation, not the bug: `start_pos` repeats a value on one
step per request — 65, 66, 66, 67, 68 — so one decode step reuses the previous
step's device seed. It is uniform across all rows, so it cannot split a batch,
and it is not touched by this fix.)*

### H2 — the divergent rows are a fixed set of *batch slots* (a per-core defect), not a race

**Experiment.** Four instrumented 32-way trials, recording which batch slot's
token stream diverged versus which slots happened to hold a `seed=0` request
(`doc/tti_release/seeding/seed_trials.py`):

| trial | seed=0 slots ∩ {0,11,22} | slots that diverged |
|---|---|---|
| 1 | {22} | {22} |
| 2 | {0,22} | {0,22} |
| 3 | {0,22} | {0,22} |
| 4 | {} | none — **all 16 identical** |

**Verdict: VERIFIED.** The divergent set is exactly `{0, 11, 22}` intersected
with the slots holding a `seed=0` request. Those three are the first three
cores in *column* order on this device grid; the caller's "which request
diverges changes run to run" is just which client request the scheduler put in
those slots. This also explains the batch-size dependence: at 2/4/8-way the
`seed=0` requests rarely land on one of them.

### H3 — a compute kernel running between `ttnn.manual_seed` and `ttnn.sampling` destroys the per-core PRNG state

`ttnn.manual_seed` installs the RNG state as a **register on each core**
(`manual_seed/device/kernels/compute/manual_seed_receive_all_data.cpp` →
`rand_tile_init(seed)`); `ttnn.sampling`'s compute kernel then advances it with
`rand_tile()` and the writer reads element 0 of the resulting tile
(`sampling/device/kernels/dataflow/writer_interleaved.cpp:112`). Nothing
carries that state in a tensor. `tt_sampling.py` called `manual_seed` and then
ran `_adjust_values_for_tiebreak` — ~17 elementwise ops — before `sampling`.

**Experiment (device, no model in the loop).** 32 users, identical near-uniform
top-32 distribution, the *same* seed pushed to all 32 users, so the sampled
index reports the drawn uniform directly.

```
doc/tti_release/seeding/rand_probe.py  40   -> manual_seed then sampling:  40 seeds, 0 disagreements
doc/tti_release/seeding/rand_probe2.py tiebreak 20
                         -> manual_seed, tie-break chain, sampling:
                            19 of 20 seeds disagree, always users {0, 11}
```

Prefix bisect of the chain (`doc/tti_release/seeding/rand_probe3.py`): clean through op 12, breaks
at op **13**, `ttnn.typecast(int32 -> bfloat16)`. Single-op confirmation
(`doc/tti_release/seeding/rand_probe4.py`):

```
none               0 bad     typecast bf16->i32  0 bad
typecast i32->bf16 6/6 bad {0,11}    typecast i32->u32   0 bad
typecast u32->bf16 6/6 bad {0,11}    abs / add / max / eq / untilize  0 bad
exp bf16           6/6 bad {0,1}
```

So it is a property of *which* kernel runs in between (and of which cores it
lands on — `exp` picks a different pair), not of dispatch in general.

**Verdict: VERIFIED — this is the root cause.**

**Fix (smallest, shared code):** move the `ttnn.manual_seed(...)` call in
`TTSampling.__call__` so it is the **last** op before `ttnn.sampling(...)`,
i.e. after `_adjust_values_for_tiebreak`. One call moved, plus the comment that
records why the order is load-bearing. No autoport-local change; the autoport's
`tt/generator.py` and `tt/generator_vllm.py` are untouched, and the diagnostic
instrumentation used above was removed before every proof run below.

**Verification (op level):** with the intervening op moved *before* the seed
call (`SEED_LAST=1 doc/tti_release/seeding/rand_probe5.py`), `i32_to_bf16`, `u32_to_bf16` and
`exp` all give 0 disagreements over 6 seeds each.

`models/common/modules/sampling/sampling_1d.py` — the other production in-tree caller of
`ttnn.manual_seed` — already seeds immediately before `ttnn.sampling` with no
intervening compute, so it needs no change.

## Final Status

**Fixed.** One file changed: `models/common/sampling/tt_sampling.py` (shared
sampler). Nothing else in the working tree is modified; nothing is committed.

Proof, all against the release server relaunched from
`doc/tti_release/bench/serve_release.sh` with the fix in and the diagnostics
out:

* **the original failing check, 3 trials** —
  `cd .../tt-inference-server && .workflow_venvs/.venv_workflow_run_script/bin/python -m pytest llm_module/test_vllm_chat_completions.py -k test_non_uniform_seeding --output-path $OUT --task-name vllm_chat_completions --endpoint-url http://127.0.0.1:8000/v1/chat/completions --model-name meta-models/Muse-Glimmer-30B -q`
  → `1 passed` ×3.
* **the whole conformance file** — same command without `-k` →
  **`22 passed in 373.54s`** (was `2 failed, 20 passed`;
  `test_penalties[presence_penalty-1.2-repeat_trap-messages0]`, the other
  failure, also passed in this run — that one was already classified by the
  vLLM-integration stage as prompt-margin-bound rather than a defect, and one
  passing run is not by itself proof that it is now stable).
* **32-way seed=0 reproduction, 3 trials** — `python doc/tti_release/seeding/seed_repro2.py` →
  `distinct-per-trial: [1, 1, 1]`, no outliers.
* **both halves, batch sweep** — `python doc/tti_release/seeding/seed_repro.py` →
  `distinct-seed0-outputs by batch size: {2: 1, 4: 1, 8: 1, 32: 1}`, and at 32
  the non-zero half is `seed!=0 requests: 16, distinct: 16`.
* **shared plugin sampling suite** —
  `python -m pytest /home/ttuser/dev/vllm-tt-plugin/tests/tt/test_seeding_and_variety.py /home/ttuser/dev/vllm-tt-plugin/tests/tt/test_request_isolation.py -q --tt-server-url=http://127.0.0.1:8000 --tt-model-name=meta-models/Muse-Glimmer-30B --tt-max-num-seqs=32`
  → **`29 passed`**. These files held the 6–7 failures the vLLM-integration and
  optimized-vLLM stages classified as *"seeded reproducibility at batch > 1"*
  (`test_seeding`, `test_same_seeds_reproduce_across_batches`,
  `test_uniform_seed_deterministic[10-0/1][32-0/1]`,
  `test_specific_seed_reproducible[0]`, `test_mixed_params_batch`). That
  documented limitation is resolved, and both stage READMEs now overstate it.
* **unseeded-serving regression** —
  `bash models/autoports/meta_models_muse_glimmer_30b/doc/tti_release/bench/qualitative.sh`
  → ends `No degenerate output detected.  exit=0`, and
  `doc/tti_release/qualitative/qualitative_vllm_vs_datatype_sweep_chat.json`
  still reports `first_divergence: 2` on all six prompts (the OpenAI API
  stripping one special token).

### Remaining risks / follow-up

* **The underlying kernel behaviour is not fixed, only avoided.** Some SFPU
  kernels (`typecast -> bfloat16`, `exp`) clobber the per-core RNG state that
  `ttnn.manual_seed` installs; others (`add`, `abs`, `max`, `eq`,
  `typecast -> int32`) do not. Any future op inserted between the seed call and
  `ttnn.sampling` will silently re-break seeded reproducibility for the users
  on the cores it touches, with no error. This is worth raising upstream
  against `ttnn.manual_seed` / `ttnn.sampling`: the seed op has no way to
  express "this state must survive to the next op", and a durable fix would
  either pass the per-user seed into `ttnn.sampling` itself or make
  `rand_tile_init` state a tensor. A unit test in
  `tests/ttnn/unit_tests/operations/reduce/test_manual_seed.py` that puts a
  `typecast -> bfloat16` between the two ops and asserts 32 equally-seeded
  users agree would catch a regression at op level; it is not added here
  because this stage's scope is the model port.
* The `start_pos` value repeated on one decode step (noted under H1) is
  unexplained and untouched. It costs one token of seed-stream advance, is
  identical across every row, and no check in scope observes it.
* `test_penalties[presence_penalty-...]` passing is reported as observed, not
  claimed as fixed.
