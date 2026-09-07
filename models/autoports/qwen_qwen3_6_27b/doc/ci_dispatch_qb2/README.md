# Running QB2 CI for Qwen3.8-27B on the autoport

What it takes to get `tt-agentic-bringup-qb2` to run benchmarks and evals
against *our* implementation, and why each piece is needed. Everything below is
measured or read out of the code that consumes it, not inferred.

## The dispatch was never wrong

`tt-shield-dispatch.yml` is the pre-rename name of
`manual-tt-shield-dispatch.yml` (commit `79921d5`, "rename to fit naming
convention"); it is the only workflow in the repo (id `342177897`). The callee
maps the inputs correctly:

```yaml
# tt-shield/.github/workflows/on-dispatch.yml
model: ${{ inputs.custom-model || inputs.model }}
inference-server-sha: ${{ inputs.inference-server-git-ref }}
```

Five runs failed with `No model spec matches model='Qwen3.8-27B'` for one
reason: **`inference-server-git-ref` defaults to `main`, and tt-inference-server
main has no Qwen3.8-27B spec.** The onboarding lives on branches.

Runs: 34113034709, 34113046572, 34113175558, 34113186560, 34113429291.

## Which tt-inference-server ref

Qwen3.8-27B exists on exactly two branches, both descended from
`vvukoman/add-8-models-to-release-flow`:

| | `vvukoman/add-8-models-to-release-flow` | `mvasiljevic/qwen38-autoport-qb2` |
| --- | --- | --- |
| `impl` | `qwen36_blackhole` — the **demo** | `qwen36_autoport` |
| bundle | none | `EXTRA_MODELS_DIR` + `QWEN_AUTOPORT_MODEL_ID/REVISION` |
| `trace_region_size` | `1073741824` | `200000000` |

vvukoman's spec serves the demo, so a green run there measures nothing of this
branch. Its 1 GB trace region is also the value that OOMs at startup (below).

## The main merge is not optional

`mvasiljevic/qwen38-autoport-qb2` (vvukoman + 3 commits) still fails, one stage
later:

```
ERROR: Failed to determine server type: resolve_model_spec not found in model_spec.py
```

`resolve_model_spec` was added to tt-inference-server main by #5027 ("Harden
release resolution against ambiguous catalog entries"). Neither Qwen3.8 branch
has it — both fork 63 commits back.

The subtlety: qb2 calls
`on-dispatch.yml@vvukoman/enable-on-dispatch-cross-repo-trigger`, and the script
at *that* SHA (`ba2f0331`) does **not** use `resolve_model_spec` — nothing under
`.github/scripts/` there does. But `workflow_determine-server-type.yml` checks
tt-shield out at `${{ github.job_workflow_sha }}`, which resolves to **tt-shield
main**:

```
git fetch ... +refs/heads/main:refs/remotes/origin/main
git checkout --force -B main refs/remotes/origin/main
```

So the dispatch always runs *main's* `determine_server_type.py`, whatever ref
`uses:` names — and main's version requires `resolve_model_spec`. Pinning a
tt-shield ref cannot avoid it; the tt-inference-server ref must carry it.

Measured A/B, same tt-shield SHA both times:

| `inference-server-git-ref` | `determine-server-type` |
| --- | --- |
| `mvasiljevic/qwen38-autoport-qb2` (no main) | failed — `resolve_model_spec not found` |
| `mvasiljevic/qwen38-autoport-ci` (+ main) | **success** |

Main also re-keyed the benchmark catalog by full HF repo id (#4722); the lookup
is `model_performance_reference.get(hf_model_repo)`, which no longer matches a
bare basename, so the branch's six models are re-keyed on merge.

## Working branch

`tt-inference-server` `mvasiljevic/qwen38-autoport-ci` =
`vvukoman/add-8-models-to-release-flow` + merge of `main` + one commit.

```bash
gh workflow run manual-tt-shield-dispatch.yml \
  --repo tenstorrent/tt-agentic-bringup-qb2 --ref main \
  -f model=Qwen/Qwen3.8-27B \
  -f runner-label=bh-qb-ge -f device-type=p300x2 \
  -f workflow=benchmarks \
  -f tt-metal-git-ref=mvasiljevic/qwen38-deltanet-kda \
  -f inference-server-git-ref=mvasiljevic/qwen38-autoport-ci \
  -f vllm-git-ref=main -f impl-of-model=default
```

`resolve_model_spec` accepts either `Qwen/Qwen3.8-27B` (matches
`hf_model_repo`) or the bare `Qwen3.8-27B` (matches `model_name`, via
`Path(model).name`). Both resolve; the full id is unambiguous.

## The four changes that are actually needed

Each was checked against its consumer. This is the complete set.

### Required to serve our implementation, not the demo

- **`EXTRA_MODELS_DIR`** — `Qwen3_5ForConditionalGeneration` resolves in the
  vLLM plugin's built-in map to `models.demos.blackhole.qwen36`, so without this
  the server silently serves the **demo**. The plugin registers every bundle
  under this directory *ahead* of its built-in map, so the autoport claims the
  architecture first. Per `models/autoports/vllm_bundles/README.md`, serving the
  demo "silently invalidates any autoport release report".
- **`QWEN_AUTOPORT_MODEL_ID` / `QWEN_AUTOPORT_MODEL_REVISION`** —
  `tt/functional_decoder.py` defaults these to `Qwen/Qwen3.6-27B`:
  ```python
  MODEL_ID = os.environ.get("QWEN_AUTOPORT_MODEL_ID", "Qwen/Qwen3.6-27B")
  ```
  Without them the autoport loads **3.6 weights and reports them as 3.8**.

### Required to start at all

- **`trace_region_size` 1073741824 -> 200000000** — the value is subtracted
  *per bank* (8.6 GB/device) and the engine dies during startup:
  `Out of Memory: ... 476544000 B DRAM buffer across 8 banks ... bank size is
  3198599552 B`. 200 MB is what the autoport derives its serving capacity
  against. Measured on p300x2.
- **`llm_module/runner.py`: 1200 s timeouts -> `None`** —
  `wait_for_healthy()` / `capture_traces()` treat `None` as
  `DEFAULT_WAIT_HEALTHY_TIMEOUT_S` (3600 s), verified on current main:
  ```python
  effective_timeout = DEFAULT_WAIT_HEALTHY_TIMEOUT_S if timeout is None else float(timeout)
  ```
  The runner defeated that with a hardcoded 1200 s no caller could raise.
  Qwen3.8-27B on p300x2 spends ~7 min staging weights plus ~15 min loading, so
  1200 s killed the server mid-load (bring-up run 33405666767) for reasons
  unrelated to the model. This is the one change not scoped to Qwen3.8; it only
  relaxes a cap, and no caller passes these arguments. Unaffected by the prefill
  work — this is model *load* time.

### Not required to run: attribution only

- **`impl: qwen36_blackhole -> qwen36_autoport`** (plus its `ImplSpec`).
  `ImplSpec.code_path` feeds the generated docs/report link, not the runner —
  execution is byte-identical either way. Without it the release report credits
  autoport numbers to `models/demos/blackhole/qwen36`. Drop this hunk if the new
  `impl_id` is unwelcome; nothing else depends on it.

### Deliberately unchanged

`TT_QWEN35_TEXT_VER` stays `qwen36_blackhole`. The upstream selector whitelists
only that value and raises on anything else; the bundle README says "leave unset
(or `qwen36_blackhole`)". Setting it to `qwen36_autoport` hard-fails at
registration.

## Merge fallout that is not ours

Main and vvukoman's branch each added **DiffusionGemma** and
**gemma-4-26B-A4B-it** independently. The catalog rejects the collision
outright, and this breaks *every* model, not just those two:

```
ValueError: Duplicate model spec leaf identity:
  ('google/diffusiongemma-26B-A4B-it', 'P300X2', 'vLLM', 'diffusion_gemma')
```

So a dedup is forced before Qwen3.8 can resolve at all. Which copy to keep:

- **gemma-4-26B-A4B-it — vvukoman's copy.** No test distinguishes them.
- **DiffusionGemma — main's copy**, in `llm.yaml` and `eval_config.py` alike.
  Not a preference: main carries eight tests that validate main's version
  (`test_diffusiongemma_dev_spec_matches_validated_256k_contract`,
  `TestDiffusionGemmaEvalContract`, `test_run_vllm_api_server`, ...) and
  vvukoman's earlier draft fails all eight.
- **`test_suites/llm.json`** — main's DiffusionGemma suite kept, and the model
  dropped from vvukoman's generic conformance suite. Listing it in both collides
  on the generated suite id `diffusiongemma-26b-a4b-it-p300x2` and silently
  shadows the model-specific suite.
- **Benchmark targets and eval published scores stay vvukoman's**, including its
  gemma-4-31B-it figures over main's.

## Verification

- `resolve_model_spec("Qwen/Qwen3.8-27B", "p300x2")` returns
  `impl=qwen36_autoport`, `code_path=models/autoports/qwen_qwen3_6_27b`, all
  four env vars set, `trace_region_size=200000000`.
- Eval config resolves `r1_gpqa_diamond`, `terminal_bench_2_1`,
  `swe_bench_verified`; benchmark targets resolve for `p300x2`.
- `tests/`: **5 failed / 1751 passed** on the merged branch and **5 failed /
  1751 passed** on clean `origin/main` — the same five, all pre-existing
  (`test_logging_utils`, `test_requirements_*`). No regression from the merge.
  (`tests/test_helm_generator` and `test_promote_dev_spec_to_prod` cannot be
  collected in this environment: `ModuleNotFoundError: ruamel`, on main too.)
- `determine-server-type` succeeds; runs 34116873055 (benchmarks) and
  34116898537 (evals) proceed to building the inference server.

## Long-ISL evals that used to time out

All numbers below are **measured on device**, not estimated.

### Under vLLM, like for like

The recorded baseline is the vLLM serving sweep in
`qwen38_checkpoint_swap/benchmarks_c1/sweep_summary.json`. Re-running the same
ISL 65536 point under vLLM (`vllm_isl65536_result.json`):

| ISL | metric | recorded (2026-08-31) | now | |
| --- | --- | --- | --- | --- |
| 65536 | `mean_ttft_ms` | 1837053.3 | **569324.7** | **3.23x** |
| 65536 | `mean_tpot_ms` | 69.67 | 56.36 | 1.24x |
| 65536 | completed / failed | 1 / 0 | 1 / 0 | |
| 131072 | `mean_ttft_ms` | **TIMEOUT** (rc124, no number) | **1150441.5** | -- |
| 131072 | `mean_tpot_ms` | -- | 57.96 | |
| 131072 | completed / failed | never completed | 1 / 0 | |

ISL 131072 is the point the recorded sweep could not finish at all; it now
returns a number. Because the baseline is a timeout rather than a measurement,
there is no ratio to quote for it.

The TPOT change is **not** attributable to the prefill work, which cannot affect
decode; it is most likely the fused KDA conv in the decode path, and possibly
`max_num_seqs` (see the caveat below). Do not report it as a prefill result.

A useful cross-check at both points: the bare harness measured 573.907 s and
1151.208 s TTFT, vLLM measures 569.325 s and 1150.441 s -- 0.8% and 0.07% apart.
At these ISLs vLLM serving overhead is negligible, so bare-harness prefill
numbers can be read as serving numbers.

**Caveat on the baseline.** The sweep artifacts never recorded the server's
`max_num_seqs` (`benchmarks_c1` names the *client* concurrency, a different
setting), so one variable in the 3.23x is unverifiable. Anything reported from
that sweep should be treated as a soft baseline until a server-side config is
recorded alongside it.

### `max_num_seqs 32` cannot prefill past 32768 tokens at all

The first three attempts at this point all died identically:

```
TT_FATAL: Out of Memory: Not enough space to allocate 10737418240 B DRAM buffer
across 8 banks, where each bank needs to store 1342177280 B, but bank size is
4247339392 B (allocated: 3354056896 B, free: 893282496 B)
```

It is not a leak. The size is exact arithmetic --
`32 x 32768 x 5120 x 2 = 10737418240` -- for the embedding of one streaming
prefill chunk: `self.batch` x `PREFILL_STACK_CHUNK_SIZE` x `hidden_size` x bf16.
It fails on the very first request (`step_counter=0`, `kv_cache_usage=0.038`),
and three runs reported byte-identical free/allocated figures.

The waste is that only one sequence is ever being prefilled
(`num_running_reqs=1`), yet the embedding is materialised for all 32 padded
batch rows -- 31/32 of that 10.7 GB is zeros. Ruled out as causes, each by
measurement:

| Suspect | Test | Result |
| --- | --- | --- |
| the prefill optimizations | rerun with `QWEN36_PREFILL_SCAN=hillis`, `QWEN36_SCAN_MATMUL_GRID=0` | identical OOM, same bytes |
| fabric / sampling mode | rerun with the sweep's `FABRIC_1D` + `decode_only` | identical OOM, same bytes |
| a regression on this branch | `PREFILL_STACK_CHUNK_SIZE` and `_prefill_forward_streaming` | unchanged since 2026-08-14 |

Fixes, cheapest first:

1. `--max-num-seqs 1` -- config only, allocation drops to 336 MB. This is what
   produced the numbers above.
2. `PREFILL_STACK_CHUNK_SIZE` 32768 -> 4096 (`tt/model.py:89`) -- 1.34 GB, keeps
   32-slot serving, costs more outer-loop iterations. The inner scan already
   works in 32-token units, so throughput should barely move.
3. Embed and prefill only the active rows -- removes the waste at any
   `max_num_seqs`. The proper fix.

**This affects CI.** The tt-inference-server spec sets `max_concurrency: 32`, so
any long-context benchmark or eval in a dispatched run will hit this. The ISL
128 graded point is unaffected. Not introduced by this branch: peak prefill
memory is `batch x 32768 x hidden` with nothing bounding the product, and the
constant only bites once ISL exceeds 32768, which is why no shorter-ISL test
caught it.

## Prefilling only the active row: 23.7x at the serving batch

The batch-32 OOM above was the visible symptom of a bigger waste. Fixing the
memory only bought the right to be slow: prefill then ran, and took ~4.9 h for
one ISL 65536 request. Both come from the same cause.

### What the waste is

`platform.py` disables chunked prefill for `model_type=qwen3_5`, so vLLM
prefills **one request per scheduler step**. The adapter builds `full_tokens` as
`[32, maxlen]` with one row real, runs the whole layer stack, and discards 31
rows at `logits[slots]`. Because every such call has identical shapes, the
program cache compiles **one** program and reuses it -- so this is one graph at
batch-32 shapes executed 32 times, not 32 batch-1 graphs. The waste is in
execution, not compilation.

Inactive rows were already *correct* before this change -- zero-length masks,
conv selectors that preserve state, a -1 cache page table -- just not free.

Decode is the opposite and always was fine: one batch-32 graph per token with
all 32 slots real. The 32-wide batch pays off in decode and is pure padding in
prefill, and that asymmetry is the whole bug.

### Measured

Full model, batch 32, ISL 128, one active slot:

| | full width | narrowed | |
| --- | --- | --- | --- |
| median prefill | 27060.0 ms | **1143.8 ms** | **23.66x** |
| 4-layer control | 2131.8 ms | 85.6 ms | 24.91x |

Two cross-checks: 27060 ms matches the 27051 ms full-model batch-32 TTFT
measured independently, and the narrowed 1143.8 ms lands within 1.5% of the
1127 ms a batch-**1** server pays. That is the ideal -- one request costs one
request whatever the slot count -- and it caps the achievable gain at ~24x here,
below the naive 32x because fixed per-call costs do not scale with batch.

Reproduce with `tests/prefill_active_row_speedup.py --batch 32 --length 128`.

### It hits benchmarks and evals both

| | client concurrency | prefills per run | each pays |
| --- | --- | --- | --- |
| benchmarks | 1 | 1 per point | 32 rows to fill 1 |
| evals | 32 | 32 sequential | 32 rows to fill 1 |

Evals default to `max_concurrent: 32` in `EvalTask`, clamped by
`eval_command.py` to the spec's `max_concurrency` (also 32). The neighbouring
`batch_size: 1` is a *client* knob, not the server batch: the eval class is
`local-completions` over HTTP, where a client batch would just fan out into more
parallel calls and make the effective parallelism unreadable, so the file pins
it to 1 and expresses all parallelism through `max_concurrent`. Concurrency 32
does **not** fill the 32 rows -- prefill is still one request per step.

Consequence for the graded ISL 128 point: TTFT 27060 ms -> 1144 ms against a
62 ms target.

This also kills the `max_concurrency: 1` workaround floated earlier in this
file: it would clamp *eval* concurrency to 1 as well, making evals ~32x slower
in the decode phase they actually parallelise.

### Why the recorded benchmark baselines are not comparable

A 32-slot server cannot reach ISL 65536 at all (it dies allocating
`32 x 32768 x 5120 x 2`), and its ISL 128 TTFT is 27060 ms, not the 3643 ms the
recorded sweep reports. Both facts fit one explanation: **the recorded sweep ran
against a 1-slot server.** `doc/vllm_rerun_on_new_base` reached the same
conclusion from the TTFT alone ("a ~4 s TTFT is what a batch-1-slot server
pays"); the OOM is independent evidence for it. The sweep never recorded its
server-side `max_num_seqs`, so this stays inference rather than fact -- but any
comparison against those numbers should assume a 1-slot baseline.

### How it is safe

Narrowing the batch is only sound if per-slot state still lands in the right
slot.

- **Full attention needs nothing.** `caches["key"]`/`["value"]` are a global
  block pool addressed by `page_table`, so a one-row page table plus
  `batch_indices=[0]` writes exactly that slot's blocks.
- **Linear attention** indexes `conv`/`recurrent` state by row, so
  `MultichipModel.single_slot_prefill_view(slot)` swaps in one-row copies and
  splices them back on exit.
- **Every decoder layer carries its own `batch`** and shapes its per-chunk
  reshapes from it, so the layers must be narrowed too, not just the model.
  Missing this fails loudly with a reshape volume mismatch, which is the good
  case -- it cannot corrupt anything silently.

The generator narrows `tokens`/`prompt_lens`/`batch`/`page_table` immediately
after validation, so positions, masks and selectors are built at batch 1 on the
host at no device cost. Logits are re-widened on both the host and
device-sampler paths, because callers address rows by absolute slot
(`generator_vllm` does `logits[slots]`).

`QWEN36_PREFILL_NARROW=0` restores the full-width path. It is the reference the
per-slot test compares against, and an escape hatch that needs no rebuild.

### Per-slot validation

`tests/prefill_active_row_pcc.py` sweeps **every** slot with a slot-dependent
prompt (`(arange*7 + slot*101 + 3) % vocab`) so a mis-routed write cannot
coincidentally match. Slot 0 alone would prove nothing: a narrowed run that
always wrote slot 0 would look perfect there and silently corrupt every other
user.

All 8 slots, length 96, 4 layers:

| metric | range |
| --- | --- |
| logits PCC | 0.99966 - 0.99978 |
| argmax match | **True, every slot** |
| linear state min PCC | 0.99857 - 0.99904 |
| paged KV state min PCC | 0.99943 - 0.99956 |
| **untouched neighbour** | **bit-exact, every slot** |

The neighbour check is the one that catches state landing in the wrong slot, and
it holds exactly.

**State is compared by PCC, not equality, and that criterion was loosened after
seeing results.** The first version of the test demanded `torch.equal` on
computed state and failed all 8 slots. That was wrong: narrowing the batch
changes matmul shapes and therefore accumulation order. Equality is kept only
for slots the prefill must not touch at all. Flagged because a bar moved after
seeing data is exactly where one gets quietly fitted to it -- the state bar sits
at 0.998 against a measured 0.99857 minimum.

**Recurrent state diverges more than the logits it produces** (0.9986 vs
0.9997), because it accumulates a per-step difference across the prompt and then
feeds decode. Harmless at 96 tokens and 4 layers; it is the number to watch at
65536 tokens and 64 layers, which is unmeasured. A generated-token comparison
over a long prompt would settle it.

### Chunk bound, kept

`_streaming_prefill_chunk_size` also gained a `batch` argument that divides the
token budget, holding peak embedding memory at what batch 1 already costs
(335.5 MB) for any batch. It is now largely redundant for the vLLM path, which
narrows to batch 1 and so gets the full 32768 chunk back, but it still protects
any caller that genuinely prefills a wide batch. Batch 1 is unchanged at 32768
and the legacy 2-arg call returns identical values, so the three pre-existing
contract assertions pass untouched.

Note it was **not** chosen as the largest chunk that fits: the allocator had
room for ~21760 at batch 32. Parity with a configuration already proven to work
was preferred over a maximum derived from the embedding term alone, which is
necessary but not sufficient -- the layer stack allocates more per chunk. A
tighter bound is solvable at startup from
`ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)`
(`num_banks`, `largest_contiguous_bytes_free_per_bank`), which is what a future
pipeline stage should do rather than hardcoding a constant.

### Regression

`regression_active_row.log`, all green, on the commit that adds this:

- 21 host contract tests
- dense-tap conv, 4 configurations: `DENSE_TAPS OK`, fused-vs-composite 0.9999997
- prefill PCC both scan modes: 0.99999 at b32/S128 and b1/S33
- decode, 6 configurations: PCC 1.0, times within 0.3% of recorded
  (single-chip 15.856/8.196 ms, multichip 2.376/0.793/0.628/0.509 ms)

Decode is provably untouched, which is the property this change must not
disturb.

## Reusing the CI image instead of rebuilding

A tt-inference-server change needs no rebuild: the test stage checks out
tt-shield and tt-inference-server fresh, and runs `python3 run.py
--override-docker-image <image>`. Only tt-metal is baked in, via
`--tt-metal-commit`.

The image from build job `101725713973` is

```
ghcr.io/tenstorrent/tt-agentic-bringup-qb2/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.21.0-<tt-metal-sha>-<hash>-<job-id>
```

`src-dev` means tt-metal *source* is present, so a bind mount can shadow it.
Pulling it needs `read:packages` on the gh token (`gh auth refresh -h github.com
-s read:packages`); the repo's own Actions token has it, a plain user token does
not, and the failure is a `403` on the manifest HEAD, after a successful
`docker login`.

**Verified against that exact image** -- same container, one bind mount the only
difference:

| | signature at `model.py:29` | md5 |
| --- | --- | --- |
| without mount | `(max_chunk, page_size)` | `a51137a2...` |
| with mount | `(max_chunk, page_size, batch: int = 1)` | `4deca343...` |

and the mounted code evaluated in-container: batch 1 -> 32768, batch 32 -> 1024,
legacy 2-arg -> 32768.

A first probe reported the fix present in **both** cases. That was a bug in the
probe -- it tested for the substring `"batch: int = 1"`, which occurs elsewhere
in `model.py`. The md5 and the signature line are the reliable discriminator.
Worth recording: a mount check that cannot distinguish mounted from unmounted
will happily report success.

`run_docker_server.py` already does exactly this for tt-inference-server's own
directories under `--dev-mode` (which *requires* `--override-docker-image`), so
"reuse an image, mount live source" is the intended workflow -- it simply does
not cover tt-metal. Extending it needs a sparse
`git clone --depth 1 --filter=blob:none` of one `models/autoports/<model>`
directory on the runner plus one `--mount`, driven by a new spec field, all
tt-inference-server-side. **Not yet implemented.**

Scope any such mount to `models/autoports/...` only, never the whole tt-metal
tree: the image carries compiled tt-metal, so this is sound **only for
Python-only deltas**. Mounting a tree whose C++ also changed pairs new Python
with older binaries, and that failure would be confusing rather than loud.
