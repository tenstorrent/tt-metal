# DiffusionGemma — vLLM serving contract (#47466 / #47488)

Status: current. **Deliberately over the 180-line cap** (~400): this file absorbs six deleted files
(1466 lines) and every line that pushed it over is a refutation, an open contradiction, a
measurement trap, a current default, or a reproduction path/artifact — the six categories that are
never cut for length.
Owns: launch flags, the fresh-host vLLM install recipe, serving flag defaults, the block-granular
emission contract, per-block metric semantics, the served-context ceiling + admission stall, the
prefill-pad fix, and the live-serving evidence from the deleted `work_log.md`,
`vllm_speed_by_context.md` and the four deleted dated sweep files.
See also: [refuted list](../REFUTED.md) · [vLLM-native plan](vllm_native_plan.md) ·
[traced serving](traced_serving.md) · [PR #47488](PR_47488.md) · [plan](../../plan.md)

## Execution modes

1. **Model-lifetime Metal trace** — one trace/controller captured at startup with reveal masking,
   device Gumbel, K=48, window-1 early halt. An **unset** `DG_UPFRONT_CAPTURE` selects this.
2. **Eager fallback** — `DG_UPFRONT_CAPTURE=0`, set *explicitly*; unsetting it does not disable
   capture. Eager is the only path emitting per-step trajectory records (a replayed trace does not),
   and is a diagnostic, not traced-serving evidence.

Trace hazards and the up-front capture contract: [optimize_perf hub](../optimize_perf/README.md).

```bash
# env: see plan.md
export DG_UPFRONT_CAPTURE=1                                                  # default ON; explicit
export DG_UPFRONT_PREFILL_WARMUP_LENS=<all-admitted-aligned-prompt-lengths>  # required, no default
export DG_TRACE_REGION_SIZE=<validated-positive-reservation>                 # required, no default
export DG_DENOISE_REVEAL_PMAX=<positive-tile-aligned-served-cap>             # optional
export DG_VLLM_GUMBEL_MODE=device                                            # default
export DG_DENOISE_SLIDING_WINDOW=1                                           # default ON since 07-27
```

Both required knobs are fail-loud by design. The prefill shape list cannot be derived from anything
the wrapper knows — every aligned prefill length the server can admit must be listed and compiled
before denoise capture, and an unseen runtime shape fails instead of compiling with resident traces.
The trace region cannot be read back (Metal takes it as an open-time constructor argument with no
getter), so a default would silence the guard without reserving anything, and a trace-region
overflow poisons the device (`tt-smi -r`). Reserve it with `--additional-config
tt.trace_region_size` and mirror the identical value in `DG_TRACE_REGION_SIZE`.
`DG_DENOISE_REVEAL_PMAX` unset becomes the tile-rounded `--max-model-len` and is logged; both paths
share one validation — positive, tile-aligned, fits prompt + one canvas, within the allocated KV span.

## Server command and launch-flag requirements

```bash
# env: see plan.md
python -m vllm.entrypoints.openai.api_server \
  --model <checkpoint> --served-model-name diffusiongemma-26B-A4B-it \
  --generation-config vllm --max-model-len <served-limit> \
  --max-num-batched-tokens <at-least-largest-whole-prompt> \
  --max-num-seqs 1 --block-size 64 \
  --additional-config '{"tt": {"sample_on_device_mode": "all", "enable_model_warmup": false}}'
```

- `--block-size 64` is **required**: `get_num_available_blocks_tt` multiplies by
  `cache_config.block_size`, which vLLM leaves `None` for this arch, failing at KV-cache init with
  `TypeError: unsupported operand type(s) for *: 'NoneType' and 'int'`.
- Without `--generation-config vllm` the checkpoint overrides `max_tokens` to 256, so a 1024-token
  request emits block 0 only and never calls `decode_forward`.
- `--max-num-batched-tokens` must cover the whole prompt: the TT scheduler provides no
  chunked-prefill admission, so an over-budget prompt sits in `Waiting` with no model execution (the
  2026-07-10 3072-token prefill needed `--max-num-batched-tokens 4096` for exactly this).
- `enable_model_warmup: false` skips the AR two-phase trace warmup; block diffusion warms lazily.
- `VLLM_ENABLE_V1_MULTIPROCESSING=0` gives single-process V1 so tracebacks surface in the log.
- Request `temperature` / `top_p` / `top_k` / seed are ignored by the DG adapter; process-level DG
  sampling config is authoritative.
- `--max-num-seqs 1`: one contiguous model cache backs one active sequence. Concurrency needs #47488
  paged-cache ownership + #47557 batched canvas decode ([plan](vllm_native_plan.md)).
- Datatype served: bf16 weights + bf16 KV + bf16 CCL, self-conditioning softmax / soft-embedding in
  fp32, matching the gemma4 vLLM bridge. bfp8 experts were measured and rejected:
  [datatype sweep](../datatype_sweep/README.md).
- `model_capabilities`: `supports_prefix_caching=False`, `supports_async_decode=False`,
  `supports_sample_on_device=True`.

## Serving flag defaults

| flag | default | why |
|---|---|---|
| `DG_UPFRONT_CAPTURE` | ON (unset) | see execution modes |
| `DG_DENOISE_SLIDING_WINDOW` | 1 since 2026-07-27 (#51080) | HF sliding layers retain only `sliding_window - 1` = 1023 committed tokens and 25 of 30 layers are sliding |
| `DG_DENOISE_HIDE_PREFILL_PADS` | ON since 2026-07-29 | the canvas otherwise attends up to 31 prefill pad keys |
| `DG_VLLM_GUMBEL_MODE` | `device` | ~53.6 vs ~36.3 tokens/block/s against the deleted `host` (~1.48x) |
| `DG_PREFILL_RAGGED_LONG` | ON | prompts above 4096 use 4096-token ragged top-8 slices |

`chunked` and `argmax` Gumbel are not materialized full-tensor sources and are rejected by the
up-front controller, leaving `device` as **the only mode valid under up-front capture** (the adapter
still accepts the other two under `DG_UPFRONT_CAPTURE=0`). The device default depends on the
Blackhole PRNG kernel fix and is pinned by
`tests/ttnn/nightly/unit_tests/operations/rand/test_rand_independence.py`; if that gate regresses,
revisit it. Mechanism and the deleted `host` arm:
[device Gumbel restored](../decision_fidelity/device_gumbel_restored.md).

Below a 1023-token committed prefix the sliding retention mask is bit-identical, so that change is
confined to the regime where TT was wrong. Its GPQA-Diamond gate repaired **52** of the 64 collapsed
questions and regressed **1** of 67 clean ones, moving TT from **58/131** to **86/131** against the
CUDA reference's 97/131; it is also a throughput fix (blocks that halt 72% → 99%, per-block latency
0.652x ≈ 1.53x faster), because a block that cannot settle burns all 48 denoise steps and still
commits an unsettled canvas. Evidence:
[device Gumbel restored](../decision_fidelity/device_gumbel_restored.md) sections 10, 17, 19; gate
scripts under `doc/decision_fidelity/gate/`. The bounded sliding read composes with hidden pads
since `a25adba5260` because its key axis already carries absolute positions; it is unconditional and
`DG_DENOISE_SLIDING_SPAN` no longer exists.

Removed knobs still visible in dated artifacts do nothing — see
[flag triage](../optimize_perf/flag_triage_20260728.md). No deleted flag is a live knob here.

## Metric semantics, measurement traps, evidence hygiene

- **METRIC RULE:** report physical per-block metrics from `DG_VLLM_METRIC` (prefill, TTFT, denoise
  steps/latency, commit, block latency, `256 / block_latency`). **Never `1000/mean_tpot_ms`** —
  there is no per-token TPOT in block diffusion. Pure prefill, serving prefill and DG "TTFT" are
  three different metrics: [three-metrics rule](../optimize_perf/README.md).
- **TRAP:** API `completion_tokens / wall_time` includes EOS trimming and queueing; with
  `max_num_seqs=1` a curl wall time may contain a previous request. Not a device measurement.
- **TRAP:** never use `ignore_eos=true` for qualitative judgment — it deliberately surfaces the
  physical 256-token canvas tail after EOS.
- **TRAP:** GPQA needs three denominators and a `prefill_block0` count check against the question
  count: [GPQA traps](../decision_fidelity/README.md).
- **EVIDENCE INTEGRITY (2026-07-10 sweeps):** recompute every referenced log digest from disk before
  trusting a result (`OK: recomputed 11 step-log hashes and 1 warmed-context log hash`;
  warmed-context digest `757ce2d3af346f48fd33078215b7ab1845bdf741f4c9ef00e80668ce088a895b`), require
  the log to end with the FastAPI `Application shutdown complete.` marker, and scan afterwards for
  leftover vLLM / EngineCore / sweep / `tt-smi` processes.
- **EXCLUDED-FAULT RULE:** a run killed by hardware is classified `excluded_hardware_fault`, its log
  and failed JSON are preserved, it is excluded from **all** performance aggregates, and exactly one
  retry is permitted after a bounded no-process check → `timeout 60 tt-smi -ls --local` →
  `timeout 180 tt-smi -r` → list again → a passing `MeshShape(1, 4)` open/close smoke.
- **NOISE FLOOR:** the same 256-token K=48 config measured 18.246764 t/s inherited and 18.276320 t/s
  isolated — +0.162%, immaterial run-to-run variation.
- Session host gates: touched-DG host pytest **276 passed / 1 skipped in 88.70 s** (the skip is the
  opt-in `DG_RUN_DEVICE=1` serving smoke), focused tt-vllm plugin pytest **13 passed in 4.64 s**,
  reverse-apply check for the runner/scheduler/host-test patches, `git diff --check` in **both**
  repos. `ruff` was unavailable there, so py_compile + Black + focused tests + JSON parsing + patch
  checks stood in; a nanobind leak warning after clean shutdown is benign.
- No-shared-edits gate ([rule](../../AGENTS.md)): `git status --porcelain -- models/demos/gemma4/`
  must be empty, and the `git diff main -- models/demos/gemma4/` delta is pre-existing branch state
  (main advanced with #47817/#47556/#47172), not this stage. Scripted form uses
  `DG_BASE_REF=origin/diffusion-gemma-function` → `OK: no shared-directory edits ...`.

## Served context — an open contradiction

> **OPEN CONTRADICTION (unexplained):** the served-context ceiling is answered three incompatible
> ways. (1) This file's context contract: HF advertises **262144**
> (`text_config.max_position_embeddings`, 256x1024) but that is **not** a validated live-vLLM served
> ceiling. (2) The deleted `doc/vllm_integration/work_log.md`: `max_model_len = 262144` with **no
> reduction below the advertised context**, citing a standalone full-depth 256K weights+KV fit at
> 29.704 GiB/chip used / 2.163 GiB free. (3) The deleted
> `live_context_sweep_256_to_256k_20260714.md`: at `max_model_len=262144` the KV cache is ~15 GiB
> which with 13.25 GiB bf16 weights plus a trace region overflows 32 GB/chip — **two confirmed
> build-time OOMs in `init_kv_cache`** — giving a **~128K practical traced ceiling** (that run used
> `max_model_len=131072`, which builds fine). A fourth reading is in
> [QB2 memory budget](../../QB2_MEMORY_BUDGET.md). Not explained.

Facts that survive whichever reading holds: full-vocab Gumbel materialization OOMs at 256K; reaching
256K would need bf8 weights (~6.5 GiB), or dropping traced denoise, or expert parallelism; the
largest allocated context actually probed is **32768** and the largest real prompt measured is
**16384 tokens**, which establishes 32768 as passing and does **not** establish the absolute ceiling.
Record the exact tested `--max-model-len`; never convert a standalone fit into a serving claim. Any
valid prompt length within the tested limit is accepted — the 256-token *output* block granularity is
not an input constraint, and prefill pads to a 32-tile multiple internally.

### The admission stall (root-caused and fixed, 2026-07-14)

A 32768-token request emitted no block markers and sat at `Running: 0, Waiting: 1, GPU KV cache
0.0%` for a full hour: no eth-core hang, no OOM, no compute. **REFUTED first guess:** a
"kernel-compile storm" — `Waiting` plus 0.0% is a *scheduler admission* failure, not a compute stall.
Root cause: `get_kv_cache_spec` emitted a `SlidingWindowSpec` for the 25 sliding layers, so vLLM
split the KV cache into **6 groups** (1 full + 5 sliding) sharing one block pool; a whole-prompt
single-shot prefill allocates `cdiv(L/64)` blocks in *every* group (sliding skipping is 0 on the
first chunk), so demand is `6*cdiv(L/64)`. With `num_gpu_blocks=2049` (2048 free, sized for one
group) admission caps at `(2048//6)*64 = 21824` tokens: 16384 needs 1536 blocks and admits, 32768
needs 3072 and `allocate_slots` returns `None` → `break` → permanent `WAITING`.
`--num-gpu-blocks-override` is clobbered by the plugin, so this was a code fix, not config.

**Fix (verified present in `tt/generator_vllm.py` today):** emit `FullAttentionSpec` for the sliding
layers too, merging all layers into ONE group backed by the whole pool (demand `cdiv(L/64)`).
Memory-neutral — the model owns the physical KV and the spec is vLLM bookkeeping — and consistent
with the inherited `_HYBRID_KV_CACHE_GROUPS_ENABLED = False`. Evidence
`verify_32k_admission_20260714.json` at `max_model_len=65536` (old cap 10880): 16384 prefill 5.30 s,
32768 prefill 11.96 s, matching the ~10.8 s standalone. Related: at `--max-model-len 1024` the
server reports `GPU KV cache size: 21,824 tokens`.

## The prefill-pad fix (`DG_DENOISE_HIDE_PREFILL_PADS`, ON since 2026-07-29)

Prefill right-pads the prompt to a tile multiple and writes K/V for the pad tokens while the reveal
predicate uses the *padded* length, so the canvas attends up to 31 garbage keys immediately before
itself — destroying the thinking-template prefix at canvas positions 0–4, the entire accept budget
block 0 bootstraps from. Same 11 prompts (those that drifted on TT, plus clean controls) at the same
5632-token budget on both platforms, so neither platform nor budget is confounded:

| | drift | guard kills | empty | mean chars |
|---|---|---|---|---|
| A100 reference | **0/11** | — | 0 | 11069 |
| TT, pads attended | **5/9 non-empty** | 3 | 2 | 2507 |
| TT, pads hidden | **0/11** | **0** | **0** | **8514** |

The **length** recovery (2507 → 8514 chars, 24% → 77% of the reference) is what shows causation
rather than coincidence; the A100 does not pad at all, which is why it never drifts. Clearest case,
`doc 8`: 0.41 CJK across 2181 chars with 43 Latin words becomes clean English, 12902 chars, 1746
words. On device, hiding the pads fixed **7 of 7** block-0 collapses, and injecting the same padding
geometry into the HF reference reproduces the failure there. Accuracy on those 11 (lm_eval
flexible-extract): TT pads hidden **8/11 = 72.7%**, A100 @5632 **7/11 = 63.6%**, TT pads attended
**5/11 = 45.5%**, reference rep2 @126976 **8/11 = 72.7%**. Per question the fix is one-directional —
three go wrong→right (two were the empty replies), **none** go right→wrong.

- **TRAP:** n=11 means one question is 9 pp. The table shows **no regression**; it does **not** show
  superiority.
- **TRAP:** pads-attended's *lower* 15.4 mean denoise steps was a bad sign, not a good one — it
  converged fast because it converged to a short, poisoned block. With pads hidden the steps are
  healthy at mean 18.3 / median 16, 1% of blocks reaching the 48-step cap, in line with `cot_rerun`'s
  20.0 mean over 2119 blocks.
- **VOID MEASUREMENT:** the 2026-07-28 revert read "repairs 3, breaks 28" off a paired 44-prompt
  comparison. Both arms ran on the token-gather denoise MoE, whose entropy plateaus at ~0.46 against
  the 0.005 halt threshold so the early halt never fires; that path was deleted in `7417bd7d69d`.
- **VOID WINDOW:** any absolute degeneracy or drift rate recorded between 2026-07-28 ~21:25 and the
  concat MoE default flip is void for the same reason.
- **STILL OWED (open):** the full 198-question GPQA score against the 70.71% / 70.20% reference bar.
  The 11 prompts were selected *for* drift, so their absolute value says nothing about the bar.
- **STILL OWED (open):** the clean-question mechanism arm `gate/padfix_regression_arm.sh`, unfinished
  at 3 of 30 pairs.

> **OPEN CONTRADICTION (unexplained):** this file credits the pad fix to commit `d0936d4da4f` while
> also recording that the 2026-07-28 flip was reverted the same day on a void measurement;
> [decision fidelity](../decision_fidelity/README.md) credits `205e87956cc`. Both kept. Not explained.

## Live serving evidence

**Eager control (pre-trace):** ~7.1–7.4 output tok/s, **flat across context** — block latency 34–37 s
and TTFT 35–37 s at prompt_len 10 / 61 / 265 (7.29 / 7.07 / 7.43 t/s), all blocks running the full 48
steps with `halted=False`. Because block latency is independent of `prompt_len`, the per-block cost is
MoE-compute bound (48 steps x 30 layers), not attention or prefix bound. This is ~6.8x over the
original dense-128 baseline of ~1.08 t/s, and is the control the traced path (2.61x on top) is
measured against: [traced serving](traced_serving.md).

**2026-07-10 warmed context sweep** (real `tenstorrent/vllm` `/v1/completions`, msl=4096, three timed
requests per target, nine steady blocks each, zero first-use compile markers): 32 / 256 / 1024 / 2048
logical tokens → **18.495 / 18.270 / 17.571 / 16.722 output tok/s**; median prefill 0.203 / 2.969 /
15.070 / 31.686 s; median block-0 TTFT 130.840 / 135.619 / 148.964 / 166.759 s. The 3072 warmed rerun
was intentionally omitted at the priority handoff; its raw file is labelled `interrupted`, not
`running`. Allocation-only scaling is flat — a fixed 32-token prompt measured 18.826 / 18.852 /
18.892 / 18.850 t/s at msl 4096 / 8192 / 16384 / 32768 — while real 6144 / 8192 / 16384-token prompts
measured **12.681 / 11.884 / 9.489 t/s** with trace-resident DRAM 15.158 / 15.675 / 16.710 GiB/chip.
The frozen-prefix *read* is material at real long context; allocation alone is not.

**2026-07-10 denoise-step cap sweep** (256-token context, one isolated server per budget, four blocks):
K=1/4/8/12/16/20/24/32/40/48 → **166.800 / 108.281 / 72.936 / 54.877 / 44.458 / 37.063 / 31.998 /
25.538 / 21.337 / 18.276 output tok/s**, i.e. 9.127x down to 1.000x versus K=48 — well below the ideal
48/K ratio, because commit and other fixed block work do not shrink. **Performance-only:** K=48 stays
the model-faithful budget under #48291 and smaller caps can change diffusion decisions and quality.

**Trace-proof contract** asserted for every request in both 2026-07-10 sweeps: block 0 emits one
capture event with exactly K distinct Metal trace IDs and executes them once; later blocks replay the
same IDs (`4*K` total / `3*K` steady for the K sweep; 192 / 144 for the 48-trace context sweep); no
eager fallback, no recapture after block 0; prefill and commit stay outside the denoise trace;
completion emits the trace-release marker before vLLM removes the row; zero `Building trisc` markers.
**REJECTED DENSE CONTROL:** before the explicit performance stack the traced path measured 1.225 t/s
at 32 prompt tokens and 1.218 at 256; retained only as `rejected_dense_control`.

**2026-07-14 ragged-prefill sweep.** Chunked ragged prefill (default on) made real vLLM prefill
**20–25x faster** past the old 4096 cap: 1024-token prompt 15.07 → **0.60 s** (25x), 16384-token
prompt ~270 → **13.52 s** (20x); 256 → 0.65 s, 4096 → 2.36 s. DRAM stayed ~22.9 GiB used of 27.9
usable through 16K. The >4096 prefill cliff is gone in the serving path. Prefill authority:
[chunked ragged prefill](../optimize_perf/chunked_ragged_prefill.md).

Same run, **generation regression, root-caused:** steady generation fell to ~3.6 output tok/s (from
~18) at 71–84 s/block because every block recaptures the Metal trace (`recapture_after_block0=true`,
`steady_replay=false`). Commit `ec5b64b4891` grows the cross-attention prefix by 256 per block and
added a `prompt_len`-keyed trace-invalidation guard in `traced_denoise.py`; the denoise mask shape
depends on `prompt_len` (`tt/denoise_forward.py`), so the trace invalidates every block. **Do NOT
revert** — it is a genuine correctness fix (the old replay left committed tokens invisible to later
blocks) delivered suboptimally. Recorded remedy: capture once against a fixed max-context mask and
feed the growing prefix as a written replay input (the KV-decode pattern). Priority follow-ups:
restore denoise trace replay (~5x, the biggest serving win) and bf8 weights for >128K context.
Harness changes that run: the trace-region guard was relaxed to [2 GiB, 10 GiB] because long context
needs the DRAM back, and the strict capture-once trace-event assertion was softened to *record*
actual capture/replay/release counts (timing still comes from `DG_VLLM_METRIC` markers).

**TRAP:** the 2026-07-10 and 2026-07-13 same-ID cross-block rows held the denoise prefix at the
initial *prompt* length — same-shape performance provenance, not current default-server TTFT, not
growing-prefix multi-block throughput, not evidence for an implicit vLLM launch. Run conditions for
both dated sweeps: `--generation-config vllm`, `--max-num-seqs 1`, `--block-size 64`, on-device
sampling, temperature 0, ignored EOS, 48 denoise steps/block.

**The live context-sweep executable was REMOVED, not fixed**, because it accepted arbitrary prompt
text whose aligned token length was known only after server startup — which cannot satisfy the
up-front prefill warmup contract. Those measurements are therefore not re-runnable as written; the
artifacts survive: `live_context_sweep_results_20260710.json`,
`live_denoise_step_sweep_results_20260710.json`,
`live_denoise_step_k{01,04,08,12,16,20,24,32,40,48}_20260710.json` +
`live_denoise_step_k04_20260710_retry.json` (the compact JSON records each server log's SHA-256),
`live_context_sweep_256_to_128k_20260714.json` (filename says 128k, its doc title said 256k),
`live_context_sweep_256_to_256k_20260714.json`, `verify_32k_admission_20260714.json`,
`live_vllm_serving.json`, `serving_test_suite.json`, `vllmtraced_msl{4096,32768}.json`.

## Architecture and the block-granular contract

The whole denoise loop lives inside the model forward. `tt/generator_vllm.py`
(`DiffusionGemmaForCausalLM(HybridAttentionForCausalLM)`) is a thin interface over the vLLM-free
block-emission core `tt/serving.py` (`BlockDiffusionServingSession`), which delegates to
`tt/generate.py`:

```
prefill_forward -> session.prefill -> generate.prefill_prompt_tokens (prompt K/V, causal)
                                   -> make_generation_logits_fn_builder_from_checkpoint_state
                -> session.decode_block (block 0) -> generate.denoise_and_commit_block
                     -> denoise_loop.denoise_block  (<=48 steps: sampling.gumbel_max /
                        token_entropy -> entropy_budget_accept -> renoise)
                     -> generate.commit_canvas_tokens
decode_forward  -> one session.decode_block per active request (block N)
```

Other delegated `tt/generate.py` methods: `decode_generation`, `tokenize_prompt`,
`make_seeded_host_canvas_init_fn`, `make_seeded_host_noise_tokens_fn`,
`make_seeded_{host,chunked,}_gumbel_noise_fn`. The adapter emits a **256-token block per decode step** (`canvas_length`) and position advances by
`canvas_length` (`next_pos += 256`). Only per-step `[B,L]` decision tensors (argmax / entropy /
sampled / accept) are read back for the data-dependent halt; `[B,L,vocab]` logits never leave the
device. The async-decode contract is redefined **per-block** — `supports_async_decode`, stale-input
refresh and page-table update happen once per emitted block, not per token — and is declared `False`
because the per-block async path is unproven without the #47488 runner. Cache ownership and the
three-phase KV machine: [vLLM-native plan](vllm_native_plan.md). Early-halt criterion and its
firing-status contradiction: [early halt](../optimize_perf/early_halt.md).

## Registration, fork patches, install

> **Migrating off the fork:** the serving work is being moved to the standalone
> `tenstorrent/vllm-tt-plugin` on upstream vLLM 0.24.0. The port, the four 0.24 API breaks it
> exposed, and a paired device smoke are in
> [`plugin_migration_024.md`](plugin_migration_024.md). Read its Break 5 before comparing any
> plugin GPQA score against a fork number.


HF arch `DiffusionGemmaForBlockDiffusion` → the plugin auto-prefixes `TT` →
`TTDiffusionGemmaForBlockDiffusion`, registered via `_register_model_if_missing` in
`register_tt_models()`. The fork is not vendored in tt-metal, so the exact edit is saved as
`doc/vllm_integration/plugin_registration.patch`. Three fork patches must be applied to
`/home/zni/tt-vllm` — `plugin_47488_registration.patch`, `plugin_47488_model_runner.patch`,
`plugin_47488_scheduler.patch` — plus the `tt/generator_vllm.py` `stop_token_ids=[]` session-stop
deferral. Rationale: [PR #47488](PR_47488.md).

Fresh-host install (no container): venv `/home/zni/venvs/tt-diffusion-gemma` (Python 3.12.12, `ttnn`
editable from `/home/zni/tt-metal`, transformers 5.12.1, torch 2.11.0+cpu); fork `/home/zni/tt-vllm`
branch `dev` head `6b4a3a7`; checkpoint `/home/zni/dg_models/diffusiongemma-26B-A4B-it`;
`MESH_DEVICE=P150x4`.

```bash
# env: see plan.md
source /home/zni/venvs/tt-diffusion-gemma/bin/activate && cd /home/zni/tt-vllm
VLLM_TARGET_DEVICE=empty uv pip install -e . \
  --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
uv pip install -e plugins/vllm-tt-plugin
```

`requirements/common.txt` pins **no** torch, so the venv's torch / transformers / ttnn survive; build
isolation pulls torch 2.10 into a throwaway env only. Result `vllm 0.1.dev1+g6b4a3a7b4.empty` +
`vllm-tt-plugin 0.0.0`, `platform_plugin() -> vllm_tt_plugin.platform.TTPlatform`. **Never use the
stale ghcr image `0.14.0-80180b9-7678b70`** — its baked tt-metal predates
`models/experimental/diffusion_gemma`. `models/common/readiness_check/run_vllm_server` does not exist
in this checkout (nor does its `check_degenerate_output.py` gate), so the live serve is driven
directly against `vllm.entrypoints.openai.api_server`.

## Reproduction

```bash
# env: see plan.md
python -m models.experimental.diffusion_gemma.demo.serving_smoke \
  --mesh P150x4 --num-layers 1 --max-seq-len 1024 --num-blocks 2 --canvas-length 256 \
  --max-denoising-steps 2 --gumbel-mode argmax --local-files-only \
  --metrics-json doc/vllm_integration/serving_smoke_reduced.json
DG_RUN_DEVICE=1 python -m pytest models/experimental/diffusion_gemma/tests/test_serving_block_contract.py -q
```

`serving_smoke` baselines (non-256-aligned prompts, `(1,4)` mesh, canvas 256): reduced 1-layer /
2 steps → TTFT 8.24 s, 6.89 s/block, 37.1 tok/block/s, 32→544, DRAM 0.0 → 1.202 → 2.616 GiB/chip;
full-depth 4 steps EOS-stop → TTFT 65.98 s, 64.45 s/block, 3.97 tok/block/s, post-build DRAM 13.268
GiB/chip; full-depth 16 steps no-EOS-stop → TTFT 111.61 s, 110.20 s/block, 2.32 tok/block/s, halted at
step 11. Artifacts `serving_smoke_{reduced,fulldepth,fulldepth_visible}.json`.

Live suite (QB2, 2026-07-03): 7 real OpenAI requests — 4 `/v1/completions` + 2
`/v1/chat/completions` + the 2-block serve — with non-256-aligned prompt lengths 6, 5, 25, 31, 21,
every one HTTP 200; `DG_RUN_DEVICE=1 pytest tests/test_serving_block_contract.py` → 7 passed. The live
2-block serve is `prefill block0 32→288 (35 steps, 178.2 s)` then `decode block=1 288→544 (48 steps,
232.8 s, stop=False)`. Qualitative controls: the visible-dialogue RUN control output
`你好！I'm doing well, thank you for asking. How can I help you today?` and a live chat request
producing `The vast blue expanse holds endless secrets beneath its rolling waves.` — so the adapter is
not a serving regression. Adapter commits `faebfbcc358` (block-granular serving adapter) and
`4d320be2615` (KV-cache ownership tightening post-review); a fresh independent xhigh stage review
(2026-07-03) returned clean-pass.

## Device hygiene and open items

- Serialize every device run with `flock /tmp/dg-mesh.lock`. No Tracy / tt-perf-report /
  device-profiler collection is done in the vLLM stage, per skill.
- An abruptly killed EngineCore leaves ethernet core 29-25 un-reset (`TT_THROW ...
  assert_active_ethernet_cores_to_reset`; `Device 0: Timed out while waiting for active ethernet core
  29-25 to become active again`, `llrt.cpp:581`). `tt-smi -r` plus a `(1,4)` mesh smoke recovers it;
  graceful SIGTERM avoids it. One session needed 3 bounded recoveries, each succeeding first try.
- **WORKTREE TRAP:** a git worktree (`/home/zni/tt-metal-apc`, `/home/zni/tt-metal-chunk`) has no
  built runtime, so `TT_METAL_HOME` must point at the built `/home/zni/tt-metal` while `PYTHONPATH`
  points at the worktree. `TT_METAL_HOME=$PWD` fails the kernel link with a missing
  `runtime/hw/toolchain/blackhole/*.ld` / `firmware_brisc.ld`. `ttnn` resolves from the editable
  install either way.
- **OPEN:** concurrent batched multi-sequence serving = #47488 paged-cache ownership + #47557 batched
  canvas decode.
- **OPEN:** the adapter-class execution-coverage gap. The 2026-07-10 live sweep exercised
  `initialize_vllm_model`, `get_kv_cache_spec`, `allocate_kv_cache*`, `prefill_forward` and
  `decode_forward` through real OpenAI requests, but its harness has since been removed, so that
  coverage cannot be re-established by running anything today.
- Text quality is bounded by the bf16 diffusion floor, not by serving:
  [decision fidelity](../decision_fidelity/README.md).
