# DiffusionGemma 26B-A4B-it — tt-metal bring-up plan, spec & status

Single source of truth for the DiffusionGemma bring-up branch. **This file merges the former `plan.md`, `DEVICE_LOOP.md`, and `DEVICE_LOOP_W2B.md`** (and the earlier `STATUS.md`, which had already been folded into `DEVICE_LOOP.md`) into one document.

- **Tracking issue:** [tenstorrent/tt-metal#47452](https://github.com/tenstorrent/tt-metal/issues/47452) (label `DiffusionGemma`)
- **Work branch:** `diffusion-gemma-function` (current; earlier work on `zni/diffusion-gemma-bringup`)
- **New module root:** `models/experimental/diffusion_gemma/`
- **Companion docs (kept separate):** [`AGENTS.md`](./AGENTS.md) — terse working context for agents; [`QB2_MEMORY_BUDGET.md`](./QB2_MEMORY_BUDGET.md) — QB2 memory budget & batch ceiling (#47487).

> **TL;DR.** The text backbone is identical to the in-repo **Gemma-4 26B-A4B MoE** (`models/demos/gemma4/`). The real work is the **generation procedure**, not the backbone: a block-autoregressive multi-canvas *text-diffusion* loop with **bidirectional canvas attention**, a **three-phase KV-cache state machine**, **entropy-budget acceptance sampling**, and **self-conditioning**. Bring up text-first on QB2.

**How this document is organized:**
- **Part 0 — Current execution contract + dated status history** — the 2026-07-17 launch,
  metric, quality, and prefill contract is authoritative; older RUN-first/two-gap narratives are
  retained below as history. **Read first.**
- **Part I — The plan** — goals, model summary, reuse-vs-build, milestones, the issue-level dependency graph, per-issue workstreams, validation strategy, the risk register, serving notes, and references.
- **Part II — Device execution spec** — the `/loop` protocol, env/run recipe, ground rules, the decision-fidelity bar, and the W1–W4 device workstream specs + acceptance.
- **Part III — Implementation status** — environment constraints, the per-workstream status table, session notes, the 2026-06-26 / 06-29 / 06-30 code reviews + fix verification, and build order.
- **Appendix A — W2b long-prompt attention (RESOLVED)** — the archived spike-first plan and status log for the >32768 non-causal denoise path.

> **Note on overlap:** Part I §5/§6 (the issue-level dependency graph + workstreams) and Part II (device execution specs W1–W4) describe the same work at different altitudes — the plan vs. the device loop. Part 0 and Part III hold the live status; Parts I–II are the standing plan/spec.

---

## Part 0 — Current status & roadmap
### Current execution contract (2026-07-17 — authoritative)

Read this section before using any older status table or benchmark below. Dated July-02/10/13
sections are retained as history and do not define the current serving launch or performance
baseline.

**vLLM launch requirements**

- The TT adapter defaults are conservative, not performant: `DG_SPARSE_MOE` defaults OFF,
  `DG_DEDUP_ARGMAX` defaults OFF, `DG_VLLM_TRACE` defaults OFF, and
  `DG_VLLM_MAX_DENOISE_STEPS` defaults to the released K=48 budget. A server that does not set
  these explicitly runs dense-128 eager denoise.
- Always pass `--generation-config vllm` for requests beyond one 256-token canvas. Without it,
  checkpoint `generation_config.json` overrides `max_tokens` to 256; `max_tokens=1024` still emits
  block 0 only and no `decode_forward` rows.
- Set `--max-num-batched-tokens` at least as large as the largest whole prompt scheduled by the TT
  backend. Chunked prefill is not a scheduler feature on this path; otherwise long requests can sit
  in `Waiting` without entering the model.
- `ignore_eos=true` is a transport stress control only. It exposes all physical canvas tokens,
  including the tail after the first EOS; do not use that tail for a quality verdict.
- HTTP `temperature`, `top_p`, `top_k`, and per-request seed are not currently consumed by the
  DiffusionGemma denoise loop. Sampling mode/step budget are process-level DG settings.

**Use an explicit profile**

- Correctness / production-sampler control: `DG_SPARSE_MOE=1`,
  `DG_SPARSE_MOE_TUNED=1`, `DG_VLLM_GUMBEL_MODE=chunked`, K=48. Keep normal EOS handling.
- Fast functional/transport control: `DG_SPARSE_MOE=1`, `DG_SPARSE_MOE_TUNED=1`,
  `DG_DEDUP_ARGMAX=1`, `DG_VLLM_GUMBEL_MODE=argmax`, and an explicitly labeled reduced K such as
  12. This is not the production-sampler quality gate.
- Trace benchmark only: additionally set `DG_VLLM_TRACE=1` and a validated
  `DG_TRACE_REGION_SIZE`. Report capture-inclusive block 0 separately. Growing contiguous-prefix
  shapes still require recapture; historical same-ID cross-block rows used a prompt-only prefix.

**Metric contract**

- One model step emits a physical 256-token block. Report `prefill_s`, prefill+block-0 TTFT,
  `denoise_steps`, denoise/commit/block latency, and `256 / block_latency` output tok/s.
- API-visible `completion_tokens / wall_time` depends on EOS trimming and queueing and is not a
  device throughput metric. With `max_num_seqs=1`, curl wall time may include another request.
- The July-10 18 tok/s tables are historical warmed same-shape trace replay, not first-request TTFT
  and not current growing-prefix multi-block throughput.

**Current qualitative interpretation**

- The July-15 fp32/bf16 control shows TT can produce coherent prompt-correct output at the intrinsic
  bf16 diffusion floor. Persistent garbage from a server launch is therefore a configuration or
  serving regression to investigate, not an expected blanket consequence of #48291.

**Prefill measurement contract**

- Distinguish pure `prefill_prompt_tokens` timing from serving TTFT and from
  `BlockDiffusionServingSession.prefill`, which also constructs generation state.
- `tt/prefill_moe.py` defaults `DG_PREFILL_RAGGED_LONG=1`: every multi-token prefill uses ragged
  top-8 expert execution, and sequences above 4096 are processed in 4096-token slices. The
  4K→16K dense-MoE cliff belongs to the pre-fix `ec5b64b4891` control only. Current pure-prefill
  evidence is `context_window_prefill_only_chunkedlong_20260713_msl65536.json` at `233b88276ab`.

> **Priority (2026-07-02): RUN-first — get the loop emitting text end-to-end on hardware; `#48291` correctness is explicitly deferred.** The near-term goal is a prompt→text device run at small scale, *accepting degenerate output* (EOS-heavy / garbage): the bf16/MoE backbone only ~50% argmax-agrees with HF and diffusion commits the clean argmax, so text quality is *known* to be poor until fidelity work happens — that is acceptable for the RUN milestone. *(The "~50% argmax → poor/degenerate output" premise is superseded 2026-07-15: see the Status addendum above — the real full-trajectory output is coherent and TT is at/above the intrinsic bf16 floor.)* Everything fidelity-shaped (#48291, R0.5/R0.6 replay, the staged-GQA PCC test, the shared-gemma4 decode gate-vs-rebaseline) is moved to the **Deferred — correctness track** below and is **not** on the RUN critical path.
>
> The original four device workstreams (W1–W4, Part II) are now mostly done or blocked. This roadmap is authoritative for "what to do next"; W1–W4 in Part II are the detailed specs for the pieces they cover.

### Current live-serving/performance addendum (updated 2026-07-13)

This addendum is the current status and supersedes older "open", "immediate next
step", and "Beyond Functional" wording in the dated June/July 2 narrative below.
The older sections remain as bring-up history and standing design context.

- The real full-depth `tenstorrent/vllm` OpenAI `/v1/completions` path is live on
  QB2. The companion fork has the #47488 block-granular runner and scheduler
  accounting, model registration, and request-release callback; four-block
  1024-token requests have run through the real engine.
- Traced serving is live rather than inferred from the reduced driver. The historical
  July-10 context/perf sweeps replayed block-0 IDs while holding a prompt-only prefix;
  current growing-prefix correctness releases/recaptures each changed prefix shape.
- Production bounded-memory chunked-Gumbel tracing is now live on QB2 (2026-07-13).
  A DG-local device-seeded uniform kernel reads a persistent seed tile and reuses one
  1024-vocab chunk buffer; the seed is refreshed between single-step replays, so block/step
  random streams do not repeat. Full 30-layer K=48 two-block evidence at
  `max_seq_len=1024` uses correct growing-prefix visibility: commit advances the prefix
  `32→288`, invalidates/releases the first 48 traces, and captures 48 new traces for block 1.
  Across two blocks it captured/executed 96 traces and measured 180.82 s for block 1 including
  recapture (1.42 output tok/s). Eager-vs-traced committed output is exact in the reduced
  control, and a frozen-prefix A/B changes block 1 while leaving block 0 unchanged. This closes
  production-Gumbel tracing and growing-prefix correctness, not traced 256K, paged/fixed-shape
  cross-block trace reuse, or the remaining host-seeded renoise-token RNG work.
- Primary warmed context evidence is complete for logical prompts
  32/256/1024/2048 at `max_model_len=4096`; bounded allocated contexts through
  32768 and real prompts through 16384 passed. The lower-priority 3072 warmed
  rerun was intentionally omitted at handoff.
- The isolated fixed-256-token-context K=1/4/8/12/16/20/24/32/40/48 sweep
  measured 166.80/108.28/72.94/54.88/44.46/37.06/32.00/25.54/21.34/18.28
  output tok/s. This is performance-only; K=48 remains model-faithful under
  #48291.
- Current serving is still one active sequence on the model-owned contiguous
  cache (`max_num_seqs=1`). Paged-cache ownership/concurrency, the absolute
  served-context ceiling, near-limit prefill hardening, production quality
  (#48291), and multimodal serving remain open. Whole-canvas context overruns
  are rejected before denoise/commit execution.
- Current evidence lives in `doc/vllm_integration/live_context_sweep_results_20260710.md`,
  `live_denoise_step_sweep_results_20260710.md`, and
  `traced_chunked_gumbel_20260713.json`; the July-10 rows are historical same-shape
  performance provenance.

### Status addendum (2026-07-15)

This addendum supersedes the older RUN-first "expect degenerate output / ~50% argmax"
framing below (the 2026-07-02 Priority note and the two-gaps table). Grounded in the
2026-07-15 issue updates (#48291/#47466/#49526/#47488/#47465) and
`doc/decision_fidelity/`.

**#48291 — DECISION: TT is at the intrinsic bf16 floor; the strict `0.95` gate is
mis-specified (recommendation recorded; production pass/fail unchanged pending owner
sign-off).**

- A zero-TT self-consistency control (same HF model, fp32 vs bf16, identical seeded
  FP32 Gumbel noise, committing the clean argmax) shows the block-diffusion trajectory
  is chaotic at the bf16 rounding scale: HF-fp32 vs HF-bf16 committed = **0.863 / 0.914**
  (seeds 0/1) at the 8-step gate and **0.867 / 0.914** at the full 48-step production
  budget — the reference **cannot match itself** to `0.95`, so **no bf16 implementation
  can clear the strict gate**. (`doc/decision_fidelity/`: `measure_bf16_floor.py`,
  `README.md`, `work_log.md`; commits `78ab3cda0ef`, `82cb69e6c45`.)
- **TT sits at/above that floor.** At the 48-step production config TT tracks the fp32
  *ideal* at **TT-vs-fp32 = 0.992** — closer than the bf16 reference itself (0.914); at
  the 8-step gate TT-vs-fp32 = 0.980 (seed 1) and equals the floor exactly on seed 0.
  **TT decoded output is content-identical to fp32 and COHERENT** (non-degenerate, 36
  content tokens, halted).
- **This REVISES the earlier framing.** The old "~50% argmax agreement → expect
  degenerate output" narrative is withdrawn: the floor is the intrinsic
  *diffusion-trajectory* sensitivity to bf16 that the HF reference shares, **not** a
  TT/gemma4 MoE defect. The 2026-07-03 "inherent shared Gemma-4 BF16 ceiling" comment is
  likewise withdrawn.
- **Recommendation (recorded as a recommendation, not a done deal):** product-accept the
  current coherent output; re-spec the gate to sound, reachable criteria (fidelity to the
  fp32 ideal, alignment-robust agreement, `sound_entropy_step_fidelity` as a
  converged-step diagnostic); pursue an fp32 MoE backbone as a **separate owned effort**
  (blocked today by `ttnn.topk` `TT_FATAL` on FLOAT32 + fp32 experts exceeding QB2 DRAM;
  DG must not edit `models/demos/gemma4/`).
- **Production stage-gate pass/fail is deliberately UNCHANGED (still red)** against the
  mis-specified bf16 reference — flipping the pass criteria is a correctness-policy change
  left for **owner sign-off**. Floor characterized on one canonical prompt at two seeds so
  far; a broader prompt/seed sweep would tighten the "usable bar" claim.

**One-line status on the serving/perf issues (2026-07-15, all on
`diffusion-gemma-function`; no shared-gemma4 edits in this work):**

- **#47466 (chunked/vLLM prefill):** live TTFT **20-25x faster** on the explicitly exercised
  `tenstorrent/vllm` chunked/ragged path (prefill 1024 15.07→0.60 s, 16384 ~270→13.5 s);
  that path removes the >4096 MoE cliff (bit-identical, `233b88276ab`). The direct
  `prefill_prompt_tokens` path now uses the same default-on 4096-token chunked-ragged dispatcher:
  current 64K-build pure-prefill evidence measures 16K **5.55 s**, 32K **10.84 s**, and 64K
  **35.58 s**. The 32K serving
  stall is root-caused + fixed (`FullAttentionSpec` for sliding layers,
  QB2-verified 32768 prefill 11.96 s, `210a60f0ac1`). Open: traced-denoise replay recovery,
  256K KV-memory-bound (~128K bf16 ceiling), upstream #47488, single-sequence cache.
- **#49526 (traced denoise):** production chunked-Gumbel tracing + growing-prefix
  correctness landed and QB2-verified (`ec5b64b4891`), **but** the growing prefix
  reintroduced per-block trace recapture → steady serving regressed **~18 → 3.6 tok/s**; a
  fixed-shape paged-read recovery is designed (not implemented, ~2-4 d,
  `denoise_replay_recovery_plan.md`, `6e2cde6f9f1`).
- **#47488 (block-granular serving):** the >21824-token admission stall is fixed in
  tt-metal (`FullAttentionSpec`, `210a60f0ac1`, QB2 32768 prefill 11.96 s); the runner/
  scheduler N-token contract remains fork patches with no upstream PR; paged-cache
  ownership is unimplemented so `--max-num-seqs 1` stays mandatory (#47557).
- **#47465 (decode/denoise perf):** historical prompt-only-prefix throughput vs denoise-step `K`
  measured (`K=48` ~18 tok/s; device-proven traced sparse-MoE @16 = 78.3, @6 = 155.9 tok/s,
  bit-identical, `6edc78938f4`). These are explicit tuned trace rows, not plain-server defaults or
  growing-prefix serving throughput. Step time is MoE-machinery-dominated; the fused MoE
  dispatch kernel landed (`DG_MOE_DISPATCH_FUSED2`, default off); decode opts
  (`DG_SPARSE_MOE`/`DG_DENOISE_TRACED`) stay opt-in pending PCC + t/s validation.

### Where we are (2026-06-26)

Foundation (torch reference + PCC harness #47468 ✅ closed, causal backbone #47461 ✅, QB2 fit #47487) plus the device attention/KV/sampling pieces are validated: KV-phase machine (W1/#47474) ✅, bidirectional masked SDPA — **both** ≤32768 (W2a) **and** the long-prompt >32768 path (W2b, resolved with regular non-causal SDPA — no new kernel) (#47462) ✅, on-device canvas sampling (W4/#47472) ✅. The decode-loop control flow (W3/#47463) is built and validated on synthetic logits but **blocked on decision fidelity (#48291)**.

**Runs end-to-end at small scale on real weights (2026-06-30).** Spike R0.1–R0.5 on QB2 joined the two halves: build the real 26B (13.236 GiB/chip), prefill a prompt, build the real-checkpoint denoise adapter (`logits (1,1,256,262144)`), run one full 256-token denoise→commit block, and measure device-vs-HF committed-argmax. Device commit-append, per-block position advancement, full-model + self-conditioning assembly from the real checkpoint, tokenizer/text I/O, and a device-vs-HF acceptance measurement all exist and have run. **Open:** remaining host-seeded renoise-token RNG (production chunked-Gumbel tracing landed 2026-07-13), intra-block device-side early stop, multi-block + long-context fit, the #48291 entropy/accept fidelity drift, and the shared-gemma4 decode regression risk introduced to fit the block (see R-new below).

### Two distinct gaps — do not conflate them

| Gap | What it is | Cost | Tracking |
|---|---|---|---|
| **Make it RUN** (emit *some* text) | Integration glue: join the pieces into one prompt→text device loop | Large but tractable net-new engineering (~weeks) | #47464 |
| **Make it CORRECT** (match HF) | The bf16/MoE/TP=4 **decision-fidelity bar** — diffusion commits the *clean argmax*, and the shared backbone shows only ~50% argmax agreement | Core gemma4 MoE-precision work **or** a product decision; possibly multi-week or unachievable on current kernels | #48291 |

The model can be made to *run* (and emit text) **without** resolving fidelity — it just will not be *correct*. **We are taking exactly that path: pursue RUN now, defer #48291.** Diffusion has no temperature/top-p cushion (it commits the clean argmax), so the ~50% backbone argmax ceiling maps almost directly to wrong tokens — **expect degenerate output from the first runs.** *(Superseded 2026-07-15: see the Status addendum above — the full-trajectory output is coherent; the ~50% causal-prefill proxy did not predict the real diffusion-trajectory fidelity.)* #48291 still determines whether the integration *eventually* yields usable output, so it must be decided *before shipping* — just not before running.

### Critical path to a first RUN (emit text) — correctness deferred

Dependency-ordered. RUN target: **short prompt → ≥2 committed blocks → detokenized text on QB2 with real 26B weights, without crashing.** Output correctness is *not* a gate here — see the **Deferred — correctness track** below. Steps 1–6 are the existing integration pieces (code exists and is smoke-tested; only step 4's single-block fit has actually run on real hardware — their remaining 🚧 is *correctness/validation*, not missing code). **The immediate step is R-b (the real multi-block run) at short prompt / small context.** R-a (maskless attention) is a *conditional* blocker — it only bites once prompts exceed ~768 tok, so it is **not** required for the short-prompt first RUN.

| # | Step | Status | Issue |
|---|---|---|---|
| R-b | **Real multi-block hardware run (short prompt / small context)** — prompt → 2 committed 256-token blocks → detokenized output path on QB2 with real 26B weights. Passed on QB2 2026-07-02 with `text_demo.py --max-seq-len 512 --canvas-length 256 --max-denoising-steps 1 --max-new-tokens 512 --num-blocks 2 --seed 0` after fixing mesh-tensor decision readback. Exit 0, post-build DRAM **13.236 GiB used / 18.631 GiB free / 31.867 GiB total**. Output is expectedly degenerate/empty after skip-special decoding; correctness remains deferred. | ✅ **RUN milestone passed (short prompt / small context)** | #47464 |
| R-a | **No attention crash at long prompts** — the denoise SDPA fallback (`tt/diffusion_attention.py::_sdpa_q_chunked`) recovers from the L1 static-CB clash **only when `attn_mask is None`; with a mask it re-raises → hard crash.** The production denoise path now defaults to maskless all-attend (`denoise_hidden_forward(..., use_explicit_sliding_mask=False)`), while the HF sliding-window mask is preserved as an explicit A/B-test opt-in. This accepts sliding-window over-attend as a RUN-first correctness compromise. Passed full-depth QB2 long-prompt forced 2-block smoke (`prompt_len=1024`, `canvas_length=32`, `blocks=2`, `generated_tokens=64`, `DG_TEXT_DEMO_SUCCESS`) after fixing single-chunk RoPE lifetime and switching prompt-KV reads to lazy per-layer DRAM slices. A full-depth short-prompt 256K context smoke with canonical 256-token canvases also now runs across two blocks with the RUN-first argmax sampler. | ✅ full-depth 2-block long-prompt smoke; ✅ full-depth 256K/canvas=256 two-block argmax smoke; 🚧 production Gumbel / near-limit prefill | #47464 |
| 0 | ~~Decide the decision-fidelity bar (#48291)~~ → **moved to the Deferred — correctness track below** (not a RUN gate) | ⤵ deferred | #48291 |
| 1 | **Device commit-append** — `tt/generate.py::commit_canvas_tokens` writes a committed canvas into the KV cache with `COMMIT_APPEND`; `denoise_and_commit_block` now composes denoise→commit for one block | ✅ RUN-validated by forced two-block QB2 smoke (`generated_tokens=512`, `blocks=2`) | #47464 |
| 2 | **Per-block RoPE/position advancement** — block N at `prompt_len + N·256`; `denoise_and_commit_block` drives adapter `q_rope_offset=start_pos` and `generate_blocks` advances `next_pos` per block | ✅ RUN-validated by parsed success summary (`prompt_len=32`, `next_pos=544`) and long-prompt smoke (`prompt_len=1024`, `next_pos=1088`) | #47464 |
| 3 | **Join full 26B + device self-conditioning + 30-layer prompt-KV** from the real checkpoint — all-layer prompt KV cache reader, self-conditioning builders, and raw/remapped checkpoint adapter/generation-builder helpers exist (`read_prompt_kv_cache_by_layer`, `build_self_conditioning`, `build_self_conditioning_embedding_weight`, `make_generation_logits_fn_builder_from_checkpoint_state`); real 26B device-vs-HF acceptance still open | ✅ RUN-validated at full 30-layer depth for short-prompt and long-prompt two-block smokes; 🚧 device-vs-HF acceptance remains deferred correctness | #47464 |
| 4 | **Measure the integrated real-size denoise step fits** on the (1,4) mesh (full-canvas logits + 262k soft-embed matmul + 30-layer `[P+C]` KV concat) | ✅ (≤512 ctx) — measured on QB2 2026-06-30 via R0.1/R0.3/R0.4 (build 13.236 GiB/chip; logits `(1,1,256,262144)`; full block). Open part is the >32768 / 256K-context fit, **and** it required ungated shared-gemma4 decode-footprint edits (see R0.4 row + R-new). | #47464 / #47487 |
| 5 | **Entry point `tt/generate.py`** — prompt→text wrappers exist (`generate_text` composes tokenization, prompt prefill, optional post-prefill logits adapter construction via raw/remapped checkpoint generation builders, multi-block commit loop, block-level EOS stop, full-sequence/decode, and host-visible EOS/length trim; `generate_text_from_checkpoint_state` builds the raw-checkpoint logits builder first, defaults to released `DiffusionConfig()`, defaults adapter config from `tt_model.hf_config`, can infer `num_blocks` from `max_new_tokens`, can infer `vocab_size` from tokenizer/model metadata for seeded host canvas init / denoise gumbel / renoise hooks, defaults EOS stop/trim from `tokenizer.eos_token_id`, defaults decode to skip special tokens, validates length/block/canvas/vocab/logits inputs, and fast-paths zero-token generation without canvas/logits/prefill work) and are smoke-tested on QB2 for prompt string → post-prefill logits builder → two committed blocks → decoded text using seeded random-token canvas init; production device RNG and intra-block device-side early stop still open | ✅ RUN entry point smoke + QB2 pipeline target; 🚧 production device RNG / intra-block device-side early stop remain deferred | #47464 |
| 6 | **e2e acceptance test** — CPU HF `generate()` vs `reference/generate_blocks` token-equal is ✅; `reference/generate_blocks` can now replay fixed per-block canvases plus injected denoise noise via `make_replay_canvas_init_fn` / `make_replay_noise_fn` for device-vs-reference tests; device-vs-HF on a short prompt with injected reference noise remains | 🚧 | #47464 |

> **RUN-first status (updated 2026-07-02):** the first short-prompt, small-context real-26B multi-block hardware run has passed. R-a's RUN-first mitigation is also in code: denoise is maskless by default, with explicit masks reserved for A/B tests. Next RUN hardening: keep this smoke reproducible, add a less-noisy success marker / regression target, and only then expand prompt length/context. **Do not** block on the deferred correctness track below.
>
> **Simple-dialogue visible-output smoke (2026-07-02):** a user-run full-30-layer short-context prompt (`"Hello, how are you?"`, `max_seq_len=1024`, `canvas_length=256`, `max_denoising_steps=32`, `num_blocks=1`, `--chunked-gumbel-sampling`, `--disable-eos-stop`) completed with `DG_TEXT_DEMO_SUCCESS generated_tokens=256 blocks=1 prompt_len=32 next_pos=288 sequence_len=275 text_chars=66` and printed visible text: `你好！I'm doing well, thank you for asking! How can I help you today?`. This is the first recorded simple-dialogue run with non-empty decoded output. It is still RUN evidence, not a #48291 correctness sign-off.
>
> **F1/F2 decode isolation (2026-07-02):** F1 is required before the next full RUN: revert the ungated shared-Gemma4 decode footprint edits from `models/demos/gemma4/tt/{attention/decode.py,attention/operations.py,experts/decode.py,rms_norm.py}` so `git diff main -- <those four files>` is empty. F2 then moves the commit-append workaround into DiffusionGemma-local code (`tt/commit_decode.py`) and routes `commit_canvas_tokens()` through that local path instead of shared `Gemma4Model.ttnn_decode_forward`. Validated after isolation with reduced-depth QB2 short+long two-block smokes and full-depth QB2 short-prompt two-block smoke.
>
> **Success/failure markers (2026-07-02):** `text_demo.py` now emits a single greppable `DG_TEXT_DEMO_SUCCESS …` line on success **and** a matching `DG_TEXT_DEMO_FAILURE mode=… mesh=… error_type=…` line on any uncaught exception (`main()` logs it then re-raises). Because the RUN-first denoise path emits expected `TT_THROW` fallback noise even on success, these two markers are the authoritative run outcome — grep them instead of scanning the fallback logs. Covered by CPU `tests/test_text_demo.py` (8 passed).
>
> **RUN regression target (2026-07-02):** the canonical R-b command is pinned as a device-gated pytest `tests/test_device_text_demo_run.py` (skips unless `DG_RUN_DEVICE=1`; checkpoint via `DG_CKPT`, defaults to `/home/zni/dg_models/diffusiongemma-26B-A4B-it`; optional `DG_TEXT_DEMO_NUM_LAYERS` for a cheaper reduced-depth smoke). The RUN gate is exit 0, not text quality. The test now passes `--disable-eos-stop` and `--max-seq-len 1024` so degenerate EOS-heavy output still commits **two** 256-token blocks (`prompt_len + 2*canvas = 544`, which would overrun a 512-token RoPE/cache span), and it parses the captured `DG_TEXT_DEMO_SUCCESS` summary via `_parse_success_summary()` to assert `generated_tokens=512`, `blocks=2`, `prompt_len=32`, and `next_pos=544` instead of checking exit code only. Collects + skips clean on CPU (1 skipped). **Validated on QB2 (2026-07-02): reduced-depth `DG_TEXT_DEMO_NUM_LAYERS=1` passed in 14.91 s before the summary assertions and passed again after tightening; full-30-layer forced-two-block pytest passed before tightening (XML time 96.16 s), after tightening (`1 test, 0 failures/errors/skips`, XML time 94.01 s), and after parser-backed assertions (`1 test, 0 failures/errors/skips`, XML time 94.09 s).** Wired into the QB2 Blackhole manifest as `bh-diffusion-gemma-run-smoke` with `DG_TEXT_DEMO_NUM_LAYERS=1` and `TT_LOGGER_LEVEL=ERROR` to suppress the ROW_MAJOR Metal warning flood; it safely skips if the checkpoint is unavailable and runs the cheap RUN smokes where the checkpoint exists. The same file now also pins the short-prompt 256K-allocation smoke, a host-injected Gumbel 256K/small-canvas smoke, and a chunked-Gumbel 256K/full-canvas smoke; the manifest-shaped reduced-depth command passed all five targets on QB2 (`5 passed`, 67.90 s; chunked full-canvas testcase 17.66 s).
>
> **Long-prompt RUN regression target (2026-07-02):** `tests/test_device_text_demo_run.py` also pins the R-a maskless denoise path with `hello * 1000` (`prompt_len=1024`) → two 32-token blocks, `--disable-eos-stop`, and `--max-seq-len 1536`. It defaults to one layer to stay cheap (`DG_TEXT_DEMO_LONG_PROMPT_NUM_LAYERS=full` opt-in for all layers) and asserts `generated_tokens=64`, `blocks=2`, `prompt_len=1024`, `next_pos=1088` through `_parse_success_summary()`. Validated on QB2 reduced-depth alongside the short-prompt smoke: `2 tests, 0 failures/errors/skips`, XML time 18.80 s (long-prompt testcase 5.46 s). The exact pytest entry also passes at full 30-layer depth with `DG_TEXT_DEMO_LONG_PROMPT_NUM_LAYERS=full`: `1 test, 0 failures/errors/skips`, XML time 44.16 s.
>
> **Google/HF generation defaults (2026-07-02):** local `google/diffusiongemma-26B-A4B-it` sets `text_config.max_position_embeddings=262144`, `canvas_length=256`, and `generation_config.max_new_tokens=256`. HF `DiffusionGemmaGenerationMixin.generate()` computes `max_length = input_len + max_new_tokens` and `max_new_canvases = ceil(max_new_tokens / canvas_length)`. Therefore a 256K context window is a capacity limit for `prompt + generated`, not a requirement that input prompts be 256K tokens.
>
> **Context-window guard (2026-07-02):** `tt/generate.py` now validates the physical TT generation span before block execution: aligned prompt cache length plus `num_blocks * canvas_length` must fit the current model context window (`tt_model.max_seq_len` first, then HF config context fields). This encodes the Google semantics above while respecting the current TT implementation detail that commit append writes whole canvas blocks even when host-visible `max_new_tokens` trims the final output. CPU `tests/test_tt_generate.py` covers context overrun rejection and the exact 262144 boundary (`247 passed`); QB2 reduced-depth short+long text-demo regressions also pass after the guard (`2 passed`, 17.87 s).
>
> **Short-prompt 256K allocation smoke (2026-07-02):** to validate the corrected context-window semantics without conflating it with a huge input prompt, a reduced-depth QB2 run used a normal short prompt but allocated the full `max_seq_len=262144` context (`num_layers=1`, `canvas_length=32`, `num_blocks=1`, `max_denoising_steps=1`, `--disable-eos-stop`). It completed with `DG_TEXT_DEMO_SUCCESS generated_tokens=32 blocks=1 prompt_len=32 next_pos=64`; post-build DRAM was 3.194 GiB/chip. This is positive evidence that short-prompt generation can run with a 256K context allocation. It is now pinned as `test_short_prompt_256k_context_allocation_exits_clean` in `tests/test_device_text_demo_run.py` and validated on QB2 reduced-depth (6.97 s). It is still **not** evidence that a single-shot near-262K prompt prefill is healthy.
>
> **Full-depth 256K allocation probe (2026-07-02):** the same short prompt / `max_seq_len=262144` / `canvas_length=32` / one-block run without `--num-layers` built the full 30-layer model and allocated the 256K caches successfully (`post-build` DRAM **29.704 GiB/chip used**, **2.163 GiB/chip free**), then failed at the first denoise step while generating device Gumbel noise: `ttnn.rand` requested a **1 GiB** DRAM buffer and hit `Out of Memory` (`largest free block ~101 MiB/bank`). It emitted `DG_TEXT_DEMO_FAILURE mode=generate ... RuntimeError`, and a follow-up reduced-depth short-prompt health smoke still passed (`1 passed`, 12.89 s), so the board/runtime recovered cleanly. Finding: full-depth 256K **weights+KV fit**, but full-vocab on-device Gumbel/noise allocation does not fit in the remaining fragmented DRAM; reduced-depth 256K allocation remains valid.
>
> **Full-depth 256K argmax RUN smoke (2026-07-02):** added explicit `--argmax-sampling` for RUN-first fit smokes. It routes the denoise sampler through clean argmax (`gumbel_noise=None`) instead of allocating a full-vocab Gumbel tensor, leaving the default seeded Gumbel path unchanged. With the same full-depth short-prompt `max_seq_len=262144`, `canvas_length=32`, one-block config, the run completed successfully: `post-build` DRAM stayed **29.704 GiB/chip used**, **2.163 GiB/chip free**, and `DG_TEXT_DEMO_SUCCESS generated_tokens=32 blocks=1 prompt_len=32 next_pos=64` (`text_chars=6`, decoded "Paris"). CPU tests cover the no-Gumbel hook and CLI threading (`test_text_demo.py` + `test_tt_denoise_loop.py`, `18 passed`); the 256K device regression now uses `--argmax-sampling` automatically when `DG_TEXT_DEMO_256K_NUM_LAYERS=full`. Follow-up canonical-canvas smokes also passed at full depth with `canvas_length=256`: one block completed with `generated_tokens=256`, `next_pos=288`, elapsed **272.8 s**; two blocks completed with `generated_tokens=512 blocks=2 prompt_len=32 next_pos=544`, block starts `32 → 288`, post-build DRAM **29.704 GiB/chip used**, elapsed **500.8 s**. This is RUN evidence for full-depth 256K + full 256-token canvas + cross-block commit/position advancement; it is still not a production RNG/correctness solution.
>
> **Host-injected Gumbel probe (2026-07-02):** added `--host-gumbel-sampling` / `use_host_gumbel_noise=True`, which generates seeded Gumbel noise on host and injects it through the existing device Gumbel-max path. This avoids `ttnn.rand` and keeps default device RNG unchanged. CPU coverage: `test_text_demo.py` + `test_tt_generate.py` (`265 passed`). Device evidence: full-depth 256K with `canvas_length=32`, one block, succeeds (`DG_TEXT_DEMO_SUCCESS generated_tokens=32 blocks=1 prompt_len=32 next_pos=64`; post-build **29.704 GiB/chip used**). The small-canvas path is now pinned as `test_short_prompt_256k_host_gumbel_small_canvas_exits_clean` in `tests/test_device_text_demo_run.py` and validated on QB2 reduced-depth with the rest of the text-demo suite (`4 passed`, 49.52 s). But full-depth 256K with canonical `canvas_length=256`, one block, still fails before denoise logits with a DRAM allocation for the injected Gumbel tensor: **128 MiB** requested, **16 MiB/bank** needed, largest free block only **~8 MiB/bank** after prefill/init; it emits `DG_TEXT_DEMO_FAILURE ... RuntimeError`. A reduced-depth 256K health smoke passed afterward (`1 passed`, 9.62 s). Finding: host injection is useful for reduced/small-canvas validation, but full-canvas production-Gumbel needs a chunked/no-materialize sampler or other memory strategy.
>
> **Chunked/no-materialize Gumbel sampler (2026-07-02):** added `--chunked-gumbel-sampling` / `use_chunked_gumbel_noise=True` and `gumbel_vocab_chunk_size` (default 1024). Instead of allocating a full `[B, canvas, vocab]` Gumbel tensor, `tt/sampling.py::gumbel_max_with_chunked_noise` generates one vocab chunk at a time, computes local `max/argmax(logits/T + gumbel)`, then reduces the per-chunk winners to global token ids. This directly targets the full-depth 256K/canonical-canvas Gumbel OOM above while preserving default seeded-Gumbel behavior. Coverage: CPU `test_text_demo.py` + `test_tt_generate.py` + `test_tt_denoise_loop.py` pass, and the QB2 op-level regression `test_device_entropy_harness.py::test_chunked_gumbel_max_matches_materialized_chunked_noise` passes against the existing materialized chunked-noise path. A reduced-depth 256K / `canvas_length=256` text-demo smoke also passes (`num_layers=1`, one block, `DG_TEXT_DEMO_SUCCESS generated_tokens=256 blocks=1 prompt_len=32 next_pos=288`, ~21.6 s), proving the CLI/generate/denoise integration path; it is now pinned as `test_short_prompt_256k_chunked_gumbel_full_canvas_exits_clean` with `DG_TEXT_DEMO_CHUNKED_GUMBEL_NUM_LAYERS=full` as the full-depth opt-in. The full-depth 30-layer 256K / `canvas_length=256` smoke now also passes with chunked Gumbel: post-build DRAM **29.704 GiB/chip used**, **2.163 GiB/chip free**, `DG_TEXT_DEMO_SUCCESS generated_tokens=256 blocks=1 prompt_len=32 next_pos=288 sequence_len=274 text_chars=0`, elapsed **263.3 s**. This validates full-canvas production-shaped Gumbel memory fit; it is not a correctness claim.
>
> **Near-limit prompt stress probe (2026-07-02):** do **not** conflate "256K context window" with "normal input is 256K tokens." The window is the upper bound for `prompt + generated`; ordinary RUN smokes can use short prompts while allocating `max_seq_len=262144`. A deliberate upper-bound stress attempt used `max_seq_len=262144`, `canvas_length=32`, `num_blocks=1`, `num_layers=1`, prompt `"hello " * 262080` (`raw prompt_len=262093`, padded prefill `cache_len=262112`, so `cache_len + canvas = 262144`). It built the 1-layer model successfully (`post-build` DRAM 3.194 GiB/chip) but produced no prefill/generate progress for ~5 minutes after post-build and had to be killed. It did not emit `DG_TEXT_DEMO_SUCCESS` or `DG_TEXT_DEMO_FAILURE`. A subsequent cheap short-prompt 1-layer health smoke also hung past pytest timeout until killed, so the board/runtime was reset with `sudo /home/zni/.local/bin/tt-smi -r`; after reset, the reduced-depth short-prompt health smoke (`DG_TEXT_DEMO_NUM_LAYERS=1`, `test_short_prompt_two_block_run_exits_clean`) passed on QB2 in 12.88 s. The finding is specifically **single-shot near-262K prompt prefill / cleanup after kill**, not proof that short-prompt generation with a 256K context allocation is broken, and not denoise SDPA (already op-validated to 262144).

#### Deferred — correctness track (NOT on the RUN critical path)

Previously top-of-queue, but **none of these advance RUN**; revisit after the model runs and before shipping. Nothing here should block a first hardware run.

| Item | Why deferred | Tracking |
|---|---|---|
| **Decide the decision-fidelity bar** — engineering MoE-precision fix vs product-accept a degraded floor | Pure correctness; the model runs without it (output just degenerate). Decide before *shipping*, not before *running*. | **#48291** |
| **R0.5 / R0.6 fidelity replay** — non-degenerate committed-argmax vs HF, entropy/accept drift, top-logits capture at positions `[2,3,4]` | Measures *how wrong*, not *whether it runs*. | #47464 / #48291 |
| **Staged-GQA fallback PCC test** vs `torch.F.scaled_dot_product_attention` | ✅ Covered on QB2 2026-07-02 by `test_device_bidirectional_sdpa.py::test_staged_gqa_fallback_matches_torch` using maskless GQA (`q_heads=4`, `kv_heads=2`, `scale=1.0`, PCC ≥0.99). Also cleaned up per-head fallback output tensor lifetimes after concat. | #47464 |
| **Gate-vs-rebaseline the ungated shared-gemma4 decode edits** (R-new) | Regression hygiene for plain Gemma-4 decode, not a diffusion RUN blocker. | #47462 / #47464 |
| **Production device RNG**, intra-block device-side early stop | Host-seeded noise hooks + bounded host readback suffice for a functional run. | #47463 / #47464 |
| **Long-context / 256K fit**, TP-across-mesh, perf | Run small-context, batch=1 first (Phase 4). | #47464 / #47487 / #47465 |

### Spike R0 — real-26B single-block denoise on QB2 (`bh-qbge-06`)  (#47464 + #48291)

**Why:** one run answers the two unmeasured unknowns at once — (a) does the **real-size** denoise step *fit + run* on the (1,4) mesh (critical-path step 4, ⬜), and (b) what is the **real-checkpoint** committed-argmax fidelity vs HF (the #48291 number, so far only inferred from the ~50% causal-prefill proxy — no real denoise trajectory has ever run). Short prompt first (≤512 ctx), **not** 256K — fit/fidelity, not long-context.

**Prereqs (all met):** DG ckpt on bhqb (`~/.cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it`, 11 safetensors / 49 GB / config+index+chat_template ✅); helpers exist — `weight_mapping.remap_state_dict` (backbone + self-cond split), `DiffusionGemma4Model`, `make_generation_logits_fn_builder_from_checkpoint_state`, `build_self_conditioning`, `read_prompt_kv_cache_by_layer`, `tt_denoise_block`/`denoise_and_commit_block`, replay helpers `make_replay_canvas_init_fn`/`make_replay_noise_fn`.

| # | Step | Acceptance | Risk |
|---|---|---|---|
| R0.1 | ✅ **Build real `DiffusionGemma4Model` on (1,4) mesh** from DG ckpt: load 11 shards → `remap_state_dict` → backbone_state (gemma4-keyed) + self_cond_state; build at `max_seq_len≈512, num_layers=30, create_kv_cache=True`, experts **TP-sharded** (column/row parallel per #47487). | Passed on QB2 2026-06-30: full 30 layers build-only, no OOM, per-chip DRAM post-build **13.236 GiB used / 18.631 GiB free / 31.867 GiB total**. Demo now logs baseline/post-build DRAM via `ttnn.get_memory_view`. | done |
| R0.2 | ✅ **Prefill a short prompt (causal)** via chat template (~32 tok) → write 30-layer prompt KV (`prefill_prompt_tokens` / `collect_prompt_kv_by_layer`). | Passed on QB2 2026-06-30: full 30-layer `--prefill-only` ran default chat prompt, `prompt_len=18`, aligned `cache_len=32`, post-prefill per-chip DRAM **13.237 GiB used / 18.630 GiB free / 31.867 GiB total**. | done |
| R0.3 | ✅ **Build denoise logits adapter from the real ckpt** (`make_generation_logits_fn_builder_from_checkpoint_state(..., prompt_len)`), then call `adapter(canvas_tokens, step=0)` once. | Passed on QB2 2026-06-30: full build + prompt prefill + real-checkpoint adapter call completed with `logits_shape=(1, 1, 256, 262144)`. The denoise SDPA path still first hits the known 832-byte L1 static-CB clash (`L1 buffer allocated at 423872`, `static CB end 424704`), then falls back to a staged TTNN GQA attention path (`matmul → softmax → matmul`) for the 32-token chunks. Follow-up reduced fallback temporary lifetime by freeing per-head expanded KV / score / probability intermediates without releasing source K/V aliases too early. | done for single adapter call — **BUT the staged TTNN GQA fallback (`_manual_gqa_attention`, `tt/diffusion_attention.py:217`) that actually executes on QB2 is UNVALIDATED numerically**: its only test (`test_gemma4_prefill_guards.py:183`) monkeypatches ttnn and asserts call structure, not values. Coverage inversion — `test_device_bidirectional_sdpa` PCC-validates the ttnn SDPA kernel, which is exactly the path bypassed on device. The staged math is logically correct (scale=1.0 matches q_norm pre-scaling; contiguous repeat-interleave GQA grouping; `softmax dim=-1`) but unverified; it is also **maskless-only** (re-raises on `attn_mask`). TODO before trusting R0.5: device PCC test driving `_sdpa_q_chunked` into the fallback vs `torch.F.scaled_dot_product_attention`. (Fallback TT_THROW logging is still noisy.) |
| R0.4 | ✅ **Run ONE denoise block** (`denoise_and_commit_block`) with released `DiffusionConfig()` (cap steps), seeded host canvas init + injected reference Gumbel noise. | Passed on QB2 2026-06-30: default 1-step smoke (`text_demo.py --max-seq-len 512 --canvas-length 256 --max-new-tokens 256 --num-blocks 1 --max-denoising-steps 1 --seed 0`) exits 0 after real denoise + commit append. The path now uses a lower-footprint Gemma4 decode sequence: decode per-head q/k/v RMSNorm takes the width-sharded fast path for batch=1; decode RoPE uses the manual per-user elementwise path instead of fused single-position RoPE; decode SDPA uses a `1x1` grid with `k_chunk_size=32`; Q is moved to DRAM before SDPA; decode SDPA output is freed after concat-heads; K/V update tensors are freed immediately after cache update; router RMSNorm and experts decode lifetimes were already reduced. This moves past the prior commit blockers: sliding SDPA (`390784` vs `483200`), router RMSNorm (`390336` vs `533248`), experts down sparse-matmul (`179392` vs `183552`), per-head q_norm (`292032` vs `303872`), fused RoPE (`390336` vs `512768`), and later global SDPA (`357568` vs `391040`). The run still emits known denoise fallback `TT_THROW` noise before Python catches the R0.3 SDPA clash, but the process completes. **⚠️ These decode-footprint changes were made IN-PLACE in shared `models/demos/gemma4` and are NOT diffusion-gated** — they alter plain Gemma-4 26B-A4B decode (RoPE op for batch=1 → `apply_rope_decode_peruser` for all batches; SDPA `1x1` grid + `k_chunk_size=32` for all layers; weightless-router + per-head RMSNorm width-sharded path; expert down-proj + Q L1→DRAM). This **re-couples the backbone** — the #47462 "backbone untouched" claim no longer holds for decode (see R-new regression risk). | done for one real 256-token block, but with the ungated shared-gemma4 decode re-coupling above; next: gate-or-rebaseline decision + R0.5 non-degenerate fidelity. |
| R0.5 | 🚧 **Measure decision fidelity vs HF** — replay the SAME initial canvas/noise through TT and HF at real 256-token scale, compare per-position committed-argmax agreement. | Passed the R0.4-aligned 1-step measurement on QB2 2026-06-30: fixed seed-0 host canvas (`canvas_sum=33550399`), prompt `prompt_len=18` with TT/HF padded `cache_len=32`, `max_denoise_steps=1`; padded HF `generate(..., decoder_input_ids=host_canvas)` and TT `generate_text_from_checkpoint_model_inputs(..., init_canvas_fn=make_host_canvas_init_fn(...))` both committed all EOS token `1`, agreement **256/256 = 100% — but DEGENERATE** (constant-vs-constant: both sides emit all-EOS at every position, so 100% measures no real-text decision). Follow-up full released-halt replay now runs 2 denoise steps with injected zero Gumbel/renoise noise after multi-step fixes: `DenoiseLogitsAdapter` retains prev logits for self-conditioning instead of letting `denoise_block` deallocate them, `TtSelfConditioning.soft_embedding` streams `softmax(prev_logits) @ embedding` over vocab chunks instead of materializing the full production-vocab softmax, and TT entropy now uses shifted-logsum `log(sum(exp(z-zmax))) - E[z-zmax]` with full-vocab intermediates in DRAM to avoid cancellation/L1 clashes. The latest 2-step comparison again gives committed/sampled/argmax **100%** (`[1.0, 1.0]`) — **again all-EOS-degenerate** — while the non-degenerate entropy signal improves but accept still drifts: entropy PCC `[0.934, 0.878]` (from `[0.706, -0.203]`), entropy max abs `[0.376, 0.064]`, accept IoU `[0.0, 0.871]`, canvas agreement `[0.984, 0.883]`, accept counts HF `[2, 233]` vs TT `[2, 203]`. A focused non-degenerate 1-step cap (`"Complete the sentence: Once upon a time"`, seed 1, zero Gumbel/renoise) finally exercises real non-EOS decisions: HF commits token `496` at 3 positions (`253` EOS), TT commits all EOS, so committed/argmax/sample/canvas agreement is **253/256 = 98.83%**, entropy PCC `0.958`, entropy max abs `1.673`, accept IoU `0.0`, accept counts HF `1` vs TT `2`. Re-running the committed harness (`/tmp/dg_replay_harness_seed1_full.pt`) reproduces exactly and localizes the 3 commit misses to step-0 positions `[2,3,4]`: HF argmax/sample `496`, TT argmax/sample EOS `1`, both accept `False`, so the non-EOS miss is **already logits/clean-argmax-level**, not caused by entropy accept or renoise. The accept/canvas drift is separate at EOS-only positions `[236,241,254]` where argmax/sample agree but entropy ordering changes the accept mask. Artifacts: `/tmp/dg_replay_harness_seed1_full.pt`, `/tmp/dg_r05_nondeg_seed1_1step_compare.pt`, `/tmp/dg_r05_entropy_dram_direct_2step_compare.pt`, plus earlier `/tmp/dg_r05_hf*.pt` and `/tmp/dg_r05_direct_2step_compare.pt`. | **Committed-argmax fidelity is now known to miss a small non-degenerate slice:** the all-EOS 2-step parity was trivial, and the non-EOS 1-step cap exposes 3 HF-vs-TT argmax/commit misses. The 3-token miss is now narrowed to the denoise logits path (backbone/prompt-cache/adapter numerics), not sampling/accept. #48291 remains an accept/canvas drift issue plus a small but real non-EOS committed-argmax mismatch; next capture/compare top logits around positions `[2,3,4]` to locate which layer/path flips token `496` to EOS. |
| R0.6 | 🚧 **Record** → update this roadmap (steps 3–4), #48291 (real-scale fidelity), #47464. | R0.4 fit + R0.5 1-/2-step fidelity recorded; issue comments #48291/#47462/#47463/#47472 updated; this roadmap refreshed for the gemma4 re-coupling + R0.5 degeneracy, and now includes the first non-degenerate 1-step miss artifact. Reusable replay harness added at `demo/replay_hf_tt.py`; HF-only smoke reproduces the seed-1 non-degenerate prompt with `hf_committed_non_eos=3` via `python models/experimental/diffusion_gemma/demo/replay_hf_tt.py --checkpoint <hf-snapshot> --local-files-only --hf-only --output /tmp/dg_replay_harness_hf_only.pt`; full smoke reproduces the TT miss and now emits compact per-step diff details in `summary["per_step_diffs"]`. | 🚧 partial — still pending: correct the #47462 "backbone untouched" comment for decode, post the non-degenerate fidelity writeup, and add a logits/top-k capture around positions `[2,3,4]` to isolate whether the clean-argmax flip comes from backbone logits drift, prompt/cache state, or adapter numerics. |

> **RUN-first note (2026-07-02):** R0.1–R0.4 (fit + one real block) are the RUN-relevant steps and are ✅. **R0.5/R0.6 (fidelity replay) are deferred to the correctness track above** — the measurements below are kept as record, not as a gate on running.
>
> Sequence: R0.1→R0.2→R0.3 is the **fit** answer; R0.4→R0.5 is the **fidelity** answer. If R0.3 OOMs, that *is* the finding (real-size step doesn't fit → right-size soft-embed / chunk the vocab matmul before proceeding). Device is the shared QB2 (`bh-qbge-06`) — coordinate the device lock (a long-running vLLM has held it; `sudo tt-smi -r` recovers a wedged board).

### Phased roadmap

**Phase 0 — Foundation** ✅ done — torch reference + PCC harness (#47468 ✅ closed), causal backbone PCC (#47461 ✅), QB2 memory fit (#47487).

**Phase 1 — Device pieces (W1–W4)** — built; one blocked:
- W1 KV-phase machine ✅ (#47474) — bounded-sliding commit-append wrap now has a discriminating QB2 regression test.
- W2a bidirectional masked SDPA, prompt+canvas ≤ 32768 ✅ (#47462).
- W2b long-prompt (>32768) attention ✅ via regular non-causal SDPA (D1, no new kernel) — SDPA op + RoPE reachability + integrated tiny-model denoise pass PCC≥0.99 through **262144** for **both** full and sliding layers, wired into the BH QuietBox2 pipeline (`tests/pipeline_reorg/blackhole_e2e_tests.yaml`). Residual: integration is a tiny config (hidden=128, head_dim=32); real-26B denoise integration is #47464 (#47462).
- W4 on-device canvas sampling ✅ (#47472) — SAMP-3 mesh-mapper `TT_FATAL` fixed; residual: regenerated-noise unvalidated at production vocab.
- W3 decode-loop control flow ✅ built & validated on synthetic logits. **Not a RUN blocker** — the control flow runs; only *correct* output is gated by #48291 (deferred). (#47463).

**Phase 2 — Integration to a first RUN** (#47464) — ✅ **short-prompt RUN achieved; hardening active.** The single-block real-26B path ran on QB2 2026-06-30 (R0.1–R0.4), the first real multi-block run passed on QB2 2026-07-02 (R-b: 2 blocks, 512 generated tokens requested, 1 denoise step/block, exit 0), and the R-a maskless long-prompt smoke is now pinned in `tests/test_device_text_demo_run.py` and validated at full 30-layer depth. Everything else previously listed here — production device RNG, intra-block device-side early stop, long-context fit, the #48291 fidelity drift, the ungated shared-gemma4 decode re-coupling (R-new) — is **deferred to the correctness track / Phase 4**, not part of the first RUN milestone.

**Phase 3 — Correctness** (#48291) — 🔴 **deferred by the current RUN-first priority** (was framed as *the* gating decision). Resolve the decision-fidelity bar (MoE precision work, or product acceptance of a degraded floor), measured via a real-checkpoint denoise trajectory. **Must be decided before shipping usable output; explicitly not before a first RUN.**

**Phase 4 — Functional milestone** (#47464) — after a first run (RUN-first) and the Phase 3 correctness decision — full 256K context (the **W2b** long-prompt non-causal attention prerequisite is now validated through 262144), TP across the mesh, perf optimization (#47465).

**Beyond Functional** — batched canvas decode (#47557), vLLM runner + TT-plugin integration (#47466 / #47488), CI + perf-regression pipelines (#47489), multimodal T+I / T+V (#47467), quantized checkpoint (#47475).

### Biggest risks

**RUN risks — these can block or corrupt a first hardware run:**

1. **Long-prompt denoise avoids explicit-mask fallback; full-context expansion is narrowed to near-limit-prompt hardening (R-a).** `_manual_gqa_attention` is the path that actually runs on QB2 (the ttnn SDPA hits an L1 static-CB clash and falls back), and the fallback is **maskless-only** — `_sdpa_q_chunked` re-raises when `attn_mask is not None`. RUN-first mitigation is implemented and now regression-tested: production denoise is maskless by default, explicit HF sliding masks are opt-in for A/B tests, and a fresh full-depth 256K-KV / canonical-canvas smoke passes all 48 steps with the production memory-bounded chunked-Gumbel sampler and the selected self-conditioning L1 chain (2026-07-10; `doc/optimize_perf/selfcond_logits_l1_256k_chunked.json`).
2. **Multi-block + cross-block KV hardware path is now exercised and regression-tested.** The forced two-block QB2 smoke asserts `generated_tokens=512`, `blocks=2`, and `next_pos=544`; the long-prompt variant asserts `generated_tokens=64`, `blocks=2`, and `next_pos=1088`. Commit-append still issues sequential single-token decode calls per block (functional, future perf item).
3. **Per-block position advancement** (RUN step 5) was an easy-to-miss requirement; the parser-backed success-marker assertions now cover it for both short- and long-prompt smokes. Full-context position/cache growth remains a later Phase 4 expansion.

**Deferred risks — correctness / hygiene, off the RUN path:**

4. **#48291 may be unachievable** on the current bf16/MoE/TP=4 kernels without core gemma4 MoE-precision work (fp32-faithful router top-k is blocked by `ttnn.topk` `TT_FATAL` on FLOAT32; fp32 experts exceed QB2 DRAM). Determines whether output is ever *usable*; does not block RUN. Decide before shipping.
5. **🆕 Shared-gemma4 decode regression from ungated R0.3/R0.4 footprint edits (R-new).** Fitting the real denoise→commit block required **unconditional, non-diffusion-gated** changes to the shared `models/demos/gemma4` decode path: batch=1 decode RoPE → `apply_rope_decode_peruser` (~0.99999 PCC, costlier) [`attention/decode.py`]; decode SDPA `1x1` grid + `k_chunk_size=32` for all layers (was `8x4`/full-grid, `k=64`) — single-core attention, a large perf hit; weightless-router + per-head RMSNorm width-sharded path [`rms_norm.py`, `attention/operations.py`]; expert down-proj + Q L1→DRAM [`experts/decode.py`]. No plain Gemma-4 26B decode PCC/throughput re-baseline has been run. **Mitigation: gate behind a diffusion param (default = old behavior) or re-baseline gemma4 decode PCC + perf.** Invalidates the "backbone untouched" claim in #47462's comment for the decode path. Hygiene, not a RUN blocker.
6. **🆕 R0.5 fidelity: a small, characterized non-EOS committed-argmax miss + accept/canvas drift.** The all-EOS 2-step parity was degenerate (constant-vs-constant); a non-degenerate 1-step cap exposes 3 HF-vs-TT argmax/commit misses localized to positions `[2,3,4]`, already at the logits/clean-argmax level (not sampling/accept). A reusable committed replay harness now exists (`demo/replay_hf_tt.py`) and reproduces the miss exactly. Remaining: a per-layer top-logits capture at `[2,3,4]` to locate the flip — deferred with #48291.
7. **W2b** (long-prompt > 32768 non-causal denoise attention) is **resolved** for the attention path: SDPA, RoPE, and integrated tiny-model denoise PCC all pass through 262144 for both full and sliding layers. The full W2b suite is now wired into the QB2 Blackhole pipeline; remaining 256K risk sits in full-model #47464 integration / #48291 decision fidelity, not W2b kernel feasibility.

---

## Part I — The plan

### 1. Goals & success criteria

From the parent issue §4. Each criterion is mapped to an owning workstream in §6.

| # | Success criterion | Owner(s) | Milestone |
|---|---|---|---|
| 1 | Near-term **QB2 only**; BHG (Galaxy) later, then broader HW | #47487 | Functional / Functional+ |
| 2 | **Max context 256K** (256 × 1024) | #47474, #47462, #47463, #47464 | Functional |
| 3 | Inputs: text · T+I · T+V | text=Functional; T+I/T+V=#47467 | Functional / Functional+ |
| 4 | **Max possible batch** given context | #47557 (canvas batch) + #47487 (budget/ceiling) | Functional+ |
| 5 | All input resolutions | #47467 | Functional+ |
| 6 | **On-device sampling** | #47472 | Functional |
| 7 | **vLLM** integration (tenstorrent/vllm TT plugin) | #47466, #47488 | Functional |
| 8 | Automatic Prefix Caching (APC) | best-effort — **not a Functional gate** (see §9, R-APC) | — |
| 9 | _Optional:_ quantized ckpt loading (FP8 / NVFP4) | #47475 | Infra/optional |

---

### 2. Model summary

**Text backbone (identical to Gemma-4 26B-A4B MoE):** 30 layers · hidden 2816 · 16 heads / 8 KV · head_dim 256 · MoE 128 experts top-8 + 1 shared MLP · `moe_intermediate` 704 · sliding-window 1024 interleaved with full-attention · dual RoPE (θ = 1e6 full / 1e4 sliding) · final logit softcap 30 · vocab 262144 · `canvas_length` 256. _(All verified against `config.json` / HF card / vLLM blog.)_

**Vision tower (Functional+):** `gemma4_vision`, 27 layers · hidden 1152 · `patch_size` 16 · `vision_soft_tokens_per_image` 280 — all confirmed in `config.json`. _("SigLIP" is an author-applied family label, **not** a primary-source name (config = `gemma4_vision`)._ For variable resolutions use the token budgets 70/140/280/560/1120 from #47467.)

#### 2.1 How it generates — block-autoregressive multi-canvas diffusion

The **same backbone, shared weights**, runs in three phases per 256-token block, selected by attention mode:

1. **Prefill (encoder, causal)** — encode the prompt; write KV.
2. **Denoise (decoder, bidirectional)** — iteratively denoise a 256-token *canvas*. Cross-attends to the prompt by **concatenating encoder K/V in front of canvas K/V** (prefix-style, no separate cross-attn module). **Read-only** on the prompt/committed KV; the canvas's own K/V is **recomputed every step** (a 256-token mini-prefill against the frozen prefix) and is **never written into the frozen cache until commit**.
3. **Commit (encoder, causal)** — re-encode the finished canvas, append its KV, emit 256 tokens. Then the next block.

**Noise = RANDOM tokens, not `[MASK]`.** The canvas is initialized to random token ids; rejected positions are re-noised to random tokens (uniform discrete diffusion, not absorbing-mask).

**Per denoise step** (≤ 48 steps; `12–16` typical-halt is anecdotal, not a design constant): temperature-scale (linear 0.8 → 0.4) → **Gumbel-max** `argmax(logits/T + gumbel)` → **entropy-budget acceptance** (accept most→least confident until accumulated entropy exceeds a budget) → re-noise the rest → stop when the argmax canvas is stable AND mean entropy < threshold, or step cap. **Commit = clean argmax**, not the noisy sampled values.

**Self-conditioning** (extra weights beyond the backbone): previous-step softmax → probability-weighted average of token embeddings → small **gated MLP** → added to canvas embeddings. Active only in denoise; **zeroed on encoder passes**.

> Algorithm reference: transformers `modeling_diffusion_gemma.py` (`DiffusionGemmaForBlockDiffusion`); vLLM blog <https://vllm-project.github.io/2026/06/10/diffusion-gemma.html>.

---

### 3. Reuse vs build

#### Reuse — the backbone is already in-repo
`models/demos/gemma4/` is a near-complete, trace-compatible on-device Gemma-4 26B-A4B MoE: `tt/model.py`, `tt/moe.py`, `tt/router.py`, `tt/experts/`, `tt/shared_mlp.py`, `tt/attention/`, weight loading, CCL/TP. MoE / softcap / dual-RoPE / weight-loading match the target. On-device sampling reference: `models/common/sampling/generator.py` (AR/last-token-shaped — a canvas/per-position variant is net-new).

**Already present — do NOT rebuild:**
- **K=V tying for full-attn (global) layers only** — flag `attention_k_eq_v` (`tt/model_config.py:45`), gated `… and not self.is_sliding` (`tt/attention/__init__.py:34`), impl `v_w = k_w` (`tt/attention/weights.py:73`). **Sliding/local layers keep a real separate V** — this matters for the bidirectional local-window path (#47462).
- Scaleless V-norm (`tt/attention/prefill.py:61` and `:214`, `decode.py:84`).
- Bounded-sliding hybrid KV cache; tokenizer / chat-template.

#### Net-new — the real work
| # | Item | Owner | Notes |
|---|---|---|---|
| N1 | Bidirectional canvas attention + 2D mask geometry + non-causal long-context path | #47462 | gemma4 SDPA is hardcoded `is_causal=True` |
| N2 | Three-phase KV-cache state machine | #47474 | prereq → #47462/#47463 |
| N3 | Discrete-diffusion decode loop (entropy / Gumbel-max / renoise / accept / commit) | #47463 | **spike acceptance first** |
| N4 | Self-conditioning gated MLP — **loader → #47461**, **runtime (denoise-only) → #47463** | #47461 / #47463 | the one net-new weight module |
| N5 | On-device canvas sampling (per-position over 256) | #47472 | keep logits/probs on device |

---

### 4. Milestones & exit criteria

| Milestone | HW | Scope | Exit criteria | Perf |
|---|---|---|---|---|
| **Foundation** | QB2 | Correctness gate (not a product target) | Causal 26B-A4B backbone PCC vs HF on **QB2** (#47461) on the DiffusionGemma ckpt; torch ref + PCC harness (#47468) live | — |
| **Functional** | **QB2** (gated on #47487) | text-only · batch 1 · max ctx 256K · on-device sampling · vLLM | E2E text generation matches torch ref at 256K on QB2; served via TT plugin | TTFT ~50%, t/s/u ~100% |
| **Functional +** | + BHG, then broader HW | + T+I / T+V · all resolutions · batch>1 | Multimodal e2e; batched decode at PCC parity to batch=1 | t/s/u ~200% |
| **Complete** | all | everything | — | — |

**Foundation's #47461 exit = causal backbone logits PCC vs HF on the _DiffusionGemma_ ckpt — measured on QB2 2026-06-24 and passes the 0.83 baseline** (PCC **0.877** 5-tok / **0.847** 24-tok; see the Part III status table). The PCC-vs-baseline gate is therefore **met**; what is *not* met is coherent greedy generation — argmax-match is ~50% (the shared bf16/MoE/TP=4 ceiling), which is the deferred **#48291** decision-fidelity bar, not a #47461 gap. Keep this distinct from #47487's HW-enablement fact that 26B-A4B *fits + runs* on `P150x4` (TP=4, no OOM, experts TP-sharded): "fits + runs" is a memory/plumbing result measured on the *plain gemma* ckpt and does **not** validate DiffusionGemma (different fine-tuned weights, extra self-cond, bidirectional denoise). #47487 owns the QB2 memory budget / 256K batch ceiling (and, later, Galaxy 4×8 TP); #47461 owns the PCC validation on the DiffusionGemma weights.

---

### 5. Critical path & dependencies

```mermaid
graph TD
  47468["#47468 torch ref + PCC harness<br/>(oracle — upstream of all TT validation)"]
  47461["#47461 causal backbone on QB2<br/>+ self-conditioning loader"]
  47487["#47487 HW enablement<br/>QB2 fit + Galaxy 4×8 TP"]
  47474["#47474 KV phase state machine"]
  47462["#47462 bidirectional forward<br/>+ 2D mask geometry"]
  47463["#47463 decode loop<br/>(spike acceptance first)"]
  47472["#47472 on-device canvas sampling"]
  47464["#47464 functional e2e (text)"]
  47465["#47465 perf"]
  47466["#47466 vLLM integration"]
  47488["#47488 vLLM block-granular runner<br/>(upstream dev PR)"]
  47557["#47557 batched decode"]
  47467["#47467 multimodal"]
  47475["#47475 quant dequant"]
  47489["#47489 CI"]

  47468 --> 47461
  47461 --> 47487
  47461 --> 47474
  47474 --> 47462
  47474 --> 47463
  47472 --> 47463
  47462 --> 47464
  47463 --> 47464
  47472 --> 47464
  47487 --> 47464
  47464 --> 47465
  47463 -. spike outcome gates perf approach .-> 47465
  47464 --> 47557
  47466 --> 47488
  47464 --> 47466
  47467 -. needs plugin video modality .-> 47488
```

**Dependency notes (verified against sub-issue bodies):**
- **#47468 (oracle) is upstream of all TT validation.** The torch reference + harness scaffolding has *no* dependency on the TT-side loader and should be built first. Only the **self-conditioning module-level PCC** within #47468 needs the #47461 loader to exist — captured by the N4 ownership tag, not by a "loader gates harness" edge.
- **#47474 is a hard prereq** of #47462 and #47463 (both structurally depend on the three-storage-class KV contract).
- **#47463's acceptance spike outcome gates #47465's perf approach** — see R1.
- **#47488 depends on #47466** (promoted out of it) and **lives in tenstorrent/vllm `dev`** (separate repo, lands via PR).
- **#47557 (batch>1) depends on the #47464 batch-1 baseline**; interacts with #47488 (block-granular) and #47474 (per-request KV).
- **#47487 budget must include #47474 canvas scratch + #47462 non-causal mask** — see R3.

**Critical path to Functional.** The backbone/loop spine converges on #47464, which then forks into a perf tail and a serving tail, alongside a co-critical HW track:

`#47468 → #47461 → #47474 → {#47462, #47463 (spike first), #47472} → #47464 → { #47465 perf ; #47466 → #47488 serving }`

- **#47487 is a co-critical parallel track, not ordinary parallel** — it `→ #47464` (a hard prereq, see §6). A slow QB2 fit / Galaxy 4×8 TP enablement directly delays #47464 and becomes shared critical path.
- **The serving tail is real and the least-controllable.** vLLM is a hard Functional criterion (criterion 7); #47466 → #47488 depends on #47464 and extends **past** #47465. **#47488 is an upstream tenstorrent/vllm `dev` PR** (separate repo / review — R6) — the least-controllable tail.
- **#47472** has no upstream dependency but is consumed by #47463/#47464 (must land before they complete).

---

### 6. Workstreams

Each workstream lists **scope**, **depends-on**, **key code anchors**, **approach**, and **acceptance**.

#### Foundation

##### #47468 — Torch reference model + PCC harness
- **Scope:** Vendor the HF `DiffusionGemmaForBlockDiffusion` torch reference with **deterministic** `generate` (fixed seed + fixed schedule) as the PCC oracle. Build a harness validating per-op / per-module / full-denoising-trajectory fidelity — including **entropy values** and **Gumbel-max argmax agreement** — by injecting the torch run's exact noise into the TT path.
- **Depends-on:** none (build first).
- **Anchors:** `tests/ttnn/utils_for_testing.py` (`assert_with_pcc`); `tech_reports/ttnn/comparison-mode.md` (auto per-op PCC).
- **Approach:** Stand up the oracle independent of any TT code. Add hooks to inject the torch run's Gumbel noise + random-renoise token ids (on-device RNG won't bit-match — see R5). The harness must validate **diffusion decisions**, not just logits (bfp8 small-probability drift can flip accept/renoise).
- **Acceptance:** Deterministic torch trajectory reproducible; harness can PCC any TT module against the oracle and diff entropy/argmax decisions per step.

##### #47461 — Causal Gemma-4 26B-A4B backbone on QB2
- **Scope:** Bring up the existing causal backbone on the **DiffusionGemma checkpoint**; validate it reproduces HF logits (PCC + coherent greedy generation) on **QB2**. Implement the **net-new self-conditioning gated MLP loader** (no such module in `gemma4/tt/`).
- **Depends-on:** #47468 oracle (to validate against).
- **Approach — two stages, do not conflate:** (1) bring up the unchanged gemma4 path and PCC vs HF on the **gemma4 ckpt**; (2) repoint to the **DiffusionGemma ckpt** and validate text weight mapping + causal-pass PCC — catch missing/renamed weight keys, the extra self-conditioning weights, and config diffs (per-layer q/k/v-norm + K=V reconciliation vs `modeling_diffusion_gemma.py`, `canvas_length`, …) — **before** any diffusion delta.
- **De-risk surface:** MoE-128/top-8 sparse routing, shared MLP, final softcap, dual-θ RoPE, 256K ctx, multi-device TP.
- **Acceptance:** Causal backbone logits PCC vs HF on QB2 on the DiffusionGemma ckpt; self-cond loader lands so #47468's self-cond module PCC can run.

##### #47487 — HW enablement: QB2 fit + Galaxy 4×8 TP
- **Scope:** Fit/run the causal backbone on **QB2** (1×4 Blackhole) with a **documented memory budget + batch ceiling**, and wire a **4×8 TP/EP/SP mesh** for Galaxy/BHG. (PCC *validation* belongs to #47461; this issue owns fit / budget / ceiling / mesh and consumes #47461's PCC as a gate.)
- **Depends-on:** #47461 (validated backbone).
- **Facts:** 26B-A4B fits + runs on QB2 (`P150x4`, TP=4) — verified, no OOM (experts are TP-sharded ≈5.7 GB/chip). `is_galaxy` is currently used only for sampling args (`tt/model.py:407`) — no 4×8 mesh wired in `tt/`. Weight memory ≈ bf16 51.7 GB / bfp8 26 GB.
- **Approach:** Reuse `gemma4/tt/{config.py,ccl.py}`. **The QB2 budget must additionally account for the per-step canvas K/V scratch (#47474 storage class ii) and the non-causal long-context mask buffers (#47462)** — neither is exercised by the short-prompt causal-backbone run (R3). Size the batch ceiling against weights + 256K KV + canvas scratch + mask.
- **Acceptance:** Documented QB2 memory budget + batch ceiling at 256K; model **fits + runs** on QB2 and (later) Galaxy 4×8. (Backbone *PCC* is #47461's acceptance — not duplicated here.)

#### Functional core

##### #47474 — KV-cache phase state machine *(prereq)*
- **Scope:** Per-phase KV state machine — prefill writes prompt KV; denoise **reads the frozen prefix without writing**; commit appends the finished 256-token block — with cache zones preventing the bounded sliding-window circular buffer from wrapping/corrupting the live window, on both full-attention and sliding layers, paged across the mesh.
- **Depends-on:** #47461 backbone.
- **Anchors:** `decode.py:119-224,76`; `prefill.py:91-96,77-90`; `kv_cache_hybrid.py:53-98`; `model.py:162-177,609`. Note: the existing `is_kv_shared` flag is architectural cross-layer KV-sharing (E2B/E4B), **not** a per-phase freeze. Circular sizing = `ceil(sliding_window / block_size)`.
- **Approach:** Define **three storage classes** — (i) frozen prompt/committed KV (paged, read-only); (ii) **per-step canvas K/V** recomputed each denoise step (ephemeral activations or a dedicated scratch zone — **must NOT be written into the cache during denoise** or it corrupts the frozen prompt KV); (iii) commit-append (write the finished canvas's KV once). Specify the page / circular-buffer mapping for local and full-attention layers.
- **Acceptance:** Encode-once + multi-block commit-append; PCC vs HF over **≥ 2 committed blocks**.

##### #47462 — Bidirectional encoder-decoder forward
- **Scope:** Non-causal SDPA + symmetric sliding-window (baked mask) + encoder-decoder K/V concat + K=V reuse + long-context (>32768) non-causal path + random-token canvas state + an **explicit 2D mask geometry contract**.
- **Depends-on:** #47474.
- **Anchors:** gemma4 SDPA hardcoded `is_causal=True` (`prefill.py:126,264`, `operations.py:333`). Non-causal refs: `models/experimental/pi0/tt/ttnn_gemma.py:320` (`scaled_dot_product_attention(attn_mask=…, is_causal=False)`), `models/tt_dit/encoders/gemma/model_gemma.py:253`. SDPA guard: `sliding_window_size` and `attn_mask` are **mutually exclusive** (`sdpa_device_operation.cpp:67-68`) ⇒ **bake the symmetric window into the mask**. Causal-only chunked long-context path: `operations.py:25-29`, `prefill.py:106-130`.
- **Approach:** Define the `[256, prompt_len+256]` mask geometry explicitly: canvas↔canvas bidirectional (local layers symmetric 2W+1 — resolve window units: vLLM `|q−k|<W` vs ttnn centered); canvas→prompt visibility per layer type; canvas absolute/RoPE positions offset by `prompt_len`; and how the mask is **chunked for long prompts**. Build a **non-causal chunked long-context path** (the existing one is causal-only and silently returns wrong results for seq>32768). Remember sliding/local layers keep a real separate V.
- **Acceptance:** Per-layer + final-logits PCC vs HF; correctness at >32768 up to 256K.

##### #47463 — Discrete-diffusion decode loop
- **Scope:** Temperature-scale → Gumbel-max → **entropy-budget acceptance** → random-token renoise → convergence/stop → **clean-argmax commit**, plus self-conditioning **runtime** (denoise-only, zeroed on encoder passes). The **entire loop runs inside the tt-metal model's `prefill_forward`/`decode_forward`**.
- **Depends-on:** #47474; uses #47472 sampling; acceptance gate needs #47468 noise injection.
- **Primitives:** entropy = softmax+log+mul+sum; Gumbel-max; renoise = where+scatter. `ttnn.sort` (values **and** indices), `ttnn.cumsum`, `ttnn.topk`, `ttnn.scatter` all exist.
- **⚠️ Spike first.** Entropy-budget acceptance = sort-by-confidence + cumulative-entropy threshold + scatter-back. The **unproven** parts: (a) scatter-back / inverse-permutation mapping accept decisions to original canvas positions; (b) the **data-dependent cutoff** under **static Metal Trace**; (c) trace-ability + perf of the whole sort→cumsum→scatter chain over the 256-position axis. The spike may conclude a new op/kernel or a small host fallback (256-element readback) is needed. **Treat "no new kernels" as a hypothesis.** The spike outcome **gates #47465** (R1).
- **Acceptance:** Multi-step trajectory PCC vs torch (entropy values + per-step argmax agreement) under injected noise; commit emits the clean argmax of all 256 positions.

##### #47472 — On-device canvas sampling
- **Scope:** User-facing sampling primitives — temperature schedule, Gumbel-max, seed/reproducibility, (forward-looking) top-k/top-p — across **all 256 canvas positions per step**, keeping logits/probs on device. Plumb through the TT plugin via `model_capabilities["supports_sample_on_device"]` + `TTSamplingParams`.
- **Depends-on:** no upstream dependency, but **consumed by #47463 and #47464** (must land before they can complete). Boundary with #47463: that issue owns the *control flow*; this owns the sampling *primitives*.
- **Anchors:** `models/common/sampling/generator.py` `SamplingGenerator` is AR/last-token-shaped (reference only) — a canvas/per-position variant is net-new. `TTSamplingParams` lives in the vLLM plugin.
- **Note:** top-k/top-p is **NOT shipped** in the reference (transformers defers it; vLLM PR #45429 open/unmerged) — ship temperature + Gumbel-max + entropy-budget first; treat top-k/p as forward-looking, not a gate.
- **Acceptance:** Matches HF reference distribution under fixed seed; temperature honored; **no per-step host readback of `[256, vocab]`**.

##### #47464 — Functional text-only e2e
- **Scope:** Integrate backbone + bidirectional forward + decode loop + on-device sampling into an e2e **text-only, batch-1, on-device-sampling** demo under `demo/`, validated at the **full 256K context** on {QB2, BHG}, with quality matching the torch reference. This is the block-autoregressive multi-canvas loop end-to-end (denoise → encode/commit → append → next canvas).
- **Depends-on:** #47462, #47463, #47472, **#47487 (HW — co-critical, see §5)**.
- **Anchors:** reuse `gemma4/tt/ccl.py`. The existing 256K demo (`text_demo_v2.py`) is validated only for the **12B** ckpt — the net-new gap is **full-256K canvas decode for 26B-A4B** (batched portion deferred to #47557).
- **Acceptance:** Coherent text output at 256K on QB2 matching torch-ref quality.

##### #47465 — Functional perf
- **Scope:** Optimize the functional text-only path to **TTFT ~50%, t/s/u ~100%** via Metal Trace of the per-step decoder, 2 command queues, and op-level tuning (bfloat8_b / LoFi / sharding / dealloc).
- **Depends-on:** #47464; **#47463 spike outcome** (R1).
- **Central tension:** adaptive early-stop (#47463) vs static-trace fixed shapes. Resolve as **fixed-max-steps + masked no-op** (traceable) vs an **untraced adaptive** path. If acceptance needs per-step host readback, the host sync defeats trace and puts this gate at risk.
- **Anchors:** reuse `gemma4` `generator_trace.py`; `tech_reports/.../AdvancedPerformanceOptimizationsForModels.md`; `tools/tracy/profile_this.py`.
- **Acceptance:** Perf targets met on QB2; trace strategy documented.

##### #47466 — vLLM integration (TT plugin)
- **Scope:** Serve via the [tenstorrent/vllm](https://github.com/tenstorrent/vllm) TT plugin (`dev`). Implement a registered TT model class (`DiffusionGemmaForCausalLM`, copyable from the gemma4 bridge) that **owns its forward + attention + KV**; document which serving features end up enabled.
- **Depends-on:** #47464 (working model).
- **Anchors:** `initialize_vllm_model` (`loader.py:39`), `prefill_forward` (`model_runner.py:1999`), `decode_forward` (`async_decode.py:473`), `allocate_kv_cache_per_layer`, `model_capabilities`; register in `register_tt_models()` (`platform.py`; HF arch auto-prefixed `TT`). Copy `Gemma4ForCausalLM` (registered as `TTGemma4ForCausalLM`).
- **Plugin constraints (verified on `dev`):** spec-decode hard-blocked (`platform.py:342`); chunked prefill unsupported (`platform.py:339-341`); phase-based continuous batching (a step is all-prefill OR all-decode); APC force-disabled for sliding-window models (`platform.py:512-521`); multimodal is image+text only today.
- **Approach:** The bidirectional attention + the whole denoise loop live **inside** the model's `prefill_forward`/`decode_forward` (loop internally, commit a 256-block, advance `start_pos`). vLLM's GPU `model_states` / per-request causal tensor / `DiffusionSampler` do **not** run here.
- **Acceptance:** Served e2e through the plugin; serving-feature matrix documented.

##### #47488 — vLLM block-granular runner/scheduler
- **Scope:** Define and land (on tenstorrent/vllm `dev`) the runner/scheduler contract so a decode step advances `start_pos` / page-tables / token-accounting by a **committed 256-token block** instead of one token, and serves coherent multi-block generation. Includes token-streaming semantics (whole-block vs incremental) and `num_computed_tokens` accounting.
- **Depends-on:** **#47466** (promoted out of it). **Different repo — lands via PR on `dev` (separate review cycle).** Blocker for serving.
- **Acceptance:** Multi-block generation served e2e via the plugin.

##### #47557 — Batched canvas decode
- **Scope:** Batch the diffusion canvas decode across requests — per-request canvas state (incl. self-conditioning + accept/freeze), batched bidirectional attention + MoE, paged/per-request KV + commit-append. **Batch=1 first, then batch=4**, with a **documented max-batch-given-context ceiling**.
- **Depends-on:** #47464 baseline; interacts with #47488, #47474.
- **Anchors:** gemma4 decode is currently single-user (batch>1 silently uses only user 0; `demos/gemma4/demo/text_demo.py:266-267,483`).
- **Acceptance:** PCC parity to batch=1; documented batch ceiling.

#### Functional +

##### #47467 — Multimodal (image + video)
- **Scope:** Wire the `gemma4_vision` tower into the encoder prefix for **T+I** and **T+V** at all resolutions (variable 70/140/280/560/1120 token budgets; video ≤ 60s). Functional+ perf target t/s/u ~200%.
- **Depends-on:** **T+V needs plugin-side modality support** (the TT plugin is image+text only today — cross-ref #47466/#47488).
- **Anchors:** reuse `models/demos/multimodal/gemma3/tt/` vision encoder + projector; `llama_vision_model.py` for variable-res tiling.
- **Acceptance:** Coherent text output for T+I and T+V at all resolutions.

#### Infra / optional

##### #47475 — Quantized checkpoint dequant converter
- **Scope:** Offline DeepSeek-style **dequant** converter turning RedHatAI FP8 / NVFP4 (compressed-tensors) checkpoints into bf16/bfp8 so they load and run PCC-matching the bf16 reference. **On-device FP8/NVFP4 arithmetic is explicitly deferred.**
- **Anchors:** `gemma4/tt/precision.py` `_DTYPE_BY_NAME` supports only bf16/bfp8/fp32; no dequant-on-load today (cf. DeepSeek `dequantize_hf_checkpoint.py`).
- **Note:** **NOT a load blocker** — the bf16 26B-A4B (~51.7 GB) loads directly. Optional / serving-perf.

##### #47489 — CI & perf-regression pipelines
- **Scope:** Wire into CI like gemma4 — per-module unit tests + diffusion-specific tests (bidirectional forward, denoise loop, self-conditioning), an e2e accuracy (PCC / denoise-trajectory) test, a perf-regression entry (TTFT + t/s/u), pipeline/threshold config gated on {QB2, BHG} runner availability.
- **Depends-on:** #47468 (oracle), #47465 (perf targets), #47487 (runners).
- **Anchors:** reuse gemma4 scaffolding (unit suite, `pcc_thresholds.json`, `vllm_harness.py`, `test_vllm_parity.py`); add `tests/pipeline_reorg/models_{unit,e2e}_tests.yaml` + a `pcc_thresholds.json`.

---

### 7. Validation / PCC strategy

Layered, oracle-driven (all via #47468):

1. **gemma4 ckpt, causal:** unchanged gemma4 path PCC vs HF — proves the backbone import is sound.
2. **DiffusionGemma ckpt, causal:** validate text weight mapping + causal-pass PCC (renamed keys, self-cond weights, config diffs) **before any diffusion delta**.
3. **Per-module (diffusion):** bidirectional forward (per-layer + final logits), self-conditioning module, sampling decisions — under **injected** torch noise.
4. **Full denoise trajectory:** entropy values + Gumbel-max argmax agreement across all ≤48 steps over ≥2 committed blocks.
5. **E2E quality:** generation matches torch ref at 256K on QB2.

**Determinism:** token-for-token PCC requires injecting the torch run's exact Gumbel noise + random-renoise token ids (on-device RNG won't bit-match). Reserve regenerated noise for distributional checks. Validate **decisions**, not just logits — bfp8 small-probability drift can flip accept/renoise.

---

### 8. Risk register

| ID | Risk | Impact | Mitigation |
|---|---|---|---|
| **R1** | Entropy-budget acceptance (data-dependent cutoff + scatter-back) may not be trace-able under static Metal Trace | Could force per-step host readback → defeats trace → **Functional perf gate (#47465) at risk** | **Spike #47463 acceptance before committing the loop.** Fallback: fixed-max-steps + masked no-op (traceable) or bounded host readback. Spike outcome is a hard input to #47465. |
| **R2** | Canvas K/V written into the frozen cache during denoise corrupts prompt KV | Silent correctness failure | #47474 three storage classes; per-step canvas K/V never written until commit; PCC over ≥2 committed blocks. |
| **R3** | QB2 memory fit — weights + 256K KV + **canvas scratch + non-causal mask** | OOM / reduced batch ceiling | #47487 budget must include #47474 scratch + #47462 mask; **a short-prompt causal-backbone PCC does not de-risk the 256K fit.** |
| **R4** | Long-context (>32768) non-causal path — existing chunked path is **causal-only** and silently wrong above 32768 | Wrong results at long ctx | Build a net-new non-causal chunked path in #47462; do not reuse the causal chunked op. |
| **R5** | On-device RNG won't bit-match torch | Can't do token-for-token PCC | Inject torch's exact Gumbel + renoise ids (#47468); validate decisions. |
| **R6** | vLLM block-granular emission is an **upstream `dev` PR** in a separate repo | Serving blocked on external review | #47488 depends on #47466; plan the upstream PR early; separate review cycle. |
| **R7** | Sliding/local layers keep a **real separate V** (K=V tying is global-only) | Bidirectional symmetric-window path gets V wrong if assumed tied | Reconcile per-layer in #47462; do not assume K=V on local layers. |

---

### 9. Serving notes (tenstorrent/vllm TT plugin)

- **R-APC.** APC is force-disabled for sliding-window models (`platform.py:512-521`) and Gemma uses sliding-window layers. The parent #47452 lists APC under success criteria, but it is **best-effort / outside the Functional gate** unless the plugin's sliding-window gating changes (matches the issue's 2026-06-19 review follow-up).
- The TT model owns its forward + attention + KV; the runner passes only tokens / page_table / kv_cache / start_pos / prompt_lens / sampling.
- spec-decode and chunked prefill are unsupported; continuous batching is phase-based (`docs/SCHEDULING.md`).

---

### 10. Open questions

- **2D mask geometry (#47462):** does the local symmetric window extend over the prompt prefix, or is the prompt fully visible to the canvas? Per-layer-type canvas→prompt visibility?
- **Acceptance cutoff (#47463):** fixed-max-steps + masked no-op vs untraced adaptive — decided by the spike (R1).
- **Galaxy 4×8 mesh (#47487):** TP/EP/SP split for 26B-A4B (none wired today).
- **Self-conditioning weight keys:** exact key names / shapes in the DiffusionGemma ckpt (reconciled during #47461 stage 2).

---

### 11. References & conventions

- Methodology: `tech_reports/ttnn/TTNN-model-bringup.md` · `models/docs/model_bring_up.md` · `tech_reports/LLMs/llms.md` · `tech_reports/ttnn/comparison-mode.md`.
- PCC: `tests/ttnn/utils_for_testing.py`. Profiling: `tools/tracy/profile_this.py`.
- Algorithm: transformers `modeling_diffusion_gemma.py`; vLLM blog <https://vllm-project.github.io/2026/06/10/diffusion-gemma.html>.
- **Convention:** commit messages must NOT include a `Co-Authored-By` trailer.

---

## Part II — Device execution spec

### 0. Loop protocol (do this every iteration)

> **Progress lives in Part III of this file** (the *Status by workstream* table). This branch
> no longer has a separate `STATUS.md` — it was folded into Part III below. Everywhere this
> spec says "update the status table" it means that Part III table. Confirm with the user which
> branch the device loop runs on before iteration 1.

1. **Read the Roadmap** (top of this file) → find the first incomplete step on the critical
   path / the active phase. W1–W4 are mostly done; the live work is **Phase 2 integration**
   (#47464), and **Phase 3 correctness is gated by the #48291 decision** — surface that, don't
   silently pick around it. Cross-check the Part III status table for per-workstream detail.
2. **Do the smallest shippable increment** of that task (one module / one device test).
3. **Validate on device** (recipe §1). For RUN-first integration increments, assert the
   executable surface directly (exit 0, `DG_TEXT_DEMO_SUCCESS`, block/position fields,
   no uncaught `DG_TEXT_DEMO_FAILURE`). For correctness increments, the oracle is always
   `reference/` — assert the ttnn output matches it (PCC or `torch.equal`), never assert
   against a fresh guess.
4. **Update the Part III status table**: flip the row, note the test file + measured PCC + date.
5. **Commit ONLY IF commits are explicitly enabled** for this loop (user/loop config says
   so) AND the device test in step 3 passed. Never commit a half-finished increment, a
   failing test, or an environment workaround (e.g. the erisc reset, a local build). If
   commits are not enabled, leave the work in the tree and record progress in the Part III
   table instead. When you do commit: see §2 ground rules — **NO `Co-Authored-By` trailer**.
6. If blocked (HW fault, missing op, oracle disagreement you can't resolve in one
   step): write the blocker into the Part III table under the workstream, leave the row 🚧
   with a one-line reason, and move to the next *independent* task if one exists;
   otherwise stop the loop and surface the blocker.
7. **Never mark a row ✅ without a passing device test** (or, for a deliberately
   host-side fallback, a passing CPU test + a one-line note saying why it's host-side).

**RUN-first definition of done:** a single entry point runs prompt → prefill →
denoise → commit → decode on QB2 with real 26B weights, and the regression target
asserts the success marker plus committed-block/position advancement fields. This
is now covered by the short- and long-prompt two-block smokes in
`tests/test_device_text_demo_run.py` and the `bh-diffusion-gemma-run-smoke` QB2
pipeline entry. The older trajectory-PCC requirement remains the **correctness**
definition of done and is deferred to #48291 / the correctness track.

---

### 1. Environment + device run recipe (QB2, `bh-qbge-06`)

```bash
source /home/zni/venvs/tt-diffusion-gemma/bin/activate
export PYTHONPATH=/home/zni/tt-metal:/home/zni/tt-metal/ttnn
export TT_METAL_HOME=/home/zni/tt-metal
export TT_METAL_RUNTIME_ROOT=/home/zni/tt-metal
export MESH_DEVICE=P150x4          # QB2 = 4× Blackhole, TP=4
export DG_RUN_DEVICE=1             # device tests skip unless this is set
```

- **Build consistency matters** (this bit us once): if the prebuilt `.so` and the
  source kernels drift, JIT compile fails with a `tt_memmove` overload mismatch in
  the permute reader. Fix = build the source tree:
  `./build_metal.sh --disable-profiler`, then run with the PYTHONPATH/runtime-root above.
- **erisc 29-25 teardown re-hangs each device run** (board fw 19.9.0 ahead of tt-metal's
  tested 19.5.0; env quirk, not our bug). **Reset between device runs**; use
  `@pytest.mark.use_module_device` so a test opens/closes the mesh **once**. Minimize
  device churn — batch assertions into one device session where possible.
- QB2 box is **shared**. If `ttnn` can't see 4 devices, another job holds them — wait/retry.

---

### 2. Ground rules

- **QB2 only.** No Galaxy (4×8) work in this loop (that's #47487's later half).
- **Oracle = `reference/`.** It's reconciled bit-for-bit vs the canonical HF source
  (`reference/_upstream.py` guards drift). If device output disagrees with `reference/`,
  the device code is wrong until proven otherwise — but if you suspect the oracle,
  check `reference/_upstream.py` / `tests/test_upstream_parity.py` before "fixing" device code.
- **Determinism = inject noise.** Token-for-token PCC requires feeding the torch run's
  **exact** Gumbel noise + random-renoise token ids into the device path (on-device RNG
  won't match torch bit-exactly). `tt/sampling.gumbel_max` already takes injected `noise`.
  Reserve regenerated-noise runs for distributional checks only.
- **Exclusive prefix.** Entropy-budget accept is `(cum - sorted_vals) <= budget`
  (HF `accept_canvas`), NOT inclusive `cum <= budget`. Already correct in
  `tt/sampling`-adjacent code and `test_device_entropy_accept.py` — keep it.
- **Commits: NO `Co-Authored-By` trailer** (project requirement). Conventional-commit
  style, scope `diffusion_gemma`, reference the issue number.
- **bf16/MoE/TP=4 precision ceiling is known:** backbone logits PCC ≈0.877 with only
  ~50% argmax-match. Don't chase >0.88 logit PCC — that's a separate precision
  follow-up. DO measure whether this ceiling **flips diffusion decisions** (§3).

---

### 3. The decision-fidelity bar (the thing this model actually needs)

Logit PCC is necessary but **not sufficient** — DiffusionGemma's correctness lives in
the per-step accept/remask **decisions**, which bf16/bfp8 small-probability drift can
flip. For every device step, `compare_trajectories` must check, against the oracle:

- clean argmax canvas, per-token entropy, Gumbel-sampled ids, **accept mask**, renoised canvas.

**Open decision (escalate, do not silently pick):** the pass/fail bar for
**accept/remask flips under bf16 (and later bfp8)** is a product-correctness call, not
an engineering default. Until set, **record the flip count** (`#positions where device
accept != oracle accept`, per step and summed over the block) in every trajectory test
and in the Part III status table. Target hypothesis = **0 flips over the block under bf16**; treat any
nonzero flip as a finding to report, not to suppress.

---

### W1 — #47474 KV-cache phase state machine  ✅ done **(was the prereq for W2/W3)** — residual: bounded-sliding commit-append wrap exercised but not yet verified (see Roadmap Phase 1 + Part III fix verification)

**Why first:** gemma4 (re)writes KV on every forward and uses a bounded-sliding
circular cache that would *wrap/corrupt* on a commit-append. The denoise loop reads a
frozen prefix while recomputing canvas K/V every step — that storage discipline does
not exist yet, and W2/W3 can't run correctly without it.

**Define three KV storage classes** (per `AGENTS.md` net-new #2):
1. **Frozen prompt/committed KV** — paged cache, **read-only** during denoise.
2. **Per-step canvas K/V** — recomputed every denoise step (256-token mini-prefill
   against the frozen prefix). **Ephemeral / scratch zone. MUST NOT be written into the
   frozen cache during denoise** (writing it corrupts the prompt KV).
3. **Commit-append** — write the *finished* canvas's KV into the cache exactly once.

**Reuse / touch:**
- `models/demos/gemma4/tt/attention/kv_cache_hybrid.py`, `kv_cache.py` — the existing
  hybrid sliding/full cache. Add a phase flag (`PREFILL_WRITE` / `DENOISE_READONLY` /
  `COMMIT_APPEND`) rather than forking the cache.
- `models/demos/gemma4/tt/model.py:479` `__call__` — already has `is_decode`,
  `sequential_kv_write`, `page_tables_per_layer`. Thread a `kv_phase` through here.
- Specify the page / circular-buffer mapping for **both** local (sliding) and
  full-attention layers — the canvas scratch zone sizing differs per layer type.

**Acceptance (device):**
- A test that runs prefill(write) → denoise(read-only, N steps, canvas K/V recomputed) →
  commit(append), and asserts: (a) the **populated frozen prompt KV region** (the prompt
  positions' written K/V entries) is **byte-identical** before vs after the denoise phase
  (no corruption). **Scope the byte check to that region only** — exclude scratch/canvas
  zones, unused page padding, page metadata, and any uninitialized allocator memory, or it
  false-positives on allocator/uninit differences. (b) after commit the cache contains
  prompt+canvas KV matching a reference re-encode (PCC, not byte-exact — recompute differs
  in low bits). New: `tests/test_device_kv_phase.py`.
- Document the per-chip scratch-zone bytes for the canvas K/V (feeds #47487's 256K budget).

---

### W2 — #47462 bidirectional canvas attention (integration)  ✅ W2a · ✅ W2b (regular SDPA, no new kernel)

**State:** mask geometry reference (`reference/attention_mask.py`, 8 tests) ✅ and an
**isolated** non-causal SDPA spike (`test_device_bidirectional_sdpa.py`, 4/4) ✅ are done.
**Net-new here = wiring the non-causal path into the real attention module**, not another
isolated SDPA call.

**The problem:** gemma4 prefill SDPA is hardcoded `is_causal=True`
(`models/demos/gemma4/tt/attention/prefill.py:126,264`, `operations.py:333`). Add a
non-causal path driven by an explicit `attn_mask`.

#### ⚠️ CANONICAL DENOISE MASK = HF BIDIRECTIONAL VISIBILITY (read before touching the mask)

`reference/attention_mask.py` is the oracle. After the 2026-06-29 H2 review against installed
transformers, the rule is:

- **Full-attention layers:** all-attend `[C, P+C]` (zeros / maskless fast path).
- **Sliding-attention layers:** short prompts with `P+C-1 <= sliding_window` are also all-attend,
  but long prompts must use HF's bidirectional sliding visibility
  `abs(q_idx - kv_idx) <= sliding_window`, which drops prompt tokens before the window.
- **Do not pass `sliding_window_size` in the denoise path.** `attn_mask` and
  `sliding_window_size` remain mutually exclusive in the ttnn SDPA op
  (`sdpa_device_operation.cpp:67-68`), so the sliding visibility is baked into the
  dense mask only when needed.

Use `build_canvas_denoise_mask(..., layer_type="sliding_attention", sliding_window=...)`
for long-prompt sliding layers; use the default/full-attention all-zero mask or maskless
fast path otherwise. `local_window=True` remains a non-canonical op-capability test only.

#### W2a — non-causal masked SDPA, prompt + canvas ≤ 32768  ✅ (the real W2 deliverable)
- Reference non-causal SDPA usage: `models/experimental/pi0/tt/ttnn_gemma.py:320`
  (`scaled_dot_product_attention(attn_mask=…, is_causal=False)`) and
  `models/tt_dit/encoders/gemma/model_gemma.py:253`.
- Add an `is_causal=False` / `attn_mask=` branch to `prefill.py` SDPA; gate by `kv_phase`
  from W1 (denoise → non-causal HF bidirectional visibility, prefill/commit → causal).
- Build the `[256, prompt_len+256]` mask from `build_canvas_denoise_mask(...)` when a
  materialized mask is required. Canvas absolute/RoPE positions are offset by `prompt_len`
  (`canvas_positions`). Cover the canvas→prompt prefix, not just an isolated 256 canvas.
- **Do NOT pass `sliding_window_size` in the denoise path — even for sliding layers.**
  `attn_mask` ⊥ `sliding_window_size` (`sdpa_device_operation.cpp:67-68`); passing both
  trips the mutual-exclusion guard. (`sliding_window_size` stays only on the causal
  prefill/commit paths.)
- **Acceptance (device):** `tests/test_device_bidirectional_attention.py` — a full-attn
  layer and a sliding layer run with denoise `is_causal=False` and no `sliding_window_size`;
  output PCC ≥ 0.99 vs a torch reference forward built from the corresponding
  `build_canvas_denoise_mask(...)`, including the prompt prefix.
- **Optional op-capability test (does NOT gate W2):** a separate
  `test_device_windowed_mask_path` driving `local_window=True` purely to prove the ttnn
  SDPA masked path handles a windowed mask. Clearly label it non-canonical.

#### W2b — long-prompt masked attention, prompt + canvas > 32768  ✅ RESOLVED with regular SDPA (no new kernel)
> 📋 **Detailed spike-first plan + status: [Appendix A](#appendix-a--w2b-long-prompt-attention-resolved).** Outcome: the gating spike **S1 PASSED** — regular non-causal SDPA returns correct results (PCC≥0.99 vs an independent fp32 oracle) at `[256 × Sk]` up to **262144**, so the original "new kernel" framing was wrong. W2b reduced to **D1**: re-key the `prefill.py` `long_seq` guard against K-length + run `is_causal=False` without `sliding_window_size`; materialize a dense mask only when HF sliding-layer visibility requires it. SDPA op, RoPE reachability, and integrated tiny-model denoise (both full & sliding layers) all pass through 262144, wired into the BH QuietBox2 pipeline. Real-26B denoise integration is #47464. **The original framing below is superseded** (kept for the source map / decision history).

**Do NOT bundle this into W2a acceptance.** The existing gemma4 chunked-prefill long-context
path is **causal-only** (`operations.py:25-29`, `prefill.py:106-130`) and `attn_mask` is
mutually exclusive with the windowing it relies on — so a **non-causal masked chunked path
is new kernel/path-level work, not wiring**. Track it as its own risk item: scope a spike
first (can ttnn SDPA chunk a `[256, >32768]` explicit mask at all, or does it need a new
op / a tiled-mask streaming scheme?). Functional milestone for short/medium prompts does
**not** depend on W2b; flag it to the user/manager as a standalone effort + likely-new-kernel risk.

---

### W3 — #47463 discrete-diffusion decode loop (device)  🔴 blocked on bf16 decision bar (control-flow implemented & validated)

**State:** all the *primitives* are validated in isolation — entropy/Gumbel
(`tt/sampling.py`, `test_device_entropy_harness.py`), entropy-budget accept full chain
`sort→cumsum→exclusive-prefix→scatter` (`test_device_entropy_accept.py`, 5/5),
self-conditioning gated MLP on device (`tt/self_conditioning.py`,
`test_device_self_conditioning.py`). **Net-new here = the loop that assembles them**,
running on device, matching `reference/denoise_loop.py` step-for-step.

**The loop (per 256-block, mirror `reference/denoise_loop.py` `StepRecord`):**
1. temperature-scale (HF reversed-step schedule `t_min+(t_max-t_min)·cur_step/N`, 0.8→0.4)
2. Gumbel-max `argmax(logits/T + injected_noise)` (`tt/sampling.gumbel_max`)
3. per-token entropy `H=−Σp·logp` (`tt/sampling.token_entropy`)
4. entropy-budget acceptance (sort-by-confidence + exclusive cumulative-entropy cutoff +
   scatter-back) — the **R1 risk**, validated in isolation; now run it in the loop
5. renoise rejected positions to **random token ids** (no `[MASK]` token) — `where`/`scatter`
6. self-conditioning: prev-step softmax → prob-weighted token-embedding avg → gated MLP
   → add to canvas embeds; **zeroed on encoder passes**
7. **commit = clean argmax** (not the noisy sampled values)
8. **halt** when argmax canvas stable AND mean entropy < `confidence_threshold=0.005`
   (`stability_threshold=1`), else step cap (`max_denoising_steps=48`)

**Watch (from the spike):**
- `min_accept` was **omitted** in the accept spike (host/slice op). Decide in the loop:
  small host-side slice vs on-device — and whether it survives **static Metal Trace** with
  a **data-dependent cutoff** (the unproven part). If trace can't express the data-dependent
  halt/cutoff, a small host readback of the `[256]` accept/entropy vector per step is the
  documented fallback (256 elements, not the `[256, vocab]` logits — keep logits on device).
- Re-run accept at the **real 256 canvas** (the spike used L=128) and in **bf16** end-to-end
  (the spike used fp32 to isolate chain logic) — then record decision-flip count (§3).

**Acceptance (device):**
- `tests/test_device_denoise_loop.py`: one block, fixed seed + **injected** Gumbel/renoise
  ids, run on QB2; `compare_trajectories` matches `reference/denoise_loop.py` on every
  decision class (argmax/entropy/sampled-ids/accept/canvas). Report the per-step and total
  accept-flip count. Logits/probs stay on device (no per-step `[256,vocab]` readback).

---

### W4 — #47472 on-device canvas sampling  ✅ done — SAMP-3 mesh-mapper `TT_FATAL` fixed; residual: regenerated-noise unvalidated at prod vocab (see Roadmap Phase 1 + Part III fix verification)

**State:** `tt/sampling.py` has `temperature_scale`, `token_entropy`, `gumbel_max`,
`softmax` (all device, validated). **Net-new here = the user-facing per-position canvas
sampler + its plumbing through the tenstorrent/vllm TT plugin.**

**Scope (ship the *released* sampling first):**
- Per-position over all 256 canvas positions: temperature schedule (0.8→0.4), Gumbel-max
  candidate draw, seed/reproducibility. This is **not** last-token AR sampling — the AR
  path `models/common/sampling/generator.py` (`SamplingGenerator`, local `SamplingParams`)
  is **reference only**; build a canvas/per-position variant.
- **top-k/top-p is NOT shipped in the reference** (transformers defers it; vLLM PR #45429
  open/unmerged). Treat top-k/p as forward-looking — **do not gate this issue on it**.
- Keep logits/probs **on device** (≤48 steps/block; per-step `[256,vocab]` host readback
  would be host-bound).
- **Boundary vs W3:** W3 owns the denoise *control flow* (accept/renoise/halt/commit).
  W4 owns the *user-controllable sampling primitives* (temperature/Gumbel/seed, top-k/p
  forward-looking) the loop calls, + their plug-through.

**vLLM plumbing (mirror the gemma4 bridge):**
- Reference: `models/demos/gemma4/tt/generator_vllm.py:381` `initialize_vllm_model`.
- Declare `model_capabilities["supports_sample_on_device"]=True`; consume `TTSamplingParams`
  (temperature/top_k/top_p/seed — `TTSamplingParams` lives in vLLM, duck-typed), map onto
  the canvas sampler. (Full TT-model-class registration is #47466 — here just the sampling seam.)

**Acceptance — TWO distinct tests (don't conflate; "matches the distribution under a fixed
seed" alone is not testable and goes flaky):**
- **Deterministic test (primary, exact):** `tests/test_device_canvas_sampling_exact.py` —
  feed the torch run's **injected** Gumbel noise; assert the device sampled ids are
  **token-exact** (`torch.equal`) vs `reference/sampling.py`, and temperature scaling is
  honored bit-close (PCC on `logits/T`). No tolerance knobs — this is the deterministic path.
- **Distributional test (secondary, regenerated noise):** `tests/test_device_canvas_sampling_dist.py`
  — draw **N samples** (state N, e.g. 4096) on device vs torch with the SAME fixed seed
  governing the comparison, and assert with **explicit statistical tolerances**: per-position
  argmax-frequency within ε, and a KL or KS bound on the sampled-id histogram. Pick N + ε so
  the test is robust (no flaky single-draw asserts). Document the chosen N/ε in the test.
- Both: temperature honored; **no per-step full-canvas `[256, vocab]` logit readback**.

---

### Sequencing summary

```
W1 (#47474 KV phase machine)  ──► W2 (#47462 bidirectional attn) ──► W3 (#47463 decode loop) ──► W4 (#47472 sampling seam)
   prereq for everything           needs W1 phase flag              needs W1+W2              calls W3's primitives
```

W4's *primitives* already exist, so its **sampler** can be drafted in parallel once W3's
loop shape is settled; its **vLLM plumbing** can land last. Everything else is strictly ordered.

**When all four are ✅:** run the end-to-end block test (prefill→denoise→commit, trajectory
PCC), update the Part III status table + the parent tracker #47452, and report the decision-flip
numbers so the §3 bar can be set. Then stop the loop.

---

## Part III — Implementation status

Maps the **Part I** workstreams to what is implemented in this
directory. Updated as work lands so progress is trackable per commit.

### Environment constraints (read first)

This box is **`bh-qbge-06` — a QB2 (4× Blackhole `p300c`, `/dev/tenstorrent/0..3`)**, so **device work is NOT blocked on hardware**. The remaining gates are software + data:

- **Dedicated env:** `/home/zni/venvs/tt-diffusion-gemma` (Python 3.12, **transformers 5.12.1** — bumped from 5.10.2 on 2026-06-23, torch 2.11+cpu, ttnn editable from the repo) — isolated from the default `python_env` (4.53.0 for LTX). Verified at 5.12.1: `transformers.models.gemma4` imports, `transformers.models.diffusion_gemma` imports, **`ttnn` sees 4 QB2 devices**, **64 reference tests pass**. Use: `source /home/zni/venvs/tt-diffusion-gemma/bin/activate && export PYTHONPATH=/home/zni/tt-metal TT_METAL_HOME=/home/zni/tt-metal`.
- **`diffusion_gemma` SHIPS since transformers 5.12** (absent in 5.10.2): at **5.12.1 the working env can load the real `DiffusionGemmaForBlockDiffusion` directly** — no separate transformers-main env needed (the `dg-tf-main` 5.13.0.dev0 venv remains as a cross-check). `from_pretrained` takes `dtype=` (primary since 5.12; `torch_dtype` kept for BC). The canonical source is also vendored at `/home/zni/dg_ref_src/` and used to reconcile the `reference/` layer 1:1; `reference/_upstream.py` is the bit-for-bit parity guard.
- **Checkpoints NOW downloaded (2026-06-22, ungated — `gated=False` on HF):**
  - `google/gemma-4-26B-A4B-it` — 51.6 GB, **Stage-1 stepping-stone only** (sanity that the reused gemma4 path runs + reproduces HF gemma4 on QB2). **NOT the #47461 target**: passing on this ckpt does not validate DiffusionGemma. Verified complete + openable.
  - `google/diffusiongemma-26B-A4B-it` — 51.7 GB, **the #47461 target ckpt** — backbone PCC must be measured on THIS (fine-tuned weights + extra self-cond + bidirectional denoise all differ from plain gemma). Carries the stage-2 weight mapping + self-cond weight values.
  - `google/gemma-4-12B-it` — dense, the QB2 device-flow proof (smaller, no MoE skip).
- **QB2 is present** (4× Blackhole, this box); fitting 26B-A4B on QB2 (1×4) is itself net-new (#47487), and the in-repo gemma4 **12B** path is QB2-supported and can validate the on-device flow on this exact HW first.

So work proceeds **env-independent-first**: pure-torch reference logic + config
+ tests that run on CPU, with checkpoint/transformers-gated pieces scaffolded
and marked `TODO(env)`. **HW + env + checkpoints are no longer blockers — QB2 is local, the dedicated transformers-5.12.1 env is built, and all three checkpoints are downloaded (ungated, 2026-06-22).**

### Status by workstream

| Item | Plan | Status |
|---|---|---|
| Module scaffolding | — | ✅ package + config |
| `config.py` (verified hyperparams) | §2 | ✅ done |
| Config reconciliation vs real 26B-A4B config.json (`from_hf_config`) | #47461 | ✅ done — `tests/test_config.py`, all fields confirmed in sync |
| **Diffusion sampling primitives (reference, pure torch)** | #47463 spike / #47468 oracle | ✅ **reconciled 1:1 vs canonical source** (2026-06-22) — exclusive-prefix entropy-bound accept (`cum-e<=bound`), HF reversed-step temperature, multinomial `sample_canvas` + Gumbel-max equivalence. `reference/sampling.py` |
| PCC trajectory harness (validates decisions) | #47468 | ✅ done — `tests/trajectory_pcc.py` |
| **Upstream parity guard (drift oracle)** | #47468 | ✅ **NEW** — `reference/_upstream.py` (verbatim canonical extractions) + `tests/test_upstream_parity.py`; reference matches HF bit-for-bit (temperature / accept / confidence / self-cond) |
| HF reference adapter seam | #47468 | ✅ done — `reference/hf_reference.py` (real load when transformers-main present; reconciled `reference/` is the env-independent oracle) |
| **Config reconciliation vs generation_config** | #47468/#47463 | ✅ **NEW** — all TODO(confirm) resolved: `confidence_threshold=0.005`, `stability_threshold=1`, `t_max/t_min`, `entropy_bound`, + `intermediate_size=2112`, `num_global_key_value_heads=2`, `global_head_dim=512` |
| Causal backbone bring-up (gemma4 reuse) — code | #47461 | ✅ **code enabled**: QB2=`MESH_DEVICE=P150x4`; gemma4 path mesh-agnostic; weight-remap keyset validated. Device PCC = the three rows below. |
| **DiffusionGemma→gemma4 weight remap + self-cond loader** | #47461 (N4) | ✅ **NEW** — `weight_mapping.py` + `SelfConditioning.load_from_state_dict`; **validated vs real ckpts**: remapped backbone == gemma4 keyset exactly; 4 self-cond tensors load with config shapes. `tests/test_weight_mapping.py` |
| **Self-conditioning gated MLP (reference)** | #47461/#47463 | ✅ **reconciled** — added `pre_norm` + scaleless `post_norm`; forward is `post_norm(emb+gated_mlp(pre_norm(signal)))` (was a bare delta). `reference/self_conditioning.py` |
| **QB2 memory budget + batch ceiling** | #47487 | ✅ **NEW doc** `QB2_MEMORY_BUDGET.md`: ~32 GB/chip (8×4 GB banks); experts sharded-vs-replicated is the fit gate (code favors sharded → fits); EP is the fallback. Empirical measure pending device |
| QB2 fit + run (no OOM; experts sharded) — **plain gemma ckpt** | **#47487** | ✅ done — `gemma-4-26B-A4B-it` ran on `P150x4` TP=4 (110 s, no OOM). **HW-enablement fact, NOT a DiffusionGemma validation.** |
| Causal backbone PCC — **Stage-1 (gemma4 ckpt)** | #47461 (stage 1) | ✅ stepping-stone — 0.8665 vs HF (threshold 0.83). Subsumed by Stage-2; the ~0.87 ceiling is now confirmed **shared-backbone** (not ckpt-specific) — a bf16/MoE/TP=4 precision follow-up. |
| Causal backbone PCC — **Stage-2 (DiffusionGemma ckpt)** — the real #47461 gate | #47461 (stage 2) | ✅ **measured on QB2 (2026-06-24)** — `tests/test_device_backbone_pcc.py` (`-k 1x4`, TP=4): logits PCC **0.877** (5-tok) / **0.847** (24-tok) vs the HF DiffusionGemma causal backbone (`model.model.encoder`→`lm_head`→softcap), passes the 0.83 baseline. ≈ plain-gemma 0.866 ⇒ fine-tuned weights add **no** extra error; argmax-match ~50% is the shared-backbone bf16/MoE/TP=4 ceiling (precision follow-up, not DG-specific). Bidirectional forward → #47462. |
| KV-cache phase state machine | #47474 | ✅ done — `KVCachePhase` plumbing landed through Gemma4 model/layer/attention and Generator-compatible prefill/decode/verify wrappers; explicit `DENOISE_READONLY` skips cache writes. Validated 2026-06-25 with `tests/test_kv_phase.py` (3 passed), QB2 `test_single_layer_model[blackhole-sliding_only-1x4]` PCC **0.999936**, and QB2 `tests/test_device_kv_phase.py`: readonly denoise leaves prompt K/V frozen-region byte-identical; `COMMIT_APPEND` decode writes the next cache position without mutating the prompt region; a 256-token canvas commit loop writes the full canvas region; 256-token commit-append canvas K/V matches one-shot re-encode by PCC; bounded sliding commit-append now uses a discriminating wrap test (`position=sliding_window+block_size`) that proves the wrapped physical slot changes while the no-wrap slot stays untouched. Canvas K/V scratch sizing added in `memory_budget.py`: QB2 TP=4 bf16 batch=1 ≈ **15 MiB/chip**. Page/circular-buffer mapping added in `kv_phase.py`: full-attn commit uses absolute positions; sliding commit uses `absolute_pos % sliding_window`. |
| Canvas mask geometry (reference, pure torch) | #47462 | ✅ done — `reference/attention_mask.py`, 8 tests pass |
| Bidirectional canvas SDPA on QB2 (device) | #47462 | ✅ **validated on QB2** — 4/4 PCC≥0.99 (full / symmetric-window / prompt-visible / GQA 16-8) on sfpi 7.60.0. ⚠️ device *teardown* re-hangs erisc 29-25 → reset between device runs. NOT a firmware issue: board fw is **19.9.0** (newer than tt-metal's tested 19.5.0); the assert's "min 18.10.0" is a hardcoded boilerplate string, not a version readout. Root cause undiagnosed (possibly fw ahead of the local UMD checkout); treat as an env quirk, work around with reset. |
| Self-conditioning gated MLP (reference, pure torch) | #47461/#47463 | ✅ done — `reference/self_conditioning.py`, 6 tests pass |
| Entropy-budget acceptance on QB2 (device) | #47463 (R1) | ✅ **validated on device (2026-06-22) — full chain `ttnn.sort`→`cumsum`→exclusive-prefix→`scatter` matches the oracle, 5/5 (`test_device_entropy_accept.py`).** The 2026-06-19 "device `ttnn.sort` returns garbage" conclusion was **WRONG** — it was a **degraded-board** artifact (erisc 29-25 fault), not a `ttnn.sort`-on-BH bug. On healthy HW `ttnn.sort` is correct: `test_sort_standard[…64…]` all pass; standalone repro (bf16/fp32, 2D `[64,64]`, 4D `[1,1,64,64]`, `[…,256]`) gives correct values+indices. **Host-side sort is unnecessary — the device chain works.** Two things were needed to validate: (1) a **consistent build** — the prebuilt `.so` (dev20260616) JIT-compiled source kernels (dev20260618) against its own headers → `tt_memmove` overload mismatch in the permute reader kernel; fixed by building the source tree (`build_metal.sh --disable-profiler`, run with `PYTHONPATH=$TT_METAL_HOME/ttnn:$TT_METAL_HOME` + `TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME`); (2) the device chain must use the **exclusive** prefix `(cum - sorted_vals) <= budget` to match HF `accept_canvas`, not inclusive `cum <= budget` (off-by-one at the boundary element). (Teardown still re-hangs erisc 29-25 each run — see SDPA row — so minimize device churn.) |
| Multi-canvas generation loop (reference, pure torch) | #47464 | ✅ done — `reference/generate.py`, 3 tests (commit-append, prefix-grows) |
| Bidirectional canvas attention (device SDPA integration, short/medium prompts) | #47462 (W2a) | ✅ **validated on QB2** — mask reference done; isolated non-causal SDPA spike is ✅ (`test_device_bidirectional_sdpa.py`, 4/4). Real Gemma4 prefill attention now accepts explicit `attn_mask` and routes to `is_causal=False` without `sliding_window_size`; rectangular denoise support lets canvas Q attend `[prompt; canvas]` K/V with canvas RoPE offset. `tt/denoise_forward.py` exposes W2 product wrappers: `denoise_attention_forward`, `denoise_logits_forward`, `denoise_logits_from_tokens`, `collect_prompt_hidden_by_layer` (legacy hidden-source shim), `collect_prompt_kv_by_layer` (projected prompt K/V), and `read_prompt_kv_cache_slice` (non-paged Gemma4 KV cache → projected prefix K/V via `ttnn.experimental.nlp_kv_cache_load_slice`). Validated with `tests/test_device_bidirectional_attention_integration.py` (4 passed): square all-attend smoke; prompt-prefix attention PCC≥0.99 for both `sliding_attention` and `full_attention`; token-driven full-canvas logits wrapper PCC≥0.98 after `PREFILL_WRITE` writes the prompt cache and denoise reads that cache slice, plus real `TtSelfConditioning` softmax→embedding→gated-MLP hook on mesh (full logits include known bf16 MoE/lm_head ceiling). `tests/test_device_self_conditioning.py` still passes 4/4 for the standalone module. |
| Paged / long-prompt denoise cache reader + masked chunking | #47462 (W2b) | ✅ done — S1/S2 resolved the core SDPA risk in favor of D1: regular non-causal SDPA passes `[256 × Sk]` through `Sk=262144`, `head_dim ∈ {256,512}` on QB2, both maskless and explicit masked. D1 is wired so full layers and short-prompt sliding layers keep the maskless fast path, while long-prompt sliding layers materialize HF's bidirectional sliding mask (`abs(q-k) <= sliding_window`). S4 RoPE reachability is verified at 262144, and integrated tiny-model denoise attention PCC passes at `P+C=33280` and `P+C=262144` for both full and sliding layers. Full W2b suite: `test_device_long_sdpa_w2b.py` 29 passed with `DG_W2B_SDPA_SWEEP=full`, and the same full-sweep command is now in the QB2 Blackhole pipeline (`bh-diffusion-gemma-w2b-full-sweep`). Real 26B e2e generation remains #47464/#48291, not a W2b SDPA blocker. |
| Reference denoise trajectory (pure torch) | #47463/#47468 | ✅ done — `reference/denoise_loop.py`, 4 tests pass |
| Discrete-diffusion decode loop (device) | #47463 | 🔴 **blocked on bf16 decision bar / full-logits precision** — local `ttnn` build unblocked by syncing `tt_metal/third_party/tracy` and `tt_metal/third_party/umd` to the superproject pins, then rebuilding with `./build_metal.sh --disable-profiler`. W3 control-loop implementation is in place and validated on QB2 (2026-06-25): `tests/test_device_denoise_loop.py` 3 passed, entropy/accept harnesses 12 passed, and real-W2-logits integration tests passed (`test_denoise_logits_adapter_threads_prev_logits_for_self_conditioning`, `test_denoise_controller_real_logits_records_decision_flips`). `tt/denoise_loop.py` composes Gumbel-max, logsumexp-form entropy, exclusive-prefix accept, uint32-safe renoise, multi-step carry, clean-argmax commit, and stable+confident halting against `reference/denoise_loop.py`; the synthetic trajectory smoke uses 256 canvas positions, injected zero Gumbel + renoise ids, halts after 2 steps, passes `compare_trajectories`, and records 0 accept flips. `tt/denoise_forward.py` exposes `DenoiseLogitsAdapter`, a stateful W2 callback that threads previous-step logits into real self-conditioning for the controller while keeping logits on device; it also accepts controller-shaped `[1,1,L,1]` TILE token canvases. Real W2 logits smoke (1-layer, vocab=256, 2 denoise steps) runs end-to-end and records **accept_flips=[0,0]**, but also a precision finding: **argmax_flips=[225,222]**, **canvas_flips=[1,1]**, entropy PCC≈[0.624,0.653] vs torch. Triage shows drift is already present at logits: logits PCC≈[0.985,0.969] but logits argmax agreement≈[0.121,0.133]; reference top1/top2 margin is tiny (~0.005) while TT-vs-torch logits mean|Δ| is ~1.86/2.64, so argmax is margin-limited. Hidden-vs-logits diagnostic shows final hidden PCC≈0.9887 before lm_head, so this is not isolated to softcap/lm_head; dense (MoE-disabled) diagnostic improved logits PCC (~0.995/0.984) but still had ~0.125 argmax agreement and even accept_flips=[2,2], so this is not MoE-only. W3 should not be marked ✅ until either backbone/full-logits precision drift is reduced enough for the decision bar, or the bf16 diffusion decision bar is explicitly accepted/escalated by product. Since control-flow is implemented and blocked on that decision, the loop can proceed to independent W4 sampler work. |
| On-device canvas sampling | #47472 | ✅ done — deterministic exact path validated on QB2 (2026-06-25): `tests/test_device_canvas_sampling_exact.py` 3 passed. `tt/sampling.py` exposes `canvas_sample(logits, temperature, injected_gumbel_noise)` as the W4 released per-position draw (`argmax(logits/T + gumbel)`); tests feed the torch run's injected Gumbel noise and match sampled ids token-exact, including the params-routed seam, plus verify temperature scaling PCC≥0.9999. W4 sampling-params seam is in place: `tt/sampling_params.py` exposes `MODEL_CAPABILITIES["supports_sample_on_device"]=True`, duck-types vLLM `TTSamplingParams` fields (temperature/top_k/top_p/seed) into a per-step `CanvasSamplingConfig`, and `canvas_sample_from_params(...)` maps those params onto the device sampler; `tests/test_sampling_params.py` 5 passed. Seed-regenerated sampling now defaults to the permuted-vocab RNG path, which keeps one `ttnn.rand` draw per logits element but generates vocab as an outer axis before permuting back; explicit distributional tolerances pass on QB2 for both direct and params-routed paths (`N=4096`, max top1-frequency error≈0.0282, mean KL≈0.0129), while the slower vocab-chunk diagnostic also passes (`max top1-frequency error≈0.0324`, mean KL≈0.0035). The raw single-call `ttnn.rand[..., vocab]` path remains as a strict-xfailed diagnostic (`max top1-frequency error≈0.179`, mean KL≈0.651`) because torch consuming the same raw noise exactly reproduces the biased samples, proving the issue is RNG axis correlation rather than sampler arithmetic/argmax; this raw path is not the released params default. |
| Functional e2e / perf / vLLM / batched / multimodal / quant / CI | #47464+ | 🚧 started — CPU e2e outer-loop oracle is now pinned against the real HF `DiffusionGemmaGenerationMixin.generate` outer loop without loading 26B weights: `test_reference_generate_blocks_matches_hf_generate_outer_loop` uses a fake HF generation model that keeps HF sampler/denoise/commit-append semantics and a prefix-length-dependent logits model, then asserts `reference.generate_blocks` commits the same multi-block token sequence. Validated with `test_generate.py` + `test_hf_reference.py` + `test_real_transformers_parity.py` (15 passed, 1 skipped). Per-block position plumbing started: `denoise_attention_forward` / `denoise_hidden_forward` / `DenoiseLogitsAdapter` now accept explicit `q_rope_offset`, so block N can use `prompt_len + N*256` instead of being tied to the current prefix tensor length; CPU wiring tests passed (7/7) and QB2 denoise-attention regression passed for sliding+full (2/2). Device commit-append is now in `tt/` instead of being test-only: `commit_canvas_tokens` appends host committed ids through Gemma4 decode with `KVCachePhase.COMMIT_APPEND`, and QB2 `test_commit_append_canvas_kv_matches_reencode_pcc` passed (1/1). `denoise_and_commit_block` now composes one generated block: it sets adapter `q_rope_offset=start_pos`, runs W3 `denoise_block`, commits the clean argmax canvas, and returns `next_pos`. `generate_blocks` now loops that helper across blocks with injected init canvases, concatenates committed host tokens, and advances `next_pos`; `host_canvas_to_device` / `make_host_canvas_init_fn` let device-vs-HF tests replay fixed torch/HF initial canvases in controller layout. QB2 `test_generate_blocks_runs_device_denoise_and_commit` passed: two 32-token blocks run W3 denoise + commit through `generate_blocks`, produce expected committed tokens, advance `next_pos`, and write both cache block regions. Demo bring-up now has a build-only smoke path; QB2 `text_demo --checkpoint /home/zni/dg_models/diffusiongemma-26B-A4B-it --local-files-only --mesh P150x4 --num-layers 1 --max-seq-len 512 --build-only` loaded one real checkpoint layer, built the TT model, and closed the mesh successfully on 2026-06-29. Generated-token demo smoke can now request small `DiffusionConfig` knobs and initializes FABRIC_1D for multi-device TP; this advanced the QB2 command `text_demo --checkpoint /home/zni/dg_models/diffusiongemma-26B-A4B-it --local-files-only --mesh P150x4 --num-layers 1 --max-seq-len 512 --canvas-length 32 --max-denoising-steps 1 --max-new-tokens 1` past embedding all-gather, then exposed the next blocker in shared Gemma4 prefill: short prompt length 18 is padded to K=32 while V remains 18, so SDPA rejects mismatched K/V sequence lengths. Device-vs-HF short-prompt acceptance, tokenizer/prompt prefill/production canvas RNG, and generated-token `text_demo` remain open. |

> **Status refresh — 2026-07-02 (#47464 RUN-first):** the historical Functional e2e row above is superseded for RUN bring-up. The generated-token demo no longer stops at the padded-K/V mismatch blocker: `tests/test_device_text_demo_run.py` now covers prompt string → prefill → real-checkpoint denoise adapter → commit append → decode on QB2. The short-prompt forced two-block smoke is validated at full 30-layer depth (`generated_tokens=512`, `blocks=2`, `prompt_len=32`, `next_pos=544`) and the long-prompt maskless smoke is validated at full 30-layer depth (`generated_tokens=64`, `blocks=2`, `prompt_len=1024`, `next_pos=1088`). Both are wired into `bh-diffusion-gemma-run-smoke` as reduced-depth CI smokes. 256K-KV/full-depth/canonical-canvas generation also now passes in RUN-first argmax mode for two blocks (`generated_tokens=512`, `next_pos=544`). Remaining work in this row is no longer "make it run"; it is correctness (#48291), production device RNG / intra-block device-side early stop, near-limit prompt hardening, vLLM, batching, perf, multimodal, quant, and CI hardening.

Legend: ✅ done · 🚧 in progress · ⛔ blocked on environment · ⬜ not started

### Session 2026-06-29 — #47464 generated-token demo smoke

Tick 86 advanced the real-checkpoint tiny generated-token demo:
`text_demo --checkpoint /home/zni/dg_models/diffusiongemma-26B-A4B-it --local-files-only --mesh P150x4 --num-layers 1 --max-seq-len 512 --canvas-length 32 --max-denoising-steps 1 --max-new-tokens 1`.
Two prefill blockers were fixed: Gemma4 prefill RoPE outputs are restored to their logical sequence length so K/V lengths stay consistent, and DiffusionGemma prompt prefill pads device input tokens to a 32-token tile while preserving the real prompt length for block positions. Host validation passed (`test_prefill_prompt_tokens_embeds_and_writes_kv`), and the QB2 smoke now reaches denoise adapter construction. The next blocker is non-32-aligned prompt KV extraction: `read_prompt_kv_cache_slice(prompt_len=18)` rejects non-tile-aligned bounds.

### Session 2026-06-30 — #47464 R0.1 full-26B build fit

R0.1 passed on QB2 with the real DiffusionGemma 26B-A4B checkpoint:
`text_demo --checkpoint /home/zni/dg_models/diffusiongemma-26B-A4B-it --local-files-only --mesh P150x4 --max-seq-len 512 --build-only`.
The full 30-layer model loaded through layer 29, initialized on-device sampling for vocab 262144, and closed the mesh cleanly. DRAM logging now records `ttnn.get_memory_view` before and after build; the measured per-chip budget was baseline **0.000 GiB used / 31.867 GiB total**, post-build **13.236 GiB used / 18.631 GiB free / 31.867 GiB total**. Next R0 step: short-prompt causal prefill through all 30 layers (R0.2).

### Session 2026-06-30 — #47464 R0.2 full-26B prompt prefill

R0.2 passed on QB2 with the same real checkpoint using the new prefill-only demo mode:
`text_demo --checkpoint /home/zni/dg_models/diffusiongemma-26B-A4B-it --local-files-only --mesh P150x4 --max-seq-len 512 --prefill-only`.
The full 30-layer model built, tokenized the default chat prompt to `prompt_len=18`, padded/wrote a `cache_len=32` frozen-prefix span through `prefill_prompt_tokens`, and closed the mesh cleanly. DRAM was baseline **0.000 GiB used / 31.867 GiB total**, post-build **13.236 GiB used / 18.631 GiB free / 31.867 GiB total**, post-prefill **13.237 GiB used / 18.630 GiB free / 31.867 GiB total**. Next R0 step: build the real-checkpoint denoise logits adapter and call it once on a 256-token canvas (R0.3).

### Session 2026-06-30 — #47464 R0.3 adapter-only path ready, QB2 blocked

R0.3 now has a narrow adapter-only demo mode:
`text_demo --checkpoint /home/zni/dg_models/diffusiongemma-26B-A4B-it --local-files-only --mesh P150x4 --max-seq-len 512 --canvas-length 256 --adapter-only`.
The path builds the full TT model, pre-fills the prompt KV, constructs the real-checkpoint generation logits builder (`make_generation_logits_fn_builder_from_checkpoint_state`), creates a seeded host canvas, converts it to the W3 controller layout, calls `adapter(canvas, step=0)` once, logs the logits shape, then resets/deallocates the adapter logits and canvas. CPU validation passed via `test_text_demo.py`.

The first QB2 run did not reach the adapter call: UMD blocked on `CHIP_IN_USE_0_PCIe`, held by another user's `VLLM::EngineCore` (PID 1085625 at the time). The local waiting demo process was stopped so it would not queue behind the lock.

### Session 2026-06-30 — #47464 R0.3 real adapter fit failure

After clearing the stale vLLM owner and recovering the known ERISC 29-25 hang with `sudo /home/zni/.local/bin/tt-smi -r`, R0.3 reached the real adapter path. The full 30-layer model built, prompt prefill passed (`prompt_len=18`, `cache_len=32`), and post-build DRAM was still **13.236 GiB used / 18.631 GiB free / 31.867 GiB total**. The failure is not a top-level DRAM OOM: `adapter(canvas, step=0)` fails in first-step self-conditioning post_norm, where `ttnn.rms_norm` on the production `[1,1,256,2816]` canvas hidden tensor reports `Statically allocated circular buffers ... clash with L1 buffers` (`L1 buffer allocated at 424320`, static CB end `533248`).

Two exact-math hardenings landed locally before retest: `prev_logits=None` now skips the zero-signal gated MLP and returns `post_norm(inputs_embeds)` directly, and self-conditioning RMSNorm requests DRAM outputs / sequence chunks. CPU tests pass, but the production-width RMSNorm program still does not fit even at 32-token chunks, so R0.3 remains blocked on a sharded-width RMSNorm or a dedicated self-conditioning norm program.

### Session 2026-06-30 — #47464 R0.3 progressed to denoise SDPA fit blocker

Follow-up R0.3 hardening moved the adapter-only failure point substantially forward. Self-conditioning RMSNorm now uses a width-sharded RMSNorm path for production-width `[1,1,32,2816]` chunks, denoise layer norms are called through a 32-token chunked wrapper, the MoE router's scaleless norm goes through the same sharded RMSNorm helper, and denoise RoPE is computed with a diffusion-local chunked `x*cos + rotate_half(x)*sin` path to avoid the rotary embedding kernel's static-CB footprint. Denoise SDPA was also reduced to 32x32 chunks, requested DRAM output, and forces Q/K/V to DRAM before launch.

CPU validation passed with:
`pytest models/experimental/diffusion_gemma/tests/test_gemma4_prefill_guards.py models/experimental/diffusion_gemma/tests/test_denoise_forward.py models/experimental/diffusion_gemma/tests/test_tt_self_conditioning.py models/experimental/diffusion_gemma/tests/test_text_demo.py -q` (**39 passed**).

QB2 R0.3 still fails, but no longer in self-conditioning or RoPE. The latest `--adapter-only` run builds the full 30-layer model, pre-fills the prompt (`prompt_len=18`, `cache_len=32`), and then fails in first-layer denoise `ttnn.transformer.scaled_dot_product_attention` with `Statically allocated circular buffers ... clash with L1 buffers` on core range `[0-0 - 7-0]` (`L1 buffer allocated at 423872`, static CB end `457472`). This is not a top-level DRAM OOM; post-build DRAM remains **13.236 GiB used / 18.631 GiB free / 31.867 GiB total**. Next fit step is to avoid a single full-canvas SDPA launch, likely by chunking Q/head groups around SDPA and concatenating attention output, or by finding a smaller denoise SDPA program/kernel.

### Session 2026-07-02 — #47464 R-b two-block RUN smoke

R-b passed on QB2 with the real DiffusionGemma 26B-A4B checkpoint after fixing host readback for mesh-distributed decision tensors. Initial attempt hit `TT_FATAL: Can't convert a tensor distributed on MeshShape([1, 4]) mesh to row-major logical tensor`; `tt/denoise_loop.py::_to_host_torch` now detects multi-device tensors and reads the first replicated shard directly before `ttnn.to_torch`, matching the device-test helper pattern.

Command:
`text_demo.py --checkpoint /home/zni/dg_models/diffusiongemma-26B-A4B-it --local-files-only --mesh P150x4 --max-seq-len 512 --canvas-length 256 --max-denoising-steps 1 --max-new-tokens 512 --num-blocks 2 --seed 0`.

Result: exit 0 in **258.8 s**. Full 30-layer build reported post-build DRAM **13.236 GiB used / 18.631 GiB free / 31.867 GiB total**. The run still emits expected denoise SDPA L1-clash `TT_THROW` lines before the Python fallback handles them; no uncaught `TT_FATAL`/`Traceback` remains. The decoded output is degenerate/empty after skip-special decoding, which is expected under the RUN-first/correctness-deferred policy.

### Session 2026-07-02 — #47464 R-a long-prompt maskless smoke

After the short-prompt RUN passed, a cheaper 1-layer R-a smoke tested prompts long enough to trip the old sliding-mask condition. `hello * 1000` tokenizes to `prompt_len=1013` (aligned `cache_len=1024`), so with `canvas_length=32` it exercises `prompt_len + canvas_len - 1 > sliding_window` without intentionally exceeding the 1024-token sliding cache capacity.

Command:
`DG_PROMPT="$(python3 -c "print('hello ' * 1000)")" text_demo.py --checkpoint /home/zni/dg_models/diffusiongemma-26B-A4B-it --local-files-only --mesh P150x4 --num-layers 1 --max-seq-len 1536 --canvas-length 32 --max-denoising-steps 1 --max-new-tokens 32 --num-blocks 1 --seed 0`.

Initial result: the run no longer failed because a sliding-layer mask reached the SDPA fallback, confirming the maskless default was taking effect, but it failed later in denoise prefix assembly with `TT_FATAL: Input Tensor is not allocated`. Follow-up localized that to `_apply_rope_chunked`: when `canvas_length=32` produced exactly one sequence chunk, the final single-element concat could return the same tensor that the helper then deallocated. After preserving the single chunk instead of concat/deallocating it, the same command exits 0 and logs `DG_TEXT_DEMO_SUCCESS generated_tokens=32 blocks=1 prompt_len=1024 next_pos=1056 sequence_len=1045 text_count=1 text_chars=0`.

Full-depth follow-up (same prompt/canvas, no `--num-layers`) initially reached post-build (**13.301 GiB used / 18.567 GiB free**) but failed before denoise compute while constructing the adapter's prompt K/V list: `read_prompt_kv_cache_by_layer` called `nlp_kv_cache_load_slice` for every layer and hit L1 OOM (`1048576 B`, largest free block `411392 B`). The fix is two-part: build prompt K/V lazily per layer and use ordinary `ttnn.slice(..., memory_config=DRAM)` for non-paged cache reads. With those changes, the same full-depth command exits 0 and logs `DG_TEXT_DEMO_SUCCESS generated_tokens=32 blocks=1 prompt_len=1024 next_pos=1056 sequence_len=1045 text_count=1 text_chars=2`.

A forced 2-block long-prompt smoke was added via `text_demo.py --disable-eos-stop` so EOS-heavy degenerate output cannot terminate before block 2. The 1-layer command with `--num-blocks 2 --max-new-tokens 64 --disable-eos-stop` exits 0 and logs `DG_TEXT_DEMO_SUCCESS generated_tokens=64 blocks=2 prompt_len=1024 next_pos=1088 sequence_len=1077 text_count=1 text_chars=171`, confirming per-block RoPE advancement remains wired on the long-prompt path. The same forced 2-block smoke also passes at full 30-layer depth: post-build DRAM **13.301 GiB used / 18.567 GiB free**, exit 0 in **95.3 s**, and `DG_TEXT_DEMO_SUCCESS generated_tokens=64 blocks=2 prompt_len=1024 next_pos=1088 sequence_len=1077 text_count=1 text_chars=12`.

### Code review — 2026-06-26 (multi-agent review of the 2026-06-25 branch)

Adversarial multi-agent review of the 48 commits `d13c3ad0c91..HEAD` (~3834 LOC) across #47474/#47462/#47463/#47472/#47464, plus a gemma4-regression sentinel and a test-rigor pass. **27 findings confirmed, 3 dismissed as false positives** (listed at the end so they are not re-raised). Each finding was independently re-verified against the code before landing here.

**Headline — no production gemma4 regression.** Every new parameter on the shared `models/demos/gemma4/tt/**` path (`kv_phase=None`, `write_kv_cache=True`, `attn_mask=None`, `kv_hidden_states=None`, `prefix_kv=None`, `q_rope_offset=0`) defaults to the pre-branch op sequence / dtype / order **bit-for-bit**: `coerce_kv_cache_phase(None)` → `PREFILL_WRITE`/`COMMIT_APPEND` → `write_kv_cache=True`; the new readonly/masked/prefix-KV branches are non-default-gated and raise on misuse; `prefill_sdpa_program_config` returns identical chunk sizes on the power-of-2 prefill buckets production actually uses. Risk is concentrated in **(a) device-loop deallocation discipline** and **(b) a few device tests that gate on nothing**.

#### 🔴 Must-fix

- ✅ **Fixed 2026-06-26 [#47472] `gumbel_max` / `canvas_sample` leaked the full `[B,L,vocab]` intermediates every call → 48-step loop OOM** — `tt/sampling.py:73-93`. `z = temperature_scale(...)` (new tensor when T≠1, i.e. the 0.8→0.4 schedule) and `perturbed = ttnn.add(z, noise)` are now deallocated after the output tensor is created; the same temporary-scale cleanup is applied to `token_entropy` / `softmax`. Validated on QB2 with `test_device_canvas_sampling_exact.py` (3 passed) plus `test_device_entropy_harness.py`, `test_device_denoise_loop.py`, and `test_device_self_conditioning.py` (14 passed).
- ✅ **Fixed 2026-06-26 [#47462/#47463] `test_denoise_controller_real_logits_records_decision_flips` no longer disables every trajectory threshold without replacement** — `tests/test_device_bidirectional_attention_integration.py:631-702`. The test is explicitly a diagnostic for the known bf16 decision-bar blocker, but now gates both MoE and dense modes on real-logits PCC, top-8 contains-reference-argmax, and a bounded accept-flip count. Validated on QB2 with `HF_MODEL=/home/zni/dg_models/gemma-4-26B-A4B-it`: MoE logits PCC **0.985/0.969**, accept flips **0/0**; dense logits PCC **0.995/0.984**, accept flips **2/2**; both parametrized cases passed.
- ✅ **Fixed 2026-06-26 [#47463/#47464] accept decision path now has an explicit on-device `ttnn.sort` regression guard** — `tt/denoise_loop.py:43-72`, `tests/test_device_entropy_accept.py`. `entropy_budget_accept` now cross-references the Part II decision-fidelity bar, and `test_production_entropy_budget_accept_guards_device_sort_at_canvas_256` directly validates the production sort→cumsum→exclusive-prefix→scatter path against the host oracle at the real 256-token canvas length, including accept count and mask equality. Validated on QB2 with `test_device_entropy_accept.py` + `test_device_denoise_loop.py` (9 passed).

#### 🟡 Should-fix

- ✅ **Fixed 2026-06-26 [#47463/#47464] `denoise_block` deallocates per-step decision tensors and superseded canvas tensors** — `tt/denoise_loop.py:156-230`. After host readback, `res.argmax/entropy/sampled/accept_mask` are freed; each consumed canvas is deallocated when replaced, and the final device canvas is freed before returning the host-only trajectory. Validated on QB2 with `test_device_denoise_loop.py` (3 passed) and the real-logits controller target in `test_device_bidirectional_attention_integration.py` (2 passed).
- ✅ **Fixed 2026-06-26 [#47474] `decode` + `DENOISE_READONLY` is rejected before `decode_forward` can drop the current token's K/V** — `models/demos/gemma4/tt/attention/kv_phase.py`, `tests/test_kv_phase.py`. `coerce_kv_cache_phase(..., is_decode=True)` now raises `ValueError` for `DENOISE_READONLY`, preserving the default decode `COMMIT_APPEND` path and the prefill-only readonly path. Validated with `test_kv_phase.py` (4 passed) and QB2 `test_device_kv_phase.py` (4 passed).
- ✅ **Fixed 2026-06-26 [#47474] Bounded-sliding commit-append wrap is now covered in the device path** — `tests/test_device_kv_phase.py`. `test_bounded_sliding_commit_append_wraps_cache_slot` builds a one-layer sliding Gemma4 model with `bounded_sliding_kv_cache=True`, a small paged sliding window, and a vLLM-style zero-padded per-layer page table; it drives `COMMIT_APPEND` decode at `position == sliding_window` and asserts the wrapped physical cache slot changes for both K and V. Validated with `test_kv_phase_mapping.py` + QB2 `test_device_kv_phase.py` (13 passed).
- ✅ **Fixed 2026-06-26 [#47464] Multi-step device-loop constant-logits test is now explicitly scoped as a control-flow smoke** — `tests/test_device_denoise_loop.py`. The synthetic test was renamed to `test_multi_step_denoise_control_flow_smoke_matches_reference` and documents that it does not exercise canvas→backbone→renoise cycling; real W2 logits cycling remains covered by `test_denoise_controller_real_logits_records_decision_flips` in `test_device_bidirectional_attention_integration.py`. Validated on QB2 with `test_device_denoise_loop.py` (3 passed) and the real-logits controller target (2 passed).

#### 🟢 Low / hardening

- ✅ **Fixed 2026-06-26 [#47474]** `attention/__init__.py:161` — three-state KV phase no longer collapses to one `write_kv_cache` bool without mode validation. `coerce_kv_cache_phase` now rejects decode+`PREFILL_WRITE`, decode+`DENOISE_READONLY`, and prefill+`COMMIT_APPEND`, while preserving default prefill/write and decode/append behavior. Validated with `test_kv_phase.py` and QB2 `test_device_kv_phase.py`.
- ✅ **Fixed 2026-06-26 [#47474]** `attention/operations.py:176-181` — `_largest_tile_divisor` now starts from the largest 32-aligned candidate at or below `min(preferred, length)`, so non-tile-aligned lengths such as 100 fall back to a tile-multiple chunk instead of returning 100. Covered by `test_gemma4_attention_operations.py`.
- ✅ **Fixed 2026-06-26 [#47474]** `diffusion_gemma/kv_phase.py:1-67` — relabeled as a reference/spec helper instead of a runtime Gemma4 cache mapping. The module docstring now states that runtime cache updates use the shared Gemma4 phase enum plus paged page-table/modulo plumbing, and `test_kv_phase_mapping.py` locks that distinction.
- ✅ **Fixed 2026-06-26 [#47462]** `attention/prefill.py:29-32, 87-102` — `q_rope_offset` and `_slice_rope_cache` now require 32-token tile alignment, and batch>1 prefill rejects nonzero `q_rope_offset` instead of silently ignoring it. Covered by `test_gemma4_prefill_guards.py`.
- ✅ **Fixed 2026-06-26 [#47463]** `tests/trajectory_pcc.py:30-33` — `_pearson` restores explicit constant-sequence handling: identical constants return 1.0, constant-but-different or one-sided-constant sequences return 0.0. `test_constant_mean_entropy_offset_fails_trajectory_pcc` covers the mean-entropy offset case.
- ✅ **Fixed 2026-06-26 [#47463]** `tt/denoise_loop.py:48-55` — `budget_t` now uses `entropy.get_dtype()` instead of hardcoded fp32, avoiding mixed-dtype threshold compares on bf16 entropy. `test_production_entropy_budget_accept_uses_entropy_dtype_for_budget` covers the bf16 production path.
- ✅ **Fixed 2026-06-26 [#47463]** `tt/denoise_forward.py:372-391` — `denoise_block` now duck-types `logits_fn.reset()` on both halt and max-step exits, so `DenoiseLogitsAdapter.reset()` can release the final cached logits without relying on callers. `test_multi_step_denoise_control_flow_smoke_matches_reference` asserts the reset hook fires once.
- ✅ **Fixed 2026-06-26 [#47472]** `tt/sampling_params.py:88-128` — `use_vocab_permuted_noise` now defaults to `False`; the permuted-vocab regenerated-noise workaround must be explicitly opted into until validated at production vocab scale. The permuted-vocab distribution test now passes `use_vocab_permuted_noise=True` explicitly.
- ✅ **Fixed 2026-06-26 [#47472]** `tt/sampling.py:112-134` — regenerated Gumbel noise now passes an explicit replicated `ttnn.MeshMapperConfig([ttnn.PlacementReplicate()])` for multi-device meshes in plain, permuted-vocab, and chunked rand paths, avoiding implicit mesh placement for QB2 1×4 RNG.
- ✅ **Fixed 2026-06-26 [#47472]** `tests/test_device_canvas_sampling_dist.py:91` — distribution smoke thresholds are now named constants with an explicit fixed-seed QB2 margin note: observed toy-vocab max top-1 error is ~0.03 at N=4096, 0.05 is only a large-regression smoke gate, and production-vocab RNG validation remains separate.
- ✅ **Fixed 2026-06-26 [#47464]** `tests/test_device_backbone_pcc.py:268` — default known-QB2 xfail floor is now 0.84 instead of 0.83, and a 0.90 ratchet ceiling makes improved-but-still-subtarget PCC fail instead of xfail so the local exception must be tightened.
- ✅ **Fixed 2026-06-26 [#47464]** `memory_budget.py:37-80` — module/function docstrings now state that the estimate covers only per-step denoise canvas K/V scratch, not prompt-prefix K/V, weights, paged cache, activations, or SDPA concat buffers measured by QB2 probes.
- ✅ **Fixed 2026-06-26 [gemma4]** `attention/prefill.py:29-32` — `_slice_rope_cache` now documents that the no-op fast path is correct only when callers pre-slice RoPE caches to the active `seq_len`.
- ✅ **Fixed 2026-06-26 [test]** `tests/test_device_canvas_sampling_dist.py:51-63` — renamed the readback-noise arithmetic test to `test_canvas_sample_matches_torch_argmax_with_readback_device_noise`, clarifying that RNG distribution is owned by the KL/frequency tests.
- ✅ **Fixed 2026-06-26 [test]** `tests/test_device_canvas_sampling_exact.py:58-75` — exact params test now asserts parsed `top_k`/`top_p` values are carried while `top_k_top_p_supported is False`, locking current no-op behavior until those filters are implemented.
- ✅ **Fixed 2026-06-26 [test, nit]** `tests/test_device_bidirectional_attention_integration.py:424-506` — self-conditioning adapter test now documents that it is a device-vs-device wiring equivalence; HF-golden logits tests own numerical correctness.

#### Dismissed — verified false positives (no action)

- **`ttnn.sort` risk "rides on" the integration test** — refuted: the `accept_flips==0` assertion is itself a host-sort-vs-device-sort cross-check, and `test_single_denoise_step_matches_reference` validates the sort chain element-exact. (The residual doc-vs-fact divergence is captured as the Must-fix above.)
- **Synthetic fp32 loop test is the only coverage of decision fidelity** — refuted: the real bf16 path is covered by the controller diagnostic test (which is the H2 finding's actual weakness — thresholds disabled, not absence of the test).
- **Removing `test_full_model[blackhole-1x4]=0.83` from the shared `pcc_thresholds.json` reverts to 0.99 and would fail** — refuted: the 26B MoE `test_full_model` `pytest.skip`s at `tp<8`, so on a 1×4 mesh (tp=4) it never reaches `compare_tensors`; the removed entry was dead for that combo. The removal correctly de-pollutes the shared production gate; DiffusionGemma's PCC gap is handled in `test_device_backbone_pcc.py`.

#### Fix verification — 2026-06-26 (independent re-check of the 23-commit fix campaign)

A second multi-agent pass independently verified all 23 fix commits at snapshot `03c40727c48` — for each fix: (a) does it resolve the finding, (b) does it introduce a regression, (c) is the added test real (would it fail if the bug regressed)? Every not-clean verdict was adversarially re-checked against the code. **Result: 20/25 fixes fully clean; the production `gemma4` path is provably unchanged.**

**Production `gemma4` cleared.** The three shared-code fixes are bit-for-bit safe: every production caller (`ttnn_prefill/decode/verify_forward`) passes `kv_phase=None` → safe default and never trips the new `coerce_kv_cache_phase` guards (isolated-run confirmed); `_largest_tile_divisor` is identical to the old `min()` on every power-of-2 prefill bucket (brute-forced); `q_rope_offset=0` / `_slice_rope_cache(start=0)` pass the new asserts. The H1/M1/M2/DENO-* dealloc+guard fixes are confirmed with **no double-free / use-after-free** (`committed` is the host copy of `res.argmax`, freed device tensors are not read; `z` from `temperature_scale` is always a fresh tensor).

- ✅ **Fixed 2026-06-26 [#47472/#47464] SAMP-3 mesh mapper fatal** — `_rand_mesh_mapper` now uses `ttnn.MeshMapperConfig(placements=[PlacementReplicate()], mesh_shape_override=ttnn.MeshShape([device.get_num_devices()]))`, so the single replicate placement maps over a flattened QB2 1×4 mesh instead of tripping `placements.size() == device.shape().dims()`. Guarded by CPU mapper tests (`test_tt_sampling.py`, 2 passed) and a real QB2 1×4 regenerated-Gumbel smoke (`test_device_sampling_mesh.py`, 1 passed).
- ✅ **Fixed 2026-06-26 [#47474/#47464] M3 wrap test is now discriminating** — `tests/test_device_kv_phase.py` drives bounded-sliding `COMMIT_APPEND` at `position=sliding_window+block_size`, so correct modulo writes physical slot 32 while a broken no-wrap path would write slot 96. The test now asserts the wrapped slot changes and the no-wrap slot stays untouched. Validated on QB2 with `test_bounded_sliding_commit_append_wraps_cache_slot` (1 passed).
- 🟢 **Fix-correct but regression-unguarded (low):**
  - **H1 / M1** — the dealloc fixes are correct, but no allocator high-water-mark test exists and the loop test halts at 2 steps (never the 48-step cap), so a *re-introduced* leak would be silent in CI. (M1 also leaks `init_canvas` in the degenerate `max_denoise_steps==0` config — non-production.) ⇒ resolves the W4 header caveat: the `gumbel_max` leak **is** fixed; only the leak-regression *guard* is missing.
  - **KV-P-4** — fix is production-identical; only the `(100,100)==32` assertion distinguishes new from old — `(384,256)==192` / `(512,256)==256` pass under both (sanity checks, not regression guards).

**Net:** the campaign is solid; the known SAMP-3 mesh-mapper fatal is fixed, and neither the remaining leak-regression guard gap nor the RNG distribution/fidelity caveats touch the production `gemma4` path.

### Code review — 2026-06-29 (multi-agent review of the #47464 integration layer)

Adversarial multi-agent review of the 68 commits `03c40727c48..HEAD` (~4.5k LOC) — the #47464 integration push since the 2026-06-26 review: `tt/generate.py` entrypoints, device commit-append / per-block RoPE advancement, checkpoint-state adapter & generation-logits builders, the reference outer loop + replay hooks, and `config.py` / `weight_mapping.py`. 9 module-group reviewers, every finding adversarially re-verified against the code (the two headline items also against the **installed transformers 5.12.1** source). **29 findings confirmed, 18 dismissed as false positives** (notable dismissals listed at the end so they are not re-raised). Status below was refreshed after the 2026-06-29 fix sweep through commit `4d114c83fbc`.

**Headline — still no production gemma4 regression; all review bugs with clear fixes are actioned.** Shared-path defaults remain safe (re-confirmed: `coerce_kv_cache_phase(None)`→legacy write, new prefill kwargs default off). The `kv_hidden_states` fused-QKV leak is fixed, long-prompt sliding denoise masks now match HF and are threaded into the device path, replay hooks are bounded, seed/shape/config guards are in place, and the known diagnostic device trajectory test now has nonzero fail thresholds. Remaining residuals are explicitly scoped below.

#### 🔴 Must-fix

- ✅ **Fixed 2026-06-29 [#47462/gemma4] `kv_hidden_states` denoise prefill leaked the fused QKV DRAM tensor every layer·step** — `models/demos/gemma4/tt/attention/prefill.py:82-90`. `xqkv_kv = apply_qkv_projection(kv_hidden_states, ...)` is now deallocated immediately after `split_qkv_heads_prefill(...)` extracts the replacement K/V tensors, preventing one fused-QKV DRAM allocation from accumulating per decoder layer per denoise step. Validated with CPU `test_generate.py` + `test_gemma4_prefill_guards.py` (18 passed) and QB2 `test_denoise_logits_adapter_threads_prev_logits_for_self_conditioning[blackhole-1x4]` (1 passed).
- ✅ **Fixed 2026-06-29 [#47462] All-attend denoise mask diverged from HF for sliding layers when `prompt_len ≥ sliding_window`** — `reference/attention_mask.py` now exposes `layer_type="sliding_attention"` + `sliding_window` to reproduce HF's bidirectional sliding visibility (`abs(q-k) <= sliding_window`) and `tests/test_real_transformers_parity.py` compares canvas rows against installed transformers. The device wrapper now materializes this mask only for long-prompt sliding layers while preserving the maskless path for full layers and short prompts. Validated with CPU mask/denoise parity tests (28 passed).

#### 🟡 Should-fix

- ✅ **Fixed 2026-06-29 [#47472] `seed=0` made "deterministic" Gumbel noise non-reproducible** — `sampling_params.py` and all TT regenerated Gumbel helpers now reject non-positive seeds (`5cd7debf6f2`, `aa292d7ca5a`).
- ✅ **Fixed 2026-06-29 [#47474, test] `test_commit_append_decode_writes_full_256_token_canvas` `NameError`** — test now defines the missing mesh mapper before reuse (`915d81e3ba6`).
- ✅ **Fixed 2026-06-29 [#47464] `embed_canvas_tokens` hardcoded batch=1 in the post-embedding reshape** — batch >1 now fails explicitly rather than silently reshaping incorrectly (`0c37988955b`).

#### 🟢 Low / hardening

- ✅ **Fixed 2026-06-29 Denoise-loop injected-noise leaks** — consumed per-step `gumbel_noise` / `noise_tokens` are now released in `tt/denoise_loop.py` (`7c0d5642b5a`).
- ✅ **Fixed 2026-06-29 `max_denoise_steps==0` degenerate config** — `DiffusionConfig.__post_init__` rejects non-positive denoise steps (`83473879ba2`).
- ✅ **Fixed 2026-06-29 oracle parameter-edge drift** — entropy accept defaults now match HF's no-min-accept edge; long-prompt mask parity is covered against real transformers. EOS remains intentionally owned by `tt/generate.py`, not the pure reference outer loop.
- ✅ **Fixed 2026-06-29 config defaults off the 26B-A4B path** — HF MoE fields are preserved, dense MoE nulls stay null, unknown checkpoint keys are surfaced, and top-level canvas-length consistency is validated (`3454f880344`, `283ccd20ab8`, `29d9db665f4`).
- ✅ **Fixed 2026-06-29 dead / unstable code** — removed dead `reference/sampling.is_converged` and unused `tt/sampling.softmax`; also dropped the unused internal denoise `prompt_len` argument (`f6326617fdb`, `bcd2759d729`, `60836956be9`).
- ✅ **Fixed 2026-06-29 robustness footguns** — multi-layer model-level `kv_hidden_states` / `q_rope_offset` now fail fast; `prefix_kv_by_layer` length is validated before layer forward; denoise replay noise hooks are required explicitly (`a2fb4bb1aff`, `b63c3294c6c`, `3bbd872e054`).

#### Nits (no action required)

Resolved nits: dead `if num_blocks > 0` after the zero-block early return, empty-string prompt chat templating, replay noise step/block bounds, misleading `resolution_token_budgets # verified` comment, unknown-key handling in `weight_mapping.py`, and the real-logits trajectory test's zero thresholds. Remaining no-action nit: redundant double `coerce_kv_cache_phase` (`model.py` + `attention/__init__.py`) is idempotent and harmless.

#### Dismissed — verified false positives (no action)

- **`denoise_step` leaks the pre-typecast argmax tensors every step** — refuted: the subsequent deallocation frees them; the two sub-facts are correct but the leak conclusion ignores the cleanup.
- **Docstring cites a non-existent `bidirectional_mask_function`** — refuted: the function exists in the installed source (it is the all-attend mask function; the real issue is cache/mask *slicing* for sliding layers, captured as H2).
- **`top_k`/`top_p` are silently dropped, not rejected** — refuted: `sampling_params.py` parses and carries them with `top_k_top_p_supported=False`, and `canvas_sample` asserts they are unset; the no-op is explicit and tested.
- **`decode` + `write_kv_cache=False` reads stale KV** — refuted: dead-but-correct defensive code; `coerce_kv_cache_phase` forbids `DENOISE_READONLY` in decode, so it is unreachable.
- **`_largest_tile_divisor` can return a non-divisor** / **prefix_kv from a bounded circular cache wraps** / **layer/model param threading is only signature-checked** / **`SelfConditioning.condition()` can't apply per-example self-cond** / **`embed_host_tokens` / `prefill_prompt_tokens` leak** / **seeded hooks share one generator** / **EOS inside a committed block still pads the canvas** / **device accept omits `min_accept`** / **bf16 entropy cumsum flips the accept boundary** / **`soft_embedding` scale diverges from the tied table** / **fp32 zero-noise argmax floor 0.95 too loose** — all refuted against HEAD (handled elsewhere, intentional, or factually wrong).

### Code review — 2026-06-30 (multi-agent review of the function branch — tt/ production + reference)

Adversarial multi-agent review of the `diffusion-gemma-function` branch (~15.4k LOC across 70 files vs `main`, plus 4 untracked `test_w2b_*` repro files): the `tt/` device-loop production code (`generate.py`, `denoise_forward.py`, `diffusion_attention.py`, `denoise_loop.py`, `self_conditioning.py`, `sampling*.py`, `model.py`), the core support modules (`checkpoint.py`, `config.py`, `kv_phase.py`, `memory_budget.py`, `weight_mapping.py`), and the `reference/` torch/HF ground-truth implementations. 6 module-group reviewers, every finding adversarially re-verified against the code with production reachability traced through the demo path. **3 findings confirmed, 5 dismissed as false positives** (notable dismissals listed at the end so they are not re-raised). All confirmed items below are **OPEN** — nothing was fixed in this pass.

**Headline — codebase is healthy; most flagged issues did not survive verification.** No new gemma4 regression, and no silent-wrong SDPA on either path: the denoise SDPA always passes a chunked `program_config` (Q=256, q-chunk forced to a tile-divisor of the seq length), and the >32768 non-chunked prefill cliff only touches discarded prefill logits, not the `fill_cache` K/V the denoise actually reads. The one new substantive bug is a silent self-conditioning precision drift on the live demo path; the prompt-alignment crash is the already-known next blocker, now traced end-to-end.

#### 🟡 Should-fix (open)

- ✅ **Historical 2026-06-30 change; precision claim corrected 2026-07-09 [#47463/#47464].** `tt/denoise_forward.py` added `default_self_conditioning_compute_kernel_config()` (`HiFi4`, `fp32_dest_acc_en=True`, `packer_l1_acc=True`) and the adapter injects it by default. That config applies to the moderate-vocabulary full-softmax branch only. The production 262144-vocabulary path takes the ordered online-chunk branch in `tt/self_conditioning.py`, which does **not** forward this config and operates on BF16 tensors. The old statement that production self-conditioning therefore used fp32 accumulation was incorrect; the dg-08 prechunk and logits-L1 changes preserve the actual BF16 arithmetic and ordered 8192-vocabulary grouping exactly.

- ✅ **Fixed 2026-06-30 [#47464] Natural-length prompts no longer crash deep in the attention/adapter stack** — `tt/generate.py` now returns a `PromptPrefill(prompt_len, cache_len)` from `prefill_prompt_tokens`, preserving the raw host-visible prompt length while threading the 32-aligned frozen-prefix length to the denoise logits builder, KV cache reader, and block start positions. This fixes the documented default-prompt blocker (`read_prompt_kv_cache_slice(prompt_len=18)` rejecting non-tile-aligned bounds) by making the denoise prefix consistently tile-aligned. Validated with CPU `test_tt_generate.py` (242 passed), CPU `test_denoise_forward.py` (16 passed), and QB2 tiny generated-token smoke: `text_demo --checkpoint /home/zni/dg_models/diffusiongemma-26B-A4B-it --local-files-only --mesh P150x4 --num-layers 1 --max-seq-len 512 --canvas-length 32 --max-denoising-steps 1 --max-new-tokens 1` (exit 0).

#### 🟢 Low / hardening (open)

- ✅ **Fixed 2026-06-30 [#47464] `init_canvas` device tensor is released if `block_fn` raises mid-block** — `tt/generate.py::generate_blocks` now wraps each `block_fn` call in an exception guard and best-effort deallocates the just-created initial canvas before re-raising the original device failure. This preserves the normal successful ownership path while preventing repeated caught failures from orphaning canvas tensors on a live mesh. Validated with CPU `test_tt_generate.py` (243 passed), including `test_generate_blocks_deallocates_init_canvas_if_block_fails`.

#### Test rigor (open)

- ✅ **Fixed 2026-06-30 [#48549/test-rigor] W2b SDPA-cliff diagnostics are no longer auto-collected as pytest tests** — the four untracked `test_w2b_*` repro files were moved to opt-in diagnostic script names (`w2b_256k_probe.py`, `w2b_causal_repro.py`, `w2b_chunk_repro.py`, `w2b_repro.py`) with updated run instructions. This keeps them available for explicit QB2 debugging while preventing assertion-light probes from entering CI by filename convention. Validated with `python -m py_compile .../w2b_*.py`; explicit `pytest --collect-only` on those paths still finds the opt-in diagnostics when requested.

#### Dismissed — verified false positives (no action)

- **`prefill_prompt_tokens` feeds the >32768 non-chunked SDPA → corrupts the frozen prompt K/V (#48549 W2b)** — refuted: prefill writes *projected* K/V to the cache via `ttnn.fill_cache` (pre-SDPA) and the prefill SDPA logits are discarded (`generate.py:224`); the denoise reads K/V from cache, not the prefill attention output, so the cliff cannot corrupt the prefix. Also moot at the `max_seq_len` default of 4096.
- **`denoise_loop.py:123` accept-mask reshape aliases freed storage → use-after-free** — refuted: the reshape-view aliasing is real, but the freed alias is never read; the high-severity host-readback failure is not reachable in current code.
- **`config.py:115` omitting `num_experts` keeps the MoE defaults** / **`config.py:110` null-valued keys fall back to dataclass defaults** — both refuted: not reachable with the real 26B-A4B / Gemma4 configs in use.
- **`generate.py:522` decode-input leak on `ttnn_decode_forward` exception** — refuted: not impactful on the reachable path; it is the same exception-only shape as the `init_canvas` low item, which was kept as the canonical instance.

### Session 2026-06-22 — #47468 / #47461 / #47487 push (QB2-only)

Goal: implement #47468 (torch ref + PCC harness), #47461 (causal backbone + self-cond loader), #47487 (QB2 fit) — **QB2 only, not Galaxy**.

**Unblocked two stale blockers:** the canonical `modeling_/generation_/configuration_diffusion_gemma.py` are on transformers `main` (pulled to `/home/zni/dg_ref_src/`), and all three checkpoints are ungated + downloaded. This let the reference layer be reconciled to the **real** algorithm rather than plan-stated approximations.

**#47468 — torch ref + harness (DONE, env-independent, verified):**
- Reconciled `reference/` 1:1 vs canonical source — found & fixed real drift: self-conditioning was a bare additive delta (missing `pre_norm` + scaleless `post_norm`); entropy-bound accept used inclusive `cum<=bound` (real is **exclusive** `cum-e<=bound`); temperature used `/(N-1)` ascending (real is HF reversed-step `t_min+(t_max-t_min)·cur_step/N`); halting threshold was a 0.1 guess (real `confidence_threshold=0.005`, mean-entropy of temp-scaled logits).
- Added `reference/_upstream.py` (verbatim canonical extractions) + `tests/test_upstream_parity.py` — reference now matches HF **bit-for-bit** (temperature/accept/confidence/self-cond). Guards against future drift.

**#47461 — backbone + self-cond loader (loader DONE + validated; device PCC turnkey):**
- `weight_mapping.py`: DiffusionGemma `model.decoder.*` ⇄ gemma4 `model.language_model.*` is a **pure prefix swap**; self-cond is the only net-new text-backbone module. **Validated vs real checkpoints**: remapped backbone keyset == gemma4 keyset exactly (no missing/renamed); the 4 self-cond tensors load with config shapes (`intermediate_size=2112`).
- Causal backbone PCC on QB2: gemma4 path is mesh-agnostic (`MESH_DEVICE=P150x4`), test is turnkey; **gated on shared-device availability**.

**#47487 — QB2 fit (`QB2_MEMORY_BUDGET.md`):** per-chip Blackhole DRAM is **~32 GB** (8×4 GB banks — corrected a prior ~4 GB misread). The real fit gate is whether MoE experts are **sharded** (code path → ~5.7 GB/chip, fits) or **replicated** (the `test_full_model` tp<8 skip's reading → ~22.8 GB/chip, needs Expert Parallelism). Static evidence favors sharded; **empirical device measurement pending**. Added `test_full_model[blackhole-1x4]=0.83` threshold.

**CPU suite: 60 passed, 9 skipped** (device + a couple ckpt-gated). Remaining: the on-device PCC/memory run (turnkey; recipe in `QB2_MEMORY_BUDGET.md`), gated on the shared QB2 box freeing up.

### Build order (env-independent first)

1. ✅ Config + scaffolding.
2. ✅ **Reference sampling primitives** (`reference/sampling.py`) + tests — the
   `#47463` acceptance spike reference and the `#47468` oracle's sampling core.
   Pure torch, CPU-testable, no checkpoint.
3. ✅ Reference denoise loop (assembling the primitives into the per-block
   trajectory) + tests (`reference/denoise_loop.py`).
4. ✅ Canvas mask geometry (`reference/attention_mask.py`) + PCC trajectory
   harness (`tests/trajectory_pcc.py`) + self-conditioning gated MLP
   (`reference/self_conditioning.py`).

**The env-independent reference layer is complete (40 CPU tests pass).** It
pins every net-new *algorithm* — sampling/acceptance (#47463), denoise
trajectory (#47463), multi-canvas generation (#47464), mask geometry (#47462),
self-conditioning (#47461/#47463), the decision-level PCC harness (#47468), and
the HF-reference adapter seam (#47468) — so the device port and the real HF
reference both have an oracle to validate against. Remaining work is
environment-gated:

5. ⛔ Vendored HF reference wrapper — unblocks once `transformers` ships
   `diffusion_gemma` (then plug it into the trajectory harness).
6. ⛔ Device (`tt/`) implementation — backbone reuse (#47461), KV phase machine
   (#47474), bidirectional SDPA (#47462), device decode loop (#47463),
   on-device sampling (#47472) — **QB2 hardware is present (this box); env + all checkpoints are in place.** No remaining env/HW/ckpt gate — remaining work is the device implementations themselves (per the rows above).

---

## Appendix A — W2b long-prompt attention (RESOLVED)

> **Status: ✅ RESOLVED (2026-06-26).** The gating spike **S1 passed** — regular non-causal SDPA returns correct results (PCC ≥ 0.99 vs an independent fp32 oracle) at `[256 × Sk]` up to **262144**, so the original "new kernel" framing was wrong. W2b reduced to **D1** (lift the `prefill.py` guard, re-key `long_seq` against K-length, run `is_causal=False` without `sliding_window_size`, materialize a dense mask only for HF sliding-layer visibility). The full sweep is wired into the QB2 Blackhole pipeline. The original spike-first plan is archived below for the source map / decision history. Real-26B integration is #47464.

### TL;DR — the reframe (read this before assuming "new kernel")

The original framing ("non-causal masked chunked attention is new kernel work") is **probably wrong**, or at least far heavier than the actual gap. Source investigation (2026-06-26) found:

1. **Denoise attention is a `[256 × (P+256)]` rectangular ALL-ATTEND region**, not a triangular masked one. The canonical denoise mask is **all-zeros** for *both* sliding and full layers (`reference/attention_mask.py:71-76`; `tt/denoise_forward.py:36-42`, `NEG=-1e9` never actually applied in the all-attend case). The only load-bearing geometry is the **canvas RoPE offset = `prompt_len`** (`tt/denoise_forward.py:96,123`), which is already correct at any length.
2. **The non-causal, maskless SDPA path already exists in the kernel** and visits *every* K-chunk: `reader_interleaved.cpp:367-369` sets `q_high_idx = Skt` for `!is_causal`; `compute_streaming.hpp:1308-1311` documents "the only config that stamps nothing is plain non-causal attention with no mask at all." So an all-attend `[256 × longK]` SDPA needs **no mask and no causal logic**.
3. **The 32768 limit is not a hard constant** — there is no `32768`/`2^15` in the SDPA device-op index path; page/tile ids are `uint32_t` (`dataflow_common.hpp:167,229,288`), and 256K/32 = 8192 tiles/row stays well inside `uint32`. The "garbage > 32768" is an **empirically-observed** wrong-result cliff documented only in a gemma4 Python comment (`operations.py:25-29`), seen on the **causal large-Sq prefill** shape — the opposite extreme from W2b's tiny `Sq=256` / large `Sk`.

**⇒ The entire effort hinges on one cheap experiment (S1): does the existing maskless non-causal op return correct results at `[256 × Sk]` for `Sk` up to 262144?**
- **If S1 PASSES:** W2b collapses to **near-zero kernel work** — lift the `prefill.py:180-181` guard, re-derive `long_seq` against **K length** (not Q seq_len), pad `P+C` to a tile, and drop the mask. That is **D1**.
- **If S1 FAILS:** the cliff is most likely bf16 online-softmax numeric (running max/sum are `Float16_b`, `sdpa_program_factory.cpp:637-642`, accumulated over hundreds of K-chunks) → fall to **D4** (host K-chunking) or **D3** (kernel numerics), or **D5** (paged) if contiguous long-K DRAM is the ceiling.

Do **not** start any kernel work before S1.

---

### The problem (precise, source-grounded)

DiffusionGemma's denoise step needs the 256-token canvas to attend to `[prompt_prefix ; canvas]` **bidirectionally** (non-causal), and this must work when `prompt_len + 256 > 32768`. Today three facts block it:

- **Non-chunked SDPA is the only masked/non-causal path, and it's gated off above 32768.** `prefill.py:176-205` dispatches: `long_seq = seq_len > PREFILL_SDPA_MAX_SEQ` (32768); the masked branch (`prefill.py:179-190`) **hard-raises** for `long_seq` (`prefill.py:180-181`). That guard keys off **Q `seq_len`** — which for denoise is always 256 and never trips — so the guard is currently mis-keyed *and* the long-K case is untested.
- **The "chunked" long-context op is causal-only and refuses masks.** Both `chunked_scaled_dot_product_attention` overloads hardcode `is_causal=true`, `attn_mask=std::nullopt` (`sdpa.cpp:117,153,299,427`), and `validate_chunked_mode` FATALs on any mask (`sdpa_device_operation.cpp:216-218`). The causal assumption is the **K-chunk loop bound** (`compute_common.hpp:1947-1949`, `compute_streaming.hpp:1910-1913`, `reader_interleaved.cpp:361-369`), not just a triangular stamp.
- **`attn_mask` ⊥ `sliding_window_size`.** They share one L1-accumulate slot and are `static_assert`'d exclusive (`sdpa_device_operation.cpp:67-72`; `compute_streaming.hpp:1859-1860`). So sliding layers in denoise must bake any window into the dense mask — which the canonical all-attend mask already does (i.e. no window at all in denoise).

The masked non-causal SDPA **mechanism** is fully implemented (`compute_streaming.hpp:1312-1330` provided-mask streaming; `reader_interleaved.cpp:521-559` per-chunk mask reader; mask shape `[b|1, h|1, Sq, Sk]`, TILE, BF16/BFP8/BFP4, DRAM, `sdpa_device_operation.cpp:74-106`). W2a is literally this path at `Sk ≤ 32768`. **W2b is the same op (or its maskless twin) above 32768.**

---

### Loop protocol (spike-first)

1. **Run the spikes in order (S1 → …).** Each spike is a pure-op or single-layer device experiment vs a torch oracle — *no model build needed for S1/S2*. The spike outcomes route the decision tree below; do not pick a direction before its gating spike passes.
2. **One increment per iteration**, validated on QB2 (recipe in Part II §1 (the env recipe)). Oracle = `torch.softmax(QK^T·scale) @ V` (all-attend) / `reference/attention_mask.py`.
3. **Record every spike result** (Sk, head_dim, PCC, pass/fail) in the Status section at the bottom of this file.
4. **Escalate, don't guess:** if S1 fails, the root-cause bisection (S3) decides between a localized numeric fix and a paged rewrite — these have very different cost; surface the bisection result before committing.
5. **Never mark W2b ✅ without a device PCC test at `[256 × >32768]`** (and at the 131072 / 262144 milestones) — this regime is currently completely untested (`test_sdpa_prefill.py` masked tests max ~2–8K, square `Sq==Sk`).

---

### Spike sequence (the heart of the plan)

#### S1 — does the cliff even bite the W2b shape?  *(highest leverage, do first)*
- **Method:** pure `ttnn.transformer.scaled_dot_product_attention(Q=[1,nqh,256,DH], K=V=[1,nkv,Sk,DH], is_causal=False, NO attn_mask)` vs `torch.softmax(QK^T·scale)@V`. Sweep `Sk ∈ {8K, 32K, 33K (non-tile-aligned tail), 64K, 131072, 262144}`, `head_dim ∈ {512 (global, L1-tightest), 256 (sliding)}`.
- **Pass:** PCC ≥ 0.99 at every `Sk`, both head_dims (include the non-tile-aligned `Sk` to exercise the `use_padded_mask` writer, `sdpa_program_factory.cpp:240-245`).
- **Routes:** PASS → **D1** (collapse to guard-lift). FAIL → **S3**.

#### S2 — masked A/B control  *(parallel with S1, cheap)*
- **Method:** identical sweep but **with** the explicit `[1,1,256,Sk]` all-zeros bf16 mask (the W2a call, `is_causal=False` + `attn_mask`).
- **Why:** if maskless (S1) passes but masked (S2) fails at the same `Sk` (or vice-versa), the cliff is **path-specific** (provided-mask streaming reader vs padded-mask writer) and is localized. If both pass, choose **D1** and drop **D2**.

#### S3 — bisect the cliff  *(only if S1/S2 fail)*
- Force the **legacy** (non-streaming) compute kernel via `fp32_dest_acc_en` (`sdpa_program_factory.cpp:78,361` sets `use_streaming_compute=false`) and re-run the failing `Sk`. Legacy passes where streaming fails ⇒ bug is in `compute_streaming.hpp` online-softmax.
- Sweep `GEMMA4_PREFILL_SDPA_KCHUNK` at fixed `Sk`: if PCC tracks **chunk count** (not `Sk`), it confirms bf16 running-stat accumulation (`Float16_b` stats, `sdpa_program_factory.cpp:637-642`) ⇒ **D3/D4**. *(Caveat: confirm legacy compute handles non-causal maskless identically before trusting it as a control.)*

#### S4 — RoPE cache reachability  *(independent, cheap, do early)*
- `model.py:124` default `max_seq_len=131072` < `config.py:196` `max_context=262144`; `model.py:475-477` slices `cos[:,:,:seq_len,:]`. Build/pass caches sized 262144 and assert `q_rope_offset=prompt_len` up to ~256K returns a **full-length** slice (and a hard error, not a silent short slice, if exceeded). **This gates any direction above 131072 regardless of SDPA.**

#### S5 — paged non-causal corner  *(only if S1 passes but contiguous long-K DRAM is infeasible at 256K)*
- Drive `ttnn::prim::sdpa` directly (bypass the `is_causal=true` public wrappers) with `is_causal=false` + `chunk_start_idx=0` + a `page_table` covering `P+C > 32768`, `Q=[1,H,256,DH]` single chunk, **no mask**, PCC vs all-attend reference. Tests whether the existing reader/compute `is_chunked` + `!is_causal` branches **compose** before committing to **D5**.

---

### Candidate directions (cheapest / most-reuse first)

| Dir | Mechanism | Effort / Risk | Feasibility (critic) |
|---|---|---|---|
| **D1 — maskless non-causal regular SDPA (the reframe)** | Drop the all-zeros mask entirely; call non-causal SDPA over `[256 × (P+C)]` with `P+C` tile-padded. Reuses the existing zero-stamp non-causal streaming K-loop. Re-derive the `long_seq` guard against **K length**. | small / high | **MAYBE → the target.** Plumbing verified correct; only risk is the (unproven) cliff at this shape. **S1 decides.** |
| **D2 — keep the all-zeros mask, just lift the guard** | The exact W2a call at `Sk > 32768`. Legal shape (`sdpa_device_operation.cpp:104-105,118-121`). | small / high | **MAYBE, but strictly worse than D1** — materializes a full `[1,1,256,P+C]` DRAM mask (~134 MB/chip bf16 at 256K) for zero numerical benefit. Keep only as the **S2 A/B control** / a carrier for a non-canonical local-window op-test. |
| **D3 — fp32 running stats in streaming compute** | Carry online-softmax max/sum in fp32. | **large / medium** | **NO as described.** `sdpa_program_factory.cpp:637-642` asserts `im_df==stats_df`; the SALAD rescale binds out/sum/exp to one format (`compute_streaming.hpp:1596-1611`). fp32 stats ⇒ `fp32_dest_acc_en` ⇒ drops the streaming kernel. Not a localized edit. Use only if S3 proves a numeric cliff *and* D4 is rejected. |
| **D4 — Python K-chunking + host online-softmax** | Slice K/V into ≤32768 column blocks, run maskless non-causal SDPA per block, recombine with a host fp32 online-softmax (running max + exp-rescale), mirroring `operations.py:262-298` but iterating K with `Q=256` fixed. | medium / medium | **MAYBE (numeric fallback).** Fatal practical snag: the public SDPA op **does not return per-slice logsumexp/max** (`sdpa.cpp`), so recombination needs a second pass or a small op extension. Keeps every sub-op in the proven ≤32K regime. Dead weight if S1 passes outright. |
| **D5 — paged non-causal (maskless) chunked extension** | Relax `validate_chunked_mode` for `!is_causal`; add a public non-causal chunked wrapper forwarding `page_table` (+ optionally mask). Reads K from the paged cache in-kernel (`dataflow_common.hpp:256-315`) — long K never materializes contiguously. | large / very-high | **MAYBE, last resort.** Reader has independent `is_chunked`/`is_causal` branches that *appear* to compose, but compute has causal-coupled invariants (`compute_streaming.hpp:988` "KV-pad rotation mask is causal-only"). The **all-attend maskless** variant dodges the mask-page-id reconciliation. `ring_joint`'s `is_cross` proves non-causal short-Q/long-K chunked flash *runs* on BH (`ring_joint_sdpa_device_operation.cpp:319-331`) — existence proof, **not** forkable (CCL + structural mask). Pursue only if S1 passes but contiguous `[1,nkv,262144,DH]` K/V DRAM is infeasible per chip at TP=4. |

---

### Decision tree

```
S1 (maskless non-causal at [256 × Sk], Sk→262144)
├─ PASS ───────────────► D1: lift prefill.py:180-181 guard, re-key long_seq on K length,
│                            tile-pad P+C, delete mask.  (near-zero kernel work)
│                            └─ also run S4 (RoPE cache ≥262144) in parallel — independent gate.
│                            └─ if contiguous long-K DRAM infeasible at 256K → S5 → D5 (paged, maskless).
└─ FAIL ──► S3 (bisect: legacy vs streaming; chunk-count sweep)
            ├─ numeric (bf16 online-softmax) ──► D4 (host K-chunking, fallback)  or  D3 (kernel, heavy)
            └─ structural / DRAM ceiling ──────► D5 (paged non-causal, net-new kernel)
```

---

### Acceptance criteria (W2b done)

- **S1 device PCC:** non-causal maskless regular SDPA at `Sk ∈ {32K, 33K, 64K, 131072, 262144}`, `Sq=256`, PCC ≥ 0.99 vs torch `softmax(QK^T·scale)@V` on QB2 (mesh (1,4), TP=4), for `head_dim` 512 **and** 256, including a non-tile-aligned `Sk`.
- **End-to-end denoise step:** a single DiffusionGemma denoise step at `prompt_len` pushing `P+C` past 32768 (and at 131072, 262144) produces canvas hidden states matching the all-attend reference (`q_rope_offset=prompt_len`) at the gemma4 PCC convention (threshold = `floor(measured − 0.005)`, ratchet up only). Both sliding and full layers run all-attend (`sliding_window_size` forced `None`).
- **Guard re-derivation:** the `prefill.py:180-181` ValueError is re-keyed against **K length** (`P+C = tt_k.shape[-2]`), not Q `seq_len`; the W2a (`P+C ≤ 32768`) suite still passes unchanged; the long path is selected only when `P+C > PREFILL_SDPA_MAX_SEQ`.
- **Memory:** at `P+C=262144`, per-chip SDPA-input DRAM is documented and within budget — D1 materializes **no** mask (saves ~134 MB/chip bf16); the contiguous long-K `[1,nkv,P+C,DH]` footprint is measured and fits (else D5).
- **RoPE (S4):** caches sized to 262144 (or chunked slicing) verified so `q_rope_offset=prompt_len` up to ~256K returns a full-length slice, with a hard error (not a silent short slice) past the configured max.
- **If a kernel change is made (D3/D5):** the streaming path stays selected (`fp32_dest_acc_en` stays false); existing causal/sliding/joint/ring SDPA unit tests still pass; a new unit test covers the `[256, >32768]` non-causal regime.

---

### Risks & open questions

1. **Biggest unknown:** is the >32768 cliff **numeric** (bf16 `Float16_b` online-softmax running max/sum over hundreds of K-chunks) or **structural**, and does it bite the W2b shape at all? It was observed on the **causal large-Sq** prefill; W2b is the opposite extreme (`Sq=256` = one Q-chunk, one set of running stats vs very large `Sk`). These may behave completely differently. **S1 is the only way to know and it bifurcates the whole effort.**
2. **Contiguous long-K DRAM:** is `[1,nkv,262144,DH]` K/V allocatable per chip at TP=4 alongside weights/activations, or is paged KV (D5) mandatory at the top end? Decides whether the cheap D1 path scales all the way to 256K.
3. **RoPE cache** is only sized to 131072 by default — independent gate above that (S4).
4. **256K may break elsewhere:** the attention-output concat, the TP=4 allreduce/CCL, and activation residency may independently break above 32768/131072, separate from SDPA. W2b's SDPA fix is necessary but may not be sufficient for the full 256K criterion — scope a separate end-to-end-at-length check.
5. **Non-tile-aligned `P+C`:** the maskless `use_padded_mask` writer path (`sdpa_program_factory.cpp:240-245`) is unproven at >32768; S1 must include a non-aligned `Sk`. Fallback: tile-pad `P+C` in the `prefix_kv` layout.

---

### Key source map

| What | Where |
|---|---|
| 32768 cliff comment (empirical, causal large-Sq) | `models/demos/gemma4/tt/attention/operations.py:25-29` |
| Dispatch + **the guard to lift** | `models/demos/gemma4/tt/attention/prefill.py:176-205` (guard `:180-181`; W2a masked call `:182-190`; RoPE offset `:99-107`; prefix_kv concat `:116-128`) |
| Python Q-chunking (causal) / sliding chunker | `operations.py:220-298` / `:301-364`; program config `:185-217` |
| SDPA op: causal hardcode / mask reject / shape rules | `sdpa.cpp:117,153,299,427`; `sdpa_device_operation.cpp:56-60,67-72,74-106,118-121,216-218` |
| Non-causal full-K loop / maskless zero-stamp | `reader_interleaved.cpp:367-369`; `compute_streaming.hpp:1308-1311` |
| Provided-mask streaming reader / compute | `reader_interleaved.cpp:521-559`; `compute_streaming.hpp:1312-1330` |
| Online-softmax (running max / SALAD rescale) + stat format | `compute_common.hpp:2131-2177`; `compute_streaming.hpp:1596-1611`; `sdpa_program_factory.cpp:637-642` |
| Streaming vs legacy selection (`fp32_dest_acc_en`) | `sdpa_program_factory.cpp:78,361` |
| `use_padded_mask` writer | `sdpa_program_factory.cpp:240-245` |
| Paged in-kernel reader (for D5) | `dataflow_common.hpp:256-315`; uint32 ids `:167,229,288` |
| Causal-only KV-pad invariant (D5 threat) | `compute_streaming.hpp:988` |
| Non-causal short-Q/long-K chunked flash exists (not forkable) | `ring_joint_sdpa_device_operation.cpp:319-331` |
| Canonical all-attend denoise mask (all-zeros, both layer types) | `reference/attention_mask.py:71-76`; `tt/denoise_forward.py:28-49,36-42` |
| Canvas RoPE offset = prompt_len | `tt/denoise_forward.py:96,123` |
| RoPE cache size gate | `models/demos/gemma4/tt/model.py:124,475-477`; `config.py:192-197` |

---

### Out of scope / non-goals

- W2b does **not** gate the short/medium-prompt functional milestone (W2a covers `≤ 32768`). It is now resolved for the attention path; full 26B generation and quality remain separate #47464/#48291 work.
- This plan is attention-only. The decision-fidelity bar (#48291) and the rest of the e2e glue (#47464) are separate and tracked in this document.
- top-k/top-p, batching, and serving are out of scope here.

---

### Status log

| Date | Spike / step | Result |
|---|---|---|
| 2026-06-26 | Plan authored (source-grounded design + adversarial feasibility) | S1 is the gating experiment; D1 is the target pending S1; D3 ruled out as a localized edit. |
| 2026-06-26 | S1/S2 harness + smoke | Added `tests/test_device_long_sdpa_w2b.py` with a memory-bounded fp32 online-softmax oracle and opt-in full sweep. QB2 smoke passed: S1 maskless `Sk=8192,DH=256`; S1 maskless non-tile `Sk=33000,DH=256`; S2 masked `Sk=8192,DH=256`. |
| 2026-06-26 | S1 maskless non-causal `[256 × Sk]` PCC sweep | ✅ PASS on QB2: `Sk ∈ {8192, 32768, 33000, 65536, 131072, 262144}`, `head_dim ∈ {256,512}`, PCC ≥ 0.99 vs fp32 online-softmax oracle (`DG_RUN_DEVICE=1 DG_W2B_SDPA_SWEEP=full pytest models/experimental/diffusion_gemma/tests/test_device_long_sdpa_w2b.py -x -q`). Routes W2b to D1; no kernel work needed for this spike. |
| 2026-06-26 | S2 masked A/B control | ✅ PASS on QB2 for the same `Sk × head_dim` sweep with explicit all-zero bf16 mask; no path-specific cliff observed between maskless and masked regular SDPA. |
| 2026-06-26 | D1 maskless non-causal denoise path | ✅ Wired: `denoise_forward.py` defaults canonical all-attend denoise to `attn_mask=None` and passes `is_causal=False`; Gemma4 prefill attention now exposes an explicit `is_causal` knob while preserving the default causal path. Validated with CPU guards, `test_device_bidirectional_attention_integration.py` (7 passed), and the full W2b SDPA sweep (24 passed). |
| 2026-06-26 | S4 RoPE cache ≥ 262144 | ✅ PASS: `create_rope_caches` no longer allocates a hidden-width dummy for 256K caches; `_get_rope_mats` and `_slice_rope_cache` now hard-error on overrun. CPU guards passed (4/4), and QB2 `test_w2b_rope_slice_reaches_256k` verified `q_rope_offset=261888`, `canvas_len=256` returns a full 256-token slice from a 262144 cache. |
| 2026-06-26 | Integrated long-prompt denoise PCC | ✅ PASS on QB2: tiny Gemma4 denoise wrappers run through `denoise_attention_forward → Gemma4Attention → prefill_forward` with maskless `is_causal=False` at `P+C=33280` and `P+C=262144` for **both full-attention and sliding-attention layers**. The sliding cases use `prompt_len > sliding_window=1024`, so they exercise the non-windowed denoise path rather than a window no-op. Full W2b suite: 29 passed with `DG_RUN_DEVICE=1 DG_W2B_SDPA_SWEEP=full pytest models/experimental/diffusion_gemma/tests/test_device_long_sdpa_w2b.py -q`; the same command is now wired into `tests/pipeline_reorg/blackhole_e2e_tests.yaml` for QB2 (`bh-diffusion-gemma-w2b-full-sweep`). W2b attention blocker is closed; real 26B generation remains under #47464/#48291. |
| 2026-06-26 | Independent verification re-check (adversarial) | ✅ Confirmed **for the prefill path only**, and **SUPERSEDED for the DECODE path by #47464 R0.3/R0.4 (2026-06-30)** — see R0.4 row + Biggest-risk #4: the decode footprint edits (`apply_rope_decode_peruser` for all batches, `1x1` SDPA grid + `k_chunk_size=32`, weightless/per-head norm sharding, expert+Q L1→DRAM) are ungated and change plain-gemma decode; "bit-for-bit unchanged" no longer holds for decode. Original (prefill) verdict: production `gemma4` path **bit-for-bit unchanged** (`is_causal` defaults True and is never set by the model loop; rope `x_dummy` shrink is a proven no-op; the new `_slice_rope_cache`/`_get_rope_mats` guards are unreachable for production seq_lens). Maskless ≡ all-attend mask (canonical denoise mask provably all-zeros; `sliding_window_size` forced `None` on the non-causal path). Op proven to 262144 vs an **independent** fp32 online-softmax oracle (PCC≥0.99 actually asserted); both layer types covered; suite **wired into the BH QB2 pipeline** (`tests/pipeline_reorg/blackhole_e2e_tests.yaml:75`, sets both env flags). NOTE for future readers: an automated grep that scanned only `.github/` initially mis-flagged the suite as "CI-skipped / 262144 never runs" — **incorrect**; tt-metal device tests run via the `tests/pipeline_reorg/*.yaml` manifests, not `.github/`. **Sole residual:** the integrated denoise test is a tiny config (hidden=128, head_dim=32); real-26B denoise integration is #47464 (the op-level sweep already covers the real head_dims 256/512). |
