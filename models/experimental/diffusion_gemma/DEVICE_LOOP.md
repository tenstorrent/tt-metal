# DiffusionGemma device bring-up — plan, spec & status

**Audience:** an autonomous agent running in `/loop` mode on the QB2 box `bh-qbge-06`.
**Goal:** take DiffusionGemma from a set of validated pieces to a model that actually runs — a prompt string in, generated text out, on QB2.

This file is the **single source of truth** for the branch (the former `STATUS.md` was folded in). It is organized as:
- **Roadmap** (below, read first) — the forward plan: where we are, the critical path, and the phased work to a running model.
- **Part I — Execution spec** — the loop protocol, env/run recipe, ground rules, the decision-fidelity bar, and the W1–W4 workstream specs + acceptance criteria.
- **Part II — Implementation status** — environment constraints, the per-workstream status table, the 2026-06-26 code review + fix verification, session notes, and build order.

---

# Roadmap — the path to a running model

> The original four device workstreams (W1–W4, Part I) are now mostly done or blocked. The **real** remaining work to a first running generation is the integration + correctness phases below, surfaced by the 2026-06-26 gap analysis. This roadmap is authoritative for "what to do next"; W1–W4 in Part I are the detailed specs for the pieces they cover.

## Where we are (2026-06-26)

Foundation (torch reference + PCC harness #47468 ✅ closed, causal backbone #47461 ✅, QB2 fit #47487) plus three of the four device pieces are validated **in isolation**: KV-phase machine (W1/#47474) ✅, bidirectional masked SDPA ≤32768 (W2a/#47462) ✅, on-device canvas sampling (W4/#47472) ✅. The decode-loop control flow (W3/#47463) is built and validated on synthetic logits but **blocked on decision fidelity (#48291)**.

**Nothing runs end-to-end.** There is no callable — device, or even CPU-against-real-weights — that takes a prompt string and returns text. The validated halves (the full 26B backbone ↔ the device denoise loop) have **never been joined**: no device commit-append, no per-block position advancement, no full-model + self-conditioning assembly from the real checkpoint, no tokenizer/text I/O, and no end-to-end acceptance test.

## Two distinct gaps — do not conflate them

| Gap | What it is | Cost | Tracking |
|---|---|---|---|
| **Make it RUN** (emit *some* text) | Integration glue: join the pieces into one prompt→text device loop | Large but tractable net-new engineering (~weeks) | #47464 |
| **Make it CORRECT** (match HF) | The bf16/MoE/TP=4 **decision-fidelity bar** — diffusion commits the *clean argmax*, and the shared backbone shows only ~50% argmax agreement | Core gemma4 MoE-precision work **or** a product decision; possibly multi-week or unachievable on current kernels | #48291 |

The model can be made to *run* (and emit text) **without** resolving fidelity — it just will not be *correct*. Decide #48291 early: it determines whether the integration work yields usable output. Diffusion has no temperature/top-p cushion (it commits the clean argmax), so the ~50% backbone argmax ceiling maps almost directly to wrong tokens.

## Critical path to a first CORRECT generation (dependency-ordered)

| # | Step | Status | Issue |
|---|---|---|---|
| 0 | **Decide the decision-fidelity bar** — engineering MoE-precision fix, or product accepts a degraded floor (informed by a real-ckpt denoise trajectory measurement) | 🔴 open escalation | **#48291** |
| 1 | **Device commit-append** — write the committed canvas into the KV cache with `COMMIT_APPEND` (primitive exists in #47474, never called in `tt/`) | ⬜ | #47464 |
| 2 | **Per-block RoPE/position advancement** — block N at `prompt_len + N·256` (`q_rope_offset` hardcoded today) | ⬜ | #47464 |
| 3 | **Join full 26B + device self-conditioning + 30-layer prompt-KV** from the real checkpoint (self-cond tensors currently discarded; only single-layer prompt-KV lists exist) | ⬜ | #47464 |
| 4 | **Measure the integrated real-size denoise step fits** on the (1,4) mesh (full-canvas logits + 262k soft-embed matmul + 30-layer `[P+C]` KV concat) | ⬜ | #47464 / #47487 |
| 5 | **Entry point `tt/generate.py`** — tokenize + chat template → prefill → canvas init → denoise(≤48) → commit → advance → loop blocks → detokenize, with EOS/length stop | ⬜ | #47464 |
| 6 | **e2e acceptance test** — cheapest first: CPU HF `generate()` vs `reference/generate_blocks` token-equal; then device-vs-HF on a short prompt with injected reference noise | ⬜ | #47464 |

> **Cheapest unblocked step:** the CPU half of #6 (HF `generate()` vs `reference/generate_blocks` token-equal) — proves the algorithm independent of device precision, needs no QB2, and is not gated by #48291.

## Phased roadmap

**Phase 0 — Foundation** ✅ done — torch reference + PCC harness (#47468 ✅ closed), causal backbone PCC (#47461 ✅), QB2 memory fit (#47487).

**Phase 1 — Device pieces (W1–W4)** — three done, one blocked:
- W1 KV-phase machine ✅ (#47474) — residual: bounded-sliding commit-append wrap correctness still unverified (test is non-discriminating).
- W2a bidirectional masked SDPA, prompt+canvas ≤ 32768 ✅ (#47462).
- W4 on-device canvas sampling ✅ (#47472) — residual: SAMP-3 mesh-mapper `TT_FATAL` (latent), regenerated-noise unvalidated at production vocab.
- W3 decode-loop control flow ✅ built & validated on synthetic logits, 🔴 blocked on #48291 (#47463).

**Phase 2 — Integration to a first run** (#47464) — ⬜ not started; the bulk of remaining engineering — critical-path steps 1–6 above.

**Phase 3 — Correctness** (#48291) — 🔴 the gating decision — resolve the decision-fidelity bar (MoE precision work, or product acceptance of a degraded floor), measured via a real-checkpoint denoise trajectory.

**Phase 4 — Functional milestone** (#47464) — after a first correct run — full 256K context (the **W2b** long-prompt non-causal attention prerequisite is now validated through 262144), TP across the mesh, perf optimization (#47465).

**Beyond Functional** — batched canvas decode (#47557), vLLM runner + TT-plugin integration (#47466 / #47488), CI + perf-regression pipelines (#47489), multimodal T+I / T+V (#47467), quantized checkpoint (#47475).

## Biggest risks

1. **#48291 may be unachievable** on the current bf16/MoE/TP=4 kernels without core gemma4 MoE-precision work (fp32-faithful router top-k is blocked by `ttnn.topk` `TT_FATAL` on FLOAT32; fp32 experts exceed QB2 DRAM). If so, correct diffusion output needs that kernel work or an explicit product decision to ship degraded quality.
2. **W2b** (long-prompt > 32768 non-causal denoise attention) is resolved for the attention path: SDPA, RoPE, and integrated tiny-model denoise PCC all pass through 262144. Remaining 256K risk now sits in full-model #47464 integration / #48291 decision fidelity, not W2b kernel feasibility.
3. **Per-block position advancement** (step 2) is an easy-to-miss correctness requirement — without it, every block past the first is positioned wrong and the text is garbage even if fidelity were perfect.

---

# Part I — Agent loop spec

## 0. Loop protocol (do this every iteration)

> **Progress lives in Part II of this file** (the *Status by workstream* table). This branch
> no longer has a separate `STATUS.md` — it was folded into Part II below. Everywhere this
> spec says "update the status table" it means that Part II table. Confirm with the user which
> branch the device loop runs on before iteration 1.

1. **Read the Roadmap** (top of this file) → find the first incomplete step on the critical
   path / the active phase. W1–W4 are mostly done; the live work is **Phase 2 integration**
   (#47464), and **Phase 3 correctness is gated by the #48291 decision** — surface that, don't
   silently pick around it. Cross-check the Part II status table for per-workstream detail.
2. **Do the smallest shippable increment** of that task (one module / one device test).
3. **Validate on device** (recipe §1). The oracle is always `reference/` — assert the
   ttnn output matches it (PCC or `torch.equal`), never assert against a fresh guess.
4. **Update the Part II status table**: flip the row, note the test file + measured PCC + date.
5. **Commit ONLY IF commits are explicitly enabled** for this loop (user/loop config says
   so) AND the device test in step 3 passed. Never commit a half-finished increment, a
   failing test, or an environment workaround (e.g. the erisc reset, a local build). If
   commits are not enabled, leave the work in the tree and record progress in the Part II
   table instead. When you do commit: see §2 ground rules — **NO `Co-Authored-By` trailer**.
6. If blocked (HW fault, missing op, oracle disagreement you can't resolve in one
   step): write the blocker into the Part II table under the workstream, leave the row 🚧
   with a one-line reason, and move to the next *independent* task if one exists;
   otherwise stop the loop and surface the blocker.
7. **Never mark a row ✅ without a passing device test** (or, for a deliberately
   host-side fallback, a passing CPU test + a one-line note saying why it's host-side).

**Definition of done for the whole loop:** a single entry point runs
prefill → denoise(≤48 steps) → commit for one 256-token block on QB2, and its
per-step trajectory matches `reference/denoise_loop.py` under injected noise
(`tests/trajectory_pcc.compare_trajectories`), within the PCC bar agreed in §3.

---

## 1. Environment + device run recipe (QB2, `bh-qbge-06`)

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

## 2. Ground rules

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

## 3. The decision-fidelity bar (the thing this model actually needs)

Logit PCC is necessary but **not sufficient** — DiffusionGemma's correctness lives in
the per-step accept/remask **decisions**, which bf16/bfp8 small-probability drift can
flip. For every device step, `compare_trajectories` must check, against the oracle:

- clean argmax canvas, per-token entropy, Gumbel-sampled ids, **accept mask**, renoised canvas.

**Open decision (escalate, do not silently pick):** the pass/fail bar for
**accept/remask flips under bf16 (and later bfp8)** is a product-correctness call, not
an engineering default. Until set, **record the flip count** (`#positions where device
accept != oracle accept`, per step and summed over the block) in every trajectory test
and in the Part II status table. Target hypothesis = **0 flips over the block under bf16**; treat any
nonzero flip as a finding to report, not to suppress.

---

## W1 — #47474 KV-cache phase state machine  ✅ done **(was the prereq for W2/W3)** — residual: bounded-sliding commit-append wrap exercised but not yet verified (see Roadmap Phase 1 + Part II fix verification)

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

## W2 — #47462 bidirectional canvas attention (integration)  ✅ W2a · 🔴 W2b

**State:** mask geometry reference (`reference/attention_mask.py`, 8 tests) ✅ and an
**isolated** non-causal SDPA spike (`test_device_bidirectional_sdpa.py`, 4/4) ✅ are done.
**Net-new here = wiring the non-causal path into the real attention module**, not another
isolated SDPA call.

**The problem:** gemma4 prefill SDPA is hardcoded `is_causal=True`
(`models/demos/gemma4/tt/attention/prefill.py:126,264`, `operations.py:333`). Add a
non-causal path driven by an explicit `attn_mask`.

### ⚠️ CANONICAL DENOISE MASK = ALL-ATTEND (read this before touching the mask)

`reference/attention_mask.py` is the oracle and it is explicit (`modeling_diffusion_gemma.py:1399-1438`,
`bidirectional_mask_function`): the DiffusionGemma decoder is **fully bidirectional for
BOTH full-attention AND sliding layers**. `sliding_window` only shapes offsets / the SDPA
skip hint — it **NEVER restricts which keys a canvas query sees**. So the canonical denoise
mask is **all-attend**: every canvas query attends to **prompt + the entire canvas**, for
every layer type.

- **The denoise oracle is `build_canvas_denoise_mask(prompt_len, canvas_len)` with
  `local_window=False`** → all-zeros `[C, P+C]` additive mask. Use this, and only this, as
  the denoise reference.
- **The symmetric 2W+1 window is NON-canonical.** `local_window=True` exists ONLY to
  exercise the ttnn SDPA windowed-mask path (because `sliding_window_size` and `attn_mask`
  are mutually exclusive, `sdpa_device_operation.cpp:67-68`). It is an **optional
  op-capability test**, must **NOT** enter denoise acceptance, and must **NOT** be used as
  the denoise oracle. (Any earlier "local = symmetric 2W+1 for denoise" framing was wrong.)

### W2a — non-causal masked SDPA, prompt + canvas ≤ 32768  ✅ (the real W2 deliverable)
- Reference non-causal SDPA usage: `models/experimental/pi0/tt/ttnn_gemma.py:320`
  (`scaled_dot_product_attention(attn_mask=…, is_causal=False)`) and
  `models/tt_dit/encoders/gemma/model_gemma.py:253`.
- Add an `is_causal=False` / `attn_mask=` branch to `prefill.py` SDPA; gate by `kv_phase`
  from W1 (denoise → non-causal all-attend mask, prefill/commit → causal).
- Build the `[256, prompt_len+256]` mask from `build_canvas_denoise_mask(..., local_window=False)`
  (all-attend). Canvas absolute/RoPE positions are offset by `prompt_len`
  (`canvas_positions`). Cover the canvas→prompt prefix, not just an isolated 256 canvas.
- **Do NOT pass `sliding_window_size` in the denoise path — even for sliding layers.** In
  denoise you pass `attn_mask` (all-attend), and `attn_mask` ⊥ `sliding_window_size`
  (`sdpa_device_operation.cpp:67-68`); passing both trips the mutual-exclusion guard. The
  formerly-sliding layers attend fully during denoise, so the window simply does not apply.
  (`sliding_window_size` stays only on the causal prefill/commit paths.)
- **Acceptance (device):** `tests/test_device_bidirectional_attention.py` — a full-attn
  layer AND a (formerly-)sliding layer, both run with the **all-attend** denoise mask (and
  **neither** passing `sliding_window_size`); output PCC ≥ 0.99 vs a torch reference forward
  built on `build_canvas_denoise_mask(local_window=False)`, including the prompt prefix.
  **Same all-attend mask for both layer types** (that's the canonical check — a sliding
  layer must NOT window during denoise).
- **Optional op-capability test (does NOT gate W2):** a separate
  `test_device_windowed_mask_path` driving `local_window=True` purely to prove the ttnn
  SDPA masked path handles a windowed mask. Clearly label it non-canonical.

### W2b — long-prompt masked chunking, prompt + canvas > 32768  🔴 SEPARATE HIGH-RISK BLOCKER
> 📋 **Detailed spike-first plan: [`DEVICE_LOOP_W2B.md`](./DEVICE_LOOP_W2B.md).** Source investigation reframed this: the denoise attention is a `[256 × (P+C)]` *all-attend rectangle* (no mask, no causal logic), and the maskless non-causal SDPA path already exists — so W2b may be **near-zero kernel work** (lift the `prefill.py:180-181` guard), pending one gating spike (**S1**: does the existing op return correct results at `[256 × >32768]`?). If S1 fails, fall to host K-chunking or a paged kernel. See the W2b plan for the full decision tree.

**Do NOT bundle this into W2a acceptance.** The existing gemma4 chunked-prefill long-context
path is **causal-only** (`operations.py:25-29`, `prefill.py:106-130`) and `attn_mask` is
mutually exclusive with the windowing it relies on — so a **non-causal masked chunked path
is new kernel/path-level work, not wiring**. Track it as its own risk item: scope a spike
first (can ttnn SDPA chunk a `[256, >32768]` explicit mask at all, or does it need a new
op / a tiled-mask streaming scheme?). Functional milestone for short/medium prompts does
**not** depend on W2b; flag it to the user/manager as a standalone effort + likely-new-kernel risk.

---

## W3 — #47463 discrete-diffusion decode loop (device)  🔴 blocked on bf16 decision bar (control-flow implemented & validated)

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

## W4 — #47472 on-device canvas sampling  ✅ done — residual: SAMP-3 mesh-mapper `TT_FATAL` (latent) + regenerated-noise unvalidated at prod vocab (see Roadmap Phase 1 + Part II fix verification)

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

## Sequencing summary

```
W1 (#47474 KV phase machine)  ──► W2 (#47462 bidirectional attn) ──► W3 (#47463 decode loop) ──► W4 (#47472 sampling seam)
   prereq for everything           needs W1 phase flag              needs W1+W2              calls W3's primitives
```

W4's *primitives* already exist, so its **sampler** can be drafted in parallel once W3's
loop shape is settled; its **vLLM plumbing** can land last. Everything else is strictly ordered.

**When all four are ✅:** run the end-to-end block test (prefill→denoise→commit, trajectory
PCC), update the Part II status table + the parent tracker #47452, and report the decision-flip
numbers so the §3 bar can be set. Then stop the loop.

---

# Part II — Implementation status

Maps the [`plan.md`](./plan.md) workstreams to what is implemented in this
directory. Updated as work lands so progress is trackable per commit.

## Environment constraints (read first)

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

## Status by workstream

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
| KV-cache phase state machine | #47474 | ✅ done — `KVCachePhase` plumbing landed through Gemma4 model/layer/attention and Generator-compatible prefill/decode/verify wrappers; explicit `DENOISE_READONLY` skips cache writes. Validated 2026-06-25 with `tests/test_kv_phase.py` (3 passed), QB2 `test_single_layer_model[blackhole-sliding_only-1x4]` PCC **0.999936**, and QB2 `tests/test_device_kv_phase.py` (4 passed): readonly denoise leaves prompt K/V frozen-region byte-identical; `COMMIT_APPEND` decode writes the next cache position without mutating the prompt region; a 256-token canvas commit loop writes the full canvas region; 256-token commit-append canvas K/V matches one-shot re-encode by PCC. Canvas K/V scratch sizing added in `memory_budget.py`: QB2 TP=4 bf16 batch=1 ≈ **15 MiB/chip**. Page/circular-buffer mapping added in `kv_phase.py`: full-attn commit uses absolute positions; sliding commit uses `absolute_pos % sliding_window`. |
| Canvas mask geometry (reference, pure torch) | #47462 | ✅ done — `reference/attention_mask.py`, 8 tests pass |
| Bidirectional canvas SDPA on QB2 (device) | #47462 | ✅ **validated on QB2** — 4/4 PCC≥0.99 (full / symmetric-window / prompt-visible / GQA 16-8) on sfpi 7.60.0. ⚠️ device *teardown* re-hangs erisc 29-25 → reset between device runs. NOT a firmware issue: board fw is **19.9.0** (newer than tt-metal's tested 19.5.0); the assert's "min 18.10.0" is a hardcoded boilerplate string, not a version readout. Root cause undiagnosed (possibly fw ahead of the local UMD checkout); treat as an env quirk, work around with reset. |
| Self-conditioning gated MLP (reference, pure torch) | #47461/#47463 | ✅ done — `reference/self_conditioning.py`, 6 tests pass |
| Entropy-budget acceptance on QB2 (device) | #47463 (R1) | ✅ **validated on device (2026-06-22) — full chain `ttnn.sort`→`cumsum`→exclusive-prefix→`scatter` matches the oracle, 5/5 (`test_device_entropy_accept.py`).** The 2026-06-19 "device `ttnn.sort` returns garbage" conclusion was **WRONG** — it was a **degraded-board** artifact (erisc 29-25 fault), not a `ttnn.sort`-on-BH bug. On healthy HW `ttnn.sort` is correct: `test_sort_standard[…64…]` all pass; standalone repro (bf16/fp32, 2D `[64,64]`, 4D `[1,1,64,64]`, `[…,256]`) gives correct values+indices. **Host-side sort is unnecessary — the device chain works.** Two things were needed to validate: (1) a **consistent build** — the prebuilt `.so` (dev20260616) JIT-compiled source kernels (dev20260618) against its own headers → `tt_memmove` overload mismatch in the permute reader kernel; fixed by building the source tree (`build_metal.sh --disable-profiler`, run with `PYTHONPATH=$TT_METAL_HOME/ttnn:$TT_METAL_HOME` + `TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME`); (2) the device chain must use the **exclusive** prefix `(cum - sorted_vals) <= budget` to match HF `accept_canvas`, not inclusive `cum <= budget` (off-by-one at the boundary element). (Teardown still re-hangs erisc 29-25 each run — see SDPA row — so minimize device churn.) |
| Multi-canvas generation loop (reference, pure torch) | #47464 | ✅ done — `reference/generate.py`, 3 tests (commit-append, prefix-grows) |
| Bidirectional canvas attention (device SDPA integration, short/medium prompts) | #47462 (W2a) | ✅ **validated on QB2** — mask reference done; isolated non-causal SDPA spike is ✅ (`test_device_bidirectional_sdpa.py`, 4/4). Real Gemma4 prefill attention now accepts explicit `attn_mask` and routes to `is_causal=False` without `sliding_window_size`; rectangular denoise support lets canvas Q attend `[prompt; canvas]` K/V with canvas RoPE offset. `tt/denoise_forward.py` exposes W2 product wrappers: `denoise_attention_forward`, `denoise_logits_forward`, `denoise_logits_from_tokens`, `collect_prompt_hidden_by_layer` (legacy hidden-source shim), `collect_prompt_kv_by_layer` (projected prompt K/V), and `read_prompt_kv_cache_slice` (non-paged Gemma4 KV cache → projected prefix K/V via `ttnn.experimental.nlp_kv_cache_load_slice`). Validated with `tests/test_device_bidirectional_attention_integration.py` (4 passed): square all-attend smoke; prompt-prefix attention PCC≥0.99 for both `sliding_attention` and `full_attention`; token-driven full-canvas logits wrapper PCC≥0.98 after `PREFILL_WRITE` writes the prompt cache and denoise reads that cache slice, plus real `TtSelfConditioning` softmax→embedding→gated-MLP hook on mesh (full logits include known bf16 MoE/lm_head ceiling). `tests/test_device_self_conditioning.py` still passes 4/4 for the standalone module. |
| Paged / long-prompt denoise cache reader + masked chunking | #47462 (W2b) | ✅ done — S1/S2 resolved the core SDPA risk in favor of D1: regular non-causal SDPA passes `[256 × Sk]` through `Sk=262144`, `head_dim ∈ {256,512}` on QB2, both maskless and explicit all-zero masked. D1 is wired so canonical denoise uses maskless `is_causal=False` SDPA and no longer materializes the all-zero mask by default. S4 RoPE reachability is verified at 262144, and integrated tiny-model denoise attention PCC passes at `P+C=33280` and `P+C=262144`. Full W2b suite: `test_device_long_sdpa_w2b.py` 27 passed with `DG_W2B_SDPA_SWEEP=full`. Real 26B e2e generation remains #47464/#48291, not a W2b SDPA blocker. |
| Reference denoise trajectory (pure torch) | #47463/#47468 | ✅ done — `reference/denoise_loop.py`, 4 tests pass |
| Discrete-diffusion decode loop (device) | #47463 | 🔴 **blocked on bf16 decision bar / full-logits precision** — local `ttnn` build unblocked by syncing `tt_metal/third_party/tracy` and `tt_metal/third_party/umd` to the superproject pins, then rebuilding with `./build_metal.sh --disable-profiler`. W3 control-loop implementation is in place and validated on QB2 (2026-06-25): `tests/test_device_denoise_loop.py` 3 passed, entropy/accept harnesses 12 passed, and real-W2-logits integration tests passed (`test_denoise_logits_adapter_threads_prev_logits_for_self_conditioning`, `test_denoise_controller_real_logits_records_decision_flips`). `tt/denoise_loop.py` composes Gumbel-max, logsumexp-form entropy, exclusive-prefix accept, uint32-safe renoise, multi-step carry, clean-argmax commit, and stable+confident halting against `reference/denoise_loop.py`; the synthetic trajectory smoke uses 256 canvas positions, injected zero Gumbel + renoise ids, halts after 2 steps, passes `compare_trajectories`, and records 0 accept flips. `tt/denoise_forward.py` exposes `DenoiseLogitsAdapter`, a stateful W2 callback that threads previous-step logits into real self-conditioning for the controller while keeping logits on device; it also accepts controller-shaped `[1,1,L,1]` TILE token canvases. Real W2 logits smoke (1-layer, vocab=256, 2 denoise steps) runs end-to-end and records **accept_flips=[0,0]**, but also a precision finding: **argmax_flips=[225,222]**, **canvas_flips=[1,1]**, entropy PCC≈[0.624,0.653] vs torch. Triage shows drift is already present at logits: logits PCC≈[0.985,0.969] but logits argmax agreement≈[0.121,0.133]; reference top1/top2 margin is tiny (~0.005) while TT-vs-torch logits mean|Δ| is ~1.86/2.64, so argmax is margin-limited. Hidden-vs-logits diagnostic shows final hidden PCC≈0.9887 before lm_head, so this is not isolated to softcap/lm_head; dense (MoE-disabled) diagnostic improved logits PCC (~0.995/0.984) but still had ~0.125 argmax agreement and even accept_flips=[2,2], so this is not MoE-only. W3 should not be marked ✅ until either backbone/full-logits precision drift is reduced enough for the decision bar, or the bf16 diffusion decision bar is explicitly accepted/escalated by product. Since control-flow is implemented and blocked on that decision, the loop can proceed to independent W4 sampler work. |
| On-device canvas sampling | #47472 | ✅ done — deterministic exact path validated on QB2 (2026-06-25): `tests/test_device_canvas_sampling_exact.py` 3 passed. `tt/sampling.py` exposes `canvas_sample(logits, temperature, injected_gumbel_noise)` as the W4 released per-position draw (`argmax(logits/T + gumbel)`); tests feed the torch run's injected Gumbel noise and match sampled ids token-exact, including the params-routed seam, plus verify temperature scaling PCC≥0.9999. W4 sampling-params seam is in place: `tt/sampling_params.py` exposes `MODEL_CAPABILITIES["supports_sample_on_device"]=True`, duck-types vLLM `TTSamplingParams` fields (temperature/top_k/top_p/seed) into a per-step `CanvasSamplingConfig`, and `canvas_sample_from_params(...)` maps those params onto the device sampler; `tests/test_sampling_params.py` 5 passed. Seed-regenerated sampling now defaults to the permuted-vocab RNG path, which keeps one `ttnn.rand` draw per logits element but generates vocab as an outer axis before permuting back; explicit distributional tolerances pass on QB2 for both direct and params-routed paths (`N=4096`, max top1-frequency error≈0.0282, mean KL≈0.0129), while the slower vocab-chunk diagnostic also passes (`max top1-frequency error≈0.0324`, mean KL≈0.0035). The raw single-call `ttnn.rand[..., vocab]` path remains as a strict-xfailed diagnostic (`max top1-frequency error≈0.179`, mean KL≈0.651`) because torch consuming the same raw noise exactly reproduces the biased samples, proving the issue is RNG axis correlation rather than sampler arithmetic/argmax; this raw path is not the released params default. |
| Functional e2e / perf / vLLM / batched / multimodal / quant / CI | #47464+ | ⬜ not started |

Legend: ✅ done · 🚧 in progress · ⛔ blocked on environment · ⬜ not started

## Code review — 2026-06-26 (multi-agent review of the 2026-06-25 branch)

Adversarial multi-agent review of the 48 commits `d13c3ad0c91..HEAD` (~3834 LOC) across #47474/#47462/#47463/#47472/#47464, plus a gemma4-regression sentinel and a test-rigor pass. **27 findings confirmed, 3 dismissed as false positives** (listed at the end so they are not re-raised). Each finding was independently re-verified against the code before landing here.

**Headline — no production gemma4 regression.** Every new parameter on the shared `models/demos/gemma4/tt/**` path (`kv_phase=None`, `write_kv_cache=True`, `attn_mask=None`, `kv_hidden_states=None`, `prefix_kv=None`, `q_rope_offset=0`) defaults to the pre-branch op sequence / dtype / order **bit-for-bit**: `coerce_kv_cache_phase(None)` → `PREFILL_WRITE`/`COMMIT_APPEND` → `write_kv_cache=True`; the new readonly/masked/prefix-KV branches are non-default-gated and raise on misuse; `prefill_sdpa_program_config` returns identical chunk sizes on the power-of-2 prefill buckets production actually uses. Risk is concentrated in **(a) device-loop deallocation discipline** and **(b) a few device tests that gate on nothing**.

### 🔴 Must-fix

- ✅ **Fixed 2026-06-26 [#47472] `gumbel_max` / `canvas_sample` leaked the full `[B,L,vocab]` intermediates every call → 48-step loop OOM** — `tt/sampling.py:73-93`. `z = temperature_scale(...)` (new tensor when T≠1, i.e. the 0.8→0.4 schedule) and `perturbed = ttnn.add(z, noise)` are now deallocated after the output tensor is created; the same temporary-scale cleanup is applied to `token_entropy` / `softmax`. Validated on QB2 with `test_device_canvas_sampling_exact.py` (3 passed) plus `test_device_entropy_harness.py`, `test_device_denoise_loop.py`, and `test_device_self_conditioning.py` (14 passed).
- ✅ **Fixed 2026-06-26 [#47462/#47463] `test_denoise_controller_real_logits_records_decision_flips` no longer disables every trajectory threshold without replacement** — `tests/test_device_bidirectional_attention_integration.py:631-702`. The test is explicitly a diagnostic for the known bf16 decision-bar blocker, but now gates both MoE and dense modes on real-logits PCC, top-8 contains-reference-argmax, and a bounded accept-flip count. Validated on QB2 with `HF_MODEL=/home/zni/dg_models/gemma-4-26B-A4B-it`: MoE logits PCC **0.985/0.969**, accept flips **0/0**; dense logits PCC **0.995/0.984**, accept flips **2/2**; both parametrized cases passed.
- ✅ **Fixed 2026-06-26 [#47463/#47464] accept decision path now has an explicit on-device `ttnn.sort` regression guard** — `tt/denoise_loop.py:43-72`, `tests/test_device_entropy_accept.py`. `entropy_budget_accept` now cross-references the Part I decision-fidelity bar, and `test_production_entropy_budget_accept_guards_device_sort_at_canvas_256` directly validates the production sort→cumsum→exclusive-prefix→scatter path against the host oracle at the real 256-token canvas length, including accept count and mask equality. Validated on QB2 with `test_device_entropy_accept.py` + `test_device_denoise_loop.py` (9 passed).

### 🟡 Should-fix

- ✅ **Fixed 2026-06-26 [#47463/#47464] `denoise_block` deallocates per-step decision tensors and superseded canvas tensors** — `tt/denoise_loop.py:156-230`. After host readback, `res.argmax/entropy/sampled/accept_mask` are freed; each consumed canvas is deallocated when replaced, and the final device canvas is freed before returning the host-only trajectory. Validated on QB2 with `test_device_denoise_loop.py` (3 passed) and the real-logits controller target in `test_device_bidirectional_attention_integration.py` (2 passed).
- ✅ **Fixed 2026-06-26 [#47474] `decode` + `DENOISE_READONLY` is rejected before `decode_forward` can drop the current token's K/V** — `models/demos/gemma4/tt/attention/kv_phase.py`, `tests/test_kv_phase.py`. `coerce_kv_cache_phase(..., is_decode=True)` now raises `ValueError` for `DENOISE_READONLY`, preserving the default decode `COMMIT_APPEND` path and the prefill-only readonly path. Validated with `test_kv_phase.py` (4 passed) and QB2 `test_device_kv_phase.py` (4 passed).
- ✅ **Fixed 2026-06-26 [#47474] Bounded-sliding commit-append wrap is now covered in the device path** — `tests/test_device_kv_phase.py`. `test_bounded_sliding_commit_append_wraps_cache_slot` builds a one-layer sliding Gemma4 model with `bounded_sliding_kv_cache=True`, a small paged sliding window, and a vLLM-style zero-padded per-layer page table; it drives `COMMIT_APPEND` decode at `position == sliding_window` and asserts the wrapped physical cache slot changes for both K and V. Validated with `test_kv_phase_mapping.py` + QB2 `test_device_kv_phase.py` (13 passed).
- ✅ **Fixed 2026-06-26 [#47464] Multi-step device-loop constant-logits test is now explicitly scoped as a control-flow smoke** — `tests/test_device_denoise_loop.py`. The synthetic test was renamed to `test_multi_step_denoise_control_flow_smoke_matches_reference` and documents that it does not exercise canvas→backbone→renoise cycling; real W2 logits cycling remains covered by `test_denoise_controller_real_logits_records_decision_flips` in `test_device_bidirectional_attention_integration.py`. Validated on QB2 with `test_device_denoise_loop.py` (3 passed) and the real-logits controller target (2 passed).

### 🟢 Low / hardening

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

### Dismissed — verified false positives (no action)

- **`ttnn.sort` risk "rides on" the integration test** — refuted: the `accept_flips==0` assertion is itself a host-sort-vs-device-sort cross-check, and `test_single_denoise_step_matches_reference` validates the sort chain element-exact. (The residual doc-vs-fact divergence is captured as the Must-fix above.)
- **Synthetic fp32 loop test is the only coverage of decision fidelity** — refuted: the real bf16 path is covered by the controller diagnostic test (which is the H2 finding's actual weakness — thresholds disabled, not absence of the test).
- **Removing `test_full_model[blackhole-1x4]=0.83` from the shared `pcc_thresholds.json` reverts to 0.99 and would fail** — refuted: the 26B MoE `test_full_model` `pytest.skip`s at `tp<8`, so on a 1×4 mesh (tp=4) it never reaches `compare_tensors`; the removed entry was dead for that combo. The removal correctly de-pollutes the shared production gate; DiffusionGemma's PCC gap is handled in `test_device_backbone_pcc.py`.

### Fix verification — 2026-06-26 (independent re-check of the 23-commit fix campaign)

A second multi-agent pass independently verified all 23 fix commits at snapshot `03c40727c48` — for each fix: (a) does it resolve the finding, (b) does it introduce a regression, (c) is the added test real (would it fail if the bug regressed)? Every not-clean verdict was adversarially re-checked against the code. **Result: 20/25 fixes fully clean; the production `gemma4` path is provably unchanged.**

**Production `gemma4` cleared.** The three shared-code fixes are bit-for-bit safe: every production caller (`ttnn_prefill/decode/verify_forward`) passes `kv_phase=None` → safe default and never trips the new `coerce_kv_cache_phase` guards (isolated-run confirmed); `_largest_tile_divisor` is identical to the old `min()` on every power-of-2 prefill bucket (brute-forced); `q_rope_offset=0` / `_slice_rope_cache(start=0)` pass the new asserts. The H1/M1/M2/DENO-* dealloc+guard fixes are confirmed with **no double-free / use-after-free** (`committed` is the host copy of `res.argmax`, freed device tensors are not read; `z` from `temperature_scale` is always a fresh tensor).

- 🔴 **NEW BUG introduced by the SAMP-3 fix — `_rand_mesh_mapper` will `TT_FATAL` on the QB2 1×4 mesh** — `tt/sampling.py:121-124` (commit `cab7f9955e8`). The fix added `ttnn.MeshMapperConfig([ttnn.PlacementReplicate()])` (placements size 1) with **no `mesh_shape_override`**; `ttnn.rand` (`rand.cpp:69-78`) asserts `placements.size() == device.shape().dims()`, and QB2 opens as `MeshShape(1,4)` → `dims()==2`, so `1 != 2` → hard fatal on the exact multi-device mesh the feature targets (pre-fix `mesh_mapper=None` did not crash). **Latent**: regenerated-noise is opt-in/diagnostic after SAMP-2 and the production path injects host noise, so it is off the hot path — but it is a guaranteed crash once that path runs on multi-device, and untested (single-device fixture never exercises the `>1`-device branch). **Fix:** add `mesh_shape_override=ttnn.MeshShape([device.get_num_devices()])`, matching `models/common/modules/rmsnorm/rmsnorm_1d.py:388`.
- 🟡 **M3 wrap test is non-discriminating (the #1 KV hazard is still not actually verified)** — `tests/test_device_kv_phase.py:284-350` (commit `3b15e15439e`). It activates the bounded-sliding path (real model + decode + `cache_position_modulo==64` assert), but at `position=64, sliding_window=64` both correct-wrap and broken-no-wrap resolve to the **same** physical slot (block 0 row 0 — the zero-padded page-table tail also maps there), and `_assert_regions_changed` checks only that one slot changed. Reverting `cache_position_modulo` still passes. Fix: pick a position that is `block_size`-aligned but **not** `sliding_window`-aligned (so wrap vs no-wrap land on different physical blocks) and assert a non-wrapped slot stays untouched. ⇒ refines the W1 header caveat: "wrap untested" → "wrap exercised, correctness not yet verified".
- 🟢 **Fix-correct but regression-unguarded (low):**
  - **H1 / M1** — the dealloc fixes are correct, but no allocator high-water-mark test exists and the loop test halts at 2 steps (never the 48-step cap), so a *re-introduced* leak would be silent in CI. (M1 also leaks `init_canvas` in the degenerate `max_denoise_steps==0` config — non-production.) ⇒ resolves the W4 header caveat: the `gumbel_max` leak **is** fixed; only the leak-regression *guard* is missing.
  - **KV-P-4** — fix is production-identical; only the `(100,100)==32` assertion distinguishes new from old — `(384,256)==192` / `(512,256)==256` pass under both (sanity checks, not regression guards).

**Net:** the campaign is solid; the only item that can bite before use is the SAMP-3 mesh-mapper fatal (latent), then the M3 test gap. Neither touches the production `gemma4` path.

## Session 2026-06-22 — #47468 / #47461 / #47487 push (QB2-only)

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

## Build order (env-independent first)

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
