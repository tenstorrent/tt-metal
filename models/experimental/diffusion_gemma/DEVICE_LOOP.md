# DiffusionGemma device bring-up — agent loop spec

**Audience:** an autonomous agent running in `/loop` mode on the QB2 box `bh-qbge-06`.
**Goal of the loop:** take the four device-integration workstreams below from ⬜ to ✅,
in dependency order, each validated on QB2 against the pure-torch oracle in `reference/`.

This is the **Functional-core** device work. Everything it depends on (the torch
reference/oracle, the PCC harness, the causal backbone PCC, the isolated device
spikes for entropy/accept/SDPA/self-cond) is **already done** — see `STATUS.md`.
Do **not** redo it. This loop turns the validated *pieces* into an *integrated*
on-device diffusion forward.

---

## 0. Loop protocol (do this every iteration)

> **Progress tracking is branch-dependent.** The `zni/diffusion-gemma-bringup` branch has a
> `STATUS.md` / `plan.md`; the `zni/diffusion-gemma-foundation-draft` branch does **not**
> (planning docs were stripped for the foundation PR). So: **if `STATUS.md` exists, update it;
> otherwise maintain the `## Progress log` section at the bottom of THIS file.** Everywhere
> below that says "update `STATUS.md`" means "update `STATUS.md` if present, else the Progress
> log here." Confirm with the user which branch the device loop runs on before iteration 1.
>
> **STALE-ROW WARNING:** the `bringup` `STATUS.md` still carries a row
> `Bidirectional canvas attention (device SDPA) ⬜ not started`. That refers to the **integration**
> (W2a here), NOT the isolated SDPA spike — the isolated spike **is done** (`test_device_bidirectional_sdpa.py`,
> 4/4). If you run on `bringup`, fix that row first (split spike-done ✅ vs integration-not-started ⬜)
> so you don't mis-read it as "no SDPA work exists." The Progress log below already states it correctly.

1. **Read the progress source** (`STATUS.md` if present, else the Progress log below) → find
   the first workstream not yet ✅ in the order W1→W2→W3→W4. Within a workstream, pick the
   first unchecked task in its checklist.
2. **Do the smallest shippable increment** of that task (one module / one device test).
3. **Validate on device** (recipe §1). The oracle is always `reference/` — assert the
   ttnn output matches it (PCC or `torch.equal`), never assert against a fresh guess.
4. **Update `STATUS.md`**: flip the row, note the test file + measured PCC + date.
5. **Commit ONLY IF commits are explicitly enabled** for this loop (user/loop config says
   so) AND the device test in step 3 passed. Never commit a half-finished increment, a
   failing test, or an environment workaround (e.g. the erisc reset, a local build). If
   commits are not enabled, leave the work in the tree and record progress in `STATUS.md`
   instead. When you do commit: see §2 ground rules — **NO `Co-Authored-By` trailer**.
6. If blocked (HW fault, missing op, oracle disagreement you can't resolve in one
   step): write the blocker into `STATUS.md` under the workstream, leave the row 🚧
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
and in `STATUS.md`. Target hypothesis = **0 flips over the block under bf16**; treat any
nonzero flip as a finding to report, not to suppress.

---

## W1 — #47474 KV-cache phase state machine  ⬜ **(do first — prereq for W2/W3)**

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

## W2 — #47462 bidirectional canvas attention (integration)  ⬜

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

### W2a — non-causal masked SDPA, prompt + canvas ≤ 32768  (the real W2 deliverable)
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
**Do NOT bundle this into W2a acceptance.** The existing gemma4 chunked-prefill long-context
path is **causal-only** (`operations.py:25-29`, `prefill.py:106-130`) and `attn_mask` is
mutually exclusive with the windowing it relies on — so a **non-causal masked chunked path
is new kernel/path-level work, not wiring**. Track it as its own risk item: scope a spike
first (can ttnn SDPA chunk a `[256, >32768]` explicit mask at all, or does it need a new
op / a tiled-mask streaming scheme?). Functional milestone for short/medium prompts does
**not** depend on W2b; flag it to the user/manager as a standalone effort + likely-new-kernel risk.

---

## W3 — #47463 discrete-diffusion decode loop (device)  ⬜

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

## W4 — #47472 on-device canvas sampling  ⬜

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
PCC), update the progress source + the parent tracker #47452, and report the decision-flip
numbers so the §3 bar can be set. Then stop the loop.

---

## Progress log (use this when the branch has no `STATUS.md`)

Update the status + a one-line note (test file, measured PCC / flip count, date) each iteration.

| Workstream | Status | Last note |
|---|---|---|
| W1 #47474 KV phase machine | ⬜ not started | — |
| W2a #47462 non-causal masked SDPA (≤32768) | ⬜ not started | isolated SDPA spike ✅ (`test_device_bidirectional_sdpa.py` 4/4); mask ref ✅ all-attend |
| W2b #47462 long-prompt masked chunking (>32768) | 🔴 separate risk | needs spike; likely new kernel/path; not gating Functional |
| W3 #47463 device denoise loop | ⬜ not started | primitives ✅ (entropy/accept/self-cond/gumbel) |
| W4 #47472 on-device canvas sampling + vLLM seam | ⬜ not started | `tt/sampling.py` primitives ✅ |

Legend: ⬜ not started · 🚧 in progress · ✅ done · 🔴 blocked/high-risk
