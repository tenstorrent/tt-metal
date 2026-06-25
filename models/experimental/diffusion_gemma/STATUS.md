# DiffusionGemma bring-up — implementation status

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
| Bidirectional canvas attention (device SDPA integration) | #47462 | 🚧 in progress — mask reference done; isolated non-causal SDPA spike is ✅ (`test_device_bidirectional_sdpa.py`, 4/4). Real Gemma4 prefill attention now accepts explicit `attn_mask` and routes to `is_causal=False` without `sliding_window_size`; rectangular denoise support lets canvas Q attend `[prompt; canvas]` K/V with canvas RoPE offset. `tt/denoise_forward.py` now exposes the W2 product wrapper (`denoise_attention_forward`) that builds the canonical all-attend mask, concatenates prompt+canvas K/V, and runs readonly denoise attention. Validated on QB2 with `tests/test_device_bidirectional_attention_integration.py` (3 passed): square all-attend smoke plus wrapper-driven prompt-prefix PCC for both `sliding_attention` and `full_attention`. Next: lift this per-layer wrapper into a multi-layer denoise logits forward. |
| Reference denoise trajectory (pure torch) | #47463/#47468 | ✅ done — `reference/denoise_loop.py`, 4 tests pass |
| Discrete-diffusion decode loop (device) | #47463 | ⬜ not started (reference logic done) |
| On-device canvas sampling | #47472 | ⬜ not started |
| Functional e2e / perf / vLLM / batched / multimodal / quant / CI | #47464+ | ⬜ not started |

Legend: ✅ done · 🚧 in progress · ⛔ blocked on environment · ⬜ not started

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
