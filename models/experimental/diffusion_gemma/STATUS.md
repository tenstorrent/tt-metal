# DiffusionGemma bring-up — implementation status

Maps the [`plan.md`](./plan.md) workstreams to what is implemented in this
directory. Updated as work lands so progress is trackable per commit.

## Environment constraints (read first)

This box is **`bh-qbge-06` — a QB2 (4× Blackhole `p300c`, `/dev/tenstorrent/0..3`)**, so **device work is NOT blocked on hardware**. The remaining gates are software + data:

- **Dedicated env created (2026-06-19):** `/home/zni/venvs/tt-diffusion-gemma` (Python 3.12, **transformers 5.10.2**, torch 2.11+cpu, ttnn editable from the repo) — isolated from the default `python_env`, which stays at 4.53.0 for LTX. Verified: `transformers.models.gemma4` imports, **`ttnn` sees 4 QB2 devices**, `uv pip check` clean, 40 reference tests pass. Use: `source /home/zni/venvs/tt-diffusion-gemma/bin/activate && export PYTHONPATH=/home/zni/tt-metal TT_METAL_HOME=/home/zni/tt-metal`.
- **`diffusion_gemma` is NOT in transformers 5.10.2** → the *DiffusionGemma* HF reference (#47468 real load) still needs a newer transformers (main / future release) or stays on the `reference/hf_reference.py` adapter seam. The **gemma4 backbone** path (#47461 / #47487) is fully unblocked now.
- **Gated checkpoints not downloaded** — HF cache has only `gemma-3-12b-it-qat`; no `gemma-4-26B-A4B` / `diffusiongemma` (disk has ~2.4 TB free — fits the ~51.7 GB bf16 — but Gemma is gated, needs HF auth + license).
- **QB2 is present** (4× Blackhole, this box); **T3K (WH 1×8) is not** — but fitting 26B-A4B on QB2 (1×4) is itself net-new (#47487), and the in-repo gemma4 **12B** path is QB2-supported and can validate the on-device flow on this exact HW first.

So work proceeds **env-independent-first**: pure-torch reference logic + config
+ tests that run on CPU, with checkpoint/transformers-gated pieces scaffolded
and marked `TODO(env)`. **HW + env are no longer blockers — QB2 is local and the dedicated transformers-5.10.2 env is built.** The only remaining gate for gemma4-backbone device bring-up is the **gated checkpoint download** (needs `hf auth login` + Gemma license acceptance).

## Status by workstream

| Item | Plan | Status |
|---|---|---|
| Module scaffolding | — | ✅ package + config |
| `config.py` (verified hyperparams) | §2 | ✅ done |
| Config reconciliation vs real 26B-A4B config.json (`from_hf_config`) | #47461 | ✅ done — `tests/test_config.py`, all fields confirmed in sync |
| **Diffusion sampling primitives (reference, pure torch)** | #47463 spike / #47468 oracle | ✅ done — `reference/sampling.py`, 11 tests pass |
| PCC trajectory harness (validates decisions) | #47468 | ✅ done — `tests/trajectory_pcc.py`, 5 tests pass |
| HF reference adapter seam (mock-tested) | #47468 | ✅ done — `reference/hf_reference.py`, 3 tests (guard + mock end-to-end) |
| Torch reference model (real HF load) | #47468 | ⛔ blocked: transformers `diffusion_gemma` unavailable — drops into the adapter seam once installed |
| Causal backbone bring-up (gemma4 reuse) | #47461 | ⛔ blocked: ckpt + transformers 5.x + HW |
| KV-cache phase state machine | #47474 | ⬜ not started |
| Canvas mask geometry (reference, pure torch) | #47462 | ✅ done — `reference/attention_mask.py`, 8 tests pass |
| Bidirectional canvas SDPA on QB2 (device) | #47462 | ✅ **validated on QB2** — 4/4 PCC≥0.99 (full / symmetric-window / prompt-visible / GQA 16-8) on sfpi 7.60.0. ⚠️ device *teardown* re-hangs erisc 29-25 → reset between device runs. NOT a firmware issue: board fw is **19.9.0** (newer than tt-metal's tested 19.5.0); the assert's "min 18.10.0" is a hardcoded boilerplate string, not a version readout. Root cause undiagnosed (possibly fw ahead of the local UMD checkout); treat as an env quirk, work around with reset. |
| Self-conditioning gated MLP (reference, pure torch) | #47461/#47463 | ✅ done — `reference/self_conditioning.py`, 6 tests pass |
| Entropy-budget acceptance on QB2 (device) | #47463 (R1) | ⚠️ **spike conclusion (2026-06-19): device `ttnn.sort` is unusable on Blackhole → sort on host.** After `tt-smi -r` the device opens and sort *runs*, but returns **garbage** (uninitialized values, e.g. `5.5e13` / `-2.1e32` / `1.2e36`) for bf16 **and** fp32, 2D **and** 4D `[1,1,64,64]` — a `ttnn.sort`-on-BH bug, **not** dtype/shape/board-state (corroborated by the proven `test_sort_standard[bf16,[64,64]]` failing). Code committed (sort→cumsum→le→scatter vs oracle). **Decision:** acceptance reads back the 256 per-position confidences, does argsort+cumsum **on host**, scatters the accept mask back to device. (Teardown still re-hangs erisc 29-25 each run — see SDPA row — so minimize device churn.) |
| Multi-canvas generation loop (reference, pure torch) | #47464 | ✅ done — `reference/generate.py`, 3 tests (commit-append, prefix-grows) |
| Bidirectional canvas attention (device SDPA) | #47462 | ⬜ not started (mask reference done; non-causal SDPA path blocked on HW) |
| Reference denoise trajectory (pure torch) | #47463/#47468 | ✅ done — `reference/denoise_loop.py`, 4 tests pass |
| Discrete-diffusion decode loop (device) | #47463 | ⬜ not started (reference logic done) |
| On-device canvas sampling | #47472 | ⬜ not started |
| Functional e2e / perf / vLLM / batched / multimodal / quant / CI | #47464+ | ⬜ not started |

Legend: ✅ done · 🚧 in progress · ⛔ blocked on environment · ⬜ not started

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
   on-device sampling (#47472) — **QB2 hardware is present (this box)**; unblocks on transformers-5.x + checkpoint download (no longer waiting on HW).
