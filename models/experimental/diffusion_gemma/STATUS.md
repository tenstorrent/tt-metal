# DiffusionGemma bring-up — implementation status

Maps the [`plan.md`](./plan.md) workstreams to what is implemented in this
directory. Updated as work lands so progress is trackable per commit.

## Environment constraints (read first)

This dev environment **cannot run weight-validated bring-up**:

- `transformers` here is **4.53.0**; `gemma4` needs **5.10.2** and
  **`diffusion_gemma` is in neither** → the HF torch reference (#47468) is not
  importable yet.
- The gated **~46 GB** checkpoints and **T3K / QB2** hardware are not present.

So work proceeds **env-independent-first**: pure-torch reference logic + config
+ tests that run on CPU, with checkpoint/HW/transformers-gated pieces scaffolded
and marked `TODO(env)`.

## Status by workstream

| Item | Plan | Status |
|---|---|---|
| Module scaffolding | — | ✅ package + config |
| `config.py` (verified hyperparams) | §2 | ✅ done |
| **Diffusion sampling primitives (reference, pure torch)** | #47463 spike / #47468 oracle | ✅ done — `reference/sampling.py`, 11 tests pass |
| PCC trajectory harness (validates decisions) | #47468 | ✅ done — `tests/trajectory_pcc.py`, 5 tests pass |
| HF reference adapter seam (mock-tested) | #47468 | ✅ done — `reference/hf_reference.py`, 3 tests (guard + mock end-to-end) |
| Torch reference model (real HF load) | #47468 | ⛔ blocked: transformers `diffusion_gemma` unavailable — drops into the adapter seam once installed |
| Causal backbone bring-up (gemma4 reuse) | #47461 | ⛔ blocked: ckpt + transformers 5.x + HW |
| KV-cache phase state machine | #47474 | ⬜ not started |
| Canvas mask geometry (reference, pure torch) | #47462 | ✅ done — `reference/attention_mask.py`, 8 tests pass |
| Self-conditioning gated MLP (reference, pure torch) | #47461/#47463 | ✅ done — `reference/self_conditioning.py`, 6 tests pass |
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

**The env-independent reference layer is complete (34 CPU tests pass).** It
pins every net-new *algorithm* — sampling/acceptance (#47463), denoise
trajectory (#47463), mask geometry (#47462), self-conditioning (#47461/#47463),
and the decision-level PCC harness (#47468) — so the device port has an oracle
to validate against. Remaining work is environment-gated:

5. ⛔ Vendored HF reference wrapper — unblocks once `transformers` ships
   `diffusion_gemma` (then plug it into the trajectory harness).
6. ⛔ Device (`tt/`) implementation — backbone reuse (#47461), KV phase machine
   (#47474), bidirectional SDPA (#47462), device decode loop (#47463),
   on-device sampling (#47472) — unblocks on T3K/QB2 + checkpoints.
