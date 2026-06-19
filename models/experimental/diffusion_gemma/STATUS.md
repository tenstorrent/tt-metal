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
| Torch reference model (vendored HF) | #47468 | ⛔ blocked: transformers `diffusion_gemma` unavailable |
| Causal backbone bring-up (gemma4 reuse) | #47461 | ⛔ blocked: ckpt + transformers 5.x + HW |
| KV-cache phase state machine | #47474 | ⬜ not started |
| Bidirectional canvas attention | #47462 | ⬜ not started |
| Discrete-diffusion decode loop (device) | #47463 | ⬜ not started (reference logic first) |
| On-device canvas sampling | #47472 | ⬜ not started |
| Functional e2e / perf / vLLM / batched / multimodal / quant / CI | #47464+ | ⬜ not started |

Legend: ✅ done · 🚧 in progress · ⛔ blocked on environment · ⬜ not started

## Build order (env-independent first)

1. ✅ Config + scaffolding.
2. ✅ **Reference sampling primitives** (`reference/sampling.py`) + tests — the
   `#47463` acceptance spike reference and the `#47468` oracle's sampling core.
   Pure torch, CPU-testable, no checkpoint.
3. 🚧 Reference denoise loop (assembling the primitives into the per-block
   trajectory) + tests.
4. ⛔ Vendored HF reference + PCC harness — unblocks once `diffusion_gemma` is
   installable.
5. ⛔ Device (`tt/`) implementation — unblocks on HW.
