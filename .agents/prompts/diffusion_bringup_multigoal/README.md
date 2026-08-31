# diffusion_bringup_multigoal

Automated multi-goal track for bringing up a **diffusion** model (video/image/audio DiT + VAE +
text-encoder pipeline) on Tenstorrent hardware in `tt_dit`. It is the diffusion counterpart of
`model_bringup_multigoal` (which is for autoregressive LLM decoders and does not apply here — diffusion
has no KV-cache, paged decode, or token loop).

Launch with the existing runner pointed at this directory:

```bash
.agents/scripts/multigoal \
   --replace DIFFUSION_MODEL=org/Your-Model-Here \
   --replace MODEL_DIR=models/tt_dit/models/transformers/org_your_model_here \
   .agents/prompts/diffusion_bringup_multigoal/*.txt
```

Each `NN-<stage>.txt` is a `/goal` that invokes the relevant diffusion skill(s) and states checkable
completion requirements. Goals run in numeric order; a stage with a `NN-<stage>.check.sh` is gated by
that script (exit 0 pass, 1 advisory, 2 critical, 3 error).

## Stages

| # | stage | skills | gate |
|---|---|---|---|
| 01 | functional DiT block | `$functional-dit-block` `$adaln-conditioning` `$multiaxis-rope` | — |
| 02 | full DiT | `$diffusion-model-bringup` | — |
| 03 | text encoder | `$text-encoder-port` | — |
| 04 | video VAE | `$vae-port` | — |
| 05 | audio VAE | `$vae-port` | — |
| 06 | scheduler | `$denoise-loop-scheduler` | — |
| 07 | end-to-end pipeline | `$diffusion-full-pipeline` `$diffusion-qualitative-check` | `07-pipeline.check.sh` |
| 08 | multichip | `$multichip` | — |
| 09 | optimize + trace | `$optimize` `$tt-enable-tracing` | — |
| 10 | datatype sweep + readiness | `$diffusion-datatype-sweep` | `10-datatype-sweep.check.sh` |

The orchestration/contract umbrella is `$diffusion-model-bringup`.

## Contract + gate scripts (in `.agents/scripts/`)

- `check_diffusion_contract.py` — validates `capability_contract.json` (modalities, resolution, frames,
  fps, audio rate, latent shapes, denoise steps) and that any capability **reduction** carries a hard
  device-limit reason + evidence. Diffusion analog of `check_context_contract.py`.
- `check_diffusion_degenerate.py` — non-degeneracy gate for a generated artifact: no NaN/Inf, frames not
  frozen/all-black/all-white, audio not silent, in range. Diffusion analog of the LLM
  `check_degenerate_output.py`. Distinguishes degeneracy (fail) from low fidelity (allowed).

## Env vars the gates read

- `MODEL_DIR` — the model's tt_dit dir (its `doc/capability_contract.json` and
  `doc/**/selected_precision_config.json` are looked up here). Or `HF_MODEL` to slug-locate it.
- `DIFFUSION_OUT` — the generated-artifact directory (containing `frames/` and/or `audio.wav`) that
  `07-pipeline.check.sh` scores. Falls back to the newest `frames/` dir under `MODEL_DIR/doc`.

A worked example of every artifact these gates expect (a passing `capability_contract.json`,
`selected_precision_config.json`, and a non-degenerate clip) lives in the MiniMax-H3 bringup under
`models/tt_dit/models/transformers/minimax_h3/doc/`.
