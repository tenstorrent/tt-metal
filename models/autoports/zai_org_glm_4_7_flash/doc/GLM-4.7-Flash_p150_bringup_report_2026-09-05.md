# GLM-4.7-Flash on a single Blackhole p150: bring-up report

**Model:** `zai-org/GLM-4.7-Flash` (`Glm4MoeLiteForCausalLM`, 30.6B total / ~3.6B active,
47 layers, 64 experts top-4, MLA attention, vocab 154880, advertised context 202752)
**Hardware:** one Blackhole p150, 1x1 mesh, ~32 GB device DRAM
**Branch:** `ttmodelmanager/glm47-flash-probe` (pushed to `tenstorrent/tt-metal`)
**Report date:** 2026-09-05

> **Prerequisite: this branch does not load on its own.** vLLM resolves the
> checkpoint architecture `Glm4MoeLiteForCausalLM` through a registration that
> lives in a *different* repository, and it is not yet merged upstream. Without
> it you get a missing-architecture error at server start, not a working model.
> Until it merges, check out the plugin branch alongside this one:
>
> - registration: `ttmodelmanager/glm47-flash-registration` on
>   `github.com/stisiTT/vllm-tt-plugin` (a fork of `tenstorrent/vllm-tt-plugin`;
>   commit `9f2ec5d`, a single 9-line additive entry in
>   `src/vllm_tt_plugin/platform.py`)
> - open PR upstream:
>   `https://github.com/stisiTT/vllm-tt-plugin/pull/new/ttmodelmanager/glm47-flash-registration`
>
> The registration points at `models.autoports.zai_org_glm_4_7_flash`, so the two
> branches are only useful as a pair.

---

## Bottom line

The model runs on a single p150 at its **full advertised 202,752-token context with no
capability reduction**, and serves through vLLM end to end. All nine pipeline stages
completed with their runner-side gates passing. Two real latent bugs were found and
fixed on hardware, and one defect was traced to the shared stack and filed upstream.

The one qualification a reader must carry: **sampling is `smoke-gated`, not full-gated**,
by explicit owner acceptance, and the release accuracy figures are **CI-subset (5%)**,
not full-set. Neither is presented as more than it is.

---

## Headline numbers

| | value | how measured |
|---|---|---|
| **Context served** | **202,752** (full HF advertised) | prompt lengths 10000 / 65536 / 131071 / 131072 / **202751** all served post-fix |
| **TTFT** (128-token prompt) | **273.9 ms** | primary single-user profile at spec |
| **Decode, batch 1** | **29.50 ms/token = 33.9 t/s/u** | reproduced within 0.02% on an independent re-run |
| Serving burst 100/100/32 | 137.1 tok/s output, 274.2 tok/s total | CI-parity profile; not headline decode |
| Full-model traced floor | 22.99 ms/token (43.5 t/s/u) | the lower bound serving is compared against |
| Model accuracy (full model) | top-5 **1.000**, top-100 **1.000** | vs HF reference, AIME24 chat template |
| Release eval (ifeval, CI subset) | prompt-level strict **0.714**, inst-level strict **0.744** | 5% sample, not full-set |

Serving decode went from **2.0x** the traced floor to **1.28x** over the course of the
optimization stage.

---

## What was fixed

**VS-008, prefill sampling parameters reached a lane nothing reads.** A request's
temperature/top_k/top_p/penalties were written to sampler lane 0, but prefill reads the
token from lane `(seq-1) % 32`, a prompt-position index. For any prompt longer than one
token that lane held padded *greedy* defaults, so **every first token was sampled greedily
regardless of what the client asked for**. A correctness defect, not a performance one.
Fixed by describing one request's params on all 32 lanes before formatting. Confirmed on
hardware: 11 previously failing tests flip to pass, zero regressions.

**TR-001, the whole-prompt prefill activation pair was never reserved.** The release
benchmark ladder killed the vLLM engine with a hard OOM at 65,536-token prompts.
`run_layer_stack_prefill` holds the layer input and output live at once, 8192 bytes per
prompt token, and that prompt-scaled allocation sat outside the vLLM KV pool budget:
1.66 GB unaccounted at full context. The earlier VS-006 margin covered only the
chunk-scaled transients. Now reserved explicitly and derived from the committed
activation dtype rather than a guessed constant. **This is what unlocked the full
202,752 context**; before it, the context was advertised but not deliverable.

**Also fixed along the way:** per-request seed determinism on the device-sampling decode
path (VS-007), a page-table width sized from an equal-share block quota rather than
`max_seq_len` (which silently capped requests at 14,720 tokens while advertising 202,752),
and a decode MoE doing dense work over all 64 experts when only top-4 are active.

---

## Known limitations

1. **Sampling is `smoke-gated`, not full-gated.** The smoke profile passes at spec
   (3 passed, 1 skipped, 0 failed). The full profile does not. Accepted explicitly by the
   project owner; the `$vllm-integration` skill permits this status with that acceptance.
   Both logs are kept side by side.
   **The full-profile failure set is not stable.** Measured twice on the same model and
   chip, separated only by an unrelated decode-path optimisation: **11 failed / 62 passed**
   before, **8 failed / 65 passed** after. Six seeded-reproducibility tests started
   passing and three greedy/top-k tests started failing. The committed
   `sampling_tests_full_RECORD.log` holds the later (8-failure) run. Quote a count only
   with the run it came from.

2. **The full-profile failures are one upstream serving-state defect,
   [tenstorrent/tt-metal#55408](https://github.com/tenstorrent/tt-metal/issues/55408).**
   Greedy requests lose determinism in a mixed batch after a long-lived server serves a
   long host-sampled request. Every failing test passes alone against a freshly started
   server. It predates this bring-up's fixes and affects the `tt_transformers` reference
   **worse** (SmolLM2 fails the same canary at baseline, where this model passes). Nine
   hypotheses were eliminated on hardware, including #48222 (`ttnn.sampling` matched
   `torch.argmax` on 256/256 rows here) and #50512 (multi-device TP only).
   **Correction posted to that filing**
   ([comment](https://github.com/tenstorrent/tt-metal/issues/55408#issuecomment-5549545445)):
   its "greedy determinism" framing is narrower than the evidence. The failing set moves
   between runs without the sampler changing, so that framing fits the later run and not
   the earlier one, and the opposite (seed-centric) framing fits the earlier and not the
   later. The stable invariant is "a long-lived server loses per-row determinism in mixed
   batches"; the reproducer is the isolation-vs-sequence contrast plus the two canary
   triggers, not any specific test name.

3. **Release accuracy is CI-subset only** (`--limit-samples-mode ci-nightly`, 5% of each
   dataset). Not comparable to a full-set threshold without that qualification. Status is
   `release-readiness-ci-subset-pass`, exit code 0, 0 blockers.

4. **Benchmark ladder verified to 131,072**, not 202,752. Single prompts were served at
   202,751, but no benchmark *sweep* ran at the top rung.

5. **Not exercised:** prefix caching (off), KV-cache migration, multi-host or multi-rank
   serving (single chip by design).

6. **The stage-8 review budget was closed by the owner** after five rounds; residual
   findings are recorded in `doc/optimized_vllm/README.md` rather than fixed. Stage 9 was
   likewise capped after its ifeval investigation concluded.

---

## Things that looked like problems and were not

Recorded because each cost investigation time and the conclusions matter:

- **"Empty answers" in the release eval.** Not empty. `content_is_empty: false`,
  `finish_reason: "stop"`, and **identical output at concurrency 2 and 16**. The model is
  a reasoning model emitting long chains (7440 completion tokens on one ifeval prompt),
  which also explains eval items slowing from 192 s to 509 s.
- **A 4-hour silent log.** The first release run had crashed (TR-001) and its log was
  preserved as evidence; the live run was writing to a different file.
- **`ttnn.sampling` greedy correctness (#48222).** Does not reproduce single-chip:
  256/256 rows matched argmax at this model's shapes.
- **A suspected `force_argmax` trace thrash.** Cannot occur; the gate is never enabled
  for this model, verified on device.

---

## Evidence

Per-stage reports and work logs under `models/autoports/zai_org_glm_4_7_flash/doc/`:
`functional_decoder/`, `fused_decoder/`, `optimized_decoder/`, `full_model/`,
`optimized_full_model/`, `datatype_sweep/`, `vllm_integration/` (VS-001..VS-011),
`optimized_vllm/`, `tti_release/` (RUN_NOTES, AUTOFIX_prefill_dram, evals, benchmarks).

61 commits on `ttmodelmanager/glm47-flash-probe`, pushed to `tenstorrent/tt-metal`
(no PR opened). Plugin registration is one commit (`9f2ec5d`) on
`ttmodelmanager/glm47-flash-registration`, pushed to the fork
`stisiTT/vllm-tt-plugin` rather than upstream, because this account has read-only
access to `tenstorrent/vllm-tt-plugin`. See the prerequisite note at the top.

Runner-side gates, both exit 0: `check_degenerate_output --scope all` (no degenerate
output across all completions) and `check_context_contract --stage vllm`
(target = supported = 202752).
