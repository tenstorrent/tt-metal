# TTI release stage report: zai-org/GLM-4.7-Flash autoport on one Blackhole p150

Stage 11 (`$tti-release`) for the generated tt-metal autoport
`models/autoports/zai_org_glm_4_7_flash`, run 2026-09-04 on host `tt-quietbox`.
This file is the short entry point; the detail lives in the committed evidence
listed at the bottom, above all in `RUN_NOTES.md` (per-row classification,
commands, versions) and `work_log.md` (order of work and wrong turns).

## Headline

**`release-readiness-ci-subset-pass`.** `run.py --workflow release` exited 0,
acceptance is PASS with 0 blockers, and every required sampled row either passed
or carries a row-specific classification in `RUN_NOTES.md`.

| | |
|---|---|
| Release report | `reports_output/release/report_id_autoport-glm47-flash_GLM-4.7-Flash_p150_2026-09-04_20-48-56.md` |
| Server mode | external autoport vLLM server, **no Docker**, `service_port` 8000 |
| Evaluated implementation | `models/autoports/zai_org_glm_4_7_flash` (impl `autoport-glm47-flash`), proven in `run_specs/` and the report metadata; no stock `tt_transformers` / `models/demos` anywhere |
| Device | one Blackhole chip, 1x1 mesh, `MESH_DEVICE=N150`, TTI device `p150` |
| Context | full 202752, unchanged from `doc/context_contract.json`, `capability_reduction: none` |
| Benchmarks | PASS, 1 of 23 rows graded (the catalog grades one shape per device), 22 reported ungraded |
| Spec tests | PASS, 22 of 22 vLLM chat/completions parameter-conformance cases |
| Single-user serving | 128 in / 128 out, concurrency 1: TTFT 296.1 ms, TPOT 29.5 ms/token, 31.7 output tok/s |

## Accuracy is a CI subset, not full-set

The release ran with `--limit-samples-mode ci-nightly`, which propagated as
`--limit 0.05`. **Every accuracy number below is a 5% sample and must not be
compared with a full-set release threshold.** Unrestricted evals were projected
at about 16.7 h on this single chip (about 1.6 h for ifeval, about 15.1 h for
GPQA), which is why the CI subset was used; the projection and the effective
sample counts are recorded in `RUN_NOTES.md`.

| task | sampled | score | automated check |
|---|---|---|---|
| `ifeval` | 28 of 541 | prompt-level strict **0.714**, inst-level strict **0.744** | NA, no published or in-house reference exists |
| `gpqa_diamond_cot_zeroshot` | 10 of 198 | **70.0** (7 of 10, flexible-extract) | FAIL against the model card's full-set 75.2 |

`meta_ifeval` and `meta_gpqa_cot` are structurally unavailable for this
checkpoint: `EVALS_META` builds its datasets from `f"{hf_model_repo}-evals"`,
which Meta publishes only for the Llama families. `ifeval` and
`gpqa_diamond_cot_zeroshot` are the standard equivalents every non-Llama model in
the catalog uses. The GPQA row's classification, including why an n=10 subset
cannot be judged against a full-set ratio gate, is in `RUN_NOTES.md`
"Row-by-row classification".

## TR-001: prefill activation OOM, found and fixed here

The first release attempt died at benchmark sweep point 21 (`isl=65536`) with
`TT_FATAL ... Out of Memory ... 402653184 B DRAM buffer` inside `_moe_prefill`,
killing the vLLM EngineCore and taking the rest of the sweep and the whole
`spec_tests` child with it. Before the fix the advertised 202752-token context
was **not deliverable through the serving path** even though the model harness
had proven a 202751-token prefill: `get_max_tokens_all_users` handed every
remaining DRAM byte to the KV pool behind a fixed 0.75 GiB margin that reserved
nothing for the pair of whole-prompt `[1, 1, phys, 2048]` bf16 activations that
`run_layer_stack_prefill` keeps live at each of the 47 layer boundaries
(8192 B per prompt token).

Fixed with `$autofix` by deriving that reservation from `max_model_len`, the
checkpoint's own `hidden_size` and the contract's paged block size. Committed as
`c49c193f854`. Verified on hardware in one server process at prompt lengths
**10000, 65536, 131071, 131072 and 202751**, all served with `usage.prompt_tokens`
echoed exactly. KV pool 471168 -> 414656 tokens, still 2.05x the served context;
decode TPOT unchanged at 29.49 ms/token against the recorded 29.496.
Write-up in `AUTOFIX_prefill_dram.md`, machine-readable in
`autofix_prefill_dram.json`, before side in `logs/prefill_oom_before_fix.log`.

## Known limitations

1. **Accuracy is CI-subset only** (5% of each dataset). Not unrestricted
   full-set release readiness.
2. **The benchmark ladder verified serving up to ISL 131072**, not 202752. The
   full 202751-token context is proven by single-prompt probes
   (`autofix_prefill_dram.json`) and by needle-in-a-haystack retrieval, not by a
   benchmark sweep point, because the sweep's own ISL ladder stops at 131072.
3. **No in-house GPU or canonical-implementation control** exists for either
   eval, so `ifeval` has no automated verdict at all and GPQA has no
   apples-to-apples subset reference.
4. **Upstream determinism defect
   [tenstorrent/tt-metal#55408](https://github.com/tenstorrent/tt-metal/issues/55408)
   is unchanged.** Two ifeval items return an empty answer in the eval harness
   and complete normally when replayed; see the anomaly ledger in `RUN_NOTES.md`.
5. **Greedy decoding can fail to terminate** inside the reasoning block on
   open-ended creative prompts. Every graded release surface uses sampled or
   explicitly parameterised decoding, which is the model card's own operating
   point. Characterised in `qualitative/greedy_trace_probe.json`.
6. **The benchmark `ttft_ms` target compares a chat-endpoint measurement against
   a completions-endpoint one** and should be re-derived.
7. **Prefill above about 32K tokens costs 1.547 GiB of permanently reserved
   DRAM** that the KV pool no longer gets. Buying it back means changing the
   prefill compute path, which was out of scope for an OOM fix.
8. **ISL 131072 has a 1034 s TTFT.** Prefill cost grows faster than linearly
   (65536 -> 319.6 s), so release harnesses at these lengths need generous
   timeouts.

## Evidence in this directory

* `RUN_NOTES.md`: the handoff record. Server mode, host, versions and SHAs,
  exact commands, key environment, recovery actions, the smoke, the context
  contract, the ci-nightly justification, per-row classification, the
  `$qualitative-check` prompt-format decision, and the anomaly ledger.
* `work_log.md`: TR-000 to TR-006, the order of work and the wrong turns.
* `reports_output/release/`: the customer-facing release markdown and its data
  JSON.
* `run_specs/`: the TTI runtime model specs for the release and the smoke, which
  carry `impl.code_path = models/autoports/zai_org_glm_4_7_flash` and the
  `cli_args` (`docker_server=false`, `local_server=false`, `service_port=8000`).
* `benchmarks/`: the 23 release sweep points, ISL 128 to 131072, plus the smoke
  point. Per-request raw arrays were dropped on copy; each file records that.
* `evals/`: lm-eval aggregate results for both tasks plus the empty-answer probes.
* `qualitative/`: outputs, prompt-format metadata, degenerate-output check, and
  the greedy-trace characterisation.
* `logs/`: the successful release run log and the pre-fix OOM traceback.
* `non_aligned_prompts.json`: prompt lengths that are not multiples of the 1024
  serving prefill chunk, served with their exact lengths echoed.
* `AUTOFIX_prefill_dram.md`, `autofix_prefill_dram.json`: the TR-001 fix.

Whole-bring-up context, all eleven stages:
`../GLM-4.7-Flash_p150_bringup_report_2026-09-05.md`.
