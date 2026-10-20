# Prefix caching: five-task SWE candidate

This experimental release candidate retains the traced-suffix implementation
tested at Metal `6b303feaa5f46aa8f4ef84db816104a04e840306`. It does not include
the newer optional persistent-trace optimization. Production PR branches and
their draft status are unchanged. All executable validation was remote CI;
local work was limited to source/log inspection, edits and git operations.

## Standard Harbor result

[QB2 run 39255759103](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/39255759103)
completed on `qb2-120-p01t03`, with inference
`29f1648bff455a330bafdcaedcf7fd37178cb54d`, request seed 9472, prefix caching
and traced suffix enabled, B1, context 262144, and the exact model directory
mounted over the reused image. Precision identity was `bfp8_release_selected`;
this includes the documented fixed BFP4 prefill expert-down policy and is not
an assertion that every tensor uses BFP8.

| Task | Reward | Agent time (s) | Harbor input / output tokens | API calls |
|---|---:|---:|---:|---:|
| Astropy 14096 | 1 | 1300.002 | 989532 / 32023 | 59 |
| Django 11299 | 1 | 2072.156 | 3451834 / 37017 | 103 |
| Matplotlib 25332 | 0 | 1647.678 | 2711120 / 35470 | 112 |
| scikit-learn 14629 | 1 | 982.565 | 922094 / 18240 | 60 |
| Sympy 13551 | 0 | 671.313 | 569907 / 15840 | 31 |

Score: **3/5**, exceeding the requested 1/5 functional gate. All five agents
exited `Submitted`, all task exception fields were null, and Harbor recorded
zero task errors or retries. Four tool-format errors were recovered (one each
in Astropy, Django, Matplotlib and scikit-learn). This is a small fixed cohort,
not unrestricted SWE-bench accuracy.

Both failed tasks produced patches that applied successfully. Matplotlib
failed the target `test_complete[png]`; Sympy passed the issue test but
regressed `test_rational_products` and `test_special_products`. They are not
dependency or infrastructure failures.

Client wall time was 6861 seconds (114m21s), including client setup and
verification but excluding server startup. Server counters were:

- 365 requests: 364 normal stops, one repetition stop, zero server errors;
- 8749625 input tokens, 8574400 prefix-cache hits, 175225 computed tokens;
- 150069 generated tokens, 3665.220s summed TTFT, 2960.664s summed decode time;
- approximately 50.7 generated tokens/s over recorded decode time.

Harbor reports cached usage as zero despite server cache hits. Its aggregate
input/output totals (8644487 / 138590) differ from server totals, so they are
kept separate rather than silently reconciled. Cache savings are token
accounting, not a same-trajectory end-to-end speedup measurement.

## Dependency and source provenance

The clean inference candidate changes only this model's blocks in
`workflows/model_specs/dev/llm.yaml` and `reference_config/evals/eval_config.py`.
The profile explicitly enables caching and `GEMMA4_PREFIX_TRACE_SUFFIX=1`.
The SWE config uses supported `agent_env.UV_OVERRIDE` pointing to the immutable
Metal dependency file at `6b303feaa5f46aa8f4ef84db816104a04e840306`. No runtime
Harbor source replacement, shared Harbor fork change, or generic venv patch
is included. The fresh-task preflight was run 39255415691: mini-swe-agent
2.2.8, LiteLLM 1.106.0 with its MCP extra, and MCP 2.2.0 imported successfully.
Harbor source was `1da0bfd8c71cadbff17413fac984b8e391d2afc2`.

The reused image was
`ghcr.io/tenstorrent/tt-shield/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.23.0-78e87bd9380740a522ec4b5ab62a46c9498d7b35-0a76258-112700448945`.
Its baked model differs from this candidate. The QB2 source mount establishes
the result above, but passing that old image to Shield does not mount the
candidate model. Final qualification therefore requires one image build from
the exact candidate Metal/inference refs and plugin
`0a76258e14ca7237067ca9a22ec02e257bee570a`, then reuse of that image.

## Remaining qualification and limitations

Final Shield `on-dispatch.yml@main` release is pending. Keep request seed
9472: the earlier one-task unseeded evidence does not justify removing it.
Required final cohort is GPQA10 (8/10), Terminal5 (1/5), SWE5 (1/5), at the
unchanged supported context. No full-set reference parity is claimed.

Earlier focused tests cover same-cache full-vocabulary ordinary-decode
parity, partial pages, the sliding-window boundary, divergent/repeated
requests and cache lifetime. Cold full-prefill and cached continuation are
not numerically identical under their differing precision paths. Tracked
proper-chat control 39258199619 passed, but request-time trace recapture
remains; this is not generator-wide trace-lifetime compliance.

The optional persistent-trace branch measured about 0.59s less cached-request
overhead with identical outputs on a short workload. It remains separate:
its additional boundary rerun was blocked by routing to `qb2-120-p02t02`.
That worker's missing second Ethernet link between chips 2–3 persisted in
bounded no-model health probe 39266079252. No further retry on that worker is
planned without repair. Final Shield routing must avoid that known fault.

All iterative jobs must be visible in `tt-agentic-bringup-qb2/actions`; the
user explicitly permits its experiment workflow branch. Final release is
the previously requested Shield exception. No hidden remote execution,
local validation, shared workflow mutation, or production PR update is used.

## Final Shield dispatch — 2026-10-20

The user explicitly authorized final dispatch despite the unresolved runner
label limitation. Parent dispatched
[39384889094](https://github.com/tenstorrent/tt-shield/actions/runs/39384889094)
through `on-dispatch.yml@main`, runner `BH-Quietbox-2`, device `p300x2`,
workflow `release`, CI scope 10/5/5. Exact inputs were Metal
`84dd77aacd5c992700d2643add54e3f3e6c5a82f`, inference
`29f1648bff455a330bafdcaedcf7fd37178cb54d`, and plugin
`0a76258e14ca7237067ca9a22ec02e257bee570a`. The empty Docker-image input
requests the one necessary exact-source build. No duplicate dispatch or
build was started. This document may advance independently; the dispatched
model ref remains immutable.

Initial validation and SHA-resolution jobs passed. The inference-server
image build started; image tag/digest and final model results are pending.
If the known faulty worker is selected and fails fabric initialization,
preserve that infrastructure evidence and do not blindly repeat the run.

Build job `118502411741` succeeded from 07:11:47 to 08:13:39 UTC (61m52s).
The build log confirms checkout of the exact Metal commit and successful
image publication. Reuse this image rather than rebuilding:

`ghcr.io/tenstorrent/tt-shield/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.25.0-84dd77aacd5c992700d2643add54e3f3e6c5a82f-0a76258-118502411741`

Build-export identities:

- Manifest list: `sha256:2f65a018484c4dc527cdd37e47919b31ce6064e0f90a32766b5e8ea2018172b4`.
- Image manifest: `sha256:57bf823a4e0003cc3b5b5096e3accadccf4aefc5ba6974bf939a1ee78d092718`.
- Image config: `sha256:c5165fe88ca88421d12c2e1a5b8de189920cb3e83de9816fcabc3e07da78e39e`.

Registry metadata lookup with the existing read credentials returned 403
(`read:packages` unavailable); no broader credentials were requested. Runtime
pull confirmation and precision identity remain pending. Release test job
`118525081608` queued after the successful build.

The release job subsequently acquired `qb2-120-p05t09` (not the known faulty
`qb2-120-p02t02`). Runner setup, inference checkout, and image determination
passed; the `Run tests` step was active at 08:52 UTC. The job-log API still
returned `BlobNotFound`, and no evaluation artifact had been published.
Therefore runtime image/precision confirmation, model readiness, and suite
scores remain unverified at this checkpoint.
