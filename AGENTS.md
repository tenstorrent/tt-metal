# Gemma 4 31B fresh bringup

This checkout is dedicated to the twelve-stage model-bringup run. Read
`bringup/run_contract.json` and `bringup/agentic_evals/inputs.json` at startup.

Implement text-only google/gemma-4-31B-it at HF revision
842da3794eaa0b77d5f08bae87a17459d91ff475 in
models/autoports/google_gemma_4_31b_it. Pin reference and tokenizer loads to that
revision. Preserve the advertised context, with only evidence-backed physical
hardware limitations allowed by the stage contract.

The operator requested a new implementation. Existing Gemma 4 implementations
were removed from models/demos/gemma4, gemma4_31b_qb2, and gemma4_d_p.
Do not restore or copy their source from Git history, other local experiments,
remote sources, or installed packages. Use the Hugging Face reference and generic
TTNN operations/utilities. Do not copy Gemma-specific implementation branches
from the generic tt_transformers framework into the autoport. Validate that serving
registers and imports this run's autoport adapter, rather than a stock adapter.

Use this machine for local TT work; it is the primary device. Serialize all local
device commands and check for device owners before reset. Up to six QB2 CI machines
are granted as a secondary device (`qb2-ci` in `bringup/agentic_evals/inputs.json`):
dispatch, track and collect those runs yourself, following its instructions file, and
balance work between the local device and CI as the agentic-serving skill says. Use
default batches and evals from the selected skills, with the goal-12 inputs in
`bringup/agentic_evals/inputs.json`. Read current release eval names, versions,
timeouts and output limits from the pinned release spec and its harness sources; the
model spec alone may not contain the eval limits. Do not reuse the old
implementation's layout/precision/serving settings as evidence.

The operator authorizes pushing the dedicated branches `mvasiljevic/gemma4-31b-goal12-v3`
in tt-metal and tt-inference-server for CI. Publish from separate checkouts with
remotes; keep this execution checkout without a remote. Use source-only commits,
dispatch immutable SHAs, and keep weights, credentials, private logs, profiler dumps
and collected results out of publication commits. Never update main or stable. Do not
create issue comments or send messages.

Pipeline worker: gpt-6-astra, medium effort, personal ChatGPT login in
/home/mvasiljevic/.codex-gemma4-v3. Do not substitute enterprise credentials or
ambient OpenAI/Codex API keys. Use enabled workflow skills from the requested skills
branch. Stage templates authorize subagents where the selected skill requests them.

## Operator decisions (stage 9, 2026-11-05)

These answer the stage-9 blocker in
`models/autoports/google_gemma_4_31b_it/doc/vllm_integration/plugin_changes.md` and its
sampling question. They are operator messages naming the exact items.

1. PC-01 (`supports_masked_prefill_sampling` / `prefill_sample_mask`) is **not authorized**.
   Keep the pinned stock `tenstorrent/vllm-tt-plugin`. The plugin's documented rule that a
   prefill batch containing an intermediate chunk is sampled on the host is accepted as a
   shared-plugin limit, not a model defect and not a stage blocker. It counts as the
   explicit, optional host-sampling mode the stage allows: decode and all-final prefill stay
   on the canonical traced device sampler, and that device path is the one measured for
   performance. Log a per-step sampling-mode line, report how many steps were host-sampled
   under mixed chunked traffic, and record the limit in the README limitations and
   `serving_contract.json`. Do not route around it by disabling chunked prefill, reducing
   context, or constraining the scheduler.
2. Served sampling defaults: **authorized** to serve with
   `--override-generation-config '{"top_k":32}'`, keeping the checkpoint's temperature 1.0
   and top_p 0.95, so the default recipe stays within the device sampler's 32-candidate
   width. Record this as an operator decision in the work log and in every server command.

Mark `blocked.json` resolved by these decisions and continue the stage.

## Operator decisions (stage 9, 2026-11-05, 22:30 UTC)

3. Scope of "keep the pinned stock plugin" in decision 1: it covers the plugin's runtime
   (`src/`, serving defaults, capabilities, interfaces) only. PC-02 and PC-03 in
   `models/autoports/google_gemma_4_31b_it/doc/vllm_integration/` are **authorized** as
   proposed: apply the test-only patches to the local plugin checkout, run the plugin's hooks,
   host suite and the affected live checks, and make a local plugin commit. Never push it. The
   served runtime stays the stock pinned plugin, so CI and tt-inference-server run the same
   serving code. Further test-only corrections that follow the vllm-integration plugin-change
   rules need no new operator exception; runtime or shared-interface changes still do.

## Operator decisions (stage 11, 2026-11-06, 03:40 UTC)

4. PC-04 (plugin completion observer, `doc/benchmark/setup/PC-04-completion-observer.md`) is
   **not authorized**; the plugin runtime stays stock. Implement full-phase accounting in
   model-owned code instead: record lightweight host timestamps in the adapter/generator at
   submission and at the actual asynchronous completion it already observes (the token-out
   readback event it waits on), with no added device synchronization and no profiler. Define
   each phase interval as wall time between consecutive adapter-observed boundaries, so plugin
   host work after the adapter returns (host sampling, output construction, scheduling) and
   idle gaps fall inside the intervals and are counted, not separately observed. State that
   attribution in the report. Derive prefill FLOPs and decode DRAM bytes from the executed
   shapes, precision and parallelism. This counts as the stage's full-phase accounting.
5. GPQA: this host has no Hugging Face credential for the gated dataset, and none will be
   added. Record GPQA as not covered locally for that reason, keep the frozen subset
   definition, and complete stage 11 without it. Accuracy for gated suites can run in stage 12
   on the operator-defined devices that have dataset access. Do not substitute an unofficial
   mirror.
6. Rerun whatever the stage needs (fresh profiles with the collector, inside the client-stage
   budget); keep the earlier incomplete attempt as evidence.

## Operator decisions (stage 11 closure, 2026-11-06, 04:40 UTC)

7. Stage 11 is **accepted by the operator as complete** with the evidence in
   `models/autoports/google_gemma_4_31b_it/doc/benchmark/` (report, four profiles with both
   roofline estimates, GPQA/SWE-bench/Terminal-Bench recorded as not covered). The installed
   `benchmark_stage/check.py` rejects empty accuracy and caps elapsed time at 3600 s; that is a
   known defect of the skills under test. The installed plugin package stays read-only: do not
   apply the prepared checker patch. Stage 12 starts from this stage-11 state.

## Operator decisions (stage 12, 2026-11-06, 05:40 UTC)

8. Release route: this host's pipeline container gets **no Docker access**; that will not
   change. The operator **authorizes the granted `qb2-ci` device (same p300x2 device type) for
   all work that needs Docker task containers, including the dry run and the release run**.
   Keep using the local device for serving measurements that need no task containers
   (growing contexts, prefill by length, host contention, server bring-up checks). Plan the CI
   release around shared-pool queueing: dispatch the dry run early enough to fix and re-check,
   dispatch every release job well before the stage limit with queue time in the projection,
   split release work on task boundaries so each job fits the CI job limit, and keep the release
   on the same SHAs the dry run proved. GPQA, which needs gated dataset access, also runs on
   `qb2-ci`.

## Operator decisions (stage 12, 2026-11-07, 02:40 UTC)

9. CI image builds for TT `f57978407eaacf09587f5793b8ae46ca7af839ea` (EPEL HTTP 403 at the
   runners' proxy): the operator **cannot restore that build access and has no image digest for
   that commit**. This is an infrastructure failure: it is not scored and is not a model result.
   - You may retry the unchanged build through the stock workflow, at most once every 90 minutes
     and only while a slot is free; record each attempt in `compute.json` (zero device time).
   - You may use the workflow's `docker-image` input with an image an earlier successful run
     built, but only if you show that every file the logging candidate changes reaches the test
     job from the dispatched commits rather than from the image. Otherwise it is not the frozen
     source; do not use it.
   - Do not change the workflow, mirrors or dependency routes.
   - The logging candidate is optional for the release. If no build of the new commit succeeds
     in time for the dry run, run the dry run and the release on the last SHAs and image already
     proven on CI (for example the published `eef534...` route), and record the logging candidate
     as locally verified and unverified on CI. Do not hold the dry run for it.
   - Also record in `compute.json` the earlier failed benchmark-workflow runs (42239308813,
     42252352037, 42264721517) and these two build failures (42326665396, 42327188063); every run
     counts there.
