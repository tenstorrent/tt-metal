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

Use this machine for local TT work; it is the only device for this run. Serialize all local
device commands and check for device owners before reset. No CI, no GitHub, no pushes: commit
locally on branch v4-agentic-rehearsal.

Pipeline worker: gpt-6-astra, medium effort, ChatGPT login in /home/mvasiljevic/v4-agentic/codex-home.
Do not substitute other credentials or ambient OpenAI/Codex API keys. Use the enabled workflow skills.
Stage templates authorize subagents where the selected skill requests them.

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
## Operator decisions (v4 rehearsal, 2026-11-12)

5. Mode agentic, local only. The pipeline container now has Docker (host socket) for SWE-bench task
   containers, and a Hugging Face credential with access to the gated GPQA dataset. Earlier decisions that
   routed GPQA or task-container work to CI no longer apply.
6. The v3 stage 11 and 12 outputs were removed from this branch (they are in its first commit and in
   /home/mvasiljevic/v4-agentic/v3-leftovers). Do not reuse them as results.
7. Rehearsal limits are in bringup/agentic_evals/inputs.json: stage budget 10 h, phase 2 on 4 SWE tasks.
