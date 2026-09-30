/bringup

Bring up chunked prefill of Tencent Hy4 Preview (`tencent/Hy4-preview`, revision
705d81ee51566a186d645b74c974d642ef2828fe) with the gated bring-up framework in `models/demos/common/bringup`.
Repo /localdev/dnijemcevic/tt-metal2, branch dnijemcevic/ernie45_prefill (compare against origin/llk_helper_library,
never main). Box: 4x Blackhole p150b QuietBox, always FABRIC_2D.

Read first:
- The memory notes: bringup-framework-state, mimo-v2-6-bringup-plan, commit-regularly, no-approval-formalities,
  tt-metal2-env-pitfalls, branch-base-llk-helper-library.
- `hy4_preview_findings.md` in the repo root: the feasibility study (architecture, memory, component table with repo
  paths, risks, what was not verified). Treat it as a starting point, not as verified fact; check every claim you rely
  on.
- The framework overview `models/demos/common/bringup/docs/pipeline_design.html` and README.

Before any device work, check that no other job is using the device (another session may have an agent running
rms_norm tests: `ps -ef | grep -E "run_safe_pytest|tt-probe"`). One device job at a time.

## What the study found (summary; details in the .md)

- Architecture: 78 layers, hidden 6144, vocab 120832. Every layer is gated DSA sparse attention on MLA (64 heads,
  q_lora 2048, kv_lora 512, qk 192 NoPE + 64 RoPE, v 256, scale 1/16) with a learnable sink per head and a sigmoid
  output gate. An indexer (32 heads x 128, top-2048) runs on 21 "full" layers; the other 57 reuse the latest full
  layer's selection. Layer 0 dense FFN (18432); layers 1-77 MoE: 256 experts, top-8, 1 shared expert, sigmoid +
  correction bias, clamped SwiGLU at 10. A 4-stream residual (iHC: DeepSeek-V4's mHC without the Sinkhorn). BF16
  checkpoint, 780B total.
- Closest repo code: GLM-5.2 in `models/demos/deepseek_v3_d_p` (same MLA/indexer/expert geometry and indexer
  sharing), plus Kimi-K3's output gate, DeepSeek-V4's sinks-in-SDPA and mHC, and the bring-up forks
  (`ttnn/ttnn/bringup/INDEX.md`): dispatch/combine/offset_cumsum, unified_routed_expert_ffn with ClampedSiluGlu,
  rms_norm_ttnn. MoE on 2x2 is proven by `models/demos/mimo_v2_6_d_p_2x2`. No component needs a new op.
- Fit: the full model does not fit (about 104 GiB per chip even with bfp4 experts). Layers 0-5 cover all three block
  types: `dense_full` (layer 0, dense FFN, own indexer), `moe_full` (layers 1 and 5), `moe_shared` (layers 2-4). My
  estimate for layers 0-5 on 2x2 with bfp8 experts: about 18-23 GiB per chip of the 27.2 GiB budget (experts 12 GiB,
  the rest weights, KV under 0.6 GiB thanks to the 576-wide MLA latent, activations 3.5-5 GiB driven by the 4-stream
  residual and the indexer scores). The plan gate will compute the real number.
- Main risks: integrating deepseek_v3_d_p's Galaxy-tuned ttMLA/TtIndexer into the framework's component and swap
  steps; indexer top-k precision (the tests need a selection-overlap metric); sinks with sparse attention; iHC fp32
  needs; CPU goldens at 56k (HF runs eager attention only, about 800 GB of scores, so the reference needs a chunked
  sparse CPU path; the repo has `reference/cpu_deepseek_v32` and `tests/sparse_mla/sparse_mla_reference.py`).
- The modeling code is in transformers 5.17+ (`models/hy_v4/`); python_env has 5.12.1. Vendor
  `modeling_hy_v4.py` for the reference rather than upgrading the shared environment.
- Weights: layers 0-5 span 35 of the 131 shards (about 415 GB whole, about 105 GB fetching only the needed tensors).

## Intake: propose these defaults and review them with me

- Model slug `hy4_preview_d_p` (a new bring-up, no `prior`).
- Mesh 2x2, FABRIC_2D (the configuration the sparse-MLA tests and the MiMo 2x2 MoE support). Attention layout is
  the planner's call (the sparse-MLA tests use SP2xTP2).
- Layers 0-5. Target 56320 tokens in 5120-token chunks, the default ladder.
- Experts bfp8 (agent rule 7: HiFi4 everywhere during functional bring-up; bfp4 only where the checkpoint is 4-bit,
  and this one is BF16).
- Dashboards: both styles (standard + teletext). Republish both after every gate; give me both links at the start
  and at the end, and keep the per-op profile tabs.
- Owner rules for the spec's `agents.rules` (confirm the wording with me): always 2D fabric; text decoder only, skip
  MTP; the reference is a chunked sparse CPU implementation validated against HF at short lengths; point the agents
  at `hy4_preview_findings.md` and the deepseek_v3_d_p modules it lists; be aware of the ttnn.bringup ops (INDEX.md)
  and use them as they see fit.
- `agents.read`: the vendored modeling_hy_v4.py and config for the reference role; deepseek_v3_d_p (GLM-5.2 config,
  tt/mla, tt/mhc, tests/sparse_mla), mimo_v2_6_d_p_2x2/tt and the findings file for plan and implement.

Show me the spec and wait for an explicit yes, then approve the intake and launch as the skill says.

## New framework feature under test: deferral to op-gen (F46)

This is the first bring-up with the deferral feature. If an implement agent cannot find a proper TTNN op for a
component, it may defer the step: it writes an op request (evidence of what it searched and tried, the math, the
real per-chip shapes, a torch reference) under `<bringup_dir>/op_requests/<op>/`, the task becomes DEFERRED, the step
runs on the CPU through the bridge, and the rest of the bring-up continues. I review requests whenever I get to them;
launching op-gen is always my call. The skill's "Deferred steps and op-gen" section maps my plain-language answers
to the commands. As overseer:
- Review every deferral like a gate commit. For Hy4 almost every component exists in the repo (see the findings), so
  a deferral is unexpected: reject it as cheating unless the evidence shows no existing op, composition or fork fits.
- Tell me in one line when a step is deferred, and list pending requests at every status report and at the end.
- Report how the feature behaved (anything confusing, missing or broken), since this run is also its first test.

## Watch and report (as in the MiMo runs)

1. No edits to existing TTNN ops (agent rule 6). Changes go into ttnn/ttnn/bringup forks (reuse or extend first,
   INDEX.md), following skill bringup-fork-op (source tests + baseline for new forks; the extension recipe for
   existing ones, with the full regression once, and no regressions for the MiMo, Gemma and ERNIE cases).
2. O.1 at the end: every ttnn.bringup call gets a random-input case.
3. When a perf pick or an op switch changes the model, rerun the frozen component and swap tests of the affected
   layers, not only the ladder.
4. Precision of the indexer selection, sinks and iHC: watch the component metrics, not only whole-output PCC.

## Operational rules

- One device job at a time; never run device-using agents in parallel.
- Never wrap orchestrator runs in `timeout`; run them in the background with a monitor.
- Commit each verified piece right away (explicit paths). Don't push unless I ask.
- Never run tt-smi -r unless I say so. Run run_safe_pytest.sh and tt-probe.sh in the foreground.
- Decide routine things yourself. Ask me for the spec, the plan, the perf picks, op-gen requests, board resets and
  pushes.

At the end, report per-layer PCC, the 50k->55k chunk device time and 0->55k TTFT, which bringup ops and forks were
reused, extended or created, and how the deferral feature behaved.
