# GLM-5.3-Flash bring-up: setup and prompt for the second machine

## Setup on the new machine (by hand, before starting Claude)

```bash
git clone https://github.com/tenstorrent/tt-metal.git tt-metal2 && cd tt-metal2   # or reuse a checkout
git fetch origin dnijemcevic/ernie45_prefill
# own branch, from the commit that added this file; reconciled with the Hy4 branch later
git checkout -b dnijemcevic/glm53_prefill $(git log -1 --format=%H origin/dnijemcevic/ernie45_prefill -- glm53_bringup_prompt.md)
git submodule update --init --recursive                    # includes tt_metal/third_party/tt_ops_code_gen (op-gen)
./build_metal.sh && ./create_venv.sh                        # the ttnn.bringup forks are C++
mkdir -p ~/.claude/skills
for s in bringup bringup-fork-op bringup-fork-tests; do
  ln -s "$PWD/models/demos/common/bringup/skill/$s" ~/.claude/skills/$s
done
huggingface-cli login                                       # or export HF_TOKEN
```

Optional: copy the memory notes from the first machine (`~/.claude/projects/-localdev-dnijemcevic-tt-metal2/memory/`)
into this machine's matching project folder (the folder name follows the repo path).

Then run Claude in tmux (`tmux new -s glm`, `claude --dangerously-skip-permissions`) and say:
"Read glm53_bringup_prompt.md in the repo root and follow the prompt section."

## Prompt

/bringup

Bring up chunked prefill of GLM-5.3-Flash with the gated bring-up framework in `models/demos/common/bringup`. This
checkout is on branch dnijemcevic/glm53_prefill, made from dnijemcevic/ernie45_prefill at the commit that added this
file. Another
machine is bringing up Tencent Hy4 on dnijemcevic/ernie45_prefill at the same time; the two branches are reconciled
later. Commit locally on this branch and never push.

Read first:
- `glm53_flash_findings.md` in the repo root: the feasibility study (architecture, memory, component table with repo
  paths, risks, what was not verified). Treat it as a starting point and check every claim you rely on.
- The framework overview `models/demos/common/bringup/docs/pipeline_design.html`, its README and dev/BREADCRUMBS.md.
- The memory notes, if they were copied to this machine.

### Check the box first

The study assumed 4x Blackhole p150b (32 GB DRAM each, about 27 GiB usable per chip) with FABRIC_2D. This machine may
differ: check `ls /dev/tenstorrent | wc -l` and `tt-smi -ls` (never `tt-smi -r`), and redo the memory math for what
is really there before proposing a layer subset. Always use 2D fabric.

### What the study found (summary; details in the .md)

- Checkpoint `zai-org/GLM-5.3-Flash`, revision eb9eb208eb0d988989d07a6a12d0fdeb5f52574a (FP8 e4m3 with 128x128 block
  scales, 328 GB, not gated). A BF16 copy exists (`zai-org/GLM-5.3-Flash-BF16`, 643 GB). The modeling code is in
  transformers 5.17+ (`glm5_next`, `Glm5NextForConditionalGeneration`); python_env has 5.12.1, so vendor the modeling
  file for the reference rather than upgrading the shared environment.
- Architecture: 45 text layers (+ MTP and a vision tower, both skipped), hidden 4096, vocab 154880, no RoPE. 34 KDA
  linear-attention layers; 11 DSA sparse-MLA layers (every fourth, from 3) with an indexer that pools keys by 4 with a
  learned softmax and keeps the top 512 pools. mHC 4-stream residual with a 20-iteration Sinkhorn. Layers 0-2 dense;
  3-44 MoE with 288 experts, top-8, 1 shared expert, sigmoid + correction bias, clamped SwiGLU at 10.
- Closest repo code is in `models/demos/deepseek_v3_d_p` (Kimi-K3 KDA + NoPE MLA, GLM-5.2 DSA, DeepSeek-V4 mHC), plus
  the MiMo router and the ttnn.bringup forks (dispatch/combine/offset_cumsum, unified_routed_expert_ffn with
  ClampedSiluGlu, rms_norm_ttnn; `ttnn/ttnn/bringup/INDEX.md`).
- Fit (on 4x p150b): the full model does not fit (about 42 GiB per chip even with bfp4 experts). The study proposed
  layers 0-7 (3 dense + 5 MoE, 2 of them DSA; about 9 GiB of experts per chip at bfp8); the minimum covering every
  block type is 0-3. Block types: `kda_dense` (0-2), `dsa_moe` (3, 7, ...), `kda_moe` (the others from 4).
- Missing: the indexer key pooling (learned softmax over groups of 4 keys, a pooled-key cache across chunks, and a
  pool-level causal mask). Needs adapting: KDA on a 4-chip mesh (tested only on 2x4 / 1x8 / 8x4), mHC in a model
  (never done), MLA with a 512-wide latent (the cache assumes 576), sparse SDPA index compaction, the indexer without
  RoPE, mixed per-layer state (MLA latent + indexer keys + KDA recurrent state; the framework has only run GQA k/v
  state, so goldens, state metrics and the K.1 serving contract need work), a sparse CPU reference at 56k (HF's dense
  indexer and eager attention need tens of GB of scores), and per-layer loading.
- Effort: 85-110 tasks, about 1.5-2x the MiMo bring-up.

### Intake: propose these and review them with me

- Model slug `glm53_flash_d_p` (a new bring-up, no `prior`).
- Mesh and layer subset from the box check and the memory math. Target 56320 tokens in 5120-token chunks, the default
  ladder.
- Weights: dequantize the FP8 checkpoint exactly as the checkpoint defines it for the CPU reference. On device,
  experts bfp8 (agent rule 7: HiFi4 everywhere during functional bring-up; bfp4 only where the checkpoint is 4-bit,
  and this one is FP8).
- Dashboards: both styles (standard + teletext); republish both after every gate and give me both links at the start
  and at the end.
- Owner rules for `agents.rules` (confirm the wording with me): always 2D fabric; text decoder only (skip MTP and
  vision); the reference is a chunked sparse CPU implementation validated against HF at short lengths, with
  per-layer loading; point agents at `glm53_flash_findings.md` and the deepseek_v3_d_p modules it lists; be aware of
  the ttnn.bringup ops (INDEX.md) and use them as they see fit.
- Check whether the framework needs changes for mixed per-layer state before the plan (spec `state.kind` is `kv`
  today). If it does, treat it as a framework fix: gate it with selftests, log it in dev/BREADCRUMBS.md, and tell me.

Show me the spec and wait for an explicit yes, then approve the intake and launch as the skill says.

### Deferral to op-gen (F46) is likely here

The indexer key pooling may have no proper TTNN implementation. If an implement agent cannot build it from existing
ops or a fork, it may defer the step: it writes an op request under `<bringup_dir>/op_requests/<op>/`, the task
becomes DEFERRED, the step runs on the CPU through the bridge, and the bring-up continues. The skill's "Deferred steps
and op-gen" section maps my plain-language answers to the commands. As overseer:
- Review each deferral like a gate commit: accept it only if the evidence shows no existing op, composition or fork
  fits; otherwise reject it as cheating.
- Tell me in one line when a step is deferred, and list pending requests at every status report and at the end.
- Launching op-gen is always my call (it needs a push of the submodule and branch and hours of device time).
- Report how the feature behaved, since this is one of its first uses.

### Shared bring-up ops

The Hy4 run on the other machine uses the same ttnn.bringup forks. When this run extends a fork (for example the
unified expert for 72 local / 288 total experts), follow the bringup-fork-op recipe strictly: the option default off,
the default program unchanged, and the full regression once (unit suite, fork_source check, every model's cases). If
some model cases need a mesh this machine does not have, report them as not run for that reason, not as passing.
Record every fork change in its CHANGELOG so the two branches can be reconciled.

### Watch and report

1. No edits to existing TTNN ops (agent rule 6); changes go into ttnn/ttnn/bringup forks.
2. O.1 at the end: every ttnn.bringup call gets a random-input case.
3. When a perf pick or op switch changes the model, rerun the frozen component and swap tests of the affected layers.
4. Precision of the indexer selection (use a selection-overlap metric), the KDA recurrence across chunks, and mHC:
   watch component metrics, not only whole-output PCC.
5. The checkpoint trim (task R.4, F47) deletes the layers no later step reads after the HF sanity passes; make sure
   the sanity ran on the whole model first.

### Operational rules

- One device job at a time; never run device-using agents in parallel.
- Never wrap orchestrator runs in `timeout`; run them in the background with a monitor.
- Commit each verified piece right away (explicit paths). Never push.
- Never run tt-smi -r unless I say so. Run run_safe_pytest.sh and tt-probe.sh in the foreground.
- Decide routine things yourself. Ask me for the spec, the plan, the perf picks, op-gen requests and board resets.

At the end, report per-layer PCC, the 50k->55k chunk device time and 0->55k TTFT, which bring-up ops and forks were
reused, extended or created (with their CHANGELOG entries, for the reconciliation), any framework changes, and how the
deferral feature behaved.
