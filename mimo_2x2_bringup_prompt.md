/bringup

Bring up MiMo-V2.6-Flash-RL (layers 0-5) again, this time on a 2x2 mesh, seeded by the finished 1x4 bring-up
models/demos/mimo_v2_6_d_p. Repo /localdev/dnijemcevic/tt-metal2, branch dnijemcevic/ernie45_prefill (pushed at
e3e396a97b4; compare against origin/llk_helper_library, never main). Box: 4x Blackhole p150b, always FABRIC_2D. A 2x2
probe passed earlier: all_reduce on both axes, and the original DeepSeek dispatch->combine with dispatch groups of 2.

Read the memory notes first: bringup-framework-state, mimo-v2-6-bringup-plan, commit-regularly,
no-approval-formalities, tt-metal2-env-pitfalls, branch-base-llk-helper-library.

Setup:
- `python -m models.demos.common.bringup new --prior mimo_v2_6_d_p --mesh 2,2`. It creates mimo_v2_6_d_p_2x2, shares
  the 1x4 run's checkpoint and goldens, and its hooks reuse the prior's CPU reference.
- Review every copied spec field with me (box name, rules, reads), then approve the intake and launch as the skill
  says.
- The existing bring-ups keep their names (they are 1x4); only the new one has a suffix.

What should differ from 1x4 (the planner decides; I approve the plan):
- The MoE can dispatch across the 2 chips on a dispatch axis instead of each chip only to itself. The dispatch,
  combine and offset_cumsum forks behave like the originals when the axis has >1 chip.
- The sharding plan and the CCL axes change. Sequence parallel becomes possible (fusion F9).
- The serving contract's KV layout follows the new mesh.

Carry over from 1x4 (the prior's tt/ already uses these; check that the new model keeps using them where they fit):
- ttnn.bringup.rms_norm (the AI-generated norm, with a C++ host, fp32 honoured), and the fused residual add + norm
  (return_residual_sum);
- ttnn.bringup SDPA with V at 128 (no padding): the head split via nlp_create_q_heads_split, V caches at 128,
  o_proj K 2048;
- ttnn.bringup.unified_routed_expert_moe with high_precision at HiFi4, bfp8 expert weights, bf16 activations.
- Precision rule (agent rule 7): HiFi4 everywhere, bfp4 only where the checkpoint is 4-bit. The sliding-attention
  preset "S" (fp32 accumulation off) stays, as decided on 1x4.

As overseer, watch closely and report on the new mechanisms, and whether agents actually use them:
1. No edits to existing TTNN ops (agent rule 6). Any op change goes into ttnn/ttnn/bringup, either reusing or
   extending an existing fork (check INDEX.md first) or making a new one with fork_op.py. Reject a diff that touches
   ttnn/cpp/.../operations.
2. New forks follow skill bringup-fork-op:
   - tests/source.yaml with a best-effort selection of the source op's tests, plus a recorded baseline
     (testing/fork_source.py --record, one entry at a time);
   - a CHANGELOG entry and an INDEX.md row;
   - a successful ./build_metal.sh.
3. Extending an existing fork follows the recipe in section 3:
   - the recorded baseline, no rerun;
   - the new behaviour behind an option, default off, with the default program unchanged;
   - iterate on a few targeted cases plus this model's case, then the model's gate;
   - the full regression once at the end (unit suite, fork_source check, every model's cases), with no regressions
     for 1x4 MiMo, Gemma or ERNIE.
4. Task O.1 at the end: every ttnn.bringup call gets a random-input case (skill bringup-fork-tests).
5. Prior seeding is actually used:
   - briefs have a "Prior bring-up" section and the prior's files;
   - G.* tasks report golden_reused 1;
   - R.* pass on their first check;
   - implement agents copy and adapt the prior's modules (never import the prior's tt/).
6. When a perf pick or op switch changes the model, rerun the frozen component and swap tests of the affected
   layers, not only the ladder. On 1x4 a swap test failed silently for that reason.

Dashboards: both styles (`dashboard.styles: both` in the spec, so standard + teletext). Re-export and republish both
as artifacts after every gate, and give me both links at the start and at the end. Keep the per-op profile tabs
(per block type, op placement, pipelined timeline) and add the comparison against the 1x4 run.

Operational rules:
- One device job at a time. A concurrent single-card probe breaks mesh gates (known issue); never run agents that use
  the device in parallel.
- Never wrap orchestrator runs in `timeout`; run them in the background with a monitor.
- Commit each verified piece right away (explicit paths). Don't push unless I ask.
- Never run tt-smi -r unless I say so. Run run_safe_pytest.sh and tt-probe.sh in the foreground.
- Decide routine things yourself. Ask me for the spec, the plan, the perf picks, board resets and pushes.

At the end, compare against 1x4 for accuracy (per-layer PCC), the 50k->55k chunk (1x4: 204.9 ms device) and 0->55k
TTFT, and list which bringup ops and forks were reused, extended or newly created.
