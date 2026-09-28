# XiaomiMiMo/MiMo-V2.6-Flash-RL bring-up on mesh 2x2: breadcrumbs

Prior bring-up: mimo_v2_6_d_p (mesh 1x4); goldens and CPU reference shared. Append-only log, one section per task attempt: what was done, decisions and why, gotchas, the re-run command, the verdict.

## PL.1 plan (attempt 1), 2026-09-28

- Wrote `plan.yaml`, `plan.md`, `components.yaml`; re-planned from the 1x4 prior's plan and its final modules
  (`models/demos/mimo_v2_6_d_p/tt`, which use the bring-up forks: rms_norm, sdpa with V 128, unified_routed_expert_ffn
  high_precision, dispatch/combine/offset_cumsum). tasks.yaml unchanged.
- Attention and dense MLP: TP=4 over the flattened mesh, chip d = 2*row + col holds checkpoint TP rank d (ShardTensorToMesh).
  The reduce is `ttnn.all_reduce(cluster_axis=None)`: on a 2x2 mesh it runs axis 1 then axis 0 (all_reduce.cpp). Rejected
  TP=2 x SP=2: 4 KV heads / 4 TP ranks map 1:1, SP on causal chunked prefill needs KV gather or ring SDPA.
- Experts: the prior's local 1-chip dispatch needed a size-1 mesh axis; 2x2 has none. Chose the DeepSeek 2D layout:
  dispatch axis 0 (DGS 2, fabric), one dispatch group per column (128 experts), chip (r, c) holds experts 128c + 64r .. +63
  (ExpertMapping col-major). `ttnn.mesh_partition` of x/idx/wts by row at the MoE entry (S/2 per row), after combine
  all_reduce axis 1 + all_gather axis 0 -> replicated [S, 4096]; component boundaries identical to 1x4. The forks
  behave as the source ops on a 2-device axis, so no op change planned. DeepSeek's conftest runs (2, 2) with FABRIC_2D.
  Fallback (in plan.md): a default-off local-dispatch option in the forks.
- V planned at 128 (sdpa fork), so the KV state is 0.06 GiB smaller than the prior's plan. Per chip 14.59 of 27.20 GiB.
- Gate (no device): plan_fits 1, 0 unplaced, 0 plan/component/ledger errors; plan approved False until a person approves.
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup.plan.check_plan`
