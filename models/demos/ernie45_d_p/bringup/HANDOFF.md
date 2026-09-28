# Handoff: ERNIE-4.5 prefill bring-up, and the next task

Read this first in a new session. It says what was done, where every record lives, what is broken or
unfinished, and what to build next.

- Repo: `/localdev/dnijemcevic/tt-metal2`, branch `dnijemcevic/ernie45_prefill` (pushed to origin),
  forked from `llk_helper_library`.
- Box: 4x Blackhole p150b QuietBox, opened as mesh (1,4) with `FABRIC_1D_RING`.
- Owner's working preferences: gated tasks with a commit per passed gate, breadcrumbs, a live dashboard;
  writing that follows the unslop rules (https://github.com/cursor/plugins/blob/main/pstack/skills/unslop/SKILL.md).

## 1. What was done

Chunked prefill of `baidu/ERNIE-4.5-21B-A3B-PT` (28 layers: layer 0 dense MLP, layers 1-27 MoE with 64 experts,
top-6 and 2 shared experts; GQA 20 Q / 4 KV heads, head_dim 128) runs end to end on device at 55k tokens in
5,120-token chunks. All 26 gates pass.

| Gates | What they cover | Key result |
|---|---|---|
| P1.1-P1.3 | Checkpoint check, standalone CPU reference vs HF, chunked == one-shot | PCC 1.0 vs HF, 100% top-1 |
| P1.4-P1.6 | Goldens at 2k->2k, 8k->8k, 55k@5k (A Tale of Two Cities) | 55k golden took 61 min on CPU |
| P2.1-P2.2 | Mesh and collectives smoke test, sharding plan | 11.4 of 32 GB per chip |
| P2.3-P2.9 | Components: norm, embedding, RoPE, attention, MLP, router, MoE | all PCC >= 0.9999 |
| P2.10-P2.14 | Decoder blocks, full model 2k, 8k, (b) 50k prefix + last chunk, (a) 55k@5k | final hidden 0.996, top-5 100% |
| P2.15-P2.16 | Prefill-server KV contract (bf8 layout, address table, producer read-back), adapter + runtime | read-back PCC 0.994 |
| P3.1-P3.2 | Fused `unified_routed_expert_moe` MoE (needed ttnn patches) | 55k TTFT 35.9 s -> 10.9 s |
| P3.3 | Per-phase, per-chip device profile of the 50k->55k chunk | SDPA was 85% of the chunk |
| P3.4 | SDPA config A (HiFi2, fp32 acc off, approx exp, q256/k512) | 55k TTFT 10.9 s -> 4.4 s |

`python models/demos/ernie45_d_p/bringup/gate.py --status` prints every task, its verdict and its commit.

## 2. Where the records are

All under `models/demos/ernie45_d_p/bringup/`, committed and pushed:

| File | Contents |
|---|---|
| `BREADCRUMBS.md` | Narrative log per stage: decisions, gotchas, re-run commands, results. Read this second. |
| `tasks.yaml` | Gate spec per task: deps, command, thresholds, owned paths. |
| `state.json` | Verdicts and history, written only by `gate.py`. |
| `results/<task>.json` | Raw metrics per gate. `P3.3_profile.json`, `P3.4_profile.json` (per-phase, per-chip), `P3.3_routing.json`. |
| `components.yaml` | Component -> reference op -> TTNN op, NATIVE/COMPOSED/CPU tag, gating task, and the `findings:` list. |
| `gate.py`, `metrics.py` | Gate runner (`<id> --commit`, `--next`, `--status`, `--sweep [prefix]`) and metric sink. |
| `export_dashboard.py`, `dashboard/` | Dashboard generator and template. |
| `models/demos/common/bringup/docs/pipeline_design.html` | The framework overview (moved there; replaces draft 4). Published copy: https://claude.ai/artifact/L5rXDnJoEpjsEL33s3wmSC |
| `logs/<task>.log` | Full gate output. Gitignored, local only. |

Other pointers:

- Dashboard (shared with the org): https://claude.ai/artifact/AUqQ6MVBV2CXs9RNiUBPYz. Regenerate with
  `python models/demos/ernie45_d_p/bringup/export_dashboard.py`, then republish that `dashboard/index.html` to the same URL.
- Git history: `git log --oneline --grep='\[ernie45_d_p\]'`. Gate commits are tagged `[ernie45_d_p][<id>]`.
- Kimi K2.7 4x4 plan artifact, the informal reference for sharding plans:
  https://claude.ai/artifact/4MC3c1hvkErCYxByJwdPZ9
- Tensor caching research: `/localdev/dnijemcevic/tt-metal2/tensor_caching_research.md` (untracked, the owner's file).

## 3. Code map

- `models/demos/ernie45_d_p/reference/`: `ernie_ref.py` (standalone CPU model with recorder hooks and a KV cache),
  `check_*.py` (P1 gates), `generate_golden.py`.
- `models/demos/ernie45_d_p/tt/`: `common.py` (mesh helpers, caches, `Golden` reader, `signpost` and section profiler),
  `ops.py` (RMSNorm, RoPE, SwiGLU), `attention.py` (TP4 attention, bf16 attention cache, SDPA presets via
  `ERNIE_SDPA_CFG`), `moe.py` (router, dense-EP MoE), `moe_unified.py` (fused MoE, the default; `ERNIE_MOE_IMPL=dense`
  restores the old one), `model.py`, `kv_contract.py`, `runners/adapters/ernie45.py`.
- `models/demos/ernie45_d_p/tests/`: `pcc/` component tests, `test_model_chunked.py` (ladder), `test_kv_contract.py`,
  `test_adapter.py`, `perf/test_profile_chunk.py`.
- Shared code changed on this branch:
  - `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/{offset_cumsum,dispatch,combine}`: skip fabric setup on a
    1-device dispatch axis. Needs `./build_metal.sh` after checkout.
  - `models/demos/common/prefill/adapter.py`: the `ernie45_d_p` registry entry.
  - `models/demos/common/prefill/runners/prefill_producer.py`: ERNIE branch, no HF->Meta K permutation.

## 4. Data locations (not in git)

- HF checkpoint: `/localdev/dnijemcevic/hf_data/.cache/huggingface/hub/models--baidu--ERNIE-4.5-21B-A3B-PT/`.
- Goldens: `generated/ernie45_d_p/golden/` (44 GB).
- TTNN weight cache: `generated/ernie45_d_p/tt_cache/` (82 GB). Its keys do not include the mesh shape.
- The design moves all of these under `/localdev/$USER/bringup/<model>/`.

## 5. Environment traps

- The shell has `PYTHONPATH=/localdev/dnijemcevic/tt-metal` (another checkout). `gate.py` pins it to this repo.
  Set `PYTHONPATH=$PWD` when running tests by hand.
- Run device code only through `scripts/run_safe_pytest.sh` or `scripts/tt-probe.sh`, in the foreground.
- The precompile pass in `run_safe_pytest.sh` stubs `comp_pcc` to return 0.999999, and the stub can leak into the real
  pass. Tests use their own `pcc()`; `test_adapter.py` runs with `--no-precompile`.
- Tracy host capture (`--profile`) crashes on this box. The device profiler works with `TT_METAL_DEVICE_PROFILER=1
  TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1` (see the P3.3 gate).
- The first run with new SDPA kernels compiles one program per chunk offset. Measure performance on a warm run.

## 6. Unfinished work on ERNIE itself

- The tt-d-gen audit found gaps inside the Blaze contract scope: layer acks fire before the KV is on device (sync
  before `sink()` in `tt/runners/adapters/ernie45.py`); `prefill_chunk` expects a torch tensor, but the engine passes a
  uint32 device tensor with `0xFFFFFFFF` padding; the cache asserts 64-aligned starts where the engine allows 32;
  `PREFILL_NUM_USERS` is missing from the manifest.
- Two KV caches: SDPA reads a bf16 attention cache, and the address table describes the bf8 contract cache.
- SDPA config A passed but is not the default (`ERNIE_SDPA_CFG=A`). The SDPA call should use `chunk_start_idx_tensor`
  so one program serves every chunk offset.
- The MoE router, dispatch and combine now cost about twice the fused experts (100 ms vs 54 ms per 50k->55k chunk).

## 7. The next task: build the framework

Build the pipeline described in `docs/pipeline_design.html`. Read it fully before planning. Decisions already made
(recorded in the doc): test author and implementer share one agent definition with two roles; a zero stub validates a
test; the debugger is `ttnn-expert-debugger` after a WIP commit, 3 attempts each; the contract stops at what the Blaze
prefill CI checks; fixed thresholds; artifacts under `/localdev/$USER/bringup/`; one orchestrator script launched by a
conversational intake skill; swap tests on one layer per block type; every run records agent-definition hashes;
performance ends in a human-picked opportunity list ranked by measured time.

Proposed order, each piece gated and committed:

1. Extract the generic parts of `bringup/` into `models/demos/common/bringup/`: gate runner, metrics, dashboard,
   and the generic tests (`check_vs_hf`, `check_chunked`, `generate_golden`, ladder, KV contract, adapter,
   profile), parametrized by a model spec.
2. Prove the extraction by re-running the ERNIE sweep through the framework and checking the metrics match.
3. Add test freezing (hashes in the ledger), the stub check, resume, rerun, fork, and agent-definition hashes.
4. Seed the repo map and the known-issues file from `BREADCRUMBS.md`, `components.yaml` findings and the design doc.
5. Write the orchestrator script and the `/bringup` intake skill, and the per-step agent briefs.
6. Dry-run on a second model, chosen with the owner.

A prompt to start the new session with:

> Read `models/demos/ernie45_d_p/bringup/HANDOFF.md`, then `BREADCRUMBS.md`, then
> `docs/pipeline_design.html`. Build the framework in section 7, starting with step 1. Keep the gated workflow:
> a task per piece, a commit per passed gate, breadcrumbs, and dashboard updates. Ask before changing shared code
> outside `models/demos/common/bringup/`.
