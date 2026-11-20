# GLM-5.3-Flash (layers 0-4) on 4x Blackhole: run it, profile it, dig into the mHC

For a Claude Code session on the same box as the bring-up (`/localdev/dnijemcevic/...`, 4x Blackhole p300c, mesh 2x2,
FABRIC_2D). Paste the "Prompt" section into the session.

## Prompt

You are picking up a finished bring-up of GLM-5.3-Flash chunked prefill (layers 0-4 of 45; text decoder only) on this
box. Your job: get it running, reproduce the timing breakdown, then study how the mHC (the 4-stream "manifold hyper-
connection" residual) is implemented on the device and whether it can be made faster or more accurate. Report what
you find with measurements; do not change the model's default behaviour without the owner's say-so.

### Where things are

- Code: branch `dnijemcevic/glm53_prefill` on origin (tt-metal). The owner's checkout is
  `/localdev/dnijemcevic/tt-metal`; use your own clone (`git fetch origin dnijemcevic/glm53_prefill`, check it out,
  `git submodule update --init --recursive`, `./build_metal.sh`, `./create_venv.sh` - the branch has C++ changes: a
  ttnn.bringup sdpa fork option and two KDA commits cherry-picked from main). Do not work in the owner's checkout.
- Model: `models/demos/glm53_flash_d_p/`
  - `tt/model.py` (TtGlmBlock, the all-device model, split step table), `tt/common.py` (mesh helpers, the split
    layout's gathers / scatters), `bringup/hooks.py` (GlmDeviceModel: what the tests drive).
  - mHC: `tt/mhc.py` (coefficients: projection + 20-step Sinkhorn via DeepSeek's `mhc_split_sinkhorn`),
    `tt/collapse.py` (the 4 streams -> 1 before attention / MLP), `tt/residual.py` (the residual mix, P.1), `tt/rms_norm.py`.
  - CPU reference: `reference/glm_ref.py` (`hc_weights`, `hc_collapse`, `hc_residual`), HF code vendored in `reference/hf/`.
- Bring-up records (read these first): `bringup/BREADCRUMBS.md` (per-task notes; sections "P.1 perf" and "P.2 perf"
  are the mHC optimisations), `bringup/plan.md` (sharding), `bringup/supervision.md`, `bringup/findings.yaml`,
  `bringup/results/*.json` (every gate's metrics; `X.1_profile.json`, `P.1.json`, `P.2.json`, `X.3_profile.json`),
  `models/demos/common/bringup/knowledge/known_issues.md` (search "mHC") and `repo_map.md`.
- On-disk artifacts (read-only, owned by dnijemcevic; do not modify or delete):
  - weights: `/localdev/dnijemcevic/bringup/glm53_flash_d_p/hf` (zai-org/GLM-5.3-Flash @ eb9eb208eb0d, FP8 e4m3 +
    128x128 block scales, trimmed to layers 0-4 + embed / norm / lm_head, 18 GB)
  - goldens: `/localdev/dnijemcevic/bringup/glm53_flash_d_p/golden/{s4096_c2048,s16384_c8192,s56320_c5120}` (16 GB,
    CPU reference dumps per layer, per chunk, with state snapshots)
  - prompt tokens: `.../input/`; profiles: `.../profiles/*.json`; orchestrator logs + agent transcripts: `.../runs/run1/`
- Dashboards: https://claude.ai/artifact/J7Qb5GkEhxXevKJ7WVXuX9 (standard, shared with the org).

### The bring-up framework (models/demos/common/bringup)

Read `README.md`, `docs/pipeline_design.html` and `dev/BREADCRUMBS.md` (framework changes F1..F52) there. In short:
- A model is a spec (`bringup/spec.yaml`), a ledger of gated tasks (`bringup/tasks.yaml`, status in `state.json`,
  metrics in `results/<task>.json`) and hooks (`bringup/hooks.py`: `reference`, `device_component`, `device_model`,
  `contract_state_pcc`, ...). An orchestrator ran one agent per task; a task passes only when its gate command meets
  its metric thresholds. `python -m models.demos.common.bringup status --spec $BRINGUP_SPEC` lists them. Do not run
  the orchestrator or `rerun` on this model: the run is finished, and they rewrite the ledger.
- Test layers, all driving the same hooks against the CPU goldens:
  - component tests `models/demos/glm53_flash_d_p/tests/bringup/test_c_<block>_<step>.py`: one step of one block type
    on the device vs the golden at that step's boundary (PCC plus extra checks the test author calibrated against
    injected bugs; frozen: never edit them to make a change pass). `BRINGUP_IMPL=reference|stub` runs the CPU
    reference or a stub through the same test.
  - swap tests `test_swap_<block>_<nn>_<step>.py`: the whole block with steps 1..nn on the device and the rest on the
    CPU (the hybrid harness), checking the block output.
  - ladder `models/demos/common/bringup/tests/test_ladder.py` (rungs s4096, s16384, last, s56320 from the spec): the
    all-device model over real chunks, per-layer PCC and state PCC (KDA recurrent / conv, MLA latent, indexer keys).
  - contract `tests/test_contract.py`: the model driven through the prefill engine's adapter API, as the inference
    server does (padded last chunk, acks, KV table, fixed-state read-back).
  - profile `tests/test_profile.py`, positions `tests/test_positions.py`: device time per section and per chip.
- Framework selftests (CPU): `scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/common/bringup/selftest`.

### Rules on this box

- `source python_env/bin/activate; export PYTHONPATH=$PWD` in your checkout (the shell default points elsewhere);
  `export BRINGUP_SPEC=$PWD/models/demos/glm53_flash_d_p/bringup/spec.yaml`.
- Run device tests only through `scripts/run_safe_pytest.sh` (it locks the 4 cards and resets them); one device job at
  a time; never `tt-smi -r`. A 56k accuracy run takes a few minutes; a profile about 1 minute.
- Ad-hoc runs write metrics to `generated/bringup_adhoc/` (set `BRINGUP_RESULTS_DIR` to change it) and profiles to
  `/localdev/dnijemcevic/bringup/glm53_flash_d_p/profiles/`; do not commit into `bringup/results` or `state.json`.

### Reproduce

```bash
# accuracy: last chunk (51200 -> 56320) after the 50k golden prefix; per-layer PCC + state PCC
BRINGUP_RUNG=last scripts/run_safe_pytest.sh --no-precompile --run-all models/demos/common/bringup/tests/test_ladder.py
# the whole 56k in 11 chunks (s4096 / s16384 for quicker rungs)
BRINGUP_RUNG=s56320 scripts/run_safe_pytest.sh --no-precompile --run-all models/demos/common/bringup/tests/test_ladder.py
# warm per-section, per-chip device profile of one 5120-token chunk (device_ms_<section>)
PROF="TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=4000"
env $PROF scripts/run_safe_pytest.sh --no-precompile --run-all models/demos/common/bringup/tests/test_profile.py
# + per-op rows and the full 0 -> 55k prefill time
BRINGUP_FULL_PREFILL=1 BRINGUP_PROFILE_OPS=1 env $PROF scripts/run_safe_pytest.sh --run-all models/demos/common/bringup/tests/test_profile.py
# one mHC component vs its golden (block types kda_dense L0, dsa_moe L3, kda_moe L4; steps attn_hc, attn_collapse,
# attn_norm, attn_residual, ffn_hc, ffn_collapse, ffn_norm, ffn_residual)
scripts/run_safe_pytest.sh --run-all models/demos/glm53_flash_d_p/tests/bringup/test_c_kda_dense_attn_hc.py
```

Switches (env): `GLM_RESIDUAL_MIX=matmul|addcmul` (P.1; matmul default), `GLM_RESIDUAL_LAYOUT=split|replicated` (P.2;
split default), `GLM_MLA_SDPA=fork|source`, `GLM_INDEXER_SCORE=heads|op`, `GLM_KDA_DECAY`, `BRINGUP_HYBRID=1` (CPU
reference with device steps swapped in, for debugging one step).

### Where it stands (warm chunk 51200 -> 56320, ms device, slowest chip per section)

| Section | Start | After P.1 (matmul mix) | After P.2 (split by sequence) |
|---|---|---|---|
| total | 528.4 | 428.4 | 276.2 |
| attention (KDA L0-2, L4 + sparse MLA L3) | 80.2 | 82.1 | 76.7 |
| routed experts | 55.8 | 57.3 | 51.5 |
| indexer (L3) | 46.2 | 46.1 | 46.2 |
| dense MLP (L0-2) | 48.1 | 47.7 | 40.1 |
| attn_residual / ffn_residual (mHC mix) | 82.9 / 82.8 | 31.5 / 31.5 | 8.0 / 8.0 |
| attn_hc / ffn_hc (mHC coefficients + Sinkhorn) | 31.4 / 31.4 | 31.3 / 31.2 | 8.7 / 8.7 |
| attn_collapse / ffn_collapse | 24.4 / 24.4 | 24.4 / 24.4 | 6.4 / 6.4 |
| norms, q_a, router, shared expert, moe_add | 20.7 | 20.8 | 15.9 |

Full prefill 0 -> 55k: 2.88 s (4.50 s before P.1 / P.2). Accuracy at 56k: every layer PCC >= 0.9995; worst state
PCC 0.982 (layer 1 KDA recurrent state, drifting with length: 0.9992 at 4k, 0.993 at 16k).

### What to investigate (mHC)

1. Read P.1 / P.2 in BREADCRUMBS.md and the mHC entries in known_issues.md. mHC is now ~46 ms of 276 (was 277 of 528).
2. Per-op breakdown of attn_hc / ffn_hc (projection [16384 -> 24] in fp32, the 20-iteration Sinkhorn, the concat of
   pre | post | comb), the collapses and the residual mix: which ops dominate, how many programs, device idle gaps.
3. Ideas to test, each measured against the component tests and the `last` rung: fuse coefficients + collapse (both
   read the same [S/4, 4H] rows); fewer or fused Sinkhorn iterations only if the component PCC holds; the P.1 mix reads
   fp32 coefficients as TF32 (known_issues: small comb entries run a few percent low) - check whether a higher-
   precision path costs anything; the norms after P.2 carry gathers (ffn_norm 1.2 -> 3.2 ms), so check the CCLs the
   split layout added (BREADCRUMBS P.2 lists them).
4. Keep accuracy: every mHC component test (24 files `tests/bringup/test_c_*_{attn,ffn}_{hc,collapse,norm,residual}.py`)
   and the `last` + `s56320` rungs must still pass; put any new path behind a switch with the current one as default.

Write findings to a new section at the end of `models/demos/glm53_flash_d_p/bringup/BREADCRUMBS.md` in your branch.
