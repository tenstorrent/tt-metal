# Device-PLI hand-off: continue Phase 1

2026-10-09. Branch `rsalman-e2b-device-pli` on `tenstorrent/tt-metal`, based on
`rsalman-ltx-prompt-enhancing` @ `f78531d`. Tree on the cluster:
`/data/rsalman/tt-metal-pli` (build artifacts already on NFS).

**Goal:** move Gemma-4-E2B per-layer inputs (PLI) into the device decode trace.
Full plan: https://claude.ai/artifact/Ss6fYA8Bk5b9gt7ALQCehy

## How to work

- Login: `ssh rsalman@slurm-login.exabox.tenstorrent.com`
- Tool: `tt-sbatch` (docs: exabox-infra `docs/exabox-tt-sbatch.md`)
- Python-only change in `models/`: `tt-sbatch test` alone, no rebuild
- C++/ttnn change: `tt-sbatch run` (build + test chained)
- Galaxy runs: `--topo galaxy --mode single`; add `--timeout <min> --triage`
- Warm node from this work: `-- -w bh-glx-110-d07u02` (optional)
- Cmd files: `/data/rsalman/jobs/cmds/metal-galaxy-e2b-{phase0,timing}.sh`,
  `metal-galaxy-ltx-{ab,serving-check}.sh`
- Results per job: `/data/rsalman/jobs/run/<jobid>/`, logs `/data/rsalman/jobs/logs/<jobid>.log`
- Hang on exabox: cards reset per job; just resubmit

## Done (commits on the branch)

| Commit | What | Measured |
|---|---|---|
| `ccaa3a4` | `pli_host_microbench.py` | PLI host cost 9–22 ms/tok; `proj_w.float()` dominates |
| `4e6eb1f` | quick wins: cached fp32 proj, direct `compute_host_pli` | decode 43.9 → **17.1 ms/step** (56 tok/s), bit-exact |
| `848e97b` | enhancer decode via device sampler (`SamplingParams`) | opt-out `LTX_ENHANCER_HOST_SAMPLE=1` |
| `8712e1c` | `LTX_ENHANCER_TRACE=1` knob, `AB_HOST_PARITY=0` gate | traced rewrite in-pipeline **55 tok/s** |

Numbers from `bh-glx-110-d07u02`, 188-step greedy, `e2b_bringup.py::test_e2b_decode_timing`.

## Open bug (do NOT re-debug blind; file or hand to runtime)

- Traced enhancer + LTX work deadlocks the mesh after trace **replays**
- Signature: fetch-queue wait timeout; then "device unrecoverable"
- Every chip stuck on physical cores **15-2, 15-3** (the 2-link sampling all-gather ring)
- Repros (deterministic): `tt-sbatch test --topo galaxy --mode single --timeout 45 --triage --test metal-galaxy-ltx-ab -- ...`
  (untraced DiT: hangs at first eager DiT dispatch after replay) and
  `--test metal-galaxy-ltx-serving-check` (LTX_TRACED=1: hangs after gen 0 stage 1)
- Evidence: jobs 128420, 128467, 128486 under `/data/rsalman/jobs/{logs,run}/`
- Separate tooling bug: `tt-triage` fails, tt-exalens 0.4.1 installed vs 0.3.32 pinned;
  workaround `--skip-version-check` (manual run worked partially)
- Standalone enhancer (no LTX resident) is unaffected: capture+replay fine

## Next: Phase 1, device PLI (plan phases 1–2)

Target design is in the plan artifact ("Target design" + "Work plan"). Essentials:

| Weight | Placement |
|---|---|
| `embed_tokens_per_layer` [262144, 8960] | column-sharded TP=8 (1120/chip), replicated over 4 rows, row-major DRAM bf16 |
| `per_layer_model_projection` → [1536, 8960] | replicated, tile, bf16 |
| `per_layer_projection_norm` [256] | replicated; weight used as-is, NOT (1+w) |

Op chain inside `ttnn_decode_forward`: embedding lookup on sharded table →
all-gather → ×16 ; matmul from `embed_tokens` output → reshape [B,35,256] →
×1536^-0.5 → RMSNorm×w → add → ×2^-0.5 → `[1,1,35,256]`.

- Files: `models/demos/gemma4/tt/model.py` (load at `__init__` near
  `embedding_weight` ~line 373; compute; wire into `ttnn_decode_forward` ~2256;
  return None PLI from `prepare_decode_inputs_host` ~2218)
- Gate behind `GEMMA4_DEVICE_PLI=1`; keep host path as reference
- Allocate all new tensors at load, before any capture
- Exit: device-vs-host PCC ≥ 0.999 over ~50 tokens; four-arm greedy parity
  in `e2b_bringup.py` holds; `e2b_parity_probe.py` no worse than host baseline
- Then plan Phase 2: flip `_tt_vllm_always_refresh_decode_trace_inputs` (model.py:301),
  token feedback on device
- Note: device PLI's all-gather lives INSIDE the decode trace — the open CCL
  hang is about replay-vs-LTX coexistence, not in-trace CCLs; measure standalone
  first (`metal-galaxy-e2b-timing.sh`), LTX integration stays blocked on the bug

## Survey other Gemma4 implementations (required)

- 55 tok/s greedy is slow for a whole Galaxy
- E2B is 2B params; device step ~11 ms is high
- Multiple teams contribute Gemma4 code to tt-metal
- Find single-Galaxy Gemma4 runs outside `models/demos/gemma4`
- Start points: `models/tt_transformers`, demo/test yamls, perf dashboards
- List their optimizations: sampling, CCL, layout, trace scope, batching
- Known waste here: rows 1–3 duplicate compute (DP unused, batch 1)
- Compare per-step device time, not tok/s alone
- Adopt what applies; note what does not and why

## Environment gotchas (cost us time; don't rediscover)

- Container `$HOME` is node-local; caches go under `/data/$USER/cache/`
- `/mnt/models` mounted in test containers; HF snapshot path is in `e2b_bringup.py`
- No `python_env` in tree: test stage installs the wheel; `PYTHONPATH=$TT_METAL_HOME`
- Single host: `TT_MESH_ID=0 TT_MESH_HOST_RANK=0`, no mpirun
- Step times vary by node: plan says 26.2 ms/step on bh-glx-120, we measured
  43.9 ms baseline on bh-glx-110 — compare like with like, pin the node
