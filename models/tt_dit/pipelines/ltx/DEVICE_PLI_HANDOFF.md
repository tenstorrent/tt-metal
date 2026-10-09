# Device-PLI hand-off: Phases 1 and 2 done

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
- Two warm nodes, one job per node at a time:
  - `bh-glx-120-b02u08`: `-p bh_wan_machines -- -w bh-glx-120-b02u08`. Quiet; use it for timing.
  - `bh-glx-110-d07u02`: `-- -w bh-glx-110-d07u02`. Noisier today (p90 24-26 ms); use it for correctness.
- Env vars set at submit time do NOT reach the container. Put knobs in the cmd file.
- Cmd files in `/data/rsalman/jobs/cmds/`:
  - `metal-galaxy-e2b-timing.sh`: host-PLI timing baseline
  - `metal-galaxy-e2b-device-pli.sh`: Phase 1 gate (PCC test + four-arm parity)
  - `metal-galaxy-e2b-device-pli-timing.sh`: device-PLI timing
  - `metal-galaxy-e2b-parity-probe.sh`: teacher-forced vs HF, host and device PLI
  - `metal-galaxy-e2b-feedback.sh`: Phase 2 gate
  - `metal-galaxy-ltx-{ab,serving-check}.sh`: LTX hang repros
- Baselines: `/data/rsalman/jobs/baselines/` (host-PLI token stream; HF reference JSONs under `hf/`)
- Results per job: `/data/rsalman/jobs/run/<jobid>/`, logs `/data/rsalman/jobs/logs/<jobid>.log`
- Hang on exabox: cards reset per job; just resubmit
- A new cache tensor is written to NFS on first load. Don't run two first loads at once.

## Done (commits on the branch)

| Commit | What | Measured |
|---|---|---|
| `ccaa3a4` | `pli_host_microbench.py` | PLI host cost 9–22 ms/tok; `proj_w.float()` dominates |
| `4e6eb1f` | quick wins: cached fp32 proj, direct `compute_host_pli` | decode 43.9 → **17.1 ms/step** (56 tok/s), bit-exact |
| `848e97b` | enhancer decode via device sampler (`SamplingParams`) | opt-out `LTX_ENHANCER_HOST_SAMPLE=1` |
| `8712e1c` | `LTX_ENHANCER_TRACE=1` knob, `AB_HOST_PARITY=0` gate | traced rewrite in-pipeline **55 tok/s** |
| `b392577` | **Phase 1: device PLI** behind `GEMMA4_DEVICE_PLI=1` | trace-device **17.4 → 14.5 ms/step** (65.5 tok/s), see below |
| `2fbc3ce` | `GEMMA4_IMPL_SURVEY.md` | comparison of other Gemma4 implementations, ranked next steps |
| (Phase 2) | **on-device token feedback** with device PLI | trace-device **13.5 ms/step**, p90 13.5 (74 tok/s); 1 host restage per request |

Numbers come from 188-step greedy runs of `e2b_bringup.py::test_e2b_decode_timing`. The first four rows were measured on `bh-glx-110-d07u02`.

## Phase 1 result (device PLI, decode only)

- **Code:** `models/demos/gemma4/tt/model.py`, in `_load_device_pli_weights` and `compute_device_pli`.
  - Wired into `ttnn_decode_forward`. `prepare_decode_inputs_host` skips host PLI when the flag is on.
  - All weights are allocated at init, before any trace capture.
  - Prefill PLI stays on host (plan Phase 4).
- **Placement:**

  | Weight | Placement |
  |---|---|
  | `embed_tokens_per_layer` | TP=8 column-parallel: 1120 cols/chip (~0.55 GiB), replicated over the 4 rows |
  | projection | transposed `[1536, 8960]`, replicated, HiFi4/fp32 acc |
  | norm weight | used as-is |

- **Scale fold:** the H^-0.5 projection scale is folded into the RMSNorm eps (eps·H). This is exact, because RMSNorm is scale-invariant apart from eps.
- **Gates:**

  | Gate | Job | Result |
  |---|---|---|
  | (a) device vs host PCC, 50 ids, eager + traced | 128888 | min PCC 0.999992, min per-layer 0.999978; all 32 chips identical; traced == eager |
  | (b) four-arm greedy parity, 188 steps | 128888 | all four arms agree with each other |
  | (c) `e2b_parity_probe.py`, teacher-forced vs HF | 128891 | host **61/64**, device **61/64** top-1; logit corr 0.9969 / 0.9970 |

- **Note on (b)/(c): the greedy stream differs from the host-PLI stream at index 2.** It is the contested token from bring-up: host PLI picks " cinematic", HF picks " documentary".
  - Device PLI now matches HF at index 2 (KL 0.082 → 0.009) and flips a different near-tie at index 3 (device top-1/top-2 gap 0.125).
  - Mismatches 28 and 45 are shared with host PLI.
  - So use the probe, not exact equality with the host stream, to judge parity.
- **Timing, trace-device arm:**

  | Node | Host PLI | Device PLI |
  |---|---|---|
  | b02u08 | 17.4 ms median, 17.1 min (job 128881) | **14.4–14.7 ms median, 14.1 min** (job 128889) |
  | d07u02 | 20.1–20.8 ms (job 128880, noisy) | 13.8 ms (job 128888) |

## Phase 2: on-device token feedback (done)

- **Change:** with `GEMMA4_DEVICE_PLI=1`, `_tt_vllm_always_refresh_decode_trace_inputs` is False for PLI models (model.py ~l.310).
  - This turns on the non-PLI feedback path: a `[1,1,1,32]` token buffer that the sampler writes into, and `plus_one` on the device positions.
  - `GEMMA4_ALWAYS_REFRESH_DECODE=1` restores the Phase 1 behaviour.
- **Harness:** `e2b_bringup.py` decodes trace-device with `reload_inputs=False` after step 0 when feedback is on.
  - It counts `prepare_decode_inputs_host` calls per repeat.
  - `E2B_FEEDBACK_POISON=1` passes token 0 / position 0 from the host on every feedback step. The output must not change.
- **Gate (`metal-galaxy-e2b-feedback.sh`):** feedback tokens must equal the device-PLI restage stream over 188 steps, and in-run equal trace-host. The poisoned run must also equal it.
- **Result (job 128893, b02u08):**
  - Feedback, restage, trace-host and the poisoned run all produced identical tokens over 188 steps, in every repeat.
  - Host staging per warm repeat: 1 call (step 0), down from 188. The cold repeat has 3: compile, trace prep, step 0.
  - trace-device: 14.5 ms (restage) → **13.49 ms median, p90 13.51 ms**.
- **What the timing means:** the steady step time is now flat, so the loop is probably device-bound. That puts device time per step at about 13.5 ms, not the unverified ~11 ms.
  - The harness still reads each token back before queuing the next step.
  - Remaining gains therefore come from device time (see the survey), plus lagged readback (plan Phase 3). Lagged readback can only hide host work and readback latency.
- `GEMMA4_ALWAYS_REFRESH_DECODE=1` gives back the Phase 1 behaviour.
- **Not covered:**
  - vLLM: `generator_vllm.py` still disables async decode for PLI models.
  - `text_demo.py` E2B with the flag.
  - Temperature > 0 with feedback.

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
- Device PLI adds one all-gather inside the decode trace. It is untested together with LTX resident, so LTX integration stays blocked on this bug.

## Survey of other Gemma4 implementations

Done: `GEMMA4_IMPL_SURVEY.md` next to this file.
- Next step: a Tracy op profile of one decode step on b02u08. The "~11 ms device per step" figure has no profile behind it.
- The leading hypotheses are per-op overhead and latency-bound CCLs, not weight bandwidth.

## Environment gotchas (cost us time; don't rediscover)

- Container `$HOME` is node-local; caches go under `/data/$USER/cache/`
- `/mnt/models` mounted in test containers; HF snapshot path is in `e2b_bringup.py`
- No `python_env` in tree: test stage installs the wheel; `PYTHONPATH=$TT_METAL_HOME`
- Single host: `TT_MESH_ID=0 TT_MESH_HOST_RANK=0`, no mpirun
- The container cwd is not `$TT_METAL_HOME`: `cd` first, or use absolute test paths
- Step times vary by node and by day. Compare like with like on the same node, measured the same day.
- The 26.2 vs 43.9 ms gap in the plan was plan-era vs pre-Phase-0 code, not hardware.
