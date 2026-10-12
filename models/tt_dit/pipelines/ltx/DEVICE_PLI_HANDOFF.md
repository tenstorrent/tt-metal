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
| `312ce72` | **on-device token feedback** with device PLI | trace-device **13.5 ms/step**, p90 13.5 (74 tok/s); 1 host restage per request |

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

## Phases 3 and 4: feasibility review (2026-10-11, read-only, no jobs)

**Phase 3 (enhancer integration and run-ahead).** Doable standalone now. Inside LTX it delivers nothing until the open bug below is fixed: a traced enhancer hangs even with an untraced DiT (`metal-galaxy-ltx-ab.sh`), and feedback/run-ahead only exist on the traced path.
- **Stage A, ship first.** Add an `LTX_ENHANCER_DEVICE_PLI` knob that sets `GEMMA4_DEVICE_PLI=1` around model construction in `ensure_loaded`, and keep `reload_inputs=True`.
  - About 14.5 ms/step, ~0.5 s saved per 188-token rewrite. Works at the serving temperature of 0.7.
  - Test separately: device PLI alone makes the eager sampler write into the feedback buffer (gemma4 generator.py ~1150, ~1281).
- **Stage B, greedy only.** At prompt_enhancer.py ~752/760, set `reload = step == 0 or not use_feedback`. About 1 ms/step more, ~0.19 s per rewrite. Serving at temperature 0.7 doesn't get it until temperature > 0 with feedback is validated.
- **Stage C, lagged readback: defer.** Gain ≤ ~0.1 s per rewrite, and it is risky:
  - The sampled token tensor *is* the feedback buffer, so token N must be read before step N+1 is dispatched.
  - The `read_from_device=False` return shape differs from the current `out, _ =` unpacking.
- **Stale test.** `test_ltx_prompt_enhancer.py:308-310` still asserts `sampling_params is None`, which predates `848e97b`. Fix it before Stage A.
- **Unmeasured:** DRAM headroom for device PLI's ~615 MB/chip with LTX resident.
- **Scale of the win.** Against a ~30 s request, Stage A is ~2%. At the 512-token serving cap the rewrite is ~7.7 s even with Stage A.

**Phase 4 (device PLI for prefill).** Feasible (Python only, prefill stays eager, so the bug doesn't apply). **Not recommended.**
- The whole 976-token prefill is already ~0.24 s after `4e6eb1f` (jobs 128320, 128881). The old 0.61 s figure is stale.
- Do system-prompt KV prefix reuse (PROMPT_ENHANCER_HANDOFF.md open item 5) first. It makes Phase 4 moot.
- Cheaper fallbacks, if prefix reuse is ruled out:
  - one `[1,1,S,8960]` upload plus device slices, replacing the 35 per-layer `from_torch` calls;
  - a per-prefix PLI cache.
- If Phase 4 is ever done, `compute_device_pli` only handles the `[1,1,35,256]` decode shape. Prefill needs its own column-slice chain.
- First experiment, if any: time the host PLI share of eager prefill at S=1024/2048 on b02u08. Also explain the 0.669 s prefill outlier in job 128880.

## Deadlock: root cause found, fix validated (2026-10-12)

**Cause.** Persistent buffers allocated after a trace was captured got overwritten each time the trace replayed. The E2B decode trace was captured at the *start* of LTX warmup. The whole LTX warmup then allocated under it: in 128486 the tracker flagged 3507 buffers alive at the first E2B replay, including 1786 weight uploads and 80 global semaphores. The E2B replays wrote over some of them. The next LTX op that used a clobbered semaphore waited forever, which is the deadlock. Attempt 1 of the 10-06 A/B was the same bug corrupting data instead of a semaphore.

**Fix (`03c33ae`, opt-in `LTX_ENHANCER_LATE_CAPTURE=1`).**
- The enhancer warms eagerly, for both prefill lengths, at the start of warmup.
- It captures its decode trace at the *end* of `warmup_buffers`, after every LTX persistent buffer exists.
- The cmd file is `metal-galaxy-ltx-late-capture.sh`.

**Job 130120 (bh-glx-120-b03u02): PASSED.**
- 3 generations ran with `LTX_TRACED=1` and a traced enhancer. This is the 128486 config, which hung after gen 0 stage 1.
- Latents are finite and the rewrites are coherent.
- Buffers the tracker flagged as alive at each E2B replay:

  | | Count |
  |---|---|
  | Gen 0 | **1** (vs 3507 before; the eager prefill's logits `Slice`) |
  | Gen 1 | 23, allocated when LTX gen 0 captured its DiT/VAE traces after the E2B capture (`to_device`, `allocate_tensor_on_device`, `Copy`, `Clone`) |
  | Gen 2 | 4 (E2B's own prefill outputs) |

- Total: 107 unique buffers over the run (enhancer 6, LTX 101).

**Still open.**
- **(a) Gen-1 leftovers.** The 23 buffers allocated during LTX's own trace captures are still alive at E2B replays. They didn't hang this run, but they are the same class of hazard. Analysis (read-only workflow, 2026-10-12):
  - 22 of the 23 are LTX and 1 is the enhancer. The tracker flags anything allocated while a trace exists; it does no address check.
  - The LTX tensors (prompts, timesteps, Tracer I/O) are rewritten before every read, so they are not at risk.
  - The real residual risk is **program-cache kernel binaries** compiled after the E2B capture (`Copy` for held inputs; the enhancer's per-prompt-length `Slice` / `PagedFillCache` / lm_head slice). They are written once and re-read on every launch. They are allocated top-down in DRAM, so overlap with E2B's bottom-up scratch is unlikely but unproven.
  - **Fix A (`393bc9a`):** seed the DiT held inputs and compile their traced `Copy` programs in warmup.
  - **`2238a4b`:** the enhancer opts out of gemma4's sliding-tail stash, whose slice compiled once per prompt length.
  - **Still flagged:** Tracer I/O, VAE/vocoder/BWE `Clone`/`_update_input` copies, encoder copies, and the enhancer's lm_head and `PagedFillCache` per-length programs. The complete fix (a traced LTX pass before the E2B capture) would invert the hazard onto E2B's capture-time buffers, so it needs its own tracker run.
- **(a) Validation run: job 130195** (b04u08, code at `630eb95`: late capture + Fix A + stash opt-out, tracker on) **PASSED** 3/3 generations. Unit tests: 39 passed. Buffers flagged alive at each E2B replay, vs 130120:

  | | 130120 | 130195 | What's left |
  |---|---|---|---|
  | Gen 0 | 1 | **0** | the stash slice is gone |
  | Gen 1 | 23 | **13** | LTX Tracer I/O (DiT outputs, VAE/vocoder/BWE in and out), all rewritten before read, plus 3 Tracer prep-run `Clone` binaries, used only during capture |
  | Gen 2 | 4 | **3** | the enhancer's `PagedFillCache` ×2 and lm_head `Slice`, which are per-prompt-length programs |

  - Unique unsafe buffers: 94 (was 107).
  - The only remaining flagged buffers that get re-read are the enhancer's per-length programs. The read-only analysis gives fixes for them: fill the full padded tensor behind an enhancer flag, and a compile-only lm_head slice sweep in warmup. Both are medium risk and need an output-equality check.
  - Warmup took 1778 s on b04u08 because that node compiles kernels about 6× slower: the same ~4350 kernels took about 15,200–15,700 s in total in both its runs (130193 and 130195), against about 2,540 s on b03u02 and b07u02. It is not a code regression.
- **(b) Rewrite latency: resolved, it was the tracker.** Job 130196 (b07u02, late capture, tracker off, 2026-10-12) PASSED: prompt enhance took 3.7 / 2.8 / 3.2 s and total compute was 50.0 / 11.8 / 10.0 s for gens 0/1/2 (gen 0 includes the LTX trace captures). Warmup took 876 s; the late E2B capture took 29.8 s.
- **(c) Control: the hang reproduces with the knob off.** Job 130193 (bh-glx-120-b04u08, 2026-10-12) ran the same config as 130120 with `LTX_ENHANCER_LATE_CAPTURE` unset, on code at `2238a4b`. The tracker flagged 3550 buffers alive at the gen-0 E2B replay (130120 had 1). Stage 1 finished (34.2 s), and then the device timed out at the same point as 128486, on a third node. Only the knob differs between the two runs, so the late capture is what fixes the hang.
- **(d) Teardown noise: fixed in `2238a4b`.** The E2B Generator's `__del__` released the decode trace after the submesh had closed. `release_traces()` now unloads the enhancer while the mesh is open. 130196 logged no TT_THROW.
- **(e) Done in `7818a12`:** late capture is now the default. `LTX_ENHANCER_LATE_CAPTURE=0` restores the old order; the control cmd `metal-galaxy-ltx-serving-check.sh` pins it to 0.

**Hardware.** In bh_wan_machines, b02u08, b03u08, b04u02 and b05u02 all show only 24/32 chips *before* reset (`Issuing USER_RESET on 24 devices`), so the failures are not caused by our jobs (130110, 130111, 130153, 130154, 130157). They need infra. b03u02 and b04u08 are healthy (32/32), and so is b07u02 in `bh_sc5_B2B9_D12`.

## Open bug (history; root cause above)

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
