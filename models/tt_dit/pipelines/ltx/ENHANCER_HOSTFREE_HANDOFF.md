# Enhancer host-free decode: hand-off for the login-node move

2026-10-13. Branch `rsalman-e2b-device-pli` @ `45f27e82016`, pushed. Tree `/data/rsalman/tt-metal-pli`.
Earlier history (device PLI phases, the LTX deadlock and its fix) is in `DEVICE_PLI_HANDOFF.md` next to this file.

**Goal (from the user):** make the Gemma-4-E2B prompt enhancer for LTX-2.3 distilled as host-free as
possible, the same way the rest of the video model runs on device. Work independently and use workflows.

## Why we are moving

`slurm-login` stopped accepting jobs. Its root disk is 100% full, and the Slurm controller runs on that
host (`StateSaveLocation=/var/lib/slurm/slurmctld`). Every `sbatch` fails with
`I/O error writing script/environment to file`, on any partition or node, including held jobs.
- Ruled out: inodes (23% used), a read-only mount, and a downed controller (it is up).
- Usage, from a read-only scan: 103 GB in `/home` (333 users, a long tail), 25 GB of Docker images
  (17 GB reclaimable: djstefanovic's untagged `bringup-ui` builds), 18.5 GB in `/tmp` (dmadic 5 GB,
  nkapre 3 GB, dgolubovic 2.7 GB). Our own share is about 1 GB.

**On the new login node, check first:** `/data` is mounted, `tt-sbatch` is on PATH, and a held probe submits.
```
sbatch -H -p bh_sc5_B2B9_D12 --wrap hostname   # then scancel <id>
```

## State

| Item | Where |
|---|---|
| Code | `rsalman-e2b-device-pli` @ `45f27e82016` (pushed). Dev worktree `/data/rsalman/tt-metal-pli-dev`, branch `rsalman-enh-hostfree-dev`, at the same commit |
| Plan (mapping and design workflow output, judge-corrected) | `/data/rsalman/scratch/hostfree_plan.json`, key `merged`, experiments E0–E11 |
| Baseline result | job 130695, `/data/rsalman/jobs/run/130695/enhancer_exp/enhancer_bench/base.json` |
| Next job's cmd file | `/data/rsalman/jobs/cmds/metal-galaxy-enhancer-hostfree.sh` |
| Old session transcripts | `/data/rsalman/claude-migration/` |

### Commits this round

| Commit | What |
|---|---|
| `30b6c7d` | `prompt_enhancer_experiments/enhancer_bench.py`: the enhancer alone on the traced LTX galaxy ring params (no DiT, VAE or encoder). Fixed prompt set × temperatures × repeats, per-request timings, host-call counters, token ids, `BENCH_REF` comparison. About 5 min per arm instead of 30+ min inside LTX. Also adds `DevicePromptEnhancer.last_token_ids` |
| `45f27e8` | E0–E3, every knob off by default (table below). Not yet run on hardware: the submit is what failed |

### Baseline (job 130695, b09u08, host PLI, restage every token)

| T | ms/token | new tokens | prefill | wall per rewrite |
|---|---|---|---|---|
| 0.0 | 18.4 | 183 | 0.24 s | 3.59 s |
| 0.7 | 18.4 | 189 | 0.27 s | 3.73 s |

- Per token: one `prepare_decode_inputs_host` (full restage) and one host PLI computation.
- All 12 (prompt, T) pairs were identical across repeats, so the bench is deterministic, T=0.7 included.
- Load 76 s, eager warmup 73 s, capture 31 s.

## Knobs added in `45f27e8` (all default off)

| Knob | Step | Effect |
|---|---|---|
| `LTX_ENHANCER_DEVICE_PLI` | E1 | Builds gemma4 with `GEMMA4_DEVICE_PLI` pinned to 1 (or 0 when off) in `ensure_loaded` |
| `LTX_ENHANCER_FEEDBACK` | E2 | `reload_inputs` only on decode step 0. Requires traced decode, device sampling and device PLI |
| `LTX_ENHANCER_FEEDBACK_POISON` | E2 test | Passes token 0 and position `prompt_len` from the host on feedback steps; output must not change |
| `GEMMA4_SEED_ALL_ROWS` | E2 | gemma4 `sample_decode_on_device` replicates the one active seed slot to all 128 lanes, so the four mesh rows sample the same token at T>0 |
| `LTX_ENHANCER_WARM_SAMPLED` | E3 | One short T=`self.temperature` rewrite in warmup (eager) and in capture (traced), on a forked torch RNG |
| `LTX_ENHANCER_STEP_TIMING` | E0 | Splits each step into dispatch (`read_from_device=False`) and read; medians go into `last_stats` |
| `BENCH_*` | bench | `BENCH_TEMPS`, `BENCH_REPEATS`, `BENCH_MAX_NEW`, `BENCH_IGNORE_STOP`, `BENCH_REF[_TAG]`, `BENCH_ROW_CHECK` |

Without `GEMMA4_SEED_ALL_ROWS`, mesh rows 1–3 get SKIP seeds under feedback and free-run their RNG at T=0.7.
Output is still correct, because only device 0's token is read. A judge agent found that seeding slots
0/32/64/96 does not fix this: SeedManager salts duplicate seeds, so each row would get a different stream.

## Next step: submit the host-free bench

```
tt-sbatch test --root /data/rsalman/tt-metal-pli --topo galaxy -p bh_sc5_B2B9_D12 --mode single \
  --timeout 120 --triage --test metal-galaxy-enhancer-hostfree -J enh-hostfree -- -w bh-glx-120-b09u08
```
Unit tests run first, then ten arms, each a fresh pytest process. Expect about 60–70 min.

| Arm | Ref | Pass condition |
|---|---|---|
| base | – | self-equal across repeats (re-baseline on the same node) |
| dpli | – | self-equal. Greedy may differ from base at index 2 (the known contested token); this is not a failure |
| fb | dpli | **token-for-token equal at T=0 and T=0.7**; `prepare_decode_inputs_host` = 1 per request |
| fbpoison | fb | equal |
| fbrows | fb | equal; mesh rows equal |
| fbnorows | fb | documents row divergence at T=0.7 |
| fbwarm | fbrows | equal; plus dispatch and read split |
| base-, dpli-, fb-ignorestop | dpli-ignorestop for fb | 512-token fixed-length latency at T=0.7 |

Expected: device PLI about 18.4 → 14.5 ms/token, feedback about −1 ms more (Phase 1/2 numbers from `e2b_bringup`).

## After that (plan order, details in the plan JSON)

1. **E3 in-LTX gate.** Copy `metal-galaxy-ltx-late-{capture,notracker}.sh` to `*-dpli-fb.sh` with the E1/E2/E3 knobs.
   - Pass: 3/3 generations, flagged buffers ≤ 130195's 0/13/3, no sampler binaries among the flags, DRAM headroom logged.
   - Then flip the defaults.
2. **E4: system-prompt KV prefix reuse.** The ~975-token system prompt is fixed per mode.
   - Prefill from `P_aligned` with `start_pos`; no sliding-window stash is needed (unbounded identity page table).
   - Cuts prefill to the suffix bucket. `supports_prefix_caching=False` for E2B, so it needs a PCC gate.
3. **E5: lagged readback** (`read_from_device=False` + `read_decode_output(async_read=True)` + event).
4. **E6: sample the prefill token on device.**
5. **E7–E9: toward one `execute_trace` per token.**
   - E7: device-side seed advance.
   - E8: device token history, stop check every K steps.
   - E9: sampling inside the decode trace. High risk: #48037 CCL semaphores.
6. **E10–E11:** prefill token straight into the feedback buffer; final in-LTX validation and default flips.

Still on the host after all of this: chat template, tokenize and detokenize, the prompt-ids upload, and one
readback per K tokens for the stop check. The LTX text encoder also re-tokenizes the rewrite on the host
(a separate project).

## Working rules (unchanged)

- Commit as you go with `git -c user.name="Rizwan Salman" -c user.email=rsalman@tenstorrent.com commit`,
  trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Push after milestones. No PR.
- `tt-sbatch test` for Python-only changes, `run` if the wheel changes. Always `--timeout` and `--triage`.
- One job per node. Check runs by hand, not only with a monitor.
- Jobs rsync the tree when they start. Edit in the dev worktree, then fast-forward the main tree when no job is pending.
- Env vars set at submit time do not reach the container: put knobs in the cmd file.

### Nodes

| Node | State |
|---|---|
| b09u08 | Healthy, ran 130695 |
| b03u02, b07u02 | Healthy earlier, but busy today |
| b04u08 | Compiles about 6× slower |
| b02u08, b03u08, b04u02, b05u02 | Showed 24/32 chips (b04u02 again today) |

Galaxies are mostly allocated, so queue with `-x <bad nodes>` if no node is free.
