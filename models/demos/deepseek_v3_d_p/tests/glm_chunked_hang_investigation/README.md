# GLM chunked-prefill layer-77 hang — repro tooling

See `FINDINGS.md` first for the full writeup. This directory has the
scripts used to hunt the hang; they're plain bash, no special tooling
required, meant to be run from a normal terminal (tmux/screen recommended
since a single iteration takes ~20-30 minutes).

## Prerequisites

- A Blackhole Galaxy box with the 8x4 mesh config this test targets.
- GLM-5.2 weights/cache/trace reachable at the default
  `/mnt/models/deepseek-prefill-cache/...` paths (same paths CI's own
  workflow sets — override via `GLM52_HF_MODEL`,
  `TT_GLM52_PREFILL_TTNN_CACHE`, `PREFILL_TRACE_DIR` env vars if your box
  differs).
- Ideally a genuinely high-power (`>=130W` TDP) box, so
  `test_prefill_transformer_chunked.py`'s `is_high_power()` `skipif` guard
  passes without needing to touch the test file. Check with `tt-smi -s`
  (look for `TDP_LIMIT_MAX` under `smbus_telem`, hex value >= `0x82` ==
  130).

## Usage

Terminal 1 — run the loop:

```
cd models/demos/deepseek_v3_d_p/tests/glm_chunked_hang_investigation
./run_loop.sh            # 15 iterations, default range 1..15
# or, to also reset the device before every single iteration:
./run_loop_reset_each.sh # default range 16..30 (keeps numbering distinct
                         # from run_loop.sh's logs in the same LOGDIR)
```

Terminal 2 — watch for a stall, in parallel:

```
cd models/demos/deepseek_v3_d_p/tests/glm_chunked_hang_investigation
./poller_loop.sh
```

The poller stays completely silent until it sees 5 minutes of no log
growth on the *current* iteration while the process is still alive. When
that happens it will, in order: run `tools/tt-triage.py -vv` and wait for
that output to actually land on disk, *then* kill the hung test process,
*then* run `tt-smi -glx_reset`. It never disturbs a live process before
triage is captured — that's the whole point.

If you run both `run_loop.sh` and `run_loop_reset_each.sh` back-to-back
against the same `LOGDIR`, restart `poller_loop.sh` for the second one
with a marker argument that only the second loop's completion line
contains (e.g. `./poller_loop.sh 'NO_HANG_AFTER_30_ITERATIONS'`) —
otherwise it will match the first loop's leftover completion line in
`summary.log` and report "no hang found" immediately without ever having
watched the second loop run. (This bit us during the original
investigation — see `FINDINGS.md`.)

## Where things land

Default log directory: `generated/glm_chunked_hang_repro/` (override with
`GLM_HANG_LOGDIR`). Key files:

- `summary.log` — high-level progress (iteration start/finish/rc, tails
  of any failure)
- `run_<N>.log` — full raw output of iteration N
- `current_run.info` — machine-readable pointer the poller reads
- `triage_pid<PID>_<timestamp>.log` — `tt-triage.py -vv` output, written
  once a hang is confirmed
- `glx_reset_after_hang_<timestamp>.log` — the recovery `tt-smi -glx_reset`
  output, written after triage

## Cleanup

Both loop scripts and the poller are plain background-friendly bash — if
you back them with `nohup ... & disown` (recommended so a closed terminal
doesn't kill them), stop them with:

```
pkill -f run_loop.sh; pkill -f run_loop_reset_each.sh; pkill -f poller_loop.sh
pkill -9 -f test_glm_prefill_transformer_chunked_no_pcc   # if a hung test needs killing
tt-smi -glx_reset                                          # after killing anything fabric-related
```
