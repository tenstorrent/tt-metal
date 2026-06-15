# RECOVERY — Kimi prefill perf overnight run

If this session (or your laptop) died, everything you need is here. The investigation runs
**independently of Claude** in a tmux session on `bh-glx-d04u02`; results land on disk as they complete.

## TL;DR — what's running
- A tmux session **`kimi_perf`** runs `orchestrator.sh`, which executes experiment files from
  `queue/` one at a time on the shared 8x4 mesh, appends findings to `RESULTS.md`, and moves each
  finished `*.exp` into `done/`. When the queue is empty it idles and rescans every 30s.
- Dir: `/home/ppopovic/kimi_perf_overnight/` (this folder).
- Code instrumentation is **committed to git** on branch `ppopovic/investigation` (so it survives even
  if this folder is lost). All probes are env-gated and OFF by default.

## The question being investigated
Runner prefills a 5120-tok chunk in **~3.3 s**; the no-PCC transformer test does the same 61-layer
chunk in **~1.94 s**. Gap = **~1.4 s constant/additive per chunk**, prefix-independent. All
measurement-side hypotheses already ruled out (see `tt-metal/models/demos/deepseek_v3_d_p/tt/runners/RUNNER_PERF_INVESTIGATION.md`).
Prime suspect: `mla_seq_len` = 61440 (runner) vs 56320 (test).

## How to look at what happened
```bash
cd /home/ppopovic/kimi_perf_overnight
cat RESULTS.md              # <-- the durable findings, one section per experiment
tail -f orchestrator.log    # <-- live progress / which experiment is running
ls done/                    # experiments already completed
ls queue/                   # experiments still pending
ls -t logs/                 # raw runner/producer/test logs per experiment
```

## Attach to the live run
```bash
tmux attach -t kimi_perf      # watch it live (Ctrl-b d to detach without killing)
tmux capture-pane -t kimi_perf -p | tail -40   # snapshot without attaching
```

## Is it still alive / healthy?
```bash
tmux ls                                   # is 'kimi_perf' listed?
pgrep -af "runners.prefill_runner|runners.prefill_h2d_producer"   # current device user
tt-smi -s                                 # device health
```

## Stop it
```bash
touch /home/ppopovic/kimi_perf_overnight/STOP   # graceful: finishes current exp, then exits
# or hard: tmux kill-session -t kimi_perf  &&  the EXIT trap kills child runner/producer
```

## Add more experiments (live, while it runs)
Drop a new `NN_name.exp` into `queue/`. Format (runner example):
```
EXP_NAME=06_my_test
EXP_TYPE=runner            # or: test
EXP_DESC="one line of what/why"
RUNNER_ENV="PREFILL_FOO=1" # extra env appended to the runner baseline (last wins)
PRODUCER_ENV=""            # extra env appended to the producer baseline
# for EXP_TYPE=test instead:  TEST_ENV="PREFILL_SECTION_TIMING=1"
```
The orchestrator picks it up on its next scan (≤30s when idle).

## Experiment queue (first wave)
| # | type | what |
|---|------|------|
| 00 | runner | baseline 61440, no instrumentation — confirm ~3.3 s + no probe-code regression |
| 01 | runner | **MAX_SEQ_LEN=56320** — decisive mla_seq_len test |
| 02 | runner | 61440 + section timing + construction/input dump — localize the 1.4 s |
| 03 | test | no-PCC test + section timing + construction/input dump — fast-path localize + config diff |
| 04 | runner | SKIP_ACK_SYNC=1 — drop the 61 per-layer syncs the runner does (test does 0) |
| 05 | runner | MAX_SEQ_LEN=56320 + section timing — if 01 helped, show which section dropped |

## The env-gated probes (in committed code)
- `PREFILL_SECTION_TIMING=1` — per-chunk embed/mla/moe breakdown (`perf_probe.section`, logged via
  `[section-timing]`). Sync-brackets each section, so absolute ms inflate; compare runner-vs-test deltas.
- `PREFILL_DUMP_CONSTRUCTION=1` — one-shot `[construction-dump]` (config + `_chunked_kv_buf` spec) and
  `[input-dump]` (first forward_chunk token/cache tensor specs).
- `PREFILL_SKIP_ACK_SYNC=1` — keep the per-layer ack inject, drop its `synchronize_device` (mla.py).
- Pre-existing: `PREFILL_PREFILL_SYNC`, `PREFILL_PRESYNC`, `PREFILL_DISABLE_LAYER_ACK`.

Files instrumented: `tt/perf_probe.py` (new), `tt/tt_prefill_transformer.py`, `tt/tt_prefill_block.py`,
`tt/mla/mla.py`. All edits env-gated; default behavior unchanged.

## Gotchas (carried over)
- Mesh is shared; one runner/test at a time. The orchestrator serializes and force-cleans between runs.
- Killing a runner mid-init while another starts races the chip lock and crashes the new one — the
  orchestrator's `cleanup_procs` waits for full process death before the next launch.
- Never delete `TT_UMD_LOCK.*` shm files (normal robust-mutexes).
- DEVICE_FP32 gate kernel already built; these are Python-only changes (no rebuild needed).
