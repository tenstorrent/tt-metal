<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
SPDX-License-Identifier: Apache-2.0
-->
# Qwen3 decode with sampling on the L2CPU (x280) cores

Demonstration of the l2cpu components on a real model: Qwen3 (8B and 0.6B, tt_transformers model code, unmodified)
with token sampling moved from the host or the Tensix cores onto the SiFive x280 harts of L2CPU tile 0, or, at
batch 32, of tiles 0-3 sharing the batch (`--tiles`).
Results: `RESULTS.md`.

Two loops:
- **Host-mediated ("Plan A")**: `decode_harness.py`. Per step the host runs the decode trace, pushes the logits
  rows into the firmware's zone, bumps the request and reads the tokens back (`l2cpu_sampler.py`); the same harness
  runs the reference samplers (`dh_samplers.py`: host C library = bit-exact reference, torch argmax).
- **Device-resident ("Plan B")**: `plan_b.py`. One trace per step = [decode forward with logits on device, untilize,
  push (+ doorbell), wait]; the x280 writes the next token into the trace's token input in place; the host only
  enqueues `execute_trace` and drains the output ring. Streaming at batch > 1 (push and x280 overlap).
  With `--tiles N` (2 or 4, batch > 1) the trace is [forward, untilize, split push (one core per tile, each ringing
  its tile first), wait for every tile]: tile i serves users 32i/N .. 32(i+1)/N - 1 from its own region (tile 3's
  in a second buffer: tiles 2 and 3 share DRAM bank 7) and writes their tokens into the one token tensor.

Requires the bring-up component (`tools/l2cpu`), the Tensix link (`tools/l2cpu/tensix`) and the sampling library +
firmware (`tools/l2cpu/sampling`); `deps.py` is the single place where this example binds to them. The firmware's
arena embeds the link block (`l2cpu_link.h`) at a fixed offset; `l2cpu_ops.py` passes `arena + offset` to the link
kernels and the logits zones as offsets from it.

## Commands (one process per command from a fresh chip reset, watcher off)

    source tools/l2cpu/scripts/l2cpu_env.sh      # TT_METAL_HOME, PYTHONPATH (l2cpu package), PY
    # every device command below runs as: tools/l2cpu/scripts/l2cpu_run.sh "<reason>" $PY <script> <args>

    $PY tools/l2cpu/examples/qwen3/run_plan_b.py --model Qwen/Qwen3-8B --batch 1 --max-new-tokens 256 --temperature 0.7 --top-k 50 --top-p 0.9 --seed 1234
    $PY tools/l2cpu/examples/qwen3/run_plan_b.py --model Qwen/Qwen3-8B --batch 32 --max-new-tokens 256 --per-user-mix [--tiles 4]
    $PY tools/l2cpu/examples/qwen3/accept_plan_b.py --model Qwen/Qwen3-8B --batch 1 --prompts 5 --steps 256
    $PY tools/l2cpu/examples/qwen3/accept_plan_b.py --model Qwen/Qwen3-8B --batch 32 --steps 256 [--tiles 2|4] [--negative-control [--neg-user U] | --repeat 20]
    $PY tools/l2cpu/examples/qwen3/accept_plan_b.py --model Qwen/Qwen3-8B --batch 32 --steps 256 --retry-on-timeout 16 --inject-timeout-at 100 [--tiles 4 --inject-tile 1]
    tools/l2cpu/examples/qwen3/bench_row.sh --arm x280-planB --batch 32 --setting T0.7k50p0.9 --planb plan_b:planb_bench [--tiles 4]
    $PY tools/l2cpu/examples/qwen3/bench.py --table

Environment: `TT_METAL_HOME` / `PYTHONPATH` of the tt-metal tree; on a multi-chip board one chip
(`TT_VISIBLE_DEVICES`, mesh graph descriptor of a single-chip system). `TT_CACHE_PATH` or `L2CPU_QWEN3_CACHE`
(weights cache root, default `~/.cache/tt-l2cpu-qwen3`), `L2CPU_QWEN3_OUT` (results, default `./l2cpu_qwen3_out`),
`L2S_FW_IMAGE` (sampling firmware image, default `tools/l2cpu/fw/build/bh-irq-sampling/fw.bin`, built by
`make -C tools/l2cpu/sampling`), `X280_PLANB_STREAMED=0|1` (default: on at batch > 1), `X280_PLANB_WAIT=all|each`
(several tiles: one wait kernel polling every tile, default, or one wait program per tile), `L2CPU_WAIT_TIMEOUT_US`
(wait-op bound, default 50000), `L2CPU_MAX_STEP_MS` (host bound per step for the ring drain, default 200).

## Conventions
- Step / user / seed: the prefill token is sampling step 0 on the host (C library); decode step s is firmware
  request `step_seq_base + s + 1`, sampled with step s + 1 and user index b (row b); per-user seeds and settings in
  the firmware's parameter block (`sampling_mix.py` for the mixed batch: user u -> setting u % 4, seed 1234 + u).
  The reference (host C library in the harness) uses the same indices, so the token lists are comparable 1:1.
- Several tiles: tile i holds its users as local 0..n-1 (rows, params, output ring) with `user_base` = its first
  global user; the draw and the token write use the global index, so the tokens of a split run equal the one-tile
  run and the same host reference. Batch 1 always runs on tile 0 (the one-tile trace, unchanged).
- No stop conditions (by design): generation continues past EOS; the demo cuts the displayed text.

## Waits, bounds and failure paths
| Where | Bound | At the bound |
|---|---|---|
| wait op (`tensix/kernels/l2cpu_wait.cpp`) | `L2CPU_WAIT_TIMEOUT_US` (50 ms) wall clock | writes `0xDEAD0000 \| req` to the wait status word, returns; the trace continues |
| several tiles: wait op (`l2cpu_wait_all.cpp`, or one `l2cpu_wait.cpp` per tile) | one 50 ms bound for all tiles (per tile with `each`) | writes the status word of every tile that is not done; the tiles that answered keep serving the next steps |
| host ring drain (`PlanB.run`) | n x (`L2CPU_MAX_STEP_MS` + wait bound) + 5 s; also stops early when a wait status word is set (any tile); counts the steps every tile published | raises `StepTimeout` |
| firmware stream wait (streamed push) | firmware `timeout_us` (0 = 100 ms) | hart parks with STREAM_TIMEOUT, nothing published -> wait op bound |
| monitor thread (`l2cpu_monitor.py`, 100 ms, one per tile) | heartbeat unchanged 2 s, firmware error word, wait status word | dumps hart states + log ring, `os._exit(17)` (no device close: queued waits drain on their own; next run resets the chip); with `exit_on_fail=False` it only records (retry mode) |
| `--retry-on-timeout CHUNK` | chunk of steps between host checks | keeps the chunk's good tokens (steps every tile published), marks nothing in flight, warm-restarts the firmware of every active tile (bring-up control: mailbox park, then RNMI), rewrites the inputs of the failed step and its step index, retries; the retried tokens equal the reference |
| host waits for firmware READY / mailbox | 5 s / 1 s | exception |

## Limits
Verified on one P300 (chip 0, L2CPU tile 0; tiles 0-3 for `--tiles`, 110 Tensix cores) with Qwen3-8B and Qwen3-0.6B, batch 1 and 32 only,
up to 256 generated tokens with 128-token prompts (max_seq_len 1024 is not checked during a run). The L2CPU clock
runs only while the process holds the chip. See `RESULTS.md` for the deviations of the comparison.
