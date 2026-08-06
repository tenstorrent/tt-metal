# Audio decode — how to run the benchmarks

Everything below was run on **bh-glx-110-c09u14** (32 Blackhole chips, mesh 1x1 for these tests).
`tt-smi -r` is forbidden on this cluster (STATE.md) — use `tt-smi -glx_reset`, and reset after every
kill.

## Environment

```bash
cd /data/rshirvani/tt-metal
source python_env/bin/activate
export TT_METAL_HOME=/data/rshirvani/tt-metal
export MINIMAX_H3_DIFFUSERS_DIR=/data/cglagovich/MiniMax-H3-diffusers
export MINIMAX_H3_MODEL_PATH=/data/cglagovich/MiniMax-H3-diffusers
export TT_DIT_CACHE_DIR=/data/kevinmi/tt_dit_cache   # unset degrades SILENTLY: 713 s instead of ~64 s
```

To run the modified code without touching your checkout, prepend the worktree. `ttnn` resolves
through a meta-path finder pinned to the main checkout, so only the model code is overridden and no
rebuild is needed:

```bash
export PYTHONPATH=/data/rshirvani/tt-metal/.claude/worktrees/audio-kernels
```

## 1. Wall-clock decode time — the headline number

```bash
timeout 2400 python -m pytest \
  models/tt_dit/tests/models/minimax_h3/test_performance_vae_minimax_h3.py \
  -k audio_decoder_durations -s --timeout 2100
```

Logs `PERF audio_decode_5s / _10s / _15s`. This is the number to move. Measured on this branch at the
default: **1.286 / 2.482 / 3.929 s**. Beware run-to-run spread — STATE.md am. 82 records ±8 % at
identical shape and seed, so a single pair of runs cannot resolve a small change.

## 1b. Bit-exactness of a routing change

`/home/rshirvani/.claude/jobs/00644216/tmp/bitexact.py`. Compares two env configurations
output-to-output at the production batch (B=2) rather than against a golden, which is what catches a
path returning a *different* answer instead of a less precise one.

```bash
python bitexact.py
```

Run this at **B=2**. The `s5_up` C=16 defect that killed the L1 mode does not reproduce at B=1, where
the two routes agree to four significant figures of rel_rmse.

Note: never pipe a device run to `tail` — buffering hides hangs (STATE.md).

## 2. Per-shape FIR sweep — which path each filter takes, and what it costs

`/home/rshirvani/.claude/jobs/00644216/tmp/fir_bench.py` (copy it somewhere durable). Walks the five
production depthwise-FIR shapes and reports rel_rmse against a float64 golden plus median wall time
for each of: L1 conv1d, DRAM conv1d, operand-split conv1d, and the exact MAC form.

```bash
python fir_bench.py
```

Needs no weights and no diffusers — it builds the kaiser taps directly, so it is the fast loop for
anything touching `depthwise_tap_filter`.

## 3. End-to-end accuracy of a change

`/home/rshirvani/.claude/jobs/00644216/tmp/decode_accuracy.py`. Decodes the real 207-latent clip under
several env configurations, each in its own subprocess, and scores them against the decoder's own
`MINIMAX_H3_AUDIO_ACCURATE=1` output.

```bash
python decode_accuracy.py
```

The golden is a proxy, not a reference: STATE.md am. 113 measures `ACCURATE=1` at 0.45 % rel RMSE
against torch, ~25x better than the default path's 10.46 %, so it resolves a change in the default
path without being exact itself. Use it to detect a *regression*, not to certify absolute accuracy.

## 4. Tracy profile — where the time goes

```bash
timeout 1800 python -m tracy -p -r -v --op-support-count 40000 -m pytest \
  models/tt_dit/tests/models/minimax_h3/test_performance_vae_minimax_h3.py \
  -k tracy_audio_decode -s --timeout 900 &> tracy_audio.log

tt-perf-report --print-signposts <csv>          # ALWAYS run this first
tt-perf-report --start-signpost start --end-signpost stop --group-by category <csv>
```

`--op-support-count 40000` is required: at 8000 the capture fails, and at any lower-but-passing value
it silently truncates — which is what produced the bogus "1680 ops / 224 ms" figure in am. 64. Without
an explicit signpost range the tool takes the *last* signpost and reports "No device operations found".
Unset `TTNN_CONFIG_PATH` and do not combine with `TT_METAL_WATCHER`.

## 5. Correctness gates

These three need no diffusers and are the ones that gate `depthwise_tap_filter`:

```bash
timeout 2400 python -m pytest \
  models/tt_dit/tests/models/minimax_h3/test_audio_vae_minimax_h3.py \
  -k "depthwise_mac or tap_matmul or operand_split" -s --timeout 2100
```

The end-to-end PSNR gates (`test_decode`, `test_encode`, `test_roundtrip`,
`test_audio_baselines_and_roundtrip`) are **currently unrunnable**: the installed diffusers is 0.38.0
and no longer exports `AutoencoderKLMiniMaxH3Audio`, which the H3 weights want at `0.36.0.dev0`.
Restoring them needs that class back, ideally in a venv of its own rather than in `python_env`.

## Env knobs for the audio path

| var | values | default | effect |
|---|---|---|---|
| `MINIMAX_H3_AUDIO_CONV1D_L1` | off / safe / aggressive | **off** | new: L1_FULL routing for the depthwise FIR. `safe` is accuracy-neutral; `aggressive` is faster but costs 16 dB PSNR — see the docstring. |
| `MINIMAX_H3_AUDIO_DEPTHWISE_SPLIT` | off / weight / full | off | new: operand-split the FIR. Measured a bad trade — see the docstring. |
| `MINIMAX_H3_AUDIO_DEPTHWISE_MAC` | 0 / 1 | 0 | exact FIR, 2–26x slower per call. |
| `MINIMAX_H3_AUDIO_ACCURATE` | 0 / 1 | 0 | turns on MAC + conv split + tap matmul together. |
| `MINIMAX_H3_AUDIO_CONV_SPLIT` | off / weight / full | off | operand split for the conv3d paths. |
| `MINIMAX_H3_AUDIO_TAP_MATMUL` | 0 / 1 | 0 | stride-1 convs as shifted matmuls. |
