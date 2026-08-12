# Audio decode — how to run the benchmarks

> **Note:** some `audio_perf/` scripts cited below were removed once their conclusions were
> captured here and in `ITEM1_RESULT.md` / `ITEM2_RESULT.md`. Recover any of them with
> `git log -- models/tt_dit/tests/models/minimax_h3/audio_perf`. See `README.md` for what survives.

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

## Current profile — start the next round of fusion here

`PROFILE_2026_08_06.txt`, produced by `analyze_csv.py` over the signposted Tracy window on the code as
committed. **1401 ms device FW over 6955 ops.** `tt-perf-report` is not installed on this box, so
`analyze_csv.py` groups the CSV directly (it also finds the signpost rows itself, which is the step
that silently returns "No device operations found" when done wrong).

| role | ms | % | made of |
|---|---|---|---|
| **FIR scaffolding** | **~514** | **37** | Untilize 138.9, PaddedSlice 112.3, ReshapeView 110.5, Slice 53.2, Halo 39.8, I2S+S2I 36.4, SliceWrite 19.1, Move 4.8 |
| **Concat** | **285.3** | **20** | 469 calls: replicate/zero T-pad, polyphase merge, channel-align pad, C-chunk reassembly |
| Conv3d | 210.9 | 15 | 136 calls, the AMP resblocks |
| Snake (ternary) | 140.3 | 10 | was 235.9 before the tile-fold |
| **FIR compute (Conv2d)** | **81.3** | **6** | 870 calls -- the actual convolution |
| BinaryNg / Permute | 138.1 | 10 | residual adds; 57 BCT<->BTC permutes |

The wrapper-to-work ratio on the FIR is still ~6:1, and concat is now the single largest op. Neither
is arithmetic. Both die if `Activation1d` becomes one fused band that keeps the 2x-upsampled tensor in
L1 -- today it is written to DRAM and re-read about ten times.

## Levers closed by measurement — do not re-walk these

Each of these looked promising and was killed by a number. The scripts that killed them are in this
directory, so any of them can be re-opened with evidence rather than argument.

| lever | verdict | measurement |
|---|---|---|
| **Trace** | dead | vocoder 1.2214 s untraced vs 1.2191 s traced. Not dispatch-bound. |
| **conv1d L1_FULL** | worth ~nothing | `verify` mode: 1 of 42 production shapes fits L1. The 1.2–2.3x from a per-shape sweep was at B=1; production is B=2 and longer. Also returns a *wrong answer* at C=16 (tile width — padding C to 32 is bit-exact). |
| **Depthwise operand split** | harmful | plateaus ~6e-04 at 2.5–6x cost; after the SFPU fix the unsplit path is 5.4e-08, so splitting is strictly worse. |
| **Algebraic band fusion** | exact, neutral | rel_rmse 8.5e-08, but removing the 2x tensor doubles the band's op count. e2e inside the noise. |
| **L1-sharded intermediates** | impossible | `to_layout(TILE)`, `snake_beta`, `add` all reject sharded tensors; `concat` accepts and spills to DRAM. |
| **Cheaper replicate pad** | ~2–6 % at best | `ttnn.pad` beats the 12-piece concat 2.1x at s4 but only 1.11x at s6 — the cost is the full-tensor copy, which no primitive avoids, and `pad` is zeros so replicate needs a correction that adds the ops back. |

The common thread: every one of them is limited by a full-tensor DRAM round trip that op-level work
cannot remove. Padding, interleaving and the activation's layout changes are all free *inside* a
kernel that has the rows in L1 already, and all unavoidable outside one. That is the case for K5, and
it is now made by measurement rather than by inference.

## Env knobs for the audio path

| var | values | default | effect |
|---|---|---|---|
| `MINIMAX_H3_AUDIO_CONV1D_L1` | off / safe / aggressive | **off** | new: L1_FULL routing for the depthwise FIR. `safe` is accuracy-neutral; `aggressive` is faster but costs 16 dB PSNR — see the docstring. |
| `MINIMAX_H3_AUDIO_DEPTHWISE_SPLIT` | off / weight / full | off | new: operand-split the FIR. Measured a bad trade — see the docstring. |
| `MINIMAX_H3_AUDIO_DEPTHWISE_MAC` | 0 / 1 | 0 | exact FIR, 2–26x slower per call. |
| `MINIMAX_H3_AUDIO_ACCURATE` | 0 / 1 | 0 | turns on MAC + conv split + tap matmul together. |
| `MINIMAX_H3_AUDIO_CONV_SPLIT` | off / weight / full | off | operand split for the conv3d paths. |
| `MINIMAX_H3_AUDIO_TAP_MATMUL` | 0 / 1 | 0 | stride-1 convs as shifted matmuls. |
