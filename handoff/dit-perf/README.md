# DiT perf matrix: Wan2.2 and MiniMax-H3

One script times Wan2.2 T2V/I2V (720p) and MiniMax-H3 T2VA/FL2VA/Ref2VA (15 s; Ref2VA with `image`,
`video_with_sound`, `mixed` references) on any Blackhole or Wormhole Galaxy, scores the clips, and builds a
side-by-side page. Same TDP on every machine; the script refuses to run if the chips don't read it back.

| File | Job |
|---|---|
| `run_matrix.py` | `probe`, `plan`, `run`, `score`, `setup-scoring`, `check-tdp` |
| `score_clips.py` | VBench + CLIP + frame sanity on the clips a run kept (called by `run_matrix.py score`) |
| `make_report.py` | `results.json` from one or more machines to one HTML page |

## Procedure on a new machine

1. **Check out and build.** Use a branch with the Bin6 fixes if the chip has a DRAM bank harvested
   (`jameslee/dit-bin6-perf`); on a full-DRAM chip plain main is enough. `./build_metal.sh -c`. The first run
   on a build tree compiles kernels for 35-50 minutes; later runs take about 10 minutes per case.
2. **Weights.** `MINIMAX_H3_MODEL_PATH=/mnt/weka/model-weights/video/MiniMaxAI/MiniMax-H3`,
   `HF_HOME=/mnt/weka/model-weights/video/huggingface` with `HF_HUB_OFFLINE=1` (the Wan weights come from
   there; the per-user HF cache is empty stubs), `TT_DIT_CACHE_DIR=~/.cache/tt-dit`.
3. **Scoring deps, once.** `python run_matrix.py setup-scoring` installs vbench, decord and an `ffmpeg`/`unzip`
   shim into `~/dit-perf-pydeps` and `~/dit-perf-bin`. Then make `~/dit-perf-pydeps-run` containing a symlink to
   `~/dit-perf-pydeps/imageio_ffmpeg` (Wan's clip export needs it on the test's own path).
4. **Ref2VA video reference, if the usual file is missing.** The test wants a 5 s clip with sound at
   `REF2VA_VIDEO_FILE` (default under `/data/DC-deploy/...`). Without it `video_with_sound` and `mixed` SKIP.
   Cut one from any H3 clip: `ffmpeg -i t2va_15s.mp4 -t 5 -c:v libx264 -pix_fmt yuv420p -c:a aac ref5s.mp4`.
5. **Look first.** `python run_matrix.py probe` (machine, chip count, DRAM banks), `plan` (resolved test ids),
   `check-tdp --tdp 130` (opens the devices and requires every chip to read 130 W).
6. **Run it detached** (it takes hours; use `setsid nohup`, never a foreground shell):

   ```bash
   cd <checkout>
   source <checkout or tt-metal>/python_env/bin/activate
   export HF_HOME=/mnt/weka/model-weights/video/huggingface HF_HUB_OFFLINE=1 TT_DIT_CACHE_DIR=$HOME/.cache/tt-dit
   export MINIMAX_H3_MODEL_PATH=/mnt/weka/model-weights/video/MiniMaxAI/MiniMax-H3
   export REF2VA_VIDEO_FILE=$HOME/ref5s.mp4          # step 4
   export REF2VA_PRESET_DEFAULTS=1                   # only on a DRAM-harvested chip, see below
   OUT=~/dit-perf-runs/$(hostname)-$(date +%m%d)
   python handoff/dit-perf/run_matrix.py run --out $OUT --keep-going
   python handoff/dit-perf/run_matrix.py score --out $OUT
   ```

   Each case gets its own `tt-smi` reset, a wall-clock limit, and a TDP check from the runtime log; progress is
   in `$OUT/matrix.log`, per-case logs in `$OUT/<case>/pytest.log`, results in `$OUT/results.json`. `--only
   wan-t2v,h3-fl2va` runs a subset.
7. **Build the page.** `python handoff/dit-perf/make_report.py --run "Machine A=$OUT_A" --run "Machine B=$OUT_B"
   --out report.html` (use `--pending "Machine B"` for a machine not run yet, `--note` for extra lines). Publish
   with the Artifact tool.

## Statuses

`PASS`; `PERF_LIMIT` (finished and measured, but a Wan stage time fell outside the limits the Wan test uses for
a full-DRAM device, which a DRAM-harvested chip can trip; the time is real); `INVALID_TDP` (chips did not read
the requested limit); `SKIPPED` (a test precondition is missing, e.g. the reference clip); `FAIL`; `HUNG`.

## Bin6 (DRAM-harvested) specifics

* H3 needs #60190's endpoint commit; without it the video is noise. Check with a contact sheet or the CLIP score
  (a clean fox clip scores about 36; noise about 20).
* Wan needs the DRAM-harvest VAE chunk (`vae_t_chunk_size_dram_harvested`) or the VAE decode runs out of DRAM.
* H3 Ref2VA: the test normally forces traced, bucketed denoise, which does not fit 7 banks at 15 s. Set
  `REF2VA_PRESET_DEFAULTS=1` to use the 4x8 preset's own mode. A full-DRAM comparison run should leave it
  unset, so note which mode each column used.
* Wan I2V enables DBCache by default and T2V does not, so I2V denoise is about half of T2V's.

## Pitfalls seen

* `pkill -f` / `pgrep -f` with a pattern that appears in your own command line kills or matches your own shell.
  Find PIDs by `ps -eo pid,comm` instead.
* Reset (`tt-smi -glx_reset`) after killing any run, before the next.
* VBench runs on CPU, about 30 minutes per 15 s clip. Score after the timed runs, not during, or the host load
  skews the timings.
* Timed numbers are one warm generation; weight loading and kernel compile are excluded.
