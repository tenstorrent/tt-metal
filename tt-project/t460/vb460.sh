#!/bin/bash
# t460 VBench (g15blx02, CPU, no device): integrated (1080p device upscale) as candidate vs t397 (768p)
# upscaled to 1080p on the host (ffmpeg bicubic, crf 8; the harness needs equal frame sizes)
# as reference on the five 6 s seeds, with the ltx_eval harness tree from t421 (state/t421/evaltree).
set -eo pipefail
D=/home/smarton/fasth3/tt-metal/tt-project/state/t460
cd /home/smarton/fasth3/tt-metal/tt-project/state/t421/evaltree
LTX_EVAL_THREADS=8 timeout 5400 nice -n 10 /home/smarton/fasth3/tt-metal/python_env/bin/python -m models.tt_dit.tests.models.ltx.tools.ltx_eval \
  batch --cand-dir $D/vb/int --ref-dir $D/vb/base --out $D/vbench --vbench-ref --jobs 2
