#!/bin/bash
# Experiment 1: Lightricks DistilledPipeline on CPU at the served shape (153f/25fps/1088x1920), seed 10, default test prompt.
source /home/rsalman/ltx2-ref-venv/bin/activate
cd /home/rsalman/tt-metal
OMP_NUM_THREADS=56 python ltx_exp1/ref_driver.py --prompt "$(cat ltx_exp1/prompt_default.txt)" --seed 10 --frames 153 --fps 25 --height 1088 --width 1920 --threads 56 --decode-video --out ltx_exp1/ref_153f25_seed10
