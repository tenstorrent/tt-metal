#!/bin/bash
source /home/rsalman/ltx2-ref-venv/bin/activate
cd /home/rsalman/tt-metal
OMP_NUM_THREADS=32 python ltx_exp1/ref_driver.py --prompt "A close-up of a woman singing softly into a vintage microphone" --seed 10 --frames 17 --fps 25 --height 256 --width 384 --threads 32 --decode-video --out ltx_exp1/ref_smoke_17f_256x384
