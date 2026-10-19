#!/bin/bash
# Rebuild host libraries for mmanzoor/gemma4-sdpa-small-chunk-sweep (base main 6ba9d172b63) on bh-glx-120-c04u02.
cd /home/mmanzoor/compiler/tt-metal
source python_env/bin/activate
ninja -C build && ninja -C build install
