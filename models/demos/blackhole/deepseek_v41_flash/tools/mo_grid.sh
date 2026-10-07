#!/bin/bash
# usage (run ON the host): mo_grid.sh <codeW> <logname> <sessions>   e.g. "isl4k_b16@a,isl4k_b16@c,isl4k_b16@a"
# MoE-collective-overlap paired A/B in ONE process with the grid's exact env (no MEMLOG, SPEC=0, 40 layers, build slots/stagger); the model is built once with the
# split shared-expert weights (DSV41_MO_OVERLAP=prep) and each scenario applies its mode (env switch + new chunk-trace capture):
#   a = baseline (batched router off, linear topology, overlap off)   b = batched own-row router   c = b + shared expert || dispatch (sub-devices, segmented trace)
#   d = c + ring topology   e = ring topology only   f = overlap only (slices router)
W=$1; N=$2; SS=$3; M=/mnt/tt-data/ssinghal/tests/tt-metal; H=$(hostname -s | tr -dc 0-9 | tail -c 2)
exec flock -w 14400 /tmp/dsv4_dev.lock bash -c "
for v in \$(compgen -e | grep -E '^(DSV41_|MOE_COMPUTE_|TT_METAL_)'); do unset \$v; done
cd $M && source python_env/bin/activate
export MOE_COMPUTE_FP32_ACC=1 MOE_COMPUTE_BFP8_WEIGHTS=1 TT_METAL_CACHE=/mnt/tt-data/ssinghal/tt-metal-cache/h$H TT_METAL_HOME=$M PYTHONPATH=$W:$M
export DSV41_LAYERS=${LAYERS:-0-39} DSV41_ENGRAM_RAM=1 DSV41_BUILD_SLOTS=10 DSV41_BUILD_STAGGER_S=${STAG:-480} DSV41_SPEC=0 DSV41_SESSION=$SS DSV41_MO_OVERLAP=prep
export DSV41_MODE_A=DSV41_MO_OVERLAP=prep,DSV41_UNI_ROUTER=slices,DSV41_UNI_TOPO=linear
export DSV41_MODE_B=DSV41_MO_OVERLAP=prep,DSV41_UNI_ROUTER=batched,DSV41_UNI_TOPO=linear
export DSV41_MODE_C=DSV41_MO_OVERLAP=1,DSV41_UNI_ROUTER=batched,DSV41_UNI_TOPO=linear
export DSV41_MODE_D=DSV41_MO_OVERLAP=1,DSV41_UNI_ROUTER=batched,DSV41_UNI_TOPO=ring
export DSV41_MODE_E=DSV41_MO_OVERLAP=prep,DSV41_UNI_ROUTER=slices,DSV41_UNI_TOPO=ring
export DSV41_MODE_F=DSV41_MO_OVERLAP=1,DSV41_UNI_ROUTER=slices,DSV41_UNI_TOPO=linear
echo MOGRID_ENV \$(hostname -s) code=$W commit=\$(git -C $W rev-parse --short=11 HEAD); env | grep -E '^(DSV41_|MOE_COMPUTE_|TT_METAL_|PYTHONPATH)' | sort | sed 's/^/MOGRID_ENV /'
timeout 43200 pytest -x -s -q -o junit_suite_name=mogrid_$N $W/models/demos/blackhole/deepseek_v41_flash/demo/text_demo.py -k session"
