#!/bin/bash
cd /home/tt-admin/ssinghal/qwen36/new/tt-metal
export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) HF_MODEL=Qwen/Qwen3.6-27B MESH_DEVICE=BH_GLX
export QWEN36_FORCE_SWITCH_DECODE=1 QWEN36_DECODE_L1_RESIDUAL=1 QWEN36_RESIDUAL_BUF_BF16=1 QWEN36_LM_HEAD_PLAIN_DECODE=1
export QWEN36_SEQ_CORES_PER_HEAD=4 QWEN36_FULLATTN_WO_TUNED=1 QWEN36_DELTA_OP_TUNED=1 QWEN36_CCL_NUM_LINKS_DELTA=2
# isolate the links lever: RS-async OFF (orthogonal, separately validated)
export QWEN36_DECODE_RS_ASYNC=0
# THE LEVER: BH fabric links 1 (default) vs 2 (confirmed max).
export QWEN36_GALAXY_NUM_LINKS="${QWEN36_GALAXY_NUM_LINKS:-1}"
export QWEN36_N_LAYERS=64 QWEN36_GEN_DECODE_STEPS=32 QWEN36_PERF_T_PREFILL=128
export QWEN36_TEMP=0.5 QWEN36_TOP_K=50 QWEN36_TOP_P=0.95
source python_env/bin/activate
echo "=== START 64L decode, GALAXY_NUM_LINKS=$QWEN36_GALAXY_NUM_LINKS $(date) ==="
python -m pytest --noconftest models/demos/qwen3_6_galaxy_v2/demo/text_demo_qwen36.py -v -s \
    -k test_qwen36_demo_generator_batch1
echo "=== EXIT $? $(date) ==="
