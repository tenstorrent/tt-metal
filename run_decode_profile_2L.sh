#!/bin/bash
cd /home/tt-admin/ssinghal/qwen36/new/tt-metal
export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) HF_MODEL=Qwen/Qwen3.6-27B MESH_DEVICE=BH_GLX
export QWEN36_FORCE_SWITCH_DECODE=1 QWEN36_DECODE_L1_RESIDUAL=1 QWEN36_RESIDUAL_BUF_BF16=1 QWEN36_LM_HEAD_PLAIN_DECODE=1
export QWEN36_SEQ_CORES_PER_HEAD=4 QWEN36_FULLATTN_WO_TUNED=1 QWEN36_DELTA_OP_TUNED=1 QWEN36_CCL_NUM_LINKS_DELTA=2
# 2-layer model: layer0 = GDN (linear), layer1 = full-attention -> isolate 1 of each type
export QWEN36_PATTERN_OVERRIDE=linear,full QWEN36_N_LAYERS=2 QWEN36_GEN_DECODE_STEPS=6 QWEN36_PERF_T_PREFILL=128
export QWEN36_TEMP=0.5 QWEN36_TOP_K=50 QWEN36_TOP_P=0.95
source python_env/bin/activate
echo "=== START decode profile 2L [GDN,FA] $(date) ==="
python -m tracy -p -v -r -m --op-support-count 20000 pytest --noconftest \
    models/demos/qwen3_6_galaxy_v2/tests/test_decode_generator_profile.py -s \
    -k test_decode_generator_profile_traced
echo "=== EXIT $? $(date) ==="
