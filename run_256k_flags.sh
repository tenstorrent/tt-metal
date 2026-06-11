#!/bin/bash
cd /home/tt-admin/ssinghal/qwen36/new/tt-metal
export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) HF_MODEL=Qwen/Qwen3.6-27B MESH_DEVICE=BH_GLX
export QWEN36_FORCE_SWITCH_DECODE=1 QWEN36_DECODE_L1_RESIDUAL=1 QWEN36_RESIDUAL_BUF_BF16=1 QWEN36_LM_HEAD_PLAIN_DECODE=1
export QWEN36_SEQ_CORES_PER_HEAD=4 QWEN36_FULLATTN_WO_TUNED=1 QWEN36_DELTA_OP_TUNED=1 QWEN36_CCL_NUM_LINKS_DELTA=2
export QWEN36_PERF_T_PREFILL=262144 QWEN36_PREFILL_CHUNK=4096
source python_env/bin/activate
echo "=== START 256k full-flags $(date) ==="
python -m pytest --noconftest models/demos/qwen3_6_galaxy_v2/demo/text_demo_qwen36.py -v -s -k test_qwen36_demo_batch1
echo "=== EXIT $? $(date) ==="
