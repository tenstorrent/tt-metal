#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, sixth review check 2: the multi-chip Blackhole models' binary op census. Each ttnn add, subtract and
# multiply call site with its operands (EB_SITE, eb_census_plugin.py) and each binary_ng program built (EB_CALL), from the
# models' own CI tests on QuietBox 2, LoudBox and Galaxy runners. usage: r13inv.sh <set>
cd /work
export EB_R3_LOG_CALLS=/tmp/eb_calls.txt TT_METAL_CACHE=/tmp/r13cache PYTHONPATH=/work:/work/tests/eb_r3_ci:${PYTHONPATH:-}
rm -f $EB_R3_LOG_CALLS; mkdir -p $TT_METAL_CACHE
run() { echo "##### $(date -u +%T) $*"; echo "EB_RUN $*" >> $EB_R3_LOG_CALLS; timeout -s INT -k 60 ${EB_RUN_LIMIT:-3600} bash -c "$*" > /tmp/r13_last.txt 2>&1; rc=$?; echo "rc=$rc $(grep -E 'passed|failed|error' /tmp/r13_last.txt | tail -1 | cut -c1-200)"; [[ $rc != 0 ]] && grep -E "^E  |Error|error:|Exception" /tmp/r13_last.txt | grep -v "digest\|teardown" | sort | uniq -c | sort -rn | head -12 | cut -c1-300; }
P="-p eb_census_plugin -p no:cacheprovider"
# any cached snapshot of a hub model (the models pin revisions that the runner's cache may not hold)
snap() { local d; for h in ${HF_HOME:-} /mnt/MLPerf/huggingface /mnt/models/huggingface; do d=$(ls -d $h/hub/models--$1/snapshots/* 2>/dev/null | head -1); [[ -n $d ]] && { echo $d; return; }; done; }
diag() { echo "##### diag: $(env | grep -i -E '^hf_|huggingface' | tr '\n' ' ')"; ls /mnt 2>&1 | head; for h in /mnt/MLPerf/huggingface/hub /mnt/models/huggingface/hub; do echo "== $h"; ls $h 2>/dev/null | grep -i -E "gemma-4|llama-3.1-8b|qwen3.8|qwen3.6|qwen3-32b" ; done; }
case $1 in
  qb2gemma)
    diag
    uv pip install -q -r models/demos/gemma4/requirements.txt > /dev/null 2>&1
    export MESH_DEVICE=P300x2 EXTRA_MODELS_DIR=$PWD/models/demos TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0
    unset HF_MODEL; export HF_MODEL=$(snap google--gemma-4-31B-it); echo "HF_MODEL=$HF_MODEL"
    run pytest $P --timeout 900 models/demos/gemma4_31b_qb2/tests/test_decoder.py -k "'1025-2 or 1-13'"
    run pytest $P --timeout 900 models/demos/gemma4_31b_qb2/tests/test_chunked_prefill.py
    ;;
  qb2llama)
    diag
    export TT_LLAMA_TEXT_VER=llama31_8b_qb2 MESH_DEVICE=P300x2
    export LLAMA_MODEL_PATH=$(snap meta-llama--Llama-3.1-8B-Instruct); echo "LLAMA_MODEL_PATH=$LLAMA_MODEL_PATH"
    run pytest $P --timeout 300 models/demos/llama31_8b_qb2/tests
    ;;
  qb2qwen38)
    diag
    export MESH_DEVICE=P300x2 MODEL_WEIGHTS_DIR=$(snap Qwen--Qwen3.8-27B); echo "MODEL_WEIGHTS_DIR=$MODEL_WEIGHTS_DIR"
    run pytest $P --timeout 600 models/demos/qwen38_27b_qb2/tests/unit models/demos/qwen38_27b_qb2/tests/test_decode_conv.py
    ;;
  qb2qwen36)
    export HF_HOME=/mnt/MLPerf/huggingface HF_MODEL=Qwen/Qwen3.6-27B TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/Qwen/Qwen3.6-27B
    run pytest $P --timeout 300 models/demos/blackhole/qwen36/tests/test_attention_tp.py models/demos/blackhole/qwen36/tests/test_mlp_tp.py
    run pytest $P --timeout 300 models/demos/blackhole/qwen36/tests/test_gdn_tp.py -k "'not test_gdn_tp_prefill'"
    run pytest $P --timeout 600 models/demos/blackhole/qwen36/tests/test_model_tp.py -k "'not test_model_tp_decode_batched and not test_model_tp_long_prefill_traced'"
    run pytest $P --timeout 300 models/demos/blackhole/qwen36/tests/test_mtp_tp.py models/demos/blackhole/qwen36/tests/test_vision_block.py models/demos/blackhole/qwen36/tests/test_fused_recurrent_gdn.py -k "'not perf'"
    export HF_MODEL=Qwen/Qwen3.6-35B-A3B TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/Qwen/Qwen3.6-35B-A3B MESH_DEVICE=P150x4
    run pytest $P --timeout 300 models/demos/blackhole/qwen36/tests/test_moe_tp.py models/demos/blackhole/qwen36/tests/unit/test_moe.py
    ;;
  qb2dsdp)
    run pytest $P --tb=short models/demos/deepseek_v3_d_p/tests/pcc/ -k "'not matmul_pcc and not perf-host-64 and ((fabric2d-mesh-2x2 or fabric2d-2x2 or torus-x-1x4) or not (moe_gate_prefill2d or moe_routing_setup or ttnn_moe)) and (not hca or (flash and chunk5120-varying))'" --timeout 1800
    ;;
  lbdsdp)
    export DEEPSEEK_V3_HF_MODEL=/mnt/MLPerf/huggingface/hub/models--deepseek-ai--DeepSeek-R1-0528
    run pytest $P --tb=short models/demos/deepseek_v3_d_p/tests/pcc/ -k "'not matmul_pcc and not perf-host-64 and ((fabric2d-mesh-4x2 or fabric2d-mesh-2x4 or fabric2d-2x4) or not (moe_gate_prefill2d or moe_routing_setup or ttnn_moe)) and not (ttnn_moe and kimi and perf) and not pad50 and (not hca or (flash and chunk5120-varying))'" --timeout 1800
    run pytest $P --tb=short models/demos/deepseek_v3_d_p/tests/kda/layer/test_acceptance.py::test_synthetic_kimi_k3_accuracy_and_determinism -k SP2xTP4
    run pytest $P --tb=short models/demos/deepseek_v3_d_p/tests/test_prefill_block_loop.py -k "'fabric2d-mesh-2x4-2link and layer0 and device'"
    run pytest $P --tb=short models/demos/deepseek_v3_d_p/tests/attn_res/
    ;;
  glxqwen)
    export HF_HOME=/mnt/MLPerf/huggingface HF_MODEL=Qwen/Qwen3-32B TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/Qwen/Qwen3-32B QWEN_BH_PREFETCHER=1
    run pytest $P --timeout 900 models/demos/llama3_70b_galaxy/demo/text_qwen_demo.py -k ci-token-matching
    run pytest $P --timeout 900 models/demos/llama3_70b_galaxy/tests/unit_tests/test_qwen_decoder.py models/demos/llama3_70b_galaxy/tests/unit_tests/test_qwen_decoder_prefill.py
    ;;
esac
echo "##### EB census lines (first occurrence order)"
[[ -f $EB_R3_LOG_CALLS ]] && awk '!seen[$0]++' $EB_R3_LOG_CALLS | cut -c1-700
echo "##### end"
