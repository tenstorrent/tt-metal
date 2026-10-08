#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, rules of 16:40 and 17:40: which binary_ng configurations and which opted-in kernels the Blackhole
# models' own tests reach (the merged head; no A/B). One line per binary_ng program built (EB_R3_LOG_CALLS), then the
# compute kernels the JIT built. usage: r10inv.sh <set>
cd /work
export EB_R3_LOG_CALLS=/tmp/eb_calls.txt TT_METAL_CACHE=/tmp/r10cache
rm -f $EB_R3_LOG_CALLS; mkdir -p $TT_METAL_CACHE
run() { echo "##### $(date -u +%T) $*"; timeout -s INT -k 60 ${EB_RUN_LIMIT:-2400} bash -c "$*" > /tmp/r10_last.txt 2>&1; echo "rc=$? $(grep -E 'passed|failed|error' /tmp/r10_last.txt | tail -1 | cut -c1-200)"; }
case $1 in
  gemma)
    uv pip install -q -r models/demos/gemma4/requirements.txt > /dev/null 2>&1
    export HF_MODEL=models/demos/gemma4/configs/gemma-4-E2B-it
    run pytest --timeout 600 --ignore=models/demos/gemma4/tests/unit/test_lm_head.py --ignore=models/demos/gemma4/tests/unit/test_prefill_trace_parity.py --ignore=models/demos/gemma4/tests/unit/test_prefill_trace_perf.py --ignore=models/demos/gemma4/tests/unit/test_prefill_trace_tracy_csv.py --ignore=models/demos/gemma4/tests/unit/test_batched_prefill_perf.py --ignore=models/demos/gemma4/tests/unit/test_packed_verify.py --deselect models/demos/gemma4/tests/unit/test_model.py::test_full_model -k "'not test_bridge_ and not test_full_model_parity and not test_attention_decode_paged_batched and not test_sampling_'" models/demos/gemma4/tests/unit/
    ;;
  whisper)
    export HF_HUB_OFFLINE=0 HF_HOME=/mnt/MLPerf/huggingface HF_HUB_CACHE=/mnt/MLPerf/huggingface/hub HF_DATASETS_CACHE=/tmp/huggingface/datasets
    run pytest --timeout 600 models/demos/audio/whisper/tests/test_whisper_modules.py
    ;;
  sdvae)
    export HF_HUB_OFFLINE=0
    run pytest --timeout 600 models/demos/stable_diffusion_xl_base/vae/tests/pcc --ignore=models/demos/stable_diffusion_xl_base/vae/tests/pcc/test_welford_state_leak_regression.py
    ;;
  sdbase)
    export HF_HUB_OFFLINE=0
    run pytest --timeout 600 models/demos/stable_diffusion_xl_base/tests/pcc --ignore=models/demos/stable_diffusion_xl_base/tests/pcc/test_unet_loop.py --ignore=models/demos/stable_diffusion_xl_base/tests/pcc/test_euler_discrete_scheduler.py
    ;;
  misc)
    run pytest --timeout 300 models/demos/minimax_m3/tests/unit/ -k "'1x1 and not msa_layer'"
    run pytest --timeout 600 models/demos/gpt_oss_d_p/tests/unit/
    export HF_HOME=/mnt/MLPerf/huggingface
    HF_MODEL=PaddlePaddle/PaddleOCR-VL-1.6 TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/PaddlePaddle/PaddleOCR-VL-1.6 run pytest --timeout 170 models/demos/blackhole/paddleocr_vl/tests/test_text_decoder_pcc.py models/demos/blackhole/paddleocr_vl/tests/test_vision_permutation.py
    run pytest --timeout 600 models/experimental/janus_pro/tests/test_ci_dispatch.py
    run pytest -sv --timeout 600 models/experimental/functional_unet/tests/test_unet_perf.py -k "'test_unet_trace_perf and not test_unet_trace_perf_multi_device'"
    ;;
  qwen3)
    export HF_MODEL=Qwen/Qwen3-0.6B HF_HOME=/mnt/MLPerf/huggingface TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/Qwen/Qwen3-0.6B
    run pytest --timeout 720 models/tt_transformers/demo/simple_text_demo.py -k "performance-ci-1" --sampling_params "'{\"temperature\": 1.0, \"top_k\": 1, \"top_p\": 0.5}'"
    ;;
  llama8b)
    export TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=10000 HF_MODEL=meta-llama/Llama-3.1-8B-Instruct TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/meta-llama/Llama-3.1-8B-Instruct
    run pytest --timeout 600 models/tt_transformers/tests/test_device_perf.py -k "prefill-llama3_8b-2-131072-2-2-1-1-False"
    run pytest --timeout 600 models/tt_transformers/tests/test_device_perf.py -k "decode-llama3_8b-2-131072-2-10-1-1-False"
    ;;
esac
echo "##### EB_CALL configs (count, line)"
[[ -f $EB_R3_LOG_CALLS ]] && sort $EB_R3_LOG_CALLS | uniq -c | sort -rn | cut -c1-600
echo "##### compiled compute kernels"
find $TT_METAL_CACHE -name "*.elf" 2>/dev/null | sed -n 's#.*/kernels/\([^/]*\)/.*#\1#p' | sort | uniq -c | sort -rn | head -400
echo "##### end"
