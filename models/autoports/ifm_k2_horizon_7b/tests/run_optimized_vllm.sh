#!/usr/bin/env bash
# Reproducible stage-10 runner invocation. Run from the tt-metal root.
set -euo pipefail
source python_env/bin/activate
source /home/vkovacevic/k2-horizon/lane-env.sh
export HF_HUB_OFFLINE=1 TT_METAL_TRACE_ALLOC_TRACKING=1 OMP_NUM_THREADS=16
export K2_VLLM_ALLOW_HOST_SAMPLING="${K2_VLLM_ALLOW_HOST_SAMPLING:-0}"
export K2_VLLM_FORCE_HOST_SAMPLING="${K2_VLLM_FORCE_HOST_SAMPLING:-0}"
unset TT_METAL_DEVICE_PROFILER TT_METAL_PROFILER_CPP_POST_PROCESS
unset TT_METAL_PROFILER_MID_RUN_DUMP TT_METAL_PROFILER_TRACE_TRACKING
unset TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT TT_METAL_WATCHER
stage="${1:?serve, benchmark, sampling or qualitative}"
shift
args=(--stages "$stage" --model-dir models/autoports/ifm_k2_horizon_7b
      --hf-model IFM/K2-Horizon-7B --mesh-device P300x2 --max-num-seqs 32
      --max-model-len 524288 --block-size 32 --sampling-profile full
      --tt-config '{"trace_region_size":200000000,"fabric_config":"FABRIC_1D_RING","trace_mode":"all"}')
if [[ "$stage" == serve ]]; then
    args+=(--additional-server-args="--trust-remote-code --async-scheduling --max-logprobs -1 --revision 036114ce8d46c32b24c15423211069abb9c5d25e --code-revision 036114ce8d46c32b24c15423211069abb9c5d25e --tokenizer-revision 036114ce8d46c32b24c15423211069abb9c5d25e --served-model-name IFM/K2-Horizon-7B")
else
    args+=(--server-url http://localhost:8000)
fi
if [[ "$stage" == benchmark ]]; then
    args+=(--additional-benchmark-args="--tokenizer /mnt/models/huggingface/hub/models--IFM--K2-Horizon-7B/snapshots/036114ce8d46c32b24c15423211069abb9c5d25e --trust-remote-code --save-detailed --num-warmups 1")
fi
exec python -m readiness_check.run_vllm_server "${args[@]}" "$@"
