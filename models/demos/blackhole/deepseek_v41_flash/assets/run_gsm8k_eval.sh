#!/bin/bash
# Run the GSM8K evaluation (dsv41_gsm8k_cot, 8-shot chain of thought, chat template, temperature 1.0 / top-p 0.95, all 1319 questions)
# of tt-inference-server's `evals` workflow against a RUNNING DeepSeek-V4.1-Flash server, from a second container of the same image.
# See GETTING_STARTED.md, section "Run the GSM8K evaluation".
#
# usage:  IMAGE=<image> CKPT=<checkpoint dir> SPEC=<runtime spec json> STATE=<scratch dir> ./run_gsm8k_eval.sh
# optional: URL (default http://127.0.0.1:8000), CONCURRENCY (default 16 = the 16-user server), HF_CACHE (default $STATE/eval/hf), DOCKER (default docker)
# extra arguments are passed to run.py, e.g. `./run_gsm8k_eval.sh --limit-samples-mode smoke-test` (8 questions) for a quick check
set -euo pipefail
: "${IMAGE:?set IMAGE}" "${CKPT:?set CKPT}" "${SPEC:?set SPEC}" "${STATE:?set STATE}"
URL=${URL:-http://127.0.0.1:8000}; CONCURRENCY=${CONCURRENCY:-16}; DOCKER=${DOCKER:-docker}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
E=$STATE/eval; HF_CACHE=${HF_CACHE:-$E/hf}
mkdir -p $E/home $E/cache_root $E/logs $E/model_specs $E/workflow_logs $E/workflow_venvs $HF_CACHE
# eval_config.py names the tokenizer by this path (the checkpoint's site path): mount the checkpoint there
CKPT_IN=/mnt/tt-data/ssinghal/deepseek-v41-flash
curl -sf -o /dev/null "$URL/health" || { echo "no server answering $URL/health"; exit 1; }
H=/home/container_app_user
$DOCKER run --rm --network host --user $(id -u):$(id -g) --entrypoint /entry.sh \
  -v $HERE/eval_client_entry.sh:/entry.sh:ro \
  -e HOME=$H/h -e UV_PYTHON_INSTALL_DIR=$H/h/uv-python \
  -e HF_TOKEN=${HF_TOKEN:-local-checkpoint-unused} -e HF_HOME=/hf -e MODEL_SPECS_ENV=dev \
  -e DSV41_EVAL_TASKS=dsv41_gsm8k_cot -e DSV41_EVAL_CONCURRENCY=$CONCURRENCY \
  -v $E/home:$H/h -v $E/cache_root:$H/cache_root -v $E/logs:$H/logs -v $E/model_specs:$H/model_specs \
  -v $E/workflow_logs:$H/tt-inference-server/workflow_logs -v $E/workflow_venvs:$H/tt-inference-server/.workflow_venvs \
  -v $HF_CACHE:/hf \
  -v $CKPT:$CKPT_IN:ro -v /etc/passwd:/etc/passwd:ro \
  -v $SPEC:/specs/spec.json:ro \
  -w $H/tt-inference-server \
  $IMAGE \
  --model DeepSeek-V4.1-Flash --tt-device blackhole_galaxy --dev-mode \
  --tt-metal-home $H/tt-metal --tt-metal-python-venv-dir $H/tt-metal/python_env \
  --no-auth --skip-system-sw-validation --host-weights-dir $CKPT_IN \
  --server-url $URL --service-port 8000 \
  --workflow evals --runtime-model-spec-json /specs/spec.json "$@"
echo "reports: $E/workflow_logs/reports_output/evals/"
