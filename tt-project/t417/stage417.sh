#!/bin/bash
# t417: stage source (ttp/t409 head) and weights at /data/smarton/fasth3/t417 on exabox-login. No build, no device.
set -eo pipefail
F=/data/smarton/fasth3; T=$F/t417; W=$T/src; M=$T/models
BR=ttp/t409-port-vsa-sparse-attention-into-the-fasth; REV=${1:-1d8e15320e3}
trap 'echo "STAGE417_RC=$?" > $T/stage417.done' EXIT
rm -f $T/stage417.done; mkdir -p $T/home/.cache $M
export HOME=$T/home HF_HOME=$T/home/.cache/huggingface HF_HUB_DISABLE_TELEMETRY=1
step() { echo "[s417] $(date -u '+%F %T') $*"; }
step source
cd $F/t48
git fetch --filter=blob:none origin "+refs/heads/$BR:refs/remotes/origin/$BR"
[ -d $W/.git ] || [ -f $W/.git ] || git worktree add --detach $W "$REV"
cd $W; git checkout --detach "$REV"
for n in tracy umd tt-cluster-descriptors; do git submodule update --init --depth 1 -- tt_metal/third_party/$n; done
step "source at $(git rev-parse HEAD)"
HF=$F/python_env/bin/hf
# hf 1.16.1 exits non-zero (click Exit(0) traceback) even after a good download: judge by the files instead.
dl() { local rc=0; nice -n 10 ionice -c3 timeout 14400 $HF download "$@" || rc=$?; echo "[s417] hf rc=$rc"; [ $rc -ne 124 ]; }
need() { [ -s "$1" ] || { echo "[s417] missing $1"; exit 3; }; }
step lora-v1.0
dl lightx2v/Minimax-h3-Turbo minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors --local-dir $M/lightx2v-h3-turbo
need $M/lightx2v-h3-turbo/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors
step lora-sla
dl lightx2v/Minimax-h3-Turbo-SLA minimax_h3_fl2v_turbo_4step_v0.1_768p_sla_bf16.safetensors --local-dir $M/lightx2v-h3-turbo-sla
need $M/lightx2v-h3-turbo-sla/minimax_h3_fl2v_turbo_4step_v0.1_768p_sla_bf16.safetensors
step base
dl MiniMaxAI/MiniMax-H3 --local-dir $M/MiniMax-H3
# -print -quit: with pipefail, grep -q closing the pipe early makes find die of SIGPIPE (rc 141).
[ -n "$(find $M/MiniMax-H3 -name "*.safetensors" -not -path "*/.cache/*" -print -quit)" ] || { echo "[s417] no safetensors in MiniMax-H3"; exit 3; }
touch $M/MiniMax-H3/READ_OK
step "done; $(du -sh $M | cut -f1)"
