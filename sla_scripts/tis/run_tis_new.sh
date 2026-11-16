#!/bin/bash
# tt-inference-server main + self-built image (tt-metal f0a85c864bf, plugin 042a948), TP32 dev spec, WH Galaxy.
until [ -f ~/teja/runs/mmlu_tp32_v5.done ]; do sleep 30; done
pkill -f server_example_tt.py; sleep 20
tmux kill-window -t teja:vllmsrv 2>/dev/null
tt-smi -glx_reset_auto > ~/teja/runs/reset_tis_new.log 2>&1
cd ~/teja/tis-main/tt-inference-server
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
python3 run.py --model meta-llama/Llama-3.3-70B-Instruct --tt-device galaxy --workflow server --docker-server --dev-mode \
  --override-docker-image ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.26.1-f0a85c864bf744b85008be365e37de3ba187c949-042a948-llama70b \
  --host-weights-dir /home/ubuntu/teja/weights/Llama-3.3-70B-Instruct --no-auth \
  --vllm-override-args '{"no-async-scheduling": true, "enable_prompt_tokens_details": true}' > ~/teja/runs/tis_new_launch.log 2>&1
echo "LAUNCH EXIT $?" >> ~/teja/runs/tis_new_launch.log
