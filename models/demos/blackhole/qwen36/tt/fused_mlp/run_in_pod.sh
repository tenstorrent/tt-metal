#!/bin/bash
# usage: run_fm.sh TAG script.py [ENV=VAL ...]   (inside the d27u08 dev pod)
TAG=$1; S=$2; shift 2
export TT_METAL_HOME=/home/container_app_user/tt-metal PYTHONPATH=/home/container_app_user/tt-metal:/home/container_app_user/tt-metal/ttnn ARCH_NAME=blackhole TT_METAL_INSPECTOR_RPC=0
cd /tmp/r16/fm && env "$@" timeout 1800 $TT_METAL_HOME/python_env/bin/python /tmp/r16/fm/$S > /tmp/r16/fm/$TAG.log 2>&1; echo "rc=$?"
grep -E '^\[(fm8|fm2|sil)\]|Error|error|TT_FATAL|TT_THROW' /tmp/r16/fm/$TAG.log | grep -v '^\[fm8\]   x landing' | tail -12
