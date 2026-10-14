#!/bin/bash
# Entry point of the eval CLIENT container (run_gsm8k_eval.sh mounts it at /entry.sh).
# run.py asks `git rev-parse HEAD` of its own tree; the tree in the image is an empty `git init` owned by another uid,
# so trust it and give it a commit (the tree is world-writable), then run run.py with the image's own python env.
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0='*'
cd /home/container_app_user/tt-inference-server
if ! git rev-parse HEAD >/dev/null 2>&1; then
  git add -A -f >/dev/null 2>&1
  git -c user.name=image -c user.email=image@local commit -q -m "tt-inference-server (image tree)" >/dev/null 2>&1
fi
exec /home/container_app_user/tt-metal/python_env/bin/python run.py "$@"
