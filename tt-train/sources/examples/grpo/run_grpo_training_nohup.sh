#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../" && pwd)"
nohup python tt-train/sources/examples/grpo/grpo_training.py &
echo "PID: $!"
