#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
export TT_MODEL_BRINGUP_ROOT=/home/mvasiljevic/qwen38-full-rerun/codex-home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.4
export TT_AUTODEBUG_ROOT=/home/mvasiljevic/qwen38-full-rerun/codex-home/plugins/cache/tenstorrent-skills/tt-autodebug/0.1.5
BRINGUP_EXPORTS=$(python_env/bin/python "$TT_MODEL_BRINGUP_ROOT/scripts/environment.py")
eval "$BRINGUP_EXPORTS"
export PYTHONPATH=.:$PYTHONPATH
multichip_name=$1
shift
multichip_doc=models/autoports/qwen_qwen3_8_27b/doc/multichip_decoder
printf '%q ' "$0" "$multichip_name" "$@" >> "$multichip_doc/commands.log"
printf '\n' >> "$multichip_doc/commands.log"
mkdir -p "$multichip_doc/sources"
multichip_hash=$(sha256sum models/autoports/qwen_qwen3_8_27b/tt/multichip_decoder.py | cut -d ' ' -f 1)
cp models/autoports/qwen_qwen3_8_27b/tt/multichip_decoder.py "$multichip_doc/sources/$multichip_hash.py.txt"
printf '%s\n' "$multichip_hash" > "$multichip_doc/$multichip_name.source.sha256"
timeout -k 10 900 python_env/bin/python models/autoports/qwen_qwen3_8_27b/tests/run_multichip_decoder.py --output "$multichip_doc/$multichip_name.json" "$@" > "$multichip_doc/$multichip_name.log" 2>&1
