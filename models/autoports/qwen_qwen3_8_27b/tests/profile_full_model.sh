#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
export TT_MESH_PASS_THROUGH_THREAD_POOL=${TT_MESH_PASS_THROUGH_THREAD_POOL:-1}
export PYTHONPATH=.:$PYTHONPATH
profile_name=$1
shift
profile_doc=models/autoports/qwen_qwen3_8_27b/doc/full_model
trap 'profile_status=$?; printf "%s\n" "$profile_status" > "$profile_doc/$profile_name.exit_status"' EXIT
printf '%q ' "$0" "$profile_name" "$@" >> "$profile_doc/commands.log"
printf '\n' >> "$profile_doc/commands.log"
sha256sum models/autoports/qwen_qwen3_8_27b/tt/*.py models/autoports/qwen_qwen3_8_27b/tests/run_full_model.py > "$profile_doc/$profile_name.source.sha256"
python_env/bin/python -m tracy -r -p -v -o "$profile_doc/tracy/$profile_name" \
 -m models.autoports.qwen_qwen3_8_27b.tests.run_full_model \
 --profile --length 128 --generate 2 --output "$profile_doc/$profile_name.json" "$@" > "$profile_doc/$profile_name.log" 2>&1
