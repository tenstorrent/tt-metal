#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
repo_root=$(git rev-parse --show-toplevel)
model_root="$repo_root/models/autoports/ornith_ai_ornith_1_5_9b"
art="$model_root/doc/functional_decoder"
# This symlink changes only a profiler source-location ID; the kernel bytes are identical.
python - "$repo_root" "$art" <<'PY'
import os
import sys
from pathlib import Path
repo, art = map(Path, sys.argv[1:])
kernel = Path('ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/device/kernels/dataflow/reader_chunk_gdn_prep.cpp')
for source in [kernel, *[kernel.parent / name for name in ('chunk_gdn_fused_map.hpp', 'chunk_gdn_fused_rt_args.hpp', 'chunk_gdn_handoff.hpp', 'chunk_gdn_gate_read.hpp')]]:
    target = art / 'profiler_kernel_overlay' / source
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.symlink_to(os.path.relpath(repo / source, target.parent))
    assert target.read_bytes() == (repo / source).read_bytes()
PY
cd "$model_root"
if [[ $# -eq 0 ]]; then set -- 0 3; fi
for layer in "$@"; do
    kind=linear
    if [[ "$layer" == 3 ]]; then kind=full; fi
    echo "START profile $kind"
    timeout 1800 env TRACY_NO_WEB_SERVER=1 TT_METAL_HOME="$repo_root" \
        TT_METAL_KERNEL_PATH="$art/profiler_kernel_overlay" \
        PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}" \
        python -m tracy -r -p -v -o "$art/tracy/$kind" \
        -m models.autoports.ornith_ai_ornith_1_5_9b.tests.validate \
        --layer "$layer" --lengths 1024 --profile --output "$art/${kind}_profile.json" \
        > "$art/${kind}_profile.log" 2>&1
    test -s "$art/${kind}_profile.json"
    rg -q "^VALIDATION_PASS" "$art/${kind}_profile.log"
    if rg -q "Traceback|TT_FATAL|TT_THROW" "$art/${kind}_profile.log"; then exit 1; fi
    echo "PASS profile $kind"
done
