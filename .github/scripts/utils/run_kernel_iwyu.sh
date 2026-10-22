#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# The ci-test image ships IWYU 0.24 with Clang 20. Its driver replays
# every database entry in its own cwd, including repeated wrapper paths.
# --check_also reaches the #included kernel sources and project headers;
# --keep preserves the JIT's deliberate implementation-file includes.
# Default exit status is zero for recommendations, nonzero for errors.
include-what-you-use --version > /work/kernel_tidy/iwyu-version.txt
status=0
iwyu_tool.py -p /work/kernel_tidy -j 8 -- \
  -Xiwyu "--check_also=$PWD/*" \
  -Xiwyu '--check_also=*/tt_metal/*' \
  -Xiwyu '--check_also=*/ttnn/*' \
  -Xiwyu '--check_also=*/tt-metal-cache/*' \
  -Xiwyu '--keep=*.cpp' \
  -Xiwyu '--keep=*.cc' \
  > /work/kernel_tidy/iwyu.txt 2>&1 || status=$?
echo "$status" > /work/kernel_tidy/iwyu-exit-code.txt
{
  echo '### Kernel Include What You Use (non-blocking)'
  echo ''
  cat /work/kernel_tidy/iwyu-version.txt
  echo ''
  echo "Analyzer exit code: $status (0 means analysis completed, not that includes are clean)."
  echo 'Recommendations and parse errors: `iwyu.txt` in the per-leg kernel-clang-tidy artifact.'
  echo 'Advice is per captured configuration; review across roles before applying changes.'
} >> "$GITHUB_STEP_SUMMARY"
if [ "$status" -ne 0 ]; then
  echo "::warning::IWYU reported analysis failures; see iwyu.txt in the capture artifact."
fi
exit "$status"
