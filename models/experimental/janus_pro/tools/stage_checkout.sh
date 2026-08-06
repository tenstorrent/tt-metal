#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Put the tower's compute path at one historical commit while keeping the trace harness at HEAD,
# so every stage in PERF.md's change log is measured by the same test. Without this the early
# stages cannot be measured at all: forward_device, prepare_patches and the perf test itself
# only arrive in a72f222a6fb, part-way through the sequence.
#
#   tools/stage_checkout.sh <sha>     put the compute path at <sha>
#   tools/stage_checkout.sh --restore return everything to HEAD
#
# COMPUTE files carry the optimizations and are what a stage measurement varies.
# HARNESS files only gained device-side entry points; holding them at HEAD is what makes the
# numbers comparable. model_config.py is compute: it holds the program and kernel configs.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

J=models/experimental/janus_pro
COMPUTE=(
  "$J/tt/janus_pro_image_attention.py"
  "$J/tt/janus_pro_image_block.py"
  "$J/tt/janus_pro_image_mlp.py"
  "$J/tt/janus_pro_vision_aligner.py"
  "$J/tt/janus_pro_layernorm.py"
  "$J/tt/model_config.py"
)

if [[ "${1:-}" == "--restore" ]]; then
  git checkout HEAD -- "$J/tt/" "$J/tests/"
  echo "restored to HEAD"
  exit 0
fi

SHA="${1:?usage: stage_checkout.sh <sha> | --restore}"
git rev-parse --verify "$SHA^{commit}" >/dev/null

for f in "${COMPUTE[@]}"; do
  # A file may not exist yet at an early commit (janus_pro_layernorm.py arrives late). Leaving
  # the HEAD copy in place is harmless there, because nothing at that commit imports it.
  if git cat-file -e "$SHA:$f" 2>/dev/null; then
    git checkout "$SHA" -- "$f"
  fi
done

echo "compute path at $SHA ($(git log -1 --format=%s "$SHA" | cut -c1-60))"
echo "harness held at HEAD; run the perf command from PERF.md, then tools/perf_stage_report.py"
