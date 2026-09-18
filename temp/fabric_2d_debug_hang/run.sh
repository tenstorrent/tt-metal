#!/usr/bin/env bash
# Run from your tt-metal checkout root. Keep the bundle files together.
set -euo pipefail
bundle="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
case "${1:-}" in
  control) name=Debug2DControl; seconds=120 ;;
  hang)    name=Debug2DHang;    seconds=0 ;;
  *) echo 'Usage: bash path/to/fabric_2d_debug_hang/run.sh control|hang' >&2; exit 2 ;;
esac
exe=./build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric
[[ -x "$exe" ]] || { echo 'Run from a tt-metal checkout with test_tt_fabric built.' >&2; exit 2; }
kernel=tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp
[[ -f "$kernel" ]] || { echo 'Router source not found in this checkout.' >&2; exit 2; }
if [[ "$name" == Debug2DHang ]] && ! grep -q 'BEGIN TEST_ONLY_2D_RECEIVER_STALL' "$kernel"; then
  echo 'Apply patch_receiver.py --apply first. Without the patch, this is ordinary 2D traffic.' >&2
  exit 2
fi
export TT_METAL_RUNTIME_ROOT="$PWD"
export TT_METAL_KERNEL_PATH="$PWD"
export TT_METAL_FORCE_JIT_COMPILE=1
export TT_METAL_OPERATION_TIMEOUT_SECONDS="$seconds"
exec "$exe" \
  --test_config "$bundle/test_fabric_2d_debug_hang.yaml" \
  --filter "name.$name" \
  --master-seed 1 \
  --dump-built-tests \
  --show-workers
