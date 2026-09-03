#!/usr/bin/env bash
# Quasar pool channel-width sweep runner — same craq-sim env as run_qpool_sim.sh, but with a
# timeout budget sized for the full C sweep in one pytest session (per-case banners in the log
# identify the case in flight if it hangs).
set -euo pipefail

# =============================== CONFIG — edit me ===============================
SIM_SO="$HOME/sim/qsr/libttsim.so"
TIMEOUT_S=2400
# =================================================================================

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)

if [[ ! -f "$SIM_SO" ]]; then
    echo "QPOOL-SWEEP: no sim library at $SIM_SO — see run_qpool_sim.sh for build instructions." >&2
    exit 1
fi
if [[ ! -f "$(dirname "$SIM_SO")/soc_descriptor.yaml" ]]; then
    cp "$REPO_ROOT/tt_metal/soc_descriptors/quasar_32_arch.yaml" "$(dirname "$SIM_SO")/soc_descriptor.yaml"
fi

export TT_METAL_SIMULATOR="$SIM_SO"
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_FORCE_JIT_COMPILE=1
unset TT_METAL_LLK_ASSERTS TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS 2>/dev/null || true

cd "$REPO_ROOT"
set +e
timeout --foreground "$TIMEOUT_S" pytest -q -s "$SCRIPT_DIR/test_qpool_sweep.py" "$@"
rc=$?
set -e
if [[ $rc -eq 124 ]]; then
    echo "QPOOL-SWEEP: TIMED OUT after ${TIMEOUT_S}s — the last 'QPOOL-SWEEP: C=...' banner names the hung case."
fi
exit $rc
