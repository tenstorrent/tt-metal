#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# DiffVAE regression-gate runner: runs the `diffvae_gate`-marked tests strict (a skip = a fail),
# records each gate's PCC to a ledger, and compares it against the committed baseline.
#
#   run_diffvae_gates.sh [extra pytest args, e.g. --mesh or -k]
#
# Env:
#   LTX_CORE_SRC        path to LTX-2/packages/ltx-core/src   (required by the stage-5 gate)
#   DIFFVAE_CAPTURE     capture dump                          (required by e2e/det gates)
#   DIFFVAE_CHECKPOINT  gated bf16 VAE checkpoint             (defaults to ~/.cache/...)
#   DIFFVAE_GATES_STRICT  1 (default) -> skip becomes fail;  0 -> allow skips
#   DIFFVAE_GATES_TOL     max allowed PCC drop vs baseline    (default 0.0002)
#   RECORD_BASELINE       1 -> record this run as the new baseline instead of comparing
set -uo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$ROOT"
HERE="models/tt_dit/tests/models/vae"
LEDGER="$ROOT/generated/diffvae_gates.jsonl"
BASELINE="$HERE/diffvae_gate_baseline.json"

export DIFFVAE_GATES_STRICT="${DIFFVAE_GATES_STRICT:-1}"
export DIFFVAE_GATES_LEDGER="$LEDGER"
export PYTHONPATH="${LTX_CORE_SRC:+$LTX_CORE_SRC:}$ROOT:${PYTHONPATH:-}"
mkdir -p "$(dirname "$LEDGER")"; : > "$LEDGER"   # fresh ledger each run

# Preflight: ltx_core's module-level importorskip skips at *collection*, before the strict hook
# can turn it into a failure — so guard it here.
if [ "$DIFFVAE_GATES_STRICT" = "1" ]; then
  python -c "import ltx_core" 2>/dev/null || {
    echo "GATE PRECONDITION FAILED: ltx_core not importable — set LTX_CORE_SRC to LTX-2/packages/ltx-core/src"
    exit 2
  }
fi

python -m pytest "$HERE" -m diffvae_gate -p no:cacheprovider "$@"
pytest_rc=$?

if [ "${RECORD_BASELINE:-0}" = "1" ]; then
  python "$HERE/diffvae_gate_compare.py" "$LEDGER" "$BASELINE" --record
  exit $pytest_rc
fi

python "$HERE/diffvae_gate_compare.py" "$LEDGER" "$BASELINE" --tol "${DIFFVAE_GATES_TOL:-0.0002}"
compare_rc=$?

[ $pytest_rc -ne 0 ] && exit $pytest_rc
exit $compare_rc
