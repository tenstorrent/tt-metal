#!/usr/bin/env bash
#
# ir_to_emit.sh — turn a TTNN IR dump (.mlir) into a readable, flat ttnn Python emit.
#
# This is the IR analogue of a tt-forge `model_ttnn.py`: it lowers the TTNN
# dialect graph through tt-mlir's TTNN->EmitPy pipeline and translates the
# result to Python. The emit is FLAT (one big trace/forward function) and its
# weights are synthetic (`ttnn.ones`) — it is a *structure/math* reference, not
# a runnable model. Use it exactly the way the forge skill uses model_ttnn.py:
# to read the op sequence, eps, activations, fusion order, and (via
# create_weights_for__main) the arg->HF-weight-key mapping.
#
# Usage:
#   ir_to_emit.sh <input.mlir> <out_prefix>
# Produces:
#   <out_prefix>.emitpy.mlir   (EmitPy-dialect IR)
#   <out_prefix>.py            (the flat ttnn Python emit)
#
# Env:
#   TTMLIR   tt-mlir checkout (default: /home/mvasiljevic/tt-mlir). Must be built.
#
set -euo pipefail

SRC="${1:?usage: ir_to_emit.sh <input.mlir> <out_prefix>}"
OUT="${2:?usage: ir_to_emit.sh <input.mlir> <out_prefix>}"
: "${TTMLIR:=/home/mvasiljevic/tt-mlir}"

OPT="$TTMLIR/build/bin/ttmlir-opt"
TR="$TTMLIR/build/bin/ttmlir-translate"
[ -x "$OPT" ] || { echo "ERROR: $OPT not found — build tt-mlir first" >&2; exit 1; }

# tt-mlir env (puts venv python on PATH, sets TTMLIR_ENV_ACTIVATED which CMake checks).
if [ -z "${TTMLIR_ENV_ACTIVATED:-}" ]; then
  # shellcheck disable=SC1091
  ( cd "$TTMLIR" && source env/activate ) >/dev/null 2>&1 || true
  export PATH="$TTMLIR/build/bin:$TTMLIR/../ttmlir-toolchain/venv/bin:/opt/ttmlir-toolchain/venv/bin:$PATH"
fi

TMP="$(mktemp --suffix=.mlir)"
trap 'rm -f "$TMP"' EXIT

# --- Known schema-skew normalizations between nightly-produced IR and the local
# --- tt-mlir build. Each is a signedness/attr fix, not a semantic change. If a
# --- new op trips the converter with a "failed to satisfy constraint: NN-bit
# --- {signed,signless} integer attribute" error, add a targeted line here
# --- (scope the sed to the op name so you do not flip attrs on other ops).
#   * ttnn.argmax `dim` must be signless i32 (nightly emits si32).
sed -E "/ttnn.argmax/ s/(dim = -?[0-9]+) : si32/\1 : i32/g" "$SRC" > "$TMP"

"$OPT" --ttnn-common-to-emitpy-pipeline "$TMP" -o "$OUT.emitpy.mlir"
"$TR" --mlir-to-python "$OUT.emitpy.mlir" > "$OUT.py"

echo "wrote $OUT.py ($(wc -l < "$OUT.py") lines) and $OUT.emitpy.mlir"
