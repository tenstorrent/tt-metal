#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Download the openpi pi05_base checkpoint (gs://openpi-assets/checkpoints/pi05_base,
# JAX/Orbax, public) and convert it to the torch safetensors layout this package loads,
# using openpi's own exporter. Then verify it with Pi0_5WeightLoader, link it at
# weights/pi05_base (the PCC/perf test default), and delete the JAX checkpoint.
#
# Usage (from anywhere):
#   models/experimental/pi0_5/p150/weights/download_pi05_base.sh
#
# Env knobs:
#   PI05_CACHE   work/output dir, outside the repo  (default: $HOME/pi05_cache)
#   KEEP_JAX=1   keep the downloaded JAX checkpoint (default: delete after conversion)
#
# Needs: curl, git, uv, python3; ~27 GB free disk while converting (~14.5 GB after);
# ~44 GB RAM for the CPU-only conversion.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_HOME="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
PI05_CACHE="${PI05_CACHE:-$HOME/pi05_cache}"
JAX_DIR="$PI05_CACHE/pi05_base_jax"   # exporter needs "pi05" in this path (selects the adaRMS mapping)
OUT_DIR="$PI05_CACHE/pi05_base"
OPENPI_DIR="$PI05_CACHE/openpi"
OPENPI_COMMIT=215abfb217dbac7d5f1273282331b9b1866c0479  # exporter version this was validated with
GCS_PREFIX=checkpoints/pi05_base
LINK="$SCRIPT_DIR/pi05_base"

for tool in curl git uv python3; do
    command -v "$tool" >/dev/null || { echo "ERROR: '$tool' not found on PATH" >&2; exit 1; }
done
mkdir -p "$PI05_CACHE"

# --- 1. Download the Orbax checkpoint from the public bucket (resumable: skips files of the right size)
echo "[1/5] Downloading gs://openpi-assets/$GCS_PREFIX -> $JAX_DIR"
mkdir -p "$JAX_DIR"
listing=$(curl -sfL "https://storage.googleapis.com/storage/v1/b/openpi-assets/o?prefix=$GCS_PREFIX/&fields=items(name,size)")
echo "$listing" | python3 -c "import json,sys; [print(i['name'], i['size']) for i in json.load(sys.stdin)['items']]" \
    > "$JAX_DIR/.manifest"
cd "$JAX_DIR"
# shellcheck disable=SC2016
xargs -P 8 -n 2 sh -c '
    f=${0#'"$GCS_PREFIX"'/}
    [ -f "$f" ] && [ "$(stat -c %s "$f")" = "$1" ] && exit 0
    mkdir -p "$(dirname "$f")"
    curl -sfL --retry 5 -o "$f" "https://storage.googleapis.com/openpi-assets/$0"
' < .manifest
while read -r name size; do
    f=${name#"$GCS_PREFIX"/}
    [ "$(stat -c %s "$f" 2>/dev/null || echo -1)" = "$size" ] || { echo "ERROR: size mismatch for $f" >&2; exit 1; }
done < .manifest
echo "      $(wc -l < .manifest) files, sizes verified"

# --- 2. openpi env + its transformers patch (AdaRMS). UV_LINK_MODE=copy keeps the patch
#        inside openpi's .venv — uv's default hardlink mode would also patch the shared uv cache.
echo "[2/5] Setting up openpi @ ${OPENPI_COMMIT:0:7} in $OPENPI_DIR"
if [ ! -d "$OPENPI_DIR/.git" ]; then
    git clone -q https://github.com/Physical-Intelligence/openpi.git "$OPENPI_DIR"
fi
cd "$OPENPI_DIR"
git fetch -q --depth 1 origin "$OPENPI_COMMIT" 2>/dev/null || true
git checkout -q "$OPENPI_COMMIT"
UV_LINK_MODE=copy GIT_LFS_SKIP_SMUDGE=1 uv sync -q
cp -r src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/

# --- 3. Convert. pi05_aloha = Pi0Config(pi05=True) with all defaults -> config.json gets
#        action_horizon=50 (pi05_base's horizon); the weights don't depend on the config.
#        float32: the Orbax params are fp32 (the exporter's bfloat16 default would round them).
echo "[3/5] Converting to torch safetensors -> $OUT_DIR"
JAX_PLATFORMS=cpu .venv/bin/python examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir "$JAX_DIR" --config_name pi05_aloha --precision float32 --output_path "$OUT_DIR"
# The exporter looks for assets/ next to --checkpoint_dir, not inside it.
cp -r "$JAX_DIR/assets" "$OUT_DIR/"

# --- 4. Verify with this package's own loader
echo "[4/5] Verifying with Pi0_5WeightLoader"
cd "$TT_METAL_HOME"
PYTHONPATH="$TT_METAL_HOME" "$TT_METAL_HOME/python_env/bin/python" - "$OUT_DIR" <<'EOF'
import sys
from models.experimental.pi0_5.p150.common.weight_loader import Pi0_5WeightLoader
from models.experimental.pi0_5.p150.common.checkpoint_meta import action_horizon_from_checkpoint
d = sys.argv[1]
ah = action_horizon_from_checkpoint(d)
cw = Pi0_5WeightLoader(d).categorized_weights
ae = cw["action_expert"]
need = ["model.layers.0.input_layernorm.dense.weight", "model.norm.dense.weight"]
assert ah == 50, f"action_horizon={ah}, expected 50"
assert all(k in ae for k in need), f"missing adaRMS tensors in action_expert: {need}"
assert {"time_mlp_in.weight", "time_mlp_out.weight"} <= set(cw["pi0_projections"]), "missing time_mlp_*"
print("      OK: action_horizon=50, " + ", ".join(f"{k}={len(v)}" for k, v in cw.items()))
EOF

# --- 5. Link at the test default path, drop the JAX checkpoint
echo "[5/5] Linking $LINK -> $OUT_DIR"
if [ -L "$LINK" ] || [ ! -e "$LINK" ]; then
    ln -sfn "$OUT_DIR" "$LINK"
else
    echo "      $LINK exists and is not a symlink; leaving it. export PI05_CHECKPOINT_DIR=$OUT_DIR"
fi
if [ "${KEEP_JAX:-0}" != 1 ]; then
    rm -rf "$JAX_DIR"
    echo "      removed $JAX_DIR (KEEP_JAX=1 to keep)"
fi
echo "Done. export PI05_CHECKPOINT_DIR=$OUT_DIR"
