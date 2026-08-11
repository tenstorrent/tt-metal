#!/bin/bash
# Verify the fused snake end to end, from any machine that mounts /data/rshirvani.
#
# WHY this exists: the environment is not guessable. TT_METAL_HOME must be the worktree (device
# kernels are JIT-compiled from it), PYTHONPATH needs the diffusers-main overlay or the reference
# model will not import, and TMPDIR must be off /tmp because parallel jobs there clobber the
# assembler's temp files and produce spurious build failures.
#
# The host C++ was built on c09u14 into the shared /data/rshirvani/tt-metal/build_Release, so no
# rebuild is needed on the new machine -- only the JIT kernel cache is per-machine and it repopulates
# itself on the first run.
#
# Stage 0 is a bare eltwise add. If that hangs the card is wedged and nothing after it means
# anything, which is exactly the state c09u14 was left in; recover with `tt-smi -r 0` and re-run.
set -u

WT=/data/rshirvani/tt-metal/.claude/worktrees/audio-kernels
AP=$WT/models/tt_dit/tests/models/minimax_h3/audio_perf
LOGDIR=${LOGDIR:-$HOME/snake_verify_logs}

source /data/rshirvani/tt-metal/python_env/bin/activate
export TT_METAL_HOME=$WT
export PYTHONPATH=/data/rshirvani/audio_ref_pkgs:$WT
export TMPDIR=${TMPDIR:-$HOME/ttcc}
mkdir -p "$TMPDIR" "$LOGDIR"
cd "$WT" || exit 1

echo "host=$(hostname)  TT_METAL_HOME=$TT_METAL_HOME  logs=$LOGDIR"

# --- Stage 0: is the card alive at all? ------------------------------------------------
echo
echo "=== [0/3] device health (bare eltwise add, 180s cap)"
timeout 180 python - <<'PY' > "$LOGDIR/health.log" 2>&1
import torch, ttnn
d = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=65536)
try:
    a = ttnn.from_torch(torch.randn(1, 1, 32, 32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=d)
    print("HEALTH OK sum=%.3f" % float(ttnn.to_torch(ttnn.add(a, a)).sum()), flush=True)
finally:
    ttnn.close_mesh_device(d)
PY
if ! grep -q "HEALTH OK" "$LOGDIR/health.log"; then
    echo "FAILED -- the card is wedged (a bare add did not complete)."
    echo "Recover with:  tt-smi -r 0     then re-run this script."
    echo "Last log lines:"; tail -5 "$LOGDIR/health.log"
    exit 1
fi
grep "HEALTH OK" "$LOGDIR/health.log"

# --- Stage 1: the default path must be untouched ---------------------------------------
# The snake code is all behind TT_CONV1D_SNAKE_PARAMS, so with the var unset this must stay exactly
# as it was. Catches a kernel edit that breaks the build or the ungated path.
echo
echo "=== [1/3] default-path regression (depthwise_mac / channel_padding, 1800s cap)"
timeout 1800 python -m pytest \
    models/tt_dit/tests/models/minimax_h3/test_audio_vae_minimax_h3.py \
    -k "depthwise_mac or channel_padding" -q > "$LOGDIR/regress.log" 2>&1
echo "exit=$?"; tail -3 "$LOGDIR/regress.log"

# --- Stage 2: the fused snake itself ---------------------------------------------------
# Bar is rel_rmse ~1e-07 against the float64 golden. A hang here (rather than a wrong number) is the
# mcast handshake being unpaired between sender and receiver.
echo
echo "=== [2/3] fused snake vs float64 golden (900s cap)"
TT_CONV1D_SNAKE_PARAMS=1 timeout 900 python "$AP/snake_fused_verify.py" > "$LOGDIR/snake.log" 2>&1
echo "exit=$?"
grep -E "prepared weight|widened weight|conv alone|fused snake|PASS|FAIL|differs from plain|FUSED RUN FAILED" \
    "$LOGDIR/snake.log" || tail -15 "$LOGDIR/snake.log"

# --- Stage 3: the full gate ------------------------------------------------------------
echo
echo "=== [3/3] full test file (17 expected, 3600s cap)"
timeout 3600 python -m pytest \
    models/tt_dit/tests/models/minimax_h3/test_audio_vae_minimax_h3.py \
    -q > "$LOGDIR/full.log" 2>&1
echo "exit=$?"; tail -4 "$LOGDIR/full.log"

echo
echo "Done. Full logs in $LOGDIR"
