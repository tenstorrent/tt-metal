#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# h3/h3.sh: synthetic MiniMax-H3 ref2va load on one galaxy (reliability.sh h3-random and h3-full steps).
#   h3.sh --mode=random --table <tsv> --stop-at <epoch> --log <dir>   gens and pauses from the table (h3/random-table.tsv)
#   h3.sh --mode=full [--cooldowns <tsv> [--cooldowns-end <s>]] [--time-scale <x>] --stop-at <epoch> --log <dir>
#                                                                     gens back to back, with the zero-load cooldowns of the
#                                                                     table (h3/full-cooldowns.tsv; offsets and durations times
#                                                                     --time-scale, default 1) at gen boundaries;
#                                                                     none starts at or runs past --cooldowns-end s (scaled) from
#                                                                     the first gen: continuous load from there
#   h3.sh --prepare [--log <dir>]                                    set up the ttnn wheel route only, then exit
#   --no-pcie-traffic   no LTX-like PCIe traffic (development only): 'WARNING: ... DISABLED by --no-pcie-traffic' is the
#                       first output line, with every gen in gens.log and on the end line
# PCIe traffic (default on; h3/pcie_traffic.py, volume and seed per mode in h3/pcie-traffic.tsv): a
# checked weight reload at step start (3328 MB per chip, 112 GB per direction per box) and checked per-gen writes and
# readbacks; 'H3 PCIE FAIL <bdf> tray <t> ASIC <a> direction=...' fails the step. Before and after the step h3.sh reads
# each chip's and root port's correctable AER count (sysfs, reads only): 'H3 AER' lines in gens.log, a rise is
# 'H3 AER FINDING'.
# Runs h3/synth_gen.py with pytest on ttnn from the pinned wheel: when traytest/ttnn-wheel.env
# pins a wheel (TTNN_WHEEL_SHA256) and $H3_TT_METAL_BUILD is unset, the venv is the tray-test venv when it already holds
# the wheel and H3's packages (.ttnn-wheel-sha256, .h3-reqs-sha256), else traytest/venv-ttnn-wheel made by
# traytest/setup.sh (TRAY_TTNN=wheel), plus the pinned model-code archive (H3_MODELS_*, a few MB) unpacked on node-local
# disk. Both are sha256-checked; then 'import ttnn' and its version are checked. TRAY_TTNN_WHEEL_PATH and
# TRAY_H3_MODELS_PATH name local copies for a host with no GitHub access. --prepare does only this (reliability.sh
# --install and run start call it): exit 0 ready, 3 with an 'ERROR:' line. A step whose wheel route fails logs an ERROR
# line and tries the site build ($H3_TT_METAL_BUILD, else the default SITE path for h3/tt-metal.sha's pin) or the
# h3-dist archive once, never waiting or compiling; with neither it ends at once (exit 3, 'ERROR: no usable ttnn ...').
# No gen starts at or after --stop-at.
# Writes <dir>/gens.log ('H3 GEN <n> <start> <end> <s> ok|fail:<why>' plus CASE/POWER/SETUP and COOLDOWN lines); pytest's output,
# those lines included, goes to stdout.
# Exit 0 when every gen passed; 1 on a failed gen or test error; 2 usage; 3 no usable ttnn (ERROR line with the cause);
# 4 signal; 5 another process holds the chips (off Slurm; under our Slurm job it is one log line and the run goes on).
# No reset here: reliability.sh owns resets (tt-smi -glx_reset only).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
MODE="" TABLE="" COOLDOWNS="" COOL_END=0 TSCALE=1 STOP_AT="" LOG="" PREP="" PCIE=1
while [ $# -gt 0 ]; do
  case $1 in
    --mode=*) MODE=${1#--mode=} ;;
    --mode) MODE=${2:?}; shift ;;
    --table) TABLE=${2:?}; shift ;;
    --cooldowns) COOLDOWNS=${2:?}; shift ;;
    --cooldowns-end) COOL_END=${2:?}; shift ;;
    --time-scale) TSCALE=${2:?}; shift ;;
    --stop-at) STOP_AT=${2:?}; shift ;;
    --log) LOG=${2:?}; shift ;;
    --prepare) PREP=1 ;;
    --no-pcie-traffic) PCIE=0 ;;
    *) echo "h3.sh: unknown argument $1" >&2; exit 2 ;;
  esac; shift
done
if [ -n "$PREP" ]; then LOG=${LOG:-${TMPDIR:-/tmp}/h3-prepare-$(id -un)}; else
case $MODE in random) [ -r "$TABLE" ] || { echo "h3.sh: --mode=random needs --table <tsv>" >&2; exit 2; } ;; full) ;; *) echo "h3.sh: --mode=random|full" >&2; exit 2 ;; esac
[ -z "$COOLDOWNS" ] || { [ "$MODE" = full ] && [ -r "$COOLDOWNS" ]; } || { echo "h3.sh: --cooldowns <tsv> is for --mode=full and must be readable" >&2; exit 2; }
[[ $COOL_END =~ ^[0-9]+$ ]] || { echo "h3.sh: --cooldowns-end <seconds>" >&2; exit 2; }
[[ $TSCALE =~ ^[0-9]*\.?[0-9]+$ ]] || { echo "h3.sh: --time-scale <number>" >&2; exit 2; }
[[ $STOP_AT =~ ^[0-9]+$ ]] || { echo "h3.sh: --stop-at <epoch>" >&2; exit 2; }
[ -n "$LOG" ] || { echo "h3.sh: --log <dir>" >&2; exit 2; }
fi
OFFW=""; [ "$PCIE" = 1 ] || [ -n "$PREP" ] || { OFFW="WARNING: H3 PCIe traffic (weight reloads, per-gen writes and checked readbacks) DISABLED by --no-pcie-traffic"; echo "$OFFW"; }
mkdir -p "$LOG"; LOG=$(cd "$LOG" && pwd)
log() { echo "[$(date -u +%FT%TZ)] h3.sh: $*"; }

PIN=$(cat "$HERE/tt-metal.sha")
SITE=${H3_TT_METAL_BUILD:-$HOME/h3-tt-metal/$PIN}
ready() { [ -f "$1/READY" ] && [ -x "$1/python_env/bin/python" ] && [ -f "$1/tt-metal/ttnn/ttnn/_ttnn.so" ]; }
why_not() { if [ ! -e "$1" ]; then echo "$1 does not exist"; elif [ ! -f "$1/READY" ]; then echo "$1 has no READY (build not finished)";
  elif [ ! -x "$1/python_env/bin/python" ]; then echo "$1 has no venv python"; else echo "$1 has no ttnn/_ttnn.so"; fi; }
trap 'log "stopped by signal during setup"; exit 4' TERM INT HUP
TT=""; for d in "$HERE/../traytest" "$HERE/.."; do [ -f "$d/setup.sh" ] && [ -f "$d/ttnn-wheel.env" ] && { TT=$(cd "$d" && pwd); break; }; done
ROUTE=build
if [ -z "${H3_TT_METAL_BUILD:-}" ] && [ -n "$TT" ]; then
  # shellcheck disable=SC1091
  . "$TT/lib/pinned.sh"
  pinned_load "$TT/ttnn-wheel.env" 2> "$LOG/pin.err" || { log "ERROR: ttnn wheel pin unreadable: $(tail -n 1 "$LOG/pin.err")"; exit 3; }
  [ -z "$TTNN_WHEEL_SHA256" ] || ROUTE=wheel
fi
# The kernel source files (paths relative to the tree root) that H3's python (<tree>/models/tt_dit) names, one per line.
kernel_refs() {
  (cd "$1" && { grep -rlE '^[[:space:]]*KERNEL_DIR[[:space:]]*=[[:space:]]*"' --include='*.py' models/tt_dit 2> /dev/null || true; } | while read -r f; do
    d=$(sed -nE 's/^[[:space:]]*KERNEL_DIR[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/p' "$f" | head -n 1)
    { grep -oE '\{KERNEL_DIR\}/[A-Za-z0-9_./-]+\.(cpp|hpp|h)' "$f" || true; } | sed "s|{KERNEL_DIR}|$d|"
  done
  { grep -rhoE '"models/[A-Za-z0-9_./-]+\.(cpp|hpp|h)"' --include='*.py' models/tt_dit 2> /dev/null || true; } | tr -d '"') | sort -u
}
# The wheel route (venv, model code, import check); on failure WHY names the cause and it returns 3. Idempotent: on a
# prepared box (reliability.sh --install and run start call h3.sh --prepare) it only re-checks.
wheel_setup() {
  t_setup=$(date +%s)
  [ -n "$H3_MODELS_SHA256" ] || { WHY="ttnn wheel pinned, but $TT/ttnn-wheel.env pins no H3 model code (H3_MODELS_SHA256)"; return 3; }
  PYENV=${H3_TRAY_VENV:-$TT/venv}
  # H3's own python packages (h3/requirements-wheel.txt) go only into venv-ttnn-wheel: a venv is reused when it holds
  # the pinned wheel and those packages (marker .h3-reqs-sha256), and run.sh's venv is never changed
  REQS=$HERE/requirements-wheel.txt; REQS_SHA=$(sha256sum "$REQS" | cut -d' ' -f1)
  if [ "$(cat "$PYENV/.ttnn-wheel-sha256" 2> /dev/null)" = "$TTNN_WHEEL_SHA256" ] && [ -x "$PYENV/bin/python" ] \
    && [ "$(cat "$PYENV/.h3-reqs-sha256" 2> /dev/null)" = "$REQS_SHA" ]; then
    log "ttnn wheel $TTNN_WHEEL_VERSION: reusing the tray-test venv $PYENV"
  else
    PYENV=$TT/venv-ttnn-wheel
    log "ttnn wheel $TTNN_WHEEL_VERSION: the tray-test venv does not hold it with H3's packages; setting up $PYENV"
    TRAY_TTNN=wheel TRAY_VENV=venv-ttnn-wheel "$TT/setup.sh" > "$LOG/setup-wheel.log" 2>&1 \
      || { WHY="ttnn wheel setup failed: $(grep -v '^\s*$' "$LOG/setup-wheel.log" | tail -n 1 | cut -c1-300)"; return 3; }
    if [ "$(cat "$PYENV/.h3-reqs-sha256" 2> /dev/null)" != "$REQS_SHA" ]; then
      log "installing H3's python packages ($REQS) into $PYENV"
      "$PYENV/bin/pip" install -q -r "$REQS" >> "$LOG/setup-wheel.log" 2>&1 \
        || { WHY="H3 python packages did not install: $(grep -v '^\s*$' "$LOG/setup-wheel.log" | tail -n 1 | cut -c1-300)"; return 3; }
      # torchvision (models/conftest.py imports it, job 173321) must match the venv's CPU torch: pinning that torch
      # makes pip pick the matching torchvision from the same index and leaves torch as it is
      tv=$("$PYENV/bin/python" -c 'import torch; print("torch", torch.__version__)' 2>> "$LOG/setup-wheel.log" | awk '{print $2}')
      [ -n "$tv" ] || { WHY="no torch in $PYENV to match torchvision to"; return 3; }
      "$PYENV/bin/pip" install -q torchvision "torch==$tv" --index-url https://download.pytorch.org/whl/cpu >> "$LOG/setup-wheel.log" 2>&1 \
        || { WHY="torchvision for torch $tv did not install: $(grep -v '^\s*$' "$LOG/setup-wheel.log" | tail -n 1 | cut -c1-300)"; return 3; }
      echo "$REQS_SHA" > "$PYENV/.h3-reqs-sha256"
    fi
  fi
  miss=$("$PYENV/bin/python" -c "import importlib.util as u
print(' '.join(m for m in 'pytest pytest_timeout loguru diffusers transformers safetensors huggingface_hub PIL av tqdm yaml pytz torch torchvision ttnn tt_umd'.split() if not u.find_spec(m)))" 2>&1) \
    || { WHY="python check of $PYENV failed: $miss"; return 3; }
  [ -z "$miss" ] || { WHY="venv $PYENV lacks H3 python modules: $miss"; return 3; }
  if [ -n "${H3_MODELS_ROOT:-}" ]; then MROOT=$H3_MODELS_ROOT   # a fresh, empty cache (tools/reliability-smoke)
  elif [ -n "${SLURM_JOB_ID:-}" ]; then MROOT=${SLURM_TMPDIR:-/tmp/$(id -un)}/h3-models; else MROOT=$HOME/.cache/h3-models; fi
  case $(realpath -m "$MROOT") in /data|/data/*) WHY="model-code root $MROOT is on shared /data; use node-local disk"; return 3 ;; esac
  W=$MROOT/$H3_MODELS_SHA256
  mkdir -p "$MROOT"; exec 7> "$MROOT/$H3_MODELS_SHA256.lock"; flock 7
  msrc=cached
  if [ ! -f "$W/.h3models" ]; then
    pinned_fetch "H3 model code" "$H3_MODELS_SHA256" "$H3_MODELS_PATH" "$H3_MODELS_URL" "$MROOT/$H3_MODELS_SHA256.tar.zst" 2> "$LOG/pin.err" \
      || { WHY="$(tail -n 1 "$LOG/pin.err")"; return 3; }
    msrc=$PINNED_SRC
    log "H3 model code archive $(( ($(stat -c %s "$MROOT/$H3_MODELS_SHA256.tar.zst") + 524288) / 1048576 )) MB"
    rm -rf "${W:?}.part"; mkdir -p "$W.part"
    zstd -q -dc "$MROOT/$H3_MODELS_SHA256.tar.zst" | tar -xf - -C "$W.part" || { rm -rf "${W:?}.part"; WHY="model code archive did not unpack"; return 3; }
    echo "$H3_MODELS_SHA256" > "$W.part/.h3models"; rm -rf "${W:?}"; mv "$W.part" "$W"; rm -f "$MROOT/$H3_MODELS_SHA256.tar.zst"
  fi
  exec 7>&-
  [ -d "$W/models/tt_dit/tests/models/minimax_h3" ] || { WHY="model code $W has no models/tt_dit/tests/models/minimax_h3"; return 3; }
  # import ttnn itself, from a neutral directory with no tt-metal tree on the path: the wheel's own runtime
  got=$(cd / && env -u TT_METAL_HOME -u PYTHONPATH "$PYENV/bin/python" -c "import ttnn, importlib.metadata as m; print(m.version('ttnn'))" 2> "$LOG/import.err" | tail -n 1) \
    || { WHY="import ttnn in $PYENV failed: $(grep -v '^\s*$' "$LOG/import.err" | tail -n 1 | cut -c1-300)"; return 3; }
  [ "$got" = "$TTNN_WHEEL_VERSION" ] || { WHY="$PYENV has ttnn $got, not the pinned $TTNN_WHEEL_VERSION"; return 3; }
  # import what an H3 step imports from the model tree, the way the step does (cwd and PYTHONPATH the tree): a stale
  # archive fails here at --prepare, not as pytest exit 2 at step start
  bad=$(cd "$W" && env -u TT_METAL_HOME PYTHONPATH="$W" "$PYENV/bin/python" -c "# h3 model-tree import check
import importlib, sys
for m in sys.argv[1:]:
    try:
        importlib.import_module(m)
    except Exception as e:
        print(m + (': missing module ' + (e.name or m) if isinstance(e, ModuleNotFoundError) else ': ' + type(e).__name__ + ': ' + str(e))[:300])
        sys.exit(1)" tt_py_test_utils_common tests.tests_common models.tt_dit.tests.models.minimax_h3 2> "$LOG/import-models.err") \
    || { WHY="model code $W does not import with PYTHONPATH=$W: ${bad:-$(grep -v '^\s*$' "$LOG/import-models.err" | tail -n 1 | cut -c1-300)} (archive $H3_MODELS_SHA256 is stale)"; return 3; }
  # every tt_dit kernel source the model code names (KERNEL_DIR = "<dir>" with f"{KERNEL_DIR}/<file>", or a literal
  # "models/.../<file>.cpp") must be in the unpacked tree: tt-metal finds it there through TT_METAL_KERNEL_PATH=$W, and a
  # missing one fails here, not as TT_THROW kernel.cpp:92 'Kernel file ... doesn't exist' at the first gen
  kmiss=$(kernel_refs "$W" | while read -r k; do [ -f "$W/$k" ] || echo "$k"; done)
  [ -z "$kmiss" ] || { WHY="model code $W lacks kernel sources: $(echo $kmiss | cut -c1-300) (archive $H3_MODELS_SHA256 is stale)"; return 3; }
  log "SETUP ttnn wheel $TTNN_WHEEL_VERSION (sha256 $TTNN_WHEEL_SHA256), venv $PYENV, model code $W ($(du -sm "$W" | cut -f1) MB), $(( $(date +%s) - t_setup )) s"
  log "ROUTE ttnn wheel source=$(cat "$PYENV/.ttnn-wheel-source" 2> /dev/null || echo unknown) models=$msrc"
}

# The fallback after a failed wheel route: the site build ($H3_TT_METAL_BUILD, else the default SITE path for the pin),
# else the pin's prebuilt archive unpacked by h3/dist/fetch.sh (H3_DIST=0: off). One try, never a wait and never a
# compile: with neither, the step ends at once with an ERROR line.
build_once() {
  if ready "$SITE"; then BUILD=$SITE; return 0; fi
  cause="site build: $(why_not "$SITE")"
  if [ -z "${H3_TT_METAL_BUILD:-}" ] && [ "${H3_DIST:-1}" != 0 ]; then
    if b=$(H3_DIST_BUILD=0 "$HERE/dist/fetch.sh" build "$PIN" 2> "$LOG/fetch.err") && ready "$b"; then BUILD=$b; log "build from the prebuilt archive: $b"; return 0; fi
    cause+="; archive: $(grep -v '^\s*$' "$LOG/fetch.err" | tail -n 1 | cut -c1-300)"
  fi
  return 3
}
WHY="" cause=""
if [ -n "$PREP" ]; then
  [ "$ROUTE" = wheel ] || { log "prepare: no ttnn wheel route (${H3_TT_METAL_BUILD:+H3_TT_METAL_BUILD is set}${H3_TT_METAL_BUILD:-no wheel pinned}): nothing to prepare"; exit 0; }
  wheel_setup || { log "ERROR: ttnn wheel prepare failed: $WHY"; exit 3; }
  exit 0
fi
if [ "$ROUTE" = wheel ]; then
  if wheel_setup; then BUILD=$PYENV
  else exec 7>&-; log "ERROR: ttnn wheel route failed: $WHY; trying the site build or the h3-dist archive once"; ROUTE=build; fi
fi
if [ "$ROUTE" = build ]; then
  build_once || { log "ERROR: no usable ttnn for synthetic H3${WHY:+ (wheel: $WHY)}; $cause; the step ends now, no wait for a build"; exit 3; }
  W=$BUILD/tt-metal PYENV=$BUILD/python_env
fi

# Read-only: nobody else holds the chips (our own process tree may). Under our own Slurm allocation (SLURM_JOB_ID set)
# others' leftovers get one log line naming each as pid(user/command) and the run goes on (holding our own
# reservation is enough to run); we never signal them. Off Slurm they refuse the step (exit 5).
others=""
for p in $(timeout 30 fuser /dev/tenstorrent/* 2> /dev/null || true); do
  p=${p//[!0-9]/}; [ -n "$p" ] || continue
  q=$p; while [ -n "$q" ] && [ "$q" -gt 1 ] && [ "$q" != $$ ] && [ "$q" != "$PPID" ]; do q=$(awk '{print $4}' "/proc/$q/stat" 2> /dev/null || echo 1); done
  [ "$q" = $$ ] || [ "$q" = "$PPID" ] || others+=" $p($(ps -o user=,comm= -p "$p" 2> /dev/null | tr -s ' ' '/'))"
done
if [ -n "$others" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
  log "other holders of /dev/tenstorrent under our Slurm job $SLURM_JOB_ID, not touched, going on:$others"
elif [ -n "$others" ]; then log "refused: other holders of /dev/tenstorrent:$others"; exit 5; fi

# The test runs from inside the tt-metal tree (its relative imports); the build may be shared by several nodes, so
# each run copies under its own names and removes only those.
TAG=$(hostname -s | tr -c 'a-zA-Z0-9\n' _)_$$
TD=$W/models/tt_dit/tests/models/minimax_h3
TEST=$TD/test_h3_synth_gen_$TAG.py
WMOD=$TD/synth_weights_$TAG.py
PMOD=$TD/pcie_traffic_$TAG.py
# Scratch and caches stay node-local, never under shared /data: a TMPDIR or TT_METAL_CACHE there is
# replaced by a dir under $SLURM_TMPDIR, else /tmp/$USER-h3-<pid>, removed at exit.
LOCAL=${SLURM_TMPDIR:-/tmp/$(id -un)-h3-${SLURM_JOB_ID:-$$}}
under_data() { case $(realpath -m "$1") in /data|/data/*) return 0 ;; *) return 1 ;; esac; }
if under_data "${TMPDIR:-/tmp}"; then export TMPDIR=$LOCAL/tmp; mkdir -p "$TMPDIR"; log "TMPDIR was on /data, now $TMPDIR"; fi
if [ -n "${TT_METAL_CACHE:-}" ] && under_data "$TT_METAL_CACHE"; then export TT_METAL_CACHE=$LOCAL/tt-metal-cache; mkdir -p "$TT_METAL_CACHE"; log "TT_METAL_CACHE was on /data, now $TT_METAL_CACHE"; fi
WORK=$(mktemp -d "${TMPDIR:-/tmp}/h3synth.XXXXXX")
# Under Slurm a node-local tree goes only through slurm-log-archive.sh: every log in WORK (H3
# logs included) and in LOCAL to one shared-storage <jobid>-<node>-h3-<run>.tar.zst first, the tree kept with
# an ERROR line when that write fails. Outside Slurm the trees are removed as before.
SLA=$HERE/../reliability/slurm/slurm-log-archive.sh
gone() {   # <tree> <name>
  if [ -n "${SLURM_JOB_ID:-}" ]; then
    if [ -s "$SLA" ] && . "$SLA"; then sla_archive "$1" "$2" >&2 || true; else log "ERROR: no slurm-log-archive.sh: node-local $1 kept"; fi
  else rm -rf "${1:?}"; fi; }   # not a Slurm job: no slurm-log-archive.sh step
cleanup() { [ -z "${WPID:-}" ] || kill "$WPID" 2> /dev/null || true; [ -z "${MPID:-}" ] || kill "$MPID" 2> /dev/null || true
  rm -f "${TEST:?}" "${WMOD:?}" "${PMOD:?}"; rm -f "$TD"/__pycache__/*"_$TAG".*.pyc 2> /dev/null || true
  gone "$WORK" "h3-${TAG}-${WORK##*.}"; if [ -z "${SLURM_TMPDIR:-}" ] && [ -d "$LOCAL" ]; then gone "$LOCAL" "h3-${TAG}-local"; fi; }
trap cleanup EXIT
trap 'exit 4' TERM INT HUP
cp "$HERE/synth_weights.py" "$WMOD"; cp "$HERE/pcie_traffic.py" "$PMOD"
sed -e "s/^from \.synth_weights import/from .synth_weights_$TAG import/" -e "s/^from \.pcie_traffic import/from .pcie_traffic_$TAG import/" "$HERE/synth_gen.py" > "$TEST"
grep -q "^from .synth_weights_$TAG import" "$TEST" || { log "could not point the test at synth_weights_$TAG"; exit 1; }
grep -q "^from .pcie_traffic_$TAG import" "$TEST" || { log "could not point the test at pcie_traffic_$TAG"; exit 1; }

export H3_SYNTH_MODE=$MODE H3_SYNTH_TABLE=${TABLE:+$(cd "$(dirname "$TABLE")" && pwd)/$(basename "$TABLE")}
export H3_SYNTH_COOLDOWNS=${COOLDOWNS:+$(cd "$(dirname "$COOLDOWNS")" && pwd)/$(basename "$COOLDOWNS")} H3_SYNTH_TIME_SCALE=$TSCALE H3_SYNTH_COOLDOWN_END_S=$COOL_END
export H3_SYNTH_STOP_AT=$STOP_AT H3_SYNTH_GENLOG=$LOG/gens.log H3_SYNTH_CONFIG=$HERE/config H3_SYNTH_WORK=$WORK
export H3_PCIE_TABLE=$HERE/pcie-traffic.tsv H3_PCIE_TRAFFIC=$PCIE
# Correctable AER of each chip and its root port before and after the step (sysfs reads only)
aer() {   # snap <file> | diff: the 'H3 AER' lines go to gens.log and to stdout
  if [ "$1" = snap ]; then python3 "$HERE/pcie_traffic.py" aer-snap > "$2" 2> "$2.err" || log "AER read before or after the step failed: $(tail -n 1 "$2.err")"
  else python3 "$HERE/pcie_traffic.py" aer-diff "$LOG/aer-before.tsv" "$LOG/aer-after.tsv" 2>&1 | tee -a "$LOG/gens.log" | sed "s/^/[$(date -u +%FT%TZ)] h3.sh: /" || true; fi; }
aer_after() { [ -z "${AER_DONE:-}" ] || return 0; AER_DONE=1; aer snap "$LOG/aer-after.tsv"; aer diff; }
aer snap "$LOG/aer-before.tsv"
# A warm kernel cache for this pin and chip arch, when the site has one (h3/dist/pack.sh kcache). H3_DIST_KCACHE=0: off.
if [ -z "${TT_METAL_CACHE:-}" ] && [ -f "$BUILD/.h3dist" ] && [ "${H3_DIST_KCACHE:-1}" != 0 ] \
  && kc=$("$HERE/dist/fetch.sh" kcache "$(cat "$HERE/tt-metal.sha")" "$BUILD"); then
  export TT_METAL_CACHE=$kc; log "warm kernel cache $kc"
fi
if [ "$ROUTE" = wheel ]; then   # the wheel carries tt-metal's runtime; the model tree is on the import path, and
  # tt-metal looks up the model's own kernel sources (models/tt_dit/.../kernels/*.cpp) in it whatever the cwd
  unset TT_METAL_HOME; export PYTHONPATH=$W TT_METAL_KERNEL_PATH=$W
else export TT_METAL_HOME=$W PYTHONPATH=$W:$W/ttnn:$W/tools; fi
export PYTHONDONTWRITEBYTECODE=1 VIRTUAL_ENV=$PYENV PATH=$PYENV/bin:$PATH
log "${OFFW:+$OFFW; }mode $MODE${TABLE:+ table $TABLE}${COOLDOWNS:+ cooldowns $COOLDOWNS x$TSCALE$( [ "$COOL_END" = 0 ] || echo " until $COOL_END s")}, stop at $(date -d "@$STOP_AT" +%FT%T%z), ${ROUTE/wheel/ttnn wheel venv} $BUILD, host $(hostname -s)"
rc=0; t_launch=$(date +%s)
memav() { awk '/^MemAvailable:/ { printf "%.0f", $2 / 1048576 }' /proc/meminfo 2> /dev/null || true; }   # GB
mem_launch=$(memav)
# Host memory before the launch: synthetic H3 h3-full peaks at 16.6 GiB of python RSS (job
# 195814, 7 gens). One cluster node lost about 14 GB of MemAvailable per smoke job to memory
# held outside any process (594 GB total, 1.2 GB available, about 50 GB in processes, page cache, slab and hugepages;
# probe job 195269), and pytest was OOM-killed in jobs 194664, 194881 and 194927. Below H3_MEM_NEED_GB (32) the step is
# not started: exit 3 with an ERROR line naming the node's memory, a node finding, not an OOM kill mid-load.
mem_need=${H3_MEM_NEED_GB:-32}
if [[ $mem_launch =~ ^[0-9]+$ ]] && [ "$mem_launch" -lt "$mem_need" ]; then
  aer_after   # first, so the ERROR line is h3.log's last line: reliability.sh's missed-H3 note reads it (tail -n 1)
  log "ERROR: host memory short: MemAvailable $mem_launch GB, synthetic H3 needs $mem_need GB; $(awk -v ma="$mem_launch" '/^(MemTotal|AnonPages|Hugetlb|Slab|Cached|Shmem):/ { m[$1] = $2 / 1048576 }
    END { printf "MemTotal %.1f GB, held outside processes, page cache, slab and hugepages %.1f GB", m["MemTotal:"],
      m["MemTotal:"] - m["AnonPages:"] - m["Hugetlb:"] - m["Slab:"] - m["Cached:"] - ma }' /proc/meminfo 2> /dev/null || true); the step is not started"
  exit 3
fi
(cd "$W" && exec python -m pytest -p no:cacheprovider -s -x "$TEST") 2>&1 &
PY=$!
# Host memory every H3_MEM_POLL s (5) while pytest runs, to $LOG/mem.tsv (smoke job 194001's pytest was
# SIGKILLed while loading the transformer weights): MemAvailable, Shmem, pytest's VmRSS and VmHWM (peak), hugepages (kB)
mem_watch() {   # it waits with read -t on a pipe no one writes, not a sleep child, so a TERM from finish or cleanup at any
  # point leaves nothing behind (a TERM between 'sleep &' and its pid's assignment left a sleep 5 running)
  trap 'exit 0' TERM; exec 9<> <(:)
  printf 'epoch\tmem_avail_kb\tshmem_kb\tpy_rss_kb\tpy_hwm_kb\thugepages_total\thugepages_free\n' > "$LOG/mem.tsv"
  while kill -0 "$PY" 2> /dev/null; do
    { awk -v t="$(date +%s)" '/^(MemAvailable|Shmem|HugePages_Total|HugePages_Free):/ { m[$1] = $2 }
        END { printf "%s\t%s\t%s", t, m["MemAvailable:"], m["Shmem:"]; h = m["HugePages_Total:"] "\t" m["HugePages_Free:"] }
        FILENAME != "/proc/meminfo" && /^(VmRSS|VmHWM):/ { p[$1] = $2 }
        END { printf "\t%s\t%s\t%s\n", p["VmRSS:"], p["VmHWM:"], h }' /proc/meminfo "/proc/$PY/status" 2> /dev/null || true; } >> "$LOG/mem.tsv"
    read -r -t "${H3_MEM_POLL:-5}" -u 9 _ || true
  done; }
mem_watch & MPID=$!
# exit 137: the least MemAvailable and the highest pytest peak in mem.tsv, and any OOM-killer line the kernel log lets us read
mem_oom() {
  awk -F'\t' 'NR > 1 && $2 != "" { if (m == "" || $2 < m) m = $2; if ($5 > h) h = $5 }
    END { printf "least MemAvailable %.1f GB, pytest peak RSS %.1f GB (mem.tsv)", m / 1048576, h / 1048576 }' "$LOG/mem.tsv" 2> /dev/null || true
  k=$({ timeout 10 dmesg -T 2> /dev/null || timeout 10 journalctl -k -q --no-pager --since "@$t_launch" 2> /dev/null || true; } |
    grep -aE 'oom-kill|Out of memory|Killed process' | tail -n 3 || true)
  if [ -n "$k" ]; then printf '; kernel: %s' "$(tr '\n' '|' <<< "$k")"; else printf '; no OOM-killer line readable in the kernel log'; fi; }
finish() {   # <pytest exit>: the end summary and h3.sh's exit (0 only when pytest exited 0)
  rc=$1
  kill "$MPID" 2> /dev/null || true
  aer_after
  n_ok=$(grep -c '^H3 GEN .* ok$' "$LOG/gens.log" 2> /dev/null || true); n_fail=$(grep -c '^H3 GEN .* fail:' "$LOG/gens.log" 2> /dev/null || true)
  # every gens.log read below may find no file (the 4x8 mesh did not open, no gen ran): never an exit under set -euo pipefail
  first=$(sed -nE 's/.*H3 GEN [0-9]+ [0-9.]+ ([0-9.]+) [0-9.]+ ok$/\1/p' "$LOG/gens.log" 2> /dev/null | head -n 1 || true)
  [ -z "$first" ] || log "first ok gen ended $(( ${first%.*} - t_launch )) s after launch (load plus first gen)"
  n_pf=$(grep -c '^H3 PCIE FAIL ' "$LOG/gens.log" 2> /dev/null || true); n_ar=$(grep -c '^H3 AER FINDING ' "$LOG/gens.log" 2> /dev/null || true)
  pt=$({ grep '^H3 PCIE TOTAL ' "$LOG/gens.log" 2> /dev/null || true; } | tail -n 1 | sed 's/^H3 PCIE TOTAL //')
  # exit 137 is a SIGKILL h3.sh never sends: the kernel OOM killer or the caller's outer timeout (smoke job
  # 193913: pytest killed 74 s into the weight load, on a node whose OOM killer had taken an H3 python of 55 GB before)
  [ "$rc" != 137 ] || log "pytest ended by SIGKILL (exit 137), not sent by h3.sh: the kernel OOM killer or the caller's outer timeout; MemAvailable ${mem_launch:-?} GB at launch, $(memav || true) GB now; $(mem_oom)"
  log "${OFFW:+$OFFW; }end: pytest exit $rc, gens ok ${n_ok:-0}, failed ${n_fail:-0}, PCIe readback FAILs ${n_pf:-0}, AER rises ${n_ar:-0}${pt:+, PCIe traffic $pt}"
  [ "$rc" = 0 ] && exit 0
  exit 1; }
# On a stop signal pytest gets TERM; synth_gen.py's handler raises, the fixture teardown closes the mesh and pytest exits
# Up to H3_TERM_GRACE s (15) is a clean exit; past it, h3.sh adds no KILL and waits for pytest until the
# caller's outer timeout (timeout -k, reliability.sh's dev_run) or the broker's kill ends it. A second signal is ignored.
# Exit 4, except when pytest had already passed (exit 0) as the stop came: then its result stands (a stopped
# pytest exits 2, interrupted, or by the signal).
stop_pytest() {
  local g=${H3_TERM_GRACE:-15} t0 i=0 prc=0; t0=$(date +%s)
  trap "" TERM INT HUP; log "stopped by signal: ending pytest $PY (TERM; closes the mesh device, up to $g s)"
  kill -TERM "$PY" 2> /dev/null || true
  while kill -0 "$PY" 2> /dev/null && [ "$i" -lt "$g" ]; do sleep 1; i=$((i + 1)); done
  if kill -0 "$PY" 2> /dev/null; then
    log "pytest $PY still running $g s after TERM: waiting for it until the outer timeout or the broker's kill ends it"
    wait "$PY" || prc=$?
  else wait "$PY" || prc=$?; log "pytest $PY ended $(( $(date +%s) - t0 )) s after TERM"; fi
  if [ "$prc" = 0 ]; then log "pytest $PY had passed (exit 0) as the stop came: its result stands"; finish 0; fi
  aer_after; exit 4; }
trap stop_pytest TERM INT HUP
# Inside a tt-device broker job (TT_DEVICE_MCP_JOB_TAG set), the broker's PREEMPTING notice comes 60 s before its SIGTERM;
# ending at the notice means the job ends itself with the device closed, so the broker needs no reset.
# H3_STOP_FILE: reliability.sh creates it to end the step for work the same way. Read-only: /health every 5 s.
preempt_watch() {
  local id=${TT_DEVICE_MCP_JOB_TAG:-} sock=${H3_BROKER_SOCK:-${TT_DEVICE_MCP_SOCKET:-/run/tt-device-broker/broker.sock}} h why
  id=${id%-*}   # the tag is '<job id>-<12 hex>'
  while kill -0 "$PY" 2> /dev/null; do
    why=""
    if [ -n "${H3_STOP_FILE:-}" ] && [ -e "$H3_STOP_FILE" ]; then why="stop file $H3_STOP_FILE"
    elif [ -n "${TT_DEVICE_MCP_JOB_TAG:-}" ] && [ -S "$sock" ]; then
      h=$(timeout 5 curl -s -m 4 --unix-socket "$sock" http://localhost/health 2> /dev/null || true)
      if grep -q '"preempt": *"PREEMPTING' <<< "$h" && grep -q "\"preempt_job_id\": *\"$id\"" <<< "$h"; then
        why="broker PREEMPTING notice for job $id ($(grep -o '"preempt": *"[^"]*"' <<< "$h" | head -n 1))"; fi
    fi
    if [ -n "$why" ]; then log "$why: ending the step now with the device closed"; kill -TERM $$ 2> /dev/null || true; return 0; fi
    sleep "${H3_PREEMPT_POLL:-5}"
  done; }
WPID=""
if [ -n "${TT_DEVICE_MCP_JOB_TAG:-}${H3_STOP_FILE:-}" ]; then preempt_watch & WPID=$!; fi
wait "$PY" || rc=$?
# pytest ended on its own: a stop that comes now (the watcher's TERM to $$ racing pytest's end) cannot turn its result into
# exit 4; the watcher goes
trap "" TERM INT HUP; [ -z "$WPID" ] || kill "$WPID" 2> /dev/null || true
finish "$rc"
