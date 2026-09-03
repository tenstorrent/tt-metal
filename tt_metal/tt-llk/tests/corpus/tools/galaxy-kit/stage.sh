#!/usr/bin/env bash
# galaxy-kit stage.sh — compile-and-ship.
#
#   stage.sh -t <toolchain sfpi dir> -f <tt-llk farm dir> -w <workdir> \
#            [-o <ops>] [-b <FINAL-BOARD.tsv>] [-d <dest>] [-p <python>] \
#            [--relink-sfpi] [--no-ship]
#
# Builds every benchmark ELF on THIS machine at the given toolchain (one
# pytest --compile-producer session per flag/env group; resume-safe via
# DONE markers), packs the execute-only bundle (farm + builds + specs +
# worker; runtime binutils only, no compiler ships), and streams it to the
# cluster through the Mac relay (never touching the relay's disk).
set -uo pipefail
KIT=$(cd "$(dirname "$0")" && pwd)
source "$KIT/lib/remote.sh"

TOOLCHAIN=${LK_TOOLCHAIN:-/home/ttuser/sfpi-uplift/sfpi/build/sfpi}
FARMDIR=""
WORK=""
OPS=""
BOARD=${LK_BOARD:-/home/ttuser/sfpi-uplift/laneFM-evidence-20260822/FINAL-BOARD.tsv}
PYBIN=""
SHIP=1
RELINK=0
while [ $# -gt 0 ]; do
  case "$1" in
    -t) TOOLCHAIN=$2; shift 2;;
    -f) FARMDIR=$2; shift 2;;
    -w) WORK=$2; shift 2;;
    -o) OPS=$2; shift 2;;
    -b) BOARD=$2; shift 2;;
    -d) LK_DEST=$2; shift 2;;
    -p) PYBIN=$2; shift 2;;
    --relink-sfpi) RELINK=1; shift;;
    --no-ship) SHIP=0; shift;;
    *) echo "unknown arg $1"; exit 2;;
  esac
done
: "${FARMDIR:?-f <tt-llk dir> required}"
: "${WORK:?-w <workdir> required}"
mkdir -p "$WORK"
PYDIR=$FARMDIR/tests/python_tests
[ -f "$PYDIR/conftest.py" ] || { echo "REFUSE: $FARMDIR is not a tt-llk dir"; exit 2; }
if [ -z "$PYBIN" ]; then
  for c in "$FARMDIR/tests/.venv/bin/python" \
           /home/ttuser/sfpi-uplift/tt-metal/tt_metal/tt-llk/tests/.venv/bin/python; do
    [ -x "$c" ] && PYBIN=$c && break
  done
fi
[ -x "${PYBIN:-}" ] || { echo "REFUSE: no venv python found (-p)"; exit 2; }

# ---- toolchain identity + farm wiring ----
CC1=$TOOLCHAIN/compiler/libexec/gcc/riscv-tt-elf/15.1.0/cc1plus
[ -f "$CC1" ] || { echo "REFUSE: no cc1plus under $TOOLCHAIN"; exit 2; }
CCSHA=$(sha256sum "$CC1" | cut -c1-12)
SFPILINK=$FARMDIR/tests/sfpi
if [ "$(readlink -f "$SFPILINK" 2>/dev/null)" != "$(readlink -f "$TOOLCHAIN")" ]; then
  if [ "$RELINK" = 1 ]; then
    ln -sfn "$(readlink -f "$TOOLCHAIN")" "$SFPILINK"
    echo "relinked tests/sfpi -> $TOOLCHAIN"
  else
    echo "REFUSE: $SFPILINK does not resolve to $TOOLCHAIN (pass --relink-sfpi"
    echo "        only for a farm this kit owns; never repoint a shared farm)"
    exit 2
  fi
fi
{
  echo "toolchain=$TOOLCHAIN"
  echo "cc1plus_sha12=$CCSHA"
  echo "farm=$FARMDIR"
  echo "farm_git=$(git -C "$FARMDIR" rev-parse --short HEAD 2>/dev/null || echo n/a)"
  echo "board=$BOARD"
  echo "date=$(date -u +%FT%TZ)"
} > "$WORK/TOOLCHAIN.txt"
cat "$WORK/TOOLCHAIN.txt"

# ---- specs ----
python3 "$KIT/lib/gen_spec.py" --farm "$FARMDIR" --board "$BOARD" \
  --work "$WORK" ${OPS:+--ops "$OPS"} || exit 2

# ---- compile-producer per group (ONE session per group: PRODUCE rmtree's
# the artefact dir per session) ----
mkdir -p "$WORK/builds" "$WORK/producer-logs"
rc_all=0
while IFS=$'\x1f' read -r group fk env n; do
  [ "$group" = "group" ] || [ -z "$group" ] && continue
  BUILD=$WORK/builds/$group
  if [ -f "$BUILD/DONE" ]; then echo "produce: skip $group (done)"; continue; fi
  rm -rf "$BUILD"; mkdir -p "$BUILD"
  FLAGS=$(cat "$WORK/flags/$fk.txt")
  ENVARGS=()
  [ -n "$env" ] && IFS=';' read -ra ENVARGS <<< "$env"
  echo "== produce $group: $n nodes (flags=$fk env='$env')"
  mapfile -t NODES < "$WORK/producer-nodes/$group.txt"
  ( cd "$PYDIR" && \
    env ${ENVARGS[@]+"${ENVARGS[@]}"} CHIP_ARCH=blackhole LLK_HOME="$FARMDIR" \
      RUNNER_TEMP="$BUILD" TT_LLK_EXTRA_COMPILER_OPTIONS="$FLAGS" \
      PYTHONUNBUFFERED=1 \
      "$PYBIN" -m pytest -o addopts= -q -n 16 --compile-producer \
      "${NODES[@]}" > "$WORK/producer-logs/$group.log" 2>&1 )
  rc=$?
  tail -1 "$WORK/producer-logs/$group.log"
  if [ $rc -eq 0 ]; then touch "$BUILD/DONE"; else
    echo "PRODUCE-FAIL $group rc=$rc"; rc_all=1
  fi
done < <(sed $'s/\t/\x1f/g' "$WORK/producer-groups.tsv")
[ $rc_all -eq 0 ] || exit 2

# ---- pack the execute-only bundle ----
SHIPD=$WORK/ship
rm -rf "$SHIPD"
mkdir -p "$SHIPD/farm/tt_metal"
rsync -a --copy-links \
  --exclude '.git' --exclude '.venv' --exclude 'tests/sfpi' \
  --exclude 'perf_data' --exclude '__pycache__' \
  "$FARMDIR/" "$SHIPD/farm/tt_metal/tt-llk/"
# runtime binutils only (perf tests exec riscv-tt-elf-size/objdump at RUN
# time for the TEXT_SIZE metric) — the compiler itself never ships
mkdir -p "$SHIPD/farm/tt_metal/tt-llk/tests/sfpi/compiler/bin"
for t in size objdump objcopy nm addr2line readelf strings; do
  cp "$TOOLCHAIN/compiler/bin/riscv-tt-elf-$t" \
     "$SHIPD/farm/tt_metal/tt-llk/tests/sfpi/compiler/bin/"
done
mkdir -p "$SHIPD/builds"
for g in "$WORK"/builds/*/; do
  name=$(basename "$g")
  mkdir -p "$SHIPD/builds/$name"
  rsync -a --exclude DONE "$g/tt-llk-build" "$SHIPD/builds/$name/"
done
cp -r "$WORK/flags" "$SHIPD/flags"
cp "$WORK/ARMS.tsv" "$WORK/ROWS.tsv" "$WORK/TOOLCHAIN.txt" "$SHIPD/"
cp "$KIT/lib/worker.py" "$KIT/lib/seed.py" "$KIT/lib/galaxy_launch.sh" "$SHIPD/"
cp "$FARMDIR/tests/requirements.txt" "$SHIPD/requirements.txt"
tar -C "$SHIPD" -cf - . | zstd -T8 -8 -q -o "$WORK/ship.tar.zst" -f
sha256sum "$WORK/ship.tar.zst" | tee "$WORK/ship.tar.zst.sha256"

[ "$SHIP" = 1 ] || { echo "STAGE-DONE (no-ship)"; exit 0; }

# ---- ship: stream through the relay, untar, venv ----
route_check || exit 2
echo "shipping $(du -h "$WORK/ship.tar.zst" | cut -f1) -> $LK_DEST"
exa_put "$LK_DEST/ship.tar.zst" < "$WORK/ship.tar.zst" || exit 2
REMOTE_SHA=$(exa "cd $LK_DEST && sha256sum ship.tar.zst" | cut -d' ' -f1)
LOCAL_SHA=$(cut -d' ' -f1 "$WORK/ship.tar.zst.sha256")
[ "$REMOTE_SHA" = "$LOCAL_SHA" ] || { echo "REFUSE: ship sha mismatch"; exit 2; }
exa "cd $LK_DEST && (command -v zstd >/dev/null && zstd -dc ship.tar.zst | tar xf - || tar --zstd -xf ship.tar.zst) && echo UNPACKED"
# venv: reuse a compatible existing venv on /data if its imports work,
# else build fresh from the shipped requirements.txt (PyPI reachable)
exa "cd $LK_DEST && if [ -x venv/bin/python ] && venv/bin/python -c 'import torch, ttexalens, pytest' 2>/dev/null; then echo VENV-KEPT; elif [ -x /data/nkapre/craq-bitexact/venv/bin/python ] && /data/nkapre/craq-bitexact/venv/bin/python -c 'import torch, ttexalens, pytest' 2>/dev/null; then ln -sfn /data/nkapre/craq-bitexact/venv venv && echo VENV-REUSED-KC; else python3 -m venv venv && ./venv/bin/pip install -q --upgrade pip && ./venv/bin/pip install -q -r requirements.txt && echo VENV-FRESH; fi; ./venv/bin/python -c 'import torch, ttexalens, pytest; print(\"LK-VENV-OK\")'"
echo "STAGE-DONE dest=$LK_DEST cc1plus=$CCSHA"
