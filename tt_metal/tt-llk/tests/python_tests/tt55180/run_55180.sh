#!/bin/bash
# tt-metal #55180 -- two-arm deterministic reproducer.
#
#   ARM_FIX   TT55180_REPRO_NOPS=6 TT55180_FIX_UNPACK_STALL=1   -> both runs pass
#   ARM_HANG  TT55180_REPRO_NOPS=6 TT55180_FIX_UNPACK_STALL=0   -> run 2 hangs
#
# Protocol: run the FIX arm first and time it, then give the HANG arm a budget of
# 2x that per-run time. Exceeding the budget IS the hang signal, so nothing depends
# on parsing a harness timeout message.
#
# THE TWO RUNS MATTER. The hang is on the SECOND run after a device reset; run 1
# always passes. A single run from a cold device CANNOT reproduce this, so a
# simulator session must execute the kernel twice without restarting in between.
#
# Usage:   ./run_55180.sh [--sim] [--out DIR]
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTEST_DIR="$(cd "$HERE/.." && pwd)"
LLK_ROOT="$(cd "$PYTEST_DIR/../.." && pwd)"
SIM=0; OUT="$HERE/results"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sim) SIM=1; shift ;;
        --out) OUT="$2"; shift 2 ;;
        *) echo "usage: $0 [--sim] [--out DIR]" >&2; exit 2 ;;
    esac
done
mkdir -p "$OUT"
REPORT="$OUT/report.txt"; : > "$REPORT"
log() { echo "$*" | tee -a "$REPORT"; }

export CHIP_ARCH="${CHIP_ARCH:-blackhole}"
export LLK_HOME="$LLK_ROOT"
export TT_METAL_SLOW_DISPATCH_MODE=1
[[ -x "$LLK_ROOT/tests/.venv/bin/python3" ]] && export PATH="$LLK_ROOT/tests/.venv/bin:$PATH"
export PYTHONPATH="$PYTEST_DIR${PYTHONPATH:+:$PYTHONPATH}"
OD="$LLK_ROOT/tests/sfpi/compiler/bin/riscv-tt-elf-objdump"
CASE='test_fused.py::test_fuser[tiny_tiles_multi_op_op1]'
CORE="${CORE:-0,0}"     # TENSIX_LOCATION is R,C -- NOT the X-Y form

reset_device() {
    if (( SIM )); then
        log "  [sim] NOT resetting: on a simulator both runs must share ONE session."
    else
        tt-smi -r >/dev/null 2>&1; sleep 3
    fi
}

# Classify from a CAPTURED string. Never `| grep -q`: in a pipeline the match is
# unreliable (it has been observed to miss a "Timeout reached" that is present),
# and a two-way matcher silently scores every failure as a pass.
run_once() {   # $1 = timeout seconds; $2 = label -> prints "verdict seconds"
    local budget="$1" label="$2" out rc t0 t1
    t0=$(date +%s.%N)
    out=$(cd "$PYTEST_DIR" && timeout "$budget" python3 -m pytest --compile-consumer -q -p no:randomly "$CASE" 2>&1); rc=$?
    t1=$(date +%s.%N)
    printf '%s\n' "$out" > "$OUT/$label.log"
    local secs; secs=$(python3 -c "print(f'{$t1-$t0:.1f}')")
    if   (( rc == 124 ));                          then echo "TIMEOUT $secs"
    elif grep -q "Timeout reached" <<<"$out";      then echo "hang $secs"
    elif grep -q "1 passed"        <<<"$out";      then echo "pass $secs"
    else                                                echo "OTHER $secs"; fi
}

build_arm() {  # $1 = fix flag
    export TT_LLK_EXTRA_DEFINES="-DTT55180_REPRO_NOPS=6 -DTT55180_FIX_UNPACK_STALL=$1"
    ( cd "$PYTEST_DIR" && flock /tmp/ttnop-build-$CHIP_ARCH.lock \
        python3 -m pytest --compile-producer -q "$CASE" ) > "$OUT/compile_fix$1.log" 2>&1
    local rc=$?
    local elf; elf=$(ls -t /tmp/tt-llk-build/sources/fused_tests/tiny_tiles_multi_op_op1.cpp/*/elf/unpack.elf 2>/dev/null | head -1)
    # VALIDATE THE LEVER. A silently-unarmed arm looks exactly like a working fix.
    local nops sw
    nops=$("$OD" -d "$elf" 2>/dev/null | grep -c ttunpacr_nop)
    sw=$("$OD"   -d "$elf" 2>/dev/null | grep -c ttstallwait)
    local cond
    cond=$("$OD" -d "$elf" 2>/dev/null | grep -m1 -oE "ttstallwait[[:space:]]+[0-9]+,[0-9]+" | grep -oE "[0-9]+,[0-9]+")
    log "  build(fix=$1) rc=$rc  elf_unpacr_nops=$nops (want 6)  ttstallwait=$sw  cond=$cond"
    (( rc == 0 )) || { log "  !! COMPILE FAILED -- see $OUT/compile_fix$1.log"; return 1; }
    # 8,1024 = STALL_UNPACK / TRISC_CFG (shipped). 8,1030 = + UNPACK (the fix).
    local want; [[ "$1" == "1" ]] && want="8,1030" || want="8,1024"
    [[ "$cond" == "$want" ]] || { log "  !! WRONG ARM: stallwait is '$cond', expected '$want'"; return 1; }
    [[ "$nops" == "6" ]] || { log "  !! LEVER NOT ARMED -- aborting"; return 1; }
    echo "$elf"
}

# Optional: set TT55180_DUMP_HOOK to your own capture script to snapshot device
# state at each step. Called as: $TT55180_DUMP_HOOK <outfile> <core>. Unset = skip.
dump() {
    [[ -n "${TT55180_DUMP_HOOK:-}" && -x "${TT55180_DUMP_HOOK}" ]] || return 0
    "$TT55180_DUMP_HOOK" "$OUT/$1.json" "$CORE" >/dev/null 2>&1 \
        && log "  captured $1.json" || log "  (capture failed: $1)"
}

log "=== #55180 two-arm reproducer ($(date -u +%FT%TZ)) ==="
log "arch=$CHIP_ARCH sim=$SIM core=$CORE out=$OUT"

log ""
log "--- ARM_FIX (TT55180_FIX_UNPACK_STALL=1): expect pass / pass ---"
build_arm 1 >/dev/null || exit 1
reset_device; dump fix_A_afterreset
r=$(run_once 300 fix_run1); f1v=${r% *}; f1s=${r#* }; log "  run1: $f1v (${f1s}s)"
dump fix_B_afterrun1
r=$(run_once 300 fix_run2); f2v=${r% *}; f2s=${r#* }; log "  run2: $f2v (${f2s}s)"
dump fix_C_afterrun2

BUDGET=$(python3 -c "import math;print(max(60,int(math.ceil(2*max($f1s,$f2s)))))")
log ""
log "--- ARM_HANG (TT55180_FIX_UNPACK_STALL=0): expect pass / HANG ---"
log "  per-run budget = 2x the FIX arm's slowest run = ${BUDGET}s"
build_arm 0 >/dev/null || exit 1
reset_device; dump hang_A_afterreset
r=$(run_once "$BUDGET" hang_run1); h1v=${r% *}; h1s=${r#* }; log "  run1: $h1v (${h1s}s)"
dump hang_B_afterrun1
r=$(run_once "$BUDGET" hang_run2); h2v=${r% *}; h2s=${r#* }; log "  run2: $h2v (${h2s}s)"
dump hang_C_afterrun2

log ""
log "=== VERDICT ==="
log "  ARM_FIX : run1=$f1v run2=$f2v"
log "  ARM_HANG: run1=$h1v run2=$h2v   (budget ${BUDGET}s)"
if [[ "$f1v" == pass && "$f2v" == pass && "$h1v" == pass && ( "$h2v" == hang || "$h2v" == TIMEOUT ) ]]; then
    log "  REPRODUCED: the fix arm passes twice; the unfixed arm hangs on run 2."
else
    log "  NOT the expected pattern -- read $OUT/*.log before drawing any conclusion."
fi
log ""
log "full report: $REPORT"
