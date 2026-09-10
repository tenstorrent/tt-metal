#!/usr/bin/env bash
# Sweep pack-loop alignment on Versim: N nops before the loop, one deterministic
# TILE_LOOP value per build.
#
# Silicon is bistable: it picks one of two states at run start, and code alignment
# decides whether the fast state is reachable. Versim is deterministic, so the same
# effect shows as "some 16-byte-aligned placements give a fast value, others a slow
# one". The first pass found 2,772 at N=0..3 (small wobble) and 2,708 at N=4.
#
# Self-contained: creates the worktree, links the toolchain and venv, applies the
# config-B / loop-factor-16 / L1_TO_L1-only edits, then loops over N.
#
# usage: versim_nop_sweep.sh [N_MAX]        (default 12; sweeps 0..N_MAX)
#        NOPS="0 4 8 12" versim_nop_sweep.sh  (explicit list)
set -uo pipefail
say() { echo "=== $* -- $(date -u +%H:%M:%SZ) ==="; }

MAIN=~/tt-metal; WT=~/versim-wt; OUT=~/versim-sweep
VERSIM="${VERSIM:-/proj_sw/user_dev/lpremovic/tt-umd-simulators/build/versim-wormhole-b0}"
CFG_INDEX="${CFG_INDEX:-5742}"; LF="${LF:-16}"
N_MAX="${1:-12}"; NOPS="${NOPS:-$(seq -s' ' 0 "$N_MAX")}"
[ -f "$VERSIM/run.sh" ] || { echo "FATAL: no Versim build at $VERSIM"; exit 1; }
mkdir -p "$OUT"

# ---- worktree of the harness branch, with the main tree's toolchain and venv ----
if [ ! -d "$WT/.git" ] && [ ! -f "$WT/.git" ]; then
    say "creating worktree $WT"
    ( cd "$MAIN" && git worktree prune && git fetch -q origin lpremovic/versim-harness \
        && git worktree add -q "$WT" origin/lpremovic/versim-harness ) \
        || { echo "FATAL: worktree"; exit 1; }
fi
LLK="$WT/tt_metal/tt-llk"; PT="$LLK/tests/python_tests"; SRC="$LLK/tests/sources"
[ -e "$LLK/tests/sfpi" ]  || ln -s "$MAIN/tt_metal/tt-llk/tests/sfpi"  "$LLK/tests/sfpi"
[ -e "$LLK/tests/.venv" ] || ln -s "$MAIN/tt_metal/tt-llk/tests/.venv" "$LLK/tests/.venv"
PY="$LLK/tests/.venv/bin/python"

# ---- always start from the branch's pristine files ----
cd "$LLK" && git checkout -- tests/ 2>/dev/null
trap 'cd "$LLK" && git checkout -- tests/ 2>/dev/null; echo "=== restored ==="' EXIT

# ---- harness edits: one config, loop factor, L1_TO_L1 only, long simulator timeouts ----
cd "$PT"
sed -i "s/^@pytest.mark.perf\$/ALL_TEST_PARAMS = [ALL_TEST_PARAMS[$CFG_INDEX]]\n\n@pytest.mark.perf/" perf_math_matmul.py
sed -i "s/^            LOOP_FACTOR(1024),\$/            LOOP_FACTOR($LF),/" perf_math_matmul.py
sed -i '/PerfRunType.UNPACK_ISOLATE,$/d; /PerfRunType.MATH_ISOLATE,$/d; /PerfRunType.PACK_ISOLATE,$/d; /PerfRunType.L1_CONGESTION,$/d' perf_math_matmul.py
sed -i "s/return 600 if self.run_simulator else 2/return 7200 if self.run_simulator else 2/; s/return 600 if self.run_simulator else 1/return 7200 if self.run_simulator else 1/" helpers/target_config.py
grep -q "ALL_TEST_PARAMS\[$CFG_INDEX\]" perf_math_matmul.py || { echo "FATAL: config sed"; exit 1; }
grep -q "LOOP_FACTOR($LF),"              perf_math_matmul.py || { echo "FATAL: loop factor sed"; exit 1; }
[ "$(grep -c 'PerfRunType\.' perf_math_matmul.py)" -ge 1 ] || { echo "FATAL: run types"; exit 1; }
grep -q "return 7200"                    helpers/target_config.py || { echo "FATAL: timeout sed"; exit 1; }
say "config index $CFG_INDEX, loop factor $LF, L1_TO_L1 only"
"$PY" - <<'PY'
import sys; sys.path.insert(0, '.')
from perf_math_matmul import ALL_TEST_PARAMS
fid, mc, thr = ALL_TEST_PARAMS[0]; td, fl = mc.tile_dimensions, mc.face_layout_config
print(f"  {fid}  {mc.formats}  {mc.dest_acc}  {mc.dest_sync}  dst {mc.dst_index}")
print(f"  tile_cnt {td.tile_cnt}  ct {td.ct_dim}  in0_tile_r {td.in0_tile_r_dim}  partial_math {fl.partial_face_math}")
PY

export CHIP_ARCH=wormhole LLK_HOME="$LLK" TT_METAL_SIMULATOR="$VERSIM"
export TT_SIMULATOR_LOCALHOST=1 NNG_SOCKET_NAME="llk_$$" RUNNER_TEMP="$HOME/llk-versim-build"
export USER=gitlab-ci    # no VCD

RESULTS="$OUT/results.csv"
[ -f "$RESULTS" ] || echo "nops,bytes,text_size,tile_loop,kernel,init,seconds" > "$RESULTS"

KERNEL_ANCHOR='_llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();'
for N in $NOPS; do
    say "N=$N  ($((4*N)) bytes)"
    cd "$LLK" && git checkout -- "$SRC/math_matmul_perf.cpp"
    for _ in $(seq "$N"); do
        sed -i "/$KERNEL_ANCHOR/a asm volatile(\"nop\");" "$SRC/math_matmul_perf.cpp"
    done
    GOT=$(grep -c 'asm volatile("nop")' "$SRC/math_matmul_perf.cpp")
    [ "$GOT" -eq "$N" ] || { echo "FATAL: wanted $N nops, file has $GOT"; exit 1; }

    cd "$PT"; rm -rf "$RUNNER_TEMP/tt-llk-build"
    T0=$SECONDS
    "$PY" -u -m pytest --run-simulator -p no:randomly -q -m perf perf_math_matmul.py > "$OUT/n${N}.log" 2>&1
    RC=$?; SECS=$((SECONDS-T0))
    if [ $RC -ne 0 ]; then
        say "N=$N FAILED rc=$RC after ${SECS}s -- see $OUT/n${N}.log"
        tail -5 "$OUT/n${N}.log"; continue
    fi
    F=$(ls -td "$LLK"/perf_data/runs/*/ | head -1)perf_math_matmul/perf_math_matmul.csv
    "$PY" - "$F" "$N" "$SECS" "$RESULTS" <<'PY'
import sys, pandas as pd
f, n, secs, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
d = pd.read_csv(f, low_memory=False).set_index("marker")
tl, ke, ini = (int(d.loc[m, "mean(L1_TO_L1)"]) for m in ("TILE_LOOP", "KERNEL", "INIT"))
ts = int(d.loc["TILE_LOOP", "TEXT_SIZE(L1_TO_L1)"])
open(out, "a").write(f"{n},{4*n},{ts},{tl},{ke},{ini},{secs}\n")
print(f"  N={n:2d}  text {ts}  TILE_LOOP {tl}  KERNEL {ke}  INIT {ini}  ({secs}s)")
PY
    mv -f versim_*.log "$OUT/" 2>/dev/null
done

say DONE
echo
"$PY" - "$RESULTS" <<'PY'
import sys, pandas as pd
d = pd.read_csv(sys.argv[1]).drop_duplicates("nops", keep="last").sort_values("nops")
d["pos16"] = (4 * d.nops) % 16
base = d.loc[d.nops == 0, "tile_loop"].iloc[0] if (d.nops == 0).any() else d.tile_loop.iloc[0]
d["vs_N0"] = d.tile_loop - base
d["step_ok"] = (d.text_size.diff().fillna(4) == 4)
print(d[["nops", "bytes", "pos16", "text_size", "step_ok", "tile_loop", "vs_N0", "kernel", "seconds"]].to_string(index=False))
lo, hi = d.tile_loop.min(), d.tile_loop.max()
print(f"\nTILE_LOOP range {lo}..{hi}  ({(hi-lo)/hi*100:.2f}%)")
fast = d[d.tile_loop < lo + (hi - lo) / 2]
print(f"fast placements (nops): {sorted(fast.nops.tolist())}")
print(f"slow placements (nops): {sorted(d[~d.index.isin(fast.index)].nops.tolist())}")
print("\nRead: pos16 groups the same alignment class. A ~13-cycle wobble across classes")
print("is fetch alignment. A ~60-cycle drop at some aligned slots and not others is the")
print("silicon pattern: alignment is necessary but not sufficient for the fast state.")
PY
