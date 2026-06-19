#!/usr/bin/env bash
# Sweep N distinct programs (5/10/20/30/50/70/100), measure e2e wall for two configs:
#   A) local-inline   : server OFF, NO precompile -> kernels compile on-demand during the real run.
#                       e2e = the run itself (single device-open; local box ~8 cores).
#   B) remote-farm     : server ON, precompile warm pass @128 on the farm, THEN a warm real run.
#                       e2e = warm_pass + warm_run (TWO device-opens; compile offloaded to 128 cores).
# Both start fully cold (fresh local JIT cache each iteration; server .elf dewarmed for B). Real runs
# DISPATCH on device (rms_norm is stable) under a timeout; a hang resets via the dirty flag. Keepalive
# ON for the farm path. Finer-grained timing via date +%s.%N.
set -u
cd "$(git rev-parse --show-toplevel)"
source python_env/bin/activate 2>/dev/null
export PYTHONPATH="$PWD:$PWD/tt_metal/third_party/tt_ops_code_gen"
C=bgdepyc01-special-mstaletovic-for-reservation-24729
R=tt_metal/impl/jit_server/repro
LCACHE="$PWD/$R/fvi_cache"
RESULTS="$PWD/$R/farm_vs_inline_results.csv"
NS=(5 10 20 30 50 70 100)
TMO="timeout --signal=INT --kill-after=15 300"

export LOGURU_LEVEL=ERROR PYTHONUNBUFFERED=1
# FAIR cold-vs-cold: the login profile sets TT_METAL_CCACHE_KERNEL_SUPPORT=1, which would let the
# local-inline path hit ccache (objects from the firmware prime + nested sets) and understate inline
# time — while the farm path can't use it (preprocess-shipped .ii is uncacheable). Disable ccache on
# the client and point it at an isolated dir we inspect afterward to PROVE it stayed cold.
# NOTE: TT_METAL_CCACHE_KERNEL_SUPPORT=0 does NOT disable ccache (the build wraps `ccache g++`
# unconditionally). Use ccache's OWN kill switch CCACHE_DISABLE=1 -> pure passthrough to the real
# compiler every time (always a full cold compile, no hits/writes). Prove it via the probe dir.
export TT_METAL_CCACHE_KERNEL_SUPPORT=0
export CCACHE_DISABLE=1
export CCACHE_DIR="$PWD/$R/fvi_ccache_probe"; rm -rf "$CCACHE_DIR"; mkdir -p "$CCACHE_DIR"
srv(){ ssh -o ConnectTimeout=10 bgdepyc01 "docker exec $C bash -c '$1'" 2>/dev/null; }
dewarm_local(){ rm -rf "$LCACHE"; mkdir -p "$LCACHE"; }
dewarm_server(){ srv 'find /tmp/tt-metal-cache -type d -name "rms_norm*" -exec rm -rf {} + 2>/dev/null'; }
srv_count(){ srv 'grep -aoE "count=[0-9]+" /localdev/mstaletovic/tt-metal/jit_compile_server.log | tail -1 | grep -oE "[0-9]+"'; }
now(){ date +%s.%N; }
el(){ awk "BEGIN{printf \"%.1f\", $2-$1}"; }

# config A: local inline real run (server off, no precompile). echoes elapsed seconds.
run_inline(){ local set="$1" log="$2" t0 t1
    export TT_METAL_CACHE="$LCACHE"
    unset TT_METAL_JIT_SERVER_ENABLE TT_METAL_JIT_SERVER_ENDPOINT TT_METAL_JIT_PREPROCESS
    t0=$(now); $TMO pytest $(cat "$set") -p no:cacheprovider -q >"$log" 2>&1; local rc=$?; t1=$(now)
    [[ $rc -eq 124 || $rc -eq 137 ]] && { echo "HANG"; return; }
    el "$t0" "$t1"
}
# config B step 1: farm warm pass (server on @128, NO_DISPATCH collect+compile).
run_warm(){ local set="$1" log="$2" t0 t1
    export TT_METAL_CACHE="$LCACHE"
    export TT_METAL_JIT_SERVER_ENABLE=1 TT_METAL_JIT_SERVER_ENDPOINT=bgdepyc01:54210 TT_METAL_JIT_PREPROCESS=1
    export TT_METAL_JIT_SERVER_KEEPALIVE=1 TT_METAL_JIT_SERVER_TIMEOUT_S=120
    export UP_FRONT_COLLECT=1 UP_FRONT_REAL_ALLOC=1 UP_FRONT_COLLECT_WORKERS=128
    t0=$(now); $TMO pytest $(cat "$set") -p tests.plugins.up_front_collect -p no:cacheprovider -q >"$log" 2>&1; t1=$(now)
    unset UP_FRONT_COLLECT UP_FRONT_REAL_ALLOC UP_FRONT_COLLECT_WORKERS
    el "$t0" "$t1"
}
# config B step 2: warm real run (server OFF; cache is warm on disk).
run_warm_run(){ local set="$1" log="$2" t0 t1
    export TT_METAL_CACHE="$LCACHE"
    unset TT_METAL_JIT_SERVER_ENABLE TT_METAL_JIT_SERVER_ENDPOINT TT_METAL_JIT_PREPROCESS
    t0=$(now); $TMO pytest $(cat "$set") -p no:cacheprovider -q >"$log" 2>&1; local rc=$?; t1=$(now)
    [[ $rc -eq 124 || $rc -eq 137 ]] && { echo "HANG"; return; }
    el "$t0" "$t1"
}

echo "N,programs,kernels,inline_s,farm_warm_s,farm_run_s,farm_total_s,fallbacks,speedup" > "$RESULTS"
exec 9>/tmp/tt-device.lock; echo "FVI: acquiring lock"; flock 9; echo "FVI: lock acquired"; touch /tmp/tt-device.dirty
dewarm_local; dewarm_server
# prime firmware (build once, not counted)
echo "FVI: priming firmware..."; run_inline "$R/sets/p5.txt" "$R/fvi_prime.log" >/dev/null

for N in "${NS[@]}"; do
    SET="$R/sets/p${N}.txt"
    # --- A: local inline ---
    dewarm_local
    T_IN=$(run_inline "$SET" "$R/fvi_inline_${N}.log")
    PROG=$(grep -aoE "[0-9]+ passed" "$R/fvi_inline_${N}.log" | tail -1 | grep -oE "[0-9]+")
    # --- B: remote farm (warm pass + warm run) ---
    dewarm_local; dewarm_server
    c0=$(srv_count)
    T_WARM=$(run_warm "$SET" "$R/fvi_warm_${N}.log")
    c1=$(srv_count)
    KERN=$((c1 - c0)); [[ $c1 -lt $c0 ]] && KERN=NA
    FB=$(grep -ac "falling back to local" "$R/fvi_warm_${N}.log")
    UPROG=$(grep -aoE "[0-9]+ unique programs" "$R/fvi_warm_${N}.log" | head -1 | grep -oE "^[0-9]+")
    T_RUN=$(run_warm_run "$SET" "$R/fvi_warmrun_${N}.log")
    T_FARM=$(awk "BEGIN{printf \"%.1f\", $T_WARM+$T_RUN}")
    SPD=$(awk "BEGIN{if($T_FARM>0)printf \"%.2f\", $T_IN/$T_FARM; else print \"NA\"}")
    echo "${N},${UPROG:-$PROG},${KERN},${T_IN},${T_WARM},${T_RUN},${T_FARM},${FB},${SPD}" >> "$RESULTS"
    echo "FVI: N=$N prog=${UPROG:-$PROG} kern=$KERN | inline=${T_IN}s | farm warm=${T_WARM}s + run=${T_RUN}s = ${T_FARM}s | fb=$FB | speedup=${SPD}x"
done
rm -rf "$LCACHE"
echo "FVI: DONE"; column -t -s, "$RESULTS"
