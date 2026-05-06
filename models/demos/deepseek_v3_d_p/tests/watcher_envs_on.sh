# Mirror the watcher + auto-triage env vars used by CI's t3000-apc-fast-tests
# job (run-id 25378509226, setup-job action with `enable-watcher: true` and
# `auto-triage: true`).
#
# Usage:  source models/demos/deepseek_v3_d_p/tests/watcher_envs_on.sh
#   then: ./models/demos/deepseek_v3_d_p/tests/run_mla_stability_100k.sh ...
#
# Pair with watcher_envs_off.sh to unset everything.
#
# This file MUST be sourced — it mutates the current shell's environment.
# `TT_METAL_WATCHER=2` causes the next pytest run to JIT-recompile kernels
# with `-DWATCHER_ENABLED` (so ASSERT() macros become real assert_and_hang
# calls). First run will be slower because of the recompile.

(return 0 2>/dev/null) || {
    echo "ERROR: $0 must be sourced, not executed."
    echo "  Use:  source $0   (or:  . $0)"
    exit 1
}

# --- Watcher group ----------------------------------------------------------
# Polling interval in seconds. 2 matches CI.
export TT_METAL_WATCHER=2
# Append to existing watcher log instead of overwriting (preserves history
# across iterations of a stability loop).
export TT_METAL_WATCHER_APPEND=1
# Disable kernel-function inlining so post-hang stack traces are readable.
export TT_METAL_WATCHER_NOINLINE=1

# Deliberately NOT setting TT_METAL_WATCHER_DISABLE_ETH (CI sets it to 1).
# Our local hangs converge on erisc kernels (fabric_erisc_router); leaving
# Eth-core watcher enabled is the whole point of running this locally.
unset TT_METAL_WATCHER_DISABLE_ETH

# --- Auto-triage on dispatch timeout ---------------------------------------
# 5s host-side dispatch timeout matches CI. When the dispatcher hits this,
# it runs the command in TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE.
export TT_METAL_OPERATION_TIMEOUT_SECONDS=5
export TT_TRIAGE_ENABLE_AGGREGATED_CALLSTACKS=1

# Local equivalent of CI's hang_report.py + tt-triage pipeline (CI's exact
# command bakes in /__w/... paths that don't exist locally). Drops triage
# output into ./triage_output.txt next to the cwd you launched pytest from.
export TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE='./tools/tt-triage.py --disable-progress --triage-summary-path=triage_summary.txt 2>&1 | tee triage_output.txt 1>&2'

echo "Watcher envs ON:"
echo "  TT_METAL_WATCHER=${TT_METAL_WATCHER}"
echo "  TT_METAL_WATCHER_APPEND=${TT_METAL_WATCHER_APPEND}"
echo "  TT_METAL_WATCHER_NOINLINE=${TT_METAL_WATCHER_NOINLINE}"
echo "  TT_METAL_WATCHER_DISABLE_ETH (unset; eth cores will be watched)"
echo "  TT_METAL_OPERATION_TIMEOUT_SECONDS=${TT_METAL_OPERATION_TIMEOUT_SECONDS}"
echo "  TT_TRIAGE_ENABLE_AGGREGATED_CALLSTACKS=${TT_TRIAGE_ENABLE_AGGREGATED_CALLSTACKS}"
echo "  TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE=<set>"
echo "Note: first pytest run will recompile kernels with -DWATCHER_ENABLED."
