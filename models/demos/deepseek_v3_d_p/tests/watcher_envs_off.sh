# Unset everything watcher_envs_on.sh exported.
#
# Usage:  source models/demos/deepseek_v3_d_p/tests/watcher_envs_off.sh
#
# Pair with watcher_envs_on.sh.
#
# This file MUST be sourced — it mutates the current shell's environment.

(return 0 2>/dev/null) || {
    echo "ERROR: $0 must be sourced, not executed."
    echo "  Use:  source $0   (or:  . $0)"
    exit 1
}

unset TT_METAL_WATCHER
unset TT_METAL_WATCHER_APPEND
unset TT_METAL_WATCHER_NOINLINE
unset TT_METAL_WATCHER_DISABLE_ETH
unset TT_METAL_OPERATION_TIMEOUT_SECONDS
unset TT_TRIAGE_ENABLE_AGGREGATED_CALLSTACKS
unset TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE

echo "Watcher envs OFF (all related vars unset)."
