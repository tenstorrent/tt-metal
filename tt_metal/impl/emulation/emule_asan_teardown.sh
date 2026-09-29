# =============================================================================
# emule_asan_teardown.sh — undo everything emule_asan_setup.sh did
#
#   USAGE:  source tt_metal/impl/emulation/emule_asan_teardown.sh
#           (source it — do NOT execute — so the environment changes persist)
#
# Restores every variable emule_asan_setup.sh exported to its pre-setup value
# (or unsets it if it was not set before), and removes the emule_preflight /
# emule_postflight helper functions — as if the setup script was never sourced.
# =============================================================================

# Refuse to be executed instead of sourced (env changes would be lost).
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    echo "ERROR: source this file, do not execute it:  source tt_metal/impl/emulation/emule_asan_teardown.sh"
    exit 1
fi

_EMULE_ASAN_ENV_VARS="TT_METAL_HOME ARCH_NAME TT_METAL_EMULE_MODE TT_METAL_SLOW_DISPATCH_MODE TT_METAL_MOCK_CLUSTER_DESC_PATH TT_METAL_EMULE_ASAN"

if [ -n "${_EMULE_ASAN_ENV_SAVED:-}" ]; then
    # Setup was sourced in this shell: restore each variable to its snapshot.
    for _v in $_EMULE_ASAN_ENV_VARS; do
        _had="_EMULE_ASAN_HAD_$_v"
        _saved="_EMULE_ASAN_SAVED_$_v"
        if [ "${!_had:-}" = "1" ]; then
            export "$_v=${!_saved}"
        else
            unset "$_v"
        fi
        unset "$_had" "$_saved"
    done
    unset _v _had _saved _EMULE_ASAN_ENV_SAVED
else
    # No snapshot (setup not sourced in this shell). Unset only the emule-
    # specific variables; leave TT_METAL_HOME / ARCH_NAME alone since they are
    # part of the normal tt-metal environment and their prior values are unknown.
    echo "[teardown] note: no setup snapshot found in this shell; unsetting emule vars only (TT_METAL_HOME / ARCH_NAME left untouched)"
    unset TT_METAL_EMULE_MODE TT_METAL_SLOW_DISPATCH_MODE TT_METAL_MOCK_CLUSTER_DESC_PATH TT_METAL_EMULE_ASAN
fi
unset _EMULE_ASAN_ENV_VARS

unset -f emule_preflight emule_postflight 2>/dev/null

echo "[teardown] emule + ASAN environment removed."
