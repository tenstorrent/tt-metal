#!/usr/bin/env bash
# Configure environment variables for LLK_ASSERT + DEVICE_PRINT debug runs.
#
# Usage (must be sourced, not executed):
#   source tools/setup_llk_assert_env.sh <assert_output_path> <dprint_output_path>
#
# Arguments (both required, no defaults):
#   assert_output_path   Path where tt-triage.py dumps lightweight asserts on dispatch timeout.
#   dprint_output_path   Path where TT_METAL_DPRINT_FILE writes DEVICE_PRINT output.
#
# After sourcing, run the test with:
#   TT_METAL_LLK_ASSERTS=1 pytest <test_path>

_usage() {
    echo "usage: source ${BASH_SOURCE[0]:-setup_llk_assert_env.sh} <assert_output_path> <dprint_output_path>" >&2
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "error: this script must be sourced, not executed." >&2
    _usage
    exit 1
fi

if [[ $# -ne 2 ]]; then
    echo "error: expected 2 arguments, got $#." >&2
    _usage
    return 1
fi

_assert_out="$1"
_dprint_out="$2"

if [[ -z "${TT_METAL_HOME:-}" ]]; then
    echo "error: TT_METAL_HOME is not set; run this from an activated tt-metal environment." >&2
    return 1
fi

_triage="${TT_METAL_HOME}/tools/tt-triage.py"
if [[ ! -f "${_triage}" ]]; then
    echo "error: tt-triage.py not found at ${_triage}." >&2
    return 1
fi

# Shell-quote paths so spaces or metacharacters in the inputs don't break the timeout hook.
printf -v _triage_q '%q' "${_triage}"
printf -v _assert_out_q '%q' "${_assert_out}"
export TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE="python3 ${_triage_q} --run=dump_lightweight_asserts > ${_assert_out_q} && tt-smi -r"
export TT_METAL_OPERATION_TIMEOUT_SECONDS=5.0
export TT_RUN_DISABLED_TRIAGE_SCRIPTS_IN_CI=1
export TT_METAL_DPRINT_CORES=all
export TT_METAL_DEVICE_PRINT=1
export TT_METAL_DPRINT_FILE="${_dprint_out}"

echo "LLK_ASSERT debug env configured:"
echo "  TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE=${TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE}"
echo "  TT_METAL_OPERATION_TIMEOUT_SECONDS=${TT_METAL_OPERATION_TIMEOUT_SECONDS}"
echo "  TT_RUN_DISABLED_TRIAGE_SCRIPTS_IN_CI=${TT_RUN_DISABLED_TRIAGE_SCRIPTS_IN_CI}"
echo "  TT_METAL_DPRINT_CORES=${TT_METAL_DPRINT_CORES}"
echo "  TT_METAL_DEVICE_PRINT=${TT_METAL_DEVICE_PRINT}"
echo "  TT_METAL_DPRINT_FILE=${TT_METAL_DPRINT_FILE}"
echo "Now run: TT_METAL_LLK_ASSERTS=1 pytest <test>"

unset _assert_out _dprint_out _triage _triage_q _assert_out_q
unset -f _usage
