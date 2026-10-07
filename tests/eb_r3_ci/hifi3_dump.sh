#!/usr/bin/env bash
# HiFi3 rule dumps: the two sides differ in the math fidelity only, so the code check groups variants without MATH_FIDELITY
[[ -n "${HWLOCK_HELD:-}" || -n "${GITHUB_ACTIONS:-}" || -n "${TT_METAL_MOCK_CLUSTER_DESC_PATH:-}" ]] || { echo "not under hwlock" >&2; exit 2; }
export EB_ELFAB="eltwise_binary_no_bcast=EB_R3_|MATH_FIDELITY eltwise_binary_col_bcast=EB_R3_|MATH_FIDELITY eltwise_binary_scalar_bcast=EB_R3_|MATH_FIDELITY eltwise_binary_row_bcast=EB_R3_|MATH_FIDELITY eltwise_binary_scalar=EB_R3_|MATH_FIDELITY eltwise_binary=EB_R3_|MATH_FIDELITY"
bash tests/eb_r3_ci/dump_run.sh "$1"
