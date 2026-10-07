#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary: the binary tests that draw their scalar from Python's random unseeded, seeded, HiFi4 against the HiFi2 rule.
cd /work
bash tests/eb_r3_ci/bits_env.sh EB_R3_NO_HIFI2 -p eb_seed_plugin tests/ttnn/unit_tests/operations/eltwise/test_binary_scalar.py tests/ttnn/unit_tests/operations/eltwise/test_binaryng_ND.py -k "test_ND_scalar_bcast or test_ND_subtile_bcast or test_binary_scalar_ops"
