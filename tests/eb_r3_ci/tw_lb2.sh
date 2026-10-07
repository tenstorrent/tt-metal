#!/usr/bin/env bash
# Round 3 eltwise binary (#58723 review, twin round): Blackhole LoudBox, ring joint MLA SDPA device time A/B with the opt-in,
# one test per process under tracy (the repetition plugin's extra calls abort at submesh close), JIT root /work, CI unset
# (the tests' uncollect_if keys on it): the 2x4 perf case (one compile launch and five measured) and CI's 4x2 pcc_check case.
[[ -n "${HWLOCK_HELD:-}" || -n "${GITHUB_ACTIONS:-}" || -n "${TT_METAL_MOCK_CLUSTER_DESC_PATH:-}" ]] || { echo "not under hwlock" >&2; exit 2; }
cd /work
export TT_METAL_RUNTIME_ROOT=/work
env -u CI -u TT_GH_CI_INFRA EB_RUN_LIMIT=900 bash tests/eb_r3_ci/ab_plain.sh tests/eb_r3_ci/optin_ring.txt tests/eb_r3_ci/spec_mla_lb.txt 3 "^RingJointSDPA"
python3 tests/eb_r3_ci/elf_set_diff.py /tmp/ebplain/cache_main /tmp/ebplain/cache_optin 2>&1 | grep -v "ELFSET identical"
