#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Compare production (TT/TTI) and HAL codegen of one thread for one test.
# usage: compare_codegen.sh <test_x.py> <unpack|math|pack> [--perf] [--match REGEX]
#   --perf         build the perf variants (asserts off, as perf runs) instead of the functional ones
#   --match REGEX  pick the variant whose build.h matches REGEX (default: the first variant built by both)
# Run from a tt-llk worktree. Builds land in private trees, so it is safe next to other agents' builds.
set -euo pipefail
TEST=$1; THREAD=$2; shift 2
PERF=""; MATCH=""
while [[ $# -gt 0 ]]; do case "$1" in --perf) PERF="--perf"; shift ;; --match) MATCH="$2"; shift 2 ;; *) echo "unknown arg $1"; exit 2 ;; esac; done
LLK=$(pwd)
[[ -x "$LLK/.claude/scripts/run_test.sh" || -f "$LLK/.claude/scripts/run_test.sh" ]] || { echo "run from a tt-llk directory"; exit 2; }
OUT=$(mktemp -d /tmp/bh-port-cg.XXXXXX)
OD=$LLK/tests/sfpi/compiler/bin/riscv-tt-elf-objdump
SZ=$LLK/tests/sfpi/compiler/bin/riscv-tt-elf-size
for impl in tt hal; do
  if [[ $impl == hal ]]; then IMPL="hal_llk"; else IMPL=""; fi
  RUNNER_TEMP=$OUT/$impl TT_LLK_DISABLE_ASSERTS=1 TT_LLK_IMPL_INCLUDE_DIRS=$IMPL \
    bash .claude/scripts/run_test.sh compile --worktree "$LLK" --arch blackhole --test "$TEST" $PERF --jobs 6 2>&1 \
    | grep -E "VERDICT|error:" | sed "s#.*/tests/##;s/^/[$impl] /" | sort -u | head -12
done
python3 - "$OUT" "$THREAD" "$MATCH" "$OD" "$SZ" <<'EOF'
import sys, glob, hashlib, os, re, subprocess, difflib
out, thread, match, od, sz = sys.argv[1:]
def idx(r):
    d = {}
    for b in glob.glob(f'{r}/tt-llk-build/sources/*/*/build.h'):
        v = os.path.dirname(b)
        if os.path.exists(f'{v}/elf/{thread}.elf'):
            d[hashlib.md5(open(b, 'rb').read()).hexdigest()] = v
    return d
tt, hal = idx(f'{out}/tt'), idx(f'{out}/hal')
keys = [k for k in tt if k in hal and (not match or re.search(match, open(f'{tt[k]}/build.h').read()))]
if not keys:
    sys.exit(f'no paired variant (tt {len(tt)}, hal {len(hal)} built); see the compile errors above')
print(f'paired variants: {len(keys)}; comparing {os.path.basename(tt[keys[0]])}')
MASK = [(r'^\s*[0-9a-f]+:\s*', ''), (r'<[^>]*>', ''), (r'#.*', ''), (r'\b(a[0-7]|t[0-6]|s[0-9]|s1[01]|ra|sp|gp|zero)\b', 'R'), (r'\b[0-9a-f]{4,}\b', 'ADDR')]
def dis(v):
    txt = subprocess.run([od, '-d', '--no-show-raw-insn', '-C', f'{v}/elf/{thread}.elf'], capture_output=True, text=True).stdout
    m = re.search(r'<run_kernel\(RuntimeParams const&\)>:\n(.*?)(\n\n|\Z)', txt, re.S)
    lines = m.group(1).split('\n') if m else []
    norm = []
    for l in lines:
        for a, b in MASK:
            l = re.sub(a, b, l)
        norm.append(l.strip())
    size = sum(int(x.split()[1]) for x in subprocess.run([sz, '-A', f'{v}/elf/{thread}.elf'], capture_output=True, text=True).stdout.splitlines() if x.startswith('.text'))
    return lines, norm, size
total = {'tt': 0, 'hal': 0}
for k in keys:
    for impl, d in (('tt', tt), ('hal', hal)):
        total[impl] += dis(d[k])[2]
print(f'.text summed over {len(keys)} variants: tt {total["tt"]}  hal {total["hal"]}  ({100 * (total["hal"] / total["tt"] - 1):+.2f}%)')
res = {}
for impl, d in (('tt', tt), ('hal', hal)):
    lines, norm, size = dis(d[keys[0]])
    res[impl] = norm
    mn = {}
    for l in lines:
        mm = re.search(r'\t(tt\w+)', l)
        if mm:
            mn[mm.group(1)] = mn.get(mm.group(1), 0) + 1
    print(f'== {impl}: .text={size} run_kernel_instr={len(lines)} tensix_immediate={sum(mn.values())}')
    print('   ' + ' '.join(f'{v}x{k}' for k, v in sorted(mn.items(), key=lambda x: -x[1])))
diff = list(difflib.unified_diff(res['tt'], res['hal'], 'tt', 'hal', lineterm='', n=1))
open(f'{out}/diff.txt', 'w').write('\n'.join(diff))
print(f'== normalised run_kernel diff ({len(diff)} lines) saved to {out}/diff.txt; first 60:')
print('\n'.join(diff[:60]))
EOF
