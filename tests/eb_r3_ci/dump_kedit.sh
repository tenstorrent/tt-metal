#!/usr/bin/env bash
# Round 3 eltwise binary (#58818 review): bit dumps of ops whose hand-off is an opt-in define in the kernel .cpp. Side main:
# the listed define lines removed from the kernel files; side PR: the files as committed. Each side in its own process with
# its own kernel cache (the JIT hash does not cover the .cpp text); the module saves (main) and compares (PR).
# usage: dump_kedit.sh <toggle file: "path|define line" per line> <pytest args...>
[[ -n "${HWLOCK_HELD:-}" || -n "${GITHUB_ACTIONS:-}" || -n "${TT_METAL_MOCK_CLUSTER_DESC_PATH:-}" ]] || { echo "not under hwlock" >&2; exit 2; }
cd /work
TOG=$1; shift
export PYTHONPATH=/work:/work/tests/eb_r3_ci:${PYTHONPATH:-}
O=${EB_KEDIT_DIR:-/tmp/ebkedit}; mkdir -p $O
export EB_DUMP_DIR=$O/out
F='(/DUMP|passed|failed|skipped|rror|Traceback|^E |Timeout|timed out/) && !seen[$0]++ {print substr($0, 1, 1500); fflush()}'
declare -a FILES
while IFS='|' read -r KFILE DEFINE; do
  [[ -z "$KFILE" || "$KFILE" == \#* ]] && continue
  FILES+=("$KFILE|$DEFINE"); cp "$KFILE" "$O/$(echo $KFILE | tr / _).orig"
  echo "toggle $KFILE: '$DEFINE' present $(grep -c -x -F "$DEFINE" "$KFILE") time(s)"
done < "$TOG"
restore() { for e in "${FILES[@]}"; do KFILE=${e%%|*}; cp "$O/$(echo $KFILE | tr / _).orig" "$KFILE"; done; }
remove() {
  for e in "${FILES[@]}"; do KFILE=${e%%|*}; DEFINE=${e#*|}
    python3 - "$KFILE" "$DEFINE" <<'PY'
import sys
p, d = sys.argv[1], sys.argv[2]
s = open(p).read().splitlines(keepends=True)
n = len(s)
s = [l for l in s if l.rstrip("\n") != d]
open(p, "w").write("".join(s))
print(f"  removed {n - len(s)} line(s) from {p}")
PY
  done
}
for side in main pr; do
  restore; [[ $side == main ]] && remove
  echo "##### $side $(date -u +%T)"
  export TT_METAL_CACHE=$O/cache_$side; mkdir -p $TT_METAL_CACHE
  st=save; [[ $side == pr ]] && st=cmp
  EB_DUMP_STAGE=$st timeout -s INT -k 60 ${EB_RUN_LIMIT:-3000} python3 -u -m pytest -p no:cacheprovider --timeout=0 -q -s -rfE "$@" < /dev/null 2>&1 | sed -u -E 's/^tests\/eb_r3_ci\/[^ ]* //' | awk "$F"
done
restore
# the toggled kernels compiled to different code on the two sides (same kernel hash, different executable sections)
python3 - "$O" "${FILES[@]}" <<'PY'
import hashlib, os, sys
sys.path.insert(0, "/work/tests/eb_r3_ci")
sys.argv_saved = list(sys.argv)
exec(open("/work/tests/eb_r3_ci/elf_ab.py").read().split("root = sys.argv[1]")[0])
o, files = sys.argv[1], sys.argv[2:]
def elfs(root, name):
    out = {}
    for dp, dn, fn in os.walk(root):
        parts = dp.split(os.sep)
        if "kernels" in parts:
            i = len(parts) - 1 - parts[::-1].index("kernels")
            if len(parts) > i + 2 and parts[i + 1] == name:
                for f in fn:
                    if f.endswith(".elf") and f.startswith("trisc"):
                        out[os.path.relpath(os.path.join(dp, f), os.path.join(root))] = code_hash(os.path.join(dp, f))
    return out
for e in files:
    name = os.path.basename(e.split("|")[0]).rsplit(".", 1)[0]
    m, p = elfs(os.path.join(o, "cache_main"), name), elfs(os.path.join(o, "cache_pr"), name)
    common = sorted(set(m) & set(p))
    ndiff = sum(1 for k in common if m[k] != p[k])
    print(f"DUMP elf {name}: trisc ELFs main {len(m)}, PR {len(p)}, same path {len(common)}, executable sections differ {ndiff}")
PY
echo "##### end $(date -u +%T)"; du -sh $O 2>/dev/null
