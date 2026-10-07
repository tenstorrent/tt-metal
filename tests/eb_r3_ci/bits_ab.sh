#!/usr/bin/env bash
# Round 3 eltwise binary: run whole test modules on main's kernels and with opt-in defines, hash every output (eb_bits_plugin),
# then compare outcomes and bits per test. usage: bits_ab.sh <optin file: "path|define line" per line> <pytest args...>
[[ -n "${HWLOCK_HELD:-}" || -n "${GITHUB_ACTIONS:-}" || -n "${TT_METAL_MOCK_CLUSTER_DESC_PATH:-}" ]] || { echo "not under hwlock" >&2; exit 2; }
set -uo pipefail
cd /work
OPT=$1; shift
O=/tmp/ebbits; mkdir -p $O
export PYTHONPATH=/work:/work/tests/eb_r3_ci:${PYTHONPATH:-}
declare -a FILES
while IFS='|' read -r KFILE DEFINE; do [[ -z "$KFILE" || "$KFILE" == \#* ]] && continue; FILES+=("$KFILE|$DEFINE"); cp "$KFILE" "$O/$(echo $KFILE | tr / _).orig"; done < "$OPT"
for v in main optin; do
  if [[ $v == optin ]]; then
    for e in "${FILES[@]}"; do KFILE=${e%%|*}; DEFINE=${e#*|}
      python3 - "$KFILE" "$DEFINE" <<'PY'
import sys
p, d = sys.argv[1], sys.argv[2]
s = open(p).read().splitlines(keepends=True)
i = next(k for k, l in enumerate(s) if l.startswith("#include"))
s.insert(i, d + "\n")
open(p, "w").write("".join(s))
PY
    done
  fi
  export TT_METAL_CACHE=$O/cache_$v EB_HASH_OUT=$O/hash_$v.json; mkdir -p $TT_METAL_CACHE
  timeout -s INT -k 60 ${EB_RUN_LIMIT:-3000} python3 -m pytest -p eb_bits_plugin -p no:cacheprovider -o timeout_method=thread -q -rfE "$@" > $O/log_$v.txt 2>&1
  echo "--- $v rc=$?: $(grep -E 'passed|failed|error' $O/log_$v.txt | tail -1)"
  grep -E "^(FAILED|ERROR)" $O/log_$v.txt | cut -c1-220 | head -40
  [[ -n "${EB_SHOW_ERR:-}" ]] && grep -E "error:|TT_THROW|TT_FATAL|Timeout|timed out|^E  " $O/log_$v.txt | sort | uniq -c | sort -rn | head -12 | cut -c1-500
done
for e in "${FILES[@]}"; do KFILE=${e%%|*}; cp "$O/$(echo $KFILE | tr / _).orig" "$KFILE"; done
python3 - <<'PY'
import json
a = json.load(open("/tmp/ebbits/hash_main.json")); b = json.load(open("/tmp/ebbits/hash_optin.json"))
tests = sorted(set(a["outcome"]) | set(b["outcome"]))
same_out = sum(1 for t in tests if a["outcome"].get(t) == b["outcome"].get(t))
print(f"tests {len(tests)}, same outcome {same_out}")
for t in tests:
    if a["outcome"].get(t) != b["outcome"].get(t):
        print(f"OUTCOME DIFFERS {t}: main {a['outcome'].get(t)} optin {b['outcome'].get(t)}")
ident = diff = nohash = 0
for t in tests:
    ha, hb = a["hashes"].get(t), b["hashes"].get(t)
    if not ha and not hb:
        nohash += 1
    elif ha == hb:
        ident += 1
    else:
        diff += 1
        print(f"BITS DIFFER {t}: {len(ha or [])} vs {len(hb or [])} outputs, first differing index {next((i for i, (x, y) in enumerate(zip(ha or [], hb or [])) if x != y), None)}")
print(f"bits: identical {ident}, differ {diff}, no outputs {nohash}")
PY
