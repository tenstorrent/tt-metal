#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary: run whole test modules with the listed kernels' ELTWISE_BINARY_PER_TILE_HANDOFF* defines removed
# ("main") and as committed ("optin"), hash every output (eb_bits_plugin), compare outcomes and bits per test. The JIT reads
# every file from /work. usage: bits_strip.sh <file: one kernel path per line> <pytest args...>
set -uo pipefail
cd /work
OPT=$1; shift
O=/tmp/ebbits; mkdir -p $O
export PYTHONPATH=/work:/work/tests/eb_r3_ci:${PYTHONPATH:-}
declare -a FILES
export TT_METAL_RUNTIME_ROOT=/work
while IFS='|' read -r KFILE DEFINE; do [[ -z "$KFILE" || "$KFILE" == \#* ]] && continue; FILES+=("$KFILE"); cp "$KFILE" "$O/$(echo $KFILE | tr / _).orig"; done < "$OPT"
for v in main optin; do
  for KFILE in "${FILES[@]}"; do
    if [[ $v == main ]]; then grep -v "^#define ELTWISE_BINARY_PER_TILE_HANDOFF" "$O/$(echo $KFILE | tr / _).orig" > "$KFILE"; else cp "$O/$(echo $KFILE | tr / _).orig" "$KFILE"; fi
  done
  export TT_METAL_CACHE=$O/cache_$v EB_HASH_OUT=$O/hash_$v.json; mkdir -p $TT_METAL_CACHE
  timeout -s INT -k 60 ${EB_RUN_LIMIT:-3000} python3 -m pytest -p eb_bits_plugin -p no:cacheprovider -o timeout_method=thread -q -rfE "$@" > $O/log_$v.txt 2>&1
  echo "--- $v rc=$?: $(grep -E 'passed|failed|error' $O/log_$v.txt | tail -1)"
  grep -E "^(FAILED|ERROR)" $O/log_$v.txt | cut -c1-220 | head -40
  [[ -n "${EB_SHOW_ERR:-}" ]] && grep -E "error:|TT_THROW|TT_FATAL|Timeout|timed out|^E  " $O/log_$v.txt | sort | uniq -c | sort -rn | head -12 | cut -c1-500
done
for KFILE in "${FILES[@]}"; do cp "$O/$(echo $KFILE | tr / _).orig" "$KFILE"; done
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
