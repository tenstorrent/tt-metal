#!/usr/bin/env bash
# Round 3 eltwise binary: whole test modules with an environment toggle set ("main") and unset ("optin"), every output hashed
# (eb_bits_plugin), outcomes and bits compared per test. usage: bits_env.sh <VAR> <pytest args...>
set -uo pipefail
cd /work
VAR=$1; shift
O=/tmp/ebbenv; mkdir -p $O
export PYTHONPATH=/work:/work/tests/eb_r3_ci:${PYTHONPATH:-}
for v in main optin; do
  if [[ $v == main ]]; then export $VAR=1; else unset $VAR; fi
  export TT_METAL_CACHE=$O/cache_$v EB_HASH_OUT=$O/hash_$v.json; mkdir -p $TT_METAL_CACHE
  timeout -s INT -k 60 ${EB_RUN_LIMIT:-3000} python3 -m pytest -p eb_bits_plugin -p no:cacheprovider -o timeout_method=thread -q -rfE "$@" > $O/log_$v.txt 2>&1
  echo "--- $v rc=$?: $(grep -E 'passed|failed|error' $O/log_$v.txt | tail -1)"
  grep -E "^(FAILED|ERROR)" $O/log_$v.txt | cut -c1-220 | head -40
done
python3 - <<'PY'
import json
a = json.load(open("/tmp/ebbenv/hash_main.json")); b = json.load(open("/tmp/ebbenv/hash_optin.json"))
tests = sorted(set(a["outcome"]) | set(b["outcome"]))
print(f"tests {len(tests)}, same outcome {sum(1 for t in tests if a['outcome'].get(t) == b['outcome'].get(t))}")
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
        print(f"BITS DIFFER {t}: {len(ha or [])} vs {len(hb or [])} outputs")
print(f"bits: identical {ident}, differ {diff}, no outputs {nohash}")
PY
