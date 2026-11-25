#!/bin/bash
# Determinism control, one leg: build main's code twice in the same full tree; compare ELFs, raw and after --strip-debug.
set -uo pipefail
ROOT=$PWD; T=$ROOT/tt_metal/tt-llk/tests; PT=$T/python_tests; TRIAL=$ROOT/smoke_trial
mkdir -p "$OUT"; BUILD="$RUNNER_TEMP/tt-llk-build"
MARK="not perf and not nightly and not quasar and not accuracy"
SPLIT=(--splits "$NGROUPS" --group "$GROUP"); [[ $ARCH == blackhole ]] && SPLIT+=(--splitting-algorithm least_duration)
cp "$DURATIONS" "$PT/.test_durations"
cp "$TRIAL/main/test_config.py" "$PT/helpers/test_config.py"
cd "$PT"
for b in a b; do
  rm -rf "$BUILD"
  python3 -m pytest -q --override-ini=log_cli=false -n auto --dist=worksteal -m "$MARK" "${SPLIT[@]}" --timeout=60 --compile-producer -p no:cacheprovider -o "python_files=test_*.py" . > "$OUT/build_$b.log" 2>&1
  tail -1 "$OUT/build_$b.log"
  mv "$BUILD" "$RUNNER_TEMP/build-$b"
done
python3 - "$T/sfpi/compiler/bin/riscv-tt-elf-objcopy" "$RUNNER_TEMP" "$OUT" <<'PY'
import hashlib, json, os, subprocess, sys, tempfile
objcopy, temp, out = sys.argv[1:]
def elfs(b):
    r = os.path.join(temp, b)
    return {os.path.relpath(os.path.join(d, f), r) for d, _, fs in os.walk(r) if os.path.basename(d) == "elf" for f in fs if f.endswith(".elf")}
def digest(p, strip):
    if not strip:
        return hashlib.sha256(open(p, "rb").read()).hexdigest()
    with tempfile.NamedTemporaryFile() as t:
        subprocess.run([objcopy, "--strip-debug", p, t.name], check=True)
        return hashlib.sha256(open(t.name, "rb").read()).hexdigest()
A, B = elfs("build-a"), elfs("build-b")
res = {"same_set": A == B, "count": len(A)}
for strip in (False, True):
    d = sorted(e for e in A & B if digest(f"{temp}/build-a/{e}", strip) != digest(f"{temp}/build-b/{e}", strip))
    res["strip" if strip else "raw"] = {"different": len(d), "files": d}
json.dump(res, open(f"{out}/control.json", "w"), indent=1)
print(json.dumps({k: (v if not isinstance(v, dict) else v["different"]) for k, v in res.items()}))
PY
