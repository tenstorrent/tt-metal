#!/usr/bin/env bash
set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }

: << 'END'
Usage:
  -f TEST        : test command to run (quote if it has spaces)
  -g GOOD_SHA    : known good commit
  -b BAD_SHA     : known bad commit
  -t TIMEOUT     : per-iteration timeout (default 30m)
  -p             : enable Tracy profiling
  -r RETRIES     : number of retries (default 3)
  -n             : non-deterministic mode (run exactly RETRIES, any failure => bad; record pass rate)
END

timeout_duration_iteration="30m"
test=""
good_commit=""
bad_commit=""
tracy_enabled=0
retries=3
nondeterministic=0

while getopts ":f:g:b:t:pr:n" opt; do
  case "$opt" in
    f) test="$OPTARG" ;;
    g) good_commit="$OPTARG" ;;
    b) bad_commit="$OPTARG" ;;
    t) timeout_duration_iteration="$OPTARG" ;;
    p) tracy_enabled=1 ;;
    r) retries="$OPTARG" ;;
    n) nondeterministic=1 ;;
    \?) die "Invalid option: -$OPTARG" ;;
    :)  die "Option -$OPTARG requires an argument." ;;
  esac
done

[ -n "$test" ] || die "Please specify -f TEST."
[ -n "$good_commit" ] || die "Please specify -g GOOD_SHA."
[ -n "$bad_commit" ] || die "Please specify -b BAD_SHA."


echo "TT_METAL_HOME: $TT_METAL_HOME"
echo "PYTHONPATH: $PYTHONPATH"
echo "ARCH_NAME: ${ARCH_NAME:-}"
echo "pwd: $(pwd)"
if [ "$tracy_enabled" -eq 1 ]; then
  echo "Tracy profiling enabled for builds."
fi
if [ "$nondeterministic" -eq 1 ]; then
  echo "Non-deterministic mode enabled: will run exactly $retries attempts per commit and record pass rates."
fi

# Creating virtual environment where we can install ttnn
./create_venv.sh
pip install -r models/tt_transformers/requirements.txt

git cat-file -e "$good_commit^{commit}" 2>/dev/null || die "Invalid good commit: $good_commit"
git cat-file -e "$bad_commit^{commit}" 2>/dev/null  || die "Invalid bad commit: $bad_commit"

echo "Good: $good_commit"
echo "Bad : $bad_commit"
echo "Branch: $(git rev-parse --abbrev-ref HEAD)"
echo "Commit: $(git rev-parse HEAD)"
echo "Status:"
git status --porcelain=v1

# Keep CPM cache across iterations for speed, but ensure it's in the workspace
export CPM_SOURCE_CACHE="${CPM_SOURCE_CACHE:-$TT_METAL_HOME/.cpmcache}"
mkdir -p "$CPM_SOURCE_CACHE"
export CMAKE_ARGS="-DCPM_SOURCE_CACHE=$CPM_SOURCE_CACHE -DCPM_DOWNLOAD_ALL=ON -DCPM_USE_LOCAL_PACKAGES=OFF"

# Helper to ensure repo matches rev and all build products are fresh
fresh_clean() {
  # Match submodules exactly for current rev
  git submodule sync --recursive
  git submodule update --init --recursive --force

  # Nuke build outputs but keep venv/cache
  git reset --hard
  rm -rf build build_Release build_Debug || true
}

# After building, verify we import from the workspace
verify_import_path() {
  python - <<'PY'
import ttnn, sys
print(ttnn.get_arch_name())
print("ttnn imported from:", ttnn.__file__)
PY
}

echo "Starting git bisect…"
git bisect start "$bad_commit" "$good_commit"

found=false
while [[ "$found" == "false" ]]; do
  rev="$(git rev-parse --short=12 HEAD)"
  echo "::group::Building $rev"

  fresh_clean

  build_rc=0
  if [ "$tracy_enabled" -eq 1 ]; then
    ./build_metal.sh \
      --build-all \
      --enable-ccache \
      --enable-profiler || build_rc=$?
  else
    ./build_metal.sh \
      --build-all \
      --enable-ccache || build_rc=$?
  fi

  echo "::endgroup::"

  if [ $build_rc -ne 0 ]; then
    echo "Build failed (rc=$build_rc); skipping this commit"
    git bisect skip
    continue
  fi

  echo "::group::Import sanity ($rev)"
  if ! verify_import_path; then
    echo "Import path check failed; skipping this commit"
    git bisect skip
    echo "::endgroup::"
    continue
  fi
  echo "::endgroup::"

  echo "::group::Testing $rev"
  output_file="bisect_test_output.log"

  # Prepare CSV logging when in non-deterministic mode
  csv_path="generated/bisect_pass_rates.csv"
  if [ "$nondeterministic" -eq 1 ]; then
    mkdir -p "$(dirname "$csv_path")"
    if [ ! -f "$csv_path" ]; then
      echo "commit_sha,short_rev,attempts,successes,pass_rate_percent,result" > "$csv_path"
    fi
  fi

  if [ "$nondeterministic" -eq 1 ]; then
    attempts=$retries
    successes=0
    failures=0
    saw_timeout=0
    for run in $(seq 1 $attempts); do
      echo "Run $run/$attempts: $test"
      if timeout -k 10s "$timeout_duration_iteration" bash -lc "$test" 2>&1 | tee "$output_file"; then
        successes=$((successes + 1))
        echo "Success $successes/$attempts"
      else
        rc=$?
        failures=$((failures + 1))
        echo "Test failed (code $rc)"
        echo "--- Logs (run $run) ---"
        sed -n '1,200p' "$output_file" || true
        echo "------------------------------"
        if [ $rc -eq 124 ] || [ $rc -eq 137 ] || [ $rc -eq 143 ]; then
          saw_timeout=1
        fi
      fi
    done

    pass_rate_percent=$(( successes * 100 / attempts ))
    result="good"
    if [ $failures -gt 0 ]; then
      result="bad"
    fi
    if [ $saw_timeout -eq 1 ]; then
      result="skip"
    fi

    echo "Pass rate for $rev: $successes/$attempts (${pass_rate_percent}%) => $result"
    if [ -f "$csv_path" ]; then
      echo "$(git rev-parse HEAD),$rev,$attempts,$successes,$pass_rate_percent,$result" >> "$csv_path"
    fi
  else
    # Normal mode: stop on first success, up to $retries attempts
    attempts=$retries
    successes=0
    last_rc=1
    for run in $(seq 1 $attempts); do
      echo "Run $run/$attempts: $test"
      if timeout -k 10s "$timeout_duration_iteration" bash -lc "$test" 2>&1 | tee "$output_file"; then
        successes=$((successes + 1))
        last_rc=0
        echo "Success on attempt $run; stopping early"
        echo "--- Final Logs ---"
        sed -n '1,200p' "$output_file" || true
        echo "------------------------------"
        break
      else
        last_rc=$?
        echo "Test failed (code $last_rc)"
        echo "--- Logs (run $run) ---"
        sed -n '1,200p' "$output_file" || true
        echo "------------------------------"
      fi
    done
  fi
  echo "::endgroup::"

  if [ "$nondeterministic" -eq 1 ]; then
    if [ "$result" = "skip" ]; then
      echo "Timeout/kill detected; skipping this commit"
      git bisect skip
      continue
    elif [ "$result" = "good" ]; then
      out="$(git bisect good || true)"
    else
      out="$(git bisect bad || true)"
    fi
  else
    if [ $successes -ge 1 ]; then
      out="$(git bisect good || true)"
    elif [ $last_rc -eq 124 ] || [ $last_rc -eq 137 ] || [ $last_rc -eq 143 ]; then
      echo "Timeout/kill detected; skipping this commit"
      git bisect skip
      continue
    else
      out="$(git bisect bad || true)"
    fi
  fi

  first_line="$(printf '%s\n' "$out" | head -n1)"
  case "$first_line" in
    *"is the first bad commit"*)
      echo "FOUND IT: $first_line"
      found=true
      ;;
    *"There are only 'skip'ped commits left to test."*)
      echo "Bisect inconclusive: only skipped commits left."
      echo "Last bisect output: $first_line"
      break
      ;;
    "")
      echo "git bisect produced no output; stopping to avoid an infinite loop."
      echo "Last bisect output: $first_line"
      break
      ;;
  esac
done

git bisect reset || true
