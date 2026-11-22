#!/bin/bash
# C0 item 4 (torrentci#3525, design r2.1): replay the parity gaps G1-G6 on the
# ci-build image's clang-tidy-23 through ctcache 1.2.0. For each case: cache
# the clean TU, apply the edit, then record whether ctcache hits (same
# digest, verdict "clean") and what clang-tidy-23 itself says. A known pass is
# "same digest, cached clean, real verdict fails". A control edit must miss.
set -uo pipefail
tools=$(cd "$(dirname "$0")" && pwd)
tidy=${C0_TIDY:-clang-tidy-23}
cxx=${C0_CXX:-clang++-20}
command -v "$tidy" >/dev/null || { echo "no $tidy"; exit 1; }
command -v "$cxx" >/dev/null || { echo "no $cxx"; exit 1; }
"$tidy" --version | head -2
"$cxx" --version | head -1
root=/tmp/c0-parity
rm -rf "$root"
export CTCACHE_LOCAL=1 CTCACHE_KEEP_COMMENTS=1 CTCACHE_NO_LOCAL_STATS=1
failures=0
TIDY_EXTRA=()

setup() { # <case>
  dir="$root/$1"
  mkdir -p "$dir/inc" "$dir/third_party"
  cat >"$dir/.clang-tidy" <<'EOF'
Checks: '-*,bugprone-macro-parentheses,bugprone-reserved-identifier,modernize-use-nullptr,clang-diagnostic-unused-variable'
WarningsAsErrors: '*'
HeaderFilterRegex: '.*'
ExcludeHeaderFilterRegex: 'third_party'
EOF
  export CTCACHE_DIR="$dir/cache" C0_LOG="$dir/log.jsonl"
}

run_cached() { # <flags...>; prints "rc digest hit"
  (cd "$dir" && "$tools/ctcache-shim.py" "$tidy" --warnings-as-errors='*' ${TIDY_EXTRA[@]+"${TIDY_EXTRA[@]}"} src.cpp -- \
    "$cxx" -std=c++20 -Iinc -Ithird_party "$@" -o src.o -c src.cpp >/dev/null 2>&1)
  local rc=$?
  python3 -c 'import json,sys; r=json.loads(open(sys.argv[1]).read().splitlines()[-1]); print(sys.argv[2], r["digest"], r["hit"])' "$C0_LOG" "$rc"
}

run_real() { # <flags...>; prints the rc and the diagnostics names
  local output rc
  output=$(cd "$dir" && "$tidy" --warnings-as-errors='*' ${TIDY_EXTRA[@]+"${TIDY_EXTRA[@]}"} src.cpp -- "$cxx" -std=c++20 -Iinc -Ithird_party "$@" -c src.cpp 2>&1)
  rc=$?
  echo "$rc $(grep -o -E '\[[a-z0-9.,-]+\]' <<<"$output" | sort -u | tr '\n' ' ')"
}

report() { # <case> <description> <base flags> -- <edit flags>
  local name=$1 what=$2 base=$3 after=$4
  read -r rc_base digest_base hit_base <<<"$base"
  read -r rc_after digest_after hit_after <<<"$after"
  read -r rc_real diags <<<"$real"
  local verdict
  if [ "$digest_base" = "$digest_after" ] && [ "$hit_after" = True ] && [ "$rc_after" = 0 ] && [ "$rc_real" != 0 ]; then
    verdict="KNOWN-PASS (gap reproduced: cached clean, real verdict fails)"
  elif [ "$digest_base" != "$digest_after" ]; then
    verdict="MISS (digest changed; gap does not reproduce on this toolchain)"
  else
    verdict="OTHER"
  fi
  echo "PARITY $name: $what | base rc=$rc_base stored-digest=${digest_base:0:12} | edit: cached rc=$rc_after hit=$hit_after digest=${digest_after:0:12} | real $tidy rc=$rc_real $diags | $verdict"
}

# Control: a real code change must miss.
setup control
printf 'int f() { return 1; }\n' >"$dir/src.cpp"
base=$(run_cached)
printf 'int f() { return 2; }\n' >"$dir/src.cpp"
after=$(run_cached); real=$(run_real)
read -r _ d1 _ <<<"$base"; read -r _ d2 h2 <<<"$after"
if [ "$d1" != "$d2" ] && [ "$h2" = False ]; then echo "PARITY control: code edit misses (ok)"; else echo "PARITY control: FAILED ($base / $after)"; failures=$((failures + 1)); fi

setup G1
printf 'int f() { return 1; }\n' >"$dir/src.cpp"
base=$(run_cached)
printf '#define SQ(x) (x * x)\n#define _Bad 1\nint f() { return 1; }\n' >"$dir/src.cpp"
after=$(run_cached); real=$(run_real)
report G1 "unused bad #define" "$base" "$after"

setup G2
printf 'void g() {\n  // NOLINTNEXTLINE(modernize-use-nullptr)\n  int *p = 0;\n  (void)p;\n}\n' >"$dir/src.cpp"
base=$(run_cached)
printf 'void g() {\n  // NOLINTNEXTLINE(modernize-use-nullptr)\n\n  int *p = 0;\n  (void)p;\n}\n' >"$dir/src.cpp"
after=$(run_cached); real=$(run_real)
report G2 "blank line after NOLINTNEXTLINE" "$base" "$after"

setup G3
printf '#define ZERO 0\nint *h() { return ZERO; }\n' >"$dir/src.cpp"
base=$(run_cached)
printf '#define ZERO 0\nint *h() { return 0; }\n' >"$dir/src.cpp"
after=$(run_cached); real=$(run_real)
report G3 "macro use replaced by its expansion" "$base" "$after"

setup G4
printf '#include "tp.h"\nint *k() { return tp(); }\n' >"$dir/src.cpp"
printf '#pragma once\ninline int *tp() { return 0; }\n' >"$dir/third_party/tp.h"
base=$(run_cached)
mv "$dir/third_party/tp.h" "$dir/inc/tp.h"
after=$(run_cached); real=$(run_real)
report G4 "header moved from third_party/ to inc/ (same #include, both on -I)" "$base" "$after"

# G4 again with the header filters on the command line: clang-tidy-20 was
# seen to ignore the config file's HeaderFilterRegex for this fixture.
setup G4cli
TIDY_EXTRA=(--header-filter=.* --exclude-header-filter=third_party/)
printf '#include "tp.h"\nint *k() { return tp(); }\n' >"$dir/src.cpp"
printf '#pragma once\ninline int *tp() { return 0; }\n' >"$dir/third_party/tp.h"
base=$(run_cached)
mv "$dir/third_party/tp.h" "$dir/inc/tp.h"
after=$(run_cached); real=$(run_real)
report G4cli "as G4, header filters passed on the command line" "$base" "$after"
TIDY_EXTRA=()

setup G5
printf 'void u() { int x; }\n' >"$dir/src.cpp"
base=$(run_cached -Wunused-variable -Wno-unused-variable)
after=$(run_cached -Wno-unused-variable -Wunused-variable); real=$(run_real -Wno-unused-variable -Wunused-variable)
report G5 "flags reordered (-Wx -Wno-x -> -Wno-x -Wx)" "$base" "$after"

setup G6
printf '#define SQ(x) (x * x) // NOLINT(bugprone-macro-parentheses)\nint s() { return SQ(2); }\n' >"$dir/src.cpp"
base=$(run_cached)
printf '#define SQ(x) (x * x)\nint s() { return SQ(2); }\n' >"$dir/src.cpp"
after=$(run_cached); real=$(run_real)
report G6 "NOLINT removed from a #define line" "$base" "$after"

# The -frandom-seed carrier for the r3.1 follow-up (K0): no finding under '*'.
setup K0
printf 'int f() { return 1; }\n' >"$dir/src.cpp"
printf "Checks: '*'\nWarningsAsErrors: '*'\n" >"$dir/.clang-tidy"
real=$(cd "$dir" && "$tidy" --warnings-as-errors='*' --extra-arg=-frandom-seed=0123abcd src.cpp -- "$cxx" -std=c++20 -Weverything -c src.cpp 2>&1 | grep -o -E '\[[a-z0-9.,-]+\]' | sort -u | tr '\n' ' ')
nocarrier=$(cd "$dir" && "$tidy" --warnings-as-errors='*' src.cpp -- "$cxx" -std=c++20 -Weverything -c src.cpp 2>&1 | grep -o -E '\[[a-z0-9.,-]+\]' | sort -u | tr '\n' ' ')
echo "PARITY K0: -frandom-seed carrier findings: [${real}] without carrier: [${nocarrier}]"
command -v /usr/lib/llvm-23/bin/clang++ >/dev/null && echo "PARITY K0: /usr/lib/llvm-23/bin/clang++ present" || echo "PARITY K0: /usr/lib/llvm-23/bin/clang++ absent"
exit "$failures"
