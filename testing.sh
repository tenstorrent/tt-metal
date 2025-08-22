#!/usr/bin/env bash
set -euo pipefail

CC="clang++-20"
CXXSTD="-std=c++20"
CFLAGS="-g -O0 -fPIC"
# Base: trace-pc (no guard)
COV_BASE="-fsanitize-coverage=trace-pc"
# With function mode: required for fun:/src: allowlist to apply
COV_FUNC="-fsanitize-coverage=func,trace-pc"

echo "[1/8] Creating minimal TU and allowlists"
cat > mini.cpp <<'EOF'
int foo() { return 42; }
int main() { return foo(); }
EOF

# All functions
printf 'fun:.*\n' > allow_fun_all.txt
# Only main by name
printf 'fun:^main$\n' > allow_fun_main.txt
# Exact absolute path (matches what DWARF shows below)
printf 'src:%s/mini.cpp\n' "$PWD" > allow_src_exact.txt
# Regex absolute path
printf 'src:.*/mini\\.cpp\n' > allow_src_regex.txt

check_cov() {
  local obj="$1"
  if nm -C "$obj" 2>/dev/null | grep -Eq '__sanitizer_cov_trace_pc(_guard)?'; then
    echo "INSTRUMENTED  - $obj"
  else
    echo "REJECTED      - $obj"
  fi
}

check_ir() {
  local ll="$1"
  if grep -Eq '__sanitizer_cov_trace_pc(_guard)?' "$ll" 2>/dev/null; then
    echo "IR: INSTRUMENTED - $ll"
  else
    echo "IR: REJECTED     - $ll"
  fi
}

echo "[2/8] Baseline (no allowlist) — expect INSTRUMENTED"
$CC $CXXSTD $CFLAGS $COV_BASE -c mini.cpp -o mini_no_list.o
check_cov mini_no_list.o

echo "[3/8] Allowlist fun:.* WITH func,trace — expect INSTRUMENTED"
$CC $CXXSTD $CFLAGS $COV_FUNC -fsanitize-coverage-allowlist="$PWD/allow_fun_all.txt" -c mini.cpp -o mini_fun_all.o
check_cov mini_fun_all.o
$CC $CXXSTD $CFLAGS $COV_FUNC -fsanitize-coverage-allowlist="$PWD/allow_fun_all.txt" -S -emit-llvm mini.cpp -o mini_fun_all.ll
check_ir mini_fun_all.ll

echo "[4/8] Allowlist fun:.* WITH ONLY trace-pc — expect REJECTED (fail-closed)"
$CC $CXXSTD $CFLAGS $COV_BASE -fsanitize-coverage-allowlist="$PWD/allow_fun_all.txt" -c mini.cpp -o mini_fun_all_trace_only.o || true
check_cov mini_fun_all_trace_only.o || true
$CC $CXXSTD $CFLAGS $COV_BASE -fsanitize-coverage-allowlist="$PWD/allow_fun_all.txt" -S -emit-llvm mini.cpp -o mini_fun_all_trace_only.ll || true
check_ir mini_fun_all_trace_only.ll || true

echo "[5/8] Allowlist fun:^main$ WITH func,trace — expect INSTRUMENTED"
$CC $CXXSTD $CFLAGS $COV_FUNC -fsanitize-coverage-allowlist="$PWD/allow_fun_main.txt" -c mini.cpp -o mini_fun_main.o
check_cov mini_fun_main.o

echo "[6/8] Allowlist src:<absolute/regex> WITH func,trace — expect INSTRUMENTED"
$CC $CXXSTD $CFLAGS $COV_FUNC -fsanitize-coverage-allowlist="$PWD/allow_src_exact.txt" -c mini.cpp -o mini_src_exact.o
check_cov mini_src_exact.o
$CC $CXXSTD $CFLAGS $COV_FUNC -fsanitize-coverage-allowlist="$PWD/allow_src_regex.txt" -c mini.cpp -o mini_src_regex.o
check_cov mini_src_regex.o

echo "[7/8] Show DWARF path shape (what src: must match)"
$CC -g -O0 -fPIC -c mini.cpp -o mini_dbg.o
if command -v llvm-dwarfdump >/dev/null 2>&1; then
  echo "--- DWARF debug-line (first 120 lines) ---"
  llvm-dwarfdump --debug-line mini_dbg.o | sed -n '1,120p'
else
  echo "llvm-dwarfdump not found; skipping DWARF preview."
fi

echo "[8/8] Prove cc1 and allowlist ingestion"
echo "--- cc1 command (-###) ---"
$CC -### $COV_FUNC -fsanitize-coverage-allowlist="$PWD/allow_fun_all.txt" -c mini.cpp -o /dev/null 2>&1 | sed -n '1,200p'
echo "--- strace open of allowlist ---"
if command -v strace >/dev/null 2>&1; then
  strace -f -e openat -s 200 -- $CC $COV_FUNC -fsanitize-coverage-allowlist="$PWD/allow_fun_all.txt" -c mini.cpp -o /dev/null 2>&1 | grep -F "allow_fun_all.txt" || true
else
  echo "strace not found; skipping."
fi

echo
echo "Summary:"
check_cov mini_no_list.o
check_cov mini_fun_all.o
check_cov mini_fun_all_trace_only.o || true
check_cov mini_fun_main.o
check_cov mini_src_exact.o
check_cov mini_src_regex.o
