# Test Plan: `--compiler` flag and auto-detection in `build_metal.sh`

Issue: [#37907](https://github.com/tenstorrent/tt-metal/issues/37907)
Branch: `ivoitovych/issue-37907-fix-fedora-toolchain-selection`

## Overview

This test plan achieves 100% branch coverage for all code paths introduced by the
`--compiler` flag, compiler-to-toolchain mapping, and auto-detection fallback in
`build_metal.sh`.

All tests use `--configure-only` to avoid actual builds. Compiler visibility is
controlled via `PATH` manipulation using temporary directories with fake compiler
scripts.

## Test Setup

```bash
# Create temp directory with fake compiler stubs
FAKE_DIR=$(mktemp -d)
EMPTY_DIR=$(mktemp -d)  # empty = no compilers at all

# Create a fake compiler stub (repeat for each needed binary)
create_fake() {
    echo '#!/bin/sh' > "$FAKE_DIR/$1"
    echo 'echo "fake $1"' >> "$FAKE_DIR/$1"
    chmod +x "$FAKE_DIR/$1"
}

# Minimal PATH with only system essentials (bash, getopt, cmake, python3, etc.)
# but no compilers. Adjust per system.
SYSTEM_BIN="/usr/bin:/bin"

# To hide all real compilers, prepend FAKE_DIR (with only desired stubs) to a
# compiler-free base PATH. To hide everything, use EMPTY_DIR only.
```

## Section 1: `--compiler` flag — Clang search (lines 231–250)

| # | Test | PATH setup | Command | Expected output | Lines |
|---|------|-----------|---------|-----------------|-------|
| 1 | `--compiler clang` finds clang++-20 | Normal (has clang++-20) | `--compiler clang --configure-only` | `INFO: --compiler clang: found clang++-20`, TOOLCHAIN_FILE=`clang-20-libstdcpp` | 246 |
| 2 | `--compiler clang` finds clang++-19 | Fake clang++-19 only, no clang++-20 | `--compiler clang --configure-only` | `found clang++-19`, CMAKE_CXX_COMPILER=`.../clang++-19` | 248–249 |
| 3 | `--compiler clang` finds clang++-18 | Fake clang++-18 only | `--compiler clang --configure-only` | `found clang++-18`, CMAKE_CXX_COMPILER | 248–249 |
| 4 | `--compiler clang` finds clang++-17 | Fake clang++-17 only | `--compiler clang --configure-only` | `found clang++-17`, CMAKE_CXX_COMPILER | 248–249 |
| 5 | `--compiler clang` finds unversioned clang++ | Fake clang++ only (no versioned) | `--compiler clang --configure-only` | `found clang++`, TOOLCHAIN_FILE=`clang-libstdcpp` | 247 |
| 6 | `--compiler clang` finds nothing | Empty PATH (no compilers) | `--compiler clang --configure-only` | `ERROR: No clang++ found in PATH.`, exit 1 | 241–242 |

## Section 2: `--compiler` flag — GCC search (lines 254–271)

| # | Test | PATH setup | Command | Expected output | Lines |
|---|------|-----------|---------|-----------------|-------|
| 7 | `--compiler gcc` finds g++-14 | Fake g++-14 | `--compiler gcc --configure-only` | `found g++-14`, TOOLCHAIN_FILE=`gcc-14` | 266 |
| 8 | `--compiler gcc` finds g++-13 | Fake g++-13, no g++-14 | `--compiler gcc --configure-only` | `found g++-13`, CMAKE_CXX_COMPILER=`.../g++-13` | 269–270 |
| 9 | `--compiler gcc` finds g++-12 | Fake g++-12, no g++-14/13 | `--compiler gcc --configure-only` | `found g++-12`, TOOLCHAIN_FILE=`gcc-12` | 267 |
| 10 | `--compiler gcc` finds unversioned g++ | Fake g++ only | `--compiler gcc --configure-only` | `found g++`, TOOLCHAIN_FILE=`gcc` | 268 |
| 11 | `--compiler gcc` finds nothing | Empty PATH (no compilers) | `--compiler gcc --configure-only` | `ERROR: No g++ found in PATH.`, exit 1 | 260–262 |

## Section 3: `--compiler` flag — Pinned versions (lines 272–279)

| # | Test | PATH setup | Command | Expected output | Lines |
|---|------|-----------|---------|-----------------|-------|
| 12 | `--compiler clang-20` | Normal | `--compiler clang-20 --configure-only` | TOOLCHAIN_FILE=`clang-20-libstdcpp` | 273 |
| 13 | `--compiler clang-20-libcpp` | Normal | `--compiler clang-20-libcpp --configure-only` | TOOLCHAIN_FILE=`clang-20-libcpp` | 274 |
| 14 | `--compiler gcc-12` | Normal | `--compiler gcc-12 --configure-only` | TOOLCHAIN_FILE=`gcc-12` | 277 |
| 15 | `--compiler gcc-14` | Normal | `--compiler gcc-14 --configure-only` | TOOLCHAIN_FILE=`gcc-14` | 278 |

## Section 4: `--compiler` flag — Error handling (lines 280–283)

| # | Test | PATH setup | Command | Expected output | Lines |
|---|------|-----------|---------|-----------------|-------|
| 16 | `--compiler foobar` (unknown) | Normal | `--compiler foobar --configure-only` | `ERROR: Unknown compiler 'foobar'`, help text, exit 1 | 280–283 |

## Section 5: Auto-detection fallback — Clang found (lines 287–305)

These tests run with NO `--compiler`, `--cxx-compiler-path`, or `--toolchain-path` flags.

| # | Test | PATH setup | Command | Expected output | Lines |
|---|------|-----------|---------|-----------------|-------|
| 17 | No fallback (clang++-20 present) | Normal (has clang++-20) | `--configure-only` | TOOLCHAIN_FILE=`clang-20-libstdcpp`, NO warning | 288 (guard false) |
| 18 | Fallback finds clang++-19 | Fake clang++-19, no clang++-20 | `--configure-only` | `WARNING`, `Auto-selected clang++-19`, CMAKE_CXX_COMPILER | 302–303 |
| 19 | Fallback finds clang++-18 | Fake clang++-18 only | `--configure-only` | `Auto-selected clang++-18`, CMAKE_CXX_COMPILER | 302–303 |
| 20 | Fallback finds clang++-17 | Fake clang++-17 only | `--configure-only` | `Auto-selected clang++-17`, CMAKE_CXX_COMPILER | 302–303 |
| 21 | Fallback finds unversioned clang++ | Fake clang++ only | `--configure-only` | `Auto-selected clang++`, TOOLCHAIN_FILE=`clang-libstdcpp` | 301 |

## Section 6: Auto-detection fallback — No Clang, GCC found (lines 306–323)

| # | Test | PATH setup | Command | Expected output | Lines |
|---|------|-----------|---------|-----------------|-------|
| 22 | Fallback finds g++-14 (no clang) | Fake g++-14, no clang at all | `--configure-only` | `WARNING`, `Auto-selected g++-14`, TOOLCHAIN_FILE=`gcc-14` | 317 |
| 23 | Fallback finds g++-13 (no clang) | Fake g++-13 only | `--configure-only` | `Auto-selected g++-13`, CMAKE_CXX_COMPILER | 320–321 |
| 24 | Fallback finds g++-12 (no clang) | Fake g++-12 only | `--configure-only` | `Auto-selected g++-12`, TOOLCHAIN_FILE=`gcc-12` | 318 |
| 25 | Fallback finds unversioned g++ (no clang) | Fake g++ only | `--configure-only` | `Auto-selected g++`, TOOLCHAIN_FILE=`gcc` | 319 |

## Section 7: Auto-detection fallback — Nothing found (lines 324–326)

| # | Test | PATH setup | Command | Expected output | Lines |
|---|------|-----------|---------|-----------------|-------|
| 26 | Fallback finds nothing | Empty PATH (no compilers) | `--configure-only` | `WARNING`, `ERROR: No C++ compiler found.`, exit 1 | 325–326 |

## Section 8: Auto-detection guard conditions (line 288)

These verify that auto-detection is skipped when any explicit override is set.
All tests run WITHOUT clang++-20 in PATH to ensure the fallback would trigger
if the guard didn't prevent it.

| # | Test | Command | Expected output | Guard |
|---|------|---------|-----------------|-------|
| 27 | `--compiler` set skips auto-detect | `--compiler clang-20 --configure-only` | TOOLCHAIN_FILE=`clang-20`, NO warning | `compiler != ""` |
| 28 | `--cxx-compiler-path` set skips auto-detect | `--cxx-compiler-path /usr/bin/g++ --configure-only` | CMAKE_CXX_COMPILER=`/usr/bin/g++`, NO warning | `cxx_compiler_path != ""` |
| 29 | `--toolchain-path` set skips auto-detect | `--toolchain-path cmake/x86_64-linux-gcc-14-toolchain.cmake --configure-only` | TOOLCHAIN_FILE=`gcc-14`, NO warning | `toolchain_path_explicitly_set = true` |

## Section 9: Search priority ordering (lines 234, 254)

| # | Test | PATH setup | Expected | Verifies |
|---|------|-----------|----------|----------|
| 30 | clang++-20 preferred over clang++-19 | Both clang++-20 and clang++-19 present | `found clang++-20` | Descending order |
| 31 | clang++-18 preferred over clang++ | Both clang++-18 and clang++ present | `found clang++-18` | Versioned before unversioned |
| 32 | g++-14 preferred over g++-13 | Both g++-14 and g++-13 present | `found g++-14` | Descending order |
| 33 | g++-12 preferred over g++ | Both g++-12 and g++ present | `found g++-12` | Versioned before unversioned |

## Section 10: Help text and CMake presets

| # | Test | Command | Expected |
|---|------|---------|----------|
| 34 | `--help` shows `--compiler` | `build_metal.sh --help` | Output contains `--compiler compiler_name` and lists `clang`, `gcc` |
| 35 | CMakePresets.json valid | `cmake --list-presets 2>&1` | Output lists `gcc-system` and `clang-system` presets |

## Section 11: `--toolchain-path` protection flag (line 208)

| # | Test | Command | Expected | Verifies |
|---|------|---------|----------|----------|
| 36 | `--toolchain-path` sets explicit flag | `--toolchain-path cmake/x86_64-linux-gcc-toolchain.cmake --configure-only` (no clang++-20) | Uses `gcc-toolchain`, NO auto-detect override | `toolchain_path_explicitly_set=true` |
| 37 | Default has explicit flag false | `--configure-only` (with clang++-20) | Uses default `clang-20-libstdcpp` | `toolchain_path_explicitly_set=false` |

## Coverage Summary

| Code section | Lines | Tests |
|-------------|-------|-------|
| `--compiler clang` search | 231–250 | 1–6, 30–31 |
| `--compiler gcc` search | 251–271 | 7–11, 32–33 |
| `--compiler` pinned versions | 272–279 | 12–15 |
| `--compiler` error (unknown) | 280–283 | 16 |
| Auto-detect: clang++-20 present | 288 | 17 |
| Auto-detect: clang fallback | 289–305 | 18–21 |
| Auto-detect: gcc fallback | 306–323 | 22–25 |
| Auto-detect: nothing found | 324–326 | 26 |
| Auto-detect: guard conditions | 288 | 27–29 |
| `--toolchain-path` explicit flag | 208 | 28–29, 36–37 |
| Help text | 53 | 34 |
| CMakePresets.json | — | 35 |
| **Total** | | **37 tests** |
