# Test Plan: `--compiler` flag for `build_metal.sh` and `install_dependencies.sh`

Issue: [#37907](https://github.com/tenstorrent/tt-metal/issues/37907)
Branch: `ivoitovych/issue-37907-fix-fedora-toolchain-selection`

## Overview

This test plan covers all code paths introduced by the `--compiler` flag,
compiler-to-toolchain mapping, and auto-detection fallback in `build_metal.sh`
(tests 1–37), and the `--compiler` flag, `prep_redhat_system()`, `install_llvm()`
rewrite, and warning fixes in `install_dependencies.sh` (tests 38–55).

`build_metal.sh` tests use `--configure-only` and PATH manipulation.
`install_dependencies.sh` tests run in Docker containers (require root).

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

## Data tables and helper functions reference

| Symbol | Lines | Purpose |
|--------|-------|---------|
| `CLANG_SEARCH` | 18 | Clang binary search order |
| `GCC_SEARCH` | 19 | GCC binary search order |
| `BINARY_TOOLCHAIN` | 23–31 | Binary name → toolchain ID map |
| `FLAG_TOOLCHAIN` | 34–39 | CLI flag → toolchain ID map |
| `toolchain_file()` | 42–44 | Toolchain ID → full cmake path |
| `find_compiler()` | 48–56 | Find first available binary from candidate list |
| `use_compiler()` | 60–76 | Set `toolchain_path` or `cxx/c_compiler_path` |

## Section 1: `--compiler` flag — Clang search (lines 292–296)

| # | Test | PATH setup | Command | Expected output | Lines |
|---|------|-----------|---------|-----------------|-------|
| 1 | `--compiler clang` finds clang++-20 | Normal (has clang++-20) | `--compiler clang --configure-only` | `INFO: --compiler clang: found clang++-20`, TOOLCHAIN_FILE=`clang-20-libstdcpp` | 293, 296 → BINARY_TOOLCHAIN:24 |
| 2 | `--compiler clang` finds clang++-19 | Fake clang++-19 only, no clang++-20 | `--compiler clang --configure-only` | `found clang++-19`, CMAKE_CXX_COMPILER=`.../clang++-19` | 293, 296 → use_compiler:66 |
| 3 | `--compiler clang` finds clang++-18 | Fake clang++-18 only | `--compiler clang --configure-only` | `found clang++-18`, CMAKE_CXX_COMPILER | 293, 296 → use_compiler:66 |
| 4 | `--compiler clang` finds clang++-17 | Fake clang++-17 only | `--compiler clang --configure-only` | `found clang++-17`, CMAKE_CXX_COMPILER | 293, 296 → use_compiler:66 |
| 5 | `--compiler clang` finds unversioned clang++ | Fake clang++ only (no versioned) | `--compiler clang --configure-only` | `found clang++`, TOOLCHAIN_FILE=`clang-libstdcpp` | 293, 296 → BINARY_TOOLCHAIN:25 |
| 6 | `--compiler clang` finds nothing | Empty PATH (no compilers) | `--compiler clang --configure-only` | `ERROR: No clang++ found in PATH.`, exit 1 | 293–294 |

## Section 2: `--compiler` flag — GCC search (lines 297–301)

| # | Test | PATH setup | Command | Expected output | Lines |
|---|------|-----------|---------|-----------------|-------|
| 7 | `--compiler gcc` finds g++-14 | Fake g++-14 | `--compiler gcc --configure-only` | `found g++-14`, TOOLCHAIN_FILE=`gcc-14` | 298, 301 → BINARY_TOOLCHAIN:26 |
| 8 | `--compiler gcc` finds g++-13 | Fake g++-13, no g++-14 | `--compiler gcc --configure-only` | `found g++-13`, CMAKE_CXX_COMPILER=`.../g++-13` | 298, 301 → use_compiler:66 |
| 9 | `--compiler gcc` finds g++-12 | Fake g++-12, no g++-14/13 | `--compiler gcc --configure-only` | `found g++-12`, TOOLCHAIN_FILE=`gcc-12` | 298, 301 → BINARY_TOOLCHAIN:27 |
| 10 | `--compiler gcc` finds unversioned g++ | Fake g++ only | `--compiler gcc --configure-only` | `found g++`, TOOLCHAIN_FILE=`gcc` | 298, 301 → BINARY_TOOLCHAIN:28 |
| 11 | `--compiler gcc` finds nothing | Empty PATH (no compilers) | `--compiler gcc --configure-only` | `ERROR: No g++ found in PATH.`, exit 1 | 298–299 |

## Section 3: `--compiler` flag — Pinned versions (lines 302–304)

| # | Test | PATH setup | Command | Expected output | Lines |
|---|------|-----------|---------|-----------------|-------|
| 12 | `--compiler clang-20` | Normal | `--compiler clang-20 --configure-only` | TOOLCHAIN_FILE=`clang-20-libstdcpp` | 303–304, FLAG_TOOLCHAIN:35 |
| 13 | `--compiler clang-20-libcpp` | Normal | `--compiler clang-20-libcpp --configure-only` | TOOLCHAIN_FILE=`clang-20-libcpp` | 303–304, FLAG_TOOLCHAIN:36 |
| 14 | `--compiler gcc-12` | Normal | `--compiler gcc-12 --configure-only` | TOOLCHAIN_FILE=`gcc-12` | 303–304, FLAG_TOOLCHAIN:38 |
| 15 | `--compiler gcc-14` | Normal | `--compiler gcc-14 --configure-only` | TOOLCHAIN_FILE=`gcc-14` | 303–304, FLAG_TOOLCHAIN:37 |

## Section 4: `--compiler` flag — Error handling (lines 305–309)

| # | Test | PATH setup | Command | Expected output | Lines |
|---|------|-----------|---------|-----------------|-------|
| 16 | `--compiler foobar` (unknown) | Normal | `--compiler foobar --configure-only` | `ERROR: Unknown compiler 'foobar'`, help text, exit 1 | 305–309 |

## Section 5: Auto-detection fallback — Clang found (lines 313–322)

These tests run with NO `--compiler`, `--cxx-compiler-path`, or `--toolchain-path` flags.

| # | Test | PATH setup | Command | Expected output | Lines |
|---|------|-----------|---------|-----------------|-------|
| 17 | No fallback (clang++-20 present) | Normal (has clang++-20) | `--configure-only` | TOOLCHAIN_FILE=`clang-20-libstdcpp`, NO warning | 315 (guard false) |
| 18 | Fallback finds clang++-19 | Fake clang++-19, no clang++-20 | `--configure-only` | `WARNING`, `Auto-selected clang++-19`, CMAKE_CXX_COMPILER | 317, 320 → use_compiler:66 |
| 19 | Fallback finds clang++-18 | Fake clang++-18 only | `--configure-only` | `Auto-selected clang++-18`, CMAKE_CXX_COMPILER | 317, 320 → use_compiler:66 |
| 20 | Fallback finds clang++-17 | Fake clang++-17 only | `--configure-only` | `Auto-selected clang++-17`, CMAKE_CXX_COMPILER | 317, 320 → use_compiler:66 |
| 21 | Fallback finds unversioned clang++ | Fake clang++ only | `--configure-only` | `Auto-selected clang++`, TOOLCHAIN_FILE=`clang-libstdcpp` | 317, 320 → BINARY_TOOLCHAIN:25 |

## Section 6: Auto-detection fallback — No Clang, GCC found (lines 313–322)

| # | Test | PATH setup | Command | Expected output | Lines |
|---|------|-----------|---------|-----------------|-------|
| 22 | Fallback finds g++-14 (no clang) | Fake g++-14, no clang at all | `--configure-only` | `WARNING`, `Auto-selected g++-14`, TOOLCHAIN_FILE=`gcc-14` | 317, 320 → BINARY_TOOLCHAIN:26 |
| 23 | Fallback finds g++-13 (no clang) | Fake g++-13 only | `--configure-only` | `Auto-selected g++-13`, CMAKE_CXX_COMPILER | 317, 320 → use_compiler:66 |
| 24 | Fallback finds g++-12 (no clang) | Fake g++-12 only | `--configure-only` | `Auto-selected g++-12`, TOOLCHAIN_FILE=`gcc-12` | 317, 320 → BINARY_TOOLCHAIN:27 |
| 25 | Fallback finds unversioned g++ (no clang) | Fake g++ only | `--configure-only` | `Auto-selected g++`, TOOLCHAIN_FILE=`gcc` | 317, 320 → BINARY_TOOLCHAIN:28 |

## Section 7: Auto-detection fallback — Nothing found (lines 317–318)

| # | Test | PATH setup | Command | Expected output | Lines |
|---|------|-----------|---------|-----------------|-------|
| 26 | Fallback finds nothing | Empty PATH (no compilers) | `--configure-only` | `WARNING`, `ERROR: No C++ compiler found.`, exit 1 | 317–318 |

## Section 8: Auto-detection guard conditions (line 314)

These verify that auto-detection is skipped when any explicit override is set.
All tests run WITHOUT clang++-20 in PATH to ensure the fallback would trigger
if the guard didn't prevent it.

| # | Test | Command | Expected output | Guard |
|---|------|---------|-----------------|-------|
| 27 | `--compiler` set skips auto-detect | `--compiler clang-20 --configure-only` | TOOLCHAIN_FILE=`clang-20`, NO warning | `compiler != ""` |
| 28 | `--cxx-compiler-path` set skips auto-detect | `--cxx-compiler-path /usr/bin/g++ --configure-only` | CMAKE_CXX_COMPILER=`/usr/bin/g++`, NO warning | `cxx_compiler_path != ""` |
| 29 | `--toolchain-path` set skips auto-detect | `--toolchain-path cmake/x86_64-linux-gcc-14-toolchain.cmake --configure-only` | TOOLCHAIN_FILE=`gcc-14`, NO warning | `toolchain_path_explicitly_set = true` |

## Section 9: Search priority ordering (lines 18–19, via 293, 298)

| # | Test | PATH setup | Expected | Verifies |
|---|------|-----------|----------|----------|
| 30 | clang++-20 preferred over clang++-19 | Both clang++-20 and clang++-19 present | `found clang++-20` | CLANG_SEARCH descending order |
| 31 | clang++-18 preferred over clang++ | Both clang++-18 and clang++ present | `found clang++-18` | Versioned before unversioned |
| 32 | g++-14 preferred over g++-13 | Both g++-14 and g++-13 present | `found g++-14` | GCC_SEARCH descending order |
| 33 | g++-12 preferred over g++ | Both g++-12 and g++ present | `found g++-12` | Versioned before unversioned |

## Section 10: Help text and CMake presets

| # | Test | Command | Expected |
|---|------|---------|----------|
| 34 | `--help` shows `--compiler` | `build_metal.sh --help` | Output contains `--compiler compiler_name` and lists `clang`, `gcc` |
| 35 | CMakePresets.json valid | `cmake --list-presets 2>&1` | Output lists `gcc-system` and `clang-system` presets |

## Section 11: `--toolchain-path` protection flag (line 272)

| # | Test | Command | Expected | Verifies |
|---|------|---------|----------|----------|
| 36 | `--toolchain-path` sets explicit flag | `--toolchain-path cmake/x86_64-linux-gcc-toolchain.cmake --configure-only` (no clang++-20) | Uses `gcc-toolchain`, NO auto-detect override | `toolchain_path_explicitly_set=true` (line 272) |
| 37 | Default has explicit flag false | `--configure-only` (with clang++-20) | Uses default `clang-20-libstdcpp` | `toolchain_path_explicitly_set=false` (line 150) |

---

# `install_dependencies.sh` tests

All tests below run as root in Docker containers. Use `--docker` flag to avoid
systemd-dependent steps. Verify behavior via output messages (grep for INFO,
WARNING, ERROR).

## Section 12: `--compiler` flag — validation (install_dependencies.sh)

| # | Test | Container | Command | Expected output |
|---|------|-----------|---------|-----------------|
| 38 | Valid `--compiler clang` | Ubuntu 22.04 | `./install_dependencies.sh --docker --compiler clang` | `[INFO] Compiler selection: clang`, LLVM 20 installed |
| 39 | Valid `--compiler gcc` | Ubuntu 22.04 | `./install_dependencies.sh --docker --compiler gcc` | `[INFO] Compiler selection: gcc`, `[INFO] Skipping LLVM installation (--compiler gcc)` |
| 40 | Valid `--compiler clang-20` | Ubuntu 22.04 | `./install_dependencies.sh --docker --compiler clang-20` | `[INFO] Compiler selection: clang-20`, LLVM 20 installed |
| 41 | Valid `--compiler gcc-14` | Ubuntu 22.04 | `./install_dependencies.sh --docker --compiler gcc-14` | `[INFO] Compiler selection: gcc-14`, `[INFO] Skipping LLVM installation (--compiler gcc-14)` |
| 42 | Invalid `--compiler foobar` | Ubuntu 22.04 | `./install_dependencies.sh --docker --compiler foobar` | `[ERROR] Unknown compiler 'foobar'. Allowed: clang gcc clang-20 clang-20-libcpp gcc-12 gcc-14.`, exit 1 |
| 43 | Default (no --compiler) | Ubuntu 22.04 | `./install_dependencies.sh --docker` | LLVM 20 installed, no `[INFO] Compiler selection` |

## Section 13: `--compiler` flag — LLVM skip on GCC variants

| # | Test | Container | Command | Expected output |
|---|------|-----------|---------|-----------------|
| 44 | `--compiler gcc` skips LLVM | Ubuntu 22.04 | `./install_dependencies.sh --docker --compiler gcc` | `[INFO] Skipping LLVM installation (--compiler gcc)` |
| 45 | `--compiler gcc-12` skips LLVM | Ubuntu 22.04 | `./install_dependencies.sh --docker --compiler gcc-12` | `[INFO] Skipping LLVM installation (--compiler gcc-12)` |
| 46 | `--compiler gcc-14` skips LLVM | Ubuntu 24.04 | `./install_dependencies.sh --docker --compiler gcc-14` | `[INFO] Skipping LLVM installation (--compiler gcc-14)` |
| 47 | `--compiler clang-20-libcpp` installs LLVM | Ubuntu 22.04 | `./install_dependencies.sh --docker --compiler clang-20-libcpp` | LLVM 20 installed (no skip message) |

## Section 14: `install_llvm()` — RedHat messaging

| # | Test | Container | Command | Expected output |
|---|------|-----------|---------|-----------------|
| 48 | Default on Fedora: distro clang | Fedora 40 | `./install_dependencies.sh --docker` | `[INFO] Using distro-provided LLVM/Clang for fedora:` (no WARNING) |
| 49 | `--compiler gcc` on Fedora | Fedora 40 | `./install_dependencies.sh --docker --compiler gcc` | `[INFO] Skipping LLVM installation (--compiler gcc)` |
| 50 | `--compiler clang` on Fedora | Fedora 40 | `./install_dependencies.sh --docker --compiler clang` | `[INFO] Using distro-provided LLVM/Clang for fedora:` |

## Section 15: `prep_redhat_system()` — repo setup

| # | Test | Container | Command | Expected output |
|---|------|-----------|---------|-----------------|
| 51 | Fedora: no extra repos | Fedora 40 | `./install_dependencies.sh --docker` | `[INFO] Preparing Red Hat family system...` then proceeds (no EPEL) |
| 52 | Rocky: EPEL + CRB | Rocky 9 | `./install_dependencies.sh --docker` | `[INFO] Installing EPEL repository...`, `[INFO] Enabling CRB repository` |

## Section 16: Warning fixes — MPI ULFM and hugepages

| # | Test | Container | Command | Expected output |
|---|------|-----------|---------|-----------------|
| 53 | MPI ULFM on Fedora: INFO not WARNING | Fedora 40 | `./install_dependencies.sh --docker` | `[INFO] MPI ULFM is only available as a .deb package; skipping on fedora` |
| 54 | Hugepages on Fedora: INFO not WARNING | Fedora 40 | `./install_dependencies.sh --docker --hugepages` | `[INFO] Hugepages package is only available as a .deb package; skipping on fedora` |

## Section 17: Help text (install_dependencies.sh)

| # | Test | Command | Expected output |
|---|------|---------|-----------------|
| 55 | `--help` shows `--compiler` | `./install_dependencies.sh --help` | Output contains `[--compiler name]` and lists `clang gcc clang-20 clang-20-libcpp gcc-12 gcc-14` |

## Coverage Summary

**`build_metal.sh`:**

| Code section | Lines | Tests |
|-------------|-------|-------|
| `CLANG_SEARCH` / `GCC_SEARCH` arrays | 18–19 | 1–11, 18–26, 30–33 |
| `BINARY_TOOLCHAIN` map | 23–31 | 1, 5, 7, 9–10, 21–22, 24–25 |
| `FLAG_TOOLCHAIN` map | 34–39 | 12–15 |
| `toolchain_file()` | 42–44 | 1, 5, 7, 9–10, 12–15, 21–22, 24–25 |
| `find_compiler()` search | 48–56 | 1–11, 18–26, 30–33 |
| `use_compiler()` toolchain path | 60–64 | 1, 5, 7, 9–10, 21–22, 24–25 |
| `use_compiler()` direct path | 65–75 | 2–4, 8, 18–20, 23 |
| `--compiler clang` case | 292–296 | 1–6, 30–31 |
| `--compiler gcc` case | 297–301 | 7–11, 32–33 |
| `--compiler` pinned (FLAG_TOOLCHAIN) | 302–304 | 12–15 |
| `--compiler` error (unknown) | 305–309 | 16 |
| Auto-detect: clang++-20 present | 314–315 | 17 |
| Auto-detect: compiler fallback | 316–321 | 18–25 |
| Auto-detect: nothing found | 317–318 | 26 |
| Auto-detect: guard conditions | 314 | 27–29 |
| `--toolchain-path` explicit flag | 150, 272 | 28–29, 36–37 |
| Help text | 109 | 34 |
| CMakePresets.json | — | 35 |

**`install_dependencies.sh`:**

| Code section | Tests |
|-------------|-------|
| `COMPILER_FLAGS` array + validation | 38–43, 55 |
| `--compiler gcc*` skips `install_llvm()` | 39, 41, 44–46 |
| `--compiler clang*` installs LLVM | 38, 40, 47 |
| `install_llvm()` RedHat: distro clang INFO | 48, 50 |
| `install_llvm()` RedHat: skip on gcc | 49 |
| `prep_redhat_system()` Fedora: no-op | 51 |
| `prep_redhat_system()` RHEL/Rocky: EPEL+CRB | 52 |
| MPI ULFM warning → INFO | 53 |
| Hugepages warning → INFO | 54 |
| Help text | 55 |

| Script | Tests |
|--------|-------|
| `build_metal.sh` | 37 |
| `install_dependencies.sh` | 18 |
| **Total** | **55 tests** |
