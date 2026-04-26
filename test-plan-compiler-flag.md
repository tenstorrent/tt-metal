# Test Plan: `--compiler` flag for `build_metal.sh` and `install_dependencies.sh`

Issue: [#37907](https://github.com/tenstorrent/tt-metal/issues/37907)
Branch: `ivoitovych/issue-37907-fix-fedora-toolchain-selection`

## Overview

This test plan covers all code paths introduced by the `--compiler` flag,
compiler-to-toolchain mapping, and auto-detection fallback in `build_metal.sh`
(tests 1–41), and the `--compiler` flag, `prep_redhat_system()`, `install_llvm()`
rewrite, LLVM tarball installation, versioned GCC installation, and warning fixes
in `install_dependencies.sh` (tests 42–68).

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
| `BINARY_TOOLCHAIN` | 23–33 | Binary name → toolchain ID map (10 entries) |
| `COMPILER_FLAGS` | 36 | Valid `--compiler` flag values |
| `FLAG_TOOLCHAIN` | 39–48 | CLI flag → toolchain ID map (8 entries) |
| `toolchain_file()` | 51–53 | Toolchain ID → full cmake path |
| `find_compiler()` | 57–65 | Find first available binary from candidate list |
| `use_compiler()` | 69–89 | Set `toolchain_path` or `cxx/c_compiler_path` |

## Section 1: `--compiler` flag — Clang search (lines 310–314)

| # | Test | PATH setup | Command | Expected output | Lines |
|---|------|-----------|---------|-----------------|-------|
| 1 | `--compiler clang` finds clang++-20 | Normal (has clang++-20) | `--compiler clang --configure-only` | `INFO: --compiler clang: found clang++-20`, TOOLCHAIN_FILE=`clang-20-libstdcpp` | 311, 314 → BINARY_TOOLCHAIN:24 |
| 2 | `--compiler clang` finds clang++-19 | Fake clang++-19 only, no clang++-20 | `--compiler clang --configure-only` | `found clang++-19`, TOOLCHAIN_FILE=`clang-19-libstdcpp` | 311, 314 → BINARY_TOOLCHAIN:25 |
| 3 | `--compiler clang` finds clang++-18 | Fake clang++-18 only | `--compiler clang --configure-only` | `found clang++-18`, TOOLCHAIN_FILE=`clang-18-libstdcpp` | 311, 314 → BINARY_TOOLCHAIN:26 |
| 4 | `--compiler clang` finds clang++-17 | Fake clang++-17 only | `--compiler clang --configure-only` | `found clang++-17`, TOOLCHAIN_FILE=`clang-17-libstdcpp` | 311, 314 → BINARY_TOOLCHAIN:27 |
| 5 | `--compiler clang` finds unversioned clang++ | Fake clang++ only (no versioned) | `--compiler clang --configure-only` | `found clang++`, TOOLCHAIN_FILE=`clang-libstdcpp` | 311, 314 → BINARY_TOOLCHAIN:28 |
| 6 | `--compiler clang` finds nothing | Empty PATH (no compilers) | `--compiler clang --configure-only` | `ERROR: No clang++ found in PATH.`, exit 1 | 311–312 |

## Section 2: `--compiler` flag — GCC search (lines 315–319)

| # | Test | PATH setup | Command | Expected output | Lines |
|---|------|-----------|---------|-----------------|-------|
| 7 | `--compiler gcc` finds g++-14 | Fake g++-14 | `--compiler gcc --configure-only` | `found g++-14`, TOOLCHAIN_FILE=`gcc-14` | 316, 319 → BINARY_TOOLCHAIN:29 |
| 8 | `--compiler gcc` finds g++-13 | Fake g++-13, no g++-14 | `--compiler gcc --configure-only` | `found g++-13`, TOOLCHAIN_FILE=`gcc-13` | 316, 319 → BINARY_TOOLCHAIN:30 |
| 9 | `--compiler gcc` finds g++-12 | Fake g++-12, no g++-14/13 | `--compiler gcc --configure-only` | `found g++-12`, TOOLCHAIN_FILE=`gcc-12` | 316, 319 → BINARY_TOOLCHAIN:31 |
| 10 | `--compiler gcc` finds unversioned g++ | Fake g++ only | `--compiler gcc --configure-only` | `found g++`, TOOLCHAIN_FILE=`gcc` | 316, 319 → BINARY_TOOLCHAIN:32 |
| 11 | `--compiler gcc` finds nothing | Empty PATH (no compilers) | `--compiler gcc --configure-only` | `ERROR: No g++ found in PATH.`, exit 1 | 316–317 |

## Section 3: `--compiler` flag — Pinned versions (lines 320–322)

| # | Test | PATH setup | Command | Expected output | Lines |
|---|------|-----------|---------|-----------------|-------|
| 12 | `--compiler clang-20` | Normal | `--compiler clang-20 --configure-only` | TOOLCHAIN_FILE=`clang-20-libstdcpp` | 321–322, FLAG_TOOLCHAIN:40 |
| 13 | `--compiler clang-19` | Normal | `--compiler clang-19 --configure-only` | TOOLCHAIN_FILE=`clang-19-libstdcpp` | 321–322, FLAG_TOOLCHAIN:41 |
| 14 | `--compiler clang-18` | Normal | `--compiler clang-18 --configure-only` | TOOLCHAIN_FILE=`clang-18-libstdcpp` | 321–322, FLAG_TOOLCHAIN:42 |
| 15 | `--compiler clang-17` | Normal | `--compiler clang-17 --configure-only` | TOOLCHAIN_FILE=`clang-17-libstdcpp` | 321–322, FLAG_TOOLCHAIN:43 |
| 16 | `--compiler clang-20-libcpp` | Normal | `--compiler clang-20-libcpp --configure-only` | TOOLCHAIN_FILE=`clang-20-libcpp` | 321–322, FLAG_TOOLCHAIN:44 |
| 17 | `--compiler gcc-14` | Normal | `--compiler gcc-14 --configure-only` | TOOLCHAIN_FILE=`gcc-14` | 321–322, FLAG_TOOLCHAIN:45 |
| 18 | `--compiler gcc-13` | Normal | `--compiler gcc-13 --configure-only` | TOOLCHAIN_FILE=`gcc-13` | 321–322, FLAG_TOOLCHAIN:46 |
| 19 | `--compiler gcc-12` | Normal | `--compiler gcc-12 --configure-only` | TOOLCHAIN_FILE=`gcc-12` | 321–322, FLAG_TOOLCHAIN:47 |

## Section 4: `--compiler` flag — Error handling (lines 323–327)

| # | Test | PATH setup | Command | Expected output | Lines |
|---|------|-----------|---------|-----------------|-------|
| 20 | `--compiler foobar` (unknown) | Normal | `--compiler foobar --configure-only` | `ERROR: Unknown compiler 'foobar'`, help text, exit 1 | 323–327 |

## Section 5: Auto-detection fallback — Clang found (lines 331–340)

These tests run with NO `--compiler`, `--cxx-compiler-path`, or `--toolchain-path` flags.

| # | Test | PATH setup | Command | Expected output | Lines |
|---|------|-----------|---------|-----------------|-------|
| 21 | No fallback (clang++-20 present) | Normal (has clang++-20) | `--configure-only` | TOOLCHAIN_FILE=`clang-20-libstdcpp`, NO warning | 333 (guard false) |
| 22 | Fallback finds clang++-19 | Fake clang++-19, no clang++-20 | `--configure-only` | `WARNING`, `Auto-selected clang++-19`, TOOLCHAIN_FILE=`clang-19-libstdcpp` | 335, 338 → BINARY_TOOLCHAIN:25 |
| 23 | Fallback finds clang++-18 | Fake clang++-18 only | `--configure-only` | `Auto-selected clang++-18`, TOOLCHAIN_FILE=`clang-18-libstdcpp` | 335, 338 → BINARY_TOOLCHAIN:26 |
| 24 | Fallback finds clang++-17 | Fake clang++-17 only | `--configure-only` | `Auto-selected clang++-17`, TOOLCHAIN_FILE=`clang-17-libstdcpp` | 335, 338 → BINARY_TOOLCHAIN:27 |
| 25 | Fallback finds unversioned clang++ | Fake clang++ only | `--configure-only` | `Auto-selected clang++`, TOOLCHAIN_FILE=`clang-libstdcpp` | 335, 338 → BINARY_TOOLCHAIN:28 |

## Section 6: Auto-detection fallback — No Clang, GCC found (lines 331–340)

| # | Test | PATH setup | Command | Expected output | Lines |
|---|------|-----------|---------|-----------------|-------|
| 26 | Fallback finds g++-14 (no clang) | Fake g++-14, no clang at all | `--configure-only` | `WARNING`, `Auto-selected g++-14`, TOOLCHAIN_FILE=`gcc-14` | 335, 338 → BINARY_TOOLCHAIN:29 |
| 27 | Fallback finds g++-13 (no clang) | Fake g++-13 only | `--configure-only` | `Auto-selected g++-13`, TOOLCHAIN_FILE=`gcc-13` | 335, 338 → BINARY_TOOLCHAIN:30 |
| 28 | Fallback finds g++-12 (no clang) | Fake g++-12 only | `--configure-only` | `Auto-selected g++-12`, TOOLCHAIN_FILE=`gcc-12` | 335, 338 → BINARY_TOOLCHAIN:31 |
| 29 | Fallback finds unversioned g++ (no clang) | Fake g++ only | `--configure-only` | `Auto-selected g++`, TOOLCHAIN_FILE=`gcc` | 335, 338 → BINARY_TOOLCHAIN:32 |

## Section 7: Auto-detection fallback — Nothing found (lines 335–336)

| # | Test | PATH setup | Command | Expected output | Lines |
|---|------|-----------|---------|-----------------|-------|
| 30 | Fallback finds nothing | Empty PATH (no compilers) | `--configure-only` | `WARNING`, `ERROR: No C++ compiler found.`, exit 1 | 335–336 |

## Section 8: Auto-detection guard conditions (line 332)

These verify that auto-detection is skipped when any explicit override is set.
All tests run WITHOUT clang++-20 in PATH to ensure the fallback would trigger
if the guard didn't prevent it.

| # | Test | Command | Expected output | Guard |
|---|------|---------|-----------------|-------|
| 31 | `--compiler` set skips auto-detect | `--compiler clang-20 --configure-only` | TOOLCHAIN_FILE=`clang-20-libstdcpp`, NO warning | `compiler != ""` |
| 32 | `--cxx-compiler-path` set skips auto-detect | `--cxx-compiler-path /usr/bin/g++ --configure-only` | CMAKE_CXX_COMPILER=`/usr/bin/g++`, NO warning | `cxx_compiler_path != ""` |
| 33 | `--toolchain-path` set skips auto-detect | `--toolchain-path cmake/x86_64-linux-gcc-14-toolchain.cmake --configure-only` | TOOLCHAIN_FILE=`gcc-14`, NO warning | `toolchain_path_explicitly_set = true` |

## Section 9: Search priority ordering (lines 18–19, via 311, 316)

| # | Test | PATH setup | Expected | Verifies |
|---|------|-----------|----------|----------|
| 34 | clang++-20 preferred over clang++-19 | Both clang++-20 and clang++-19 present | `found clang++-20` | CLANG_SEARCH descending order |
| 35 | clang++-18 preferred over clang++ | Both clang++-18 and clang++ present | `found clang++-18` | Versioned before unversioned |
| 36 | g++-14 preferred over g++-13 | Both g++-14 and g++-13 present | `found g++-14` | GCC_SEARCH descending order |
| 37 | g++-12 preferred over g++ | Both g++-12 and g++ present | `found g++-12` | Versioned before unversioned |

## Section 10: Help text and CMake presets

| # | Test | Command | Expected |
|---|------|---------|----------|
| 38 | `--help` shows `--compiler` | `build_metal.sh --help` | Output contains `--compiler compiler_name` and lists `clang`, `gcc` |
| 39 | CMakePresets.json valid | `cmake --list-presets 2>&1` | Output lists `gcc-system`, `gcc-13`, `clang-system`, `clang-19`, `clang-18`, `clang-17` presets |

## Section 11: `--toolchain-path` protection flag (line 290)

| # | Test | Command | Expected | Verifies |
|---|------|---------|----------|----------|
| 40 | `--toolchain-path` sets explicit flag | `--toolchain-path cmake/x86_64-linux-gcc-toolchain.cmake --configure-only` (no clang++-20) | Uses `gcc-toolchain`, NO auto-detect override | `toolchain_path_explicitly_set=true` (line 290) |
| 41 | Default has explicit flag false | `--configure-only` (with clang++-20) | Uses default `clang-20-libstdcpp` | `toolchain_path_explicitly_set=false` (line 165) |

---

# `install_dependencies.sh` tests

All tests below run as root in Docker containers. Use `--docker` flag to avoid
systemd-dependent steps. Verify behavior via output messages (grep for INFO,
WARNING, ERROR).

## Section 12: `--compiler` flag — validation (install_dependencies.sh)

| # | Test | Container | Command | Expected output |
|---|------|-----------|---------|-----------------|
| 42 | Valid `--compiler clang` | Ubuntu 22.04 | `./install_dependencies.sh --docker --compiler clang` | `[INFO] Compiler selection: clang`, LLVM 20 installed |
| 43 | Valid `--compiler gcc` | Ubuntu 22.04 | `./install_dependencies.sh --docker --compiler gcc` | `[INFO] Compiler selection: gcc`, `[INFO] Skipping LLVM installation (--compiler gcc)` |
| 44 | Valid `--compiler clang-20` | Ubuntu 22.04 | `./install_dependencies.sh --docker --compiler clang-20` | `[INFO] Compiler selection: clang-20`, LLVM 20 installed |
| 45 | Valid `--compiler gcc-14` | Ubuntu 22.04 | `./install_dependencies.sh --docker --compiler gcc-14` | `[INFO] Compiler selection: gcc-14`, `[INFO] Skipping LLVM installation (--compiler gcc-14)` |
| 46 | Invalid `--compiler foobar` | Ubuntu 22.04 | `./install_dependencies.sh --docker --compiler foobar` | `[ERROR] Unknown compiler 'foobar'. Allowed: clang gcc clang-20 clang-19 clang-18 clang-17 clang-20-libcpp gcc-14 gcc-13 gcc-12.`, exit 1 |
| 47 | Default (no --compiler) | Ubuntu 22.04 | `./install_dependencies.sh --docker` | LLVM 20 installed, no `[INFO] Compiler selection` |

## Section 13: `--compiler` flag — LLVM skip on GCC variants

| # | Test | Container | Command | Expected output |
|---|------|-----------|---------|-----------------|
| 48 | `--compiler gcc` skips LLVM | Ubuntu 22.04 | `./install_dependencies.sh --docker --compiler gcc` | `[INFO] Skipping LLVM installation (--compiler gcc)` |
| 49 | `--compiler gcc-12` skips LLVM | Ubuntu 22.04 | `./install_dependencies.sh --docker --compiler gcc-12` | `[INFO] Skipping LLVM installation (--compiler gcc-12)` |
| 50 | `--compiler gcc-14` skips LLVM | Ubuntu 24.04 | `./install_dependencies.sh --docker --compiler gcc-14` | `[INFO] Skipping LLVM installation (--compiler gcc-14)` |
| 51 | `--compiler gcc-13` skips LLVM | Ubuntu 22.04 | `./install_dependencies.sh --docker --compiler gcc-13` | `[INFO] Skipping LLVM installation (--compiler gcc-13)` |
| 52 | `--compiler clang-20-libcpp` installs LLVM | Ubuntu 22.04 | `./install_dependencies.sh --docker --compiler clang-20-libcpp` | LLVM 20 installed (no skip message) |

## Section 14: `install_llvm()` — RedHat messaging

| # | Test | Container | Command | Expected output |
|---|------|-----------|---------|-----------------|
| 53 | Default on Fedora: distro clang | Fedora 40 | `./install_dependencies.sh --docker` | `[INFO] Using distro-provided LLVM/Clang for fedora:` (no WARNING) |
| 54 | `--compiler gcc` on Fedora | Fedora 40 | `./install_dependencies.sh --docker --compiler gcc` | `[INFO] Skipping LLVM installation (--compiler gcc)` |
| 55 | `--compiler clang` on Fedora | Fedora 40 | `./install_dependencies.sh --docker --compiler clang` | `[INFO] Using distro-provided LLVM/Clang for fedora:` |

## Section 15: `install_llvm_from_tarball()` — LLVM tarball installation

| # | Test | Container | Command | Expected output |
|---|------|-----------|---------|-----------------|
| 56 | `--compiler clang-20` installs LLVM 20 tarball | Fedora 40 | `./install_dependencies.sh --docker --compiler clang-20` | Downloads `LLVM-20.1.8-Linux-X64.tar.xz`, installs to `/usr/local/llvm-20` |
| 57 | `--compiler clang-19` installs LLVM 19 tarball | Fedora 40 | `./install_dependencies.sh --docker --compiler clang-19` | Downloads `LLVM-19.1.7-Linux-X64.tar.xz`, installs to `/usr/local/llvm-19` |
| 58 | `--compiler clang-17` installs LLVM 17 tarball | Fedora 40 | `./install_dependencies.sh --docker --compiler clang-17` | Downloads `clang+llvm-17.0.6-x86_64-linux-gnu-ubuntu-22.04.tar.xz` (old naming), installs to `/usr/local/llvm-17` |

## Section 16: `prep_redhat_system()` — repo setup

| # | Test | Container | Command | Expected output |
|---|------|-----------|---------|-----------------|
| 59 | Fedora: no extra repos | Fedora 40 | `./install_dependencies.sh --docker` | `[INFO] Preparing Red Hat family system...` then proceeds (no EPEL) |
| 60 | AlmaLinux: EPEL + CRB | AlmaLinux 10 | `./install_dependencies.sh --docker` | `[INFO] Installing EPEL repository...`, `[INFO] Enabling CRB repository` |
| 61 | Oracle Linux: Oracle EPEL + CRB | Oracle Linux 10 | `./install_dependencies.sh --docker` | `oracle-epel-release-el10` installed, `ol10_codeready_builder` enabled |

## Section 17: `verify_compiler()` — GCC version guard

| # | Test | Container | Command | Expected output |
|---|------|-----------|---------|-----------------|
| 62 | GCC >= 12 passes | Ubuntu 22.04 (GCC 12) | `./install_dependencies.sh --docker --compiler gcc` | `[OK] Compiler verified:` |
| 63 | GCC < 12 rejected | System with GCC 11 | `./install_dependencies.sh --docker --compiler gcc` | `[ERROR] GCC 11 is too old. tt-metal requires GCC >= 12.`, exit 1 |

## Section 18: `use_compiler()` — C compiler derivation check (build_metal.sh)

| # | Test | PATH setup | Command | Expected output |
|---|------|-----------|---------|-----------------|
| 64 | Missing C compiler detected | Fake `g++` in PATH, no `gcc` | `--compiler gcc --configure-only` | `ERROR: C compiler 'gcc' not found (derived from 'g++')`, exit 1 |
| 65 | Missing C compiler (clang) | Fake `clang++` in PATH, no `clang` | `--compiler clang --configure-only` | `ERROR: C compiler 'clang' not found (derived from 'clang++')`, exit 1 |

## Section 19: Warning fixes — MPI ULFM and hugepages

| # | Test | Container | Command | Expected output |
|---|------|-----------|---------|-----------------|
| 66 | MPI ULFM on Fedora: INFO not WARNING | Fedora 40 | `./install_dependencies.sh --docker` | `[INFO] MPI ULFM is only available as a .deb package; skipping on fedora` |
| 67 | Hugepages on Fedora: INFO not WARNING | Fedora 40 | `./install_dependencies.sh --docker --hugepages` | `[INFO] Hugepages package is only available as a .deb package; skipping on fedora` |

## Section 20: Help text (install_dependencies.sh)

| # | Test | Command | Expected output |
|---|------|---------|-----------------|
| 68 | `--help` shows `--compiler` | `./install_dependencies.sh --help` | Output contains `[--compiler name]` and lists `clang gcc clang-20 clang-19 clang-18 clang-17 clang-20-libcpp gcc-14 gcc-13 gcc-12` |

## Coverage Summary

**`build_metal.sh`:**

| Code section | Lines | Tests |
|-------------|-------|-------|
| `CLANG_SEARCH` / `GCC_SEARCH` arrays | 18–19 | 1–11, 22–30, 34–37 |
| `BINARY_TOOLCHAIN` map | 23–33 | 1–5, 7–10, 22–29 |
| `FLAG_TOOLCHAIN` map | 39–48 | 12–19 |
| `toolchain_file()` | 51–53 | 1–5, 7–10, 12–19, 22–29 |
| `find_compiler()` search | 57–65 | 1–11, 22–30, 34–37 |
| `use_compiler()` toolchain path | 72–73 | 1–5, 7–10, 22–29 |
| `use_compiler()` C compiler guard | 83–86 | 64–65 |
| `--compiler clang` case | 310–314 | 1–6, 34–35 |
| `--compiler gcc` case | 315–319 | 7–11, 36–37 |
| `--compiler` pinned (FLAG_TOOLCHAIN) | 320–322 | 12–19 |
| `--compiler` error (unknown) | 323–327 | 20 |
| Auto-detect: clang++-20 present | 332–333 | 21 |
| Auto-detect: compiler fallback | 334–339 | 22–29 |
| Auto-detect: nothing found | 335–336 | 30 |
| Auto-detect: guard conditions | 332 | 31–33 |
| `--toolchain-path` explicit flag | 165, 290 | 32–33, 40–41 |
| Help text | 123 | 38 |
| CMakePresets.json | — | 39 |

**`install_dependencies.sh`:**

| Code section | Tests |
|-------------|-------|
| `COMPILER_FLAGS` array + validation | 42–47, 68 |
| `--compiler gcc*` skips `install_llvm()` | 43, 45, 48–51 |
| `--compiler clang*` installs LLVM | 42, 44, 52 |
| `install_llvm()` RedHat: distro clang INFO | 53, 55 |
| `install_llvm()` RedHat: skip on gcc | 54 |
| `install_llvm_from_tarball()` LLVM 20/19 (new naming) | 56, 57 |
| `install_llvm_from_tarball()` LLVM 17 (old naming) | 58 |
| `prep_redhat_system()` Fedora: no-op | 59 |
| `prep_redhat_system()` RHEL/Alma: EPEL+CRB | 60 |
| `prep_redhat_system()` Oracle: EPEL+CRB | 61 |
| `verify_compiler()` GCC >= 12 guard | 62–63 |
| MPI ULFM warning → INFO | 66 |
| Hugepages warning → INFO | 67 |
| Help text | 68 |

| Script | Tests |
|--------|-------|
| `build_metal.sh` | 41 |
| `install_dependencies.sh` | 27 |
| **Total** | **68 tests** |
