# Clang-Tidy in tt-metal

tt-metal uses clang-tidy for static analysis. The CI runs it automatically on every PR
and on `main` daily. This page explains how to run it locally and how to manage checks.

## Running locally (dev container — recommended)

The dev container ships clang-tidy 20 and all required tools. No extra installation needed.

```sh
docker pull ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-24.04-dev-amd64:latest
docker run -it --rm -v $(pwd):/work -w /work \
  ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-24.04-dev-amd64:latest bash
```

```sh
# Configure
cmake --preset clang-tidy

# Scan a specific target (e.g. tt_metal, ttnn, tt-train)
cmake --build --preset clang-tidy --target <target>

# Or scan everything
cmake --build --preset clang-tidy
```

For a quick spot-check on specific files before committing:

```sh
# Configure (from repo root, disable PCHs which interfere with clang-tidy)
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DCMAKE_DISABLE_PRECOMPILE_HEADERS=ON \
      -DCMAKE_C_COMPILER=clang-20 -DCMAKE_CXX_COMPILER=clang++-20

# Scan specific file(s)
run-clang-tidy -p ./build -quiet -use-color \
  -header-filter "$(pwd)" \
  -source-filter "$(pwd)" \
  -exclude-header-filter "(.*CPM.*|.*tt-train.*|.*third_party.*)" \
  ./path/to/your/file.cpp
```

The key flags: `-p ./build` points to `compile_commands.json`; `-header-filter` and
`-source-filter` restrict scanning to the repo tree; `-exclude-header-filter` skips
third-party headers.

## Prerequisites (without the dev container)

If you are building outside the dev container, you need:

| Tool | Version | Ubuntu package |
|------|---------|----------------|
| clang / clang++ | **20** | `apt install clang-20` |
| clang-tidy | **20** | `apt install clang-tidy-20` |
| run-clang-tidy | **20** | `apt install clang-tools-20` |
| cmake | ≥ 3.22 | `apt install cmake` |
| ninja | any | `apt install ninja-build` |

> **Version matters.** `apt install clang-tidy` without a version suffix installs an
> older version on most distros and will produce different results from CI.

## How CI runs clang-tidy

The workflow ([`code-analysis.yaml`](../.github/workflows/code-analysis.yaml)) chooses
between a full scan and an incremental scan depending on context.

**Full scan** — runs when:
- A commit lands on `main` (also runs daily at 2 AM UTC)
- The `.clang-tidy` config changes (even on a PR branch)

Full scan analyzes the entire codebase in a single job (~25 minutes on CI runners).

**Incremental scan** — runs on PR branches when code or CMake files change. Diffs
against the merge-base and analyzes only files touched by the PR. Split into 4 parallel
jobs to keep wall-clock time down:

```
Job             Targets scanned   Prereq targets built first
─────────────── ───────────────── ──────────────────────────
Metalium        tt_metal          —
TTNN            ttnn              tt_metal
tt-train        tt-train          tt_metal  ttnn
catchall        (everything else) tt_metal  ttnn
```

**Skip** — if no code or CMake files changed, clang-tidy is skipped entirely.

> We scan during a CMake build rather than pointing `run-clang-tidy` directly at
> `compile_commands.json`. This lets CMake exclude third-party code that ships its own
> `.clang-tidy` files we cannot override.

## Suppressing diagnostics

Prefer narrow, named suppressions over blanket `NOLINT`:

```cpp
some_call();  // NOLINT(check-name)    — suppress on this line
// NOLINTNEXTLINE(check-name)
some_call();                           — suppress the following line
```


## Enable a check

1. Remove the `-check-name` line from [`.clang-tidy`](../.clang-tidy). Check the
   [Clang-Tidy Checks](https://clang.llvm.org/extra/clang-tidy/checks/list.html) list
   for aliases — if another name refers to the same check, remove that too.
2. Run a scan locally to count violations. If there are hundreds, consider opening a
   tracking issue before proceeding.
3. Fix all violations (see below).
4. Open a PR with the `.clang-tidy` change and all fixes together.

> We disable specific checks explicitly (rather than enabling a curated subset) so the
> disable list is a visible TODO, not a silent omission.

## Fix violations

### Automatic (checks with FIX-ITs)

If the check has FIX-ITs (see the "Offers fixes" column on the checks page), use the
parallel preset: it exports fixes to YAML files first, then applies them all at once
with `clang-apply-replacements-20`. This avoids the race condition where parallel
clang-tidy instances corrupt headers that are included by multiple TUs.

```sh
# Step 1: build and collect fixes (does not modify source yet)
cmake --preset clang-tidy-fix-parallel
cmake --build --preset clang-tidy-fix-parallel --target clean
cmake --build --preset clang-tidy-fix-parallel

# Step 2: apply all collected fixes atomically
clang-apply-replacements-20 .build/clang-tidy-fix-parallel/fixes
```

This will take several hours on a 12-core machine. When complete, review the diffs and
re-run both steps until no new fixes are applied. Then do a final clean build to confirm
nothing is broken.

**Note:** some checks only fix a subset of what they diagnose (e.g.,
`performance-unnecessary-value-param` skips templates and lambdas), and some fixes
leave the repo in an unbuildable state (e.g., fixing a definition without updating a
forward declaration). Always do a clean build after applying fixes.

### Manual

Same build flow as above (`clang-tidy-fix-parallel` or just `clang-tidy`), but after
each build review the log and address each diagnostic by hand.

