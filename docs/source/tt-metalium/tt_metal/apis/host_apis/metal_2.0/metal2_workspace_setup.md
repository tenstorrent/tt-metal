# Metal 2.0 — Workspace Setup for Porting

Use this when you have **no existing tt-metal checkout**. If you've been given an existing clone or worktree path, skip to **Run tests**.

This doc is referenced by the [port recipe](port_op_to_metal2_recipe.md) under "Before you begin." Scope: bootstrap only — clone, environment, build, test invocation patterns. The port itself lives in the recipe.

## Clone

```bash
git clone git@github.com:tenstorrent/tt-metal.git --recurse-submodules
cd tt-metal
```

If SSH isn't configured, fall back to HTTPS:

```bash
git clone https://github.com/tenstorrent/tt-metal.git --recurse-submodules
```

## Python environment

```bash
export PYTHONPATH=$(pwd)
./create_venv.sh
source python_env/bin/activate
```

`PYTHONPATH` must point to your actual clone. Use `$(pwd)` from inside the clone — do not hardcode a path copied from someone else's environment.

## Build

```bash
./build_metal.sh --build-tests
```

Use `--build-tests` (Metal **and** TTNN tests). Do **not** use `--build-metal-tests` alone — that flag omits the TTNN gtests an op porter needs.

Cold build on a warm farm node: ~7 minutes for Metal tests alone; expect a few minutes more for the full `--build-tests`. Subsequent incremental builds are fast.

For iterative rebuilds during a port, target binaries directly:

```bash
cmake --build build_Release --target ttnncpp unit_tests_ttnn -j 8
```

## Run tests

**Metal 2.0 unit tests** (sanity-check the build):

```bash
./build/test/tt_metal/unit_tests_api --gtest_filter="*ProgramSpec*"
```

**Op-specific gtests** live under `./build/test/ttnn/`. The umbrella binary is `unit_tests_ttnn`; specialized variants (`unit_tests_ttnn_tensor`, `unit_tests_ttnn_ccl`, etc.) exist alongside it.

```bash
# List what's there
ls ./build/test/ttnn/

# Discover the test names without running anything (safe without HW)
./build/test/ttnn/unit_tests_ttnn --gtest_list_tests | head

# Run tests filtered to your op (requires HW)
./build/test/ttnn/unit_tests_ttnn --gtest_filter="*<YourOp>*"
```

Tests that exercise the device require attached hardware. On a farm node with a reservation, this is satisfied.

## Run tests for the op you ported

Tests for an op `<op>` live in two predictable places:

- **C++ gtests:** `tests/ttnn/unit_tests/gtests/test_<op>.cpp` (linked into the umbrella binary `./build/test/ttnn/unit_tests_ttnn`). UDM/specialized variants live in subdirs like `gtests/udm/<op>/` and link into sibling binaries (`unit_tests_ttnn_udm`, `unit_tests_ttnn_tensor`, `unit_tests_ttnn_ccl`, ...).
- **Python pytests:** `tests/ttnn/unit_tests/operations/<op-family-slug>/` (plus nightly variants under `tests/ttnn/nightly/unit_tests/operations/<op-family-slug>/`).

⚠ The pytest directory uses the **op-family slug**, not always the literal op name (e.g., reduction's tests live at `tests/ttnn/unit_tests/operations/reduce/`, not `reduction/`). The recipe asks the invoker to supply this path — confirm with `find` if unsure.

### Find the tests for `<op>`

```bash
# C++ sources
find tests/ttnn -path '*<op>*' -name '*.cpp'

# Python sources
find tests/ttnn -path '*operations/<op>*' -name 'test_*.py'

# Which gtest binary owns the C++ tests (discovery only, no HW needed)
for b in ./build/test/ttnn/unit_tests_ttnn*; do
  echo "== $b =="; "$b" --gtest_list_tests 2>/dev/null | grep -i '<op>'
done

# Verify pytest collection without running
pytest tests/ttnn/unit_tests/operations/<op>/ --collect-only -q
```

### Run them (requires HW)

```bash
./build/test/ttnn/unit_tests_ttnn --gtest_filter='*<Op>*'      # fast, ~op-scoped
pytest tests/ttnn/unit_tests/operations/<op>/ -x               # broader coverage
```

**Recommended order:** run the gtest filter first — it's faster and exercises the C++ op directly, so it fails fast on a broken port. Run pytests after gtests are green to cover the Python-API surface, dtype/layout sweeps, and program-cache behavior. Nightly pytests are heavy; skip unless you suspect a regression they'd catch.

### Worked example: `reduction`

```bash
# C++: test_reduction.cpp -> unit_tests_ttnn (tests: SumTensor*, MinMaxTensor* fixtures)
./build/test/ttnn/unit_tests_ttnn --gtest_list_tests | grep -iE '(sum|minmax)tensor'
./build/test/ttnn/unit_tests_ttnn --gtest_filter='SumTensor*:MinMaxTensor*'

# UDM reduction lives in a sibling binary
./build/test/ttnn/unit_tests_ttnn_udm --gtest_filter='*Reduction*'

# Python
pytest tests/ttnn/unit_tests/operations/reduce/ -x
```

Note: the pytest directory is `reduce/`, not `reduction/` — directory names are op-family slugs and not always the obvious noun. Use the `find` command above to confirm.

## Friction surfaced during validation

1. **`--build-metal-tests` is insufficient** for op porting — it omits TTNN gtests entirely. Always use `--build-tests` (or pass both `--build-metal-tests` and `--build-ttnn-tests` together).
2. **`PYTHONPATH`** must point to your actual clone path. Copying a hardcoded `/proj_sw/user_dev/$USER/tt-metal` from someone else's instructions will silently set the wrong path if you cloned elsewhere.
3. **`build/`** is a symlink (e.g., to `build_Release/`). Either path works in invocations.
4. **`reduce/` vs `reduction/`** — pytest directories use op-family slugs. Always verify with `find`.
