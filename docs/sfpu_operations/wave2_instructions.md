# Wave 2 (Final): Remaining SFPU Operations Instructions

**For**: Next Claude session
**Date**: 2026-04-04
**Branch**: `vignjatijevic/sfpu-agent-codegen_kernel_bench`
**Plan file**: `/home/vignjatijevic/.claude/plans/zesty-floating-otter.md`

---

## Background

Waves 0 and 1 implemented 10 SFPU unary operations for the kernel_bench benchmark. Only **4 more operations** have kernel_bench benchmarks AND lack `vignjatijevic-sfpu-generator` submissions. This is the final wave.

### Previous Waves

**Wave 0** (5 ops): hardsigmoid 100%, hardtanh 97.72%, cbrt 82.05%, cosh 48.46%, selu 43.96%
**Wave 1** (5 ops): results in `VIGNJATIJEVIC_LEADERBOARD.md`

### Lessons from Waves 0-1

1. **Fix build before generators** -- commit all stubs/fixes to main first
2. **Per-worktree cleanup** -- delete surviving `ckernel_sfpu_*.h` for target op only
3. **Monitor for primitive reuse** -- if generator finds existing code, kill/clean/retry
4. **kernel_bench is ground truth** -- generator tests may pass but kernel_bench may fail
5. **Merge one branch at a time** -- all branches modify same shared files
6. **Exponential-heavy ops score poorly** -- sinh will likely struggle (same pattern as cosh/selu)

---

## Wave 2 Operations (4 ops)

| Op | Formula | Parametrized | kernel_bench path |
|----|---------|-------------|-------------------|
| **FRAC** | `x - floor(x)` | No | `benchmark/frac/` |
| **HARDSWISH** | `x * min(max(x+3, 0), 6) / 6` | No | `benchmark/hardswish/` |
| **SINH** | `(exp(x) - exp(-x)) / 2` | No | `benchmark/sinh/` |
| **SOFTSHRINK** | `x - lambda if x > lambda; x + lambda if x < -lambda; 0 otherwise` | Yes (lambda, default=0.5) | `benchmark/softshrink/` |

---

## Execution Steps

### Step 0: Clean Up Wave 1 Worktrees

```bash
for op in atanh softsign lgamma rpow swish; do
    git worktree remove .claude/worktrees/gen-${op} --force
done
```

### Step 1: Check for Surviving Kernel Files

The batch nuke missed many `ckernel_sfpu_*.h` files. Check and delete survivors for the 4 target ops:

```bash
grep -rn 'frac\|hardswish\|softshrink' tt_metal/hw/ckernels/*/metal/llk_api/llk_sfpu/
grep -rn 'sinh' tt_metal/hw/ckernels/*/metal/llk_api/llk_sfpu/  # careful: may match cosh/asinh
```

**Known survivors to check:**
- `hardswish_kernel.cpp` and `hardswish_kernel_sfpu.cpp` in `ttnn/.../kernels/compute/` -- DELETE these
- `ckernel_sfpu_trigonometry.h` may contain `calculate_sinh` -- surgical edit to remove sinh only
- `ckernel_sfpu_softshrink.h` if it exists -- DELETE

Delete survivors and commit before creating worktrees.

### Step 2: Handle SOFTSHRINK Registration Gap

SOFTSHRINK has `UnaryOpType::SOFTSHRINK` in the enum but **NO** `REGISTER_UNARY_OPERATION` macro in `unary.hpp` and **NO** nanobind binding. The generator should handle this (it discovers the enum and creates the registration), but verify after generation.

### Step 3: Create 4 Worktrees

```bash
git worktree add .claude/worktrees/gen-frac -b gen-frac
git worktree add .claude/worktrees/gen-hardswish -b gen-hardswish
git worktree add .claude/worktrees/gen-sinh -b gen-sinh
git worktree add .claude/worktrees/gen-softshrink -b gen-softshrink
```

### Step 4: Per-Worktree Cleanup

In each worktree, delete ONLY its target op's surviving files:

- **gen-hardswish**: Delete `hardswish_kernel.cpp`, `hardswish_kernel_sfpu.cpp`, any `hardswish.h` compute API header
- **gen-sinh**: Check `ckernel_sfpu_trigonometry.h` for `calculate_sinh` -- remove surgically if present
- **gen-frac**: Should be clean (verify)
- **gen-softshrink**: Should be clean (verify)

Commit cleanup in each worktree before launching generator.

### Step 5: Launch 4 Generators in Parallel

Use `ttnn-unary-sfpu-operation-generator` with `run_in_background: true`:

- **FRAC**: name=`frac`, definition=`x - floor(x)` (fractional part), no params
- **HARDSWISH**: name=`hardswish`, definition=`x * min(max(x+3, 0), 6) / 6`, no params
- **SINH**: name=`sinh`, definition=`(exp(x) - exp(-x)) / 2`, no params
- **SOFTSHRINK**: name=`softshrink`, definition=`x - lambda if x > lambda; x + lambda if x < -lambda; 0 otherwise`, param: lambda (float, default=0.5)

Add to each prompt: `IMPORTANT: Work inside the worktree at /localdev/vignjatijevic/tt-metal/.claude/worktrees/gen-{op}/ — this is your working directory.`

### Step 6: Monitor, Merge, Build, Test

1. Monitor for reuse -- verify NEW `ckernel_sfpu_*.h` created in each worktree
2. Merge one branch at a time into main, resolve additive conflicts
3. `./build_metal.sh` -- must pass
4. Run tests:
```bash
source python_env/bin/activate
for op in frac hardswish sinh softshrink; do
    scripts/run_safe_pytest.sh --dev tests/ttnn/unit_tests/operations/eltwise/test_${op}.py -v
done
```

### Step 7: kernel_bench Submissions

```bash
cd /localdev/vignjatijevic/kernel_bench
for op in frac hardswish sinh softshrink; do
    mkdir -p submissions/${op}/vignjatijevic-sfpu-generator/result/1/code
    ln -sf 1 submissions/${op}/vignjatijevic-sfpu-generator/result/latest
    # Create ttnn_{op}_impl.py (see below)
    uv run eval run --operation ${op} --submission vignjatijevic-sfpu-generator
done
uv run eval report --all
```

**Standard impl** (frac, hardswish, sinh):
```python
import ttnn
def ttnn_{op}(*args, **kwargs):
    return ttnn.{op}(args[0])
```

**Parametrized impl** (softshrink):
```python
import ttnn
def ttnn_softshrink(*args, **kwargs):
    lambd = kwargs.get("lambd", 0.5)
    return ttnn.softshrink(args[0], lambd)
```

### Step 8: Update Leaderboard & Push

Update `/localdev/vignjatijevic/kernel_bench/VIGNJATIJEVIC_LEADERBOARD.md`:
- Add Wave 2 section with 4 new results
- Update totals (should be 14 ops total across 3 waves)

```bash
git add submissions/ VIGNJATIJEVIC_LEADERBOARD.md README.md
git commit -m "Add Wave 2 (final) vignjatijevic-sfpu-generator results"
git push origin vignjatijevic/add-rrelu-benchmark
```

---

## Key Paths

| What | Path |
|------|------|
| tt-metal repo | `/localdev/vignjatijevic/tt-metal` |
| kernel_bench repo | `/localdev/vignjatijevic/kernel_bench` |
| kernel_bench branch | `vignjatijevic/add-rrelu-benchmark` |
| Agent definitions | `tt_metal/third_party/tt_ops_code_gen/agents/sfpu-agents/` |
| Leaderboard | `/localdev/vignjatijevic/kernel_bench/VIGNJATIJEVIC_LEADERBOARD.md` |
| Worktrees | `.claude/worktrees/gen-{op}/` |
| Surviving custom kernels | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/` |

## Agent Constraints (already configured from Wave 0)

- No internet (WebFetch, WebSearch, DeepWiki, Confluence, Glean removed)
- No git history (`git show` of deleted code forbidden)
- Isolated worktrees (each generator sees only its own work)

## Verification

1. Build passes after merge
2. `python -c "import ttnn"` succeeds
3. All 4 test suites pass
4. kernel_bench eval produces results for all 4 ops
5. VIGNJATIJEVIC_LEADERBOARD.md updated with Wave 2 results
6. Push to `vignjatijevic/add-rrelu-benchmark` succeeds
