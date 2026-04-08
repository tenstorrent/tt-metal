# Wave 1: SFPU Operation Regeneration Instructions

**For**: Next Claude session
**Date**: 2026-04-04
**Branch**: `vignjatijevic/sfpu-agent-codegen_kernel_bench`
**Master plan**: `/home/vignjatijevic/.claude/plans/merry-twirling-turing.md`
**Wave 1 plan**: `/home/vignjatijevic/.claude/plans/breezy-wandering-wall.md`

---

## Background

We're benchmarking the `ttnn-unary-sfpu-operation-generator` agent's ability to implement SFPU unary eltwise operations from scratch. All 109 ops were nuked (commit `db3f683e0a5`). Generators run offline (no internet, no git history access) in isolated worktrees. Results are submitted to kernel_bench and appear on the dashboard.

### Wave 0 Results (complete)

| Op | kernel_bench Pass Rate | Key Issue |
|----|----------------------|-----------|
| hardsigmoid | **100.00%** (261/261) | None |
| hardtanh | **97.72%** (257/263) | Rank-0 tensor edge cases |
| cbrt | **82.05%** (224/273) | Precision in Newton-Raphson approximation |
| cosh | 48.46% (110/227) | Exponential precision (`_calculate_exponential_piecewise_`) |
| selu | 43.96% (120/273) | `exp(x)-1` catastrophic cancellation + constant precision |

### Lessons Learned

1. **Fix build BEFORE launching generators** -- Wave 0 agents wasted 30+ min fixing nuke aftermath stubs
2. **Per-worktree cleanup** -- remove surviving `ckernel_sfpu_*.h` files for each target op ONLY in its worktree
3. **Monitor for primitive reuse** -- if a generator finds and reuses a surviving kernel file, kill it, clean up, retry
4. **kernel_bench eval is the ground truth** -- generator's own tests pass but kernel_bench may fail (different shapes, stricter tolerances)
5. **Exponential-heavy ops score poorly** -- ops using `exp()` got 44-48%; pure clamp/comparison ops got 98-100%
6. **Merge carefully** -- 5 branches all modify the same files; merge one at a time, verify syntax after each

---

## Wave 1 Operations

| Op | Formula | Family | Parametrized | kernel_bench path |
|----|---------|--------|-------------|-------------------|
| **ATANH** | 0.5 * ln((1+x)/(1-x)) | TRIG_FAMILY | No | `benchmark/atanh/` |
| **SOFTSIGN** | x / (1 + \|x\|) | ACTIVATIONS | No | `benchmark/softsign/` |
| **LGAMMA** | ln(\|Gamma(x)\|) | LGAMMA (standalone) | No | `benchmark/lgamma/` |
| **RPOW** | base^x | RPOW (standalone) | Yes (base float) | `benchmark/rpow/` |
| **SWISH** | x * sigmoid(x) | SILU (standalone) | No | `benchmark/swish/` |

---

## Execution Steps

### Step -1: Nuke MISH and LOGIT

These 2 ops were excluded from the original batch nuke but should be removed so generators don't discover them.

**MISH** -- delete: `mish_kernel.cpp`, dispatch cases, bindings, `REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(mish, MISH)`, Python golden.
**LOGIT** -- delete: `logit_kernel.cpp`, dispatch cases, bindings, stub `logit()` to throw, Python golden.
**Keep**: IDENTITY, DROPOUT, ZERO_POINT (too deeply integrated).

Verify build passes after.

### Step 0: Commit Build Stubs

The HARDTANH tester from Wave 0 left uncommitted build-fix stubs on main (SfpuType enum restoration, unary.hpp stubs, nanobind `#if 0` blocks, Python import guards). These must be committed so worktrees inherit a working build.

Files to check/commit:
- `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu_types.h`
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp`
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp`
- `ttnn/cpp/ttnn/operations/eltwise/unary_ng/unary_ng.{hpp,cpp}`
- `ttnn/ttnn/__init__.py`, `ttnn/ttnn/graph.py`
- `ttnn/ttnn/operations/__init__.py`, `ttnn/ttnn/operations/unary.py`
- `ttnn/ttnn/experimental_loader/golden_functions.py`

Run: `./build_metal.sh` -- must pass.
Run: `source python_env/bin/activate && python -c "import ttnn"` -- must succeed.

### Step 1: Create 5 Worktrees

```bash
git worktree add .claude/worktrees/gen-atanh -b gen-atanh
git worktree add .claude/worktrees/gen-softsign -b gen-softsign
git worktree add .claude/worktrees/gen-lgamma -b gen-lgamma
git worktree add .claude/worktrees/gen-rpow -b gen-rpow
git worktree add .claude/worktrees/gen-swish -b gen-swish
```

### Step 2: Per-Worktree Cleanup

In each worktree, delete ONLY the target op's surviving kernel files + SfpuType enum entry. The `tt_llk` submodule is empty in worktrees (not auto-initialized), so shared primitives are already gone.

Check for surviving files:
```bash
# In each worktree:
grep -rn '{op_name}' tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/
```

If files found: delete them and remove the SfpuType enum entry. Commit cleanup before launching generator.

### Step 3: Launch 5 Generators in Parallel

Use `ttnn-unary-sfpu-operation-generator` agent type with `run_in_background: true`. Each gets:

- **ATANH**: name=`atanh`, definition=`0.5 * ln((1+x)/(1-x))`, inverse hyperbolic tangent, no params
- **SOFTSIGN**: name=`softsign`, definition=`x / (1 + |x|)`, no params
- **LGAMMA**: name=`lgamma`, definition=`ln(|Gamma(x)|)`, log-gamma special function, no params
- **RPOW**: name=`rpow`, definition=`base^x` where base is a float parameter, param: base (float)
- **SWISH**: name=`swish`, definition=`x * sigmoid(x) = x / (1 + exp(-x))`, no params. Note: swish is also known as SiLU. The UnaryOpType enum entry is SILU.

Add to each prompt: `IMPORTANT: Work inside the worktree at /localdev/vignjatijevic/tt-metal/.claude/worktrees/gen-{op}/ — this is your working directory.`

### Step 4: Monitor for Reuse

After each generator completes, check:
1. Did it create a NEW `ckernel_sfpu_*.h` file? (good)
2. Did it say "already implemented" or reuse a surviving file? (bad -- kill, cleanup, retry)
3. Check `git log` in the worktree for implementation commits

### Step 5: Merge & Build

Merge one branch at a time into main:
```bash
git merge gen-atanh --no-edit
# verify no syntax errors
git merge gen-softsign --no-edit
# ...etc for all 5
./build_metal.sh
```

Conflicts are expected in shared files (all additive). Resolve by keeping BOTH additions.

### Step 6: Run Tests

```bash
source python_env/bin/activate
for op in atanh softsign lgamma rpow swish; do
  scripts/run_safe_pytest.sh --dev tests/ttnn/unit_tests/operations/eltwise/test_${op}.py -v
done
```

### Step 7: kernel_bench Submissions

For each op, create submission and run eval:
```bash
cd /localdev/vignjatijevic/kernel_bench
# For swish, kernel_bench operation name is "swish" (not "silu")
for op in atanh softsign lgamma rpow swish; do
  mkdir -p submissions/${op}/vignjatijevic-sfpu-generator/result/1/code
  ln -sf 1 submissions/${op}/vignjatijevic-sfpu-generator/result/latest
  # Create ttnn_${op}_impl.py with: import ttnn; def ttnn_${op}(*args, **kwargs): return ttnn.${op}(args[0])
  # For swish: ttnn.silu(args[0]) since swish=silu in ttnn
  uv run eval run --operation ${op} --submission vignjatijevic-sfpu-generator
done
uv run eval report --all
```

### Step 8: Update Leaderboard & Push

Update `/localdev/vignjatijevic/kernel_bench/VIGNJATIJEVIC_LEADERBOARD.md`:
- Add Wave 1 results to the table
- Update summary stats
- Add wave-by-wave comparison section

```bash
git add submissions/ VIGNJATIJEVIC_LEADERBOARD.md README.md
git commit -m "Add Wave 1 vignjatijevic-sfpu-generator results"
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
| Key notes per op | `docs/sfpu_operations/key_notes/{op}_key_notes.md` |
| Leaderboard | `/localdev/vignjatijevic/kernel_bench/VIGNJATIJEVIC_LEADERBOARD.md` |
| Worktrees | `.claude/worktrees/gen-{op}/` |
| Generator output | `.claude-analysis/{op}-1/` (inside worktree) |

## Agent Constraints (already configured)

- No internet (WebFetch, WebSearch, DeepWiki, Confluence, Glean removed from agent defs)
- No git history (`git show` of deleted code forbidden in agent instructions)
- Isolated worktrees (each generator sees only its own work)
