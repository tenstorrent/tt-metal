# The Generality Problem in Op Evaluation

## What we measure today

The current eval infrastructure answers one question:

> Did the agent produce a working op for the exact configuration the prompt asked for?

For `layer_norm_rm`, that configuration is:
- **Memory**: DRAM interleaved (`ttnn.DRAM_MEMORY_CONFIG`)
- **Layout**: ROW_MAJOR input and output
- **Dtype**: bfloat16
- **Alignment**: All shapes tile-aligned (H, W divisible by 32)

The golden test suite has ~100 parametrized cases, but every single one uses the
same `to_ttnn()` helper that hardcodes these four choices. The variation is only
in tensor shape and operation parameters (gamma/beta/epsilon). This is a thorough
test of one specific point in a large configuration space.

## Why this is insufficient

In TTNN, an operation is not just "tensor in, tensor out." The same mathematical
operation (layer norm, reduce, matmul) needs to work across a combinatorial space
of hardware configurations. The key axes are:

### 1. Memory configuration

Where the tensor lives and how it is laid out across banks:

| Config | Description | Difficulty |
|--------|-------------|------------|
| DRAM interleaved | Pages striped across DRAM banks. Simplest. | Baseline |
| L1 interleaved | Pages striped across L1 banks (faster, smaller) | Moderate |
| Height sharded | Tensor rows distributed across cores | Hard |
| Width sharded | Tensor columns distributed across cores | Hard |
| Block sharded | 2D blocks distributed across core grid | Hardest |

An op that only works with DRAM interleaved is functionally correct but limited.
Real model deployments use sharded memory to avoid DRAM bandwidth bottlenecks.
The sharded configs require the kernels to know their shard boundaries, read from
local L1 instead of issuing NoC reads, and handle cases where the shard size
doesn't evenly divide the tensor.

### 2. Tile alignment

Tensix cores operate on 32x32 tiles. When H or W is not divisible by 32, the
implementation must handle padding — either explicitly (pad on host, unpad after)
or implicitly (the kernel reads partial tiles and masks out garbage values).

- **Tile-aligned**: H=128, W=256 — tiles pack perfectly, no edge cases
- **Non-tile-aligned**: H=33, W=65 — last tile row/column is partially filled

Tile-aligned is what the prompt asks for. Non-tile-aligned is what real models
need (e.g. sequence lengths of 2048+7, hidden dims of 768).

### 3. Data types

- **bfloat16**: Standard for training. 16-bit, hardware-native.
- **float32**: Higher precision, needed for accumulation and some inference.
- **bfloat8_b / bfloat4_b**: Quantized formats for inference efficiency.

The kernel math changes depending on dtype (different tile sizes for 8-bit, different
accumulation strategies). An op that only handles bfloat16 works for training but
not for quantized inference.

### 4. Input layout

- **ROW_MAJOR**: Data stored row-by-row. The kernel must tilize (convert to tiles)
  before compute and untilize after.
- **TILE_LAYOUT**: Data already in 32x32 tiles. Skips tilize/untilize overhead.

The prompt explicitly asks for ROW_MAJOR, but a general op should accept either.

## The evaluation gap

Today's scoring is one-dimensional: golden pass rate (70/70 = good, 50/70 = bad).
All tests exercise the same memory/alignment/dtype configuration, so the score
only tells you how well the agent handled shapes and parameter variations within
that single configuration.

The agent builds exactly what the prompt asks for — nothing more. If the prompt
asks for tile-aligned DRAM interleaved bfloat16, that is what gets built. The
agent will not spontaneously add sharding support or non-tile-aligned handling.

This means that to evaluate generality, **the prompt itself must ask for it**, and
the golden tests must cover the configurations the prompt specifies. The current
setup has one prompt at one difficulty level with one set of golden tests. There
is no way to measure how the agent performs when asked for progressively harder
configurations.

## What "broader" evaluation means

We want to measure agent capability across different difficulty levels of the
*same* operation. This requires:

1. **Multiple prompts at different difficulty levels.** Each prompt explicitly
   asks for a specific set of configurations. Easy prompt: DRAM interleaved,
   tile-aligned. Medium prompt: adds L1 interleaved and non-tile-aligned. Hard
   prompt: adds sharding. Each gets its own golden tests that match what was
   asked for.

2. **Structured difficulty levels.** Not all configurations are equally hard to
   implement. L1 interleaved is a small step from DRAM interleaved; block
   sharding is a large one. The prompt/test structure must reflect this.

3. **Failure categorization per difficulty level.** When the agent fails at a
   harder prompt, we want to know *why* — was it a hang (kernel deadlock on
   sharded data), OOM (circular buffers too small for L1), numerical (padding
   not handled), or a signature error (the Python entry point doesn't accept a
   memory_config arg)? This tells us what the agent struggles with at each level.

4. **Cross-run comparison at the same difficulty level.** "Run 3 of the medium
   prompt passed 80% vs Run 2's 60%" — and separately — "the agent has never
   passed the hard prompt's sharding tests."

## Execution model: single-shot vs iterative

### Current model: single-shot

Today's eval is fire-and-forget. The prompt goes in, the agent runs once
(`--max-turns 150`), and whatever it produces gets tested. If the op is broken,
we record the failure and that's it. There is no second chance.

This is a poor match for how real development works. A human writing a TTNN op
iterates: write something, test it, see failures, fix them, test again. The
current eval denies the agent this loop — it must get everything right in one
shot, or the run is scored as-is.

### Proposed model: iterative reprompting

A better model is: start with the simplest possible prompt, let the agent build
it, run the golden tests, and if tests fail, **feed the failures back to the
agent and let it fix them**. Repeat until either everything passes or a budget
is exhausted.

This changes the eval from "can the agent produce a perfect op in one shot?" to
"can the agent converge to a working op through iteration?" — which is closer to
what actually matters.

The iteration loop would look like:

```
1. Run initial prompt → agent produces op
2. Run golden tests → get test_results.json
3. If all pass → done (record total iterations, time, etc.)
4. If failures → construct a follow-up prompt with:
   - Which tests failed
   - Failure categories (hang, numerical, OOM, compilation, etc.)
   - The actual error messages (truncated)
5. Feed follow-up prompt to the agent in the same clone
6. Agent fixes the op
7. Go to step 2
8. If iteration budget exhausted → record partial results
```

This raises several design questions:

- **Iteration budget**: How many reprompt cycles? Too few and the agent can't
  recover from hard bugs. Too many and we're measuring persistence, not capability.
- **What goes in the follow-up prompt?** Raw pytest output? Classified failure
  summaries? Just test names and categories? The amount of context matters —
  too little and the agent can't diagnose; too much and it drowns.
- **Same Claude session or fresh?** If the agent keeps its conversation context
  across iterations, it remembers what it tried. If each reprompt is a fresh
  `claude -p`, it starts from scratch every time (but the code on disk persists).
  Same session is more efficient; fresh session tests whether the code is
  self-explanatory enough to debug from cold.
- **How does this interact with difficulty levels?** Does the iteration loop
  run only within a single difficulty level? Or does the system automatically
  escalate: "level 0 passes, now reprompt with level 1 requirements"?
- **Scoring**: A run that passes on iteration 1 is better than one that passes
  on iteration 5. The score should capture both the final pass rate AND the
  number of iterations it took.

### Implementation in the current infrastructure

The current `run_eval.sh` runs claude once per prompt. Iterative reprompting
would require:

1. A loop in `run_eval.sh` (or a new script) that alternates between
   `claude -p` and `eval_test_runner.sh`
2. A follow-up prompt generator that reads `test_results.json` and constructs
   a reprompt
3. DB schema changes to track iteration count and per-iteration results
4. Dashboard changes to show convergence over iterations

The `quick_ingest.py` path could also support this — run tests, see failures,
manually reprompt, run `quick_ingest` again to record the new state.

## Golden test visibility: hidden vs known

Should the agent know the golden tests exist?

### Option 1: Hidden tests (agent cannot see them)

The golden tests live in `eval/golden_tests/` and the agent is never told about
them. The agent builds the op based solely on the prompt, using its own TDD
stages to validate correctness. After the agent finishes, the eval harness runs
the golden tests as an independent check.

**Pros:**
- Tests whether the agent can build a correct op from a specification alone
- No risk of the agent overfitting to specific test cases
- Golden tests are a genuine independent validation

**Cons:**
- The agent has no signal about what "correct" means beyond its own TDD tests
- If the golden tests have specific expectations (e.g. exact error types for
  validation tests), the agent may produce something functionally correct but
  incompatible with the test expectations
- The API contract (`api_contract.md`) already partially bridges this gap, but
  only covers the interface, not behavioral expectations

### Option 2: Visible tests (agent can see and run them)

The agent is told: "there are golden tests at `eval/golden_tests/layer_norm_rm/`.
Your op must pass them." The agent can read and run these tests during development.

**Pros:**
- Clear, unambiguous success criteria
- Agent can iterate against real tests, not just its own TDD stages
- Eliminates mismatches between what the agent builds and what the tests expect

**Cons:**
- Risk of the agent "teaching to the test" — making the tests pass without
  genuine understanding of the operation
- The agent might hardcode workarounds for specific test shapes rather than
  building a general implementation
- Less useful as an evaluation of the agent's ability to work from specs

### Option 3: Hybrid — visible contract, hidden tests

The agent sees the API contract (`api_contract.md`) and a small set of example
tests, but the full golden test suite is hidden. The examples show the expected
interface and a few representative cases. The hidden tests cover edge cases,
stress shapes, and failure modes.

**Pros:**
- Agent knows the interface expectations (avoids dumb contract mismatches)
- Agent still has to build a genuinely correct implementation
- Edge cases remain a real test of generality

**Cons:**
- More complex to maintain (contract + examples + hidden tests)
- The boundary between "shown" and "hidden" is arbitrary

### How this interacts with reprompting

If tests are hidden: the reprompt contains failure information but the agent
can't look at the test source to understand why it failed. This tests debugging
from error messages alone.

If tests are visible: the reprompt can say "test X failed" and the agent can
read the test to understand the expectation. This is more realistic (developers
always read failing tests) but easier.

If tests are hybrid: reprompt reveals which hidden tests failed and their error
output, but the agent can only read the example tests for reference. This is
a middle ground.

## Open questions

### What exactly should the difficulty levels be?

One possible organization (each level corresponds to a prompt + its golden tests):

| Level | Name | What the prompt asks for |
|-------|------|--------------------------|
| 0 | Core | Tile-aligned, DRAM interleaved, bf16, RM (current) |
| 1 | Memory | Level 0 + L1 interleaved support |
| 2 | Alignment | Level 1 + non-tile-aligned shapes |
| 3 | Sharding | Level 2 + height/width/block sharded memory |
| 4 | Dtype | Level 3 + float32 and quantized formats |

But there are tradeoffs:
- Should levels be cumulative (each includes all previous) or independent axes?
- Is "L1 interleaved" really easier than "non-tile-aligned"? It depends on the op.
- Should sharding subtypes (height/width/block) be one level or three?
- Cumulative levels are easier to score but constrain the evaluation to a single
  difficulty ladder. Independent axes allow finer-grained measurement but make
  prompt and test management more complex.

### How should the op interface change across difficulty levels?

The current (level 0) prompt produces an op with this signature:
```python
def layer_norm_rm(input_tensor, gamma=None, beta=None, *, epsilon=1e-5)
```

A harder prompt that asks for memory config support would need the agent to
produce something like:
```python
def layer_norm_rm(input_tensor, gamma=None, beta=None, *, epsilon=1e-5,
                  memory_config=None, output_memory_config=None)
```

The prompt must explicitly ask for the extended interface, and the golden tests
for that prompt must use it. This means each difficulty level has its own API
contract — the golden tests cannot test an interface the prompt didn't ask for.

### How should prompts be structured?

The agent builds what the prompt asks for. So the prompt defines what gets tested.

Option A: **One prompt per difficulty level.** `layer_norm_rm_easy.txt` asks for
DRAM interleaved + tile-aligned. `layer_norm_rm_medium.txt` adds L1 interleaved
and non-tile-aligned shapes. `layer_norm_rm_hard.txt` adds sharding. Each has
its own golden tests that match the prompt's requirements. This cleanly measures
agent capability at each level.

Option B: **One prompt with explicit tiers.** A single prompt lists requirements
in priority order: "Must support DRAM interleaved. Should support L1 interleaved.
Stretch: support height sharding." Golden tests cover all mentioned configs. This
tests whether the agent can handle prioritized requirements in a single run.

Option C: **Parametric prompts.** A template prompt with slots for memory config,
alignment, dtype. The eval harness generates concrete prompts from the template.
This is the most systematic but adds complexity to the prompt management.

Each approach has different implications for how we organize golden tests, how we
track results in the DB, and how we compare across runs.

### How do we write sharding tests without being op-specific?

Sharding tests need shard specs (core grid, shard shape) that depend on:
- The tensor shape
- The device's core grid (8x8 for Wormhole, 8x10 for Blackhole)
- The memory layout

Writing a generic shard spec calculator is non-trivial. Do we:
- Hardcode shard specs per test case? (Brittle, lots of manual work)
- Build a helper that computes valid shard specs for a given shape? (Reusable but complex)
- Test only shapes where sharding is "obvious" (e.g. H=256 across 8 cores = 32 rows/core)?

### Tolerance and scoring

Should higher tiers have relaxed tolerances? Non-tile-aligned shapes might have
slightly worse numerical accuracy due to padding. float32 should arguably have
*tighter* tolerances.

For scoring: a simple weighted sum (tier 0 = weight 1.0, tier 1 = 0.5, etc.)
gives a single generality number. But the weights are arbitrary. Is "sharding
support" really worth 30% of the score? That depends on the deployment context.

### What about performance?

Generality isn't just correctness. An op might produce correct results with L1
interleaved memory but be 10x slower than DRAM interleaved because the kernel
wasn't designed for it. Should we measure throughput per tier? That's a much
larger instrumentation effort.

## Current infrastructure constraints

Things to keep in mind when designing the solution:

1. **SQLite schema**: The eval DB has a flat `test_results` table. Adding tier
   tracking requires schema changes. The `runs` table would need per-tier
   aggregations.

2. **JUnit XML pipeline**: `eval_test_runner.sh` → `classify_failures.py` → `ingest.py`.
   Tier information needs to flow through this pipeline. The easiest way is to
   embed it in the test file name or pytest markers.

3. **Device serialization**: `eval_test_runner.sh` uses `flock` for device access.
   Sharding tests may need different device configs. Tests that corrupt device
   state (hangs from bad shard specs) need the existing dirty-flag mechanism.

4. **Golden test isolation**: Currently all golden tests for an op run in one
   pytest invocation. If tier 2 tests hang, they block tier 3+4. Should tiers
   run as separate pytest invocations? That's safer but slower.

5. **Op-specific vs generic**: The tier test *infrastructure* should be generic
   (any op can define tiers). The actual tier *tests* are necessarily op-specific
   (layer norm's sharding behavior differs from matmul's). How much can we
   template?

## Next steps

This document captures the problem space. Before writing code, we should decide:

1. **Execution model**: Single-shot or iterative reprompting? If iterative, what
   is the iteration budget and what goes in the follow-up prompt?
2. **Test visibility**: Hidden, visible, or hybrid? How does this interact with
   the chosen execution model?
3. **Difficulty structure**: Cumulative levels vs independent axes
4. **Prompt organization**: One per level, one with tiers, or parametric templates
5. **Golden test mapping**: Each prompt gets its own test set, or shared tests
   with markers
6. **Sharding test complexity**: Hardcoded specs, computed specs, or simple
   shapes only
7. **Scoring model**: Per-level pass rate, weighted aggregate, iteration count,
   or something else
8. **Performance**: In scope or only correctness
