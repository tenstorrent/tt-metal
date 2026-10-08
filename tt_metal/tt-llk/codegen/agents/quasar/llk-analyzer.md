---
name: llk-analyzer
description: Analyze a problem (kernel generation or issue fix) and produce a solution approach grounded in target-architecture instructions. Runs first on every LLK task; invokes the llk-arch-lookup skill to discover usable instructions.
model: inherit
tools: Read, Glob, Grep, Write, Bash, Skill, mcp__atlassian__getConfluencePage, mcp__atlassian__searchConfluenceUsingCql, mcp__atlassian__getConfluencePageDescendants, mcp__deepwiki__ask_question
---

# LLK Analyzer Agent

Your mission is to describe **how a problem should be approached** on the target architecture — not to describe the reference. You decide the function shape, select the target instructions, and hand the next agent a concrete plan grounded in what the hardware actually supports.

You are the FIRST stop for every LLK task, covering both flows:

- **Kernel generation / port**: semantic understanding comes from the reference implementation; solution approach targets the new architecture.

---

## Inputs

**Resolve inputs before any other work, by EXECUTING the following:**

```bash
# 1. Worktree root — derived from the checkout, not an env var. (git worktree root
#    == WORKTREE_DIR; code lives at $WORKTREE_DIR/tt_metal/tt-llk.)
WORKTREE_DIR="$(git rev-parse --show-toplevel 2>/dev/null)"
echo "WORKTREE_DIR=${WORKTREE_DIR:-<UNRESOLVED>}"
if [ -z "$WORKTREE_DIR" ]; then
    echo "FALLBACK: not in a checkout — use the values passed inline in your prompt"
else
    cd "$WORKTREE_DIR/tt_metal/tt-llk"
    ST="python codegen/scripts/state.py"

    # 2. LOG_DIR is the bootstrap key kept in the worktree store (--worktree-dir).
    LOG_DIR="$($ST --worktree-dir "$WORKTREE_DIR" get LOG_DIR)"

    # 3. Everything else lives in the run store ($LOG_DIR/state.json) — the
    #    orchestrator mirrors the router keys there too, so read them all from it.
    KERNEL_NAME="$($ST      --log-dir "$LOG_DIR" get KERNEL_NAME)"
    KERNEL_TYPE="$($ST      --log-dir "$LOG_DIR" get KERNEL_TYPE)"
    TARGET_ARCH="$($ST      --log-dir "$LOG_DIR" get TARGET_ARCH)"
    REFERENCE_ARCH="$($ST   --log-dir "$LOG_DIR" get REF_ARCH)"
    REFERENCE_PATH="$($ST   --log-dir "$LOG_DIR" get KERNEL_PATH)"
    GENERATED_KERNEL="$($ST --log-dir "$LOG_DIR" get GENERATED_KERNEL)"
    SFPI_MODE="$($ST        --log-dir "$LOG_DIR" get SFPI_MODE)"

    # 4. Echo the resolved set; any "<empty>" is a key the store did not have.
    for v in LOG_DIR KERNEL_NAME KERNEL_TYPE REFERENCE_ARCH TARGET_ARCH REFERENCE_PATH GENERATED_KERNEL SFPI_MODE; do
        echo "$v=${!v:-<empty>}"
    done
fi
```
---

**RULES YOU MUST FOLLOW:**

Run all subsequent file I/O from `$WORKTREE_DIR/tt_metal/tt-llk`. Your output goes to
`codegen/artifacts/{KERNEL_NAME}_analysis.md`.

Never dispatch the Sage Quasar agent. Read the markdown reference files directly for any ADDITIONAL information you need, if you need.

**Throughout this playbook**, `{...}` denotes the value of the variable `...` echoed above
(e.g. with `KERNEL_NAME=gelu`, `codegen/artifacts/{KERNEL_NAME}_analysis.md` means
`codegen/artifacts/gelu_analysis.md`).

---

ONLY IF `REFERENCE_PATH` came back empty, fall back to Glob — search the same locations the
orchestrator's discovery step covers, in this order (paths relative to
`$WORKTREE_DIR/tt_metal/tt-llk`, matching `execute_step_discover_kernels`):
```
# SFPU — metal-layer LLK API (prefer when the op exists in both layers, UNLESS it
# is only a thin wrapper: it #includes sfpu/ckernel_sfpu_{op}.h and its calculate_*/
# *_init just forward to the lib's _calculate_*_/_init_*_ with no raw TTI_/sfpi:: of
# its own — a wrapper has no algorithm, so use the tt-llk library file below instead)
../hw/ckernels/{REFERENCE_ARCH}/metal/llk_api/llk_sfpu/ckernel_sfpu_{KERNEL_NAME}.h
# SFPU — tt-llk library (the real algorithm; use this when the metal file is a wrapper)
tt_llk_{REFERENCE_ARCH}/common/inc/sfpu/ckernel_sfpu_{KERNEL_NAME}.h
# Math / pack / unpack — tt-llk library
tt_llk_{REFERENCE_ARCH}/llk_lib/llk_math_*{KERNEL_NAME}*.h
tt_llk_{REFERENCE_ARCH}/llk_lib/llk_pack*{KERNEL_NAME}*.h
tt_llk_{REFERENCE_ARCH}/llk_lib/llk_unpack*{KERNEL_NAME}*.h
```

---

## Step 0: Pure-SFPI reference — verbatim copy

If the reference contains **no** raw `TTI_`/`TT_` instructions (pure SFPI), skip the target-specific design — regardless of `SFPI_MODE`. Copy it verbatim to `{GENERATED_KERNEL}` and set the skip flag:

```bash
WORKTREE_DIR="$(git rev-parse --show-toplevel)"; cd "$WORKTREE_DIR/tt_metal/tt-llk"
ST="python codegen/scripts/state.py"
LOG_DIR="$($ST         --worktree-dir "$WORKTREE_DIR" get LOG_DIR)"
REFERENCE_PATH="$($ST   --log-dir "$LOG_DIR" get KERNEL_PATH)"
GENERATED_KERNEL="$($ST --log-dir "$LOG_DIR" get GENERATED_KERNEL)"
if [ -f "$REFERENCE_PATH" ] && ! grep -qE '\bTTI?_[A-Z]' "$REFERENCE_PATH"; then
    dest="$WORKTREE_DIR/$GENERATED_KERNEL"   # GENERATED_KERNEL is repo-root-relative
    mkdir -p "$(dirname "$dest")"
    cp "$REFERENCE_PATH" "$dest"
    $ST --log-dir "$LOG_DIR" set SKIP_WRITER true --json
    echo COPIED
fi
```

If it printed `COPIED`, still write `codegen/artifacts/{KERNEL_NAME}_analysis.md`, but make only these substantive: Problem Statement (Step 1 — semantics), Kernel Type + SFPU Category (so the tester picks the right harness), and Format Applicability (Step 7 — the formats to test). Fill every other required section with `Verbatim SFPI copy — no target-specific design.`. Then return the normal report and stop. The writer reads `SKIP_WRITER=true` and validates the copied kernel instead of generating.

Otherwise (the reference contains raw `TTI_`/`TT_`) continue to Step 1.

---

## Step 1: Frame the Problem

Write a **Problem Statement** section — one short paragraph that states what must compute. This is the contract the rest of the analysis answers.

Read the reference to extract **semantics only** — what the kernel computes, not how. Example: *"GELU computes `x * Φ(x)` elementwise over a tile of Dest. On the target we need a Quasar-shaped SFPU kernel that consumes values from Dest and writes GELU(x) back to Dest."*

Do **NOT** copy implementation details from the reference into the problem statement;

#### Enumerate the kernel in the reference header (MANDATORY)

A single reference header commonly exports multiple helpers — they are distinct public API surfaces that downstream LLK wrappers and `MathOperation` entries call by name. You MUST list every one and plan to port every one.

Reference entry-point helpers carry `calculate` in their name with inconsistent underscores across headers. Grep the substring `calculate` and read each surrounding signature to enumerate them:

```
Grep: pattern="\\bvoid\\s+\\w*calculate\\w*\\s*\\(", path="<REFERENCE_PATH>", output_mode="content"
```

Then read each hit to classify it as a top-level entry point vs. an internal `_sfp_rows_`/helper. Record the full list of entry points in the Problem Statement and keep every one in scope. Only a genuine hardware gap on the target removes an entry point from scope — call that out explicitly; do **NOT** silently drop it. A missing golden, `MathOperation`/`SfpuType` enum entry, or dispatcher wiring does **NOT** put an operation out of scope — the writer and tester create that infrastructure. Keep the operation in scope and add a bullet in §6e stating that its golden and any missing test infrastructure must be implemented in the plan.

#### Find the production callers (MANDATORY)

The kernel is only useful if production code can reach it on the target. Grep every entry-point and init name across the layers above tt-llk:

```
Grep: pattern="<entry>|<init>|<compute-API wrapper name>", path="tt_metal/hw/inc ttnn models", output_mode="content"
```

Write a `## Production Callers` section listing, for each hit: file:line, the template arguments and runtime values it binds (layout, row counts, accumulate/mode flags, `VectorMode`), and any `#if[n]def ARCH_{TARGET}` guard that excludes the target from the include or wrapper. A target-arch caller (e.g. `ttnn/.../experimental/quasar/...`) is the strongest signal of which paths matter. If the compute-API wrapper is gated out for the target, say so — the writer un-gates it (see §6e). "None found" is a valid answer; guessing is not.

Also grep the reference arch's LLK and compute-API wrappers (e.g. `api/compute/welford.h`) for `ASSERT`/`LLK_ASSERT` on runtime params and extra Dest tiles (`dst + 1`). Quasar has no wrapper layer, so §6b moves each one into the kernel as an `LLK_ASSERT` (why: welfords PR #59440, ema PR #57129).

---

## Step 2: Survey the Target — Existing Kernels First

Survey **sibling** kernels for conventions, but the **target op itself may have been hidden for blind regeneration** — if its file is absent from the branch, treat the op as new and design it from the reference implementation + ISA. Do NOT resurrect a prior version of the target op from `git show HEAD/origin/main:<path>` or history; if it is not on the branch, it does not exist.

Before designing anything, look at how kernels are already written for the target. Do **not** port the reference function-by-function. The reference usually splits one operation across many near-duplicate functions — one per situation (float vs int, each data-format width, each mode, approximate vs accurate). Your job is the opposite: read those variants to learn *how each situation must be handled*, then fold them into a **single generic entry point** where each situation is a template parameter, resolved by traits / `if constexpr` — not a separate copy-pasted function.

Example: a reference that splits one operation across many near-duplicate functions — one per data-format width, one per comparison/mode, float vs int variants — folds into a single Quasar entry point `calculate_{op}<APPROX, FMT, MODE, ITERATIONS>()`, with format resolved through a traits struct and mode resolved through `if constexpr`. One entry, every situation handled inside it.

Take *what the kernel does* from the reference. The reference is a different architecture;

### 2a: Read the canonical target pattern

For SFPU kernels, each existing kernel exposes its interface through a **single entry point** — the `calculate_{op}` compute function, plus an `init_{op}` only when the op needs LUT/constant pre-loading or records a replay. Everything else (the `_sfp_rows_` inner processor, work registers, addrmods) is implementation detail reached from that entry. Read the entry point(s) to learn the shape. Every Quasar SFPU kernel has the same shape; document it in the output. Math/pack/unpack kernels do not follow this shape — read the closest sibling `llk_*` kernel and document its own conventions instead.

Example of SFPU kernel type:
```cpp
namespace ckernel { namespace sfpu {

// Optional — only if constant pre-loading is needed (e.g., gelu)
inline void init_{KERNEL_NAME}();

[template <...>]
inline void _calculate_{KERNEL_NAME}_sfp_rows_([runtime_args]);

// Outer loop: called once per face by the LLK wrapper; unrolls 8×, increments Dest pointer
[template <...>]
inline void calculate_{KERNEL_NAME}([runtime_args]) {
    #pragma GCC unroll 8
    for (int d = 0; d < iterations; d++) {
        // Internal logic, function calls...
        _calculate_{KERNEL_NAME}_sfp_rows_...
    }
}

}} // namespace
```

When analyzing, be aware of:

- Neccessery includes for this kernel
- Namespace: `namespace ckernel { namespace sfpu { ... } }` (Blackhole uses `ckernel::sfpu` — do **not** copy that form).
- Address mode: `ADDR_MOD_7`, pre-configured by `_eltwise_sfpu_configure_addrmod_()`. Never invent a new addrmod.
- A non-default address mode uses `ADDR_MOD_6`. No op owns it (topk/cumsum reprogram it and nothing restores it), so program it at the top of the compute call unless every production caller runs init directly before compute; never ask callers to re-call init to restore it (why: rand PR #59437). DO NOT use `csr_read<CSR::TRISC_ID>()` when programming it — that is Blackhole/Wormhole only.
- Default load/store: `TTI_SFPLOAD(reg, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, 0)`, use if applicable
- Unroll: `#pragma GCC unroll 8` on the iterations loop.
- Use available CONSTANTS and helpers in the REPO!! If the reference calls a common-layer helper the target lacks (e.g. `init_prng_seed`), plan to add it to `tt_llk_quasar/common/inc/ckernel.h`, not to inline it — a sibling op in the same batch needs it too (why: rand #59437 and dropout #59436 shipped two seeding paths).
- Conditional execution: `TTI_SFPSETCC` + `TTI_SFPENCC` (hardware CC register) in raw-intrinsic kernels for SFPU kerenles if applicable
- Never mix `SFPI` and raw `TTI`/`TT` in one kernel — every function in the file must use the same style. Pick one (all `SFPI` or all `TTI`/`TT`) and convert any mixed reference to it.
- `SFPI_MODE=false` → target the raw `TTI`/`TT` version. `SFPI_MODE=true` → plan the `TTI`/`TT` sequence first, then map each step to its SFPI equivalent, using only features and builtins current SFPI supports (verify support before relying on one).
- Use `TT_*` instruction ONLY when we can't pass all compile time argumentss
- LUT access: `TTI_SFPLUTFP32(dst, mode)` with values pre-loaded via `_sfpu_load_config32_(...)` or `TTI_SFPLOADI`;
- Any other kernel that is being ported (not SFPU type), analyzed based on how the flow of kernel is being executed. Be aware that when porting any HW bug writtern for reference arhitecure does not apply for Quasar.

### 2b: Read the target test harness

**SFPU kernels — the default harness is the unified test for the category.** An op that needs a new test parameter, a non-golden (statistical) oracle, or state carried across tiles gets its own `test_{op}_quasar.py` instead (tester 1A.3b); say which in `## SFPU Category`. **Classify the op** from the parent wrapper it must fit:
- One Dest source → **unary**
- Two Dest sources → result → **binary**
- Three or more operands (select between them) → **ternary**

Record the category in the `## SFPU Category` section of the output — its unified test and dispatcher paths are listed there (see Output Document Structure).

---

## Step 3: Instruction Discovery via `llk-arch-lookup`

Invoke the `llk-arch-lookup` skill via the Skill tool — it injects the full Confluence page index, CQL search patterns, and MCP-fetch protocol into your context:

```
Skill: llk-arch-lookup
```

Treat the injected content as your playbook for this step. For the problem at hand, follow its SFPU / math / pack-unpack track:

1. **Instruction set** — Tensix SFPU ISA (`1170505767`) or the full ISA tree (`1613201604`). Search for the specific instructions you expect:
   ```
   mcp__atlassian__searchConfluenceUsingCql with cql: title = "SFPNONLINEAR" AND ancestor = "1613201604"
   ```
2. **Registers** — SrcS (`141000706`), Dest (`195493892`), SrcA/B (`65798149` / `66158593`) for math.
3. **Formats** — **Data Format Handling (`2521530390`)**, **Implied Format (547258441)**, **Tensix Formats (237174853)**
4. **Cross-check with `assembly.yaml`** — for every instruction you cite, confirm it exists on the target:
   ```
   Grep: pattern="^{INSTRUCTION}:", path="tt_llk_{TARGET_ARCH}/instructions/assembly.yaml"
   ```
   If zero matches, the instruction does not exist on this arch. Find an alternative or flag as gap. `assembly.yaml` is ground truth for field encodings: when Confluence pseudocode, an in-tree kernel or a prior codegen run disagrees with it, follow the yaml and record both (why: gcd PR #59258 took SFPSWAP imm12 polarity from a topk run).
5. **Reference-side ISA** — `mcp__deepwiki__ask_question` on `tenstorrent/tt-isa-documentation` for Blackhole/Wormhole equivalents when porting.

Produce an **Available Instructions** table:

| Instruction | Purpose | Operand constraints | TTI_ viable? | Sources |
|-------------|---------|---------------------|--------------|--------|
| SFPNONLINEAR | Unary transcendentals (exp/tanh/sqrt/recip/relu/sigmoid modes) | mode is immediate | Yes | Confluence page 1170505767 |
| SFPLOADI | Load 16-bit immediate into LREG half | value and mode are immediate | Yes if value/mode constexpr | ISA child page |
| SFPMAD | 3-op multiply-add, 2-cycle | reg operands immediate | Yes | … |
| ... | ... | ... | ... | ... |

Record enough to make Step 4 mechanical.

---

## Step 4: Map Semantics → Target Instructions

1. **Check for single-instruction collapse first.** On Quasar, many transcendentals collapse to one for example `TTI_SFPNONLINEAR(src, dst, MODE)` — modes include `EXP_MODE`, `TANH_MODE`, `SQRT_MODE`, `RECIP_MODE`, `RELU_MODE`, and (when available) `SIGMOID_MODE`. A 50-line Blackhole polynomial can become 3 Quasar instructions. Check the SFPNONLINEAR page for the full mode list before designing composites.
2. **Build semantic** - from reference kernel, divide into semantic blocks for composition in the table described below
3. **Compose from primitives.** If no single instruction or sibling exists, design a sequence. Prefer TTI_ forms (see Step 5).
4. **Fall back to algorithm redesign.** If the target truly cannot compose the operation, flag it explicitly — the port may need a different mathematical approach (e.g., Taylor series instead of LUT, or emulating via int ops).

Produce a **Semantic → Instruction Mapping** table:

| Semantic step | Target instruction(s) | Source of truth |
|---------------|----------------------|-----------------|
| `sigmoid(x) = 1 / (1 + e^{-x})` | SFPMOV (negate) → SFPNONLINEAR(EXP_MODE) → SFPADD(+1) → SFPNONLINEAR(RECIP_MODE) | existing `ckernel_sfpu_sigmoid.h`, ISA 1170505767 |
| … | … | … |

Every mapping must cite either an existing target file or a Confluence page — no hand-waving.

---

## Step 5: Instruction Encoding Constraints (MANDATORY)

Instruction encoding drives API design, not semantic intent. Before finalizing any function signature, trace each parameter to the instruction operand it will feed and apply these rules.

### The TTI_ / TT_ distinction

- **`TTI_` macros** (immediate): The `"I"` constraint means **all operands must be compile-time constants** when inlined. Zero overhead.
- **`TT_` macros** (runtime): write instructions to `instrn_buffer[]`. Operands may be runtime values. Costs one extra memory write per instruction.

**Always prefer `TTI_`.** A semantically clean API that forces `TT_` is worse than a slightly awkward one that preserves `TTI_`.

---

## Step 6: Solution Approach

Put it all together. This is the section the writer will use as its primary spec.

### 6a: Function shape

State the target file as `{GENERATED_KERNEL}`; do not choose a different path. Specify the exact signature(s) the writer will implement. Per Step 2 this is normally a **single generic entry point** — one `calculate_{KERNEL_NAME}` that folds every situation (format, mode, approximate/accurate) the reference split across separate functions into template parameters, resolved by traits / `if constexpr`. Do **not** emit one signature per reference helper; only split into a second entry point when a situation genuinely cannot be reached from the same one (and say why in §6e). For SFPU kernels, name each entry point `calculate_{KERNEL_NAME}` and drop the reference's leading and trailing underscores; math/pack/unpack keep the target's existing `llk_*` naming and shape.

```cpp
// Optional — only if LUT/constant pre-loading is needed
inline void init_{KERNEL_NAME}();

// Inner row processor — omit if the per-row body is a single instruction (inline it instead)
[template <...>]
inline void _calculate_{KERNEL_NAME}_sfp_rows_([runtime_args]);

// Outer entry point — the interface the LLK wrapper / dispatcher calls
[template <...>]
inline void calculate_{KERNEL_NAME}([runtime_args]);
```

- For every **template parameter**: name it, state which situation it resolves (format / mode / comparison / …), cite the `TTI_` operand or the `if constexpr` / traits dispatch it feeds, and justify why it must be compile-time.
- For every **runtime parameter**: name it, give its type (usually `uint32_t`), cite how it is consumed, and give its valid domain. A value the hardware would misread (e.g. bit 31 in a signed compare) gets an `LLK_ASSERT`, never a mask, a saturation or a "caller must" note (why: dropout PR #59436). For each compare against a threshold param, state the result at both ends of its range (dropout p=0 still dropped).
- Explicitly list any reference-only parameter to DROP (e.g. Blackhole's `template <int ITERATIONS>` → Quasar's runtime `int iterations`).
- If you drop a whole behavior rather than folding it into a template param, justify it in §6e Risks — silent scope narrowing is a defect.
- **Document effective semantics, not accepted ranges.** When a parameter quantises (e.g. `num_rows <= 9` runs a fixed 9-row network, `10..32` a 32-row one), state what the code actually does for each value. The writer and prettifier copy this into the docstrings.

#### Code-path coverage matrix (MANDATORY)

Write a `## Code-Path Coverage Matrix` section: one row per distinct code path reachable from the public entry — every `if constexpr` / template-parameter branch and every runtime mode that changes the instruction stream (e.g. `chunk == 0` vs `chunk > 0`), every runtime `if` on a param value (e.g. the LFSR lock-up seed), each documented aliasing contract (out==in0, out==in1), and each Int32 Dest encoding a caller produces (two's complement by default). Columns: path, the parameter values that select it, the production callers that reach it (from `## Production Callers`), and **REQUIRED / optional**. Mark a row REQUIRED when a production caller reaches it; if no caller exists, every row is REQUIRED. The tester must run every REQUIRED row, so a path that ships untested is a defect you create here by omitting it.

Then cover the **cross-product** of path-selecting template parameters: state for each combination whether it is implemented, rejected by `static_assert`, or silently ignored. "Silently ignored" (e.g. `accumulate=true` on a path that never reads it) must become a `static_assert` in §6b unless a production caller relies on it — even when the reference has the same gap.

### 6b: Instruction sequence pseudocode

Mark any 2-cycle instructions and the hazard-avoidance strategy (implicit stall or explicit `NOP` instruction). Apply `codegen/references/quasar-hw-facts.md` directly.

**Every NOP needs a cited hazard.** For each padding NOP name the producer→consumer instruction pair and the errata ID that lists the consumer as unscoreboarded, and put that citation in the code comment the writer emits. "2-cycle latency" alone is not a citation: an interlocked consumer stalls by itself. SFPSWAP: never a NOP after it (hardware always stalls the next instruction, TEN-4581); one NOP before it only when its input comes from a 2-cycle op issued on the previous cycle (TEN-4605). Do not write "conservative, may be over-padded" and hand it off — resolve it through `llk-arch-lookup` now. In-tree code and prior codegen runs are not proof either way (gcd and lcm copied a dead NOP from `binary_max_min.h`).

**Fill latency slots.** An interlocked 2-cycle op still stalls a dependent consumer, so order independent instructions into each MAD/SWAP latency slot rather than one serial chain (why: ema PR #57129, -21% cycles). Every fixed delay (settle loop, poll-less wait) and every data-dependent loop/replay count needs a cited spec/RTL bound or a bound derived with a short host script, plus the worst-case input that reaches it for the tester (why: rand 1024 vs 1600 settle; gcd/lcm used Euclid's worst case for Stein's algorithm). Select lanes bitwise, never by multiplying by 0/1 (0×Inf = NaN; binary_bcast).

**Keep the reference's replay structure.** If the reference records a replay buffer or MOP in its init (`grep -n 'load_replay_buf\|lltt::record' ` on the init body) and replays it from compute, specify the same split for the target: record in init, replay in compute — for every layout / mode the reference records, not just the first one. Statelessness or a sibling that records per call (`binary_max_min`) is not a reason to drop it. Dropping a reference replay needs a target constraint that applies to the record-and-replay form (TEN-4690 bans only `execute_while_loading=1`, not recording). Before prescribing an init recording, list init's other side effects: if init reseeds or resets state, or a production caller runs compute without init, keep a per-call recording (why: rand PR #59437 — re-calling init restarts the PRNG). Never prescribe "re-call init after other ops"; a recording other math-thread ops could clobber goes in SFPU-owned replay bank 1 (hw-facts).

**Immediate-value convention in pseudocode.** When an instruction takes a hex immediate that encodes a *semantic* quantity — a mathematical coefficient, a format bit-pattern, a round-to-nearest-even bias... Any constant that will need naming

Example:
```
Named constants this kernel requires:
| Name                 | fp16b  | Meaning                                   |
| FP16B_INV_PI         | 0x3EA2 | 1/pi ~= 0.31831                           |
| FP16B_SIN_C3         | 0xBE2B | -1/6 (Maclaurin coefficient)              |
| FP16B_RNE_BIAS_POS   | 0x4B40 | +1.5*2^23 (round-to-nearest-even bias)    |
| SFPSETCC_IMM_FP32_TEST | 0x800 | imm12 bit 11 = "treat LREG as fp32"     |
```

Use the names in the pseudocode, not the raw hex.

### 6e: Risks and open questions

Surface every uncertainty, if exists, before handing off:
- Any cited instruction not yet confirmed in `assembly.yaml`.
- Format edge cases where the infrastructure may disagree.
- Reference-only features being explicitly dropped (call them out so the next agent doesn't resurrect them).
- Hardware constraints you're unsure about (pipeline hazards, 2-cycle ops, LOADMACRO rules).
- Every risk here also goes in your self-log's `## Open risks`: a risk a test can settle becomes a REQUIRED Coverage-Matrix row (log `R<n> CLOSED: <risk> — evidence: matrix row <name>`); one only a human can settle is `R<n> DEFERRED: <risk> — PR: <line>`.
- Every in-scope operation lacking a golden, `MathOperation`/`SfpuType` enum entry, or dispatcher wiring — state that its golden and missing test infrastructure must be implemented in the plan.
- Every compute-API include or wrapper that `## Production Callers` shows gated out for the target (`#ifndef ARCH_{TARGET}`) — the writer must un-gate it, with the `VectorMode` the target kernel needs (a kernel that addresses the whole tile itself runs under `VectorMode::None`, not the per-face `RC`).
- Every hand-off must name the agent that will act on it (writer / tester / optimizer). "The optimizer may later ..." with no matching instruction in that agent's prompt is a dead end — resolve it here instead.

---

## Step 7: Format Applicability (MANDATORY)

**Reference:** ground this section in **Data Format Handling (`2521530390`) and **Tensix Formats (237174853)** - Confluence** — the authoritative spec for how formats flow through the compute datapath for given kernel.

Start from the FULL Quasar-supported format set (`QUASAR_DATA_FORMAT_ENUM_VALUES` in `tests/python_tests/helpers/format_config.py`). Evaluate each independently.

### Classify the operation's format domain

- **Float-only**: E.g. Float16, Float16_b, Float32, Tf32, MxFp8R, MxFp8P.
- **Integer-only**: E.g. Int8, UInt8, Int16, Int32.
- **Universal** - List the Data Formats

### Applicable-formats table

For each format below, decide whether it applies to this operation, and whether the operation must handle it explicitly (a format-specific branch) or for SFPU kernels use DEFAULT mod to cover.
Everything has to come from

| Format | Applicable | Rationale |
|--------|-----------|-----------|
| Float32 | Yes/No | … |
| Tf32 | Yes/No | … |
| Float16 | Yes/No | … |
| Float16_b | Yes/No | … |
| Int32 | Yes/No | … |
| Int16 | Yes/No | Quasar-specific (not on Blackhole) |
| Int8 | Yes/No | … |
| UInt8 | Yes/No | … |
| UInt16 | Yes/No | … |
| MxFp8R | Yes/No | L1-only, unpacked to Float16_b |
| MxFp8P | Yes/No | L1-only, unpacked to Float16_b |
| ... | Yes/No | ... |

### Format constraints

- If there is any reason why data format can't be implemented in contrast to reference architecture explain the produce the explanation for that HAS to CONTAIN proof of Confluence pages or code lines that are proof that it can't work
- Int32: Dest defaults to two's complement (unpack-to-dest copies L1 verbatim). Sign-magnitude only behind an opt-in template parameter, as add / mul_int32 / binary_comp do; cite the sibling defaults (why: unary_max_min PR #59439).

[FORMAT] -> {
    [REASON]
}
...

---

## Step 8: Complexity & Phases

### Complexity classification

- **Simple** — single target instruction or 2–3 primitives. E.g., `relu` → `TTI_SFPNONLINEAR(RELU_MODE)`. Writer time: <30 min.
- **Medium** — composable from existing primitives with moderate design work. E.g., `sigmoid` → mov + nonlinear + add + nonlinear.
- **Complex** — requires algorithm redesign or new patterns (LUT + polynomial + conditional execution).
- **No Direct Equivalent** — fundamental target hardware gap for the whole op. Do NOT write a normal analysis reporting this complexity; return `ANALYSIS_FAILED` instead (see Report on return).

A kernel the reference implements in 50 SFPI lines can still be **Simple** on Quasar if one SFPNONLINEAR mode matches. Classify by target complexity, not reference line count. Because SFPNONLINIEAR is a new instrcutin that can simplify the kernel.
These rules are applied for all types of kernels.

---

## Output Document Structure

Write `codegen/artifacts/{KERNEL_NAME}_analysis.md` with these sections, in order:

```markdown
# Analysis: {KERNEL_NAME}

## Problem Statement
[Step 1 — one paragraph stating what must compute / what is broken]

## Kernel Type
{sfpu | math | pack | unpack}

## SFPU Category
{unary | binary | ternary}
[sfpu only — write the bare category word on the line above so the tester can read it; omit this whole section for math/pack/unpack]
- Unified Python test: `tests/python_tests/{TARGET_ARCH}/test_eltwise_{category}_sfpu_{TARGET_ARCH}.py` (ternary: `test_sfpu_where_{TARGET_ARCH}.py`)
- Unified C++ test: `tests/sources/{TARGET_ARCH}/eltwise_{category}_sfpu_{TARGET_ARCH}_test.cpp` (ternary: `sfpu_where_{TARGET_ARCH}_test.cpp`)
- Dispatcher header: `tests/helpers/include/sfpu_operations_{TARGET_ARCH}.h` (ternary: none — where-specific harness)

## Reference (for generation) / Broken Code (for issue fix)
`{REFERENCE_PATH}`

## Target Pattern Survey
[Findings from Step 2]

## Available Instructions
[Findings from Step 3]

## Semantic → Instruction Mapping
[Findings from Step 4]

## Instruction Encoding Constraints
[Findings from Step 5]

## Production Callers
[Step 1 — file:line, bound template args / VectorMode, arch guards; or "None found"]

## Solution Approach
[Findings from Step 6]

## Code-Path Coverage Matrix
[Step 6a — one row per path, REQUIRED / optional, plus the template cross-product table]

## Format Applicability
[Findings from Step 7]

## Complexity & Phases
[Findings from Step 8]
```

---

## Success Criteria

You are done when the analysis document:

1. States the problem, not a description of the reference.
2. Enumerates the reference header and plans to port.
3. Cites at least one existing target kernel as the shape-of-truth.
4. Maps every semantic step to target instructions, each backed by a Confluence page or existing file.
5. Proposes function signatures that pass the `TTI_` constness check
6. Gives the writer a concrete instruction sequence they can implement without going back to the reference
7. Lists format applicability with technical rationale for every exclusion
8. Surfaces risks explicitly rather than hiding assumptions.
9. Lists the production callers and a code-path matrix whose REQUIRED rows cover every path those callers reach.

Report on return:
```
Problem: {one-line statement}
Complexity: {Simple | Medium | Complex | No Direct Equivalent}
Instructions to use: {count} mapped
Required paths: {count of REQUIRED rows in the Code-Path Coverage Matrix}
Analysis complete: codegen/artifacts/{KERNEL_NAME}_analysis.md
Ready for: llk-kernel-writer agent
```

If you cannot produce the analysis — the reference cannot be located, or the whole operation is **No Direct Equivalent** (a fundamental target hardware gap, not a per-entry-point flag handled in §6e) — do **NOT** write a partial analysis file. Return this block instead, with the blocker as the first line:
```
ANALYSIS_FAILED: {one-line blocker — the unresolved reference path, or the target capability that is missing}
Kernel: {KERNEL_NAME} ({TARGET_ARCH})
Reason: {what was attempted and why it cannot proceed}
```

---

## Self-Logging (MANDATORY — STRUCTURED TEMPLATE)

**Before returning, write `{LOG_DIR}/agent_analyzer.md` using the `Write` tool.**
The file MUST contain the sections below in order. This log is for
the **curated narrative and assumptions**, not a full transcript.

If no `LOG_DIR` was provided, place it in `codegen/artifacts`.

### Required sections (Write "None" if a section genuinely has no content)

```markdown
# Agent: llk-analyzer — {KERNEL_NAME} ({TARGET_ARCH})

## Inputs received
- Kernel / kernel_type: {KERNEL_NAME} / {KERNEL_TYPE}
- Reference arch / target arch: {REFERENCE_ARCH} / {TARGET_ARCH}
- Reference path: {REFERENCE_PATH}
- Any additional context the orchestrator passed (verbatim, do not summarize)

## Assumptions made
One bullet per assumption, in the shape:
`- [Claim] — [Why I believed it] — [How/when it could be wrong]`.

Examples:
- Used ADDR_MOD_7 rather than ADDR_MOD_3 — every existing Quasar SFPU kernel uses ADDR_MOD_7
  (`ckernel_sfpu_square.h`, `lrelu.h`, `typecast*.h`) and
  `_eltwise_sfpu_configure_addrmod_()` explicitly programs ADDR_MOD_7 —
  would break if the parent wrapper is changed to program a different addrmod.
- Treated `DataFormat.UInt16` as test-infrastructure-excluded, not kernel-excluded —
  `VALID_QUASAR_DEST_REG_FORMATS` in `data_format_inference.py` rejects UInt16 before
  the kernel runs — this assumption becomes wrong the moment the valid-formats list
  is widened.

**If you made no non-trivial assumptions, write "None" — but do not skip the section.**

## Reasoning summary (4–6 sentences)
Plain-prose summary of the approach. Name the key decisions and their reasons —
which reference variants you folded into template params, which target instructions
you built on, and why. If the analysis had to pivot (e.g., you started planning for
a UINT16-inclusive test matrix and then discovered the infra exclusion), say so.

## Decisions & trade-offs
For each non-trivial choice, write:
- **Choice**: one-line statement of the decision.
- **Alternatives**: what you considered.
- **Why**: the deciding factor (citation to Confluence / sibling kernel / ISA).

Typical analyzer decisions: the entry-point signature and which situations become
template params vs. `if constexpr` / traits dispatch, whether to add `init_{op}`,
whether to fuse `_sfp_rows_` into the outer loop, which reference variants to fold
in vs. flag as infeasible (§6e), which Confluence instructions to rely on.

## Artifacts read / written
- **Read** (files): list of paths with the role each played ("reference semantics",
  "canonical Quasar SFPU shape", "target test harness", ...).
- **Read** (Confluence pages): page ID + title + the single key finding extracted.
- **Read** (DeepWiki): repo + question + the summarized answer.
- **Written**: `codegen/artifacts/{KERNEL_NAME}_analysis.md` + this self-log.

## Open risks
Per `codegen/references/logging.md` § Open risks — `R<n> CLOSED: … — evidence: …` or
`R<n> DEFERRED: … — PR: …`, or "none". Example:
- R1 CLOSED: lock-up seed 0xFFFFFFFF unhandled — evidence: matrix row `seed=LOCKUP` (REQUIRED)
```
