# tt-agent: Scaffold + tt-skill-creator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the `tt-agent/` directory structure inside tt-metal and implement `tt-skill-creator` — the meta-skill used to build all subsequent tt-agent skills.

**Architecture:** `tt-agent/` lives at `tt-metal/tt-agent/`. It contains `skills/`, `knowledge/`, `adapters/`, and meta docs. `tt-skill-creator` lives at `skills/meta/tt-skill-creator/` and wraps `/skill-creator` with TT-specific guidelines. All future skills are built using `tt-skill-creator`, proving it works by using it.

**Tech Stack:** Markdown with YAML frontmatter, Python 3 (YAML validation via `pyyaml`), bash.

**Spec:** `docs/superpowers/specs/2026-03-26-tt-agent-design.md`

---

## File Map

**Created:**
```
tt-agent/
  tt-agent.yaml
  README.md
  DESIGN.md
  CONTRIBUTING.md
  adapters/
    claude-code/
      CLAUDE.md
    codex/
      AGENTS.md
  skills/
    meta/
      tt-skill-creator/
        SKILL.md
        tt-guidelines.md
  knowledge/
    hardware/
      tensix-architecture.md
      circular-buffer-model.md
      quirks.md
    references/
      operators.md
      kernels.md
      sharding.md
      matmul.md
      ccl.md
      models.md
notes/
  .gitkeep
tt-agent/tests/
  test_skill_frontmatter.py
```

**Modified:**
- `CLAUDE.md` (root) — update `skills/` reference to `tt-agent/skills/`

---

## Task 1: Write skill frontmatter validation test (TDD)

Write the test first. It will fail until Task 6 creates the SKILL.md.

**Files:**
- Create: `tt-agent/tests/test_skill_frontmatter.py`

- [ ] **Step 1: Create test file**

```python
"""Validates that all SKILL.md files in tt-agent/skills/ have correct YAML frontmatter."""
import os
import yaml
import pytest
from pathlib import Path

SKILLS_ROOT = Path(__file__).parent.parent / "skills"

def find_skill_files():
    return list(SKILLS_ROOT.rglob("SKILL.md"))

def test_skill_files_exist():
    """At least one SKILL.md must exist."""
    files = find_skill_files()
    assert len(files) > 0, f"No SKILL.md files found under {SKILLS_ROOT}"

@pytest.mark.parametrize("skill_file", find_skill_files())
def test_skill_has_valid_frontmatter(skill_file):
    """Every SKILL.md must have valid YAML frontmatter with name and description."""
    content = skill_file.read_text()
    assert content.startswith("---\n"), f"{skill_file}: must start with YAML frontmatter (---)"

    end = content.find("\n---\n", 3)
    assert end != -1, f"{skill_file}: frontmatter block not closed"

    frontmatter_str = content[4:end]
    try:
        frontmatter = yaml.safe_load(frontmatter_str)
    except yaml.YAMLError as e:
        pytest.fail(f"{skill_file}: invalid YAML frontmatter: {e}")

    assert "name" in frontmatter, f"{skill_file}: missing 'name' field"
    assert "description" in frontmatter, f"{skill_file}: missing 'description' field"
    assert isinstance(frontmatter["name"], str) and frontmatter["name"].strip(), \
        f"{skill_file}: 'name' must be a non-empty string"
    assert isinstance(frontmatter["description"], str) and frontmatter["description"].strip(), \
        f"{skill_file}: 'description' must be a non-empty string"

@pytest.mark.parametrize("skill_file", find_skill_files())
def test_skill_name_matches_directory(skill_file):
    """Skill name in frontmatter must match the parent directory name."""
    content = skill_file.read_text()
    end = content.find("\n---\n", 4)
    frontmatter = yaml.safe_load(content[4:end])
    dir_name = skill_file.parent.name
    assert frontmatter["name"] == dir_name, \
        f"{skill_file}: name '{frontmatter['name']}' must match directory '{dir_name}'"
```

- [ ] **Step 2: Run test to verify it fails (no SKILL.md exists yet)**

```bash
cd tt-metal && pip install pyyaml pytest -q && pytest tt-agent/tests/test_skill_frontmatter.py -v
```

Expected: FAIL — `test_skill_files_exist` fails with "No SKILL.md files found"

- [ ] **Step 3: Commit test**

```bash
git add tt-agent/tests/test_skill_frontmatter.py
git commit -m "test: add SKILL.md frontmatter validation (TDD — fails until tt-skill-creator created)"
```

---

## Task 2: Create tt-agent/ scaffold

**Files:**
- Create: `tt-agent/tt-agent.yaml`
- Create: `notes/.gitkeep`
- Create: all empty skill/knowledge directories via placeholder files

- [ ] **Step 1: Create tt-agent.yaml**

```yaml
name: tt-agent
version: 0.1.0
description: Agentic tooling for Tenstorrent hardware development
mcps:
  - name: tt-device
    source: https://github.com/tenstorrent/tt-device-mcp
  - name: deepwiki
    source: deepwiki-mcp
```

- [ ] **Step 2: Create notes/.gitkeep**

Empty file at `notes/.gitkeep`. The `notes/` directory is the shared blackboard for agent and developer findings. It should be committed (so team members see the directory) but its contents are typically gitignored per team preference.

- [ ] **Step 3: Create directory placeholders for skills layers**

Create empty `.gitkeep` files to establish the directory structure:
- `tt-agent/skills/orchestration/.gitkeep`
- `tt-agent/skills/workflows/.gitkeep`
- `tt-agent/skills/tools/.gitkeep`
- `tt-agent/skills/meta/.gitkeep`
- `tt-agent/knowledge/hardware/.gitkeep`
- `tt-agent/knowledge/references/.gitkeep`
- `tt-agent/adapters/claude-code/.gitkeep`
- `tt-agent/adapters/codex/.gitkeep`
- `tt-agent/tests/.gitkeep` (already handled in Task 1)

- [ ] **Step 4: Commit scaffold**

```bash
git add tt-agent/ notes/
git commit -m "feat: add tt-agent/ scaffold and notes/ blackboard directory"
```

---

## Task 3: Create meta documentation

**Files:**
- Create: `tt-agent/README.md`
- Create: `tt-agent/DESIGN.md`
- Create: `tt-agent/CONTRIBUTING.md`

- [ ] **Step 1: Create README.md**

```markdown
# tt-agent

Agentic tooling for Tenstorrent hardware development. Enables AI agents (Claude Code,
Codex, and others) to autonomously design, write, test, profile, debug, and optimize
TT kernels, operators, and models.

## Prerequisites

- `tt-metal` cloned and built
- [tt-device-mcp](https://github.com/tenstorrent/tt-device-mcp) installed
- deepwiki-mcp configured in your agent environment

## Install (Claude Code)

Add to your root `CLAUDE.md` or ensure it references `tt-agent/adapters/claude-code/CLAUDE.md`.

## Quick Start

- "Design a new eltwise op for Wormhole B0"  → tt-designer
- "Optimize the attention block in my model"  → tt-iterator
- "This CI test is failing, fix it"           → tt-ci-fixer
- "Review this kernel for correctness"        → tt-code-review

## Learn More

- [DESIGN.md](DESIGN.md) — why things are built the way they are
- [CONTRIBUTING.md](CONTRIBUTING.md) — how to add skills and adapters
- [Spec](../docs/superpowers/specs/2026-03-26-tt-agent-design.md) — full design spec
```

- [ ] **Step 2: Create DESIGN.md**

```markdown
# tt-agent Design Decisions

This document records the non-obvious architectural decisions behind tt-agent and their
rationale. Future agents and developers: read this before changing anything structural.
Each decision is dated so you can judge whether it is still current.

---

## 2026-03-26: Co-location in tt-metal

tt-agent lives inside `tt-metal/tt-agent/` rather than a separate repo.

**Why:** Skills reference tt-metal paths deeply (API headers, programming examples,
operator source). The agent needs tt-metal checked out regardless. Co-location is
correct, not a compromise.

**Extraction path:** When tt-agent outgrows tt-metal, `git subtree split --prefix=tt-agent`
yields a clean repo with full history.

---

## 2026-03-26: Own the full stack (not built on superpowers)

tt-agent does not use the superpowers plugin framework as its base.

**Why:** Skills and knowledge must be platform-agnostic — authored once, delivered
to Claude Code, Codex, and future platforms. Owning the stack means we control the
format, versioning, and distribution. Adapters in `adapters/` handle per-platform packaging.

---

## 2026-03-26: Skills vs Knowledge vs Notes

Three content types that must not be conflated:

- **Skills** (`skills/`) — how to accomplish a task. Procedural instructions.
- **Knowledge** (`knowledge/`) — stable hardware invariants (silicon facts) + curated
  references (pointers to canonical code examples). Never volatile APIs.
- **Notes** (`notes/`) — shared blackboard. Findings written by agents and humans,
  shared across sessions and team members.

**Why the split:** The TT software stack evolves rapidly. Inlining API signatures or
implementation patterns into static files creates lies. Volatile knowledge is always
learned fresh from source via `tt-learn`.

---

## 2026-03-26: Volatile knowledge via tt-learn + deepwiki-mcp

API signatures, op implementations, sharding patterns, CCL usage — never written down.

**Why:** These change with every PR. The `tt-learn` skill researches the live codebase
via deepwiki-mcp on demand, using `knowledge/references/` as starting points. Findings
are written to `notes/` with a commit hash, so readers can judge freshness.

---

## 2026-03-26: notes/ as shared blackboard

The `notes/` directory at the repo root is the team's shared, evolving knowledge cache.
Not session memory — notes persist across sessions and are shared between developers
and multiple agent sessions. Named "notes" (not "workspace", "memory", or "context").

---

## 2026-03-26: Two MCP dependencies only

tt-device-mcp (hardware execution) and deepwiki-mcp (codebase research). Everything
else via CLI and Bash.

**Why:** MCPs add configuration burden. Only add one when CLI genuinely cannot do the job.
Hardware execution requires a persistent device connection (MCP). Semantic codebase search
is deepwiki-mcp's purpose. Everything else (build, profile invocation, file I/O) is CLI.

---

## 2026-03-26: Skill layers

Four layers visible in the filesystem:
- `orchestration/` — routes, plans, decomposes
- `workflows/` — autonomous loops (iterate until converged)
- `tools/` — single-purpose capabilities invoked during execution
- `meta/` — system-level utilities: extend the system (tt-skill-creator) and learn from it (tt-learn)

**Workflow layer is intentionally thin.** tt-iterator, tt-ci-fixer, tt-bisect share the
same base loop (hypothesize → implement → run → analyze → next hypothesis). What differs
is triggering context and convergence criteria.

---

## 2026-03-26: tt-designer as unified design-phase skill

`tt-designer` in `tools/` combines TT-specific brainstorming, performance estimation
(roofline, arithmetic intensity), and data-movement planning (CCL strategy) into one skill.

**Why unified:** These are not separate invocations — planning a TT implementation naturally
covers all three. Wraps `/superpowers:brainstorm` and adds TT hardware constraint grilling.
Bookend to `tt-code-review`: designer before writing code, code-review after.

---

## 2026-03-26: tt-skill-creator first, then use it to build everything

All skills after tt-skill-creator are built using tt-skill-creator itself.

**Why:** Validates the tool works. Every subsequent skill is both a deliverable and a
test of tt-skill-creator's quality. "Use what you build."
```

- [ ] **Step 3: Create CONTRIBUTING.md**

```markdown
# Contributing to tt-agent

## Adding a New Skill

1. **Decide the layer.** Which directory does your skill belong in?
   - `skills/orchestration/` — routes and dispatches requests
   - `skills/workflows/` — autonomous loops with convergence criteria
   - `skills/tools/` — single-purpose capabilities used during execution
   - `skills/meta/` — system-level utilities (extending or learning from the system)

2. **Use tt-skill-creator.** Invoke the `tt-skill-creator` skill to create the SKILL.md.
   It wraps `/skill-creator` and applies tt-agent conventions automatically.

3. **Follow the golden rules:**
   - Skills describe *how to do something*, not *what the API is*
   - Point to code locations, never inline API signatures: `see tt_metal/hw/inc/api/dataflow/`
   - Keep SKILL.md ≤ 150 lines; move domain content to sub-files
   - Every SKILL.md must have valid YAML frontmatter with `name` and `description`
   - `name` must match the directory name exactly

4. **Run the validation test:**
   ```bash
   pytest tt-agent/tests/test_skill_frontmatter.py -v
   ```

## Adding to knowledge/

`knowledge/hardware/` — only for silicon-stable facts (Tensix architecture, NOC topology,
tile granularity). If it could change in a software release, it does not go here.

`knowledge/references/` — curated pointers to canonical examples, one file per topic.
Format: path + one-line description. No content inlined. Update paths when examples move.

**Do not add:** API signatures, function names, implementation patterns, op lists.
These belong to `tt-learn` (fetched fresh from source), not to static files.

## Adding a Platform Adapter

Create `adapters/<platform-name>/` and add the platform's entrypoint file
(`CLAUDE.md` for Claude Code, `AGENTS.md` for Codex, etc.). Reference the same
`skills/` and `knowledge/` directories — content does not change per platform.

## PR Conventions

- One skill per PR when possible
- Include a brief note on which layer the skill belongs to and why
- Run `pytest tt-agent/tests/` before submitting
- Update `knowledge/references/` if your skill introduces canonical examples worth pointing to
```

- [ ] **Step 4: Commit meta docs**

```bash
git add tt-agent/README.md tt-agent/DESIGN.md tt-agent/CONTRIBUTING.md
git commit -m "docs: add tt-agent README, DESIGN, CONTRIBUTING"
```

---

## Task 4: Create Claude Code adapter and update root CLAUDE.md

**Files:**
- Create: `tt-agent/adapters/claude-code/CLAUDE.md`
- Create: `tt-agent/adapters/codex/AGENTS.md`
- Modify: `CLAUDE.md` (root)

- [ ] **Step 1: Create tt-agent/adapters/claude-code/CLAUDE.md**

Read the root `CLAUDE.md`, apply two changes, and write the result to `tt-agent/adapters/claude-code/CLAUDE.md`:

**Change 1** — update the skills path reference (line 6):
```
- See `skills/` for agentic workflow instructions (writing ops, debugging, etc.).
+ See `tt-agent/skills/` for agentic workflow instructions (writing ops, debugging, etc.).
```

**Change 2** — append these three lines inside the "Key Repo Directories" code block, just before the closing backticks:
```
tt-agent skills:
  tt-agent/skills/                               # Agentic workflow skills
  tt-agent/knowledge/                            # Stable hardware knowledge + references
  notes/                                         # Shared blackboard (agent + developer findings)
```

All other content is identical to root `CLAUDE.md`. This file is the canonical source of truth for the Claude Code adapter.

- [ ] **Step 2: Create tt-agent/adapters/codex/AGENTS.md stub**

```markdown
# tt-agent for Codex

See tt-agent/skills/ for agentic workflow instructions.
See tt-agent/DESIGN.md for architecture decisions.

## MCP Dependencies
- tt-device-mcp: https://github.com/tenstorrent/tt-device-mcp
- deepwiki-mcp

<!-- TODO: expand with Codex-specific configuration when adapter is fully developed -->
```

- [ ] **Step 3: Update root CLAUDE.md**

Change the single line that references `skills/` to reference `tt-agent/skills/`:

Find: `See \`skills/\` for agentic workflow instructions (writing ops, debugging, etc.).`
Replace with: `See \`tt-agent/skills/\` for agentic workflow instructions (writing ops, debugging, etc.).`

- [ ] **Step 4: Commit adapters and root update**

```bash
git add tt-agent/adapters/ CLAUDE.md
git commit -m "feat: add Claude Code and Codex adapters, update root CLAUDE.md path"
```

---

## Task 5: Create knowledge/hardware/ files

**Files:**
- Create: `tt-agent/knowledge/hardware/tensix-architecture.md`
- Create: `tt-agent/knowledge/hardware/circular-buffer-model.md`
- Create: `tt-agent/knowledge/hardware/quirks.md`

- [ ] **Step 1: Create tensix-architecture.md**

```markdown
# Tensix Architecture

Stable silicon facts. Update only when hardware changes.

## Processing Elements

Each Tensix core contains five 32-bit in-order RISC-V CPUs:

| CPU | Role | Coprocessor thread |
|-----|------|--------------------|
| DM0 (B) | Reader: moves data from DRAM/L1 into CBs | — |
| DM1 (NC) | Writer: moves data from CBs to DRAM/L1 | — |
| T0 (Unpack) | Unpacks tiles from L1 into register file | T0 |
| T1 (Math) | Executes FPU/SFPU operations | T1 |
| T2 (Pack) | Packs results from register file to L1 | T2 |

## Compute Units

**FPU (Matrix Unit):** Matrix multiplication and element-wise operations on tiles.
One operation per cycle: 8×16 × 16×16 = 8×16 output tiles.
Math fidelity controls precision/throughput trade-off (LoFi, HiFi2, HiFi4).

**SFPU (Vector Unit):** 32-lane SIMD on 32-bit float/int values. Used for
transcendental functions, activations, and operations not expressible as matmul.
Slower than FPU for bulk operations.

## Memory

**L1 SRAM:** 1.5 MB per Tensix core. Holds circular buffers, sharded tensor tiles,
and kernel stack. Statically allocated before kernel launch — no dynamic allocation.

**DRAM:** Off-chip. Accessed via NOC. High latency, high bandwidth.
Tensors too large for L1 live in DRAM and are streamed through L1 via circular buffers.

## NOC (Network on Chip)

Two NOCs run in opposite directions forming a 2D torus across the chip:
- NOC0: conventionally used for reads (DM0 pulls from DRAM/remote L1)
- NOC1: conventionally used for writes (DM1 pushes to DRAM/remote L1)

Operations: `noc_async_read`, `noc_async_write`, `noc_async_read_barrier`,
`noc_async_write_barrier`. Transfers are asynchronous — always issue a barrier
before reading data that was written via NOC.

## Execution Model

SPMD: all cores run the same kernel binary with different runtime args (typically
their own coordinates). The host distributes work by assigning different tensor
offsets to each core via runtime args.

## Hardware Targets

| Target | Notes |
|--------|-------|
| Wormhole B0 | Current production hardware |
| Blackhole | Next-gen, higher core count |
| Quasar | Research/future |
```

- [ ] **Step 2: Create circular-buffer-model.md**

```markdown
# Circular Buffer Model

The fundamental coordination primitive between reader, compute, and writer kernels.
This model is hardware-stable — the CB API rarely changes.

## Concept

A circular buffer (CB) is a typed, tile-oriented FIFO in L1 SRAM shared between
producer and consumer kernels:

```
Reader (DM0)         Compute (T0/T1/T2)       Writer (DM1)
     |                      |                       |
  DRAM → [CB_IN_0]  →  process  →  [CB_OUT_0]  →  DRAM
```

## Producer API (reader/writer kernel)

```cpp
cb_reserve_back(cb_id, num_tiles);   // wait until space is available
// ... fill tiles at cb_write_pointer(cb_id) ...
cb_push_back(cb_id, num_tiles);      // signal tiles are ready
```

## Consumer API (compute kernel)

```cpp
cb_wait_front(cb_id, num_tiles);     // wait until tiles are available
// ... read tiles at cb_read_pointer(cb_id) ...
cb_pop_front(cb_id, num_tiles);      // signal tiles are consumed
```

## Host-side Configuration

```cpp
CircularBufferConfig cb_config = CircularBufferConfig(
    total_size_bytes,
    {{cb_index, dataformat}}
).set_page_size(cb_index, tile_size_bytes);

auto cb = CreateCircularBuffer(program, core_range, cb_config);
```

See `tt_metal/api/tt-metalium/circular_buffer_config.hpp` for full API.

## Design Rules

- **CB index**: integer 0–31. Convention: 0/1 for inputs, 16 for output.
- **Size**: must fit in L1. Size all CBs at once and verify total ≤ 1.5 MB.
- **Double-buffering**: use 2× tile size to overlap data movement with compute.
- **Format**: must match the data format of the tensor being streamed.
- **Multiple CBs**: use separate CBs for each input tensor and for output.

## Approximate Tile Sizes by Format

| Format | Bytes/tile |
|--------|-----------|
| BFLOAT16 | ~2048 |
| BFLOAT8_B | ~1088 |
| BFLOAT4_B | ~576 |
| FLOAT32 | ~4096 |
```

- [ ] **Step 3: Create quirks.md**

```markdown
# Tensix Hardware Quirks

Things that look like C++ but aren't. Every TT kernel developer hits these.
Hardware-stable — these reflect silicon behavior, not software conventions.

## No Dynamic Allocation

No `malloc`, `new`, or dynamic data structures in kernels. All buffers (CBs,
semaphores, L1 scratch) are statically configured from the host before launch.
Violating this silently corrupts memory.

## Entry Point is kernel_main(), Not main()

```cpp
void kernel_main() {    // correct
    // ...
}
int main() { }          // wrong — never called
```

## 32-bit RISC-V

No `uint64_t`, `int64_t`, `double`, or 64-bit pointer arithmetic. Addresses are
32-bit. Use `uint32_t` for addresses and sizes.

## Compile-time vs Runtime Args

Two separate arg mechanisms — do not mix them:

```cpp
// Compile-time (template/constexpr, baked into binary at host compile time)
constexpr uint32_t tile_size = get_compile_time_arg_val(0);

// Runtime (passed per-launch, can vary per core)
uint32_t src_addr = get_arg_val<uint32_t>(0);
```

## ALWI: Math Inline Assembly Macro

Math operations in compute kernels use the `ALWI` macro:
```cpp
ALWI void add_tiles(uint32_t in0_cb, uint32_t in1_cb, uint32_t dst_idx) {
    // ...
}
```
Functions called from within `math_main` must be marked `ALWI`.

## No Floating Point in Dataflow Kernels

DM0 and DM1 (reader/writer) RISC-V cores have no FPU. Do not use `float` or
`double` in dataflow kernels. Address calculations must be integer-only.

## NOC Barriers Are Mandatory

`noc_async_read` and `noc_async_write` are asynchronous. Always follow with
a barrier before reading the transferred data:

```cpp
noc_async_read(src_addr, dst_addr, size);
noc_async_read_barrier();   // required — data not valid until after this
```

## NOC Direction Convention

- NOC0: reads (DM0 pulling from DRAM or remote L1)
- NOC1: writes (DM1 pushing to DRAM or remote L1)

Using the wrong NOC doesn't fail immediately — it causes throughput degradation
or deadlocks under load.

## Deadlock Conditions

Deadlock occurs when:
- Reader and writer block on the same CB simultaneously (classic circular wait)
- `cb_reserve_back` and `cb_wait_front` are called in the wrong order
- A kernel exits without consuming all tiles it was supposed to

Always verify the reader/compute/writer tile count contract before running.
```

- [ ] **Step 4: Commit hardware knowledge**

```bash
git add tt-agent/knowledge/hardware/
git commit -m "docs: add knowledge/hardware — tensix architecture, CB model, quirks"
```

---

## Task 6: Create knowledge/references/ files

**Files:**
- Create: `tt-agent/knowledge/references/operators.md`
- Create: `tt-agent/knowledge/references/kernels.md`
- Create: `tt-agent/knowledge/references/sharding.md`
- Create: `tt-agent/knowledge/references/matmul.md`
- Create: `tt-agent/knowledge/references/ccl.md`
- Create: `tt-agent/knowledge/references/models.md`

These files contain **pointers only** — path + one-line description. No content inlined.
Update paths when examples move; never add API content here.

- [ ] **Step 1: Create operators.md**

```markdown
# Operator Reference Pointers

## Canonical simple eltwise op (full device operation pattern)
`ttnn/cpp/ttnn/operations/eltwise/binary/device/`
Start here for: validate(), compute_output_specs(), select_program_factory(), hash().
The binary_device_operation.hpp shows the attribute struct + dispatch pattern.

## Program factory example
`ttnn/cpp/ttnn/operations/eltwise/binary/device/element_wise_multi_core_program_factory.cpp`
Shows: CB setup, kernel creation, per-core work distribution, runtime args.

## Multi-core work distribution
`tt_metal/programming_examples/eltwise_sf_binary_loop/`
Simpler than ttnn ops — good for understanding the host-side program setup loop.
```

- [ ] **Step 2: Create kernels.md**

```markdown
# Kernel Reference Pointers

## Canonical reader/compute/writer pattern
`tt_metal/programming_examples/eltwise_binary/`
Three kernels: reader (NOC→CB), compute (CB→CB), writer (CB→NOC). Best starting point.

## Compute-only (no data movement)
`tt_metal/programming_examples/add_2_integers_in_compute/`
Minimal example of compute kernel with compile-time and runtime args.

## Dataflow API
`tt_metal/hw/inc/api/dataflow/dataflow_api.h`
All NOC and CB functions for reader/writer kernels. Read this for exact signatures.

## Compute API
`tt_metal/hw/inc/api/compute/`
All tile operations, FPU/SFPU functions. Read this for exact compute signatures.

## Circular buffer host config
`tt_metal/api/tt-metalium/circular_buffer_config.hpp`
Host-side CB setup API.
```

- [ ] **Step 3: Create sharding.md**

```markdown
# Sharding Reference Pointers

## Tensor sharding tech report
`tech_reports/tensor_sharding/`
Covers: interleaved vs height/width/block sharding, when to use each, L1 constraints.

## ND sharding example
Search: `ttnn/cpp/ttnn/operations/` for ops that implement `height_sharded` or `block_sharded`
to see how shard spec is set up in practice.

## Sharding in ttnn tests
`tests/ttnn/unit_tests/operations/`
Tests often show the full sharding setup including MemoryConfig and ShardSpec.
```

- [ ] **Step 4: Create matmul.md**

```markdown
# Matmul Reference Pointers

## GEMM FLOPS analysis
`tech_reports/GEMM_FLOPS/GEMM_FLOPS.md`
Per-core TFLOPS by math fidelity. How to calculate arithmetic intensity and
determine compute-bound vs memory-bound.

## Matrix engine deep dive
`tech_reports/matrix_engine/matrix_engine.md`
How the FPU executes 8×16 × 16×16 tile operations. Latency, throughput, fidelity.

## Matmul op implementation
`ttnn/cpp/ttnn/operations/matmul/`
Current matmul implementation including 1D/2D/reuse strategies.

## Data formats and accuracy
`tech_reports/data_formats/data_formats.md`
bfloat16/bfloat8_b/bfloat4_b accuracy and memory trade-offs.
```

- [ ] **Step 5: Create ccl.md**

```markdown
# CCL (Collective Communications) Reference Pointers

## Ethernet and multi-chip guide
`tech_reports/EthernetMultichip/BasicEthernetGuide.md`
Ethernet link topology, bandwidth, latency. How to reason about multi-chip data movement.

## CCL operations
`ttnn/cpp/ttnn/operations/ccl/`
All-gather, all-reduce, reduce-scatter implementations. Read for current patterns.

## CCL tests
`tests/ttnn/unit_tests/operations/ccl/`
Shows how CCL ops are set up and what parameters control topology.
```

- [ ] **Step 6: Create models.md**

```markdown
# Model Reference Pointers

## Production LLM layers
`models/tt_transformers/tt/`
Attention, MLP, layernorm, embedding — how production models are structured.
LightweightModule pattern, weight loading, forward pass is device-only.

## Shared model utilities
`models/common/`
Weight loading helpers, PCC validation (assert_with_pcc), tensor save/load.

## Model demos and end-to-end tests
`models/demos/`
Full inference demos including multi-chip setups.

## Model-level tests
`models/tt_transformers/tests/`
How to test a model block in isolation: load weights, run forward, check PCC.

## Model bringup tech report
`tech_reports/ttnn/TTNN-model-bringup.md`
Step-by-step protocol for porting a new model to TT hardware.

## Advanced performance optimization
`tech_reports/AdvancedPerformanceOptimizationsForModels/`
Trace mode, 2CQ, op fusion, math fidelity tuning.
```

- [ ] **Step 7: Commit references**

```bash
git add tt-agent/knowledge/references/
git commit -m "docs: add knowledge/references — curated pointers per topic"
```

---

## Task 7: Create tt-skill-creator/SKILL.md

**Files:**
- Create: `tt-agent/skills/meta/tt-skill-creator/SKILL.md`

- [ ] **Step 1: Run test to confirm it still fails**

```bash
pytest tt-agent/tests/test_skill_frontmatter.py -v
```

Expected: FAIL — "No SKILL.md files found"

- [ ] **Step 2: Create SKILL.md**

```markdown
---
name: tt-skill-creator
description: "Create and improve skills for the tt-agent system. Use when a TT developer wants to write a new tt-agent skill, improve an existing one, or verify a skill follows tt-agent conventions. Invokes /skill-creator for base mechanics then applies Tenstorrent-specific guidelines from tt-guidelines.md."
---

# TT Skill Creator

## Purpose

Creates high-quality skills for the tt-agent system. Wraps `/skill-creator`
(which handles generic skill mechanics: format, frontmatter, progressive load
tables, description optimization, evals) and adds TT-specific guidance on top.

## When to Invoke

Use when asked to:
- Create a new tt-agent skill (kernel, op, model, workflow, or meta)
- Improve or review an existing tt-agent skill
- Verify that a skill follows tt-agent conventions

## How This Skill Works

1. **Invoke `/skill-creator`** for all base mechanics: SKILL.md format, YAML
   frontmatter structure, progressive load table layout, description wording,
   and eval design.

2. **Load `tt-guidelines.md`** and apply TT-specific rules on top of whatever
   `/skill-creator` produces.

## Progressive Load Table

| Sub-task | Load |
|---|---|
| Any skill creation or review | `tt-guidelines.md` |
| Base skill format, frontmatter, evals | Invoke `/skill-creator` |

## Output

A skill directory containing:
- `SKILL.md` with valid YAML frontmatter (`name` matches directory, `description`
  is a single rich sentence optimized for triggering)
- Sub-files for domain knowledge, loaded progressively
- Placed in the correct layer directory under `tt-agent/skills/`
```

- [ ] **Step 3: Run test to verify it passes**

```bash
pytest tt-agent/tests/test_skill_frontmatter.py -v
```

Expected: PASS — all three tests pass for `tt-skill-creator/SKILL.md`

- [ ] **Step 4: Commit**

```bash
git add tt-agent/skills/meta/tt-skill-creator/SKILL.md
git commit -m "feat: add tt-skill-creator SKILL.md — meta-skill for building TT skills"
```

---

## Task 8: Create tt-skill-creator/tt-guidelines.md

**Files:**
- Create: `tt-agent/skills/meta/tt-skill-creator/tt-guidelines.md`

- [ ] **Step 1: Create tt-guidelines.md**

```markdown
# TT Skill Guidelines

TT-specific rules for building skills in the tt-agent system. Load this alongside
`/skill-creator` when creating or reviewing any tt-agent skill.

---

## 1. Skill Layer Placement

Every skill belongs to exactly one layer. Place SKILL.md in the correct subdirectory:

| Layer | Path | Character |
|---|---|---|
| Orchestration | `skills/orchestration/` | Routes, plans, decomposes requests |
| Workflows | `skills/workflows/` | Autonomous loops — runs until convergence |
| Tools | `skills/tools/` | Single-purpose, invoked during execution |
| Meta | `skills/meta/` | System-level: extend or learn from the system |

**Decision rule:**
- Does it route work to other skills? → orchestration
- Does it run a loop until a goal is met? → workflows
- Does it do one concrete thing (profile, test, review, design)? → tools
- Does it help build or understand tt-agent itself? → meta

---

## 2. Skills vs Knowledge vs Notes

When writing a skill, be precise about where content goes:

**In the skill (SKILL.md or sub-files):**
Procedural instructions — how to accomplish a task. Steps, decision trees, patterns.

**In `knowledge/hardware/`:**
Silicon-stable facts only: Tensix architecture, NOC topology, tile granularity, CB model,
hardware quirks. If it could change in a software release, it does not go here.

**In `knowledge/references/`:**
Curated pointers to canonical examples, one file per topic. Format: path + one-line
description. No content inlined. Update paths when examples move; never add volatile info.

**Left to `tt-learn`:**
Everything volatile — API signatures, function names, op implementations, current patterns
in the codebase, sharding strategies in use. **Never write these down in a skill.** The
agent reads them fresh from source via `tt-learn` when needed.

**Written to `notes/`:**
Findings produced during work — context briefs, experiment logs, profiler results.
Skills write to `notes/`; they reference `knowledge/` for stable facts.

---

## 3. "Point to Code, Not Inline APIs"

The most common mistake: inlining API documentation that will go stale.

**Wrong — do not do this:**
```
To read from DRAM, use:
  noc_async_read(uint32_t src_noc_addr, uint32_t dst_local_l1_addr, uint32_t size)
```

**Right:**
```
For NOC read/write API, see:
  tt_metal/hw/inc/api/dataflow/dataflow_api.h
```

Sub-files should describe *patterns and intent*, not *API signatures*. The agent
reads the actual header when it needs the exact signature.

---

## 4. Progressive Load Pattern

Every SKILL.md must have a progressive load table. Only list sub-files that exist.
Do not create sub-files for content that fits in SKILL.md itself.

```markdown
## Progressive Load Table

| Sub-task | Load |
|---|---|
| [specific sub-task description] | `sub-file.md` |
```

**Size rules:**
- SKILL.md ≤ 150 lines. If it grows beyond that, move domain content to sub-files.
- Sub-files: 150–250 lines each. One clear topic per file.
- Total context per task: SKILL.md + 1-2 sub-files + source files as needed.

---

## 5. YAML Frontmatter Requirements

Every SKILL.md must start with:

```yaml
---
name: <directory-name>        # must match directory name exactly
description: "<rich single sentence optimized for triggering>"
---
```

**Description guidelines (from /skill-creator):**
- Single sentence, no period at end
- Starts with what it does, includes when to use it
- Specific enough to trigger correctly, not so broad it triggers on everything
- Include key synonyms and trigger phrases

Run `pytest tt-agent/tests/test_skill_frontmatter.py` to validate.

---

## 6. Workflow Skills Must Define Convergence

If creating a skill in `skills/workflows/`, it must explicitly define:

```markdown
## Convergence Criteria
- **Success:** [explicit condition — e.g., "PCC > 0.999 AND throughput ≥ target"]
- **Local optimum:** [e.g., "5 iterations with < 5% improvement"]
- **Escalate:** [what the agent reports when it cannot converge]
```

---

## 7. TT Quality Bar

Any skill that involves code generation must include a step that verifies:
- **PCC > 0.999** vs PyTorch reference for numerical correctness
- **Hardware-aware correctness**: CB sizing fits L1, tile alignment, NOC conventions
- Output matches patterns found in tt-metal (not invented conventions)

Reference: `tt-agent/knowledge/hardware/` for invariants. Use `tt-learn` for current
patterns in the codebase.

---

## 8. Notes Naming Convention

When a skill writes to `notes/`, use consistent naming:

| Type | Pattern | Example |
|---|---|---|
| Context brief (tt-learn output) | `context-<topic>.md` | `context-matmul.md` |
| Experiment log | `experiments-<task>.md` | `experiments-mlp-opt.md` |
| Per-task plan | `plan-<task>.md` | `plan-fuse-attention.md` |
| Profiler finding | `profile-<workload>.md` | `profile-llama-decode.md` |

Each note must include: topic, date, tt-metal commit hash, sources read.

---

## 9. tt-learn Integration

When a skill needs volatile information (current API, implementation patterns, op
details), it should invoke `tt-learn` rather than encoding the information inline.

Pattern in sub-files:
```markdown
For current [topic] patterns, invoke `tt-learn("[topic]")`.
Starting points: `knowledge/references/[relevant-file].md`
```

---

## 10. Self-Check Before Finalizing a Skill

Before handing off a new skill, verify:
- [ ] SKILL.md starts with valid YAML frontmatter
- [ ] `name` matches directory name
- [ ] `description` triggers correctly (test with /skill-creator eval guidance)
- [ ] Progressive load table is present and all referenced files exist
- [ ] No API signatures inlined — all volatile content points to source
- [ ] Skill is in the correct layer directory
- [ ] `pytest tt-agent/tests/test_skill_frontmatter.py` passes
```

- [ ] **Step 2: Commit**

```bash
git add tt-agent/skills/meta/tt-skill-creator/tt-guidelines.md
git commit -m "feat: add tt-skill-creator/tt-guidelines.md — TT-specific skill authoring rules"
```

---

## Task 9: End-to-end validation

Prove tt-skill-creator works by using it to create `tt-orchestrator` — the first real
skill in the system. This is the "use what you build" validation.

- [ ] **Step 1: Run full test suite**

```bash
pytest tt-agent/tests/test_skill_frontmatter.py -v
```

Expected: all tests pass

- [ ] **Step 2: Invoke tt-skill-creator in a fresh Claude Code session**

Start a new Claude Code session in tt-metal. Ask:
> "Use tt-skill-creator to create a new skill: tt-orchestrator. It should live in
> skills/orchestration/ and route TT development requests to the appropriate
> workflow or tool skill."

Verify the output:
- [ ] SKILL.md is created at `tt-agent/skills/orchestration/tt-orchestrator/SKILL.md`
- [ ] YAML frontmatter is valid (`name: tt-orchestrator`, description present)
- [ ] `/skill-creator` was invoked as part of the process
- [ ] TT-specific guidelines (layer placement, progressive load, no inline APIs) are applied
- [ ] `pytest tt-agent/tests/test_skill_frontmatter.py` passes with 2 SKILL.md files

- [ ] **Step 3: Commit tt-orchestrator if it looks good**

```bash
git add tt-agent/skills/orchestration/tt-orchestrator/
git commit -m "feat: add tt-orchestrator — first skill built using tt-skill-creator"
```

- [ ] **Step 4: Write a note documenting what worked and what to improve**

Create `notes/plan-tt-agent-phase1.md` capturing:
- What tt-skill-creator got right
- What it missed or got wrong
- Improvements needed in tt-guidelines.md before building more skills

```bash
git add notes/plan-tt-agent-phase1.md
git commit -m "notes: tt-agent phase 1 retrospective"
```
