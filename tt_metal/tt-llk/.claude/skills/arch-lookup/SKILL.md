---
name: arch-lookup
description: Look up Tensix architecture, instruction, or LLK implementation details across architectures. Orchestrates sage agents in parallel.
user_invocable: true
---

# /arch-lookup — Architecture Documentation Lookup

## Usage

```
/arch-lookup "How does SFPMAD work?"
/arch-lookup "How does unpack handle Float16 on Blackhole?"
/arch-lookup "What is BroadcastType?"
/arch-lookup "How do T0/T1/T2 synchronize?"
```

## Step 1: Analyze the Query

Determine which architectures are relevant:
- **Specific architecture mentioned** → launch ONLY that sage
- **No architecture specified** → launch ALL relevant sages in parallel (up to 3)
- **Pure ISA question with no arch context** → launch sage-wormhole + sage-blackhole only (they have DeepWiki access). Skip sage-quasar unless Quasar is mentioned — tt-isa-documentation has no Quasar content.

## Step 1b: Classify Question Type (MANDATORY)

Before writing the sage prompt, classify the question. This determines which source the sage must consult **first** — otherwise the sage will default to grep and miss authoritative HW information.

| Question Type | Examples | Required Primary Source |
|---------------|----------|-------------------------|
| **HW capability / spec** | "What formats does X support?", "What SFPU instructions exist?", "How wide is the FPU?", "How many DEST rows?" | **Confluence (QSR) / DeepWiki (WH,BH) FIRST**, code second |
| **HW behavior / semantics** | "How does SFPMAD handle NaN?", "When does the pipeline stall?", "What does instruction X do in HW?" | **Confluence / DeepWiki FIRST**, code second |
| **LLK implementation** | "How is matmul implemented?", "What does llk_unpack_AB do?", "Where is X wired up?" | **Code FIRST** (`tt_llk_*/`), docs second |
| **Mixed / end-to-end** | "How is Float16 handled from unpack to pack?" | Both required — docs for HW limits, code for the software path |

**Critical distinction:** LLK code shows what is **wired up in software**. It can lag or diverge from what the **hardware actually supports**. For HW-capability / HW-behavior questions, a grep-only answer is incomplete — the authoritative source is Confluence (Quasar) or DeepWiki / tt-isa-documentation (WH, BH).

## Step 2: Launch Sage Agents

Launch up to 3 agents IN PARALLEL using the Agent tool:

| Agent | Scope | Primary ISA Source |
|-------|-------|--------------------|
| `sage-wormhole` | `tt_llk_wormhole_b0/` | DeepWiki + assembly.yaml |
| `sage-blackhole` | `tt_llk_blackhole/` | DeepWiki + assembly.yaml |
| `sage-quasar` | `tt_llk_quasar/` | Confluence + assembly.yaml (NO DeepWiki) |

**Prompt construction rules:**

1. Start with the user's exact question, verbatim.
2. **If the question is HW capability / behavior (per Step 1b), you MUST explicitly instruct the sage to consult docs first.** Example phrasing to include:
   > "This is a HW-capability question. Consult Confluence first (sage-quasar) / DeepWiki tt-isa-documentation first (sage-wormhole, sage-blackhole) to get the authoritative HW answer. Use code grep only to cross-verify what is actually wired up in the LLK. If docs and code disagree, call out the conflict — do not silently prefer one."
3. Do NOT over-constrain the sage prompt with code-only framing (e.g. "grep for the enum", "cite file:line for the definition") on HW-capability questions — that will push the sage into grep-only mode and bypass its doc-consultation protocol.
4. Always allow the sage to return both doc-sourced facts (with staleness caveats) and code-sourced facts (with file:line).

## Step 3: Aggregate

After all sages return:
1. **Commonalities** — what's consistent across architectures
2. **Differences** — architecture-specific variations (highlight these clearly)
3. **Synthesize** — unified explanation with per-arch sections where they diverge

## Quality Checklist

Before responding, verify:
- [ ] Question was classified per Step 1b; sage prompt matched the classification
- [ ] For HW-capability / HW-behavior questions, the sage cited Confluence (QSR) or DeepWiki / tt-isa-documentation (WH, BH) — not just code grep
- [ ] Where HW docs and LLK code disagree, the conflict is called out (code lagging HW is common)
- [ ] WHY explained — hardware constraints documented, not just code description
- [ ] Default path identified — baseline vs variants distinguished
- [ ] All data format paths covered
- [ ] Code references include file:line; doc references include Confluence URL + last-updated date (Quasar)
- [ ] Architecture differences highlighted where they exist
