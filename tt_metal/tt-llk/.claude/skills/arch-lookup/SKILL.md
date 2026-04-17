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

## Step 2: Launch Sage Agents

Launch up to 3 agents IN PARALLEL using the Agent tool:

| Agent | Scope | Primary ISA Source |
|-------|-------|--------------------|
| `sage-wormhole` | `tt_llk_wormhole_b0/` | DeepWiki + assembly.yaml |
| `sage-blackhole` | `tt_llk_blackhole/` | DeepWiki + assembly.yaml |
| `sage-quasar` | `tt_llk_quasar/` | Confluence + assembly.yaml (NO DeepWiki) |

Prompt each sage with the user's exact question.

## Step 3: Aggregate

After all sages return:
1. **Commonalities** — what's consistent across architectures
2. **Differences** — architecture-specific variations (highlight these clearly)
3. **Synthesize** — unified explanation with per-arch sections where they diverge

## Quality Checklist

Before responding, verify:
- [ ] WHY explained — hardware constraints documented, not just code description
- [ ] Default path identified — baseline vs variants distinguished
- [ ] All data format paths covered
- [ ] Code references include file:line
- [ ] Architecture differences highlighted where they exist
