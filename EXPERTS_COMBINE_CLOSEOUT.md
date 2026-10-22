# Experts + combine: close-out plan

Goal: one planner for the flat expert + combine op that works for every model and stage, then a final measurement
of flat, combine and fused efficiency per model.

## Models

GLM-5.3, GLM-5.3-Flash, Kimi K3, Kimi K2.7, MiniMax M3, DSv4.1-Flash.

## Rules

1. DRISC only: no Tensix gate/up weight readers.
2. Full arena.
3. Experts per chip as on a Galaxy, not a LoudBox.
4. Even splits only: a split is used only if it lowers the slowest core's work. No straggler cores.
5. Planner first: given an arena size, it plans every case, including the 8x4 Galaxy.

## Even splits

Gate/up splits over 2·I (gate and up together, the existing mode), not over I.

n = tiles / ceil(tiles / budget). Budgets are 64 gate/up cores and 32 down cores.

| Model | gate/up tiles (2·I/32) | Cores × tiles | down tiles (Hr/32) | Cores × tiles |
|---|---|---|---|---|
| GLM-5.3 | 128 | 64 × 2 | 192 | 32 × 6 |
| GLM-5.3-Flash | 128 | 64 × 2 | 128 | 32 × 4 |
| K2.7 | 128 | 64 × 2 | 224 | 32 × 7 |
| M3 | 192 | 64 × 3 | 192 | 32 × 6 |
| K3 | 192 | 64 × 3 | 112 | **28 × 4** |
| DSv4.1-Flash | 144 | **48 × 3** | 160 | 32 × 5 |

- **K3:** 64 / 28. Down is uneven at 32 (Hr = 3584); the 4 freed cores become combine cells.
- **DSv4.1-Flash is the tricky one.** 144 tiles average 2.25 per core on 64 cores, so the slowest core does 3 either way:
  - 48 × 3 matches 64 uneven cores and frees 16 cells for the combine;
  - 72 × 2 lowers the slowest core but needs 72 + 32 + relays + launcher = 109 of 120 cells, leaving about 11 for the combine.

## Experts per chip

| Model | 8x4 (32 chips) | 4x4 (16 chips) |
|---|---|---|
| GLM-5.3 | 8 | 16 |
| GLM-5.3-Flash | 9 | 18 |
| K3 | 28 | 56 |
| K2.7 | 12 | 24 |
| M3 | 4 | 8 |
| DSv4.1-Flash | 12 | 24 |

Cover 4x4 too: K3 is planned there, and 56 experts per chip is the hardest case for the arena.

## Planner

- **Inputs:** arena budget, mesh (SP × TP), chunk / tokens per chip, experts per chip, H / Hr, I, top-k.
- **Output:** one typed plan. No env overrides.
- **First test, host only:** 6 models × {8x4, 4x4, 2x4} × chunks {5120, 16384}. Assert that every case fits and that every split is even.
  - 8x4 and 4x4 use the mock Galaxy cluster descriptor.
- This is consolidation step 1: it retires most of the current env knobs.

## Measurement

- **Flat alone:** on bh-lb-09 (12x10 grid), representative of a Galaxy.
- **Combine and fused:** on the LoudBox 2x4 with Galaxy experts per chip and tokens per chip.
  - Efficiency is against the LoudBox's own busiest-link floor; label it that way.
- **Check first:** DSv4.1 at 48 gate/up cores changes the DRISC multicast groups per bank. Confirm the DRISC rate there first.

## Report: per model × stage × chunk

- **Plan:** experts per chip, M per expert, gate/up cores × tiles, down cores × tiles, relays, combine cells, accumulators, ports, ring depth, landing-pool size.
- **L1:** used and free per role (gate/up, down, relay, accumulator, port). The free L1 is what is left on the table.
- **Measured:**
  - flat µs, with DRAM % of 512 GB/s and math %;
  - combine µs vs its link floor;
  - fused µs vs max(flat, combine).
