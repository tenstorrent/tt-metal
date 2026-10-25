# Router consolidation

Goals, rules and phases: `PLAN.md`. Measurements go in `../RESULTS.md` (§82 onward).

## Phase 0 documents
- `ROUTER_MAP.md`: what each ERISC does per pass and per packet in s79mc (flags 13129088 resolved), with costs from
  RESULTS, essential vs overhead; E0's inner-loop live state against the 27-register budget; per-packet budgets for
  the main traffic cases; observations for a redesign.
- `LEDGER.md`: every experiment in RESULTS §1–§81, one row per idea, classified IN / LOST-SUBSTANCE / LOST-CONTEXT /
  LOST-UNCLEAR / MIXED / OPEN, with the items a whole-router redesign should look at first.
- `V3_INVENTORY.md` (one level up) is the older keep/drop list; it predates §63–§82 and the ledger supersedes it.

## Tools
| Tool | Use |
| --- | --- |
| `offline_build.py collect` | Read logged JIT commands (`fabric_exp/results/p0o*_*.log`) into `offline/cmds.json` (O3, O2, Os; 440 router kernels each) |
| `offline_build.py build --variant DIR --out DIR [--opt O3] [--set-opt Os] [--flags=-f...] [--hashes H,..]` | Rebuild every logged router kernel from a variant (bare file names or mirrored `tt_metal/fabric/...` paths) without the device; ~25 s for 440. Needs the Phase 0 caches `~/.tt-metal-cache-p0{,-O2,-Os}` (generated headers, firmware link objects) |
| `codegen_diff.py BASE NAME=DIR...` | Per build: configs whose E0/E1 `.text` changed, instruction and frame deltas, instructions that differ in the hot E0/E1 |
| `textcmp.sh A.elf B.elf` | Same `.text` instruction stream at the same addresses |
| `perturb.py BASE OUT EDIT...` | Benign source edits (`reorder`, `field`, `hb`) |
| `p0a_runs.sh`, `p0b_runs.sh` | Broker jobs: a router at a given opt level in its own fresh cache (logs commands), or a variant swapped into the tree |
| `p0c_runs.sh` | Broker job: offline-built ELFs injected into a warm cache and run, tree untouched |
| `p0_compare.py` | Phase 0 table: per suite, mean and largest change vs -O3, and -O3 run-to-run spread |

Notes:
- `TT_METAL_FABRIC_OPT_LEVEL` needs a fresh kernel cache per level. With a warm cache the JIT reuses the other
  level's ELFs (RESULTS §82).
- Static instruction and stack counts do not predict cycles/packet (§82). Use offline builds to see whether and where
  code moved, and hardware for cost.
