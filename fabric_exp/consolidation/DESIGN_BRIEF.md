# v3 design brief (Phase 1, whole-ERISC designs)

Read first: `PLAN.md`, `PHASE1.md`, `ROUTER_MAP.md`, the header and the LOST-CONTEXT / OPEN rows of `LEDGER.md`,
`V3A_NOTES.md`, and RESULTS §82–§84 (`../RESULTS.md`, `grep -n '^## 8'`).

## Starting point
`fabric_exp/variants/v3a/`: the faithful consolidation of s79mc as `fabric_erisc_router_v3.cpp` plus a few shared
headers. Its code is instruction-for-instruction s79mc without the study's five branch hints, on all 440 logged
configs at -O3, -O2 and -Os. Measured (as s79mc without hints, RESULTS §83), cycles/packet:
E/W 16 B write 121.6, N/S 16 B 127.2, forwarded 16 B 111.6, 1-hop E/W 4352 B 132.3, 3-hop 146.8, multicast 140.2,
E/W scatter 164.9, fused 153.9, neighbor exchange 125.0. At -Os the same source is ~20% slower (§82); with VC1 it only
fits at -O2/-Os (§84). The `hook` probe moves s79m by up to +4.4 (§83).

## What a design is
A whole ERISC program, not a patch: kernel_main's setup for that ERISC, the outer loop, the inner loop, every channel
it services (VC0 and VC1), connection handling, completions and credits. Restructure freely. Ideas in the ledger that
lost "in context" may fit a better shape; ideas that lost on substance stay out.

## Hard rules
- Same protocol and behavior: every packet, credit, completion and flush the router produces today, in an order the
  other side accepts. Production paths keep working: Wormhole `#if`s, 1D/ring and the speedy path, VC1/VC2,
  deadlock avoidance/bubble flow control, first-level ack, stream-register credits, single-ERISC mode, telemetry, UDM,
  tensix mux, context switch, link retrain. If a design only changes the paths our configs use, the others must keep
  v3a's code.
- Same host interface: the compile-time and run-time args v3a reads. No builder changes in this phase. If a design
  wants a new host arg, write it down as a proposal and fall back to deriving the value in the kernel.
- E0 never does RX work. One worker channel per router. The worker connection/teardown check stays in every pass of
  the inner loop (never deferred to the outer loop).
- No branch-layout hints (`[[likely]]`, `[[unlikely]]`, `__builtin_expect`) except the one inherited from main on
  `NOC_SPARSE_MCAST_WRITE`. No register pinning, no `#pragma GCC unroll`/`optimize`/`__attribute__((optimize))`
  added to steer code generation, no layout games meant to steer scalarization or allocation. Source must not rely
  on how a particular compiler allocates registers or splits aggregates.
- Hardware-ordering needs (issuing several register/L1 reads back to back so their latencies overlap, RESULTS §17)
  may be expressed through one small, documented primitive with an explicit contract (for example an inline-asm
  block that issues N loads and returns N values), or through v3a's fences. Say which and why.
- Do not rely on trimming forwarding channels that have no upstream router: a separate team track owns that, and
  generic targets assume every serviced channel may be live.
- Code size must not grow with the number of channels the way inlining every channel does today. VC1 adds 3 sender
  channels and a receiver channel; E0 + E1 of a VC1 router must fit the 26,624 B kernel config buffer at -O2 at
  least, and the target is -O3 (today: 29,604 B at -O3).
- Keep the `// STABILITY_PROBE_*` markers working (see `perturb.py`): one DECL at file scope, PRELOOP/LOOP around and
  inside each ERISC's main loop, STATE in the per-channel state you keep.

## How a design is judged (Phase 1 evaluation)
I (the coordinating session) run hardware; designers work offline only. A design is measured on:
1. correctness: validation with RELAXED and STRICT flushes, VC1 multi-mesh sanity;
2. performance at -O3 against v3a (suite means across 2D 4352, mixes, flushed-signal mixes, 1D);
3. stability: the same at -O2 and -Os, and with the `hook` probe;
4. code size, and VC1 at -O2/-O3.
Static instruction or stack-access counts do not predict cycles (§82), so do not tune by counting instructions.
Design by removing work and by structure; use disassembly only to confirm the design compiled the way the source says.

## Tools (no device; never edit the live tree; no git state changes)
- Build every logged config: `python3 fabric_exp/consolidation/offline_build.py build --variant <dir> --out <dir>
  [--opt O3|O2|Os] [--hashes H,...] --jobs 48`. 536 configs at -O3/-O2 (440 VC0 + 96 VC1), 535 at -Os. VC1 hashes:
  `fabric_exp/consolidation/offline/vc1_hashes.txt`. All must build at all three levels (-Werror is on).
- Compare: `codegen_diff.py <base build> name=<build>`; `textcmp.sh`. Size: largest E0 + largest E1 loadable bytes
  (`riscv-tt-elf-readelf -lW`, sum of LOAD filesz) over the VC1 configs.
- Probes: `perturb.py <variant> <out> hook statefield`, then build both.
- A design touching only one ERISC must leave the other ERISC's code identical to v3a on every config
  (`codegen_diff.py` shows 0 changed for it), so its measurement is attributable.

## Deliverable
A variant dir `fabric_exp/variants/<name>/` (same layout as v3a: `fabric_erisc_router_v3.cpp` at top level, headers by
bare name or mirrored path), and `fabric_exp/consolidation/<NAME>_NOTES.md`: the design in a page (structure, state,
per-pass and per-packet work vs v3a, what was removed, which ledger ideas it uses and why), the semantic-equivalence
argument, build results at all three levels, VC1 size, probe builds, and open risks.
