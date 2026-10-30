# v3 design brief (Phase 1, whole-ERISC designs)

Read first: `PLAN.md`, `PHASE1.md`, `ROUTER_MAP.md`, `LEDGER.md` (both parts: the "Look at these first" list, the
LOST-CONTEXT / OPEN rows, and the Phase 1 part's "Applies to" / "In the lead?" columns, which say what one ERISC
program learned that the other has not used yet), `V3A_NOTES.md`, and RESULTS §82–§98 (`../RESULTS.md`,
`grep -n '^## [89]'`).

## Standing rules (user; collected 2026-10-26)
These hold for every design and override anything older below.
1. **ERISC roles.** Never put receive work on E0. One worker channel per router (mix flows on a link through forwarded
   traffic). E0's worker connection/teardown check stays in every pass of the inner loop; only the rare handling it
   triggers (open, close) may be out of line.
2. **No compiler-fragile tricks.** No register pinning, no allocation or layout steering, no branch hints, no new
   `optimize()` attributes or pragmas added for performance, no global memory clobbers. The source must not assume how
   GCC allocates, scalarizes, unrolls or peels. (One small, documented load-group primitive with an explicit contract
   remains allowed, as below; it is not a barrier.)
   **Amended 2026-10-29 (user):** placement primitives with a documented contract are allowed: `volatile`, and an
   inline-asm statement with a memory operand on one named object (`asm volatile("" : "+m"(obj))`: the object holds
   its current value in memory at that point and is reloaded after it). They push named objects out of registers at
   named points, each with its contract in a comment, behind one helper; what they guarantee is defined by the
   language or GCC's documented asm semantics, not by optimizer heuristics. Still banned: pinning anything into a
   register, and anything whose effect depends on heuristics (a runtime index to defeat scalarization, inlining or
   unrolling decisions, layout games). Placement is judged by its stability under -O3/-O2/-Os and benign edits.
3. **No rules or gates that burden other contributors or the shared toolchain** (RESULTS §96): no codegen-identity
   gates, no bans on ordinary constructs. The constant-multiply rule of §95 is withdrawn (§97); synth_mult is a
   documented gotcha with a diagnosis method (`rtlfirst.py`).
4. **NoC programming through the provided NoC APIs**, stateful `set_state` / `with_state` forms included, wrapped in
   named helpers where a pattern repeats. Where the API lacks a call, propose an API addition instead of hand-rolling
   register access. The same goes for the Ethernet TX-queue and stream-register helpers (`eth_txq_reg_read`,
   `remote_update_ptr_val`, `increment_local_update_ptr_val`, `NOC_STREAM_READ_REG`).
5. **Robust over peak.** Prefer designs whose performance is a smooth function of timing: no check whose success
   depends on landing just before or just after an event, so that work happens at once or a whole pass later
   (RESULTS §98, the early completion check). Holds and batching are bounded relative to the window and judged on
   bursty and short senders and small windows too. Do not rely on specific configuration numbers: slot counts vary by
   config. 1–2 cycles/packet is an accepted price for this.
6. **How designs are judged.** Suite means across -O3, -O2 and -Os decide, not single rows. Static counts (instructions,
   loads, the emulator's score) explain and target; hardware decides.

## Structure every design keeps (learned in Phase 1)
- Each ERISC's main loop is a function of its own that owns its state, and its once-per-round and rare work
  (termination and router-command tests, heartbeat, context-switch handshake, pause, rare send types, intermesh exit)
  lives in small functions of its own, so that the loop function holds only per-pass code. With that work inlined,
  edits to cold or per-round code re-lay out the whole loop (RESULTS §94 v3r2, §97 2x2; E0_AUDIT §1).
- Code size does not grow with the channel count the way inlining every channel does; VC1 E0 + E1 fits the 26,624 B
  buffer at -O3.
- Idle tests agree with the generic loop for every register value (exact predicates, RESULTS §93).
- E0 sets up the sender side only; E1 owns the receiver-side init (contract, V3L_NOTES §7).

## Checks every candidate passes offline (2026-10-26)
- Every logged config builds at -O3 / -O2 / -Os (536 / 536 / 534; the 2 garbled logged VC1 commands fail for every
  variant) and the 96 VC1 configs at -Os from their -O2 commands (`--set-opt Os`).
- The ERISC a design does not touch is PT_LOAD-identical to its base on every config (`scratch_tools/v3l/loadcmp.py`).
- Ordinary edits (`jitter_edits.py` coldmul, init, templates, os) leave the touched program's `run_main_loop` the same
  or moved, never DIFF, on its hot configs at all three levels (`scratch_tools/v3t/fncmp.py`).
- VC1 E0 + E1 within 26,624 B at -O3.
- `coverage.py build --opt O3` matches v3tk's case results (only the inherited `*_nofuse` and `wh_vc1` failures).
- Per-path counts against the base, with the method stated (E1: `rxscen.py` walker; E0: `emu.py`-based pass emulator).
- On hardware, besides validation and the suites: VC1 34/34 at every level with the capture armed, the jitter band
  (§96), the perturbation sweep (§98) for anything touching credit timing, and the burst benchmark for anything that
  holds credits.

## Starting point (2026-10-24; the lead is now `fabric_exp/variants/v3q`, RESULTS §98)
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
  on how a particular compiler allocates registers or splits aggregates. (Placement primitives with a documented
  contract are allowed since 2026-10-29: rule 2 above.)
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
Static instruction or stack-access counts do not predict cycles (§82): they explain and target, hardware decides
(standing rule 6). Design by removing work and by structure; use disassembly to confirm the design compiled the way
the source says.

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
