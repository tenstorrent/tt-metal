# Methods: hardware probes and simulation

Use these when the quick checks and the order/code methods show WHAT moves, but not WHY.
They cost more: a reserved board, a branch with experiment switches, or a simulator.

## Branch predictor off (diagnostic only)

When: a P2 move that follows the code address, or a P1 move that follows the previous kernel.

Switch: `LLK_TRISC_BP_OFF=<mask>` on the experiment branch `nstojictt/p58v2-versim` (not on
main). Bit 0 = unpack, bit 1 = math, bit 2 = pack; `4` = pack only, `7` = all three. The
same branch has the other no-work switches: `LLK_FN_NOPS=N`, `LLK_THREAD_FN_NOPS`,
`LLK_RELEASE_GAP`, `LLK_ZONE_RESERVE_NOPS`, `LLK_NO_PADS`, `LLK_NO_QUIET`,
`REPRO_PACK_RESYNC=N`. Read `test_config.py` on that branch for each one.

Read:
- The move goes away with the pack predictor off: branch aliasing in the pack loop (Wormhole).
- It goes away only with math off too: start timing through the branch-type cache (WH).
- Blackhole: predictor off did not remove the BH code-shift moves. BP is not the BH cause.

Pitfalls:
- With all predictors off, about 120,000 values change once, by up to 50 %. Never compare a
  BP-off build with a BP-on baseline; compare BP-off with BP-off.
- The disable bits persist until written. A kernel launch does not reset them. Clear them
  before the next arm.
- A debug halt of a TRISC (for example a register read through the debug tools) can turn its
  predictor off and runs code on that core. A state dump between two kernels therefore
  changes what the next kernel measures.

## Counters on / off

When: you need the reason for a slow state (who waits on whom), or someone asks whether
counters change the numbers.

```bash
pytest "${A[@]}" --compile-producer --enable-perf-counters -n 10 <files>     # producer must have counters too
PERF_RUN_TAG=cnt pytest "${A[@]}" --compile-consumer --dump-perf-counters --dist loadgroup -n 15 <files>
```

`--dump-perf-counters` implies `--enable-perf-counters` and writes a separate `.counters.csv`.
The L1 counters are in two mux groups: build once with `LLK_PERF_L1_MUX_GROUP=0` and once
with `=1`.

Read: equal grants with different request cycles mean the same work took longer; refused
cycles = requests − grants. Compare the two builds that give the two values.

Results 2026-10 (full suite, L1_TO_L1, on top of the stability stack): counters on vs off is
inside the noise floor on both arches, with no systematic shift (WH median 0.0000 %; BH
< 0.0005 %). The measuring pass is about 7× slower with counters.

## Board reset between tests

When: someone suspects state that only a chip reset clears.

Run the same node list twice in one session, then twice with `tt-smi -r` before every test
(one pytest call per test, `-n 0`). Compare all four with `perf_bitexact.py`.

Result 2026-10 (Blackhole, 120 tests of matmul / eltwise_binary / eltwise_unary_sfpu, non-SoL,
CI run 40677089303): 0 of 112 points differ in either pair; reset vs no reset 1 of 112,
< 0.005 %. On Blackhole a reset between tests does not change the numbers.

Pitfall: on silicon the host does not reset the Tensix between tests. Only the BRISC toggles
the TRISC soft reset before each kernel. A reset arm is a different harness.

## Versim (Wormhole) and VCS (Blackhole)

When: the card result is clear and repeatable but the cause is not. A simulator runs the same
ELF and gives the same cycles as the card, and shows every signal per cycle.

Order of work:
1. Calibrate: the simulator must give the card's cycles for both builds. If not, stop and fix
   the setup. Do not explain a simulator-only number.
2. Make the case small: one node id, one run type, a small loop factor.
3. Run both builds; diff the waveforms from the kernel release. The first cycle where they
   differ is the trigger; what follows is the mechanism.

Run: `TT_METAL_SIMULATOR=<simulator build> pytest --run-simulator ...` (alias
`TT_UMD_SIMULATOR_PATH`). Trees that use the UMD simulation server also need
`NNG_SOCKET_ADDR` and `NNG_SOCKET_LOCAL_PORT`. Ask the LLK perf infra team for the current
build and host; the WH and BH write-ups are on the team's Confluence pages about WH and BH
perf instability.

Signals that answered most cases: TRISC commit PC and mispredict, instruction-cache L1 reads,
TRISC L1 reads and writes, packer L1 write request/ready, the DEST read crossbar, and (BH) the
shared TRISC L1 port and the shared register block for wall-clock reads.

Pitfalls:
- Speed: a Wormhole matmul kernel took about 2 h on Versim; a BH test with production BRISC
  firmware 30–45 min on VCS. Pick a small config.
- A host reset in the simulator is a whole-grid reset; on silicon only the BRISC resets the
  TRISCs. Keep one session and the production BRISC firmware, or you test another harness.
- A test-only fast BRISC poll changed the results. Use the production firmware for mechanism
  work.
- Two simulator runs in one directory write the same VCD file. Use one working copy per run.
- The second kernel launch in one session crashed Versim with the 2026-10 harness port.
  Check that a two-kernel session works before you build an experiment on it.
- Keep results on shared storage, not in the container: reservations end and take `/tmp`.
