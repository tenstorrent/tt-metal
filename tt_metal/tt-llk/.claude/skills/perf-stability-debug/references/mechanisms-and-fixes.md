# Known mechanisms and fixes (status 2026-10-30)

Every row was measured on silicon, and most were confirmed in Versim (Wormhole) or VCS
(Blackhole). "Shows as" uses P1 (same build, other value) and P2 (no-work change moves it).

## Mechanisms

| # | Mechanism | Arch | Shows as | Size seen | Fix / status |
|---|---|---|---|---|---|
| 1 | **Per-core test order.** The value depends on the sequence of kernels that ran before it on the same core. `--maxschedchunk` 10 → 2000 (#53642, `fd8a72b6cbd`) made that sequence timing-dependent. | BH (and WH) | P1 | BH 1,354 L1_TO_L1 + 1,692 PACK_ISOLATE fires per shard | #57015 `--dist loadgroup` (merged 09-20); `pytest.ini` now `--maxschedchunk=200`. On the stability stack BH is repeatable without loadgroup. |
| 2 | **State left by the previous kernel in L1_TO_L1-only runs.** The gates measure only L1_TO_L1, so each measured kernel follows a kernel of another test. Exposed strongly by SFPI 7.90 code (#62327). Core identity does not matter. | WH | P1 | 696–1,976 TILE_LOOP points per matmul run, up to 12 % | Warm-ups `c75a4733ceb` + `be61600d8bb` (#58068, open): 0 of 108,366. Alternative: MATH_ISOLATE prelude (#61761, draft). The carried state is not identified. |
| 3 | **Branch predictor aliasing.** 16 untagged entries indexed by PC bits; two loop branches can share one entry and both mispredict every pass. | WH (pack loop) | P2 | +2.7 % on a tiny case; full suite +1 NOP 14,750 PACK_ISOLATE moves | #58068 layout pads + barrier restart avoid bad layouts; stage 2 labels the rest as placement. Predictor-off is diagnostic only. |
| 4 | **Branch-type cache with random replacement.** Which branch loses its entry depends on the start cycle. | WH (math loop) | P2 via start timing | MATH_ISOLATE +2.3 % on a case; 61 residual moves (max 7.2 %) in the full suite | #58068 BRISC release timing fixed; residual remains. |
| 5 | **Instruction-cache set conflict.** Three hot 16-B lines of a long loop map to one 2-way set and miss every pass. | WH (math) | P2 | up to +27 % (one matmul config) | #58068 layout pads. |
| 6 | **Packer rhythm.** Packers read DEST through a fixed-priority crossbar and lock into a fast or a slow rhythm for the whole loop; one L1 access by another core at the wrong cycle picks the slow one. | WH | P1 (random-cycle L1 traffic), P2 | 1–10 cycles per tile | Keep other cores off L1 while measuring: #61357 (host waits on overlay registers, BRISC polls every 100 µs on NOPs), #61358 (zone records after the closing clock read). L1_TO_L1 keeps some P2 by design. |
| 7 | **Host and BRISC polling L1 or the wall clock during a kernel.** | WH, BH | P1 | WH: 49 of 736 values, −7.7 % to +26.4 %; BH: INIT 176/179/180 run to run | #61357 (stack, open, approved). BH main + #61357 alone is bit-identical run to run. |
| 8 | **Profiler zone records and clock reads on shared paths.** BH: the three TRISCs share one L1 request port (records, data, code fetches) and one register block for wall-clock reads; a code shift in one thread changes which thread wins. | BH | P2 | INIT L1_TO_L1 up to +13.5 %; MATH_ISOLATE KERNEL +27 cycles | #61358 for WH-style records; BH experiments (records in TRISC local memory, `mcycle` timing for isolates) are not on main. |
| 9 | **BH packer L1-accumulate bistability.** With L1 accumulation the 8 packer ports lock into 1.0 or 1.5 cycles per beat. | BH (`perf_pack_dest_bank`) | P1 / P2 | 31 % | #61359 drain commit (stack, review required). A partial state up to 13 % remained in VCS on the stack, not seen on the card. |
| 10 | **Idle threads and mocks in isolate run types.** Peer threads' exit code and mock loops run inside the measured window. | WH | P2 | 54 values ±22–28 % | #58068 (hold peers, one-by-one release, mocks through the MOP expander). |
| 11 | **L1_CONGESTION run type.** Several arbitration states by design. | WH | P1, P2 | 1,300+ moves | Not fixable; not in the gates (they measure L1_TO_L1). |
| 12 | **LoFi matmul family.** Most throughput-bound configs; first to move under any of the above. | both | P1, P2 | 93 % of movers in the 2026-09 BH control | Check LoFi `perf_math_matmul` first; never treat its single-run moves as real without stage 2. |
| 13 | **SFPI bumps.** Change code generation broadly, therefore both work and layout, and the exposure to #2. | both | real + P2 | 7.90 WH matmul +38.5 %; 7.93 WH LoFi +13 % median (74 configs) | Pin SFPI on both sides to separate; stage 2 cannot run when base and head pin different SFPI. |
| 14 | **Assert builds.** Without `TT_LLK_DISABLE_ASSERTS=1`, INIT calls assert-only helpers. | both | false INIT moves | INIT ±1–8 cycles | Export the variable in every local run. |
| 15 | **Board and runner quality.** Some boards jitter even with a fixed order; CI runners are slower and show CI-only residual movers. | both | P1 | ~1,200 points on one bad WH board | Repeat on a second board; quote CI numbers for CI. |

## Ruled out (do not test again without new evidence)

- STALLWAIT for pack inside each test (#56807 part): no effect on the order effect.
- `run_count` > 1 and `min()`: the state binds once per run; `min == mean`.
- Per-worker core mapping (#53642's other half): not the cause; the scheduling chunk is.
- Wormhole TRISC branch-predictor contents as the carried state of #2: the TRISC reset clears
  them (Versim). SFPU lane-enable bits: equal in fast and slow cases.
- BH L1 data cache, BH branch predictor, BH THCON matmul commit (a speed change, not a
  stability fix), BH test order on the stack, BH board reset between tests.
- Counters on vs off: no systematic shift on either arch.

## Fix inventory

| Fix | Where | State 2026-10-30 |
|---|---|---|
| `--dist loadgroup` in the measuring pass | #57015 | merged 09-20 |
| `--maxschedchunk=200` | `pytest.ini` on main | merged |
| Restore missing uninits in tests | #56807 | merged 09-18 |
| Report keeps every chunk's rows; skip bcast configs with no block | #61356 | open, approved |
| Host and BRISC stay off L1 while a kernel runs | #61357 | open, approved |
| Zone records after the closing clock read | #61358 | open, approved |
| BH: unpack matmul base without THCON; no fast tile-end drain with L1 accumulation | #61359 | open, review required |
| WH: full faces on the straight path in `config_unpacker_x_end` | #61360 | open, review required |
| Barrier restart, layout pads, warm-ups, peer hold, INIT launch | #58068 | open, review required |
| MATH_ISOLATE prelude for WH gates | #61761 | draft |
| Stage 2: 5 code positions + loop fingerprint | #62981 (BH, report-only), #61761 | open |

Background issues: #55169 (matmul TILE_LOOP bistability is a placement effect), #59795
(Versim and silicon evaluation of WH instability), #59796 (instability fixes merged to main),
#62044 (stage 2 design), #63027 (stage-2 follow-ups).
