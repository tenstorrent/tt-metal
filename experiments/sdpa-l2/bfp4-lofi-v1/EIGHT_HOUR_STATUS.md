# Eight-hour continuation

- Requested window starts: **2026-09-15 05:48:43 UTC**.
- Required end: **2026-09-15 13:48:43 UTC** (eight hours).
- User explicitly asked to keep working until the window ends.
- Worktree: `/Users/cglagovich/dev/tt-metal-blackhole`.
- Do not modify the four pinned selected variants.
- Initial device: yyzo-bh-08, reservation220027. Remote worktree:
  `/localdev/cglagovich/tt-metal-blackhole-20260908`.
- Access: SSH via yyz-ird to host yyzo-bh-08, then `docker exec` as
  cglagovich in `yyzo-bh-08-special-cglagovich-for-reservation-220027`.
- JIT cache: experiment-local `.jit-cache/`, not NFS home cache.

## Working sequence

1. Wider-storage LoFi controls, right-operand pre-rounding, and BFP4 residual
   quantization simulations; verify promising configurations on device.
2. Device-side quantization/preprocessing probes and cost measurement.
3. Isolated fused/resident streaming prototypes for numerical candidates
   worth optimizing; keep Q/K sizes and timing contracts explicit.
4. Broader distributions/seeds, source-pinned reproducibility, measured
   performance comparisons, and a final research report.

## Progress at 06:23 UTC (~35 minutes elapsed)

- Reservation220027 extended by successful `ird change-timeout 3 09:30:00`.
- Isolated v2 numerical models: 640 records each for host/device QKV packing.
  One-pass per-value Q7/KV5 LoFi is about 2% normal L2; residual BFP4+BFP8
  with per-value Q/P is about 0.58% but doubles QK/PV matmul counts.
- Fully fused Q128/K512 FP32 LoFi agrees with model (~2% normal, through256K).
  BFP8 Q needed explicit SrcB reconfiguration after BF16 max usage; fixed only
  in a private experimental header. Four locked modes untouched.
- Resident Q256/K512: LoFi FP32 cheap1.039TF/core, BF16 compensated1.890,
  BF16 uncompensated2.495. CPU rounding excluded from these measurements.
- Distinct-input64-case suite completed. BF16 compensated ~4% normal;
  uncompensated LoFi drifts drastically at256K. Testing HiFi2 recurrent
  broadcast multiplies while keeping LoFi QK/PV to isolate this effect.
- Device-side RNE rounding kernel compiles/runs; first524288 values exact.
  Exhaustive ordinary BF16 patterns and110-core bandwidth tests completed;
  gathering results. Implementation in sibling `bfp4-lofi-v2/`.

Required end remains13:48:43 UTC; goal remains active.

## Progress at06:50 UTC (~one hour elapsed)

- All four selected Q256 resident controls remeasured on current allocation:
  main1.99409, FAST1.59144, balanced0.935189, accurate0.759834TF/core;
  effectively identical to locked checkpoint.
- LoFi compensated BF16 + safeHiFi2 state/output multiply:1.89088TF/core,
  32Knormal2.9566%L2,256K3.5995%. BFP8K/V works too:1.87777TF/core,
  32K3.048%L2, preserving two input buffers and fixed Q/Kchunk sizes.
- Rounding preprocessing now passes524288 values ×4wide-exponent checks
  exactly after clearing incompatible source-zero flag inFP32DSTdatacopy.
- Fused BFP4+BFP8 residual prototype:4Knormal0.6691%L2,
  PCC0.9999776. Resident0.609TF/core with explicitProunding; without that
  expensive pass0.876TF/core and~0.681%L2. Optimizations still needed.
- V-centering can create large mean quantization bias on discretized BF16
  inputs. Model correction using original-minus-representedVmean repairs
  this; a fused FP32 output-bias epilogue is being tested.
- Intermediate notebook: sibling bfp4-lofi-v2/PROGRESS.md.

## Progress at08:05 UTC (~2h16m elapsed)

- A production-style per-head unicast KV chain removes much of the redundant
  input traffic in the experimental fullchip harness. AtH10/256K/Q256/K512,
  LoFi compensated BF16+BFP8 KV reaches190.63TF/chip including preprocessing,
  L2=3.735%; matching mainBF16 control153.22TF/chip/L2=18.781%.
- Fully device-prepared BFP4+BFP8 residuals reach0.59–0.60%L2; Q prescale1.0028
  lowers Q128 tests to0.503–0.524%. This is near, not reliably below,0.5%.
- Residual fullchip chain passes all-output1024/heads2/cores6 tests including
  unequal per-core work; larger performance runs are active.
- Device fused centering passes exact normal/common/threshold/wide/zero tests;
  device K mean producer passes BF16-FPU and FP32-SFPU controls. Full mean+center
  atH10/256K costs6.31ms/13.72ms respectively, included in combined timing.
- Frozen FAST Q256 distinct-input recurrence has a stale address-modifier bug
  hidden by previous Q128 and resident coverage. Private reset repairs it;
  report in v2/FAST_Q256_CORRECTION_BUG.md. Pinned four sources untouched.
- Native BFP8 ties-away causes measurable positive magnitude bias. CPU model
  attributes about0.5–0.6% output gain to it; device RNE pre-rounding is queued.
- Scalar native BFP packing exactness passes; default width4 BFP pack framing
  fails and is now explicitly rejected. Separate compressed-P8 FP32 prototype
  and padded-BFP page-stride probe are under development.

Continue until13:48:43 UTC. More than five hours remain; do not stop early.

## Progress at08:50 UTC (~three hours elapsed)

- Native BFP8 attribution is resolved: generic packing defaults to an extra
  per-value E8M6 ties-away rounding. `bfp8_pack_precise=True` removes it;
  high-level typecast already sets this. Independent pipeline oracles pass.
- Residual4+8 Q-prescaled fullchip reaches~0.515% L2 at~70.5 useful TF/chip,
  not a Pareto improvement over balanced. Compressed P8 and BF16-compute/FP32-
  state hybrid also remain dominated after optimization and qualification.
- FP32 quadratic exp improves resident1.0347→1.0865TF/core with L2
  2.137→2.153%. Linear reaches1.1438TF/core but2.881% L2. BF16 FAST already
  uses native exp, so changing this refiner does not apply to it.
- All twelve independent K/V BFP4/BFP8 × destination smoke configurations
  pass all-output1024 checks and exact device input quantization oracles.
  A denominator-only compensated BF16 control passes too. Longer timing
  comparisons are active, including the previously missing LoFi MAIN+BFP8.
- Testing device Q-centering with high-precision mean-Q/original-K score
  correction; producer JIT compiles and centered-Q is exact, but correction
  has not yet passed its initially proposed0.01% gate. Investigate, don't
  claim qualification. Integrated prototype is being designed separately.
- Four pinned modes still untouched; only the two research directories are
  untracked. No research checkpoint commit yet.

Continue until13:48:43 UTC. About five hours remain.
