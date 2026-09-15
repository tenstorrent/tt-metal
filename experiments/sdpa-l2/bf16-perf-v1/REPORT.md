# BF16 streaming SDPA: reduce compensation-loop overhead

## Result

The independently rebuilt retained candidate runs in **2842.76 ms**, versus **2936.15 ms** for the previous compensated BF16 implementation. Relative L2 remains **3.189697%**. A fresh unmodified-main-base control runs in **2788.95 ms**.

| Implementation | Median ms | Relative L2 % | Overhead vs main |
|---|---:|---:|---:|
| Unmodified main base | 2788.95 | 18.964064 | — |
| Previous BF16 compensation | 2936.15 | 3.189697 | 5.28% |
| Retained replay-only compensation | **2842.76** | **3.189697** | **1.93%** |

This saves **93.40 ms / 3.18% of total time**, reducing the added overhead from **147.20 ms to 53.80 ms**, or **63.45%**. This preserves the previous BF16 accuracy improvement; it does not bring BF16 L2 below 0.5%.

Raw controls: [start-full.jsonl](start-full.jsonl), [main-forced.jsonl](main-forced.jsonl), [final-forced.jsonl](final-forced.jsonl). Final measured replays span **2839.15–2848.78 ms**; main spans **2780.42–2791.42 ms**. An earlier independent run of the same scheduling algorithm measured **2841.58 ms**, with the same sampled-output hash and L2 ([replay-full.jsonl](replay-full.jsonl)).

## Fixed configuration and measurement contract

Blackhole P100A on **yyzo-bh-26**, reservation **215262**, 110 compute cores (11x10). This is not a Galaxy measurement. Main base is commit `2ba6fc2339d53300ae87c5202f335ef56492cfb3`; “main” here means that original unmodified base, not a newly fetched revision.

Noncausal **B=1, H=10, Q=K=262144, D=128**, BF16 inputs/output and destination, HiFi2 QK/PV, approximate-math/exp settings. **Q chunk stays 128, K chunk stays 512. All input double-buffering settings, CB allocations, state-tile copies/packs, and reader/writer code are unchanged from the previous compensated implementation.** No changes were made to the FP32 algorithm.

Times are median **blocking trace-replay wall times**, 40 warmups and 10 measured replays, excluding compilation, transfers, reference calculation, and host preprocessing. They are not newly collected device-counter latencies or FPU-utilization measurements. Clock/power settings were not changed.

All three timing controls retain the historical Q preprocessing recipe (six-bit bit-ceil, scale compensation 1.0027). Reference uses the original BF16 Q/K/V in FP64, on 128 tail query rows per head, not every output row. The final host preprocessing took 106.54 ms and is excluded from trace timing. The correctness sweep additionally tests untouched Q.

## Retained change

Only `calculate_sdpa_compensated_state` in [ckernel_sfpu_sdpa.h](../../../tt_metal/hw/ckernels/blackhole/metal/llk_api/experimental/llk_sfpu/ckernel_sfpu_sdpa.h) changes relative to the prior implementation: **one header, 52 insertions / 15 deletions**.

The original compiled loop serializes each pair's SFPU load/add/MAD/round/subtract/round/store chain. The replacement:

- Interleaves two independent compensated-state chains to hide dependent-use latency.
- Records an 11-instruction (one pair) or 21-instruction (two pairs) program in Blackhole's per-thread replay buffer; executes the first vector while recording and replays the other 31.
- Uses the final store's address modifier to advance DST and unrolls replay issuance.

It retains the same update and nearest-conversion instructions:
`sum = (hi + lo) * correction + chunk; hi = bf16(sum); lo = bf16(sum - hi)`.

No exp coefficients, fidelity, precision terms, or Q preprocessing were changed. Both numerator and denominator compensation remain. Replay belongs to the PACK thread after the logit-exp phase; the next K chunk's existing exp initialization restores the fast-exp replay program.

The host program factory, compute_common.hpp, and compute_streaming.hpp are byte-identical to the pre-task snapshots. The existing extra compensation-state storage is unchanged.

## Correctness and verification

**All 22 retained-version cases have exactly the same sampled-output hashes and L2 as the pre-optimization implementation.** See [final-output-comparison.jsonl](final-output-comparison.jsonl). The separately tested hoisted-replay experiment also matched all 22 cases.

With untouched Q, the spread-sampled normal-input L2 remains **3.162034%**. The FP32 regression check remains **0.487625%**, with an identical sampled-output hash. Stress-distribution errors are preserved, not newly solved by this scheduling optimization.

The matched suite includes three normal seeds, seven spread-sampled distributions with untouched Q, Q/K/V common-mode offsets +8/+32, 32K/64K/128K contexts, a causal case, a D=64 fallback, and the previously improved FP32 streaming path. Each result records a SHA-256 of the sampled output. The repro also checks finite values, its FP64 reference implementation, and ordinary-versus-traced sampled-output equality.

[validate.sh](validate.sh) runs the suite; [compare_outputs.py](compare_outputs.py) checks matching input geometry and exact sampled-output hashes, as well as unchanged L2. These are sampled-output guarantees, not exhaustive equality over every 256K output element or arbitrary input distributions.

The host build and device JIT succeeded. Main and final controls each used a separate fresh kernel cache and an explicitly touched host translation unit before rebuilding; both build logs show the transformer object being compiled. This avoids the stale-object issue from earlier snapshot restores.

Checks: changed C++ hunks match clang-format 19.1.4; Python passes Black 23.10.1 and syntax compilation; shell scripts pass syntax checks; `git diff --check` and reverse-apply checks for both patches pass. [source-sha256.txt](source-sha256.txt) matches local and remote sources.

## Other experiments

Short-Q screening uses Q=1408, K=262144, H=10, the same 128/512 chunks and all 110 cores, with 2000 warmups / 20 measured replays. It does not perfectly predict sustained full-256K performance.

| Variant | Screening ms | Full-256K ms | Decision |
|---|---:|---:|---|
| SFPI outer-loop unroll 4 | 14.661 | Not measured | Superseded by replay |
| Replay-only | 14.353 | 2841.58; final 2842.76 | Retained |
| Replay + unchanged-max shortcut | 14.376 | Not measured | No screening benefit |
| Cached replay programs | 14.211 | 2849.78 | No full-shape benefit |
| Cached replay + hoisted address setup | 14.205 | 2841.17 | Within variation of simpler replay-only |

The cached/hoisted variants retained the exact sampled output but added setup-lifetime coupling without a convincing full-shape win. The initial cached-replay attempt used incorrect replay-control flags and produced zero output (100% L2); `cached-proxy` is an invalid timing/accuracy candidate. The corrected results are labeled `cached-fixed-proxy` and `cached-full`.

## Reproduce and handoff

From the configured, built remote worktree:

```bash
bash experiments/sdpa-l2/bf16-perf-v1/measure.sh fresh-label
bash experiments/sdpa-l2/bf16-perf-v1/validate.sh fresh-label
python_env/bin/python experiments/sdpa-l2/bf16-perf-v1/compare_outputs.py before fresh-label
```

Use fresh labels; measurement scripts refuse to overwrite existing result files. The final isolated JIT cache is `/localdev/cglagovich/sdpa-bf16-final-cache.TloRvs`.

[incremental-bf16.patch](incremental-bf16.patch) is the one-header delta from the previous retained implementation. [final-candidate.patch](final-candidate.patch) includes all prior FP32/BF16 accuracy work plus this change relative to the main base. No commits were made.

Remote checkout: `/localdev/cglagovich/tt-metal-blackhole-20260908`. The machine is left on the retained candidate, built and tested, not on main or a rejected variant. Raw logs and results have been copied to this local report directory. Reservation 215262 on yyzo-bh-26 uses SSH port 42837 and expires around **2026-09-10 17:13 Eastern / 21:13 UTC**, unless extended separately.
