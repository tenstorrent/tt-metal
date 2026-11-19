# Static decode terminal-scan and norm-tail candidates

These are source-checked diagnostic snapshots, not stage acceptance results. This builder agent used no hardware and imported neither Torch nor TTNN. Production and previously measured snapshots remain unchanged.

The common baseline is `v6_corrected_fma_explicit_norm_full_v5.py.snapshot`, SHA256 `52df1b042d445630ef9bdce0d100245fa2faad9dd801042dc638b24bf3fe50d9`. It retains the corrected chunk64 prefill arithmetic and the embedded original V5 full-attention runtime.

| Candidate | Snapshot | SHA256 |
|---|---|---|
| Norm tail only | `v6_corrected_fma_full_v5_norm_tail.py.snapshot` | `9362bfa3984efda2a63f383d12b419783a996831e2cff97aeb05a28ba25eaff1` |
| Terminal scan only | `v6_corrected_fma_full_v5_terminal_scan.py.snapshot` | `93bcedd5b1d44c52c0435cc856b98f4e68945f7f4d2caa9edba4e27c5f0c7adc` |
| Combined | `v6_corrected_fma_full_v5_terminal_scan_norm_tail.py.snapshot` | `99d2680b696bfbba57ceca9bd6597ae06daa217c062bf9289a25c9a10e53e5bd` |

Each snapshot has a matching `.provenance.json`, a `.py.snapshot.builder.py.snapshot`, and literal kernel source snapshots. Provenance pins the base, measured helpers, component evidence, generated source hashes, and affected method AST hashes.

## Scope and exactness checks

The norm-tail candidate embeds the measured `GatedNormTail` class with `debug=False` and a literal compute string. The only setup addition constructs its fixed epsilon/gamma constants for linear-attention layers. In public decode, native FP32 square/sum and the existing four-statement SiLU lookup remain unchanged; the generic operation replaces post-sum normalization, BF16 rounding boundaries and gate multiplication. Head packing and output projection are unchanged. The public one-token prefill false-cache-boundary branch retains the exact prior arithmetic. Removing the setup and replacement restores the complete baseline AST.

The terminal candidate embeds the measured `TerminalProjectionScan` and scheduling function with only collision-avoiding global renames; the reader, compute and writer are literal strings. It replaces three syntactic scan/crop sequences, covering router, output projection and both packed input projections. Separate FP32 product construction, native block reduction, cast and placement stay unchanged. Every replaced sequence has the exact prior sequence as its false branch.

An explicit `terminal_scan=False` keyword prevents inferring decode from K=512: public S=1 prefill also uses K=512. The two packed projections and decode output projection pass `decode and cache_boundary`; the router passes `ordered_router`, whose only corrected forward caller sets `decode and cache_boundary`. Internal prefill tails and public singleton prefill therefore keep their existing scan/crop paths. K=1024 prefill remains unchanged. The seven full-attention V5 early-return guards have identical AST hashes, including original fused setup, projection/router, norm and residual behavior. Removing only the recorded setup, keyword arguments, call keywords and three guarded scan replacements restores the full input module AST exactly.

The combined candidate applies the terminal transformation to the norm-tail snapshot. It retains that snapshot's embedded norm class and literal source without changes. No snapshot imports tests, compiles Python dynamically, or transfers runtime tensors to/from Torch. Generic operation descriptors and scheduling execute on the host as normal TTNN dispatch; tensor arithmetic and terminal-row relocation execute on device.

## Component evidence and selection limits

`v6_gated_norm_tail_fusion_device.json` and `v6_gated_norm_tail_timing_device.json` cover 14 exact cases. The complete norm adapter median improved from 0.528467567638 ms to 0.507525331341 ms in the recorded microbenchmark. This is not an integrated decoder result.

`v6_terminal_projection_scan_device.json` records 32 exact output fixtures, including two stacked prefill fixtures, 30 exact terminal-partial comparisons and the additional multiple-leading-axis fixture. Its traced complete projection adapters improved across tested shapes, while eager adapters worsened. Consequently these integrated diagnostics intentionally enable terminal scanning only for public decode. The eager prefill schedule is preserved even where the component was numerically exact. Full decoder warmed/traced performance, repeated correctness, watcher cleanliness, and acceptance remain for the parent device runs; the component gains alone do not select these snapshots.

## Reproduction

Run from the checkout root. Builders refuse to overwrite output evidence; use a fresh output directory for repeats.

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_build_norm_tail_snapshot --output /tmp/norm-tail.py.snapshot --report /tmp/norm-tail.provenance.json
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_build_terminal_scan_snapshots --output-dir /tmp/terminal-build
```

Create the fresh terminal output directory before running. The terminal builder deliberately pins the frozen repo-local norm-tail snapshot for its combined variant. Both builders run only standard-library source checks; compiling emitted Python validates syntax without executing imports or touching hardware.
