# Decode-only streaming projection integration snapshot

The standalone candidate `v6_corrected_fma_full_v5_streaming_scan_norm_tail_integration.py.snapshot` embeds the device-validated streaming helper into the existing terminal-scan/norm-tail decoder snapshot. It is an integration experiment, not a delivered production change or an integrated correctness/performance pass.

| Artifact | SHA256 |
|---|---|
| Frozen base `v6_corrected_fma_full_v5_terminal_scan_norm_tail.py.snapshot` | `99d2680b696bfbba57ceca9bd6597ae06daa217c062bf9289a25c9a10e53e5bd` |
| Candidate integration snapshot | `700c10db90b23fac1d40427c9bbcd82a6924c7704699d54768a0160c2b641e05` |
| Builder `tests/v6_build_streaming_scan_snapshot.py` | `cdcf3b8f15b7d508a1cede19587be75f777647a4181ba95c0397784b620f7a98` |
| Validated streaming helper | `b555f5caae383fb2cf1976860e92b9af3f1f2e2e39ec560e4c43d4d05bb23354` |

The parent-owned `v6_streaming_projection_scan_device.json` passed all 32 complete-adapter fixtures, all singleton full-product/partial-sum bit comparisons, multiple leading axes, and exact outputs after 100 trace replays. Compared with the earlier terminal-only component measurements, complete traced K512 adapters changed as follows:

| Adapter | Terminal-only ms | Streaming ms |
|---|---:|---:|
| Router | 0.060586 | 0.050378 |
| Attention output | 0.151061 | 0.148460 |
| Packed attention K512 | 0.188974 | 0.179659 |
| Packed Q/Z K512 | 0.186734 | 0.180624 |

K1024 streaming was slower than terminal-only, and eager timing regressed. The candidate therefore preserves the existing decode-only selection. These component measurements were collected separately and do not establish an integrated decoder gain. The parent must compare the new candidate against the best correct integrated baseline.

## Exact scope

Only two existing methods change:

- `from_state_dict`: within the existing linear-attention guard, replace the terminal helper instance with the streaming helper instance.
- `linear`: at the three existing `if terminal_scan` sites, replace the true arm with `self.streaming_projection_scan(weight, coefficients)`. Move the original materialized `ttnn.multiply` into the false arm, immediately before the unchanged native cumsum/crops. Router, attention output, and both packed projection keys are covered.

The `terminal_scan=False` signature default and all call sites are unchanged. Attention and output projections still receive `decode and cache_boundary`. Router receives `ordered_router`, whose sole public forward source remains `decode and cache_boundary`. Thus public singleton prefill, multirow prefill, internal K1024 tails, and calls with `cache_boundary=False` keep the native product/cumsum path. No new public sequence-length constraint is introduced.

The builder reverses these precise substitutions and setup change, then proves complete module AST equality with the frozen base before adding the new static helper definitions. It separately checks that the full-attention V5 guards in setup, linear, norm, MoE, forward, and prefill are identical. Every other existing method is AST-identical. Cache/state arithmetic, norm-tail fusion, output block reduction, casts, memory placement, and all full-attention V5 behavior are unchanged.

The new helper class is copied from the measured component and only its global constant/scheduling names are changed. An inverse rename proves AST equivalence. Kernel strings are embedded as static literals and checked byte-for-byte against the measured component's saved kernel snapshots. The already-embedded scheduling function is reused only after exact AST comparison with the measured helper dependency. The old terminal class remains unchanged as unused static source; its runtime instance is replaced.

The candidate contains no imports from model tests and no runtime AST, `exec`, `eval`, or `compile`. The runtime helper has no Torch arithmetic or tensor host conversion. It streams BF16 weight/coefficient tiles into FP32 FPU products and native cumsum, eliminating the full FP32 product tensor only on the guarded decode path. The helper's product-only diagnostic branch remains part of the alpha-equivalent static class, but no integrated call enables it.

## Build and provenance

Executed CPU-only command:

```bash
PYTHONPATH=. python_env/bin/python \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/v6_build_streaming_scan_snapshot.py
```

Result: `STREAMING_SCAN_STATIC_INTEGRATION_BUILD_PASS`, followed by the candidate hash above. Builder Python compilation also passes. Neither Torch nor TTNN was imported; no device was opened.

Artifacts under `doc/fused_decoder/endurance/` share the stem `v6_corrected_fma_full_v5_streaming_scan_norm_tail_integration`:

- `.py.snapshot`: standalone candidate module.
- `.provenance.json`: base/helper/evidence hashes; complete restoration checks; unchanged gates and full-attention guard digests; component timings; kernel hashes; builder provenance.
- `.builder.py.snapshot`: exact builder bytes used.
- `.reader.cpp.snapshot`, `.compute.cpp.snapshot`, `.writer.cpp.snapshot`, `.product_writer.cpp.snapshot`: exact embedded kernel strings.

The builder refuses to overwrite artifacts and verifies the base/helper hashes and all required component passes before emitting anything. Rebuilding for comparison can use `--output-dir <fresh-directory>`.

No production/shared files or frozen prior snapshots were changed. Parent-owned integration correctness, watcher run, repeated/traced behavior, and before/after decoder latency are still required. If the new integrated path fails a gate or does not beat the best correct candidate, it is not eligible for selection.
