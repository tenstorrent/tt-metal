# sampling_kernel.cpp (deepseek_v3_b1 micro_ops/sampling) — MIGRATED (API v11)

- Unit: Tier 1.6, `deepseek-b1-sampling-loop-barrier`
- Production commit: `2840fc2836177691fdc72d0d0f7ad93a72451694`
- Status: migrated at API v11 on 2026-08-16

## Implementation

The single-device NCRISC loop barrier now uses one helper-owned, signal-only `Mcast2D` Flag pipe with a
fixed final-core sender and no handshake. The host constructs the dense bounding rectangle of the sparse
101-core shard grid, so the Blackhole 11x10 route retains the raw 109-destination `EXCLUDE_SOURCE`
multicast. The helper semaphore is initialized on all 110 landed cells; kernel descriptors and per-core
runtime arguments remain restricted to the 101 active cores. Mesh mode and the operation-owned global
sampling semaphores are unchanged.

The migration removes the hand-written NoC coordinate swap, physical-coordinate conversion, destination
count, raw semaphore multicast/wait/reset sequence, and five named compile-time arguments. Per-file
production shrink gate: kernel 36 deletions / 9 additions; host 22 deletions / 10 additions.

## Correctness and cache evidence

- `./build_metal.sh`: passed.
- Cold `run_safe_pytest.sh --dev --no-precompile` 101-core argmax: passed, 0/523 JIT hits.
- Complete mapped normal selection: 4 argmax passed; 3 top-k retained their pre-existing Blackhole
  selection-mismatch skips; warm precompile/run reported 533/533 JIT hits.
- Temporarily unskipped `test_sampling_topk_single_device[test_1]` completed all 100 internal iterations,
  selected expected index 85, and failed only at the known metadata `p_scores` assertion.
- The raw implementation, rebuilt from an empty cache under the same unskipped node, produced identical
  `p_indices`, identical `p_scores`, selected the same index, and failed at the same assertion.
- The temporary test edit was restored; the test file has no diff.
- `test_mcast_pipe_source_audit.py`: 17 passed.

## Matched performance

Tracy `DEVICE KERNEL DURATION [ns]` on the same Blackhole:

| Node | Raw | API v11 | Delta |
|---|---:|---:|---:|
| argmax 101-core `[2005-100]` | 18,789 | 18,836 | +0.25% |
| top-k `test_1`, 100 iterations | 1,558,235 | 1,557,464 | -0.05% |

## Claude review

Claude approved the fixed-sender/no-handshake Flag formulation, dense 11x10 topology, NCRISC runtime
offsets, and sender-local Flag behavior with `API_EXPANSION NO`. Claude's KEEP verdict for the skipped
top-k route required an exact raw failure-signature match; that condition was satisfied. Later broad and
compact final-verdict retries returned no output and were terminated, so no approval was inferred from
those timeouts.
