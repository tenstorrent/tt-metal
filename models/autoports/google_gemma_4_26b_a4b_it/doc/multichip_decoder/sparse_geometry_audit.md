# Optional indexed-expert geometry candidates

CPU-only preparation following the historical native sparse-matmul advice. [The runner patch](sparse_geometry_runner.patch) is unapplied and changes no model defaults. It exposes independent gate/up and down overrides after construction, before cache setup or tracing. No hardware was used; no live runtime or runner was edited.

[Provenance](sparse_geometry_provenance.json) records runner base `06f0a0573dbd14e852be4a27c88b72faa610a51961b90cf33365025747898581` and candidate `c6c55f39bf683cc33f1955145fb0fc23081f2cbc5c32cfe20c17c9edfa687049`. The runtime reference is `8b59370cda6f4ff88157de123123509036f2e91e8054000c809752e21f933175`. These bases already contain the proposed selected defaults and optional attention DRAM backend. [Complete runner candidate](sparse_geometry_runner_candidate.py.txt) is preserved. Recheck patch applicability if the parent changes policy defaults.

| Candidate | Flag | Local K,N tiles | Worker grid | K block | Per-worker N / output block / subblock |
| --- | --- | --- | --- | ---: | --- |
| Current gate/up | no override | 88,12 | 6x2 = 12 workers | 44 | 1 /1 /1 |
| Gate/up N2 K44 | `--sparse-gate-geometry n2-k44` | 88,12 | 6x1 = 6 workers | 44 | 2 /2 /2 |
| Gate/up N2 K88 | `--sparse-gate-geometry n2-k88` | 88,12 | 6x1 = 6 workers | 88 | 2 /2 /2 |
| Current down | no override | 6,88 | 11x8 = 88 workers | 6 | 1 /1 /1 |
| Down N2 K6 | `--sparse-down-geometry n2-k6` | 6,88 | 11x4 = 44 workers | 6 | 2 /2 /2 |

Both flags default to `baseline`. The runner can combine an explicitly selected gate candidate and down candidate, but first measure each role independently and combine only passing winners. EP-only experts are rejected before device setup; TP-only and hybrid TP decode are supported. In a hybrid layer the override resolves `experts.decode`, leaving the separate EP prefill object intact.

The guard requires the existing packed indexed top 8 path and actual local matrices `[1,128,2816,384]` and `[1,128,192,2816]`. Only `gate_config` and/or `down_config` is replaced. Weight objects and dtypes, activation precision, compute fidelity, FP32/packer flags, GELU, routing/indices, expert mixing, prefill configs and collectives remain unchanged. The candidate preserves current LoFi decode with FP32 destination accumulation off and packer accumulation off. Gate weights remain BFP8 sliding/BFP4 full; down remains BFP4. No dense all-expert fallback or fresh weight packing is introduced. The cache page-block variable remains 32 independently of gate K44/K88.

Source support:

- `ttnn/cpp/ttnn/operations/matmul/device/sparse/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp:60-82` extracts the existing 1D program config and uses compact indexed expert slots. This proposal stays within that supported backend.
- The factory at `:158-180` requires tiled K divisible by the K block, sufficient workers for `ceil(M/per_M)*ceil(N/per_N)`, and `mcast_in0=True`. With physical M1 tile, the candidates need exactly 6 or 44 workers and divide 88 by 44/88 or 6 by 6.
- `matmul/device/sparse/sparse_matmul_device_operation.cpp:353-392` requires output subblock to divide output block, and output block to divide per-core dimensions. Explicit N2/block2/subblock2 satisfies those constraints; M dimensions remain 1. `fuse_batch=False` preserves sparse group handling.
- `tt/optimized_decoder.py:172-237` supplies eight router indices, compact slots and unchanged weights/compute to the two sparse matmuls. Overriding their program attributes does not alter expert selection.

Wider blocks trade fewer worker cores and K-loop iterations for more work/buffer space per core. From the factory's double-buffer sizing, gate K44 primary A/B buffers are approximately 180224/191488 bytes for BF16 A/BFP8 B; K88 doubles these to 360448/382976. BFP4 gate B uses 101376/202752 bytes instead. Down uses 24576 bytes A and 13824 bytes B. These exclude output, intermediate, multicast/runtime allocations and other live tensors; they are not an allocator-peak proof or a speed claim. K88 has one K block, but the current requested packer accumulation is already off.

[CPU checks](sparse_geometry_cpu_checks.json), with [preserved source](sparse_geometry_cpu_checks.py.txt), execute the extracted override against stand-in objects for baseline, each individual candidate and both explicit combinations. They verify geometry/divisibility, unchanged weight/compute/router/mix/prefill object identity and unchanged cache page-block value. The complete candidate parses, and `git apply --check` passes against the recorded base. TTNN was not imported; numerical correctness and actual allocation remain unverified.

After final native profiles identify the role as material, use the same selected precision/CCL policy, actual layer 0/layer 5 weights and paired 4096/128 traced cache-check harness as the control. Record a fresh control and each individual candidate before combining winners. Compare complete-layer timing, not isolated sparse op time alone. Preserve active 8 metadata in any native profiles, then rerun final stack/batch/capacity/Watcher gates if a candidate is selected. No default change or final acceptance follows from this source preparation.
