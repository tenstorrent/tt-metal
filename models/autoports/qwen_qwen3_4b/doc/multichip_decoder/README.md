# Qwen3-4B Multichip Decoder

This stage adds `MultichipDecoder` for `Qwen/Qwen3-4B` under
`models/autoports/qwen_qwen3_4b/tt/multichip_decoder.py`. It uses the
single-chip `OptimizedDecoder` as the numerical baseline and targets the four
local Blackhole p150b devices as a 1x4 tensor-parallel ring.

## Mesh Plan

Chosen target: `MeshShape(1, 4)` with `FabricConfig.FABRIC_1D_RING`, TP=4, one
fabric link, ring collectives on `cluster_axis=1`.

Per-device Qwen3-4B shapes:

| Tensor or state | Global logical shape | Per-device shape | Strategy |
| --- | ---: | ---: | --- |
| Layer input/output | `[1, 1, S, 2560]` | `[1, 1, S, 2560]` | Replicated hidden state |
| Q projection | `[4096, 2560]` | local Q width `1024` | Packed into QKV column-parallel shard |
| K projection | `[1024, 2560]` | local K width `256` | Packed into QKV column-parallel shard |
| V projection | `[1024, 2560]` | local V width `256` | Packed into QKV column-parallel shard |
| Packed QKV prefill | `[2560, 6144]` | `[2560, 1536]` | Sharded on output dim, packed `[V, Q, K]` per device |
| Packed QKV decode | `[2560, 6144]` | `[2560, 1536]` | Sharded on output dim, packed `[Q, K, V]` per device |
| Attention heads | Q=32, KV=8 | Q=8, KV=2 | Local SDPA/paged SDPA per device |
| O projection | `[2560, 4096]` | `[2560, 1024]` | Row parallel; ring all-reduce to hidden |
| Gate/up projection | `[9728, 2560]` each | `[2432, 2560]` each | Column parallel |
| Down projection | `[2560, 9728]` | `[2560, 2432]` | Row parallel; ring all-reduce to hidden |
| KV cache per K or V | `[2560, 8, 16, 128]` across TP | `[2560, 2, 16, 128]` | Local KV heads, replicated tensor object across mesh |
| Page table | `[1, 2560]` | `[1, 2560]` | Replicated |
| Current positions | `[batch]` | `[batch]` | Replicated |
| RoPE and masks | Context-sized | Context-sized | Replicated |

The default multichip KV config is `max_num_blocks=2560`, `block_size=16`,
`cache_dtype=BF16`, so the decoder-layer KV capacity is the full HF-advertised
context of `40960`. Per device, the full-layer-stack KV expectation is:

- One K cache plus one V cache per layer: `2 * 2560 * 2 * 16 * 128 * 2` bytes =
  `41,943,040` bytes.
- Full 36-layer decoder stack: `1,509,949,440` bytes per device for KV cache.

Qwen3-4B is a dense model, not MoE, so there is no expert routing strategy or
active-expert execution gate for this stage.

## Rejected Alternatives

`1x2` TP was rejected because the local machine exposes four Blackhole devices
and Q/KV/MLP dimensions divide cleanly by four. It would leave half the mesh
unused and double local KV heads and intermediate width.

`TP>4` was rejected because this host has four local devices.

`2D` and Galaxy-style plans were rejected because the available mesh is a local
four-chip ring and the model dimensions map naturally to 1D tensor parallelism.

Reduce-scatter-only residual flow was not used in the final code. The next
consumer after each row-parallel projection is RMSNorm over the full hidden
dimension, and the optimized single-chip baseline provides a full-hidden RMSNorm
contract. Keeping layer input/output replicated is the safe stacked-decoder
contract for this stage. The current all-reduce lowers to reduce-scatter plus
all-gather in the profiler, so the communication cost is visible in the
artifacts for a future distributed-RMSNorm optimization.

Topology/adapted-evidence table:

| Candidate | Measured or source evidence | Decision |
| --- | --- | --- |
| Replicated residual with `ttnn.all_reduce` after WO/down | Implemented path. `tt-perf-report` shows each all-reduce lowers to ReduceScatter plus AllGather. CCL costs are `77.639 us` prefill and `79.161 us` traced decode, about `19%` of the measured device time for the layer. | Selected because it preserves the optimized decoder's full-hidden RMSNorm and layer stacking contract. |
| Reduce-scatter-only row-parallel outputs | Shape-faithful probe command below produced local hidden shape `[1, 1, 16, 640]`. Full residual add failed with `Invalid subtile broadcast type`; full-width RMSNorm failed because input width `640` did not match gamma width `2560`. | Rejected for this stage because it cannot cross the next residual/RMSNorm boundary without changing layer input ownership. |
| Distributed RMSNorm helper path | The same probe showed a local-width RMSNorm can produce `[1, 1, 16, 640]` only with sharded norm weight, but the next gate/up matmul against current column-parallel weights failed with `width=640 height=2560`. Repo helpers in `models/tt_transformers/tt/ccl.py` also require TT_CCL semaphore/persistent-buffer plumbing. | Not adopted in the repo-local autoport decoder because it would require redesigning the next MLP input contract or adding a gather before MLP, which gives back the movement this stage is trying to avoid. |
| Keep residual sharded across the whole stack | Could be a future full-stack design if embeddings, every decoder layer, final norm, and logits all accept width-sharded hidden states. | Rejected for the multichip-decoder stage because the user explicitly scoped this stage to a decoder-layer baseline, not full-model stack redesign. |

Reduce-scatter residual probe:

```bash
QWEN3_4B_MULTICHIP_RUN_RS_PROBE=1 \
pytest -q -s models/autoports/qwen_qwen3_4b/tests/test_multichip_decoder.py::test_multichip_reduce_scatter_residual_contract_probe --tb=short
```

Result: `1 passed in 10.03s`. The saved log is
`reduce_scatter_residual_probe.log` and records:

- `reduce_scatter_local_shape=(1, 1, 16, 640)`
- full residual add shape/broadcast failure
- full-width RMSNorm gamma width `2560` vs input width `640` failure
- local-width RMSNorm success with shape `(1, 1, 16, 640)`
- gate/up matmul failure with input width `640` vs weight height `2560`

Memory estimates recorded in `doc/context_contract.json`:

| Item | Per-device bytes |
| --- | ---: |
| KV cache per layer, K+V, 40960 context | `41,943,040` |
| KV cache for 36 decoder layers | `1,509,949,440` |
| Decoder weights per layer, BF4 projections plus BF16 replicated norms | `12,626,432` |
| Decoder weights for 36 layers | `454,551,552` |
| Trace region used in tests | `67,108,864` |
| Representative seq-16 activation payload estimate | `578,560` |

## Correctness

Main command:

```bash
pytest -q -s models/autoports/qwen_qwen3_4b/tests/test_multichip_decoder.py --tb=short
```

Result: `8 passed, 3 skipped in 74.56s`.

PCC against the single-chip optimized baseline:

| Case | PCC |
| --- | ---: |
| Prefill seq 16 | `0.9997333805764328` |
| Prefill seq 17 | `0.9997359676733848` |
| Prefill seq 64 | `0.999684148069019` |
| Paged decode prefix 16 | `0.9980224018654454` |
| Paged decode prefix 17 | `0.9978650895682959` |
| Trace replay | `1.0` |
| Inter-device output replication | `1.0` |

The tests validate:

- Single-chip optimized baseline comparison for prefill and paged decode.
- Non-aligned logical lengths (`seq_len=17`, `decode_prefix=17`).
- Local KV cache layout: `[2560, 2, 16, 128]` per K or V cache tensor.
- Replicated page table `[1, 2560]` and current position `[1]`.
- Traced decode replay on the target mesh.
- No host fallback in `prefill_forward` or `decode_forward`.

## Performance

Uninstrumented warmed host timings:

| Mode | Single-chip ms | Multichip ms | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: |
| Prefill seq 16 | `1.743231` | `2.514297` | `0.693327` | `0.173332` |
| Traced decode at pos 16 | `0.504047` | `0.440318` | `1.144734` | `0.286184` |

Tracy/profiler timings:

| Mode | Single-chip ms | Multichip ms | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: |
| Prefill seq 16 | `1.818641` | `2.913875` | `0.624131` | `0.156033` |
| Traced decode at pos 16 | `0.537467` | `0.477777` | `1.124933` | `0.281233` |

`tt-perf-report` artifacts:

- `tt_perf_report_prefill.txt`
- `tt_perf_report_prefill.csv`
- `tt_perf_report_prefill_stacked.csv`
- `tt_perf_report_prefill_stacked.png`
- `tt_perf_report_traced_decode.txt`
- `tt_perf_report_traced_decode.csv`
- `tt_perf_report_traced_decode_stacked.csv`
- `tt_perf_report_traced_decode_stacked.png`

Profiler summary:

| Mode | Modeled DRAM roofline | Matmul time | CCL time |
| --- | ---: | ---: | ---: |
| Prefill | `6.7%`, `34 GB/s` | `213.678 us` | `77.639 us` |
| Traced decode | `6.8%`, `35 GB/s` | `211.261 us` | `79.161 us` |

The tiny seq-16 prefill case is slower on TP4 because two row-parallel
all-reduces per layer dominate the small compute payload. Traced decode is
modestly faster than the warmed single-chip optimized baseline.

## Watcher

Default full-Ethernet watcher coverage currently exposes fabric/watcher issues
outside the Qwen decoder code:

- Without `TT_METAL_WATCHER_NOINLINE=1`, 1x4 fabric initialization fails because
  active Ethernet watcher instrumentation makes the program too large for the
  kernel config buffer.
- With `TT_METAL_WATCHER_NOINLINE=1`, the model stress passes, then active
  Ethernet watcher trips in `fabric_erisc_router.cpp` packet-tag sanitization.
- Disabling only NOC sanitization still leaves an active Ethernet teardown
  timeout.

Strongest passing watcher command for this stage:

```bash
TT_METAL_WATCHER=10 \
TT_METAL_WATCHER_NOINLINE=1 \
TT_METAL_WATCHER_DISABLE_ETH=1 \
QWEN3_4B_MULTICHIP_RUN_WATCHER_STRESS=1 \
pytest -q -s models/autoports/qwen_qwen3_4b/tests/test_multichip_decoder.py::test_multichip_watcher_single_mesh_stress --tb=short
```

Result: `1 passed in 73.51s`. This keeps worker-side watcher coverage enabled
while disabling only active Ethernet watcher instrumentation. The stress keeps a
single TP4 mesh open and covers prefill, non-aligned prefill, paged decode,
KV-cache layout, and traced decode.

See `AUTOFIX.md` for the fabric-router diagnosis and saved watcher logs.
