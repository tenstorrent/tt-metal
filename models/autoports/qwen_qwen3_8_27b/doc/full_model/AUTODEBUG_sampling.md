# AutoDebug: Stage 6 common sampling integration

Investigation date: 2026-09-12. Source-only AutoFix diagnosis in an isolated agent context. No implementation changes, target execution, device access, or performance measurements were made. The reported failure was the `Sampling1D._sample_topk` strategy call; no original failing command/log was supplied to this diagnosis.

## Findings and recommendation

`Sampling1D` has a verified incomplete local-index change that prevents **both** top-k strategies from running. `TTSampling` avoids these errors and supplies better greedy tie handling and request integration, but its current wrapper assumes 32 sampling rows. Neither path directly accepts a persistent output shaped `[1,1,B,1]` for B > 1. The native sampling output `[1,1,1,B]` is also the shape that row-major embedding accepts directly.

For the smallest unmodified common path, use `SamplingGenerator` / `TTSampling` with 32 sampling rows, 32 candidates per TP shard, force-argmax disabled, and native `[1,1,1,32]` output. The model wrapper must adapt its active B rows to this contract. For actual B=1/B=8 sampling, the smallest Python crash repair is restoring `Sampling1D`'s original implicit-index top-k strategy; this requires separately supplying the request/trace state and deciding how to preserve the stronger greedy tie handling from `TTSampling`. Do not hide a B=32 expansion behind a B=8 output buffer.

### Verified Python failures

Source: `models/common/modules/sampling/sampling_1d.py`.

1. `_sample_topk:407-408` invokes `self._topk(x_bf16, active_batch)`, while `_topk_single_device:568` and `_topk_multi_device:601` accept only `(self, x_bf16)`. Python rejects the call before the selected strategy executes.
2. Merely adding the argument to the methods is insufficient. Both methods reference `self._local_indices` at lines 573/618. The entire file contains no assignment or buffer specification for that attribute. They also reference `active_batch` without a parameter or local assignment in their present definitions.
3. The single-device branch creates `indices_list` but never passes it into its `ttnn.topk` calls. Its index preparation is orphan code, rather than a necessary part of the current offset scheme.

A Python AST inspection, without importing/running the target module, returned:

```text
_sample_topk: _topk call(line,nargs) = (408,2)
_topk_single_device: args = self,x_bf16; active_batch loads = 573; stores = none
_topk_multi_device: args = self,x_bf16; active_batch loads = 618; stores = none
_local_indices: four Load references; zero Store references in the entire module
```

The narrow crash fix is to call `_topk(x_bf16)`, remove the orphan `_local_indices`, `indices_list`, and `sliced_indices` preparation/cleanup from both strategies, and remove `indices_tensor=local_indices` from the multi-device top-k call. `ttnn.topk` generates local column indices when that argument is omitted; this matches the current `TTSampling` implementation. Keeping the existing post-gather offset addition preserves TP token numbering. This proposal fixes the verified Python defects; it is not a hardware correctness or performance result.

Adding a full-vocabulary index buffer instead is unnecessary for the existing algorithm, introduces ownership/cleanup and power-of-two-padding shape obligations, and disables the Blackhole automatic large-indices route when passed as a custom index tensor (`topk.cpp:291-295`).

### Verified lower-level contracts

Source: `ttnn/cpp/ttnn/operations/reduction/sampling/device/sampling_device_operation.cpp`.

| Item | Contract |
| --- | --- |
| Candidate values | BF16, TILE, interleaved, `[1,1,B,W]` |
| Candidate token indices | UINT32 **or INT32** on Wormhole/Blackhole, ROW_MAJOR, interleaved, same logical shape as values |
| B | 1 through 32, exact logical dim 2 |
| W | Nonzero, divisible by 32; W/32 must be a power of two |
| k | UINT32 or INT32, ROW_MAJOR, exact `[B]`; supported sampling k is 1 through 32 |
| p and inverse temperature | BF16, ROW_MAJOR, exact `[B]` |
| Preallocated output | UINT32 or INT32, interleaved, exact `[1,1,1,B]` (lines 125-139); allocate ROW_MAJOR |
| Seed and user-id tensors | UINT32, ROW_MAJOR, rank 1, equal shapes and volumes; keep the same lane/core mapping for seeding and drawing |

The suspected INT32 candidate-index incompatibility is **refuted**: validation explicitly accepts INT32 (lines 47-55). UINT32 output remains the useful embedding contract on the Stage 6 hardware.

`embedding/device/embedding_device_operation.cpp:90-91` requires row-major token inputs to have dimensions 1 and 2 equal to one. Consequently `[1,1,1,B]` can feed embedding directly, whereas `[1,1,B,1]` cannot for B > 1. `reshape_view/reshape.cpp:613-641` only takes its ordinary row-major metadata-view shortcut when the last dimension is unchanged. Converting `[1,1,1,B]` to `[1,1,B,1]` changes that dimension and must not be assumed to alias a persistent buffer. Prefer native token shape through sampling and embedding, then reshape the embedding output for the decoder. If a separate public B-column token buffer is mandatory, explicitly capture the required device data movement and validate its persistent destination semantics.

`TTSampling.__init__:198-203` rounds the configured maximum batch to at least 32. Its parameters, offsets and greedy mask are all allocated for that rounded batch, and `forward` does not slice them to active B. B=8 values combined with 32-row offsets/masks are therefore not the advertised B=8 pipeline; parameter and value/index equality validation prevents simply attaching a B=8 output. The existing common tests intentionally pad logits to 32 rows, copying a valid row into inactive lanes instead of using all-negative-infinity rows (`test_tt_sampling.py:197-213`).

## TP4 candidate layout

Assume one 1D TP mesh, contiguous vocabulary sharding in rank order, valid and model-padded vocabulary both 248320, and local vocabulary width 62080.

```text
local logits[r]       [1,1,B,62080]   BF16 TILE; r=0..3
local top32 values    [1,1,B,32]      BF16 TILE
local top32 indices   [1,1,B,32]      local column IDs
gather values        [1,1,B,128]     rank-major four blocks of 32
gather indices       [1,1,B,128]     same block order as values
offsets              [1,1,B,128]     INT32 TILE, replicated:
                                      [0]*32 + [62080]*32 +
                                      [124160]*32 + [186240]*32
global indices       [1,1,B,128]     add in INT32, widen/output UINT32,
                                      then untilize for sampling
sampled output       [1,1,1,B]       UINT32 ROW_MAJOR, replicated
```

Both gathers are along tensor dimension 3, `cluster_axis=None` for a 1D mesh. The output candidate row is 128 columns / four tiles, satisfying the sampler's power-of-two tile-count condition. No sampled-token gather is present in either current implementation; each TP rank samples the same gathered candidate set with the same per-user seeds. Ensure the model uses identical seed/parameter replicas.

With runtime `k=1, p=0, inverse_temperature=1`, the result is greedy even though the candidate extraction uses local k=32. That local tile size is a storage/selection parameter, not permission to change the user's greedy request to stochastic top-32. For runtime k <= 32, the union of per-shard top32 contains the global top-k, modulo exact-tie selection. `TTSampling._adjust_values_for_tiebreak` prefers the smallest global token ID among gathered tied maxima for k=1. It explicitly cannot recover the smallest ID if more than 32 tied maxima in one shard cause that ID to be dropped before gathering. `Sampling1D` has no corresponding tie correction.

### A concrete common-path configuration

```python
from types import SimpleNamespace
from models.common.sampling.generator import (
    SamplingGenerator, SamplingParams, format_sampling_params,
)

grid = mesh.compute_with_storage_grid_size()
worker_grid = ttnn.CoreRangeSet([
    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))
])
args = SimpleNamespace(
    vocab_size=248320,
    padded_vocab_size=248320,
    max_batch_size=32,             # Current TTSampling/TTPenalties contract.
    max_top_k=32,                  # Local candidates, independent of runtime k.
    cluster_shape=mesh.shape,
    sampling_dp=1,
    sampling_all_gather_axis=0,    # Ignored in favor of None for 1D TP4.
    sub_core_grids=worker_grid,    # Pin matching seed/sampling lane placement.
    sub_core_grid_topk=None,
    pad_logits_to_power_of_2=False,  # Benchmark before choosing padding; see below.
    salt_duplicate_seeds=False,   # Independent requests with equal seeds should match.
    use_topk_logprobs=False,
    model_config={
        "SAMPLING_AG_CONFIG": {
            "allow_force_argmax": False,
            "num_links": 1,
            "topology": ttnn.Topology.Linear,
        },
    },
)
sampler = SamplingGenerator(args=args, mesh_device=mesh, tt_ccl=ccl)
params = format_sampling_params(
    SamplingParams(temperature=0.0, top_k=1, top_p=0.0), 32
)
sampler.reset_sampling_params(params)
# Allocate one replicated UINT32 ROW_MAJOR [1,1,1,32] output before tracing.
# logits32 has 32 logical rows, with active requests first and valid inactive rows.
tokens, logprobs = sampler.sample(logits32, enable_trace=False, tt_out_tok=out32)
```

`format_sampling_params` translates user temperature 0 to device `(k=1,p=0,temp=1)` and positive user temperature T to the device multiplier `1/T`. It clamps nonpositive/unbounded top-k to 32 and top-k > 32 to 32; this is restricted top-p over at most 32 selected global candidates, not unrestricted full-vocabulary nucleus sampling. For a mixed batch pass explicit per-active-user lists. Scalar `seed` is deliberately lane-scoped rather than broadcast.

Historical routing concern, resolved by the final integration review below: `TTSampling._perform_all_gather:563-568` falls back to `ttnn.all_gather` when the CCL object has no `line_all_gather`. That fallback does not forward link/topology flags. The original recommendation to add an adapter for those flags is withdrawn: current `ttnn.all_gather` deliberately ignores its deprecated `num_links` and `topology` arguments and discovers routing from the mesh/fabric. `SAMPLING_AG_CONFIG` still controls the separate force-argmax path. `Sampling1D` passing deprecated flags to ordinary `ttnn.all_gather` does not give it stronger routing control.

### Exact-width performance hypothesis, not a result

Current stock top-k multicore eligibility requires a power-of-two reduced width **strictly less than 65535**, plus grid/L1 feasibility (`topk_utils.cpp:27-43`, `topk_constants.hpp:25-26`). Consequently neither 62080 nor its next power of two 65536 is eligible. The older sampler's comment that padding selects the fast path does not establish that for this vocabulary. Blackhole has a separate automatic route; Wormhole does not gain that route by padding.

A bounded future experiment for this exact TP4 shape is:

1. Split each local 62080 shard into two tile-aligned halves of 31040.
2. Pad each half to 32768 with negative infinity; each is structurally eligible for stock multicore top32.
3. Add local half offsets 0 and 31040 to the returned indices, concatenate 64 local candidates, then reduce to 32 with `indices_tensor` carrying those corrected indices.
4. Gather the resulting 32 candidates per device and add the unchanged TP rank offsets above.

This retains the requested one-tile local candidate output. It adds a small local merge and needs an exact-shape correctness/latency comparison against direct top32. The union of each half's top32 contains the shard top32, with the same exact-tie limitation noted above. Do not change rank stride from 62080 to 65536 or 32768: padding is internal to candidate extraction and does not move real vocabulary boundaries.

## State, trace, and feature comparison

| Capability | Sampling1D | TTSampling / SamplingGenerator |
| --- | --- | --- |
| k/p/temperature | Per-call device tensors | Persistent tensors updated in place by `reset_params` |
| Batch contract | Config can use actual B; offsets sliced by active rows | Rounded 32-row common wrapper; no per-call slicing |
| Local candidate extraction | Currently broken; narrow crash repair above | Implemented implicit local indices and rank offsets |
| Greedy ties | No global-ID correction | Correction on gathered candidate values; documented >32 local ties limitation |
| Persistent token output | Forwards `tt_out_tok` to sampling | Same; C++ native shape applies to both |
| RNG | Persistent/default or per-call seeds; no request counters | SeedManager owns counters, slot remap, reset, optional duplicate salting, bounded seed hash |
| Trace ownership | Caller owns all trace state | SamplingGenerator caches traces by penalties/logprobs/force-argmax/bucket and validates tensor identity |
| Explicit seeded sampling | Caller must refresh stable seed buffer | `sample` deliberately disables its internal trace when active explicit request seeds are present |
| Penalties | Separate Penalties1D, caller-owned request/accumulator state | TTPenalties integrated before sampling and updates counts after each real token |
| Logprobs | Sampled-token interface only | Sampled-token and top-k interfaces; **both return None on TP4** |
| Release | Explicit lazy-buffer/calculator release | Generator releases traces through `reset_trace`; no equivalent module-wide buffer-release method in these files |

The logprob limitation is an explicit source gate in `tt_log_probs.py:419-426`: only total device counts 8 or 32 are supported. Enabling the flag on TP4 does not produce logprobs. Neither source path should be reported as TP4 logprob-complete without additional implementation and validation.

For a combined decoder/LM-head/sampler trace:

- Preallocate model positions, token input/output, k/p/temp, seeds and penalty state before capture. Execute the sampling computation directly inside the outer trace (`sample(..., enable_trace=False)` if using SamplingGenerator); do not nest its internal capture.
- The captured `manual_seed` call reads the persistent seed buffer immediately before sampling. Refresh that same allocation before replay for explicit per-token seeds; do not replace its tensor object. SeedManager writes the buffer with `copy_host_to_device_tensor`.
- SeedManager does not advance model positions. The model wrapper must independently update the existing position/RoPE/page-table buffers or capture the intended on-device position increment. Ensure exactly one increment per real token.
- `align_seed_counters_to_positions(..., offset=1)` uses authoritative absolute positions, and `get_new_values(active_slots)` advances/writes the derived seed. Choose the offset against the wrapper's prefill-first-sample convention explicitly. Repeatedly aligning to stale host positions freezes or repeats the RNG stream.
- For unseeded draws, SeedManager writes initial entropy seeds, then writes UINT32_MAX as the skip-reseed sentinel, then stops seed copies so device RNG advances. Leaving default non-sentinel seeds unchanged causes reseeding on every call. Restore seed state after warmup before using the first real token.
- Trace capture is a preparation step. `capture_trace` records its output references and never calls `execute_trace` itself. Its optional eager precompile uses `count_tokens=False`. Execute the intended replay before consuming a captured result, especially with `skip_precompile=True`; do not treat warmup output/counts as a real generated step.
- Penalty counts must update on every real token, including trace replay, but not during eager compilation. Preserve/restore the decoder's KV, recurrent and convolution state as well as sampling/request state around warmup. Eager precompile can still overwrite output tokens and RNG state even when `count_tokens=False`.
- Updating values of persistent parameters preserves the traced graph, while toggling penalties/logprobs/force-argmax or changing B/input allocations can change it. The outer model trace must include those choices in its own key or invalidate/rebuild accordingly. SamplingGenerator's internal keys do not manage an independently owned outer trace.

## Next evidence needed

First land only the chosen wrapper or narrow verified Python repair. Then, when the coordinating hardware owner schedules it, compare B=1/B=8/B=32 token outputs against CPU global argmax across all four vocab shards, including highest-index shard IDs, negative logits, ties, and inactive rows. Verify the first replay after warmup and consecutive replays with changing token/position/seed inputs; test equal-seed reset reproducibility and a non-greedy k>1 case. Benchmark local candidate extraction separately from full generation before claiming any speedup. This diagnosis ran no such device jobs.

## Final integration disposition: common candidate gathers use native routing

Follow-up date: 2026-09-12. Source and existing profile evidence only; no TTNN imports, hardware execution, new tests or implementation changes by this reviewer.

**Retain the final common `TTSampling` integration. No adapter or custom sampler is needed to forward ignored routing flags.** The earlier adapter recommendation was a documentation overstatement, not a demonstrated violation of the inherited decoder CCL policy.

The final wrapper constructs `TTSampling(self.mesh, model.ccl, args)` with `MeshShape(1,4)`, local vocabulary 62080, `max_top_k=32`, 32 sampler rows, and split sampling selected. `model.ccl` is `models/common/modules/tt_ccl.TT_CCL`, which provides semaphore management but no `line_all_gather`. Consequently `_perform_all_gather` uses the ordinary call:

```python
ttnn.all_gather(candidates, dim=3, cluster_axis=None,
                memory_config=ttnn.DRAM_MEMORY_CONFIG)
```

There are two such gathers: local candidate values and local candidate indices. The selected path does not gather the full vocabulary or sampled output tokens. The `SAMPLING_AG_CONFIG` values `num_links=2` and `topology=Ring` apply to the force-argmax comparison path using the experimental async API. The split path's internal `num_gather_links=1` is passed into `_perform_all_gather` but is not forwarded by this fallback; it does **not** force one native link.

Both API documentation and implementation establish the behavior:

- `ttnn/cpp/ttnn/operations/ccl/all_gather/all_gather_nanobind.cpp:51-52` explicitly documents `num_links` and `topology` as deprecated and ignored.
- `all_gather.cpp:141-161` only inspects those deprecated values for a warning. The native `prim::all_gather` invocation does not receive them; the composite fallback also receives null routing overrides. Supplying the flags would therefore not impose the intended route.
- `device/all_gather_device_operation.cpp:312-361` obtains the active fabric configuration, discovers topology for each active mesh axis and queries its available routing planes. `ccl_common.cpp:168-190` requires a wired wrap before selecting Ring. `common/host/moe_utils.cpp:107-136` derives link count from available routing planes, falling back to one only if discovery fails.

For this tile-aligned candidate layout, the native path has no gather-dimension padding or row-alignment conversion to perform. Its Blackhole program-selection heuristic chooses between multicast and unicast from page/transfer sizes; the small candidate shape falls in the multicast region in the inspected source. This is automatic selection from actual tensor/fabric properties, not a Python promise of a hard-coded routing count.

Existing post-repair profile evidence confirms the relevant runtime configuration. In `tracy/profile_final/device0_ops.csv.gz`, candidate `AllGatherDeviceOperation` rows record:

```text
fabric_config     FABRIC_1D_RING
cluster_axis      std::nullopt
axis_num_devices  {1,4}
axis_num_links    {0,2}
axis_topology     {Linear,Ring}
input             [1,1,32,32], TILE, interleaved DRAM
output            [1,1,32,128], TILE, interleaved DRAM
values dtype      BF16
indices dtype     UINT16, widened after gather before sampling
```

The sample-phase report `device0_sample_perf_report.txt`, operations 1223 and 1224, shows two worker cores and approximately 6 and 7 microseconds respectively. Each per-rank candidate input is one 2 KiB tile. The multicast factory partitions input pages across discovered links, so two-link/two-worker metadata does not imply that both links carry useful payload for a one-page input. It does establish that the native operation already discovered the available two-link Ring configuration. There is no evidence of a lost routing override to repair.

Decoder and embedding collectives retain their separate explicit policy: `MultichipDecoder` keeps BF16 CCL, `num_links=2`, `ring=True`, and its existing experimental async all-reduce/reduce-scatter/all-gather calls; the embedding entry gather likewise passes Ring/two links to `experimental.all_gather_async`. This common sampler fallback does not change those calls, inter-layer layouts, or their retained buffers.

The coordinator's final reduced sampler window is 0.4897 ms, with about 0.013 ms in the two candidate gathers (`performance.md` and the saved sample report). This is about 1.91% of the separately measured 25.605 ms full-stack token-out latency. These are existing measurements, not a new experiment or a claim that an alternative adapter would be faster. The completed top-k, trace-feedback, inactive-row, page-table and repeated-request evidence remains applicable; this routing clarification introduces no new correctness concern requiring another test.

Final wording for the integration contract: **decoder/embedding collectives preserve explicit Ring/two-link settings; common split sampling gathers one tile of local candidates per rank using native fabric routing, observed to resolve to the same two-link Ring configuration on this run.** Keep routing discovery and observed configuration distinct from ignored Python options. No sampling implementation change is warranted by this review.
