# Optimized multichip decoder — `meta-models/Muse-Glimmer-30B`

An optimization pass over the [multichip decoder](../multichip_decoder/README.md),
in place, on the same four Blackhole dies. Same public contract, same paged
semantics, same 131072-token capability, the **same correctness floor to the sixth
decimal** — and **0.53–0.55 % faster traced decode** plus **2.9–3.6 % faster
prefill device time**, taking the decode speedup against one chip from
2.39x/2.49x to **2.40x/2.50x**.

The headline is small, and the reason is worth stating up front: the two largest
candidates this stage found — persistent CCL staging buffers and the async
collective on the *decode* payload — were implemented, measured at 1.4 % together,
and then **rejected on correctness**. What ships is what survived.

| item | value |
| --- | --- |
| implementation | `models/autoports/meta_models_muse_glimmer_30b/tt/multichip_decoder.py` (optimized in place) |
| tests | `tests/test_multichip_decoder.py` + `tests/test_multichip_vs_single_chip.py`, unchanged as a gate; 4 new contract tests |
| baseline | the multichip decoder at `937ed0b5c50`, re-measured in the same harness in the same processes |
| device | 4 x Blackhole, `ClusterType::P300_X2`, `ttnn.MeshShape(1, 4)`, `FABRIC_1D_RING`, `ttnn.Topology.Ring`, 2 links |
| what changed | the **prefill** reduction calls the async CCL primitives with semaphores this layer owns; the decode residual crosses the **layer boundary width-sharded in L1** |
| what did not | parallelism, precision policy, KV-cache dtype/layout, matmul geometry, SDPA config, collective bytes, packet size, `l1_small_size` |
| capability | unchanged: 131072 tokens, batch to 32, non-aligned lengths — [`../context_contract.json`](../context_contract.json) |

## Result

`bench/layer_ab.py`, warmed, min of 3 rounds, no profiler attached. `before` and
every `after` row below come from **one invocation** of the harness
(`logs/final_layer_ab.log`), so they are like-for-like; each configuration is run
two or three times in that same process as its own repeat control.

| window | before | after | delta |
| --- | --- | --- | --- |
| traced decode, sliding @2048 | 0.4574 / 0.4571 | **0.4547 / 0.4545 / 0.4547** | **−0.55 %** |
| traced decode, full @2048 | 0.4257 / 0.4264 | **0.4238 / 0.4239 / 0.4240** | **−0.53 %** |
| prefill 8192, sliding | 19.31 / 19.09 ms | 19.64 / 18.98 / 19.20 ms | inside the spread |
| prefill 8192, full | 18.89 / 18.88 ms | 18.28 / 18.43 / 19.06 ms | inside the spread |

Traced decode reproduces to 1e-4 within a run and across runs, so 0.52 % is an
order of magnitude above the noise.

Warmed prefill is much noisier — the same configuration measured three times in
one process spans 3.4 % (`sliding`) and 4.1 % (`full`), which is larger than the
effect — so the prefill row is **not** decided by this table and is not claimed
from it. It is decided at the op level, where the collective itself is **15.2 %
faster** (§[Collectives](#collectives)), and confirmed by device time, which drops
**2.88 % (`sliding`) / 3.64 % (`full`)** on the 8192-token window.

Against one chip, both measured in the same harness on the same host
(`logs/final_layer_ab_single.log`):

| window | 1 chip | 4 chips | speedup | was |
| --- | --- | --- | --- | --- |
| traced decode, sliding @2048 | 1.0909 | **0.4545 ms/token** | **2.40x** | 2.39x |
| traced decode, full @2048 | 1.0604 | **0.4238 ms/token** | **2.50x** | 2.49x |
| prefill 8192, sliding | 44.21 | 18.98 ms | 2.33x | 2.30x |
| prefill 8192, full | 44.37 | 18.28 ms | 2.43x | 2.33x |

Device time from the eight committed Tracy windows, decode divided by its 8
replays, against the multichip stage's tables captured the same way:

| window | before | after | delta | ops/iter |
| --- | --- | --- | --- | --- |
| decode sliding / full @2048 | 441.5 / 419.0 | **438.8 / 416.4** | −0.62 / −0.62 % | 46→45 / 36→35 |
| decode sliding / full @131071 | 440.5 / 522.7 | **438.2 / 520.6** | −0.53 / −0.39 % | 46→45 / 36→35 |
| prefill 8192 sliding / full | 18211.6 / 17925.2 | **17687.1 / 17272.0** | **−2.88 / −3.64 %** | unchanged |
| prefill 128 sliding / full | 839.8 / 814.6 | 840.1 / 812.5 | +0.03 / −0.26 % | unchanged |

The decode op count drops by exactly one — the `sharded_to_interleaved` the
boundary contract removes — and `ShardedToInterleaved` falls from 5.85 μs over 5
instances to **3.98 μs over 4**. The decode collectives are unchanged
(`ReduceScatter` 29.36 + `AllGather` 24.09 = 53.45 μs against 53.24), which is the
point: decode keeps the wrappers, so the whole device-time delta is the removed
conversion.

### Where the win comes from

Each knob measured on its own, in the same invocation, by turning exactly one off:

| change | sliding | full | worth |
| --- | --- | --- | --- |
| **sharded layer boundary** (`no_sharded_io` turns it off) | 0.4573 → 0.4545 | 0.4260 → 0.4238 | **0.61 % / 0.52 %** |
| **async prefill collective** | not visible in decode | not visible in decode | **15.2 % of the prefill collective**, −2.9 / −3.6 % of prefill device time |
| async collective on *decode* too (`ccl_async`) | 0.4555 | 0.4244 | **0.22 % / 0.14 % slower** — rejected, below |
| persistent CCL staging buffers | 0.31 % faster | 0.29 % faster | **rejected on correctness**, below |

So the decode win is the boundary contract, essentially all of it, and the prefill
win is the collective. The two candidates that looked bigger did not survive.

## Collectives

The multichip stage tuned the composite wrappers as far as they go and left one
observation in its own sweep table: `ttnn.reduce_scatter` exposes
`num_workers_per_link`, and setting it to 1 was that stage's single largest
non-matmul win (33.55 → 20.93 μs), while `ttnn.all_gather` is *"(not tunable)"* at
16.39 μs. Half of every reduction had no tuning surface at all.

`ttnn.all_reduce` decomposes into `reduce_scatter_minimal_async` +
`all_gather_async`, and those primitives **do** expose it.
`MultichipDecoder._all_reduce_async` calls them directly, with semaphores the
layer owns and the all-gather's worker count set explicitly.

**It ships for prefill and not for decode**, and the number that decides the split
is not a worker count — it is a barrier semaphore.

### The barrier semaphore, and a watcher trip

Both primitives take a `barrier_semaphore`. The first implementation here passed
one to the reduce-scatter and left the all-gather's at its `None` default. The
first watcher run of that build **stopped the device**:

```
Device 0 acteth core(x= 0,y= 9) virtual(x=29,y=25): subordinate_erisc detected
invalid NOC command buffer state before starting the next kernel (write-capable
NOC packet tags must be zero so implicit transaction ID users start with
transaction ID 0).  Current kernel: fabric_erisc_router.cpp
```

Two things have to be said about that, and the second is uncomfortable:

* the missing barrier is a real defect and it is fixed. Without it the next op can
  start against a fabric router that has not drained, which is what the assert
  describes; `models/tt_transformers/tt/ccl.py` passes a barrier to both
  collectives and this layer now does too;
* **the trip is not reproduced.** The build that tripped also carried the
  persistent staging buffers that §"Persistent staging buffers" rejects, and the
  run's artifacts were overwritten by a later one (a mistake, now prevented — the
  watcher script takes a `WATCHER_TAG`). Re-running the same 35 node ids with the
  barrier deliberately removed, from committed code
  (`MG_MULTICHIP_CCL_IMPL=async MG_MULTICHIP_CCL_AG_BARRIER=0`), is **watcher-clean**:
  `logs/watcher_no_ag_barrier.log`, every counter 0.

So the honest statement is: a fabric-level watcher trip was observed once on a
build with two defects in it, one of which (persistent buffers) is independently
and reproducibly wrong, and it has not recurred since either was fixed. The
barrier is kept because it is what the op contract and every other model in the
tree do, and because it costs 0.4 % where it ships — **not** because this stage
can show it was the cause.

The decode/prefill split therefore rests on the measurement, which is sufficient
on its own:

| payload | async, with AG barrier | wrapper |
| --- | --- | --- |
| prefill, 8192 rows, BFP8 (op level) | **1348.0 μs** | 1588.7 (`all_reduce`) |
| decode, 40 KB (traced whole layer, sliding / full) | 0.4555 / 0.4244 | **0.4545 / 0.4238** |

The decode rows do not overlap across their three rounds
(`logs/final_layer_ab.log`), so 0.2 % is a real ordering, not noise. **Prefill
takes the async pair** — 15.2 % off the collective, twice per layer, which the
profiler sees as −2.9 % / −3.6 % of the 8192-token prefill window — and **decode
keeps the wrappers**, because at 40 KB the collective is pure fixed cost and one
more synchronization round costs more than the async op's tuning surface buys.
Removing the barrier would buy 0.13 % of the prefill collective (1346.3 against
1348.0) and is not worth reopening the question.

`test_collective_implementation_is_split_by_payload` asserts both halves at
dispatch level, because either one flipping silently still computes the right
answer.

### The all-gather worker count is per payload too

All rows below are from the committed post-barrier probe
(`logs/prefill_ccl_probe.log`), at the shipped prefill payload:

| all-gather `num_workers_per_link` | prefill (107 MB, op level) |
| --- | --- |
| **op default (shipped)** | **1348.0 μs** |
| 1 | 2606.3 (+93 %) |
| 2 | 1661.5 (+23 %) |
| 4 | 1351.7 (+0.3 %) |

and the reduce-scatter's, holding the all-gather at 4:

| reduce-scatter `num_workers_per_link` | prefill |
| --- | --- |
| **4 (shipped)** | **1351.7 μs** |
| 2 | 1667.5 |
| 1 | 2086.6 |

This is the same latency-versus-bandwidth split the multichip stage found for the
decode reduce-scatter, with the sign reversed by the payload: one worker wins the
40 KB decode collective and loses the 107 MB prefill one by 93 %. The decode-side
worker sweep was measured on the pre-barrier build and is not repeated here,
because decode does not ship the async op; it is in `logs/ab_ccl_async.log` and is
marked superseded in [work log §2](work_log.md).

### Semaphores, and what they cost

The wrappers create a global semaphore per **program** and leave it in `L1_SMALL`
for the life of the program cache — the thing that bounds this mesh to 24 distinct
CCL programs (multichip [limitation 7](../multichip_decoder/README.md)).

The async prefill path adds **seven** semaphores, created once per mesh and shared
by every shape and every layer (`_CCL_SEMAPHORES`, dropped by
`close_multichip_mesh`), also in `L1_SMALL`: 7 x 256 B = 1,792 B of the 6,144 B
region. They are deliberately *not* double-buffered — the layer's two reductions
are separated by the norm, the residual add and two matmuls that consume one's
output to build the other's input, so they are ordered by data dependency rather
than by convention.

They must be in `L1_SMALL`. Twelve of them in the main L1 pool (the first
implementation) sit at the top of it for the life of the mesh, and the decode step
has only 7,296 B of headroom there — it made the *next* sharded-norm program fail
with *"Statically allocated circular buffers in program 8764 clash with L1
buffers"* in 7 of the 104 acceptance tests.

**The cost, stated plainly for the full-model stage**: those 1,792 B are 7 fewer
distinct wrapper CCL programs the mesh can hold before they spill into the main L1
pool and fragment it. A test session that builds hundreds hits this — the suite's
`_bound_ccl_semaphores` floor had to move from 1,536 B to 2,560 B, and
`test_collective_implementation_is_split_by_payload` clears the program cache up
front because it runs the L1-tightest window (a 256-row prefill) *and* a decode in
one test. A stacked model does not: it dispatches two CCL shapes per layer kind
for decode and one per prefill chunk size.

### Persistent staging buffers — rejected, on correctness

OPT-009 asks for persistent or preallocated intermediate/output buffers on
repeated decode collectives. They were implemented
(`reduce_scatter_minimal_async_create_intermediate_buffer` for the chunk-paged
staging pair, plus this layer's own scattered output), they are **worth 0.31 % /
0.29 % of traced decode**, and they are **off**.

`reduce_scatter_minimal_async_create_intermediate_buffer` returns *uninitialised*
staging, and the ring algorithm reads the penult intermediate before writing it on
the first invocation. `bench/regression_bisect.py` isolates it to one column:

| configuration | decode0 | decode1–3 |
| --- | --- | --- |
| async decode, **no** persistent buffers | 1.000000 | 1.000000 |
| async decode, **persistent** buffers (`full`) | **0.739526** | 1.000000 |
| async decode, **persistent** buffers (`sliding`) | **0.774936** | 1.000000 |

A first-use fault, not a numerical one — and the reason it reached a committed
suite at all is instructive: the whole-layer A/B and the HF-reference tests all
warm the layer before they measure or assert, so every one of them passed. Only
`test_multichip_matches_single_chip[12345-4-full]`, which compares the *first*
decode step against a single-chip TTNN baseline at 0.999, caught it — at 0.7268.

Two fixes were implemented and measured: a warm-up collective through the fresh
buffers, and then that plus a `synchronize_device` and eager allocation at build
time. **Each moved the fault rather than removing it** — the first left the
default configuration at 0.9721 on the first `sliding` decode step, the second
left the wrapper-prefill/async-decode combination at 0.9605 on the first `full`
one, and the failing arm changed between runs of the same code. That is a race.

An intermittently wrong first token is not a defect a *stacking baseline* may ship
for 0.3 %: 52 layers compose it, and both the full-model and the vLLM stages decode
without prefilling through this layer, which is the arm that fails hardest. The
knob (`ccl_persistent_buffers=True`), the warm-up
(`_prewarm_decode_ccl_buffers`) and the bisect harness are all kept, because the
async ops themselves are clean in every arm and carry the win, and because this is
a TTNN first-use contract worth re-testing when the op changes.

## The inter-layer residual layout contract

**Full-model bringup should preserve this rather than rediscover it.**

The decode residual is width-sharded in L1 for the whole layer, and now across the
layer *boundary* as well:

| property | value |
| --- | --- |
| layout | `TensorMemoryLayout::WIDTH_SHARDED`, `BufferType::L1` |
| grid | 16 cores, `MULTICHIP_BOUNDARY_CORES`; the memory config is `decoder.boundary_memcfg(rows, hidden_size)` |
| shard shape | `[32, 416]` — one tile row by `6656 / 16` |
| dtype | `BFLOAT16` (the activation dtype; **not** the collective payload dtype) |
| shape | `[1, 1, 32, 6656]` — tile-padded rows, logical batch is separate ($optimize OPT-005) |
| replication | replicated across the mesh; **no** collective, gather, reshard or all-reduce between layers |
| fixed point | layer *n*'s output is exactly what layer *n+1* returns, asserted by `test_decode_boundary_layout_is_a_fixed_point` and `test_two_layers_stack` |
| ownership | the layer **does not free** a sharded input; the caller still owns it |
| compatibility | a DRAM-interleaved input is still accepted and still produces the boundary layout, so the contract is a superset of the multichip one |

`prefill_forward` is unchanged: DRAM-interleaved in, DRAM-interleaved out, which is
right for its regime (its activations are large and interleaved) and is already a
fixed point.

What it is worth, measured on the same layer in the same process
(`bench/boundary_probe.py`, `logs/boundary_probe.log`, PCC against the
DRAM-boundary arm):

| kind | layers | DRAM boundary | sharded boundary | per layer | PCC |
| --- | --- | --- | --- | --- | --- |
| sliding | 1 | 0.4508 | 0.4489 | 1.90 μs | 1.000000000 |
| sliding | 2 | 0.8906 | 0.8840 | 3.30 μs | 1.000000000 |
| full | 1 | 0.4201 | 0.4181 | 1.99 μs | 1.000000000 |
| full | 2 | 0.8373 | 0.8247 | 6.34 μs | 1.000000000 |

PCC is exactly 1.0: the same computation with fewer conversions. The two-layer
rows are the ones a stack should read, because only they contain a real
layer-to-layer *join* — isolating it gives **4.7 μs per join on `sliding`** and
**10.7 μs on `full`**, against 1.9–2.0 μs for the tail conversion a one-layer
measurement sees. Over 52 layers that is the difference between paying the join
and not.

## Operation-topology audit

The full table — every op in the measured decode path, its share, whether it is a
defect, and what was done — is [work log §1.2](work_log.md). Its conclusions:

* **four** layout conversions in the decode path. Two are removed by the boundary
  contract above; one (`sharded_to_interleaved` before
  `nlp_create_qkv_heads_decode`) is an exact op contract — the op rejects a
  sharded input whose shard width does not divide `head_dim`, and its interleaved
  DRAM reader zeroes odd Q rows on Blackhole (tt-metal #16667); one feeds the gate
  multiply. The `_reshard_to` calls around the MLP are already no-ops at TP=4.
* **four** material collectives (two reductions, each a scatter plus a gather),
  53.2 μs = 12.0 % of the step — the subject of the section above.
* **two** repeated same-input matmul groups — `wqkv`+`attn_gate` off the input
  norm, and the MLP gate/up pair. Both packings measured, both rejected, below.
* **no** fused matmul-CCL op is usable on this path, and that is a measurement
  rather than an API error — below.

## Measured and rejected

Every row is a candidate this stage ran on this mesh, not a quotation.

| candidate | verdict | evidence |
| --- | --- | --- |
| async collective on the **decode** payload | rejected: **0.4555 / 0.4244** against the wrappers' **0.4545 / 0.4238**, three non-overlapping rounds each | `logs/final_layer_ab.log` |
| omitting the all-gather barrier semaphore | not taken: worth 0.13 % of the prefill collective (1346.3 vs 1348.0); a watcher trip was observed once on a build that also carried the rejected persistent buffers and has **not** reproduced since | `logs/prefill_ccl_probe.log`, `logs/watcher_no_ag_barrier.log` |
| persistent CCL staging buffers | rejected: 0.31 % faster, intermittently wrong on the **first** decode step (PCC 0.7395 / 0.7749); two fixes moved the fault instead of removing it | `logs/regression_bisect.log`, `logs/regression_bisect_fixed.log` |
| fused `matmul_reduce_scatter_async`, DRAM-sharded config | blocked, exact contract: *"Unsupported MatmulProgramConfig type for MatmulReduceScatterAsync. Needs to be 2D Multicast."* | `logs/fused_ccl_probe.log` |
| fused `matmul_reduce_scatter_async`, 2D-multicast config | rejected **on measurement**: the fusion is worth +2.2 % / −2.6 % against its own unfused control, but the 2D-multicast matmul the op requires costs **38 %** (`o_proj`) / **103 %** (`mlp_down`) against the shipped DRAM-sharded form | `logs/fused_ccl_probe.log` |
| `all_gather_matmul_async` (gathered-input `o_proj`, OPT-008) | rejected: 64.74 μs fused / 65.84 unfused against **44.91** for the shipped decomposition — 44 % slower *with* the fusion | `logs/fused_ccl_gathered_input.log` |
| packed `wqkv`+`attn_gate` (OPT-001) | rejected: 41.05 vs **40.50** μs. `in0_block_w=13` is legal here, unlike on one chip, so this rejection is about the split cost, not the block size | `logs/packing_probe.log` |
| packed MLP gate/up (OPT-010) | rejected: 145.66 vs **142.96** μs at `in0_block_w=13`; 4 and 2 are illegal (`(shard_shape[1]/tile_width) % in0_block_w == 0`), 1 costs 87 % | `logs/packing_probe.log` |
| `o_proj` working shard at 8 cores, `in0_block_w=4` (OPT-011) | rejected **on the shipped path**: it *wins*, 0.4542 / 0.4236 against 0.4546 / 0.4238 (0.11 % / 0.05 %, rounds non-overlapping) -- and costs an extra reshard op, the single-grid invariant three structural tests assert, and 13 % of the multichip-vs-single-chip PCC headroom (0.999183 -> 0.999159 against a 0.999 bar). Not a good trade for a stacking baseline | `logs/ab_oproj_workshard_final.log`, `logs/ab_oproj_workshard.log` |
| `o_proj` at 4 / 2 / 1 cores | rejected: 4 cores slower; 2 and 1 fail L1 with the exact circular-buffer messages | `logs/ab_oproj_workshard.log` |
| BFP4 attention weights (OPT-007) | rejected on the **released checkpoint, on this topology**: 0.49 % faster decode, prefill PCC 0.9695 / 0.9732 and decode PCC 0.9818 / 0.9748 against a 0.995 bar | `logs/real_weight_precision.log` |
| BF16 attention weights | rejected: 57 % / 61 % slower decode, no PCC gain | `logs/real_weight_precision.log` |
| HiFi2 decode fidelity | rejected: 42 % / 45 % slower for ≤1e-4 of PCC — LoFi re-confirmed on this topology | `logs/real_weight_precision.log` |
| BFP8 activations | blocked, exact contract: `nlp_create_qkv_heads_decode` takes FLOAT32 or BFLOAT16 only | `logs/real_weight_precision.log` |
| BFP4 KV cache | rejected: no speed change, decode PCC 0.9781 / 0.9733 against 0.995 | `logs/real_weight_precision.log` |
| BF16 KV cache | rejected: no speed change, 2x the cache bytes, +8e-5 of PCC — BFP8 re-confirmed | `logs/real_weight_precision.log` |
| async all-gather at 1 / 2 / 4 workers (prefill) | rejected: 2606.3 / 1661.5 / 1351.7 μs against **1348.0** at the op default | `logs/prefill_ccl_probe.log` |
| async reduce-scatter at 1 / 2 workers (prefill) | rejected: 2086.6 / 1667.5 μs against **1351.7** at 4 | `logs/prefill_ccl_probe.log` |
| `chunks_per_sync` 2 / 10 / 20, `num_buffers_per_channel` 2 / 8, `num_links=1`, decode reduce-scatter at 2 workers | rejected. **Measured on a base this stage later withdrew** (async decode with persistent buffers), so they are evidence about knobs that are not live in the shipped path, not about it: 0.4553 / 0.4576 / 0.4573 and 0.4510 / 0.4511 and +6.1 % and 0.4546, against that base's 0.4509 | `logs/ab_ccl_async_tuning.log` |

## Correctness

The multichip stage's two modules are the gate, **unchanged**: a faster layer does
not get a lower bar.

**104 passed** (`logs/full_test_run.log`, [`test_results.xml`](test_results.xml))
and **4 passed** (`logs/vs_single_chip_run.log`). The four new tests are this
stage's own contracts; the other 100 are the multichip stage's, with two updated
where the *contract* changed rather than the behaviour.

The comparison that can actually see a parallelisation or scheduling fault —
multichip against a single-chip `OptimizedDecoder` on identical weights, inputs,
page tables and positions, at a 0.999 bar — reproduces the multichip stage's
numbers **exactly**:

| case | worst | multichip stage |
| --- | --- | --- |
| sliding, 2049, batch 1 | 0.999839 | 0.999839 |
| full, 2049, batch 1 | 0.999807 | 0.999807 |
| sliding, 12345, batch 4 | 0.999721 | 0.999721 |
| full, 12345, batch 4 | **0.999183** | 0.999183 |

Not "within noise" — the same six decimals. The optimized path changes which ops
dispatch and where tensors live, not what is computed.

Real-checkpoint PCC at 2049 tokens (`logs/real_weight_precision.log`): prefill
0.997755 (`sliding`) / 0.997067 (`full`), decode 0.998429 / 0.997342, against the
0.995 acceptance bar.

New tests:

* `test_collective_implementation_is_split_by_payload` — prefill dispatches the
  async primitives and no wrapper; decode dispatches the wrappers and no async
  primitive. Only a dispatch-level assertion catches a silent flip, because
  either one still computes the right answer.
* `test_decode_boundary_layout_is_a_fixed_point` — the boundary contract above:
  sharded in gives sharded out, interleaved in still works, the layer does not
  free a caller's tensor, and both paths produce bit-identical output.
* Updated `test_two_layers_stack` — asserts the boundary layout is a fixed point
  across a layer instead of asserting it equals a DRAM-interleaved input.
* Updated `test_ccl_mode_override` — pins `ccl_impl="wrapper"`, because
  `ccl_mode` selects between the two *wrapper* spellings and is inert on the async
  path. Both wrapper spellings stay live and stay tested.
* `_CollectiveSpy` now traps the async primitives as well as the wrappers, so
  every "how many reductions" contract test measures reductions rather than
  spellings.

**Runtime fallback audit clean** — `test_no_host_fallback_in_forward` traps
`ttnn.from_torch` / `to_torch` / `as_tensor` and 13 torch entry points across a
full prefill and decode; the persistent-buffer allocation path deliberately uses
`ttnn.zeros` rather than `ttnn.from_torch` for that reason. **Stress**: the
inherited 64-step soak, 3-repeat determinism and traced-replay tests all pass
against the optimized path.

## Limitations and known issues

1. **Persistent CCL staging buffers are unusable on this build.** Worth 0.3 % of
   decode; rejected for an intermittent first-use fault that two fixes moved
   rather than removed. A TTNN bug worth filing:
   `reduce_scatter_minimal_async_create_intermediate_buffer` returns
   uninitialised staging that the ring path reads before writing.
2. **The async collective loses on the decode payload once it is used safely.**
   The all-gather's barrier semaphore is mandatory (the watcher stops the device
   without it) and costs more at 40 KB than the async op's tuning surface buys.
   Decode therefore keeps the composite wrappers, and half of its reduction --
   the all-gather -- still has no tuning surface. That is a TTNN gap, not a
   choice: `ttnn.all_gather` should expose `num_workers_per_link`.
3. **The async prefill path costs 1,792 B of the `L1_SMALL` CCL-program budget**,
   i.e. 7 fewer distinct wrapper CCL programs before they spill into main L1. A
   stacked model cannot reach that; a session that builds hundreds of CCL shapes
   can, and the suite's clearing floor had to move to accommodate it.
4. **The BFP4 MLP rows are still unpack-bound at 52 % of peak**, inherited
   unchanged, still the largest single lever, still needing a TTNN change
   (`matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:240` fixes the
   worker count to the DRAM bank count).
5. **`o_proj` still runs at 62 % of peak DRAM.** The narrower working shard that
   OPT-011 points at was implemented and measured; it wins 0.20 % against the old
   collective and loses 0.07 % against the new one, and the wider block sizes fail
   L1. The enabling code is kept so the candidate stays expressible.
6. **The hidden-size RMSNorms still do not shrink with TP**, and the fractured
   residual that would quarter them is still refuted for decode on the multichip
   stage's floor-corrected measurement. For prefill it is not refuted; see
   [work log §8.1](work_log.md) for what this stage did with it.
7. **Prefill is still not traced**, so it keeps a host gap. Inherited; tracing
   prefill belongs to the stage that owns the generator loop.
8. **Whole-layer prefill A/B cannot resolve changes below ~4 %** on this harness.
   Every prefill decision in this stage is therefore made at the op level and
   reported as such.

## Artifacts

```bash
D=models/autoports/meta_models_muse_glimmer_30b/doc/optimized_multichip_decoder
# the acceptance gate -- two pytest invocations, see the note in the script
bash $D/bench/run_suites.sh
# the before/after A/B, the single-chip baseline, and the real-weight precision run
python $D/bench/layer_ab.py --mesh 1x4 \
  --candidates before,tp4,tp4b,beforeb,tp4c,no_sharded_io,ccl_wrapper,ccl_async
python $D/bench/layer_ab.py --mesh 1x1 --candidates single
python $D/bench/layer_ab.py --mesh 1x4 --real-weights --pcc-seq-len 2049 \
  --candidates tp4,attn_bfp4,attn_bf16,fid_hifi2,act_bfp8,kv_bfp4,kv_bf16
# device-time profiles (no watcher in this run) and the watcher run (no profiler)
bash $D/bench/run_tracy.sh
bash $D/bench/run_watcher.sh
# the deliberately-unsafe arm, which is why run_watcher.sh takes a WATCHER_TAG
MG_MULTICHIP_CCL_IMPL=async MG_MULTICHIP_CCL_AG_BARRIER=0 WATCHER_TAG=_no_ag_barrier \
  bash $D/bench/run_watcher.sh
# the pre-stage control: the multichip stage's collectives and layer boundary
MG_MULTICHIP_CCL_IMPL=wrapper MG_MULTICHIP_SHARDED_DECODE_IO=0 bash $D/bench/run_suites.sh
python $D/bench/layer_ab.py --list
```

| probe | question it answers |
| --- | --- |
| `bench/layer_ab.py` | whole-layer candidate ranking; extends the multichip stage's harness so a number here is comparable with one there |
| `bench/boundary_probe.py` | what the inter-layer residual layout costs, at 1 and 2 layers |
| `bench/prefill_ccl_probe.py` | the prefill reduction, per implementation and knob, at the op level |
| `bench/fused_ccl_probe.py` | fused matmul-CCL, both decompositions, against unfused controls |
| `bench/packing_probe.py` | packed vs separate same-input projections at the per-device shapes |
| `bench/regression_bisect.py` | which optimized knob broke the first decode step |
| `bench/run_tracy.sh` | the eight signposted device-time windows |
