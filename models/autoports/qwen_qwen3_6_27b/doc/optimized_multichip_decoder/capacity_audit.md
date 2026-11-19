# Stage 5 CPU capacity and traffic audit

Audit scope: Qwen/Qwen3.6-27B, TP4 on a 1x4 Blackhole mesh, 48 GDN layers and
16 full-attention layers. Source inspected on 2026-11-13. This is a pre-final
CPU audit, not a stage review or hardware acceptance. No devices, TTNN imports,
or existing files were modified. The effective policy in `integrated1.json` is
still opt-in; its inherited geometry is eight activation workers and two DRAM
readers per bank. A later selected geometry must be supplied to the helper.

**Preserve the 262144 context contract.** Nothing in this audit proves a physical
limit below it. Packed QKVG and flat prefill do not change KV dtype/layout or
recurrent-state dimensions. The newly persistent Linear reduce-scatter scratch
adds only 40 MiB per device for one warmed short-M signature in each of 64
independently owned decoders. The stage-4 full-context evidence is historical
evidence; stage-5's final path still needs its appropriate 32K–64K smoke and
allocation evidence. Do not relabel stage-4 full-context results as stage-5 runs.

## Reproducible accounting

[`../../tests/optimized_multichip_capacity.py`](../../tests/optimized_multichip_capacity.py)
is a standalone standard-library helper. It prints JSON to stdout and never
writes the context contract. Its `plan`, `layer_inventory`,
`decode_lower_bound`, and `recurrent_kernel_traffic` functions can be called by
the final accounting step. Run:

```bash
python models/autoports/qwen_qwen3_6_27b/tests/optimized_multichip_capacity.py
```

Verified CPU invariants: with stage-4 policies the new resident total exactly
matches the old helper, **16,234,340,352 bytes/device**; its auxiliary total also
matches exactly. Checked reader counts 1, 2, 4, packed/split attention, B1/B32,
S33/S513, maximum context, and one/two warmed CCL signatures. This validates
the arithmetic against source and the previous inventory, not device peaks.

## Allocation formulas

Use `A(P,t) = 8 * ceil(P/8) * align64(t)` for an interleaved eight-bank DRAM
buffer with P pages of t bytes. This follows
`tt_metal/impl/allocator/allocator.cpp::calculate_bank_size_spread`; multiplying
the returned per-bank size by eight accounts for unused last-bank pages.

For TILE shape `[..., M, N]`, `P=prod(...) * ceil(M/32) * ceil(N/32)`;
tile bytes are BF16=2048, FP32=4096, BFP8=1088, BFP4=576. For ROW_MAJOR,
`P=prod(shape[:-1])`, `t=last_dim * element_bytes`. In particular, parameter
element counts miss padded norm vectors and small metadata allocations.
These are allocated payload-plus-bank-slack formulas, not allocator metadata,
reserved firmware space, or peak-fragmentation measurements.

For each matrix role j, resolve the effective role-specific geometry from
`policy.geometry[name]` over the common policy. With eight banks, c activation
workers and r readers per bank:

```
Q_j = 32 * lcm(8*r_j, c_j)
Nphysical_j = ceil(Nlogical_j / Q_j) * Q_j
K_j % (32*c_j) == 0
W_j = (K_j/32) * (Nphysical_j/32) * tile_bytes(dtype_j)
```

The shards have `[K_j, Nphysical_j/8]` per bank, so their tile counts are exactly
balanced in this path. Correction is always BFP8 even when attention matrices
use another candidate dtype. The helper also handles unsharded matrix storage
by tile-aligning N, but the stage's selected device path must remain TP4.

| Matrix role | Per-device logical K x N | Default Nphysical |
|---|---:|---:|
| Packed MLP gate/up | 5120 x 8704 | 8704 |
| MLP down | 4352 x 5120 | 5120 |
| GDN packed QKV/Z/B/A | 5120 x 4160 | 4608 |
| GDN narrow correction QKV/B/A | 5120 x 2624 | 3072 |
| GDN output | 1536 x 5120 | 5120 |
| Full packed QKV | 5120 x 2048 | 2048 |
| Full separate gate | 5120 x 1536 | 1536 |
| Full packed QKVG, replacing both preceding rows | 5120 x 3584 | 3584 |
| Full output | 1536 x 5120 | 5120 |

Packed QKVG alone saves launches, **zero matrix bytes** under inherited r2/c8
geometry: 2048+1536=3584. It must not be credited with a nonexistent bandwidth
reduction. At r4/c8, split N2048+N2048 and packed N4096 again match, but other
geometry combinations can differ. The roles must be recomputed independently.

| Uniform readers / cores8, BFP8 | GDN read-once matrix bytes | Full read-once matrix bytes |
|---|---:|---:|
| 1 | 118,374,400 | 98,877,440 |
| 2, inherited | 121,159,680 | 98,877,440 |
| 4 | 126,730,240 | 104,448,000 |

Full-model matrix residency is `48*WGDN_resident + 16*WFull_resident`.
Each layer retains the normal matrices **plus** interleaved down/output MMRS
copies (32,030,720 bytes at inherited geometry) **plus** the independently
tile-pair-interleaved prefill SwiGLU copy (47,349,760 bytes). The MMRS copies
inherit the post-geometry physical widths; the SwiGLU copy remains `[5120,8704]`
without decode-geometry N padding. Do not count either inactive resident copy
as a second decode weight read. Packed projections remove the obsolete weights.
Default per-layer matrix residency is GDN=200,540,160 and full=178,257,920 bytes;
all 64 total 12,478,054,400 bytes/device.

Future full-model estimates retain independent BF16 embedding `[62080,5120]`,
BF16 head `[5120,62464]`, and replicated FP32 final norm: 1,275,985,920 bytes.
Head padding is explicitly a stage-4 conservative estimate, not an implemented
stage-5 head. Do not claim complete-model allocation acceptance from this plan.

## Cache, recurrent state, and auxiliary residency

Each full-attention rank owns six Q heads and one KV head of width256.
If the caller table has B rows and P columns, `create_paged_state` computes
`P'=align16(P)` when P>=16, otherwise P'=P. Physical block count is
`N=max(original_table)+1 + B*(P'-P)`. Thus sparse page IDs matter; B*P' is valid
only for a dense contiguous physical pool. Each K or V buffer is
`[N,1,32,256]` BFP8, exactly `N*8*1088` bytes. All16 layers consume
`16*2*N*8*1088`. A page table is ROW_MAJOR `[B,P']` int32; use A, not B*P'*4.

At B1/context262144 with exactly8192 pages: KV=2,281,701,376 bytes/device.
The context harness's extra continuation page rounds8193 columns to8208 and
adds4,456,448 KV bytes across16 layers, plus8192 bytes of table allocations.
Use the actual table/pool for final accounting; S alone is insufficient.

Each GDN state is BF16 `[B,12,128,128]`: **393,216*B bytes/layer**.
All48 recurrent states: 18,874,368*B. Convolution history `[B,3,2560]`
ROW_MAJOR BF16 occupies `A(3B,5120)` per layer: 40,960 at B1 and491,520 at B32.
Its logical payload is15,360*B, so allocated bank slack cannot be called traffic.
Long GDN prefill widens the active layer's state to FP32, adding786,432*B bytes
temporarily while retaining the published BF16 state; it is not a permanent
FP32 state copy in all48 layers.

The helper retains every source-allocated auxiliary tensor: both full-width and
sharded hidden norm vectors (128 of each), full Q/K norms (32), GDN norm (48),
dt_bias/A_log/neg_a vectors (144), tiled and ROW_MAJOR convolution taps (192 of
each), exchange buffers (48), eye/lower/ones (144), masks (48), 513 metadata
scalars per GDN layer, and16 page tables. At B1/context262144 these sum to
177,758,208 bytes/device. Constants are counted per layer because that is how
the current constructor owns them; hypothetical future sharing is not credited.

## Persistent CCL and the reserve

`MultichipDecoder._collective` keys buffers by
`(reduce, logical_shape, dtype, memory_config)`, per decoder. It persists only
packed M<=32 calls. Count **unique warmed signatures**, not calls, norms,
sublayers, or trace replays. Linear reduce-scatter's intermediate shape doubles
the leading axis of the padded input, matching native
`reduce_scatter_minimal_async_op_device_operation.cpp::compute_output_specs`.

For one BF16 width5120 reduce signature `[1,1,B,5120]`, B<=32:

| Buffer | Placement | Physical payload/allocation |
|---|---|---:|
| RS scratch `[2,1,32,5120]` | DRAM | 655,360 bytes |
| RS output `[1,1,B,1280]` | L1 | 81,920 payload bytes |
| Hidden AG output `[1,1,B,5120]` | L1 | 327,680 payload bytes |
| Optional stats AG `[1,1,B,128]` | L1 | 8,192 payload bytes |

L1 numbers are tile payload, not an L1 allocator-fit assertion. Stats are
allocated only when the distributed-norm path actually warms that signature.
The two sublayers normally reuse their same-shape buffers. If a model instance
warms both B1 and B32, their logical signature differs even though padding
matches: count two RS buffers (80 MiB for64 layers), not one. Additional dtype
or memory-placement signatures add their own buffers. With BFP8 CCL the
corresponding RS scratch is348,160 bytes; stats do not automatically inherit
that dtype because the cast applies only at width>=1280. Global CCL/convolution
semaphores are L1 resources, not DRAM traffic or DRAM capacity.

The subsequent placement experiment adds `ccl_decode_memory="DRAM"` for
compact rows<=32. With that policy the three output rows in the table move to
DRAM, while RS scratch remains DRAM. One hidden AG and RS output per layer add
409,600 bytes/layer (26,214,400 bytes across64); one warmed stats AG adds16,384
allocated DRAM bytes/layer where applicable (8192 payload, eight-bank slack).
The helper includes this policy and accepts
separate AG/stats signature counts. Stats count defaults tozero and must be
supplied if that path warms it. These additional output allocations are
capacity; actual DRAM CCL reads/writes still require their own traffic ledger.
`integrated1` leaves the placement override unset.

The inherited **4 GiB** reserve is a planning allowance, not an exact peak:

```
3 * allocated_BF16_hidden[B,S,1280]       # input, retained chunks, concat output
+ 2 * allocated_BF16_rope[1,1,S,64]     # replicated cos/sin, if full attention
+ allocated_FP32_state[B,12,128,128]     # one active GDN layer
+ 32 MiB trace region
+ remainder for chunk intermediates, CCL temporaries, metadata, fragmentation
```

At B1/context262144 the listed terms consume2,114,715,648 bytes, leaving
2,180,251,648 bytes. Flat prefill lowers tile padding between temporal boundaries
but does not remove state/attention padding or the512-token scheduler.
Projection output widths come from selected geometry, and GDN correction
partials are FP32. These tensors, minimal-SwiGLU workspaces, and per-call CCL
scratch still need measured high-water validation; their simultaneous lifetimes
cannot be proved from a tensor-size sum. Persistent RS scratch is accounted
outside the reserve, conservatively avoiding reuse assumptions.

The below short rows use the decoder harness's minimum33 extra tokens:
`P=ceil((S+33)/32)`, then the constructor's P' rounding. One persistent RS
signature per layer, inherited geometry and BFP8 weights:

| Batch / logical context | Physical pages/layer | All16 KV bytes | Resident bytes/device | With4 GiB reserve |
|---|---:|---:|---:|---:|
| 1 / 33 | 3 | 835,584 | 13,991,231,488 | 18,286,198,784 |
| 32 / 33 | 96 | 26,738,688 | 14,623,891,456 | 18,918,858,752 |
| 1 / 513 | 32 | 8,912,896 | 13,999,316,992 | 18,294,284,288 |
| 32 / 513 | 1024 | 285,212,672 | 14,882,398,208 | 19,177,365,504 |
| 1 / 262144, exact capacity | 8192 | 2,281,701,376 | 16,276,283,392 | 20,571,250,688 |

The last row leaves13,788,487,680 bytes below nominal32 GiB. This is ample
planning headroom, not usable-device-memory proof. B32 short-context support
does not promise32 simultaneous maximum-length users: B32 maximum KV alone
would require73,014,444,032 bytes/device. That batch/context capacity envelope
must be handled separately from the per-request262144 capability.

## Traffic and roofline: do not substitute residency

The existing `account_multichip_perf.py` uses stage-4 hardcoded matrix padding
and omits GDN state traffic. Final accounting must consume the final effective
policy and preferably cross-check actual uploaded physical shapes/dtypes in the
result JSON. A decode batch reads each projection's stored weight tiles once
in the compulsory read-once bound; do not multiply weights by B or charge
inactive resident copies. If a kernel rereads weights, show that as actual-path
overhead rather than making a slower algorithm's optimization target easier.

Use separate columns for:

1. **Compulsory/read-once bound**: chosen stored matrix tiles + visible KV tiles
   + GDN recurrent-state read/write. This gives an honest, conservative target.
2. **Source-requested/measured path traffic**: additional tensor reads/writes,
   cache-update tile RMW, reshard spills, intermediate CCL transfers and repeated
   reads, with source or counter provenance. Separate NoC/Ethernet bytes from
   DRAM bytes. Counter-inferred bandwidth is not independently measured bytes.
3. **Resident allocation**: includes duplicates, unused cache pages, bank slack,
   persistent buffers and reserve. These bytes never enter the traffic numerator
   merely because the allocator holds them.

For a GDN decode step, recurrent-state traffic is exactly
`2 * B * 12 * 16 * 2048 = 786432*B` bytes at BF16: `_LOCAL12_READER` reads16
tiles/head and `_fused_delta_row_GDN_COMPACT_WRITER` writes16 tiles/head.
It is25,165,824 bytes at B32 and cannot be omitted. Selected recurrence variants
leave this unchanged; skipped zero fills and constants are L1 work, not reduced
DRAM state traffic. Convolution-history compulsory read/write adds30,720*B
logical bytes if counted; its old/new output copies and all other intermediates
remain separate. The helper's bound deliberately excludes these small terms.

`recurrent_kernel_traffic` additionally counts exact raw-packed recurrence
source-requested Q/K/V/g/beta reads and output writes. For cumulative-exp-scalar,
110 available cores, FP32 g/BF16 beta: B1=813,184 and B32=25,555,264 bytes,
including state traffic. Q/K reuse groups restart at each core partition;
therefore B32 has164 Q/K reads and110 gate-vector reads, not merely128 and32.
Output writes include deterministic tail padding (`align32(12B)*512`). These
are kernel DRAM payload bytes, not memory-bus protocol transaction counts and
not the entire layer. The helper applies only to raw packed BF16 local12;
B1 may use the inherited nonpacked path unless explicitly enabled, so do not
attribute its raw-packed number to B1's default execution.

For full attention, if user i decodes at zero-based position p_i, visible length
is `L_i=p_i+1`, and read-once BFP8 KV tile bytes are
`2 * 8 * 1088 * sum_i(ceil(L_i/32))`. Use independent device positions, not the
allocated table size. SDPA can read key-block padding (32 or512 here), or
duplicate reads across Q-head work; charge that separately when source/profile
proves it. For S513's first decode, visible length514 needs17 pages, whereas
a512-key-block-rounded sweep can touch32 pages. Cache update writes/RMW are
also actual traffic, not all allocated-cache capacity.

At512e9 bytes/s **per chip**, `Tbound_ms = bytes_per_rank / 512e9 * 1000`.
Equivalently divide four ranks' summed bytes by four chips' bandwidth; never
divide one rank's bytes by aggregate bandwidth. Reconcile with the slowest-rank
signposted device window and warmed traced host latency from the same run.

| Kind | B | S before first decode | Read-once + KV/state bound bytes | Bound ms |
|---|---:|---:|---:|---:|
| GDN | 1 | 33 or513 | 121,946,112 | 0.238176 |
| GDN | 32 | 33 or513 | 146,325,504 | 0.285792 |
| Full | 1 | 33 | 98,912,256 | 0.193188 |
| Full | 32 | 33 | 99,991,552 | 0.195296 |
| Full | 1 | 513 | 99,173,376 | 0.193698 |
| Full | 32 | 513 | 108,347,392 | 0.211616 |

These are lower bounds, not measured achieved performance. The task's70%
objective means `Tbound / Twarmed_host >= 0.70` using a declared consistent
numerator. For example, inherited GDN/B32 would need<=0.408274 ms on this bound.
Do not raise the fraction by adding resident duplicates, twice-sized CCL scratch
capacity, idle physical cache, L1 zero fills, or redundant inter-layer movement.
Report an actual-traffic bandwidth utilization separately. Prefill also needs a
compute bound with actual precision/fidelity and chunked reuse; a decode
weight-only byte formula is not a prefill roofline. Tool P150 compute defaults
are not evidence of P300c physical compute efficiency.

## Final handoff

Recompute with the delivered effective per-role policies, actual cache pool and
warmed CCL signatures; retain trace/activation reserve as an estimate backed by
the final device peak. Record a new optimized-multichip section in the context
contract when the final path is accepted, preserving262144 and historical
stage records. Update the final perf accounting to include recurrent traffic
and selected geometry, keep compulsory and actual-path bytes distinct, and
avoid representing this CPU audit as a hardware or full-model validation.
