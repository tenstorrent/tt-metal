# Selected tensor storage and context accounting

This is **source-derived tensor payload accounting**, not a measured allocator
peak. It applies to the source snapshot SHA
`5ff391a2efb6096e7d9eaa499c6d19a8548cf9a60d108425e773d4ded72ee898`, one device,
TP=1, 128 experts, hidden width H=2816, expert width I=704, and shared width
J=2112. The selected policies are recorded in `final_policy.json`.
[The v8 source delta](source_delta_v8.md) selects full QKV M2 and directly places the existing final weighted input-normalization result in L1. Resident checkpoint/cache/RoPE payload delta is zero; no setup tensor is added. [Current validation](validated_v8_validation_summary.json) passes12 primary plus five boundary commands. Full maximum/near-maximum references are v8; sliding retains the exact v5/v6/v7/v8 inheritance map. Current native profiles and CPU accounting pass; payload calculations remain separate from allocator peaks and final review.


Earlier CPU arithmetic checks recomputed all listed resident weight/cache and
indexed-intermediate payloads. The review fixes introduce no tensor at setup
and add one small host SDPA config. Final sliding prefill skips its unused
private K/V tail copies; this is described separately below and changes none
of the resident payload totals. Sliding BFP8 prefill gate still aliases decode;
indexed experts still retain all 128 checkpoint experts. ROW_MAJOR RoPE changes
layout, not aligned payload. Program workspaces, traces and temporary tensors
remain outside the counted resident subtotals.

All counted tensors use standard 32x32 tiles. Metal's
`tt_metal/api/tt-metalium/tt_backend_api_types.hpp::tile_size` defines BF16 as
2,048 bytes, BFP8 as 1,088 bytes, and BFP4 as 576 bytes per tile. The latter
two include 64 shared-exponent bytes; counting nominal 8/4 bits alone would
underestimate storage. All listed logical matrix dimensions are tile aligned.
Bytes below include the explicit shared-decode padding and count aliases once.
MiB=2^20 bytes; GiB=2^30 bytes.

## Routed expert weights

Packed gate/up is `[1,128,2816,1408]`: 507,510,784 elements or 495,616 tiles.
Down is `[1,128,704,2816]`: 253,755,392 elements or 247,808 tiles.

| Unique allocation | Sliding bytes | Full bytes |
| --- | ---: | ---: |
| Packed gate/up, shared by prefill and decode | 539,230,208 (BFP8) | 285,474,816 (BFP4) |
| Down, shared by prefill and decode | 142,737,408 (BFP4) | 142,737,408 (BFP4) |
| **Resident expert weights** | **681,967,616 = 650.375 MiB** | **428,212,224 = 408.375 MiB** |

Sliding previously retained a separate 285,474,816-byte BFP4 prefill gate.
The BFP8 prefill repair shares the existing BFP8 decode gate, so current
resident expert payload is `967442432 - 285474816 = 681967616` bytes, a
272.25 MiB reduction in these allocations. Full attention is unchanged.
Exact phase-alias source statements and CPU checks are recorded in
`final_policy.json::historical_factory_resolution_proofs` (v4 source; the same underlying alias code is retained).

`OptimizedExperts` explicitly aliases matching phase dtypes. Its active
prefill mode does not retain the original `PackedExperts` object, and the
original BF16 gate/up/down references are replaced. BF16 copies used while
constructing the decoder are setup transients, not included here. The
all-BF16 gate/up/down payload would be 1,522,532,352 bytes (1,452 MiB).
Top-8 routing reduces per-step work; it does not reduce resident storage for
the complete 128-expert checkpoint.

Both attention kinds use decode gate/up K block 44 and down K block 22;
routed-expert prefill retains K block 11. These program tiling choices do
not change tensor shapes or storage dtypes. Their kernel circular-buffer
requirements are excluded from this payload calculation.

## Indexed decode expert intermediates

The generalized router retains its existing device top-8 index view. Both
sparse expert matmuls consume those indices; post-scale routing weights are
gathered in the same order. The expert outputs now have eight slots. The
weighted merge remains a matmul with K block 1 and inherited output geometry;
its eight logical weights pad to 32. GELU is fused into gate/up multiplication.
No checkpoint weight is truncated to eight experts.

| Individual BF16 tensor | Indexed padded shape | Indexed bytes | Expanded E=128 bytes |
| --- | --- | ---: | ---: |
| Packed gate/up output | `[1,8,32,1408]` | 720,896 | 11,534,336 |
| Activated hidden | `[1,8,32,704]` | 360,448 | 5,767,168 |
| Down output | `[1,8,32,2816]` | 1,441,792 | 23,068,672 |

These are individual padded tensor payloads, not simultaneously live sums,
allocator peaks, or measured traffic savings. Indexed selection and GELU
fusion do not change the resident expert-weight table above. Their buffers
and kernel workspaces are outside the resident-weight subtotal.

## Shared MLP and attention weights

Shared prefill intentionally delegates to its retained BF16 implementation.
Its packed gate/up `[2816,4224]` occupies 23,789,568 bytes; its down
`[2112,2816]` occupies 11,894,784 bytes: **35,684,352 bytes (34.03125 MiB)**
for either attention kind. `FusedSharedMLP` replaces the original down
closure with its owned BF16 weight and clears `source.down_proj`, so this is
one retained down allocation, not two.

The selected shared-decode DRAM layout assumes the tested eight-bank device.
`DecodeLinear` pads N to `ceil(N/(8*32*6))*(8*32*6)`, independently of reader
count. Packed gate/up therefore uses `[2816,4608]`, and down `[2112,3072]`.
Changing from one to two readers does not create another weight copy.

| Unique allocation | Sliding bytes | Full bytes |
| --- | ---: | ---: |
| Shared decode padded gate/up | 13,787,136 (BFP8) | 7,299,072 (BFP4) |
| Shared decode padded down | 6,893,568 (BFP8) | 3,649,536 (BFP4) |
| Shared MLP total, including BF16 prefill | **56,365,056** | **46,632,960** |
| Packed QKV weight shared across phases | **24,510,464** | **27,574,272** |
| Output projection weight shared across phases | **12,255,232** | **24,510,464** |

Sliding QKV stores `[2816,8192]` BFP8. Full QKV stores `[2816,9216]` BFP8:
Q plus tied K, with V restored by concatenation after projection. The full
cache still has separate K and V arrays. Output weights have shapes
`[4096,2816]` and `[8192,2816]`, respectively, both BFP8.

`DirectQKV` aliases the packed prefill/decode QKV weight. Its retained source
objects do not imply retained BF16 QKV weights: the factory replaces those
weights, sets broadcast `rows=None`, and sets `lane_mask=None`. The earlier precision
candidate JSON files confirm all cleanup flags. Relative to keeping those
three particular redundant allocations, cleanup removes an FP32 transpose,
a duplicate BFP8 phase weight, and a tile-padded BF16 `[32,2816]` lane mask:
116,965,376 bytes sliding / 131,563,520 bytes full. These savings exclude
other setup transients and do not claim a measured allocator-peak change.

Direct router projection aliases the original BF16 projection weight;
it does not upload another weight. The wrapper still retains its original
router, including the FP32 broadcast projection rows. No router-weight
cleanup saving is credited here. Router weights and gate buffers remain
outside the weight subtotals above. Sliding uses a `(4,1)` program with K22
HiFi4; full uses the same grid with K44 LoFi. Both retain FP32 input,
accumulation, and output with approximation and packer accumulation disabled.

## Attention temporaries and program policy

Native decode attention now stays BF16 through head concatenation and the
output-projection input. The removed BF16-to-FP32 promotion changes transient
payloads, not resident weights or cache allocation. A tile-padded native
output `[1,1,32,D]` occupies 16,384 bytes for sliding D=256 or 32,768 bytes
for full D=512 in BF16; FP32 would double those payloads. A tile-padded
concatenated `[1,1,32,16*D]` tensor occupies 262,144 or 524,288 bytes in BF16,
also half its FP32 payload. These are individual tensor-size calculations,
not measured traffic or peak-memory savings. Output projection still emits
FP32 into L1 and preserves the FP32 residual path.

Prefill QKV now uses `MinimalPrefillProjection` with K8 sliding/K16 full,
FP32 input/output, BFP8 weight, HiFi4 sliding and HiFi2 full compute. Full prefill output
uses the same helper with K8, BF16 input, BFP8 weight, LoFi and FP32 output.
The helper retains the selected weight and compute objects; the QKV wrapper
retains the existing decode callable and its phase alias. Neither backend
uploads or creates a weight/cache tensor. Full minimal-QKV output inherits L1 from input; sliding QKV and the separate full attention output projection retain DRAM output. Full tied K/V duplication retains its logical shape and returns DRAM after an L1 slice. Sliding output
retains the previous multicast2D grid `(11,8)`, K16, LoFi and DRAM input/output.

Each selected minimal role constructs four host config objects at setup,
N8 and subblock1×4. Full QKV Mblocks are1/2/2/2; other minimal roles retain1/2/3/4. Runtime indexes by `min(4, padded_M_tiles)`; there is no maximum-length-sized config allocation.
Sliding's unchanged multicast output still builds its 32 row-specific
configs. Program metadata and temporary buffers remain outside resident
weight/cache subtotals.

The minimal factory explicitly double-buffers A, B and output, and allocates
one intermediate block. At selected Mblock4 (sliding QKV/full output) or Mblock2 (full QKV), N8, BFP8 weights and FP32 intermediate/output give the following **per-core CB payload estimates**:

| Selected minimal role | A CB bytes | B CB bytes | Output + intermediate CB bytes | Total |
| --- | ---: | ---: | ---: | ---: |
| Sliding QKV, FP32 input, K8 | 262,144 | 139,264 | 393,216 | 794,624 |
| Full QKV, FP32 input, M2/K16 | 262,144 | 278,528 | 196,608 | 737,280 |
| Full output, BF16 input, K8 | 131,072 | 139,264 | 393,216 | 663,552 |

These are separate sequential projection programs, not simultaneous
allocations or decoder memory peaks. Kernel code, runtime arguments,
semaphores and other live tensors are additional. Smaller M reduces A/output/
intermediate buffers, not the B buffer. The estimates follow
`minimal_matmul_program_factory.cpp:309–355`; alternative weight dtypes change
B tile bytes. No padding tensor is uploaded for K16 with H/32=88: the kernel
rounds its K loop to 96 tiles and handles the partial block. The archived v5 maximum/near-maximum artifacts at 169c0d97 pass all 291
sampled rows: minimum 0.995130377831 sliding and 0.996084490295 full.
Maximum/near-maximum coverage is retained at v5 hash169c0d97 through the exact source delta: checkpoint/cache payloads and normal full-chunk math are unchanged. Current b585a21f regressions exercise both capacity-selection branches and final-tail request reuse; maximum contexts were not rerun on v6.
These are B1 context checks and do not establish simultaneous B32 maximum
context fit. They do not measure an allocator peak.

Both kinds retain all sharded norm sites for one-row hidden-width decode;
full multi-row input normalization changes only final weighted-output placement to L1; its arithmetic and other norm sites remain unchanged. No allocator-peak
change is inferred from program, backend or norm-site selection.

Full attention's later paged-prefill chunks use Q64/K256 on `(8,4)` with
HiFi2, FP32 destination, full destination sync, and disabled math/exponent
approximation and packer accumulation. The outer chunk is 1,024 tokens.
The first full-prefill chunk retains its regular SDPA program; sliding
prefill uses its regular windowed program with LoFi. These compute settings
do not alter any cache or weight payload above; their workspace and circular
buffers are excluded. Q/K environment overrides can change program geometry.

## Review delta: final private prefill tail

V6 passes `retain_prefill_tail=False` only for the final fresh-prefill chunk.
After that chunk's attention is computed, the sliding branch assigns
`self.tail=None` instead of cloning K and V. Nonfinal chunk history remains
unchanged; public decode and prefix continuation consume the paged cache.

Each removed BF16 clone is `[1,8,T,256]`, with
`T=min(1024, physical_K_rows)`. The pair payload is `8192*T` bytes, up to
**8,388,608 bytes (8 MiB)** at T=1024. Short single-chunk inputs use 32-rounded
physical rows; later short sliding chunks already pad physically to a full
window, so their eliminated pair also reaches 8 MiB. Full attention has no
sliding tail allocation. This removes two clone calls and a private per-request
stash, not resident checkpoint/cache storage or a measured allocator peak.

The full-attention boundary program is an additional setup config: default
Q64/K128, same grid/compute/dtypes as Q64/K256. It is selected only if rounded
primary reads would exceed logical page-table capacity. It can create another
cached device program and has different transient SDPA buffers; no CB-byte
saving is assumed without native/factory accounting. Weight/cache tensor
shapes, precision and requested capacity remain unchanged.

## Caller-owned RoPE table layout

`OptimizedDecoder.decode_rope_layout` advertises ROW_MAJOR. The headline,
batch, request-reuse, and long-context runners read this property at setup;
decoders without it retain TILE. Separate prefill tables remain TILE, and
legacy caller-provided TILE decode tables remain accepted. No forward-time
host conversion or arbitrary-content table cache is introduced.

For aligned extent C and head width D, each BF16 decode table occupies
`C * D * 2` bytes under either layout. The cosine/sine pair at C=5120 is
5,242,880 bytes sliding (D=256) or 10,485,760 bytes full (D=512). The ROW_MAJOR
producer avoids the embedding operation's TILE-weight untilize; this is a
source-level movement change, not a measured current-snapshot latency claim.
RoPE buffers remain excluded from the resident-weight/cache subtotals.

## Caller-owned paged K/V cache

For B slots, each with physical extent C tokens, the two cache tensors are
`[B*C/32, Hkv, 32, D]`. C must cover the logical context and required SDPA
read padding; the runners round it to at least 128-token boundaries.
Sliding uses Hkv=8, D=256; full uses Hkv=2, D=512.

For BFP8, the exact pair payload is
`2 * B * (C/32) * Hkv * (D/32) * 1088` bytes. BF16 substitutes 2048.
For unequal slot capacities, replace `B*C/32` by the total physical page
count. Per 32-token physical page, the K/V pair is 139,264 bytes sliding or
69,632 bytes full in BFP8.

| Capacity | Sliding BFP8 pair | Full BFP8 pair | Sliding / full BF16 pair |
| --- | ---: | ---: | ---: |
| B1, C=262144 | 1,140,850,688 (1.0625 GiB) | 570,425,344 (0.53125 GiB) | 2,147,483,648 / 1,073,741,824 |
| B1, logical S=262143, C=262144 | 1,140,850,688 | 570,425,344 | Same as previous row |
| B32, C=128 per slot | 17,825,792 (17 MiB) | 8,912,896 (8.5 MiB) | 33,554,432 / 16,777,216 |
| B32, C=262144 per slot | 36,507,222,016 (34 GiB) | 18,253,611,008 (17 GiB) | 68,719,476,736 / 34,359,738,368 |

The public API uses absolute paged cache positions, including for sliding
attention. A 1,024-token attention window limits reads but does not make the
caller allocation a 1,024-token ring cache. B32 at short context and B1 at
maximum context are distinct capacity tests; their combination does not
establish simultaneous B32 maximum-context fit. The parent owns the final
context-test outcome and supported aggregate-capacity statement.

These counts exclude router/norm/position buffers, caller activations and
RoPE tensors, sliding prefill tails, temporary activations, kernel circular
buffers, trace/program storage, allocator metadata, and construction peaks.
For scale, one BF16 `[1,1,262144,2816]` caller tensor alone is 1,476,395,008
bytes (1.375 GiB), and chunk outputs can coexist with concatenated outputs.
The tables therefore must not be interpreted as total device memory needed
for the maximum-context invocation.

Source basis: `OptimizedExperts`, `DirectQKV`, `MinimalPrefillQKV`,
`MinimalPrefillProjection`, `DecodeLinear`, and
`OptimizedSharedMLP` in `tt/optimized_decoder.py`; `PackedExperts`,
`FusedSharedMLP`, and `TiedQKV` in `tt/fused_decoder.py`; the imported
`SharedMLP` and expert weight loader; and Metal's tile-size definitions.
No hardware query or device execution was used to calculate these values.

## V7 fidelity-only storage delta

Full minimal-prefill QKV uses HiFi2 with the same FP32 accumulation/output, BFP8 weight, DRAM allocation, grid11×8 and K16 program. Sliding retains the complete HiFi4 compute values. Setup creates a host compute-config object and changes no resident checkpoint/cache/RoPE or retained shared-prefill/QKV storage. All byte totals above remain unchanged. Native transient program/CB cost is outside those resident subtotals. The archived v7 full maximum-context gates pass; inherited sliding maximum-context records retain their original hash in [the context contract](../context_contract.json).

## V8 producer placement and M2 workspace delta

The full minimal-QKV input is the existing final learned-weight multiply output of input normalization. Its FP32 payload is `physical_rows * 2816 * 4` bytes:11,534,336bytes (11MiB) for1024 rows, or1,081,344bytes for96 physical rows. V8 writes this tensor directly to interleaved L1 instead of DRAM; it does not allocate a persistent tensor or add a DRAM-to-L1 copy. These are whole-tensor payloads distributed across L1 banks, not per-core allocations or measured peaks. Sliding retains the original placement.

M2/K16/N8 changes per-compute-core QKV CB payload from1,196,032 to737,280bytes (−458,752bytes): A double-buffer262,144; B double-buffer278,528; output double-buffer131,072; intermediate65,536. The original M4/input-L1 collision (CB end1,307,648 above live input start1,114,112) is resolved by the smaller workspace. Other live tensors and allocator/runtime metadata are excluded, and the source estimate is not a whole-decoder peak. All current maximum contexts, tight capacity, tails and allocation-tracked reuse pass; full reuse keeps379 cached programs while the trace is live. Resident weight/cache totals above remain unchanged.

### Full QKV downstream temporary placement

The None minimal output request inherits input memory. Current native rows show full packed FP32 QKV[1,1,1024,9216] inL1 (37,748,736bytes/36MiB) and its tiedKV slice[1,1,1024,1024] inL1 (4,194,304bytes/4MiB). The following concat emits[1,1,1024,10240] intoDRAM (41,943,040bytes/40MiB). Sliding minimal-QKV output remainsDRAM. These values supplement the11MiB normalized input and per-core CB inventory; they are individual temporary payloads, not a summed live allocation or peak. Thus v8 moves input, packed output and slice intoL1 while preserving resident bytes and the concat boundary. Max/near-max, tails and reuse pass on the actual selected path.
