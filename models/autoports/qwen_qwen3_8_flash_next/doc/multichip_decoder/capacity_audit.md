# CPU capacity audit: four Blackhole chips

This is a planning audit, not an allocation, accuracy, or hardware acceptance result. No TT device was opened. The reference is `../optimized_decoder/selected_decoder_policy.json`, `../../tt/optimized_decoder.py`, and checkpoint revision `812d96e10a0dd82b69d749e68693698cf6c36daa`. The current single-chip context contract advertises 262144 autoregressive positions; its resident PLE single-call 4768 limit is a particular single-chip allocation profile, not a multichip limit to inherit.

## Configuration and representation facts

`../functional_decoder/target_text_config.json` and `checkpoint_inventory.json` give H=2560, four residual streams (10240 values/token), 48 layers: 36 GDN and 12 QSA, 512 experts/layer, top-10, expert and shared width640, Q=24 heads, KV=2 heads, attention head width256. GDN has16 key heads and48 value heads, both width128. QSA's indexer has4 query heads and1 key head at width128, compression4. There is one PLE layer (checkpoint layer1, config one-based `ple_layer_ids=[2]`).

Expert weights contain 48*512*3*2560*640 =120795955200 scalar parameters. TT 32x32 BFP8 tiles occupy1088 bytes and BFP4 tiles576 bytes (`tt_metal/tt-llk/tests/python_tests/helpers/llk_params.py`), so bytes/element are17/16 and9/16, not1 and0.5. All expert dimensions are tile aligned. EP4 stores128 whole experts/chip, with local gate/up `[128,2560,1280]` and down `[128,640,2560]`. TP4 width640/4=160 is only5 tiles, while E=512; both skill criteria favor EP. Full expert replication is physically impossible even at BFP4:67,947,724,800 bytes/chip before other tensors.

PLE has128 checkpoint tables of `[2500012,160]`, i.e.320001536 rows. The existing packed format has20 UINT32 code words plus4 UINT32 scale-ID words per row,96 bytes/row (4-bit codes, group10,16 eight-bit LUT scale indices). Logical packed storage is30,720,147,456 bytes globally or7,680,036,864 bytes/chip under equal sharding. Tile padding depends on whether each chip owns one contiguous packed row range or32 separately padded checkpoint tables; the latter is7,680,163,840 bytes/chip. Tiny FLOAT32 codebook, BF16 scale LUT, and index tensors are extra. Source: `../../tt/packed_embedding.py`, `packed_embedding_cpu.py`. PLE BF16 checkpoint storage is102,400,491,520 bytes; it must not be confused with the packed resident format. Row ownership and remote lookup combination must preserve the existing global IDs and code/scale pairing.

Sensitive baseline weights: GDN QKV/output are BF16 with precise projection, GDN packed z/a/b BF16/HiFi4; GR mixing down/up are FP32; router's exact projection retains FP32 expanded weights; QSA ordered indexer retains BF16 weights. Norm vectors often occupy32 padded tile rows and are FP32. Ordinary QSA, shared, and PLE projections are BFP8. Preserving precision does not imply replicating these matrices: output-dimension sharding of precise projections preserves full local reduction order, whereas row-sharding an exact projection plus CCL can change rounding. Sources: `precise_gdn_projection.py`, `precise_residual_projection.py`, `precise_projection.py`, `ordered_projection.py`, `decoder_components.py`.

## Full text-stack planning budget

Bytes below are per chip. They exclude vision and optional MTP execution. Embedding/head estimates use config shape `[248320,2560]` and BF16 vocabulary sharding; full-model implementation is not part of this stage. GDN packed z/a/b pads local a/b widths to legal tiles (128 global gate slots in this arithmetic). GR is output-sharded, router/indexer replicated, and ordinary weights use TP4. A quarter of the PLE table is resident on every chip. Shared-expert and final-mixer placements are proposed, not measured.

| Resident category | Per-chip bytes |
|---|---:|
| EP4 experts BFP8 (alternative) |32,086,425,600|
| EP4 experts BFP4 (candidate) |16,986,931,200|
| Packed PLE logical payload |7,680,036,864|
| GDN BF16 TP4 projections,36 layers |1,043,988,480|
| GR FP32 TP4 mixing,48 layers |629,145,600|
| QSA BFP8 TP4 projections,12 layers, duplicated KV head groups |167,116,800|
| QSA BF16 indexer replicated |39,321,600|
| FP32 router replicated,48 layers |251,658,240|
| BFP8 shared experts TP4 incl padded shared gate |63,713,280|
| BFP8 PLE projections TP4 |8,704,000|
| BF16 vocabulary-sharded embedding plus independent head |635,699,200|
| Final FP32 GR mixing TP4 |6,553,600|
| One shared RoPE cosine/sine table at262144 |67,108,864|
| Explicit estimate: norms, conv weights, padding, small tensors |268,435,456|
| **BFP4 resident planning subtotal** |**27,848,413,184**|

The preceding stage records34,178,731,008 allocatable DRAM bytes/chip in `../functional_decoder/ple_private_batch_capacity_bound.json`; this is a prior observation, to be rechecked on the selected mesh. BFP8 experts plus packed PLE alone require39,766,462,464 bytes/chip, exceeding that observed capacity by5,587,731,456 bytes before any dense weights, cache, or trace. Thus all-layer resident BFP8 EP4 cannot meet the full-stack contract even at zero context. This does not prove that BFP4 is accurate; it makes BFP4 or another evidenced weight-storage strategy necessary for this hardware.

### KV heads and exact cache arithmetic

TP4 cannot divide two KV heads into0.5 head/chip. With native per-head SDPA, each chip owns6 Q heads, one complete KV head, and duplicates KV groups across chip pairs: ranks0/1 own head0, ranks2/3 head1. This is TP2 for KV memory despite TP4 query/output compute. Local BF16 K and V have `[C/page,1,page,256]`, page32 or another supported tile-aligned size. Twelve layers therefore require12*C*2*1*256*2=12288*C bytes/chip/request: **3,221,225,472 bytes at262144**, not1.5GiB. Padding KV heads2->4 with zeros alone is incorrect unless Q-to-KV ownership is also remapped; duplicate real KV heads instead. Splitting the head dimension needs a different distributed-attention implementation and cannot be assumed to preserve native SDPA.

QSA baseline `qsa.py::make_state` also holds compressed BF16 keys `[1,1,C/4,128]`:201,326,592 bytes across12 layers at max context. FLOAT32 tiled block/token-number arrays occupy32 physical rows: `(C/4+C)*32*4*12` =503,316,480 bytes if retained separately for every request/layer. These arrays are immutable and can eventually share storage, but the conservative budget below does **not** assume that sharing. RoPE sharing across all12 QSA layers is explicitly assumed above; copying the table12 times adds738,197,504 bytes. Local GDN FP32 recurrent state `[1,12,128,128]` across36 layers is28,311,552 bytes/request; local BF16 conv history physical `[1,1,32,2560]` adds5,898,240 bytes. Other small request buffers and allocator/CCL effects are covered only by the reserve below.

| Per-request max-context category | Bytes/chip |
|---|---:|
| BF16 K/V (12 QSA layers, one local head) |3,221,225,472|
| Replicated compressed index keys |201,326,592|
| Unshared padded block/token numbers |503,316,480|
| TP4 GDN recurrence and padded conv state |34,209,792|
| **Request-state subtotal** |**3,960,078,336**|

BFP4 resident subtotal + one max-context request =31,808,491,520 bytes/chip (29.624GiB), leaving2,370,239,488 bytes before runtime reserve. Reserving1GiB for trace, CCL, transient activations, scratch and allocator fragmentation leaves1,296,497,664 bytes. This arithmetic makes262144 batch1 plausible without a context reduction. It is not a fit proof: trace reservation and transient allocations must be measured, precise weights must actually be sharded, and packed PLE placement must not create bank imbalance. A full262144-token10240-wide BF16 residual alone is5,368,709,120 bytes replicated, or1,342,177,280 bytes/chip hidden-sharded; streaming/chunked prefill and bounded live outputs are therefore part of the stack design. Do not infer a resident single-call maximum from this budget.

## Batch planning and MoE execution

With the stated resident subtotal and a1GiB reserve, independently allocating each request's complete capacity gives arithmetic ceilings: batch10 at32768, batch5 at65536, and batch1 at262144. These are estimates, not public maximum-batch contracts or tested capacities. Batch32 at4096 tokens/request fits the same estimate; with a pooled paged cache, batch32 can coexist with a token-budget admission policy rather than reserving32 maximum-length pools. Preserve the ability of a lone request to use262144. Do not lower advertised context just to raise batch count, and do not advertise the arithmetic batch ceiling as measured.

For diverse routing, expected union fraction is `1-(1-10/512)^B`: batch1 1.953%, batch8 14.60%, batch16 27.06%, batch32 46.80%. The25% boundary is reached at batch15. Thus batch32 must use per-token grouped dispatch/fused SwiGLU/combine, not a union-masked `[E,B,H]` dense-per-touched-expert path. EP4 prefill and grouped batched decode are the default plan; indexed batch1 is a separate candidate if it wins measured latency. All-chip replicated experts cannot support the text stack; any batch1 TP candidate must share/repartition EP weights without leaving duplicate full experts resident.

## Existing BFP4 evidence and open validation

`../optimized_decoder/gate4_hf0.json` uses real checkpoint weights at logical33: prefill PCC0.999895394, three decode PCCs0.999926031/0.999924660/0.999806643. `down4_hf0.json` gives prefill0.999907315 and decode0.999947965/0.999904394/0.999861181. Their `_selection.json` files identify separate gate/up-only and down-only BFP4/LoFi candidates. Their real-weight accuracy passes support testing BFP4; they do not validate combined BFP4 gate/up+down, every layer kind, TP/EP output, or long/batched multichip execution. Profiler evidence in `../optimized_decoder/tracy/final_ops_{gate4,down4}/report/` gives representative gate/up77.827us vs BFP8 105.719us and down39.156us vs49.747us, respectively; those separate captures are not whole-layer multichip speedups.

CPU safetensors-header inspection found60 present shard files and71 broken/missing shard targets in the snapshot. Representative layers0/1/3 and some other layers are available; final embedding/head tensors were not present in those readable headers, so their budget uses the config, not a claim of loading them. No whole-model checkpoint download or implementation was started. The stage owner must preserve this distinction when claiming future stack readiness.

Remaining decisive evidence: combined BFP4 real-weight PCC against optimized baseline for all layer kinds; actual rank-local weight/cache allocation sizes; all-layer reserve arithmetic against measured trace/transient peaks; 32K–64K mesh smoke; nonaligned logical lengths; local KV mapping and page permutation/current-position checks; and the selected diverse batch32 grouped-routing path. The multichip context contract should record the retained262144 capability plus these measured scopes, rather than copying the predecessor PLE4768 single-chip restriction.


## Reconciliation with the current candidate (2026-11-13, supersedes the earlier budget)

The current source now owns `MeshWorkspace.expert_output` and accepts one shared workspace at construction; `prefill_forward` allocates one output and writes each chunk directly rather than retaining every chunk; tuple decode concatenates MLP rows. These changes improve the intended stack lifetime contract. Mesh correctness and measured capacity remain unproven. The following numbers describe source-visible storage and a **proposal**, not accepted capability. `context_contract_proposal.json` is separate from the unchanged authoritative `../context_contract.json`.

The first budget assumed shared RoPE and replicated FP32 router weights. Neither matches current construction: each of12 QSA objects owns two full RoPE tables, while `MeshMoe.router` stores output-sharded BF16 weights and widens products at runtime. GR down shards have96 physical output rows (80 valid), and QSA retains an extra `index_qk` matrix alongside the ordered projector. The revised ledger includes these differences.

| Candidate storage, per chip | Bytes |
|---|---:|
| EP4 all48 layers, BFP4 experts, exact tile payload |16,986,931,200|
| PLE code/scale row ownership, each striped array tile padded |7,680,040,960|
| GDN QKV + padded ZAB + exact output weights |1,043,988,480|
| GDN FP32 conv taps, A/dt, BF16 norm vectors incl32-row padding |47,775,744|
| GR FP32 down/up, down96 padding |692,060,160|
| GR BF16 injection and FP32 norm vectors |94,371,840|
| Output-sharded BF16 router, all48 layers |31,457,280|
| TP4 BFP8 shared gate/up/down plus replicated padded scalar gate |66,846,720|
| QSA BFP8 local QKV/output |167,116,800|
| Both BF16 QSA indexer matrices,12 layers |78,643,200|
| QSA FP32 norm vectors |1,179,648|
| **Unshared RoPE tables,12 layers at262144** |**805,306,368**|
| PLE key and full value BFP8, norms and conv |15,564,800|
| Assumed BF16 TP4 embedding/head and final FP32 GR mixer |642,252,800|
| **One shared expert-output workspace** |**82,903,040**|
| Explicit estimate: final norm/mixer and other small static objects |34,603,008|
| **Revised resident planning subtotal** |**28,471,042,048**|

Exact tensor-payload rows above use physical tile geometry; they are not allocator-reserved bank sizes. The last row is explicitly an estimate, and embedding/head/final-mixer placement remains a future stack assumption. PLE value storage includes a correctness-required full `[2560,2560]` replicated matrix. At this inspection `MeshPLE` shards that matrix to640 output columns although inherited `PLE.__call__` consumes2560-wide values and norms. The current narrower allocation is not a legitimate memory saving; the revised ledger reserves the full intended value projection. No implementation repair is made by this audit.

`MeshWorkspace.capacity=16192`, so its BF16 `[16192,2560]` output is82,903,040 bytes/chip. **The caller must pass the same workspace to every serial layer.** Omitting it allocates48 workspaces and adds3,896,442,880 bytes, invalidating the max-context profile. Sharing is safe only when execution is serialized and every live trace acknowledges this deliberately overwritten allocation under a documented ownership invariant; sharing the Python object alone does not establish trace safety.

### Request-state ledger

Let C be page-aligned physical context. Current per-chip batch1 state is:

- K/V: `12288*C` bytes (one complete local BF16 KV head,12 layers).
- Compressed index keys: `768*C` bytes.
- Unshared tiled FLOAT32 token/block-number arrays: `1920*C` bytes.
- Independent ROW_MAJOR page tables: `1.5*C` bytes for page32 across12 layers.
- GDN recurrence+conv state, QSA raw history/positions, PLE conv/history and all1..32 hash input/output/control buffers:34,542,400 bytes. This includes row-major64-byte alignment and tile padding; generic-program metadata remains in the runtime reserve.

Total = `14977.5*C +34542400` bytes. At262144 this is**3,960,804,160** bytes; at32768 **525,325,120**; at65536 **1,016,107,840**. Full requested batches must not multiply262144-state pools unless physically budgeted; paged pool allocation is a future stack ownership decision.

### Construction peaks

`_create_base_state` first invokes the inherited two-head `QSA.make_state`, then replaces K and V separately with one-head caches. At C=262144, construction of one request/layer reaches**402,653,184 bytes above its eventual state footprint** while the first new cache overlaps both old caches. This is temporary but real; construct states before large prefill tensors/traces. Prefer a local-head state allocator to remove this peak without changing logical capability. No need to hold two independently constructed full stacks as a control on the same mesh.

`MeshQSA.build` constructs the replicated BF16 Q/K/V/O matrices before replacing them; the local `base` still references the old matrices until return. Conservatively allow**99,614,720 bytes** temporary duplicated projection payload for the one layer being built. Expert construction directly uploads local weights and does not require a second resident global expert bank. Build layers serially, release source-only device objects, and use one shared scratch before capturing.

Packed PLE repartitioning is host-heavy: the full code payload is25,600,122,880 bytes. The current NumPy linear reorder, all four local parts and concatenation can coexist, creating a source-derived host-array subtotal of**76,800,368,640 bytes** before conversion copies, mmap residency and checkpoint tensors. This is a host-memory construction consideration, not device DRAM or a smaller-context justification. Chunked host repacking/memmap-backed per-rank conversion would bound it if required.

### Long-prefill input/output and runtime scratch

The revised bounded output assembly removes the list of all chunk outputs. It still retains the caller input plus one complete output: `2*C*2560*2 =10240*C` bytes/chip. This equals335,544,320 bytes at32K,671,088,640 at64K, and**2,684,354,560 at262144**. Any retained previous layer outputs or reference tensors are extra. The current input extraction uses a logical slice; if that reintroduces per-start program/metadata growth, replace it with a shape-stable bounded reader before asserting a long-call bound. This audit does not assume zero program-cache growth.

Source-visible scratch to budget separately from resident weights/state includes:

| Buffer/operation | Per-chip physical payload |
|---|---:|
| Grouped dispatch activation `[16192,2560]` BF16 |82,903,040 B|
| Dispatch metadata `[16192,3]` INT32,64-byte aligned rows |1,036,288 B|
| Grouped combine `[32,10,2560]` BF16 |1,638,400 B|
| GR down FP32 product at logical32, block32,K10240 |41,943,040 B|
| QSA merged index-score product (max-context decode or32-row prefill block) |134,217,728 B|
| QSA temporary full FP32 compressed keys at max context |33,554,432 B|
| Selected K/V `[32,1,2080,256]` BF16 pair |68,157,440 B|
| Normalized residual gather `[1,1,32,10240]` BF16 |655,360 B|
| FP32 four-rank block gather `[1,4,32,2560]` |1,310,720 B|

These buffers have different lifetimes and must not all be summed as simultaneous, but neither does their largest member prove the whole peak. Native-op scratch, program-cache buffers, async temporary overlap, trace command streams, CCL and bank fragmentation remain unmeasured. Generic kernel CB allocations are L1 rather than DRAM: e.g. local selected-KV uses40,960 bytes/core, while exact recurrence retains multiple FP32 CBs/core. Measure both L1 program legality and DRAM peak; do not charge L1 CB bytes to DRAM as though they were cache tensors.

Use two explicitly provisional runtime envelopes: **1GiB** (512MiB trace/program cache +256MiB transient ops/CCL +256MiB fragmentation headroom), and a **2GiB stress budget**. Neither is a measured trace-size bound. In particular, a full48-layer trace may exceed an extrapolation from one layer; record isolated trace sizes and an all-layer estimate before calling the profile stack-ready.

| C | Weight+state headroom | Plus full input/output and1GiB runtime reserve | Bounded streaming IO and1GiB reserve |
|---:|---:|---:|---:|
|32768 |5,182,363,840 B |3,773,077,696 B |4,108,622,016 B|
|65536 |4,691,581,120 B |2,946,750,656 B |3,617,839,296 B|
|262144 |1,746,884,800 B |**-2,011,211,584 B** |673,142,976 B|

The last column excludes a small bounded chunk input/output, which must fit within its remaining headroom. A full262144-row resident input plus output **already exceeds DRAM by937,469,760 bytes before any runtime reserve**. Therefore current all-layer placement does not justify a full-context single-call resident-output prefill claim. This is not grounds for reducing262144 autoregressive context: an outer stack driver can stream bounded chunks through all layers and retain only needed outputs, while cache/state preserve all positions. That driver is future full-model work, not an implementation started here.

### Sharing recommendation and capability proposal

Share RoPE tables across the12 QSA layers: this saves**738,197,504 bytes/chip** without changing values, layouts or request state. Share immutable token/block-number arrays across equal-capacity QSA states: this saves**461,373,440 bytes/chip** at batch1 maximum context. Their consumers are read-only; do not share mutable K/V, compressed keys, raw history, positions or request page tables. Later removal of unused index arrays is a separate source audit; sharing is the conservative option.

With both immutable sharing changes, maximum-context weight+state headroom becomes**2,946,455,744 bytes/chip**. After a2GiB runtime envelope,798,972,096 bytes remain for bounded input/output and additional measured peaks. Thus **B1 context262144 is plausible with streaming IO**. Current unshared construction is plausible only under the tighter1GiB envelope; it falls400,598,848 bytes short under2GiB. Recommend both sharing changes before a full-stack fit claim. Even both sharing changes leave a full resident max-context input/output plus1GiB reserve short by811,640,640 bytes; immutable sharing does not eliminate the need for bounded stack IO.

Preserve `hf_advertised_context=262144` and the requested autoregressive-context target. Add a new multichip section with `status=planned_unvalidated`, mesh1x4, local KV geometry, the revised ledger and explicit validation scopes. Keep earlier functional/fused/optimized evidence as historical baseline evidence; do not overwrite their passed fields with multichip results or copy PLE4768 into a new multichip cap. The proposed acceptance smoke is32K or64K from empty state for every layer kind, including a nonaligned tail and final complete public decode; name whether prefix rows execute the whole decoder or only reduced state/attention components. Max-context execution is not claimed by this CPU plan. If the project treats changed local KV-head ownership as the KV-layout-change exception, record that explicitly and schedule the single required maximum-context check; avoid pretending old two-head evidence directly validates the new one-head placement.


## Final source-ledger correction (2026-11-13)

This section supersedes the intermediate construction and memory estimates above. The current source shares one RoPE pair, immutable index arrays by capacity, and one `MeshWorkspace`; it constructs local one-head KV caches directly, retains the full PLE value projection, and bounds host PLE repacking. The detailed source hash, tensor shapes, formats, tile padding, per-buffer eight-bank reservation, and assumptions are recorded in [`memory_capacity_plan.json`](memory_capacity_plan.json). This is a CPU source audit, not a measured full-model allocation or stage-acceptance claim.

The existing **31,203,443,520 bytes/device reservation target remains valid as a planning budget**. Its arithmetic is 27,704,012,800 bytes static planning subtotal plus 3,499,430,720 bytes of B1 maximum-context state payload. The static subtotal includes a 34,603,008-byte allowance; the complete target is therefore not an exact sum of instantiated allocations. Enumerated implemented static buffers require 27,044,171,776 bytes with eight-bank spread; assumed future embedding/head/final mixing requires 642,252,800 bytes; maximum-context state requires 3,502,316,032 bytes with bank spread. Their sum is 31,188,740,608 bytes, leaving **14,702,912 bytes** inside the existing target for additional small static objects. Future stack tensor placement, final normalization, allocation fragmentation, trace storage, and runtime peaks remain assumptions or runtime-budget items. The observed 34,178,731,008-byte allocatable DRAM limit leaves 2,975,287,488 bytes beyond the existing target, or 827,803,840 bytes after a provisional 2 GiB runtime envelope.

PLE tile padding adds exactly **4,096 bytes/device** to its 7,680,036,864-byte logical packed payload; equal eight-bank spread adds another 24,576 bytes. The persistent expert-output `[10240,2560]` BF16 allocation is **52,428,800 bytes** and the complete shared workspace is **54,071,296 bytes**. The repaired combine path's bounded ROW_MAJOR conversion adds transient storage of `32 * min(512, 10*n) * 2560 * 2`: 1,638,400 bytes for one token, 3,276,800 for two, and 52,428,800 for 32. Reducing that conversion to the logical expert prefix does not shrink the persistent workspace. Sharing the workspace across serial layers remains required.

The payload formula `13217.5*C + 34542400` assumes page-table row alignment contributes no extra bytes. At C=32896 or 65664 it undercounts payload by **576 bytes**; it is exact at C=32768, 65536, and 262144. Bank reservation adds further per-buffer padding. Exact B1 state reservations for these capacities are 469,971,456; 903,426,560; 468,130,304; 901,585,408; and 3,502,316,032 bytes, respectively. The JSON lists each profile explicitly. These small differences fit the existing overall allowance and do not invalidate the running reservation experiment.

Both resident residual slabs use the tile-padded logical row count: `2 * ceil(T/32)*32 * 2560 * 2`. Logical length 32769 therefore needs **335,872,000 bytes**, and 65537 needs **671,416,320 bytes**. At the existing maximum-context reservation target plus a 1 GiB runtime envelope, those leave 1,565,673,664 and 1,230,129,344 bytes, respectively. A full resident 262144-row input/output pair still exceeds the budget with that runtime envelope by **782,808,896 bytes**. Preserve B1 autoregressive context 262144 with bounded future stack IO; do not claim full resident maximum-length stack prefill from this ledger. The requested 32K–64K decoder smoke remains compatible with this arithmetic.

The local KV allocator removes the earlier 402,653,184-byte replacement peak. Current QSA construction can still transiently create a duplicate 67,108,864-byte RoPE pair before substituting the shared pair, and the conservative single-layer replicated-projection overlap remains 99,614,720 bytes. Serial construction before large prefill IO and trace capture avoids confusing these peaks with steady state. The previous 76.8 GB simultaneous host-repacking estimate is obsolete because repacking now uses bounded blocks and memmaps.

### Context versus concurrency

B1 advertised context **262144** and the independently measured **B32 capacity 128** layer/trace profile are different claims. They do not imply 32 independent maximum-context caches. For one common capacity C, mutable request state and page tables scale with batch B; immutable token/block indices are shared once. With eight-bank spread, the state reservation is:

```text
B * (34542400 + 132800 + 12288*C
     + 12*ceil((C/32)/8)*8*2048
     + 12*8*round_up(C/8,64))
+ sum(ceil(ceil(W/32)/8)*8*4096 for W in [C/4, C])
```

This assumes all 1..32 PLE hash buffer sets remain available for every request. Each additional distinct capacity cached in the workspace retains another immutable index pair. The following arithmetic ceilings retain the 27,704,012,800-byte static planning subtotal and a **2 GiB runtime envelope**, and exclude full resident input/output slabs. They are admission-budget calculations, not newly validated capability.

| Concurrent requests | Largest capacity, rounded down to 128 | Remaining bytes | Next capacity shortfall |
|---:|---:|---:|---:|
|1|262144 (advertised limit)|824,918,528|Not applicable|
|8|38656|2,362,368|11,826,176|
|16|17920|22,690,816|5,751,808|
|32|7680|4,791,296|52,061,184|

The B1 headroom here is 2,885,312 bytes lower than the headline reservation headroom because this deliberately conservative frontier preserves the entire static allowance and separately charges exact banked state. At B32 capacity 128, full-stack request state would reserve 1,166,491,648 bytes and leave 3,160,742,912 bytes after the 2 GiB envelope. At B32 capacity 262144, request state alone would reserve 110,773,878,784 bytes/device; static plus state would exceed allocatable DRAM by 104,299,160,576 bytes before runtime. A future paged admission controller may allocate a shared physical token pool; no such full-model implementation is assumed here. Measured B32 capacity 128 remains the evidence boundary, while B1 context 262144 remains the retained advertised target.

## Final single-token reducer update

The accepted single-token reducer adds two shared FP32 scalar TILE bounds to
MeshWorkspace. Payload increases8192B and the eight-bank reservation increases
65536B, both once per mesh rather than per layer. Shared workspace payload is
54079488B. The enumerated static banked ledger plus future assumptions now
leaves14637376B of the original31203443520B reservation unassigned. The total
reservation and262144 context target are unchanged. The final long-context
runs `final_mesh_l1_long32769.json` and `final_mesh_l3_long32769.json` pass with
this source under that reservation; actual allocations remain31203453440B and
31203453952B/rank. The JSON memory ledger is authoritative over earlier tables.


## Final runtime evidence reconciliation

The final implementation SHA is `af4699739ad869466bbc825909daf868e8a2f01ed64e0af149602fe6128349c1`. All three final short contracts, both final 32769-token reservation smokes, all three full watcher contracts and all three final segmented profiler captures passed on this source. [`evidence_manifest.json`](evidence_manifest.json) pins the 11 matching run provenances, baseline controls, CPU tests, report exports and excluded captures. This supersedes earlier statements here that mesh correctness or long-reservation validation was unproven; the earlier estimates remain historical records.

The adopted shared ownership bounds occupy 8,192 payload bytes and 65,536 bank-reserved bytes/device. Complete shared workspace payload is now **54,079,488 bytes**; the existing **31,203,443,520-byte** reservation target is unchanged, with **14,637,376 bytes** of its allowance remaining after the enumerated banked allocations and future-stack assumptions. The numerical tables and exact state alignment in `memory_capacity_plan.json` are authoritative for this final source.

Final layer1 PLE/GDN and layer3 QSA complete public mesh prefills used logical length 32769 from empty state, followed by three decodes. Whole-output PCC minima were **0.9998434904** and **0.9991748793**; token history or page tables/positions were exact. Actual pre-IO reserved allocations were **31,203,453,440** and **31,203,453,952 bytes/device**, respectively. This is representative-layer execution with filler allocation, not an instantiated 48-layer model. Layer1 provides shared GDN long-context coverage; no separate layer0 long run is claimed.

CPU admission checks passed six tests and 12 subtests, PLE repacking checks passed, and source precommit checks passed. Watcher and profiler evidence are separate runs. No maximum-context mesh execution, full-stack trace-size measurement, maximum batch/context Cartesian product, full-model generation or vLLM capability is added. Advertised B1 context 262144 remains retained with the recorded memory/streaming assumptions. Independent stage review, final documentation precommit and the local acceptance commit remain pending.
