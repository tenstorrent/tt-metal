# Direct prefill QKV movement: bounded non-aligned adaptation

CPU-only, unapplied proposal. Owned artifacts: `qkv_movement28.patch`, `qkv_movement28_prepare.py`, `qkv_movement28_cpu.json`, `qkv_movement28_before.csv`, this audit. No production edits, device/build execution, TTNN or Torch import. Python AST parsing and `git apply --check` pass. Native C++ compilation and hardware behavior are unverified.

The now-completed `prefill_qkv26_perf.json` passes correctness but establishes the non-aligned movement regression. Selected whole-boundary traced medians (microseconds):

| B/S | Native split/gate | Original direct110 |
|---|---:|---:|
|1/33|20.318|57.535|
|1/512|39.341|33.732|
|1/513|43.397|306.428|
|3/512|100.869|82.996|
|17/512|498.672|429.645|
|32/33|295.928|1104.902|
|32/512|933.460|797.831|
|32/513|2683.947|13211.941|

All12 cases and both worker counts are retained in the before CSV. No proposed speedup is claimed.

## Source diagnosis and changes

The old kernel gates eight-tile batching on **global `S%32==0`**. Thus S513 sends even its individually aligned full-live output tiles—every full gate tile and aligned users—through one read/write barrier per tile. Actual shifted tiles execute1024 volatile16-bit load/store selections, including integer tile-address math for each element, even when face-row runs are contiguous.

The patch keeps that exact old `KERNEL` and default API path as `implementation="reference"`. Two independently selectable alternatives are added:

- `batch_tiles`: detect alignment **per work item**, group up to eight consecutive aligned full-live items for native-style reads/barriers/writes regardless of public S, otherwise group up to four mixed output tiles. Read complete one/two source tiles into distinct slots. Assemble shifted rows using contiguous **32-bit integer words**, unrolled four words at a time, splitting only at source/destination16-row face boundaries. Write zero only for destination dead rows; preserve both16-bit payloads without arithmetic/casting.
- `face_fragments`: identical worker ownership, batching and32-bit assembly, but read only source-face spans intersecting the live row interval. Each source-face span is expanded to **64-byte/two-row boundaries** and read once. This cuts unused source bytes, at the cost of more NoC transactions, so it is an independent measured candidate rather than an assumed improvement.

No weight, norm, RoPE, cache, layout contract or main default is changed. The wrapper still consumes BF16 DRAM `[1,B*S,3584]` and returns owning Q `[B,6,S,256]`, K/V `[B,1,S,256]`, gate `[1,B*S,1536]`. All aliases/shapes/dtypes/mesh guards are retained. Public non-aligned B/S remain supported. Existing native input/gate slicing remains the independent comparison path in the probe.

The proposed probe adds `--implementations reference batch_tiles face_fragments`, with every implementation × requested core count timed/checked alongside native. Original poison fixture, per-user/per-rank exact checks, changed input, rebinding, queued replay, preallocated banks/outputs, allocation positive control and rotated timing order are reused unchanged. The source reference kernel is still available in the same process; default production callers retain it until explicit later selection.

## Alignment, ownership and memory proof

Blackhole's `internal/tt-1xx/blackhole/noc/noc_parameters.h:379` declares DRAM-read alignment64 bytes; Wormhole's32-byte constant is not applicable. `internal/debug/sanitize.h:628–663` derives the target-dependent mask and checks the local address against that requirement. `api/tensor/noc_traits.h:137–153` supplies explicit page/byte-offset addressing.

This proposal **does not issue arbitrary32-byte face-row DRAM reads**. Every fragment's source offset, local destination, and byte count is a multiple of64; the tensor tile page is2048 bytes. A live row at odd local index is rounded down one row, and the end rounded up one row, within the same512-byte face. Extra rows can contain poison and are intentionally ignored by assembly. Fragments use distinct source-face/tile regions, so there are no overlapping outstanding local writes, even when two destination-face segments use the same source face. Full-tile writes remain2048 bytes from64-aligned slots. Source and destination face-copy addresses are at least4-byte aligned; each run is a multiple of eight32-bit words.

Four source slots consume4×4096=16384 bytes; four owning assembly slots consume4×2048=8192 bytes, followed by ≤63-byte alignment slop. Scratch grows from16,448 to**24,640 bytes/core** (extra8192). The aligned fast path reuses the first16384 bytes after the previous write barrier. Four mixed slots and their outputs remain disjoint, all reads drain before CPU access, and all writes drain before reuse. No extra output allocation or length-indexed persistent cache is introduced. The additional L1 footprint needs watcher/live-layer evidence before selection.

`qkv_movement28_cpu.json` records:

- All1024 phase0..31/live1..32 combinations,524,288 exact uint32 payload comparisons, and exactly one owner for every output word, including zeroed dead rows.
- 4032 non-overlapping fragment reads; every source/destination/length is64 aligned and stays inside a physical source tile; every copied byte was fetched.
- All64 possible scratch-base alignment residues fit the declared CB.
- 4864 partition/batching plans over B1..32, S1..64 plus127/128/129/255/256/257/511/512/513/1023/1024/1025, core counts48/110:31,441,536 output tiles with exact interval coverage. Higher S repeats the same local mapping; existing uint32 geometry guards remain unchanged.

The integer payload pattern exercises both16-bit halves independently, including sign/NaN-like patterns. Hardware poison/readback tests remain mandatory because CPU proof cannot establish compiler codegen, device cache ordering or NoC acceptance.

## Serialized comparison plan after review/application

Keep all old artifacts. Run smoke with fresh filenames, watcher and allocation tracking, no timing:

```bash
TT_METAL_WATCHER=1 TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 python -m models.autoports.qwen_qwen3_6_27b.tests.probe_prefill_qkv --batches 1 3 --seqs 33 513 --cores 110 --implementations reference batch_tiles face_fragments --steps 3 --queued 50 --no-timing --output models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/qkv_movement28_smoke.json
```

Then all B1..32 at S31/33/512/513, and B1/3/17/32 with all-residue coverage, retaining unchanged bitwise/finite/per-user gates. Check all four outputs, physical padding, source poison coverage, rebound addresses and exact eager/trace results. A first compilation/API/alignment failure is a repair task; retain the failing variant's source and diagnostic artifacts before adapting it.

Uninstrumented matched perf, after correctness:

```bash
TT_METAL_TRACE_ALLOC_TRACKING=0 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 python -m models.autoports.qwen_qwen3_6_27b.tests.probe_prefill_qkv --batches 1 3 17 32 --seqs 33 512 513 --cores 48 110 --implementations reference batch_tiles face_fragments --steps 3 --queued 50 --samples 7 --replays 50 --output models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/qkv_movement28_perf.json
```

Unset watcher/profiler for timing. Capture source/native provenance. Compare the full native split+gate boundary to original and both adapted candidates; do not time only aligned subtiles or exclude assembly. If fragments lose, its read-count/byte evidence can reject only that transport variant; keep the batched-word-copy family independently. Once a winner passes, explicitly integrate/select its implementation and verify whole-decoder S33/S513 plus aligned S512 using the unchanged reference gates. Default production calls remain `reference` in this proposal, so standalone candidate numbers cannot be presented as default-path numbers.
