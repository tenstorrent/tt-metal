# Direct compact-packed prefill QKVG movement

2026-11-13. New isolated `tt/prefill_qkv_candidate.py` and
`tests/probe_prefill_qkv.py`; no main/shared-helper changes, TTNN/Torch imports,
device run or build. Native compilation, exact hardware results and performance
are unverified. This implements item2 of `remaining_bottleneck_audit25.md` as a
movement-only family, with no normalization, rotary, cache or weight changes.
Joint convolution remains frozen at its v4 source; its audit now records the
parent's exact B1/B3 watcher/tracker/queued smoke pass and v3/v4 causal evidence.

## Actual input and coherent boundary

`MultichipDecoder._qkv` receives flat projection output `[1,B*S,3584]` when
called from `flat_prefill._attention_flat(...logical_shape=(B,S))`. It currently
slices compact gate and QKV, materially reshapes QKV to independently padded
`[B,S,2048]`, then calls `split_query_key_value_and_split_heads` with six Q heads,
one K/V head and no K transpose. Output norms and partial RoPE are later calls.
The existing flat gate stays `[1,B*S,1536]`.

The helper consumes that actual BF16 TILE DRAM flat allocation and returns:

| Output | Logical shape | Physical shape |
|---|---|---|
| Q | `[B,6,S,256]` | `[B,6,ceil(S/32)*32,256]` |
| K, V | `[B,1,S,256]` | `[B,1,ceil(S/32)*32,256]` |
| gate | `[1,B*S,1536]` | `[1,ceil(B*S/32)*32,1536]` |

All outputs independently own BF16 TILE DRAM allocations. There is no
intermediate padded `[B,S,2048]`, restore to it, or material section slice.
The helper supports every positive logical S and public B1..32 within uint32
page indexing; it owns sequence/user padding. No aligned public-input contract
is introduced. Different effective lengths in a common padded logical S are
still handled by the existing downstream masks; the mover preserves all
logical input rows and does not invent a new ragged interface.

Signature:

```python
q, k, v, gate = prefill_qkv(
    packed, batch=B, seq=S,
    outputs=prepared_four_tensor_tuple,  # optional allocation convenience
    cores=110,
)
```

No persistent owner state, closures, weights or mutable source cache exist.
The four outputs must be distinct from the input and each other, with exact
logical/physical shape, dtype, memory and same-mesh placement. The caller keeps
preallocated outputs alive for trace lifetime; the function overwrites every
physical output tile, including zero padding, on each invocation.

## Addressing and native comparison

The native runtime-rows head-shuffle source in
`ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/device/kernels/dataflow/`
uses disjoint `(batch,head,tile-row)` ownership and full head widths. Its input
layout assumes each user is independently tile-padded; runtime_rows does not
remove that assumption. This helper instead computes source logical row
`user*S + output_tile_row*32` with packed input stride112 tiles. Q columns are
0..47, K48..55, V56..63 and gate64..111. Q/K/V output tile order is native
`(user,head,sequence_tile,column_tile)` order; gate keeps flat-token tile order.

A single dataflow kernel per worker performs only NoC transfers and uint16
copies. It launches no compute/TRISC arithmetic. Balanced contiguous work
intervals cover every output tile exactly once, even when a worker crosses
Q/K/V/gate boundaries.

For S divisible by32, every input/output tile is aligned and complete. Eight
tiles are transferred per read/write barrier, following the native bulk-movement
pattern. For nonaligned S, a destination tile may draw rows from two input
tiles. The reader fetches at most those two full tiles, copies only its live
rows through the four-face tile map, and writes zeros for dead rows. The final
flat gate tile is also masked. No read crosses a source tile page; the second
read is issued only when live rows cross its boundary and is bounded by the
actual compact input allocation. Full-tile reads are2048 bytes and the private
scratch pointer is explicitly64-byte aligned. Scratch is16,448 bytes/worker,
including alignment slack; the general two-tile merge needs only the first
6,144 bytes of that same allocation.

This first nonaligned implementation deliberately uses complete source-tile
reads instead of unaligned small DRAM transfers. Its worst case reads two tiles
for one output tile. That is an honest adaptation cost, not proof of a speedup.
S33 and S512 need separate complete-boundary timings. If the nonaligned reader
is limiting, adapt to coalesced aligned face-row segments with the same owner
and zeroing contract; do not restore the old per-user padded tensor or reject
on a first transfer/API error.

## Probe contract and CPU evidence

The native oracle is the exact current gate/QKV slices -> native material
reshape -> `split_query_key_value_and_split_heads` sequence, without changing
its output allocation behavior. An independent Torch reshape/permutation of
the source fixture supplies the bitwise movement oracle; it is not an HF math
oracle. Native and candidate must each equal it for Q/K/V/gate, every rank and
user. Signed zero, distinct user/rank markers, sequence edges and changed input
banks make source/owner swaps visible.

Input physical rows after B*S contain preserved NaNs. An explicit raw-tile copy
installs them in a tensor with the actual compact logical shape; physical
readback verifies all input bits and poison counts. Candidate output padding
must be zero, and input bits remain unchanged. All candidate input/output banks
are allocated before capture. Both input and output addresses are rebound in
eager tests; changed-input trace replay and50 queued steps follow. Allocation
tracking1/skip0 has a deliberately unsafe allocation positive control. The
probe always appends every per-user comparison, including nonfinite failures,
and requires16*B records per path. Nonfinite/constant PCC cannot prevent the
first-failure tensor dump, and cannot relax exact/finiteness/.995 gates.

CPU checks cover all B1..32 and S1..65 plus127/255/511/512/513:
8,960 ownership plans,8,672,112 output tiles and28,067,712 live/dead row-boundary
checks. An additional1,048,576 integer-copy checks cover every phase0..31,
live-count1..32, face and output element, including zero padding. Seven host
negative cases, scratch alignment and the diagnostic-loop AST guard pass.
The CPU command imports neither TTNN nor Torch. Python AST/Black checks pass.
These are host/source proofs, not hardware validation.

## Parent commands and remaining integration gates

Use fresh artifact names and the parent's established serial device environment:

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.probe_prefill_qkv --cpu-check

TT_METAL_WATCHER=1 TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
  python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.probe_prefill_qkv \
  --batches 1 3 --seqs 33 512 --cores 110 --no-timing \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/prefill_qkv_smoke25.json

TT_METAL_WATCHER=1 TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
  python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.probe_prefill_qkv \
  --all-batches --seqs 1 3 31 33 511 512 513 --cores 110 --no-timing \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/prefill_qkv_wide25.json

env -u TT_METAL_WATCHER -u TT_METAL_TRACE_ALLOC_TRACKING -u TT_METAL_DEVICE_PROFILER \
  python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.probe_prefill_qkv \
  --batches 1 3 17 32 --seqs 33 512 --cores 48 110 \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/prefill_qkv_perf25.json
```

`--all-residues` additionally includes S33..63; it can be combined with a bounded
B1/3/17/32 set for runtime coverage of every sequence residue. The full wide run
above is large; smoke must first compile and pass. Timing measures the entire
packed-output-to-final-heads-and-gate boundary, including native intermediate
allocations captured by its original APIs and all four candidate output writes.
It rotates family order and never times with watcher/profiler/tracker enabled.
Norm/RoPE/cache operations are deliberately unchanged and excluded from this
standalone measurement; their full consumer boundary still requires parent
whole-layer validation before keeping a runtime policy.

After standalone evidence only: add an off-default branch inside `_qkv` when
`prefill`, `logical_shape` and packed QKVG are present. Call this helper directly
on projected flat output, then converge on the existing `_norm` and `_rope`
calls. Preserve gate compact shape and the existing cache-fill/dtype/mask path.
Do not send the returned Q/K/V through the old reshape/head creation again.
Leave compact decode, separate-gate/QKV policies and nonflat prefill on their
existing branches. Whole-layer S33/S512/S513, resumed starts, cache PCC,
nonaligned batches and final profiles remain required. No default or main
integration has been made here.

Frozen initial source SHA256:

- helper `8747283312e74d5f2af8c2dc135565b9797f917ad53cf5fabd3131b6d2a73e49`
- probe `eb45693e5063751539d38f15aca7dc8cf084e5c54543d42aa59d585856583aee`
