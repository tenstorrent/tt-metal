# Next ordered-projection layout discriminator

Source/CPU only. No runtime edits or hardware execution. Current parent profile is `profile_projection_l1/tracy/layer_0/ops.csv`; its singleton pair-gather is `GatherCodegenDeviceOperation`, 64 cores, Ht=1, Wt_input=Wt_index=64 (one replay example 51.690 us). Parent reports BinaryNG about97 us and first FastReduce512 about140.5 us. The proposal below targets the gather/adapter while preserving the proven product/reduction exactly.

## Concrete candidate: compact pair gather

At setup upload a BF16-independent UINT16 TILE index tensor of shape `[1,64,32]`: every row is `[1,0,3,2,...,31,30]`. Keep existing global indices for prefill. Replace only singleton coefficient preparation:

```python
compact = ttnn.reshape(ttnn.to_memory_config(x, DRAM), (1, 64, 32))
ordered = ttnn.gather(compact, -1, self.attention_pair32_indices)
coefficients = ttnn.reshape(ordered, (2048 // projection_k, projection_k, 1, 1))
# Existing BF16 weights, FP32 product in L1 and ordered reductions unchanged.
```

For global position `k = 32*r+c`, this reads `32*r+(c xor1) = k xor1`, exactly the existing pair reversal. CPU exhaustively checked all2048 integer addresses; adjacent JSON records that proof and current source SHA. BF16 values undergo only device data movement. Both K512 decode and K1024 singleton prefill-tail policies retain their respective arithmetic order. This is only the singleton branch, not a change to public prefill chunking.

Source supports a meaningful traffic difference:

- `gather/codegen/gather_codegen_device_operation.cpp:17–35` selects tiled output ownership for Ht below the worker count and Wt_index>=2; Wt_index=1 instead selects row-buffered ownership.
- `gather/codegen/kernels/gather_reader_tiled.cpp:6–27,95–105` and its paired writer require each current output owner to stage all64 input tiles. The current64 owners therefore read4096 BF16 pages (8 MiB) in aggregate before gathering. They also execute padded index-tile loops despite one valid row.
- Candidate Ht=2/Wt=1 has two row owners, each staging one2KiB tile:4096 input bytes in aggregate. `gather_reader.cpp` selects its full-tile element loop because all32 rows/columns are valid. This removes redundant input residency, not just an unused logical dimension.
- The candidate input and static index are only4096bytes each, compared with131072bytes for each original padded vector. The final scalar-coefficient tensor remains2048 BF16 pages=4MiB. The initial compacting reshape is a real device operation, not a metadata view (`reshape_view/reshape.cpp:757–783`); its cost must be included.
- All shapes use default32x32 tiles, BF16 data, UINT16 indices and interleaved memory, within `gather_codegen_supported.cpp`. No host conversion, custom tile or native edit is needed.

Measure the entire old gather+coefficient reshape versus compact reshape+gather+coefficient reshape, then full warmed trace. Require exact finite product/projection codes and unchanged seeded endurance. There is no measured speed claim: two cores still execute a full1024-element scalar gather loop each, and an extra reshape may erase the read reduction. A separate RM gather adapter is legal but adds layout conversions; compact TILE is the narrower first discriminator.

## Why more reduction owners are not a proven improvement

Current products `[4,512,32,224]` have28 output tiles in the first reduction. `fast_reduce_nc_program_factory.cpp:169–184` splits those output tiles among28 owners. In `kernels/reduce_nc.cpp:42–76`, every owner copies its first input tile and then performs511 full-tile sequential FP32 SFPU additions. Increasing H/W padding to56 or84 owners preserves512 terms per owner and cannot shorten this serial instruction chain. It raises product/weight traffic. Splitting K into shorter parallel reductions changes FP32 association and violates the canonical left fold; ordinary sum/matmul/tile reduction is not an equivalent substitute.

Placing an owner's complete scan in its own L1 would require512*4096=2,097,152bytes/core before buffers, beyond Blackhole worker L1. Current interleaved placement distributes58,720,256bytes over110 banks, about524KiB/core. ND sharding can change address locality, but cannot make the full scan local or create more arithmetic owners without changing the operator; its specialized factory work split requires dim0, matching ND input/output, and more output shards than worker cores. Current first stage is dim1 with28 output tiles. No source-backed reduction speedup follows from relabeling this tensor ND-sharded.

## Binary vectorization limits

`binary_ng_program_factory.cpp:1150–1175` enables two-tile SFPU cycles only for no-broadcast with all operands/output natively sharded (or tensor-scalar value with sharded input/output). Current subtile scalar broadcast is excluded. Fully replicating coefficients to product shape would materialize another29,360,128-byte BF16 tensor plus its replication operation; together with BF16 weights staged to L1 and FP32 products this is117,440,512bytes, about1.02MiB/core at110 banks before CBs/live state. Such replication is not a free vectorization/layout view and has no demonstrated benefit.

The current source also explicitly disables BF16 scalar LLK broadcast on Blackhole when FP32 destination is enabled (`binary_ng_program_factory.cpp:1449–1475`, MOVB2D issue449). Current products already use the reader software-broadcast path. Widening coefficients solely to bypass LLK broadcast cannot improve this implementation choice. The older mixed-product memo's general BF16/BF16 LLK observation is superseded by this architecture guard.

No supported graph-only multiply+FastReduce fusion was found: FastReduce consumes already-materialized FP32 tiles and exposes no coefficient operand or multiply activation. Existing exact ordered operations remain the baseline.
