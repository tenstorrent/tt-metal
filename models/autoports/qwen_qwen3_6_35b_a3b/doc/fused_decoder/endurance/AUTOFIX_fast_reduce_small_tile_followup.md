# Small tiles and output-owner count for exact projection

Source/CPU inspection only. No device calls, decoder changes, or shared-native edits. Parent reports the 32x32 fast-NC candidate at 5.157 ms and owns its endurance validation; this memo makes no new performance claim.

## Conclusion

**16x32/8x32 tiles cannot be used through the existing FP32 multiply + fast_reduce_nc pipeline.** Binary multiply explicitly rejects custom tiles, and fast_reduce_nc independently assumes full 32x32 tiles in page sizes, address strides, output creation and work splitting. Upload support is insufficient. This is a native-contract blocker under the no-shared-native-edits constraint, not a candidate requiring a device trial.

Padding channels to create more output tile owners is valid but does not reduce the per-owner serial work. It increases both multiplication work and memory traffic. Keep the measured 13312-channel packing as the preferred candidate unless profiling identifies a specific memory-mapping benefit worth testing.

## Contract evidence

1. **Host tensor creation exposes custom tile configuration.** `ttnn/ttnn/operations/core.py:282–300,397–405` forwards `tile` and dtype to the tensor constructor. `tt_metal/impl/data_format/tile.cpp:19–34` lists 16x32 and 8x32 shapes; its 8-row entries are explicitly described as host-loopback shapes rather than a blanket LLK support promise. A tensor's successful construction says nothing about the downstream operator contract.

2. **FP32 binary multiply rejects both shapes.** `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.cpp:480–501` checks all TILE inputs and any preallocated output and requires height 32 / width 32. The accompanying source comment says circular buffers and subtile fills are sized for 32x32. Selecting FP32/SFPU, preallocating a smaller output, or supplying only one custom-tiled operand does not bypass this validation.

3. **Fast-NC cannot safely consume smaller tile pages either.** In `ttnn/cpp/ttnn/operations/experimental/reduction/fast_reduce_nc/device/fast_reduce_nc_program_factory.cpp`:
   - Lines 48–64 compute H tiles as H/32 and W tiles as W/32, which also fixes the reduction reader's plane stride.
   - Lines 101–105 use `tt::tile_size(format)` for input/output/zero-tile DFB entry sizes, without the tensor's custom tile dimensions.
   - Lines 111–112 divide physical volume by 1024 to count output tiles.
   - Lines 203–217 instantiate the DFBs with those full-tile byte sizes.
   - The tile-shape reads at 140–141 only test input/output compatibility for an ND-sharding optimization; they do not parameterize the full-tile arithmetic above.
   - `fast_reduce_nc_device_operation.cpp:121–130` constructs a default TILE output page configuration, rather than preserving the input's custom tile.

   Consequently, changing the input tile metadata does not produce 104 or 208 valid output owners. With a fixed logical H/W it still counts 52 full tiles; the reader/page assumptions become inconsistent with the smaller pages. A custom preallocated output does not repair those assumptions. The compute kernel also calls ordinary full-tile copy/add operations without a custom-tile mode. Absence of an explicit small-tile rejection in this op is not support.

## Padding bounds

These all retain four ordered 512-term partials, keep native 32x32 tiles and have H=32. W is the padded channel count divided by32. Ordering of every valid output remains unchanged.

| Channels per partial | Layout | Output tiles / maximum owners | Product/weight size each | Traffic increase vs current 13312 | Padding vs original 12352 |
|---:|---|---:|---:|---:|---:|
|13312|[512,4,32,416]|52|104 MiB|0%|7.77%|
|16384|[512,4,32,512]|64|128 MiB|23.08%|32.64%|
|24576|[512,4,32,768]|96|192 MiB|84.62%|98.96%|

Each output owner still reads **512 full FP32 tiles = 2 MiB**, performs 511 full-tile sequential additions, and writes one 4 KiB output tile. More owners are achieved by distributing useful output channels among more padded lanes, not by shortening their sum or reducing the full-tile instructions issued by a core. Useful lanes fall from 92.79% to 75.39% or 50.26%. While all owners fit on separate available cores, the ideal per-owner compute/read critical path is unchanged. If output tiles exceed available cores, some cores receive additional tiles and the critical path can grow.

The product-producing multiply is already parallel over a large tensor; additional padding increases its work and tensor movement. Across a standalone weight read, product write and product read, 16384 adds roughly72 MiB and 24576 adds roughly264 MiB relative to current packing, before considering caches and small operands. These are payload bounds, not measured DRAM transactions.

Source for independent tile ownership: `fast_reduce_nc_program_factory.cpp:169–184,345–370`; for each owner's complete sequential scan: `device/kernels/reduce_nc.cpp:42–76` and `reader_reduce_nc.cpp:57–87`.

## Recommendation

Do not run the custom-tile path: the source proves it unsupported without native changes. Do not interpret 64/96 owners as a proportional speedup opportunity. If root wants one padding experiment after kernel-level profiling, 16384 is the less costly diagnostic; compare product and reduction latency separately and require unchanged projection codes. Reserve 24576 for evidence of a concrete bandwidth/distribution issue that outweighs nearly doubled padded payload versus the true channel count. The current evidence alone favors retaining 13312.

Splitting each 512-term ordered sum between cores and combining partials is also unsuitable: it changes FP32 association. A truly faster 16x32 reduction would need coordinated native tile-size support in multiplication, reduction resource sizing, output specifications and compute behavior, which is outside the authorized model-only scope.
