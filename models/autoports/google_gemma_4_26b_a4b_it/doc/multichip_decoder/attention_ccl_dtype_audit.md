# Attention collective dtype audit

CPU/source-only audit; no hardware or runtime edits. BF16 is a legal candidate.
The native RS/AG source does not reject BFP8 either; no source-supported BFP8
blocker was found for this tile-aligned H2816 -> H704 contract. Runtime
correctness/performance remains unmeasured for attention BFP8 communication.

## Validation and packing evidence

- `ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_op_device_operation.cpp:67`
  invokes `experimental::ccl::reduce_scatter_common_validates`.
- Its implementation is
  `ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_common/reduce_scatter_validate_utils.cpp:15`.
  It accepts Ring/Linear, device buffers, aligned pages and supported memory
  layouts. Scatter tile count must divide device count: H2816 is88 tiles,
  and88/4=22 tiles gives H704. There is no BF16-only dtype allowlist.
  Optional output and tiled intermediate dtypes must match input dtype.
- Native output spec retains input dtype in
  `reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_op_device_operation.cpp:181`.
  Line program CB creation in
  `reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_program.cpp:1187`
  converts input dtype to its device data format and uses it for input,
  intermediate and compute-output CBs; Ring does the same at line517.
  Consequently reduced precision affects intermediate packing, not just wire
  transport after an otherwise FP32 reduction.
- `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_device_operation.cpp:80`
  checks page alignment and layouts, with matching input/output dtype. Its
  Blackhole DRAM rejection applies only to the specialized llama-sharded
  kernel; the candidate uses ordinary DRAM-interleaved AG. AG performs no
  arithmetic and preserves the reduced RS representation.
- Blackhole test
  `tests/ttnn/unit_tests/operations/ccl/blackhole_CI/box/nightly/test_minimal_reduce_scatter_async_bh.py:265`
  contains BFP8 *composite* edge cases. Those cases alone are not native
  H2816 evidence; the native acceptance conclusion above comes from the
  actual validation/data-format paths, not a claim those tests ran here.

Tile payload bytes for physical [1,1,32,2816]: FP32=360448, BF16=180224,
BFP8=95744 (88 tiles times4096/2048/1088 bytes). Nominal ring-equivalent
RS+AG per-rank traffic is1.5 times these values, excluding headers and actual
Linear scheduling. Reduced payload does not by itself establish a latency win.

## Coherent model-local candidate

Keep WO matmul precision/output unchanged initially. In `_LocalAttention.project`
(`tt/multichip_decoder.py`), compute local output, explicitly cast only that
attention contribution to BF16, then call the existing `self.reduce` binding.
That binding already selects replicated allreduce or sharded RS at construction.
For the requested replicated experiment the full boundary is:

`WO FP32 -> cast BF16 -> RS BF16 -> AG BF16 -> post-attention norm FP32 -> residual FP32`.

Leave shared/routed collective policies, grouping, geometry and expert policies
identical to the control. Prefer a setup-selected attention CCL dtype attribute,
with the FP32 default unchanged; the same boundary can test BFP8 if desired.
Apply the policy consistently to both prefill/decode, or explicitly record a
decode-only policy as such. Public residual input/output remains replicated
BF16 H2816, so stacking adds no layout conversion. Cache ownership, head split
and page tables are upstream and structurally unchanged, but output differences
can change later layer routing; real-weight full-layer and stack checks matter.

## Additional rounding and normalization

`tt/optimized_decoder.py:1319` keeps prefill/decode WO output FP32.
`tt/fused_decoder.py:155` implements the inherited `normalize`: for hidden
widthH2816 with `fuse_norm=True`, it casts its input to FP32 before native
RMSNorm, then multiplies the norm weight. Thus a BF16/BFP8 collective is
promoted back to FP32 for post-attention normalization without changing that
consumer; promotion cannot restore discarded mantissa bits.

There are two numerical changes relative to FP32 CCL: rounding each local WO
partial before summation, and reduced-format intermediate/output packing during
RS. Cancellation across row-parallel contributions may amplify that error.
Additionally `ttnn/cpp/ttnn/operations/ccl/ccl_common.cpp:28` auto-enables FP32
destination accumulation only when the input dtype is FLOAT32 and no explicit
compute config is supplied. The public RS wrapper applies that helper at
`reduce_scatter_minimal_async/reduce_scatter_minimal_async.cpp:54`.
A BF16 input does not inherit this automatic FP32 setting. If BF16 accuracy
fails, a distinct BF16-transport/explicit-FP32-accumulation candidate is possible
through the RS `compute_kernel_config` argument, but still rounds at packing
boundaries and needs its own evidence; current `MeshConfig.allreduce` does not
forward that argument.

Recommended gate: paired4096/128 both layer kinds, minimumPCC.995, exact replay,
page/cache checks, then stacked output PCC on the selected grouped/geometry1
configuration. Time the complete layer including cast and FP32 norm promotion;
do not accept an isolated CCL speed result as a whole-layer improvement.
