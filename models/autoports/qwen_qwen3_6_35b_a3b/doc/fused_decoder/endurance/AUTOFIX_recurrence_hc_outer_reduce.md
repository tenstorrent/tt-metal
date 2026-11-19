# Recurrence contraction: HC transpose and outer-dimension reshape

Source/CPU-only candidate; no runtime file edit or hardware execution. Candidate snapshot: `recurrence_hc_outer_reduce.py.snapshot`, SHA256 `166f7454f138745bee9606e22157c1c725d278261820b63c95f31478b9263df0`, derived from current F SHA256 `f8b93f756ef34f680b1aa7c70848bbdf8abe07a537f02b2cf14ab356fd0df06a`. The adjacent provenance JSON identifies the base. AST comparison after removing `_recurrent_sum` proves no other source node changed.

```python
def _recurrent_sum(self, products):
    head_rows = ttnn.permute(products, (0, 2, 1, 3))
    chunks = ttnn.reshape(head_rows, (8, 16, 32, 128))
    total = ttnn.experimental.fast_reduce_nc(chunks, dims=[0, 1], compute_kernel_config=self.compute)
    return ttnn.permute(total, (0, 2, 1, 3))
```

## Exact ordering

For `P[0,h,k,v]`, write `k=16a+b` with `a∈[0,8)`, `b∈[0,16)`. Current reshape/permutation yields `old[b,a,h,v]=P[0,h,16a+b,v]`; candidate HC transpose and reshape yield `new[a,b,h,v]=P[0,h,16a+b,v]`. Current first reduces dimension0 (b), then dimension1 (a). Candidate public `dims=[0,1]` sorts dimensions and executes them **descending**, so reduces dimension1 (b), then dimension0 (a). Both form the same sixteen-value left fold for each contiguous K chunk, then the same eight-partial left fold. The first value initializes each fold; neither version adds an extra zero.

Source: `experimental/reduction/fast_reduce_nc/fast_reduce_nc.cpp:37–56` sorts and processes descending dimensions. `device/kernels/reader_reduce_nc.cpp:59–84` reads increasing positions of the selected dimension. `device/kernels/reduce_nc.cpp:50–68` copies the first plane to Dest, then adds each next plane serially. `device/fast_reduce_nc_program_factory.cpp:289–303` selects the FP32 SFPU path and `UnpackToDest` for this FP32 input and the unchanged FP32-destination compute config. Intermediate output defaults to input dtype (`fast_reduce_nc_device_operation.cpp:125–131`), so both stages stay FP32. The default `fp32_intermediate_stages=False` does not narrow an FP32 input.

CPU proof command:

```bash
python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/recurrence_hc_outer_reduce_cpu.py
```

`recurrence_hc_outer_reduce_cpu.json` proves the index mapping over all524,288 elements and bit-exact FP32 outputs for10 cases: random values, cancellation-heavy values, and saved memory/output products at seed154 positions128/5990/5991/5992. Explicit serial adds emulate native fold order; this does not rely on Torch's potentially different `sum` schedule. Output logical shape remains `[1,32,1,128]`.

## Native shape/layout contract

- Input remains FP32 interleaved TILE `[1,32,128,128]`. `permute.cpp:165–166` recognizes `(0,2,1,3)` and calls native HC transpose, producing `[1,128,32,128]`. No BF16 cast is taken; that wrapper's conversion branch is specific to BFLOAT8_B.
- HC validation requires padded H/W tile multiples; input H128/W128 and output H32/W128 satisfy this. The HC factory derives data format from the original dtype, including FLOAT32. It transfers data through data-movement buffers; no arithmetic precision boundary is introduced. Source: `transpose_device_operation.cpp:352–359,403–418`; `transpose_hc_tiled_interleaved_program_factory.cpp:109–123,176`.
- Reshape `[1,128,32,128] -> [8,16,32,128]` leaves the inner two dimensions unchanged and splits only outer contiguous planes. `reshape_view/reshape.cpp:757–770` therefore takes the metadata-only view path, preserving buffer/layout/dtype. No padding is inserted.
- Both reduction dimensions0/1 are legal non-spatial axes for rank4 (`fast_reduce_nc_device_operation.cpp:71–83`). Shapes are `[8,16,32,128] -> [8,1,32,128] -> [1,1,32,128]`. Final HC transpose is identical to current F and gives logical `[1,32,1,128]`, with the same padded H32 storage as before.
- This proof applies to the current `_recurrent_sum` contract with32 heads and128 K/V. Both decode contractions and sequential prefill correction use that shape. It does not propose changing public sequence/chunk boundaries or BF16 cache rounding.

## What could improve, and what is not fused

Current first reshape changes spatial H128 toH16, which is not tile-aligned and does not meet the metadata-view condition. It can materialize `[32,8,32,128]` padded storage (4 MiB versus2 MiB useful input) before a general `(2,1,0,3)` permutation. The candidate replaces that pair with one recognized HC transpose plus a metadata-only outer reshape. This is a source-supported opportunity to remove movement and padding, not a measured latency guarantee.

Combining dimensions in one public `fast_reduce_nc` call **still launches two native reduction operations**. It preserves the same FP32 intermediate size (128 KiB logical) and arithmetic order; the benefit is interface simplification, not a claim of one fused reduction kernel. Parent should measure exact outputs and warmed full recurrence/decoder trace latency on this snapshot before selection, retaining the existing seed/continuation gates. No precision or performance result for the candidate is claimed here.
