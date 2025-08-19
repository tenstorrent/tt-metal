# Issue: Incorrect override runtime arguments in attn_matmul

- File: `ttnn/cpp/ttnn/operations/experimental/matmul/attn_matmul/device/attn_matmul_program_factory.cpp`
- Function: `multi_core_attn_matmul(...)` override callback

## Suspected root causes

1) Reader runtime-arg order differs between create vs override

   - During create, reader runtime args are set as:
     - Lines ~206-224: `{src0_addr, src1_addr, Mt, Kt, Nt, MtKt, in1_KtNt_skip, in1_KtNt_stride*32, num_blocks_per_core, itileA_start, itileB_start}`

   - During override, reader runtime args are set as:
     - Lines ~320-337: `{0, 0, Mt, Nt, Kt, MtKt, in1_KtNt_skip, in1_KtNt_stride*32, num_blocks_per_core, itileA_start, itileB_start}`

   - Problems:
     - `Mt`, `Kt`, `Nt` order is `Mt,Kt,Nt` at create but `Mt,Nt,Kt` at override.
     - `src0_addr` and `src1_addr` are not updated at all (zeros written), causing stale addresses on cache-hit.

2) Output writer kernel args are correctly updated with `dst_addr`, but reader input addresses are not, leading to either hangs or PCC mismatch depending on memory reuse.

3) Circular buffer size update for `cb_src0` uses `Kt * in0_single_tile_size` at override vs. `cb0_num_input_tiles * ...` at create, but `cb0_num_input_tiles` equals `Kt*2` at create. This halves the CB capacity on cache-hit and can corrupt the dataflow.

## Failure mode

- Expected: second run should reuse the same compiled program and produce identical results.
- Observed: the second run hits the cache but reuses incorrect runtime args due to the issues above, leading to a PCC mismatch (or potential hang).

## Reproduction

```bash
pytest -q program_cache/experimental/matmul/attn_matmul/failures/test_attn_matmul_program_cache_override_rtargs.py::test_attn_matmul_program_cache_override_rtargs -s --disable-warnings
```

## Suggested fix

- Make reader runtime-argument order identical between create and override:
  - Use the same ordering `{src0_addr, src1_addr, Mt, Kt, Nt, MtKt, in1_KtNt_skip, in1_KtNt_stride*32, num_blocks_per_core, itileA_start, itileB_start}` in override.
- Update `src0_addr` and `src1_addr` on every cache hit.
- Match CB sizing logic between create and override. If `cb0_num_input_tiles = Kt * 2` at create, update override to the same effective capacity or refactor to a helper to compute it consistently.
