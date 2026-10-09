# rmsnorm-prefill

## What the code does
`ttnn.experimental.dit_fused_distributed_rmsnorm` is an RMSNorm whose hidden dimension is sharded across the
4 chips of a 1x4 Blackhole mesh (tensor parallel, TP=4). Each chip computes a partial sum of squares per row,
the chips exchange those partial statistics over the fabric (an all-gather on the TP axis), and each chip then
normalizes its slice and multiplies by its slice of gamma. Inputs are bf16 in TILE layout in DRAM.

## Where the code is
Everything editable is under `ttnn/cpp/ttnn/operations/experimental/ccl/dit_fused_distributed_rmsnorm/`:
- `device/dit_fused_distributed_rmsnorm_program_factory.cpp`: core grid, circular buffers, kernel args, the
  all-gather forwarder setup (host side, needs a rebuild)
- `device/kernels/`: dataflow (reader/writer, fabric) and compute kernels (JIT-compiled at run time, no rebuild)
- the op/device-operation files and nanobind bindings

## What is measured
`tests/ttnn/nightly/unit_tests/operations/fused/test_fused_rms_norm_prefill.py`: 4 prefill shapes, 640 rows,
hidden 3584 / 4096 / 6144 / 7168 (896-1792 columns per chip). 3 warmup + 10 measured calls per shape. The value
per case is the mean per-chip DEVICE KERNEL DURATION of the measured calls, in µs, from the Tracy ops CSV.
Accuracy: PCC >= 0.99999 and max abs error <= 0.05 against torch on every shape.

## Constraints
Reduced math fidelity and approximate math modes are not allowed (see the campaign rules).
