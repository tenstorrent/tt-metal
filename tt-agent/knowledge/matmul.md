# Matmul

Topic knowledge for tt-metal matmul — external references, memory layout,
compute config, debugging, K-blocking, numerical behavior, structural
pivots. Each guideline names the pattern, the expected gain, and when to
deviate.

## External references

- **GEMM FLOPS analysis** — `tt-metal/tech_reports/GEMM_FLOPS/GEMM_FLOPS.md`
  Per-core TFLOPS by math fidelity. Arithmetic intensity, compute-bound
  vs memory-bound reasoning.
- **Matrix engine deep dive** — `tt-metal/tech_reports/matrix_engine/matrix_engine.md`
  How the FPU executes 8×16 × 16×16 tile operations. Latency,
  throughput, fidelity.
- **Matmul op implementation** — `tt-metal/ttnn/cpp/ttnn/operations/matmul/`
  Current 1D / 2D / reuse strategies.
- **Data formats and accuracy** — `tt-metal/tech_reports/data_formats/data_formats.md`
  bfloat16 / bfloat8_b / bfloat4_b accuracy and memory trade-offs.
- **PSE Matmul Configuration Guide** — authoritative for variant
  selection, L1 budgeting, and subblock rules.

## Memory layout

### Shard activations in L1 matching the matmul's grid

For matmul inputs, use sharded L1 (HEIGHT, WIDTH, or BLOCK per the
matmul's parallelization). The shard spec must match the matmul's grid.

**Gain:** ~6× over `L1_MEMORY_CONFIG` on medium activations. Interleaved
L1 forces cross-core NOC fetches — slower than DRAM interleaved.

**Deviation:** template shard spec from a known-working sibling op (Llama
MLP, SDPA). Wrong specs OOM L1 or fail correctness silently.

### Size compute data for a 1.2 MB per-core L1 budget

Per-core L1 is 1.5 MB total, but firmware, kernels, debug regions, and CB
reservations consume ~300 KB. Budget `per_core_M × per_core_N` output
tiles + `in0_block_w` reader buffers + intermediates at 1.2 MB. Sizing
against 1.5 MB OOMs at runtime. Authoritative reference: PSE Matmul
Configuration Guide.

## Compute config

### Try `fp32_dest_acc_en=False` on short-K matmul (K ≤ ~100 tiles)

Default `matmul_config()` sets it `True`. Cost: DST register count drops
8 → 4, capping `out_subblock_h × out_subblock_w ≤ 4`. That cap alone can
pin `out_subblock_w=1` on narrow N, leaving math throughput unused.

**Gain:** 2-5% typical with PCC target ≥ 0.99. BF16 DST accumulation is
accurate enough at short K; the freed DST enables bigger subblocks in a
follow-up iteration.

**Deviation:** keep `True` when PCC target 0.999 is strict and long-K
accumulation error is load-bearing.

### Test fidelity lowering once; the effect is often neutral

LoFi < HiFi2 < HiFi4 (speed-vs-quality) is real but not load-bearing on
most non-compute-bound matmuls. On an overhead-bound path, HiFi4 → HiFi2
can be near-neutral. On an explicit progcfg path, HiFi3 → HiFi2 is
typically a small (~2%) win.

One test iteration per fidelity drop; skip repeat if the first shows no
signal.

### Default `packer_l1_acc=True`

Folds partial-sum accumulation into the packer, avoiding the BF16
round-trip between matmul blocks. Leave it on.

## Debugging

### Print the realized program config on crashes

When a matmul fails with a kernel-level assertion (`num_blocks_x`
mismatch, CB overflow, dim assertion), one print root-causes fast:

```python
pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(...)
print(f"pc_<name>: {pc}")
out = ttnn.linear(..., program_config=pc)
```

The realized `per_core_M/N`, `out_subblock_h/w`, `out_block_h/w` are in
the repr. Integer-division truncation, defaulted fields, and padding are
the usual surprises.

### Use actual weight shape in progcfgs, not cached `args.<dim>`

Many tt-metal models reuse Llama progcfg lambdas that close over
`args.<dim>` values computed at config time. Those can truncate (int-div)
or drift from actual weight shape. Prefer `self.<weight>.shape[-1]` over
`self.args.<cached_dim>`.

If a shared lambda crashes on a new model, suspect the cached dim before
the progcfg shape.

### Read `num_blocks_x > num_cores_x` as a dim mismatch

`num_blocks_x=10, num_cores_x=8` means the derived `per_core_N` doesn't
match the N dim. Fix the progcfg inputs, not the grid: pick `per_core_N`
such that `ceil(N_tiles / per_core_N) ≤ num_cores_x`.

## K-blocking

### Factorize `K_tiles` before a K-blocking jump

`in0_block_w` must divide `K_tiles` exactly. Enumerate valid divisors
first.

Example: `K_tiles=68 = 4 × 17`. Valid: {1, 2, 4, 17, 34, 68}. Going from
`in0_block_w=4` to `=17` needs 1820 KB CB footprint — OOMs L1 (max 1499
KB). No `=8` or `=10` option.

When the next valid value overruns L1: reshape `per_core_M` smaller to
free L1, or accept the current block size and move the lever elsewhere.

## Numerical

### Accept fused-activation PCC shifts up to ~5× for GELU/SiLU/RELU

`matmul_config(..., fused_activation=ttnn.UnaryOpType.GELU)` and
`ttnn.gelu(...)` use different approximations. Observed shift:
0.9999596 → 0.9991 when fusing GELU on Gemma3 MLP — expected, still
within the 0.99 quality bar.

Re-check against the user's PCC bar (not the default 0.999 `pcc_abort`)
before concluding fusion broke correctness. First fusion iteration
typically drops PCC 2-5×.

## Structural pivots

### Verify a sibling's mode before borrowing structural choices

tt-metal models share kernels across PREFILL and DECODE with very
different progcfgs per mode. If the target "reused from X", verify X's
progcfg path for the *same* mode as the target.

Observed: planned L1-block-sharded activations for Gemma3 vision MLP
(PREFILL, seq=16512) based on Llama MLP's DRAM-sharded config — but
Llama MLP uses DRAM-sharded only in DECODE; its PREFILL is
DRAM-interleaved.

Check: `grep` the sibling for mode keywords (`prefill`, `decode`,
`mode==`) and confirm the progcfg branch the target inherits.
