# Preserve the incumbent statistics tree while distributing hidden RMSNorm

The parent reports that the first two-pass distributed candidate passes isolated
same-input comparisons with relative RMS below 1e-5, yet fails the unchanged
512-step decoder gate: minimum PCC .9937332 versus .9958222 for the accurate
per-user incumbent. Its compact decoder latency improves from 2.26743 ms to
2.14061 ms. These are parent-reported results, not device runs by this agent.

The repair hypothesis is that a different FP32 positive-sum association changes
rare BF16 norm ties and accumulates through the recurrent state. The isolated
error bound alone does not establish this cause. The new candidate restores the
incumbent's complete statistics arithmetic tree, then requires zero BF16 bit
mismatches against that incumbent before whole-decoder validation.

## New immutable artifacts

- `candidates/hidden_rmsnorm_distributed_exact_tree.py`: three-pass helper.
- `candidates/check_hidden_rmsnorm_exact_tree.py`: CPU/source identity proof.
- `hidden_rmsnorm_exact_tree_cpu.json`: passing CPU results.
- `../../tests/run_norm_exact_comparison.py`: hardware comparator, not run here.

Old helpers and the main decoder are unchanged. All nine device kernel sources
are embedded in the new helper. The public function stays
`hidden_rmsnorm_sfpu(x, gamma, epsilon=1e-6, output=None, scratch=None)`.
Optional reusable scratch is `(folded, inverse)`, FP32 TILE interleaved DRAM,
with shapes `[1,32,1024]` and `[1,32,32]`. Supplying output and scratch avoids
tensor allocations. Every program registers all referenced tensors and rebuilds
runtime addresses. The three generic ops execute on the same default queue.

## Exact mapping and arithmetic

Let `x[u,c]` be a logical BF16 input. The incumbent reader creates five dense
32x32 tiles on each user core, with element `(r,c)` of chunk `j` equal to
`x[u,j*1024+r*32+c]`. Its first arithmetic phase computes, in order:

```
f[u,d] = ((((x[u,d]^2 + x[u,1024+d]^2)
           + x[u,2048+d]^2) + x[u,3072+d]^2) + x[u,4096+d]^2)
```

All squares and adds are separate FP32 SFPU operations. The new pass 1 assigns
one core to each `r=0..31`. It loads original input tile pages
`r,32+r,64+r,96+r,128+r`, each containing all token rows and 32 channels.
The five-tile compute loop is literally identical to the incumbent's loop.
Each core writes its folded FP32 page, with element `[u,c]=f[u,r*32+c]`.
Pass 1 needs 14 KiB CB storage/core.

Pass 2 assigns one core per logical user. For each of 32 folded pages it reads
two 64-byte face rows. The source FP32 element address is
`(u//16)*512 + face*256 + (u%16)*16`, and the destination address is
`(r//16)*512 + face*256 + (r%16)*16`. These are precisely the old dense tile's
face-major addresses. All 1024 positions are covered exactly once; source and
destination byte offsets are divisible by 64. There are 64 small reads/user,
compared with 320 x reads and 320 gamma reads/user in the incumbent.

The reconstructed tile therefore contains the same FP32 folded values at the
same physical positions. Pass 2 then executes the literal incumbent source
span: SFPU SUM `REDUCE_COL`, SFPU SUM `REDUCE_ROW`, multiply by FP32 `1/5120`,
add epsilon, precise rsqrt. It packs FP32 and writes element zero into column
zero of that user's inverse row. Full row writes and separate padding writes
cover the entire inverse tile without overlap. Only column zero is consumed.
Pass 2 needs 12 KiB CB storage/user.

Pass 3 uses 32 channel cores and reads five whole x/gamma tiles per core plus
one inverse tile. `sfpu_normalize_bcast_col` applies `(x - +0) * inverse`,
retaining even the incumbent's zero subtraction, followed by FP32 gamma row
broadcast multiplication. Both the old scalar normalization and the new column
normalization use separate SFPADD and SFPMUL instructions; gamma stays FP32.
Explicit SFPU Float32-to-BF16 typecast preserves the same final rounding
boundary. The writer zeros nonlogical token rows and writes complete tiles.
Pass 3 needs 48 KiB CB storage/core.

Every consumed Float32 CB has `UnpackToDestFp32`; BF16 input uses Default unpack
into FP32 destination. Intermediate FP32 packing is intended to preserve all
bits, with no TF32 FPU operations, BF16 scalers, fitted compensation, or host
conversions. All NoC read/write calls have the five mandatory API arguments.

## Host validation completed

Python syntax compilation passes for helper, proof, and comparator. The CPU
proof passes at B=1,3,16,17,31,32. It verifies literal fold and reduction/scalar
source identity, full channel ownership, identical dense mapping, aligned
reads, complete inverse-row write coverage, and bit-identical reconstructed
FP32 folded values. It does not simulate SFPU lane execution.

```sh
python_env/bin/python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/check_hidden_rmsnorm_exact_tree.py > models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/hidden_rmsnorm_exact_tree_cpu.json
```

An AST-isolated host exercise also successfully loaded the four actual layer0/1
input/post-norm weights from the pinned local model revision, folded each to
FP32 gamma, and generated every stress fixture for B1/B3/B32. A scalar `.pow`
typo discovered by that exercise was corrected in the comparator before freeze.
No TTNN device was opened.

## Required serialized device checks

The parent has queued the frozen helper/test as `exact79_probe`:

```sh
python -m models.autoports.qwen_qwen3_6_27b.tests.run_norm_exact_comparison --output models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/exact79_probe.json
```

The default comparator runs 54 cases: both actual layer0 gamma tensors,
B1/B3/B32, four random scale bands, positive/signed zeros, alternating powers,
sparse cross-chunk impulses, and mixed BF16 mantissas. It reuses candidate
scratch/output and incumbent output across changed inputs, requires **zero BF16
bit mismatches**, writes incremental JSON, and saves the first failing input,
gamma, reference output, and candidate output in a `.failure.pt` file.
`--layers 0 1` includes actual layer1 gamma too. `--real-tensors FILE` accepts
additional saved dictionaries containing real activation `x` and folded
`gamma` tensors.

Device compilation, exact output matching, trace replay, watcher, latency and
the complete 512-step gate remain unverified by this agent. Extra dispatches
and FP32 intermediate traffic could erase some of the two-pass speed gain.
If the comparator still differs, inspect the saved sample and intermediate
folded/inverse buffers before attributing the discrepancy to recurrent state.
A comparator pass is necessary but does not replace the full decoder gate.
