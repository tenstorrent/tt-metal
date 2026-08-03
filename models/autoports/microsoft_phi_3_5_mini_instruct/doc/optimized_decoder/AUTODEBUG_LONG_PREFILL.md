# AutoDebug: longest non-aligned optimized prefill

## Headline finding

The `seq_len=131071` failure is an L1 planning bug in the optimized
prefill projection path, not an alignment restriction and not an SDPA failure.
`_linear_prefill` sends the entire logical sequence to an explicit
`MatmulMultiCoreReuseMultiCastProgramConfig`. Its large-M circular buffers grow
with the per-core row assignment and exceed Blackhole L1 before the first QKV
projection can run.

The smallest proven device-only repair is to keep the public input and output at
the original arbitrary logical length while slicing the sequence dimension into
internal projection chunks of at most 1024 rows, running the existing explicit
TTNN matmul on every chunk, and concatenating the device results. A 2048-row
maximum is not safe for all Phi projections: it remained 13,056 bytes over L1.
The 1024-row discriminator passed the advertised non-aligned
`seq_len=131071` case.

## Direct observations

- The failing test constructs logical input `[1, 1, 131071, 3072]` and reaches
  the first QKV call at `optimized_decoder.py:555`; the exception is raised by
  `ttnn.linear` in `_linear_prefill` before QKV head splitting, cache fill, or
  the long-prefill SDPA loop.
- `_prefill_matmul_config` computes `row_tiles=ceil(rows/32)`, distributes those
  tiles across up to eight grid rows, and assigns the resulting value to
  `per_core_M`. For 131071 logical rows, tile padding produces 4096 row tiles
  and `per_core_M=4096/8=512`.
- The QKV projection has `K=3072` (`Kt=96`) and `N=9216` (`Nt=288`).
  The 8-column grid gives `per_core_N=36`; the large-row branch selects
  `in0_block_w=1`.
- The Python construction omits `out_block_h` and `out_block_w`. The nanobind
  constructor defaults them to `per_core_M` and `per_core_N`
  (`ttnn/cpp/ttnn/operations/matmul/matmul_nanobind.cpp:166-167`), so the
  failing QKV program uses an output block of `512 x 36` tiles.
- The 2-D multicast factory sizes input CBs from `out_block_h`,
  `out_block_w`, and `in0_block_w`, double-buffers both inputs, and sizes the
  output/intermediate CB from `out_block_h * out_block_w`
  (`matmul_multicore_reuse_mcast_2d_program_factory.cpp:137-167`).
- For the default BFP4 QKV weight, the allocations are:

  | CB | Calculation | Bytes |
  |---|---:|---:|
  | input 0, BF16 | `512 * 1 * 2 * 2048` | 2,097,152 |
  | input 1, BFP4 | `36 * 1 * 2 * 576` | 41,472 |
  | shared output/intermediate, BF16 | `512 * 36 * 2048` | 37,748,736 |
  | total CB payload |  | 39,887,360 |

  Adding the observed 111,360-byte L1 allocation base gives 39,998,720 bytes,
  exactly the endpoint in the exception, versus 1,572,864 bytes of L1.

## Complete projection audit

QKV is merely the earliest messenger. If it alone were bypassed, the full-row
o-projection, packed gate/up projection, and down-projection would encounter
the same scaling error. The packed gate/up projection is the worst case because
its `N=16384` output gives `per_core_N=64`.

The existing `_mlp_prefill` batch split only bounds the serving case
`batch=32, seq=128`: each 16-user slice has 2048 fused rows. With batch 1 and a
131071-token sequence, that loop still produces one full-sequence slice, so it
does not solve long prefill. Projection chunking must operate on sequence rows,
not only on batch.

Smaller correctness cases pass because their `per_core_M` is small. For
example, 33 rows have two row tiles distributed over two grid rows, so
`per_core_M=1`. Their success does not exercise the large-M CB regime.

## Proven intervention and discriminators

Implement one shared, device-only chunking boundary in `_linear_prefill`:

1. Preserve the logical input shape and public `seq_len`.
2. If the fused row count is at most 1024, use the existing explicit matmul
   directly.
3. Otherwise slice along the logical sequence dimension into chunks no larger
   than 1024 rows, run the same explicit TTNN linear operation for each chunk,
   and concatenate the TTNN outputs along that dimension.
4. Let the final chunk remain non-aligned logically; tile padding is internal.

Observed discriminators:

- Unchunked `seq_len=131071`: fails at first QKV compile with
  `39,998,720 > 1,572,864`.
- Internal maximum 2048 rows: still fails the widest projection, with the
  static allocation 13,056 bytes over L1. This rules out treating the first
  smaller configuration as sufficient.
- Internal maximum 1024 rows: passes the advertised longest non-aligned
  `seq_len=131071` test.

Do not replace the explicit program with the framework-default matmul as the
primary fix. The optimized weights are WIDTH_SHARDED in DRAM, and the attempted
default factory rejects that B layout. A first fallback/API error therefore
does not invalidate the explicit path; the successful 1024-row explicit
configuration is the relevant adapted result.

## Contract and SDPA impact

This repair adds no public `seq_len % chunk == 0` condition. It preserves
131071 as the logical sequence length and only pads or chunks internally. It
does not reduce the advertised 131072-token capacity, change paging, or change
KV-cache dtype/layout.

The existing long-prefill SDPA path remains intact. It starts only after QKV,
RoPE, and cache population and already bounds attention queries independently
with a 32768-token initial chunk and later four-tile chunks. Projection
chunking fixes the earlier linear stage and must not replace or weaken that
attention chunking.

## Validation required after the implementation edit

- Re-run the exact `test_optimized_longest_non_aligned_prefill` gate and retain
  its `context=131071 non_aligned=true` artifact.
- Re-run representative non-aligned short lengths, real-weight prefill PCC,
  batch-1 and batch-32 warmed prefill, and watcher-clean optimized correctness.
- Confirm the optimized path still calls its explicit program config for each
  chunk and never invokes `FunctionalDecoder` or a host fallback.

## AutoDebug workflow note

The required fresh `autodebug.sh` runner was started in an isolated temporary
directory, but this host denied the runner's bubblewrap user namespace before
it could read source. The findings above were therefore checked directly
against the repo source and the supplied hardware discriminator results; no
additional hardware command was run during this inspection.
