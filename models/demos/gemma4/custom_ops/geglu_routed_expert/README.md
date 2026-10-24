# geglu_routed_expert: the Gemma 4 GeGLU routed experts over a dispatched buffer as one generic op

The 26B-A4B token-dispatch prefill (`tt/experts/dispatch_prefill.py`) runs its experts on the DeepSeek
dispatch/combine stack. Between dispatch and combine the fused routed-expert kernel computes, for every local
expert, `down(gelu_tanh(x @ gate) * (x @ up))` over that expert's rows of the dispatched buffer. The shared op
(`ttnn.experimental.unified_routed_expert_ffn`, DeepSeek's `unified_routed_expert_moe`) only knows SiLU-shaped
activations; this folder is the same program as a Gemma 4 model-local `ttnn.generic_op` with the GeGLU
activation baked into its own compute kernel, so no Gemma 4 activation variant has to live in the shared op.

`op.py`: `GegluRoutedExpert(mesh_device, gate_projs, up_projs, down_projs, idx_table, max_tokens)` builds the
`ttnn.ProgramDescriptor` (11x8 worker grid, 13 semaphores, 16 circular buffers, three kernels) and
`__call__(x, counts, region_offsets, output=None)` runs it; `golden(...)` is the torch reference.

The kernel family is Blackhole-only: the 11x8 rectangle is taken from `custom_ops/grid.compute_grid` (capped by
`GEMMA4_TEST_GRID`), and a grid that cannot host it (the Wormhole 8x8) makes `GegluRoutedExpert.unsupported_reason(device)`
non-None; the constructor refuses with that reason and the unit tests skip on it (never fail). Plan knobs
`per_core_m_max` / `in0_block_w_gu` (lab env `GEMMA4_GEGLU_PER_CORE_M_MAX` / `GEMMA4_GEGLU_IN0_BLOCK_W`) exist for
experiments; the 0451 scan found the per-expert cost independent of the gate/up K-block count.

## Contract (per device; the same program runs on every chip of the mesh with that chip's weights and buffer)

| tensor | shape / dtype / layout | role |
|---|---|---|
| `x` | `[M_buf, K]` (leading 1s allowed) ROW_MAJOR bf16 (the dispatch output, tilized in-op) or TILE bf8 | the dispatched buffer; local expert `e` owns rows `[start[gid], start[gid] + counts[gid])`, `gid = idx_table[e]` |
| `counts`, `region_offsets` | `[G]` or `[1, G]` uint32 ROW_MAJOR DRAM | per GLOBAL expert token count and first row (the routing setup's outputs) |
| `idx_table` | `[E]` uint32 ROW_MAJOR DRAM | local slot -> global expert id (sharded per chip) |
| `gate_projs`, `up_projs` | `E` x `[K, H]` TILE bf4 / bf8, DRAM interleaved or ND-sharded one tile-row tall and `per_core_N * 32` wide | `TtRoutedExpert`'s per-expert weights, reused as-is |
| `down_projs` | `E` x `[H, K]`, same placement | |
| output | `[M_buf, K]` TILE bf8: fresh for a ROW_MAJOR `x`; `x` itself (in place) for a TILE `x` unless `output=` is given | expert outputs at the expert regions, other rows untouched |

The kernels read `counts` / `idx_table` / `region_offsets` themselves (one core reads the pages and multicasts
them), size each expert's chunk loop from its own count at runtime (`kernels/adaptive_chunk.hpp`) and skip
experts with count 0, so a call needs no host-side token counts and is trace-safe.

## Kernels and program

Copied from the shared op (`ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn`):

* `reader_geglu_routed_expert.cpp` (NCRISC): x rows of the expert region (sticks or tiles) multicast along the
  M-row, gate weights read from DRAM and multicast down the N-column, the activated block multicast for the down
  matmul; the bias (`FUSE_BIAS`) path is removed.
* `writer_geglu_routed_expert.cpp` (BRISC): the `up` weights read on NoC 1 (UP_SPLIT), half of each down K-block
  (DOWN_SPLIT), the gate/up multicast on NoC 1 (IN1_WRITER_MCAST, `GEMMA4_GEGLU_WRITER_MCAST=0` or
  `DS_NO_WRITER_MCAST=1` hands it back to the reader's NoC 0), and the output drain at the expert's region offset.
* `compute_geglu_routed_expert.cpp` (TRISC): fused gate+up matmuls with packer-L1 accumulation across K-blocks,
  then `gelu_tanh_tile` on each gate tile in DST between the copy out of the accumulator and the pack (where the
  shared kernel runs `silu_tile`), the elementwise multiply with `up`, and the down matmul. The SwiGLU-OAI /
  SiTU-GLU / clamped-SiLU variants and the bias branch are gone.
* `adaptive_chunk.hpp`, `weight_runs.hpp`: verbatim (the runtime chunk picker all three kernels agree on, and
  the ND-shard-aware coalesced weight reads).

`op.py` reproduces the shared program factory host side: per-core N split (`ceil(N / 11)`), subblock widths
(ragged down grid below 3 tiles), the L1 fit of `per_core_M` x gate/up K-block width (and the widen-M rule),
chunk sizing (CBs sized to `per_core_M = 4`, runtime chunks of `<= 32` tile-rows), ring-granule checks, the
staggered DRAM-sender placement (weight sender of column `gx` on row `gx % 8`, x sender of row `gy` on column
`(gy + 1) % 11`), LoFi / bf16 DST. Compile-time args and the per-core runtime args are built once per shape
(`_build_plan`); buffer addresses are common runtime args rebuilt per call (generic_op's program cache keys on
the descriptor without them and re-applies them on a hit).

## Model wiring

`GEMMA4_GEGLU_CUSTOM_OP=1` makes `Gemma4DispatchExperts` (`tt/experts/dispatch_prefill.py`) run this op on the
weights, index table and buffers its `TtRoutedExpert` already holds instead of that module's fused op; the
default stays on `TtRoutedExpert`.

## Tests

`tests/unit/custom_ops/test_geglu_routed_expert.py`: (a) one chip and (b) the 2x4 block (every chip its own
experts, buffer, counts and region offsets), small shapes with both x layouts, and the 26B-A4B dispatch shapes
(hidden 2816, intermediate 704, 4 local experts, 4096 max tokens per expert, counts covering a full region, a
multi-chunk tail, a single partial tile and an empty expert); PCC vs the torch golden and vs the shared op on
the same inputs, and the warm per-call time of both.
