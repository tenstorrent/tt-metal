# AutoDebug: batch-32 prefill matmul block geometry

## Headline finding

The stage-review finding is valid and is not an API limitation. At the measured
batch-32, sequence-128 shape, all material prefill projections use
`in0_block_w=1` because `_prefill_matmul_config` applies one coarse
`rows >= 2048` branch to QKV, output, packed gate/up, and down. That branch was
a conservative response to the packed gate/up block-4 L1 failure, but it also
suppresses legal block-2-or-larger configurations for every role.

The first adapted experiment should keep the existing 8x8 grid and launches,
give prefill its own per-role block fields, and explicitly reduce the inner
`out_block_h` where necessary. TTNN's 2-D multicast config supports this:
`out_block_h` and `out_block_w` are independent divisors of `per_core_M` and
`per_core_N`. The current Python call omits them, so nanobind silently makes
each inner output block as large as the whole per-core output. Smaller inner
output blocks reduce all three dominant circular-buffer allocations and make
`in0_block_w >= 2` comfortably feasible.

This report is source-only. No TT hardware command was run, and no
implementation file was edited. L1 numbers below are source-derived admission
estimates; compilation, PCC, warmed latency, and `tt-perf-report` remain the
decisive experiments.

## Direct observations

### Model-side policy

- `OptimizationPolicy` has role-specific block fields only for decode
  (`optimized_decoder.py:70-88`). Prefill has no policy fields for core grid,
  K block, or inner output blocks.
- `_prefill_matmul_config` fixes `grid_x=8`, chooses at most eight grid rows,
  and forces `in0_candidates=(1,)` whenever fused rows are at least 2048
  (`optimized_decoder.py:363-385`).
- The batch-32 QKV and output projections have 4096 fused rows. The packed MLP
  is split into two batch-16 calls, so each gate/up and down projection has
  exactly 2048 fused rows (`optimized_decoder.py:484-502`). All therefore enter
  the same block-1 branch.
- The long-context sequence chunker is deliberately inactive at the measured
  serving shape. It only runs when fused rows exceed 32768
  (`optimized_decoder.py:417-449`).

### Profile evidence

`tracy/prefill_b32_perf_report.txt` records these material rows in the
22.245 ms device window:

| Role | Logical matmul | Calls | Device time per call | Report share per call |
| --- | --- | ---: | ---: | ---: |
| packed QKV | `b={32} x 128 x 3072 x 9216` | 1 | 1967 us | 8.6% |
| output | `b={32} x 128 x 3072 x 3072` | 1 | 657 us | 2.9% |
| packed gate/up | `b={16} x 128 x 3072 x 16384` | 2 | 1766/1768 us | 7.8% each |
| down | `b={16} x 128 x 8192 x 3072` | 2 | 760 us | 3.3% each |

Every one receives the same advice: "`in0_block_w=1` is small, try
`in0_block_w=2` or above." The raw Tracy attributes confirm the actual
programs, not merely the report heuristic:

| Role | Grid | `per_core_M/N` | implicit `out_block_h/w` | `in0_block_w` |
| --- | --- | --- | --- | ---: |
| QKV | 8x8 | 16 / 36 | 16 / 36 | 1 |
| output | 8x8 | 16 / 12 | 16 / 12 | 1 |
| gate/up | 8x8 | 8 / 64 | 8 / 64 | 1 |
| down | 8x8 | 8 / 12 | 8 / 12 | 1 |

The six projection calls account for about 7.68 ms of reported device time, so
this is a material optimization target rather than cosmetic advice.

## TTNN config and validation contract

The 2-D config constructor defaults omitted `out_block_h/w` to
`per_core_M/N` (`ttnn/cpp/ttnn/operations/matmul/matmul_nanobind.cpp:151-181`).
The validator requires:

- nonzero `in0_block_w`, `per_core_M`, and `per_core_N`;
- `out_subblock_h/w` divide `out_block_h/w`;
- `out_block_h/w` divide `per_core_M/N`;
- the subblock tile product fits destination registers;
- the number of M/N work blocks fits the selected grid; and
- `Kt % in0_block_w == 0`.

Those checks are at
`ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp:145-193`,
`:245-279`, `:422-448`, and `:1773-1779`. None requires
`out_block_h == per_core_M` or `out_block_w == per_core_N`.

The factory's relevant circular-buffer sizes are
(`matmul_multicore_reuse_mcast_2d_program_factory.cpp:137-167`):

```text
in0_CB = 2 * out_block_h * in0_block_w * in0_tile_bytes
in1_CB = 2 * out_block_w * in0_block_w * in1_tile_bytes
out_CB = out_block_h * out_block_w * output_tile_bytes
```

The factor two is the common multicast input buffering depth
(`matmul_utilities.hpp:20-23`). For the selected policy, BF16 activation/output
tiles are 2048 bytes and BFP4 weight tiles are 576 bytes. The output and
intermediate CB share storage for this DRAM-interleaved BF16 output.

The earlier L1 failure established a 111360-byte allocation base and
1572864-byte Blackhole L1 limit. Using those observed constants, the current
serving configs have the following estimated allocation endpoints:

| Role | Current endpoint | Estimated headroom |
| --- | ---: | ---: |
| QKV block 1, inner 16x36 | 1398016 B | 174848 B |
| output block 1, inner 16x12 | 583936 B | 988928 B |
| gate/up block 1, inner 8x64 | 1266432 B | 306432 B |
| down block 1, inner 8x12 | 354560 B | 1218304 B |

Simply changing all four roles to block 2 while retaining the implicit inner
blocks is source-legal and estimated to fit:

| Role | Block-2 endpoint | Estimated headroom |
| --- | ---: | ---: |
| QKV | 1505024 B | 67840 B |
| output | 663296 B | 909568 B |
| gate/up | 1372928 B | 199936 B |
| down | 401152 B | 1171712 B |

QKV is too close to the limit to be a good final configuration, especially for
a watcher-clean gate, but this is the smallest discriminator for the coarse
policy bug. A compile failure there must be adapted with smaller inner output
blocks; it would not prove block 2 is unsupported.

## Required experiment-policy surface

Keep decode fields unchanged and add prefill-specific fields. `None` preserves
the current heuristic, making each experiment isolated:

```python
prefill_core_grid: tuple[int, int] = (8, 8)
prefill_qkv_in0_block_w: int | None = None
prefill_o_proj_in0_block_w: int | None = None
prefill_gate_up_in0_block_w: int | None = None
prefill_down_in0_block_w: int | None = None
prefill_qkv_out_block_h: int | None = None
prefill_o_proj_out_block_h: int | None = None
prefill_gate_up_out_block_h: int | None = None
prefill_down_out_block_h: int | None = None
prefill_qkv_out_block_w: int | None = None
prefill_o_proj_out_block_w: int | None = None
prefill_gate_up_out_block_w: int | None = None
prefill_down_out_block_w: int | None = None
prefill_sequence_chunk: int | None = None
prefill_batch_chunk: int | None = None
```

Map `gate` and `up` to the gate/up fields when testing separate projections.
Pass `weight_name` into `_prefill_matmul_config` so the role selects its own
values. Validate the configured prefill grid against
`compute_with_storage_grid_size()` in `__init__`, just as decode does.

The adapted config builder should use:

```python
grid_x = min(policy.prefill_core_grid[0], n_tiles)
grid_y = min(policy.prefill_core_grid[1], row_tiles)
per_core_m = math.ceil(row_tiles / grid_y)
per_core_n = math.ceil(n_tiles / grid_x)
in0_block_w = role_override or current_heuristic
out_block_h = role_override or per_core_m
out_block_w = role_override or per_core_n

if k_tiles % in0_block_w:
    raise ValueError(...)
if per_core_m % out_block_h or per_core_n % out_block_w:
    raise ValueError(...)
out_subblock_h = 1
out_subblock_w = _largest_divisor(out_block_w, (4, 3, 2, 1))
```

Pass both `out_block_h` and `out_block_w` explicitly to
`MatmulMultiCoreReuseMultiCastProgramConfig`. Choosing `grid_y` directly rather
than only from exact divisors is necessary to exercise 8x10 or 11x10; TTNN
already supports a partial last work block via ceiling division and validates
that the block count fits the grid.

## Adapted candidate ladder

Each first API/L1 error must advance to the next adaptation instead of rejecting
the family.

### C0: minimal block-2 discriminator

Same 8x8 grid, launches, and implicit output blocks:

```python
prefill_*_in0_block_w = 2  # all four roles
```

Expected endpoints are in the block-2 table above. This directly proves or
refutes the overly broad threshold. QKV's 67840-byte estimate is intentionally
tight, so C0 is a discriminator, not the preferred final candidate.

### C1: block 2 with smaller inner M blocks

Keep one call per current projection and set:

| Role | `in0_block_w` | `out_block_h` | `out_block_w` |
| --- | ---: | ---: | ---: |
| QKV | 2 | 8 | implicit 36 |
| output | 2 | 8 | implicit 12 |
| gate/up | 2 | 4 | implicit 64 |
| down | 2 | 4 | implicit 12 |

All divisibility constraints hold. The estimated endpoints become:

| Role | Endpoint | Headroom |
| --- | ---: | ---: |
| QKV | 849664 B | 723200 B |
| output | 401152 B | 1171712 B |
| gate/up | 815872 B | 756992 B |
| down | 270080 B | 1302784 B |

This is the safest first complete candidate. It changes only the inner
scheduling blocks and retains the existing public tensor topology and number of
linear launches.

### C2: role-specific K-block sweep on C1

Sweep one role at a time at both batch 1 and batch 32, then combine only winners:

| Role | Legal source-derived sweep |
| --- | --- |
| QKV, `Kt=96` | `2, 3, 4, 6, 8` with `out_block_h=8` |
| output, `Kt=96` | `2, 3, 4, 6, 8, 12` with `out_block_h=8` |
| gate/up, `Kt=96` | `2, 3, 4, 6, 8` with `out_block_h=4` |
| down, `Kt=256` | `2, 4, 8, 16, 32` with `out_block_h=4` |

Do not infer the best block from another role: QKV/gate-up have wide N while
down has a much wider K.

### C3: smaller inner M and N blocks

This tests whether larger K blocks outperform the extra inner-output loops:

| Role | `in0_block_w` | `out_block_h/w` | Endpoint | Headroom |
| --- | ---: | --- | ---: | ---: |
| QKV | 12 | 4 / 12 | 572160 B | 1000704 B |
| output | 12 | 4 / 12 | 572160 B | 1000704 B |
| gate/up | 8 | 4 / 16 | 520960 B | 1051904 B |
| down | 32 | 4 / 12 | 1176320 B | 396544 B |

Here 12 divides QKV/output/gate Kt=96, 8 divides gate Kt=96, and 32 divides
down Kt=256. All proposed inner blocks divide their per-core dimensions. This
candidate attacks the profiler advice most aggressively without adding
slice/concat launches.

### C4: 2-D grid geometry

The profile reports 110 available worker cores, while the current prefill uses
64. Add at least these grid controls:

1. 8x8 current baseline.
2. 8x10: preserves the useful N divisors (`per_core_N=36/12/64/12`) while
   distributing the 128/64 M tiles across ten rows (`per_core_M=13/7`).
3. 10x8: tests more N columns, accepting padded final N blocks.
4. 11x10 only after 8x10: uses the full reported rectangular extent but gives
   awkward `per_core_N=27/9/47/9`, including prime 47 for gate/up.

For 8x10, set `out_block_h=1`; C3's inner N blocks remain legal. A concrete
starting point is QKV 8/1/12, output 12/1/12, gate-up 8/1/16, and down
32/1/12 (`in0_block_w/out_block_h/out_block_w`). Their estimated endpoints are
roughly 500480, 350976, 324352, and 709376 bytes respectively. Measure rather
than assuming more cores win: `out_block_h=1` creates more inner M blocks and
the multicast topology can outweigh the extra parallelism.

For 11x10, use `out_block_h=1` and only N blocks that divide
`27/9/47/9`; gate/up therefore needs `out_block_w=1` or 47. That prime-width
case is why 11x10 is an adapted follow-up, not the first geometry.

### C5: internal sequence-M chunking

If one-program candidates compile but do not improve, test
`prefill_sequence_chunk=64`. Slice only `shape[-2]`, run the same device linear
per chunk, and concatenate with `ttnn.concat(..., dim=-2)`, exactly like the
existing long-context repair. At batch 32:

- QKV/output become two 2048-row calls;
- each existing batch-16 MLP group becomes two 1024-row calls; and
- block 4 is estimated to fit every role.

This preserves public sequence length, including a non-aligned final chunk, and
does not alter attention, cache, or paging semantics. It adds launches, slices,
and concats, so retain it only if the block-width gain beats those costs in the
full warmed prefill.

### C6: internal batch-M chunking

As a separate hypothesis, test `prefill_batch_chunk=8`. For a 4-D tensor,
identify the unique leading dimension equal to `self.batch`, slice that axis in
groups of eight, and concatenate on the same axis. This gives 1024 fused rows
per serving call and permits block 4 under the existing small-row rule.

Do not combine batch and sequence chunking initially. QKV and output would grow
from one call to four calls, so batch chunking has a higher launch-cost risk
than the inner-block candidates. It nevertheless preserves each user's
sequence, ordering, cache mapping, and public batch 32.

## Experiment order and gates

1. Add policy fields and static validation only; preserve the default.
2. Run C0 to discriminate the coarse threshold.
3. If C0 fails or has weak watcher headroom, adapt to C1. Do not reject block 2.
4. Sweep C2 one role at a time, then test the cumulative winner.
5. Cross the cumulative winner with C3 inner-N blocks and C4 grids.
6. Test C5 and C6 only if the no-extra-launch candidates do not improve the
   23.011 ms batch-32 baseline.
7. For every candidate, measure warmed prefill and PCC at both batch 1 and 32.
   A serving-only improvement is insufficient if batch 1 regresses materially.
8. Reprofile the best candidate. Require all material rows to show
   `in0_block_w >= 2`, check that device-op growth is justified, and inspect new
   advice rather than relying only on wall time.
9. Run real-weight prefill PCC, non-aligned lengths 31/33/65, longest
   non-aligned 131071, repeated runs, and a separate watcher-clean optimized
   correctness test. Chunk policies must never introduce a public alignment
   restriction.

Suggested policy names in `optimized_decoder_perf.py`:

```text
prefill_b2_default_blocks
prefill_b2_inner_m
prefill_role_<role>_b<width>
prefill_inner_mn
prefill_grid_8x10
prefill_grid_10x8
prefill_grid_11x10
prefill_seq64
prefill_batch8
```

Record compile/L1 failures as candidate evidence with the exact requested and
available bytes. A first validation or allocation error is a routing signal to
reduce `out_block_h/w`, then M chunk size, not grounds to dismiss larger
`in0_block_w`.

## Semantics and scope

All proposed candidates stay in `optimized_decoder.py` and its tests/docs.
They do not touch decode, multichip, full-model, or vLLM work. Inner
`out_block_h/w` and core-grid changes are program scheduling only. Optional M
chunking uses device `slice`/`concat`, preserves logical batch and sequence
dimensions, and leaves paged KV fill and SDPA after the reconstructed QKV
tensor. No candidate requires Torch conversion, host fallback, or a public
`seq_len % chunk == 0` restriction.

## AutoDebug workflow note

The required fresh `.agents/scripts/autodebug.sh` runner was started in an
isolated temporary directory. Its Codex sandbox could not create the bubblewrap
user namespace, so neither it nor its delegated processes could read the
checkout; the blocked run was terminated without an output report. The findings
above were then independently checked against the current model source, raw and
filtered Tracy artifacts, nanobind constructor, TTNN validation, and 2-D
multicast CB-sizing source. No TT hardware was touched during this
investigation.
