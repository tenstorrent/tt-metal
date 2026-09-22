# AutoDebug: three-reader DRAM-sharded decoder projections

Status: source diagnosis experimentally confirmed by the main agent's serialized
device experiment; output-storage policy fix retained. This investigator only
inspected source and artifacts and wrote this report; implementation edits and
device execution were performed by the main agent.

## Verified follow-through

After the initial source finding was communicated, the main agent changed only
the `per_core_N` formula to derive storage width from bank-shard width times bank
count, as shown below. The real-weight, real-activation three-reader full layer
now completes. Independently inspected artifacts:

- `readers3_storage_fixed.json` and `readers3_storage_fixed.log`;
  exact invocation is recorded under that name in `commands.log`.
- Source hash: `aaa91d4825fdb8f757b81d6cd04af840ecfdc534f301e0d245694847e83906ea`.
- Layer 3, length 128, batch 1: prefill and traced decode PCC both
  `0.9999038577079773`; changed-input trace PCC `0.9998857975006104`.
- `repeat_bitwise_equal=true`, 30 traced timing samples,
  `runtime_fallback_audit=passed`, `unused_pages_unchanged=true`.
- Traced decode median is approximately 0.901 ms. This proves candidate
  legality/correctness at this boundary; it does not establish that three
  readers are the fastest policy.

The generalized output-storage formula was tested as one changed program-config
field across the projection calls. A separate down-only override was not run.
Batch 32, both layer kinds and final-stage coverage remain the main agent's
responsibility. The hypotheses and initial focused test design below preserve
the reasoning that led to this verified fix.

## Evidence and scope

- Stage 3, single-device Qwen/Qwen3.8-27B; starting commit `ad43d1388fd`.
  Scope is `tt/optimized_decoder.py`, model tests, and docs; no native changes.
- `geometry_r3_b{4,8,16}.log` all report successful prefill PCC
  `0.9999038577079773`, then fail while constructing the **MLP down projection**
  program in `_finish`, at native factory line 814:
  `Worker 7-2 has no storage area assigned`.
- The source hash saved with `geometry_r3_b4` is
  `446a4e2ae523a12b7b98e35f815bd86e982879fe6858ebebb12f3c7e18c3909f`.
  The shared live implementation is being extended by the main agent; line
  numbers in the original traceback refer to that saved candidate.
- The saved work log records Blackhole, an 11x10 compute grid and eight DRAM
  banks. Its measured environment takes precedence over generic runner text.
- Existing one-reader and two-reader geometry runs provide controls. The
  three-reader failure occurs after the attention and gate/up projection calls
  return; their placement is therefore not inherently unsupported.
- Exact original invocations are preserved in `commands.log` under
  `geometry_r3_b4`, `geometry_r3_b8`, and `geometry_r3_b16`. All three retain
  `down_block=17`; only attention/gate/up block widths vary.

## Ranked hypotheses

### 1. Output storage ends before the final padded reader begins — verified

The current weight upload correctly pads each bank's **shard allocation** to an
integer number of reader tiles. `_linear` independently computes
`per_core_N=ceil((logical_N/32)/input_storage_cores)`. For the down projection,
that sizes output storage to logical N with no spare output tiles.

Native source establishes the mismatch:

- `matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:127–167` removes
  whole padding-only banks, then chooses
  `reader_tiles=ceil(N_tiles/active_workers)`. Multi-reader bank shards must equal
  `readers_per_bank * reader_tiles`.
- The factory's `N_tiles` comes from **the weight tensor's padded shape**, not
  the full bank allocation (`:950–951`, `:1046`). Enlarging `ShardSpec.shape[1]`
  does not itself make that argument equal the whole bank allocation.
- The program's `per_core_N` becomes **output storage width**, not the width
  computed by each reader (`:1069–1071`, `:61`, `:159–160`).
- `matmul_device_operation.cpp:2513–2549` builds
  `ceil(N_tiles/per_core_N)` output storage cores, independently of the input
  core count and reader count.
- The writer planner computes that same output core count at factory `:761`.
  When `reader_tiles < storage_tiles`, every worker must start before its end
  (`:813–848`). This branch can clip the last worker's second write, but does
  not allow a later worker to start entirely beyond output storage.

For the failing model boundary, K=17408, N=5120, N_tiles=160, input storage
cores=8, input shard width=68 tiles, and `in0_block_w=17`:

| Readers/bank | Bank shard width (tiles) | Active workers | Tiles/reader | Scheduled width (tiles) | Output storage | Last reader start |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 8 | 20 | 160 | 8 x 20 = 160 | 140 |
| 2 | 20 | 16 | 10 | 160 | 8 x 20 = 160 | 150 |
| 3, failing | 21 | 24 | 7 | 168 | 8 x 20 = 160 | **161** |
| 3, proposed | 21 | 24 | 7 | 168 | 8 x 21 = 168 | 161 |

No bank is padding-only: 168-160=8 padding tiles is less than the 21-tile bank
shard. All 24 readers remain. With the failing 20-tile output shard, worker i=22
begins at tile 154, writes the remaining six tiles, and advances the storage
cursor past the eighth output core. Worker i=23 begins at tile 161 and hits
line 814. With a 21-tile output shard, worker i=23 begins at tile 161 within the
last shard and writes its final seven tiles. Logical output remains 5120.

The precise condition for the failing planner branch is:

```text
reader_tiles < storage_tiles
and (active_workers - 1) * reader_tiles >= ceil(N_tiles/storage_tiles) * storage_tiles
```

Full storage for every reader tile is a conservative fix; the current planner
can also pass with less storage if only the tail of the final reader is clipped.
For example, gate/up currently allocate 550 tiles for 552 scheduled tiles, but
the last reader starts at 529, so it receives a destination. Down is different:
its final reader starts entirely outside the allocation.

### 2. Bank allocation padding is being mistaken for tensor-shape padding — related, secondary

The bank allocation for down is 5376 columns, but logical N and normal tile-padded
N remain 5120. Both quantities are valid and intentional; they must be kept
distinct when selecting output storage. The test helper at
`test_matmul_dram_sharded.py:81–93` likewise pads the bank storage layout while
keeping logical N unchanged. Its BFP4 three-reader case uses N=5376 (168 tiles),
specifically divisible by the reader counts (`:267–270`), so it does not prove
the failing N=5120, eight-output-core combination works unchanged.

Do not enlarge only the weight shard again: native `:162–167` rejects bank
widths that do not match the inferred reader width, and the existing oversized
storage rejection test at `:315–347` explicitly covers this constraint.

### 3. Rank or input-K sharding is the cause — low confidence for this failure

The model supplies A as rank four `[1,1,batch,K]` before sharding, and B as rank
two `[K,N]`; native planning reads the trailing K/N dimensions. The existing
failure reaches writer assignment, after reader geometry and K divisibility
checks. Rank changes alone cannot alter the numerical storage mismatch above.
The down input also divides exactly: 544 K tiles / 8 cores = 68 tiles/core,
and 68 is divisible by block 17. There is no source evidence for changing this
input geometry to address line 814.

## Smallest model-local fix and focused experiment

Initial proposed experiment: change **only down `per_core_N` from 20 to 21** for the three-reader case.
Keep weight objects, rank, dtype, fidelity, input cores, block, and reader count
identical. This is the discriminating A/B: it changes the destination allocation
without changing the computation or weight layout.

The verified model-local conservative policy derives
the output storage width from the existing uploaded bank shard:

```python
# Scalar shape orchestration only; no host tensor data access.
bank_tiles = dram_weight.memory_config().shard_spec.shape[1] // 32
bank_count = self.device.dram_grid_size().x
storage_tiles = (bank_tiles * bank_count + cores - 1) // cores
# MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(per_core_N=storage_tiles)
```

For the current eight-bank, three-reader model shapes this yields:

| Projection | K | Logical N | Bank allocation N | Input cores | Proposed per_core_N | Actual output allocation (tiles) |
|---|---:|---:|---:|---:|---:|---:|
| Full Q/K/V/gate | 5120 | 14336 | 14592 | 10 | 46 | 10 x 46 = 460 |
| Linear QKV/Z/B/A | 5120 | 16512 | 16896 | 10 | 53 | 10 x 53 = 530 |
| Attention output | 6144 | 5120 | 5376 | 12 | 14 | 12 x 14 = 168 |
| MLP gate or up | 5120 | 17408 | 17664 | 10 | 56 | 10 x 56 = 560 |
| MLP down | 17408 | 5120 | 5376 | 8 | 21 | 8 x 21 = 168 |

The attention dimensions are taken from the pinned snapshot config: 24 full
attention heads, four KV heads, head dimension 256; linear key/value heads
16/48 with dimension 128. Linear B/A packing rounds each 48-wide piece to 64.

For future arbitrary shapes, verify the **actual** output core count
`ceil(N_tiles/storage_tiles)` and the condition above; do not assume it always
equals the requested input core count. For this model the table provides the
required output coverage. Keeping the formula scoped to the failing down
projection initially minimizes unproven performance/layout changes.

Originally suggested rerun, after that one-variable change; this exact label was
**not run** by this source-only investigator. The main agent ran the same policy
under `readers3_storage_fixed`, using the generalized formula as described above:

```bash
models/autoports/qwen_qwen3_8_27b/tests/run_optimization_experiment.sh geometry_r3_b4_storage21 \
  --layer 3 --length 128 --batch 1 --benchmark \
  --activations /home/mvasiljevic/qwen38-full-rerun/tt-metal-cache/optimized_decoder_activations \
  --policy '{"dram":true,"sharded_norm":true,"carry_residual":true,"attention_block":4,"gate_block":4,"up_block":4,"down_block":17,"attention_dtype":"bfloat4_b","attention_fidelity":"LoFi","gate_dtype":"bfloat4_b","gate_fidelity":"LoFi","up_dtype":"bfloat4_b","up_fidelity":"LoFi","down_dtype":"bfloat4_b","down_fidelity":"LoFi","attention_readers":3,"gate_readers":3,"up_readers":3,"down_readers":3}'
```

At the narrow projection boundary, record A/B logical and padded shapes, weight
bank shard spec, `per_core_N`, and output shard spec. Compare the same real
weight/input projection with the one-reader control after conversion to a
common interleaved shape. Then require the original real-activation runner's
traced PCC, repeat bitwise equality, changed-input trace correctness, and page
preservation checks. Repeat the adapted three-reader b8/b16 candidates only
after the b4 control validates the cause. Batch 32 and both layer kinds remain
necessary for final stage acceptance; no performance claim follows from source
arithmetic.

## Further legal adaptations if the focused fix exposes another boundary

1. **Explicit padded projection, then crop:** at weight setup, create a zero-
   padded rank-four down weight `[1,1,17408,5376]`, shard it as
   `[17408,672]` across eight banks, use `per_core_N=21`, compute output at
   5376 columns, and crop to 5120 before the residual/norm boundary. Keep the
   original weight for prefill and track semantic N separately. This makes the
   factory's N=168 tiles and storage/reader widths agree explicitly. Preserve
   batch shape when cropping. It is more invasive than the storage-only fix
   and should be a separate experiment, not bundled with it.
2. **One reader-width per output shard:** retaining logical N=5120 and bank
   width 21, set `per_core_N=7`. Output receives 23 seven-tile shards. The
   planner's other branch (`:849–906`) permits the final padding-only reader to
   have zero writes. This is another source-predicted legal model-local layout,
   but changes output placement more than the proposed 21-tile layout and must
   be measured through the following residual reshard.
3. If testing a rank hypothesis, reshape B to `[1,1,K,N]` while preserving its
   padded dimensions and memory geometry; run this separately. Merely promoting
   rank does not add storage. An explicit padded-shape view must be backed by
   actual zero-filled physical padding; a view is not a substitute for safely
   materializing new weight columns.

The current evidence supports adapting output storage and retesting. It does
not support rejecting the three-reader candidate as unsupported on this device.
