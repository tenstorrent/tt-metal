# KDA multi-group wrap plan

Status: approved design, implementation planned
Parent work: `tt-metal_tracker-ea6`

## Verdict

Restore the normal performance-selected group count for split offsets. The
boundary device will summarize its physical groups into two logical sets,
`head` and `tail`; the existing SP reduction consumes only `head`, the existing
SP prefix result supplies the tail seed, and one device-local segmented affine
prefix produces the correct seed for every group. The recurrent scan remains
one invocation and processes every group once in parallel.

## Problem and goal

- **Observed:** `C=2560` contains 80 32-row chunks and normally uses four groups
  of 20 chunks (`tt/kda/config.py:12,35-39,71-78`).
- **Observed:** every split currently sets `group_chunks = num_chunks`, forcing
  those 80 chunks into one serial group (`tt/kda/recurrence.py:452-464`).
- **Observed:** the 5k hardware run passes but measures +17.42% to +20.48% split
  latency with ~18.5% run drift; this proves a large problem exists but is not a
  precise program-level attribution (`kda_device_local_wrap_plan.md:116-125`).
- **Goal:** choose `group_chunks` only from geometry, configuration, and worker
  budget, never from whether an offset splits a device.
- **Non-goals:** remove final-state replication, change convolution routing,
  change the public `KdaState` contract, or add another recurrent scan call.

## Current system

For an unsplit device, each group produces one affine summary. The summaries
are reduced to one device transform, the SP prefix computes the device entry
state, and `affine_exclusive_scan` derives every local group seed before the
groups scan in parallel (`tt/kda/recurrence.py:478-539`).

For a split, the summary kernel already snapshots the head transform and the
recurrent kernel already reloads `tail_state` at the wrap
(`recurrent_chunk_scan.cpp:225-303,305-364`). This is sufficient for one group,
because the same core carries the reloaded state through the entire tail. It is
insufficient for multiple groups, because later tail groups execute concurrently
and require their entry states before recurrent scanning begins.

## Approved design

For boundary device 0, the global order is:

```text
D0 head groups -> D1 groups -> ... -> Dlast groups -> D0 tail groups
```

### 1. Produce head and tail summary sets

One summary invocation emits four tensors with fixed group-shaped outputs:

```text
head_A, head_B: used by normal device reduction and SP prefix
tail_A, tail_B: retained locally for the boundary tail prefix
```

On ordinary devices, `head` contains every full group transform and `tail` is
ignored. On the boundary device:

- groups before the wrap contribute to `head`;
- groups after the wrap contribute to `tail`;
- if the wrap straddles a group, that core snapshots the head transform,
  reinitializes its affine-summary state, and emits the tail transform without
  reprocessing any chunk;
- inactive positions are affine identity `(I, 0)`, keeping shapes uniform.

The summary output arity changes only in the explicitly requested segmented
mode; the existing two-output API and baseline program remain untouched.

### 2. Keep reduction and SP prefix algebra unchanged

`reduce_affine_transforms(head_A, head_B, G)` produces:

- the whole-device transform on ordinary devices;
- only the boundary head transform on the boundary device.

The existing SP prefix composes those device transforms in virtual order. Its
final carry is already the entry state for the boundary tail; no second SP
prefix or collective is added.

### 3. Add one device-local segmented affine-prefix program

Extend the affine exclusive-scan primitive with optional tail summaries,
`tail_state`, wrap indicator, and wrap group metadata. It still returns exactly
one `[B*H*G,K,V]` group-seed tensor.

- Ordinary devices execute the existing single prefix and ignore tail inputs.
- The boundary device executes a head prefix seeded by its device entry state
  and a tail prefix seeded by the SP prefix's final carry inside the same device
  program.
- The output chooses the head-derived seed for every head group and for a
  straddling group; it chooses the tail-derived seed for groups wholly after the
  wrap.
- This is one TTNN operation. Do not construct both seed candidates mesh-wide
  and select with generic tensor operations.

### 4. Run the existing parallel recurrent scan

Restore `G > 1` on split offsets. The program factory converts the device-wide
wrap chunk into `(wrap_group, chunk_within_group)`:

- only the boundary device's straddling group reloads `tail_state` inside its
  chunk loop;
- other boundary groups and every ordinary-device group run without a reset;
- all groups process their normal fixed chunk count once and concurrently;
- select the last physical group's state before the existing split-state
  replication.

## Contracts and invariants

- `group_chunks` and `groups_per_head` are identical for offset zero, rotation,
  and split offsets at the same geometry.
- Every prepared chunk is summarized once and recurrently scanned once.
- The boundary head participates in SP reduction; the boundary tail never does.
- The SP prefix final carry is the sole tail-prefix seed.
- A straddling group receives its head entry as its group seed and reloads the
  tail seed exactly at the wrap chunk.
- Groups wholly after the wrap receive tail-derived entry states upfront.
- Ordinary devices are bit-identical to the existing grouped path when their
  local wrap indicator is zero.
- Offset zero preserves the existing two-output summary path and program shape.
- One summary, one SP prefix, one affine exclusive-scan operation, and one
  recurrent scan appear per layer; no generic select graph is introduced.
- Output placement and replicated recurrent/convolution state contracts remain
  unchanged.

## Implementation map

1. `recurrent_chunk_scan` summary mode
   (`ttnn/.../recurrent_chunk_scan/`): add segmented-summary output mode, derive
   each folded worker's group index, emit head/tail `(A,B)`, validate fixed
   shapes, and update its performance model.
2. `affine_exclusive_scan` (`ttnn/.../affine_exclusive_scan/`): add an optional
   device-local second seed and summary set; execute the second prefix only on
   the wrap device and write one selected group-seed tensor.
3. Python recurrence (`models/.../tt/kda/recurrence.py`): remove the split
   `G=1` override, feed only head summaries to reduction, pass the SP final carry
   and tail summaries into the segmented prefix, map the wrap to one group-local
   reset, and select the last group state for replication.
4. Tests and profiling: extend op oracles first, then component/layer matrices,
   trace/program-count assertions, and interleaved performance measurement.

## Validation

| Contract | Evidence required |
| --- | --- |
| Correct head/tail summaries | Op tests for `G={1,2,4}` and wrap before, inside, and after group boundaries; composition of head + middle devices + tail equals the unsplit full transform |
| Correct group seeds | Segmented affine-prefix tests with head/tail seeds separated by five orders of magnitude; ordinary-device result bit-identical to current exclusive scan |
| One-pass recurrence | Grouped scan oracle for wraps at chunk 1, 19, 20, 21, 40, and 79; every chunk appears exactly once |
| Mesh correctness | `SP2xTP4` and `SP4xTP2`, boundary on each SP rank, output/recurrent/convolution state PCC gates |
| Trace safety | repeated trace output bit-identical; cache-hit tensors rebound; all 32-aligned offsets use one recurrent scan |
| Performance | native C++ build; interleaved `C=640` and `C=2560` runs; 5k device profile confirms four groups and attributes remaining delta |

Required commands include:

```text
cmake --build build --target ttnn/install -j 24
scripts/run_safe_pytest.sh --run-all -q tests/ttnn/nightly/unit_tests/operations/experimental/kda/test_summarize_chunk_recurrence.py --tt-arch blackhole
scripts/run_safe_pytest.sh --run-all -q tests/ttnn/nightly/unit_tests/operations/experimental/kda/test_recurrent_chunk_scan.py --tt-arch blackhole
scripts/run_safe_pytest.sh --run-all -q models/demos/deepseek_v3_d_p/tests/kda/layer/test_offset.py --tt-arch blackhole
scripts/run_safe_pytest.sh --run-all -q models/demos/deepseek_v3_d_p/tests/kda/perf/test_offset_perf.py -k T5120 --tt-arch blackhole
```

## Tracked execution

```text
tt-metal_tracker-ea6.5   emit head/tail summaries
tt-metal_tracker-ea6.13  add one device-local two-seed affine prefix
              \          /
               ea6.6     integrate G>1 wrap recurrence
                 |
               ea6.7     exhaustive correctness and performance gate
```

`ea6.6` depends on both `ea6.5` and `ea6.13`; `ea6.7` depends on `ea6.6`.
The old G=1 prototype `ea6.10` is superseded by the completed device-local
implementation `ea6.11` and is not part of this plan.

## Decisions and risks

- **Approved:** two logical prefix seeds on the boundary device; tail summaries
  remain local and do not participate in SP reduction.
- **Proposed:** implement both logical prefixes inside one device operation.
  Two mesh-wide `affine_exclusive_scan` calls would repeat launch and prefix work
  on ordinary devices.
- **Risk:** fixed identity slots and four tail-summary outputs increase local
  summary traffic. Measure it; do not trade away group parallelism without data.
- **Risk:** a straddling worker participates in the head output but starts the
  tail prefix. Kernel tests must cover both roles explicitly.
- **Constraint:** `B*H*G` must remain within the existing 128-worker coordinate
  table. K3's `24*4=96` fits; group selection continues to enforce this budget.
- **Unknown:** exact 5k program-level savings remain unmeasured because the
  current benchmark profiles only `C=640` and the 5k wall run drifts heavily.
