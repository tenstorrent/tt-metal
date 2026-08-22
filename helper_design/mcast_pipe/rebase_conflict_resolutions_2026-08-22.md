# `sjovic/mcast-migration` rebase conflict resolutions (2026-08-22)

## Rebase setup

- Source branch: `sjovic/mcast-migration`
- Original source tip: `6d26d60e2aca90fff8a511f4ad49fe6a3d5c5a04`
- Old helper baseline / replay cut point: `dc9282be7d5e9d5a4b9137c1bf327de8d923e18e`
- New helper baseline: `llk_helper_library` at
  `ce76cb1faa08ebee0a255979845abf289d0298b1`
- Rebase command:

  ```bash
  git rebase --onto llk_helper_library \
      dc9282be7d5e9d5a4b9137c1bf327de8d923e18e \
      sjovic/mcast-migration
  ```

This selects the 81 migration commits after the old helper baseline and replays
them on the latest helper branch.

## Approved resolution 1: matmul in1 compile-time argument offsets

- Original commit: `bcf1c9d5b5e` (`Migrate matmul in1 to multicast host helper`)
- Conflicted file:
  `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp`

The new helper baseline added `compact_output`, while the migration made the
multicast argument prefix variable-sized. Preserve both changes:

```cpp
constexpr bool compact_output = get_compile_time_arg_val(in1_post_mcast_ct_offset + 18);
constexpr auto in1_args = TensorAccessorArgs<in1_post_mcast_ct_offset + 19>();
```

Rationale:

- The baseline's fixed `TensorAccessorArgs<33>()` assumes the legacy fixed-size
  multicast layout.
- The migration's `TensorAccessorArgs<in1_post_mcast_ct_offset + 18>()` predates
  `compact_output` and would decode that scalar as accessor data.
- The factories append the two fused-op arguments, then `compact_output`, then
  the tensor accessor arguments.

## Approved resolution 2: multicast control-signal tests

- Original commit: `594374f8acc` (`Add mcast counter signal coverage for sort`)
- Conflicted files:
  - `tests/ttnn/unit_tests/kernel_lib/kernels/pipe_signal_receiver.cpp`
  - `tests/ttnn/unit_tests/kernel_lib/kernels/pipe_signal_sender.cpp`
  - `tests/ttnn/unit_tests/kernel_lib/test_mcast_pipe.py`

Keep the `llk_helper_library` versions of all three files.

Rationale: the helper baseline already contains the migration commit's
non-handshaked counter coverage and extends it with both handshake policies,
explicit Flag/control-value coverage, and matching sender/receiver
`control_value` compile-time arguments. Taking the older migration versions
would remove newer coverage and shift `TensorAccessorArgs` incorrectly.

The non-conflicting documentation changes from `594374f8acc` were retained
during the attempted rebase.

## Unresolved stop point

The rebase was stopped and aborted at original commit `46a781123f1` (`Migrate
sort control channel to mcast pipe`). It conflicts in the three single-row
multi-core sort dataflow kernels and `sort_program_factory.cpp`. No resolution
for that conflict was approved or applied.

The latest helper baseline has ported this sort implementation to the newer
`ProgramSpec`, named argument, named semaphore, and `DataflowBuffer` APIs. A
future retry should preserve that new structure and transplant only the
multicast-pipe control-channel behavior rather than taking the older migration
files wholesale.
