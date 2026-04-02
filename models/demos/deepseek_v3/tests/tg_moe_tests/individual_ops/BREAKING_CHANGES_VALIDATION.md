# MoE Breaking Changes Validation

This document lists breaking changes that are confirmed to be caught by the TG MoE E2E test and/or individual ops tests.

## Confirmed Breaking Changes

The following off-by-one errors in MoE operations are validated to cause test failures:

1. **all_to_all_dispatch_metadata_device_operation.cpp**
   - Location: `dispatch_devices` calculation
   - Type: Off-by-one error in device count

2. **writer_all_to_all_dispatch_metadata.cpp**
   - Location: `token_indices` pointer offset
   - Type: Off-by-one error in pointer arithmetic

3. **selective_reduce_combine reader.cpp**
   - Location: `rem` calculation
   - Type: Off-by-one error in remainder calculation

4. **selective_reduce_combine writer.cpp**
   - Location: `t_idx` calculation
   - Type: Off-by-one error in token index

5. **selective_reduce_combine_program_factory.cpp**
   - Location: `total_tokens` calculation
   - Type: Off-by-one error in token count

6. **moe_compute_device_operation.cpp**
   - Location: `tilize_input_shape[0]` indexing
   - Type: Off-by-one error in shape indexing

7. **moe_compute_program_factory.cpp**
   - Location: `experts_per_device` calculation
   - Type: Off-by-one error in expert count

## Additional Coverage (Likely Validated)

Brief manual checks suggest the test would also catch the following types of breaking changes:

- Runtime argument ordering or omission
- Circular buffer (CB) index changes
- CB page count modifications
- CB property changes
- Removed synchronization barriers
- Deleted runtime overrides

## Expanding Coverage

Test coverage can be expanded based on feedback and specific requirements. Current validation focuses on the most critical breaking changes identified during development.
