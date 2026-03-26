# Operator Reference Pointers

## Canonical simple eltwise op (full device operation pattern)
`ttnn/cpp/ttnn/operations/eltwise/binary/device/`
Start here for: validate(), compute_output_specs(), select_program_factory(), hash().
The binary_device_operation.hpp shows the attribute struct + dispatch pattern.

## Program factory example
`ttnn/cpp/ttnn/operations/eltwise/binary/device/element_wise_multi_core_program_factory.cpp`
Shows: CB setup, kernel creation, per-core work distribution, runtime args.

## Multi-core work distribution
`tt_metal/programming_examples/eltwise_sf_binary_loop/`
Simpler than ttnn ops — good for understanding the host-side program setup loop.
