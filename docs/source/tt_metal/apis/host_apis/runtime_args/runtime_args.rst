Runtime Arguments
==================

.. doxygenfunction:: SetRuntimeArgs(const Program &program, KernelID kernel_id, const CoreCoord &logical_core, const std::vector<uint32_t> &runtime_args)

.. doxygenfunction:: SetRuntimeArgs(const Program &program, KernelID kernel_id, const CoreRange &core_range, const std::vector<uint32_t> &runtime_args)

.. doxygenfunction:: SetRuntimeArgs(const Program &program, KernelID kernel_id, const CoreRangeSet &core_range_set, const std::vector<uint32_t> &runtime_args)

.. doxygenfunction:: GetRuntimeArgs
