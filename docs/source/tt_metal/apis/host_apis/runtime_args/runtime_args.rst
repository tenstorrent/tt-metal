Runtime Arguments
==================

.. doxygenfunction:: SetRuntimeArgs(KernelHandle kernel_id, const std::variant<CoreCoord,CoreRange,CoreRangeSet> &logical_core, const std::vector<uint32_t> &runtime_args)

.. doxygenfunction:: SetRuntimeArgs(KernelHandle kernel, const std::vector< CoreCoord > & core_spec, const std::vector< std::vector<uint32_t> > &runtime_args)

.. doxygenfunction:: GetRuntimeArgs
