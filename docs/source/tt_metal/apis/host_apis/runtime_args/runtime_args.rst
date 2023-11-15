Runtime Arguments
==================

.. doxygenfunction:: SetRuntimeArgs(const Program &program, KernelID kernel_id, const std::variant<CoreCoord,CoreRange,CoreRangeSet> &logical_core, const std::vector<uint32_t> &runtime_args)

.. doxygenfunction:: SetRuntimeArgs(const Program &program, KernelID kernel, const std::vector< CoreCoord > & core_spec, const std::vector< std::vector<uint32_t> > &runtime_args)

.. doxygenfunction:: GetRuntimeArgs

.. doxygenfunction:: UpdateRuntimeArg
