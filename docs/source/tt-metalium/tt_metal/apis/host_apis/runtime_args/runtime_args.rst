Runtime Arguments
==================

.. doxygenfunction:: tt::tt_metal::SetRuntimeArgs(const Program &program, KernelHandle kernel_id, const std::variant<CoreCoord,CoreRange,CoreRangeSet> &logical_core, stl::Span<const uint32_t> runtime_args)

.. doxygenfunction:: tt::tt_metal::SetRuntimeArgs(const Program &program, KernelHandle kernel_id, const std::variant<CoreCoord,CoreRange,CoreRangeSet> &logical_core, std::initializer_list<uint32_t> runtime_args)

.. doxygenfunction:: tt::tt_metal::SetRuntimeArgs(const Program &program, KernelHandle kernel, const std::vector< CoreCoord > & core_spec, const std::vector< std::vector<uint32_t> > &runtime_args)

.. doxygenfunction:: tt::tt_metal::GetRuntimeArgs(const Program &program, KernelHandle kernel_id, const CoreCoord &logical_core)

.. doxygenfunction:: tt::tt_metal::GetRuntimeArgs(const Program &program, KernelHandle kernel_id)

.. doxygenfunction:: tt::tt_metal::SetCommonRuntimeArgs(const Program &program, KernelHandle kernel_id, stl::Span<const uint32_t> runtime_args)

.. doxygenfunction:: tt::tt_metal::SetCommonRuntimeArgs(const Program &program, KernelHandle kernel_id, std::initializer_list<uint32_t> runtime_args)

.. doxygenfunction:: tt::tt_metal::GetCommonRuntimeArgs(const Program &program, KernelHandle kernel_id)
