Runtime Arguments
==================

.. doxygenfunction:: SetRuntimeArgs(ProgramHandle handle, KernelHandle kernel_id, const std::variant<CoreCoord,CoreRange,CoreRangeSet> &logical_core, const std::vector<uint32_t> &runtime_args)

.. doxygenfunction:: SetRuntimeArgs(ProgramHandle handle, KernelHandle kernel, const std::vector< CoreCoord > & core_spec, const std::vector< std::vector<uint32_t> > &runtime_args)

.. doxygenfunction:: SetRuntimeArgs(Device* device, const std::shared_ptr<Kernel> kernel, const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec, std::shared_ptr<RuntimeArgs> runtime_args)

.. doxygenfunction:: SetRuntimeArgs(Device* device, const std::shared_ptr<Kernel> kernel, const std::vector< CoreCoord > & core_spec, const std::vector<std::shared_ptr<RuntimeArgs>> runtime_args)

.. doxygenfunction:: GetRuntimeArgs(ProgramHandle handle, KernelHandle kernel_id, const CoreCoord &logical_core)

.. doxygenfunction:: GetRuntimeArgs(ProgramHandle handle, KernelHandle kernel_id)

.. doxygenfunction:: SetCommonRuntimeArgs(ProgramHandle handle, KernelHandle kernel_id, const std::vector<uint32_t> &runtime_args)

.. doxygenfunction:: GetCommonRuntimeArgs(ProgramHandle handle, KernelHandle kernel_id)
