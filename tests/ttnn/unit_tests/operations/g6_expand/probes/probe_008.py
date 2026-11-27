import ttnn

print(
    hasattr(ttnn, "Mcast"),
    hasattr(ttnn, "McastDataReady"),
    hasattr(ttnn, "attach_absent_mcast"),
    hasattr(ttnn, "compute_program_descriptor_hash"),
    hasattr(ttnn, "is_tensor_storage_on_device"),
)
k = ttnn.KernelDescriptor()
k.common_runtime_args = [5]
print(list(k.common_runtime_args))
d = ttnn.ProgramDescriptor()
d.kernels = [k]
ks = d.kernels
ks[0].common_runtime_args = [7]
d.kernels = ks
print(list(d.kernels[0].common_runtime_args))
