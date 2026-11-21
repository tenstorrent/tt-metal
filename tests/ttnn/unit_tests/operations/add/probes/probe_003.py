import ttnn
E = ttnn.experimental
c = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)
print("CCD attrs", [a for a in dir(c) if not a.startswith("_")])
print("dst_full_sync_en", c.dst_full_sync_en)
print("Precision", hasattr(ttnn, "Precision"), [a for a in dir(E) if "recision" in a], [a for a in dir(ttnn) if "recision" in a])
hw = E.ComputeHardwareConfig(fpu_math_fidelity=c.math_fidelity, enable_32_bit_dest=True, double_buffer_dest=True)
print("hw", hw.fpu_math_fidelity, hw.enable_32_bit_dest, hw.double_buffer_dest, hw.sfpu_precision_mode)
