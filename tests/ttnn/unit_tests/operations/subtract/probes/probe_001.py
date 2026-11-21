import ttnn
E = ttnn.experimental
c = ttnn.ComputeConfigDescriptor()
print("CCD fields", [a for a in dir(c) if not a.startswith("_")])
print("CCD defaults", c.math_fidelity, c.fp32_dest_acc_en, c.math_approx_mode, c.dst_full_sync_en)
h = E.ComputeHardwareConfig()
print("HW defaults", h.fpu_math_fidelity, h.sfpu_precision_mode, h.enable_32_bit_dest, h.double_buffer_dest)
print("Precision", [a for a in dir(E.Precision) if not a.startswith("_")] if hasattr(E, "Precision") else "no E.Precision", hasattr(ttnn, "Precision"))
