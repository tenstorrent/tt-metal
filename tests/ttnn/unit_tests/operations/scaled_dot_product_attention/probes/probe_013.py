import ttnn

config = ttnn.ComputeConfigDescriptor()
print("Default unpack_to_dest_mode:", type(config.unpack_to_dest_mode))
print("VectorUnpackToDestMode available:", hasattr(ttnn._ttnn.program_descriptor, "VectorUnpackToDestMode"))
# Try list assignment
try:
    config.unpack_to_dest_mode = [ttnn.UnpackToDestMode.Default] * 32
    print("List assignment OK")
except Exception as e:
    print("List assignment failed:", e)
# Try VectorUnpackToDestMode construction
try:
    v = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode([ttnn.UnpackToDestMode.Default] * 32)
    print("VectorUnpackToDestMode construction OK; len =", len(v))
    config.unpack_to_dest_mode = v
    print("Vector assignment OK")
except Exception as e:
    print("Vector assignment failed:", e)
