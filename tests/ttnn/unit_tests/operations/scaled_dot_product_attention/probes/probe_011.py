import ttnn

device = ttnn.GetDefaultDevice() if hasattr(ttnn, "GetDefaultDevice") else None
