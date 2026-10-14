import torch, ttnn

device = ttnn.open_device(device_id=0)
try:
    for shp in [[64], [], [1, 1, 50, 50]]:
        spec = ttnn.TensorSpec(ttnn.Shape(shp), ttnn.bfloat16, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.DRAM)
        t = ttnn.allocate_tensor_on_device(spec, device)
        print("SHAPE", shp, "logical", t.shape, "padded", t.padded_shape)
    spec = ttnn.TensorSpec(
        ttnn.Shape([1, 1, 64, 128]), ttnn.bfloat16, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.DRAM
    )
    t = ttnn.allocate_tensor_on_device(spec, device)
    v = ttnn.experimental.view(t, ttnn.Shape([1, 1, 50, 50]), ttnn.Shape([1, 1, 64, 128]))
    print("VIEW", v.shape, v.padded_shape, v.buffer_address() == t.buffer_address())
    for lg, pd in [([64], [32, 64]), ([], [32, 32]), ([64], [2048])]:
        spec = ttnn.TensorSpec(ttnn.Shape(pd), ttnn.bfloat16, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.DRAM)
        t = ttnn.allocate_tensor_on_device(spec, device)
        try:
            v = ttnn.experimental.view(t, ttnn.Shape(lg), ttnn.Shape(pd))
            print(
                "VIEWR", lg, v.shape, v.padded_shape, ttnn.to_torch(v).shape, v.cpu().to_torch_with_padded_shape().shape
            )
        except Exception as e:
            print("VIEWR FAIL", lg, pd, str(e)[:300])
finally:
    ttnn.close_device(device)
