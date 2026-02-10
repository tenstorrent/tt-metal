import ttnn
import torch
import math


def load_tensor(file_path: str, layout, dtype, device, memory_config) -> ttnn.Tensor:
    loaded_tensor = ttnn.load_tensor(file_path)

    assert loaded_tensor.device() is None, "loaded tensor must be on host"

    if loaded_tensor.layout != layout:
        loaded_tensor = ttnn.to_layout(loaded_tensor, layout)
    if loaded_tensor.dtype != dtype:
        loaded_tensor = ttnn.to_dtype(loaded_tensor, dtype)
    if device is not None:
        loaded_tensor = ttnn.to_device(loaded_tensor, device, memory_config)

    return loaded_tensor


class DeviceGetter:
    _instance = None
    _mesh_shape = None
    l1_small_size = 1 << 15

    def __init__(self):
        raise RuntimeError("This is Singleton, invoke get_device() instead.")

    def __del__(self):
        if self._instance is not None:
            ttnn.close_mesh_device(self._instance)
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    @classmethod
    def get_device(cls, mesh_shape):
        if cls._instance == None:
            if (
                not isinstance(mesh_shape, (list, tuple))
                or len(mesh_shape) == 0
                or not all(isinstance(x, int) and x > 0 for x in mesh_shape)
            ):
                raise ValueError(
                    f"mesh_shape must be a non-empty list or tuple of positive integers, got {mesh_shape}"
                )
            cls._mesh_shape = mesh_shape

            if math.prod(mesh_shape) >= 2:
                ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
            cls._instance = ttnn.open_mesh_device(
                mesh_shape=ttnn.MeshShape(mesh_shape),
                l1_small_size=cls.l1_small_size,
            )
            print(f"Device: {cls._instance}")

        # Compare requested mesh_shape with _mesh_shape used to initialize the device
        if tuple(cls._mesh_shape) != tuple(mesh_shape):
            raise ValueError(
                f"Device already initialized with mesh_shape={cls._mesh_shape}, but got mesh_shape={mesh_shape}"
            )

        return cls._instance


def compute_pcc(x: torch.Tensor, y: torch.Tensor):
    x_flat, y_flat = x.flatten(), y.flatten()
    vx = x_flat - x_flat.mean()
    vy = y_flat - y_flat.mean()
    denom = vx.norm() * vy.norm()
    if denom == 0:
        return float("nan")
    return ((vx @ vy) / denom).item()


def test_lt():
    device = DeviceGetter.get_device((1, 1))

    # --- Load input as torch (bfloat16) ---
    ttnn_raw = load_tensor(
        "arg0.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    input_bf16 = ttnn.to_torch(ttnn_raw)  # torch bfloat16 [1,8400,4]

    # --- Prepare inputs in torch (all on CPU) ---
    input_f32 = input_bf16.to(torch.float32)                          # typecast
    const_f32 = torch.full([1, 8400, 4], 0.99000000953674316,
                           dtype=torch.float32)                        # full

    # --- CPU less-than ---
    cpu_result = (input_f32 < const_f32)

    # --- TTNN less-than (only op on device) ---
    ttnn_input = ttnn.from_torch(input_f32, dtype=ttnn.DataType.FLOAT32,
                                 layout=ttnn.Layout.TILE, device=device,
                                 memory_config=ttnn.MemoryConfig(
                                     ttnn.TensorMemoryLayout.INTERLEAVED,
                                     ttnn.BufferType.DRAM, None))
    ttnn_const = ttnn.from_torch(const_f32, dtype=ttnn.DataType.FLOAT32,
                                 layout=ttnn.Layout.TILE, device=device,
                                 memory_config=ttnn.MemoryConfig(
                                     ttnn.TensorMemoryLayout.INTERLEAVED,
                                     ttnn.BufferType.DRAM, None))
    ttnn_result = ttnn.lt(ttnn_input, ttnn_const,
                        #   dtype=ttnn.DataType.BFLOAT16,
                          memory_config=ttnn.MemoryConfig(
                              ttnn.TensorMemoryLayout.INTERLEAVED,
                              ttnn.BufferType.DRAM, None))
    tt_result = ttnn.to_torch(ttnn_result)  # torch bfloat16

    # --- Compare ---
    cpu_f64 = cpu_result.to(torch.float64)
    tt_f64 = tt_result.to(torch.float64)
    diff = (cpu_f64 - tt_f64).abs()

    print(f"CPU  dtype: {cpu_result.dtype}, shape: {cpu_result.shape}")
    print(f"TTNN dtype: {tt_result.dtype}, shape: {tt_result.shape}")
    print(f"torch.equal:    {torch.equal(cpu_result, tt_result)}")
    print(f"torch.allclose: {torch.allclose(cpu_f64, tt_f64)}")
    print(f"PCC:            {compute_pcc(cpu_f64, tt_f64)}")
    print(f"Max abs diff:   {diff.max().item()}")
    print(f"Mean abs diff:  {diff.mean().item()}")
    print(f"Mismatched:     {(diff > 0).sum().item()} / {diff.numel()}")
