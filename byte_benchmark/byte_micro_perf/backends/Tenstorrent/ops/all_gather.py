import sys
import pathlib
import atexit

sys.path.insert(0, str(pathlib.Path(__file__).absolute().parents[3]))

# Clean up ttnn's namespace pollution before importing from core
from backends.Tenstorrent import _cleanup_ttnn_namespace  # noqa: F401

import torch
from core.utils import OpTensorInfo
from core.op import BasicOp

# Override test_mode to "mesh_single" for Tenstorrent native CCL
# ttnn CCL uses single-process MeshDevice instead of multi-process gloo
# mesh_single mode: only 1 subprocess, manages all devices via MeshDevice
TEST_MODE = "mesh_single"

OP_MAPPING = {}


def _cleanup_mesh_device():
    """Cleanup handler for MeshDevice - suppress teardown errors."""
    try:
        if TenstorrentNativeAllGatherOp._mesh_device is not None:
            import ttnn

            ttnn.close_mesh_device(TenstorrentNativeAllGatherOp._mesh_device)
            TenstorrentNativeAllGatherOp._mesh_device = None
    except Exception:
        pass


class TenstorrentNativeAllGatherOp(BasicOp):
    """
    Tenstorrent native all_gather operation using ttnn CCL with MeshDevice.

    Matches NVIDIA GPU benchmark pattern:
    - Tensors are pre-created on device (not CPU)
    - Run function only measures device-to-device communication
    - No CPU<->Device transfer in the measured path
    """

    # Class-level mesh device cache
    _mesh_device = None
    _mesh_device_count = None

    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    @classmethod
    def _get_mesh_device(cls, device_count=4):
        """Get or create cached mesh device."""
        if cls._mesh_device is None or cls._mesh_device_count != device_count:
            import ttnn

            ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
            mesh_shape = ttnn.MeshShape(1, device_count)
            cls._mesh_device = ttnn.open_mesh_device(mesh_shape)
            cls.device = cls._mesh_device
            cls._mesh_device_count = device_count
            print(f"✓ Opened ttnn MeshDevice with shape (1, {device_count})")

        return cls._mesh_device

    def prepare(self):
        import ttnn

        self._ttnn = ttnn

        self.arg_type = self.args_dict["arg_type"]
        if self.arg_type not in ["default", "llm"]:
            raise NotImplementedError

        self.dtype = self.args_dict["dtype"]
        # ttnn supports bfloat16 and float32 for CCL operations
        # Skip unsupported dtypes (like float16, int8 which ttnn doesn't have)
        if self.dtype not in ["bfloat16", "float32", "bfloat8_b"]:
            raise NotImplementedError(f"dtype={self.dtype} not supported by ttnn CCL")

        self.torch_dtype = torch.float32 if self.dtype == "float32" else torch.bfloat16
        if self.dtype == "bfloat16":
            self.dtype_size = 2
            self.ttnn_dtype = ttnn.bfloat16
        elif self.dtype == "float32":
            self.ttnn_dtype = ttnn.float32
            self.dtype_size = 4
        elif self.dtype == "bfloat8_b":
            self.dtype_size = 1
            self.ttnn_dtype = ttnn.bfloat8_b

        # Get actual device count
        try:
            device_ids = ttnn.get_device_ids()
            actual_device_count = len(device_ids)
        except:
            actual_device_count = 4

        # Get requested world_size from config, default to actual device count
        requested_world_size = self.args_dict.get("world_size", actual_device_count)

        # Skip if requested world_size exceeds actual devices
        if requested_world_size > actual_device_count:
            raise NotImplementedError(
                f"Requested world_size={requested_world_size} exceeds available devices={actual_device_count}"
            )

        # Use actual device count for operations
        self.world_size = actual_device_count
        # Update args_dict to reflect actual world_size used
        self.args_dict["world_size"] = actual_device_count

        if self.arg_type == "default":
            self.batch_size = self.args_dict["batch_size"]
            self.dim_size = self.args_dict["dim_size"]
        elif self.arg_type == "llm":
            self.batch_size = self.args_dict["num_tokens"]
            self.dim_size = self.args_dict["hidden_size"]

        # Calculate tileable dimensions
        input_elements = self.batch_size * self.dim_size // self.world_size
        output_elements = self.batch_size * self.dim_size

        # Tile-friendly shapes (32x32 tiles)
        input_height = 32
        input_width = max(32, ((input_elements + 31) // 32) * 32 // input_height)
        if input_width * input_height < input_elements:
            input_height = ((input_elements + input_width - 1) // input_width + 31) // 32 * 32

        self.input_ttnn_shape = [1, 1, input_height, input_width]
        self.input_total_elements = input_height * input_width

        output_height = 32
        output_width = max(32, ((output_elements + 31) // 32) * 32 // output_height)
        if output_width * output_height < output_elements:
            output_height = ((output_elements + output_width - 1) // output_width + 31) // 32 * 32

        self.output_ttnn_shape = [1, 1, output_height, output_width]
        self.output_total_elements = output_height * output_width

        # Input/output tensor info - matching NVIDIA pattern
        self.input_tensor_info = {
            "src": OpTensorInfo(
                shape=[self.input_total_elements], dtype=self.torch_dtype, device="ttnn_device", creator=torch.randn
            )
        }
        self.output_tensor_info = {
            "dst": OpTensorInfo(
                shape=[self.output_total_elements], dtype=self.torch_dtype, device="ttnn_device", creator=torch.randn
            )
        }

        # Size calculations
        self.input_tensor_size = self.input_total_elements * self.dtype_size
        self.output_tensor_size = self.output_total_elements * self.dtype_size
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self.algo_size = self.output_tensor_size
        self.bus_size = (self.world_size - 1) * self.algo_size / self.world_size

        self.calc_flops = 0

        self.mesh_device = self._get_mesh_device(self.world_size)

        self._create_tensors_func = self._create_device_tensors
        self._run_func = self.all_gather_run

    def _create_device_tensors(self, instance_num):
        """Create tensors that are already on device."""
        import ttnn

        all_tensor_list = []
        for _ in range(instance_num):
            # Create input tensor and shard across devices
            torch_tensor = torch.randn(self.input_ttnn_shape, dtype=self.torch_dtype)

            ttnn_tensor = ttnn.from_torch(
                torch_tensor,
                dtype=self.ttnn_dtype,
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=-1),
            )

            tensor_mapping = {"src": ttnn_tensor}
            all_tensor_list.append(tensor_mapping)

        return all_tensor_list

    def all_gather_run(self, tensor_mapping):
        """
        Execute ttnn native all_gather.

        This ONLY measures device-to-device communication.
        """
        import ttnn

        src = tensor_mapping["src"]

        # Perform native ttnn all_gather - THIS IS THE ONLY MEASURED OPERATION
        output = ttnn.all_gather(src, dim=0)

        return output

    def core_run(self, tensor_mapping):
        """Core run function called by benchmark framework."""
        return self.all_gather_run(tensor_mapping)


OP_MAPPING["torch"] = TenstorrentNativeAllGatherOp

# Register cleanup handler
atexit.register(_cleanup_mesh_device)
