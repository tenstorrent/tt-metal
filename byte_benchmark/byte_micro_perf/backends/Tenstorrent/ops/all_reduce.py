import sys
import pathlib
import atexit

sys.path.insert(0, str(pathlib.Path(__file__).absolute().parents[3]))

# Clean up ttnn's namespace pollution before importing from core
from backends.Tenstorrent import _cleanup_ttnn_namespace  # noqa: F401

import torch
from core.utils import OpTensorInfo
from core.op import BasicOp


def _cleanup_mesh_device():
    """
    Cleanup handler for MeshDevice.

    This is registered with atexit to ensure proper device cleanup.
    We suppress errors since the ttnn teardown can fail due to known
    driver issues (see: https://github.com/tenstorrent/tt-metal/issues/30194)
    """
    try:
        if TenstorrentNativeAllReduceOp._mesh_device is not None:
            import ttnn

            print("Closing MeshDevice...")
            ttnn.close_mesh_device(TenstorrentNativeAllReduceOp._mesh_device)
            TenstorrentNativeAllReduceOp._mesh_device = None
            TenstorrentNativeAllReduceOp._mesh_device_count = None
    except Exception:
        # Suppress teardown errors - known issue with tt-metal driver cleanup
        pass


# Override test_mode to "mesh_single" for Tenstorrent native CCL
# ttnn CCL uses single-process MeshDevice instead of multi-process gloo
# mesh_single mode: only 1 subprocess, manages all devices via MeshDevice
TEST_MODE = "mesh_single"

OP_MAPPING = {}


class TenstorrentNativeAllReduceOp(BasicOp):
    """
    Tenstorrent native all_reduce operation using ttnn CCL with MeshDevice.

    Matches NVIDIA GPU benchmark pattern:
    - Tensors are pre-created on device (not CPU)
    - Run function only measures device-to-device communication
    - No CPU<->Device transfer in the measured path
    """

    # Class-level mesh device cache to avoid repeated initialization
    _mesh_device = None
    _mesh_device_count = None

    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    @classmethod
    def _get_mesh_device(cls, device_count=4):
        """Get or create cached mesh device."""
        if cls._mesh_device is None or cls._mesh_device_count != device_count:
            import ttnn

            # Set fabric configuration
            ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)

            # Open mesh device
            mesh_shape = ttnn.MeshShape(1, device_count)
            cls._mesh_device = ttnn.open_mesh_device(mesh_shape)
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
        # Skip unsupported dtypes (like float16 which ttnn doesn't have)
        if self.dtype not in ["bfloat16", "float32"]:
            raise NotImplementedError(f"dtype={self.dtype} not supported by ttnn CCL, only bfloat16/float32")
        self.torch_dtype = getattr(torch, self.dtype)

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

        # For ttnn, we need tensors that can be tiled (multiples of 32)
        total_elements = self.batch_size * self.dim_size

        # Calculate tileable dimensions (ttnn uses 32x32 tiles)
        height = 32
        width = max(32, ((total_elements + 31) // 32) * 32 // height)
        if width * height < total_elements:
            height = ((total_elements + width - 1) // width + 31) // 32 * 32

        self.ttnn_shape = [1, 1, height, width]
        self.total_elements = height * width

        # Input/output tensor info - matching NVIDIA pattern
        # Note: We use a special "ttnn_device" marker to indicate on-device tensors
        self.input_tensor_info = {
            "src": OpTensorInfo(
                shape=[self.total_elements],
                dtype=self.torch_dtype,
                device="ttnn_device",  # Special marker for ttnn device tensors
                creator=torch.randn,
            )
        }
        self.output_tensor_info = {}  # In-place operation

        # Size calculations for performance metrics
        self.input_tensor_size = self.total_elements * torch.tensor([], dtype=self.torch_dtype).element_size()
        self.output_tensor_size = 0  # In-place operation
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.input_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self.algo_size = self.input_tensor_size
        self.bus_size = 2 * (self.world_size - 1) * self.algo_size / self.world_size

        self.calc_flops = self.batch_size * self.dim_size

        # Get mesh device
        self.mesh_device = self._get_mesh_device(self.world_size)

        self._create_tensors_func = self._create_device_tensors
        self._run_func = self.all_reduce_run

    def _create_device_tensors(self, instance_num):
        """
        Create tensors that are already on device.

        This matches NVIDIA's pattern where _create_tensors_func returns
        tensors that are already on the GPU, not on CPU.
        """
        import ttnn

        # Map torch dtype to ttnn dtype
        ttnn_dtype = ttnn.bfloat16 if self.torch_dtype == torch.bfloat16 else ttnn.float32

        all_tensor_list = []
        for _ in range(instance_num):
            # Create new random data
            torch_tensor = torch.randn(self.ttnn_shape, dtype=self.torch_dtype)

            # Create ttnn tensor on device (this is setup, not measured)
            ttnn_tensor = ttnn.from_torch(
                torch_tensor,
                dtype=ttnn_dtype,
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

            # Store the ttnn tensor directly - no CPU tensor involved in run
            tensor_mapping = {"src": ttnn_tensor}
            all_tensor_list.append(tensor_mapping)

        return all_tensor_list

    def all_reduce_run(self, tensor_mapping):
        """
        Execute ttnn native all_reduce.

        This ONLY measures device-to-device communication.
        No CPU<->Device transfer happens here - matching NVIDIA's pattern.
        """
        import ttnn

        # Get the ttnn tensor that's already on device
        src = tensor_mapping["src"]

        # Perform native ttnn all_reduce - THIS IS THE ONLY MEASURED OPERATION
        output = ttnn.all_reduce(src)

        return output

    def core_run(self, tensor_mapping):
        """Core run function called by benchmark framework."""
        return self.all_reduce_run(tensor_mapping)


OP_MAPPING["torch"] = TenstorrentNativeAllReduceOp

# Register cleanup handler
atexit.register(_cleanup_mesh_device)
