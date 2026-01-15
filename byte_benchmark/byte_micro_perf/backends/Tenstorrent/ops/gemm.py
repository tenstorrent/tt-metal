import sys
import pathlib
import atexit

sys.path.insert(0, str(pathlib.Path(__file__).absolute().parents[3]))

# Clean up ttnn's namespace pollution before importing from core
from backends.Tenstorrent import _cleanup_ttnn_namespace  # noqa: F401

import torch
from core.utils import OpTensorInfo
from core.op import BasicOp

OP_MAPPING = {}


def _cleanup_device():
    """Cleanup handler for device - suppress teardown errors."""
    try:
        if TenstorrentGemmOp._device is not None:
            import ttnn

            ttnn.close_device(TenstorrentGemmOp._device)
            TenstorrentGemmOp._device = None
    except Exception:
        pass


class TenstorrentGemmOp(BasicOp):
    """
    Tenstorrent native GEMM operation using ttnn.matmul.

    Performs matrix multiplication on Tenstorrent hardware:
    C = A @ B where A is [M, K] and B is [K, N]
    """

    # Class-level device cache
    _device = None

    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    @classmethod
    def _get_device(cls):
        """Get or create cached device."""
        if cls._device is None:
            import ttnn

            cls._device = ttnn.open_device(device_id=0)
            print(f"Opened ttnn device 0")
        return cls._device

    def prepare(self):
        import ttnn

        self._ttnn = ttnn

        self.arg_type = self.args_dict["arg_type"]
        if self.arg_type not in ["default", "llm"]:
            raise NotImplementedError

        self.dtype = self.args_dict["dtype"]
        # ttnn supports bfloat16 and float32 for matmul
        # Skip unsupported dtypes
        if self.dtype not in ["bfloat16", "float32"]:
            raise NotImplementedError(f"dtype={self.dtype} not supported by ttnn matmul, only bfloat16/float32")

        self.torch_dtype = getattr(torch, self.dtype)
        self.ttnn_dtype = ttnn.bfloat16 if self.dtype == "bfloat16" else ttnn.float32

        if self.arg_type == "default":
            self.M = self.args_dict["M"]
            self.K = self.args_dict["K"]
            self.N = self.args_dict["N"]
        elif self.arg_type == "llm":
            self.M = self.args_dict["num_tokens"]
            self.K = self.args_dict["hidden_size"]
            self.N = self.args_dict["new_hidden_size"]

        # ttnn requires tile-aligned dimensions (multiples of 32)
        self.M_padded = ((self.M + 31) // 32) * 32
        self.K_padded = ((self.K + 31) // 32) * 32
        self.N_padded = ((self.N + 31) // 32) * 32

        # Input/output tensor info
        self.input_tensor_info = {
            "a": OpTensorInfo(
                shape=[self.M_padded, self.K_padded], dtype=self.torch_dtype, device="ttnn_device", creator=torch.randn
            ),
            "b": OpTensorInfo(
                shape=[self.K_padded, self.N_padded], dtype=self.torch_dtype, device="ttnn_device", creator=torch.randn
            ),
        }
        self.output_tensor_info = {
            "c": OpTensorInfo(
                shape=[self.M_padded, self.N_padded], dtype=self.torch_dtype, device="ttnn_device", creator=torch.zeros
            )
        }

        # Size calculations for metrics (use original sizes for fair comparison)
        dtype_size = torch.tensor([], dtype=self.torch_dtype).element_size()
        self.input_tensor_size = (self.M * self.K + self.K * self.N) * dtype_size
        self.output_tensor_size = self.M * self.N * dtype_size
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self.calc_flops = self.M * self.N * self.K * 2

        self.device = self._get_device()

        self._create_tensors_func = self._create_device_tensors
        self._run_func = self.gemm_run

    def _create_device_tensors(self, instance_num):
        """Create tensors on device for benchmark."""
        import ttnn

        all_tensor_list = []
        for _ in range(instance_num):
            # Create torch tensors and convert to ttnn
            torch_a = torch.randn(1, 1, self.M_padded, self.K_padded, dtype=self.torch_dtype)
            torch_b = torch.randn(1, 1, self.K_padded, self.N_padded, dtype=self.torch_dtype)

            # Convert to ttnn tensors on device
            ttnn_a = ttnn.from_torch(torch_a, dtype=self.ttnn_dtype, device=self.device, layout=ttnn.TILE_LAYOUT)
            ttnn_b = ttnn.from_torch(torch_b, dtype=self.ttnn_dtype, device=self.device, layout=ttnn.TILE_LAYOUT)

            tensor_mapping = {"a": ttnn_a, "b": ttnn_b}
            all_tensor_list.append(tensor_mapping)

        return all_tensor_list

    def gemm_run(self, tensor_mapping):
        """
        Execute GEMM using ttnn.matmul on Tenstorrent hardware.
        """
        import ttnn

        a = tensor_mapping["a"]
        b = tensor_mapping["b"]

        # Perform matmul on device
        c = ttnn.matmul(a, b)

        # Synchronize to ensure operation completes before timing ends
        ttnn.synchronize_device(self.device)

        return c

    def core_run(self, tensor_mapping):
        """Core run function called by benchmark framework."""
        return self.gemm_run(tensor_mapping)


OP_MAPPING["torch"] = TenstorrentGemmOp

# Register cleanup handler
atexit.register(_cleanup_device)
