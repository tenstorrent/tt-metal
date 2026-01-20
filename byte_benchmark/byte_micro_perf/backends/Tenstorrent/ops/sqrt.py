import sys
import pathlib
import atexit

sys.path.insert(0, str(pathlib.Path(__file__).absolute().parents[3]))

from backends.Tenstorrent import _cleanup_ttnn_namespace  # noqa: F401

import torch
from core.utils import OpTensorInfo
from core.op import BasicOp

OP_MAPPING = {}


def _cleanup_device():
    """Cleanup handler for device."""
    try:
        if TenstorrentSqrtOp._device is not None:
            import ttnn

            ttnn.close_device(TenstorrentSqrtOp._device)
            TenstorrentSqrtOp._device = None
    except Exception:
        pass


class TenstorrentSqrtOp(BasicOp):
    """Tenstorrent native sqrt operation using ttnn."""

    _device = None

    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    @classmethod
    def _get_device(cls):
        if cls._device is None:
            import ttnn

            cls._device = ttnn.open_device(device_id=0)
        return cls._device

    def prepare(self):
        import ttnn

        self._ttnn = ttnn

        self.dtype = self.args_dict["dtype"]
        if self.dtype not in ["bfloat16", "float32", "bfloat8_b"]:
            raise NotImplementedError(f"dtype={self.dtype} not supported by ttnn")

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

        self.batch_size = self.args_dict["batch_size"]
        self.dim_size = self.args_dict["dim_size"]

        # Pad to tile-aligned (32x32)
        self.height = ((self.batch_size + 31) // 32) * 32
        self.width = ((self.dim_size + 31) // 32) * 32

        self.input_tensor_info = {
            "src": OpTensorInfo(
                shape=[self.height, self.width], dtype=self.torch_dtype, device="ttnn_device", creator=torch.randn
            ),
        }
        self.output_tensor_info = {
            "dst": OpTensorInfo(
                shape=[self.height, self.width], dtype=self.torch_dtype, device="ttnn_device", creator=torch.zeros
            )
        }

        self.input_tensor_size = self.batch_size * self.dim_size * self.dtype_size
        self.output_tensor_size = self.batch_size * self.dim_size * self.dtype_size
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self.calc_flops = self.batch_size * self.dim_size

        self.device = self._get_device()
        self._create_tensors_func = self._create_device_tensors
        self._run_func = self.sqrt_run

    def _create_device_tensors(self, instance_num):
        import ttnn

        all_tensor_list = []
        for _ in range(instance_num):
            # Use positive values for sqrt
            torch_src = torch.randn(1, 1, self.height, self.width, dtype=self.torch_dtype).abs() + 0.1

            ttnn_src = ttnn.from_torch(torch_src, dtype=self.ttnn_dtype, device=self.device, layout=ttnn.TILE_LAYOUT)

            all_tensor_list.append({"src": ttnn_src})
        return all_tensor_list

    def sqrt_run(self, tensor_mapping):
        import ttnn

        src = tensor_mapping["src"]
        dst = ttnn.sqrt(src)
        return dst

    def core_run(self, tensor_mapping):
        return self.sqrt_run(tensor_mapping)


OP_MAPPING["torch"] = TenstorrentSqrtOp
atexit.register(_cleanup_device)
