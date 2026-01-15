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
        if TenstorrentDivOp._device is not None:
            import ttnn

            ttnn.close_device(TenstorrentDivOp._device)
            TenstorrentDivOp._device = None
    except Exception:
        pass


class TenstorrentDivOp(BasicOp):
    """Tenstorrent native div operation using ttnn."""

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
        if self.dtype not in ["bfloat16", "float32"]:
            raise NotImplementedError(f"dtype={self.dtype} not supported by ttnn")

        self.torch_dtype = getattr(torch, self.dtype)
        self.ttnn_dtype = ttnn.bfloat16 if self.dtype == "bfloat16" else ttnn.float32

        self.batch_size = self.args_dict["batch_size"]
        self.dim_size = self.args_dict["dim_size"]

        # Pad to tile-aligned (32x32)
        self.height = ((self.batch_size + 31) // 32) * 32
        self.width = ((self.dim_size + 31) // 32) * 32

        self.input_tensor_info = {
            "a": OpTensorInfo(
                shape=[self.height, self.width], dtype=self.torch_dtype, device="ttnn_device", creator=torch.randn
            ),
            "b": OpTensorInfo(
                shape=[self.height, self.width], dtype=self.torch_dtype, device="ttnn_device", creator=torch.randn
            ),
        }
        self.output_tensor_info = {
            "c": OpTensorInfo(
                shape=[self.height, self.width], dtype=self.torch_dtype, device="ttnn_device", creator=torch.zeros
            )
        }

        dtype_size = torch.tensor([], dtype=self.torch_dtype).element_size()
        self.input_tensor_size = 2 * self.batch_size * self.dim_size * dtype_size
        self.output_tensor_size = self.batch_size * self.dim_size * dtype_size
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self.calc_flops = self.batch_size * self.dim_size

        self.device = self._get_device()
        self._create_tensors_func = self._create_device_tensors
        self._run_func = self.div_run

    def _create_device_tensors(self, instance_num):
        import ttnn

        all_tensor_list = []
        for _ in range(instance_num):
            torch_a = torch.randn(1, 1, self.height, self.width, dtype=self.torch_dtype)
            # Avoid division by zero
            torch_b = torch.randn(1, 1, self.height, self.width, dtype=self.torch_dtype).abs() + 0.1

            ttnn_a = ttnn.from_torch(torch_a, dtype=self.ttnn_dtype, device=self.device, layout=ttnn.TILE_LAYOUT)
            ttnn_b = ttnn.from_torch(torch_b, dtype=self.ttnn_dtype, device=self.device, layout=ttnn.TILE_LAYOUT)

            all_tensor_list.append({"a": ttnn_a, "b": ttnn_b})
        return all_tensor_list

    def div_run(self, tensor_mapping):
        import ttnn

        a, b = tensor_mapping["a"], tensor_mapping["b"]
        c = ttnn.div(a, b)
        ttnn.synchronize_device(self.device)
        return c

    def core_run(self, tensor_mapping):
        return self.div_run(tensor_mapping)


OP_MAPPING["torch"] = TenstorrentDivOp
atexit.register(_cleanup_device)
