import sys
import pathlib
import atexit

sys.path.insert(0, str(pathlib.Path(__file__).absolute().parents[3]))

from backends.Tenstorrent import _cleanup_ttnn_namespace  # noqa: F401

import torch
from core.utils import OpTensorInfo, calc_tensor_size
from core.op import BasicOp

OP_MAPPING = {}


def _cleanup_device():
    """Cleanup handler for device."""
    try:
        if TenstorrentLayerNormOp._device is not None:
            import ttnn

            ttnn.close_device(TenstorrentLayerNormOp._device)
            TenstorrentLayerNormOp._device = None
    except Exception:
        pass


class TenstorrentLayerNormOp(BasicOp):
    """Tenstorrent native layer_norm operation using ttnn."""

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

        self.arg_type = self.args_dict["arg_type"]
        if self.arg_type not in ["default", "llm"]:
            raise NotImplementedError

        self.dtype = self.args_dict["dtype"]
        if self.dtype not in ["bfloat16", "float32"]:
            raise NotImplementedError(f"dtype={self.dtype} not supported by ttnn")

        self.torch_dtype = getattr(torch, self.dtype)
        self.ttnn_dtype = ttnn.bfloat16 if self.dtype == "bfloat16" else ttnn.float32

        if self.arg_type == "default":
            self.batch_size = self.args_dict["batch_size"]
            self.dim_size = self.args_dict["dim_size"]
        elif self.arg_type == "llm":
            self.batch_size = self.args_dict["num_tokens"]
            self.dim_size = self.args_dict["hidden_size"]

        # Pad to tile-aligned (32x32)
        self.height = ((self.batch_size + 31) // 32) * 32
        self.width = ((self.dim_size + 31) // 32) * 32

        self.input_tensor_info = {
            "src": OpTensorInfo(
                shape=[self.height, self.width], dtype=self.torch_dtype, device="ttnn_device", creator=torch.randn
            ),
            "weight": OpTensorInfo(
                shape=[1, self.width], dtype=self.torch_dtype, device="ttnn_device", creator=torch.ones
            ),
            "bias": OpTensorInfo(
                shape=[1, self.width], dtype=self.torch_dtype, device="ttnn_device", creator=torch.zeros
            ),
        }
        self.output_tensor_info = {
            "dst": OpTensorInfo(
                shape=[self.height, self.width], dtype=self.torch_dtype, device="ttnn_device", creator=torch.zeros
            )
        }

        # Calculate sizes based on original (non-padded) dimensions for accurate metrics
        dtype_size = torch.tensor([], dtype=self.torch_dtype).element_size()
        self.input_tensor_size = (self.batch_size * self.dim_size + 2 * self.dim_size) * dtype_size
        self.output_tensor_size = self.batch_size * self.dim_size * dtype_size
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self.algo_size = 0
        self.bus_size = 0

        # flops for layer_norm: similar to rms_norm
        self.calc_flops = self.batch_size * (3 * self.dim_size + 4)

        self.device = self._get_device()
        self._create_tensors_func = self._create_device_tensors
        self._run_func = self.layer_norm_run

    def _create_device_tensors(self, instance_num):
        import ttnn

        all_tensor_list = []
        for _ in range(instance_num):
            torch_src = torch.randn(1, 1, self.height, self.width, dtype=self.torch_dtype)
            torch_weight = torch.ones(1, 1, 1, self.width, dtype=self.torch_dtype)
            torch_bias = torch.zeros(1, 1, 1, self.width, dtype=self.torch_dtype)

            ttnn_src = ttnn.from_torch(torch_src, dtype=self.ttnn_dtype, device=self.device, layout=ttnn.TILE_LAYOUT)
            ttnn_weight = ttnn.from_torch(
                torch_weight, dtype=self.ttnn_dtype, device=self.device, layout=ttnn.TILE_LAYOUT
            )
            ttnn_bias = ttnn.from_torch(torch_bias, dtype=self.ttnn_dtype, device=self.device, layout=ttnn.TILE_LAYOUT)

            all_tensor_list.append({"src": ttnn_src, "weight": ttnn_weight, "bias": ttnn_bias})
        return all_tensor_list

    def layer_norm_run(self, tensor_mapping):
        import ttnn

        src = tensor_mapping["src"]
        weight = tensor_mapping["weight"]
        bias = tensor_mapping["bias"]
        dst = ttnn.layer_norm(src, weight=weight, bias=bias)
        ttnn.synchronize_device(self.device)
        return dst

    def core_run(self, tensor_mapping):
        return self.layer_norm_run(tensor_mapping)


OP_MAPPING["torch"] = TenstorrentLayerNormOp
atexit.register(_cleanup_device)
