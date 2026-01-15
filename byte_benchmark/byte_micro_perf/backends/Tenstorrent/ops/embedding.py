import sys
import pathlib
import atexit
import random

sys.path.insert(0, str(pathlib.Path(__file__).absolute().parents[3]))

from backends.Tenstorrent import _cleanup_ttnn_namespace  # noqa: F401

import torch
from core.utils import OpTensorInfo, calc_tensor_size
from core.op import BasicOp

OP_MAPPING = {}


def _cleanup_device():
    """Cleanup handler for device."""
    try:
        if TenstorrentEmbeddingOp._device is not None:
            import ttnn

            ttnn.close_device(TenstorrentEmbeddingOp._device)
            TenstorrentEmbeddingOp._device = None
    except Exception:
        pass


class TenstorrentEmbeddingOp(BasicOp):
    """Tenstorrent native embedding operation using ttnn."""

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
        if self.arg_type not in ["default"]:
            raise NotImplementedError

        self.dtype = self.args_dict["dtype"]
        if self.dtype not in ["bfloat16", "float32"]:
            raise NotImplementedError(f"dtype={self.dtype} not supported by ttnn")

        self.torch_dtype = getattr(torch, self.dtype)
        self.ttnn_dtype = ttnn.bfloat16 if self.dtype == "bfloat16" else ttnn.float32

        self.src_batch_size = self.args_dict["src_batch_size"]
        self.dst_batch_size = self.args_dict["dst_batch_size"]
        self.dim_size = self.args_dict["dim_size"]

        # Pad to tile-aligned (32x32)
        self.height = ((self.src_batch_size + 31) // 32) * 32
        self.width = ((self.dim_size + 31) // 32) * 32
        self.dst_height = ((self.dst_batch_size + 31) // 32) * 32

        self.input_tensor_info = {
            "src": OpTensorInfo(
                shape=[self.height, self.width], dtype=self.torch_dtype, device="ttnn_device", creator=torch.randn
            ),
            "index": OpTensorInfo(shape=[self.dst_batch_size], dtype=torch.int64, device="ttnn_device"),
        }
        self.output_tensor_info = {
            "dst": OpTensorInfo(
                shape=[self.dst_height, self.width], dtype=self.torch_dtype, device="ttnn_device", creator=torch.zeros
            )
        }

        # Calculate sizes based on original (non-padded) dimensions for accurate metrics
        dtype_size = torch.tensor([], dtype=self.torch_dtype).element_size()
        self.src_tensor_size = self.src_batch_size * self.dim_size * dtype_size
        self.index_tensor_size = self.dst_batch_size * 8  # int64 = 8 bytes
        self.dst_tensor_size = self.dst_batch_size * self.dim_size * dtype_size

        self.input_tensor_size = self.src_tensor_size + self.index_tensor_size
        self.output_tensor_size = self.dst_tensor_size
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.dst_tensor_size + self.index_tensor_size
        self.write_bytes = self.dst_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self.calc_flops = self.dst_batch_size * self.dim_size

        self.device = self._get_device()
        self._create_tensors_func = self._create_device_tensors
        self._run_func = self.embedding_run

    def _create_device_tensors(self, instance_num):
        import ttnn

        all_tensor_list = []
        for _ in range(instance_num):
            torch_src = torch.randn(1, 1, self.height, self.width, dtype=self.torch_dtype)

            # Create random indices
            random_index = []
            for value in range(self.dst_batch_size):
                random_index.append(value % self.src_batch_size)
            random.shuffle(random_index)
            torch_index = torch.tensor(random_index, dtype=torch.int32)

            ttnn_src = ttnn.from_torch(torch_src, dtype=self.ttnn_dtype, device=self.device, layout=ttnn.TILE_LAYOUT)
            # For embedding indices, we use uint32 and ROW_MAJOR layout
            ttnn_index = ttnn.from_torch(
                torch_index.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                dtype=ttnn.uint32,
                device=self.device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

            all_tensor_list.append({"src": ttnn_src, "index": ttnn_index})
        return all_tensor_list

    def embedding_run(self, tensor_mapping):
        import ttnn

        src = tensor_mapping["src"]
        index = tensor_mapping["index"]
        # Use ttnn.embedding for lookup
        dst = ttnn.embedding(index, src)
        ttnn.synchronize_device(self.device)
        return dst

    def core_run(self, tensor_mapping):
        return self.embedding_run(tensor_mapping)


OP_MAPPING["torch"] = TenstorrentEmbeddingOp
atexit.register(_cleanup_device)
