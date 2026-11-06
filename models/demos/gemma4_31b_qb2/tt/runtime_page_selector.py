"""Ordered-CQ0 raw tiled page selection using existing tensor-start slice.

Setup owns padded controls. No cache writes, quantization, or trace-policy change.
"""
import torch
import ttnn
from models.demos.gemma4_31b_qb2.tt.kv_precision import selected_kv_dtypes


class RuntimePageSelector:
    def __init__(self, mesh):
        self.mesh = mesh
        self.allowed_dtypes = frozenset(selected_kv_dtypes())
        self._assert_queue()
        self.starts = self._allocate()
        self.ends = self._allocate()

    @staticmethod
    def _assert_queue():
        if int(ttnn.get_current_command_queue_id_for_thread()) != 0:
            raise ValueError("Page selector requires one ordered CQ0 caller")

    def _allocate(self):
        # Native reader reads one data tile (max1088 BF8 bytes) from each control.
        return ttnn.from_torch(torch.zeros(512, dtype=torch.int32), device=self.mesh,
            dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh))

    def buffer_ids(self):
        return [self.starts.buffer_unique_id(), self.ends.buffer_unique_id()]

    @staticmethod
    def _refresh(destination, words):
        values = torch.zeros(512, dtype=torch.int32)
        values[:4] = torch.tensor(words, dtype=torch.int32)
        source = ttnn.from_torch(values, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(source, destination, cq_id=0)

    def select(self, cache, page):
        self._assert_queue()
        shape = tuple(cache.shape)
        if (len(shape) != 4 or shape[1:] != (4, 128, 256)
                or tuple(cache.padded_shape) != shape
                or cache.dtype not in self.allowed_dtypes or cache.layout != ttnn.TILE_LAYOUT
                or cache.memory_config() != ttnn.DRAM_MEMORY_CONFIG):
            raise ValueError("Expected selected interleaved sliding cache")
        if type(page) is not int or not 0 <= page < shape[0]:
            raise ValueError("Physical page index outside cache")
        self._refresh(self.starts, (page, 0, 0, 0))
        self._refresh(self.ends, (page + 1, 4, 128, 256))
        return ttnn.slice(cache, self.starts, self.ends, slice_dim=0,
                          num_devices=shape[0], memory_config=ttnn.DRAM_MEMORY_CONFIG)
