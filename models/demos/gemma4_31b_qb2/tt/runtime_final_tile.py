"""Setup-owned final-tile selectors; one model and ordered CQ0 caller only.

Native tensor slice reads one BF16 data tile from each control buffer. Allocate
512 uint32 words (2048 bytes), although only the first four words are meaningful.
This removes scalar tile coordinates only; residual row selection stays bounded.
No program preparation or trace-retirement policy is changed here.
"""
import torch
import ttnn


class RuntimeFinalTile:
    def __init__(self, mesh):
        self._assert_queue()
        self.mesh = mesh
        self.starts = self._allocate()
        self.ends = self._allocate()
        self.initial_buffer_ids = self.buffer_ids()

    @staticmethod
    def _assert_queue():
        if int(ttnn.get_current_command_queue_id_for_thread()) != 0:
            raise ValueError("Final-tile selectors require one ordered CQ0 caller")

    def _allocate(self):
        return ttnn.from_torch(
            torch.zeros(512, dtype=torch.int32), device=self.mesh,
            dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )

    def buffer_ids(self):
        return {"starts": self.starts.buffer_unique_id(), "ends": self.ends.buffer_unique_id()}

    @staticmethod
    def _refresh(destination, words):
        values = torch.zeros(512, dtype=torch.int32)
        values[:4] = torch.tensor(words, dtype=torch.int32)
        source = ttnn.from_torch(values, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(source, destination, cq_id=0)

    def select(self, hidden, tile_start):
        self._assert_queue()
        logical, padded = tuple(hidden.shape), tuple(hidden.padded_shape)
        if (len(logical) != 4 or logical[:2] != (1, 1)
                or logical[-1] not in (1344, 5376) or padded[:2] != (1, 1)
                or padded[-1] != logical[-1] or padded[2] % 32
                or hidden.dtype != ttnn.bfloat16 or hidden.layout != ttnn.TILE_LAYOUT
                or hidden.memory_config() != ttnn.DRAM_MEMORY_CONFIG):
            raise ValueError("Expected original interleaved BF16 TP4 hidden stream")
        if type(tile_start) is not int or tile_start < 0 or tile_start % 32 or tile_start >= logical[2]:
            raise ValueError("Final tile must begin on an aligned real hidden row")
        if tile_start + 32 > padded[2]:
            raise ValueError("Final tile exceeds physical padded height")
        # Expose existing tile padding as an alias; never change its geometry.
        if logical != padded:
            hidden = ttnn.reshape(hidden, ttnn.Shape(padded), ttnn.Shape(padded))
        self._refresh(self.starts, (0, 0, tile_start, 0))
        self._refresh(self.ends, (1, 1, tile_start + 32, logical[-1]))
        return ttnn.slice(hidden, self.starts, self.ends, slice_dim=2,
                          num_devices=padded[2] // 32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
