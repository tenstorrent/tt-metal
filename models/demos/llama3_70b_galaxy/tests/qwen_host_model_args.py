# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The real TtQwenModelArgs, built on the host without opening a mesh device, for host-only tests.

The model args read only the mesh shape and device count from the mesh device, so a stand-in object
with those two fields is enough. HF_MODEL must point at the Qwen3-32B checkpoint directory (config only;
dummy_weights=True loads no weights)."""

import functools
import os

import ttnn
from models.demos.llama3_70b_galaxy.tt.qwen_model_config import TtQwenModelArgs


class _HostOnlyGalaxyMesh:
    """Stand-in for an 8x4 (32-chip) Wormhole Galaxy mesh: exposes only what TtQwenModelArgs reads.
    The two grids feed max_grid_size and dram_weight_grid only (not the prefill padding or SDPA configs)."""

    shape = (8, 4)

    def get_num_devices(self):
        return 32

    def compute_with_storage_grid_size(self):
        return ttnn.CoreCoord(7, 10)  # Galaxy worker grid the Qwen configs assume (SDPA_PROGCFG uses (7, 10))

    def dram_grid_size(self):
        return ttnn.CoreCoord(12, 1)  # Wormhole: 12 DRAM banks

    def __getattr__(self, name):
        raise AttributeError(f"host-only mesh stand-in has no {name}; the test needs a real device")


@functools.lru_cache(maxsize=None)
def qwen_model_args(max_seq_len=40960):
    assert os.getenv("HF_MODEL"), "set HF_MODEL to the Qwen3-32B checkpoint directory (config is read from it)"
    return TtQwenModelArgs(_HostOnlyGalaxyMesh(), max_batch_size=1, max_seq_len=max_seq_len, dummy_weights=True)
