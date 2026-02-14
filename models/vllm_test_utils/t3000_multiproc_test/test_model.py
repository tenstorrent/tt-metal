# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path
from types import ModuleType

import torch
from loguru import logger

# Create a namespace package for 'tests' pointing to local directory
# This ensures 'tests.*' imports resolve to the local tests directory instead of vllm's
_project_root = Path(__file__).resolve().parent.parent.parent.parent
_tests_dir = _project_root / "tests"
if _tests_dir.exists():
    _tests_module = ModuleType("tests")
    _tests_module.__path__ = [str(_tests_dir)]
    sys.modules["tests"] = _tests_module

import ttnn
from tests.nightly.t3000.ccl.test_minimal_reduce_scatter_async import run_reduce_scatter_impl


class DummyT3000MultiProcessModel:
    """
    Dummy model class for testing simulated multihost on T3000 with 2x4 MeshDevice.
    This model runs a reduce scatter async test instead of actual inference.
    """

    def __init__(self, mesh_device, max_batch_size, vocab_size):
        self.mesh_device = mesh_device
        self.submesh_device = mesh_device.create_submesh(ttnn.MeshShape((1, 4)))
        self.max_batch_size = max_batch_size
        self.vocab_size = vocab_size

    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, **kwargs):
        vocab_size = hf_config.vocab_size
        return cls(mesh_device, max_batch_size, vocab_size)

    def prefill_forward(self, *args, **kwargs):
        logger.info("Dummy prefill: running reduce scatter async test (for 2x4 MeshDevice)")
        run_reduce_scatter_impl(
            self.submesh_device,
            self.submesh_device.get_num_devices(),
            rs_input_shape=[1, 1, 8, 7168],
            dim=3,
            num_links=1,
            rs_input_dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mem_config_input=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            mem_config_rs=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            rs_topology=ttnn.Topology.Linear,
            enable_trace=False,
            num_iters=3,
            ones_tensor=False,
            use_barrier=True,
            use_persistent_buffers=False,
            cluster_axis=1,
        )
        logger.info("Minimal reduce scatter async test completed")
        tokens = kwargs.get("tokens")
        return torch.zeros(tokens.shape[0], 1, self.vocab_size)

    def decode_forward(self, *args, **kwargs):
        # Run nothing for decode forward in this dummy model
        return torch.zeros(self.max_batch_size, 1, self.vocab_size)

    def allocate_kv_cache(self, *args, **kwargs):
        return None

    def warmup_model_prefill(self, *args, **kwargs):
        pass

    def warmup_model_decode(self, *args, **kwargs):
        pass
