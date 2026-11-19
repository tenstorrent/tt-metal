# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Isolated experiment: owned sampler outputs with barrier-preserving broadcast AG.

This is deliberately not imported by the production generator. The checked,
instance-local forward copy models one optional ownership hook in shared sampling;
it never patches ttnn.deallocate or the shared TTSampling class.
"""

import hashlib
import inspect
import textwrap
from types import MethodType

import torch

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tt.generator import SamplerCCL
from models.common.sampling.tt_sampling import TTSampling


class BroadcastPersistentSamplerCCL(SamplerCCL):
    def __init__(self, model, *, num_links=1):
        super().__init__(model)
        assert num_links in (1, 2)
        self.num_links = num_links
        self.frozen = False
        self.calls = 0
        self.buffers = {
            "SAMPLING_VALUES": model.upload(torch.zeros(1, 1, 32, 128), ttnn.bfloat16),
            "SAMPLING_INDICES": model.upload(torch.zeros(1, 1, 32, 128, dtype=torch.int32), ttnn.uint16),
        }
        self.addresses = {key: value.buffer_address() for key, value in self.buffers.items()}

    def owns_persistent_tensor(self, tensor):
        return any(
            tensor.buffer_address() == value.buffer_address()
            and tensor.dtype == value.dtype
            and tensor.memory_config() == value.memory_config()
            for value in self.buffers.values()
        )

    def freeze(self):
        self.frozen = True
        self.assert_live()

    def assert_live(self):
        for key, value in self.buffers.items():
            assert value.is_allocated(), f"Persistent sampler buffer was deallocated: {key}"
            assert value.buffer_address() == self.addresses[key], key

    def line_all_gather(self, tensor, *, dim, cluster_axis, memory_config, num_links, buffer_key=None):
        assert buffer_key in self.buffers, buffer_key
        shape = tuple(tensor.shape)
        mesh_shape = tuple(self.model.mesh.shape)
        # TTSampling._get_sampling_cluster_axis returns None for a 1x4 mesh.
        # Production SamplerCCL resolves that degenerate-2D default to axis 1.
        # Accept either equivalent spelling; reject other mesh/axis contracts.
        assert -len(shape) <= dim < len(shape), ("dim", dim, shape, cluster_axis, mesh_shape)
        normalized_dim = dim % len(shape)
        assert normalized_dim == 3 and cluster_axis in (None, 1) and mesh_shape == (1, 4), (
            "unsupported sampler gather",
            dim,
            normalized_dim,
            cluster_axis,
            mesh_shape,
            shape,
            buffer_key,
        )
        assert tuple(tensor.shape) == (1, 1, 32, 32), tuple(tensor.shape)
        assert tuple(tensor.padded_shape) == (1, 1, 32, 32)
        assert tensor.layout == ttnn.TILE_LAYOUT
        assert memory_config == ttnn.DRAM_MEMORY_CONFIG
        output = self.buffers[buffer_key]
        assert tensor.dtype == output.dtype, (buffer_key, tensor.dtype, output.dtype)
        self.assert_live()
        ccl = self.model.workspace.ccl
        self.calls += 1
        result = ttnn.experimental.all_gather_async(
            tensor,
            dim=3,
            cluster_axis=1,
            mesh_device=self.model.mesh,
            topology=ttnn.Topology.Linear,
            multi_device_global_semaphore=ccl.get_ag_ping_pong_semaphore(),
            barrier_semaphore=ccl.get_barrier_semaphore(),
            num_links=self.num_links,
            memory_config=memory_config,
            persistent_output_tensor=output,
            use_broadcast=True,
        )
        assert result.buffer_address() == output.buffer_address(), (
            "native AG did not return the supplied output",
            buffer_key,
            result.buffer_address(),
            output.buffer_address(),
        )
        return result


def install_persistent_sampler_probe(generator, *, num_links=1):
    sampler = generator.sampler.tt_sampling
    assert type(sampler) is TTSampling
    assert sampler.sampling_memory_config == ttnn.DRAM_MEMORY_CONFIG
    assert not sampler._force_argmax_sampling
    assert not sampler.pad_to_power_of_2
    source = textwrap.dedent(inspect.getsource(TTSampling.forward))
    anchor = "    ttnn.deallocate(topk_values_gathered_bf16_interleaved)"
    assert source.count(anchor) == 1, "Shared sampler ownership site changed; re-audit the private experiment"
    patched = source.replace(
        anchor,
        "    if not self.tt_ccl.owns_persistent_tensor(topk_values_gathered_bf16_interleaved):\n"
        "        ttnn.deallocate(topk_values_gathered_bf16_interleaved)",
    )
    namespace = dict(TTSampling.forward.__globals__)
    exec(compile(patched, "<isolated-sampler-persistent-forward>", "exec"), namespace)
    ccl = BroadcastPersistentSamplerCCL(generator.model, num_links=num_links)
    # SeedManager retains the same sampler object. Only this candidate's method
    # and cached collective binding change; baseline and shared class are intact.
    sampler.forward = MethodType(namespace["forward"], sampler)
    sampler.tt_ccl = ccl
    sampler._line_all_gather = ccl.line_all_gather
    sampler._line_all_gather_supports_buffer_key = True
    ccl.forward_source_sha256 = hashlib.sha256(source.encode()).hexdigest()
    ccl.patched_forward_sha256 = hashlib.sha256(patched.encode()).hexdigest()
    return ccl
