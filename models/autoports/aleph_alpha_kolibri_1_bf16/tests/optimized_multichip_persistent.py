# SPDX-License-Identifier: Apache-2.0
"""Persistent RS/AG family, retaining fractured residual through consuming norms."""

import argparse

import ttnn

from . import multichip_checks
from .optimized_multichip_candidates import Candidate


class PersistentCandidate(Candidate):
    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        self = super().from_state_dict(state_dict, **kwargs)
        self.buffers = {}
        return self

    def _buffer(self, key, shape, dtype, mem):
        if key not in self.buffers:
            self.buffers[key] = ttnn.empty(
                shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=mem
            )
        return self.buffers[key]

    def _gather(self, x):
        if x.shape[-2] > 32:
            return super()._gather(x)
        shape = tuple(x.shape)
        out_shape = (*shape[:-1], shape[-1] * 4)
        key = ("ag", shape, str(x.dtype))
        output = self._buffer(key, out_shape, x.dtype, x.memory_config())
        return ttnn.experimental.all_gather_async(
            x,
            dim=3,
            cluster_axis=1,
            mesh_device=self.device,
            persistent_output_tensor=output,
            topology=self.ccl.topology,
            multi_device_global_semaphore=self.ccl.get_ag_ping_pong_semaphore(),
            barrier_semaphore=self.ccl.get_barrier_semaphore(),
            num_links=self.ccl.num_links,
            memory_config=x.memory_config(),
        )

    def _allreduce(self, x):
        if x.shape[-2] != 1 or x.shape[-1] == 640:
            return super()._allreduce(x)
        x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        if x.dtype != getattr(ttnn, self.policy.ccl_dtype):
            x = ttnn.typecast(x, getattr(ttnn, self.policy.ccl_dtype))
        shape = tuple(x.shape)
        # Linear RS stages two directions in an input-shaped, doubled leading dimension.
        intermediate = self._buffer(("rs_inter", shape, str(x.dtype)), (2, *shape[1:]), x.dtype, x.memory_config())
        output = self._buffer(("rs_out", shape, str(x.dtype)), (*shape[:-1], 640), x.dtype, x.memory_config())
        result = ttnn.experimental.reduce_scatter_minimal_async(
            x,
            persistent_output_buffers=[intermediate, output],
            dim=3,
            cluster_axis=1,
            topology=self.ccl.topology,
            multi_device_global_semaphore=self.ccl.get_rs_ping_pong_semaphore(),
            barrier_semaphore=self.ccl.get_barrier_semaphore(),
            num_links=self.ccl.num_links,
            memory_config=x.memory_config(),
        )
        if self.policy.residual_layout != "sharded":
            result = self._gather(result)
        return ttnn.typecast(result, ttnn.bfloat16) if result.dtype != ttnn.bfloat16 else result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--repetitions", type=int, default=100)
    a = p.parse_args()
    a.baseline = False
    multichip_checks.MultichipDecoder = PersistentCandidate
    multichip_checks.run(a)
