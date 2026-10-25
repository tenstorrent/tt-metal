# SPDX-License-Identifier: Apache-2.0
"""One fused AG+gate/SiLU, sharing its gathered output with the up projection."""
import hashlib
import json
from pathlib import Path
import sys
import ttnn
from . import run_prefill_projection_candidate as projection
from . import run_prefill_column_candidate as column
from . import run_multichip_decoder as harness

_IMPORTED_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class SharedGateCandidate(projection.ProjectionCandidate):
    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        model = super().from_state_dict(*args, **kwargs)
        view = model._prefill_impl
        assert list(view.gate.shape) == [5120,4352]
        assert view.policy['gate']['readers'] == 1
        # Exact N=4352 is divisible by eight cores, so no tail in fused AG.
        view.shared_gate_weight = ttnn.reshape(ttnn.to_memory_config(view.gate, ttnn.DRAM_MEMORY_CONFIG), [1,1,5120,4352])
        return model

    def _finish(self, residual, mixer):
        if not self.policy.get('_prefill_phase') or mixer.shape[1] == 1:
            return super()._finish(residual, mixer)
        x = ttnn.add(residual, mixer, memory_config=mixer.memory_config())
        h = self._norm(x, self.post_norm)
        shape = list(h.shape)
        assert shape == [1,4096,1280]
        h = ttnn.reshape(ttnn.typecast(h, ttnn.bfloat8_b), [1,1,4096,1280])
        pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8,8),
            allowed_worker_cores=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0,0),ttnn.CoreCoord(7,7))]),
            in0_block_w=self.policy['shared_gate_k'], per_core_M=16, per_core_N=17,
            out_block_h=16, out_block_w=17, out_subblock_h=8, out_subblock_w=1,
            transpose_mcast=False, fuse_batch=False,
            fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU))
        gathered, gate = ttnn.experimental.all_gather_matmul_async(
            h, self.shared_gate_weight, persistent_output_buffer=None, dim=3,
            multi_device_global_semaphore=self.ccl.get_and_cycle_ag_semaphore_handles(),
            barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=2, topology=ttnn.Topology.Ring, all_gather_core_grid_offset=(0,8),
            memory_config_ag=ttnn.DRAM_MEMORY_CONFIG, memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
            program_config=pc, compute_kernel_config=self.matmul_ckc, dtype=ttnn.bfloat16,
            chunks_per_sync=10, num_workers_per_link=1, num_buffers_per_channel=2)
        gathered = ttnn.reshape(gathered, [1,4096,5120])
        gate = ttnn.reshape(gate, [1,4096,4352])
        up = self._linear(gathered, self.up)
        inter = ttnn.multiply(gate, up)
        del h, gathered, gate, up
        mlp = self._linear(inter, self.down)
        return ttnn.add(x, mlp, memory_config=mlp.memory_config())


if __name__ == '__main__':
    column.OptimizedDecoder._linear = column.column_linear
    harness.MultichipDecoder = SharedGateCandidate
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    try:
        harness.main()
    finally:
        if output.exists():
            value = json.loads(output.read_text())
            for module in (projection, column, projection.padded, projection.interleaved):
                value['provenance']['tests/'+Path(module.__file__).name] = module._IMPORTED_SHA
            value['provenance']['tests/run_prefill_shared_gate_candidate.py'] = _IMPORTED_SHA
            value['candidate_adapter'] = 'One BF8 fused AG+gate/SiLU returns shared gathered input for separate up; native RS down and decode unchanged.'
            output.write_text(json.dumps(value, indent=2)+'\n')
