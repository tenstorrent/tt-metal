"""Exact control for PLE compact norm rows, including the failed512 boundary."""

import json
from pathlib import Path
import torch
import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt.checkpoint import CheckpointState
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import MeshOps, ParallelPLE, _hidden_parts
from models.autoports.qwen_qwen3_8_flash_next.tests.multichip_components import CHECKPOINT


def main():
    torch.set_num_threads(8)
    torch.manual_seed(2026)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    try:
        ops = MeshOps(mesh)
        obj = ParallelPLE.__new__(ParallelPLE)
        obj.ops = ops
        obj.mesh = mesh
        obj.eps = 1e-6
        obj.workspaces = {}
        obj.flat_norm_weights = {}
        obj._norm_host_parts = {}
        state = CheckpointState(CHECKPOINT, "model.language_model.layers.1.ple.")
        names = ("norm_key", "norm_query", "norm_conv")
        for name in names:
            parts = [p.reshape(1, 1, 4, 640) for p in _hidden_parts(1 + state[name + ".weight"].float())]
            weight = ops.shards(parts, ttnn.float32)
            setattr(obj, name, weight)
            obj._norm_host_parts[id(weight)] = parts
        report = []
        for n in (1, 31, 32, 33, 128, 511, 512, 513):
            obj.prepare(batch=1, physical_seq_len=n)
            cpu = torch.randn(1, 1, n, 10240).bfloat16()
            x = ops.shards(_hidden_parts(cpu))
            for name in names:
                weight = getattr(obj, name)
                original = ttnn.typecast(ttnn.reshape(x, (1, n, 4, 640)), ttnn.float32)
                old = ttnn.typecast(ops.norm(original, weight, obj.eps), ttnn.bfloat16)
                expected = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(old)]
                del original, old
                fast = obj.norm(ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG), weight, n)
                actual = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(fast)]
                equal = all(torch.equal(a, b) for a, b in zip(expected, actual))
                error = max((a.float() - b.float()).abs().max().item() for a, b in zip(expected, actual))
                row = dict(length=n, norm=name, exact=equal, max_abs_error=error)
                print(row, flush=True)
                report.append(row)
                assert equal, row
                del fast
            del x
        Path("models/autoports/qwen_qwen3_8_flash_next/doc/full_model/ple_norm_control.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
