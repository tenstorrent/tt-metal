"""Precision-locked head fidelity control with real reduced-model activations."""
import json
import statistics
import time
from pathlib import Path

import torch
from ttnn.types import BlackholeComputeKernelConfig

import ttnn

from ..tt.generator import Gemma4Generator


def main():
    torch.set_num_threads(8)
    root = Path("models/autoports/google_gemma_4_31b_it/doc/optimized_full_model")
    prompt = json.loads((root / "aime24_prompt.json").read_text())["tokens"]
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024)
    results = []
    try:
        gen = Gemma4Generator(mesh, layer_indices=[0, 5])
        gen.prefill.stage(torch.tensor(prompt), slot=0, start=0, size=192, valid=len(prompt))
        hidden = gen.prefill.last_row(gen.prefill.hidden(size=192, start=0, kv_cache=gen.kv_cache))
        ref = None
        for fidelity in (ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2):
            gen.model.kernel = BlackholeComputeKernelConfig(
                math_fidelity=fidelity, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
            )
            out = gen.model.terminal(hidden)
            actual = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float()
            if ref is None:
                ref = actual
            pcc = float(torch.corrcoef(torch.stack([ref.flatten(), actual.flatten()]))[0, 1])
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            out = gen.model.terminal(hidden)
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            samples = []
            for _ in range(3):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                start = time.perf_counter()
                for _ in range(128):
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                samples.append((time.perf_counter() - start) * 1e6 / 128)
            ttnn.release_trace(mesh, tid)
            results.append(
                dict(
                    fidelity=str(fidelity),
                    traced_host_us=samples,
                    median_us=statistics.median(samples),
                    pcc=pcc,
                    greedy_equal=bool(torch.equal(ref.argmax(-1), actual.argmax(-1))),
                )
            )
            (root / "head_fidelity.json").write_text(
                json.dumps(dict(scope="reduced real activation terminal only", results=results), indent=2) + "\n"
            )
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
