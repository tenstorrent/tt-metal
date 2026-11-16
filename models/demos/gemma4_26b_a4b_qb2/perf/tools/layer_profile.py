"""Batch-1 decode per layer (real layers 0 sliding and 5 full): traced time, or a device profile of one call.

  python models/demos/gemma4_26b_a4b_qb2/perf/tools/layer_profile.py [--iters N] [--layers 0,5]
      prints LAYER_US layer=<i> kind=<type> us=<median of 5 x N back-to-back trace replays>
  python -m tracy -r -p --no-web-server -o <dir> models/demos/gemma4_26b_a4b_qb2/perf/tools/layer_profile.py --profile
      one warm untraced decode per layer between LAYER_<i>_START / _END signposts; feed the ops CSV
      (<dir>/reports/*/ops_perf_results_*.csv) to ops_scope.py for the program count per scope.

The layer input is an interleaved bf16 row; in the model the batch-1 residual arrives width-sharded, so the first
input reshard of each layer is bench-only (ops_scope.py labels it "bench"). RoPE rows are gathered once outside the
layer, as the model does per step.
"""

import argparse
import statistics
import time

import torch

import ttnn
from models.demos.gemma4_26b_a4b_qb2.tt.generator import Gemma4Generator
from models.demos.gemma4_26b_a4b_qb2.tt.generator_vllm import Gemma4ForCausalLM
from models.demos.utils.trace_region_sizes import build_trace_device_params


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--layers", default="0,5")
    args = p.parse_args()
    torch.manual_seed(0)
    ttnn.set_fabric_config(**Gemma4ForCausalLM.model_capabilities["fabric_config"])
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 4), l1_small_size=16384, **build_trace_device_params("gemma4-26b-a4b-qb2-decoder")
    )
    try:
        gen = Gemma4Generator(mesh, max_seq_len=2048, layer_indices=tuple(int(x) for x in args.layers.split(",")))
        model = gen.model
        caches, table = model.allocate_cache(slots=1, context=256)
        device_table = model.upload(table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        length = 100
        position = model.upload(torch.tensor([[length]], dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        cache_position = model.upload(torch.tensor([length], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        hidden = model.upload(torch.randn(1, 1, 1, model.config.hidden_size) * 2.0)
        for n, index in enumerate(model.layer_indices):
            layer, kind = model.layers[n], model.config.layer_types[index]
            rope = model.rope_decode[kind]
            if getattr(layer.layer.self_attn, "fused_heads", False):
                from models.demos.gemma4_26b_a4b_qb2.tt.multichip_decoder import DecodeRopeRows

                rope = DecodeRopeRows.gather(rope, position)  # once per step in the model

            def forward():
                return layer.decode_forward(
                    hidden,
                    rope_mats=rope,
                    current_pos=position,
                    cache_pos=cache_position,
                    page_table=device_table,
                    kv_cache=caches[n],
                )

            forward()
            ttnn.synchronize_device(mesh)
            if args.profile:
                from tracy import signpost

                signpost(f"LAYER_{index}_START")
                forward()
                ttnn.synchronize_device(mesh)
                signpost(f"LAYER_{index}_END")
                ttnn.ReadDeviceProfiler(mesh)
                continue
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            forward()
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            for _ in range(10):
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            samples = []
            for _ in range(5):
                t0 = time.perf_counter()
                for _ in range(args.iters):
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                samples.append((time.perf_counter() - t0) * 1e6 / args.iters)
            ttnn.release_trace(mesh, trace)
            print(
                f"LAYER_US layer={index} kind={kind} us={statistics.median(samples):.1f} "
                f"samples={[round(s, 1) for s in samples]}",
                flush=True,
            )
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
