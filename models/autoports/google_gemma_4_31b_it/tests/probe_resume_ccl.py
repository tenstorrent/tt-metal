# SPDX-License-Identifier: Apache-2.0
"""Test ring reduction parity using identical real local projection tensors."""

import argparse
import bz2
import json
from pathlib import Path
from unittest.mock import patch

import torch

import ttnn

from ..tt.generator import Gemma4Generator
from ..tt.optimized_decoder import OptimizedDecoder
from .probe_prefill_geometry import compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("models/autoports/google_gemma_4_31b_it/readiness_vllm/resume_ccl.json")
    )
    args = parser.parse_args()
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024)
    gen = None
    try:
        gen = Gemma4Generator(mesh, layer_indices=(0, 5), batch_size=1, max_seq_len=4096)
        gen.prepare_decode()
        layer = gen.model.layers[0]
        text = bz2.open("models/tt_transformers/tests/tale-of-two-cities.txt.bz2", "rt").read()
        tokens = torch.tensor(gen.tokenizer.encode(text, add_special_tokens=True)[:1024], dtype=torch.int32)
        original = OptimizedDecoder._linear
        saved = []

        def capture(self, x, name):
            out = original(self, x, name)
            if self is layer and not self.decoding and name == "self_attn.o_proj":
                assert not saved and out.shape[2] == 1024
                saved.append(
                    ttnn.clone(ttnn.typecast(ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG), ttnn.bfloat8_b))
                )
            return out

        gen.reset()
        with patch.object(OptimizedDecoder, "_linear", capture):
            for _, _, hidden in gen._prefill_hidden_chunks(tokens, slot=0, start=0, end=1024):
                del hidden
        source = saved[0]
        assert source.shape[-1] == 5376

        def reduce(x, workers=None):
            return ttnn.experimental.reduce_scatter_minimal_async(
                x,
                dim=3,
                multi_device_global_semaphore=layer.ccl.get_and_cycle_rs_semaphore_handles(),
                barrier_semaphore=layer.ccl.get_and_cycle_barrier_semaphore_handle(),
                topology=layer.topology,
                num_links=2,
                num_workers_per_link=workers,
                num_buffers_per_channel=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        def host(tensor):
            return torch.cat(
                [ttnn.to_torch(chip).float().reshape(tensor.shape[2], -1) for chip in ttnn.get_device_tensors(tensor)],
                dim=1,
            )

        def rows(tensor, start, end):
            return ttnn.slice(tensor, (0, 0, start, 0), (1, 1, end, tensor.shape[-1]))

        def zeros(count):
            return ttnn.zeros(
                (1, 1, count, source.shape[-1]),
                dtype=source.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        baseline = host(reduce(source))
        results = []
        for start in (32, 64, 128, 256):
            suffix = rows(source, start, 1024)
            source_identity = compare(host(source)[start:], host(suffix))
            assert source_identity["exact_equal"]
            lead = start % 256
            for mode in ("direct", "fixed_workers_2", "tail_pad_1024", "leading_phase_pad", "absolute_leading_pad"):
                padding = lead if mode == "leading_phase_pad" else start if mode == "absolute_leading_pad" else 0
                x = ttnn.concat([zeros(padding), suffix], dim=2) if padding else suffix
                if mode == "tail_pad_1024":
                    x = ttnn.concat([x, zeros(start)], dim=2)
                output = host(reduce(x, workers=2 if mode == "fixed_workers_2" else None))
                candidate = output[padding : padding + 1024 - start]
                results.append(
                    {
                        "start": start,
                        "mode": mode,
                        "leading_rows": padding,
                        "physical_rows": x.shape[2],
                        "input_exact": source_identity["exact_equal"],
                        **compare(baseline[start:], candidate),
                    }
                )
        report = {
            "scope": "real_layer0_local_output_projection_ccl_only",
            "source_dtype": str(source.dtype),
            "source_shape": list(source.shape),
            "mesh": [1, 4],
            "num_links": 2,
            "precision_changed": False,
            "hypothesis": "42 output tiles per row and 8-tile ring chunks give a 256-token phase period; leading alignment preserves rank accumulation order.",
            "comparisons": results,
            "limitations": [
                "Correctness-only host reads; no performance claim.",
                "All cases use slices of the same real local projection, not regenerated projections.",
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"output": str(args.output), "comparisons": len(results)}))
    finally:
        if gen is not None:
            gen.release_persistent_capture()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
