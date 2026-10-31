"""Full-stack timings with diagnostics disabled; direct logit determinism gate."""

import argparse
import json
import os
from pathlib import Path

import torch

import ttnn

from ..tt.generator import Gemma4Generator
from .full_generation_evidence import performance
from .run_full_batch import inventory, memory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 32, 16, 8])
    args = parser.parse_args()
    for key in (
        "TT_METAL_TRACE_ALLOC_TRACKING",
        "TT_METAL_DEVICE_PROFILER",
        "TT_METAL_WATCHER",
        "TT_METAL_DPRINT_CORES",
    ):
        if os.environ.get(key, "0") not in ("", "0"):
            raise ValueError(f"Clean timing requires {key} disabled")
    torch.set_num_threads(8)
    root = Path("models/autoports/google_gemma_4_31b_it/doc/full_model")
    prompt = json.loads((root / "aime24_prompt.json").read_text())["tokens"]
    ttnn.CONFIG.throw_exception_on_fallback = True
    for batch in args.batches:
        # Mesh teardown clears fabric configuration; restore it for each open.
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024)
        try:
            gen = Gemma4Generator(mesh, batch_size=batch)
            mem = dict(
                construction_stages=gen.model.allocation_bytes, constructed=memory(mesh), inventory=inventory(gen)
            )
            gen.prepare_decode()
            mem["prepared"] = memory(mesh)
            (root / f"memory_clean_b{batch}.json").write_text(json.dumps(mem, indent=2) + "\n")
            if batch == 32:
                snapshots = []
                for _ in range(2):
                    gen.generate_batch([prompt] * batch, 4, ignore_eos=True)
                    logits = gen._host_logits(gen.logits).reshape(batch, -1)
                    assert torch.equal(logits, logits[0].expand_as(logits))
                    snapshots.append(logits)
                assert torch.equal(*snapshots)
                (root / "logit_determinism.json").write_text(
                    json.dumps(
                        dict(
                            scope="full60layers",
                            batch=batch,
                            prompt_tokens=len(prompt),
                            generated_tokens=4,
                            comparison="bitwise equality of full vocabulary logits after traced decode",
                            repeated_equal=True,
                            equal_across_all_slots=True,
                        ),
                        indent=2,
                    )
                    + "\n"
                )
            old = root / f"performance_b{batch}.json"
            if old.exists():
                prior = json.loads(old.read_text())
                if prior.get("diagnostics", {}).get("TT_METAL_TRACE_ALLOC_TRACKING", "1") == "1":
                    old.rename(root / f"performance_instrumented_b{batch}.json")
            performance(gen, root)
        finally:
            ttnn.close_mesh_device(mesh)
        del gen


if __name__ == "__main__":
    main()
