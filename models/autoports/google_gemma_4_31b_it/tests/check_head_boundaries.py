"""Compare terminal layouts on real reduced-model activations at CB boundaries."""
import json
from pathlib import Path

import torch

import ttnn

from ..tt.generator import Gemma4Generator


def main():
    torch.set_num_threads(8)
    root = Path("models/autoports/google_gemma_4_31b_it/doc/optimized_full_model")
    prompt = json.loads((root / "aime24_prompt.json").read_text())["tokens"]
    prompt = (prompt * 8)[:1024]
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024)
    try:
        gen = Gemma4Generator(mesh, layer_indices=[0, 5])
        gen.prefill.stage(torch.tensor(prompt), slot=0, start=0, size=1024, valid=1024)
        hidden = gen.prefill.hidden(size=1024, start=0, kv_cache=gen.kv_cache)
        splits = gen.model.lm_head
        pieces = [ttnn.to_memory_config(w, ttnn.DRAM_MEMORY_CONFIG) for w in splits]
        baseline = ttnn.concat(pieces, dim=-1)
        del pieces
        records = []
        for rows in (32, 512, 544, 768, 800, 1024):
            x = ttnn.slice(hidden, (0, 0, 0, 0), (1, 1, rows, 5376))
            gen.model.head_mode = "interleaved"
            gen.model.lm_head = baseline
            ref = gen.model.terminal(x)
            gen.model.head_mode = "dram"
            gen.model.lm_head = splits
            out = gen.model.terminal(x)
            pccs = []
            maxima = []
            for a, b in zip(ttnn.get_device_tensors(ref), ttnn.get_device_tensors(out)):
                ar = ttnn.to_torch(a).float().flatten()
                br = ttnn.to_torch(b).float().flatten()
                pccs.append(float(torch.corrcoef(torch.stack([ar, br]))[0, 1]))
                maxima.append(float((ar - br).abs().max()))
            assert min(pccs) > 0.9999, (rows, pccs)
            records.append(dict(rows=rows, pcc_by_device=pccs, max_abs_by_device=maxima))
            (root / "head_boundaries.json").write_text(
                json.dumps(dict(scope="terminal only, real reduced-model activations", records=records), indent=2)
                + "\n"
            )
            del x, ref, out, ar, br
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
