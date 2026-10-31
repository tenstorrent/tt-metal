"""All-layer eager prefill smoke and allocator accounting; no throughput claim."""

import json
from pathlib import Path

import torch

import ttnn

from ..tt.model import Gemma4Model


def main():
    torch.set_num_threads(8)
    root = Path("models/autoports/google_gemma_4_31b_it/doc/full_model")
    prompt = json.loads((root / "aime24_prompt.json").read_text())["tokens"]
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024)

    def memory():
        view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
        return {
            key: int(getattr(view, key))
            for key in (
                "num_banks",
                "total_bytes_per_bank",
                "total_bytes_allocated_per_bank",
                "largest_contiguous_bytes_free_per_bank",
            )
        }

    report = {"scope": "all60layers", "execution": "eager smoke only", "memory": {"empty": memory()}}
    try:
        model = Gemma4Model(mesh)
        report["memory"]["weights_rope_ccl"] = memory()
        caches, pages = model.allocate_cache(batch=1, context=262144)
        report["memory"]["with_full_context_cache"] = memory()
        logits = None
        for start in range(0, len(prompt), 1024):
            valid = min(1024, len(prompt) - start)
            physical = (valid + 31) // 32 * 32
            tokens = torch.zeros(1, physical, dtype=torch.int32)
            tokens[0, :valid] = torch.tensor(prompt[start : start + valid])
            ids = model.upload(tokens, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
            pos = model.upload(
                torch.arange(start, start + physical, dtype=torch.int32).reshape(1, -1),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            chunk = [ttnn.slice(p, (0, start // 32), (1, (start + physical) // 32)) for p in pages]
            print("PREFILL_CHUNK", start, valid, flush=True)
            logits = model.prefill_chunk(
                ids,
                pos,
                kv_cache=caches,
                page_tables=pages,
                chunk_tables=chunk,
                chunk_start=start,
                valid_tokens=physical,
            )
            host = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).reshape(physical, -1)
        torch.save(host[valid - 1].clone(), "bringup/artifacts/full_model_aime_prefill_last.pt")
        report["last_token"] = int(host[valid - 1].argmax())
        report["prompt_length"] = len(prompt)
        report["memory"]["after_prefill"] = memory()
        print("PREFILL_RESULT", report, flush=True)
    finally:
        root.joinpath("full_prefill_smoke.json").write_text(json.dumps(report, indent=2) + "\n")
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
