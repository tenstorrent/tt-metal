"""Reduced real-layer wrapper probe; never full-model acceptance evidence."""

import time

import torch

import ttnn
from models.autoports.google_gemma_4_31b_it.tt.model import Gemma4Model


def main():
    torch.set_num_threads(16)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024)
    try:
        start = time.perf_counter()
        model = Gemma4Model(mesh, layer_indices=(0, 5))
        print("MODEL_LOADED", time.perf_counter() - start, flush=True)
        caches, pages = model.allocate_cache(batch=1, context=256)
        token = model.upload(
            torch.tensor([[2] * 32], dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        pos = model.upload(
            torch.arange(32, dtype=torch.int32).reshape(1, 32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        chunks = [ttnn.slice(p, (0, 0), (1, 1)) for p in pages]
        out = model.prefill_chunk(
            token, pos, kv_cache=caches, page_tables=pages, chunk_tables=chunks, chunk_start=0, valid_tokens=32
        )
        prefill_host = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1))
        print(
            "PREFILL", out.shape, ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float().abs().max().item(), flush=True
        )
        token = model.upload(torch.tensor([[2]], dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        pos = model.upload(torch.tensor([[32]], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        out = model.decode(token, pos, kv_cache=caches, page_tables=pages)
        decode_host = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1))
        torch.save({"prefill": prefill_host, "decode": decode_host}, "bringup/artifacts/full_model_reduced_logits.pt")
        print(
            "DECODE", out.shape, ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float().abs().max().item(), flush=True
        )
        from types import SimpleNamespace

        from models.common.sampling.tt_sampling import TTSampling

        sampler = TTSampling(
            mesh,
            model.ccl,
            SimpleNamespace(
                vocab_size=model.config.vocab_size, max_batch_size=32, max_top_k=32, cluster_shape=tuple(mesh.shape)
            ),
        )
        feedback = model.upload(
            torch.zeros(1, 1, 1, 32, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        sampled = sampler(out, tt_out_tok=feedback)
        print("SAMPLED", ttnn.to_torch(ttnn.get_device_tensors(feedback)[0]), flush=True)
        print("PROBE_SECONDS", time.perf_counter() - start, flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
