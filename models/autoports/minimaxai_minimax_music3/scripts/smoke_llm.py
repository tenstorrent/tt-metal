# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bring-up smoke for MusicLLM: build, prefill the golden prompt, run a few decode steps, time them.

    source ~/mm3-bringup/common.sh && cd $MM3_WT && with_hw_lock timeout 3600 $MM3_PY \
        models/autoports/minimaxai_minimax_music3/scripts/smoke_llm.py
"""
import time

import torch
from loguru import logger

import ttnn
from models.autoports.minimaxai_minimax_music3.reference import hf_llm as R
from models.autoports.minimaxai_minimax_music3.tt.llm import MusicLLM


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=90_000_000)
    mesh.enable_program_cache()
    try:
        t0 = time.time()
        llm = MusicLLM(mesh)
        logger.info(f"build {time.time()-t0:.0f}s; dtypes {llm.dtype_report()}")
        logger.info(f"kv {llm.kv_cache_bytes()}")
        text_ids = torch.load(R.reference_dir() / "text_ids.pt")
        emb = llm.embed_tokens(text_ids)
        logger.info(f"embed_tokens -> {emb.shape} {emb.dtype} {emb.layout}")
        emb_host = ttnn.to_torch(emb)[0].float()
        ref_emb = R.load_embed_weight()[text_ids].float()
        logger.info(f"embedding max abs diff vs table: {(emb_host - ref_emb).abs().max().item()}")
        t0 = time.time()
        h, l = llm.prefill(emb)
        logger.info(
            f"prefill {text_ids.shape} in {time.time()-t0:.2f}s -> hidden {h.shape} logits {l.shape} finite={torch.isfinite(h).all().item()} {torch.isfinite(l).all().item()}"
        )
        logger.info(f"prefill argmax logits per row: {l.argmax(-1).tolist()}; hidden norm {h.norm(dim=-1).tolist()}")
        codes = torch.load(R.reference_dir() / "sampled_codes.pt")
        audio_emb = R.load_audio_embeddings()
        embed_w = R.load_embed_weight()
        S = text_ids.shape[1]
        for i in range(3):
            fc = codes[i].unsqueeze(0).expand(2, -1)
            x = R.embed_audio_frame(embed_w, audio_emb, fc)
            t0 = time.time()
            h, l = llm.decode(x, torch.tensor([S + i, S + i]))
            logger.info(
                f"decode step {i} pos {S+i}: {time.time()-t0:.3f}s finite={torch.isfinite(h).all().item()} argmax {l.argmax(-1).tolist()} stats {llm.decode_stats}"
            )
        # device-side embed_frame path
        res = R.residual_embedding_sum(audio_emb, codes[3].unsqueeze(0).expand(2, -1))
        xdev = llm.embed_frame(codes[3, 0].repeat(2), res)
        xhost = ttnn.to_torch(xdev)[0, 0, :2].float()
        xref = R.embed_audio_frame(embed_w, audio_emb, codes[3].unsqueeze(0).expand(2, -1)).float()
        logger.info(
            f"embed_frame max abs diff vs host formula: {(xhost - xref).abs().max().item()} (ref abs max {xref.abs().max().item()})"
        )
        h, l = llm.decode(xdev, S + 3)
        logger.info(f"decode from device embed ok; stats {llm.decode_stats}")
        # warmed timing
        ttnn.synchronize_device(mesh)
        t0 = time.time()
        n = 20
        for i in range(n):
            llm.decode(x, S + 4 + i)
        ttnn.synchronize_device(mesh)
        logger.info(f"decode step (with host readback) avg {(time.time()-t0)/n*1000:.2f} ms; stats {llm.decode_stats}")
        llm.release()
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
