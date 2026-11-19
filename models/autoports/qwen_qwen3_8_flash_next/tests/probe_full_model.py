# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Small real-layer wrapper probe; never substitutes for all-layer acceptance."""
import argparse
import json
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt.model import QwenModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        default="/home/ttuser/qwen38-flash/hf/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/812d96e10a0dd82b69d749e68693698cf6c36daa",
    )
    p.add_argument("--packed-cache", default="/home/ttuser/qwen38-flash/packed_ple_lloyd_lsq_group10_scale8_wide")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--all-layers", action="store_true")
    p.add_argument("--generator", action="store_true")
    p.add_argument("--strict-fallback", action="store_true")
    p.add_argument("--capacity", type=int, default=128)
    p.add_argument("--reference", type=Path, default=Path("/home/ttuser/qwen38-flash/reference/full_model_reduced.pt"))
    a = p.parse_args()
    if a.strict_fallback:
        ttnn.CONFIG.throw_exception_on_fallback = True
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0, l1_small_size=32768)
    report = {"passed": False, "scope": "full" if a.all_layers else "reduced", "regime": "serial"}
    report["throw_exception_on_fallback"] = bool(ttnn.CONFIG.throw_exception_on_fallback)
    started = time.monotonic()
    try:
        model = QwenModel(
            a.checkpoint, mesh, packed_cache=a.packed_cache, layer_indices=None if a.all_layers else (0, 1, 3)
        )
        if a.generator:
            from transformers import AutoTokenizer

            from models.autoports.qwen_qwen3_8_flash_next.tt.generator import QwenGenerator

            gen = QwenGenerator(model, AutoTokenizer.from_pretrained(a.checkpoint), max_seq_len=a.capacity)
            report["predictions"] = gen.generate([785, 13458, 13], 4)
            report["autoregressive_counters"] = dict(gen.counters)
            repeat = gen.generate([785, 13458, 13], 4)
            report["deterministic"] = repeat == report["predictions"]
            forced = gen.generate([785, 13458, 13], 4, next_input=lambda step, prediction: [785, 13458, 13, 0][step])
            report["teacher_predictions"] = forced
            reference = torch.load(a.reference, weights_only=False)["steps"]
            ranks = torch.cat(
                [reference[0]["top100_ids"].reshape(-1, 100)[-1:]]
                + [step["top100_ids"].reshape(-1, 100) for step in reference[1:]]
            )
            report["teacher_topk"] = {
                str(k): float((torch.tensor(forced)[:, None] == ranks[:, :k]).any(-1).float().mean())
                for k in (1, 5, 100)
            }
            report["counters"] = dict(gen.counters)
            report["passed"] = (
                report["deterministic"] and report["teacher_topk"]["5"] >= 0.98 and report["teacher_topk"]["100"] == 1
            )
            assert report["passed"], report
            gen.close()
            return
        cache = model.create_cache(capacity=a.capacity)
        report["loaded_seconds"] = time.monotonic() - started
        view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
        report["memory_view"] = {
            key: getattr(view, key)
            for key in (
                "num_banks",
                "total_bytes_per_bank",
                "total_bytes_allocated_per_bank",
                "total_bytes_free_per_bank",
                "largest_contiguous_bytes_free_per_bank",
            )
        }
        ids = torch.tensor([[785, 13458, 13]], dtype=torch.int32)
        words = torch.zeros((1, 3, 16), dtype=torch.int32)
        words[:, :, 0] = ids
        words[:, :, 1] = 1
        tok = ttnn.from_torch(ids, device=mesh, dtype=ttnn.uint32)
        word = ttnn.from_torch(words, device=mesh, dtype=ttnn.uint32)
        out = model.forward_chunk(tok, word, cache=cache)
        host = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=3))
        report["shape"] = list(host.shape)
        report["finite"] = bool(torch.isfinite(host).all())
        report["predictions"] = host.argmax(-1).flatten().tolist()
        report["passed"] = report["finite"] and host.shape[-1] == model.config.vocab_size
        if not a.all_layers:
            reference = torch.load(a.reference, weights_only=False)["steps"][0]
            expected = reference["logits"].float().reshape_as(host)
            report["logit_pcc"] = float(torch.corrcoef(torch.stack([expected.flatten(), host.float().flatten()]))[0, 1])
            predictions = host.argmax(-1).reshape(-1, 1)
            ranks = reference["top100_ids"].reshape(-1, 100)
            report["topk"] = {str(k): float((predictions == ranks[:, :k]).any(-1).float().mean()) for k in (1, 5, 100)}
            report["passed"] &= report["topk"]["5"] >= 0.98 and report["topk"]["100"] == 1.0
    finally:
        report["wall_seconds"] = time.monotonic() - started
        a.output.write_text(json.dumps(report, indent=2) + "\n")
        ttnn.close_mesh_device(mesh)
    assert report["passed"], report


if __name__ == "__main__":
    main()
