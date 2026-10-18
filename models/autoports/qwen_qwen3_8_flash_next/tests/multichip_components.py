"""Serialized real-weight single-chip versus TP4 component controls."""

import argparse, json, time
from pathlib import Path
import torch, ttnn
from transformers import AutoConfig
from models.autoports.qwen_qwen3_8_flash_next.tt.checkpoint import CheckpointState
from models.autoports.qwen_qwen3_8_flash_next.tt.optimized_decoder import OptimizedHyperConnection, OptimizedMoE
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import (
    MeshOps,
    ParallelHyperConnection,
    ParallelMoE,
    _hidden_parts,
)
from models.autoports.qwen_qwen3_8_flash_next.tests.metrics import pcc_metrics

CHECKPOINT = Path(
    "/mnt/models/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/de4b8e4d43b917e7706784d8bb445c9af86a3540"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["single", "mesh"], required=True)
    ap.add_argument("--part", choices=["hc", "moe"], required=True)
    ap.add_argument("--length", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(8)
    cfg = AutoConfig.from_pretrained(CHECKPOINT, local_files_only=True).text_config
    state = CheckpointState(CHECKPOINT, "model.language_model.layers.0.")
    root = Path("models/autoports/qwen_qwen3_8_flash_next/doc/multichip_decoder")
    root.mkdir(exist_ok=True)
    cache = Path("/home/vkovacevic/qwen38-flash/tt-metal-cache/qwen_qwen3_8_flash_next/multichip")
    cache.mkdir(exist_ok=True)
    torch.manual_seed(2026)
    h = torch.randn(1, 1, a.length, 2560).repeat(1, 1, 1, 4) * 0.05 + torch.randn(1, 1, a.length, 10240) * 0.005
    x = h if a.part == "hc" else h[..., :2560]
    x = x.bfloat16()
    if a.impl == "mesh":
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    m = ttnn.open_mesh_device(ttnn.MeshShape(1, 4 if a.impl == "mesh" else 1), trace_region_size=0)
    try:
        if a.impl == "mesh":
            ops = MeshOps(m)
            model = (
                ParallelHyperConnection(state, "attn_hyper_connection", cfg, ops)
                if a.part == "hc"
                else ParallelMoE(state, cfg, ops)
            )
            xd = ops.shards(_hidden_parts(x)) if a.part == "hc" else ops.upload(x)
        else:
            model = (
                OptimizedHyperConnection.from_state_dict(
                    state, prefix="attn_hyper_connection", hf_config=cfg, mesh_device=m
                )
                if a.part == "hc"
                else OptimizedMoE.from_state_dict(state, hf_config=cfg, mesh_device=m)
            )
            xd = ttnn.from_torch(x, device=m, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

        def call():
            if a.part == "moe":
                return [model(xd)]
            mixed, inject = model.read(xd)
            update = mixed
            if a.impl == "mesh":
                # Read provides full mixed input; use a known local update to
                # test the write boundary without incorrectly summing copies.
                update = local_update
            return [mixed, inject, model.write(xd, update, inject)]

        if a.impl == "mesh" and a.part == "hc":
            local_update = ops.shards([x[..., :2560][..., r * 640 : (r + 1) * 640] for r in range(4)])
        # For write comparison use identical known update on both contracts.
        if a.part == "hc":
            mixed, inject = model.read(xd)
            upd = (
                local_update
                if a.impl == "mesh"
                else ttnn.from_torch(x[..., :2560], device=m, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
            )
            outputs = [mixed, inject, model.write(xd, upd, inject)]
        else:
            outputs = call()
        ttnn.synchronize_device(m)
        hosts = []
        for idx, t in enumerate(outputs):
            if a.impl == "single":
                hosts.append(ttnn.to_torch(t))
                continue
            shards = [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]
            if a.part == "moe":
                host = torch.cat(shards, -1)
            elif idx < 2:
                for other in shards[1:]:
                    assert torch.equal(shards[0], other)
                host = shards[0]
            else:
                host = torch.cat([d.reshape(1, 1, a.length, 4, 640) for d in shards], -1).reshape(x.shape)
            hosts.append(host)
        path = cache / f"{a.impl}_{a.part}_{a.length}.pt"
        torch.save(hosts, path)
        if a.impl == "mesh":
            reference = torch.load(cache / f"single_{a.part}_{a.length}.pt", weights_only=True)
            metrics = [pcc_metrics(r, h) for r, h in zip(reference, hosts)]
            print(json.dumps(metrics), flush=True)
            (root / f"component_{a.part}_{a.length}.json").write_text(json.dumps(metrics, indent=2))
            assert min(q["pcc"] for q in metrics) >= 0.995
        print("COMPONENT_OK", a.impl, a.part, a.length, flush=True)
    finally:
        ttnn.close_mesh_device(m)


if __name__ == "__main__":
    main()
