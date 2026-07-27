# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Standalone correctness probe for the GLM fused KV cache branch."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import ttnn

from models.common.rmsnorm import RMSNorm
from models.experimental.glm4_moe_lite.tt.config import Glm4MoeLiteHParams
from models.experimental.glm4_moe_lite.tt.decoder_layer_tt import _fused_kv_branch_forward
from models.experimental.glm4_moe_lite.tt.layer_weights import _prepare_fused_kv_branch_weights
from models.experimental.glm4_moe_lite.tt.weights import load_glm_lazy_state_dict


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return float(torch.corrcoef(torch.stack((a, b)))[0, 1])


def main() -> None:
    snapshot = Path(
        "/home/tt-admin/.cache/huggingface/hub/models--zai-org--GLM-4.7-Flash/"
        "snapshots/7dd20894a642a0aa287e9827cb1a1f7f91386b67"
    )
    with (snapshot / "config.json").open() as f:
        hparams = Glm4MoeLiteHParams.from_hf_config(SimpleNamespace(**json.load(f)))
    state = load_glm_lazy_state_dict(snapshot)

    mesh_raw = os.environ.get("TT_TEST_MESH_SHAPE", "1x1").lower().replace("x", ",")
    mesh_rows, mesh_cols = (int(part) for part in mesh_raw.split(","))
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(mesh_rows, mesh_cols))
    try:
        fused = _prepare_fused_kv_branch_weights(
            device=mesh,
            state=state,
            layer_idx=0,
            hparams=hparams,
            cache_dir=None,
            dense_dtype=ttnn.bfloat8_b,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        fused["rmsnorm_fn"] = RMSNorm(
            device=mesh,
            dim=hparams.kv_lora_rank,
            eps=hparams.rms_norm_eps,
            state_dict=state,
            state_dict_prefix="model.layers.0.self_attn.",
            weight_key="kv_a_layernorm",
            weight_cache_path=None,
            weight_dtype=ttnn.bfloat16,
            is_distributed=False,
        )

        torch.manual_seed(0)
        x = torch.randn((1, 1, 1, int(hparams.hidden_size)), dtype=torch.bfloat16)
        cos = torch.ones((1, 1, 1, int(hparams.qk_rope_head_dim)), dtype=torch.bfloat16)
        sin = torch.zeros_like(cos)
        mapper = ttnn.ReplicateTensorToMesh(mesh)

        def to_tt(t: torch.Tensor) -> ttnn.Tensor:
            return ttnn.from_torch(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )

        actual_tt = _fused_kv_branch_forward(
            device=mesh,
            x=to_tt(x),
            fused_kv=fused,
            cos_batch=to_tt(cos),
            sin_batch=to_tt(sin),
        )
        ttnn.synchronize_device(mesh)
        actual_shards = [ttnn.to_torch(shard)[:, :, :1, :576] for shard in ttnn.get_device_tensors(actual_tt)]
        actual = actual_shards[0]

        weight = state["model.layers.0.self_attn.kv_a_proj_with_mqa.weight"].float()
        projected = torch.matmul(x.float(), weight.t())
        nope, rope = torch.split(projected, [512, 64], dim=-1)
        gamma = state["model.layers.0.self_attn.kv_a_layernorm.weight"].float()
        nope = nope * torch.rsqrt(nope.pow(2).mean(dim=-1, keepdim=True) + float(hparams.rms_norm_eps)) * gamma
        golden = torch.cat((nope, rope), dim=-1)

        full_pcc = _pcc(golden, actual)
        nope_pcc = _pcc(golden[..., :512], actual[..., :512])
        rope_pcc = _pcc(golden[..., 512:], actual[..., 512:])
        print(f"full PCC={full_pcc:.6f}")
        print(f"nope PCC={nope_pcc:.6f}")
        print(f"rope PCC={rope_pcc:.6f}")
        print(f"golden range=({golden.min().item():.4f}, {golden.max().item():.4f})")
        print(f"actual range=({actual.min().item():.4f}, {actual.max().item():.4f})")
        print(f"actual nope range=({actual[..., :512].min().item():.4f}, " f"{actual[..., :512].max().item():.4f})")
        print(f"actual rope range=({actual[..., 512:].min().item():.4f}, " f"{actual[..., 512:].max().item():.4f})")
        print(f"golden[:8]={golden.flatten()[:8].tolist()}")
        print(f"actual[:8]={actual.flatten()[:8].tolist()}")
        actual_flat = actual.flatten()
        golden_flat = golden.flatten()
        for actual_block in range(18):
            scores = [
                _pcc(
                    golden_flat[golden_block * 32 : (golden_block + 1) * 32],
                    actual_flat[actual_block * 32 : (actual_block + 1) * 32],
                )
                for golden_block in range(18)
            ]
            best = max(range(18), key=lambda i: scores[i])
            print(f"block {actual_block:02d} best_golden={best:02d} pcc={scores[best]:.4f}")
        assert torch.isfinite(actual).all(), "fused KV output contains NaN or Inf"
        assert full_pcc >= 0.999, f"full fused KV PCC too low: {full_pcc}"
        assert nope_pcc >= 0.999, f"no-PE projection PCC too low: {nope_pcc}"
        assert rope_pcc >= 0.999, f"RoPE projection PCC too low: {rope_pcc}"
        for shard_idx, shard in enumerate(actual_shards):
            shard_pcc = _pcc(golden, shard)
            assert torch.isfinite(shard).all(), f"fused KV shard {shard_idx} contains NaN or Inf"
            assert shard_pcc >= 0.999, f"fused KV shard {shard_idx} PCC too low: {shard_pcc}"
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
