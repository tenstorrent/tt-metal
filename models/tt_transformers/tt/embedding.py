# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from models.common.lightweightmodule import LightweightModule


class Embedding(LightweightModule):
    def __init__(
        self,
        mesh_device,
        args,
        weight_cache_path,
        state_dict,
        dtype,
    ):
        super().__init__()

        self.mesh_device = mesh_device

        base_name = args.get_state_dict_prefix("", None) + "tok_embeddings.weight"
        self._debug_cpu_weight = state_dict[base_name].detach().cpu()        
        torch_weight = state_dict[base_name].unsqueeze(0).unsqueeze(0)
        cache_name = None if args.dummy_weights else weight_cache_path / base_name
        print("EMB base_name:", base_name)
        print("EMB in state_dict:", base_name in state_dict)
        if base_name in state_dict:
            w = state_dict[base_name]
            print("EMB torch_weight shape:", tuple(w.shape), "dtype:", w.dtype, "norm:", float(w.float().norm()))
        print("EMB cache_file_name:", cache_name)
        self.weights = ttnn.as_tensor(
            torch_weight,
            dtype=dtype,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device=mesh_device, dims=(None, 3), mesh_shape=args.cluster_shape),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=args.get_model_config()["EMB_WEIGHTS_MEMCFG"],
            cache_file_name=cache_name,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        out = ttnn.embedding(x, self.weights, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    
        # ===== DEBUG: inspect token ids reaching Embedding =====
        try:
            import torch
            x_host = ttnn.to_torch(ttnn.get_device_tensors(x)[0])
            flat = x_host.reshape(-1).tolist()
            print("\n=== EMB INPUT IDS (device0) ===")
            print("x_host shape:", tuple(x_host.shape))
            print("first 32 ids:", flat[:32])
        
            target = 11  # HF token at prompt idx=16
            pos = None
            for i, t in enumerate(flat[:512]):  # search first 512 positions
                if t == target:
                    pos = i
                    break
            print("first occurrence of token_id 11 in first512:", pos)
            print("=== END EMB INPUT IDS ===\n")
        except Exception as e:
            print("EMB INPUT IDS debug failed:", repr(e))
        # ===== END DEBUG =====
        
    
        return out




class ScaledEmbedding(Embedding):
    def __init__(self, mesh_device, args, weight_cache_path, state_dict, dtype, embed_scale: float = 1.0):
        super().__init__(mesh_device, args, weight_cache_path, state_dict, dtype)
        self.embed_scale = embed_scale

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        e = ttnn.embedding(x, self.weights, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        s = ttnn.multiply(e, self.embed_scale, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return s
