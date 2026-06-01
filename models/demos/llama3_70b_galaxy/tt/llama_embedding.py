# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtLlamaEmbedding(LightweightModule):
    def __init__(
        self,
        mesh_device,
        args,
        weight_cache_path,
        state_dict,
        dtype,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.args = args

        base_name = args.get_state_dict_prefix("", None) + "tok_embeddings.weight"
        torch_weight = self.state_dict[base_name].unsqueeze(0).unsqueeze(0)
        cache_name = None if args.dummy_weights else weight_cache_path / base_name
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
        x = ttnn.reshape(x, ttnn.Shape((1, 1, 1, x.shape[-2] * x.shape[-1])))
        use_decode_memcfg = x.shape[-1] <= 32
        # Decode token embedding feeds the attention-norm input path first.
        # Keep it on the ATTN input ring memcfg (32-height) even if later decode residuals are 128-height.
        out_memcfg = (
            self.args.model_config["SHARDED_ATTN_INPUT_RING_MEMCFG"] if use_decode_memcfg else ttnn.DRAM_MEMORY_CONFIG
        )
        x = ttnn.embedding(
            x,
            self.weights,
            layout=ttnn.TILE_LAYOUT,
            memory_config=out_memcfg,
            dtype=ttnn.bfloat8_b
            if x.shape[-1] > 32
            else ttnn.bfloat16,  # Keep bfloat16 for decode, bfloat8_b for prefill
        )
        x = ttnn.reshape(x, ttnn.Shape((1, 1, x.shape[-2], x.shape[-1])))
        return x
