# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
from typing import List
from tqdm import tqdm
import torch
import ttnn
from ttnn import ShardTensorToMesh, ReplicateTensorToMesh

from models.utility_functions import nearest_32, profiler
from models.demos.tg.llama3_70b.tt.llama_decoder_galaxy import TtLlamaDecoder_galaxy
from models.demos.tg.llama3_70b.tt.llama_embedding_galaxy import TtLlamaEmbedding_galaxy
from models.demos.t3000.llama2_70b.tt.llama_common import (
    freqs_to_rotation_matrix,
    get_rotation_mat,
    precompute_freqs,
    get_rot_transformation_mat,
    num_to_corerange,
    gather_cos_sin,
    ShardTensor2dMesh,
    ConcatMesh2DToTensor,
)
from models.demos.tg.llama3_70b.tt.llama_common import (
    tt_all_reduce,
    tt_all_gather,
)
from models.demos.t3000.falcon40b.tt.model_utils import (
    matmul_2d_config_from_tensor_shapes,
    matmul_1d_config_from_tensor_shapes,
)


class TtLlamaModel_galaxy:
    def __init__(
        self,
        mesh_device,
        cluster_shape,
        state_dict,
        base_url,
        n_layers,
        model_config,
        configuration,
        cache_path=None,
        read_cache=False,
    ):
        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.num_devices = mesh_device.get_num_devices()
        self.model_config = model_config
        self.read_cache = read_cache
        self.cluster_shape = cluster_shape

        self.hidden_size = configuration.dim
        self.n_heads = configuration.n_heads
        self.n_local_heads = self.n_heads // self.num_devices
        self.padded_local_heads = 32
        self.head_dim = self.hidden_size // self.n_heads
        self.max_seq_len = configuration.max_seq_len
        self.vocab_size = configuration.vocab_size
        self.padded_vocab_size = 128 * 1024
        self.norm_eps = configuration.norm_eps
        self.llama3 = self.vocab_size == 128256
        self.rope_theta = configuration.rope_theta if self.llama3 else 10000.0

        self.cache_path = cache_path
        # Transformation matrix for rotary embeddings
        transformation_mat_torch = get_rot_transformation_mat(32)  # 32 for tile size
        transformation_mats = ttnn.as_tensor(
            transformation_mat_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ReplicateTensorToMesh(mesh_device),
        )

        logger.info("Creating Layers")
        self.layers = [
            TtLlamaDecoder_galaxy(
                mesh_device,
                cluster_shape,
                state_dict,
                base_url,
                layer_num,
                model_config,
                configuration,
                transformation_mats,
                cache_path=cache_path,
                read_cache=read_cache,
            )
            for layer_num in tqdm(range(n_layers))
        ]
        logger.info("Done creating layers")

        # Rotary Embedding
        self.cos, self.sin = precompute_freqs(
            self.head_dim, self.max_seq_len * 2, self.rope_theta, use_scaled=False
        )  # for prefill
        self.rot_emb = freqs_to_rotation_matrix(self.cos, self.sin)  # for decode
        # Embedding
        self.tt_embd = TtLlamaEmbedding_galaxy(
            mesh_device,
            cluster_shape,
            state_dict,
            cache_path,
        )
        self.get_model_config()
        self.load_weights()

    def set_model_config(self, model_config):
        self.model_config = model_config
        for layer in self.layers:
            layer.set_model_config(model_config)
        self.get_model_config()

    def get_model_config(self):
        self.LN_COMPUTE_KERNEL_CONFIG = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        self.COMPUTE_KERNEL_CONFIG = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.LM_HEAD_ACT_MEMCFG = ttnn.create_sharded_memory_config(
            shape=(32, 2048 // 32),
            core_grid=ttnn.CoreGrid(y=4, x=8),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        if self.model_config["LLM_MODE"] == "prefill":
            # seq_len is 32 if we slice LM head input
            hidden_size_per_chip = self.hidden_size // self.cluster_shape[0]
            self.LM_HEAD_PROGCFG = matmul_1d_config_from_tensor_shapes(
                (
                    1,
                    1,
                    self.model_config["PADDING_LENGTH"],
                    hidden_size_per_chip,
                ),  # get only last padding_length (32) tokens
                (
                    1,
                    1,
                    hidden_size_per_chip,
                    self.padded_vocab_size // self.cluster_shape[1],
                ),  # (1, 1, 2048, 16 * 1024)
                grid=ttnn.CoreGrid(x=8, y=4),
                overwrite_subblock_h=1,
                overwrite_subblock_w=1,
            )

    def load_weights(self):
        norm_str = "norm.weight"
        lm_head_str = "output.weight"

        norm_sharded_cache_str = "norm_sharded_galaxy.weight"
        lm_head_cache_str = "output_galaxy.weight"

        if not self.read_cache:
            H = 8 * 1024
            if self.llama3:
                PADDED_VOCAB = 128 * 1024
            else:
                PADDED_VOCAB = 32 * 1024
            padded_lm_head = torch.zeros(1, 1, H, PADDED_VOCAB)
            padded_lm_head[:, :, :, : self.vocab_size] = self.state_dict[lm_head_str].transpose(-2, -1)

            pt_norm_weight = self.state_dict[norm_str].reshape([1, 1, -1, 32])
        else:
            padded_lm_head = None
            pt_norm_weight = None

        self.lm_head = ttnn.as_tensor(
            padded_lm_head,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ShardTensor2dMesh(self.mesh_device, dims=(2, 3), cluster_shape=self.cluster_shape),
            cache_file_name=self.cache_path / lm_head_cache_str,
        )

        self.norm_sharded = ttnn.as_tensor(
            pt_norm_weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ShardTensor2dMesh(self.mesh_device, (2, None), self.cluster_shape),
            cache_file_name=self.cache_path / norm_sharded_cache_str,
        )

    def prepare_inputs(self, inp_ids, start_pos, valid_seq_len=None, attn_mask=None):
        assert inp_ids.dim() == 2
        batch, seq_len = inp_ids.shape

        cache_name = lambda name: self.cache_path / (f"{'llama3_' if self.llama3 else ''}{name}")

        if self.model_config["LLM_MODE"] == "decode":
            inp_ids = inp_ids.reshape(seq_len, 1, 1, batch)
        else:
            inp_ids = inp_ids.reshape(batch, 1, 1, seq_len)

        x = ttnn.as_tensor(
            inp_ids,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ReplicateTensorToMesh(self.mesh_device),
        )

        xs = self.tt_embd(x)

        if self.model_config["LLM_MODE"] == "decode":
            assert seq_len == 1, "Decode mode only supports seq_len=1"
            assert xs.shape == (seq_len, 1, batch, self.hidden_size // self.cluster_shape[0])

            ACT_MEMCFG = ttnn.create_sharded_memory_config(
                shape=(xs.shape[2], xs.shape[3] // 8),
                core_grid=ttnn.CoreGrid(y=1, x=8),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

            xs = ttnn.to_memory_config(xs, memory_config=ACT_MEMCFG)

            if isinstance(start_pos, int):
                cache_idxs = torch.tensor([start_pos for _ in range(batch // self.cluster_shape[0])], dtype=torch.int64)
            else:
                raise ValueError("start_pos must be an int, different start_pos for each user not supported yet")
                cache_idxs = start_pos

            # TODO : Create different rot_mat for each user_groups in the cluster
            rot_mat = get_rotation_mat(self.rot_emb, cache_idxs, seq_len, batch // self.cluster_shape[0])
            assert rot_mat.size() == (1, batch // self.cluster_shape[0], self.head_dim, self.head_dim)

            shard_spec_n_cores_grid = ttnn.CoreRangeSet({num_to_corerange(batch // 4)})
            ROT_MAT_MEMCFG = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    shard_spec_n_cores_grid,
                    [
                        self.head_dim,
                        self.head_dim,
                    ],
                    ttnn.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            )

            rot_mats = ttnn.as_tensor(
                rot_mat,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                cache_file_name=cache_name(f"rot_mat_decode_galaxy_{start_pos}"),
                memory_config=ROT_MAT_MEMCFG,
                mesh_mapper=ReplicateTensorToMesh(self.mesh_device),
            )

            attn_masks = None
        elif self.model_config["LLM_MODE"] == "prefill":
            assert seq_len % 32 == 0 and seq_len > 0, "Prefill mode only supports seq_len > 0 and seq_len % 32"

            assert xs.shape == (batch, 1, seq_len, self.hidden_size // self.cluster_shape[0])

            cos_gathered, sin_gathered = gather_cos_sin(
                torch.arange(start_pos, start_pos + seq_len), self.cos, self.sin
            )

            cos_gathereds = ttnn.as_tensor(
                cos_gathered,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                # cache_file_name=cache_name(f"cos_gathered_prefill_galaxy_{start_pos}"),
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ReplicateTensorToMesh(self.mesh_device),
            )
            sin_gathereds = ttnn.as_tensor(
                sin_gathered,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                # cache_file_name=cache_name(f"sin_gathered_prefill_galaxy_{start_pos}"),
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ReplicateTensorToMesh(self.mesh_device),
            )

            rot_mats = [cos_gathereds, sin_gathereds]
            if not attn_mask:
                attn_mask = torch.full((seq_len, seq_len), torch.finfo(torch.float32).min)
                attn_mask = torch.triu(attn_mask, diagonal=1)
                attn_mask = attn_mask.expand(1, batch, -1, -1)
                attn_masks = ttnn.as_tensor(
                    attn_mask,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    # cache_file_name=cache_name(f"attn_mask_prefill_{seq_len}"),
                    mesh_mapper=ReplicateTensorToMesh(self.mesh_device),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    device=self.mesh_device,
                )
            else:
                attn_masks = attn_mask

        return (
            xs,
            start_pos,
            rot_mats,
            attn_masks,
        )

    def __call__(
        self,
        xs: List[ttnn.Tensor],
        rot_mats: List[ttnn.Tensor],
        start_pos: int,
        attn_masks: List[ttnn.Tensor],
        user_id: int = 0,
    ) -> ttnn.Tensor:
        if self.model_config["LLM_MODE"] == "decode":
            return self.decode_forward(xs, rot_mats, start_pos, attn_masks)
        elif self.model_config["LLM_MODE"] == "prefill":
            return self.prefill_forward(xs, rot_mats, start_pos, attn_masks, user_id)
        else:
            raise ValueError(f"Unknown llm_mode: {self.model_config['LLM_MODE']}")

    def tt_distributed_rmsnorm(self, inp, epsilon, gamma):
        # Run distributed rmsnorm part 1
        tt_stats = ttnn.rms_norm_pre_all_gather(
            inp, compute_kernel_config=self.LN_COMPUTE_KERNEL_CONFIG, dtype=ttnn.bfloat16
        )

        padded_shape = (1, 1, inp.shape[-2], 32)
        tt_stats = ttnn.reshape(tt_stats, ttnn.Shape(padded_shape, padded_shape))  # TODO: Figure out why we need this
        tt_stats = tt_all_gather(
            tt_stats,
            mesh_device=self.mesh_device,
            dim=3,
            cluster_axis=1,
            num_links=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Run distributed rmsnorm part 2
        tt_out = ttnn.rms_norm_post_all_gather(
            inp, tt_stats, epsilon=epsilon, weight=gamma, compute_kernel_config=self.LN_COMPUTE_KERNEL_CONFIG
        )

        tt_stats.deallocate(True)

        return tt_out

    def decode_forward(
        self,
        xs: List[ttnn.Tensor],
        rot_mats: List[ttnn.Tensor],
        start_pos: int,
        attn_masks: List[ttnn.Tensor],
    ) -> ttnn.Tensor:
        ### Run all layers
        for layer in self.layers:
            xs = layer(xs, rot_mats, start_pos, attn_masks)  # xs is fractured

        xs_interleaved = ttnn.to_memory_config(xs, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        norm_out = self.tt_distributed_rmsnorm(
            xs_interleaved,
            epsilon=self.norm_eps,
            gamma=self.norm_sharded,
        )

        norm_out = ttnn.to_memory_config(norm_out, memory_config=self.LM_HEAD_ACT_MEMCFG)

        ### Each device does an LM head fracture
        lm_head_out = ttnn.matmul(
            norm_out,
            self.lm_head,
            # program_config=(
            #     self.model_config["LLAMA3_LM_HEAD_MM_PROGCFG"]
            #     if self.llama3
            #     else self.model_config["LM_HEAD_MM_PROGCFG"]
            # ),
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.COMPUTE_KERNEL_CONFIG,
        )
        norm_out.deallocate(True)

        lm_head_out = tt_all_reduce(
            lm_head_out,
            mesh_device=self.mesh_device,
            cluster_axis=1,
            dim=0,
            num_links=2,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )

        return lm_head_out

    def prefill_forward(
        self,
        xs: List[ttnn.Tensor],
        rot_mats: List[ttnn.Tensor],
        start_pos: int,
        attn_masks: List[ttnn.Tensor],
        user_id: int,
    ) -> ttnn.Tensor:
        ### Run all layers
        for id, layer in enumerate(self.layers):
            xs = layer(xs, rot_mats, start_pos, attn_masks, user_id)

        norm_out = self.tt_distributed_rmsnorm(
            xs,
            epsilon=self.norm_eps,
            gamma=self.norm_sharded,
        )
        # Slice out last padding_length(32) tokens in LM head to produce next token
        # TODO: Does not work for perplexity, or if we padded input to current sequence length
        seq_len = norm_out.shape[2]
        dmodel = norm_out.shape[3]
        norm_out = ttnn.slice(
            norm_out,
            [0, 0, seq_len - self.model_config["PADDING_LENGTH"], 0],
            [0, 0, seq_len - 1, dmodel - 1],
        )

        ### Each device does an LM head fracture
        lm_head_out = ttnn.matmul(
            norm_out,
            self.lm_head,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.COMPUTE_KERNEL_CONFIG,
            program_config=self.LM_HEAD_PROGCFG,
        )

        lm_head_out = tt_all_reduce(
            lm_head_out,
            mesh_device=self.mesh_device,
            cluster_axis=1,
            dim=0,
            num_links=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        return lm_head_out
