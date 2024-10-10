# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
from typing import List
from tqdm import tqdm
import torch
import ttnn
from ttnn import ShardTensorToMesh, ReplicateTensorToMesh


from models.utility_functions import nearest_32, profiler
from models.demos.t3000.llama2_70b.tt.llama_decoder_optimized import TtLlamaDecoder_optimized
from models.demos.t3000.llama2_70b.tt.llama_embedding import TtLlamaEmbedding
from models.demos.t3000.llama2_70b.tt.llama_common import (
    freqs_to_rotation_matrix,
    get_rotation_mat,
    precompute_freqs,
    gather_cos_sin,
    get_rot_transformation_mat,
)
from models.demos.t3000.falcon40b.tt.model_utils import matmul_2d_config


class TtLlamaModel_optimized:
    def __init__(
        self,
        mesh_device,
        state_dict,
        base_url,
        n_layers,
        model_config,
        configuration,
        cache_path=None,
        read_cache=False,
        paged_attention_config=None,
        vllm=False,
    ):
        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.num_devices = mesh_device.get_num_devices()
        self.model_config = model_config
        self.read_cache = read_cache
        self.vllm = vllm

        self.hidden_size = configuration.dim
        self.n_heads = configuration.n_heads
        self.n_local_heads = self.n_heads // self.num_devices
        self.padded_local_heads = 32
        self.head_dim = self.hidden_size // self.n_heads
        self.max_seq_len = configuration.max_seq_len
        self.vocab_size = configuration.vocab_size
        self.norm_eps = configuration.norm_eps
        self.llama3 = self.vocab_size == 128256
        self.rope_theta = configuration.rope_theta if self.llama3 else 10000.0
        self.use_scaled_rope = getattr(configuration, "use_scaled_rope", False)

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
        transformation_mats = ttnn.to_device(transformation_mats, mesh_device)

        logger.info("Creating Layers")
        self.layers = [
            TtLlamaDecoder_optimized(
                mesh_device,
                state_dict,
                base_url,
                layer_num,
                model_config,
                configuration,
                transformation_mats,
                cache_path=cache_path,
                read_cache=read_cache,
                paged_attention_config=paged_attention_config,
                vllm=vllm,
            )
            for layer_num in tqdm(range(n_layers))
        ]
        logger.info("Done creating layers")

        # Rotary Embedding
        self.cos, self.sin = precompute_freqs(
            self.head_dim, self.max_seq_len * 2, self.rope_theta, self.use_scaled_rope
        )  # for prefill
        self.rot_emb = freqs_to_rotation_matrix(self.cos, self.sin)  # for decode
        # Embedding
        self.tt_embd = TtLlamaEmbedding(
            mesh_device,
            state_dict,
            cache_path,
        )
        self.load_weights()

    def set_model_config(self, model_config):
        self.model_config = model_config
        for layer in self.layers:
            layer.set_model_config(model_config)

    def load_weights(self):
        norm_str = "norm.weight"
        norm_sharded_str = "norm_sharded.weight"
        lm_head_str = "output.weight"

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

        padded_lm_head_ttnn = ttnn.as_tensor(
            padded_lm_head,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ShardTensorToMesh(self.mesh_device, dim=3),
            cache_file_name=self.cache_path / lm_head_str,
        )
        self.lm_head = ttnn.to_device(padded_lm_head_ttnn, self.mesh_device)

        norm_ttnn = ttnn.as_tensor(
            pt_norm_weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ReplicateTensorToMesh(self.mesh_device),
            cache_file_name=self.cache_path / norm_str,
        )
        self.norm = ttnn.to_device(norm_ttnn, self.mesh_device)

        norm_sharded_ttnn = ttnn.as_tensor(
            pt_norm_weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ShardTensorToMesh(self.mesh_device, dim=2),
            cache_file_name=self.cache_path / norm_sharded_str,
        )
        self.norm_sharded = ttnn.to_device(norm_sharded_ttnn, self.mesh_device)

    def validate_input_shape(self, inp_ids):
        assert inp_ids.dim() == 2
        batch, seq_len = inp_ids.shape
        assert (
            batch <= self.model_config["MAX_BATCH_SIZE"]
        ), f"Batch size {batch} exceeds MAX_BATCH_SIZE {self.model_config['MAX_BATCH_SIZE']}"
        assert (
            seq_len <= self.model_config["MAX_CONTEXT_LEN"]
        ), f"Sequence length {seq_len} exceeds MAX_CONTEXT_LEN {self.model_config['MAX_CONTEXT_LEN']}"

    def prepare_inputs(self, inp_ids, start_pos, valid_seq_len=None, mode="decode"):
        """
        Prepare inputs for decode mode. Assume that current token is at
        start_pos, and KV cache has valid data up to start_pos.
        inp_ids: (batch, seq)
        start_pos: int
        valid_seq_len: int, optional for mask padding

        returns:
        xs: [(seq, batch, hidden_dim)] * num_devices
        start_pos: int
        rot_mats: [(1, 1, head_dim, head_dim)] * num_devices  for decode
                  [(1, 1, seq, head_dim), (1, 1, seq, head_dim)] * num_devices  for prefill
        """
        self.validate_input_shape(inp_ids)
        batch, seq_len = inp_ids.shape

        cache_name = lambda name: self.cache_path / (f"{'llama3_' if self.llama3 else ''}{name}")

        if mode == "decode":
            inp_ids = inp_ids.reshape(seq_len, 1, 1, batch)
            # Pad to PADDED_BATCH_SIZE
            inp_ids = torch.nn.functional.pad(inp_ids, (0, self.model_config["PADDED_BATCH_SIZE"] - batch), value=0)
        else:
            inp_ids = inp_ids.reshape(batch, 1, 1, seq_len)

        x = ttnn.as_tensor(
            inp_ids,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ReplicateTensorToMesh(self.mesh_device),
        )

        if mode == "prefill":
            assert seq_len % 32 == 0 and seq_len > 0, "Prefill mode only supports seqlen as a multiple of 32"
            assert batch == 1, "prefill mode only supports batch size 1"

            x = ttnn.to_device(x, self.mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

            xs = self.tt_embd(x)
            assert xs.shape == (batch, 1, seq_len, self.hidden_size // self.num_devices)

            cos_gathered, sin_gathered = gather_cos_sin(
                torch.arange(start_pos, start_pos + seq_len), self.cos, self.sin
            )
            assert cos_gathered.size() == (1, 1, seq_len, self.head_dim)
            assert sin_gathered.size() == (1, 1, seq_len, self.head_dim)

            cos_gathereds = ttnn.as_tensor(
                cos_gathered,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                cache_file_name=cache_name(f"cos_gathered_prefill_{seq_len}"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                device=self.mesh_device,
                mesh_mapper=ReplicateTensorToMesh(self.mesh_device),
            )
            sin_gathereds = ttnn.as_tensor(
                sin_gathered,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                cache_file_name=cache_name(f"sin_gathered_prefill_{seq_len}"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                device=self.mesh_device,
                mesh_mapper=ReplicateTensorToMesh(self.mesh_device),
            )
            cos_gathereds = ttnn.to_device(cos_gathereds, self.mesh_device)
            sin_gathereds = ttnn.to_device(sin_gathereds, self.mesh_device)
            rot_mats = [cos_gathereds, sin_gathereds]

            cache_idxs_tt = None  # unused in prefill mode

        elif mode == "decode":
            assert seq_len == 1, "Decode mode only supports seq_len=1"
            xs = x
            # User can provide a single start pos which applies to the whole batch or a list of start positions
            if isinstance(start_pos, int):
                cache_idxs = torch.tensor([start_pos for _ in range(batch)], dtype=torch.int64)
            else:
                cache_idxs = start_pos.to(dtype=torch.int64)

            rot_cache_idxs = torch.maximum(
                cache_idxs, torch.tensor(0, dtype=torch.int64)
            )  # Ensure position indices are non-negative
            rot_mat = get_rotation_mat(self.rot_emb, rot_cache_idxs, seq_len, batch=batch)
            assert rot_mat.size() == (1, batch, self.head_dim, self.head_dim)

            rot_mats = ttnn.as_tensor(
                rot_mat,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ReplicateTensorToMesh(self.mesh_device),
            )

            cache_idxs_tt = ttnn.as_tensor(
                cache_idxs,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ReplicateTensorToMesh(self.mesh_device),
            )

        return (xs, start_pos, rot_mats, cache_idxs_tt)

    def __call__(
        self,
        xs: List[ttnn.Tensor],
        rot_mats: List[ttnn.Tensor],
        start_pos: int,
        user_id: int = 0,
        cache_idxs=None,
        last_token_idx=None,
        page_table=None,
        kv_cache=None,
        mode="decode",
    ) -> ttnn.Tensor:
        if self.vllm:
            assert page_table is not None
            assert kv_cache is not None
        if mode == "prefill":
            return self.prefill_forward(
                xs,
                rot_mats,
                start_pos,
                user_id,
                last_token_idx=last_token_idx,
                page_table=page_table,
                kv_cache=kv_cache,
            )
        elif mode == "decode":
            return self.decode_forward(xs, rot_mats, start_pos, cache_idxs, page_table=page_table, kv_cache=kv_cache)
        else:
            raise ValueError(f"Unknown llm_mode: {mode}")

    def decode_forward(
        self,
        xs: List[ttnn.Tensor],
        rot_mats: List[ttnn.Tensor],
        start_pos: int,
        cache_idxs,
        page_table=None,
        kv_cache=None,
    ) -> ttnn.Tensor:
        ### Run all layers
        for layer in self.layers:
            xs = layer(
                xs, rot_mats, start_pos, cache_idxs=cache_idxs, page_table=page_table, kv_cache=kv_cache, mode="decode"
            )  # xs is sharded

        xs = ttnn.all_gather(
            xs,
            dim=3,
            num_links=self.model_config["ALL_GATHER_NUM_LINKS"],
            memory_config=self.model_config["FINAL_ALL_GATHER_OUTPUT_MEMCFG"],
        )

        # In-place RMSNorm
        norm_out_replicated = ttnn.rms_norm(
            xs,
            epsilon=self.norm_eps,
            weight=self.norm,
            program_config=self.model_config["LN_F_PROGCFG"],
            memory_config=self.model_config["FINAL_ALL_GATHER_OUTPUT_MEMCFG"],
            compute_kernel_config=self.model_config["LN_COMPUTE_KERNEL_CONFIG"],
        )

        ### Each device does an LM head fracture
        lm_head_out = ttnn.matmul(
            norm_out_replicated,
            self.lm_head,
            program_config=(
                self.model_config["LLAMA3_LM_HEAD_MM_PROGCFG"]
                if self.llama3
                else self.model_config["LM_HEAD_MM_PROGCFG"]
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
        )
        norm_out_replicated.deallocate(True)

        return lm_head_out

    def tt_distributed_rmsnorm(self, inp, epsilon, gamma):
        # Run distributed rmsnorm part 1
        tt_stats = ttnn.rms_norm_pre_all_gather(
            inp, compute_kernel_config=self.model_config["LN_COMPUTE_KERNEL_CONFIG"], dtype=ttnn.bfloat16
        )

        # AllGather stats
        tt_stats = ttnn.all_gather(
            tt_stats,
            dim=3,
            num_links=self.model_config["ALL_GATHER_NUM_LINKS"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Run distributed rmsnorm part 2
        tt_out = ttnn.rms_norm_post_all_gather(
            inp,
            tt_stats,
            epsilon=epsilon,
            weight=gamma,
            compute_kernel_config=self.model_config["LN_COMPUTE_KERNEL_CONFIG"],
        )

        tt_stats.deallocate(True)

        return tt_out

    def prefill_forward(
        self,
        xs: List[ttnn.Tensor],
        rot_mats: List[ttnn.Tensor],
        start_pos: int,
        user_id: int = 0,
        last_token_idx=None,
        page_table=None,
        kv_cache=None,
    ) -> ttnn.Tensor:
        ### Run all layers
        for layer in self.layers:
            xs = layer(
                xs, rot_mats, start_pos, user_id, page_table=page_table, kv_cache=kv_cache, mode="prefill"
            )  # xs is sharded

        # Distributed rmsnorm
        norm_out = self.tt_distributed_rmsnorm(xs, self.norm_eps, self.norm_sharded)
        norm_out_replicated = ttnn.all_gather(
            norm_out,
            dim=3,
            num_links=self.model_config["ALL_GATHER_NUM_LINKS"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Deallocate original input to rmsnorm
        xs.deallocate(True)

        ### Each device does an LM head fracture

        _, _, seq_len, dmodel = norm_out_replicated.shape

        if last_token_idx:
            last_token_tile = last_token_idx // 32
            norm_out_replicated = ttnn.slice(
                norm_out_replicated,
                (0, 0, last_token_tile * 32, 0),
                (1, 1, (last_token_tile + 1) * 32, dmodel),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            pc_lm_head = (
                self.model_config["PREFILL_LLAMA3_LM_HEAD_MM_PROGCFG"]
                if self.llama3
                else self.model_config["PREFILL_LM_HEAD_MM_PROGCFG"]
            )
        else:
            max_mm_seq_len = self.model_config["MAX_MM_SEQ_LEN"]
            if seq_len >= max_mm_seq_len:
                if seq_len % max_mm_seq_len != 0:
                    raise ValueError(f"Sequence length {seq_len} is not divisible by {max_mm_seq_len}")
                batch_dim = seq_len // max_mm_seq_len  # Find the division factor
                norm_out_replicated = ttnn.reshape(norm_out_replicated, (1, batch_dim, seq_len // batch_dim, -1))
                pc_lm_head = (
                    self.model_config["PREFILL_LLAMA3_LM_HEAD_MM_PROGCFG"]
                    if self.llama3
                    else self.model_config["PREFILL_LM_HEAD_MM_PROGCFG"]
                )
            elif seq_len == 128:
                pc_lm_head = (
                    self.model_config["PREFILL_LLAMA3_LM_HEAD_MM_PROGCFG_128"]
                    if self.llama3
                    else self.model_config["PREFILL_LM_HEAD_MM_PROGCFG_128"]
                )
            else:
                pc_lm_head = matmul_2d_config(
                    m=norm_out_replicated.shape[2],
                    k=norm_out_replicated.shape[3],
                    n=self.lm_head.shape[3],
                    overwrite_per_core_k=1,
                    grid=ttnn.CoreGrid(y=min(8, norm_out_replicated.shape[2] // 32), x=8),
                    is_fp32_accumulate=False,
                    overwrite_subblock_h=1,
                    overwrite_subblock_w=1,
                )

        lm_head_out = ttnn.linear(
            norm_out_replicated,
            self.lm_head,
            # TODO: increase precision?
            compute_kernel_config=self.model_config["COMPUTE_KERNEL_FP16_ACC_CONFIG"],
            core_grid=ttnn.CoreGrid(y=8, x=8) if not pc_lm_head else None,
            dtype=ttnn.bfloat16,
            program_config=pc_lm_head,
        )

        norm_out_replicated.deallocate(True)

        if not last_token_idx and seq_len >= max_mm_seq_len:
            # Prefill Reshape fix (reverse)
            lm_head_out = ttnn.reshape(lm_head_out, (1, 1, seq_len, -1))

        return lm_head_out
