# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import math
import json
import torch
import numpy as np
import tempfile
from pathlib import Path
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
import ttnn
from loguru import logger
import pytest
from models.utility_functions import skip_for_blackhole

from models.demos.deepseek_v3.tt.rope import RotarySetup
from models.demos.deepseek_v3.tt.rms_norm import RMSNorm
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.model import RMSNorm as ReferenceRMSNorm
from models.demos.deepseek_v3.reference.deepseek.rope_helpers import (
    precompute_freqs_cis,
    apply_rotary_emb,
)
from models.utility_functions import nearest_y
from transformers import AutoConfig
from types import SimpleNamespace

from models.demos.deepseek_v3.utils.run_config import create_run_config

TP = 8
DP = 4


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class DecodeModelConfig:
    def __init__(self, hf_config):
        self.args = hf_config
        self.args.qk_head_dim = self.args.qk_nope_head_dim + self.args.qk_rope_head_dim

        self.grid_size = (8, 8)
        self.bsz = 128
        self.configs = {}

        #################
        ### MLA Configs
        #################

        # wq_a
        self.configs["WQA_IN0_SHAPE"] = (1, 1, self.bsz // DP, self.args.hidden_size // TP)
        self.configs["WQA_IN1_SHAPE"] = (1, 1, self.args.hidden_size // TP, self.args.q_lora_rank)
        self.configs["WQA_IN0_DTYPE"] = ttnn.bfloat8_b
        self.configs["WQA_IN1_DTYPE"] = ttnn.bfloat4_b
        self.configs["WQA_PROGRAM_CFG"] = None
        self.configs["WQA_IN0_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.configs["WQA_IN1_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.configs["WQA_OUT_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG

        # wq_b
        self.configs["WQB_IN0_SHAPE"] = (1, 1, self.bsz // DP, self.args.q_lora_rank)
        self.configs["WQB_IN1_SHAPE"] = (
            1,
            1,
            self.args.q_lora_rank,
            (self.args.num_attention_heads * self.args.qk_head_dim) // TP,
        )
        self.configs["WQB_IN0_DTYPE"] = ttnn.bfloat8_b
        self.configs["WQB_IN1_DTYPE"] = ttnn.bfloat4_b
        self.configs["WQB_PROGRAM_CFG"] = None
        self.configs["WQB_IN0_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.configs["WQB_IN1_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.configs["WQB_OUT_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG

        # wkv_a
        self.configs["WKV_A_IN0_SHAPE"] = (1, 1, self.bsz // DP, self.args.hidden_size // TP)
        self.configs["WKV_A_IN1_SHAPE"] = (
            1,
            1,
            self.args.hidden_size // TP,
            self.args.kv_lora_rank + self.args.qk_rope_head_dim,
        )
        self.configs["WKV_A_IN0_DTYPE"] = ttnn.bfloat8_b
        self.configs["WKV_A_IN1_DTYPE"] = ttnn.bfloat4_b
        self.configs["WKV_A_PROGRAM_CFG"] = None
        self.configs["WKV_A_IN0_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.configs["WKV_A_IN1_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.configs["WKV_A_OUT_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG

        # wkv_b1
        self.configs["WKV_B1_IN0_SHAPE"] = (
            1,
            self.args.num_attention_heads // TP,
            self.bsz // DP,
            self.args.qk_nope_head_dim,
        )
        self.configs["WKV_B1_IN1_SHAPE"] = (
            1,
            self.args.num_attention_heads // TP,
            self.args.qk_nope_head_dim,
            self.args.kv_lora_rank,
        )
        self.configs["WKV_B1_IN0_DTYPE"] = ttnn.bfloat8_b
        self.configs["WKV_B1_IN1_DTYPE"] = ttnn.bfloat4_b
        self.configs["WKV_B1_PROGRAM_CFG"] = None
        self.configs["WKV_B1_IN0_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.configs["WKV_B1_IN1_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.configs["WKV_B1_OUT_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG

        # wkv_b2
        self.configs["WKV_B2_IN0_SHAPE"] = (
            1,
            self.args.num_attention_heads // TP,
            self.bsz // DP,
            self.args.kv_lora_rank,
        )
        self.configs["WKV_B2_IN1_SHAPE"] = (
            1,
            self.args.num_attention_heads // TP,
            self.args.kv_lora_rank,
            self.args.v_head_dim,
        )
        self.configs["WKV_B2_IN0_DTYPE"] = ttnn.bfloat8_b
        self.configs["WKV_B2_IN1_DTYPE"] = ttnn.bfloat4_b
        self.configs["WKV_B2_PROGRAM_CFG"] = None
        self.configs["WKV_B2_IN0_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.configs["WKV_B2_IN1_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.configs["WKV_B2_OUT_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG

        # wo
        self.configs["WO_IN0_SHAPE"] = (1, self.bsz // DP, self.args.num_attention_heads * self.args.v_head_dim)
        self.configs["WO_IN1_SHAPE"] = (
            1,
            1,
            self.args.num_attention_heads * self.args.v_head_dim,
            self.args.hidden_size // TP,
        )
        self.configs["WO_IN0_DTYPE"] = ttnn.bfloat8_b
        self.configs["WO_IN1_DTYPE"] = ttnn.bfloat4_b
        self.configs["WO_PROGRAM_CFG"] = None
        self.configs["WO_IN0_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.configs["WO_IN1_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.configs["WO_OUT_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG

        # q_rope
        self.configs["QROPE_SHAPE"] = (
            1,
            self.bsz // DP,
            self.args.num_attention_heads // TP,
            self.args.qk_rope_head_dim,
        )
        self.configs["QROPE_DTYPE"] = ttnn.bfloat16

        q_rope_shard_height = nearest_y(self.configs["QROPE_SHAPE"][2], ttnn.TILE_SIZE)
        q_rope_shard_width = self.configs["QROPE_SHAPE"][3]
        q_rope_num_cores = self.configs["QROPE_SHAPE"][1]
        q_rope_core_grid = ttnn.num_cores_to_corerangeset(q_rope_num_cores, self.grid_size, row_wise=True)
        self.configs["QROPE_MEM_CFG"] = ttnn.create_sharded_memory_config(
            shape=(q_rope_shard_height, q_rope_shard_width),
            core_grid=q_rope_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )

        # k_rope
        self.configs["KROPE_SHAPE"] = (1, self.bsz // DP // TP, 1, self.args.qk_rope_head_dim)
        self.configs["KROPE_DTYPE"] = ttnn.bfloat16
        k_rope_shard_height = nearest_y(self.configs["KROPE_SHAPE"][2], ttnn.TILE_SIZE)
        k_rope_shard_width = self.configs["KROPE_SHAPE"][3]
        k_rope_num_cores = self.configs["KROPE_SHAPE"][1]
        k_rope_core_grid = ttnn.num_cores_to_corerangeset(k_rope_num_cores, self.grid_size, row_wise=True)
        self.configs["KROPE_MEM_CFG"] = ttnn.create_sharded_memory_config(
            shape=(k_rope_shard_height, k_rope_shard_width),
            core_grid=k_rope_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )

        # KVPE Cache
        self.configs["KVPE_SHAPE"] = (1, self.bsz // DP // TP, 1, self.args.kv_lora_rank + self.args.qk_rope_head_dim)
        self.configs["KVPE_DTYPE"] = ttnn.bfloat16
        kvpe_shard_height = nearest_y(self.configs["KVPE_SHAPE"][2], ttnn.TILE_SIZE)
        kvpe_shard_width = self.configs["KVPE_SHAPE"][3]
        kvpe_num_cores = self.configs["KVPE_SHAPE"][1]
        kvpe_core_grid = ttnn.num_cores_to_corerangeset(kvpe_num_cores, self.grid_size, row_wise=True)
        self.configs["KVPE_MEM_CFG"] = ttnn.create_sharded_memory_config(
            shape=(kvpe_shard_height, kvpe_shard_width),
            core_grid=kvpe_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )

        self.configs["KVPE_CACHE_DTYPE"] = ttnn.bfloat4_b

        # q_norm
        self.configs["QNORM_SHAPE"] = (1, 1, self.bsz // DP, self.args.q_lora_rank)
        self.configs["QNORM_DTYPE"] = ttnn.bfloat16
        self.configs["QNORM_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.configs["QNORM_CATEGORY"] = "q_norm"

        # k_norm
        self.configs["KNORM_SHAPE"] = (1, 1, self.bsz // DP // TP, self.args.kv_lora_rank + self.args.qk_rope_head_dim)
        self.configs["KNORM_DTYPE"] = ttnn.bfloat16
        self.configs["KNORM_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.configs["KNORM_CATEGORY"] = "k_norm"
        # # TODO: Debug, gives bad PCC
        # knorm_num_cores = min(np.prod(self.grid_size), math.ceil(self.configs["KNORM_SHAPE"][3] / ttnn.TILE_SIZE))
        # knorm_core_grid = ttnn.num_cores_to_corerangeset(knorm_num_cores, self.grid_size, row_wise=True)
        # knorm_shard_height = nearest_y(self.configs["KNORM_SHAPE"][2], ttnn.TILE_SIZE) * np.prod(self.configs["KNORM_SHAPE"][:2])
        # knorm_shard_width = nearest_y(self.configs["KNORM_SHAPE"][3] // knorm_num_cores, ttnn.TILE_SIZE)
        # self.configs["KNORM_MEM_CFG"] = ttnn.create_sharded_memory_config(
        #     shape=(knorm_shard_height, knorm_shard_width),
        #     core_grid=knorm_core_grid,
        #     strategy=ttnn.ShardStrategy.WIDTH,
        #     use_height_and_width_as_shard_shape=True,
        # )


class PrefillModelConfig:
    def __init__(self, hf_config):
        self.args = hf_config
        self.args.qk_head_dim = self.args.qk_nope_head_dim + self.args.qk_rope_head_dim

        self.grid_size = (8, 8)
        self.max_batch_size_per_device = 4
        self.configs = {}

        #################
        ### MLA Configs
        #################

        # wq_a
        self.configs["WQA_IN0_SHAPE"] = lambda seq_len: (1, 1, seq_len, self.args.hidden_size // TP)
        self.configs["WQA_IN1_SHAPE"] = (1, 1, self.args.hidden_size // TP, self.args.q_lora_rank)
        self.configs["WQA_IN0_DTYPE"] = ttnn.bfloat8_b
        self.configs["WQA_IN1_DTYPE"] = ttnn.bfloat4_b
        self.configs["WQA_PROGRAM_CFG"] = None
        self.configs["WQA_IN0_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.configs["WQA_IN1_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.configs["WQA_OUT_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG

        # wq_b
        self.configs["WQB_IN0_SHAPE"] = lambda seq_len: (1, 1, seq_len, self.args.q_lora_rank)
        self.configs["WQB_IN1_SHAPE"] = (
            1,
            1,
            self.args.q_lora_rank,
            (self.args.num_attention_heads * self.args.qk_head_dim) // TP,
        )
        self.configs["WQB_IN0_DTYPE"] = ttnn.bfloat8_b
        self.configs["WQB_IN1_DTYPE"] = ttnn.bfloat4_b
        self.configs["WQB_PROGRAM_CFG"] = None
        self.configs["WQB_IN0_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.configs["WQB_IN1_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.configs["WQB_OUT_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG

        # wkv_a
        self.configs["WKV_A_IN0_SHAPE"] = lambda seq_len: (1, 1, seq_len, self.args.hidden_size // TP)
        self.configs["WKV_A_IN1_SHAPE"] = (
            1,
            1,
            self.args.hidden_size // TP,
            self.args.kv_lora_rank + self.args.qk_rope_head_dim,
        )
        self.configs["WKV_A_IN0_DTYPE"] = ttnn.bfloat8_b
        self.configs["WKV_A_IN1_DTYPE"] = ttnn.bfloat4_b
        self.configs["WKV_A_PROGRAM_CFG"] = None
        self.configs["WKV_A_IN0_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.configs["WKV_A_IN1_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.configs["WKV_A_OUT_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG

        # wkv_b1
        self.configs["WKV_B1_IN0_SHAPE"] = lambda seq_len: (
            1,
            self.args.num_attention_heads // TP,
            seq_len,
            self.args.qk_nope_head_dim,
        )
        self.configs["WKV_B1_IN1_SHAPE"] = (
            1,
            self.args.num_attention_heads // TP,
            self.args.qk_nope_head_dim,
            self.args.kv_lora_rank,
        )
        self.configs["WKV_B1_IN0_DTYPE"] = ttnn.bfloat8_b
        self.configs["WKV_B1_IN1_DTYPE"] = ttnn.bfloat4_b
        self.configs["WKV_B1_PROGRAM_CFG"] = None
        self.configs["WKV_B1_IN0_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.configs["WKV_B1_IN1_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.configs["WKV_B1_OUT_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG

        # wkv_b2
        self.configs["WKV_B2_IN0_SHAPE"] = lambda seq_len: (
            1,
            self.args.num_attention_heads // TP,
            seq_len,
            self.args.kv_lora_rank,
        )
        self.configs["WKV_B2_IN1_SHAPE"] = (
            1,
            self.args.num_attention_heads // TP,
            self.args.kv_lora_rank,
            self.args.v_head_dim,
        )
        self.configs["WKV_B2_IN0_DTYPE"] = ttnn.bfloat8_b
        self.configs["WKV_B2_IN1_DTYPE"] = ttnn.bfloat4_b
        self.configs["WKV_B2_PROGRAM_CFG"] = None
        self.configs["WKV_B2_IN0_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.configs["WKV_B2_IN1_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.configs["WKV_B2_OUT_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG

        # wo
        self.configs["WO_IN0_SHAPE"] = lambda seq_len: (
            1,
            seq_len,
            self.args.num_attention_heads * self.args.v_head_dim,
        )
        self.configs["WO_IN1_SHAPE"] = (
            1,
            1,
            self.args.num_attention_heads * self.args.v_head_dim,
            self.args.hidden_size // TP,
        )
        self.configs["WO_IN0_DTYPE"] = ttnn.bfloat8_b
        self.configs["WO_IN1_DTYPE"] = ttnn.bfloat4_b
        self.configs["WO_PROGRAM_CFG"] = None
        self.configs["WO_IN0_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.configs["WO_IN1_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.configs["WO_OUT_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG

        # q_rope
        self.configs["QROPE_SHAPE"] = lambda seq_len: (
            1,
            self.args.num_attention_heads // TP,
            seq_len,
            self.args.qk_rope_head_dim,
        )
        self.configs["QROPE_DTYPE"] = ttnn.bfloat16
        self.configs["QROPE_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG

        # k_rope
        self.configs["KROPE_SHAPE"] = lambda seq_len: (1, 1, seq_len, self.args.qk_rope_head_dim)
        self.configs["KROPE_DTYPE"] = ttnn.bfloat16
        self.configs["KROPE_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG

        # KVPE Cache
        self.configs["KVPE_SHAPE"] = lambda seq_len: (
            1,
            1,
            seq_len,
            self.args.kv_lora_rank + self.args.qk_rope_head_dim,
        )
        self.configs["KVPE_DTYPE"] = ttnn.bfloat8_b
        self.configs["KVPE_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.configs["KVPE_CACHE_DTYPE"] = ttnn.bfloat8_b

        # q_norm
        self.configs["QNORM_SHAPE"] = lambda seq_len: (1, 1, seq_len, self.args.q_lora_rank)
        self.configs["QNORM_DTYPE"] = ttnn.bfloat16
        self.configs["QNORM_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.configs["QNORM_CATEGORY"] = "q_norm"

        # k_norm
        self.configs["KNORM_SHAPE"] = lambda seq_len: (
            1,
            1,
            seq_len,
            self.args.kv_lora_rank + self.args.qk_rope_head_dim,
        )
        self.configs["KNORM_DTYPE"] = ttnn.bfloat16
        self.configs["KNORM_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.configs["KNORM_CATEGORY"] = "k_norm"


hugging_face_config = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-R1-0528", trust_remote_code=True)
hugging_face_config.max_seq_len = 16 * 1024  # Set max sequence length for testing

decode_cfg = DecodeModelConfig(hugging_face_config)
prefill_cfg = PrefillModelConfig(hugging_face_config)


#################
### Helper Funcs
#################


def run_matmul_impl(
    device,
    shapes,
    dtypes,
    program_config,
    memory_configs,
    seq_len=None,
):
    layout = ttnn.TILE_LAYOUT

    if seq_len is not None:  # Prefill
        # Check that the first shapes is a function
        assert callable(shapes[0]), "Shapes must be callable for prefill tests with variable sequence length."
        in0_shape = shapes[0](seq_len)
        in1_shape = shapes[1]
    else:  # Decode
        in0_shape, in1_shape = shapes
    in0_dtype, in1_dtype = dtypes
    in0_mem_config, in1_mem_config, out_mem_config = memory_configs

    # Log configs
    logger.debug("Running matmul with the following configurations:")
    logger.debug(f"Input 0 Shape: {in0_shape}, Dtype: {in0_dtype}, Memory Config: {in0_mem_config}")
    logger.debug(f"Input 1 Shape: {in1_shape}, Dtype: {in1_dtype}, Memory Config: {in1_mem_config}")
    logger.debug(f"Output Memory Config: {out_mem_config}")
    logger.debug(f"Program Config: {program_config}")

    #################
    ### Torch
    #################
    in0 = torch.randn(in0_shape).float()
    in1 = torch.randn(in1_shape).float()
    out_torch = in0 @ in1

    #################
    ### TT-NN
    #################
    tt_in0 = ttnn.from_torch(
        in0,
        device=device,
        dtype=in0_dtype,
        memory_config=in0_mem_config,
        layout=layout,
    )

    tt_in1 = ttnn.from_torch(
        in1,
        device=device,
        dtype=in1_dtype,
        memory_config=in1_mem_config,
        layout=layout,
    )

    tt_out = ttnn.matmul(
        tt_in0,
        tt_in1,
        memory_config=out_mem_config,
        program_config=program_config,
    )

    tt_out_torch = ttnn.to_torch(tt_out)

    #################
    ### Validation
    #################
    pcc_threshold = 0.99

    out_pass, out_pcc = comp_pcc(tt_out_torch, out_torch, pcc_threshold)
    logger.info(f"Output PCC: {out_pcc}")

    assert out_pass, f"Output mismatch: PCC {out_pcc} < {pcc_threshold}"


def run_rope_impl(
    device,
    shape,
    dtype,
    mem_config,
    seq_len=None,
):
    layout = ttnn.TILE_LAYOUT

    if seq_len is not None:  # Prefill
        # Check that the shape is a function
        assert callable(shape), "Shape must be callable for prefill tests with variable sequence length."

        # TT Shape: [bsz, nheads, seq_len, head_dim]
        input_shape = shape(seq_len)
        bsz = input_shape[0]
        mode = "prefill"
    else:  # Decode
        # TT Shape: [seq_len, bsz, nheads, head_dim]
        input_shape = shape
        bsz = input_shape[1]
        mode = "decode"

    logger.debug("Running rope with the following configurations:")
    logger.debug(f"Shape: {input_shape}, Dtype: {dtype}, Memory Config: {mem_config}")

    assert (
        input_shape[-1] == decode_cfg.args.qk_rope_head_dim
    ), "Input shape's last dimension must match qk_rope_head_dim."

    #################
    ### Torch
    #################
    position_ids = torch.randint(0, decode_cfg.args.max_seq_len, (bsz,))
    input_torch = torch.randn(input_shape).float()

    # Args expected by DeepSeek impl RoPE
    rope_args = SimpleNamespace(
        qk_rope_head_dim=decode_cfg.args.qk_rope_head_dim,
        max_seq_len=decode_cfg.args.max_seq_len,
        beta_fast=decode_cfg.args.rope_scaling["beta_fast"],
        beta_slow=decode_cfg.args.rope_scaling["beta_slow"],
        rope_theta=decode_cfg.args.rope_theta,
        rope_factor=decode_cfg.args.rope_scaling["factor"],
        original_seq_len=decode_cfg.args.rope_scaling["original_max_position_embeddings"],
    )
    freqs_cis = precompute_freqs_cis(rope_args)

    if mode == "prefill":
        # For prefill, we use the first seq_len positions
        freqs_cis = freqs_cis[:seq_len, :]
    else:
        # For decode, we use the position_ids to index into freqs_cis
        freqs_cis = freqs_cis[position_ids, :]

    out_torch = apply_rotary_emb(
        input_torch if mode == "decode" else input_torch.transpose(1, 2), freqs_cis  # Heads is flipped for prefill
    )

    if mode == "prefill":
        # For prefill, we need to transpose the output to match TT-NN's expected shape
        out_torch = out_torch.transpose(1, 2)

    #################
    ### TT-NN
    #################
    rope_setup = RotarySetup(
        device=device,
        batch_size=bsz,
        hf_config=decode_cfg.args,
    )

    if mode == "prefill":
        tt_cos, tt_sin, tt_trans_mat = rope_setup.get_rot_mats_table(seq_len)
    else:
        tt_cos, tt_sin, tt_trans_mat = rope_setup.get_rot_mats(position_ids)

    tt_input = ttnn.from_torch(
        input_torch,
        device=device,
        dtype=dtype,
        memory_config=mem_config,
        layout=layout,
    )

    tt_out = ttnn.experimental.rotary_embedding_llama(
        tt_input,
        tt_cos,
        tt_sin,
        tt_trans_mat,
        is_decode_mode=(mode == "decode"),
    )

    tt_out_torch = ttnn.to_torch(tt_out)

    #################
    ### Validation
    #################
    pcc_threshold = 0.99

    out_pass, out_pcc = comp_pcc(tt_out_torch, out_torch, pcc_threshold)
    logger.info(f"Output PCC: {out_pcc}")

    assert out_pass, f"Output mismatch: PCC {out_pcc} < {pcc_threshold}"


def run_update_cache_impl(
    device,
    shape,
    dtype,
    mem_config,
    cache_dtype,
):
    layout = ttnn.TILE_LAYOUT
    max_seq_len = decode_cfg.args.max_seq_len

    logger.debug("Running update cache with the following configurations:")
    logger.debug(f"Shape: {shape}, Dtype: {dtype}, Memory Config: {mem_config}")
    logger.debug(f"Max Seq Len: {max_seq_len}")

    _, bsz, nkv, head_dim = shape

    #################
    ### Torch
    #################
    cache_torch = torch.randn((bsz, nkv, max_seq_len, head_dim)).float()
    input_torch = torch.randn(shape).float()
    current_pos = torch.randint(0, max_seq_len, (bsz,))

    #################
    ### TT-NN
    #################
    tt_cache = ttnn.from_torch(
        cache_torch,
        device=device,
        dtype=cache_dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=layout,
    )

    tt_input = ttnn.from_torch(
        input_torch,
        device=device,
        dtype=dtype,
        memory_config=mem_config,
        layout=layout,
    )

    tt_current_pos = ttnn.from_torch(
        current_pos,
        device=device,
        dtype=ttnn.int32,
    )

    ttnn.experimental.paged_update_cache(
        tt_cache,
        tt_input,
        update_idxs_tensor=tt_current_pos,
    )
    tt_cache_torch = ttnn.to_torch(tt_cache)

    #################
    ### Validation
    #################
    for b in range(bsz):
        inp = input_torch[:, b, ...].unsqueeze(1)  # [seq_len, b, nkv, head_dim]
        inp = inp.permute(1, 2, 0, 3)  # [b, nkv, seq_len, head_dim]

        pos = current_pos[b].item()

        cache_torch[b, :, pos : pos + 1, :] = inp

    pcc_threshold = 0.9999
    if cache_dtype == ttnn.bfloat4_b:
        pcc_threshold = 0.99

    out_pass, out_pcc = comp_pcc(tt_cache_torch, cache_torch, pcc_threshold)
    logger.info(f"Output PCC: {out_pcc}")

    assert out_pass, f"Output mismatch: PCC {out_pcc} < {pcc_threshold}"


def run_fill_cache_impl(
    device,
    shape,
    dtype,
    mem_config,
    cache_dtype,
    seq_len,
):
    layout = ttnn.TILE_LAYOUT
    max_seq_len = prefill_cfg.args.max_seq_len

    assert callable(shape), "Shape must be callable for prefill tests with variable sequence length."
    input_shape = shape(seq_len)
    _, nkv, seq_len, head_dim = input_shape

    logger.debug("Running update cache with the following configurations:")
    logger.debug(f"Shape: {input_shape}, Dtype: {dtype}, Memory Config: {mem_config}")
    logger.debug(f"Max Seq Len: {max_seq_len}")

    #################
    ### Torch
    #################
    cache_torch = torch.randn((prefill_cfg.max_batch_size_per_device, nkv, max_seq_len, head_dim)).float()
    input_torch = torch.randn(input_shape).float()
    batch_idx = torch.randint(0, prefill_cfg.max_batch_size_per_device, (1,)).item()  # Randomly select a batch index
    logger.debug(f"Selected batch index: {batch_idx}")

    cache_torch[batch_idx, :, :seq_len, :] = input_torch

    #################
    ### TT-NN
    #################
    tt_cache = ttnn.from_torch(
        cache_torch,
        device=device,
        dtype=cache_dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=layout,
    )

    tt_input = ttnn.from_torch(
        input_torch,
        device=device,
        dtype=dtype,
        memory_config=mem_config,
        layout=layout,
    )

    ttnn.fill_cache(
        tt_cache,
        tt_input,
        batch_idx=batch_idx,
    )
    tt_cache_torch = ttnn.to_torch(tt_cache)

    #################
    ### Validation
    #################
    pcc_threshold = 0.9999
    if cache_dtype == ttnn.bfloat4_b:
        pcc_threshold = 0.99

    out_pass, out_pcc = comp_pcc(tt_cache_torch, cache_torch, pcc_threshold)
    logger.info(f"Output PCC: {out_pcc}")

    assert out_pass, f"Output mismatch: PCC {out_pcc} < {pcc_threshold}"


def run_rmsnorm_impl(
    device,
    shape,
    dtype,
    mem_config,
    norm_category,
    temp_dir,
    seq_len=None,
):
    layout = ttnn.TILE_LAYOUT
    hf_config = hugging_face_config

    if seq_len is not None:  # Prefill
        # Check that the shape is a function
        assert callable(shape), "Shape must be callable for prefill tests with variable sequence length."
        input_shape = shape(seq_len)
        mode = "prefill"
    else:  # Decode
        input_shape = shape
        mode = "decode"

    logger.debug("Running RMSNorm with the following configurations:")
    logger.debug(f"Shape: {input_shape}, Dtype: {dtype}, Memory Config: {mem_config}")

    head_dim = input_shape[-1]

    #################
    ### Torch
    #################
    input_torch = torch.randn(input_shape).float()
    rms_norm = ReferenceRMSNorm(head_dim, eps=hf_config.rms_norm_eps)
    out_torch = rms_norm(input_torch)

    #################
    ### TT-NN
    #################
    tt_input = ttnn.from_torch(
        input_torch,
        device=device,
        dtype=dtype,
        memory_config=mem_config,
        layout=layout,
    )

    # Setup: Convert weights and get weight_config
    state_dict = {"weight": rms_norm.weight.unsqueeze(0)}
    weight_config = RMSNorm.convert_weights(hf_config, state_dict, temp_dir, device, norm_category=norm_category)

    # Generate appropriate config
    if mode == "prefill":
        model_config = RMSNorm.prefill_model_config(hf_config, device, norm_category=norm_category)
    else:
        model_config = RMSNorm.decode_model_config(hf_config, device, norm_category=norm_category)

    model_state = RMSNorm.create_state(hf_config, mesh_device=device)

    # Create RunConfig using both weight_config and model_config
    run_config = create_run_config(model_config, weight_config, model_state)

    if mode == "prefill":
        tt_out = RMSNorm.forward_prefill(tt_input, run_config)
    else:
        tt_out = RMSNorm.forward_decode(tt_input, run_config)
    tt_out_torch = ttnn.to_torch(tt_out)

    #################
    ### Validation
    #################
    pcc_threshold = 0.9999

    out_pass, out_pcc = comp_pcc(tt_out_torch, out_torch, pcc_threshold)
    logger.info(f"Output PCC: {out_pcc}")

    assert out_pass, f"Output mismatch: PCC {out_pcc} < {pcc_threshold}"


#################
### Tests
#################
@pytest.mark.parametrize(
    "shapes, dtypes, program_config, memory_configs",
    [
        (  # wq_a
            [decode_cfg.configs["WQA_IN0_SHAPE"], decode_cfg.configs["WQA_IN1_SHAPE"]],
            [decode_cfg.configs["WQA_IN0_DTYPE"], decode_cfg.configs["WQA_IN1_DTYPE"]],
            decode_cfg.configs["WQA_PROGRAM_CFG"],
            [
                decode_cfg.configs["WQA_IN0_MEM_CFG"],
                decode_cfg.configs["WQA_IN1_MEM_CFG"],
                decode_cfg.configs["WQA_OUT_MEM_CFG"],
            ],
        ),
        (  # wq_b
            [decode_cfg.configs["WQB_IN0_SHAPE"], decode_cfg.configs["WQB_IN1_SHAPE"]],
            [decode_cfg.configs["WQB_IN0_DTYPE"], decode_cfg.configs["WQB_IN1_DTYPE"]],
            decode_cfg.configs["WQB_PROGRAM_CFG"],
            [
                decode_cfg.configs["WQB_IN0_MEM_CFG"],
                decode_cfg.configs["WQB_IN1_MEM_CFG"],
                decode_cfg.configs["WQB_OUT_MEM_CFG"],
            ],
        ),
        (  # wkv_a
            [decode_cfg.configs["WKV_A_IN0_SHAPE"], decode_cfg.configs["WKV_A_IN1_SHAPE"]],
            [decode_cfg.configs["WKV_A_IN0_DTYPE"], decode_cfg.configs["WKV_A_IN1_DTYPE"]],
            decode_cfg.configs["WKV_A_PROGRAM_CFG"],
            [
                decode_cfg.configs["WKV_A_IN0_MEM_CFG"],
                decode_cfg.configs["WKV_A_IN1_MEM_CFG"],
                decode_cfg.configs["WKV_A_OUT_MEM_CFG"],
            ],
        ),
        (  # wkv_b1
            [decode_cfg.configs["WKV_B1_IN0_SHAPE"], decode_cfg.configs["WKV_B1_IN1_SHAPE"]],
            [decode_cfg.configs["WKV_B1_IN0_DTYPE"], decode_cfg.configs["WKV_B1_IN1_DTYPE"]],
            decode_cfg.configs["WKV_B1_PROGRAM_CFG"],
            [
                decode_cfg.configs["WKV_B1_IN0_MEM_CFG"],
                decode_cfg.configs["WKV_B1_IN1_MEM_CFG"],
                decode_cfg.configs["WKV_B1_OUT_MEM_CFG"],
            ],
        ),
        (  # wkv_b2
            [decode_cfg.configs["WKV_B2_IN0_SHAPE"], decode_cfg.configs["WKV_B2_IN1_SHAPE"]],
            [decode_cfg.configs["WKV_B2_IN0_DTYPE"], decode_cfg.configs["WKV_B2_IN1_DTYPE"]],
            decode_cfg.configs["WKV_B2_PROGRAM_CFG"],
            [
                decode_cfg.configs["WKV_B2_IN0_MEM_CFG"],
                decode_cfg.configs["WKV_B2_IN1_MEM_CFG"],
                decode_cfg.configs["WKV_B2_OUT_MEM_CFG"],
            ],
        ),
        (  # wo
            [decode_cfg.configs["WO_IN0_SHAPE"], decode_cfg.configs["WO_IN1_SHAPE"]],
            [decode_cfg.configs["WO_IN0_DTYPE"], decode_cfg.configs["WO_IN1_DTYPE"]],
            decode_cfg.configs["WO_PROGRAM_CFG"],
            [
                decode_cfg.configs["WO_IN0_MEM_CFG"],
                decode_cfg.configs["WO_IN1_MEM_CFG"],
                decode_cfg.configs["WO_OUT_MEM_CFG"],
            ],
        ),
    ],
    ids=[
        "wq_a",
        "wq_b",
        "wkv_a",
        "wkv_b1",
        "wkv_b2",
        "wo",
    ],
)
def test_decode_matmuls(
    device,
    shapes,
    dtypes,
    program_config,
    memory_configs,
    function_level_defaults,
    reset_seeds,
):
    run_matmul_impl(
        device,
        shapes=shapes,
        dtypes=dtypes,
        program_config=program_config,
        memory_configs=memory_configs,
    )


@pytest.mark.parametrize(
    "seq_len",
    [128, 1024, 8096],
)
@pytest.mark.parametrize(
    "shapes, dtypes, program_config, memory_configs",
    [
        (  # wq_a
            [prefill_cfg.configs["WQA_IN0_SHAPE"], prefill_cfg.configs["WQA_IN1_SHAPE"]],
            [prefill_cfg.configs["WQA_IN0_DTYPE"], prefill_cfg.configs["WQA_IN1_DTYPE"]],
            prefill_cfg.configs["WQA_PROGRAM_CFG"],
            [
                prefill_cfg.configs["WQA_IN0_MEM_CFG"],
                prefill_cfg.configs["WQA_IN1_MEM_CFG"],
                prefill_cfg.configs["WQA_OUT_MEM_CFG"],
            ],
        ),
        (  # wq_b
            [prefill_cfg.configs["WQB_IN0_SHAPE"], prefill_cfg.configs["WQB_IN1_SHAPE"]],
            [prefill_cfg.configs["WQB_IN0_DTYPE"], prefill_cfg.configs["WQB_IN1_DTYPE"]],
            prefill_cfg.configs["WQB_PROGRAM_CFG"],
            [
                prefill_cfg.configs["WQB_IN0_MEM_CFG"],
                prefill_cfg.configs["WQB_IN1_MEM_CFG"],
                prefill_cfg.configs["WQB_OUT_MEM_CFG"],
            ],
        ),
        (  # wkv_a
            [prefill_cfg.configs["WKV_A_IN0_SHAPE"], prefill_cfg.configs["WKV_A_IN1_SHAPE"]],
            [prefill_cfg.configs["WKV_A_IN0_DTYPE"], prefill_cfg.configs["WKV_A_IN1_DTYPE"]],
            prefill_cfg.configs["WKV_A_PROGRAM_CFG"],
            [
                prefill_cfg.configs["WKV_A_IN0_MEM_CFG"],
                prefill_cfg.configs["WKV_A_IN1_MEM_CFG"],
                prefill_cfg.configs["WKV_A_OUT_MEM_CFG"],
            ],
        ),
        (  # wkv_b1
            [prefill_cfg.configs["WKV_B1_IN0_SHAPE"], prefill_cfg.configs["WKV_B1_IN1_SHAPE"]],
            [prefill_cfg.configs["WKV_B1_IN0_DTYPE"], prefill_cfg.configs["WKV_B1_IN1_DTYPE"]],
            prefill_cfg.configs["WKV_B1_PROGRAM_CFG"],
            [
                prefill_cfg.configs["WKV_B1_IN0_MEM_CFG"],
                prefill_cfg.configs["WKV_B1_IN1_MEM_CFG"],
                prefill_cfg.configs["WKV_B1_OUT_MEM_CFG"],
            ],
        ),
        (  # wkv_b2
            [prefill_cfg.configs["WKV_B2_IN0_SHAPE"], prefill_cfg.configs["WKV_B2_IN1_SHAPE"]],
            [prefill_cfg.configs["WKV_B2_IN0_DTYPE"], prefill_cfg.configs["WKV_B2_IN1_DTYPE"]],
            prefill_cfg.configs["WKV_B2_PROGRAM_CFG"],
            [
                prefill_cfg.configs["WKV_B2_IN0_MEM_CFG"],
                prefill_cfg.configs["WKV_B2_IN1_MEM_CFG"],
                prefill_cfg.configs["WKV_B2_OUT_MEM_CFG"],
            ],
        ),
        (  # wo
            [prefill_cfg.configs["WO_IN0_SHAPE"], prefill_cfg.configs["WO_IN1_SHAPE"]],
            [prefill_cfg.configs["WO_IN0_DTYPE"], prefill_cfg.configs["WO_IN1_DTYPE"]],
            prefill_cfg.configs["WO_PROGRAM_CFG"],
            [
                prefill_cfg.configs["WO_IN0_MEM_CFG"],
                prefill_cfg.configs["WO_IN1_MEM_CFG"],
                prefill_cfg.configs["WO_OUT_MEM_CFG"],
            ],
        ),
    ],
    ids=[
        "wq_a",
        "wq_b",
        "wkv_a",
        "wkv_b1",
        "wkv_b2",
        "wo",
    ],
)
def test_prefill_matmuls(
    device,
    shapes,
    dtypes,
    program_config,
    memory_configs,
    seq_len,
    function_level_defaults,
    reset_seeds,
):
    run_matmul_impl(
        device,
        shapes=shapes,
        dtypes=dtypes,
        program_config=program_config,
        memory_configs=memory_configs,
        seq_len=seq_len,
    )


@skip_for_blackhole("See GH Issue #24926.")
@pytest.mark.parametrize(
    "shape, dtype, mem_config",
    [
        (  # q_rope
            decode_cfg.configs["QROPE_SHAPE"],
            decode_cfg.configs["QROPE_DTYPE"],
            decode_cfg.configs["QROPE_MEM_CFG"],
        ),
        (  # k_rope
            decode_cfg.configs["KROPE_SHAPE"],
            decode_cfg.configs["KROPE_DTYPE"],
            decode_cfg.configs["KROPE_MEM_CFG"],
        ),
    ],
    ids=[
        "q_rope",
        "k_rope",
    ],
)
def test_decode_ropes(
    device,
    shape,
    dtype,
    mem_config,
    function_level_defaults,
    reset_seeds,
):
    run_rope_impl(
        device,
        shape=shape,
        dtype=dtype,
        mem_config=mem_config,
    )


@pytest.mark.parametrize(
    "seq_len",
    [128, 1024, 8096],
)
@pytest.mark.parametrize(
    "shape, dtype, mem_config",
    [
        (  # q_rope
            prefill_cfg.configs["QROPE_SHAPE"],
            prefill_cfg.configs["QROPE_DTYPE"],
            prefill_cfg.configs["QROPE_MEM_CFG"],
        ),
        (  # k_rope
            prefill_cfg.configs["KROPE_SHAPE"],
            prefill_cfg.configs["KROPE_DTYPE"],
            prefill_cfg.configs["KROPE_MEM_CFG"],
        ),
    ],
    ids=[
        "q_rope",
        "k_rope",
    ],
)
def test_prefill_ropes(
    device,
    shape,
    dtype,
    mem_config,
    seq_len,
    function_level_defaults,
    reset_seeds,
):
    run_rope_impl(
        device,
        shape=shape,
        dtype=dtype,
        mem_config=mem_config,
        seq_len=seq_len,
    )


@pytest.mark.parametrize(
    "shape, dtype, mem_config, cache_dtype",
    [
        (
            decode_cfg.configs["KVPE_SHAPE"],
            decode_cfg.configs["KVPE_DTYPE"],
            decode_cfg.configs["KVPE_MEM_CFG"],
            decode_cfg.configs["KVPE_CACHE_DTYPE"],
        ),
    ],
    ids=["kvpe"],
)
def test_update_caches(
    device,
    shape,
    dtype,
    mem_config,
    cache_dtype,
    function_level_defaults,
    reset_seeds,
):
    run_update_cache_impl(
        device,
        shape=shape,
        dtype=dtype,
        mem_config=mem_config,
        cache_dtype=cache_dtype,
    )


@pytest.mark.parametrize(
    "seq_len",
    [128, 1024, 8096],
)
@pytest.mark.parametrize(
    "shape, dtype, mem_config, cache_dtype",
    [
        (
            prefill_cfg.configs["KVPE_SHAPE"],
            prefill_cfg.configs["KVPE_DTYPE"],
            prefill_cfg.configs["KVPE_MEM_CFG"],
            prefill_cfg.configs["KVPE_CACHE_DTYPE"],
        ),
    ],
    ids=["kvpe"],
)
def test_fill_caches(
    device,
    shape,
    dtype,
    mem_config,
    cache_dtype,
    seq_len,
    function_level_defaults,
    reset_seeds,
):
    run_fill_cache_impl(
        device,
        shape=shape,
        dtype=dtype,
        mem_config=mem_config,
        cache_dtype=cache_dtype,
        seq_len=seq_len,
    )


@pytest.mark.parametrize(
    "shape, dtype, mem_config, norm_category",
    [
        (
            decode_cfg.configs["QNORM_SHAPE"],
            decode_cfg.configs["QNORM_DTYPE"],
            decode_cfg.configs["QNORM_MEM_CFG"],
            decode_cfg.configs["QNORM_CATEGORY"],
        ),
        (
            decode_cfg.configs["KNORM_SHAPE"],
            decode_cfg.configs["KNORM_DTYPE"],
            decode_cfg.configs["KNORM_MEM_CFG"],
            decode_cfg.configs["KNORM_CATEGORY"],
        ),
    ],
    ids=["q_norm", "k_norm"],
)
def test_decode_rmsnorms(
    device,
    shape,
    dtype,
    mem_config,
    norm_category,
    temp_dir,
    function_level_defaults,
    reset_seeds,
):
    run_rmsnorm_impl(
        device,
        shape=shape,
        dtype=dtype,
        mem_config=mem_config,
        norm_category=norm_category,
        temp_dir=temp_dir,
    )


@pytest.mark.parametrize(
    "seq_len",
    [128, 1024, 8096],
)
@pytest.mark.parametrize(
    "shape, dtype, mem_config, norm_category",
    [
        (
            prefill_cfg.configs["QNORM_SHAPE"],
            prefill_cfg.configs["QNORM_DTYPE"],
            prefill_cfg.configs["QNORM_MEM_CFG"],
            prefill_cfg.configs["QNORM_CATEGORY"],
        ),
        (
            prefill_cfg.configs["KNORM_SHAPE"],
            prefill_cfg.configs["KNORM_DTYPE"],
            prefill_cfg.configs["KNORM_MEM_CFG"],
            prefill_cfg.configs["KNORM_CATEGORY"],
        ),
    ],
    ids=["q_norm", "k_norm"],
)
def test_prefill_rmsnorms(
    device,
    shape,
    dtype,
    mem_config,
    norm_category,
    temp_dir,
    seq_len,
    function_level_defaults,
    reset_seeds,
):
    run_rmsnorm_impl(
        device,
        shape=shape,
        dtype=dtype,
        mem_config=mem_config,
        norm_category=norm_category,
        temp_dir=temp_dir,
        seq_len=seq_len,
    )
