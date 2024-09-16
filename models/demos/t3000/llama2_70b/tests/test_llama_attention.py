# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import torch
import ttnn
from ttnn import ReplicateTensorToMesh, ConcatMeshToTensor

from models.demos.t3000.llama2_70b.reference.llama.llama import Llama
from models.demos.t3000.llama2_70b.tt.llama_attention_optimized import TtLlamaAttention_optimized
from models.demos.t3000.llama2_70b.reference.llama.llama.model import precompute_freqs_cis

from models.utility_functions import skip_for_grayskull
from models.demos.t3000.llama2_70b.tt.llama_common import (
    setup_llama_env,
    check_mesh_device,
    extract_pcc_from_log,
    generate_rot_emb,
    get_rotation_mat,
    gather_cos_sin,
    precompute_freqs,
    MAX_SEQ_LEN,
    MAX_SEQ_LEN_LLAMA3,
    BASE_URL,
    UNIT_TEST_N_LAYER,
    UNIT_TEST_LAYER_NUM,
    UNIT_TEST_START_POS,
    UNIT_TEST_GENERATION_LENGTH,
    comp_pcc,
    get_rot_transformation_mat,
    should_skip_model_load,
    check_kv_cache,
)
from models.utility_functions import nearest_32

from dataclasses import dataclass


@dataclass
class PagedAttentionConfig:
    block_size = 64
    max_num_blocks = 2048


class PytorchLlamaAttentionModel(torch.nn.Module):
    def __init__(self, hf_reference_model, layer_num):
        super().__init__()
        self.attention = hf_reference_model.layers[layer_num].attention

        # Disable dropout
        self.attention.eval()

        configuration = hf_reference_model.params
        self.n_heads = configuration.n_heads
        hidden_dim = configuration.dim
        self.head_dim = hidden_dim // self.n_heads
        self.max_seq_len = configuration.max_seq_len

    def prepare_inputs(self, x, start_pos):
        """
        Prepare inputs for decode mode. Assume that current token is at
        start_pos, and KV cache has valid data up to start_pos.
        """
        batch = x.size(0)
        freqs_cis = precompute_freqs_cis(self.head_dim, self.max_seq_len * 2)
        freqs_cis = freqs_cis[start_pos : start_pos + 1]

        attn_mask = torch.zeros(batch, 1, 1, start_pos + 1)
        # attn_mask[:, :, :, : start_pos + 1] = -1e9
        attn_mask = attn_mask.expand(-1, self.n_heads, -1, -1)

        return x, start_pos, freqs_cis, attn_mask

    def prepare_inputs_prefill(self, x, start_pos):
        """
        Prepare inputs for decode mode. Assume that current token is at
        start_pos, and KV cache has valid data up to start_pos.
        """
        batch = x.size(0)
        seq_len = x.size(1)
        freqs_cis = precompute_freqs_cis(self.head_dim, self.max_seq_len * 2)
        freqs_cis = freqs_cis[start_pos : start_pos + seq_len]

        attn_mask = torch.full((seq_len, seq_len), float("-inf"))
        attn_mask = torch.triu(attn_mask, diagonal=1)
        attn_mask = attn_mask.expand(batch, self.n_heads, -1, -1)

        return x, start_pos, freqs_cis, attn_mask

    def forward(self, x, start_pos, freqs_cis, mask):
        """
        x: (batch, seq, hidden_dim)
        start_pos: int
        freqs_cis: ?
        mask: ?

        return: (batch, seq, hidden_dim)
        """
        result = self.attention(
            x,
            start_pos,
            freqs_cis,
            mask,
        )
        return result


def tt_llama_attention_prepare_inputs(llama_attention_model, x, start_pos, mode):
    assert len(x.size()) == 3
    batch, seq_len, _ = x.shape

    cache_name = lambda name: llama_attention_model.cache_path / (f"{name}")

    if mode == "prefill":
        assert (
            seq_len % 128 == 0 and seq_len > 0 and seq_len <= 8192
        ), "Prefill mode only supports seqlen as a multiple of 128 up to 8k"
        assert batch == 1, "prefill mode only supports batch size 1"
        x = x.unsqueeze(0)
        assert x.shape == (1, batch, seq_len, llama_attention_model.hidden_size)
        xs = ttnn.as_tensor(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ReplicateTensorToMesh(llama_attention_model.mesh_device),
            device=llama_attention_model.mesh_device,
        )
        xs = ttnn.to_device(xs, llama_attention_model.mesh_device)

        cos, sin = precompute_freqs(llama_attention_model.head_dim, llama_attention_model.max_seq_len * 2)
        cos_gathered, sin_gathered = gather_cos_sin(torch.arange(start_pos, start_pos + seq_len), cos, sin)
        assert cos_gathered.size() == (1, 1, seq_len, llama_attention_model.head_dim)
        assert sin_gathered.size() == (1, 1, seq_len, llama_attention_model.head_dim)

        cos_gathereds = ttnn.as_tensor(
            cos_gathered,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=cache_name(f"cos_gathered_prefill_{seq_len}"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            device=llama_attention_model.mesh_device,
            mesh_mapper=ReplicateTensorToMesh(llama_attention_model.mesh_device),
        )
        sin_gathereds = ttnn.as_tensor(
            sin_gathered,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=cache_name(f"sin_gathered_prefill_{seq_len}"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            device=llama_attention_model.mesh_device,
            mesh_mapper=ReplicateTensorToMesh(llama_attention_model.mesh_device),
        )

        cos_gathereds = ttnn.to_device(cos_gathereds, llama_attention_model.mesh_device)
        sin_gathereds = ttnn.to_device(sin_gathereds, llama_attention_model.mesh_device)
        rot_mats = [cos_gathereds, sin_gathereds]

        cache_idxs_tt = None

    elif mode == "decode":
        assert seq_len == 1, "Only supporting decode mode"
        x = x.transpose(0, 1).unsqueeze(1)
        assert x.shape == (seq_len, 1, batch, llama_attention_model.hidden_size)
        # Pad x to match the padded batch size
        x = torch.nn.functional.pad(
            x, (0, 0, 0, llama_attention_model.model_config["PADDED_BATCH_SIZE"] - batch), value=0
        )

        xs = ttnn.as_tensor(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ReplicateTensorToMesh(llama_attention_model.mesh_device),
            device=llama_attention_model.mesh_device,
        )
        xs = ttnn.to_device(xs, llama_attention_model.mesh_device)
        xs = ttnn.interleaved_to_sharded(xs, llama_attention_model.model_config["FINAL_ALL_GATHER_OUTPUT_MEMCFG"])

        cache_idxs = torch.tensor([start_pos for _ in range(batch)])

        rot_emb = generate_rot_emb(llama_attention_model.head_dim, llama_attention_model.max_seq_len * 2)
        rot_mat = get_rotation_mat(rot_emb, cache_idxs, seq_len, batch=batch)
        assert rot_mat.size() == (1, batch, llama_attention_model.head_dim, llama_attention_model.head_dim)

        rot_mats = ttnn.as_tensor(
            rot_mat,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ReplicateTensorToMesh(llama_attention_model.mesh_device),
            device=llama_attention_model.mesh_device,
        )
        rot_mats = ttnn.to_device(rot_mats, llama_attention_model.mesh_device)

        rot_mats = ttnn.interleaved_to_sharded(rot_mats, llama_attention_model.model_config["ROT_MAT_MM_IN1_MEMCFG"])

        cache_idxs_tt = ttnn.as_tensor(
            cache_idxs,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=llama_attention_model.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ReplicateTensorToMesh(llama_attention_model.mesh_device),
        )

    return (
        xs,
        start_pos,
        rot_mats,
        cache_idxs_tt,
    )


def run_test_LlamaAttention_inference(
    t3k_mesh_device,
    max_batch_size,
    batch,
    seq_len,
    pcc,
    model_config,
    llama_version,
    ckpt_dir,
    tokenizer_path,
    cache_path,
    paged_attention,
):
    # Prepare paths and devices
    skip_model_load = should_skip_model_load()

    # Prepare configs
    hugging_face_reference_model = Llama.build(
        ckpt_dir,
        tokenizer_path,
        max_seq_len=MAX_SEQ_LEN if llama_version == "llama2" else MAX_SEQ_LEN_LLAMA3,
        max_batch_size=max_batch_size,
        n_layers=UNIT_TEST_N_LAYER,
        skip_model_load=skip_model_load,
    ).model
    hugging_face_reference_model.eval()
    state_dict = hugging_face_reference_model.state_dict()
    logger.info(state_dict.keys())
    torch.manual_seed(0)
    configuration = hugging_face_reference_model.params

    # PyTorch model --------------------------------------------------------------------
    pytorch_LlamaAttention_model = PytorchLlamaAttentionModel(hugging_face_reference_model, UNIT_TEST_LAYER_NUM)
    # TT model -------------------------------------------------------------------------
    transformation_mat_torch = get_rot_transformation_mat(32)  # 32 for tile size

    transformation_mats = ttnn.as_tensor(
        transformation_mat_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=t3k_mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ReplicateTensorToMesh(t3k_mesh_device),
    )
    transformation_mats = ttnn.to_device(transformation_mats, t3k_mesh_device)

    page_table_tt = None
    paged_attention_config = None
    if paged_attention:
        paged_attention_config = PagedAttentionConfig()

        # Implied shuffling of blocks
        permutation = torch.randperm(paged_attention_config.max_num_blocks)
        # Page table which maps virtua blocks to physical
        reverse_permutation = torch.argsort(permutation)
        page_table = reverse_permutation.reshape(
            max_batch_size, paged_attention_config.max_num_blocks // max_batch_size
        )
        page_table_tt = ttnn.as_tensor(
            page_table,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ReplicateTensorToMesh(t3k_mesh_device),
        )
        page_table_tt = ttnn.to_device(page_table_tt, t3k_mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    mode = "prefill" if seq_len > 1 else "decode"

    tt_LlamaAttention_model = TtLlamaAttention_optimized(
        t3k_mesh_device,
        state_dict,
        BASE_URL,
        UNIT_TEST_LAYER_NUM,
        model_config,
        configuration,
        transformation_mats,
        cache_path=cache_path,
        paged_attention_config=paged_attention_config,
    )

    all_tests_pass, all_pccs = True, []
    if mode == "prefill":
        generation_start_pos = 0
        generation_length = 1
    else:
        generation_start_pos = UNIT_TEST_START_POS
        generation_length = UNIT_TEST_GENERATION_LENGTH

    for i in range(generation_length):
        # Prepare input
        pt_inp_ids = torch.randint(0, configuration.vocab_size, (batch, seq_len))
        pt_inp = hugging_face_reference_model.tok_embeddings(pt_inp_ids)
        pt_inp_normed = hugging_face_reference_model.layers[UNIT_TEST_LAYER_NUM].attention_norm(pt_inp)
        tt_input = pt_inp_normed.clone()
        start_pos = generation_start_pos + i

        # PyTorch output --------------------------------------------------------------------
        if mode == "prefill":
            attention_input, start_pos, freqs_cis, attn_mask = pytorch_LlamaAttention_model.prepare_inputs_prefill(
                pt_inp_normed, start_pos
            )
        else:
            attention_input, start_pos, freqs_cis, attn_mask = pytorch_LlamaAttention_model.prepare_inputs(
                pt_inp_normed, start_pos
            )

        pytorch_out = pytorch_LlamaAttention_model(
            attention_input,
            start_pos,
            freqs_cis,
            attn_mask,
        )

        # TT hardware execution -------------------------------------------------------------
        attention_input, start_pos, rot_mat, cache_idxs = tt_llama_attention_prepare_inputs(
            tt_LlamaAttention_model, tt_input, start_pos, mode
        )
        tt_out = tt_LlamaAttention_model(
            attention_input,
            rot_mat,
            start_pos,
            cache_idxs=cache_idxs if mode == "decode" else None,
            page_table=page_table_tt,
            mode=mode,
        )

        tt_out = ttnn.from_device(tt_out)
        tt_out = ttnn.to_torch(tt_out, mesh_composer=ConcatMeshToTensor(t3k_mesh_device, dim=3))
        tt_out = tt_out.permute(2, 1, 0, 3).squeeze(1)  # [batch, seq_len, hidden_dim]
        if mode == "decode":
            tt_out = tt_out[:batch]

        # check outputs ----------------------------------------------------------------------
        does_pass, output_pcc = comp_pcc(pytorch_out, tt_out, pcc)
        logger.info(f"Output: {output_pcc}")
        all_pccs.append(extract_pcc_from_log(output_pcc))

        if does_pass:
            logger.info(f"[start_pos={start_pos}] {llama_version} Attention output Passed!")
        else:
            logger.warning(
                f"[start_pos={start_pos}] {llama_version} Attention output Failed! PCC value is lower than {pcc}"
            )
            all_tests_pass = False

    logger.info(f"Average PCC over {len(all_pccs)} tokens: {sum(all_pccs) / len(all_pccs)}")

    # Check kv cache
    # PyTorch output --------------------------------------------------------------------
    pytorch_layer_present = [
        pytorch_LlamaAttention_model.attention.cache_k.clone().permute(0, 2, 1, 3)[
            :batch, ...
        ],  # [batch, n_kv_heads, seq, head_dim]
        pytorch_LlamaAttention_model.attention.cache_v.clone().permute(0, 2, 1, 3)[
            :batch, ...
        ],  # [batch, n_kv_heads, seq, head_dim]
    ]
    # TT hardware output ----------------------------------------------------------------

    # concat the pasts by heads
    tt_layer_present_all = [ttnn.from_device(lp) for lp in tt_LlamaAttention_model.layer_past]
    if paged_attention:
        tt_layer_present_all = [ttnn.from_device(lp) for lp in tt_LlamaAttention_model.layer_past]
        tt_layer_present_all = [
            (
                ttnn.to_torch(lp, mesh_composer=ConcatMeshToTensor(t3k_mesh_device, dim=1))[reverse_permutation]
                .reshape(
                    max_batch_size,
                    paged_attention_config.max_num_blocks // max_batch_size,
                    configuration.n_kv_heads,
                    paged_attention_config.block_size,
                    tt_LlamaAttention_model.head_dim,
                )
                .transpose(1, 2)
                .reshape(max_batch_size, configuration.n_kv_heads, -1, tt_LlamaAttention_model.head_dim)[:batch, ...]
            )
            for lp in tt_layer_present_all
        ]
    else:
        tt_layer_present_all = [
            ttnn.to_torch(lp, mesh_composer=ConcatMeshToTensor(t3k_mesh_device, dim=1))[:batch, ...]
            for lp in tt_layer_present_all
        ]

    cache_test_pass = check_kv_cache(
        pytorch_layer_present,
        tt_layer_present_all,
        generation_start_pos,
        generation_length,
        seq_len,
        mode == "prefill",
        pcc,
    )

    all_tests_pass = all_tests_pass and cache_test_pass

    if all_tests_pass:
        logger.info(f"{llama_version} Attention output Passed!")
    assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "llama_version",
    (
        ("llama2"),
        ("llama3"),
    ),
)
@pytest.mark.parametrize(
    "batch, seq_len, pcc",
    ((32, 1, 0.9997), (16, 1, 0.999), (1, 128, 0.9997), (1, 2048, 0.9997), (1, 8192, 0.99)),
    ids=("decode", "decodeb16", "prefill_128", "prefill_2k", "prefill_8k"),
)
@pytest.mark.parametrize(
    "max_batch_size, max_context_len",
    (
        (32, 2048),
        (16, 8192),
    ),
    ids=(
        "short_context",
        "long_context",
    ),
)
@pytest.mark.parametrize(
    "paged_attention",
    (True, False),
    ids=("paged_attention", "non_paged_attention"),
)
def test_LlamaAttention_inference(
    batch,
    seq_len,
    pcc,
    t3k_mesh_device,
    max_batch_size,
    max_context_len,
    llama_version,
    paged_attention,
    use_program_cache,
):
    if seq_len == 1 and batch != max_batch_size:
        pytest.skip(f"Input batch size should match max_batch_size")

    if batch == 1 and seq_len > max_context_len:
        pytest.skip(f"Prefill with seq_len={seq_len} is not supported with short context")

    if llama_version == "llama2" and seq_len > 2048:
        pytest.skip(f"Llama2 with seq_len={seq_len} is not supported (max 2048)")

    model_config, ckpt_dir, tokenizer_path, cache_path = setup_llama_env(
        llama_version=llama_version,
        batch=batch,
        seq_len=seq_len,
        max_batch_size=max_batch_size,
        max_context_len=max_context_len,
    )

    check_mesh_device(t3k_mesh_device, model_config)
    run_test_LlamaAttention_inference(
        t3k_mesh_device,
        max_batch_size,
        batch,
        seq_len,
        pcc,
        model_config,
        llama_version,
        ckpt_dir,
        tokenizer_path,
        cache_path,
        paged_attention,
    )
