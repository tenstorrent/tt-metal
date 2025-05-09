# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import pytest
import scipy
import torch
from loguru import logger
from sklearn.metrics import top_k_accuracy_score

import ttnn
from models.demos.t3000.llama2_70b.reference.llama.llama import Llama
from models.demos.t3000.llama2_70b.tests.test_llama_attention import PagedAttentionConfig
from models.demos.t3000.llama2_70b.tt.llama_common import (
    BASE_URL,
    UNIT_TEST_GENERATION_LENGTH,
    check_kv_cache,
    check_mesh_device,
    comp_pcc,
    extract_pcc_from_log,
    setup_llama_env,
    should_skip_model_load,
)
from models.demos.t3000.llama2_70b.tt.llama_model_optimized import TtLlamaModel_optimized
from models.utility_functions import skip_for_grayskull
from ttnn import ConcatMeshToTensor

DEVICE_PERF_START_SIGNPOST = "START_PERF_RUN"
DEVICE_PERF_END_SIGNPOST = "END_PERF_RUN"


class PytorchLlamaModel(torch.nn.Module):
    def __init__(self, hf_reference_model):
        super().__init__()
        self.model = hf_reference_model

        # Disable dropout
        self.model.eval()

        configuration = hf_reference_model.params
        self.n_heads = configuration.n_heads
        hidden_dim = configuration.dim
        self.head_dim = hidden_dim // self.n_heads
        self.max_seq_len = configuration.max_seq_len

    def forward(self, x, start_pos):
        """
        x: (batch, seq)
        start_pos: int

        return: (batch, seq, hidden_dim)
        """
        return self.model(x, start_pos)


def run_test_LlamaModel_inference(
    t3k_mesh_device,
    batch,
    seq_len,
    max_batch_size,
    max_context_len,
    pcc,
    model_config,
    n_layers,
    llama_version,
    ckpt_dir,
    tokenizer_path,
    cache_path,
    prompt_file=None,
    generation_start_pos=0,
    device_perf=False,  # set to True when measuring device perf
    paged_attention=False,
    chunk_size=None,
):
    if device_perf:  # Enable tracy signpost support in device perf runs only
        from tracy import signpost

    # Load prompt file if provided
    prompt = None
    if prompt_file:
        assert os.path.isfile(prompt_file), "Input file does not exist!"
        with open(prompt_file, "r") as f:
            prompt = f.read()

    # Prepare paths and devices
    skip_model_load = should_skip_model_load()

    logger.info(f"Running num_layer: {n_layers}")
    hugging_face_reference = Llama.build(
        ckpt_dir,
        tokenizer_path,
        max_seq_len=max_context_len,
        max_batch_size=batch,
        n_layers=n_layers,
        skip_model_load=skip_model_load,
    )
    hugging_face_reference_model = hugging_face_reference.model
    hugging_face_reference_model.eval()
    state_dict = hugging_face_reference_model.state_dict()
    logger.info(state_dict.keys())
    torch.manual_seed(0)
    configuration = hugging_face_reference_model.params

    # PyTorch model --------------------------------------------------------------------
    pytorch_model = PytorchLlamaModel(hugging_face_reference_model)
    # TT model -------------------------------------------------------------------------

    page_table = None
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

    tt_model = TtLlamaModel_optimized(
        t3k_mesh_device,
        state_dict,
        BASE_URL,
        n_layers,
        model_config,
        configuration,
        cache_path=cache_path,
        paged_attention_config=paged_attention_config,
    )

    mode = "prefill" if seq_len > 1 else "decode"

    if mode == "prefill" or device_perf:
        generation_length = 1
    else:
        generation_length = UNIT_TEST_GENERATION_LENGTH

    # Pre-process inputs in prompt mode
    if prompt:
        tokenizer = hugging_face_reference.tokenizer
        tokenized = tokenizer.encode(prompt, bos=True, eos=False)
        tokenized = torch.tensor(tokenized).unsqueeze(0)

        # Sliding window across sequence dimension for generation_length iterations
        tokenized = tokenized[:, : (seq_len + generation_length - 1) * batch]
        tokenized = torch.reshape(tokenized, (batch, (seq_len + generation_length - 1)))

        logger.info("Finished converting prompt to tokens.")

    all_tests_pass = True
    all_pccs, all_top1, all_top5 = [], [], []
    for i in range(generation_length):
        # Prepare input
        if prompt:
            pt_inp_ids = tokenized[:, i : i + seq_len]  # Slide window
            assert pt_inp_ids.shape == (batch, seq_len), f"Inputs must have shape {(batch, seq_len)}"
        else:
            pt_inp_ids = torch.randint(0, configuration.vocab_size, (batch, seq_len))
        tt_inp_ids = pt_inp_ids.clone()

        start_pos = generation_start_pos + i

        # PyTorch output --------------------------------------------------------------------
        pytorch_out = pytorch_model(
            pt_inp_ids,
            start_pos,
        )
        if mode == "decode":
            pytorch_out = pytorch_out.squeeze().reshape(batch, -1)  # [batch, hidden_dim]
        else:
            pytorch_out = pytorch_out.squeeze().reshape(seq_len, -1)  # [seq, hidden_dim]

        if device_perf:
            signpost(DEVICE_PERF_START_SIGNPOST)  # start for device perf measurement
        # TT hardware execution -------------------------------------------------------------
        if chunk_size is not None:
            tt_out = []
            assert mode == "prefill"
            for chunk_start in range(0, seq_len, chunk_size):
                logger.info(f"Chunk start: {chunk_start}")
                chunk_end = chunk_start + chunk_size
                assert chunk_end <= seq_len, "Chunk end should be less than seq_len"
                chunk_page_table = page_table[
                    :,
                    chunk_start // paged_attention_config.block_size : chunk_end // paged_attention_config.block_size,
                ]
                # SDPA requires that the page table batch dim matches the input batch dim, which must be 1 in prefill
                prefill_page_table = page_table[0:1, :]

                chunk_tt_input = tt_inp_ids[:, chunk_start:chunk_end]
                # TT hardware execution -------------------------------------------------------------
                (
                    tt_inp_emb,
                    start_pos,
                    rot_mat,
                    rot_idxs_tt,
                    cache_idxs,
                    page_table_tt,
                    chunk_page_table_tt,
                ) = tt_model.prepare_inputs(
                    chunk_tt_input,
                    chunk_start,
                    mode=mode,
                    page_table=prefill_page_table,
                    chunk_page_table=chunk_page_table,
                )

                tt_chunk_out = tt_model(
                    tt_inp_emb,
                    rot_mat,
                    start_pos,
                    cache_idxs=cache_idxs,
                    mode=mode,
                    page_table=page_table_tt,
                    chunk_page_table=chunk_page_table_tt,
                    chunk_start_idx=chunk_start,
                )

                tt_chunk_out = ttnn.from_device(tt_chunk_out)
                tt_chunk_out = ttnn.to_torch(tt_chunk_out, mesh_composer=ConcatMeshToTensor(t3k_mesh_device, dim=3))
                tt_chunk_out = tt_chunk_out[..., : configuration.vocab_size]
                tt_chunk_out = tt_chunk_out.permute(2, 1, 0, 3).squeeze()  # [batch, seq_len, hidden_dim]

                tt_out.append(tt_chunk_out)

            tt_out = torch.cat(tt_out, dim=0).float()

        else:
            if mode == "decode":
                tt_inp_emb, start_pos, rot_mat, cache_idxs, *_ = tt_model.prepare_device_inputs_decode(
                    tt_inp_ids, start_pos, mode=mode
                )
            else:
                tt_inp_emb, start_pos, rot_mat, cache_idxs, *_ = tt_model.prepare_inputs(
                    tt_inp_ids, start_pos, mode=mode
                )

            tt_out = tt_model(
                tt_inp_emb,
                rot_mat,
                start_pos,
                cache_idxs=cache_idxs,
                mode=mode,
            )
            del tt_inp_emb, rot_mat

            tt_out = ttnn.from_device(tt_out)
            tt_out = ttnn.to_torch(tt_out, mesh_composer=ConcatMeshToTensor(t3k_mesh_device, dim=3))

            if device_perf:
                signpost(DEVICE_PERF_END_SIGNPOST)  # end for device perf measurement

            tt_out = tt_out[..., : configuration.vocab_size]
            tt_out = tt_out.permute(2, 1, 0, 3).squeeze()  # [batch, hidden_dim]
            if mode == "decode":
                tt_out = tt_out[:batch]
            tt_out = tt_out.float()

        # check outputs ----------------------------------------------------------------------
        does_pass, output_pcc = comp_pcc(pytorch_out, tt_out, pcc)
        logger.info(f"Output: {output_pcc}")
        all_pccs.append(extract_pcc_from_log(output_pcc))

        kl_divs = scipy.stats.entropy(
            torch.nn.functional.softmax(pytorch_out, dim=-1), torch.nn.functional.softmax(tt_out, dim=-1), axis=-1
        )
        logger.info(f"Mean KL Divergence: {kl_divs.mean()}")

        reference_top1 = np.argmax(pytorch_out, axis=-1)
        top1_acc = top_k_accuracy_score(reference_top1, tt_out, k=1, labels=np.arange(tt_out.shape[-1]))
        top5_acc = top_k_accuracy_score(reference_top1, tt_out, k=5, labels=np.arange(tt_out.shape[-1]))

        all_top1.append(top1_acc)
        all_top5.append(top5_acc)

        logger.info(f"Mean Top-1: {top1_acc}")
        logger.info(f"Mean Top-5: {top5_acc}")

        if does_pass:
            logger.info(f"[start_pos={start_pos}] {llama_version} Model output Passed!")
        else:
            logger.warning(
                f"[start_pos={start_pos}] {llama_version} Model output Failed! PCC value is lower than {pcc}"
            )
            all_tests_pass = False

    logger.info(f"Average PCC over {len(all_pccs)} tokens: {sum(all_pccs) / len(all_pccs)}")
    logger.info(f"Average Top-1 over {len(all_top1)} tokens: {sum(all_top1) / len(all_top1)}")
    logger.info(f"Average Top-5 over {len(all_top5)} tokens: {sum(all_top5) / len(all_top5)}")
    # Check kv cache
    # PyTorch output --------------------------------------------------------------------
    tt_layer_present_all = [ttnn.from_device(lp) for lp in tt_model.layers[0].attention.layer_past]
    if paged_attention:
        tt_layer_present_all = [
            (
                ttnn.to_torch(lp, mesh_composer=ConcatMeshToTensor(t3k_mesh_device, dim=1))[reverse_permutation]
                .reshape(
                    max_batch_size,
                    paged_attention_config.max_num_blocks // max_batch_size,
                    configuration.n_kv_heads,
                    paged_attention_config.block_size,
                    tt_model.head_dim,
                )
                .transpose(1, 2)
                .reshape(max_batch_size, configuration.n_kv_heads, -1, tt_model.head_dim)[:batch, ...]
            )
            for lp in tt_layer_present_all
        ]
    else:
        tt_layer_present_all = [
            ttnn.to_torch(lp, mesh_composer=ConcatMeshToTensor(t3k_mesh_device, dim=1))[:batch, ...]
            for lp in tt_layer_present_all
        ]

    pytorch_layer_present = [
        pytorch_model.model.layers[0]
        .attention.cache_k.clone()
        .permute(0, 2, 1, 3)[:batch, ...],  # [batch, n_kv_heads, seq, head_dim]
        pytorch_model.model.layers[0]
        .attention.cache_v.clone()
        .permute(0, 2, 1, 3)[:batch, ...],  # [batch, n_kv_heads, seq, head_dim]
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
        logger.info(f"{llama_version} output Passed!")

    assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"


@pytest.mark.timeout(240000)
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "llama_version",
    (
        ("llama2"),
        ("llama3"),
    ),
)
@pytest.mark.parametrize(
    "pcc,n_layers",
    (
        (0.997, 1),
        (0.996, 2),
        (0.99, 4),
        (0.99, 6),
        (0.99, 7),
        (0.99, 8),
        (0.99, 10),
        (0.99, 20),
        (0.99, 40),
        (0.99, 80),
    ),
    ids=("1L", "2L", "4L", "6L", "7L", "8L", "10L", "20L", "40L", "80L"),
)
@pytest.mark.parametrize(
    "batch, seq_len",
    ((32, 1), (16, 1), (1, 1), (1, 128), (1, 2048), (1, 8192), (1, 128 * 1024), (1, 3 * 1024)),
    ids=("decode", "decodeb16", "decodeb1", "prefill_128", "prefill_2k", "prefill_8k", "prefill_128k", "prefill_3k"),
)
@pytest.mark.parametrize(
    "max_batch_size, max_context_len",
    (
        (32, 2048),
        (16, 8192),
        (1, 128 * 1024),
    ),
    ids=(
        "short_context",
        "8k_context",
        "128k_context",
    ),
)
@pytest.mark.parametrize(
    "prompt_file",
    ("models/demos/t3000/llama2_70b/demo/data/a_tale_of_two_cities.txt", None),
    ids=("prompt_input", "rand_input"),
)
@pytest.mark.parametrize(
    "paged_attention, chunk_size",
    (
        (True, 128),
        (False, None),
    ),
    ids=("chunked_paged_attention", "unpaged_attention"),
)
def test_LlamaModel_inference(
    batch,
    seq_len,
    pcc,
    n_layers,
    t3k_mesh_device,
    max_batch_size,
    max_context_len,
    llama_version,
    prompt_file,
    paged_attention,
    chunk_size,
    use_program_cache,
):
    if chunk_size is not None and seq_len == 1:
        pytest.skip("Chunked prefill is not valid for decode mode tests")

    if chunk_size is not None:
        assert paged_attention, "Chunked prefill is only valid for paged attention"
        assert chunk_size > 0, "Chunk size must be greater than 0"
        assert seq_len % chunk_size == 0, "Sequence length must be divisible by chunk size"

    is_decode = seq_len == 1
    if is_decode and batch != max_batch_size:
        pytest.skip(f"Input batch size should match max_batch_size")

    if not is_decode and seq_len > max_context_len:
        pytest.skip(f"Prefill with seq_len={seq_len} is not supported with short context")

    if llama_version == "llama2" and seq_len > 2048:
        pytest.skip(f"Llama2 with seq_len={seq_len} is not supported (max 2048)")

    model_config, ckpt_dir, tokenizer_path, cache_path = setup_llama_env(
        llama_version=llama_version,
        max_batch_size=max_batch_size,
        max_context_len=max_context_len,
    )

    check_mesh_device(t3k_mesh_device, model_config)

    run_test_LlamaModel_inference(
        t3k_mesh_device,
        batch,
        seq_len,
        max_batch_size,
        max_context_len,
        pcc,
        model_config,
        n_layers,
        llama_version,
        ckpt_dir,
        tokenizer_path,
        cache_path,
        prompt_file=prompt_file,
        paged_attention=paged_attention,
        chunk_size=chunk_size,
    )
