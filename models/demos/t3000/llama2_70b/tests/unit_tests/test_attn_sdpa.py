# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import math
import torch.nn.functional as F

import ttnn
from models.demos.t3000.llama2_70b.reference.llama.llama import Llama
from models.demos.t3000.llama2_70b.reference.llama.llama.model import repeat_kv
from models.demos.t3000.llama2_70b.tt.model_config import (
    get_model_config,
    # get_tt_cache_path,
)

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
from models.utility_functions import torch2tt_tensor, tt2torch_tensor


class TtLlamaSDPA(torch.nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        base_url,
        layer_num,
        hidden_size: int,
        n_heads,
        n_kv_heads,
        model_config,
        tt_cache_path,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = hidden_size // n_heads

        self.state_dict = state_dict
        self.device = device
        self.hidden_size = hidden_size
        self.model_config = model_config

        layer_name = f"{base_url}.{layer_num}"

        wo_str = f"{layer_name}.attention.wo.weight"

        self.wo = torch2tt_tensor(
            torch.transpose(
                self.state_dict[wo_str],
                -2,
                -1,
            ),
            self.device,
        )

    def scale_mask_softmax_decomposed(self, attn, scale, attn_mask):
        # scores = scores / math.sqrt(self.head_dim)
        # if attn_mask is not None:
        #     scores = scores + attn_mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        # scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        attn = ttnn.multiply(attn, scale)

        ## Need to figure out how to broadcast in t dim
        # attn_mask = ttnn.repeat(attn_mask, [1, attn.shape()[1], 1, 1])  # this causes memory error as the broadcast result is too big
        # attn_mask = tt2torch_tensor(attn_mask)
        # attn_mask = attn_mask.repeat(1, attn.shape()[1], 1, 1)
        # attn_mask = torch2tt_tensor(attn_mask, self.device)

        attn = ttnn.add(attn, attn_mask)
        attn = ttnn.softmax(attn)
        return attn

    def forward(self, xq, keys, values, attn_mask):
        # xq = [seqlen, n_heads, batch, head_dim]
        # keys = [batch, num_kv_heads, cache_len + seqlen, dhead]
        # values = [batch, num_kv_heads, cache_len + seqlen, dhead]
        # attn_mask = [seq_len, n_heads, batch,cache_len + seqlen]

        keys = ttnn.transpose(keys, -1, -2)  #  [batch, num_kv_heads, dhead, cache_len + seqlen]
        attn = ttnn.experimental.group_attn_matmul(
            xq,
            keys,
            compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            # output_mem_config=self.model_config["PRE_SOFTMAX_MM_OUTPUT_MEMCFG"],
            # output_dtype=self.model_config["PRE_SOFTMAX_MM_OUTPUT_DTYPE"],  # Must be BFLOAT16
        )  # seqlen, n_heads, batch, cache_len + seqlen

        # TODO: This op expects attn_mask to be sharded such that each core has 1 head
        # This is illegal on single chip since we need 8x8 coregrid to shard
        # 64 heads on. Until we fracture on multi-chip, we can't use this op.
        # attn = ttnn.scale_mask_softmax_in_place(
        #         attn,
        #         1 / math.sqrt(self.head_dim),
        #         attn_mask,
        #         # program_config=self.model_config["SOFTMAX_PROGCFG"],
        #         is_causal_mask=True,
        #     )  # seqlen, n_heads, batch, cache_len + seqlen
        scale = 1 / math.sqrt(self.head_dim)
        attn = self.scale_mask_softmax_decomposed(attn, scale, attn_mask)

        attn_output = ttnn.experimental.group_attn_matmul(
            attn,
            values,
            compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            # output_mem_config=self.model_config["POST_SOFTMAX_MM_OUTPUT_MEMCFG"],
            # output_dtype=self.model_config["POST_SOFTMAX_MM_OUTPUT_DTYPE"],  # Must be BFLOAT16
        )  # seqlen, n_heads, batch, dhead

        attn_output = ttnn.experimental.nlp_concat_heads(
            attn_output,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            # output_mem_config=self.model_config["CONCAT_HEADS_OUTPUT_MEMCFG"],
        )  # seqlen, 1, batch, hidden_size

        attb_output = ttnn.matmul(
            attn_output,
            self.wo,
        )  # seqlen, 1, batch, hidden_size
        return attb_output


class PytorchLlamaSDPA(torch.nn.Module):
    def __init__(self, hf_reference_model, layer_num):
        super().__init__()
        self.n_heads = hf_reference_model.params.n_heads
        self.n_kv_heads = hf_reference_model.params.n_kv_heads
        self.head_dim = hf_reference_model.params.dim // self.n_heads
        self.attn = hf_reference_model.layers[layer_num].attention
        self.n_rep = self.n_heads // self.n_kv_heads

    def forward(self, xq, keys, values, attn_mask):
        """
        Take k_new, v_new. Return updated layer_past.
        """
        # xq = [seqlen, n_heads, batch, head_dim]
        # keys = [batch, num_kv_heads, cache_len + seqlen, dhead]
        # values = [batch, num_kv_heads, cache_len + seqlen, dhead]
        # attn_mask = [seq_len, 1, batch,cache_len + seqlen]
        attn_mask = attn_mask.permute(2, 1, 0, 3)  # [batch, n_heads, seq_len, cache_len + seqlen]
        bsz = xq.size(2)
        seqlen = xq.size(0)

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys.transpose(1, 2), self.n_rep).transpose(
            1, 2
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(values.transpose(1, 2), self.n_rep).transpose(
            1, 2
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.permute(2, 1, 0, 3)  # (bs, n_local_heads, seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3))
        scores = scores / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores = scores + attn_mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.attn.wo(output)  # (bs, seqlen, dim)


def run_test_LlamaSDPA(
    device,
    model_version,
    batch,
    seq_len,
    pcc,
    model_config,
    # tt_cache_path,
    # model_location_generator,
):
    # model_name = model_location_generator(model_version, model_subdir="Falcon")

    ckpt_dir = "/proj_sw/user_dev/llama-data-repacked/llama-2-70b/"
    tokenizer_path = "/proj_sw/user_dev/llama-data/tokenizer.model"

    hugging_face_reference_model = Llama.build(
        ckpt_dir, tokenizer_path, max_seq_len=4096, max_batch_size=1, n_layers=1, skip_model_load=True
    ).model

    hugging_face_reference_model.eval()
    configuration = hugging_face_reference_model.params
    n_heads = configuration.n_heads
    n_kv_heads = configuration.n_kv_heads
    hidden_dim = configuration.dim
    head_dim = hidden_dim // n_heads

    state_dict = hugging_face_reference_model.state_dict()

    # Prepare input
    # xq, keys, values, attn_mask
    torch.manual_seed(0)
    max_seq_len = configuration.max_seq_len
    inp = [
        (torch.rand(seq_len, n_heads, batch, head_dim) * 2) - 1,
        (torch.rand(batch, n_kv_heads, max_seq_len, head_dim) * 2) - 1,
        (torch.rand(batch, n_kv_heads, max_seq_len, head_dim) * 2) - 1,
        # Match attn mask shape from falcon40b model_preprocessing
        # [seqlen, n_heads, batch, max_seq_len]
        (torch.rand(seq_len, 1, batch, max_seq_len).expand(-1, n_heads, -1, -1) * 2) - 1,
    ]

    layer_num = 0

    base_url = "layers"

    # PyTorch output --------------------------------------------------------------------
    pytorch_model = PytorchLlamaSDPA(hugging_face_reference_model, layer_num)
    pytorch_out = [pytorch_model(*inp).squeeze()]

    # TT hardware execution -------------------------------------------------------------
    tt_model = TtLlamaSDPA(
        device,
        state_dict,
        base_url,
        layer_num,
        hidden_dim,
        n_heads,
        n_kv_heads,
        model_config,
        tt_cache_path=None,
    )

    tt_inp = [torch2tt_tensor(i, device) for i in inp]
    # attn_mask = tt_inp[3]
    # attn_mask_mem_config = model_config["ATTN_MASK_MEMCFG"]
    # attn_mask_shard_shape = attn_mask_mem_config.shard_spec.shape
    # attn_mask_shard_shape[-1] = max_seq_len
    # attn_mask_mem_config.shard_spec.shape = attn_mask_shard_shape

    # attn_mask = ttnn.interleaved_to_sharded(attn_mask, attn_mask_mem_config)
    # tt_inp[3] = attn_mask

    tt_out = tt_model(*tt_inp)
    tt_out = [tt2torch_tensor(tt_out).permute(2, 0, 1, 3).squeeze()]  # [batch, seq_len, hidden_dim]

    # check outputs ----------------------------------------------------------------------

    for i in range(len(pytorch_out)):
        logger.info(comp_allclose(pytorch_out[i], tt_out[i]))

    does_pass = True
    for i in range(len(pytorch_out)):
        out_pass, output_pcc = comp_pcc(pytorch_out[i], tt_out[i], pcc)
        # Check each shape matches
        assert pytorch_out[i].shape == tt_out[i].shape
        logger.info(f"PCC value: {output_pcc}")
        does_pass = does_pass and out_pass

        mae = torch.mean(torch.abs(pytorch_out[i] - tt_out[i]))
        logger.info(f"MAE: {mae}")

        max_incorrect = torch.max(torch.abs(pytorch_out[i] - tt_out[i]))
        logger.info(f"Max incorrect: {max_incorrect}")

        max_gt = torch.max(torch.abs(pytorch_out[i]))
        logger.info(f"Max ground truth: {max_gt}")

    if does_pass:
        logger.info("Llama QKV output Passed!")
    else:
        logger.warning("Llama QKV output Failed!")
        assert does_pass, f"PCC value is lower than {pcc}"


@pytest.mark.parametrize(
    "model_version, batch, seq_len, pcc",
    (
        (
            "llama-2-70B",
            32,
            1,
            0.98,
        ),
    ),
)
@pytest.mark.parametrize("model_config_str", ("BFLOAT16-DRAM",))
def test_LlamaSDPA_inference(
    model_version,
    batch,
    seq_len,
    pcc,
    model_config_str,
    # model_location_generator,
    device,
):
    model_config = get_model_config(model_config_str)
    # tt_cache_path = get_tt_cache_path(model_version)

    run_test_LlamaSDPA(
        device,
        model_version,
        batch,
        seq_len,
        pcc,
        model_config,
        # tt_cache_path,
        # model_location_generator,
    )
