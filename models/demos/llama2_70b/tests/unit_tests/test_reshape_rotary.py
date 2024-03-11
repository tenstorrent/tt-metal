# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger

import tt_lib
from models.demos.llama2_70b.reference.llama.llama import Llama
from models.demos.llama2_70b.reference.llama.llama.model import apply_rotary_emb, precompute_freqs_cis
from models.demos.llama2_70b.tt.model_config import get_model_config
from models.demos.llama2_70b.tt.llama_common import (
    precompute_freqs as tt_precompute_freqs,
    freqs_to_rotation_matrix,
    gather_rotary_emb as tt_gather_rotary_emb,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
from models.utility_functions import torch2tt_tensor, tt2torch_tensor


def get_rotation_mat(dhead, end, start_pos, seqlen, batch):
    cos, sin = tt_precompute_freqs(dhead, end)
    rot_mat = freqs_to_rotation_matrix(cos, sin)
    position_ids = torch.ones(seqlen, batch, dtype=torch.long) * start_pos
    rot_emb = tt_gather_rotary_emb(rot_mat, position_ids)
    return rot_emb


class TtLlamaRotary(torch.nn.Module):
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

    def forward(self, xq, xk, xv, rot_mat):
        seqlen = xq.shape()[0]
        bsz = xq.shape()[2]

        xqkv_fused = tt_lib.tensor.concat([xq, xk, xv], dim=-1)
        (
            q_heads,  # [seqlen, n_heads, bsz, head_dim]
            k_heads,  # [seqlen, n_kv_heads, bsz, head_dim]
            v_heads,  # [seqlen, n_kv_heads, bsz, head_dim]
        ) = tt_lib.tensor.nlp_create_qkv_heads(
            xqkv_fused, num_heads=self.n_heads, num_kv_heads=self.n_kv_heads, transpose_k_heads=False
        )

        # Have to put bsz back in dim 1 to match rot_mat shape
        q_heads = tt_lib.tensor.transpose(q_heads, 1, 2)
        k_heads = tt_lib.tensor.transpose(k_heads, 1, 2)

        q_heads = tt_lib.tensor.bmm(
            q_heads, rot_mat  # [seqlen, bsz, n_heads, head_dim]  # [1, bsz, head_dim, head_dim]
        )
        k_heads = tt_lib.tensor.bmm(
            k_heads, rot_mat  # [seqlen, bsz, n_kv_heads, head_dim]  # [1, bsz, head_dim, head_dim]
        )

        q_heads = tt_lib.tensor.transpose(q_heads, 1, 2)
        k_heads = tt_lib.tensor.transpose(k_heads, 1, 2)

        return q_heads, k_heads, v_heads


class PytorchLlamaRotaryModel(torch.nn.Module):
    def __init__(self, hf_reference_model, layer_num):
        super().__init__()
        self.n_heads = hf_reference_model.params.n_heads
        self.n_kv_heads = hf_reference_model.params.n_kv_heads
        self.head_dim = hf_reference_model.params.dim // self.n_heads

    def forward(self, xq, xk, xv, freqs_cis):
        seqlen = xq.size(0)
        bsz = xq.size(2)
        xq = xq.view(seqlen, bsz, self.n_heads, self.head_dim)
        xk = xk.view(seqlen, bsz, self.n_kv_heads, self.head_dim)
        xv = xv.view(seqlen, bsz, self.n_kv_heads, self.head_dim).transpose(1, 2)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)

        return xq, xk, xv


def run_test_LlamaReshape(
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
    torch.manual_seed(0)
    inp = [
        (torch.rand(seq_len, 1, batch, hidden_dim) * 2) - 1,
        (torch.rand(seq_len, 1, batch, int(head_dim * n_kv_heads)) * 2) - 1,
        (torch.rand(seq_len, 1, batch, int(head_dim * n_kv_heads)) * 2) - 1,
    ]
    freqs_cis = precompute_freqs_cis(
        # Note that self.params.max_seq_len is multiplied by 2 because the token limit for the Llama 2 generation of models is 4096.
        # Adding this multiplier instead of using 4096 directly allows for dynamism of token lengths while training or fine-tuning.
        hidden_dim // n_heads,
        configuration.max_seq_len * 2,
    )  # torch.Size([8192, 64])
    start_pos = 1000  # Must pick non-zero start pos to get non-zero freqs_cis
    freqs_cis = freqs_cis[start_pos : start_pos + 1]  # Imagine start_pos = 1, seqlen = 1
    freqs_cis = freqs_cis.expand(batch, -1)  # torch.Size([32, 64])

    inp.append(freqs_cis)
    layer_num = 0

    base_url = "layers"

    # PyTorch output --------------------------------------------------------------------
    pytorch_model = PytorchLlamaRotaryModel(hugging_face_reference_model, layer_num)
    pytorch_out = pytorch_model(*inp)

    # TT hardware execution -------------------------------------------------------------
    tt_model = TtLlamaRotary(
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

    rot_mat = get_rotation_mat(
        dhead=head_dim, end=configuration.max_seq_len * 2, start_pos=start_pos, seqlen=seq_len, batch=batch
    )
    inp[3] = rot_mat
    tt_inp = [torch2tt_tensor(i, device) for i in inp]

    tt_out = tt_model(*tt_inp)
    tt_out = [tt2torch_tensor(tt_out_tensor) for tt_out_tensor in tt_out]

    # check outputs ----------------------------------------------------------------------

    for i in range(3):
        logger.info(comp_allclose(pytorch_out[i], tt_out[i]))

    does_pass = True
    for i in range(3):
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
@pytest.mark.parametrize("model_config_str", ("BFLOAT16-SHARDED",))
def test_LlamaReshape_inference(
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

    run_test_LlamaReshape(
        device,
        model_version,
        batch,
        seq_len,
        pcc,
        model_config,
        # tt_cache_path,
        # model_location_generator,
    )
