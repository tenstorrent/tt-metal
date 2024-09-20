# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import torch
from torch import nn
import ttnn

from models.demos.t3000.llama2_70b.reference.llama.llama import Llama
from models.demos.t3000.llama2_70b.reference.llama.llama.model import precompute_freqs_cis, apply_rotary_emb
from models.demos.t3000.llama2_70b.tt.model_config import (
    get_model_config,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, skip_for_grayskull, get_devices_for_t3000
from models.demos.t3000.llama2_70b.tt.llama_common import (
    get_llama_path,
    extract_pcc_from_log,
    MAX_SEQ_LEN,
    BASE_URL,
    UNIT_TEST_N_LAYER,
    UNIT_TEST_LAYER_NUM,
    UNIT_TEST_START_POS,
    UNIT_TEST_GENERATION_LENGTH,
)

from models.demos.t3000.llama2_70b.tt.llama_common import precompute_freqs, freqs_to_rotation_matrix, gather_rotary_emb


def get_rotation_mat(dhead, end, start_pos, seqlen, batch):
    cos, sin = precompute_freqs(dhead, end)
    rot_mat = freqs_to_rotation_matrix(cos, sin)
    position_ids = torch.ones(seqlen, batch, dtype=torch.long) * start_pos
    rot_emb = gather_rotary_emb(rot_mat, position_ids)
    return rot_emb


def get_rotation_mat_prefill(dhead, end, start_pos, seqlen, batch):
    cos, sin = precompute_freqs(dhead, end)
    rot_mat = freqs_to_rotation_matrix(cos, sin)
    position_ids = torch.ones(batch, seqlen, dtype=torch.long) * torch.arange(start_pos, start_pos + seqlen).unsqueeze(
        0
    )
    rot_emb = gather_rotary_emb(rot_mat, position_ids)
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
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = hidden_size // n_heads

    def forward(self, xq, xk, rot_mat):
        # xq = xq.transpose(1, 2).squeeze(0)
        # xk = xk.transpose(1, 2).squeeze(0)
        # rot_mat = rot_mat.squeeze(0)

        # xq = torch.bmm(
        #     xq,
        #     rot_mat,
        # )
        # xk = torch.bmm(
        #     xk,
        #     rot_mat,
        # )
        # xq = xq.unsqueeze(0).transpose(1, 2)
        # xk = xk.unsqueeze(0).transpose(1, 2)

        xq = ttnn.pad(xq, [1, 32, 128, self.head_dim], [0, 0, 0, 0], 0.0)
        xq = ttnn.transpose(xq, 1, 2)

        xq = ttnn.matmul(
            xq,
            rot_mat,
            # compute_kernel_config=self.model_config["ROT_MAT_COMPUTE_KERNEL_CONFIG"]
        )

        xq = ttnn.transpose(xq, 1, 2)
        xq = ttnn.slice(
            xq,
            [0, 0, 0, 0],
            [1, 8, 128, self.head_dim],
        )

        xk = ttnn.pad(xk, [1, 32, 128, self.head_dim], [0, 0, 0, 0], 0.0)

        xk = ttnn.transpose(xk, 1, 2)

        xk = ttnn.matmul(
            xk,
            rot_mat,
            # compute_kernel_config=self.model_config["ROT_MAT_COMPUTE_KERNEL_CONFIG"],
        )

        xk = ttnn.transpose(xk, 1, 2)
        xk = ttnn.slice(
            xk,
            [0, 0, 0, 0],
            [1, 1, 128, self.head_dim],
        )

        return xq, xk


class PytorchLlamaRotaryModel(torch.nn.Module):
    def __init__(self, hf_reference_model, layer_num):
        super().__init__()
        self.n_heads = hf_reference_model.params.n_heads
        self.n_kv_heads = hf_reference_model.params.n_kv_heads
        self.head_dim = hf_reference_model.params.dim // self.n_heads

    def forward(self, xq, xk, freqs_cis):
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)

        return xq, xk


def run_test_LlamaReshape(
    devices,
    batch,
    seq_len,
    pcc,
    model_config,
    n_devices,
    emulated=False,
):
    # Prepare paths and devices
    devices, ckpt_dir, tokenizer_path, cache_path = get_llama_path(devices, model_config, n_devices, emulated)
    device = devices[0]

    hugging_face_reference_model = Llama.build(
        ckpt_dir, tokenizer_path, max_seq_len=4096, max_batch_size=1, n_layers=1, skip_model_load=False
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
        (torch.rand(batch, 8, seq_len, head_dim) * 2) - 1,
        (torch.rand(batch, 1, seq_len, head_dim) * 2) - 1,
    ]
    freqs_cis = precompute_freqs_cis(
        # Note that self.params.max_seq_len is multiplied by 2 because the token limit for the Llama 2 generation of models is 4096.
        # Adding this multiplier instead of using 4096 directly allows for dynamism of token lengths while training or fine-tuning.
        hidden_dim // n_heads,
        configuration.max_seq_len * 2,
    )  # torch.Size([8192, 64])

    start_pos = 0  # Must pick non-zero start pos to get non-zero freqs_cis
    freqs_cis = freqs_cis[start_pos : start_pos + seq_len]
    # freqs_cis = freqs_cis.expand(batch, -1)  # torch.Size([32, 64])
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
    )

    rot_mat = get_rotation_mat_prefill(
        dhead=head_dim, end=configuration.max_seq_len * 2, start_pos=start_pos, seqlen=seq_len, batch=batch
    )
    inp[2] = rot_mat
    tt_inp = [torch2tt_tensor(i, device) for i in inp]

    tt_out = tt_model(*tt_inp)
    tt_out = [tt2torch_tensor(tt_out_tensor) for tt_out_tensor in tt_out]

    # check outputs ----------------------------------------------------------------------

    # for i in range(2):
    #     logger.info(comp_allclose(pytorch_out[i], tt_out[i]))

    does_pass = True
    for i in range(2):
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


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "n_devices, emulated",
    (
        (8, False),
        (8, True),
        (32, True),
    ),
    ids=(
        "8chip-T3000",
        "8chip-emulated",
        "32chip-emulated",
    ),
)
@pytest.mark.parametrize(
    "batch, seq_len",
    (
        (32, 1),
        (1, 128),
    ),
    ids=("decode", "prefill"),
)
@pytest.mark.parametrize("model_config_str, pcc", (("BFLOAT16-DRAM", 0.9997),))
def test_LlamaAttention_inference(
    batch,
    seq_len,
    pcc,
    model_config_str,
    n_devices,
    all_devices,
    emulated,
):
    devices = get_devices_for_t3000(all_devices, num_devices=n_devices if not emulated else 1)
    model_config = get_model_config(model_config_str, num_devices=n_devices, seq_len=seq_len)
    compute_grid_size = devices[0].compute_with_storage_grid_size()
    if len(devices) < n_devices and not emulated:
        pytest.skip(f"Requires at {n_devices} devices to run")
    if compute_grid_size.x < model_config["MAX_GRID_SIZE"][0] or compute_grid_size.y < model_config["MAX_GRID_SIZE"][1]:
        pytest.skip(f"Requires grid size of at least {model_config['MAX_GRID_SIZE']} to run")

    run_test_LlamaReshape(
        devices,
        batch,
        seq_len,
        pcc,
        model_config,
        n_devices,
        emulated,
    )
