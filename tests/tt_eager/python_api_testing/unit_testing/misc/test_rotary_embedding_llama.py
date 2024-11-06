# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import torch
import ttnn

from models.demos.t3000.llama2_70b.reference.llama.llama.model import precompute_freqs_cis, apply_rotary_emb
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_pcc,
)
from models.utility_functions import skip_for_grayskull, skip_for_blackhole
from models.demos.t3000.llama2_70b.tt.llama_common import precompute_freqs, freqs_to_rotation_matrix, gather_rotary_emb
from models.demos.t3000.llama2_70b.tt.llama_rope import TtLlamaRotarySetup

MAX_SEQ_LEN = 128 * 1024


def get_rotation_mat(dhead, end, start_pos, seqlen, batch):
    cos, sin = precompute_freqs(dhead, end)
    rot_mat = freqs_to_rotation_matrix(cos, sin)
    position_ids = torch.ones(seqlen, batch, dtype=torch.long) * start_pos
    rot_emb = gather_rotary_emb(rot_mat, position_ids)
    return rot_emb


class TtLlamaRotary(torch.nn.Module):
    def __init__(
        self,
        device,
        head_dim: int,
        mode: str,
        datatype=ttnn.bfloat16,
    ):
        super().__init__()

        self.head_dim = head_dim
        self.device = device
        self.mode = mode

        self.transformation_mat = ttnn.from_torch(
            get_rot_transformation_mat(dhead=ttnn.TILE_SIZE), device=device, layout=ttnn.TILE_LAYOUT, dtype=datatype
        )

    def apply_rotary(self, q, k, cos, sin):
        # n_head = 8 for Q
        # n_head = 1 for K

        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            # math_fidelity=ttnn.MathFidelity.LoFi,
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=True,
            fp32_dest_acc_en=(True if self.head_dim <= 128 else False),
            packer_l1_acc=True,
        )
        # q = torch.ones(1, 32, 32, 32)
        # q = ttnn.from_torch(q, device=self.device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        # q_mem_config = ttnn.create_sharded_memory_config(
        #     shape=(ttnn.TILE_SIZE, 32),
        #     core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
        #     strategy=ttnn.ShardStrategy.HEIGHT,
        #     orientation=ttnn.ShardOrientation.ROW_MAJOR,
        #     use_height_and_width_as_shard_shape=True,
        # )
        # q = ttnn.interleaved_to_sharded(q, q_mem_config)

        # k = torch.ones(1, 32, 32, 32)
        # k = ttnn.from_torch(k, device=self.device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        # k_mem_config = ttnn.create_sharded_memory_config(
        #     shape=(ttnn.TILE_SIZE, 32),
        #     core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 4), ttnn.CoreCoord(7,7))}),
        #     strategy=ttnn.ShardStrategy.HEIGHT,
        #     orientation=ttnn.ShardOrientation.ROW_MAJOR,
        #     use_height_and_width_as_shard_shape=True,
        # )
        # k = ttnn.interleaved_to_sharded(k, k_mem_config)

        # cos = torch.concat((torch.ones(1, 32, 1, 32), 1*torch.ones(1, 32, 1, 32)), dim=1)
        # cos = ttnn.from_torch(cos, device=self.device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        # cos_mem_config = ttnn.create_sharded_memory_config(
        #     shape=(32, 32),
        #     core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}),
        #     strategy=ttnn.ShardStrategy.HEIGHT,
        #     orientation=ttnn.ShardOrientation.ROW_MAJOR,
        #     use_height_and_width_as_shard_shape=True,
        # )
        # cos = ttnn.interleaved_to_sharded(cos, cos_mem_config)

        # sin = torch.ones(1, 64, 1, 32)
        # sin = ttnn.from_torch(sin, device=self.device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        # sin_mem_config = ttnn.create_sharded_memory_config(
        #     shape=(32, 32),
        #     core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}),
        #     strategy=ttnn.ShardStrategy.HEIGHT,
        #     orientation=ttnn.ShardOrientation.ROW_MAJOR,
        #     use_height_and_width_as_shard_shape=True,
        # )
        # sin = ttnn.to_memory_config(sin, sin_mem_config)

        # self.transformation_mat = torch.ones(1, 1, 64 * 32, 32)
        # self.transformation_mat = ttnn.from_torch(
        #     self.transformation_mat, device=self.device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        # )
        # self.transformation_mat_mem_config = ttnn.create_sharded_memory_config(
        #     shape=(32, 32),
        #     core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}),
        #     strategy=ttnn.ShardStrategy.HEIGHT,
        #     orientation=ttnn.ShardOrientation.ROW_MAJOR,
        #     use_height_and_width_as_shard_shape=True,
        # )
        # self.transformation_mat = ttnn.interleaved_to_sharded(self.transformation_mat, self.transformation_mat_mem_config)

        rotary_output1, rotary_output2 = ttnn.experimental.rotary_embedding_llama_fused_qk(
            q,
            k,
            cos,
            sin,
            self.transformation_mat,
            compute_kernel_config=compute_kernel_config,
        )
        breakpoint()

        return rotary_output1, rotary_output2

    def forward(self, xq, xk, cos, sin):
        xq, xk = self.apply_rotary(xq, xk, cos, sin)
        # xk = self.apply_rotary(xk, cos, sin)
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


def get_rot_transformation_mat(dhead):
    rot_emb_matrix = torch.zeros(1, 1, dhead, dhead)
    rot_emb_matrix[..., torch.arange(0, dhead, 2), torch.arange(1, dhead, 2)] = 1
    rot_emb_matrix[..., torch.arange(1, dhead, 2), torch.arange(0, dhead, 2)] = -1
    return rot_emb_matrix


def compute_gather_cos_sin(dhead, end, position_ids):
    cos, sin = precompute_freqs(dhead, end)
    position_id_expanded = position_ids.unsqueeze(1).expand(-1, cos.shape[-1])
    cos = cos.gather(0, position_id_expanded)
    sin = sin.gather(0, position_id_expanded)
    cos = torch.stack([cos, cos], dim=-1).flatten(-2).unsqueeze(0).unsqueeze(0)
    sin = torch.stack([sin, sin], dim=-1).flatten(-2).unsqueeze(0).unsqueeze(0)
    return cos, sin


def run_test_rotary_embedding_llama(
    device,
    batch,
    seq_len,
    pcc,
    n_heads,
    n_kv_heads,
    head_dim,
    max_seq_len,
    datatype=ttnn.bfloat16,
):
    # Prepare input
    torch.manual_seed(0)
    mode = "decode" if seq_len == 1 else "prefill"

    if mode == "decode":
        max_seq_len = MAX_SEQ_LEN

    inp = [
        (torch.rand(batch, n_heads, seq_len, head_dim) * 2) - 1,
        (torch.rand(batch, n_kv_heads, seq_len, head_dim) * 2) - 1,
    ]

    # To test with different position ids, assume that batch
    # dimension is the seq len dimension when passing inputs to torch
    if mode == "decode":
        inp = [x.permute(2, 1, 0, 3) for x in inp]
        # inp: [seq_len, n_heads, batch, head_dim]

    freqs_cis = precompute_freqs_cis(
        # Note that self.params.max_seq_len is multiplied by 2 because the token limit for the Llama 2 generation of models is 4096.
        # Adding this multiplier instead of using 4096 directly allows for dynamism of token lengths while training or fine-tuning.
        head_dim,
        max_seq_len * 2,  # In decode, precompute for all positions
    )  # torch.Size([8192, 64])

    start_pos = 0  # Must pick non-zero start pos to get non-zero freqs_cis

    position_ids = torch.arange(batch) if mode == "decode" else slice(start_pos, start_pos + seq_len)

    freqs_cis = freqs_cis[position_ids]

    # PyTorch Ground Truth output --------------------------------------------------------------------
    torch_xq = inp[0].transpose(1, 2)
    torch_xk = inp[1].transpose(1, 2)

    torch_xq, torch_xk = apply_rotary_emb(torch_xq, torch_xk, freqs_cis=freqs_cis)

    torch_xq = torch_xq.transpose(1, 2)
    torch_xk = torch_xk.transpose(1, 2)

    pytorch_out = (torch_xq, torch_xk)

    # TT hardware / Modified PyTorch execution -------------------------------------------------------------
    tt_model = TtLlamaRotary(device, head_dim, mode, datatype)

    if mode == "decode":
        rope_setup_decode = TtLlamaRotarySetup(device, head_dim, max_seq_len)
        cos, sin = rope_setup_decode.get_rot_mats(position_ids)
        tt_model.transformation_mat = rope_setup_decode.transformation_mat

        # For decode, TTNN expects inputs to be [1, batch, nh, dhead]
        inp = [x.transpose(1, 2) for x in inp]
        # inp: [seq_len, batch, n_heads, head_dim]

        grid = (
            ttnn.num_cores_to_corerangeset(batch, rope_setup_decode.core_grid, row_wise=True).bounding_box().grid_size()
        )

        q_input_mem_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, head_dim),
            core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        k_input_mem_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, head_dim),
            core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 4), ttnn.CoreCoord(7, 7))}),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        tt_inp = [
            ttnn.from_torch(
                inp[0], device=device, dtype=datatype, memory_config=q_input_mem_config, layout=ttnn.TILE_LAYOUT
            ),
            ttnn.from_torch(
                inp[1], device=device, dtype=datatype, memory_config=k_input_mem_config, layout=ttnn.TILE_LAYOUT
            ),
        ]
        tt_inp += [cos, sin]  # Append cos and sin to the input list
    else:
        cos, sin = compute_gather_cos_sin(
            dhead=head_dim,
            end=max_seq_len * 2,
            position_ids=torch.arange(start_pos, start_pos + seq_len),
        )

        tt_inp = [inp[0], inp[1], cos, sin]
        tt_inp = [ttnn.from_torch(i, device=device, dtype=datatype, layout=ttnn.TILE_LAYOUT) for i in tt_inp]

    tt_out = tt_model(*tt_inp)
    tt_out = [ttnn.to_torch(tt_out_tensor) for tt_out_tensor in tt_out]

    if mode == "decode":
        tt_out = [x.transpose(1, 2) for x in tt_out]
        # tt_out: [seq_len, n_heads, batch, head_dim]

    # check outputs ----------------------------------------------------------------------
    assert len(pytorch_out) == len(tt_out), "Lengths of pytorch and tt outputs do not match!"
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


@skip_for_blackhole("Requires eth connected devices to run, only single chip BH available. See #12349")
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "batch, seq_len",
    (
        (1, 32),  # To test single core implementation
        (1, 128),
        (1, 256),
        (1, 512),
        (1, 2048),
        (1, 3 * 1024),  # To test non-power of 2
        (1, 4096),
        (1, 8192),
        (1, 16384),
        (1, 128 * 1024),
        # (64, 1),
        (32, 1),
        # (16, 1),
        # (8, 1),
        # (1, 1),
    ),
    ids=(
        "prefill_32",
        "prefill_128",
        "prefill_256",
        "prefill_512",
        "prefill_2k",
        "prefill_3k",
        "prefill_4k",
        "prefill_8k",
        "prefill_16k",
        "prefill_128k",
        # "decode_64",
        "decode_32",
        # "decode_16",
        # "decode_8",
        # "decode_1",
    ),
)
@pytest.mark.parametrize(
    "n_heads, n_kv_heads, head_dim",
    (
        (1, 1, 32),
        # (8, 1, 128),
        # (11, 3, 128),
        # (71, 32, 64),
        # (8, 1, 96),
        # (8, 1, 256),
    ),
)
@pytest.mark.parametrize("datatype", (ttnn.bfloat16,))
@pytest.mark.parametrize("pcc", (0.9997,))
def test_rotary_embedding_llama(
    batch,
    seq_len,
    n_heads,
    n_kv_heads,
    head_dim,
    datatype,
    pcc,
    device,
):
    compute_grid_size = device.compute_with_storage_grid_size()
    if compute_grid_size.x < 8 or compute_grid_size.y < 8:
        pytest.skip(f"Requires grid size of at least {(8, 8)} to run")

    if seq_len == 128 * 1024 and (n_heads, n_kv_heads, head_dim) != (8, 1, 128):
        pytest.skip("Only testing for (8, 1, 128) due to time constraints")

    if seq_len == 1 and (n_heads > ttnn.TILE_SIZE or n_kv_heads > ttnn.TILE_SIZE):
        pytest.skip("n_heads or n_kv_heads cannot be greater than ttnn.TILE_SIZE for decode mode")

    max_seq_len = max(4096, seq_len)

    run_test_rotary_embedding_llama(device, batch, seq_len, pcc, n_heads, n_kv_heads, head_dim, max_seq_len, datatype)

    # shift input/output tensor by creating very small tensor between loop
    inp = torch.randn(1, 1, 32, 32)
    test_tensor = (
        ttnn.Tensor(
            inp.reshape(-1).tolist(),
            inp.shape,
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        .to(ttnn.TILE_LAYOUT)
        .to(device)
    )


@skip_for_blackhole("Requires eth connected devices to run, only single chip BH available. See #12349")
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "batch, seq_len",
    (
        (1, 2048),
        (1, 4096),
        (1, 8192),
        (64, 1),
        (32, 1),
        (16, 1),
        (8, 1),
        (1, 1),
    ),
    ids=(
        "prefill_2k",
        "prefill_4k",
        "prefill_8k",
        "decode_64",
        "decode_32",
        "decode_16",
        "decode_8",
        "decode_1",
    ),
)
@pytest.mark.parametrize(
    "n_heads, n_kv_heads, head_dim",
    ((8, 1, 128),),
)
@pytest.mark.parametrize("datatype", (ttnn.bfloat16,))
@pytest.mark.parametrize("pcc", (0.9997,))
def test_rotary_embedding_llama_with_program_cache(
    batch,
    seq_len,
    n_heads,
    n_kv_heads,
    head_dim,
    datatype,
    pcc,
    device,
    use_program_cache,
):
    compute_grid_size = device.compute_with_storage_grid_size()
    if compute_grid_size.x < 8 or compute_grid_size.y < 8:
        pytest.skip(f"Requires grid size of at least {(8, 8)} to run")

    max_seq_len = max(4096, seq_len)

    mode = "decode" if seq_len == 1 else "prefill"

    cache_tensors = []
    for _ in range(3):
        run_test_rotary_embedding_llama(
            device, batch, seq_len, pcc, n_heads, n_kv_heads, head_dim, max_seq_len, datatype
        )

        # shift input/output tensor by creating very small tensor between loop
        inp = torch.randn(1, 1, 32, 32)
        test_tensor = (
            ttnn.Tensor(
                inp.reshape(-1).tolist(),
                inp.shape,
                ttnn.bfloat16,
                ttnn.ROW_MAJOR_LAYOUT,
            )
            .to(ttnn.TILE_LAYOUT)
            .to(device)
        )

        cache_tensors.append(test_tensor)

    num_ops = 2  # 2 * rope
    if mode == "decode":
        num_ops += 4  # embedding + transpose + pad + interleaved_to_sharded

        # When batch size is 1, transpose is a no-op
        if batch == 1:
            num_ops -= 1

    assert device.num_program_cache_entries() == num_ops
