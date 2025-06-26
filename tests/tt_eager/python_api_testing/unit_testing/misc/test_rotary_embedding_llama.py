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
from models.utility_functions import skip_for_grayskull, skip_for_blackhole, nearest_32, skip_for_wormhole_b0
from models.tt_transformers.tt.common import (
    precompute_freqs,
    get_rot_transformation_mat,
)
from models.tt_transformers.tt.rope import RotarySetup
from models.demos.llama3_subdevices.tt.llama_rope import TtLlamaRotarySetup

MAX_SEQ_LEN = 128 * 1024


class TtLlamaRotary(torch.nn.Module):
    def __init__(
        self,
        device,
        head_dim: int,
        mode: str,
        datatype=ttnn.bfloat16,
        fuse_qk=False,
    ):
        super().__init__()

        self.head_dim = head_dim
        self.device = device
        self.mode = mode
        self.fuse_qk = fuse_qk

        self.transformation_mat = ttnn.from_torch(
            get_rot_transformation_mat(dhead=ttnn.TILE_SIZE), device=device, layout=ttnn.TILE_LAYOUT, dtype=datatype
        )

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            # math_fidelity=ttnn.MathFidelity.LoFi,
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=True,
            fp32_dest_acc_en=(True if self.head_dim <= 128 else False),
            packer_l1_acc=True,
        )

    def apply_rotary(self, x, cos, sin):
        # n_head = 8 for Q
        # n_head = 1 for K

        rotary_output = ttnn.experimental.rotary_embedding_llama(
            x,
            cos,
            sin,
            self.transformation_mat,
            is_decode_mode=self.mode == "decode",
            compute_kernel_config=self.compute_kernel_config,
        )

        return rotary_output

    def apply_fused_rotary(self, q, k, cos, sin):
        # n_head = 8 for Q
        # n_head = 1 for K
        rotary_output_q, rotary_output_k = ttnn.experimental.rotary_embedding_llama_fused_qk(
            q,
            k,
            cos,
            sin,
            self.transformation_mat,
            compute_kernel_config=self.compute_kernel_config,
        )

        return rotary_output_q, rotary_output_k

    def forward(self, xq, xk, cos, sin):
        if self.fuse_qk:
            xq, xk = self.apply_fused_rotary(xq, xk, cos, sin)
        else:
            xq = self.apply_rotary(xq, cos, sin)
            xk = self.apply_rotary(xk, cos, sin)
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


def compute_gather_cos_sin(dhead, end, position_ids):
    cos, sin = precompute_freqs(
        dhead, end, theta=10000.0, scale_factor=None, orig_context_len=131072
    )  # Using reference defaults (no scaling)
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
    fuse_qk=False,
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
    tt_model = TtLlamaRotary(device, head_dim, mode, datatype, fuse_qk)

    if mode == "decode":
        # For decode, TTNN expects inputs to be [1, batch, nh, dhead]
        inp = [x.transpose(1, 2) for x in inp]
        # inp: [seq_len, batch, n_heads, head_dim]

        if fuse_qk:
            # Set up rope with 2 * batch size (for fused qk) (no scaling)
            rope_setup_decode = RotarySetup(
                device, batch * 2, head_dim, max_seq_len, rope_theta=10000, scale_factor=None, orig_context_len=131072
            )
            tt_model.transformation_mat = rope_setup_decode.transformation_mat
            cos, sin = rope_setup_decode.get_rot_mats(position_ids.repeat(2))

            assert (
                batch % 8 == 0 or batch == 1
            ), "Batch size must be a multiple of 8 or less than 8 for fused_qk rotary embedding"
            if batch == 1:
                q_core_grid_start = (0, 0)
                q_core_grid_end = (0, 0)
                k_core_grid_start = (1, 0)
                k_core_grid_end = (1, 0)
            else:
                q_core_grid_start = (0, 0)
                q_core_grid_end = ((batch - 1) % 8, (batch // 8) - 1)
                k_core_grid_start = (0, (batch // 8))
                k_core_grid_end = ((batch - 1) % 8, (batch // 8) * 2 - 1)
            q_input_mem_config = ttnn.create_sharded_memory_config(
                shape=(nearest_32(n_heads), head_dim),
                core_grid=ttnn.CoreRangeSet(
                    {ttnn.CoreRange(ttnn.CoreCoord(*q_core_grid_start), ttnn.CoreCoord(*q_core_grid_end))}
                ),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            k_input_mem_config = ttnn.create_sharded_memory_config(
                shape=(nearest_32(n_kv_heads), head_dim),
                core_grid=ttnn.CoreRangeSet(
                    {ttnn.CoreRange(ttnn.CoreCoord(*k_core_grid_start), ttnn.CoreCoord(*k_core_grid_end))}
                ),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            input_mem_configs = [q_input_mem_config, k_input_mem_config]

        else:
            # Set up rope with batch size (no scaling)
            rope_setup_decode = RotarySetup(
                device, batch, head_dim, max_seq_len, rope_theta=10000, scale_factor=None, orig_context_len=131072
            )

            tt_model.transformation_mat = rope_setup_decode.transformation_mat
            cos, sin = rope_setup_decode.get_rot_mats(position_ids)

            grid = ttnn.num_cores_to_corerangeset(batch, rope_setup_decode.core_grid, row_wise=True)
            input_mem_configs = [
                ttnn.create_sharded_memory_config(
                    shape=(ttnn.TILE_SIZE, head_dim),
                    core_grid=grid,
                    strategy=ttnn.ShardStrategy.HEIGHT,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                for _ in range(len(inp))
            ]

        tt_inp = [
            ttnn.from_torch(
                x, device=device, dtype=datatype, memory_config=input_mem_configs[i], layout=ttnn.TILE_LAYOUT
            )
            for i, x in enumerate(inp)
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


def run_test_row_major_rotary_embedding_llama(
    device,
    batch,
    seq_len,
    pcc,
    n_heads,
    n_kv_heads,
    head_dim,
    max_seq_len,
    datatype=ttnn.bfloat16,
    fuse_qk=False,
):
    torch.manual_seed(0)

    max_seq_len = MAX_SEQ_LEN

    inp = [
        (torch.rand(seq_len, batch, n_heads, head_dim) * 2) - 1,
        (torch.rand(seq_len, batch, n_kv_heads, head_dim) * 2) - 1,
    ]

    freqs_cis = precompute_freqs_cis(head_dim, max_seq_len * 2)

    start_pos = 0  # Must pick non-zero start pos to get non-zero freqs_cis

    position_ids = torch.arange(batch)

    freqs_cis = freqs_cis[position_ids]

    # PyTorch Ground Truth output --------------------------------------------------------------------
    torch_xq = inp[0]
    torch_xk = inp[1]

    torch_xq, torch_xk = apply_rotary_emb(torch_xq, torch_xk, freqs_cis=freqs_cis)

    pytorch_out = (torch_xq, torch_xk)

    # Set up rope with 2 * batch size (for fused qk) (no scaling)
    rope_setup_decode = TtLlamaRotarySetup(
        device, batch, head_dim, max_seq_len, rope_theta=10000, use_scaled_rope=False, scale_factor=None
    )
    transformation_mat = rope_setup_decode.transformation_mat
    cos, sin = rope_setup_decode.get_rm_rot_mats(position_ids)
    sub_core_grids = rope_setup_decode.sub_core_grids

    q_core_grid = ttnn.num_cores_to_corerangeset_in_subcoregrids(ttnn.CoreCoord(1, 0), 8, sub_core_grids, row_wise=True)
    k_core_grid = ttnn.num_cores_to_corerangeset_in_subcoregrids(ttnn.CoreCoord(3, 2), 8, sub_core_grids, row_wise=True)

    q_input_mem_config = ttnn.create_sharded_memory_config(
        shape=(n_heads, head_dim),
        core_grid=q_core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    k_input_mem_config = ttnn.create_sharded_memory_config(
        shape=(n_kv_heads, head_dim),
        core_grid=k_core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    input_mem_configs = [q_input_mem_config, k_input_mem_config]

    tt_inp = [
        ttnn.from_torch(
            x,
            device=device,
            dtype=datatype,
            memory_config=input_mem_configs[i],
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(device, dims=(None, 1), mesh_shape=list(device.shape)),
        )
        for i, x in enumerate(inp)
    ]

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    rotary_output_q, rotary_output_k = ttnn.experimental.rotary_embedding_llama_fused_qk(
        tt_inp[0],
        tt_inp[1],
        cos,
        sin,
        transformation_mat,
        compute_kernel_config=compute_kernel_config,
    )
    tt_out = [
        ttnn.to_torch(rotary_output_q, mesh_composer=ttnn.ConcatMesh2dToTensor(device, dims=(0, 1), mesh_shape=(8, 4)))[
            0, ...
        ].unsqueeze(0)
    ]
    tt_out += [
        ttnn.to_torch(rotary_output_k, mesh_composer=ttnn.ConcatMesh2dToTensor(device, dims=(0, 1), mesh_shape=(8, 4)))[
            0, ...
        ].unsqueeze(0)
    ]
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
        (64, 1),
        (32, 1),
        (15, 1),
        (8, 1),
        (1, 1),
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
        "decode_64",
        "decode_32",
        "decode_15",
        "decode_8",
        "decode_1",
    ),
)
@pytest.mark.parametrize(
    "n_heads, n_kv_heads, head_dim",
    (
        (8, 1, 64),
        (8, 1, 128),
        (11, 3, 128),
        (71, 32, 64),
        (8, 1, 96),
        (8, 1, 256),
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
        num_ops += 4  # untilize cos/sin + embedding + transpose + interleaved_to_sharded

        if batch % ttnn.TILE_SIZE != 0:
            num_ops += 1  # slice

    assert device.num_program_cache_entries() == num_ops


def apply_rotary_emb_qk_real(
    x: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> torch.Tensor:
    """
    Ground truth implementation which is required when cos/sin have num_heads > 1.
    """
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    # Apply rotation
    cos_part = x_even * freqs_cos - x_odd * freqs_sin
    sin_part = x_even * freqs_sin + x_odd * freqs_cos

    out = torch.stack([cos_part, sin_part], dim=-1).flatten(-2)
    return out


@skip_for_blackhole("Requires eth connected devices to run, only single chip BH available. See #12349")
@pytest.mark.parametrize(
    "batch, seq_len",
    (
        (1, 2048),
        (1, 3 * 1024),  # To test non-power of 2
        (1, 4096),
        (2, 1024),  # Test batch > 1
    ),
    ids=("prefill_2048", "prefill_3072", "prefill_4096", "batch2_1024"),
)
@pytest.mark.parametrize(
    "n_heads, head_dim",
    (
        (24, 128),
        (3, 128),
    ),
)
@pytest.mark.parametrize("datatype", (ttnn.bfloat16,))
@pytest.mark.parametrize("pcc", (0.9997,))
def test_rotary_embedding_llama_per_head(
    batch,
    seq_len,
    n_heads,
    head_dim,
    datatype,
    pcc,
    device,
):
    """
    This test is for Mochi-style rotary embeddings, where attention is MHA
    and each head has independent rotary embeddings.
    """
    x = torch.randn(batch, n_heads, seq_len, head_dim)
    cos = torch.randn(1, n_heads, seq_len, head_dim // 2)
    sin = torch.randn(1, n_heads, seq_len, head_dim // 2)

    # ttnn implementation requires stacked cos, sin
    cos_reshape = torch.stack([cos, cos], dim=-1).flatten(-2)
    sin_reshape = torch.stack([sin, sin], dim=-1).flatten(-2)
    trans_mat = get_rot_transformation_mat(None)

    # Apply ground truth implementation with unstacked cos, sin
    gt = apply_rotary_emb_qk_real(x, cos, sin)
    x_tt = ttnn.from_torch(x, device=device, dtype=datatype, layout=ttnn.TILE_LAYOUT)
    cos_tt = ttnn.from_torch(cos_reshape, device=device, dtype=datatype, layout=ttnn.TILE_LAYOUT)
    sin_tt = ttnn.from_torch(sin_reshape, device=device, dtype=datatype, layout=ttnn.TILE_LAYOUT)
    trans_mat_tt = ttnn.from_torch(trans_mat, device=device, dtype=datatype, layout=ttnn.TILE_LAYOUT)

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=(True if head_dim <= 128 else False),
        packer_l1_acc=True,
    )

    out_tt = ttnn.experimental.rotary_embedding_llama(
        x_tt, cos_tt, sin_tt, trans_mat_tt, is_decode_mode=False, compute_kernel_config=compute_kernel_config
    )

    out = ttnn.to_torch(out_tt)
    passing, out_pcc = comp_pcc(gt, out, pcc)
    logger.info(out_pcc)
    assert passing
