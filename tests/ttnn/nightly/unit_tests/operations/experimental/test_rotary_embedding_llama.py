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
from models.common.utility_functions import skip_for_blackhole, nearest_32, skip_for_wormhole_b0
from models.tt_transformers.tt.common import (
    precompute_freqs,
    get_rot_transformation_mat,
)
from models.tt_transformers.tt.rope import RotarySetup
from models.demos.llama3_70b_galaxy.tt.llama_rope import TtLlamaRotarySetup

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
                device, batch * 2, head_dim, max_seq_len, rope_theta=10000, rope_scaling=None
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
            rope_setup_decode = RotarySetup(device, batch, head_dim, max_seq_len, rope_theta=10000, rope_scaling=None)

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
        # RotarySetup stores cos/sin in ROW MAJOR layout (no untilize is needed in embedding)
        num_ops += 3  # embedding + transpose + interleaved_to_sharded

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


def run_test_rotary_embedding_llama_prefill_cos_sin_and_trans_mat_sharding(
    trans_mat_sharded,
    cos_sin_sharded,
    seq_len,
    batch,
    num_heads,
    head_dim,
    datatype,
    pcc,
    device,
):
    """
    Validate height-sharded `cos`/`sin` and `trans_mat` support in Prefill mode. Also, validates
    decode (seq_len=1) mode emulated as prefill mode (batch, seq_len exchanged) for Deepseek.
    Input/Output are interleaved, cos/sin and trans_mat may be interleaved or sharded.

    `cos_sin_sharded` and `trans_mat_sharded` accept:
        0  -> interleaved (not sharded)
        -1 -> sharded across all device cores (tests globally-allocated CB fast path)
        N  -> sharded across exactly N cores (tests TensorAccessor reload path with fewer shards)
    """
    torch.manual_seed(42)
    max_seq_len = MAX_SEQ_LEN
    compute_grid_size = device.compute_with_storage_grid_size()
    device_num_cores = compute_grid_size.x * compute_grid_size.y

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=(True if head_dim <= 128 else False),
        packer_l1_acc=True,
    )

    trans_mat_torch = get_rot_transformation_mat(dhead=ttnn.TILE_SIZE)  # 32x32 tile
    head_dim_padded = nearest_32(head_dim)  # to allow sharded config with arbitrary (even) head_dim

    if seq_len == 1:
        # Decode emulation: input (1, num_heads, batch, head_dim)
        num_positions = batch
        position_ids = torch.arange(batch)
        # This code path emulates decode done as prefill instead by interchanging `seq_len=1` and `batch`
        # as prefill expects the order [batch, nheads, seq_len, head_dim] but deepseek decode
        # (to be emulated with prefill) is (to be) done with shape [seq_len (=1), nheads (=1), batch, head_dim]
        x_torch = (torch.rand(1, num_heads, batch, head_dim) * 2) - 1
    else:
        # Prefill: input (batch, num_heads, seq_len, head_dim), compare vs PyTorch (covers COS_SIN_SHARDED_RELOAD)
        num_positions = seq_len
        position_ids = torch.arange(seq_len)  # can technically be seq_len x batch_size
        x_torch = (torch.rand(batch, num_heads, seq_len, head_dim) * 2) - 1

    cos_torch, sin_torch = compute_gather_cos_sin(dhead=head_dim, end=max_seq_len * 2, position_ids=position_ids)
    cos_half = cos_torch[..., 0::2]
    sin_half = sin_torch[..., 0::2]
    out_a_torch = apply_rotary_emb_qk_real(x_torch, cos_half, sin_half)

    # Path B: Prefill mode (interleaved input)
    if cos_sin_sharded:
        cs_num_cores = device_num_cores if cos_sin_sharded == -1 else min(cos_sin_sharded, device_num_cores)
        cs_grid = ttnn.num_cores_to_corerangeset(cs_num_cores, compute_grid_size, row_wise=True)
        cs_mem_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, head_dim_padded),
            core_grid=cs_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        # Mirrors the Deepseek approach: create cos/sin at natural shape as interleaved,
        # then convert to HEIGHT_SHARDED via to_memory_config (framework handles padding).
        cos_tt = ttnn.from_torch(cos_torch, device=device, dtype=datatype, layout=ttnn.TILE_LAYOUT)
        sin_tt = ttnn.from_torch(sin_torch, device=device, dtype=datatype, layout=ttnn.TILE_LAYOUT)
        cos_tt = ttnn.to_memory_config(cos_tt, memory_config=cs_mem_config)
        sin_tt = ttnn.to_memory_config(sin_tt, memory_config=cs_mem_config)
    else:
        cos_tt = ttnn.from_torch(cos_torch, device=device, dtype=datatype, layout=ttnn.TILE_LAYOUT)
        sin_tt = ttnn.from_torch(sin_torch, device=device, dtype=datatype, layout=ttnn.TILE_LAYOUT)

    if trans_mat_sharded:
        tm_num_cores = device_num_cores if trans_mat_sharded == -1 else min(trans_mat_sharded, device_num_cores)
        tm_grid = ttnn.num_cores_to_corerangeset(tm_num_cores, compute_grid_size, row_wise=True)
        tm_mem_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, ttnn.TILE_SIZE),
            core_grid=tm_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        trans_mat_repeated = trans_mat_torch.repeat(1, 1, tm_num_cores, 1)
        trans_mat_tt = ttnn.from_torch(
            trans_mat_repeated,
            device=device,
            dtype=datatype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=tm_mem_config,
        )
    else:
        trans_mat_tt = ttnn.from_torch(trans_mat_torch, device=device, dtype=datatype, layout=ttnn.TILE_LAYOUT)

    x_tt_b = ttnn.from_torch(x_torch, device=device, dtype=datatype, layout=ttnn.TILE_LAYOUT)
    logger.info(
        f"Shapes: x_tt_b: {x_tt_b.shape}, cos_tt: {cos_tt.shape}, sin_tt: {sin_tt.shape}, trans_mat_tt: {trans_mat_tt.shape}"
    )
    out_b = ttnn.experimental.rotary_embedding_llama(
        x_tt_b,
        cos_tt,
        sin_tt,
        trans_mat_tt,
        is_decode_mode=False,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        compute_kernel_config=compute_kernel_config,
    )
    out_b_torch = ttnn.to_torch(out_b)

    passing, out_pcc = comp_pcc(out_a_torch, out_b_torch, pcc)
    logger.info(f"PCC against Torch: {out_pcc}")
    assert out_a_torch.shape == out_b_torch.shape, f"Shape mismatch: {out_a_torch.shape} vs {out_b_torch.shape}"
    assert passing, f"Prefill PCC {out_pcc} below threshold {pcc}"


@skip_for_blackhole("Requires eth connected devices to run, only single chip BH available. See #12349")
@pytest.mark.parametrize(
    "trans_mat_sharded",
    (
        0,
        -1,
        32,
    ),
    ids=(
        "tm_interleaved",
        "tm_sharded_all",
        "tm_sharded_32",
    ),
)
@pytest.mark.parametrize(
    "cos_sin_sharded",
    (
        0,
        -1,
        32,
    ),
    ids=(
        "cs_interleaved",
        "cs_sharded_all",
        "cs_sharded_32",
    ),
)
@pytest.mark.parametrize(
    "seq_len",
    (
        1,
        128,
    ),
    ids=("decode", "prefill"),
)
@pytest.mark.parametrize(
    "batch",
    (
        1,
        32,
        33,
    ),
    ids=(
        "batch1",
        "batch32",
        "batch33",
    ),
)
@pytest.mark.parametrize(
    "num_heads",
    (
        1,
        33,
    ),
    ids=(
        "heads1",
        "heads33",
    ),
)
@pytest.mark.parametrize(
    "head_dim",
    (
        2,
        64,
        256,
    ),
    ids=(
        "head_dim2",
        "head_dim64",
        "head_dim256",
    ),
)
@pytest.mark.parametrize("datatype", (ttnn.bfloat16,))
@pytest.mark.parametrize("pcc", (0.9997,))
def test_rotary_embedding_llama_prefill_cos_sin_and_trans_mat_sharding(
    trans_mat_sharded,
    cos_sin_sharded,
    seq_len,
    batch,
    num_heads,
    head_dim,
    datatype,
    pcc,
    device,
):
    """
    Validate height-sharded `cos`/`sin` and `trans_mat` support in Prefill mode.
    Shard count values: 0=interleaved, -1=all device cores, N=exactly N cores (fewer than device).
    """
    compute_grid_size = device.compute_with_storage_grid_size()
    if compute_grid_size.x < 8 or compute_grid_size.y < 8:
        pytest.skip(f"Requires grid size of at least (8, 8) to run")

    run_test_rotary_embedding_llama_prefill_cos_sin_and_trans_mat_sharding(
        trans_mat_sharded,
        cos_sin_sharded,
        seq_len,
        batch,
        num_heads,
        head_dim,
        datatype,
        pcc,
        device,
    )


def run_test_rotary_embedding_llama_bwr(
    num_positions,
    num_heads,
    head_dim,
    datatype,
    device,
    use_compute_config=True,
    shard_num_cores=-1,
):
    """
    Bitwise reproducibility (BWR) test for rotary_embedding_llama.

    Uses the all-interleaved prefill path as the golden baseline and verifies
    that every other supported calling combination produces an identical result:
      Case 2a: Height-sharded cos/sin, interleaved trans_mat  (prefill)
      Case 2b: Interleaved cos/sin, height-sharded trans_mat  (prefill)
      Case 2c: Height-sharded cos/sin AND trans_mat           (prefill)
      Case 3:  Height-sharded decode  (batch and seq_len exchanged)
    """
    torch.manual_seed(42)
    max_seq_len = MAX_SEQ_LEN
    compute_grid_size = device.compute_with_storage_grid_size()
    device_num_cores = compute_grid_size.x * compute_grid_size.y
    head_dim_padded = nearest_32(head_dim)

    compute_kernel_config = (
        ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=(True if head_dim <= 128 else False),
            packer_l1_acc=True,
        )
        if use_compute_config
        else None
    )

    # Prefill input: [batch=1, num_heads, seq_len=num_positions, head_dim]
    x_prefill = (torch.rand(1, num_heads, num_positions, head_dim) * 2) - 1
    trans_mat = get_rot_transformation_mat(dhead=ttnn.TILE_SIZE)
    position_ids = torch.randint(1, MAX_SEQ_LEN, (num_positions,))
    cos, sin = compute_gather_cos_sin(dhead=head_dim, end=max_seq_len * 2, position_ids=position_ids)
    # cos, sin shape: [1, 1, num_positions, head_dim]

    # ── Case 1: all-interleaved prefill (BASELINE) ──────────────────────
    logger.info("BWR Case 1: all-interleaved prefill (baseline)")
    x_tt = ttnn.from_torch(x_prefill, device=device, dtype=datatype, layout=ttnn.TILE_LAYOUT)
    cos_tt = ttnn.from_torch(cos, device=device, dtype=datatype, layout=ttnn.TILE_LAYOUT)
    sin_tt = ttnn.from_torch(sin, device=device, dtype=datatype, layout=ttnn.TILE_LAYOUT)
    tm_tt = ttnn.from_torch(trans_mat, device=device, dtype=datatype, layout=ttnn.TILE_LAYOUT)

    baseline = ttnn.to_torch(
        ttnn.experimental.rotary_embedding_llama(
            x_tt,
            cos_tt,
            sin_tt,
            tm_tt,
            is_decode_mode=False,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=compute_kernel_config,
        )
    )

    # Quick PCC sanity against PyTorch golden
    cos_half, sin_half = cos[..., 0::2], sin[..., 0::2]
    golden = apply_rotary_emb_qk_real(x_prefill, cos_half, sin_half)
    passing, pcc_val = comp_pcc(golden, baseline, 0.9997)
    logger.info(f"Baseline PCC vs PyTorch golden: {pcc_val}")
    assert passing, f"Baseline PCC {pcc_val} below 0.9997"

    mae = torch.mean(torch.abs(golden - baseline)).item()
    max_abs_err = torch.max(torch.abs(golden - baseline)).item()
    max_golden = torch.max(torch.abs(golden)).item()
    logger.info(f"Golden vs baseline: MAE={mae:.6e}, MaxAE={max_abs_err:.6e}, MaxGolden={max_golden:.6e}")
    assert mae < 0.01, f"MAE {mae} too large (golden vs baseline)"
    assert max_abs_err < 0.1, f"Max absolute error {max_abs_err} too large (golden vs baseline)"

    # ── Shared sharding configs ─────────────────────────────────────────
    # shard_cores: how many cores to spread sharded tensors across
    shard_cores = device_num_cores if shard_num_cores == -1 else min(shard_num_cores, device_num_cores)

    # cos/sin shard: one tile-row per shard
    cs_grid = ttnn.num_cores_to_corerangeset(shard_cores, compute_grid_size, row_wise=True)
    cs_mem = ttnn.create_sharded_memory_config(
        shape=(ttnn.TILE_SIZE, head_dim_padded),
        core_grid=cs_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # trans_mat shard: one 32x32 tile per shard
    tm_grid = ttnn.num_cores_to_corerangeset(shard_cores, compute_grid_size, row_wise=True)
    tm_mem = ttnn.create_sharded_memory_config(
        shape=(ttnn.TILE_SIZE, ttnn.TILE_SIZE),
        core_grid=tm_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    tm_repeated = trans_mat.repeat(1, 1, shard_cores, 1)

    def run_prefill(cos_sharded, tm_sharded):
        """Run a prefill variant and return the torch output."""
        xt = ttnn.from_torch(x_prefill, device=device, dtype=datatype, layout=ttnn.TILE_LAYOUT)

        if cos_sharded:
            ct = ttnn.to_memory_config(
                ttnn.from_torch(cos, device=device, dtype=datatype, layout=ttnn.TILE_LAYOUT), cs_mem
            )
            st = ttnn.to_memory_config(
                ttnn.from_torch(sin, device=device, dtype=datatype, layout=ttnn.TILE_LAYOUT), cs_mem
            )
        else:
            ct = ttnn.from_torch(cos, device=device, dtype=datatype, layout=ttnn.TILE_LAYOUT)
            st = ttnn.from_torch(sin, device=device, dtype=datatype, layout=ttnn.TILE_LAYOUT)

        if tm_sharded:
            tt = ttnn.from_torch(
                tm_repeated,
                device=device,
                dtype=datatype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=tm_mem,
            )
        else:
            tt = ttnn.from_torch(trans_mat, device=device, dtype=datatype, layout=ttnn.TILE_LAYOUT)

        return ttnn.to_torch(
            ttnn.experimental.rotary_embedding_llama(
                xt,
                ct,
                st,
                tt,
                is_decode_mode=False,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=compute_kernel_config,
            )
        )

    # ── Case 2a: sharded cos/sin, interleaved trans_mat ─────────────────
    logger.info("BWR Case 2a: sharded cos/sin, interleaved trans_mat")
    out = run_prefill(cos_sharded=True, tm_sharded=False)
    assert torch.equal(baseline, out), "BWR FAILED Case 2a: sharded cos/sin != baseline"
    logger.info("BWR Case 2a PASSED")

    # ── Case 2b: interleaved cos/sin, sharded trans_mat ─────────────────
    logger.info("BWR Case 2b: interleaved cos/sin, sharded trans_mat")
    out = run_prefill(cos_sharded=False, tm_sharded=True)
    assert torch.equal(baseline, out), "BWR FAILED Case 2b: sharded trans_mat != baseline"
    logger.info("BWR Case 2b PASSED")

    # ── Case 2c: sharded cos/sin AND trans_mat ──────────────────────────
    logger.info("BWR Case 2c: sharded cos/sin AND trans_mat")
    out = run_prefill(cos_sharded=True, tm_sharded=True)
    assert torch.equal(baseline, out), "BWR FAILED Case 2c: both sharded != baseline"
    logger.info("BWR Case 2c PASSED")

    # ── Case 3: height-sharded decode ───────────────────────────────────
    logger.info("BWR Case 3: height-sharded decode")
    if num_positions > device_num_cores:
        logger.info("Skipping BWR Case 3: num_positions > device_num_cores not supported in decode mode sharding")
    elif num_heads > ttnn.TILE_SIZE:
        logger.info("Skipping BWR Case 3: num_heads > TILE_SIZE not supported in decode mode sharding")
    else:
        # Interchange batch <-> seq_len:
        #   prefill [1, NH, NP, HD] -> decode [1, NP, NH, HD]
        x_decode = x_prefill.transpose(1, 2)
        cos_decode = cos.transpose(1, 2)  # [1, NP, 1, HD]
        sin_decode = sin.transpose(1, 2)

        decode_grid = ttnn.num_cores_to_corerangeset(num_positions, compute_grid_size, row_wise=True)
        decode_shard = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, head_dim),
            core_grid=decode_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        x_tt = ttnn.from_torch(
            x_decode,
            device=device,
            dtype=datatype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=decode_shard,
        )
        cos_tt = ttnn.from_torch(
            cos_decode,
            device=device,
            dtype=datatype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=decode_shard,
        )
        sin_tt = ttnn.from_torch(
            sin_decode,
            device=device,
            dtype=datatype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=decode_shard,
        )
        tm_decode_mem = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, ttnn.TILE_SIZE),
            core_grid=decode_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        tm_tt = ttnn.from_torch(
            trans_mat.repeat(1, 1, num_positions, 1),
            device=device,
            dtype=datatype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=tm_decode_mem,
        )

        decode_out = ttnn.to_torch(
            ttnn.experimental.rotary_embedding_llama(
                x_tt,
                cos_tt,
                sin_tt,
                tm_tt,
                is_decode_mode=True,
                compute_kernel_config=compute_kernel_config,
            )
        )

        # Transpose decode output back to prefill layout for comparison
        decode_as_prefill = decode_out.transpose(1, 2)
        assert torch.equal(baseline, decode_as_prefill), "BWR FAILED Case 3: decode != baseline"
        logger.info("BWR Case 3 PASSED")

    logger.info("All BWR cases PASSED!")


@skip_for_blackhole("Requires eth connected devices to run, only single chip BH available. See #12349")
@pytest.mark.parametrize(
    "num_positions, num_heads, head_dim",
    (
        (1024, 32, 256),
        (512, 64, 64),
        (32, 1, 64),
        (32, 1, 128),
        (32, 8, 128),
        (64, 1, 128),
    ),
    ids=(
        "pos1024_heads32_dim256",
        "pos512_heads64_dim64",
        "pos32_heads1_dim64",
        "pos32_heads1_dim128",
        "pos32_heads8_dim128",
        "pos64_heads1_dim128",
    ),
)
@pytest.mark.parametrize("datatype", (ttnn.bfloat16,))
@pytest.mark.parametrize(
    "use_compute_config",
    (True, False),
    ids=("with_compute_config", "default_compute_config"),
)
@pytest.mark.parametrize(
    "shard_num_cores",
    (-1, 32),
    ids=("shard_all_cores", "shard_32_cores"),
)
def test_rotary_embedding_llama_bwr(
    num_positions,
    num_heads,
    head_dim,
    datatype,
    use_compute_config,
    shard_num_cores,
    device,
):
    """
    Bitwise reproducibility: all calling combinations of rotary_embedding_llama
    (interleaved prefill, sharded prefill, sharded decode) must produce
    identical results.
    """
    compute_grid_size = device.compute_with_storage_grid_size()
    if compute_grid_size.x < 8 or compute_grid_size.y < 8:
        pytest.skip(f"Requires grid size of at least (8, 8) to run")

    if shard_num_cores > 0:
        cos_sin_tile_rows = (num_positions + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE
        if shard_num_cores < cos_sin_tile_rows:
            pytest.skip(f"shard_num_cores={shard_num_cores} too small for {cos_sin_tile_rows} cos/sin tile-rows")

    run_test_rotary_embedding_llama_bwr(
        num_positions,
        num_heads,
        head_dim,
        datatype,
        device,
        use_compute_config,
        shard_num_cores,
    )
