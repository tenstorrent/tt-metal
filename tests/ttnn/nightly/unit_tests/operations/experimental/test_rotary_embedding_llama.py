# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from models.demos.llama3_70b_galaxy.tests.unit_tests.rotary_embedding_llama_runners import (
    MAX_SEQ_LEN,
    TtLlamaRotary,
    compute_gather_cos_sin,
    run_test_rotary_embedding_llama,
    run_test_row_major_rotary_embedding_llama,
)

import pytest
from loguru import logger
import torch
import ttnn

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_pcc,
)
from models.common.reference_rope import apply_rotary_emb, precompute_freqs_cis
from tt_py_test_utils_common.utility_functions import skip_for_blackhole, nearest_32, skip_for_wormhole_b0
from models.tt_transformers.tt.common import (
    precompute_freqs,
    get_rot_transformation_mat,
)
from models.tt_transformers.tt.rope import RotarySetup
from models.demos.llama3_70b_galaxy.tt.llama_rope import TtLlamaRotarySetup


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


@pytest.mark.parametrize("seq_len", (64, 256), ids=("prefill_64", "prefill_256"))
@pytest.mark.parametrize("n_heads, n_kv_heads, head_dim", ((8, 1, 128), (16, 4, 256), (32, 8, 128)))
def test_rotary_embedding_llama_prefill_head_split(seq_len, n_heads, n_kv_heads, head_dim, device):
    """Short prefill: a few sequence tiles leave most cores idle, so the op also splits heads across cores."""
    run_test_rotary_embedding_llama(device, 1, seq_len, 0.9997, n_heads, n_kv_heads, head_dim, max(4096, seq_len))


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

    assert device.cache_entries_counter.total == num_ops


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


def _host_first_from_torch(torch_tensor, device, **kwargs):
    memory_config = kwargs.pop("memory_config", None)
    host_tensor = ttnn.from_torch(torch_tensor, device=None, **kwargs)
    if memory_config is None:
        return ttnn.to_device(host_tensor, device=device)
    return ttnn.to_device(host_tensor, device=device, memory_config=memory_config)


def _direct_from_torch(torch_tensor, device, **kwargs):
    return ttnn.from_torch(torch_tensor, device=device, **kwargs)


def _shard_cos_sin(cos_tt, sin_tt, device, head_dim, orientation=ttnn.ShardOrientation.ROW_MAJOR):
    compute_grid_size = device.compute_with_storage_grid_size()
    num_cores = compute_grid_size.x * compute_grid_size.y
    core_grid = ttnn.num_cores_to_corerangeset(num_cores, compute_grid_size, row_wise=True)
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core_grid, [ttnn.TILE_SIZE, nearest_32(head_dim)], orientation),
    )
    return ttnn.to_memory_config(cos_tt, memory_config), ttnn.to_memory_config(sin_tt, memory_config)


def _run_rotary_embedding_llama_direct_cos_padding_tail_case(device, q_seq_len, rope_seq_len, cos_sin_sharded):
    torch.manual_seed(0)
    head_dim = 128
    q = torch.randn((1, 64, q_seq_len, head_dim), dtype=torch.float32)
    cos, sin = compute_gather_cos_sin(
        dhead=head_dim,
        end=2 * rope_seq_len,
        position_ids=torch.arange(rope_seq_len),
    )
    trans_mat = get_rot_transformation_mat(dhead=head_dim)

    tensor_kwargs = {
        "dtype": ttnn.bfloat16,
        "layout": ttnn.TILE_LAYOUT,
        "memory_config": ttnn.DRAM_MEMORY_CONFIG,
    }

    q_tt = _host_first_from_torch(q, device, **tensor_kwargs.copy())
    sin_tt = _host_first_from_torch(sin, device, **tensor_kwargs.copy())
    trans_mat_tt = _host_first_from_torch(trans_mat, device, **tensor_kwargs.copy())

    host_cos_tt = _host_first_from_torch(cos, device, **tensor_kwargs.copy())
    direct_cos_tt = _direct_from_torch(cos, device, **tensor_kwargs.copy())
    direct_sin_tt = sin_tt
    if cos_sin_sharded:
        direct_cos_tt, direct_sin_tt = _shard_cos_sin(
            direct_cos_tt, _direct_from_torch(sin, device, **tensor_kwargs.copy()), device, head_dim
        )

    host_out_tt = ttnn.experimental.rotary_embedding_llama(
        q_tt, host_cos_tt, sin_tt, trans_mat_tt, is_decode_mode=False
    )
    direct_out_tt = ttnn.experimental.rotary_embedding_llama(
        q_tt, direct_cos_tt, direct_sin_tt, trans_mat_tt, is_decode_mode=False
    )

    host_out = ttnn.to_torch(host_out_tt)
    direct_out = ttnn.to_torch(direct_out_tt)

    _, output_pcc = comp_pcc(host_out, direct_out, 0.9997)
    logger.info(f"PCC direct vs host: {output_pcc}")

    abs_diff = torch.abs(host_out.float() - direct_out.float())
    logger.info(f"MAE direct vs host: {torch.mean(abs_diff)}")
    logger.info(f"Max abs diff direct vs host: {torch.max(abs_diff)}")
    logger.info(f"Max host output: {torch.max(torch.abs(host_out.float()))}")
    logger.info(f"Max direct output: {torch.max(torch.abs(direct_out.float()))}")

    assert torch.isfinite(direct_out.float()).all()
    assert direct_out.shape == host_out.shape
    assert torch.equal(direct_out, host_out)

    tail_start = nearest_32(rope_seq_len)
    if q_seq_len > tail_start:
        tail = direct_out[:, :, tail_start:, :]
        assert torch.equal(tail, torch.zeros_like(tail))


@skip_for_blackhole("Requires eth connected devices to run, only single chip BH available. See #12349")
@pytest.mark.parametrize(
    "q_seq_len, rope_seq_len",
    (
        (33, 32),
        (128, 16),
    ),
    ids=(
        "q33_rope32",
        "q128_rope16",
    ),
)
@pytest.mark.parametrize("cos_sin_sharded", (False, True), ids=("interleaved_cos_sin", "sharded_cos_sin"))
def test_rotary_embedding_llama_direct_cos_padding_tail(q_seq_len, rope_seq_len, cos_sin_sharded, device):
    compute_grid_size = device.compute_with_storage_grid_size()
    if compute_grid_size.x < 8 or compute_grid_size.y < 8:
        pytest.skip(f"Requires grid size of at least {(8, 8)} to run")

    _run_rotary_embedding_llama_direct_cos_padding_tail_case(device, q_seq_len, rope_seq_len, cos_sin_sharded)


# Full-grid sharded cos/sin, one sequence tile per core: core k's own shard is sequence tile k.
# With batch > 1 the cores of later batches restart at tile 0, so their own shard is the wrong row.
@skip_for_blackhole("Requires eth connected devices to run, only single chip BH available. See #12349")
@pytest.mark.parametrize("batch", (2, 4))
def test_rotary_embedding_llama_sharded_cos_sin_batch(batch, device):
    compute_grid_size = device.compute_with_storage_grid_size()
    if compute_grid_size.x < 8 or compute_grid_size.y < 8:
        pytest.skip(f"Requires grid size of at least {(8, 8)} to run")

    torch.manual_seed(0)
    head_dim = 128
    seq_len = 64
    q = torch.randn((batch, 8, seq_len, head_dim), dtype=torch.float32)
    cos, sin = compute_gather_cos_sin(dhead=head_dim, end=2 * seq_len, position_ids=torch.arange(seq_len))
    trans_mat = get_rot_transformation_mat(dhead=head_dim)

    tensor_kwargs = {"dtype": ttnn.bfloat16, "layout": ttnn.TILE_LAYOUT, "memory_config": ttnn.DRAM_MEMORY_CONFIG}
    q_tt = ttnn.from_torch(q, device=device, **tensor_kwargs)
    cos_tt = ttnn.from_torch(cos, device=device, **tensor_kwargs)
    sin_tt = ttnn.from_torch(sin, device=device, **tensor_kwargs)
    trans_mat_tt = ttnn.from_torch(trans_mat, device=device, **tensor_kwargs)
    sharded_cos_tt, sharded_sin_tt = _shard_cos_sin(cos_tt, sin_tt, device, head_dim)

    expected = ttnn.to_torch(
        ttnn.experimental.rotary_embedding_llama(q_tt, cos_tt, sin_tt, trans_mat_tt, is_decode_mode=False)
    )
    actual = ttnn.to_torch(
        ttnn.experimental.rotary_embedding_llama(
            q_tt, sharded_cos_tt, sharded_sin_tt, trans_mat_tt, is_decode_mode=False
        )
    )

    assert torch.equal(actual, expected)


# One sequence tile per core: a COL_MAJOR shard puts tile k on a different core than the one processing it.
@skip_for_blackhole("Requires eth connected devices to run, only single chip BH available. See #12349")
@pytest.mark.parametrize(
    "orientation",
    (ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR),
    ids=("row_major", "col_major"),
)
def test_rotary_embedding_llama_sharded_cos_sin_orientation(orientation, device):
    compute_grid_size = device.compute_with_storage_grid_size()
    if compute_grid_size.x < 8 or compute_grid_size.y < 8:
        pytest.skip(f"Requires grid size of at least {(8, 8)} to run")

    torch.manual_seed(0)
    head_dim = 128
    seq_len = 2048
    q = torch.randn((1, 8, seq_len, head_dim), dtype=torch.float32)
    cos, sin = compute_gather_cos_sin(dhead=head_dim, end=2 * seq_len, position_ids=torch.arange(seq_len))
    trans_mat = get_rot_transformation_mat(dhead=head_dim)

    tensor_kwargs = {"dtype": ttnn.bfloat16, "layout": ttnn.TILE_LAYOUT, "memory_config": ttnn.DRAM_MEMORY_CONFIG}
    q_tt = ttnn.from_torch(q, device=device, **tensor_kwargs)
    cos_tt = ttnn.from_torch(cos, device=device, **tensor_kwargs)
    sin_tt = ttnn.from_torch(sin, device=device, **tensor_kwargs)
    trans_mat_tt = ttnn.from_torch(trans_mat, device=device, **tensor_kwargs)
    sharded_cos_tt, sharded_sin_tt = _shard_cos_sin(cos_tt, sin_tt, device, head_dim, orientation)

    expected = ttnn.to_torch(
        ttnn.experimental.rotary_embedding_llama(q_tt, cos_tt, sin_tt, trans_mat_tt, is_decode_mode=False)
    )
    actual = ttnn.to_torch(
        ttnn.experimental.rotary_embedding_llama(
            q_tt, sharded_cos_tt, sharded_sin_tt, trans_mat_tt, is_decode_mode=False
        )
    )

    assert torch.equal(actual, expected)


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
