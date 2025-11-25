# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
import ttnn

from ....utils.tensor import bf16_tensor
from ....models.transformers.hunyuan.attention_hunyuan import HunyuanAttention
from ....parallel.manager import CCLManager
from ....parallel.config import DiTParallelConfig, ParallelFactor
from models.tt_transformers.tt.common import get_rot_transformation_mat


def stack_cos_sin(cos, sin):
    cos = torch.stack([cos, cos], dim=-1).flatten(-2)
    sin = torch.stack([sin, sin], dim=-1).flatten(-2)
    return cos, sin


def reshape_and_merge_qkv(q_state, k_state, v_state):
    # Rearrange QKV projections such column-fracturing shards the heads
    n_local_heads = 4
    n_local_key_value_heads = 1
    head_dim = 128

    def _merge_tensors(q, k, v):
        n_dev = 8
        q, k, v = q.T, k.T, v.T
        q = q.reshape(q.shape[0], n_dev, n_local_heads, head_dim)
        k = k.reshape(k.shape[0], n_dev, n_local_key_value_heads, head_dim)
        v = v.reshape(v.shape[0], n_dev, n_local_key_value_heads, head_dim)
        qkv = torch.cat([q, k, v], dim=2)
        qkv = qkv.reshape(qkv.shape[0], n_dev * (n_local_heads + 2 * n_local_key_value_heads) * head_dim)
        qkv = qkv.T
        return qkv

    weight = _merge_tensors(q_state, k_state, v_state)
    return weight


@pytest.mark.parametrize(
    "mesh_device, sp_axis, tp_axis, num_links",
    [
        [(4, 8), 0, 1, 4],
        # [(4, 8), 1, 0, 4],
    ],
    ids=[
        "4x8sp0tp1",
        # "4x8sp1tp0",
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    ("B, seq_len"),
    [
        (1, 128),
    ],
    ids=["128_seq"],
)
# @pytest.mark.parametrize("is_fsdp", [True, False], ids=["yes_fsdp", "no_fsdp"])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
def test_hunyuan_attention(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    B: int,
    seq_len: int,
    # is_fsdp: bool,
) -> None:
    torch_dtype = torch.float32

    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]

    # Model configuration
    hidden_dim = 4096
    num_attention_heads = 32
    num_key_value_heads = 8
    head_dim = 128
    bias = False
    out_dim = hidden_dim
    eps = 1e-5
    layer_id = 0

    MIN_PCC = 0.99

    state_dict = {
        # "to_q.weight": torch.ones(num_attention_heads * head_dim, hidden_dim),
        "to_q.weight": torch.linspace(0, 1, num_attention_heads * head_dim, dtype=torch_dtype)
        .view(num_attention_heads * head_dim, 1)
        .repeat(1, hidden_dim),
        "to_k.weight": torch.ones(num_key_value_heads * head_dim, hidden_dim),
        "to_v.weight": torch.ones(num_key_value_heads * head_dim, hidden_dim),
        "to_out.0.weight": torch.ones(hidden_dim, (num_attention_heads) * head_dim),
        "norm_q.weight": torch.ones(head_dim),
        "norm_k.weight": torch.ones(head_dim),
    }

    # Create CCL manager
    ccl_manager = CCLManager(
        mesh_device=mesh_device,
        num_links=num_links,
        topology=ttnn.Topology.Ring,
    )

    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=sp_factor),
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
    )

    # Create TT model
    tt_model = HunyuanAttention(
        hidden_dim=hidden_dim,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        eps=eps,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
    )

    tt_model.load_state_dict(state_dict)

    # Initialize weights randomly for testing
    torch.manual_seed(0)
    # Create input tensors
    # input_tensor = torch.ones((1, B, seq_len, hidden_dim), dtype=torch_dtype)
    input_tensor = (
        torch.linspace(0, seq_len - 1, seq_len, dtype=torch_dtype).view(1, 1, seq_len, 1).repeat(1, B, 1, hidden_dim)
    )

    # TODO: Use real ROPE embeddings
    rope_cos = torch.randn(seq_len, num_attention_heads, head_dim // 2)
    rope_sin = torch.randn(seq_len, num_attention_heads, head_dim // 2)

    # Sequence fractured spatial
    tt_input = bf16_tensor(input_tensor, device=mesh_device, mesh_axis=sp_axis, shard_dim=-2)
    tt_rope_cos = bf16_tensor(rope_cos, device=mesh_device, mesh_axis=sp_axis, shard_dim=-2)
    tt_rope_sin = bf16_tensor(rope_sin, device=mesh_device, mesh_axis=sp_axis, shard_dim=-2)

    # Create transformation matrix for RoPE
    trans_mat = get_rot_transformation_mat(None)
    tt_trans_mat = bf16_tensor(trans_mat, device=mesh_device)

    # Run TT model
    tt_output = tt_model(
        tt_input,
        tt_rope_cos,
        tt_rope_sin,
        tt_trans_mat,
    )

    qkv_tt = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 3), mesh_shape=(4, 8)),
    )

    qkv_state = reshape_and_merge_qkv(state_dict["to_q.weight"], state_dict["to_k.weight"], state_dict["to_v.weight"])
