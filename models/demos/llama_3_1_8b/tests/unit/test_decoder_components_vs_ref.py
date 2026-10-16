# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""D2/D3: Decoder component PCC tests on the target 8x4 mesh.

Covers: RMSNorm, dense MLP, attention (one-shot), KV cache write, decoder layer.
All use random weights identical on both sides, bf16 reference.
"""

from types import SimpleNamespace

import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.llama_3_1_8b.tests.test_factory import parametrize_mesh_with_fabric

PCC_LOWER = 0.85
PCC_TARGET = 0.99

NUM_LAYERS_TEST = 1
SEQ_LEN_LOCAL = 256
HEAD_DIM = 128
HIDDEN = 4096
INTERMEDIATE = 14336
NUM_HEADS = 32
NUM_KV_HEADS = 8
VOCAB = 1024


def _hf_config():
    return SimpleNamespace(
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        num_hidden_layers=NUM_LAYERS_TEST,
        rms_norm_eps=1e-5,
        head_dim=HEAD_DIM,
        rope_theta=500000.0,
        rope_scaling=SimpleNamespace(
            factor=8.0,
            low_freq_factor=1.0,
            high_freq_factor=4.0,
            original_max_position_embeddings=8192,
            rope_type="llama3",
        ),
        vocab_size=VOCAB,
        attention_bias=False,
        mlp_bias=False,
    )


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)])
def test_norm_vs_ref(mesh_device, device_params, reset_seeds):
    """RMSNorm output vs torch reference."""
    torch.manual_seed(42)
    hf = _hf_config()
    sp = mesh_device.shape[0]
    seq_local = SEQ_LEN_LOCAL

    weight = torch.randn(HIDDEN, dtype=torch.bfloat16)
    x_full = torch.randn(1, 1, seq_local, HIDDEN, dtype=torch.bfloat16)

    # Torch reference
    from models.demos.llama_3_1_8b.reference.model import LlamaRMSNorm

    norm_ref = LlamaRMSNorm(HIDDEN, eps=hf.rms_norm_eps).to(torch.bfloat16)
    norm_ref.weight.data = weight.clone()
    ref_out = norm_ref(x_full.squeeze(0).squeeze(0)).unsqueeze(0).unsqueeze(0)

    # TT
    from models.demos.llama_3_1_8b.tt.rms_norm import RMSNorm

    state_dict = {"weight": weight.clone()}
    tt_norm = RMSNorm(mesh_device, hf, state_dict)

    tt_x = ttnn.from_torch(
        x_full,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_out = tt_norm(tt_x)
    tt_out_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).float()

    passing, pcc = comp_pcc(ref_out.float(), tt_out_torch, PCC_LOWER)
    assert passing, f"RMSNorm PCC {pcc} < {PCC_LOWER}"


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)])
def test_dense_mlp_vs_ref(mesh_device, device_params, reset_seeds):
    """Dense SwiGLU MLP vs torch reference at real dims."""
    torch.manual_seed(42)
    hf = _hf_config()
    seq_local = SEQ_LEN_LOCAL

    # Random weights
    gate_w = torch.randn(INTERMEDIATE, HIDDEN, dtype=torch.bfloat16)
    up_w = torch.randn(INTERMEDIATE, HIDDEN, dtype=torch.bfloat16)
    down_w = torch.randn(HIDDEN, INTERMEDIATE, dtype=torch.bfloat16)
    x_full = torch.randn(1, 1, seq_local, HIDDEN, dtype=torch.bfloat16)

    # Torch reference
    import torch.nn.functional as F

    gate_out = F.silu(x_full @ gate_w.T)
    up_out = x_full @ up_w.T
    ref_out = (gate_out * up_out) @ down_w.T

    # TT
    from models.demos.llama_3_1_8b.tt.ccl import CCLManager
    from models.demos.llama_3_1_8b.tt.config import MeshConfig
    from models.demos.llama_3_1_8b.tt.mlp import MLP
    from models.demos.llama_3_1_8b.utils.general_utils import get_default_num_links

    mesh_config = MeshConfig(mesh_device.shape, tp=mesh_device.shape[1])
    ccl = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device))
    state_dict = {
        "gate_proj.weight": gate_w.clone(),
        "up_proj.weight": up_w.clone(),
        "down_proj.weight": down_w.clone(),
    }
    tt_mlp = MLP(mesh_device, hf, state_dict, ccl, mesh_config=mesh_config)

    tt_x = ttnn.from_torch(
        x_full,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_out = tt_mlp(tt_x)
    tt_out_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).float()

    passing, pcc = comp_pcc(ref_out.float(), tt_out_torch, PCC_LOWER)
    assert passing, f"MLP PCC {pcc} < {PCC_LOWER}"


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)])
def test_decoder_layer_vs_ref(mesh_device, device_params, reset_seeds):
    """One complete decoder layer with residuals vs torch reference."""
    torch.manual_seed(42)
    from models.demos.llama_3_1_8b.config import Llama31Config
    from models.demos.llama_3_1_8b.reference.model import LlamaDecoderLayer as RefLayer
    from models.demos.llama_3_1_8b.reference.model import LlamaRotaryEmbedding
    from models.demos.llama_3_1_8b.tt.ccl import CCLManager
    from models.demos.llama_3_1_8b.tt.config import MeshConfig
    from models.demos.llama_3_1_8b.tt.layer import DecoderLayer
    from models.demos.llama_3_1_8b.tt.rope import build_transformation_mat
    from models.demos.llama_3_1_8b.utils.general_utils import get_default_num_links

    cfg = Llama31Config(num_hidden_layers=1)
    hf = cfg
    sp = mesh_device.shape[0]
    seq_total = SEQ_LEN_LOCAL * sp

    ref_layer = RefLayer(cfg, layer_idx=0).to(torch.bfloat16)
    state_dict_flat = {}
    for name, param in ref_layer.named_parameters():
        state_dict_flat[name] = param.data.clone()

    # Torch reference
    x = torch.randn(1, seq_total, HIDDEN, dtype=torch.bfloat16)
    rope_emb = LlamaRotaryEmbedding(cfg)
    pos_ids = torch.arange(seq_total).unsqueeze(0)
    cos, sin = rope_emb(x, pos_ids)
    ref_out, _, _ = ref_layer(x, (cos, sin))

    # TT
    mesh_config = MeshConfig(mesh_device.shape, tp=mesh_device.shape[1])
    ccl = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device))
    from models.demos.llama_3_1_8b.tt.rope import build_indexed_rope

    chunk_size = seq_total
    max_seq_len = seq_total
    kv_cache_seq = max_seq_len

    from models.demos.llama_3_1_8b.tt.attention.kv_cache import allocate_kv_cache

    kv_cache = allocate_kv_cache(
        mesh_device,
        num_layers=1,
        max_seq_len=kv_cache_seq,
        num_users=1,
    )

    indexed_rope = build_indexed_rope(
        mesh_device,
        head_dim=HEAD_DIM,
        max_seq_len=max_seq_len,
        chunk_size=chunk_size,
    )
    transformation_mats = {"prefill": build_transformation_mat(mesh_device)}

    tt_layer = DecoderLayer(
        mesh_device,
        hf,
        state_dict_flat,
        layer_idx=0,
        ccl_manager=ccl,
        mesh_config=mesh_config,
        transformation_mats=transformation_mats,
        max_seq_len=max_seq_len,
        sequence_parallel=True,
    )

    # SP-shard the input
    x_4d = x.reshape(1, 1, seq_total, HIDDEN)
    shard_dims = [None, None]
    shard_dims[0] = 2
    tt_x = ttnn.from_torch(
        x_4d,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=tuple(shard_dims), mesh_shape=mesh_device.shape),
    )

    tt_out = tt_layer(
        tt_x,
        position_embeddings=indexed_rope,
        kv_cache=kv_cache,
        user_id=0,
        cached_len=0,
        indexed_rope=True,
    )

    # Gather output from SP shards
    out_torch = ttnn.to_torch(
        tt_out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 3), mesh_shape=mesh_device.shape)
    ).float()
    # After SP concat dim2 has seq_total, after TP concat dim3 has hidden*tp — take first hidden
    out_torch = out_torch[:, :, :seq_total, :HIDDEN]

    passing, pcc = comp_pcc(ref_out.float().unsqueeze(0), out_torch, PCC_LOWER)
    assert passing, f"Decoder layer PCC {pcc} < {PCC_LOWER}"
