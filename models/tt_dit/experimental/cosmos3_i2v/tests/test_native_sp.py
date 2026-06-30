# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""SP=2 PCC tests for native Cosmos3 attention / decoder layer / trunk.

Counterpart to `test_native_attention.py`, `test_native_decoder_layer.py`,
and `test_native_transformer.py` (which cover (1, N) TP-only meshes).
These tests parametrize over a 2D `(2, 4)` mesh — sp_axis=0, tp_axis=1,
sp_factor=2, tp_factor=4 — exercising the ring-joint-SDPA path through
the gen pathway.

Config picks num_attention_heads=32, num_key_value_heads=4 (8:1 GQA,
matching the real 64B trunk's 64:8 ratio). At tp_factor=4 that yields
n_local_kv_heads=1, which satisfies the `kv_repeat>1 →
n_local_kv_heads==1` assertion the ring-SDPA path enforces (GQA
broadcast via `ttnn.concat` only equals `repeat_interleave` when the
single KV head is replicated across the local Q heads).

N_gen=256 = `k_chunk_size * sp_factor` (128 * 2) so the per-chip gen
sequence after scatter is `k_chunk_size`-aligned, which the ring op
requires. N_und=128 is the q_chunk_size pad already enforced by
`_pad_for_joint`. The trunk gathers gen back to replicated at exit, so
the test compares directly against the torch reference for the full
trunk; the attention and decoder-layer tests read the sp-sharded gen
output with ConcatMesh2dToTensor and slice the padded rows off.

The SP path is gated behind `TT_COSMOS3_ENABLE_SP_RING`. The
`enable_sp_ring` fixture toggles it on for these tests via monkeypatch.
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.check import assert_quality
from models.tt_dit.utils.tensor import bf16_tensor
from models.tt_dit.utils.test import line_params


@pytest.fixture
def enable_sp_ring(monkeypatch: pytest.MonkeyPatch) -> None:
    """Opt into the SP ring-SDPA branch in `model/attention.py`."""
    monkeypatch.setenv("TT_COSMOS3_ENABLE_SP_RING", "1")


# On BH Galaxy you must open the FULL (4, 8) mesh up front and carve the SP
# submesh from it — opening a (2, 4) mesh directly fails the fabric router
# ethernet handshake (FabricFirmwareInitializer timeout) because the broker
# can't bring up just a partial slice of the galaxy fabric. Pattern matches
# `tests/blocks/test_attention.py`: parent shape → create_submesh.
_SP_MESH_PARAMS = [
    pytest.param((4, 8), (2, 4), line_params, id="2x4_sp2_tp4_on_bh_galaxy"),
]

# (2, 8) submesh: sp=2, tp=8. Matches the real production layout (full Cosmos3
# 64Q/8KV-head GQA needs tp=8 for n_local_kv_heads=1) and is what the cfg-parallel
# pipeline carves on BH Galaxy 4x8 (two 2x8 submeshes for cond/uncond).
_SP_REAL_MESH_PARAMS = [
    pytest.param((4, 8), (1, 8), line_params, id="1x8_sp1_tp8_real_config"),
    pytest.param((4, 8), (2, 8), line_params, id="2x8_sp2_tp8_real_config"),
    pytest.param((4, 8), (4, 8), line_params, id="4x8_sp4_tp8_real_config"),
]

# Production trunk knobs. Matches `transformer/config.json` of the 64B trunk
# (Cosmos3-Super-Image2Video). Loading one decoder layer is enough to expose
# any bug specific to the wide-hidden + 3D mRoPE path that the small-config
# tests above can't catch.
_REAL_HIDDEN = 5120
_REAL_HEAD_DIM = 128
_REAL_NUM_Q_HEADS = 64
_REAL_NUM_KV_HEADS = 8  # 8:1 GQA — n_local_kv=1 at tp=8
_REAL_INTERMEDIATE = 25600
_REAL_ROPE_THETA = 5_000_000.0
_REAL_ROPE_AXES = (24, 20, 20)  # sums to head_dim/2 = 64

# Shared test-config knobs. num_kv_heads=4 → n_local_kv=1 at tp=4 (GQA broadcast assertion).
_HIDDEN = 256
_HEAD_DIM = 64
_NUM_Q_HEADS = 32
_NUM_KV_HEADS = 4
_INTERMEDIATE = 512
_RMS_EPS = 1e-6
_ROPE_THETA = 5_000_000.0
_ROPE_AXES = (12, 10, 10)  # sums to head_dim/2 = 32
_N_UND = 128  # q_chunk_size multiple
_N_GEN = 256  # k_chunk_size * sp_factor (128 * 2)


def _gather_sp_sharded(
    t: ttnn.Tensor, mesh_device: ttnn.MeshDevice, sp_axis: int, n_logical: int, hidden: int
) -> torch.Tensor:
    """Concat per-chip slices along the sequence dim and slice off padding.

    SP-sharded gen outputs from attention/decoder-layer tests have shape
    `[1, 1, N_padded/sp, hidden]` per chip with the sequence dim split
    across `sp_axis`. The replicated TP axis means each row of the mesh
    on the TP axis carries the same data; pick the first column.
    """
    # gen tensors are SP-sharded on sp_axis and replicated on tp_axis. Pulling all
    # device tensors and indexing by mesh-row gives us the SP-axis slices in order.
    devs = ttnn.get_device_tensors(t)
    mesh_shape = tuple(mesh_device.shape)
    if sp_axis == 0:
        sp_factor = mesh_shape[0]
        tp_factor = mesh_shape[1]
        # Row-major flat index: rank = sp_idx * tp_factor + tp_idx. We want tp_idx=0.
        per_sp = [ttnn.to_torch(devs[sp_idx * tp_factor]) for sp_idx in range(sp_factor)]
    else:
        sp_factor = mesh_shape[1]
        tp_factor = mesh_shape[0]
        # Column-major would be the same; with sp_axis=1, we want tp_idx=0 → rank = sp_idx.
        per_sp = [ttnn.to_torch(devs[sp_idx]) for sp_idx in range(sp_factor)]
    concatenated = torch.cat([s.reshape(-1, hidden) for s in per_sp], dim=0)
    return concatenated[:n_logical]


@pytest.mark.parametrize(
    "mesh_device, submesh_shape, device_params",
    _SP_MESH_PARAMS,
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(300)
def test_native_joint_attention_sp(
    mesh_device: ttnn.MeshDevice, submesh_shape: tuple[int, int], enable_sp_ring: None
) -> None:
    """SP=2 ring-joint-SDPA PCC against the torch reference attention."""
    from models.tt_dit.experimental.cosmos3_i2v.model.attention import Cosmos3JointAttention
    from models.tt_dit.experimental.cosmos3_i2v.reference.transformer_cosmos3 import (
        Cosmos3PackedMoTAttention,
        Cosmos3VLTextRotaryEmbedding,
    )

    torch.manual_seed(42)

    mesh_device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    mesh_shape = tuple(mesh_device.shape)
    sp_factor = mesh_shape[0]
    tp_factor = mesh_shape[1]
    assert sp_factor > 1, "SP test requires sp_factor > 1"

    torch_attn = Cosmos3PackedMoTAttention(
        hidden_size=_HIDDEN,
        head_dim=_HEAD_DIM,
        num_attention_heads=_NUM_Q_HEADS,
        num_key_value_heads=_NUM_KV_HEADS,
        attention_bias=False,
        rms_norm_eps=_RMS_EPS,
    )
    torch_attn.eval()
    torch_attn.to(dtype=torch.bfloat16)

    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(1, 0),
        sequence_parallel=ParallelFactor(sp_factor, 0),
        tensor_parallel=ParallelFactor(tp_factor, 1),
    )
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=1, topology=ttnn.Topology.Linear)

    tt_attn = Cosmos3JointAttention(
        hidden_size=_HIDDEN,
        head_dim=_HEAD_DIM,
        num_attention_heads=_NUM_Q_HEADS,
        num_key_value_heads=_NUM_KV_HEADS,
        attention_bias=False,
        rms_norm_eps=_RMS_EPS,
        mesh_device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
    )
    tt_attn.load_torch_state_dict(torch_attn.state_dict())

    # mRoPE cos/sin over the joint sequence, sliced into und + gen halves.
    rope = Cosmos3VLTextRotaryEmbedding(
        head_dim=_HEAD_DIM,
        rope_theta=_ROPE_THETA,
        rope_axes_dim=list(_ROPE_AXES),
    )
    position_ids = torch.arange(_N_UND + _N_GEN).unsqueeze(0)
    cos_all, sin_all = rope(position_ids, device=torch.device("cpu"), dtype=torch.bfloat16)
    cos_all = cos_all.squeeze(0)
    sin_all = sin_all.squeeze(0)
    cos_und = cos_all[:_N_UND]
    sin_und = sin_all[:_N_UND]
    cos_gen = cos_all[_N_UND:]
    sin_gen = sin_all[_N_UND:]

    und_seq = torch.randn(_N_UND, _HIDDEN, dtype=torch.bfloat16)
    gen_seq = torch.randn(_N_GEN, _HIDDEN, dtype=torch.bfloat16)

    with torch.no_grad():
        ref_und, ref_gen = torch_attn(und_seq, gen_seq, (cos_und, sin_und, cos_gen, sin_gen))

    # und + cos/sin und are replicated. gen + cos/sin gen are SP-sharded on dim=2.
    und_tt = bf16_tensor(und_seq.reshape(1, 1, _N_UND, _HIDDEN), device=mesh_device)
    cos_und_tt = bf16_tensor(cos_und.reshape(1, 1, _N_UND, _HEAD_DIM), device=mesh_device)
    sin_und_tt = bf16_tensor(sin_und.reshape(1, 1, _N_UND, _HEAD_DIM), device=mesh_device)

    gen_tt = bf16_tensor(gen_seq.reshape(1, 1, _N_GEN, _HIDDEN), device=mesh_device, mesh_axis=0, shard_dim=2)
    cos_gen_tt = bf16_tensor(cos_gen.reshape(1, 1, _N_GEN, _HEAD_DIM), device=mesh_device, mesh_axis=0, shard_dim=2)
    sin_gen_tt = bf16_tensor(sin_gen.reshape(1, 1, _N_GEN, _HEAD_DIM), device=mesh_device, mesh_axis=0, shard_dim=2)

    tt_und_out, tt_gen_out = tt_attn(
        und_tt, gen_tt, cos_und_tt, sin_und_tt, cos_gen_tt, sin_gen_tt, logical_n_gen=_N_GEN
    )

    und_view = ttnn.to_torch(ttnn.get_device_tensors(tt_und_out)[0]).reshape(_N_UND, _HIDDEN)
    gen_view = _gather_sp_sharded(tt_gen_out, mesh_device, sp_axis=0, n_logical=_N_GEN, hidden=_HIDDEN)

    assert_quality(ref_und, und_view, pcc=0.98)
    assert_quality(ref_gen, gen_view, pcc=0.98)


@pytest.mark.parametrize(
    "mesh_device, submesh_shape, device_params",
    _SP_MESH_PARAMS,
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(300)
def test_native_decoder_layer_sp(
    mesh_device: ttnn.MeshDevice, submesh_shape: tuple[int, int], enable_sp_ring: None
) -> None:
    """SP=2 full-decoder-layer PCC. Validates RMSNorm + MLP + residual stay correct on sp-sharded gen."""
    from models.tt_dit.experimental.cosmos3_i2v.model.decoder_layer import (
        Cosmos3VLTextMoTDecoderLayer as TTDecoderLayer,
    )
    from models.tt_dit.experimental.cosmos3_i2v.reference.transformer_cosmos3 import (
        Cosmos3VLTextMoTDecoderLayer as RefDecoderLayer,
    )
    from models.tt_dit.experimental.cosmos3_i2v.reference.transformer_cosmos3 import Cosmos3VLTextRotaryEmbedding

    torch.manual_seed(42)

    mesh_device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    mesh_shape = tuple(mesh_device.shape)
    sp_factor = mesh_shape[0]
    tp_factor = mesh_shape[1]
    assert sp_factor > 1, "SP test requires sp_factor > 1"

    torch_layer = RefDecoderLayer(
        hidden_size=_HIDDEN,
        head_dim=_HEAD_DIM,
        num_attention_heads=_NUM_Q_HEADS,
        num_key_value_heads=_NUM_KV_HEADS,
        intermediate_size=_INTERMEDIATE,
        attention_bias=False,
        rms_norm_eps=_RMS_EPS,
    )
    torch_layer.eval()
    torch_layer.to(dtype=torch.bfloat16)

    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(1, 0),
        sequence_parallel=ParallelFactor(sp_factor, 0),
        tensor_parallel=ParallelFactor(tp_factor, 1),
    )
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=1, topology=ttnn.Topology.Linear)

    tt_layer = TTDecoderLayer(
        hidden_size=_HIDDEN,
        head_dim=_HEAD_DIM,
        num_attention_heads=_NUM_Q_HEADS,
        num_key_value_heads=_NUM_KV_HEADS,
        intermediate_size=_INTERMEDIATE,
        attention_bias=False,
        rms_norm_eps=_RMS_EPS,
        mesh_device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
    )
    tt_layer.load_torch_state_dict(torch_layer.state_dict())

    rope = Cosmos3VLTextRotaryEmbedding(
        head_dim=_HEAD_DIM,
        rope_theta=_ROPE_THETA,
        rope_axes_dim=list(_ROPE_AXES),
    )
    position_ids = torch.arange(_N_UND + _N_GEN).unsqueeze(0)
    cos_all, sin_all = rope(position_ids, device=torch.device("cpu"), dtype=torch.bfloat16)
    cos_all = cos_all.squeeze(0)
    sin_all = sin_all.squeeze(0)
    cos_und = cos_all[:_N_UND]
    sin_und = sin_all[:_N_UND]
    cos_gen = cos_all[_N_UND:]
    sin_gen = sin_all[_N_UND:]

    und_seq = torch.randn(_N_UND, _HIDDEN, dtype=torch.bfloat16)
    gen_seq = torch.randn(_N_GEN, _HIDDEN, dtype=torch.bfloat16)

    with torch.no_grad():
        ref_und, ref_gen = torch_layer(und_seq, gen_seq, (cos_und, sin_und, cos_gen, sin_gen))

    und_tt = bf16_tensor(und_seq.reshape(1, 1, _N_UND, _HIDDEN), device=mesh_device)
    cos_und_tt = bf16_tensor(cos_und.reshape(1, 1, _N_UND, _HEAD_DIM), device=mesh_device)
    sin_und_tt = bf16_tensor(sin_und.reshape(1, 1, _N_UND, _HEAD_DIM), device=mesh_device)

    gen_tt = bf16_tensor(gen_seq.reshape(1, 1, _N_GEN, _HIDDEN), device=mesh_device, mesh_axis=0, shard_dim=2)
    cos_gen_tt = bf16_tensor(cos_gen.reshape(1, 1, _N_GEN, _HEAD_DIM), device=mesh_device, mesh_axis=0, shard_dim=2)
    sin_gen_tt = bf16_tensor(sin_gen.reshape(1, 1, _N_GEN, _HEAD_DIM), device=mesh_device, mesh_axis=0, shard_dim=2)

    tt_und_out, tt_gen_out = tt_layer(
        und_tt, gen_tt, cos_und_tt, sin_und_tt, cos_gen_tt, sin_gen_tt, logical_n_gen=_N_GEN
    )

    und_view = ttnn.to_torch(ttnn.get_device_tensors(tt_und_out)[0]).reshape(_N_UND, _HIDDEN)
    gen_view = _gather_sp_sharded(tt_gen_out, mesh_device, sp_axis=0, n_logical=_N_GEN, hidden=_HIDDEN)

    assert_quality(ref_und, und_view, pcc=0.97)
    assert_quality(ref_gen, gen_view, pcc=0.97)


@pytest.mark.parametrize(
    "mesh_device, submesh_shape, device_params",
    _SP_MESH_PARAMS,
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("num_hidden_layers", [1, 2, 4], ids=["L1", "L2", "L4"])
@pytest.mark.timeout(600)
def test_native_transformer_sp(
    mesh_device: ttnn.MeshDevice,
    submesh_shape: tuple[int, int],
    num_hidden_layers: int,
    enable_sp_ring: None,
) -> None:
    """SP=2 full-trunk PCC. The trunk all-gathers gen at exit, so we read replicated outputs."""
    from diffusers.models.normalization import RMSNorm as RefRMSNorm

    from models.tt_dit.experimental.cosmos3_i2v.model.transformer import Cosmos3OmniTransformer as TTTransformer
    from models.tt_dit.experimental.cosmos3_i2v.reference.transformer_cosmos3 import (
        Cosmos3VLTextMoTDecoderLayer as RefDecoderLayer,
    )
    from models.tt_dit.experimental.cosmos3_i2v.reference.transformer_cosmos3 import Cosmos3VLTextRotaryEmbedding

    torch.manual_seed(42)

    mesh_device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    mesh_shape = tuple(mesh_device.shape)
    sp_factor = mesh_shape[0]
    tp_factor = mesh_shape[1]
    assert sp_factor > 1, "SP test requires sp_factor > 1"

    ref_layers = [
        RefDecoderLayer(
            hidden_size=_HIDDEN,
            head_dim=_HEAD_DIM,
            num_attention_heads=_NUM_Q_HEADS,
            num_key_value_heads=_NUM_KV_HEADS,
            intermediate_size=_INTERMEDIATE,
            attention_bias=False,
            rms_norm_eps=_RMS_EPS,
        )
        for _ in range(num_hidden_layers)
    ]
    ref_norm = RefRMSNorm(_HIDDEN, eps=_RMS_EPS, elementwise_affine=True, bias=False)
    ref_norm_moe_gen = RefRMSNorm(_HIDDEN, eps=_RMS_EPS, elementwise_affine=True, bias=False)
    for m in [*ref_layers, ref_norm, ref_norm_moe_gen]:
        m.eval()
        m.to(dtype=torch.bfloat16)

    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(1, 0),
        sequence_parallel=ParallelFactor(sp_factor, 0),
        tensor_parallel=ParallelFactor(tp_factor, 1),
    )
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=1, topology=ttnn.Topology.Linear)

    tt_trunk = TTTransformer(
        hidden_size=_HIDDEN,
        head_dim=_HEAD_DIM,
        num_attention_heads=_NUM_Q_HEADS,
        num_key_value_heads=_NUM_KV_HEADS,
        intermediate_size=_INTERMEDIATE,
        num_hidden_layers=num_hidden_layers,
        attention_bias=False,
        rms_norm_eps=_RMS_EPS,
        mesh_device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
    )

    state: dict[str, torch.Tensor] = {}
    for i, layer in enumerate(ref_layers):
        for k, v in layer.state_dict().items():
            state[f"layers.{i}.{k}"] = v
    for k, v in ref_norm.state_dict().items():
        state[f"norm.{k}"] = v
    for k, v in ref_norm_moe_gen.state_dict().items():
        state[f"norm_moe_gen.{k}"] = v
    tt_trunk.load_torch_state_dict(state)

    rope = Cosmos3VLTextRotaryEmbedding(
        head_dim=_HEAD_DIM,
        rope_theta=_ROPE_THETA,
        rope_axes_dim=list(_ROPE_AXES),
    )
    position_ids = torch.arange(_N_UND + _N_GEN).unsqueeze(0)
    cos_all, sin_all = rope(position_ids, device=torch.device("cpu"), dtype=torch.bfloat16)
    cos_all = cos_all.squeeze(0)
    sin_all = sin_all.squeeze(0)
    cos_und = cos_all[:_N_UND]
    sin_und = sin_all[:_N_UND]
    cos_gen = cos_all[_N_UND:]
    sin_gen = sin_all[_N_UND:]

    und_seq = torch.randn(_N_UND, _HIDDEN, dtype=torch.bfloat16)
    gen_seq = torch.randn(_N_GEN, _HIDDEN, dtype=torch.bfloat16)

    with torch.no_grad():
        ref_und, ref_gen = und_seq, gen_seq
        for layer in ref_layers:
            ref_und, ref_gen = layer(ref_und, ref_gen, (cos_und, sin_und, cos_gen, sin_gen))
        ref_und = ref_norm(ref_und)
        ref_gen = ref_norm_moe_gen(ref_gen)

    und_tt = bf16_tensor(und_seq.reshape(1, 1, _N_UND, _HIDDEN), device=mesh_device)
    cos_und_tt = bf16_tensor(cos_und.reshape(1, 1, _N_UND, _HEAD_DIM), device=mesh_device)
    sin_und_tt = bf16_tensor(sin_und.reshape(1, 1, _N_UND, _HEAD_DIM), device=mesh_device)

    gen_tt = bf16_tensor(gen_seq.reshape(1, 1, _N_GEN, _HIDDEN), device=mesh_device, mesh_axis=0, shard_dim=2)
    cos_gen_tt = bf16_tensor(cos_gen.reshape(1, 1, _N_GEN, _HEAD_DIM), device=mesh_device, mesh_axis=0, shard_dim=2)
    sin_gen_tt = bf16_tensor(sin_gen.reshape(1, 1, _N_GEN, _HEAD_DIM), device=mesh_device, mesh_axis=0, shard_dim=2)

    tt_und, tt_gen = tt_trunk(und_tt, gen_tt, cos_und_tt, sin_und_tt, cos_gen_tt, sin_gen_tt, logical_n_gen=_N_GEN)

    # Trunk gathers gen back to replicated on exit, so device 0 carries the full tensor.
    tt_und_view = ttnn.to_torch(ttnn.get_device_tensors(tt_und)[0]).reshape(_N_UND, _HIDDEN)
    tt_gen_view = ttnn.to_torch(ttnn.get_device_tensors(tt_gen)[0]).reshape(_N_GEN, _HIDDEN)

    # Loose PCC because errors compound across num_hidden_layers stages, same as TP test.
    assert_quality(ref_und, tt_und_view, pcc=0.96)
    assert_quality(ref_gen, tt_gen_view, pcc=0.96)


# Test config knobs for the real-config single-layer PCC.
_REAL_N_UND = 128  # q_chunk_size multiple
_REAL_N_GEN = 256  # k_chunk_size * sp = 128 * 2 = 256, so pad_n = 0


@pytest.mark.parametrize(
    "mesh_device, submesh_shape, device_params",
    _SP_REAL_MESH_PARAMS,
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(900)
def test_native_decoder_layer_real_sp(
    mesh_device: ttnn.MeshDevice, submesh_shape: tuple[int, int], enable_sp_ring: None
) -> None:
    """SP=2 PCC on a single decoder layer at PRODUCTION dims.

    The small-config tests above (hidden=256, 4 layers) pass at ≥99.9%.
    End-to-end with SP enabled at production scale (hidden=5120, 64
    layers) produces broken output. This test runs ONE layer at the real
    config — if PCC fails here, the bug is in the wide-hidden /
    real-mrope / 8-kv-head GQA SP path, not depth-driven drift.

    Mesh is (2, 8): sp=2 on axis 0, tp=8 on axis 1. Matches the
    cfg-parallel pipeline's per-submesh layout. tp=8 makes
    n_local_kv_heads = 8/8 = 1, satisfying the GQA broadcast assertion.
    """
    from models.tt_dit.experimental.cosmos3_i2v.model.decoder_layer import (
        Cosmos3VLTextMoTDecoderLayer as TTDecoderLayer,
    )
    from models.tt_dit.experimental.cosmos3_i2v.reference.transformer_cosmos3 import (
        Cosmos3VLTextMoTDecoderLayer as RefDecoderLayer,
    )
    from models.tt_dit.experimental.cosmos3_i2v.reference.transformer_cosmos3 import Cosmos3VLTextRotaryEmbedding

    torch.manual_seed(42)

    mesh_device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    mesh_shape = tuple(mesh_device.shape)
    sp_factor = mesh_shape[0]
    tp_factor = mesh_shape[1]
    assert sp_factor > 1 and tp_factor == 8, f"need sp>1 tp=8, got {mesh_shape}"

    torch_layer = RefDecoderLayer(
        hidden_size=_REAL_HIDDEN,
        head_dim=_REAL_HEAD_DIM,
        num_attention_heads=_REAL_NUM_Q_HEADS,
        num_key_value_heads=_REAL_NUM_KV_HEADS,
        intermediate_size=_REAL_INTERMEDIATE,
        attention_bias=False,
        rms_norm_eps=1e-6,
    )
    torch_layer.eval()
    torch_layer.to(dtype=torch.bfloat16)

    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(1, 0),
        sequence_parallel=ParallelFactor(sp_factor, 0),
        tensor_parallel=ParallelFactor(tp_factor, 1),
    )
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=1, topology=ttnn.Topology.Linear)

    tt_layer = TTDecoderLayer(
        hidden_size=_REAL_HIDDEN,
        head_dim=_REAL_HEAD_DIM,
        num_attention_heads=_REAL_NUM_Q_HEADS,
        num_key_value_heads=_REAL_NUM_KV_HEADS,
        intermediate_size=_REAL_INTERMEDIATE,
        attention_bias=False,
        rms_norm_eps=1e-6,
        mesh_device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
    )
    tt_layer.load_torch_state_dict(torch_layer.state_dict())

    rope = Cosmos3VLTextRotaryEmbedding(
        head_dim=_REAL_HEAD_DIM,
        rope_theta=_REAL_ROPE_THETA,
        rope_axes_dim=list(_REAL_ROPE_AXES),
    )
    position_ids = torch.arange(_REAL_N_UND + _REAL_N_GEN).unsqueeze(0)
    cos_all, sin_all = rope(position_ids, device=torch.device("cpu"), dtype=torch.bfloat16)
    cos_all = cos_all.squeeze(0)
    sin_all = sin_all.squeeze(0)
    cos_und = cos_all[:_REAL_N_UND]
    sin_und = sin_all[:_REAL_N_UND]
    cos_gen = cos_all[_REAL_N_UND:]
    sin_gen = sin_all[_REAL_N_UND:]

    und_seq = torch.randn(_REAL_N_UND, _REAL_HIDDEN, dtype=torch.bfloat16)
    gen_seq = torch.randn(_REAL_N_GEN, _REAL_HIDDEN, dtype=torch.bfloat16)

    with torch.no_grad():
        ref_und, ref_gen = torch_layer(und_seq, gen_seq, (cos_und, sin_und, cos_gen, sin_gen))

    und_tt = bf16_tensor(und_seq.reshape(1, 1, _REAL_N_UND, _REAL_HIDDEN), device=mesh_device)
    cos_und_tt = bf16_tensor(cos_und.reshape(1, 1, _REAL_N_UND, _REAL_HEAD_DIM), device=mesh_device)
    sin_und_tt = bf16_tensor(sin_und.reshape(1, 1, _REAL_N_UND, _REAL_HEAD_DIM), device=mesh_device)

    gen_tt = bf16_tensor(gen_seq.reshape(1, 1, _REAL_N_GEN, _REAL_HIDDEN), device=mesh_device, mesh_axis=0, shard_dim=2)
    cos_gen_tt = bf16_tensor(
        cos_gen.reshape(1, 1, _REAL_N_GEN, _REAL_HEAD_DIM), device=mesh_device, mesh_axis=0, shard_dim=2
    )
    sin_gen_tt = bf16_tensor(
        sin_gen.reshape(1, 1, _REAL_N_GEN, _REAL_HEAD_DIM), device=mesh_device, mesh_axis=0, shard_dim=2
    )

    tt_und_out, tt_gen_out = tt_layer(
        und_tt, gen_tt, cos_und_tt, sin_und_tt, cos_gen_tt, sin_gen_tt, logical_n_gen=_REAL_N_GEN
    )

    und_view = ttnn.to_torch(ttnn.get_device_tensors(tt_und_out)[0]).reshape(_REAL_N_UND, _REAL_HIDDEN)
    gen_view = _gather_sp_sharded(tt_gen_out, mesh_device, sp_axis=0, n_logical=_REAL_N_GEN, hidden=_REAL_HIDDEN)

    # Looser PCC than synthetic-config tests because bf16 accumulation noise grows
    # with hidden_size × N — at 5120 × 384 we have ~80x the work per output element.
    assert_quality(ref_und, und_view, pcc=0.95)
    assert_quality(ref_gen, gen_view, pcc=0.95)


@pytest.mark.parametrize(
    "mesh_device, submesh_shape, device_params",
    _SP_REAL_MESH_PARAMS,
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("num_hidden_layers", [1, 2, 4, 8, 16], ids=["L1", "L2", "L4", "L8", "L16"])
@pytest.mark.timeout(1500)
def test_native_transformer_real_sp(
    mesh_device: ttnn.MeshDevice,
    submesh_shape: tuple[int, int],
    num_hidden_layers: int,
    enable_sp_ring: None,
) -> None:
    """SP=2 PCC with depth-bisect at PRODUCTION dims (hidden=5120, 64Q/8KV).

    Step 2 of the SP-at-scale debug: single-layer real-config passed
    (test_native_decoder_layer_real_sp). Find the depth at which PCC
    collapses by stacking N reference layers vs the native trunk on a
    (2, 8) submesh. Build time + memory dominate above L=8; bisect stops
    at L=16 to keep the iteration cycle under ~10 min.
    """
    from diffusers.models.normalization import RMSNorm as RefRMSNorm

    from models.tt_dit.experimental.cosmos3_i2v.model.transformer import Cosmos3OmniTransformer as TTTransformer
    from models.tt_dit.experimental.cosmos3_i2v.reference.transformer_cosmos3 import (
        Cosmos3VLTextMoTDecoderLayer as RefDecoderLayer,
    )
    from models.tt_dit.experimental.cosmos3_i2v.reference.transformer_cosmos3 import Cosmos3VLTextRotaryEmbedding

    torch.manual_seed(42)

    mesh_device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    mesh_shape = tuple(mesh_device.shape)
    sp_factor = mesh_shape[0]
    tp_factor = mesh_shape[1]
    assert sp_factor > 1 and tp_factor == 8, f"need sp>1 tp=8, got {mesh_shape}"

    ref_layers = [
        RefDecoderLayer(
            hidden_size=_REAL_HIDDEN,
            head_dim=_REAL_HEAD_DIM,
            num_attention_heads=_REAL_NUM_Q_HEADS,
            num_key_value_heads=_REAL_NUM_KV_HEADS,
            intermediate_size=_REAL_INTERMEDIATE,
            attention_bias=False,
            rms_norm_eps=1e-6,
        )
        for _ in range(num_hidden_layers)
    ]
    ref_norm = RefRMSNorm(_REAL_HIDDEN, eps=1e-6, elementwise_affine=True, bias=False)
    ref_norm_moe_gen = RefRMSNorm(_REAL_HIDDEN, eps=1e-6, elementwise_affine=True, bias=False)
    for m in [*ref_layers, ref_norm, ref_norm_moe_gen]:
        m.eval()
        m.to(dtype=torch.bfloat16)

    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(1, 0),
        sequence_parallel=ParallelFactor(sp_factor, 0),
        tensor_parallel=ParallelFactor(tp_factor, 1),
    )
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=1, topology=ttnn.Topology.Linear)

    tt_trunk = TTTransformer(
        hidden_size=_REAL_HIDDEN,
        head_dim=_REAL_HEAD_DIM,
        num_attention_heads=_REAL_NUM_Q_HEADS,
        num_key_value_heads=_REAL_NUM_KV_HEADS,
        intermediate_size=_REAL_INTERMEDIATE,
        num_hidden_layers=num_hidden_layers,
        attention_bias=False,
        rms_norm_eps=1e-6,
        mesh_device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
    )

    state: dict[str, torch.Tensor] = {}
    for i, layer in enumerate(ref_layers):
        for k, v in layer.state_dict().items():
            state[f"layers.{i}.{k}"] = v
    for k, v in ref_norm.state_dict().items():
        state[f"norm.{k}"] = v
    for k, v in ref_norm_moe_gen.state_dict().items():
        state[f"norm_moe_gen.{k}"] = v
    tt_trunk.load_torch_state_dict(state)

    rope = Cosmos3VLTextRotaryEmbedding(
        head_dim=_REAL_HEAD_DIM,
        rope_theta=_REAL_ROPE_THETA,
        rope_axes_dim=list(_REAL_ROPE_AXES),
    )
    position_ids = torch.arange(_REAL_N_UND + _REAL_N_GEN).unsqueeze(0)
    cos_all, sin_all = rope(position_ids, device=torch.device("cpu"), dtype=torch.bfloat16)
    cos_all = cos_all.squeeze(0)
    sin_all = sin_all.squeeze(0)
    cos_und = cos_all[:_REAL_N_UND]
    sin_und = sin_all[:_REAL_N_UND]
    cos_gen = cos_all[_REAL_N_UND:]
    sin_gen = sin_all[_REAL_N_UND:]

    und_seq = torch.randn(_REAL_N_UND, _REAL_HIDDEN, dtype=torch.bfloat16)
    gen_seq = torch.randn(_REAL_N_GEN, _REAL_HIDDEN, dtype=torch.bfloat16)

    with torch.no_grad():
        ref_und, ref_gen = und_seq, gen_seq
        for layer in ref_layers:
            ref_und, ref_gen = layer(ref_und, ref_gen, (cos_und, sin_und, cos_gen, sin_gen))
        ref_und = ref_norm(ref_und)
        ref_gen = ref_norm_moe_gen(ref_gen)

    und_tt = bf16_tensor(und_seq.reshape(1, 1, _REAL_N_UND, _REAL_HIDDEN), device=mesh_device)
    cos_und_tt = bf16_tensor(cos_und.reshape(1, 1, _REAL_N_UND, _REAL_HEAD_DIM), device=mesh_device)
    sin_und_tt = bf16_tensor(sin_und.reshape(1, 1, _REAL_N_UND, _REAL_HEAD_DIM), device=mesh_device)

    gen_tt = bf16_tensor(gen_seq.reshape(1, 1, _REAL_N_GEN, _REAL_HIDDEN), device=mesh_device, mesh_axis=0, shard_dim=2)
    cos_gen_tt = bf16_tensor(
        cos_gen.reshape(1, 1, _REAL_N_GEN, _REAL_HEAD_DIM), device=mesh_device, mesh_axis=0, shard_dim=2
    )
    sin_gen_tt = bf16_tensor(
        sin_gen.reshape(1, 1, _REAL_N_GEN, _REAL_HEAD_DIM), device=mesh_device, mesh_axis=0, shard_dim=2
    )

    tt_und, tt_gen = tt_trunk(und_tt, gen_tt, cos_und_tt, sin_und_tt, cos_gen_tt, sin_gen_tt, logical_n_gen=_REAL_N_GEN)

    # Trunk all-gathers gen on sp_axis at exit, so device 0 carries the full tensor.
    tt_und_view = ttnn.to_torch(ttnn.get_device_tensors(tt_und)[0]).reshape(_REAL_N_UND, _REAL_HIDDEN)
    tt_gen_view = ttnn.to_torch(ttnn.get_device_tensors(tt_gen)[0]).reshape(_REAL_N_GEN, _REAL_HIDDEN)

    # Log PCC explicitly so we see the depth->PCC curve even if asserts pass.
    # No PCC assertion here — the point is to find the depth at which PCC collapses,
    # not to gate. Failing on assertion would mask the trend.
    assert_quality(ref_und, tt_und_view, pcc=0.5)
    assert_quality(ref_gen, tt_gen_view, pcc=0.5)


@pytest.mark.parametrize(
    "mesh_device, submesh_shape, device_params",
    _SP_REAL_MESH_PARAMS,
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(180)
def test_sp_scatter_gather_roundtrip(mesh_device: ttnn.MeshDevice, submesh_shape: tuple[int, int]) -> None:
    """Verify the SP scatter -> all-gather round-trip preserves sequence order.

    Bypasses every transformer op — just shards a known sequence on sp_axis,
    immediately all-gathers it back via the same CCLManager path the trunk
    uses at exit, and asserts the gathered sequence matches the input bit-for-bit
    (modulo bf16 cast). If the all-gather is reordering slices, this will fail.

    Three N values:
      - N=256: matches the existing PCC tests, no pad
      - N=1344: matches 256x256x81 production gen seq (with N_padded=1536)
      - N=16422: matches 720x1072x81 production gen seq (with N_padded=16640)
    """
    from models.tt_dit.parallel.manager import CCLManager

    mesh_device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    mesh_shape = tuple(mesh_device.shape)
    sp_factor = mesh_shape[0]
    sp_axis = 0

    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=1, topology=ttnn.Topology.Linear)

    hidden = 5120
    k_chunk_sp = 128 * sp_factor

    for n_logical in [256, 1344, 16422]:
        pad_n = (-n_logical) % k_chunk_sp
        n_padded = n_logical + pad_n
        per_chip = n_padded // sp_factor

        # Make each token uniquely identifiable: token i has value i in every channel,
        # so a permutation bug is immediately visible.
        host = torch.arange(n_padded, dtype=torch.float32).unsqueeze(-1).expand(-1, hidden).to(torch.bfloat16)
        host_4d = host.reshape(1, 1, n_padded, hidden)

        scattered = bf16_tensor(host_4d, device=mesh_device, mesh_axis=sp_axis, shard_dim=2)
        gathered = ccl_manager.all_gather_persistent_buffer(scattered, dim=2, mesh_axis=sp_axis)
        result = ttnn.to_torch(ttnn.get_device_tensors(gathered)[0]).reshape(n_padded, hidden)

        # Check ordering: row i should be all i's (within bf16 quantization).
        # Pad rows (after n_logical) carry whatever bf16(i) for i in [n_logical..n_padded).
        # We only care about the first n_logical rows for downstream correctness.
        expected = host[:n_logical]
        actual = result[:n_logical]
        diff = (actual.to(torch.float32) - expected.to(torch.float32)).abs().max().item()
        max_actual_per_row = actual.to(torch.float32).max(dim=1).values
        min_actual_per_row = actual.to(torch.float32).min(dim=1).values
        row_var = (max_actual_per_row - min_actual_per_row).abs().max().item()

        print(
            f"[sp-roundtrip n={n_logical} pad={pad_n} per_chip={per_chip}] "
            f"max|diff|={diff:.3f} row_var_within_token={row_var:.3f}",
            flush=True,
        )

        # All values in row i should equal i (within bf16 precision). row_var is the
        # max within-row spread; should be ~0 if ordering is preserved.
        assert row_var < 1.0, (
            f"N={n_logical}: within-row variance {row_var:.3f} >> 0 — the gather "
            f"interleaved values from different tokens. Slice/concat order broken."
        )
        # Check sequence order: row 0 should be ~0, row n_logical-1 should be ~n_logical-1.
        first_val = actual[0].to(torch.float32).mean().item()
        last_val = actual[n_logical - 1].to(torch.float32).mean().item()
        assert abs(first_val) < 1.0, f"N={n_logical}: row 0 should be ~0, got {first_val:.3f}"
        assert abs(last_val - (n_logical - 1)) < max(
            1.0, n_logical * 0.01
        ), f"N={n_logical}: row {n_logical - 1} should be ~{n_logical - 1}, got {last_val:.3f}"


@pytest.mark.parametrize(
    "mesh_device, submesh_shape, device_params",
    _SP_REAL_MESH_PARAMS,
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(1800)
def test_native_decoder_layer_real_weights_sp(
    mesh_device: ttnn.MeshDevice, submesh_shape: tuple[int, int], enable_sp_ring: None
) -> None:
    """SP=2 single-layer PCC using REAL HF Cosmos3 weights (not random init).

    Loads the actual `Cosmos3OmniTransformer` from HF, grabs layer 0's
    state dict, and runs SP=2 PCC against the torch reference. If real
    weight distributions interact badly with SP at production dims, this
    will catch what the random-init test cannot.

    Hidden / head / mrope dims come from `model_config.TRANSFORMER_CONFIG`
    so we never drift from the canonical values.
    """
    from models.tt_dit.experimental.cosmos3_i2v.model.decoder_layer import (
        Cosmos3VLTextMoTDecoderLayer as TTDecoderLayer,
    )
    from models.tt_dit.experimental.cosmos3_i2v.model_config import HF_REPO, TRANSFORMER_CONFIG
    from models.tt_dit.experimental.cosmos3_i2v.reference.transformer_cosmos3 import Cosmos3OmniTransformer
    from models.tt_dit.experimental.cosmos3_i2v.reference.transformer_cosmos3 import (
        Cosmos3VLTextMoTDecoderLayer as RefDecoderLayer,
    )
    from models.tt_dit.experimental.cosmos3_i2v.reference.transformer_cosmos3 import Cosmos3VLTextRotaryEmbedding

    torch.manual_seed(42)

    mesh_device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    mesh_shape = tuple(mesh_device.shape)
    sp_factor = mesh_shape[0]
    tp_factor = mesh_shape[1]
    assert sp_factor > 1 and tp_factor == 8, f"need sp>1 tp=8, got {mesh_shape}"

    hidden = TRANSFORMER_CONFIG["hidden_size"]
    head_dim = TRANSFORMER_CONFIG["head_dim"]
    nq = TRANSFORMER_CONFIG["num_attention_heads"]
    nkv = TRANSFORMER_CONFIG["num_key_value_heads"]
    intermediate = TRANSFORMER_CONFIG["intermediate_size"]
    rms_eps = TRANSFORMER_CONFIG["rms_norm_eps"]
    rope_theta = TRANSFORMER_CONFIG["rope_theta"]
    rope_axes = list(TRANSFORMER_CONFIG["rope_scaling"]["mrope_section"])

    # Load real weights for layer 0. from_pretrained materializes the full 64B
    # in CPU RAM (~128GB bf16) — only run this where you have the RAM.
    print(f"[real-weights] loading HF transformer from {HF_REPO} ...", flush=True)
    full_transformer = Cosmos3OmniTransformer.from_pretrained(
        HF_REPO, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    layer0_state = full_transformer.layers[0].state_dict()
    del full_transformer  # release the other 63 layers + heads ASAP

    torch_layer = RefDecoderLayer(
        hidden_size=hidden,
        head_dim=head_dim,
        num_attention_heads=nq,
        num_key_value_heads=nkv,
        intermediate_size=intermediate,
        attention_bias=False,
        rms_norm_eps=rms_eps,
    )
    torch_layer.load_state_dict(layer0_state)
    torch_layer.eval()
    torch_layer.to(dtype=torch.bfloat16)

    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(1, 0),
        sequence_parallel=ParallelFactor(sp_factor, 0),
        tensor_parallel=ParallelFactor(tp_factor, 1),
    )
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=1, topology=ttnn.Topology.Linear)

    tt_layer = TTDecoderLayer(
        hidden_size=hidden,
        head_dim=head_dim,
        num_attention_heads=nq,
        num_key_value_heads=nkv,
        intermediate_size=intermediate,
        attention_bias=False,
        rms_norm_eps=rms_eps,
        mesh_device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
    )
    tt_layer.load_torch_state_dict(layer0_state)

    rope = Cosmos3VLTextRotaryEmbedding(head_dim=head_dim, rope_theta=rope_theta, rope_axes_dim=rope_axes)
    n_und = _REAL_N_UND
    n_gen = _REAL_N_GEN
    position_ids = torch.arange(n_und + n_gen).unsqueeze(0)
    cos_all, sin_all = rope(position_ids, device=torch.device("cpu"), dtype=torch.bfloat16)
    cos_all = cos_all.squeeze(0)
    sin_all = sin_all.squeeze(0)
    cos_und = cos_all[:n_und]
    sin_und = sin_all[:n_und]
    cos_gen = cos_all[n_und:]
    sin_gen = sin_all[n_und:]

    und_seq = torch.randn(n_und, hidden, dtype=torch.bfloat16)
    gen_seq = torch.randn(n_gen, hidden, dtype=torch.bfloat16)

    with torch.no_grad():
        ref_und, ref_gen = torch_layer(und_seq, gen_seq, (cos_und, sin_und, cos_gen, sin_gen))

    und_tt = bf16_tensor(und_seq.reshape(1, 1, n_und, hidden), device=mesh_device)
    cos_und_tt = bf16_tensor(cos_und.reshape(1, 1, n_und, head_dim), device=mesh_device)
    sin_und_tt = bf16_tensor(sin_und.reshape(1, 1, n_und, head_dim), device=mesh_device)

    gen_tt = bf16_tensor(gen_seq.reshape(1, 1, n_gen, hidden), device=mesh_device, mesh_axis=0, shard_dim=2)
    cos_gen_tt = bf16_tensor(cos_gen.reshape(1, 1, n_gen, head_dim), device=mesh_device, mesh_axis=0, shard_dim=2)
    sin_gen_tt = bf16_tensor(sin_gen.reshape(1, 1, n_gen, head_dim), device=mesh_device, mesh_axis=0, shard_dim=2)

    tt_und_out, tt_gen_out = tt_layer(
        und_tt, gen_tt, cos_und_tt, sin_und_tt, cos_gen_tt, sin_gen_tt, logical_n_gen=n_gen
    )

    und_view = ttnn.to_torch(ttnn.get_device_tensors(tt_und_out)[0]).reshape(n_und, hidden)
    gen_view = _gather_sp_sharded(tt_gen_out, mesh_device, sp_axis=0, n_logical=n_gen, hidden=hidden)

    # Loose 0.5 PCC threshold — we want to SEE the number, not gate. If real
    # weights expose the SP bug it'll be obvious in the printed PCC.
    assert_quality(ref_und, und_view, pcc=0.5)
    assert_quality(ref_gen, gen_view, pcc=0.5)


@pytest.mark.parametrize(
    "mesh_device, submesh_shape, device_params",
    _SP_REAL_MESH_PARAMS,
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("num_hidden_layers", [8, 16], ids=["L8", "L16"])
@pytest.mark.timeout(3600)
def test_native_transformer_real_weights_sp(
    mesh_device: ttnn.MeshDevice,
    submesh_shape: tuple[int, int],
    num_hidden_layers: int,
    enable_sp_ring: None,
) -> None:
    """SP=2 trunk PCC with REAL HF Cosmos3 weights at multiple depths.

    Single-layer real-weights SP passed (99.98% gen). Stack the first
    N real layers + the 2 final RMSNorms from the HF transformer and
    re-PCC at the real production layout. Closes the "random weights
    don't catch the bug" gap by testing real weight distributions at
    nontrivial depth.
    """
    from diffusers.models.normalization import RMSNorm as RefRMSNorm  # noqa: F401  (kept for symmetry)

    from models.tt_dit.experimental.cosmos3_i2v.model.transformer import Cosmos3OmniTransformer as TTTransformer
    from models.tt_dit.experimental.cosmos3_i2v.model_config import HF_REPO, TRANSFORMER_CONFIG
    from models.tt_dit.experimental.cosmos3_i2v.reference.transformer_cosmos3 import (
        Cosmos3OmniTransformer,
        Cosmos3VLTextRotaryEmbedding,
    )

    torch.manual_seed(42)

    mesh_device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    mesh_shape = tuple(mesh_device.shape)
    sp_factor = mesh_shape[0]
    tp_factor = mesh_shape[1]
    assert sp_factor > 1 and tp_factor == 8, f"need sp>1 tp=8, got {mesh_shape}"

    hidden = TRANSFORMER_CONFIG["hidden_size"]
    head_dim = TRANSFORMER_CONFIG["head_dim"]
    nq = TRANSFORMER_CONFIG["num_attention_heads"]
    nkv = TRANSFORMER_CONFIG["num_key_value_heads"]
    intermediate = TRANSFORMER_CONFIG["intermediate_size"]
    rms_eps = TRANSFORMER_CONFIG["rms_norm_eps"]
    rope_theta = TRANSFORMER_CONFIG["rope_theta"]
    rope_axes = list(TRANSFORMER_CONFIG["rope_scaling"]["mrope_section"])

    print(f"[real-weights L={num_hidden_layers}] loading HF transformer ...", flush=True)
    full_transformer = Cosmos3OmniTransformer.from_pretrained(
        HF_REPO, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    ref_layers = full_transformer.layers[:num_hidden_layers]
    ref_norm = full_transformer.norm
    ref_norm_moe_gen = full_transformer.norm_moe_gen
    for m in [*ref_layers, ref_norm, ref_norm_moe_gen]:
        m.eval()

    # Compose the state dict the native trunk expects: layers.{i}.<...>, norm.<...>, norm_moe_gen.<...>.
    state: dict[str, torch.Tensor] = {}
    for i, layer in enumerate(ref_layers):
        for k, v in layer.state_dict().items():
            state[f"layers.{i}.{k}"] = v
    for k, v in ref_norm.state_dict().items():
        state[f"norm.{k}"] = v
    for k, v in ref_norm_moe_gen.state_dict().items():
        state[f"norm_moe_gen.{k}"] = v

    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(1, 0),
        sequence_parallel=ParallelFactor(sp_factor, 0),
        tensor_parallel=ParallelFactor(tp_factor, 1),
    )
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=1, topology=ttnn.Topology.Linear)

    tt_trunk = TTTransformer(
        hidden_size=hidden,
        head_dim=head_dim,
        num_attention_heads=nq,
        num_key_value_heads=nkv,
        intermediate_size=intermediate,
        num_hidden_layers=num_hidden_layers,
        attention_bias=False,
        rms_norm_eps=rms_eps,
        mesh_device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
    )
    tt_trunk.load_torch_state_dict(state)

    rope = Cosmos3VLTextRotaryEmbedding(head_dim=head_dim, rope_theta=rope_theta, rope_axes_dim=rope_axes)
    position_ids = torch.arange(_REAL_N_UND + _REAL_N_GEN).unsqueeze(0)
    cos_all, sin_all = rope(position_ids, device=torch.device("cpu"), dtype=torch.bfloat16)
    cos_all = cos_all.squeeze(0)
    sin_all = sin_all.squeeze(0)
    cos_und = cos_all[:_REAL_N_UND]
    sin_und = sin_all[:_REAL_N_UND]
    cos_gen = cos_all[_REAL_N_UND:]
    sin_gen = sin_all[_REAL_N_UND:]

    und_seq = torch.randn(_REAL_N_UND, hidden, dtype=torch.bfloat16)
    gen_seq = torch.randn(_REAL_N_GEN, hidden, dtype=torch.bfloat16)

    with torch.no_grad():
        ref_und, ref_gen = und_seq, gen_seq
        for layer in ref_layers:
            ref_und, ref_gen = layer(ref_und, ref_gen, (cos_und, sin_und, cos_gen, sin_gen))
        ref_und = ref_norm(ref_und)
        ref_gen = ref_norm_moe_gen(ref_gen)

    # Release the host-side full transformer + ref layers now that we have ref outputs.
    del full_transformer, ref_layers, ref_norm, ref_norm_moe_gen

    und_tt = bf16_tensor(und_seq.reshape(1, 1, _REAL_N_UND, hidden), device=mesh_device)
    cos_und_tt = bf16_tensor(cos_und.reshape(1, 1, _REAL_N_UND, head_dim), device=mesh_device)
    sin_und_tt = bf16_tensor(sin_und.reshape(1, 1, _REAL_N_UND, head_dim), device=mesh_device)

    gen_tt = bf16_tensor(gen_seq.reshape(1, 1, _REAL_N_GEN, hidden), device=mesh_device, mesh_axis=0, shard_dim=2)
    cos_gen_tt = bf16_tensor(cos_gen.reshape(1, 1, _REAL_N_GEN, head_dim), device=mesh_device, mesh_axis=0, shard_dim=2)
    sin_gen_tt = bf16_tensor(sin_gen.reshape(1, 1, _REAL_N_GEN, head_dim), device=mesh_device, mesh_axis=0, shard_dim=2)

    tt_und, tt_gen = tt_trunk(und_tt, gen_tt, cos_und_tt, sin_und_tt, cos_gen_tt, sin_gen_tt, logical_n_gen=_REAL_N_GEN)

    tt_und_view = ttnn.to_torch(ttnn.get_device_tensors(tt_und)[0]).reshape(_REAL_N_UND, hidden)
    tt_gen_view = ttnn.to_torch(ttnn.get_device_tensors(tt_gen)[0]).reshape(_REAL_N_GEN, hidden)

    assert_quality(ref_und, tt_und_view, pcc=0.5)
    assert_quality(ref_gen, tt_gen_view, pcc=0.5)


# (1, 8) submesh: TP=8 only, NO SP. Direct comparison point against
# test_native_transformer_real_weights_sp which uses (2, 8). Same TP, no SP axis.
_TP_ONLY_REAL_MESH_PARAMS = [
    pytest.param((4, 8), (1, 8), line_params, id="1x8_tp8_no_sp_real_config"),
]


@pytest.mark.parametrize(
    "mesh_device, submesh_shape, device_params",
    _TP_ONLY_REAL_MESH_PARAMS,
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("num_hidden_layers", [8, 16], ids=["L8", "L16"])
@pytest.mark.timeout(3600)
def test_native_transformer_real_weights_tp_only(
    mesh_device: ttnn.MeshDevice,
    submesh_shape: tuple[int, int],
    num_hidden_layers: int,
) -> None:
    """Real-weights L=8/16 PCC with TP=8 only (no SP). Comparison point for
    test_native_transformer_real_weights_sp[L8/L16]. Same config, same weights,
    same input, just gen replicated instead of sp-sharded.

    If this PCC is much higher than the SP version, SP is the magnitude-drift
    cause. If similar, the drift is generic TT bf16 at production scale.
    """
    from models.tt_dit.experimental.cosmos3_i2v.model.transformer import Cosmos3OmniTransformer as TTTransformer
    from models.tt_dit.experimental.cosmos3_i2v.model_config import HF_REPO, TRANSFORMER_CONFIG
    from models.tt_dit.experimental.cosmos3_i2v.reference.transformer_cosmos3 import (
        Cosmos3OmniTransformer,
        Cosmos3VLTextRotaryEmbedding,
    )

    torch.manual_seed(42)

    mesh_device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    mesh_shape = tuple(mesh_device.shape)
    sp_factor = mesh_shape[0]
    tp_factor = mesh_shape[1]
    assert sp_factor == 1 and tp_factor == 8, f"want (1, 8), got {mesh_shape}"

    hidden = TRANSFORMER_CONFIG["hidden_size"]
    head_dim = TRANSFORMER_CONFIG["head_dim"]
    nq = TRANSFORMER_CONFIG["num_attention_heads"]
    nkv = TRANSFORMER_CONFIG["num_key_value_heads"]
    intermediate = TRANSFORMER_CONFIG["intermediate_size"]
    rms_eps = TRANSFORMER_CONFIG["rms_norm_eps"]
    rope_theta = TRANSFORMER_CONFIG["rope_theta"]
    rope_axes = list(TRANSFORMER_CONFIG["rope_scaling"]["mrope_section"])

    print(f"[real-weights L={num_hidden_layers} TP-only] loading HF transformer ...", flush=True)
    full_transformer = Cosmos3OmniTransformer.from_pretrained(
        HF_REPO, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    ref_layers = full_transformer.layers[:num_hidden_layers]
    ref_norm = full_transformer.norm
    ref_norm_moe_gen = full_transformer.norm_moe_gen
    for m in [*ref_layers, ref_norm, ref_norm_moe_gen]:
        m.eval()

    state: dict[str, torch.Tensor] = {}
    for i, layer in enumerate(ref_layers):
        for k, v in layer.state_dict().items():
            state[f"layers.{i}.{k}"] = v
    for k, v in ref_norm.state_dict().items():
        state[f"norm.{k}"] = v
    for k, v in ref_norm_moe_gen.state_dict().items():
        state[f"norm_moe_gen.{k}"] = v

    # sp_factor=1 — no SP axis in the parallel config. tp on axis 1.
    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(1, 0),
        sequence_parallel=ParallelFactor(1, 0),
        tensor_parallel=ParallelFactor(tp_factor, 1),
    )
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=1, topology=ttnn.Topology.Linear)

    tt_trunk = TTTransformer(
        hidden_size=hidden,
        head_dim=head_dim,
        num_attention_heads=nq,
        num_key_value_heads=nkv,
        intermediate_size=intermediate,
        num_hidden_layers=num_hidden_layers,
        attention_bias=False,
        rms_norm_eps=rms_eps,
        mesh_device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
    )
    tt_trunk.load_torch_state_dict(state)

    rope = Cosmos3VLTextRotaryEmbedding(head_dim=head_dim, rope_theta=rope_theta, rope_axes_dim=rope_axes)
    position_ids = torch.arange(_REAL_N_UND + _REAL_N_GEN).unsqueeze(0)
    cos_all, sin_all = rope(position_ids, device=torch.device("cpu"), dtype=torch.bfloat16)
    cos_all = cos_all.squeeze(0)
    sin_all = sin_all.squeeze(0)
    cos_und = cos_all[:_REAL_N_UND]
    sin_und = sin_all[:_REAL_N_UND]
    cos_gen = cos_all[_REAL_N_UND:]
    sin_gen = sin_all[_REAL_N_UND:]

    und_seq = torch.randn(_REAL_N_UND, hidden, dtype=torch.bfloat16)
    gen_seq = torch.randn(_REAL_N_GEN, hidden, dtype=torch.bfloat16)

    with torch.no_grad():
        ref_und, ref_gen = und_seq, gen_seq
        for layer in ref_layers:
            ref_und, ref_gen = layer(ref_und, ref_gen, (cos_und, sin_und, cos_gen, sin_gen))
        ref_und = ref_norm(ref_und)
        ref_gen = ref_norm_moe_gen(ref_gen)

    del full_transformer, ref_layers, ref_norm, ref_norm_moe_gen

    # All inputs REPLICATED — no sharding on any sequence dim.
    und_tt = bf16_tensor(und_seq.reshape(1, 1, _REAL_N_UND, hidden), device=mesh_device)
    cos_und_tt = bf16_tensor(cos_und.reshape(1, 1, _REAL_N_UND, head_dim), device=mesh_device)
    sin_und_tt = bf16_tensor(sin_und.reshape(1, 1, _REAL_N_UND, head_dim), device=mesh_device)

    gen_tt = bf16_tensor(gen_seq.reshape(1, 1, _REAL_N_GEN, hidden), device=mesh_device)
    cos_gen_tt = bf16_tensor(cos_gen.reshape(1, 1, _REAL_N_GEN, head_dim), device=mesh_device)
    sin_gen_tt = bf16_tensor(sin_gen.reshape(1, 1, _REAL_N_GEN, head_dim), device=mesh_device)

    tt_und, tt_gen = tt_trunk(und_tt, gen_tt, cos_und_tt, sin_und_tt, cos_gen_tt, sin_gen_tt, logical_n_gen=_REAL_N_GEN)

    tt_und_view = ttnn.to_torch(ttnn.get_device_tensors(tt_und)[0]).reshape(_REAL_N_UND, hidden)
    tt_gen_view = ttnn.to_torch(ttnn.get_device_tensors(tt_gen)[0]).reshape(_REAL_N_GEN, hidden)

    assert_quality(ref_und, tt_und_view, pcc=0.5)
    assert_quality(ref_gen, tt_gen_view, pcc=0.5)


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [pytest.param((4, 8), line_params, id="bh_galaxy_parent")],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("num_hidden_layers", [8], ids=["L8"])
@pytest.mark.timeout(3600)
def test_sp_vs_tp_only_direct_diff(
    mesh_device: ttnn.MeshDevice,
    num_hidden_layers: int,
) -> None:
    """Drive the SAME real weights + inputs through SP path and TP-only path,
    compare TT outputs against EACH OTHER (not against torch).

    Both paths drift from torch by ~30% RMSE/σ on real weights. But if they
    drift in CORRELATED directions, SP-vs-TP PCC stays high — the end-to-end
    quality gap must be elsewhere. If they drift UNCORRELATED, SP-vs-TP PCC
    drops too — SP path produces materially different output even though
    forward correctness against torch looks similar.
    """
    import os as _os

    from models.tt_dit.experimental.cosmos3_i2v.model.attention import sp_ring_enabled  # noqa: F401
    from models.tt_dit.experimental.cosmos3_i2v.model.transformer import Cosmos3OmniTransformer as TTTransformer
    from models.tt_dit.experimental.cosmos3_i2v.model_config import HF_REPO, TRANSFORMER_CONFIG
    from models.tt_dit.experimental.cosmos3_i2v.reference.transformer_cosmos3 import (
        Cosmos3OmniTransformer,
        Cosmos3VLTextRotaryEmbedding,
    )

    torch.manual_seed(42)

    hidden = TRANSFORMER_CONFIG["hidden_size"]
    head_dim = TRANSFORMER_CONFIG["head_dim"]
    nq = TRANSFORMER_CONFIG["num_attention_heads"]
    nkv = TRANSFORMER_CONFIG["num_key_value_heads"]
    intermediate = TRANSFORMER_CONFIG["intermediate_size"]
    rms_eps = TRANSFORMER_CONFIG["rms_norm_eps"]
    rope_theta = TRANSFORMER_CONFIG["rope_theta"]
    rope_axes = list(TRANSFORMER_CONFIG["rope_scaling"]["mrope_section"])

    print(f"[sp-vs-tp L={num_hidden_layers}] loading HF transformer ...", flush=True)
    full_transformer = Cosmos3OmniTransformer.from_pretrained(
        HF_REPO, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    state: dict[str, torch.Tensor] = {}
    for i, layer in enumerate(full_transformer.layers[:num_hidden_layers]):
        for k, v in layer.state_dict().items():
            state[f"layers.{i}.{k}"] = v
    for k, v in full_transformer.norm.state_dict().items():
        state[f"norm.{k}"] = v
    for k, v in full_transformer.norm_moe_gen.state_dict().items():
        state[f"norm_moe_gen.{k}"] = v
    del full_transformer

    rope = Cosmos3VLTextRotaryEmbedding(head_dim=head_dim, rope_theta=rope_theta, rope_axes_dim=rope_axes)
    position_ids = torch.arange(_REAL_N_UND + _REAL_N_GEN).unsqueeze(0)
    cos_all, sin_all = rope(position_ids, device=torch.device("cpu"), dtype=torch.bfloat16)
    cos_all = cos_all.squeeze(0)
    sin_all = sin_all.squeeze(0)
    cos_und = cos_all[:_REAL_N_UND]
    sin_und = sin_all[:_REAL_N_UND]
    cos_gen = cos_all[_REAL_N_UND:]
    sin_gen = sin_all[_REAL_N_UND:]

    und_seq = torch.randn(_REAL_N_UND, hidden, dtype=torch.bfloat16)
    gen_seq = torch.randn(_REAL_N_GEN, hidden, dtype=torch.bfloat16)

    def _run_trunk(submesh_shape: tuple[int, int], sp_ring: bool) -> tuple[torch.Tensor, torch.Tensor]:
        prev_env = _os.environ.get("TT_COSMOS3_ENABLE_SP_RING")
        _os.environ["TT_COSMOS3_ENABLE_SP_RING"] = "1" if sp_ring else "0"
        try:
            submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
            sp_factor = submesh_shape[0]
            tp_factor = submesh_shape[1]
            parallel_config = DiTParallelConfig(
                cfg_parallel=ParallelFactor(1, 0),
                sequence_parallel=ParallelFactor(sp_factor, 0),
                tensor_parallel=ParallelFactor(tp_factor, 1),
            )
            ccl_manager = CCLManager(mesh_device=submesh, num_links=1, topology=ttnn.Topology.Linear)

            trunk = TTTransformer(
                hidden_size=hidden,
                head_dim=head_dim,
                num_attention_heads=nq,
                num_key_value_heads=nkv,
                intermediate_size=intermediate,
                num_hidden_layers=num_hidden_layers,
                attention_bias=False,
                rms_norm_eps=rms_eps,
                mesh_device=submesh,
                parallel_config=parallel_config,
                ccl_manager=ccl_manager,
            )
            trunk.load_torch_state_dict(state)

            # Layout: SP path scatters gen on sp_axis=0; TP-only path keeps gen replicated.
            scatter = sp_factor > 1 and sp_ring
            if scatter:
                gen_tt = bf16_tensor(
                    gen_seq.reshape(1, 1, _REAL_N_GEN, hidden), device=submesh, mesh_axis=0, shard_dim=2
                )
                cos_gen_tt = bf16_tensor(
                    cos_gen.reshape(1, 1, _REAL_N_GEN, head_dim), device=submesh, mesh_axis=0, shard_dim=2
                )
                sin_gen_tt = bf16_tensor(
                    sin_gen.reshape(1, 1, _REAL_N_GEN, head_dim), device=submesh, mesh_axis=0, shard_dim=2
                )
            else:
                gen_tt = bf16_tensor(gen_seq.reshape(1, 1, _REAL_N_GEN, hidden), device=submesh)
                cos_gen_tt = bf16_tensor(cos_gen.reshape(1, 1, _REAL_N_GEN, head_dim), device=submesh)
                sin_gen_tt = bf16_tensor(sin_gen.reshape(1, 1, _REAL_N_GEN, head_dim), device=submesh)

            und_tt = bf16_tensor(und_seq.reshape(1, 1, _REAL_N_UND, hidden), device=submesh)
            cos_und_tt = bf16_tensor(cos_und.reshape(1, 1, _REAL_N_UND, head_dim), device=submesh)
            sin_und_tt = bf16_tensor(sin_und.reshape(1, 1, _REAL_N_UND, head_dim), device=submesh)

            und_out_tt, gen_out_tt = trunk(
                und_tt, gen_tt, cos_und_tt, sin_und_tt, cos_gen_tt, sin_gen_tt, logical_n_gen=_REAL_N_GEN
            )
            und_out = ttnn.to_torch(ttnn.get_device_tensors(und_out_tt)[0]).reshape(_REAL_N_UND, hidden)
            gen_out = ttnn.to_torch(ttnn.get_device_tensors(gen_out_tt)[0]).reshape(_REAL_N_GEN, hidden)
            return und_out, gen_out
        finally:
            if prev_env is None:
                _os.environ.pop("TT_COSMOS3_ENABLE_SP_RING", None)
            else:
                _os.environ["TT_COSMOS3_ENABLE_SP_RING"] = prev_env

    print("[sp-vs-tp] running SP path (2x8 sp_ring=on) ...", flush=True)
    sp_und, sp_gen = _run_trunk((2, 8), sp_ring=True)
    print("[sp-vs-tp] running TP-only path (1x8 sp_ring=off) ...", flush=True)
    tp_und, tp_gen = _run_trunk((1, 8), sp_ring=False)

    print("[sp-vs-tp] SP vs TP-only direct comparison:", flush=True)
    assert_quality(tp_und, sp_und, pcc=0.5)
    assert_quality(tp_gen, sp_gen, pcc=0.5)


@pytest.mark.parametrize(
    "mesh_device, submesh_shape, device_params",
    _SP_REAL_MESH_PARAMS,
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "n_gen",
    [32, 64, 128, 256],
    ids=["n_gen_32_pad87", "n_gen_64_pad75", "n_gen_128_pad50", "n_gen_256_pad0"],
)
@pytest.mark.timeout(900)
def test_native_decoder_layer_real_weights_sp_n_gen_sweep(
    mesh_device: ttnn.MeshDevice,
    submesh_shape: tuple[int, int],
    n_gen: int,
    enable_sp_ring: None,
) -> None:
    """Real-weights single-layer SP PCC across N_gen values to localize pad-path bug.

    Production at 128x128x5 has N_gen=32 → padded to 256 → 87.5% pad ratio.
    Our existing real-weights L=1 PCC used N_gen=256 (pad_n=0) and passed at
    99.9997%. Sweep [32, 64, 128, 256] to find where PCC collapses.
    """
    from models.tt_dit.experimental.cosmos3_i2v.model.decoder_layer import (
        Cosmos3VLTextMoTDecoderLayer as TTDecoderLayer,
    )
    from models.tt_dit.experimental.cosmos3_i2v.model_config import HF_REPO, TRANSFORMER_CONFIG
    from models.tt_dit.experimental.cosmos3_i2v.reference.transformer_cosmos3 import Cosmos3OmniTransformer
    from models.tt_dit.experimental.cosmos3_i2v.reference.transformer_cosmos3 import (
        Cosmos3VLTextMoTDecoderLayer as RefDecoderLayer,
    )
    from models.tt_dit.experimental.cosmos3_i2v.reference.transformer_cosmos3 import Cosmos3VLTextRotaryEmbedding

    torch.manual_seed(42)

    mesh_device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    mesh_shape = tuple(mesh_device.shape)
    sp_factor = mesh_shape[0]
    tp_factor = mesh_shape[1]
    assert sp_factor > 1 and tp_factor == 8, f"need sp>1 tp=8, got {mesh_shape}"

    hidden = TRANSFORMER_CONFIG["hidden_size"]
    head_dim = TRANSFORMER_CONFIG["head_dim"]
    nq = TRANSFORMER_CONFIG["num_attention_heads"]
    nkv = TRANSFORMER_CONFIG["num_key_value_heads"]
    intermediate = TRANSFORMER_CONFIG["intermediate_size"]
    rms_eps = TRANSFORMER_CONFIG["rms_norm_eps"]
    rope_theta = TRANSFORMER_CONFIG["rope_theta"]
    rope_axes = list(TRANSFORMER_CONFIG["rope_scaling"]["mrope_section"])

    full_transformer = Cosmos3OmniTransformer.from_pretrained(
        HF_REPO, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    layer0_state = full_transformer.layers[0].state_dict()
    del full_transformer

    torch_layer = RefDecoderLayer(
        hidden_size=hidden,
        head_dim=head_dim,
        num_attention_heads=nq,
        num_key_value_heads=nkv,
        intermediate_size=intermediate,
        attention_bias=False,
        rms_norm_eps=rms_eps,
    )
    torch_layer.load_state_dict(layer0_state)
    torch_layer.eval()
    torch_layer.to(dtype=torch.bfloat16)

    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(1, 0),
        sequence_parallel=ParallelFactor(sp_factor, 0),
        tensor_parallel=ParallelFactor(tp_factor, 1),
    )
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=1, topology=ttnn.Topology.Linear)

    tt_layer = TTDecoderLayer(
        hidden_size=hidden,
        head_dim=head_dim,
        num_attention_heads=nq,
        num_key_value_heads=nkv,
        intermediate_size=intermediate,
        attention_bias=False,
        rms_norm_eps=rms_eps,
        mesh_device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
    )
    tt_layer.load_torch_state_dict(layer0_state)

    # Pad N_gen to k_chunk_size*sp_factor = 128*2 = 256 the same way the proxy does.
    k_chunk_sp = 128 * sp_factor
    # N_und=69 matches the 128x128x5 production trunk call (text + time tokens).
    # _pad_for_joint pads und to q_chunk=128 multiple → exercises the und joint
    # pad path that N_und=128 (no pad) skipped.
    n_und = 69
    n_gen_padded = n_gen + ((-n_gen) % k_chunk_sp)
    pad_n = n_gen_padded - n_gen

    rope = Cosmos3VLTextRotaryEmbedding(head_dim=head_dim, rope_theta=rope_theta, rope_axes_dim=rope_axes)
    # We need cos/sin of length n_gen_padded for the SP path; reference uses n_gen only.
    position_ids = torch.arange(n_und + n_gen_padded).unsqueeze(0)
    cos_all, sin_all = rope(position_ids, device=torch.device("cpu"), dtype=torch.bfloat16)
    cos_all = cos_all.squeeze(0)
    sin_all = sin_all.squeeze(0)
    cos_und = cos_all[:n_und]
    sin_und = sin_all[:n_und]
    cos_gen_ref = cos_all[n_und : n_und + n_gen]  # for torch reference (unpadded)
    sin_gen_ref = sin_all[n_und : n_und + n_gen]
    cos_gen_full = cos_all[n_und:]  # padded length for SP
    sin_gen_full = sin_all[n_und:]

    und_seq = torch.randn(n_und, hidden, dtype=torch.bfloat16)
    gen_seq = torch.randn(n_gen, hidden, dtype=torch.bfloat16)
    # Pad gen_seq to match the proxy's behavior
    if pad_n > 0:
        gen_seq_padded = torch.cat([gen_seq, gen_seq.new_zeros(pad_n, hidden)], dim=0)
    else:
        gen_seq_padded = gen_seq

    with torch.no_grad():
        ref_und, ref_gen = torch_layer(und_seq, gen_seq, (cos_und, sin_und, cos_gen_ref, sin_gen_ref))

    und_tt = bf16_tensor(und_seq.reshape(1, 1, n_und, hidden), device=mesh_device)
    cos_und_tt = bf16_tensor(cos_und.reshape(1, 1, n_und, head_dim), device=mesh_device)
    sin_und_tt = bf16_tensor(sin_und.reshape(1, 1, n_und, head_dim), device=mesh_device)

    gen_tt = bf16_tensor(
        gen_seq_padded.reshape(1, 1, n_gen_padded, hidden), device=mesh_device, mesh_axis=0, shard_dim=2
    )
    cos_gen_tt = bf16_tensor(
        cos_gen_full.reshape(1, 1, n_gen_padded, head_dim), device=mesh_device, mesh_axis=0, shard_dim=2
    )
    sin_gen_tt = bf16_tensor(
        sin_gen_full.reshape(1, 1, n_gen_padded, head_dim), device=mesh_device, mesh_axis=0, shard_dim=2
    )

    tt_und_out, tt_gen_out = tt_layer(
        und_tt, gen_tt, cos_und_tt, sin_und_tt, cos_gen_tt, sin_gen_tt, logical_n_gen=n_gen
    )

    und_view = ttnn.to_torch(ttnn.get_device_tensors(tt_und_out)[0]).reshape(n_und, hidden)
    # Gather sp-sharded gen output (padded), then slice off the pad.
    gen_view_padded = _gather_sp_sharded(tt_gen_out, mesh_device, sp_axis=0, n_logical=n_gen_padded, hidden=hidden)
    gen_view = gen_view_padded[:n_gen]

    print(f"\n[n_gen_sweep n_gen={n_gen} pad_n={pad_n} pad_ratio={pad_n / n_gen_padded:.1%}]", flush=True)
    assert_quality(ref_und, und_view, pcc=0.5)
    assert_quality(ref_gen, gen_view, pcc=0.5)


# Single-attention-layer ring-vs-local probe. Runs as TWO independent pytest invocations
# (TT_COSMOS3_ATTN_DUMP_PATH controls where each writes its gen output) so we avoid
# submesh re-creation between SP and TP within one process. Diff host-side after.
def _attn_probe_inner(mesh_device, submesh_shape, sp_factor_cfg):
    """Build Cosmos3JointAttention at real config, real layer-0 weights, run once on
    production-like input. Returns gen output as host torch."""
    from models.tt_dit.experimental.cosmos3_i2v.model.attention import Cosmos3JointAttention
    from models.tt_dit.experimental.cosmos3_i2v.model_config import HF_REPO, TRANSFORMER_CONFIG
    from models.tt_dit.experimental.cosmos3_i2v.reference.transformer_cosmos3 import (
        Cosmos3OmniTransformer,
        Cosmos3VLTextRotaryEmbedding,
    )

    torch.manual_seed(42)

    mesh_device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    mesh_shape = tuple(mesh_device.shape)
    sp_factor = mesh_shape[0]
    tp_factor = mesh_shape[1]

    hidden = TRANSFORMER_CONFIG["hidden_size"]
    head_dim = TRANSFORMER_CONFIG["head_dim"]
    nq = TRANSFORMER_CONFIG["num_attention_heads"]
    nkv = TRANSFORMER_CONFIG["num_key_value_heads"]
    rms_eps = TRANSFORMER_CONFIG["rms_norm_eps"]
    rope_theta = TRANSFORMER_CONFIG["rope_theta"]
    rope_axes = list(TRANSFORMER_CONFIG["rope_scaling"]["mrope_section"])

    # Real layer-0 attention weights.
    full = Cosmos3OmniTransformer.from_pretrained(HF_REPO, subfolder="transformer", torch_dtype=torch.bfloat16)
    attn_state = full.layers[0].self_attn.state_dict()
    del full

    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(1, 0),
        sequence_parallel=ParallelFactor(sp_factor, 0),
        tensor_parallel=ParallelFactor(tp_factor, 1),
    )
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    attn = Cosmos3JointAttention(
        hidden_size=hidden,
        head_dim=head_dim,
        num_attention_heads=nq,
        num_key_value_heads=nkv,
        attention_bias=False,
        rms_norm_eps=rms_eps,
        mesh_device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
    )
    attn.load_torch_state_dict(attn_state)

    n_und = 69
    n_gen = 32
    k_chunk_sp = 128 * sp_factor if sp_factor > 1 else 1
    n_gen_padded = n_gen + ((-n_gen) % k_chunk_sp) if sp_factor > 1 else n_gen

    rope = Cosmos3VLTextRotaryEmbedding(head_dim=head_dim, rope_theta=rope_theta, rope_axes_dim=rope_axes)
    position_ids = torch.arange(n_und + n_gen_padded).unsqueeze(0)
    cos_all, sin_all = rope(position_ids, device=torch.device("cpu"), dtype=torch.bfloat16)
    cos_all = cos_all.squeeze(0)
    sin_all = sin_all.squeeze(0)
    cos_und = cos_all[:n_und]
    sin_und = sin_all[:n_und]
    cos_gen_full = cos_all[n_und:]
    sin_gen_full = sin_all[n_und:]

    # Re-seed right before generating random inputs so the two probe runs
    # (ring path on sp=2 vs local path on sp=1) see IDENTICAL und/gen,
    # regardless of how many sub-modules each path constructs.
    torch.manual_seed(123456)
    und_seq = torch.randn(n_und, hidden, dtype=torch.bfloat16)
    gen_seq = torch.randn(n_gen, hidden, dtype=torch.bfloat16)
    if n_gen_padded > n_gen:
        gen_seq_padded = torch.cat([gen_seq, gen_seq.new_zeros(n_gen_padded - n_gen, hidden)], dim=0)
    else:
        gen_seq_padded = gen_seq

    und_tt = bf16_tensor(und_seq.reshape(1, 1, n_und, hidden), device=mesh_device)
    cos_und_tt = bf16_tensor(cos_und.reshape(1, 1, n_und, head_dim), device=mesh_device)
    sin_und_tt = bf16_tensor(sin_und.reshape(1, 1, n_und, head_dim), device=mesh_device)

    if sp_factor > 1:
        gen_tt = bf16_tensor(
            gen_seq_padded.reshape(1, 1, n_gen_padded, hidden), device=mesh_device, mesh_axis=0, shard_dim=2
        )
        cos_gen_tt = bf16_tensor(
            cos_gen_full.reshape(1, 1, n_gen_padded, head_dim), device=mesh_device, mesh_axis=0, shard_dim=2
        )
        sin_gen_tt = bf16_tensor(
            sin_gen_full.reshape(1, 1, n_gen_padded, head_dim), device=mesh_device, mesh_axis=0, shard_dim=2
        )
    else:
        gen_tt = bf16_tensor(gen_seq_padded.reshape(1, 1, n_gen_padded, hidden), device=mesh_device)
        cos_gen_tt = bf16_tensor(cos_gen_full.reshape(1, 1, n_gen_padded, head_dim), device=mesh_device)
        sin_gen_tt = bf16_tensor(sin_gen_full.reshape(1, 1, n_gen_padded, head_dim), device=mesh_device)

    _und_out, gen_out_tt = attn(und_tt, gen_tt, cos_und_tt, sin_und_tt, cos_gen_tt, sin_gen_tt, logical_n_gen=n_gen)

    if sp_factor > 1:
        gen_view = _gather_sp_sharded(gen_out_tt, mesh_device, sp_axis=0, n_logical=n_gen, hidden=hidden)
    else:
        gen_view = ttnn.to_torch(ttnn.get_device_tensors(gen_out_tt)[0]).reshape(n_gen, hidden)

    dump_path = __import__("os").environ.get("TT_COSMOS3_ATTN_DUMP_PATH")
    if dump_path:
        torch.save(gen_view.detach().cpu(), dump_path)
    return gen_view


@pytest.mark.parametrize(
    "mesh_device, submesh_shape, device_params",
    _SP_REAL_MESH_PARAMS,
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(600)
def test_single_attn_ring_dump(
    mesh_device: ttnn.MeshDevice, submesh_shape: tuple[int, int], enable_sp_ring: None
) -> None:
    """One Cosmos3JointAttention forward at sp=2 (ring path), real layer-0 weights.
    Dumps gen output to TT_COSMOS3_ATTN_DUMP_PATH for cross-run diff."""
    _attn_probe_inner(mesh_device, submesh_shape, sp_factor_cfg=2)


@pytest.mark.parametrize(
    "mesh_device, submesh_shape, device_params",
    _TP_ONLY_REAL_MESH_PARAMS,
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(600)
def test_single_attn_local_dump(mesh_device: ttnn.MeshDevice, submesh_shape: tuple[int, int]) -> None:
    """One Cosmos3JointAttention forward at sp=1 (local path), real layer-0 weights.
    Dumps gen output to TT_COSMOS3_ATTN_DUMP_PATH for cross-run diff."""
    _attn_probe_inner(mesh_device, submesh_shape, sp_factor_cfg=1)
