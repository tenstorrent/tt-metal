# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Device tests for overlapped (fused) weight extraction.

Tests all three constituents of get_tt_q_ab_proj_and_kv_a_proj_weights
(using an NCRISC kernel to extract each sub-tensor, then verifying
against an independently preprocessed + bfp8 round-tripped reference):
  - q_a_proj (packed)
  - q_b_proj (shuffled)
  - kv_a_proj (shard-reordered)

Tests all constituents of get_tt_o_proj_and_gate_mm_weights
(using CopyToOutput to extract each sub-tensor, then verifying
against a dtype round-tripped reference):
  - o_proj  (BFP8)
  - gate_mm (BFP16)
  - attn_norm (BFP16, 1x32 tile)
  - q_norm (BFP16, 1x32 tile)
  - kv_norm (BFP16, 1x32 tile)
  - ffn_norm (BFP16, 1x32 tile)

Tests both constituents of get_tt_kv_b12_proj_weights
(using CopyToOutput to extract each sub-tensor, then verifying
against a dtype round-tripped reference):
  - kv_b1_proj (HEIGHT_SHARDED, BFP8)
  - kv_b2_proj (HEIGHT_SHARDED, BFP8)

Tests both constituents of get_tt_moe_shared_expert_weights
(using CopyToOutput to extract each sub-tensor, then verifying
against a dtype round-tripped reference):
  - gate_proj (block-sharded HEIGHT_SHARDED, BFP4)
  - up_proj   (block-sharded HEIGHT_SHARDED, BFP4)
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.blitz_decode_weights import (
    GATE_UP_PROJ_SINGLE_DEVICE_OVERLAP_SPEC,
    KVB12_PROJ_SINGLE_DEVICE_OVERLAP_SPEC,
    O_PROJ_GATE_MM_RMSNORM_GAMMA_SINGLE_DEVICE_OVERLAP_SPEC,
    QAB_KVA_PROJ_SINGLE_DEVICE_OVERLAP_SPEC,
    BlitzDecodeWeights,
)
from models.demos.deepseek_v3_b1.tests.blitz_weights_tests.op import CopyToOutput


# ---------------------------------------------------------------------------
# Device tensor helpers
# ---------------------------------------------------------------------------
def _shard_shape(height, width, num_cores, sharding):
    """Compute the per-core shard shape for the given sharding strategy."""
    if sharding == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        return (height // num_cores, width)
    return (height, width // num_cores)


def _create_output_device_tensor(
    height,
    width,
    core_range_set,
    device,
    dtype=ttnn.bfloat8_b,
    sharding=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    tile=None,
    mesh_mapper=None,
):
    """Allocate a zeroed output tensor on device."""
    num_cores = core_range_set.num_cores()
    shard = _shard_shape(height, width, num_cores, sharding)
    shard_spec = ttnn.ShardSpec(core_range_set, shard, ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(sharding, ttnn.BufferType.L1, shard_spec)
    kwargs = {}
    if tile is not None:
        kwargs["tile"] = tile
    if mesh_mapper is not None:
        kwargs["mesh_mapper"] = mesh_mapper
    return ttnn.from_torch(
        torch.zeros(height, width, dtype=torch.bfloat16),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem_config,
        **kwargs,
    )


def _get_roundtrip_reference(
    torch_data,
    core_range_set,
    device,
    dtype=ttnn.bfloat8_b,
    sharding=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    tile=None,
    mesh_mapper=None,
    mesh_composer=None,
):
    """Send torch data through a dtype round-trip on device to get ground truth."""
    height, width = torch_data.shape
    num_cores = core_range_set.num_cores()
    shard = _shard_shape(height, width, num_cores, sharding)
    shard_spec = ttnn.ShardSpec(core_range_set, shard, ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(sharding, ttnn.BufferType.L1, shard_spec)
    kwargs = {}
    if tile is not None:
        kwargs["tile"] = tile
    if mesh_mapper is not None:
        kwargs["mesh_mapper"] = mesh_mapper
    tt = ttnn.from_torch(
        torch_data,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem_config,
        **kwargs,
    )
    to_torch_kwargs = {}
    if mesh_composer is not None:
        to_torch_kwargs["mesh_composer"] = mesh_composer
    result = ttnn.to_torch(tt, **to_torch_kwargs)
    ttnn.deallocate(tt)
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mesh_rows, mesh_cols", [(4, 2), (1, 1)])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_q_ab_proj_kv_a_proj_overlap(bh_2d_mesh_device, mesh_rows, mesh_cols):
    """Verify all three constituents of the q_ab_proj + kv_a_proj overlap.

    Creates the fused tensor once via BlitzDecodeWeights, then extracts
    each sub-tensor with CopyToOutput and checks it against an
    independently preprocessed + bfp8 round-tripped reference.

    For multi-device (4x2 mesh), mla_tp=2 across mesh columns:
    q_b_proj is TP-sharded on the width, q_a_proj and kv_a_proj are replicated.

    Per-device layout::

        q_a_proj (7168, 1536) packed to (3584, 3072) as bfloat8_b
          WIDTH_SHARDED on 96 cores (12x8), shard (3584, 32)

        q_b_proj (1536, 12288) as bfloat8_b
          WIDTH_SHARDED on 96 cores (12x8), shard (1536, 128)

        kv_a_proj (7168, 576) as bfloat8_b, shard-reordered
          WIDTH_SHARDED on 18 cores (9x2), shard (7168, 32)
    """
    num_devices = mesh_rows * mesh_cols
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than are available on this platform")

    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((mesh_rows, mesh_cols)))
    mla_tp = mesh_cols
    cfg = QAB_KVA_PROJ_SINGLE_DEVICE_OVERLAP_SPEC

    torch.manual_seed(42)
    q_a_raw = torch.randn(cfg.q_a_proj_shape, dtype=torch.bfloat16)
    q_b_raw = torch.randn(cfg.q_b_proj_shape[0], cfg.q_b_proj_shape[1] * mla_tp, dtype=torch.bfloat16)
    kv_raw = torch.randn(cfg.kv_a_proj_shape, dtype=torch.bfloat16)

    bdw = BlitzDecodeWeights(submesh)
    logger.info("Building fused q_ab_proj + kv_a_proj weights ...")
    q_a, q_b, kv = bdw.get_tt_q_ab_proj_and_kv_a_proj_weights(q_a_raw, q_b_raw, kv_raw)

    replicate = ttnn.ReplicateTensorToMesh(submesh)
    composer = ttnn.ConcatMeshToTensor(submesh, dim=0)
    single_device = submesh.create_submesh(ttnn.MeshShape((1, 1)))

    # -- Extract all three sub-tensors while fused buffer is alive -----------
    logger.info("Extracting q_a_proj from fused buffer ...")
    q_a_out = _create_output_device_tensor(
        *q_a.tensor_shape, q_a.core_range_set, submesh, dtype=q_a.dtype, mesh_mapper=replicate
    )
    q_a_out = CopyToOutput.op(q_a.fused_tensor, q_a_out, byte_offset=q_a.byte_offset)
    q_a_result = ttnn.to_torch(q_a_out, mesh_composer=composer)
    ttnn.deallocate(q_a_out)

    logger.info("Extracting q_b_proj from fused buffer ...")
    q_b_out = _create_output_device_tensor(
        *q_b.tensor_shape, q_b.core_range_set, submesh, dtype=q_b.dtype, mesh_mapper=replicate
    )
    q_b_out = CopyToOutput.op(q_b.fused_tensor, q_b_out, byte_offset=q_b.byte_offset)
    q_b_result = ttnn.to_torch(q_b_out, mesh_composer=composer)
    ttnn.deallocate(q_b_out)

    logger.info("Extracting kv_a_proj from fused buffer ...")
    kv_out = _create_output_device_tensor(
        *kv.tensor_shape, kv.core_range_set, submesh, dtype=kv.dtype, mesh_mapper=replicate
    )
    kv_out = CopyToOutput.op(kv.fused_tensor, kv_out, byte_offset=kv.byte_offset)
    kv_result = ttnn.to_torch(kv_out, mesh_composer=composer)
    ttnn.deallocate(kv_out)

    ttnn.deallocate(q_a.fused_tensor)

    # -- Build references (bfp8 round-trip on single device) -----------------
    logger.info("Building q_a_proj round-trip reference ...")
    q_a_ref = _get_roundtrip_reference(
        cfg.shuffle_q_a(q_a_raw),
        q_a.core_range_set,
        single_device,
        dtype=q_a.dtype,
    )
    logger.info("Building kv_a_proj round-trip reference ...")
    kv_ref = _get_roundtrip_reference(
        cfg.shuffle_kv_a(kv_raw),
        kv.core_range_set,
        single_device,
        dtype=kv.dtype,
    )

    logger.info("Building q_b_proj per-TP round-trip references ...")
    q_b_tp_refs = []
    for tp_idx in range(mla_tp):
        q_b_slice = cfg.get_q_b_slice(q_b_raw, tp_idx, mla_tp)
        ref = _get_roundtrip_reference(
            cfg.shuffle_q_b(q_b_slice),
            q_b.core_range_set,
            single_device,
            dtype=q_b.dtype,
        )
        q_b_tp_refs.append(ref)

    ttnn.close_mesh_device(single_device)

    q_a_h = q_a.tensor_shape[0]
    q_b_h = q_b.tensor_shape[0]
    kv_h = kv.tensor_shape[0]

    logger.info("Verifying per-device results ...")
    for device_idx in range(num_devices):
        tp_group = device_idx % mesh_cols

        q_a_slice = q_a_result[device_idx * q_a_h : (device_idx + 1) * q_a_h]
        assert torch.equal(q_a_slice, q_a_ref), f"q_a_proj mismatch on device {device_idx}"

        q_b_slice = q_b_result[device_idx * q_b_h : (device_idx + 1) * q_b_h]
        assert torch.equal(
            q_b_slice, q_b_tp_refs[tp_group]
        ), f"q_b_proj mismatch on device {device_idx} (TP={tp_group})"

        kv_slice = kv_result[device_idx * kv_h : (device_idx + 1) * kv_h]
        assert torch.equal(kv_slice, kv_ref), f"kv_a_proj mismatch on device {device_idx}"

        logger.info(f"Device {device_idx} (TP={tp_group}): q_a_proj, q_b_proj, kv_a_proj overlap passed")


@pytest.mark.parametrize("mesh_rows, mesh_cols", [(4, 2), (1, 1)])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_o_proj_gate_mm_rmsnorm_gamma_overlap(bh_2d_mesh_device, mesh_rows, mesh_cols):
    """Verify all constituents of the o_proj + gate_mm + rmsnorm gamma overlap.

    Creates the fused tensor once via BlitzDecodeWeights, then extracts
    each sub-tensor with CopyToOutput and checks it against an
    independently preprocessed + dtype round-tripped reference.

    For multi-device (4x2 mesh), mla_tp=2 across mesh columns:
    o_proj is TP-sharded on the inner dim (8192), everything else is replicated.

    Per-device layout::

        o_proj (8192, 7168) as bfloat8_b
          WIDTH_SHARDED on 112 cores, shard (8192, 64)

        gate_mm (7168, 256) as bfloat16
          WIDTH_SHARDED on 8 cores, shard (7168, 32)

        attn_norm (1, 7168) as bfloat16, 1x32 tiles, on core (12, 9)
        q_norm   (1, 1536) as bfloat16, 1x32 tiles, on core (12, 9)
        kv_norm  (1, 512)  as bfloat16, 1x32 tiles, on core (0, 8)
        ffn_norm (1, 7168) as bfloat16, 1x32 tiles, on core (12, 9)
    """
    num_devices = mesh_rows * mesh_cols
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than are available on this platform")

    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((mesh_rows, mesh_cols)))
    mla_tp = mesh_cols
    cfg = O_PROJ_GATE_MM_RMSNORM_GAMMA_SINGLE_DEVICE_OVERLAP_SPEC
    tile_1x32 = ttnn.Tile([1, 32])

    torch.manual_seed(42)
    o_proj_raw = torch.randn(cfg.o_proj_shape[0] * mla_tp, cfg.o_proj_shape[1], dtype=torch.bfloat16)
    gate_mm_raw = torch.randn(cfg.gate_mm_shape, dtype=torch.bfloat16)
    attn_norm_raw = torch.randn(cfg.attn_norm_shape, dtype=torch.bfloat16)
    q_norm_raw = torch.randn(cfg.q_norm_shape, dtype=torch.bfloat16)
    kv_norm_raw = torch.randn(cfg.kv_norm_shape, dtype=torch.bfloat16)
    ffn_norm_raw = torch.randn(cfg.ffn_norm_shape, dtype=torch.bfloat16)

    bdw = BlitzDecodeWeights(submesh)
    logger.info("Building fused o_proj + gate_mm + rmsnorm gamma weights ...")
    o_proj, gate_mm, attn_norm_g, q_norm_g, kv_norm_g, ffn_norm_g = bdw.get_tt_o_proj_and_gate_mm_weights(
        o_proj_raw,
        gate_mm_raw,
        attn_norm_raw,
        q_norm_raw,
        kv_norm_raw,
        ffn_norm_raw,
    )

    replicate = ttnn.ReplicateTensorToMesh(submesh)
    composer = ttnn.ConcatMeshToTensor(submesh, dim=0)
    single_device = submesh.create_submesh(ttnn.MeshShape((1, 1)))

    # -- Extract o_proj (BFP8) while fused buffer is alive -------------------
    logger.info("Extracting o_proj from fused buffer ...")
    o_out = _create_output_device_tensor(
        *o_proj.tensor_shape, o_proj.core_range_set, submesh, dtype=o_proj.dtype, mesh_mapper=replicate
    )
    o_out = CopyToOutput.op(o_proj.fused_tensor, o_out, byte_offset=o_proj.byte_offset)
    o_result = ttnn.to_torch(o_out, mesh_composer=composer)
    ttnn.deallocate(o_out)

    # -- Extract gate_mm (BFP16) while fused buffer is alive -----------------
    logger.info("Extracting gate_mm from fused buffer ...")
    g_out = _create_output_device_tensor(
        *gate_mm.tensor_shape, gate_mm.core_range_set, submesh, dtype=gate_mm.dtype, mesh_mapper=replicate
    )
    g_out = CopyToOutput.op(gate_mm.fused_tensor, g_out, byte_offset=gate_mm.byte_offset)
    g_result = ttnn.to_torch(g_out, mesh_composer=composer)
    ttnn.deallocate(g_out)

    # -- Extract gammas (BFP16 1x32 tiles) while fused buffer is alive ------
    gamma_results = {}
    for name, ov in [
        ("attn_norm", attn_norm_g),
        ("q_norm", q_norm_g),
        ("kv_norm", kv_norm_g),
        ("ffn_norm", ffn_norm_g),
    ]:
        out = _create_output_device_tensor(
            *ov.tensor_shape,
            ov.core_range_set,
            submesh,
            dtype=ov.dtype,
            sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            tile=tile_1x32,
            mesh_mapper=replicate,
        )
        logger.info(f"Extracting {name} from fused buffer ...")
        out = CopyToOutput.op(ov.fused_tensor, out, byte_offset=ov.byte_offset)
        gamma_results[name] = ttnn.to_torch(out, mesh_composer=composer)
        ttnn.deallocate(out)

    ttnn.deallocate(o_proj.fused_tensor)

    # -- Build references (dtype round-trip on single device) ----------------
    logger.info("Building o_proj per-TP round-trip references ...")
    per_device_o_h = cfg.o_proj_shape[0]
    o_tp_refs = []
    for tp_idx in range(mla_tp):
        o_slice = o_proj_raw[tp_idx * per_device_o_h : (tp_idx + 1) * per_device_o_h, :]
        ref = _get_roundtrip_reference(
            o_slice,
            o_proj.core_range_set,
            single_device,
            dtype=o_proj.dtype,
        )
        o_tp_refs.append(ref)

    logger.info("Building gate_mm round-trip reference ...")
    g_ref = _get_roundtrip_reference(
        gate_mm_raw,
        gate_mm.core_range_set,
        single_device,
        dtype=gate_mm.dtype,
    )

    gamma_refs = {}
    for name, raw, ov in [
        ("attn_norm", attn_norm_raw, attn_norm_g),
        ("q_norm", q_norm_raw, q_norm_g),
        ("kv_norm", kv_norm_raw, kv_norm_g),
        ("ffn_norm", ffn_norm_raw, ffn_norm_g),
    ]:
        logger.info(f"Building {name} round-trip reference ...")
        gamma_refs[name] = _get_roundtrip_reference(
            raw,
            ov.core_range_set,
            single_device,
            dtype=ov.dtype,
            sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            tile=tile_1x32,
        )

    ttnn.close_mesh_device(single_device)

    o_h = o_proj.tensor_shape[0]
    g_h = gate_mm.tensor_shape[0]

    logger.info("Verifying per-device results ...")
    for device_idx in range(num_devices):
        tp_group = device_idx % mesh_cols

        o_slice = o_result[device_idx * o_h : (device_idx + 1) * o_h]
        assert torch.equal(o_slice, o_tp_refs[tp_group]), f"o_proj mismatch on device {device_idx} (TP={tp_group})"

        g_slice = g_result[device_idx * g_h : (device_idx + 1) * g_h]
        assert torch.equal(g_slice, g_ref), f"gate_mm mismatch on device {device_idx}"

        for name in ["attn_norm", "q_norm", "kv_norm", "ffn_norm"]:
            gamma_h = gamma_results[name].shape[0] // num_devices
            gamma_slice = gamma_results[name][device_idx * gamma_h : (device_idx + 1) * gamma_h]
            assert torch.equal(gamma_slice, gamma_refs[name]), f"{name} mismatch on device {device_idx}"

        logger.info(f"Device {device_idx} (TP={tp_group}): o_proj, gate_mm, gammas overlap passed")


@pytest.mark.parametrize("mesh_rows, mesh_cols", [(4, 2), (1, 1)])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_kv_b12_proj_overlap(bh_2d_mesh_device, mesh_rows, mesh_cols):
    """Verify both constituents of the kv_b1 + kv_b2 overlap.

    Creates the fused tensor once via BlitzDecodeWeights, then extracts
    each sub-tensor with CopyToOutput and checks it against an
    independently preprocessed + bfp8 round-tripped reference.

    For multi-device (4x2 mesh), mla_tp=2 across mesh columns:
    kv_b1_proj is TP-sharded on height (8192), kv_b2_proj on width (8192).

    Per-device layout::

        kv_b1_proj (8192, 512) as bfloat8_b
          HEIGHT_SHARDED on 64 cores (8x8 Qnope grid), shard (128, 512)

        kv_b2_proj (512, 8192) pre-transposed to (8192, 512) as bfloat8_b
          HEIGHT_SHARDED on 64 cores, shard (128, 512)

    For multi-device (4x2 mesh), mla_tp=2 across mesh columns:
    both tensors are TP-sharded on the 8192 (heads) dim.
    """
    num_devices = mesh_rows * mesh_cols
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than are available on this platform")

    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((mesh_rows, mesh_cols)))
    mla_tp = mesh_cols
    cfg = KVB12_PROJ_SINGLE_DEVICE_OVERLAP_SPEC

    torch.manual_seed(42)
    kv_b1_raw = torch.randn(cfg.kv_b1_proj_shape[0] * mla_tp, cfg.kv_b1_proj_shape[1], dtype=torch.bfloat16)
    kv_b2_raw = torch.randn(cfg.kv_b2_proj_shape[0], cfg.kv_b2_proj_shape[1] * mla_tp, dtype=torch.bfloat16)

    bdw = BlitzDecodeWeights(submesh)
    logger.info("Building fused kv_b1 + kv_b2 weights ...")
    kv_b1, kv_b2 = bdw.get_tt_kv_b12_proj_weights(kv_b1_raw, kv_b2_raw)

    replicate = ttnn.ReplicateTensorToMesh(submesh)
    composer = ttnn.ConcatMeshToTensor(submesh, dim=0)
    single_device = submesh.create_submesh(ttnn.MeshShape((1, 1)))

    # -- Extract kv_b1 (HEIGHT_SHARDED, BFP8) while fused buffer is alive ----
    logger.info("Extracting kv_b1_proj from fused buffer ...")
    kv_b1_out = _create_output_device_tensor(
        *kv_b1.tensor_shape,
        kv_b1.core_range_set,
        submesh,
        dtype=kv_b1.dtype,
        sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        mesh_mapper=replicate,
    )
    kv_b1_out = CopyToOutput.op(kv_b1.fused_tensor, kv_b1_out, byte_offset=kv_b1.byte_offset)
    kv_b1_result = ttnn.to_torch(kv_b1_out, mesh_composer=composer)
    ttnn.deallocate(kv_b1_out)

    # -- Extract kv_b2 (HEIGHT_SHARDED, BFP8) while fused buffer is alive -----
    logger.info("Extracting kv_b2_proj from fused buffer ...")
    kv_b2_out = _create_output_device_tensor(
        *cfg.kv_b1_proj_shape,
        kv_b2.core_range_set,
        submesh,
        dtype=kv_b2.dtype,
        sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        mesh_mapper=replicate,
    )
    kv_b2_out = CopyToOutput.op(kv_b2.fused_tensor, kv_b2_out, byte_offset=kv_b2.byte_offset)
    kv_b2_result = ttnn.to_torch(kv_b2_out, mesh_composer=composer)
    ttnn.deallocate(kv_b2_out)

    ttnn.deallocate(kv_b1.fused_tensor)

    # -- Build per-TP references (bfp8 round-trip on single device) ----------
    per_device_b1_h = cfg.kv_b1_proj_shape[0]
    per_device_b2_w = cfg.kv_b2_proj_shape[1]

    kv_b1_tp_refs = []
    kv_b2_tp_refs = []
    for tp_idx in range(mla_tp):
        logger.info(f"Building kv_b1_proj round-trip reference (TP={tp_idx}) ...")
        b1_slice = kv_b1_raw[tp_idx * per_device_b1_h : (tp_idx + 1) * per_device_b1_h, :]
        ref = _get_roundtrip_reference(
            b1_slice,
            kv_b1.core_range_set,
            single_device,
            dtype=kv_b1.dtype,
            sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        )
        kv_b1_tp_refs.append(ref)

        logger.info(f"Building kv_b2_proj round-trip reference (TP={tp_idx}) ...")
        b2_slice = kv_b2_raw[:, tp_idx * per_device_b2_w : (tp_idx + 1) * per_device_b2_w]
        b2_physical = cfg.shuffle_kv_b2(b2_slice)
        ref = _get_roundtrip_reference(
            b2_physical,
            kv_b2.core_range_set,
            single_device,
            dtype=kv_b2.dtype,
            sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        )
        kv_b2_tp_refs.append(ref)

    ttnn.close_mesh_device(single_device)

    kv_b1_h = kv_b1.tensor_shape[0]

    logger.info("Verifying per-device results ...")
    for device_idx in range(num_devices):
        tp_group = device_idx % mesh_cols

        b1_slice = kv_b1_result[device_idx * kv_b1_h : (device_idx + 1) * kv_b1_h]
        assert torch.equal(
            b1_slice, kv_b1_tp_refs[tp_group]
        ), f"kv_b1_proj mismatch on device {device_idx} (TP={tp_group})"

        b2_slice = kv_b2_result[device_idx * kv_b1_h : (device_idx + 1) * kv_b1_h]
        assert torch.equal(
            b2_slice, kv_b2_tp_refs[tp_group]
        ), f"kv_b2_proj mismatch on device {device_idx} (TP={tp_group})"

        logger.info(f"Device {device_idx} (TP={tp_group}): kv_b1_proj, kv_b2_proj overlap passed")


@pytest.mark.parametrize("mesh_rows, mesh_cols", [(4, 2), (1, 1)])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_gate_up_proj_overlap(bh_2d_mesh_device, mesh_rows, mesh_cols):
    """Verify both constituents of the gate + up projection overlap.

    Creates the fused tensor once via BlitzDecodeWeights, then extracts
    each sub-tensor with CopyToOutput and checks it against an
    independently preprocessed + bfp4 round-tripped reference.

    For multi-device (4x2 mesh), moe_tp=8 across all devices:
    both gate_proj and up_proj are TP-sharded on the outer dim (N=2048,
    per-device N=256).

    Per-device layout::

        gate_proj (7168, 256) as bfloat4_b, block-sharded
          stacked (57344, 32), shard (896, 32), 64 A cores

        up_proj (7168, 256) as bfloat4_b, block-sharded
          stacked (57344, 32), shard (896, 32), 64 B cores

        combined: 128 cores, HEIGHT_SHARDED (114688, 32)
    """
    num_devices = mesh_rows * mesh_cols
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than are available on this platform")

    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((mesh_rows, mesh_cols)))
    moe_tp = num_devices
    cfg = GATE_UP_PROJ_SINGLE_DEVICE_OVERLAP_SPEC

    torch.manual_seed(42)
    gate_raw = torch.randn(cfg.gate_proj_shape[0], cfg.gate_proj_shape[1] * moe_tp, dtype=torch.bfloat16)
    up_raw = torch.randn(cfg.up_proj_shape[0], cfg.up_proj_shape[1] * moe_tp, dtype=torch.bfloat16)
    down_raw = torch.randn(256 * moe_tp, 7168, dtype=torch.bfloat16)

    replicate = ttnn.ReplicateTensorToMesh(submesh)
    composer = ttnn.ConcatMeshToTensor(submesh, dim=0)
    single_device = submesh.create_submesh(ttnn.MeshShape((1, 1)))

    bdw = BlitzDecodeWeights(submesh)
    logger.info("Building fused gate + up proj weights ...")
    gate, up, _ = bdw.get_tt_moe_shared_expert_weights(gate_raw, up_raw, down_raw)

    # -- Extract gate (block-sharded BFP4) while fused buffer is alive -------
    logger.info("Extracting gate_proj from fused buffer ...")
    gate_out = _create_output_device_tensor(
        *cfg.stacked_shape,
        gate.core_range_set,
        submesh,
        dtype=gate.dtype,
        sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        mesh_mapper=replicate,
    )
    gate_out = CopyToOutput.op(gate.fused_tensor, gate_out, byte_offset=gate.byte_offset)
    gate_result = ttnn.to_torch(gate_out, mesh_composer=composer)
    ttnn.deallocate(gate_out)

    # -- Extract up (block-sharded BFP4) while fused buffer is alive ---------
    logger.info("Extracting up_proj from fused buffer ...")
    up_out = _create_output_device_tensor(
        *cfg.stacked_shape,
        up.core_range_set,
        submesh,
        dtype=up.dtype,
        sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        mesh_mapper=replicate,
    )
    up_out = CopyToOutput.op(up.fused_tensor, up_out, byte_offset=up.byte_offset)
    up_result = ttnn.to_torch(up_out, mesh_composer=composer)
    ttnn.deallocate(up_out)

    ttnn.deallocate(gate.fused_tensor)

    # -- Build per-TP references (bfp4 round-trip on single device) ----------
    per_device_n = cfg.gate_proj_shape[1]

    logger.info("Building per-TP gate_proj round-trip references ...")
    gate_tp_refs = []
    for tp_idx in range(moe_tp):
        gate_slice = gate_raw[:, tp_idx * per_device_n : (tp_idx + 1) * per_device_n]
        ref = _get_roundtrip_reference(
            cfg.reshuffle_block_to_height_sharded(gate_slice, cfg.gate_core_range_set),
            gate.core_range_set,
            single_device,
            dtype=gate.dtype,
            sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        )
        gate_tp_refs.append(ref)

    logger.info("Building per-TP up_proj round-trip references ...")
    up_tp_refs = []
    for tp_idx in range(moe_tp):
        up_slice = up_raw[:, tp_idx * per_device_n : (tp_idx + 1) * per_device_n]
        ref = _get_roundtrip_reference(
            cfg.reshuffle_block_to_height_sharded(up_slice, cfg.up_core_range_set),
            up.core_range_set,
            single_device,
            dtype=up.dtype,
            sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        )
        up_tp_refs.append(ref)

    ttnn.close_mesh_device(single_device)

    # -- Verify per-device ---------------------------------------------------
    gate_h = cfg.stacked_shape[0]
    up_h = cfg.stacked_shape[0]

    logger.info("Verifying per-device results ...")
    for device_idx in range(num_devices):
        tp_group = device_idx

        gate_slice = gate_result[device_idx * gate_h : (device_idx + 1) * gate_h]
        assert torch.equal(
            gate_slice, gate_tp_refs[tp_group]
        ), f"gate_proj mismatch on device {device_idx} (TP={tp_group})"

        up_slice = up_result[device_idx * up_h : (device_idx + 1) * up_h]
        assert torch.equal(up_slice, up_tp_refs[tp_group]), f"up_proj mismatch on device {device_idx} (TP={tp_group})"

        logger.info(f"Device {device_idx} (TP={tp_group}): gate_proj, up_proj overlap passed")
