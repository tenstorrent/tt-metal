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

Tests both constituents of get_tt_gate_up_proj_weights
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
    GATE_UP_PROJ_OVERLAP_CFG,
    KVB12_PROJ_OVERLAP_CFG,
    O_PROJ_GATE_MM_RMSNORM_GAMMA_OVERLAP_CFG,
    QAB_KVA_PROJ_OVERLAP_CFG,
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

    For multi-device (4x2 mesh), TP=2 across mesh columns:
    q_b_proj is sharded, q_a_proj and kv_a_proj are replicated.
    """
    num_devices = mesh_rows * mesh_cols
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than are available on this platform")

    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((mesh_rows, mesh_cols)))
    tp = mesh_cols
    cfg = QAB_KVA_PROJ_OVERLAP_CFG

    torch.manual_seed(42)
    q_a_raw = torch.randn(cfg.q_a_proj_shape, dtype=torch.bfloat16)
    q_b_raw = torch.randn(cfg.q_b_proj_shape[0], cfg.q_b_proj_shape[1] * tp, dtype=torch.bfloat16)
    kv_raw = torch.randn(cfg.kv_a_proj_shape, dtype=torch.bfloat16)

    bdw = BlitzDecodeWeights(submesh)
    q_a, q_b, kv = bdw.get_tt_q_ab_proj_and_kv_a_proj_weights(q_a_raw, q_b_raw, kv_raw)

    replicate = ttnn.ReplicateTensorToMesh(submesh)
    composer = ttnn.ConcatMeshToTensor(submesh, dim=0)

    # -- Extract all three sub-tensors while fused buffer is alive -----------
    q_a_out = _create_output_device_tensor(
        *q_a.tensor_shape, q_a.core_range_set, submesh, dtype=q_a.dtype, mesh_mapper=replicate
    )
    q_a_out = CopyToOutput.op(q_a.fused_tensor, q_a_out, byte_offset=q_a.byte_offset)
    q_a_result = ttnn.to_torch(q_a_out, mesh_composer=composer)
    ttnn.deallocate(q_a_out)

    q_b_out = _create_output_device_tensor(
        *q_b.tensor_shape, q_b.core_range_set, submesh, dtype=q_b.dtype, mesh_mapper=replicate
    )
    q_b_out = CopyToOutput.op(q_b.fused_tensor, q_b_out, byte_offset=q_b.byte_offset)
    q_b_result = ttnn.to_torch(q_b_out, mesh_composer=composer)
    ttnn.deallocate(q_b_out)

    kv_out = _create_output_device_tensor(
        *kv.tensor_shape, kv.core_range_set, submesh, dtype=kv.dtype, mesh_mapper=replicate
    )
    kv_out = CopyToOutput.op(kv.fused_tensor, kv_out, byte_offset=kv.byte_offset)
    kv_result = ttnn.to_torch(kv_out, mesh_composer=composer)
    ttnn.deallocate(kv_out)

    ttnn.deallocate(q_a.fused_tensor)

    # -- Build references (bfp8 round-trip) after fused buffer is freed ------
    # q_a and kv_a are replicated: build a single reference and check all devices.
    q_a_ref = _get_roundtrip_reference(
        cfg.shuffle_q_a(q_a_raw),
        q_a.core_range_set,
        submesh,
        dtype=q_a.dtype,
        mesh_mapper=replicate,
        mesh_composer=composer,
    )
    kv_ref = _get_roundtrip_reference(
        cfg.shuffle_kv_a(kv_raw),
        kv.core_range_set,
        submesh,
        dtype=kv.dtype,
        mesh_mapper=replicate,
        mesh_composer=composer,
    )

    # q_b is TP-sharded: build per-TP references.
    per_device_q_b_w = cfg.q_b_proj_shape[1]
    q_b_tp_refs = []
    for tp_idx in range(tp):
        q_b_slice = q_b_raw[:, tp_idx * per_device_q_b_w : (tp_idx + 1) * per_device_q_b_w]
        ref = _get_roundtrip_reference(
            cfg.shuffle_q_b(q_b_slice),
            q_b.core_range_set,
            submesh,
            dtype=q_b.dtype,
            mesh_mapper=replicate,
            mesh_composer=composer,
        )
        q_b_tp_refs.append(ref[: q_b.tensor_shape[0]])

    q_a_h = q_a.tensor_shape[0]
    q_b_h = q_b.tensor_shape[0]
    kv_h = kv.tensor_shape[0]

    for device_idx in range(num_devices):
        tp_group = device_idx % mesh_cols

        q_a_slice = q_a_result[device_idx * q_a_h : (device_idx + 1) * q_a_h]
        q_a_ref_slice = q_a_ref[device_idx * q_a_h : (device_idx + 1) * q_a_h]
        assert torch.equal(q_a_slice, q_a_ref_slice), f"q_a_proj mismatch on device {device_idx}"

        q_b_slice = q_b_result[device_idx * q_b_h : (device_idx + 1) * q_b_h]
        assert torch.equal(
            q_b_slice, q_b_tp_refs[tp_group]
        ), f"q_b_proj mismatch on device {device_idx} (TP={tp_group})"

        kv_slice = kv_result[device_idx * kv_h : (device_idx + 1) * kv_h]
        kv_ref_slice = kv_ref[device_idx * kv_h : (device_idx + 1) * kv_h]
        assert torch.equal(kv_slice, kv_ref_slice), f"kv_a_proj mismatch on device {device_idx}"

        logger.info(f"Device {device_idx} (TP={tp_group}): q_a_proj, q_b_proj, kv_a_proj overlap passed")


def test_o_proj_gate_mm_rmsnorm_gamma_overlap(device):
    """Verify all constituents of the o_proj + gate_mm + rmsnorm gamma overlap.

    Creates the fused tensor once via BlitzDecodeWeights, then extracts
    each sub-tensor with CopyToOutput and checks it against an
    independently preprocessed + dtype round-tripped reference.
    """
    cfg = O_PROJ_GATE_MM_RMSNORM_GAMMA_OVERLAP_CFG
    tile_1x32 = ttnn.Tile([1, 32])

    torch.manual_seed(42)
    o_proj_raw = torch.randn(cfg.o_proj_shape, dtype=torch.bfloat16)
    gate_mm_raw = torch.randn(cfg.gate_mm_shape, dtype=torch.bfloat16)
    attn_norm_raw = torch.randn(cfg.attn_norm_shape, dtype=torch.bfloat16)
    q_norm_raw = torch.randn(cfg.q_norm_shape, dtype=torch.bfloat16)
    kv_norm_raw = torch.randn(cfg.kv_norm_shape, dtype=torch.bfloat16)
    ffn_norm_raw = torch.randn(cfg.ffn_norm_shape, dtype=torch.bfloat16)

    bdw = BlitzDecodeWeights(device)
    o_proj, gate_mm, attn_norm_g, q_norm_g, kv_norm_g, ffn_norm_g = bdw.get_tt_o_proj_and_gate_mm_weights(
        o_proj_raw,
        gate_mm_raw,
        attn_norm_raw,
        q_norm_raw,
        kv_norm_raw,
        ffn_norm_raw,
    )

    # -- Extract o_proj (BFP8) while fused buffer is alive -------------------
    o_out = _create_output_device_tensor(*o_proj.tensor_shape, o_proj.core_range_set, device, dtype=o_proj.dtype)
    o_out = CopyToOutput.op(o_proj.fused_tensor, o_out, byte_offset=o_proj.byte_offset)
    o_result = ttnn.to_torch(o_out)
    ttnn.deallocate(o_out)

    # -- Extract gate_mm (BFP16) while fused buffer is alive -----------------
    g_out = _create_output_device_tensor(*gate_mm.tensor_shape, gate_mm.core_range_set, device, dtype=gate_mm.dtype)
    g_out = CopyToOutput.op(gate_mm.fused_tensor, g_out, byte_offset=gate_mm.byte_offset)
    g_result = ttnn.to_torch(g_out)
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
            device,
            dtype=ov.dtype,
            sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            tile=tile_1x32,
        )
        out = CopyToOutput.op(ov.fused_tensor, out, byte_offset=ov.byte_offset)
        gamma_results[name] = ttnn.to_torch(out)
        ttnn.deallocate(out)

    ttnn.deallocate(o_proj.fused_tensor)

    # -- Build references (dtype round-trip) after fused buffer is freed -----
    o_ref = _get_roundtrip_reference(o_proj_raw, o_proj.core_range_set, device, dtype=o_proj.dtype)
    assert torch.equal(o_result, o_ref), "o_proj overlap has mismatch"
    logger.info("o_proj overlap comparison passed")

    g_ref = _get_roundtrip_reference(gate_mm_raw, gate_mm.core_range_set, device, dtype=gate_mm.dtype)
    assert torch.equal(g_result, g_ref), "gate_mm overlap has mismatch"
    logger.info("gate_mm overlap comparison passed")

    for name, raw, ov in [
        ("attn_norm", attn_norm_raw, attn_norm_g),
        ("q_norm", q_norm_raw, q_norm_g),
        ("kv_norm", kv_norm_raw, kv_norm_g),
        ("ffn_norm", ffn_norm_raw, ffn_norm_g),
    ]:
        ref = _get_roundtrip_reference(
            raw,
            ov.core_range_set,
            device,
            dtype=ov.dtype,
            sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            tile=tile_1x32,
        )
        assert torch.equal(gamma_results[name], ref), f"{name} overlap has mismatch"
        logger.info(f"{name} overlap comparison passed")


def test_kv_b12_proj_overlap(device):
    """Verify both constituents of the kv_b1 + kv_b2 overlap.

    Creates the fused tensor once via BlitzDecodeWeights, then extracts
    each sub-tensor with CopyToOutput and checks it against an
    independently preprocessed + bfp8 round-tripped reference.

    kv_b1_proj (8192, 512) is HEIGHT_SHARDED on the 8x8 Qnope grid.
    kv_b2_proj (512, 8192) is pre-transposed, reshaped internally to
    (8192, 512) and HEIGHT_SHARDED on the remaining 64 cores.
    """
    cfg = KVB12_PROJ_OVERLAP_CFG

    torch.manual_seed(42)
    kv_b1_raw = torch.randn(*cfg.kv_b1_proj_shape, dtype=torch.bfloat16)
    kv_b2_raw = torch.randn(*cfg.kv_b2_proj_shape, dtype=torch.bfloat16)

    bdw = BlitzDecodeWeights(device)
    kv_b1, kv_b2 = bdw.get_tt_kv_b12_proj_weights(kv_b1_raw, kv_b2_raw)

    kv_b2_physical = cfg.shuffle_kv_b2(kv_b2_raw)

    # -- Extract kv_b1 (HEIGHT_SHARDED, BFP8) while fused buffer is alive ----
    kv_b1_out = _create_output_device_tensor(
        *kv_b1.tensor_shape,
        kv_b1.core_range_set,
        device,
        dtype=kv_b1.dtype,
        sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    )
    kv_b1_out = CopyToOutput.op(kv_b1.fused_tensor, kv_b1_out, byte_offset=kv_b1.byte_offset)
    kv_b1_result = ttnn.to_torch(kv_b1_out)
    ttnn.deallocate(kv_b1_out)

    # -- Extract kv_b2 (HEIGHT_SHARDED, BFP8) while fused buffer is alive -----
    kv_b2_out = _create_output_device_tensor(
        *cfg.kv_b1_proj_shape,
        kv_b2.core_range_set,
        device,
        dtype=kv_b2.dtype,
        sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    )
    kv_b2_out = CopyToOutput.op(kv_b2.fused_tensor, kv_b2_out, byte_offset=kv_b2.byte_offset)
    kv_b2_result = ttnn.to_torch(kv_b2_out)
    ttnn.deallocate(kv_b2_out)

    ttnn.deallocate(kv_b1.fused_tensor)

    # -- Build references (bfp8 round-trip) after fused buffer is freed ------
    kv_b1_ref = _get_roundtrip_reference(
        kv_b1_raw,
        kv_b1.core_range_set,
        device,
        dtype=kv_b1.dtype,
        sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    )
    assert torch.equal(kv_b1_result, kv_b1_ref), "kv_b1_proj overlap has mismatch"
    logger.info("kv_b1_proj overlap comparison passed")

    kv_b2_ref = _get_roundtrip_reference(
        kv_b2_physical,
        kv_b2.core_range_set,
        device,
        dtype=kv_b2.dtype,
        sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    )
    assert torch.equal(kv_b2_result, kv_b2_ref), "kv_b2_proj overlap has mismatch"
    logger.info("kv_b2_proj overlap comparison passed")


def test_gate_up_proj_overlap(device):
    """Verify both constituents of the gate + up projection overlap.

    Creates the fused tensor once via BlitzDecodeWeights, then extracts
    each sub-tensor with CopyToOutput and checks it against an
    independently preprocessed + bfp4 round-tripped reference.

    Both gate_proj and up_proj are block-sharded (stacked HEIGHT_SHARDED)
    on 64 non-rectangular compute cores each.
    """
    cfg = GATE_UP_PROJ_OVERLAP_CFG

    torch.manual_seed(42)
    gate_raw = torch.randn(cfg.gate_proj_shape, dtype=torch.bfloat16)
    up_raw = torch.randn(cfg.up_proj_shape, dtype=torch.bfloat16)

    bdw = BlitzDecodeWeights(device)
    gate, up = bdw.get_tt_gate_up_proj_weights(gate_raw, up_raw)

    # -- Extract gate (block-sharded BFP4) while fused buffer is alive -------
    gate_out = _create_output_device_tensor(
        *gate.tensor_shape,
        gate.core_range_set,
        device,
        dtype=gate.dtype,
        sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    )
    gate_out = CopyToOutput.op(gate.fused_tensor, gate_out, byte_offset=gate.byte_offset)
    gate_result = ttnn.to_torch(gate_out)
    ttnn.deallocate(gate_out)

    # -- Extract up (block-sharded BFP4) while fused buffer is alive ---------
    up_out = _create_output_device_tensor(
        *up.tensor_shape,
        up.core_range_set,
        device,
        dtype=up.dtype,
        sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    )
    up_out = CopyToOutput.op(up.fused_tensor, up_out, byte_offset=up.byte_offset)
    up_result = ttnn.to_torch(up_out)
    ttnn.deallocate(up_out)

    ttnn.deallocate(gate.fused_tensor)

    # -- Build references (bfp4 round-trip) after fused buffer is freed ------
    gate_ref = _get_roundtrip_reference(
        cfg.reshuffle_block_to_height_sharded(gate_raw),
        gate.core_range_set,
        device,
        dtype=gate.dtype,
        sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    )
    assert torch.equal(gate_result, gate_ref), "gate_proj overlap has mismatch"
    logger.info("gate_proj overlap comparison passed")

    up_ref = _get_roundtrip_reference(
        cfg.reshuffle_block_to_height_sharded(up_raw),
        up.core_range_set,
        device,
        dtype=up.dtype,
        sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    )
    assert torch.equal(up_result, up_ref), "up_proj overlap has mismatch"
    logger.info("up_proj overlap comparison passed")
