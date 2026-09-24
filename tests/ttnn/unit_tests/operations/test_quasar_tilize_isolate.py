# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""craq-sim (Quasar functional simulator) fp32-tilize hang — isolation / discriminator suite.

See CRAQSIM_FP32_TILIZE_HANG.md at the repo root for the full root cause and proposed sim fixes.

Summary: on craq-sim, producing an fp32 tile hangs; bf16 works. The tt-metal op code is the standard
lossless-fp32 tilize path — the gap is in the sim's Quasar Tensix backend, which never retires the fp32
ELWADD lossless-datacopy, so MATH_PACK is never posted, DEST never goes valid, PACR0 stalls forever, and the
writer waits on the output DFB (WFW) until readback/device-sync hangs.

Run on craq-sim (single sim device; bound hangs with `timeout`, bf16 passes in a few seconds):

    source python_env/bin/activate
    export TT_METAL_HOME=$PWD TT_METAL_RUNTIME_ROOT=$PWD PYTHONPATH=$PWD:$PYTHONPATH
    export TT_METAL_SIMULATOR=/abs/path/to/libttsim.so TT_SIMULATOR_LOCALHOST=1 TT_METAL_SLOW_DISPATCH_MODE=1
    export ARCH_NAME=quasar CHIP_ARCH=quasar MESH_DEVICE=N150 TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="3,2"
    timeout -k1 90 python -m pytest -q -p no:cacheprovider <this_file>::<test> -k 32x32

Discriminator matrix at 32x32 (1 tile, 1 core) — expected on current craq-sim:
    test_mainline_tilize_bf16      bf16 in  -> bf16 tile  : PASS  (fast_tilize path, no ELWADD datacopy)
    test_mainline_tilize_fp32      fp32 in  -> fp32 tile  : HANG  (production case; lossless ELWADD datacopy)
    test_experimental_tilize_fp32  fp32 in  -> fp32 tile  : HANG  (same LLK path via the quasar-native op)
    test_disc_fp32in_bf16out       fp32 in  -> bf16 tile  : HANG  (isolates the fp32 UNPACK / ELWADD side)
    test_disc_bf16in_fp32out       bf16 in  -> fp32 tile  : ERROR (immediate, no hang: qsr_convert_pack_value
                                                                    in_format=5 out_format=0 unimplemented)
The two hangs share `fp32 input -> ELWADD lossless datacopy`; the ERROR is the separate fp32-PACK-convert gap.
"""
import pytest
import torch
from loguru import logger
import ttnn


def _readback(tt, mesh_device):
    try:
        num = mesh_device.get_num_devices()
    except Exception:
        num = 1
    if num > 1:
        return ttnn.to_torch(tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    return ttnn.to_torch(tt)


_SHAPES = [(32, 32), (32, 2048)]


@pytest.mark.parametrize("shape", _SHAPES, ids=[f"{h}x{w}" for (h, w) in _SHAPES])
def test_mainline_tilize_bf16(mesh_device, shape):
    """bf16 row-major upload (no device tilize), then MAINLINE ttnn.tilize (no typecast)."""
    h, w = shape
    torch.manual_seed(0)
    t = torch.randn(1, 1, h, w, dtype=torch.bfloat16)
    logger.info(f"[iso-mainline-tilize] upload row-major (1,1,{h},{w})")
    rm = ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
    )
    logger.info(f"[iso-mainline-tilize] ttnn.tilize begin (1,1,{h},{w})")
    tt = ttnn.tilize(rm, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    logger.info(f"[iso-mainline-tilize] tilize done (1,1,{h},{w}); reading back")
    out = _readback(tt, mesh_device).float()
    logger.info(f"[iso-mainline-tilize] readback complete (1,1,{h},{w})")
    assert tuple(out.shape)[-2:] == (h, w)
    assert torch.isfinite(out).all()
    assert torch.allclose(out.reshape(t.float().shape), t.float(), atol=0.05, rtol=0.05)


@pytest.mark.parametrize("shape", _SHAPES, ids=[f"{h}x{w}" for (h, w) in _SHAPES])
def test_mainline_tilize_fp32(mesh_device, shape):
    """fp32 row-major upload, then MAINLINE ttnn.tilize (fp32->fp32 tile, NO typecast).

    Isolates the fp32 tilize step of from_torch(TILE) from the fp32->bf16 typecast that follows it.
    """
    h, w = shape
    torch.manual_seed(0)
    t = torch.randn(1, 1, h, w, dtype=torch.float32)
    logger.info(f"[iso-fp32-tilize] upload row-major fp32 (1,1,{h},{w})")
    rm = ttnn.from_torch(
        t,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
    )
    logger.info(f"[iso-fp32-tilize] ttnn.tilize begin (1,1,{h},{w})")
    tt = ttnn.tilize(rm, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    logger.info(f"[iso-fp32-tilize] tilize done (1,1,{h},{w}); reading back")
    out = _readback(tt, mesh_device).float()
    logger.info(f"[iso-fp32-tilize] readback complete (1,1,{h},{w})")
    assert tuple(out.shape)[-2:] == (h, w)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("shape", _SHAPES, ids=[f"{h}x{w}" for (h, w) in _SHAPES])
def test_experimental_tilize_fp32(mesh_device, shape):
    """experimental.quasar.tilize with FP32 input (no typecast). If this hangs like the mainline fp32
    tilize, the differentiator is fp32-vs-bf16, not mainline-vs-experimental."""
    h, w = shape
    torch.manual_seed(0)
    t = torch.randn(1, 1, h, w, dtype=torch.float32)
    logger.info(f"[iso-exp-fp32] upload row-major fp32 (1,1,{h},{w})")
    rm = ttnn.from_torch(
        t,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
    )
    logger.info(f"[iso-exp-fp32] quasar.tilize(fp32) begin (1,1,{h},{w})")
    tt = ttnn.experimental.quasar.tilize(rm, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.float32)
    logger.info(f"[iso-exp-fp32] tilize done; reading back (1,1,{h},{w})")
    out = _readback(tt, mesh_device).float()
    logger.info(f"[iso-exp-fp32] readback complete (1,1,{h},{w})")
    assert tuple(out.shape)[-2:] == (h, w)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("shape", _SHAPES, ids=[f"{h}x{w}" for (h, w) in _SHAPES])
def test_disc_bf16in_fp32out(mesh_device, shape):
    """DISCRIMINATOR: bf16 rowmajor input -> quasar.tilize(dtype=fp32). Standard tilize path
    (fp32 output disables fast tilize), bf16 UNPACK (no UnpackToDest / no 32-bit-dest unpack),
    but 32-bit (fp32) PACK. If this hangs, the culprit is the fp32 PACK side."""
    h, w = shape
    torch.manual_seed(0)
    t = torch.randn(1, 1, h, w, dtype=torch.bfloat16)
    rm = ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
    )
    logger.info(f"[disc-bf16in-fp32out] quasar.tilize begin (1,1,{h},{w})")
    tt = ttnn.experimental.quasar.tilize(rm, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.float32)
    out = _readback(tt, mesh_device).float()
    logger.info(f"[disc-bf16in-fp32out] readback complete (1,1,{h},{w})")
    assert tuple(out.shape)[-2:] == (h, w)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("shape", _SHAPES, ids=[f"{h}x{w}" for (h, w) in _SHAPES])
def test_disc_fp32in_bf16out(mesh_device, shape):
    """DISCRIMINATOR: fp32 rowmajor input -> quasar.tilize(dtype=bf16). Standard tilize path
    (lossless override for fp32 input), 32-bit-dest UNPACK (UnpackToDest), but bf16 (2-byte) PACK.
    If this hangs, the culprit is the fp32 UNPACK / 32-bit-dest side."""
    h, w = shape
    torch.manual_seed(0)
    t = torch.randn(1, 1, h, w, dtype=torch.float32)
    rm = ttnn.from_torch(
        t,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
    )
    logger.info(f"[disc-fp32in-bf16out] quasar.tilize begin (1,1,{h},{w})")
    tt = ttnn.experimental.quasar.tilize(rm, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
    out = _readback(tt, mesh_device).float()
    logger.info(f"[disc-fp32in-bf16out] readback complete (1,1,{h},{w})")
    assert tuple(out.shape)[-2:] == (h, w)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("shape", _SHAPES, ids=[f"{h}x{w}" for (h, w) in _SHAPES])
def test_mainline_typecast_after_tilize(mesh_device, shape):
    """fp32 tilize (via experimental quasar tilize to avoid confounding), then MAINLINE ttnn.typecast
    fp32->bf16. Isolates the typecast step."""
    h, w = shape
    torch.manual_seed(0)
    t = torch.randn(1, 1, h, w, dtype=torch.float32)
    rm = ttnn.from_torch(
        t,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
    )
    tiled = ttnn.experimental.quasar.tilize(rm, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.float32)
    logger.info(f"[iso-typecast] ttnn.typecast fp32->bf16 begin (1,1,{h},{w})")
    tt = ttnn.typecast(tiled, ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    logger.info(f"[iso-typecast] typecast done (1,1,{h},{w}); reading back")
    out = _readback(tt, mesh_device).float()
    logger.info(f"[iso-typecast] readback complete (1,1,{h},{w})")
    assert tuple(out.shape)[-2:] == (h, w)
    assert torch.isfinite(out).all()
