# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Standalone device-RoPE verification for the e2e DECODE path: ``ttnn.experimental.rotary_embedding_llama``
(separate Q and K, ``is_decode_mode=True``, HEIGHT_SHARDED).

Purpose: confirm the model's decode RoPE runs ON DEVICE and is NUMERICALLY correct on Quasar, so we know
whether flipping ``LLAMA_QSR_DEVICE_ROPE=1`` in the e2e is viable. By default the e2e applies RoPE on the
HOST (``_install_quasar_host_rope``); this test is the gate that would let us remove that host step.

WHY THE NON-FUSED OP (not fused_qk): the e2e model is built with ``use_qk_fused=False``
(models/llama32_1b/model.py:838), so ``_bind_forward_methods`` binds ``_rotary_embed_decode`` to
``_rotary_embed_decode_nonfused`` (attention_1d.py:826), which calls ``ttnn.experimental.rotary_embedding_llama``
SEPARATELY for Q and K (attention_1d.py:997,1000, ``is_decode_mode=True``). The fused ``rotary_embedding_llama_fused_qk``
op is NOT on the e2e path, and it cannot even finalize on Quasar (it binds 10 intra-tensix DFBs; the HW
remapper allows only 8 = 16 tile counters / 2 ClientL+shadow per DFB). The non-fused sharded op binds exactly
8 DFBs (input, cos, sin, trans_mat, rotated_interm, cos_interm, sin_interm, out) and its sharded compute kernel
is Quasar-ported by commit 9d3abcb5c85 (#58196: dummy_pack for resident pushes + pack_init on each pack-output
switch). So this test should PASS on Quasar.

RoPE math (GPT-J adjacent-pair rotation): rot(x) = x @ trans, trans +1 at (2i,2i+1), -1 at (2i+1,2i)
=> [-x1, x0, -x3, x2, ...]; out = x*cos + rot*sin. The device trans_mat is the base 32x32 rotation applied
tile-wise (adjacent pairs never straddle the 32-element tile boundary, so the tiled 32x32 == the 64x64 matrix).

Run (Quasar sim, 2-node emulator, SLOW dispatch; large batch needs `batch` cores per call):
    TTSIM_QSR_TC_LEGACY_TRUNCATION_ALIAS=0 TT_METAL_SIMULATOR=~/sim/libttsim.so TT_METAL_SLOW_DISPATCH_MODE=1 \
        TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="3,2" MESH_DEVICE=N150 TT_METAL_LLK_ASSERTS=1 \
        pytest models/experimental/llama32_1b_quasar/tests/debug_ops/test_quasar_rope_device.py
"""

import pytest
import torch

import ttnn
from models.experimental.llama32_1b_quasar.tensor_utils import get_rot_transformation_mat
from models.experimental.llama32_1b_quasar.tests.ops import op_utils as U


def _adjacent_rotate_mat(hd):
    """get_rot_transformation_mat equivalent at full head_dim: +1 at (2i,2i+1), -1 at (2i+1,2i).
    x @ m = [-x1, x0, -x3, x2, ...] (the GPT-J adjacent-pair rotation)."""
    m = torch.zeros(hd, hd)
    even = torch.arange(0, hd, 2)
    odd = torch.arange(1, hd, 2)
    m[even, odd] = 1.0
    m[odd, even] = -1.0
    return m


def _rope_ref(x, cos1, sin1):
    """Torch reference RoPE. x [1,batch,heads,hd]; cos1/sin1 [1,batch,1,hd] broadcast over heads."""
    trans = _adjacent_rotate_mat(x.shape[-1])
    rot = x @ trans
    return x * cos1 + rot * sin1


def _rope_decode_device(mesh_device, x_t, cos1, sin1, batch, start_core=None):
    """Run the model's non-fused decode RoPE op on a HEIGHT_SHARDED input and return the device output.

    x_t [1,batch,heads,hd]; cos1/sin1 [1,batch,1,hd]. Mirrors _rotary_embed_decode_nonfused: one
    ttnn.experimental.rotary_embedding_llama(..., is_decode_mode=True) call on sharded inputs."""
    hd = x_t.shape[-1]
    # cos/sin broadcast over all TILE rows so the single decode position the kernel reads matches the ref.
    cos_dev = cos1.expand(1, batch, U.TILE, hd).contiguous()  # [1,batch,TILE,hd]
    sin_dev = sin1.expand(1, batch, U.TILE, hd).contiguous()

    x_memcfg = U.height_sharded_batch_memcfg(mesh_device, batch, (U.TILE, hd), start_core=start_core)
    cos_sin_memcfg = U.height_sharded_batch_memcfg(mesh_device, batch, (U.TILE, hd), start_core=start_core)
    trans_memcfg = U.height_sharded_batch_memcfg(mesh_device, batch, (U.TILE, U.TILE), start_core=start_core)

    x = U.to_tt(x_t, mesh_device, memory_config=x_memcfg)
    cos = U.to_tt(cos_dev, mesh_device, memory_config=cos_sin_memcfg)
    sin = U.to_tt(sin_dev, mesh_device, memory_config=cos_sin_memcfg)
    # Decode trans-mat is the base 32x32 rotation tiled per core (one TILE×TILE tile per user core).
    trans_mat = U.to_tt(get_rot_transformation_mat().repeat(1, 1, batch, 1), mesh_device, memory_config=trans_memcfg)

    return ttnn.experimental.rotary_embedding_llama(x, cos, sin, trans_mat, is_decode_mode=True)


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("batch", [pytest.param(b, id=f"decode-batch{b}") for b in U.DECODE_BATCHES])
def test_rope_nonfused_decode_device(mesh_device, batch):
    """Device decode RoPE (the e2e non-fused path) vs a torch reference (PCC). Q and K are separate
    ttnn.experimental.rotary_embedding_llama calls, each HEIGHT_SHARDED over `batch` cores. Should PASS on
    Quasar (sharded kernel ported by #58196; 8 DFBs fit the remapper limit)."""
    torch.manual_seed(0)
    hd = U.HEAD_DIM  # 64

    q_t = U.torch_rand((1, batch, U.N_HEADS, hd))
    k_t = U.torch_rand((1, batch, U.N_KV_HEADS, hd))
    # One real cos/sin per user (position row 0), broadcast over all TILE rows.
    cos1 = U.torch_rand((1, batch, 1, hd))
    sin1 = U.torch_rand((1, batch, 1, hd))

    # Separate Q and K calls (non-fused). Each is height-sharded over `batch` cores; running sequentially,
    # both can start at core 0 (no non-overlap requirement, unlike fused_qk).
    q_out = _rope_decode_device(mesh_device, q_t, cos1, sin1, batch)
    k_out = _rope_decode_device(mesh_device, k_t, cos1, sin1, batch)
    ttnn.synchronize_device(mesh_device)

    q_ref = _rope_ref(q_t.float(), cos1.float(), sin1.float())  # [1,batch,32,64]
    k_ref = _rope_ref(k_t.float(), cos1.float(), sin1.float())  # [1,batch,8,64]
    # k_out head axis is tile-padded to 32; assert_pcc trims got to the reference's element count (real heads
    # 0..7 are the leading rows of the row-major flatten).
    U.assert_pcc(q_ref, q_out, pcc=0.99, mesh_device=mesh_device)
    U.assert_pcc(k_ref, k_out, pcc=0.99, mesh_device=mesh_device)
