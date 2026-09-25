# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Minimal Quasar repro: fp32 tilize needs unpack-to-DEST, which Quasar's tilize LLK API does not do.

One 32x32 float32 tile, one core, one op (ttnn.tilize). No typecast, no model code.

What happens on main:
  tilize_metal2.cpp picks Fp32Mode::Lossless for a Float32 input, so the factory enables UnpackToDest
  and the JIT sets the input operand's unpack_dst_format to Float32. Quasar's llk_unpack_tilize_init /
  llk_unpack_tilize_block ignore UnpackToDestEn and always target UNP_A (SrcA). Float32 -> Float32 is
  only a legal unpacker conversion on the DEST path, so:
    - emu-quasar-2x3 (RTL): Neo0TRISC0 hardware fault, cause UNPACKER_0 (ILLEGAL_FORMAT_CONVERSION)
    - craq-sim (ttsim):     hang (math waits on an UNPACK_MATH post that never comes)

Run from the tt-metal root (emulator; set NNG_SOCKET_ADDR to your reservation's host:port):
  TT_METAL_SIMULATOR=<tt-umd-simulators>/build/emu-quasar-2x3/ NNG_SOCKET_ADDR=tcp://<host>:<port> \
  NNG_SOCKET_LOCAL_PORT=5555 TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 \
  TT_METAL_WATCHER=1 TT_METAL_LLK_ASSERTS=1 TT_METAL_WATCHER_DISABLE_NOC_SANITIZE=1 \
  pytest -sv --timeout=1800 <this file>

bfloat16 is the control: same op and shape, no unpack-to-DEST, passes.
"""

import pytest
import torch
import ttnn


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16_control", "fp32"])
def test_quasar_fp32_tilize_unpack_to_dest_repro(mesh_device, dtype):
    torch.manual_seed(0)
    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    x = torch.randn(1, 1, 32, 32, dtype=torch_dtype)

    x_rm = ttnn.from_torch(x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)
    y = ttnn.tilize(x_rm, use_multicore=False)
    out = ttnn.to_torch(y)

    # Tilize only reorders data, so the round trip must be bit-exact.
    assert torch.equal(out.reshape(x.shape), x)
