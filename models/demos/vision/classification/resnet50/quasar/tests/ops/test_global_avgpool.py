# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone de-risk for the ResNet-50 GLOBAL AVERAGE POOL on Quasar.

Right before the FC classifier, resnet50 does a global avg-pool over the full layer4 feature map
(spatial 7x7, 2048 channels) -> [N, 2048, 1, 1]. The model issues it as (see
ttnn_functional_resnet50.py ~line 1131):

  x = ttnn.experimental.quasar.avg_pool2d(
        input_tensor=x,                       # WIDTH_SHARDED, TILE layout, [1,1,N*H*W, C]
        kernel_size=[H, W], stride=[1,1], padding=[0,0,0,0],
        output_layout=ttnn.TILE_LAYOUT,       # -> the OUTPUT_TILED compute path
        dtype=ttnn.bfloat16,
        compute_kernel_config=LoFi)

This exercises a DIFFERENT Quasar pool path than the stem max-pool: output_layout=TILE_LAYOUT takes
the OUTPUT_TILED branch of compute_pool_2d.cpp, which packs straight into the real out_cb via
tilize_block (fast_tilize is unported on Quasar) and NEVER touches the scratch-roundtrip scaffold
that the max-pool uses -- so it is an independent de-risk. Known Quasar concerns on this path:
  - fast_tilize unported -> QSR uses tilize_init/tilize_block with an explicit llk_pack_init(out_cb)
    retarget (else PCC 0.0 / all-zero output);
  - AVG reduce uses fp32 accumulation (is_avg_pool) which caps tiles/reduction to 4;
  - width-sharded channel tiling across cores.

Gates:
  - PCC vs torch F.avg_pool2d (global) golden;
  - no value inflation / deflation blow-up: the per-channel avg must lie within [min, max] of that
    channel's input (an average can never exceed the input range) -- catches a bad reduce/scale or a
    stale-L1 leak (the "got.max=2.0" class of bug noted in the pool reader).

The width-sharded memory config mirrors the model (fit_width_sharded_cores + create_sharded_memory_config_),
so it adapts to the 1-2 core emulator as well as WH.

Run (emulator / WH, slow dispatch + forced JIT):
  TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 \
  pytest -s models/demos/vision/classification/resnet50/quasar/tests/ops/test_global_avgpool.py
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import nearest_32
from models.demos.vision.classification.resnet50.quasar.tt.ttnn_functional_resnet50 import fit_width_sharded_cores
from tests.ttnn.utils_for_testing import assert_with_pcc

PCC = 0.99

# ResNet-50 final feature map is 7x7. Channels swept from small -> the real 2048.
_H = 7
_W = 7
# (channels, id)
_CASES = [
    (64, "C64"),
    (256, "C256"),
    (512, "C512"),
    (2048, "C2048"),  # the true resnet50 layer4 channel count
]


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("channels,cid", _CASES, ids=[c[1] for c in _CASES])
def test_quasar_global_avgpool(mesh_device, channels, cid):
    device = mesh_device
    torch.manual_seed(0)

    batch_size = 1
    input_h, input_w = _H, _W

    # torch golden in NCHW: global avg over the full HxW window -> [N, C, 1, 1].
    x_nchw = torch.rand((batch_size, channels, input_h, input_w), dtype=torch.float32)
    golden = torch.nn.functional.avg_pool2d(x_nchw, kernel_size=(input_h, input_w))  # [N, C, 1, 1]

    # ttnn avg_pool2d expects [1, 1, N*H*W, C] (flattened NHW, C).
    x_nhwc = x_nchw.permute(0, 2, 3, 1).reshape(1, 1, batch_size * input_h * input_w, channels)

    # WIDTH_SHARDED input, tile layout -- exactly as the model builds it for the global avg-pool.
    num_cores, core_grid = fit_width_sharded_cores(channels, 8 * 8, device)
    width_mem_config = ttnn.create_sharded_memory_config_(
        [nearest_32(batch_size * input_h * input_w), channels // num_cores],
        core_grid,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.ShardOrientation.ROW_MAJOR,
        tile_layout=True,
        use_height_and_width_as_shard_shape=True,
    )

    x = ttnn.from_torch(x_nhwc.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    x = x.to(device, width_mem_config)

    out = ttnn.experimental.quasar.avg_pool2d(
        input_tensor=x,
        batch_size=batch_size,
        input_h=input_h,
        input_w=input_w,
        channels=channels,
        kernel_size=[input_h, input_w],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        output_layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        compute_kernel_config=ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.LoFi
        ),
    )
    ttnn.synchronize_device(device)

    tt_out = ttnn.to_torch(ttnn.from_device(out)).float()
    # [.., N*1*1, C] -> [N, C, 1, 1] to line up with the torch golden.
    tt_out = tt_out.reshape(batch_size, 1, 1, channels).permute(0, 3, 1, 2)

    in_lo, in_hi = float(x_nchw.min()), float(x_nchw.max())
    dev_lo, dev_hi = float(tt_out.min()), float(tt_out.max())
    print(
        f"[global_avgpool {cid}] num_cores={num_cores} out={tuple(tt_out.shape)} golden={tuple(golden.shape)} "
        f"in=[{in_lo:.4f},{in_hi:.4f}] dev=[{dev_lo:.4f},{dev_hi:.4f}] golden=[{float(golden.min()):.4f},{float(golden.max()):.4f}]"
    )

    # -------------------- [DIAG] error-structure localizer (remove after avgpool is fixed) --------------------
    # Width-sharding splits channels CONTIGUOUSLY across cores: core k owns channels [k*Cpc, (k+1)*Cpc).
    # in_ntiles_c per core = Cpc/32; in_nblocks_c = ceil(in_ntiles_c / 4) (MAX_TILES_PER_REDUCTION for avg).
    g = golden.reshape(-1).float()  # [C], channel-major
    d = tt_out.reshape(-1).float()  # [C]
    err = (d - g).abs()
    cpc = channels // num_cores
    tiles_per_core = max(1, cpc // 32)
    n_blocks_c = (tiles_per_core + 3) // 4
    n_bad = int((err > 0.02).sum())
    print(
        f"  [DIAG {cid}] Cpc={cpc} tiles/core={tiles_per_core} nblocks_c={n_blocks_c} | "
        f"n_bad(|e|>0.02)={n_bad}/{channels} ({100.0*n_bad/channels:.1f}%) "
        f"max|e|={float(err.max()):.4f} mean(d-g)={float((d-g).mean()):+.4f} std(d-g)={float((d-g).std()):.4f}"
    )
    # per-core slice: is one core (channel half) wrong and the other right? (points at per-core reduce vs shared)
    for k in range(num_cores):
        sl = slice(k * cpc, (k + 1) * cpc)
        gk, dk = g[sl], d[sl]
        pcc_k = float(torch.corrcoef(torch.stack([gk, dk]))[0, 1]) if gk.numel() > 1 else float("nan")
        print(
            f"  [DIAG {cid}] core{k} ch[{k*cpc}:{(k+1)*cpc}] pcc={pcc_k:.4f} "
            f"mean(d-g)={float((dk - gk).mean()):+.4f} n_bad={int((err[sl] > 0.02).sum())}/{cpc}"
        )
    # per-32-channel-tile mean error within core0 (does a specific c-tile / c-block go wrong?)
    for t in range(tiles_per_core):
        sl = slice(t * 32, (t + 1) * 32)
        print(
            f"  [DIAG {cid}] core0 tile{t} (blk{t // 4}) ch[{t*32}:{(t+1)*32}] "
            f"mean(d-g)={float((d[sl] - g[sl]).mean()):+.4f} max|e|={float(err[sl].max()):.4f}"
        )
    # first 6 channels raw, so we can eyeball golden vs dev
    print(f"  [DIAG {cid}] ch0..5 golden={[round(float(x),4) for x in g[:6]]} dev={[round(float(x),4) for x in d[:6]]}")
    # ---------------------------------------------------------------------------------------------------------

    # an average can never leave the input range (catches bad scale / stale-L1 leak).
    assert dev_lo >= in_lo - 1e-2 and dev_hi <= in_hi + 1e-2, (
        f"avg out range [{dev_lo:.4f},{dev_hi:.4f}] escaped input range [{in_lo:.4f},{in_hi:.4f}] "
        f"(bad reduce scale / stale-L1 leak)"
    )
    assert_with_pcc(golden, tt_out, pcc=PCC)


# ------------------------------------------------------------------------------------------------------------
# [DIAG] Constant-input bias probe (remove after avgpool is fixed). Decomposes the observed uniform +0.075 bias:
# with a CONSTANT input c, golden == c for every channel, so
#   dev(c) == c            -> bias is data-dependent (correlated with input variance); NOT padding/scalar.
#   dev(c) == c + b        -> ADDITIVE bias b, constant across c -> extra summed rows hold a value != c
#                             (stale / nonzero pad); b = n_pad * v / 49.
#   dev(c) == c * s        -> SCALAR wrong (wrong divisor: s = 49/D).
# Two constants pin it: b = dev(0), s = dev(1) - dev(0). This is host-only (no rebuild) and does not assert,
# so it always runs to completion and prints. C64 (1 tile/core) keeps it minimal.
# ------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("fill", [0.0, 0.25, 0.5, 1.0], ids=["c0.0", "c0.25", "c0.5", "c1.0"])
def test_quasar_avgpool_bias_probe(mesh_device, fill):
    device = mesh_device
    channels = 64
    batch_size = 1
    input_h, input_w = _H, _W

    x_nchw = torch.full((batch_size, channels, input_h, input_w), float(fill), dtype=torch.float32)
    golden = torch.nn.functional.avg_pool2d(x_nchw, kernel_size=(input_h, input_w))  # == fill everywhere
    x_nhwc = x_nchw.permute(0, 2, 3, 1).reshape(1, 1, batch_size * input_h * input_w, channels)

    num_cores, core_grid = fit_width_sharded_cores(channels, 8 * 8, device)
    width_mem_config = ttnn.create_sharded_memory_config_(
        [nearest_32(batch_size * input_h * input_w), channels // num_cores],
        core_grid,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.ShardOrientation.ROW_MAJOR,
        tile_layout=True,
        use_height_and_width_as_shard_shape=True,
    )
    x = ttnn.from_torch(x_nhwc.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    x = x.to(device, width_mem_config)

    out = ttnn.experimental.quasar.avg_pool2d(
        input_tensor=x,
        batch_size=batch_size,
        input_h=input_h,
        input_w=input_w,
        channels=channels,
        kernel_size=[input_h, input_w],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        output_layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        compute_kernel_config=ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.LoFi
        ),
    )
    ttnn.synchronize_device(device)
    d = ttnn.to_torch(ttnn.from_device(out)).float().reshape(-1)  # [C], all == golden == fill

    print(
        f"  [BIASPROBE fill={fill}] golden={fill:.4f} dev mean={float(d.mean()):.4f} "
        f"min={float(d.min()):.4f} max={float(d.max()):.4f} std={float(d.std()):.4f} "
        f"| dev-golden mean={float(d.mean()) - fill:+.4f}"
    )
    # No PCC assert: this probe is purely to read out the bias structure across constants.
