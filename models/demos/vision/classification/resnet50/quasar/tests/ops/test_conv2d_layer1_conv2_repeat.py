# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Repeat test of the resnet50 layer1 3x3 conv2, for the 2-core emulators (Horizon, Trinity).

test_resnet50_e2e.py fails on Horizon and Trinity (2 worker cores) and passes on the Quasar emulator (1 worker
core). Its per-op logs show two symptoms on 2 cores:
  * layer1_module1.conv2 (the first 3x3 conv: a program-cache miss) gives the same output in every run, but not
    the Quasar output. Output pixel (0, 0), which uses the top and left padding, is different.
  * layer1_module2/3.conv2 (the same shape: program-cache hits) give a different output from run to run.

This test runs that conv (64 -> 64, 56x56, 3x3, stride 1, pad 1, HEIGHT_SHARDED, the model's conv2 Conv2dConfig)
N times on one device with the same input and weights. Iteration 0 is a program-cache miss; the others are hits.
It logs:
  * [REPEAT] for each iteration: the sha256 of the output, how many output pixels differ from iteration 0, and
    the PCC and the max error against torch;
  * [ERRMAP] for iteration 0: the error against torch on the image border, on the rows next to each shard edge
    between cores and in the interior, the 16 worst pixels as (h, w), and a 56x56 map ('x' = a pixel error
    above 4x the median pixel error);
  * [DIFFMAP] for the first iteration that differs from iteration 0: a 56x56 map of the changed pixels.
The input is seeded, so the sha256 values can be compared between platforms and between variants: a correct
conv gives the same output on 1 core and on 2 cores.

Variants:
  model     TILE input HEIGHT_SHARDED over all worker cores, reshard_if_not_optimal=False (the model's path)
  modelcfg  ROW_MAJOR input, reshard_if_not_optimal=True (test_conv2d_layer_conv2_modelcfg.py's path)
  one_core  TILE input HEIGHT_SHARDED on 1 core, reshard_if_not_optimal=False
  model_nocache  "model" with the program cache disabled, so every iteration builds a new program: if the
            output still changes, the cause is not the reuse of the cached program

RESNET_REPEAT_N sets the number of iterations (default 5). RESNET_PCC_DUMP=<dir> also saves the input, the
golden and each output (bfloat16, [H, W, C]) to <dir>/repeat_<variant>_*.pt.

Run (emulator):
  TT_METAL_SLOW_DISPATCH_MODE=1 \
    pytest -q models/demos/vision/classification/resnet50/quasar/tests/ops/test_conv2d_layer1_conv2_repeat.py
"""

import hashlib
import os

import pytest
import torch
from loguru import logger

import ttnn

IN_CH = OUT_CH = 64
SPATIAL = 56
PCC = 0.99


def _sha16(t):
    return hashlib.sha256(t.to(torch.bfloat16).view(torch.int16).numpy().tobytes()).hexdigest()[:16]


def _pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    return 1.0 if denom == 0 else float((a @ b) / denom)


def _pixel_map(mask):
    """[H, W] bool -> one line per image row, 'x' where the mask is set."""
    return "\n".join("".join("x" if v else "." for v in row) for row in mask.tolist())


def _dump(name, t):
    dump = os.environ.get("RESNET_PCC_DUMP")
    if dump:
        dump_dir = os.path.realpath(dump)
        os.makedirs(dump_dir, exist_ok=True)
        torch.save(t.to(torch.bfloat16), os.path.join(dump_dir, f"{name}.pt"))


def _cache_entries(device):
    try:
        return device.num_program_cache_entries()
    except Exception as e:  # debug log only
        return f"<{type(e).__name__}>"


def _report_errmap(tag, out, golden, num_cores):
    """Error of out against golden ([H, W, C]) per region, the worst pixels and a pixel map."""
    h, w, _ = out.shape
    err = (out - golden).abs().amax(dim=-1)  # [H, W] max error over the channels of each pixel
    hh = torch.arange(h).reshape(-1, 1).expand(h, w)
    ww = torch.arange(w).reshape(1, -1).expand(h, w)
    border = (hh == 0) | (hh == h - 1) | (ww == 0) | (ww == w - 1)
    # Pixels within one image row of a shard edge (pixel index k * nhw / num_cores, NHW order).
    p = hh * w + ww
    edge = torch.zeros(h, w, dtype=torch.bool)
    for k in range(1, num_cores):
        b = k * (h * w // num_cores)
        edge |= (p >= b - w) & (p < b + w)
    regions = {"border": border, "shard_edge": edge & ~border, "interior": ~border & ~edge}
    thr = 4 * float(err.median()) + 1e-6
    for name, m in regions.items():
        if m.any():
            e = err[m]
            logger.info(
                f"[ERRMAP] {tag} {name}: pixels={int(m.sum())} mean_err={float(e.mean()):.5f} "
                f"max_err={float(e.max()):.5f} above_thr={int((e > thr).sum())} (thr={thr:.5f})"
            )
    top = torch.topk(err.flatten(), 16)
    worst = []
    for v, i in zip(top.values.tolist(), top.indices.tolist()):
        r, c = divmod(i, w)
        region = next(n for n, m in regions.items() if m[r, c])
        worst.append(f"({r},{c}) {v:.4f} {region}")
    logger.info(f"[ERRMAP] {tag} worst pixels (h,w) err region: {worst}")
    logger.info(f"[ERRMAP] {tag} map ('x' = err > {thr:.5f}):\n{_pixel_map(err > thr)}")


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("variant", ["model", "modelcfg", "one_core", "model_nocache"])
def test_layer1_conv2_repeat(mesh_device, variant, monkeypatch):
    device = mesh_device
    torch.manual_seed(0)
    monkeypatch.setenv("TT_METAL_QSR_CONV_SPLIT_PROGRAM", "1")  # the model's split path for HS conv2
    n_iter = int(os.environ.get("RESNET_REPEAT_N", "5"))
    if variant == "model_nocache":
        device.disable_and_clear_program_cache()

    # Post-ReLU activations and He-scaled weights, so the values look like the model's (many zeros, O(1)).
    torch_input = torch.relu(torch.randn((1, IN_CH, SPATIAL, SPATIAL))).bfloat16().float()
    torch_weight = (torch.randn((OUT_CH, IN_CH, 3, 3)) * (2.0 / (IN_CH * 9)) ** 0.5).bfloat16().float()
    torch_bias = (torch.randn((OUT_CH,)) * 0.1).bfloat16().float()
    golden = torch.relu(torch.nn.functional.conv2d(torch_input, torch_weight, bias=torch_bias, padding=1))
    golden = golden[0].permute(1, 2, 0)  # [H, W, C]

    nhw = SPATIAL * SPATIAL
    flat = torch.permute(torch_input, (0, 2, 3, 1)).reshape(1, 1, nhw, IN_CH).contiguous()
    grid = device.compute_with_storage_grid_size()
    if variant == "one_core":
        num_cores = 1
    else:
        num_cores = max(c for c in range(1, grid.x * grid.y + 1) if nhw % (c * 32) == 0)
    in_mem = ttnn.create_sharded_memory_config(
        shape=(1, 1, nhw // num_cores, IN_CH),
        core_grid=ttnn.num_cores_to_corerangeset(num_cores, grid, True),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    layout = ttnn.ROW_MAJOR_LAYOUT if variant == "modelcfg" else ttnn.TILE_LAYOUT
    tt_weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16)
    tt_bias = ttnn.from_torch(torch_bias.reshape(1, 1, 1, OUT_CH), dtype=ttnn.bfloat16)
    logger.info(f"[REPEAT] variant={variant} grid={grid.x}x{grid.y} input cores={num_cores} iterations={n_iter}")
    _dump(f"repeat_{variant}_input", torch_input[0].permute(1, 2, 0))
    _dump(f"repeat_{variant}_golden", golden)

    # The model's conv2 Conv2dConfig (ttnn_functional_resnet50.py, bottleneck conv2).
    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        deallocate_activation=True,
        reallocate_halo_output=False,
        act_block_h_override=32,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        reshard_if_not_optimal=(variant == "modelcfg"),
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.LoFi, packer_l1_acc=True
    )

    outs = []
    for it in range(n_iter):
        # A fresh input each time (the conv deallocates it), with the host weights, as each model module does.
        tt_input = ttnn.from_torch(flat, dtype=ttnn.bfloat16, layout=layout).to(device, in_mem)
        out, [oh, ow], _wb = ttnn.experimental.quasar.conv2d(
            input_tensor=tt_input,
            weight_tensor=tt_weight,
            bias_tensor=tt_bias,
            in_channels=IN_CH,
            out_channels=OUT_CH,
            batch_size=1,
            input_height=SPATIAL,
            input_width=SPATIAL,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            groups=1,
            device=device,
            conv_config=conv_config,
            compute_config=compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=ttnn.bfloat16,
        )
        if it == 0:
            logger.info(f"[REPEAT] variant={variant} output memory_config={out.memory_config()}")
        res = ttnn.to_torch(ttnn.from_device(out)).float()
        ttnn.deallocate(out)
        res = res.reshape(oh, ow, res.shape[-1])[:, :, :OUT_CH]
        outs.append(res)
        _dump(f"repeat_{variant}_it{it}", res)

        changed = (res != outs[0]).any(dim=-1)  # [H, W] pixels that differ from iteration 0
        first = [divmod(int(i), ow) for i in changed.flatten().nonzero().flatten()[:8]]
        logger.info(
            f"[REPEAT] variant={variant} it={it} sha256={_sha16(res)} "
            f"pixels_changed_vs_it0={int(changed.sum())} first_changed(h,w)={first} "
            f"pcc={_pcc(res, golden):.6f} max_err={float((res - golden).abs().max()):.5f} "
            f"cache_entries={_cache_entries(device)}"
        )

    _report_errmap(f"variant={variant} it=0", outs[0], golden, num_cores)
    diffs = [it for it in range(1, n_iter) if not torch.equal(outs[it], outs[0])]
    if diffs:
        changed = (outs[diffs[0]] != outs[0]).any(dim=-1)
        logger.info(f"[DIFFMAP] variant={variant} it={diffs[0]} vs it=0 ('x' = changed pixel):\n{_pixel_map(changed)}")

    assert not diffs, f"the output changed between iterations: iterations {diffs} differ from iteration 0"
    pcc0 = _pcc(outs[0], golden)
    assert pcc0 >= PCC, f"PCC {pcc0:.6f} < {PCC}"
