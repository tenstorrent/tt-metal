# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone per-op test for the Quasar resnet50 stem `ttnn.experimental.quasar.fold`.

WHERE IT COMES FROM
-------------------
resnet50 (models/.../quasar/tt/ttnn_functional_resnet50.py, run()) folds the input image into
channels (space-to-depth) as the very first op, so the 4x4 stem conv can run at unity stride:

    fold_output_tensor = ttnn.experimental.quasar.fold(
        input_tensor,
        self.fold_stride_h, self.fold_stride_w,   # = stride = 2
        use_transpose_as_fold=True,
        padding=[self.fold_pad_h, self.fold_pad_h, self.fold_pad_w, self.fold_pad_w, 0, self.fold_pad_c],
        grid_size=self.fold_compute_grid_size,
        override_memory_config=self.override_fold_mem_config,
    )

For resnet50 the constructor is called with kernel_size=3, stride=2 and a (N, 3, 224, 224) image
(see resnet50_test_infra.py), which gives:
    fold_stride_h = fold_stride_w = 2
    fold_pad_h = fold_pad_w = kernel_size = 3
    fold_pad_c = nearest_y(3, 4) - 3 = 1              (channels padded 3 -> 4)
    padding = [3, 3, 3, 3, 0, 1]
    fold_output_shape = (N, 230//2, 230//2, 4*2*2) = (N, 115, 115, 16)

The input to fold is the ROW_MAJOR, HEIGHT_SHARDED NCHW image (setup_l1_sharded_input shards it over
the flattened N*C*H rows). This test reproduces that exact configuration; only the batch and the core
count are tied to the device so it runs on the small Quasar sim grid as well as full silicon.

WHAT IT VALIDATES
-----------------
`use_transpose_as_fold=True` implements the fold as a sequence of transposes/reshapes on device. It
is data-preserving (a pure rearrangement), so `ttnn.to_torch` of the output must equal the reference
CPU fold. A wrong Quasar LLK binding (transpose/reshape data movement, sharded reader/writer)
corrupts the data or hangs. The golden below is the well-established CPU reference used by the
existing WH/BH fold tests (tests/ttnn/unit_tests/operations/conv/data_movement/test_fold_op.py).

RUN
---
  TT_METAL_SIMULATOR=~/sim/libttsim.so \
  TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 \
  pytest models/demos/vision/classification/resnet50/quasar/tests/ops/test_fold.py
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import _nearest_y
from tests.ttnn.utils_for_testing import assert_with_pcc


def pad_and_fold_conv_activation_for_unity_stride(activation_pyt_nchw_tensor, pad_h, pad_w, stride_h, stride_w):
    """CPU reference fold (NCHW in, NCHW folded out), copied from the WH/BH fold op test."""
    assert stride_h == stride_w
    assert activation_pyt_nchw_tensor.shape[2] == activation_pyt_nchw_tensor.shape[3]
    # Pad channels to a multiple of 4 (keeps L1 read addresses 16-bit aligned), plus the conv padding.
    C = _nearest_y(activation_pyt_nchw_tensor.shape[1], 4)
    activation_pyt_padded = torch.nn.functional.pad(
        activation_pyt_nchw_tensor, (pad_w, pad_w, pad_h, pad_h, 0, C - activation_pyt_nchw_tensor.shape[1])
    )
    assert activation_pyt_padded.shape[2] % stride_h == 0
    activation_pyt_padded_folded = torch.zeros(
        [
            activation_pyt_padded.shape[0],
            C * stride_h * stride_w,
            (int)(activation_pyt_padded.shape[2] / stride_h),
            (int)(activation_pyt_padded.shape[3] / stride_w),
        ]
    )
    for h in range(0, activation_pyt_padded.shape[2], stride_h):
        for w in range(0, activation_pyt_padded.shape[3], stride_w):
            folded_h = (int)(h / stride_h)
            folded_w = (int)(w / stride_w)
            for i in range(stride_h * stride_w):
                start_c = i * C
                activation_pyt_padded_folded[:, start_c : start_c + C, folded_h, folded_w] = activation_pyt_padded[
                    :, :, h + (int)(i / stride_w), w + (int)(i % stride_w)
                ]
    return activation_pyt_padded_folded


def _fit_cores(total_rows, device):
    """Largest core count <= device cores that divides total_rows (so the height shards are exact)."""
    grid = device.compute_with_storage_grid_size()
    cap = min(total_rows, grid.x * grid.y)
    num_cores = cap
    while num_cores > 1 and total_rows % num_cores != 0:
        num_cores -= 1
    return num_cores, grid


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("batch_size", [1, 2], ids=["b1", "b2"])
# use_transpose_as_fold: True = transpose-chain fold (hits the broken WH transpose on the 2-core grid);
# False = direct data-movement fold op (ttnn::prim::qsr::fold, no WH transpose) -- the pivot under test.
@pytest.mark.parametrize("use_transpose_as_fold", [True, False], ids=["xpose", "direct"])
def test_quasar_fold(mesh_device, batch_size, use_transpose_as_fold):
    device = mesh_device
    torch.manual_seed(0)

    # resnet50 stem fold params.
    c, h, w = 3, 224, 224
    kernel_size = 3
    stride_h = stride_w = 2
    pad_h = pad_w = kernel_size  # fold_pad_h/w = kernel_size = 3
    C = _nearest_y(c, 4)  # 4
    pad_c = C - c  # 1

    torch_input = torch.rand((batch_size, c, h, w), dtype=torch.bfloat16)

    # Golden: CPU fold (NCHW folded) then permute to NHWC to match the device output layout.
    golden = pad_and_fold_conv_activation_for_unity_stride(torch_input, pad_h, pad_w, stride_h, stride_w)
    golden = torch.permute(golden, (0, 2, 3, 1))

    # HEIGHT-shard the ROW_MAJOR NCHW image over the flattened N*C*H rows, tied to the device grid
    # (mirrors setup_l1_sharded_input). Use an exact divisor so shards are unpadded.
    total_rows = batch_size * c * h
    num_cores, grid = _fit_cores(total_rows, device)
    shard_grid = ttnn.num_cores_to_corerangeset(num_cores, grid, row_wise=True)
    shard_spec = ttnn.ShardSpec(shard_grid, (total_rows // num_cores, w), ttnn.ShardOrientation.ROW_MAJOR)
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    # ------------------------------------------------------------------ DEBUG
    # Comprehensive, file-backed instrumentation. The emulator captures/loses stdout
    # and can be killed mid-run, so every marker is flushed to a log file immediately
    # (append + flush + fsync) as well as printed. Localizes WHERE the fold output
    # diverges (per-channel / per-fold-group / per-spatial / pad channels) so we can
    # tell a bad transpose from a bad untilize/pad without device-side trace.
    import os
    import time
    import traceback

    mode = "xpose" if use_transpose_as_fold else "direct"
    log_path = os.environ.get("FOLD_DBG_LOG", os.path.abspath(f"fold_dbg_b{batch_size}_{mode}.log"))

    def _dbg(msg):
        line = f"[fold-dbg b{batch_size} t={time.time():.3f}] {msg}"
        try:
            with open(log_path, "a") as fh:
                fh.write(line + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        except Exception:
            pass
        print(line, flush=True)

    def _pcc(a, b):
        a = a.detach().float().flatten()
        b = b.detach().float().flatten()
        if a.numel() != b.numel():
            return f"SHAPE-MISMATCH a={a.numel()} b={b.numel()}"
        if torch.equal(a, b):
            return 1.0
        sa, sb = a.std().item(), b.std().item()
        if sa == 0.0 or sb == 0.0:
            return f"CONSTANT a_std={sa:.3e} b_std={sb:.3e}"
        return torch.corrcoef(torch.stack([a, b]))[0, 1].item()

    def _fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    def np_unravel(flat, shape):
        idx = []
        for dim in reversed(shape):
            idx.append(int(flat % dim))
            flat //= dim
        return tuple(reversed(idx))

    def _stats(name, t):
        tf = t.detach().float()
        _dbg(
            f"{name}: shape={tuple(t.shape)} dtype={t.dtype} "
            f"min={tf.min().item():.4f} max={tf.max().item():.4f} mean={tf.mean().item():.4f} "
            f"std={tf.std().item():.4f} nnz={(tf != 0).sum().item()}/{tf.numel()} "
            f"nan={torch.isnan(tf).any().item()} inf={torch.isinf(tf).any().item()}"
        )

    _dbg(
        f"START config: mode={mode} N={batch_size} c={c} h={h} w={w} stride={stride_h}x{stride_w} "
        f"pad_hw={pad_h} pad_c={pad_c} C={C} total_rows={total_rows} num_cores={num_cores} "
        f"grid={grid.x}x{grid.y} log={log_path}"
    )
    _stats("torch_input(NCHW)", torch_input)
    _stats("golden(NHWC)", golden)

    got = None
    exc = None
    try:
        _dbg("STAGE from_torch: begin")
        tt_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=input_mem_config,
        )
        _dbg("STAGE from_torch: done")

        _dbg(f"STAGE fold: begin (use_transpose_as_fold={use_transpose_as_fold})")
        tt_out = ttnn.experimental.quasar.fold(
            tt_input,
            stride_h,
            stride_w,
            use_transpose_as_fold=use_transpose_as_fold,
            padding=[pad_h, pad_h, pad_w, pad_w, 0, pad_c],
            grid_size=shard_grid,
        )
        _dbg(
            f"STAGE fold: done; tt_out spec shape={list(tt_out.shape)} layout={tt_out.layout} "
            f"dtype={tt_out.dtype} mem={tt_out.memory_config()}"
        )

        _dbg("STAGE to_torch: begin")
        got = ttnn.to_torch(tt_out).to(torch.bfloat16)
        _dbg("STAGE to_torch: done")
    except Exception as e:  # noqa: BLE001
        exc = e
        _dbg("EXCEPTION during device fold:\n" + traceback.format_exc())

    if got is not None:
        try:
            _stats("got(device)", got)
            # Save both for offline diff.
            try:
                torch.save({"golden": golden, "got": got, "torch_input": torch_input}, log_path.replace(".log", ".pt"))
                _dbg(f"saved tensors -> {log_path.replace('.log', '.pt')}")
            except Exception as e:  # noqa: BLE001
                _dbg(f"tensor save failed: {e!r}")

            g = golden.float()
            r = got.float()
            if tuple(g.shape) != tuple(r.shape):
                _dbg(
                    f"!!! SHAPE MISMATCH golden={tuple(g.shape)} got={tuple(r.shape)} "
                    f"(cannot do elementwise; skipping localized diffs)"
                )
            else:
                _dbg(f"OVERALL pcc={_pcc(g, r)}")
                diff = (g - r).abs()
                _dbg(
                    f"abs-err: max={diff.max().item():.5f} mean={diff.mean().item():.6f} "
                    f"frac_wrong(>1e-2)={(diff > 1e-2).float().mean().item():.4f}"
                )
                # Location of the single worst element.
                fi = torch.argmax(diff).item()
                idx = np_unravel(fi, g.shape)
                _dbg(f"worst elem at {idx}: golden={g[idx].item():.4f} got={r[idx].item():.4f}")

                # Per-channel PCC (last dim = 16 folded channels).
                Cf = g.shape[-1]
                per_ch = [f"ch{ci}:{_fmt(_pcc(g[..., ci], r[..., ci]))}" for ci in range(Cf)]
                _dbg("PER-CHANNEL pcc: " + "  ".join(per_ch))

                # Per-fold-group PCC: stride_h*stride_w groups of C channels each. Group i is the
                # sub-sample at spatial offset (i//stride_w, i%stride_w). Isolates a bad transpose /
                # sub-position interleave from a bad pad/channel path.
                ngroups = stride_h * stride_w
                for i in range(ngroups):
                    sl = slice(i * C, (i + 1) * C)
                    off = (i // stride_w, i % stride_w)
                    _dbg(f"  group{i} (offset {off}, ch[{i*C}:{(i+1)*C}]) pcc={_fmt(_pcc(g[..., sl], r[..., sl]))}")

                # Pad-channel check: within each group of C=4, channel index (C-1) is the padded
                # channel (pad_c=1) and must be 0 in the golden; report got's content there.
                for i in range(ngroups):
                    pc = i * C + (C - 1)
                    gp, rp = g[..., pc], r[..., pc]
                    _dbg(
                        f"  pad-ch{pc}: golden_nnz={(gp != 0).sum().item()} "
                        f"got_nnz={(rp != 0).sum().item()} got_absmax={rp.abs().max().item():.4f}"
                    )

                # Spatial spot checks (first few output sticks).
                for yy, xx in [(0, 0), (0, 1), (1, 0), (g.shape[1] // 2, g.shape[2] // 2)]:
                    _dbg(f"  stick[0,{yy},{xx},:] golden={g[0, yy, xx, :].tolist()} " f"got={r[0, yy, xx, :].tolist()}")
        except Exception as e:  # noqa: BLE001
            _dbg("EXCEPTION during diagnostics:\n" + traceback.format_exc())

    _dbg("END diagnostics")
    if exc is not None:
        raise exc
    # ---------------------------------------------------------------- END DEBUG
    # fold is a pure data-preserving rearrangement.
    assert_with_pcc(golden.to(torch.bfloat16), got, pcc=0.999)
