# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone isolation of the Quasar `ttnn.experimental.quasar.transpose` used by the resnet50 stem
fold (`fold_with_transpose_sharded_`).

WHY
---
test_fold.py shows fold is perfect on WH (PCC 1.0) but scrambled on the Quasar emulator (PCC ~0:
pad channels contaminated, most output zeroed, worse for larger batch) and HANGS on craq-sim. The
sharded fold is a chain of 4x `quasar::transpose` + 2x `quasar::pad` + 2x `view` + `slice`, and the
transpose is the prime suspect: craq-sim wedges inside `transpose_wh_rm` (reader NARW / writer WFW /
DFB counter posted-never-acked). The corruption scales with shard SIZE — the emulator packs the
whole tensor onto its 2 cores in one huge shard ([6613,16]/[13226,16]) while WH uses 56 small
[237,16] shards — so a large-single-shard transpose stride/data-movement bug fits.

This test exercises `quasar.transpose` ALONE, at the fold's exact intermediate shapes, with the core
count TIED TO THE DEVICE (via _fit_cores) so it runs the same 2-core big-shard config on the emulator
that fold hits (the pre-existing test_quasar_transpose_wh hard-codes an 8x4 grid and SKIPS on the
emulator). If this reproduces the scramble, the transpose is the fold bug; the same fix should also
unblock the craq-sim hang.

The two transpose flavors the fold uses:
  * transpose(2,3): H<->W  (routes to TransposeWHShardedRMProgramFactory) -- the craq-sim hang site.
  * transpose(1,2): C<->H.

RUN (emulator or craq-sim, slow dispatch, forced JIT):
  TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 \
  pytest models/demos/vision/classification/resnet50/quasar/tests/ops/test_fold_transpose.py
Localized diagnostics are written to fold_transpose_dbg_*.log (flushed+fsync'd; survives a kill/hang)
as well as stdout; set FOLD_DBG_LOG to override the path.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def _fit_cores(total_rows, device):
    """Largest core count <= device cores that divides total_rows (exact, unpadded shards).
    On the 2-core Quasar emulator this yields 2 cores -> one huge shard, matching the fold config."""
    grid = device.compute_with_storage_grid_size()
    cap = min(total_rows, grid.x * grid.y)
    num_cores = cap
    while num_cores > 1 and total_rows % num_cores != 0:
        num_cores -= 1
    return num_cores, grid


# (n, c, h, w, dim0, dim1, id) -- the fold's transpose intermediates, batch 1 and 2.
TRANSPOSE_CONFIGS = [
    # First fold transpose: padded input [n, C=4, padded_h32=256, w=224], transpose(2,3) -> [n,4,224,256].
    # This is transpose_wh (TransposeWHShardedRMProgramFactory) -- the craq-sim hang site.
    (1, 4, 256, 224, 2, 3, "b1_wh_4x256x224"),
    (2, 4, 256, 224, 2, 3, "b2_wh_4x256x224"),
    # C<->H transpose the fold also uses: [n, 4, 224, 256], transpose(1,2) -> [n,224,4,256].
    (1, 4, 224, 256, 1, 2, "b1_hc_4x224x256"),
]


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "n, c, h, w, dim0, dim1", [cfg[:6] for cfg in TRANSPOSE_CONFIGS], ids=[cfg[6] for cfg in TRANSPOSE_CONFIGS]
)
def test_quasar_fold_transpose(mesh_device, n, c, h, w, dim0, dim1):
    device = mesh_device
    torch.manual_seed(0)

    import os
    import time
    import traceback

    tid = f"{n}x{c}x{h}x{w}_t{dim0}{dim1}"
    log_path = os.environ.get("FOLD_DBG_LOG", os.path.abspath(f"fold_transpose_dbg_{tid}.log"))

    def _dbg(msg):
        line = f"[xpose-dbg {tid} t={time.time():.3f}] {msg}"
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

    def _unravel(flat, shape):
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
            f"nan={bool(torch.isnan(tf).any())} inf={bool(torch.isinf(tf).any())}"
        )

    # HEIGHT-shard over the flattened first-3-dims rows (n*c*h), width w -- mirrors the fold's sharding.
    total_rows = n * c * h
    num_cores, grid = _fit_cores(total_rows, device)
    shard_grid = ttnn.num_cores_to_corerangeset(num_cores, grid, row_wise=True)
    shard_spec = ttnn.ShardSpec(shard_grid, (total_rows // num_cores, w), ttnn.ShardOrientation.ROW_MAJOR)
    in_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    torch_input = torch.randn((n, c, h, w), dtype=torch.bfloat16)
    golden = torch_input.transpose(dim0, dim1).contiguous()

    _dbg(
        f"START config: n={n} c={c} h={h} w={w} transpose({dim0},{dim1}) total_rows={total_rows} "
        f"num_cores={num_cores} grid={grid.x}x{grid.y} shard=[{total_rows // num_cores},{w}] log={log_path}"
    )
    _stats("torch_input", torch_input)
    _stats("golden(transposed)", golden)

    got = None
    exc = None
    try:
        _dbg("STAGE from_torch: begin")
        tt_in = ttnn.from_torch(
            torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_mem_config
        )
        _dbg("STAGE from_torch: done")

        _dbg(f"STAGE transpose({dim0},{dim1}): begin")
        tt_out = ttnn.experimental.quasar.transpose(tt_in, dim0, dim1)
        _dbg(
            f"STAGE transpose: done; shape={list(tt_out.shape)} layout={tt_out.layout} dtype={tt_out.dtype} "
            f"mem={tt_out.memory_config()}"
        )

        _dbg("STAGE to_torch: begin")
        got = ttnn.to_torch(tt_out).to(torch.bfloat16)
        _dbg("STAGE to_torch: done")
    except Exception as e:  # noqa: BLE001
        exc = e
        _dbg("EXCEPTION during device transpose:\n" + traceback.format_exc())

    if got is not None:
        try:
            _stats("got(device)", got)
            try:
                torch.save({"golden": golden, "got": got, "torch_input": torch_input}, log_path.replace(".log", ".pt"))
                _dbg(f"saved tensors -> {log_path.replace('.log', '.pt')}")
            except Exception as e:  # noqa: BLE001
                _dbg(f"tensor save failed: {e!r}")

            g = golden.float()
            r = got.float()
            if tuple(g.shape) != tuple(r.shape):
                _dbg(f"!!! SHAPE MISMATCH golden={tuple(g.shape)} got={tuple(r.shape)}")
            else:
                _dbg(f"OVERALL pcc={_fmt(_pcc(g, r))}")
                diff = (g - r).abs()
                _dbg(
                    f"abs-err: max={diff.max().item():.5f} mean={diff.mean().item():.6f} "
                    f"frac_wrong(>1e-2)={(diff > 1e-2).float().mean().item():.4f}"
                )
                # Partial-write signature (the fold failure mode): got has far fewer nonzeros than golden.
                gz, rz = (g != 0).sum().item(), (r != 0).sum().item()
                _dbg(
                    f"nnz golden={gz} got={rz} ({(rz / max(gz, 1)):.3f} of golden) "
                    f"-> {'PARTIAL/MISSING WRITES' if rz < 0.9 * gz else 'ok-count'}"
                )
                worst = _unravel(int(torch.argmax(diff).item()), g.shape)
                _dbg(f"worst elem at {worst}: golden={g[worst].item():.4f} got={r[worst].item():.4f}")

                # Per-outer-slice PCC over dim0 then dim1 of the OUTPUT to localize which shard-region is
                # wrong (a stride bug tends to corrupt a contiguous block of sticks/slices).
                for d in (0, 1):
                    dim_len = g.shape[d]
                    if dim_len <= 1:
                        continue
                    per = []
                    for i in range(min(dim_len, 8)):
                        sl = [slice(None)] * g.ndim
                        sl[d] = i
                        per.append(f"{i}:{_fmt(_pcc(g[tuple(sl)], r[tuple(sl)]))}")
                    _dbg(f"PER-DIM{d} pcc (first {len(per)} of {dim_len}): " + "  ".join(per))

                # Spot-check the first output row and the middle output row raw values.
                flat_g = g.reshape(-1, g.shape[-1])
                flat_r = r.reshape(-1, r.shape[-1])
                for ri in [0, 1, flat_g.shape[0] // 2, flat_g.shape[0] - 1]:
                    _dbg(
                        f"  row{ri}: golden={[round(v,3) for v in flat_g[ri, :min(8, flat_g.shape[1])].tolist()]} "
                        f"got={[round(v,3) for v in flat_r[ri, :min(8, flat_r.shape[1])].tolist()]}"
                    )
        except Exception:  # noqa: BLE001
            _dbg("EXCEPTION during diagnostics:\n" + traceback.format_exc())

    _dbg("END diagnostics")
    if exc is not None:
        raise exc
    # transpose is a pure data-preserving permutation.
    assert_with_pcc(golden.to(torch.bfloat16), got, pcc=0.999)
