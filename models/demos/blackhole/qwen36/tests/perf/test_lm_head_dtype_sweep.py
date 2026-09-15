# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Price the LM-head weight dtype: perf against accuracy, on the real checkpoint weights.

WHY
---
MEASURED (T3K TP=8, 27B, tests/perf/test_profile_model_tail_decode.py) the LM-head matmul is
**866 us/token** and reads ~169 MB/device (5120 x 31,040 bf8) at **195 GB/s** -- i.e. it is AT the
Wormhole DRAM roofline. Blocking, core count and compute config are therefore all dead ends: the
op has no program_config at all and still saturates DRAM. The only lever is fewer BYTES, and the
only byte lever left is the weight dtype, bf8 -> bfp4, which should halve it to ~430 us.

Unlike the prefill-gather narrowing (which failed because that op turned out to be LATENCY bound,
see layer.py `_ff_gather_dtype`), the perf side here is predictable: at the bandwidth roofline bytes
map linearly to time. So this file exists to answer the OTHER question -- what the 4-bit mantissa
costs -- because the LM head is the most selection-sensitive matmul in the model. It is the last
projection before argmax, and greedy decode only cares about one thing: does the ARGMAX move.

WHY TOP-1 AGREEMENT AND NOT PCC
-------------------------------
PCC on logits is the wrong gate. Greedy decode is invariant to any error that does not reorder the
top of the distribution, and catastrophically sensitive to error that does -- so a 0.999 PCC can
still flip tokens, and a lower PCC can be harmless. This reports both, but the number that decides
is TOP-1 AGREEMENT against an fp32 reference, plus the margin distribution (how close top-1 and
top-2 are, which is what determines how often a given noise level can flip them).

SCOPE: ONE VOCAB SHARD
----------------------
The comparison runs on a single ``[dim, vocab/tp]`` shard rather than the full vocab, because that
is (a) what the served per-shard argmax actually reduces over, and (b) 8x cheaper on host RAM (the
full bf16 weight is ~2.5 GB). The cross-device winner-pick that follows is exact integer/compare
work and cannot introduce dtype error, so a per-shard result carries.

Run (needs a device; skipped unless the env var is set)::

    QWEN_LMHEAD_SWEEP=1 MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen3.6-27B \\
      pytest models/demos/blackhole/qwen36/tests/perf/test_lm_head_dtype_sweep.py -v -s
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, run_for_wormhole_b0_or_blackhole

BATCH = 32
TRIALS = 8  # BATCH*TRIALS rows of top-1 agreement statistics
_SKIP = os.environ.get("QWEN_LMHEAD_SWEEP") != "1"

try:
    from tracy import signpost as _SP
except ImportError:  # pragma: no cover
    _SP = None

DTYPES = {"bf16": ttnn.bfloat16, "bf8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}


def _mesh_shape():
    name = (os.environ.get("MESH_DEVICE") or "").upper()
    explicit = {"P150": (1, 1), "N150": (1, 1), "P150X4": (1, 4), "N150X4": (1, 4), "N300": (1, 2), "T3K": (1, 8)}
    return explicit.get(name, (1, max(1, min(ttnn.get_num_devices(), 2))))


MESH_SHAPE = _mesh_shape()
_MULTI = MESH_SHAPE != (1, 1)
DEVICE_PARAMS = [
    {
        "l1_small_size": 24576,
        "num_command_queues": 2,
        **({"fabric_config": ttnn.FabricConfig.FABRIC_1D} if _MULTI else {}),
    }
]


def _load_lm_head(ckpt_dir):
    """``lm_head.weight`` [vocab, dim] as bf16, dequantizing the fp8-block form if present."""
    from safetensors import safe_open

    from models.demos.blackhole.qwen36.tt.tp_common import dequant_fp8_block

    ckpt_dir = Path(ckpt_dir)
    wm = json.load(open(ckpt_dir / "model.safetensors.index.json"))["weight_map"]
    key = next(k for k in wm if k.endswith("lm_head.weight"))
    with safe_open(str(ckpt_dir / wm[key]), framework="pt") as sf:
        w = sf.get_tensor(key)
    sk = key + "_scale_inv"
    if wm.get(sk):
        with safe_open(str(ckpt_dir / wm[sk]), framework="pt") as sf2:
            w = dequant_fp8_block(w, sf2.get_tensor(sk))
    return w.to(torch.bfloat16)


@pytest.mark.skipif(_SKIP, reason="set QWEN_LMHEAD_SWEEP=1 to run the LM-head dtype sweep")
@pytest.mark.timeout(2400)
@run_for_wormhole_b0_or_blackhole()
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_lm_head_dtype_sweep(mesh_device, device_params):
    """bf16 / bf8 / bfp4 LM head: device time, logit PCC, and top-1 agreement vs fp32."""
    del device_params
    from models.demos.blackhole.qwen36.tests.test_factory import model_path
    from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs

    mesh_device.enable_program_cache()
    os.environ.setdefault("HF_MODEL", model_path())
    args = Qwen36ModelArgs(mesh_device, max_batch_size=BATCH, max_seq_len=256)
    nd = max(1, mesh_device.get_num_devices())
    dim, vocab = args.dim, args.vocab_size
    per_shard = vocab // nd

    w_full = _load_lm_head(args.CKPT_DIR)  # [vocab, dim]
    assert w_full.shape[0] >= per_shard and w_full.shape[1] == dim, f"unexpected lm_head {tuple(w_full.shape)}"
    # One shard, transposed to the model's [dim, vocab/tp] layout (model.py stores output.weight.T).
    w_shard = w_full[:per_shard].T.contiguous()
    del w_full
    logger.info(f"lm_head dtype sweep: dim={dim} vocab={vocab} per_shard={per_shard} mesh={MESH_SHAPE}")

    torch.manual_seed(0)
    # Post-RMSNorm activations are ~unit RMS, so N(0,1) is the right scale for this projection.
    xs = [torch.randn(1, 1, BATCH, dim, dtype=torch.bfloat16) for _ in range(TRIALS)]
    wf32 = w_shard.float()
    refs = [x.float()[0, 0] @ wf32 for x in xs]  # [BATCH, per_shard] fp32 reference logits
    ref_top1 = [r.argmax(dim=-1) for r in refs]
    # Margin between top-1 and top-2 in the reference: the quantity that decides how much logit
    # noise it takes to flip a token. Reported so the agreement rate can be interpreted.
    margins = torch.cat([(r.topk(2, dim=-1).values[:, 0] - r.topk(2, dim=-1).values[:, 1]) for r in refs])
    logger.info(
        f"reference top1-top2 margin: median={margins.median():.4f} "
        f"p10={margins.kthvalue(max(1, int(0.1 * margins.numel()))).values:.4f} min={margins.min():.4f}"
    )

    rep = ttnn.ReplicateTensorToMesh(mesh_device) if _MULTI else None
    x_tts = [
        ttnn.from_torch(
            x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, **({"mesh_mapper": rep} if rep else {})
        )
        for x in xs
    ]

    results = {}
    base_top1 = None
    for name, dt in DTYPES.items():
        w_tt = ttnn.from_torch(
            w_shard,
            dtype=dt,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **({"mesh_mapper": rep} if rep else {}),
        )
        outs = [ttnn.linear(x_tt, w_tt) for x_tt in x_tts]  # warmup + correctness
        ttnn.synchronize_device(mesh_device)

        agree_ref, agree_base, pccs = 0, 0, []
        tops = []
        for i, o in enumerate(outs):
            host = ttnn.to_torch(ttnn.get_device_tensors(o)[0] if _MULTI else o).float()[0, 0]
            t1 = host.argmax(dim=-1)
            tops.append(t1)
            agree_ref += int((t1 == ref_top1[i]).sum())
            pccs.append(float(comp_pcc(refs[i], host, 0.0)[1]))
        for o in outs:
            ttnn.deallocate(o)

        if base_top1 is None and name == "bf8":
            base_top1 = tops
        if base_top1 is not None:
            agree_base = sum(int((tops[i] == base_top1[i]).sum()) for i in range(TRIALS))

        # Timed pass, signposted so tracy can slice the matmul per dtype.
        if _SP is not None:
            _SP(f"{name}_start")
        for x_tt in x_tts:
            o = ttnn.linear(x_tt, w_tt)
            ttnn.deallocate(o)
        ttnn.synchronize_device(mesh_device)
        if _SP is not None:
            _SP(f"{name}_stop")
        ttnn.deallocate(w_tt)

        n = BATCH * TRIALS
        results[name] = dict(top1_vs_fp32=agree_ref / n, pcc=min(pccs), top1_vs_bf8=agree_base / n if base_top1 else 0)
        logger.info(
            f"  {name:5} top1-vs-fp32={agree_ref}/{n} ({100.0 * agree_ref / n:.2f}%)  "
            f"worst-logit-PCC={min(pccs):.6f}"
        )

    for x_tt in x_tts:
        ttnn.deallocate(x_tt)

    logger.info("=== LM-head dtype summary ===")
    for name, r in results.items():
        logger.info(
            f"  {name:5} top1 vs fp32 {100.0 * r['top1_vs_fp32']:6.2f}%   "
            f"top1 vs bf8 {100.0 * r['top1_vs_bf8']:6.2f}%   worst PCC {r['pcc']:.6f}"
        )
    # bf8 is what ships; if IT disagrees badly with fp32 the harness or scale is wrong, not the dtype.
    assert results["bf8"]["top1_vs_fp32"] > 0.90, f"bf8 baseline itself looks wrong: {results['bf8']}"
