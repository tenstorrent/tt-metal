# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ttKDA row-local prefill on a multi-row mesh (``TtKDA.prefill_forward(..., local=True)``: ttnn's
``sequence_parallel_size=1`` on the chronology ops): every mesh row runs ITS OWN prompt of ``T`` rows (the wave
of per-row prompts, doc/mesh/galaxy_throughput_plan.md A1), the mesh stays (rows, 4) for the TP columns.

Checks on (rows, 4): (i) each row's output and final carries against HF on that row's prompt (rows must not
contaminate each other), (ii) row 0's local output against the sequence-parallel path's output for the same
prompt (the two scans agree within bf16 ulps). Needs a ttnn with the ``sequence_parallel_size`` argument
(branch arg/kda_sequence_parallel_size); the test skips on an older ttnn.

    GLM53_MESH_SHAPE=8,4 pytest models/demos/glm53_flash/tests/test_mesh_local_kda.py -s
"""
from __future__ import annotations

import inspect
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.tests.test_kda import LAYER, _hf_linear_attention, weights  # noqa: F401 (fixture)
from models.demos.glm53_flash.tests.utils import pcc
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.kda import TtKDA
from models.demos.glm53_flash.tt.mesh import GLMMesh, mesh_device_params

MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "2,4").split(","))
T = int(os.environ.get("GLM53_LOCAL_T", "1024"))  # rows per row (one chunk per row)


def _ttnn_has_sequence_parallel_size() -> bool:
    """The registered op is a ``FastOperation`` whose signature is ``(*args, **kwargs)``; the nanobind call
    method underneath carries the real signature in its docstring."""
    op = ttnn.experimental.kda.prepare_chunk_recurrence
    inner = getattr(op, "function", op)
    texts = [getattr(o, "__doc__", "") or "" for o in (op, inner, getattr(type(inner), "__call__", None))]
    try:
        texts.append(str(inspect.signature(inner)))
    except (TypeError, ValueError):
        pass
    return any("sequence_parallel_size" in t for t in texts)


def _hf_run(hf, text, x):
    from transformers import DynamicCache

    cache = DynamicCache(config=text)
    with torch.no_grad():
        y = hf(x, cache_params=cache, attention_mask=torch.ones(1, x.shape[1], dtype=torch.bool))
    return y, cache.layers[LAYER]


def _carries(mesh, new_state, cfg):
    """Per mesh row: (recurrent [1, H, K, V] over the columns, conv history [1, 3, 3*H*K] in q|k|v order)."""
    rec_shards = [ttnn.to_torch(t) for t in mesh.device_tensors(new_state.recurrent)]
    conv_shards = [ttnn.to_torch(t) for t in mesh.device_tensors(new_state.convolution)]
    if new_state.convolution.layout == ttnn.TILE_LAYOUT:  # block path: tap-major tiles [taps, B, W] -> [B, taps, W]
        conv_shards = [c.permute(1, 0, 2).contiguous() for c in conv_shards]
    out = []
    for r in range(mesh.rows):
        cols = [rec_shards[r * mesh.cols + c] for c in range(mesh.cols)]
        rec = torch.cat(cols, dim=1)
        cv = [conv_shards[r * mesh.cols + c] for c in range(mesh.cols)]
        W_loc = cv[0].shape[-1] // 3
        conv = torch.cat(
            [torch.cat([cv[c][..., i * W_loc : (i + 1) * W_loc] for c in range(mesh.cols)], -1) for i in range(3)], -1
        )
        out.append((rec, conv))
    return out


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_local_kda_prefill_one_prompt_per_row_vs_hf(mesh_device, weights):
    if not _ttnn_has_sequence_parallel_size():
        pytest.skip("this ttnn's KDA ops have no sequence_parallel_size (branch arg/kda_sequence_parallel_size)")
    snapshot, text, raw = weights
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    hf = _hf_linear_attention(snapshot, text, raw)
    attn = {k[len("self_attn.") :]: v for k, v in raw.items()}
    mesh = GLMMesh.from_device(mesh_device)
    rows = mesh.rows
    assert rows > 1, "the row-local mode is the multi-row case"
    tt = TtKDA(mesh_device, cfg, attn, layer_idx=LAYER, mesh=mesh)
    torch.manual_seed(0)
    xs = [(torch.randn(1, T, cfg.hidden_size) * 0.8).to(torch.bfloat16) for _ in range(rows)]  # one prompt per row
    refs = [_hf_run(hf, text, x) for x in xs]

    # (i) every row its own prompt, row-locally
    x_dev = mesh.upload(torch.cat(xs, dim=0), ttnn.bfloat16, rows=0)  # [1, T, 4096] per row = that row's prompt
    state = tt.allocate_state(1)
    y, new_state = tt.prefill_forward(x_dev, state, tt.scalar(0), None, local=True)
    out = mesh.to_host(y, rows=0).float()  # [rows, T, 4096]
    assert out.shape[0] == rows and out.shape[1] == T
    carries = _carries(mesh, new_state, cfg)
    worst = 1.0
    for r in range(rows):
        y_hf, layer = refs[r]
        p = pcc(y_hf, out[r : r + 1])
        p_rec = pcc(layer.recurrent_states[0], carries[r][0])
        conv_hf = layer.conv_states[0][:, :, -3:].transpose(1, 2)
        p_conv = pcc(conv_hf, carries[r][1])
        cross = max(pcc(refs[q][0], out[r : r + 1]) for q in range(rows) if q != r)
        logger.info(
            f"local ttKDA layer {LAYER} T={T} on {MESH}: row {r} output pcc vs HF {p:.6f} (vs another row's prompt "
            f"{cross:.4f}), recurrent {p_rec:.6f}, conv {p_conv:.6f}"
        )
        worst = min(worst, p, p_rec, p_conv)
        assert p > 0.995 and p_rec > 0.995 and p_conv > 0.995, f"row {r}"
        assert cross < 0.9, f"row {r}'s output matches another row's prompt: the rows share a chronology"

    # (ii) the sequence-parallel path on row 0's prompt agrees with the local output (the same math, grouped differently)
    assert T % (rows * 32) == 0
    x_sp = mesh.upload(xs[0], ttnn.bfloat16, rows=1)  # slab r = rows [r*T/rows, (r+1)*T/rows) of prompt 0
    state_sp = tt.allocate_state(1)
    y_sp, _ = tt.prefill_forward(x_sp, state_sp, tt.scalar(0), None)
    out_sp = mesh.to_host(y_sp, rows=1).float()  # [1, T, 4096]
    p_sp = pcc(out_sp, out[0:1])
    logger.info(f"  sequence-parallel vs local on prompt 0: pcc {p_sp:.6f} (worst local gate {worst:.6f})")
    assert p_sp > 0.999


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_sp_prefill_padded_ragged_chunk_carry(mesh_device, weights):
    """The served shape for the sequence-parallel KDA scan: a chunk whose valid end lies inside a bucket-padded chunk
    (the 2026-10-29 APC smoke: 3968 valid rows in a 16384-row chunk on the 8x4, rows 2-7 entirely past the end; here
    3968 of 16384 on the block: row 1 entirely past it) vs the same rows padded minimally: the valid outputs and the
    carries (recurrent state, conv history) agree with HF and with each other (different chunk lengths group the
    scan differently: bf16/fp32 rounding only)."""
    snapshot, text, raw = weights
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    hf = _hf_linear_attention(snapshot, text, raw)
    attn = {k[len("self_attn.") :]: v for k, v in raw.items()}
    mesh = GLMMesh.from_device(mesh_device)
    rows = mesh.rows
    tt = TtKDA(mesh_device, cfg, attn, layer_idx=LAYER, mesh=mesh)
    n = 3968
    torch.manual_seed(1)
    x = (torch.randn(1, n, cfg.hidden_size) * 0.8).to(torch.bfloat16)
    y_hf, layer = _hf_run(hf, text, x)

    def run(T):
        xp = torch.zeros(1, T, cfg.hidden_size, dtype=torch.bfloat16)
        xp[:, :n] = x
        x_dev = mesh.upload(xp, ttnn.bfloat16, rows=1)  # slab r = rows [r*T/rows, (r+1)*T/rows)
        state = tt.allocate_state(1)
        y, new_state = tt.prefill_forward(x_dev, state, tt.scalar(0), tt.scalar(n))
        out = mesh.to_host(y, rows=1).float()[:, :n]
        return out, _carries(mesh, new_state, cfg)[0]

    unit = 32 * rows
    y_min, (rec_min, conv_min) = run(-(-n // unit) * unit)
    y_pad, (rec_pad, conv_pad) = run(16384)
    conv_hf = layer.conv_states[0][:, :, -3:].transpose(1, 2)
    p = {
        "out min vs HF": pcc(y_hf, y_min),
        "out padded vs HF": pcc(y_hf, y_pad),
        "out padded vs min": pcc(y_min, y_pad),
        "recurrent min vs HF": pcc(layer.recurrent_states[0], rec_min),
        "recurrent padded vs HF": pcc(layer.recurrent_states[0], rec_pad),
        "recurrent padded vs min": pcc(rec_min, rec_pad),
        "conv min vs HF": pcc(conv_hf, conv_min),
        "conv padded vs HF": pcc(conv_hf, conv_pad),
    }
    logger.info(
        f"SP KDA layer {LAYER}, {n} valid rows in a {16384}-row chunk on {MESH}: "
        + ", ".join(f"{k} {v:.6f}" for k, v in p.items())
        + f"; padded vs min bit-identical: out {torch.equal(y_min, y_pad)} recurrent {torch.equal(rec_min, rec_pad)}"
    )
    assert min(p.values()) > 0.995, p
    assert p["out padded vs min"] > 0.999 and p["recurrent padded vs min"] > 0.999, p

    # The served chunk plans: a prompt continued in chunks whose START is not a multiple of the padded chunk length
    # (vLLM's budget split: [2048, 2176) then [2176, 2348) padded to 256; 2176 % 256 = 128) vs the plan whose starts
    # are multiples of their padded lengths (the adapter's own 1024 + 1024 + 300-in-512), both against HF on the whole
    # prompt: the last chunk's outputs and the carries (the 2026-10-29 budget-split state diff: KDA conv taps and
    # recurrent states wrong after the 2176-start chunk while every DSA row before 2176 matched).
    n2 = 2348
    torch.manual_seed(2)
    x2 = (torch.randn(1, n2, cfg.hidden_size) * 0.8).to(torch.bfloat16)
    y_hf2, layer2 = _hf_run(hf, text, x2)
    conv_hf2 = layer2.conv_states[0][:, :, -3:].transpose(1, 2)

    def run_plan(plan):
        """``plan`` = [(start, end, padded T)]: the chunks in order through one carried state."""
        state = tt.allocate_state(1)
        outs = []
        for s0, e, Tp in plan:
            assert Tp % unit == 0 and e - s0 <= Tp and s0 % 32 == 0, (s0, e, Tp)
            xp = torch.zeros(1, Tp, cfg.hidden_size, dtype=torch.bfloat16)
            xp[:, : e - s0] = x2[:, s0:e]
            x_dev = mesh.upload(xp, ttnn.bfloat16, rows=1)
            y, state = tt.prefill_forward(x_dev, state, tt.scalar(s0), tt.scalar(e))
            outs.append(mesh.to_host(y, rows=1).float()[:, : e - s0])
        return torch.cat(outs, dim=1), _carries(mesh, state, cfg)[0]

    plans = {
        "aligned starts 1024+1024+300/512": [(0, 1024, 1024), (1024, 2048, 1024), (2048, 2348, 512)],
        "budget split 1024+1024+128/256+172/256": [
            (0, 1024, 1024),
            (1024, 2048, 1024),
            (2048, 2176, 256),
            (2176, 2348, 256),
        ],
        "128 minimal then 172/256 from 2176": [
            (0, 1024, 1024),
            (1024, 2048, 1024),
            (2048, 2176, 128),
            (2176, 2348, 256),
        ],
        "one chunk to 2176 then 172/512": [(0, 2176, 2176), (2176, 2348, 512)],
    }
    results = {}
    for name, plan in plans.items():
        y_p, (rec_p, conv_p) = run_plan(plan)
        results[name] = {
            "last-chunk out vs HF": pcc(y_hf2[:, 2176:], y_p[:, 2176:]),
            "out vs HF": pcc(y_hf2, y_p),
            "recurrent vs HF": pcc(layer2.recurrent_states[0], rec_p),
            "conv vs HF": pcc(conv_hf2, conv_p),
        }
        logger.info(f"SP KDA layer {LAYER} plan {name}: " + ", ".join(f"{k} {v:.6f}" for k, v in results[name].items()))
    for name, r in results.items():
        assert min(r.values()) > 0.995, f"plan {name}: {r}"
