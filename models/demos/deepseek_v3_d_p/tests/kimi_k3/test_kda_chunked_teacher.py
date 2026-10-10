# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Kimi-K3's KDA layers alone on TT, chunk by chunk over the whole 1M-token prompt, fed vLLM's recorded inputs.

`test_chunked_prefill.py` runs whole layers from token IDs, so a KDA layer's input already carries every error
made upstream of it, and at long context a drifting KDA output cannot be told apart from a drifting input. This
takes the upstream away. For each of `model.layers.0-2`, the KDA module input vLLM recorded
(`kda_input_layer_i`, post-`input_layernorm`) goes straight into `ttKDA`, one 5120-token chunk at a time in
prompt order, the way chunked prefill runs it:

  * `actual_start` is the chunk's absolute position. Chunks start on multiples of SP * local rows, so the
    MLA block-cyclic row layout is the natural order and the chunk is uploaded as is;
  * the carries live in a `KdaStateCache`, zeroed once at token 0 and advanced by `commit` after every chunk. They
    are never reset and never replaced by the golden's state, so each layer carries its own TT state across all
    204 chunks, and nothing upstream of the layer runs.

Per chunk and layer, the output is scored against `kda_output_layer_i` over the chunk, and the carries against the
trace's snapshot after it (recurrent `[heads, v, k]`, transposed to the layer's `[heads, k, v]`; conv
`[3, all q | all k | all v]`).

Which trace matters. The default, `k3_4_layers_moe_kda_io`, was captured with vLLM's FlashKDA prefill kernel,
which stores the recurrent state in bf16 every 16 tokens: exact fp32 math fed the same inputs drifts from it to
PCC 0.934 on `model.layers.1` by chunk 203, so past layer 0 it is not an exact target. The recapture with vLLM's
Triton KDA kernel (fp32 state, `kda_prefill_backend="triton"`) follows exact math, PCC >= 0.99997 on every chunk
of all three layers; point `$KIMI_K3_GOLDEN_TRACE` at it to measure TT against exact KDA.

Only the setup gates the test: chunk 0 (zero carry, exact input) must match vLLM, every score must be finite, and
device DRAM must not grow across chunks. The long-context curves are the result, one JSON line per chunk.

    KIMI_K3_GOLDEN_TRACE=<1M Triton KDA trace> K3_KDA_PCC_DUMP=kda_tt.jsonl \\
        pytest models/demos/deepseek_v3_d_p/tests/kimi_k3/test_kda_chunked_teacher.py -s
"""

import json
import math
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import kimi_k3_kda_config
from models.demos.deepseek_v3_d_p.tests.kda.checkpoint_utils import load_kda_layer_state_dict
from models.demos.deepseek_v3_d_p.tests.kda.utils import (
    mla_row_permutation,
    reconstruct_convolution_at_sp_rank,
    reconstruct_sp_tp_tensor,
    reconstruct_state_at_sp_rank,
    to_sp_input,
)
from models.demos.deepseek_v3_d_p.tests.kimi_k3.golden import GOLDEN_ROOT, resolve_checkpoint, resolve_trace
from models.demos.deepseek_v3_d_p.tests.kimi_k3.test_chunked_prefill import _dram_bytes
from models.demos.deepseek_v3_d_p.tests.kimi_k3.test_transformer_depth import PLACEMENTS, SP_AXIS, TP_AXIS
from models.demos.deepseek_v3_d_p.tt.kda.config import kimi_k3_program_config
from models.demos.deepseek_v3_d_p.tt.kda.kda import ttKDA
from models.demos.deepseek_v3_d_p.tt.kimi_k3.kda_state import KdaStateCache
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl, per_axis_topology
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import make_actual_start

TRACE_4L = GOLDEN_ROOT / "structured_traces" / "k3_4_layers_moe_kda_io"
CHUNK = 5120
# The 1M KDA traces hold 1,044,480 tokens: 204 full chunks, with a state snapshot after each.
NUM_CHUNKS = int(os.environ.get("K3_KDA_NUM_CHUNKS", "204"))
# Model-layer indices, 0-based as in `model.layers.N` and the trace's `*_layer_N` streams.
LAYERS = [int(x) for x in os.environ.get("K3_KDA_LAYERS", "0,1,2").split(",")]
PCC_DUMP = os.environ.get("K3_KDA_PCC_DUMP")

# Chunk 0 starts from the zero carry on vLLM's exact input, so only arithmetic separates TT from vLLM there. The
# whole-model run measured 0.99992 / 0.99976 / 0.99949 for layers 0-2 at chunk 0, with inputs already off.
FIRST_CHUNK_PCC = 0.999


def _metrics(got, want):
    """PCC, relative L2 error and max abs error, in float64: at 36M elements a float32 PCC saturates."""
    g, w = got.double().flatten(), want.double().flatten()
    gc, wc = g - g.mean(), w - w.mean()
    return {
        "pcc": float(gc @ wc / (gc.norm() * wc.norm())),
        "rel_l2": float((g - w).norm() / w.norm()),
        "max_abs": float((g - w).abs().max()),
    }


def _dump(record):
    if PCC_DUMP:
        with open(PCC_DUMP, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


@pytest.mark.timeout(int(os.environ.get("K3_KDA_TIMEOUT_S", "14400")))
@pytest.mark.parametrize("mesh_device, device_params", PLACEMENTS, indirect=True)
def test_kda_chunked_teacher_forced(mesh_device, device_params):
    checkpoint = resolve_checkpoint()
    trace = resolve_trace(TRACE_4L)
    if checkpoint is None or trace is None:
        pytest.skip("needs KIMI_K3_CKPT (or KIMI_K3_HF_MODEL) and a 1M KDA trace")
    for idx in LAYERS:
        for stream in ("input", "output", "recurrent_state", "conv_state"):
            if not trace.has("kda", f"kda_{stream}_layer_{idx}"):
                pytest.fail(f"{trace.path.name} has no kda/kda_{stream}_layer_{idx}")
    if trace.metadata.get("kda_state_every") != CHUNK:
        pytest.fail(f"state snapshots every {trace.metadata.get('kda_state_every')} tokens, not every {CHUNK}")

    config = kimi_k3_kda_config()
    mesh_shape = tuple(mesh_device.shape)
    sp_size, tp_size = mesh_shape[SP_AXIS], mesh_shape[TP_AXIS]
    local_rows = CHUNK // sp_size
    conv_width = config.q_dim // tp_size
    # One program for every layer and every chunk: the same geometry the model builds in `build_attention`.
    program_config = kimi_k3_program_config(
        active_seq_len_local=local_rows, tp_ccl_topology=per_axis_topology()[TP_AXIS]
    )
    tt_ccl = get_tt_ccl(mesh_device)
    layers = {
        idx: ttKDA(
            mesh_device,
            config,
            load_kda_layer_state_dict(Path(checkpoint), idx, config),
            layer_idx=idx,
            weight_cache_path=None,
            tt_ccl=tt_ccl,
            sp_axis=SP_AXIS,
            tp_axis=TP_AXIS,
            program_config=program_config,
            active_seq_len=CHUNK,
        )
        for idx in LAYERS
    }
    # Zeroed carries: the head of the prompt, and the only reset.
    states = KdaStateCache(layers)
    logger.info(
        f"KDA teacher-forced chunked prefill against {trace.path.name}: layers {LAYERS}, "
        f"{NUM_CHUNKS} x {CHUNK} tokens, SP{sp_size} x TP{tp_size}"
    )

    history = {idx: {} for idx in LAYERS}
    footprints = []
    failures = []
    for chunk in range(NUM_CHUNKS):
        t0 = time.time()
        start, end = chunk * CHUNK, (chunk + 1) * CHUNK
        permutation = mla_row_permutation(start, sp_size, local_rows)
        assert torch.equal(permutation, torch.arange(CHUNK)), f"chunk {chunk} does not start on an SP slab boundary"
        actual_start = make_actual_start(mesh_device, start)
        module_pcc, state_pcc, metrics = {}, {}, {}
        for idx, kda in layers.items():
            hidden = trace.rows("kda", f"kda_input_layer_{idx}", start, end)
            hidden_tt = to_sp_input(hidden.reshape(1, CHUNK, config.hidden_size), mesh_device, SP_AXIS)
            with ttnn.manage_config("throw_exception_on_fallback", True):
                output_tt, new_state = kda.forward(hidden_tt, states.read(idx), actual_start)
            ttnn.deallocate(hidden_tt)
            states.commit(idx, new_state)

            output = reconstruct_sp_tp_tensor(output_tt, mesh_device, SP_AXIS, TP_AXIS, tp_dim=2, sp_dim=1)[0].float()
            ttnn.deallocate(output_tt)
            # The carries are SP-replicated; SP rank 0 holds the whole of each.
            carry = states.read(idx)
            recurrent = reconstruct_state_at_sp_rank(carry.recurrent, mesh_device, SP_AXIS, TP_AXIS, 0)[0].float()
            conv = reconstruct_convolution_at_sp_rank(
                carry.convolution, mesh_device, SP_AXIS, TP_AXIS, 0, local_width=conv_width
            )[0].float()

            # Snapshot row c is the state after (c + 1) * 5120 tokens; the golden's carry is [heads, v, k].
            scores = {
                "attn_out": _metrics(output, trace.rows("kda", f"kda_output_layer_{idx}", start, end)),
                "recurrent": _metrics(
                    recurrent,
                    trace.rows("kda", f"kda_recurrent_state_layer_{idx}", chunk, chunk + 1)[0].transpose(-1, -2),
                ),
                "conv": _metrics(conv, trace.rows("kda", f"kda_conv_state_layer_{idx}", chunk, chunk + 1)[0]),
            }
            key = str(idx)
            module_pcc[key] = {"attn_out": scores["attn_out"]["pcc"]}
            state_pcc[key] = {"recurrent": scores["recurrent"]["pcc"], "conv": scores["conv"]["pcc"]}
            metrics[key] = scores
            history[idx][chunk] = scores

            bad = [name for name, m in scores.items() if not all(math.isfinite(v) for v in m.values())]
            if bad:
                failures.append(f"layer {idx} chunk {chunk}: non-finite {bad}")
            if chunk == 0 and scores["attn_out"]["pcc"] < FIRST_CHUNK_PCC:
                failures.append(f"layer {idx} chunk 0 output PCC {scores['attn_out']['pcc']:.6f} < {FIRST_CHUNK_PCC}")
        ttnn.deallocate(actual_start)

        footprints.append(_dram_bytes(mesh_device))
        _dump(
            {
                "kind": "chunk",
                "source": "tt",
                "mode": "teacher",
                "trace": trace.path.name,
                "num_layers": len(LAYERS),
                "chunk": chunk,
                "start": start,
                "end": end,
                "module_pcc": module_pcc,
                "state_pcc": state_pcc,
                "metrics": metrics,
                "dram_bytes": footprints[-1],
                "seconds": round(time.time() - t0, 2),
            }
        )
        logger.info(
            f"  chunk {chunk:3d} [{start:7d}:{end:7d}]  "
            + "  ".join(
                f"L{i} out={m['attn_out']['pcc']:.5f} rec={m['recurrent']['pcc']:.5f} conv={m['conv']['pcc']:.5f}"
                for i, m in ((i, metrics[str(i)]) for i in LAYERS)
            )
            + f"  ({time.time() - t0:.1f}s)"
        )
        if failures and chunk == 0:
            break  # a setup error; the remaining chunks would only repeat it

    sampled = sorted(
        {0, NUM_CHUNKS // 4, NUM_CHUNKS // 2, 3 * NUM_CHUNKS // 4, NUM_CHUNKS - 1} & set(history[LAYERS[0]])
    )
    logger.info("  by context length (chunk: output / recurrent state PCC):")
    for idx in LAYERS:
        cells = [
            f"{c}: {history[idx][c]['attn_out']['pcc']:.4f} / {history[idx][c]['recurrent']['pcc']:.4f}"
            for c in sampled
        ]
        logger.info(f"    model.layers.{idx}  " + "   ".join(cells))

    states.deallocate()
    # The carries are the one allocation surface that lives across chunks; after chunk 0 warms the pools the
    # footprint has to be flat, or a carry or its replacement is leaking.
    steady = footprints[1:]
    if steady:
        growth = max(steady) - min(steady)
        assert growth == 0, f"device DRAM grew {growth} bytes across chunks 1..{len(footprints) - 1}: {footprints}"
    assert not failures, "; ".join(failures)
