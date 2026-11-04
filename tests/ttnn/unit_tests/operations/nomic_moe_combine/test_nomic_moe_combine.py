# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for nomic_moe_combine (immutable spec; see ttnn/cpp/ttnn/operations/experimental/nomic_embed_text_v2_moe/nomic_moe_combine/op_design.md).

out[t, :] = round_bf16(bias + w[0, t] * rows[index[0, t], :] + w[1, t] * rows[index[1, t], :]), fp32 accumulation, one
rounding, the bias added once. Phase 0 cell: expert_rows in L1, the output in the layer norm's ND-sharded layout
(nomic_residual_layer_norm.input_memory_config), aligned and ragged windows, B == 1 and B > 1.

The routing is routing_metadata's host twin (helpers.route); every expert row no token points at is NaN, so a row the
op should not read poisons the output. Tolerances are the golden suite's (helpers.TOLERANCES): the window is not a
generic eltwise, the bar is the one rounding of the incumbent.

Shapes: the op's domain starts at T = 1000 tokens (32 tile rows), so there is no single-tile case; the parametrization
covers the smallest window (ragged, single batch), multi-batch 1k / 2k / 4k windows (the perf shapes), a non-square
multi-batch window, and a ragged 4000-token window whose last norm shard is partial.
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from eval.golden_tests.nomic_moe_combine.helpers import (
    TOLERANCES,
    check_values,
    padded,
    route,
    rows_for_tokens,
)
from ttnn.operations._op_contract import SupportRefusal
from ttnn.operations.nomic_moe_combine import nomic_moe_combine, prepare_combine_bias
from ttnn.operations.nomic_residual_layer_norm import input_memory_config as norm_layout

HIDDEN = 768
TILE = 32

# PCC / relative-RMS bars keyed by dtype: the golden suite's (calibrated on the incumbent's one rounding).
TOLERANCE_BY_DTYPE = {ttnn.bfloat16: TOLERANCES[ttnn.bfloat16]}

SHAPES = [
    (1, 1000),  # T = 1000, R = 2240: smallest window, ragged (8 padding tokens), single batch
    (2, 512),  # T = 1024, R = 2272: the 1k perf shape, multi-batch
    (8, 256),  # T = 2048: non-square multi-batch window
    (4, 512),  # T = 2048, R = 4320: the 2k perf shape
    (8, 512),  # T = 4096, R = 8416: the 4k perf shape (perf focus)
    (1, 4000),  # T = 4000, R = 8224: ragged, the last norm shard holds fewer tile rows
]


def _ids(shape):
    return f"{shape[0]}x{shape[1]}"


def _inputs(batch, seqlen, device, one_hot=False):
    torch.manual_seed(42)
    tokens = batch * seqlen
    plan = route(tokens, "real_text", seed=0)
    rows = rows_for_tokens(tokens)
    values = (0.5 * torch.randn(rows, HIDDEN)).to(torch.bfloat16)
    values[~plan["slot_row"]] = float("nan")  # rows no token's choice points at: never read
    bias = (0.5 * torch.randn(1, 1, 1, HIDDEN)).to(torch.bfloat16)
    weights = plan["combine_weights"].clone()
    if one_hot:
        pick = torch.randint(0, 2, (padded(tokens),))
        weights = torch.zeros(2, padded(tokens), dtype=torch.float32)
        weights[pick, torch.arange(padded(tokens))] = 1.0
        weights[:, tokens:] = 0.0
    index = plan["combine_index"]

    tt_rows = ttnn.from_torch(
        values.reshape(1, 1, rows, HIDDEN),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    tt_index = ttnn.from_torch(
        index.to(torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_weights = ttnn.from_torch(
        weights.reshape(1, 2, -1, 1),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_bias = ttnn.from_torch(
        bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    return dict(
        values=values,
        bias=bias,
        weights=weights,
        index=index,
        tokens=tokens,
        tt=(tt_rows, tt_index, tt_weights, tt_bias),
    )


def torch_reference(values, bias, index, weights, tokens):
    """(T, 768) float64: bias + sum over k of w[k, t] * row[index[k, t]]."""
    rows = values.reshape(-1, HIDDEN).double()
    out = bias.reshape(1, HIDDEN).double().repeat(tokens, 1)
    for k in range(2):
        out += weights[k, :tokens].double().unsqueeze(1) * rows[index[k, :tokens]]
    return out


def _run(batch, seqlen, device, one_hot=False):
    data = _inputs(batch, seqlen, device, one_hot)
    tt_rows, tt_index, tt_weights, tt_bias = data["tt"]
    shape = (batch, 1, seqlen, HIDDEN)
    config = norm_layout(shape, device)
    prepared = prepare_combine_bias(tt_bias, device)
    output = nomic_moe_combine(
        tt_rows, tt_index, tt_weights, prepared, num_tokens=data["tokens"], batch=batch, memory_config=config
    )
    assert tuple(output.shape) == shape
    assert output.dtype == ttnn.bfloat16 and output.layout == ttnn.TILE_LAYOUT
    assert output.memory_config() == config
    return data, output


def _check_padding(output, bias, seqlen):
    if seqlen % TILE == 0:
        return
    pad = output.cpu().to_torch_with_padded_shape()[..., seqlen:, :HIDDEN].reshape(-1, HIDDEN)
    want = bias.reshape(1, HIDDEN).expand_as(pad).contiguous()
    assert bool((pad.view(torch.int16) == want.view(torch.int16)).all()), "padding rows are not exactly the bias"


@pytest.mark.parametrize("batch,seqlen", SHAPES, ids=[_ids(s) for s in SHAPES])
def test_nomic_moe_combine(batch, seqlen, device):
    data, output = _run(batch, seqlen, device)
    expected = torch_reference(data["values"], data["bias"], data["index"], data["weights"], data["tokens"])
    check_values(output, expected, data["tokens"], TOLERANCE_BY_DTYPE[ttnn.bfloat16])
    _check_padding(output, data["bias"], seqlen)


@pytest.mark.parametrize("batch,seqlen", [(1, 1000), (2, 512), (8, 512)], ids=["1x1000", "2x512", "8x512"])
def test_one_hot_weights_are_row_plus_bias_rounded_once(batch, seqlen, device):
    """One-hot weights: each value is its chosen row + the bias, rounded to bf16 once (ties to even or away)."""
    data, output = _run(batch, seqlen, device, one_hot=True)
    tokens = data["tokens"]
    got = ttnn.to_torch(output).reshape(-1, HIDDEN)[:tokens]
    pick = data["weights"][:, :tokens].argmax(dim=0)
    rows = data["values"].reshape(-1, HIDDEN)[data["index"][pick, torch.arange(tokens)]]
    exact = rows.float() + data["bias"].reshape(1, HIDDEN).float()
    even = exact.to(torch.bfloat16)
    away = ((exact.contiguous().view(torch.int32) + 0x8000) & ~0xFFFF).view(torch.float32).to(torch.bfloat16)
    g = got.contiguous().view(torch.int16)
    ok = (g == even.view(torch.int16)) | (g == away.view(torch.int16))
    assert bool(ok.all()), f"{int((~ok).sum())} values are not row + bias rounded once"
    _check_padding(output, data["bias"], seqlen)


def test_bias_is_added_once(device):
    """The projection of (out - weighted rows) on the bias is 1 (0 if dropped, sum(w) if inside the k loop)."""
    batch, seqlen = 4, 512
    data, output = _run(batch, seqlen, device)
    tokens = data["tokens"]
    got = ttnn.to_torch(output).reshape(-1, HIDDEN)[:tokens].double()
    weighted = torch_reference(data["values"], torch.zeros_like(data["bias"]), data["index"], data["weights"], tokens)
    b = data["bias"].reshape(HIDDEN).double()
    multiplicity = float(((got - weighted) @ b).sum() / (tokens * (b @ b)))
    assert abs(multiplicity - 1.0) < 0.02, f"bias in the output {multiplicity:.4f} times"


def test_window_below_the_domain_is_refused(device, expect_error):
    """T < 1000 is outside the op's domain: SupportRefusal before any device work."""
    tokens = 992
    data = _inputs(1, tokens, device)
    tt_rows, tt_index, tt_weights, tt_bias = data["tt"]
    prepared = prepare_combine_bias(tt_bias, device)
    with expect_error(SupportRefusal, "nomic_moe_combine"):
        nomic_moe_combine(
            tt_rows, tt_index, tt_weights, prepared, num_tokens=tokens, batch=1, memory_config=ttnn.L1_MEMORY_CONFIG
        )


def test_cached_descriptor_is_rebound_every_call(device):
    """Refinement 4: one descriptor per call-state key, built once; every call rebinds it to that call's tensors (the
    readers' address args and the output shard's CB), so a cached descriptor never carries a previous call's
    addresses. The second call's tensors sit at new addresses (the first call's are still allocated)."""
    import importlib

    op = importlib.import_module("ttnn.operations.nomic_moe_combine.nomic_moe_combine")
    batch, seqlen = 2, 512
    descriptors, outputs = [], []
    for call in range(2):
        data, output = _run(batch, seqlen, device)
        tt_rows, tt_index, tt_weights, _ = data["tt"]
        # Key: (T, B, rows shape, dtype, rows buffer, memory config, compute config, device id).
        states = [s for k, s in op._CALL_CACHE.items() if k[:2] == (data["tokens"], batch) and k[-1] == device.id()]
        assert len(states) == 1, "one call state per key"
        descriptor = states[0].descriptor
        descriptors.append(descriptor)
        outputs.append((data, output))
        for slot in range(2):  # both readers' common args 0..3 are this call's input addresses
            common = list(descriptor.kernels[slot].common_runtime_args)
            assert common[:3] == [t.buffer_address() for t in (tt_rows, tt_index, tt_weights)]
        assert descriptor.cbs[0].buffer_address() == output.buffer_address()
        tokens = data["tokens"]
        expected = torch_reference(data["values"], data["bias"], data["index"], data["weights"], tokens)
        check_values(output, expected.float(), tokens, TOLERANCE_BY_DTYPE[ttnn.bfloat16])
    assert descriptors[0] is descriptors[1], "the second call reuses the cached descriptor"
    assert outputs[0][1].buffer_address() != outputs[1][1].buffer_address()
