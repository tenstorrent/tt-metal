# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""MTP v2 step 4: the GDN and PLE multi-row paths against their 1-row paths, on a torch-backed ttnn (no device).

Rows computed in the fake do not depend on how many rows share a tensor, so the multi-row paths can be compared
with sequential 1-row steps; the chunk kernel model is pluggable. R = 5 is the verify pass, R = 32 the prefill chunk
(forward_rows at the full tile plus commit_rows with every row accepted: a = 31, r - 1 for a padded last chunk).
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from models.demos.blackhole.qwen38_flash_next.tests.helpers.mtp_verify import row_serial_torch
from models.demos.blackhole.qwen38_flash_next.tests.helpers.torch_ttnn import (
    BF16,
    FP32,
    GDN_HEAD_DIM,
    GDN_HEADS,
    ROW_MAJOR,
    TP,
    FakeTensor,
    bf16_randn,
    bits_equal,
    cat_shards,
    clone_gdn_state,
    gdn_output_tolerance,
    gdn_ring_history,
    gdn_rows_selectors,
    hidden_sharded,
    make_gdn_module,
    make_ple_module,
    ordered_calls,
    residual_rows,
    run_gdn_steps,
    seed_gdn_state,
    seed_ple_state,
    source_methods,
    source_segment,
    view_tile_pages,
)
from models.demos.blackhole.qwen38_flash_next.tt.gdn import Qwen38GDN, Qwen38GDNState
from models.demos.blackhole.qwen38_flash_next.ttnn import gdn as gdn_module
from models.demos.blackhole.qwen38_flash_next.ttnn import mtp_v2
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import CHUNK_ROWS
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gdn_rows_wrap

ROOT = Path(__file__).resolve().parents[1]
GDN_SOURCE = ROOT / "ttnn" / "gdn.py"
PLE_SOURCE = ROOT / "ttnn" / "ple.py"
LAYER_SOURCE = ROOT / "ttnn" / "layer.py"
ROWS = 5
HISTORY_PAD = [(0, 0), (0, 0), (0, 32 - 3), (0, 0)]  # the history sync pads its 3-row tile
ROWS_PAD = [(0, 0), (0, 0), (0, 32 - ROWS), (0, 0)]


def step_chunk_kernel(q, k, v, g, beta, scale, state):
    """Row 0 through the 1-row path's fp32 step arithmetic (the same torch calls the fake step ops make); rows
    past 0 are zero.  Used to show that everything around the recurrence is bitwise the 1-row path."""

    heads, head_dim = GDN_HEADS, GDN_HEAD_DIM
    q_row = q[:, 0].repeat_interleave(heads // q.shape[2], dim=1).reshape(1, heads, 1, head_dim).float() * scale
    k_row = k[:, 0].repeat_interleave(heads // k.shape[2], dim=1).reshape(1, heads, 1, head_dim).float()
    v_row = v[:, 0].reshape(1, heads, 1, head_dim)
    beta_row = beta[:, 0].reshape(1, heads, 1, 1)
    g_row = g[:, 0].reshape(1, heads, 1, 1)
    decayed = state * torch.exp(g_row)
    v_read = torch.matmul(k_row, decayed)
    delta = v_row.to(torch.float32) - v_read
    outer = torch.matmul(k_row.transpose(2, 3), delta)
    new_state = decayed + outer * beta_row
    o_row = torch.matmul(q_row, new_state)  # [1, HEADS, 1, HEAD_DIM] fp32; the rows path casts to bf16 as 1-row does
    output = torch.zeros(1, q.shape[1], heads, head_dim, dtype=torch.float32)
    output[:, 0] = o_row[:, :, 0]
    return output, new_state


def _rows_run(module, state, hidden: torch.Tensor, constants, **forward):
    rows_state = module.allocate_rows_state(constants)
    module.sync_rows_history_from_state(state, rows_state)
    return module.forward_rows(hidden_sharded(hidden), state, rows_state, **forward), rows_state


def _record_pads(rows_fake) -> list:
    pads = []
    real_pad = rows_fake.ttnn.pad
    rows_fake.ttnn.pad = lambda *args, **kwargs: pads.append(args[1]) or real_pad(*args, **kwargs)
    return pads


# --------------------------------------------------------------------------- GDN under the fake


def test_softplus_gate_never_flushes_to_zero_and_tracks_torch_softplus(rows_fake) -> None:
    """The decay gate over a + dt_bias in [-8, 0]: no exact 0, within 1e-3 of torch softplus, FP32 out."""

    xs = torch.linspace(-8.0, 0.0, 4 * 32, dtype=torch.float32).reshape(1, 1, 4, 32)
    a_fp32 = FakeTensor([xs.clone() for _ in range(TP)], FP32)
    dt_bias = FakeTensor([torch.zeros(1, 1, 1, 32) for _ in range(TP)], FP32)
    gate = gdn_module.softplus_gate(a_fp32, dt_bias, memory_config="L1")
    assert gate.dtype is FP32 and gate.shape == (1, 1, 4, 32)
    for local in gate.torch_shards():
        assert bool((local > 0.0).all()), float(local.min())
        assert float((local - F.softplus(xs)).abs().max()) <= 1e-3
    assert a_fp32.alive and dt_bias.alive  # the helper frees only its own intermediate (the add's output)


def test_forward_rows_full_tile_output_and_padded_input_tile_are_the_row_slice_bitwise(rows_fake, gdn_weights) -> None:
    """``full_tile=True`` returns the whole 32-row output (a fresh buffer): rows below R are the persistent slice's,
    rows past R are exact zeros (zero q rows -> zero recurrent output -> zero norm, gate and projection). A caller
    that hands the zero-padded 32-row input tile gets the same rows without ``ttnn.pad`` (the verify body's
    persistent input tile)."""

    rows_fake.chunk.impl = step_chunk_kernel
    module = make_gdn_module(gdn_weights[1])
    torch.manual_seed(17)
    hidden = torch.randn(1, ROWS, 2560).to(torch.bfloat16)
    state_slice = seed_gdn_state(module, 13)
    state_tile, state_padded = clone_gdn_state(module, state_slice), clone_gdn_state(module, state_slice)
    constants = module.allocate_rows_constants(ROWS)
    pads = _record_pads(rows_fake)
    sliced, rows_slice = _rows_run(module, state_slice, hidden, constants)
    assert pads == [HISTORY_PAD, ROWS_PAD]
    tile, rows_tile = _rows_run(module, state_tile, hidden, constants, full_tile=True)
    output = cat_shards(tile.hidden_rows, 3)
    assert output.shape == (1, 1, CHUNK_ROWS, 2560) and tile.hidden_rows is not rows_tile.output
    assert torch.equal(output[:, :, :ROWS], cat_shards(sliced.hidden_rows, 3))
    assert torch.count_nonzero(output[:, :, ROWS:]) == 0
    assert torch.equal(cat_shards(tile.final_state, 1), cat_shards(sliced.final_state, 1))
    padded_input = F.pad(hidden, (0, 0, 0, 32 - ROWS))
    assert hidden_sharded(padded_input).shape == (1, 1, 32, 640)
    pads.clear()
    padded, rows_padded = _rows_run(module, state_padded, padded_input, constants)
    assert pads == [HISTORY_PAD]  # no input pad
    assert torch.equal(cat_shards(padded.hidden_rows, 3), cat_shards(sliced.hidden_rows, 3))
    assert torch.equal(cat_shards(padded.final_state, 1), cat_shards(sliced.final_state, 1))
    for name, shard_dim in (("q", 2), ("k", 2), ("v", 3), ("beta", 3), ("g", 3), ("qkv", 3)):
        assert torch.equal(
            cat_shards(getattr(rows_padded, name), shard_dim), cat_shards(getattr(rows_slice, name), shard_dim)
        ), name
    # The verify layer body takes the tile (no slice landing, no pad) and owns it.
    source = inspect.getsource(mtp_v2._forward_layer_verify)
    assert "full_tile=True" in source and 'label="GDN verify output rows"' not in source
    assert "attention_owner = attention_hidden" in source


def test_forward_rows_at_one_row_is_bitwise_the_one_row_path_around_the_recurrence(rows_fake, gdn_weights) -> None:
    rows_fake.chunk.impl = step_chunk_kernel
    module = make_gdn_module(gdn_weights[1])
    torch.manual_seed(7)
    hidden = torch.randn(1, 1, 2560).to(torch.bfloat16)
    state_step = seed_gdn_state(module, 11)
    state_rows = clone_gdn_state(module, state_step)
    constants = module.allocate_rows_constants(1)

    initial = cat_shards(state_step.recurrent, 1)
    step_output = run_gdn_steps(module, state_step, hidden)[0]
    result, rows_state = _rows_run(module, state_rows, hidden, constants)

    assert result.hidden_rows.shape == (1, 1, 1, 640)
    assert torch.equal(cat_shards(result.hidden_rows, 3), step_output)
    assert torch.equal(cat_shards(result.final_state, 1), cat_shards(state_step.recurrent, 1))
    # The committed state was not written by the pass; the ring and its phase were not touched.
    assert torch.equal(cat_shards(state_rows.recurrent, 1), initial) and state_rows.conv_phase == 0
    call = rows_fake.chunk.calls[-1]
    assert call["chunk_size"] == 32 and call["rows"] == 32
    # The kernel saw the GQA-expanded q/k (H = HV = 12, no repeat inside the composite), v token-major flat
    # (no head split) and was asked for its head-major output (no untilize / row-major permute round trip).
    assert call["heads"] == (GDN_HEADS, GDN_HEADS, GDN_HEADS) and call["output_head_major"] and call["flat_v"]
    # Rows past the real one reached the kernel as zeros: k = v = 0, beta = 0, g = 0.
    for name, rows_axis, shard_dim in (("q", 1, 2), ("k", 1, 2), ("v", 2, 3), ("beta", 2, 3), ("g", 2, 3)):
        tensor = cat_shards(getattr(rows_state, name), shard_dim)
        assert torch.count_nonzero(tensor.narrow(rows_axis, 1, 31)) == 0, name


@pytest.mark.parametrize("rows,seed,state_seed", ((ROWS, 8, 12), (CHUNK_ROWS, 31, 32)), ids=("verify", "chunk"))
def test_forward_rows_matches_the_sequential_steps_and_the_cpu_oracle(
    rows_fake, gdn_weights, rows: int, seed: int, state_seed: int
) -> None:
    oracle, weights = gdn_weights
    module = make_gdn_module(weights)
    torch.manual_seed(seed)
    hidden = torch.randn(1, rows, 2560).to(torch.bfloat16)
    state_step = seed_gdn_state(module, state_seed)
    state_rows = clone_gdn_state(module, state_step)
    initial = cat_shards(state_rows.recurrent, 1)
    # The CPU oracle: device d holds q/k heads 4d..4d+3 and v heads 12d..12d+11 as its local [q|k|v] columns.
    stacked = torch.stack([cat_shards(slot, 3).reshape(TP, 2560) for slot in state_rows.conv_window()[:3]], dim=-1)
    q, k, v = (
        stacked[:, :512].reshape(2048, 3),
        stacked[:, 512:1024].reshape(2048, 3),
        stacked[:, 1024:].reshape(6144, 3),
    )
    with row_serial_torch():
        oracle_output, oracle_state = Qwen38GDN(oracle).forward(
            hidden, Qwen38GDNState(conv=torch.cat([q, k, v], dim=0).reshape(1, 10240, 3), recurrent=initial)
        )
    constants = module.allocate_rows_constants(rows)
    assert torch.count_nonzero(constants.row_mask_fp32.torch_shards()[0]) == rows
    pads = _record_pads(rows_fake)

    step_outputs = torch.cat(run_gdn_steps(module, state_step, hidden), dim=2)
    result, rows_state = _rows_run(module, state_rows, hidden, constants)
    rows_output = cat_shards(result.hidden_rows, 3)
    final_state = cat_shards(result.final_state, 1)
    step_state = cat_shards(state_step.recurrent, 1)

    # A full tile's forward pass pads nothing.
    assert pads == ([HISTORY_PAD] if rows == CHUNK_ROWS else [HISTORY_PAD, ROWS_PAD])
    assert rows_output.shape == (1, 1, rows, 2560) and result.hidden_rows is rows_state.output
    out_err = (rows_output.float() - step_outputs.float()).abs().max().item()
    state_err = (final_state - step_state).abs().max().item()
    print(f"{rows} rows vs steps: output max abs {out_err:.2e}; state max abs {state_err:.2e}")
    # The kernel scales q in bf16 (a 2**-9 relative rounding the fp32 step does not have) and the layer output
    # is bf16: four bf16 ulps of the output scale on top of the design's 2e-4 recurrence budget.
    assert out_err <= gdn_output_tolerance(step_outputs)
    torch.testing.assert_close(final_state, step_state, rtol=1e-4, atol=1e-5)  # step-2 model: rel <= 1.2e-5
    # The CPU oracle normalizes q/k with its own bf16 rounding (a 1-ulp flip in k moves the state by 4e-3), so
    # this is a wiring check (head order, conv mapping, gate/output path), not a precision gate.
    oracle_err = (rows_output.float().reshape(rows, 2560) - oracle_output.float().reshape(rows, 2560)).abs().max()
    assert oracle_err.item() <= 0.1 * oracle_output.float().abs().max().item()
    assert (final_state - oracle_state.recurrent).abs().max().item() <= 0.1 * oracle_state.recurrent.abs().max().item()
    # The committed state is untouched by the forward pass.
    assert torch.equal(cat_shards(state_rows.recurrent, 1), initial)
    assert rows_fake.chunk.calls[-1]["rows"] == 32 and rows_fake.chunk.calls[-1]["chunk_size"] == 32


COMMIT_FORMS = ("chunk", "step_on_full_rejection", "step_committed_rows")
COMMIT_CASES = (
    *((ROWS, form, accepted) for form in COMMIT_FORMS for accepted in range(ROWS)),
    *((CHUNK_ROWS, "chunk", accepted) for accepted in (31, 0, 1, 2, 8)),
    *((CHUNK_ROWS, "step_committed_rows", accepted) for accepted in (31, 4)),
)


@pytest.mark.parametrize("rows,form,accepted", COMMIT_CASES)
def test_commit_rows_lands_the_committed_prefix_state_and_history(
    rows_fake, gdn_weights, rows: int, form: str, accepted: int
) -> None:
    step_form, anchor_form = form == "step_on_full_rejection", form == "step_committed_rows"
    module = make_gdn_module(gdn_weights[1])
    torch.manual_seed(9)
    hidden = torch.randn(1, rows, 2560).to(torch.bfloat16)
    state_step = seed_gdn_state(module, 13)
    state_rows = clone_gdn_state(module, state_step)
    constants = module.allocate_rows_constants(rows)

    run_gdn_steps(module, state_step, hidden[:, : accepted + 1])  # the committed prefix, one step at a time
    result, rows_state = _rows_run(module, state_rows, hidden, constants)
    selectors = gdn_rows_selectors(accepted, constants)
    onehots = [float(t.torch_shards()[0].reshape(-1)[0]) for t in selectors.onehot_bf16]
    assert onehots == [float(j == accepted) for j in range(rows)]
    assert selectors.committed_mask.torch_shards()[0].reshape(-1).tolist() == [float(j <= accepted) for j in range(32)]
    # The per-pass history select is the stack row of this accept count: rows 0..2 pick logical window rows
    # accepted + 1 .. accepted + 3 (history rows 0..2 sit at buffer rows 0..2, new row r at 32 + r).
    select = selectors.history_select.torch_shards()[0].float().reshape(32, 64)
    assert selectors.history_select.dtype is BF16 and torch.count_nonzero(select[3:]) == 0
    for index in range(3):
        logical = accepted + 1 + index
        assert select[index].nonzero().reshape(-1).tolist() == [logical if logical < 3 else 32 + logical - 3]
    chunk_calls = len(rows_fake.chunk.calls)
    module.commit_rows(
        state_rows, rows_state, selectors, step_on_full_rejection=step_form, step_committed_rows=anchor_form
    )
    # The re-anchor does not rerun the chunk kernel; the other two forms rerun it once (the masked catch-up).
    assert len(rows_fake.chunk.calls) - chunk_calls == (0 if anchor_form else 1)

    committed = cat_shards(state_rows.recurrent, 1)
    expected = cat_shards(state_step.recurrent, 1)  # accepted + 1 committed rows: row 0 (the base token) always is
    err = (committed - expected).abs().max().item()
    print(f"commit R={rows} a={accepted} {form}: state vs {accepted + 1} steps max abs {err:.2e}")
    torch.testing.assert_close(committed, expected, rtol=1e-4, atol=1e-5)
    if anchor_form or (accepted == 0 and step_form):
        # The 1-row step arithmetic over the committed rows is bitwise the 1-row path's state.
        assert torch.equal(committed, expected)
    if accepted == rows - 1 and not anchor_form:
        assert torch.equal(committed, cat_shards(result.final_state, 1))  # the forward pass's own final state
    # The history tile's rows 0..2 are the 1-row ring's three history slots; the selection matmul leaves exact
    # zeros in rows 3..31.
    history = cat_shards(rows_state.history, 3)  # the four device tiles side by side
    assert history.shape == (1, 1, 32, 4 * 2560)
    assert torch.equal(history[:, :, :3], gdn_ring_history(state_step)) and torch.count_nonzero(history[:, :, 3:]) == 0
    # The hand-off to the 1-row path: the host sets the phase, the sync fills the ring; the next step agrees.
    state_rows.conv_phase = (accepted + 1) % 4
    module.sync_state_from_rows_history(rows_state, state_rows)
    assert torch.equal(gdn_ring_history(state_rows), gdn_ring_history(state_step))
    torch.manual_seed(35)
    next_hidden = torch.randn(1, 1, 2560).to(torch.bfloat16)
    step_next = run_gdn_steps(module, state_step, next_hidden)[0]
    rows_next = run_gdn_steps(module, state_rows, next_hidden)[0]
    assert (rows_next.float() - step_next.float()).abs().max().item() <= gdn_output_tolerance(step_next)


def test_step_anchor_keeps_the_committed_state_on_the_one_row_path_over_many_passes(rows_fake, gdn_weights) -> None:
    """R = 5 passes with random accept counts: with ``step_committed_rows`` the committed state stays bitwise the
    1-row path's after every pass and the FIR history the ring's; the chunk form's state is reported beside it."""

    module = make_gdn_module(gdn_weights[1])
    generator = torch.Generator().manual_seed(180)
    state_step = seed_gdn_state(module, 21)
    state_anchor = clone_gdn_state(module, state_step)
    state_chunk = clone_gdn_state(module, state_step)
    constants = module.allocate_rows_constants(ROWS)
    rows_anchor = module.allocate_rows_state(constants)
    rows_chunk = module.allocate_rows_state(constants)
    module.sync_rows_history_from_state(state_anchor, rows_anchor)
    module.sync_rows_history_from_state(state_chunk, rows_chunk)
    committed_rows = 0
    chunk_drift = []
    for index in range(24):
        hidden = torch.randn(1, ROWS, 2560, generator=generator).to(torch.bfloat16)
        accepted = int(torch.randint(0, ROWS, (1,), generator=generator))
        selectors = gdn_rows_selectors(accepted, constants)
        run_gdn_steps(module, state_step, hidden[:, : accepted + 1])
        for state, rows_state, anchor in ((state_anchor, rows_anchor, True), (state_chunk, rows_chunk, False)):
            module.forward_rows(hidden_sharded(hidden), state, rows_state)
            module.commit_rows(state, rows_state, selectors, step_committed_rows=anchor)
        committed_rows += accepted + 1
        expected = cat_shards(state_step.recurrent, 1)
        anchored = cat_shards(state_anchor.recurrent, 1)
        chunk_drift.append((cat_shards(state_chunk.recurrent, 1) - expected).abs().max().item())
        assert torch.equal(anchored, expected), (index, accepted, (anchored - expected).abs().max().item())
        assert torch.equal(cat_shards(rows_anchor.history, 3)[:, :, :3], gdn_ring_history(state_step))
    assert committed_rows >= 60, committed_rows
    scale = cat_shards(state_step.recurrent, 1).abs().max().item()
    print(f"{committed_rows} rows: anchored bitwise; chunk-form drift {max(chunk_drift):.2e} (scale {scale:.2f})")


def test_rows_paths_never_upload_or_take_per_pass_host_ints() -> None:
    """Trace contract of the new GDN methods: one full chunk with device-resident constants, no host tensor
    creation, the row count only from the constants, and the 1-row state's ring untouched."""

    methods = source_methods(GDN_SOURCE, "Qwen38TTNNGDN")
    rows_methods = (
        "forward_rows",
        "commit_rows",
        "_all_gather_rows",
        "_project_rows",
        "_conv_window_rows",
        "_select_rows",
        "_causal_conv_rows",
        "_make_chunk_inputs",
        "_chunk_rows",
        "_project_rows_linear",
        "_land_rows_qkv",
        "_gate_and_project_rows",
        "_rows_output_tile",
        "_advance_history_rows",
    )
    calls = {
        name: [ast.unparse(node.func) for node in ast.walk(methods[name]) if isinstance(node, ast.Call)]
        for name in rows_methods
    }
    segment = {name: source_segment(GDN_SOURCE, methods[name]) for name in rows_methods}
    for name, names in calls.items():
        for forbidden in (
            "ttnn.zeros",
            "ttnn.from_torch",
            "ttnn.as_tensor",
            "torch.",
            "ttnn.copy_host_to_device_tensor",
        ):
            assert not any(called.startswith(forbidden) for called in names), (name, forbidden)
        assert "state.advance_conv_window" not in names and "state.conv_window" not in names, name
    for name in ("forward_rows", "commit_rows"):  # the row count comes from the constants, never per call
        for argument in methods[name].args.args:
            annotation = ast.unparse(argument.annotation) if argument.annotation is not None else ""
            assert annotation != "int", f"{name} takes a per-call host int {argument.arg}"
    # The owned rows path runs the public recurrence on its rank-four buffers with the original scale.
    assert calls["_chunk_rows"].count("ttnn.transformer.chunk_gated_delta_rule") == 1, calls["_chunk_rows"]
    source_call = next(
        node
        for node in ast.walk(methods["_chunk_rows"])
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "ttnn.transformer.chunk_gated_delta_rule"
    )
    positional = [ast.unparse(argument) for argument in source_call.args[:5]]
    assert positional == ["rows_state.q", "rows_state.k", "v_rows", "g_rows", "beta_rows"], positional
    source_keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in source_call.keywords}
    scale_and_layout = (source_keywords.get("scale"), source_keywords.get("output_head_major"))
    assert scale_and_layout == ("HEAD_DIM ** (-0.5)", "True"), source_keywords
    # The forward pass reads the committed state and writes only rows buffers; the commit is the only writer.
    assert "output_tensor=state.recurrent" not in segment["forward_rows"]
    assert "_copy_inplace" in calls["commit_rows"] and "output_tensor=state.recurrent" in segment["commit_rows"]
    # Exact selects: every row read from the [history | qkv] window is a 0/1 selection matmul (HiFi4, fp32
    # accumulation), never a row-unaligned slice (a ROW_MAJOR round trip on this runtime) and never a reduce.
    assert "ttnn.matmul(" in segment["_select_rows"]
    assert "compute_kernel_config=self.compute_config" in segment["_select_rows"]
    for name in ("_advance_history_rows", "_causal_conv_rows"):
        body = segment[name]
        assert "self._select_rows(" in body and "ttnn.slice" not in body and "ttnn.sum" not in body, name
    assert "ttnn.slice" not in segment["_conv_window_rows"] and "ttnn.pad" not in segment["_causal_conv_rows"]
    assert calls["_causal_conv_rows"].count("self._select_rows") == 1  # one per tap in the comprehension
    assert calls["_make_chunk_inputs"].count("ttnn.matmul") == 1
    assert "ttnn.repeat_interleave" not in calls["_make_chunk_inputs"]
    gate = segment["_gate_and_project_rows"]
    assert "ttnn.experimental.view(" in gate and "ttnn.to_layout" not in gate and "ttnn.permute" not in gate
    # The history select lands through the matmul's own output tensor (no copy); the state copy remains.
    assert "output_tensor=rows_state.history" in segment["_advance_history_rows"]
    assert "_copy_inplace" not in calls["_advance_history_rows"]
    assert gdn_module.CHUNK_SIZE == 32 and gdn_module.CONV_HISTORY_ROWS == 3 and gdn_module.CONV_WINDOW_ROWS == 35
    assert gdn_module.CONV_WINDOW_TILE_ROWS == 64
    tiles = gdn_module.chunk_constant_tiles()
    assert tiles["masks"].shape == (1, 1, 32, 96) and tiles["masks"].sum() == 3 * 16 * 16
    assert torch.equal(tiles["eye"][0, 0], torch.eye(32)) and tiles["tril"].sum() == 32 * 33 / 2


def test_rows_selection_matmuls_reproduce_torch_indexing_bitwise() -> None:
    """The 0/1 selection matrices pick exactly the rows torch indexing picks under HiFi4 with fp32 accumulation:
    each output element has a single nonzero term, so the result is exact whatever the accumulation order."""

    tiles = gdn_module.rows_window_select_tiles(ROWS)
    taps, stack, expand = tiles["conv_taps"], tiles["history_select_stack"], tiles["qk_expand"]
    for matrix in (*taps, stack, expand):
        assert set(matrix.unique().tolist()) <= {0.0, 1.0}
        assert torch.equal(matrix.to(torch.bfloat16).float(), matrix)  # exactly representable constants
    assert (taps != 0).sum(dim=-1).eq(1).all()  # every tap row selects exactly one window row
    assert (expand != 0).sum(dim=0).eq(1).all()  # every expanded column comes from exactly one q/k column
    for accepted in range(ROWS):
        select = stack[accepted].reshape(32, 64)
        assert (select[:3] != 0).sum(dim=-1).eq(1).all() and torch.count_nonzero(select[3:]) == 0
    assert torch.count_nonzero(stack[ROWS:]) == 0

    torch.manual_seed(29)
    history_tile = bf16_randn(1, 1, 32, 2560)  # rows 3..31 are never read: random here on purpose
    qkv = bf16_randn(1, 1, 32, 2560)
    window = torch.cat([history_tile, qkv], dim=2)  # the device window: two whole tiles
    logical = torch.cat([history_tile[:, :, :3], qkv], dim=2)  # [3 history | 32 new] = 35 rows

    def select_rows(select: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        products = select.float().unsqueeze(-1) * rows.float().unsqueeze(-3)  # every product, exact in fp32
        return products.sum(dim=-2).to(torch.bfloat16)  # fp32 accumulation, one rounding

    for tap in range(3):
        assert bits_equal(select_rows(taps[tap], window[0, 0]), logical[0, 0, tap : tap + 32]), tap
    for accepted in range(ROWS):
        onehot = torch.zeros(1, 32)
        onehot[0, accepted] = 1.0
        select = select_rows(onehot.to(torch.bfloat16), stack.to(torch.bfloat16)).float().reshape(32, 64)
        assert torch.equal(select, stack[accepted].reshape(32, 64))
        selected = select_rows(select, window[0, 0])
        assert bits_equal(selected[:3], logical[0, 0, accepted + 1 : accepted + 4]), accepted
        assert torch.count_nonzero(selected[3:].float()) == 0
    q = bf16_randn(1, 1, 32, 512)
    expanded = (q[0, 0].float() @ expand.float()).to(torch.bfloat16)  # one nonzero term per output element
    assert bits_equal(
        expanded.reshape(1, 32, GDN_HEADS, GDN_HEAD_DIM), q.reshape(1, 32, 4, GDN_HEAD_DIM).repeat_interleave(3, dim=2)
    )


def test_fake_view_is_the_device_tile_order_and_folds_head_major_rows_for_free() -> None:
    """The gate path's head-major ``[1, 12, 32, 128]`` -> token-major ``[1, 1, 32, 1536]`` view is a permute +
    reshape in logical terms: tile (h, c) sits at page 4h + c in both shapes."""

    torch.manual_seed(30)
    head_major = torch.randn(1, GDN_HEADS, 32, GDN_HEAD_DIM).to(torch.bfloat16)
    viewed = view_tile_pages(
        FakeTensor([head_major.clone() for _ in range(TP)], BF16), (1, 1, 32, GDN_HEADS * GDN_HEAD_DIM)
    )
    expected = head_major.permute(0, 2, 1, 3).reshape(1, 1, 32, GDN_HEADS * GDN_HEAD_DIM)
    assert bits_equal(viewed.torch_shards()[0], expected)
    assert not torch.equal(viewed.torch_shards()[0].float(), head_major.reshape(1, 1, 32, -1).float())
    # The GR shapes the model already runs through the view on device, and a plain row-count view.
    for host, shape in ((torch.randn(1, 4, 1, 640), (1, 1, 1, 2560)), (torch.randn(12, 32, 128), (1, 12, 32, 128))):
        host = host.to(torch.bfloat16)
        assert torch.equal(view_tile_pages(FakeTensor([host] * TP, BF16), shape).torch_shards()[0], host.reshape(shape))


def test_gdn_rows_and_one_row_bodies_are_the_pinned_walks() -> None:
    """The rows-path call order (the trace body's op order) and the stage boundaries the micro-test times; the 1-row
    production path is untouched (its call order is the op-diet pins' walk)."""

    methods = source_methods(GDN_SOURCE, "Qwen38TTNNGDN")

    def walk(node, prefix: str = "self.") -> list[str]:
        return [call.removeprefix(prefix) for call in ordered_calls(node) if call.startswith(prefix + "_")]

    # forward_rows gathers and hands the body to the resolved rows form; the composed body is walked below.
    assert walk(methods["forward_rows"]) == [
        "_validate_state",
        "_validate_rows_state",
        "_all_gather_rows",
        "_rows_body",
    ]
    assert walk(ast.parse(inspect.getsource(gdn_rows_wrap.rows_body_composed)), "gdn.") == [
        "_project_rows",
        "_causal_conv_rows",
        "_make_chunk_inputs",
        "_chunk_rows",
        "_gate_and_project_rows",
    ]
    assert walk(methods["_project_rows"]) == ["_project_rows_linear", "_land_rows_qkv"]
    assert walk(methods["_gate_and_project_rows"])[-1] == "_rows_output_tile"
    assert walk(methods["commit_rows"]) == [
        "_validate_state",
        "_validate_rows_state",
        "_advance_history_rows",  # the verify-rows fold's form: the prefix-state pick (a fused call), then the history
        "_step_committed_rows_state",  # the state re-anchor form: no chunk rerun
        "_advance_history_rows",
        "_chunk_rows",
        "_step_row_state",  # the step-on-full-rejection form's row 0
        "_advance_history_rows",
    ]
    assert walk(methods["_step_committed_rows_state"]) == ["_step_row_state"]
    # the recurrence: the wrap's prim-layout form when it owns the rows state, else the public op; both return
    # through the one validation
    assert walk(methods["_chunk_rows"]) == ["_validate_chunk_rows", "_validate_chunk_rows"]
    assert walk(methods["_validate_chunk_rows"]) == []
    assert walk(methods["_causal_conv_rows"]) == ["_shifted_rows_slab", "_conv_window_rows", "_select_rows"]
    assert walk(methods["_advance_history_rows"]) == ["_conv_window_rows", "_select_rows"]
    # the conv, recurrent inputs, recurrent step and gate run inside the resolved gdn_step chain (ttnn/fused/gdn_step)
    forward = methods["forward_decode"]
    assert walk(forward) == ["_validate_state", "_all_gather_hidden", "_project", "_gdn_step", "_out_project"]
    assert "rows" not in source_segment(GDN_SOURCE, forward)


# --------------------------------------------------------------------------- PLE under the fake


@pytest.mark.parametrize("rows,accepted", [(ROWS, a) for a in range(ROWS)] + [(CHUNK_ROWS, a) for a in (31, 0, 1, 8)])
def test_ple_rows_equal_sequential_prepared_steps_and_commit_selects_the_history(
    rows_fake, rows: int, accepted: int
) -> None:
    module = make_ple_module()
    torch.manual_seed(24)
    residual = torch.randn(rows, 4, 2560).to(torch.bfloat16)
    tokens = ((17, 15, 16, 95859, 20) + tuple(int(v) for v in torch.randint(0, 200_000, (CHUNK_ROWS,))))[:rows]
    state = seed_ple_state(module)
    rows_state = module.allocate_rows_state(rows)
    rows_state.load_from_state(state)
    assert rows_state.token_context is None

    # 1-row path, threaded through the nine slots and the host context.
    deltas, histories, contexts = [], [], [None]
    for row, token in enumerate(tokens):
        prepared = module.prepare_decode_input(torch.tensor([[token]], dtype=torch.long), state)
        result = module.forward_prepared(residual_rows(residual[row : row + 1]), prepared, state)
        deltas.append(cat_shards(result.residual_delta, 3))
        histories.append(torch.cat([cat_shards(slot, 3) for slot in state.conv], dim=1))
        contexts.append(tuple(int(v) for v in state.token_context[0]))

    prepared_rows = module.prepare_rows_input(tokens, rows_state)
    assert prepared_rows.contexts == tuple(contexts) and prepared_rows.embedding_rows.layout == ROW_MAJOR
    assert prepared_rows.embedding_rows.shape == (1, 1, rows, 640)
    result = module.forward_prepared_rows(residual_rows(residual), prepared_rows, rows_state)
    delta_rows = cat_shards(result.residual_delta, 3)
    assert delta_rows.shape == (1, rows, 4, 2560)
    for row in range(rows):
        assert torch.equal(delta_rows[:, row : row + 1], deltas[row]), row

    constants = gdn_module.Qwen38TTNNGDNRowsConstants.allocate("mesh", module.mesh_contract, rows=rows)
    module.commit_rows(rows_state, gdn_rows_selectors(accepted, constants))
    module.commit_rows_host(rows_state, prepared_rows, accepted)
    assert torch.equal(cat_shards(rows_state.history, 3), histories[accepted])
    assert rows_state.token_context == contexts[accepted + 1]
    # Round trip to the nine-slot form.
    back = module.allocate_state()
    rows_state.store_to_state(back)
    assert torch.equal(torch.cat([cat_shards(slot, 3) for slot in back.conv], dim=1), histories[accepted])
    assert back.token_context.tolist() == [list(contexts[accepted + 1])]


def test_ple_rows_source_pins() -> None:
    methods = source_methods(PLE_SOURCE, "Qwen38TTNNPLE")
    for name in (
        "forward_prepared_rows",
        "_project_rows",
        "_gate_rows",
        "_convolve_rows",
        "_conv_window_rows",
        "commit_rows",
    ):
        calls = [ast.unparse(node.func) for node in ast.walk(methods[name]) if isinstance(node, ast.Call)]
        assert not any(
            called.startswith(("ttnn.from_torch", "ttnn.zeros", "torch.", "self.resident_lookup")) for called in calls
        ), name
    inject = [ast.unparse(node.func) for node in ast.walk(methods["inject_rows"]) if isinstance(node, ast.Call)]
    assert inject.count("ttnn.permute") == 2 and "self.forward_prepared_rows" in inject and "ttnn.add" in inject
    convolve = source_segment(PLE_SOURCE, methods["_convolve_rows"])
    assert "tap * CONV_DILATION" in convolve and convolve.count("ttnn.slice") == 1
    commit = source_segment(PLE_SOURCE, methods["commit_rows"])
    assert "committed + CONV_STATE_LENGTH" in commit and "fast_and_approximate_mode=False" in commit
    # The 1-row body is untouched: it still shifts the nine slots and consumes the prepared host context.
    assert "_copy_inplace(state.conv[index + 1], state.conv[index]" in source_segment(PLE_SOURCE, methods["_convolve"])
    forward_one = source_segment(PLE_SOURCE, methods["forward_prepared"])
    assert "state.token_context = prepared.next_token_context" in forward_one


def test_layer_rows_entry_points_are_thin_and_typed() -> None:
    methods = source_methods(LAYER_SOURCE, "Qwen38TTNNDecoderLayer")

    def calls(name: str) -> list[str]:
        return [ast.unparse(node.func) for node in ast.walk(methods[name]) if isinstance(node, ast.Call)]

    for name in ("forward_gdn_rows", "commit_gdn_rows", "forward_ple_rows", "commit_ple_rows"):
        assert name in methods, name
    forward_ple = calls("forward_ple_rows")
    assert (
        "ple.inject_rows" in forward_ple and "ttnn.permute" not in forward_ple
    )  # the layer permutes only in _apply_ple
    assert "gdn.forward_rows" in calls("forward_gdn_rows")
    # The 1-row layer bodies do not call into the rows path.
    for name in ("forward_decode", "forward_decode_generic", "_apply_ple"):
        assert not any(called.endswith("_rows") for called in calls(name)), name
