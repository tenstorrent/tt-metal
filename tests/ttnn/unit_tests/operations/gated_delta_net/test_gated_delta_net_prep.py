# SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Regime-pinned tests for R-PREP (prep cores feed the state cores' inboxes; the state cores run only the chunk
step's state part). PREP_MIN_TILE_ROWS = 1 forces R-PREP on short calls that the default keeps local, so a
sequence of one tile row (one prep core per group, one inbox slot), batched sequences sharing a tile row, a
partial last tile row, inf padding and one sequence's tile rows spread over several prep cores are all covered;
R-PREP off (PREP_OFFLOAD = False) is the local chunk step on the same inputs. The regime predicate is checked on
the descriptor helper."""

import pytest
import torch

import ttnn
from ttnn.operations.gated_delta_net import gated_delta_net
from ttnn.operations.gated_delta_net import gated_delta_net_program_descriptor as gdn_pd
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net import (
    ORDER,
    check_call,
    compute_config,
    make_inputs,
    reference,
    to_device,
)


def test_prep_predicate(device):
    budget = gdn_pd.l1_cb_budget(device)
    # model_prefill_4096 (bf16): 2 pieces, every idle core preps, the key head's 3 value heads per prep group.
    assert gdn_pd._prep_layout(1, 12, 4, 4096, 4, 4, 2, True, ttnn.bfloat16, 110 - 24, budget)[2] == 3
    # Short calls (2 to 16 tile rows): one value head per prep group (Refinement 11).
    for tokens in (64, 96, 128, 512):
        assert gdn_pd._prep_layout(1, 12, 4, tokens, 4, 4, 2, True, ttnn.bfloat16, 86, budget)[2] == 1
    # q / k cores (Refinement 14) on short calls, not where the three value heads already share them.
    assert gdn_pd._prep_layout(1, 12, 4, 128, 4, 4, 2, True, ttnn.bfloat16, 84, budget)[3]
    assert not gdn_pd._prep_layout(1, 12, 4, 4096, 4, 4, 2, True, ttnn.bfloat16, 84, budget)[3]
    # Too few tile rows, not chunked, one piece per pair, or no idle core: local.
    assert gdn_pd._prep_layout(1, 12, 4, 32, 4, 4, 2, True, ttnn.bfloat16, 86, budget) is None
    assert gdn_pd._prep_layout(1, 12, 4, 4096, 4, 4, 2, False, ttnn.bfloat16, 86, budget) is None
    assert gdn_pd._prep_layout(1, 12, 4, 4096, 4, 4, 1, True, ttnn.bfloat16, 98, budget) is None
    assert gdn_pd._prep_layout(1, 12, 4, 4096, 4, 4, 2, True, ttnn.bfloat16, 3, budget) is None


@pytest.mark.parametrize("offload", [True, False], ids=["prep", "local"])
@pytest.mark.parametrize(
    "inputs",
    [
        pytest.param(((1, 2, 256), (1, 3, 32, 64)), id="t2"),
        pytest.param(((1, 33, 256), (1, 3, 32, 64)), id="t33"),
        pytest.param(((1, 200, 256), (1, 3, 32, 64)), id="t200"),
        pytest.param(((1, 66, 256), (2, 3, 32, 64)), id="batched_b2_t33"),
        pytest.param(((1, 70, 2560), (1, 12, 128, 128)), id="chip_t70"),
    ],
)
@pytest.mark.parametrize("poison", [None, float("inf")], ids=["clean", "inf_padding"])
def test_prep_vs_local(device, monkeypatch, offload, inputs, poison):
    monkeypatch.setattr(gdn_pd, "PREP_MIN_TILE_ROWS", 1)
    monkeypatch.setattr(gdn_pd, "PREP_OFFLOAD", offload)
    torch.manual_seed(11)
    tensors = make_inputs(inputs)
    expected = reference(*(tensors[name] for name in ORDER))
    returned = gated_delta_net(*to_device(tensors, device, poison=poison), compute_kernel_config=compute_config())
    check_call(returned, expected, inputs)


@pytest.mark.parametrize("mask", [0, 1, 2, 4, 7])
@pytest.mark.parametrize(
    "inputs",
    [
        pytest.param(((1, 200, 256), (1, 3, 32, 64)), id="t200"),
        pytest.param(((1, 70, 2560), (1, 12, 128, 128)), id="chip_t70"),
    ],
)
def test_prep_scale_fpu(device, monkeypatch, inputs, mask):
    """Refinement 26: each head-block scaling as an SFPU or an FPU product (PREP_SCALE_FPU mask), against the
    reference."""
    monkeypatch.setattr(gdn_pd, "PREP_MIN_TILE_ROWS", 1)
    monkeypatch.setattr(gdn_pd, "PREP_SCALE_FPU", mask)
    torch.manual_seed(11)
    tensors = make_inputs(inputs)
    expected = reference(*(tensors[name] for name in ORDER))
    returned = gated_delta_net(*to_device(tensors, device), compute_kernel_config=compute_config())
    check_call(returned, expected, inputs)


@pytest.mark.parametrize("m_beta", [False, True], ids=["beta_rhs", "beta_m"])
@pytest.mark.parametrize("fused", [True, False], ids=["fused", "split"])
@pytest.mark.parametrize("poison", [None, float("inf")], ids=["clean", "inf_padding"])
@pytest.mark.parametrize(
    "inputs",
    [
        pytest.param(((1, 2, 256), (1, 3, 32, 64)), id="t2"),
        pytest.param(((1, 200, 256), (1, 3, 32, 64)), id="t200"),
        pytest.param(((1, 70, 2560), (1, 12, 128, 128)), id="chip_t70"),
    ],
)
def test_prep_m_beta(device, monkeypatch, inputs, poison, fused, m_beta):
    """Refinement 28: beta in the rhs of M or folded into M (PREP_M_BETA), with and without the fused rhs, against
    the reference."""
    monkeypatch.setattr(gdn_pd, "PREP_MIN_TILE_ROWS", 1)
    monkeypatch.setattr(gdn_pd, "PREP_M_BETA", m_beta)
    monkeypatch.setattr(gdn_pd, "STATE_FUSED_RHS", fused)
    torch.manual_seed(11)
    tensors = make_inputs(inputs)
    expected = reference(*(tensors[name] for name in ORDER))
    returned = gated_delta_net(*to_device(tensors, device, poison=poison), compute_kernel_config=compute_config())
    check_call(returned, expected, inputs)


@pytest.mark.parametrize("mid", [0, 1], ids=["norms_last", "norm_mid"])
@pytest.mark.parametrize("hold", [True, False], ids=["hold", "unpack"])
@pytest.mark.parametrize(
    "inputs",
    [
        pytest.param(((1, 200, 256), (1, 3, 32, 64)), id="t200"),
        pytest.param(((1, 20, 2560), (1, 12, 128, 128)), id="chip_t20"),
        pytest.param(((1, 70, 2560), (1, 12, 128, 128)), id="chip_t70"),
        pytest.param(((1, 1344, 2560), (1, 12, 128, 128)), id="chip_t1344"),
    ],
)
def test_prep_hold_mid(device, monkeypatch, inputs, hold, mid):
    """Refinement 28: STATE_HOLD (the upper update batch of a state core stays in DEST; chip_t20 is one step, first
    and last) and PREP_NORM_MID (chip_t1344: 42 tile rows on 21 prep cores, so the cores past the inbox depth
    normalize a unit ahead of the heads of their second row), each on and off, against the reference."""
    monkeypatch.setattr(gdn_pd, "PREP_MIN_TILE_ROWS", 1)
    monkeypatch.setattr(gdn_pd, "STATE_HOLD", hold)
    monkeypatch.setattr(gdn_pd, "PREP_NORM_MID", mid)
    torch.manual_seed(23)
    tensors = make_inputs(inputs)
    expected = reference(*(tensors[name] for name in ORDER))
    returned = gated_delta_net(*to_device(tensors, device), compute_kernel_config=compute_config())
    check_call(returned, expected, inputs)


def test_prep_state_carry(device, monkeypatch):
    """Two R-PREP calls (T = 100, 40) then a decode call, each from the previous call's states."""
    monkeypatch.setattr(gdn_pd, "PREP_MIN_TILE_ROWS", 1)
    torch.manual_seed(13)
    head = (1, 3, 32, 64)
    channels = 256
    calls = [100, 40, 1]
    full = make_inputs(((1, sum(calls), channels), head))
    hist, state = full["conv_state"], full["recurrent_state"]
    hist_dev, state_dev = None, None
    start = 0
    for tokens in calls:
        part = dict(full)
        for name in ("mixed_qkv", "z", "b", "a"):
            part[name] = full[name][:, start : start + tokens]
        part["conv_state"], part["recurrent_state"] = hist, state
        expected = reference(*(part[name] for name in ORDER))
        dev = to_device(part, device)
        if hist_dev is not None:
            dev[ORDER.index("conv_state")], dev[ORDER.index("recurrent_state")] = hist_dev, state_dev
        _, hist_dev, state_dev = check_call(
            gated_delta_net(*dev, compute_kernel_config=compute_config()), expected, ((1, tokens, channels), head)
        )
        hist, state = expected[1], expected[2]
        start += tokens


@pytest.mark.parametrize("token_dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize(
    "inputs",
    [
        pytest.param(((1, 33, 256), (1, 3, 32, 64)), id="t33"),
        pytest.param(((1, 200, 256), (1, 3, 32, 64)), id="t200"),
        pytest.param(((1, 66, 256), (2, 3, 32, 64)), id="batched_b2_t33"),
        pytest.param(((1, 128, 2560), (1, 12, 128, 128)), id="chip_t128"),
        pytest.param(((1, 300, 2560), (1, 12, 128, 128)), id="chip_t300"),
    ],
)
def test_prep_shared_qk(device, monkeypatch, inputs, token_dtype):
    """Refinement 14: q / k cores (PREP_SHARE_QK) give the same bits as each prep core's own q / k part, at one value
    head per prep group (the q / k part is then shared by the key head's groups)."""
    from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net_perf import _to_device

    monkeypatch.setattr(gdn_pd, "PREP_MIN_TILE_ROWS", 1)
    monkeypatch.setattr(gdn_pd, "PREP_HEADS", 1)
    monkeypatch.setattr(gdn_pd, "STATE_FOLD", True)  # "auto" would differ between the two plans
    monkeypatch.setattr(gdn_pd, "PREP_L2_FPU", False)  # the q / k cores keep the SFPU norm
    torch.manual_seed(17)
    tensors = make_inputs(inputs)
    expected = reference(*(tensors[name] for name in ORDER))
    results = {}
    for shared in (True, False):
        monkeypatch.setattr(gdn_pd, "PREP_SHARE_QK", shared)
        returned = gated_delta_net(*_to_device(tensors, device, token_dtype), compute_kernel_config=compute_config())
        if token_dtype == ttnn.float32:
            check_call(returned, expected, inputs)
        results[shared] = [ttnn.to_torch(t) for t in returned]
    for got, ref, name in zip(results[True], results[False], ("output", "conv_state_out", "recurrent_state_out")):
        assert torch.equal(got, ref), f"{name}: shared q / k differs from the prep cores' own"


@pytest.mark.parametrize("norm_split", [True, False], ids=["norm_split", "norm_per_row"])
@pytest.mark.parametrize("fused", [True, False], ids=["fused_rhs", "packed_reads"])
@pytest.mark.parametrize(
    "inputs",
    [
        pytest.param(((1, 200, 256), (1, 3, 32, 64)), id="t200"),
        pytest.param(((1, 66, 256), (2, 3, 32, 64)), id="batched_b2_t33"),
        pytest.param(((1, 1056, 2560), (1, 12, 128, 128)), id="chip_t1056"),
    ],
)
def test_prep_norm_split_knobs(device, monkeypatch, inputs, norm_split, fused):
    """PREP_NORM_SPLIT (prep rows first, norm units spread over the group; chip_t1056: 33 tile rows on 21 prep cores,
    so 12 cores prep two rows and normalize nothing) and STATE_FUSED_RHS, each on and off, against the reference."""
    monkeypatch.setattr(gdn_pd, "PREP_MIN_TILE_ROWS", 1)
    monkeypatch.setattr(gdn_pd, "PREP_NORM_SPLIT", norm_split)
    monkeypatch.setattr(gdn_pd, "STATE_FUSED_RHS", fused)
    torch.manual_seed(19)
    tensors = make_inputs(inputs)
    expected = reference(*(tensors[name] for name in ORDER))
    returned = gated_delta_net(*to_device(tensors, device), compute_kernel_config=compute_config())
    check_call(returned, expected, inputs)


@pytest.mark.parametrize("hold", [True, False], ids=["hold", "unpack"])
@pytest.mark.parametrize("fold", [False, True], ids=["rhs", "fold"])
@pytest.mark.parametrize(
    "inputs",
    [
        pytest.param(((1, 2, 256), (1, 3, 32, 64)), id="t2"),
        pytest.param(((1, 200, 256), (1, 3, 32, 64)), id="t200"),
        pytest.param(((1, 66, 256), (2, 3, 32, 64)), id="batched_b2_t33"),
        pytest.param(((1, 20, 2560), (1, 12, 128, 128)), id="chip_t20"),
        pytest.param(((1, 70, 2560), (1, 12, 128, 128)), id="chip_t70"),
        pytest.param(((1, 1056, 2560), (1, 12, 128, 128)), id="chip_t1056"),
    ],
)
def test_prep_state_fold(device, monkeypatch, inputs, fold, hold):
    """STATE_FOLD (M folded into the state core's operands by the prep core) with and without STATE_HOLD, against the
    reference."""
    monkeypatch.setattr(gdn_pd, "PREP_MIN_TILE_ROWS", 1)
    monkeypatch.setattr(gdn_pd, "STATE_FOLD", fold)
    monkeypatch.setattr(gdn_pd, "STATE_HOLD", hold)
    torch.manual_seed(29)
    tensors = make_inputs(inputs)
    expected = reference(*(tensors[name] for name in ORDER))
    returned = gated_delta_net(*to_device(tensors, device), compute_kernel_config=compute_config())
    check_call(returned, expected, inputs)


@pytest.mark.parametrize(
    "inputs, segs, token_dtype",
    [
        pytest.param(((1, 128, 320), (1, 3, 64, 64)), 2, ttnn.float32, id="t128_s2"),
        pytest.param(((1, 192, 320), (1, 3, 64, 64)), 3, ttnn.float32, id="t192_s3"),
        pytest.param(((1, 512, 320), (1, 3, 64, 64)), 4, ttnn.float32, id="t512_s4"),
        pytest.param(((1, 256, 2560), (1, 12, 128, 128)), 2, ttnn.bfloat16, id="chip_t256_s2"),
        pytest.param(((1, 1024, 2560), (1, 12, 128, 128)), 2, ttnn.bfloat16, id="chip_t1024_s2"),
    ],
)
def test_prep_seq_split(device, monkeypatch, tmp_path, inputs, segs, token_dtype):
    """SEQ_SPLIT: the token axis in segments whose state chains run in parallel (segments 1.. on [P | H], then the
    combine and the output correction). fp32 tokens against the reference; bf16 tokens (the plans of the model's
    shapes) against the unsplit call. The layout dump shows that the split plan ran."""
    import json

    from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net_perf import _to_device

    layout = tmp_path / "layout.json"
    monkeypatch.setenv("GDN_LAYOUT_DUMP", str(layout))
    monkeypatch.setattr(gdn_pd, "PREP_MIN_TILE_ROWS", 1)
    torch.manual_seed(31)
    tensors = make_inputs(inputs)
    results = {}
    for split in (1, segs):
        monkeypatch.setattr(gdn_pd, "SEQ_SPLIT", split)
        returned = gated_delta_net(*_to_device(tensors, device, token_dtype), compute_kernel_config=compute_config())
        assert json.loads(layout.read_text())["batch"] == 2 * split - 1, "the call fell back to the unsplit plan"
        results[split] = [ttnn.to_torch(t).to(torch.float32) for t in returned]
        if token_dtype == ttnn.float32:
            check_call(returned, reference(*(tensors[name] for name in ORDER)), inputs)
    out, cs, state = results[segs]
    out_1, cs_1, state_1 = results[1]
    assert torch.equal(cs, cs_1), "conv_state_out differs from the unsplit call's"
    state_rms = float((state - state_1).pow(2).mean().sqrt() / state_1.std())
    out_rms = float((out - out_1).pow(2).mean().sqrt() / out_1.std())
    print(f"\nSPLIT_ERR inputs={inputs} segs={segs} state_rms={state_rms:.3e} out_rms={out_rms:.3e}")
    assert state_rms < 5e-3 and out_rms < 2e-2


@pytest.mark.parametrize("tail", [1, 3, 8])
@pytest.mark.parametrize(
    "inputs",
    [
        pytest.param(((1, 200, 256), (1, 3, 32, 64)), id="t200"),
        pytest.param(((1, 66, 256), (2, 3, 32, 64)), id="batched_b2_t33"),
        pytest.param(((1, 70, 2560), (1, 12, 128, 128)), id="chip_t70"),
        pytest.param(((1, 1056, 2560), (1, 12, 128, 128)), id="chip_t1056"),
    ],
)
def test_prep_norm_tail(device, monkeypatch, inputs, tail):
    """STATE_NORM_TAIL: the state cores normalize the last tile rows of a sequence themselves (clamped below the
    sequence's rows; t200 has a partial last tile row), against the reference."""
    monkeypatch.setattr(gdn_pd, "PREP_MIN_TILE_ROWS", 1)
    monkeypatch.setattr(gdn_pd, "STATE_NORM_TAIL", tail)
    torch.manual_seed(37)
    tensors = make_inputs(inputs)
    expected = reference(*(tensors[name] for name in ORDER))
    returned = gated_delta_net(*to_device(tensors, device), compute_kernel_config=compute_config())
    check_call(returned, expected, inputs)
