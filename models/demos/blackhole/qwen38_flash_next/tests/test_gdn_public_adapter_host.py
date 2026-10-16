# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Model-owned public GDN layout, scaling and masked-buffer ownership contracts.

These host controls complement the device recurrence tests; they do not claim
that a Torch reshape proves a device tiled reshape or a native numerical path.
"""

from types import SimpleNamespace

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gdn_public_adapter as adapter
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gdn_rows_wrap as wrap


@pytest.mark.parametrize("rows", [32, 128, 2048])
def test_public_input_contract_preserves_head_and_chunk_order(monkeypatch, rows):
    # Distinct values across all axes expose swapped chunk/head/token dimensions.
    heads, dim = 12, 128
    q = torch.arange(rows * heads * dim, dtype=torch.int32).reshape(1, rows, heads, dim)
    k = -q - 1
    v = (q + 13).reshape(1, 1, rows, heads * dim)
    g = torch.arange(rows * heads, dtype=torch.float32).reshape(1, rows, heads) * -0.125
    beta = g + 0.375
    q_c, k_c = [x.permute(0, 2, 1, 3).reshape(heads, rows // 32, 32, dim) for x in (q, k)]
    g_c, beta_c = [x.permute(0, 2, 1).reshape(heads, rows // 32, 32, 1) for x in (g, beta)]
    initial = object()
    constants = tuple(object() for _ in range(4))
    config, expected_return = object(), (object(), object())
    monkeypatch.setattr(adapter.ttnn, "reshape", torch.reshape)
    monkeypatch.setattr(adapter.ttnn, "permute", torch.permute)

    def public(q_actual, k_actual, v_actual, g_actual, beta_actual, **kwargs):
        for actual, expected in zip(
            (q_actual, k_actual, v_actual, g_actual, beta_actual), (q, k, v.reshape(1, rows, heads * dim), g, beta)
        ):
            assert torch.equal(actual, expected)
        assert q_actual.ndim == k_actual.ndim == 4
        assert kwargs["scale"] == 1.0
        assert kwargs["initial_state"] is initial and kwargs["output_final_state"] is True
        assert kwargs["chunk_size"] == 32 and kwargs["output_head_major"] is True
        assert kwargs["program_config"] is config
        assert all(kwargs[key] is value for key, value in zip(("eye", "tril", "ones", "masks"), constants))
        return expected_return

    monkeypatch.setattr(adapter.ttnn.transformer, "chunk_gated_delta_rule", public)
    assert (
        adapter.chunk_public(q_c, k_c, v, beta_c, g_c, initial, constants, rows_total=rows, program_config=config)
        is expected_return
    )


@pytest.mark.parametrize("committed_rows", [None, 0, 1, 5, 31])
@pytest.mark.parametrize("fail", [False, True])
def test_masked_commit_preserves_inputs_and_releases_only_temporaries(monkeypatch, committed_rows, fail, expect_error):
    beta = torch.arange(12 * 32, dtype=torch.float32).reshape(12, 1, 32, 1) + 1
    decay = beta * -0.001
    buffers = SimpleNamespace(q_c=object(), k_c=object(), beta_c=beta, g_c=decay)
    constants = SimpleNamespace(**{key: object() for key in ("eye", "tril", "ones", "masks")})
    rows_state = SimpleNamespace(v=object(), constants=constants)
    initial, result = object(), (object(), object())
    mask = None if committed_rows is None else (torch.arange(32) < committed_rows).float().reshape(1, 1, 32, 1)
    originals = (beta.clone(), decay.clone())
    observed, released = [], []
    monkeypatch.setattr(wrap.ttnn, "multiply", lambda a, b, **kw: a * b)
    monkeypatch.setattr(wrap, "_release", lambda *args: released.extend(args))

    def public(q, k, value, beta_arg, decay_arg, state, tiles, *, rows_total):
        assert q is buffers.q_c and k is buffers.k_c and value is rows_state.v
        assert state is initial and rows_total == 32
        assert all(a is b for a, b in zip(tiles, vars(constants).values()))
        for actual, original in zip((beta_arg, decay_arg), originals):
            assert torch.equal(actual, original if mask is None else original * mask)
        observed.extend((beta_arg, decay_arg))
        if fail:
            raise RuntimeError("injected recurrence failure")
        return result

    monkeypatch.setattr(wrap, "chunk_source", public)
    if fail:
        with expect_error(RuntimeError, match="injected recurrence failure"):
            wrap.chunk(None, rows_state, buffers, initial, mask)
    else:
        assert wrap.chunk(None, rows_state, buffers, initial, mask) is result
    assert torch.equal(beta, originals[0]) and torch.equal(decay, originals[1])
    if mask is None:
        assert released == [] and observed[0] is beta and observed[1] is decay
    else:
        assert len(released) == 2 and all(a is b for a, b in zip(released, observed))
        assert all(t is not beta and t is not decay for t in released)
