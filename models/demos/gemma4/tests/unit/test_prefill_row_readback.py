# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The one-row logits readback of a traced single-user prefill (``ChunkedPrefillPageTableGuardMixin._append_prefill_result``)
returns the same last-token logits as the shared whole-tile readback (``GEMMA4_PREFILL_ROW_READBACK=0``), with the
same first token, on the served unit (a 128-token prompt on a (4,1) column). Prints the warm call wall of both and the
model's host phase accumulators. Device, no HF."""

import os
import statistics
import time

import pytest
import torch

import ttnn

from ..test_factory import TestFactory, _get_model_path, parametrize_mesh_with_fabric
from .tracy_prefill_common import build_prefill_trace_fixtures, load_prefill_trace_generator

PREFILL_LEN = int(os.environ.get("GEMMA4_ROW_READBACK_LEN", "128"))
N_CALLS = int(os.environ.get("GEMMA4_ROW_READBACK_CALLS", "16"))


def _row(out):
    t = out if torch.is_tensor(out) else out[0]
    return t.reshape(-1, t.shape[-1])[0].float()


@parametrize_mesh_with_fabric(mesh_shapes=[(4, 1)], device_params_extra={"trace_region_size": 512_000_000})
def test_prefill_row_readback_matches_tile_readback(mesh_device, reset_seeds, request, monkeypatch):
    hf_config = TestFactory.create_hf_config()
    if int(getattr(hf_config, "hidden_size_per_layer_input", 0) or 0) > 0:
        pytest.skip("PLI models disable prefill trace")
    fixtures = build_prefill_trace_fixtures(1, PREFILL_LEN, hf_config.vocab_size)
    generator, kv = load_prefill_trace_generator(mesh_device, _get_model_path(), fixtures)
    model = generator.model[0]
    mesh = model.mesh_device

    def call():
        model._g4_pf_ms = {}
        out = generator.prefill_forward_text(
            fixtures["tokens"],
            page_table=fixtures["page_table"],
            kv_cache=kv,
            prompt_lens=fixtures["prompt_lens"],
            enable_trace=True,
            warmup_prefill=False,
        )
        ttnn.synchronize_device(mesh)
        return out

    results = {}
    for mode in ("1", "0", "1"):  # row, tile, row again (the trace is captured by the first call)
        monkeypatch.setenv("GEMMA4_PREFILL_ROW_READBACK", mode)
        walls = []
        for _ in range(N_CALLS):
            t0 = time.perf_counter()
            out = call()
            walls.append(time.perf_counter() - t0)
        results[mode] = (_row(out), statistics.median(walls) * 1e3, dict(model._g4_pf_ms))
    row_logits, row_ms, row_phases = results["1"]
    tile_logits, tile_ms, tile_phases = results["0"]
    print(f"RESULT row_readback row: call median {row_ms:.2f} ms phases {row_phases}")
    print(f"RESULT row_readback tile: call median {tile_ms:.2f} ms phases {tile_phases}")
    print(f"RESULT row_readback first token row={int(torch.argmax(row_logits))} tile={int(torch.argmax(tile_logits))}")
    assert row_logits.shape == tile_logits.shape, (row_logits.shape, tile_logits.shape)
    assert torch.equal(row_logits, tile_logits), "the one-row readback must be the tile readback's row"
