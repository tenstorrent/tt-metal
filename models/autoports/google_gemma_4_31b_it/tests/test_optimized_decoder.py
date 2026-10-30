"""Default optimized runtime with real pinned weights and HF text activations."""

import json
from argparse import Namespace

import pytest

from models.autoports.google_gemma_4_31b_it.tests.record_decoder_activations import record
from models.autoports.google_gemma_4_31b_it.tests.run_optimized_decoder import run


@pytest.fixture(scope="module", params=[0, 5])
def recorded_layer(request, tmp_path_factory):
    directory = tmp_path_factory.mktemp(f"optimized_layer{request.param}")
    inputs = directory / "inputs.pt"
    record(request.param, 130, str(inputs), batch=32)
    return request.param, inputs


@pytest.mark.parametrize("batch", [1, 32, 16, 8])
def test_optimized_default(recorded_layer, batch, tmp_path):
    layer, inputs = recorded_layer
    result = tmp_path / "result.json"
    run(
        Namespace(
            layer=layer,
            batch=batch,
            seq=33,
            inputs=str(inputs),
            result=str(result),
            extended=False,
            synthetic=False,
            layout=False,
            replays=32,
        )
    )
    evidence = json.loads(result.read_text())
    assert evidence["implementation"] == "models.autoports.google_gemma_4_31b_it.tt.optimized_decoder.OptimizedDecoder"
    assert evidence["real_weights"] and evidence["deterministic"]
    assert evidence["trace_replays"] == 33
    assert evidence["caller_prefill_input_unchanged"]
    assert batch == 1 or evidence["distinct_request_inputs"]


@pytest.mark.parametrize("seq", [31, 65, 129])
def test_optimized_odd_page_capacity(recorded_layer, seq, tmp_path):
    """Decode must not read page-table padding at one, three, or five pages."""
    layer, inputs = recorded_layer
    result = tmp_path / "result.json"
    run(
        Namespace(
            layer=layer,
            batch=1,
            seq=seq,
            inputs=str(inputs),
            result=str(result),
            extended=False,
            synthetic=False,
            layout=False,
            replays=32,
        )
    )
    evidence = json.loads(result.read_text())
    assert evidence["deterministic"] and evidence["trace_replays"] == 33
    assert evidence["caller_prefill_input_unchanged"]
