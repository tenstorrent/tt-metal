# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""CPU contract for the native prompt test's setup; no device fixtures are loaded."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from ...pipelines.ltx import pipeline_ltx


@pytest.mark.parametrize("checkpoint_kind", ["local", "hub", "missing"])
def test_prompt_setup_is_encoder_only(monkeypatch, tmp_path, checkpoint_kind):
    checkpoint = tmp_path / "weights.safetensors"
    checkpoint.write_bytes(b"identity fixture; no model weights")
    reference = str(checkpoint) if checkpoint_kind == "local" else "owner/model:weights.safetensors"
    events = []

    def forbidden(*args, **kwargs):
        pytest.fail("prompt setup constructed or loaded a transformer/VAE/audio decoder")

    def resolve_hub(*, repo_id, filename):
        assert (repo_id, filename) == ("owner/model", "weights.safetensors")
        events.append("resolved")
        if checkpoint_kind == "missing":
            raise FileNotFoundError("checkpoint unavailable")
        return str(checkpoint)

    class Pair:
        _encoder_trace = True

        def __init__(self, gemma_path, **kwargs):
            assert gemma_path == "gemma-fixture"
            assert kwargs["checkpoint_name"] is None
            assert kwargs["dynamic_load"] is False
            self.gemma_path = gemma_path
            self.checkpoint_name = None

        def ensure_loaded(self):
            assert self.checkpoint_name == str(checkpoint)
            if checkpoint_kind == "hub":
                assert events == ["resolved"]
            events.append("loaded")

    class EncodingStarted(Exception):
        pass

    def encode(pipe, prompts, *, use_cache):
        assert pipe.checkpoint_name is None
        assert pipe.transformer is None and pipe.vae is None and pipe._audio_adapter is None
        assert pipe.gemma_encoder_pair.checkpoint_name == str(checkpoint)
        assert events[-1] == "loaded"
        assert prompts == ["warmup"] and use_cache is False
        raise EncodingStarted

    monkeypatch.setattr(pipeline_ltx.ttnn.device, "is_blackhole", lambda: True)
    monkeypatch.setattr(pipeline_ltx, "CCLManager", lambda mesh, **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(pipeline_ltx, "GemmaTokenizerEncoderPair", Pair)
    monkeypatch.setattr(pipeline_ltx, "hf_hub_download", resolve_hub)
    for name in ("LTXVideoVAEAdapter", "LTXTransformerCheckpoint", "LTXTransformerModel", "LTXAudioDecoderAdapter"):
        monkeypatch.setattr(pipeline_ltx, name, forbidden)
    monkeypatch.setattr(pipeline_ltx.LTXPipeline, "_prime_caches", forbidden)
    monkeypatch.setattr(pipeline_ltx.LTXPipeline, "encode_prompts", encode)

    # Execute the real test body without its device parametrization or checkpoint-discovery
    # decorators. The actual pipeline factory, constructor and resolver remain under test.
    source = Path(__file__).parents[1] / "models/ltx/test_prompt_encode_ltx.py"
    tree = ast.parse(source.read_text())
    body = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "test_prompt_conditioning"
    )
    body.decorator_list = []
    namespace = {
        "LTXPipeline": pipeline_ltx.LTXPipeline,
        "default_ltx_checkpoint": lambda filename: reference,
        "default_ltx_gemma": lambda: "gemma-fixture",
    }
    exec(compile(ast.Module(body=[body], type_ignores=[]), str(source), "exec"), namespace)
    try:
        namespace["test_prompt_conditioning"](SimpleNamespace(shape=(4, 8)))
    except EncodingStarted:
        assert checkpoint_kind != "missing"
    except FileNotFoundError as error:
        assert checkpoint_kind == "missing" and str(error) == "checkpoint unavailable"
        assert events == ["resolved"]
    else:
        pytest.fail("prompt test did not reach encoding or propagate checkpoint resolution failure")
