"""Real-checkpoint oracle for both layer kinds shipped by advisor-challenger."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
import torch

from models.demos.gemma4.config import MeshConfig, ModeConfig


ROOT = Path(__file__).resolve().parents[2]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


functional = _load(ROOT / "tests/test_functional_decoder.py", "gemma4_oracle_functional")
optimized = _load(ROOT / "tests/test_optimized_decoder.py", "gemma4_oracle_optimized")

pytestmark = pytest.mark.parametrize(
    "mesh_device,device_params",
    [pytest.param((1, 1), {"trace_region_size": 0}, id="1x1")],
    indirect=True,
)


@pytest.mark.parametrize("layer_kind", ["sliding_attention", "full_attention"])
def test_real_weight_oracle(layer_kind, mesh_device):
    from safetensors import safe_open
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer

    weight_file = Path(os.environ["GEMMA4_12B_WEIGHTS"])
    config = functional._hf_text_config()
    layer_idx = functional._find_layer_idx(config, layer_kind)
    prefix = f"model.language_model.layers.{layer_idx}."
    with safe_open(weight_file, framework="pt", device="cpu") as handle:
        state = {key[len(prefix) :]: handle.get_tensor(key) for key in handle.keys() if key.startswith(prefix)}
    assert state, f"no real tensors for {layer_kind} layer {layer_idx}"

    hf_layer = Gemma4TextDecoderLayer(config, layer_idx=layer_idx)
    hf_layer.load_state_dict(state, strict=True)
    hf_layer.to(torch.bfloat16).eval()
    Decoder = optimized._load_optimized_decoder_class()
    decoder = Decoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        mesh_config=MeshConfig(mesh_device.shape, decode=ModeConfig(tp=1)),
    )
    result = optimized._run_optimized_prefill_then_decode(
        layer_kind,
        32,
        mesh_device,
        hf_layer=hf_layer,
        decoder=decoder,
        weights_label="advisor_challenger_real",
    )
    assert result["prefill_pcc"] >= functional.PCC_THRESHOLD
    assert result["decode_pcc"] >= optimized._decode_threshold(layer_kind)
