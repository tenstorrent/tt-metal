import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest
from pathlib import Path
from tests.utils.comparison import compare_tensors

GOLDEN_DIR = Path(__file__).parent.parent / "golden_tensors"

@pytest.fixture(scope="module")
def golden_intermediates():
    tensors = {}
    for pt_file in GOLDEN_DIR.glob("*.pt"):
        tensors[pt_file.stem] = torch.load(pt_file, weights_only=False)
    return tensors

def test_golden_tensors_exist(golden_intermediates):
    assert "inputs" in golden_intermediates
    assert "model_config_dict" in golden_intermediates
    encoder_keys = [k for k in golden_intermediates if k.startswith("encoder")]
    assert len(encoder_keys) > 0, "No encoder intermediates found"

def test_self_comparison(golden_intermediates):
    for name, tensor in golden_intermediates.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        passed, details = compare_tensors(tensor, tensor)
        assert passed, f"Self-comparison failed for {name}: {details}"