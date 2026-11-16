"""Exercise the HF comparison CLI with fake inference and real saved reports."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest


@pytest.mark.parametrize("conditioning,accepted", [([10, 11], True), ([10, 12], False)])
def test_cli_checks_conditioning_before_publishing_comparison(tmp_path, conditioning, accepted):
    source = Path(__file__).resolve().parents[1] / "tools/precision_repro_hf.py"
    spec = importlib.util.spec_from_file_location("comparison_cli_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    case = tmp_path / "case.json"
    case.write_text(json.dumps({"prompt_token_ids": [10, 11]}))
    output = tmp_path / "hf.json"
    tt = tmp_path / "tt"
    tt.mkdir()
    rows = [{"position": 0, "forced_token": 42, "top1_token": 42, "forced_token_logprob": -0.5}]
    (tt / "case.json").write_text(
        json.dumps(
            {
                "continuations": {
                    "chosen": {
                        "conditioning_token_ids": conditioning,
                        "positions": rows,
                    }
                }
            }
        )
    )
    fake_torch = ModuleType("torch")
    fake_torch.bfloat16 = object()
    fake_model = SimpleNamespace(
        parameters=lambda: iter([SimpleNamespace(device=SimpleNamespace(type="cpu"), dtype=fake_torch.bfloat16)])
    )
    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoTokenizer = SimpleNamespace(from_pretrained=lambda *a, **kw: object())
    fake_loader = ModuleType("models.autoports.google_gemma_4_31b_it.tests.generate_full_reference")
    fake_loader.load_reference = lambda: fake_model
    fake_weights = ModuleType("models.autoports.google_gemma_4_31b_it.tt.weights")
    fake_weights.HF_MODEL, fake_weights.HF_REVISION = "fixture", "revision"

    def fake_run_case(*args):
        output.write_text(json.dumps({"teacher_forced_chosen": {"positions": rows}}))

    replacements = {
        "torch": fake_torch,
        "transformers": fake_transformers,
        fake_loader.__name__: fake_loader,
        fake_weights.__name__: fake_weights,
    }
    argv = ["hf", "--case", str(case), "--output", str(output), "--compare-dir", str(tt)]
    with patch.dict(sys.modules, replacements), patch.object(sys, "argv", argv), patch.object(
        module, "run_case", fake_run_case
    ):
        if accepted:
            module.main()
            comparison = json.loads(output.read_text())["tt_comparisons"][str(tt)]
            assert comparison["identity"]["conditioning_token_ids_verified"]
            assert comparison["top1_agreement"] == 1.0
        else:
            with pytest.raises(  # allow-pytest.raises: host-only --noconftest check
                ValueError, match="conditioning prompt token IDs differ"
            ):
                module.main()
            assert "tt_comparisons" not in json.loads(output.read_text())
