# SPDX-License-Identifier: Apache-2.0
"""CPU-only tiny HF model checks cache conditioning, not checkpoint accuracy."""

import json
import time
from types import SimpleNamespace

import pytest
import torch
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

from models.autoports.google_gemma_4_31b_it.tools.precision_repro_hf import run_case, validate_tt_comparison_identity


def test_chunked_hybrid_cache_preserves_tf_and_seeded_generation(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    config = Gemma4TextConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_global_key_value_heads=2,
        head_dim=8,
        global_head_dim=8,
        hidden_size_per_layer_input=0,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=4,
        pad_token_id=0,
        eos_token_id=None,
        bos_token_id=1,
    )
    config._attn_implementation = "sdpa"
    model = Gemma4ForCausalLM(config).eval()
    tokens = list(range(1, 14))
    chosen = [14, 15, 16]
    case = {"prompt_token_ids": tokens, "continuations": {"chosen": chosen}}
    source = tmp_path / "case.json"
    source.write_text(json.dumps(case))
    tokenizer = SimpleNamespace(pad_token_id=0, decode=lambda ids, **kwargs: str(ids))
    args = SimpleNamespace(prefill_chunk=3, seeds=[42, 43], tokens=4)
    out = tmp_path / "output.json"
    run_case(model, tokenizer, source, case, out, args, time.monotonic(), "tiny_random_test")
    report = json.loads(out.read_text())
    with torch.inference_mode():
        expected = model(torch.tensor([tokens + chosen[:-1]]), use_cache=False).logits[0, len(tokens) - 1 :].float()
        expected_lp = expected.log_softmax(-1)
        for i, row in enumerate(report["teacher_forced_chosen"]["positions"]):
            assert row["top1_token"] == int(expected_lp[i].argmax())
            assert abs(row["forced_token_logprob"] - float(expected_lp[i, chosen[i]])) < 1e-5
        for row in report["results"]:
            torch.manual_seed(row["seed"])
            ids = torch.tensor([tokens])
            generated = model.generate(
                ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=4,
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                top_k=20,
                pad_token_id=0,
                logits_to_keep=1,
            )[0, len(tokens) :].tolist()
            assert row["generated_token_ids"] == generated
            assert row["streamed_generated_tokens"] == len(generated)
    assert report["status"] == "complete"


def test_token_streamer_skips_prompt_and_reports_only_counts(capsys):
    from models.autoports.google_gemma_4_31b_it.tools.precision_repro_hf import TokenCountStreamer

    streamer = TokenCountStreamer(42)
    streamer.put(torch.ones((1, 257), dtype=torch.long))
    for _ in range(256):
        streamer.put(torch.tensor([7]))
    streamer.end()
    assert streamer.generated_count == 256
    assert streamer.ended
    assert capsys.readouterr().out.splitlines() == ["HF_GENERATION_PROGRESS 42 128", "HF_GENERATION_PROGRESS 42 256"]


def _comparison_fixtures(tmp_path):
    case = {"prompt_token_ids": [10, 11], "metadata": {"task": "task-a", "call": 7}}
    case_path = tmp_path / "case.json"
    case_path.write_text(json.dumps(case))
    hf_rows = [{"position": i, "forced_token": 42} for i in range(4)]
    tt_case_path = tmp_path / "tt-case.json"
    tt_case_path.write_text(json.dumps(case))
    tt_report = {
        "case": str(tt_case_path),
        "metadata": {"task": "task-a", "call": 7},
        "continuations": {
            "chosen": {
                "positions": [{"position": i, "forced_token": 42} for i in range(4)],
                "tokens": 4,
                "conditioning_token_ids": [10, 11],
            }
        },
    }
    hf_report = {"case_sha256": __import__("hashlib").sha256(case_path.read_bytes()).hexdigest()}
    return case_path, case, hf_report, tt_report, hf_rows


def test_comparison_identity_accepts_historical_schema_without_optional_hash_or_window(tmp_path):
    case_path, case, hf_report, tt_report, hf_rows = _comparison_fixtures(tmp_path)
    rows, identity = validate_tt_comparison_identity(case_path, case, hf_report, tt_report, hf_rows)
    assert len(rows) == 4
    assert identity["conditioning_token_ids_verified"] is True
    assert identity["explicit_window_checked"] is False


def test_comparison_identity_can_verify_legacy_prompt_from_saved_case_file(tmp_path):
    case_path, case, hf_report, tt_report, hf_rows = _comparison_fixtures(tmp_path)
    tt_report["continuations"]["chosen"].pop("conditioning_token_ids")
    rows, identity = validate_tt_comparison_identity(case_path, case, hf_report, tt_report, hf_rows)
    assert len(rows) == 4
    assert identity["case_prompt_ids_verified"] is True
    assert identity["conditioning_token_ids_verified"] is False


def test_comparison_identity_rejects_unrelated_conditioning_even_when_targets_match(tmp_path):
    case_path, case, hf_report, tt_report, hf_rows = _comparison_fixtures(tmp_path)
    tt_report["continuations"]["chosen"]["conditioning_token_ids"] = [10, 12]
    with pytest.raises(  # allow-pytest.raises: host-only --noconftest check
        ValueError, match="conditioning prompt token IDs differ"
    ):
        validate_tt_comparison_identity(case_path, case, hf_report, tt_report, hf_rows)


def test_comparison_identity_rejects_unrelated_case_file_even_when_report_conditioning_claims_match(tmp_path):
    case_path, case, hf_report, tt_report, hf_rows = _comparison_fixtures(tmp_path)
    other = tmp_path / "other-case.json"
    other.write_text(json.dumps({"prompt_token_ids": [10, 12], "metadata": {"task": "task-a", "call": 7}}))
    tt_report["case"] = str(other)
    with pytest.raises(  # allow-pytest.raises: host-only --noconftest check
        ValueError, match="case prompt token IDs differ"
    ):
        validate_tt_comparison_identity(case_path, case, hf_report, tt_report, hf_rows)


def test_comparison_identity_rejects_shifted_window_with_same_repeated_targets(tmp_path):
    case_path, case, hf_report, tt_report, hf_rows = _comparison_fixtures(tmp_path)
    tt_report["continuations"]["chosen"]["positions"] = [{"position": i + 1, "forced_token": 42} for i in range(4)]
    with pytest.raises(  # allow-pytest.raises: host-only --noconftest check
        ValueError, match="positions or forced-token window differ"
    ):
        validate_tt_comparison_identity(case_path, case, hf_report, tt_report, hf_rows)


def test_comparison_identity_checks_optional_case_hash_and_explicit_window(tmp_path):
    case_path, case, hf_report, tt_report, hf_rows = _comparison_fixtures(tmp_path)
    tt_report["case_sha256"] = "wrong-hash"
    with pytest.raises(ValueError, match="case hashes differ"):  # allow-pytest.raises: host-only --noconftest check
        validate_tt_comparison_identity(case_path, case, hf_report, tt_report, hf_rows)
    tt_report.pop("case_sha256")
    tt_report["window"] = {"start_token": 1, "max_tokens": 4}
    with pytest.raises(ValueError, match="window starts"):  # allow-pytest.raises: host-only --noconftest check
        validate_tt_comparison_identity(case_path, case, hf_report, tt_report, hf_rows)
