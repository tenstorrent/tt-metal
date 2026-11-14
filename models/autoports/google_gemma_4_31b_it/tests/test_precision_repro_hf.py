# SPDX-License-Identifier: Apache-2.0
"""CPU-only tiny HF model checks cache conditioning, not checkpoint accuracy."""

import json
import time
from types import SimpleNamespace

import torch
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

from models.autoports.google_gemma_4_31b_it.tools.precision_repro_hf import run_case


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
