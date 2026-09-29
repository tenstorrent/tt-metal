"""CPU full-model HF final logits for repeated-BOS and encode-once stress inputs."""

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "IFM/K2-Horizon-7B"
REVISION = "036114ce8d46c32b24c15423211069abb9c5d25e"
DOC = Path("models/autoports/ifm_k2_horizon_7b/doc/full_model")
RAW = Path("bringup/artifacts/ifm_k2_full_model_raw_20260928")
SENTENCE = "A careful scientist checks the evidence before drawing a conclusion. "


def source(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=DOC / "stress_hf_topk.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=True, local_files_only=True)
    full36_path = DOC / "continuation_l36_s31.json"
    full36 = json.loads(full36_path.read_text())
    splits = {row["split_position"]: row for row in full36["splits"]}
    predictions = {
        "tt_whole": splits[31]["full_greedy"],
        "tt_split31": splits[31]["split_greedy"],
        "tt_split32": splits[32]["split_greedy"],
    }
    assert full36["layers"] == 36
    assert predictions == {"tt_whole": 589, "tt_split31": 18, "tt_split32": 222}
    assert splits[32]["full_greedy"] == predictions["tt_whole"]
    stress = (tok.encode(SENTENCE) * 30)[:257]
    normal = tok.encode(SENTENCE * 30)[:257]
    reference_path = RAW / "hf_continuation_l3.pt"
    saved = torch.load(reference_path, map_location="cpu", weights_only=True)
    assert saved["prompt_tokens"].tolist() == [stress], "Exact historical stress prompt mismatch"
    del saved
    normal_path = DOC / "continuation_l3_s31_normal.json"
    assert json.loads(normal_path.read_text())["prompt_tokens"] == [normal]
    assert len(stress) == len(normal) == 257
    assert tok.bos_token_id not in normal[1:]
    metadata = {
        "hf_model": MODEL,
        "revision": REVISION,
        "layers": 36,
        "tokenizer_class": type(tok).__name__,
        "chat_template_present": bool(tok.chat_template),
        "prompt_mode": "raw continuation stress diagnostic; no chat-template quality verdict",
        "construction": {
            "repeated_bos": "(tokenizer.encode(sentence) * 30)[:257]",
            "encode_once": "tokenizer.encode(sentence * 30)[:257]",
            "sentence": SENTENCE,
        },
        "sources": {
            "tt_full36": source(full36_path),
            "historical_exact_prompt": source(reference_path),
            "normal_prompt": source(normal_path),
            "probe_recipe": source(Path("models/autoports/ifm_k2_horizon_7b/tests/probe_full_continuation.py")),
        },
        "device": "cpu",
        "threads": args.threads,
        "interop_threads": 1,
        "offline": True,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "forward": {
            "batch_size": 2,
            "sequence_length": 257,
            "use_cache": False,
            "logits_to_keep": 1,
            "attention_mask": "all ones",
            "generation": False,
        },
        "rank_definition": "1-based, descending logit with token ID ascending on ties; rank_strict counts only strictly larger logits",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": f"python -m models.autoports.ifm_k2_horizon_7b.tests.hf_stress_logits --threads {args.threads}",
    }
    print(json.dumps({"event": "load", "metadata": metadata}), flush=True)
    start = time.perf_counter()
    model = (
        AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, trust_remote_code=True, local_files_only=True)
        .eval()
        .to("cpu")
    )
    assert model.config.num_hidden_layers == len(model.model.layers) == 36
    metadata["model_dtype"] = str(next(model.parameters()).dtype)
    metadata["attention_backend"] = model.config._attn_implementation
    metadata["load_seconds"] = time.perf_counter() - start
    batch = torch.tensor([stress, normal], dtype=torch.int64)
    start = time.perf_counter()
    with torch.no_grad():
        logits = (
            model(batch, attention_mask=torch.ones_like(batch), use_cache=False, logits_to_keep=1)
            .logits[:, -1, :]
            .float()
        )
    metadata["forward_seconds"] = time.perf_counter() - start
    records = []
    for name, ids, values in zip(("repeated_bos", "encode_once"), (stress, normal), logits):
        assert torch.isfinite(values).all()
        ordered = torch.argsort(values, descending=True, stable=True)
        ranks = torch.empty_like(ordered)
        ranks[ordered] = torch.arange(1, len(ordered) + 1)
        probabilities = values.softmax(-1)

        def entry(token_id):
            return {
                "id": token_id,
                "text": tok.decode([token_id], skip_special_tokens=False),
                "token": tok.convert_ids_to_tokens(token_id),
                "logit": float(values[token_id]),
                "rank": int(ranks[token_id]),
                "rank_strict": int((values > values[token_id]).sum()) + 1,
                "rank_tie_end": int((values >= values[token_id]).sum()),
                "margin_to_top1": float(values[ordered[0]] - values[token_id]),
                "probability": float(probabilities[token_id]),
            }

        record = {
            "id": name,
            "prompt_token_ids": ids,
            "bos_token_count": ids.count(tok.bos_token_id),
            "prompt_tail_token_ids": ids[-40:],
            "prompt_tail_text": tok.decode(ids[-40:], skip_special_tokens=False),
            "hf_top1": entry(int(ordered[0])),
            "tt_predictions_ranked_in_hf": {label: entry(token_id) for label, token_id in predictions.items()},
            "hf_top100": [entry(token_id) for token_id in ordered[:100].tolist()],
        }
        records.append(record)
        print(
            json.dumps(
                {
                    "event": "result",
                    **{key: value for key, value in record.items() if key not in ("prompt_token_ids", "hf_top100")},
                }
            ),
            flush=True,
        )
    metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"metadata": metadata, "records": records}, indent=2) + "\n")
    print(
        json.dumps({"event": "complete", "output": str(args.output), "forward_seconds": metadata["forward_seconds"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
