"""CPU-only full-depth pinned HF capture and saved TT residual comparison."""

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

from .hf_qualitative import MODEL, REVISION

RAW = Path("bringup/artifacts/ifm_k2_full_model_raw_20260928")
DOC = Path("models/demos/k2_horizon_7b_qb2/doc/full_model/precision")
SENTENCE = "A careful scientist checks the evidence before drawing a conclusion. "


def source(path):
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest}


def metrics(reference, candidate):
    a, b = reference.float().flatten(), candidate.float().flatten()
    assert a.shape == b.shape
    return {
        "pcc": float(torch.corrcoef(torch.stack([a, b]))[0, 1]),
        "relative_l2": float((a - b).norm() / a.norm().clamp_min(1e-30)),
        "reference_norm": float(a.norm()),
        "candidate_norm": float(b.norm()),
        "finite": bool(torch.isfinite(b).all()),
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--raw", type=Path, default=RAW / "hf_continuation_l36.pt")
    parser.add_argument("--tt-artifact", type=Path, default=RAW / "full36_early3_bfp8_hifi2_fp32_m4.pt")
    parser.add_argument("--tt-metadata", type=Path, default=DOC / "full36_early3_bfp8_hifi2_fp32_m4.json")
    parser.add_argument("--output", type=Path, default=DOC / "hf_full36_early3_comparison.json")
    args = parser.parse_args()
    if not 1 <= args.threads <= 4:
        parser.error("Use one to four CPU threads")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    historical_path = RAW / "hf_continuation_l3.pt"
    historical = torch.load(historical_path, map_location="cpu", weights_only=True)
    tokens = historical["prompt_tokens"].clone()
    tt_metadata = json.loads(args.tt_metadata.read_text())
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=True, local_files_only=True)
    assert tokens.tolist() == [(tokenizer.encode(SENTENCE) * 30)[:257]]
    assert tokens.tolist() == [tt_metadata["prompt_token_ids"]]
    assert tuple(tokens.shape) == (1, 257)
    capture_metadata = {
        "model": MODEL,
        "revision": REVISION,
        "device": "cpu",
        "threads": args.threads,
        "interop_threads": 1,
        "layers": 36,
        "offline": True,
        "torch_version": str(torch.__version__),
        "transformers_version": transformers.__version__,
        "prompt_construction": "(tokenizer.encode(sentence) * 30)[:257]",
        "sentence": SENTENCE,
        "bos_positions": torch.where(tokens[0] == tokenizer.bos_token_id)[0].tolist(),
        "historical_reference": source(historical_path),
        "runner": source(Path(__file__)),
        "forward": {"batch_size": 1, "sequence_length": 257, "use_cache": False, "logits_to_keep": 1},
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if args.raw.exists():
        saved = torch.load(args.raw, map_location="cpu", weights_only=True)
        assert torch.equal(saved["prompt_tokens"], tokens)
        assert set(saved["layer_outputs"]) == set(range(36))
        assert saved["metadata"]["revision"] == REVISION
        capture_metadata = saved["metadata"]
        reused = True
    else:
        print(json.dumps({"event": "load", "model": MODEL, "revision": REVISION}), flush=True)
        start = time.perf_counter()
        model = (
            AutoModelForCausalLM.from_pretrained(
                MODEL, revision=REVISION, trust_remote_code=True, local_files_only=True
            )
            .eval()
            .to("cpu")
        )
        capture_metadata["load_seconds"] = time.perf_counter() - start
        assert model.config.num_hidden_layers == len(model.model.layers) == 36
        assert next(model.parameters()).dtype == torch.bfloat16
        assert model.config._attn_implementation == "sdpa"
        capture_metadata["model_dtype"] = str(next(model.parameters()).dtype)
        capture_metadata["attention_backend"] = model.config._attn_implementation
        outputs, hooks = {}, []
        for index, layer in enumerate(model.model.layers):

            def capture(module, inputs, output, index=index):
                value = output[0] if isinstance(output, tuple) else output
                outputs[index] = value.detach().cpu().clone()

            hooks.append(layer.register_forward_hook(capture))
        start = time.perf_counter()
        final_logits = model(tokens, attention_mask=torch.ones_like(tokens), use_cache=False, logits_to_keep=1).logits
        capture_metadata["forward_seconds"] = time.perf_counter() - start
        for hook in hooks:
            hook.remove()
        assert set(outputs) == set(range(36))
        assert all(tuple(value.shape) == (1, 257, 4096) and torch.isfinite(value).all() for value in outputs.values())
        assert torch.isfinite(final_logits).all()
        assert int(final_logits.flatten().argmax()) == 20678
        capture_metadata["historical_first_three_bitwise_equal"] = {
            str(index): torch.equal(outputs[index], historical["layer_outputs"][index]) for index in range(3)
        }
        assert all(capture_metadata["historical_first_three_bitwise_equal"].values())
        capture_metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        saved = {
            "prompt_tokens": tokens,
            "layer_outputs": outputs,
            "final_logits": final_logits.detach().cpu().clone(),
            "metadata": capture_metadata,
        }
        args.raw.parent.mkdir(parents=True, exist_ok=True)
        torch.save(saved, args.raw)
        del model
        reused = False
        print(
            json.dumps({"event": "captured", "raw": str(args.raw), "seconds": capture_metadata["forward_seconds"]}),
            flush=True,
        )
    tt = torch.load(args.tt_artifact, map_location="cpu", weights_only=True)
    comparisons = {}
    bos = torch.tensor(capture_metadata["bos_positions"])
    for mode in ("whole", "split31", "split32"):
        layers = []
        for index in range(36):
            reference = saved["layer_outputs"][index].reshape(257, 4096)
            actual = tt["records"][mode][index].reshape(-1, 4096)
            if mode.startswith("split"):
                cut = mode[len("split") :]
                prefix = tt["records"]["prefix" + cut][index].reshape(-1, 4096)
                actual = torch.cat([prefix, actual])
            assert actual.shape == reference.shape
            a, b = reference.float(), actual.float()
            row_l2 = (a - b).norm(dim=-1) / a.norm(dim=-1).clamp_min(1e-30)
            layers.append(
                {
                    "layer": index,
                    "last": metrics(reference[-1], actual[-1]),
                    "all_rows": metrics(reference, actual),
                    "bos_median_relative_l2": float(row_l2[bos].median()),
                    "bos_max_relative_l2": float(row_l2[bos].max()),
                }
            )
        comparisons[mode] = {
            "first_layer_last_pcc_below_0_995": next((v["layer"] for v in layers if v["last"]["pcc"] < 0.995), None),
            "first_layer_after_2_last_pcc_below_0_995": next(
                (v["layer"] for v in layers[3:] if v["last"]["pcc"] < 0.995), None
            ),
            "layers": layers,
        }
    logits = saved["final_logits"].float().flatten()
    top_values, top_ids = logits.topk(10)
    result = {
        "capture": capture_metadata,
        "raw_hf_artifact": source(args.raw),
        "raw_tt_artifact": source(args.tt_artifact),
        "tt_metadata": source(args.tt_metadata),
        "reused_hf_capture": reused,
        "diagnostic_pcc_threshold": 0.995,
        "threshold_note": "Localization marker only; no accepted gate is changed.",
        "reference": "Pinned BF16 HF on the exact repeated-BOS 257-token fixture",
        "hf_top10": [
            {"id": int(i), "text": tokenizer.decode([i]), "logit": float(v)} for i, v in zip(top_ids, top_values)
        ],
        "hf_top1_probability": float(logits.softmax(-1)[int(top_ids[0])]),
        "logit_comparisons": {
            "whole": metrics(saved["final_logits"], tt["whole_logits"]),
            "split32": metrics(saved["final_logits"], tt["last_split_logits"]),
            "split31": {
                "available": False,
                "reason": "Saved TT artifact retains only last split logits; per-layer split31 residuals are available.",
            },
        },
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "event": "complete",
                "output": str(args.output),
                "first_downstream_divergence": {
                    mode: rows["first_layer_after_2_last_pcc_below_0_995"] for mode, rows in comparisons.items()
                },
                "logits": result["logit_comparisons"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
