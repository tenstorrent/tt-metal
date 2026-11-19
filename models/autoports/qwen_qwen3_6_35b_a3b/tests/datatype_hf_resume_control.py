# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Matched CPU HF BF16 control for saved TT resumed-prefill logits; no waiver inference."""

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen3.6-35B-A3B"
REVISION = "345d467b117d1fd5eda9b879a5988de9ae5f47a7"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    assert "ttnn" not in sys.modules
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "result.json"
    assert not output.exists(), "Choose a fresh matched-control output directory"
    diagnostic = json.loads(args.diagnostic.read_text())
    assert diagnostic["gate_modified"] is False
    tokens, positions = diagnostic["tokens"], diagnostic["positions"]
    start = diagnostic["gate_result"]["start"]
    packaged = Path(os.environ["TT_MODEL_BRINGUP_ROOT"]) / "runtime/readiness_check/run_prefill_resume_check.py"
    helper_path = ROOT / "doc/full_model/qualitative/hf_resume_control/run_control.py"
    spec = importlib.util.spec_from_file_location("original_hf_resume_control", helper_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    helpers, helper_sha = module.checker_functions(packaged)
    thresholds = dict(pcc_mean_min=0.99, pcc_min_min=0.98, top1_min=0.98, control_min_window=64)
    assert positions == helpers["window_positions"](start, len(tokens), 256)
    tensor_path = Path(diagnostic["tensor_path"])
    tt = torch.load(tensor_path, map_location="cpu", weights_only=True)
    snapshot = Path(os.environ["HF_HOME"]) / "hub/models--Qwen--Qwen3.6-35B-A3B/snapshots" / REVISION
    result = dict(
        status="running",
        model=MODEL,
        revision=REVISION,
        checkpoint=str(snapshot),
        command=sys.argv,
        time_unix=time.time(),
        length=len(tokens),
        start=start,
        positions=positions,
        tokens=tokens,
        thresholds=thresholds,
        gate_modified=False,
        waiver_inferred=False,
        device="cpu",
        threads=8,
        torch=torch.__version__,
        transformers=transformers.__version__,
        prompt_mode="Exact saved natural-book continuation stress; no chat quality or generation judgment",
        source_sha256={
            str(p): sha(p)
            for p in (Path(__file__), helper_path, packaged, args.diagnostic, tensor_path, snapshot / "config.json")
        },
        numeric_helpers_ast_sha256=helper_sha,
        tt_runtime_precision=diagnostic["runtime_precision"],
        tt_original_gate_result=diagnostic["gate_result"],
        forwards=[],
    )

    def save():
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    model = None
    try:
        print("HF_MATCHED_LOADING", MODEL, REVISION, flush=True)
        model = AutoModelForCausalLM.from_pretrained(str(snapshot), local_files_only=True, dtype=torch.bfloat16).eval()
        assert next(model.parameters()).device.type == "cpu"
        result.update(
            model_class=type(model).__name__,
            model_dtype=str(model.dtype),
            attention_implementation=model.config._attn_implementation,
        )
        source = Path(sys.modules[type(model).__module__].__file__)
        result["source_sha256"][str(source)] = sha(source)
        save()

        def forward(ids, keep, label, cache=None, mask_length=None):
            torch.manual_seed(0)
            state = torch.random.get_rng_state().clone()
            began = time.monotonic()
            with torch.inference_mode():
                out = model(
                    input_ids=torch.tensor([ids]),
                    attention_mask=torch.ones((1, mask_length or len(ids)), dtype=torch.long),
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=keep,
                )
            assert torch.equal(state, torch.random.get_rng_state()), "Deterministic forward consumed RNG"
            assert torch.isfinite(out.logits).all()
            result["forwards"].append(
                dict(
                    label=label,
                    elapsed_s=time.monotonic() - began,
                    rng_unchanged=True,
                    cache_length=out.past_key_values.get_seq_length(),
                )
            )
            save()
            print("HF_MATCHED_FORWARD", label, result["forwards"][-1], flush=True)
            return out

        full = forward(tokens, torch.tensor(positions), "fresh")
        assert full.past_key_values.get_seq_length() == len(tokens)
        hf_fresh = full.logits.detach().cpu().clone()
        del full
        gc.collect()
        prefix = forward(tokens[:start], 1, "prefix")
        assert prefix.past_key_values.get_seq_length() == start
        resumed = forward(
            tokens[start:],
            torch.tensor([p - start for p in positions]),
            "resumed",
            cache=prefix.past_key_values,
            mask_length=len(tokens),
        )
        assert resumed.past_key_values.get_seq_length() == len(tokens)
        hf_resumed = resumed.logits.detach().cpu().clone()
        torch.save(
            {"fresh_logits": hf_fresh, "resumed_logits": hf_resumed, "positions": positions},
            args.output_dir / "hf_logits.pt",
        )
        del prefix, resumed, model
        model = None
        gc.collect()
        paths = {
            "tt_fresh": tt["fresh_logits"],
            "tt_resumed": tt["resumed_logits"],
            "hf_fresh": hf_fresh,
            "hf_resumed": hf_resumed,
        }
        assert len({tuple(x.shape) for x in paths.values()}) == 1
        pairs = [
            ("tt_fresh", "tt_resumed"),
            ("hf_fresh", "hf_resumed"),
            ("hf_fresh", "tt_fresh"),
            ("hf_fresh", "tt_resumed"),
            ("hf_resumed", "tt_fresh"),
            ("hf_resumed", "tt_resumed"),
        ]
        result["comparisons"] = {}
        for left, right in pairs:
            stats = helpers["compare_logits"](paths[left], paths[right])
            reasons = helpers["evaluate_case"](stats, thresholds)
            result["comparisons"][left + "_vs_" + right] = dict(
                metrics=stats, reasons=reasons, passes_same_numeric_thresholds=not reasons
            )
        rows = []
        for i, position in enumerate(positions):
            vectors = {key: value[0, i].float() for key, value in paths.items()}
            tops = {key: int(value.argmax()) for key, value in vectors.items()}
            row = dict(
                position=position,
                predicts_token_position=position + 1,
                top1=tops,
                top1_text={key: tokenizer.decode([token], skip_special_tokens=False) for key, token in tops.items()},
                pcc={left + "_vs_" + right: helpers["_pcc"](vectors[left], vectors[right]) for left, right in pairs},
                top1_agreement={left + "_vs_" + right: tops[left] == tops[right] for left, right in pairs},
                hf_margins={},
            )
            for hf_key in ("hf_fresh", "hf_resumed"):
                scores = vectors[hf_key]
                values, indices = scores.topk(2)
                row["hf_margins"][hf_key] = dict(
                    top2_ids=indices.tolist(),
                    top2_logits=values.tolist(),
                    top1_minus_top2=float(values[0] - values[1]),
                    comparison_tokens={
                        key: dict(
                            token_id=token,
                            rank_1based=int((scores > scores[token]).sum()) + 1,
                            hf_top1_minus_token=float(values[0] - scores[token]),
                        )
                        for key, token in tops.items()
                    },
                )
            rows.append(row)
        result["per_position"] = rows
        result["tt_mismatch_positions"] = [
            r["position"] for r in rows if not r["top1_agreement"]["tt_fresh_vs_tt_resumed"]
        ]
        result["hf_mismatch_positions"] = [
            r["position"] for r in rows if not r["top1_agreement"]["hf_fresh_vs_hf_resumed"]
        ]
        result["tt_below_min_pcc_positions"] = [
            r["position"] for r in rows if r["pcc"]["tt_fresh_vs_tt_resumed"] < thresholds["pcc_min_min"]
        ]
        result["hf_below_min_pcc_positions"] = [
            r["position"] for r in rows if r["pcc"]["hf_fresh_vs_hf_resumed"] < thresholds["pcc_min_min"]
        ]
        result["hf_logits_artifact"] = str(args.output_dir / "hf_logits.pt")
        result["status"] = "complete"
        assert "ttnn" not in sys.modules
        print("HF_MATCHED_COMPARISONS", json.dumps(result["comparisons"]), flush=True)
    except BaseException:
        result["status"] = "error"
        result["error"] = traceback.format_exc()
        raise
    finally:
        result["elapsed_seconds"] = time.time() - result["time_unix"]
        save()
    print("HF_MATCHED_DONE", str(output), flush=True)


if __name__ == "__main__":
    main()
