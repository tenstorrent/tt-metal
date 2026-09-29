"""CPU-only short-prompt HF residual capture and saved TT precision comparison."""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import create_causal_mask

from .hf_conditional_quality import MODEL, REVISION

DOC = Path("models/autoports/ifm_k2_horizon_7b/doc/full_model")
RAW = Path("bringup/artifacts/ifm_k2_full_model_raw_20260928")


def source(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def describe(logits, tokenizer, observed):
    values = logits.float().flatten()
    top = values.topk(10)
    probabilities = values.softmax(-1)
    return {
        "greedy_argmax_id": int(values.argmax()),
        "greedy_argmax_text": tokenizer.decode([int(values.argmax())]),
        "top10": [
            {
                "id": int(i),
                "text": tokenizer.decode([int(i)]),
                "logit": float(v),
                "probability": float(probabilities[i]),
            }
            for i, v in zip(top.indices, top.values)
        ],
        "top1_top2_logit_margin": float(top.values[0] - top.values[1]),
        "observed_first_tokens": {
            label: {
                "id": int(i),
                "text": tokenizer.decode([int(i)]),
                "rank_min": int((values > values[i]).sum()) + 1,
                "rank_max": int((values >= values[i]).sum()),
                "logit": float(values[i]),
                "gap_from_top1": float(top.values[0] - values[i]),
            }
            for label, i in observed.items()
        },
    }


def metrics(reference, candidate):
    a, b = reference.float().flatten(), candidate.float().flatten()
    assert a.shape == b.shape
    return {
        "pcc": float(torch.corrcoef(torch.stack([a, b]))[0, 1]),
        "relative_l2": float((a - b).norm() / a.norm().clamp_min(1e-30)),
    }


def hf_suffix(model, hidden, completed_layer):
    positions = torch.arange(hidden.shape[1]).unsqueeze(0)
    mask = create_causal_mask(
        config=model.config,
        inputs_embeds=hidden,
        attention_mask=torch.ones(hidden.shape[:2], dtype=torch.long),
        past_key_values=None,
        position_ids=positions,
    )
    rotary = model.model.rotary_emb(hidden, positions)
    for layer in model.model.layers[completed_layer + 1 :]:
        hidden = layer(
            hidden,
            position_embeddings=rotary,
            attention_mask=mask,
            position_ids=positions,
            cache_position=positions[0],
            use_cache=False,
        )
    return model.lm_head(model.model.norm(hidden)[:, -1:, :])


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--ids", nargs="+", default=["shared_1", "shared_3", "shared_4"])
    parser.add_argument("--raw", type=Path, default=RAW / "hf_quality_prefill_l36.pt")
    parser.add_argument("--tt-raw", nargs="*", type=Path, default=[])
    parser.add_argument("--same-input", action="store_true", help="Run HF layers on saved TT predecessor residuals")
    parser.add_argument("--suffix-checkpoints", nargs="*", type=int, default=[])
    parser.add_argument("--output", type=Path, default=DOC / "precision/hf_quality_prefill_l36.json")
    args = parser.parse_args()
    if not 1 <= args.threads <= 4:
        parser.error("Use one to four CPU threads")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    hf_path, tt_path = DOC / "hf_qualitative_extended.json", DOC / "qualitative_precision.json"
    hf_rows = json.loads(hf_path.read_text())
    hf = {row["id"]: row for row in hf_rows}
    tt = {row["id"]: row for row in json.loads(tt_path.read_text())}
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=True, local_files_only=True)
    for key in args.ids:
        ids = tok.apply_chat_template(hf[key]["messages"], tokenize=True, add_generation_prompt=True, return_dict=False)
        assert ids == hf[key]["prompt_token_ids"] == tt[key]["prompt_token_ids"]
    if args.raw.exists():
        saved = torch.load(args.raw, map_location="cpu", weights_only=True)
        assert saved["metadata"]["revision"] == REVISION
        assert set(args.ids) <= set(saved["records"])
    else:
        metadata = {
            "model": MODEL,
            "revision": REVISION,
            "device": "cpu",
            "threads": args.threads,
            "interop_threads": 1,
            "offline": True,
            "torch_version": str(torch.__version__),
            "transformers_version": transformers.__version__,
            "sources": {"hf": source(hf_path), "tt": source(tt_path), "runner": source(Path(__file__))},
            "command": "python_env/bin/python -u -m models.autoports.ifm_k2_horizon_7b.tests.hf_precision_quality_control "
            + " ".join(sys.argv[1:]),
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "capture_mode": "Serial batch-one unpadded exact chat prompts; no cache; final prompt-row logits",
        }
        started = time.perf_counter()
        model = (
            AutoModelForCausalLM.from_pretrained(
                MODEL, revision=REVISION, trust_remote_code=True, local_files_only=True
            )
            .eval()
            .to("cpu")
        )
        metadata["load_seconds"] = time.perf_counter() - started
        metadata["model_dtype"] = str(next(model.parameters()).dtype)
        metadata["attention_backend"] = model.config._attn_implementation
        assert next(model.parameters()).dtype == torch.bfloat16
        assert len(model.model.layers) == 36
        records = {}
        for key in args.ids:
            ids = torch.tensor([hf[key]["prompt_token_ids"]], dtype=torch.int64)
            outputs, hooks = {}, []
            for index, layer in enumerate(model.model.layers):

                def capture(module, inputs, output, index=index):
                    value = output[0] if isinstance(output, tuple) else output
                    outputs[index] = value.detach().cpu().clone()

                hooks.append(layer.register_forward_hook(capture))
            started = time.perf_counter()
            logits = model(ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=1).logits
            seconds = time.perf_counter() - started
            for hook in hooks:
                hook.remove()
            assert set(outputs) == set(range(36))
            assert all(tuple(value.shape) == (1, ids.shape[1], 4096) for value in outputs.values())
            generated = model.generate(
                ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
            assert int(generated[0, -1]) == int(logits.flatten().argmax())
            records[key] = {
                "prompt_token_ids": ids[0].tolist(),
                "layer_outputs": outputs,
                "final_logits": logits.detach().cpu().clone(),
                "forward_seconds": seconds,
                "one_step_generate_token": int(generated[0, -1]),
            }
            print(
                json.dumps(
                    {"event": "captured", "id": key, "seconds": seconds, "top1": tok.decode([int(generated[0, -1])])}
                ),
                flush=True,
            )
        # Recheck the independent control's batch-six left-masked generation shape.
        width = max(len(row["prompt_token_ids"]) for row in hf_rows)
        batch = torch.full((len(hf_rows), width), tok.eos_token_id, dtype=torch.int64)
        mask = torch.zeros_like(batch)
        for index, row in enumerate(hf_rows):
            length = len(row["prompt_token_ids"])
            batch[index, -length:] = torch.tensor(row["prompt_token_ids"])
            mask[index, -length:] = 1
        result = model.generate(
            batch,
            attention_mask=mask,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
            return_dict_in_generate=True,
            output_logits=True,
        )
        batch_logits = result.logits[0]
        metadata["batch6_first_token_recheck"] = {
            row["id"]: {
                "token_id": int(result.sequences[index, -1]),
                "matches_saved_control": int(result.sequences[index, -1]) == row["hf_token_ids"][0],
            }
            for index, row in enumerate(hf_rows)
        }
        for index, row in enumerate(hf_rows):
            if row["id"] in records:
                records[row["id"]]["batch6_first_logits"] = batch_logits[index].detach().cpu().clone()
        metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        saved = {"metadata": metadata, "records": records}
        args.raw.parent.mkdir(parents=True, exist_ok=True)
        torch.save(saved, args.raw)
        del model
    output = {"metadata": saved["metadata"], "raw": source(args.raw), "records": {}, "tt_comparisons": {}}
    for key in args.ids:
        row = saved["records"][key]
        observed = {"production_tt": tt[key]["tt_token_ids"][0], "independent_hf": hf[key]["hf_token_ids"][0]}
        output["records"][key] = {
            "prompt_length": len(row["prompt_token_ids"]),
            "forward_seconds": row["forward_seconds"],
            "one_step_generate_agrees": row["one_step_generate_token"] == int(row["final_logits"].flatten().argmax()),
            "serial_batch1": describe(row["final_logits"], tok, observed),
            "independent_shape_batch6": describe(row["batch6_first_logits"], tok, observed),
        }
    discriminator = None
    if args.same_input or args.suffix_checkpoints:
        started = time.perf_counter()
        discriminator = (
            AutoModelForCausalLM.from_pretrained(
                MODEL, revision=REVISION, trust_remote_code=True, local_files_only=True
            )
            .eval()
            .to("cpu")
        )
        output["same_input_metadata"] = {
            "runner": source(Path(__file__)),
            "load_seconds": time.perf_counter() - started,
            "command": "python_env/bin/python -u -m models.autoports.ifm_k2_horizon_7b.tests.hf_precision_quality_control "
            + " ".join(sys.argv[1:]),
            "interpretation": "HF layer on actual TT predecessor; local error versus that output, inherited error versus original HF output",
        }
        started = time.perf_counter()
        for key in args.ids:
            reference = saved["records"][key]
            positions = torch.arange(len(reference["prompt_token_ids"])).unsqueeze(0)
            for index in range(1, 36):
                hidden = reference["layer_outputs"][index - 1]
                mask = create_causal_mask(
                    config=discriminator.config,
                    inputs_embeds=hidden,
                    attention_mask=torch.ones(hidden.shape[:2], dtype=torch.long),
                    past_key_values=None,
                    position_ids=positions,
                )
                repeated = discriminator.model.layers[index](
                    hidden,
                    position_embeddings=discriminator.model.rotary_emb(hidden, positions),
                    attention_mask=mask,
                    position_ids=positions,
                    cache_position=positions[0],
                    use_cache=False,
                )
                assert torch.equal(repeated, reference["layer_outputs"][index]), (key, index)
        output["same_input_metadata"]["all_reference_layers_bitwise_reproduced"] = True
        for key in args.ids:
            for index in args.suffix_checkpoints:
                reference = saved["records"][key]
                repeated = hf_suffix(discriminator, reference["layer_outputs"][index], index)
                assert torch.equal(repeated, reference["final_logits"]), ("suffix", key, index)
        output["same_input_metadata"]["all_reference_suffix_logits_bitwise_reproduced"] = True
    for path in args.tt_raw:
        actual = torch.load(path, map_location="cpu", weights_only=True)
        comparison = {"source": source(path), "records": {}}
        for key in args.ids:
            reference, candidate = saved["records"][key], actual[key]
            assert reference["prompt_token_ids"] == candidate["prompt_token_ids"]
            length = len(reference["prompt_token_ids"])
            rows = []
            for index in range(36):
                ref = reference["layer_outputs"][index].reshape(-1, 4096)
                cand = candidate["layer_outputs"][index].reshape(-1, 4096)[:length]
                rows.append(
                    {
                        "layer": index,
                        "last": metrics(ref[-1], cand[-1]),
                        "bos": metrics(ref[0], cand[0]),
                        "all_rows": metrics(ref, cand),
                    }
                )
                if args.same_input and index > 0:
                    hidden = candidate["layer_outputs"][index - 1].reshape(-1, 4096)[:length].unsqueeze(0)
                    hidden = hidden.to(torch.bfloat16)
                    positions = torch.arange(length).unsqueeze(0)
                    mask = create_causal_mask(
                        config=discriminator.config,
                        inputs_embeds=hidden,
                        attention_mask=torch.ones(hidden.shape[:2], dtype=torch.long),
                        past_key_values=None,
                        position_ids=positions,
                    )
                    counterfactual = discriminator.model.layers[index](
                        hidden,
                        position_embeddings=discriminator.model.rotary_emb(hidden, positions),
                        attention_mask=mask,
                        position_ids=positions,
                        cache_position=positions[0],
                        use_cache=False,
                    )
                    counterfactual = counterfactual.reshape(length, 4096)
                    rows[-1]["same_input_local_last"] = metrics(counterfactual[-1], cand[-1])
                    rows[-1]["same_input_local_bos"] = metrics(counterfactual[0], cand[0])
                    rows[-1]["inherited_last"] = metrics(ref[-1], counterfactual[-1])
                    rows[-1]["inherited_bos"] = metrics(ref[0], counterfactual[0])
            comparison["records"][key] = {
                "layers": rows,
                "logits": metrics(reference["final_logits"], candidate["final_logits"]),
                "tt_logits": describe(
                    candidate["final_logits"],
                    tok,
                    {"independent_hf": hf[key]["hf_token_ids"][0], "production_tt": tt[key]["tt_token_ids"][0]},
                ),
            }
            if args.suffix_checkpoints:
                replay = {}
                for index in args.suffix_checkpoints:
                    hidden = (
                        candidate["layer_outputs"][index].reshape(-1, 4096)[:length].unsqueeze(0).to(torch.bfloat16)
                    )
                    logits = hf_suffix(discriminator, hidden, index)
                    replay[index] = {
                        "logits": metrics(reference["final_logits"], logits),
                        "distribution": describe(
                            logits,
                            tok,
                            {"independent_hf": hf[key]["hf_token_ids"][0], "production_tt": tt[key]["tt_token_ids"][0]},
                        ),
                    }
                comparison["records"][key]["hf_suffix_replay"] = replay
        output["tt_comparisons"][path.stem] = comparison
        print(json.dumps({"event": "compared", "path": str(path)}), flush=True)
    if discriminator is not None:
        output["same_input_metadata"]["forward_seconds"] = time.perf_counter() - started
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"event": "complete", "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
