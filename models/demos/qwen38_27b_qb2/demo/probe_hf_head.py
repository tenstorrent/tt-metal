# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU reference diagnostic for LM-head quantization, not an evaluation score."""

import argparse
import hashlib
import json
import time
from pathlib import Path


def report_json_default(value):
    """Transformers loading diagnostics may use sets for missing/unexpected keys."""
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"Unsupported reference report value: {type(value).__name__}")


def save_report(output, report):
    payload = json.dumps(report, indent=2, default=report_json_default, allow_nan=False) + "\n"
    temporary = output / "progress.json.tmp"
    temporary.write_text(payload)
    temporary.replace(output / "progress.json")


def main(args):
    import torch
    import transformers
    from transformers import Qwen3_5ForConditionalGeneration

    import ttnn

    args.output.mkdir()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    receipt = json.loads(args.qualification.read_text())
    prompt = receipt["prompt_tokens"]
    if not prompt or any(type(token) is not int or token < 0 for token in prompt):
        raise ValueError("Qualification must provide a nonempty public token prompt")
    report = dict(
        state="loading_reference",
        started_at=time.time(),
        scope="CPU BF16 HF hidden states and FP32 LM-head matmuls with host BFP4/BFP8 weight round trips",
        checkpoint=str(args.weights),
        checkpoint_config_sha256=hashlib.sha256((args.weights / "config.json").read_bytes()).hexdigest(),
        qualification_sha256=hashlib.sha256(args.qualification.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        torch_version=torch.__version__,
        transformers_version=transformers.__version__,
        threads=args.threads,
        prompt_tokens=len(prompt),
        requested_steps=args.steps,
        reference_steps=[],
        head_formats={},
        hardware_opened=False,
        includes_lofi_device_math=False,
        is_gpqa_score=False,
    )

    def save():
        save_report(args.output, report)

    save()
    try:
        model, loading = Qwen3_5ForConditionalGeneration.from_pretrained(
            args.weights,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            local_files_only=True,
            output_loading_info=True,
        )
        report["loading_info"] = loading
        if any(loading.get(name) for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError("HF reference did not load the exact checkpoint cleanly")
        model.eval()
        if any(parameter.device.type != "cpu" for parameter in model.parameters()):
            raise ValueError("This probe must execute on CPU only")
        report.update(state="reference_forward", loaded_at=time.time())
        save()
        tokens = torch.tensor([prompt], dtype=torch.int64)
        cache, hidden_rows, golden = None, [], []
        with torch.inference_mode():
            for step in range(args.steps):
                started = time.monotonic()
                result = model.model.language_model(
                    input_ids=tokens, past_key_values=cache, use_cache=True, output_hidden_states=True, return_dict=True
                )
                cache = result.past_key_values
                hidden = result.last_hidden_state[:, -1].clone()
                logits = model.lm_head(hidden).float()
                token = int(logits.argmax(-1).item())
                if not torch.isfinite(logits).all():
                    raise ValueError("Non-finite HF reference logits")
                hidden_rows.append(hidden)
                golden.append(
                    dict(
                        step=step,
                        input_ids=tokens.clone(),
                        layer_last_hidden=[state[:, -1].clone() for state in result.hidden_states],
                        logits=logits.clone(),
                    )
                )
                report["reference_steps"].append(dict(step=step, token=token, elapsed_s=time.monotonic() - started))
                save()
                tokens = torch.tensor([[token]], dtype=torch.int64)
                del result, logits
            torch.save(golden, args.output / "reference.pt")
            del golden, cache
            hidden = torch.cat(hidden_rows).float()
            weight = model.lm_head.weight.detach().T.contiguous()
            reference = hidden @ weight.float()
            log_reference = reference.log_softmax(-1)
            probs = log_reference.exp()
            top20 = reference.topk(20, dim=-1).indices
            report.update(
                state="head_quantization",
                reference_tensor_sha256=hashlib.sha256((args.output / "reference.pt").read_bytes()).hexdigest(),
            )
            save()
            for name in ("bfloat4_b", "bfloat8_b"):
                # The TP4 vocabulary shards are tile aligned, so this uses the
                # same [hidden,vocabulary] tile/block orientation as upload().
                packed = ttnn.from_torch(weight, dtype=getattr(ttnn, name), layout=ttnn.TILE_LAYOUT)
                restored = ttnn.to_torch(packed).float()
                actual = hidden @ restored
                if not torch.isfinite(actual).all():
                    raise ValueError("Non-finite quantized-head logits")
                diff = actual - reference
                actual_top20 = actual.topk(20, dim=-1).indices
                overlap = [len(set(left.tolist()) & set(right.tolist())) for left, right in zip(top20, actual_top20)]
                report["head_formats"][name] = dict(
                    relative_logit_rms=float(diff.square().mean().sqrt() / reference.square().mean().sqrt()),
                    mean_kl_from_reference=float((probs * (log_reference - actual.log_softmax(-1))).sum(-1).mean()),
                    top1_matches=int((actual.argmax(-1) == reference.argmax(-1)).sum()),
                    rows=hidden.shape[0],
                    top20_overlap_per_row=overlap,
                    max_absolute_logit_error=float(diff.abs().max()),
                )
                save()
                del packed, restored, actual, diff
        report.update(
            state="completed",
            finished_at=time.time(),
            interpretation="Head-only weight sensitivity on a short public prompt; decoder weights are HF BF16. This excludes device LoFi/HiFi2 arithmetic and cannot predict a GPQA score or establish its failure cause.",
        )
    except BaseException as error:
        report.update(state="failed", error=type(error).__name__, detail=str(error)[:2000], finished_at=time.time())
        raise
    finally:
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("weights", "qualification", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--steps", type=int, default=8)
    main(parser.parse_args())
