# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from loguru import logger
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForCausalLM

from models.demos.strands_agent.tt.answers import build_answer
from models.demos.strands_agent.tt.head import load_config, load_head
from models.demos.strands_agent.tt.prompting import encode_request
from models.demos.strands_agent.tt.weight_mapping import (
    STRANDS_BASE_HF_MODEL,
    STRANDS_DECIDER_HF_MODEL,
    resolve_checkpoint,
)

HERE = Path(__file__).parent


def ids_sha256(ids):
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def load_example_state(example, ckpt_dir):
    if "state_file" in example:
        return (Path(ckpt_dir) / example["state_file"]).read_text()
    return example["state"]


def apply_lora(torso, lora_dir):
    """Runtime LoRA via peft (independent of our merge code); fall back to hooks. Returns (model, method)."""
    import peft

    try:
        return peft.PeftModel.from_pretrained(torso, str(lora_dir)).eval(), "peft.PeftModel.from_pretrained"
    except Exception as e:  # noqa: BLE001
        logger.warning(f"PeftModel.from_pretrained failed ({e!r}); trying filtered LoraConfig")
    cfg = json.loads((lora_dir / "adapter_config.json").read_text())
    known = {k: v for k, v in cfg.items() if k in peft.LoraConfig.__dataclass_fields__}
    try:
        from safetensors.torch import load_file

        model = peft.get_peft_model(torso, peft.LoraConfig(**known))
        peft.set_peft_model_state_dict(model, load_file(str(lora_dir / "adapter_model.safetensors")))
        return model.eval(), "peft.LoraConfig(filtered)+set_peft_model_state_dict"
    except Exception as e:  # noqa: BLE001
        logger.warning(f"filtered LoraConfig failed ({e!r}); using forward hooks")
    from safetensors.torch import load_file

    sd = load_file(str(lora_dir / "adapter_model.safetensors"))
    scale = cfg["lora_alpha"] / cfg["r"]
    for name, mod in torso.named_modules():
        a = sd.get(f"base_model.model.{name}.lora_A.weight")
        b = sd.get(f"base_model.model.{name}.lora_B.weight")
        if a is None or b is None or not isinstance(mod, torch.nn.Linear):
            continue
        mod.register_forward_hook(
            lambda m, inp, out, a=a.float(), b=b.float(): out + scale * (inp[0].float() @ a.T) @ b.T
        )
    return torso.eval(), "forward hooks"


def main():
    parser = argparse.ArgumentParser(description="Generate CPU fp32 reference outputs for strands-decider-2B.")
    parser.add_argument("--hf-model", default=os.environ.get("HF_MODEL", STRANDS_DECIDER_HF_MODEL))
    parser.add_argument("--base-model", default=os.environ.get("STRANDS_BASE_MODEL", STRANDS_BASE_HF_MODEL))
    parser.add_argument("--output", default=str(HERE / "reference" / "decider_reference.json"))
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    ckpt_dir = resolve_checkpoint(args.hf_model)
    base_dir = resolve_checkpoint(args.base_model)
    cfg = load_config(ckpt_dir)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    text_cfg = AutoConfig.from_pretrained(base_dir).get_text_config()
    lm = Qwen3_5ForCausalLM.from_pretrained(base_dir, config=text_cfg, dtype=torch.float32)
    torso, method = apply_lora(lm.model.eval(), ckpt_dir / "lora")
    logger.info(f"lora method: {method}")
    head = load_head(ckpt_dir, text_cfg.hidden_size)
    examples = json.loads((HERE / "examples.json").read_text())

    results, hidden_rows, tops = [], {}, []
    with torch.inference_mode():
        for example in examples:
            state = load_example_state(example, ckpt_dir)
            encoded = encode_request(tokenizer, state, example["questions"], cfg.max_length)
            per_question = {}
            for q in encoded:
                ids = torch.tensor(q.input_ids)[None]
                h = torso(input_ids=ids, use_cache=False).last_hidden_state[0].float()
                query, opts = h[q.query_position], h[q.option_positions]
                logits = head(query[None], opts[None])[0]
                answer = build_answer(q, logits, cfg.temperature_by_kind, cfg.temperature, cfg.ordinal_smoothing)
                n = len(q.labels)
                t = float(cfg.temperature_by_kind.get(q.kind, cfg.temperature))
                probs = torch.softmax(logits[:n] / t, dim=-1)
                per_question[q.name] = {
                    "kind": q.kind,
                    "num_tokens": len(q.input_ids),
                    "input_ids_sha256": ids_sha256(q.input_ids),
                    "query_position": q.query_position,
                    "option_positions": q.option_positions,
                    "labels": q.labels,
                    "logits": logits.tolist(),
                    "probabilities": probs.tolist(),
                    "answer": answer,
                }
                hidden_rows[f"{example['name']}/{q.name}"] = torch.cat([query[None], opts]).contiguous()
                top = int(probs.argmax())
                tops.append((example["name"], q.name, len(q.input_ids), q.labels[top], float(probs[top])))
            results.append({"name": example["name"], "questions": per_question})

    reference = {
        "base_model": STRANDS_BASE_HF_MODEL,
        "base_revision": base_dir.name,
        "adapter_model": STRANDS_DECIDER_HF_MODEL,
        "adapter_revision": ckpt_dir.name,
        "dtype": "float32",
        "lora_method": method,
        "max_length": cfg.max_length,
        "examples": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(reference, indent=2) + "\n")
    # fp16 keeps the file under the 500 KB pre-commit limit; values stay below ~130.
    save_file({k: v.half() for k, v in hidden_rows.items()}, str(output.with_name("decider_hidden.safetensors")))

    for ex, name, num_tokens, label, prob in tops:
        logger.info(f"{ex:<15}{name:<20}{num_tokens:>7}  {label:<16}{prob:>9.5f}")
    logger.info(f"wrote {output}")


if __name__ == "__main__":
    main()
