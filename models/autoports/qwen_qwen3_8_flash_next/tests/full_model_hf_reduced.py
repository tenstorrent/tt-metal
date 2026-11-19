# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only exact HF reduced wrapper fixture; never full-model acceptance.

Execute original HF embedding, layers 0/1/3, terminal hyper-connection mixer,
LM head and cache semantics. Weights remain mmap views; PLE reads exact selected
checkpoint rows with quantization disabled. No TTNN import or device execution.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM, Qwen4ExpTextRotaryEmbedding

from models.autoports.qwen_qwen3_8_flash_next.tests.full_model_hf_reference import MODEL_ID, REVISION
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_ple_quantization import SelectedRows

LAYERS = (0, 1, 3)
MODEL_DIR = Path(__file__).resolve().parents[1]


def load_reduced(snapshot):
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True).text_config
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = Qwen4ExpForCausalLM(config)
    model.model.layers = torch.nn.ModuleList([model.model.layers[i] for i in LAYERS])
    selected = {}
    for key, filename in index.items():
        if key == "lm_head.weight":
            selected[key] = (filename, key)
        elif key.startswith("model.language_model.layers."):
            original = int(key.split(".")[3])
            if original in LAYERS and ".ngram_embedding.shard_" not in key:
                suffix = key.split(".", 4)[4]
                selected[key] = (filename, f"model.layers.{LAYERS.index(original)}.{suffix}")
        elif key.startswith("model.language_model."):
            target = "model." + key.removeprefix("model.language_model.")
            if target in model.state_dict():
                selected[key] = (filename, target)
    state = {}
    for filename in sorted({pair[0] for pair in selected.values()}):
        with safe_open(snapshot / filename, framework="pt") as handle:
            for source, (file, target) in selected.items():
                if file == filename:
                    state[target] = handle.get_tensor(source)
    result = model.load_state_dict(state, assign=True, strict=False)
    expected_missing = [
        f"model.layers.{j}.ple.ple_embedding.ngram_embedding.weight"
        for j, layer in enumerate(model.model.layers)
        if layer.ple is not None
    ]
    if sorted(result.missing_keys) != sorted(expected_missing) or result.unexpected_keys:
        raise RuntimeError(f"Unexpected reduced load result: {result}")
    for original, layer in zip(LAYERS, model.model.layers):
        if layer.ple is not None:
            prefix = f"model.language_model.layers.{original}.ple.ple_embedding.ngram_embedding."
            rows = SelectedRows(snapshot, index, prefix, method="absmax")
            rows.quantized = False
            layer.ple.ple_embedding.ngram_embedding = rows
    model.model.rotary_emb = Qwen4ExpTextRotaryEmbedding(config)
    meta = [name for name, x in list(model.named_parameters()) + list(model.named_buffers()) if x.is_meta]
    if meta:
        raise RuntimeError(f"Unmaterialized tensors: {meta}")
    return model.eval(), sum(x.numel() * x.element_size() for x in state.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path("/home/ttuser/qwen38-flash/hf/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/" + REVISION),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("/home/ttuser/qwen38-flash/reference/full_model_reduced.pt")
    )
    parser.add_argument("--provenance", type=Path, default=MODEL_DIR / "doc/full_model/hf_reduced_reference.json")
    parser.add_argument("--decode-inputs", type=int, nargs=3, default=[785, 13458, 13])
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    started = time.monotonic()
    model, weight_bytes = load_reduced(args.snapshot)
    print("HF_REDUCED_LOADED", flush=True)
    captured = {}

    def save_hidden(name):
        def hook(module, inputs, output):
            captured[name] = output.detach().clone()

        return hook

    model.model.embed_tokens.register_forward_hook(save_hidden("embedding"))
    for original, layer in zip(LAYERS, model.model.layers):
        layer.register_forward_hook(save_hidden(f"layer_{original}"))
    model.model.hyper_connection_mixer.register_forward_hook(save_hidden("terminal_hidden"))
    steps = []
    cache = None
    calls = [[785, 13458, 13]] + [[x] for x in args.decode_inputs]
    position = 0
    with torch.inference_mode():
        for step, ids in enumerate(calls):
            call_started = time.monotonic()
            captured.clear()
            tokens = torch.tensor([ids], dtype=torch.long)
            positions = torch.arange(position, position + len(ids)).unsqueeze(0)
            output = model(input_ids=tokens, position_ids=positions, past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            logits = output.logits.detach().clone()
            if not torch.isfinite(logits).all():
                raise RuntimeError(f"Nonfinite HF logits at step {step}")
            top_values, top_ids = torch.topk(logits.float(), 100, dim=-1)
            steps.append(
                dict(
                    step=step,
                    mode="prefill" if step == 0 else "teacher_forced_decode",
                    input_ids=tokens,
                    positions=positions,
                    logits=logits,
                    top100_ids=top_ids,
                    top100_logits=top_values,
                    hidden_states=dict(captured),
                )
            )
            position += len(ids)
            print(
                json.dumps(dict(step=step, seconds=time.monotonic() - call_started, top1=logits.argmax(-1).tolist())),
                flush=True,
            )
    report = dict(
        hf_model=MODEL_ID,
        revision=REVISION,
        snapshot=str(args.snapshot),
        layers=list(LAYERS),
        scope="reduced real-layer wrapper diagnostic only; not full-model acceptance",
        reference_class=type(model).__name__,
        original_layer_cache_indices=True,
        prefill_ids=calls[0],
        teacher_forced_decode_ids=args.decode_inputs,
        attention_implementation=model.config._attn_implementation,
        weight_policy="original checkpoint dtype mmap views; no casts or quantization",
        ple_policy="SelectedRows.quantized=False; exact checkpoint BF16 rows; original HF PLE maths",
        terminal="original HF hyper_connection_mixer and lm_head; no extra final norm",
        checkpoint_weight_bytes_mmap=weight_bytes,
        cpu_threads=4,
        output=str(args.output),
        command=[sys.executable, *sys.argv],
        wall_seconds=time.monotonic() - started,
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        steps=[
            dict(
                step=x["step"],
                mode=x["mode"],
                positions=x["positions"].tolist(),
                logits_shape=list(x["logits"].shape),
                top1=x["logits"].argmax(-1).tolist(),
                hidden_shapes={k: list(v.shape) for k, v in x["hidden_states"].items()},
            )
            for x in steps
        ],
        passed=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(metadata=report, steps=steps), args.output)
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(passed=True, output=str(args.output), seconds=report["wall_seconds"])), flush=True)


if __name__ == "__main__":
    main()
