# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Matched HF and canonical-generator controls for the shared vLLM prompt suite.

The HF branch deliberately never imports TTNN. Run the TT branch only in a
serialized device job under a bounded timeout. Both branches generate the full
token budget, retain the raw sequence, and trim logical output inclusively at
the first checkpoint EOS token. These are quality controls, not benchmarks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

MODEL_ID = "Qwen/Qwen3.8-Flash-Next"
REVISION = "812d96e10a0dd82b69d749e68693698cf6c36daa"
MODEL_DIR = Path(__file__).resolve().parents[1]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def eos_inclusive(tokens, eos_ids):
    for index, token in enumerate(tokens):
        if token in eos_ids:
            return tokens[: index + 1], index
    return list(tokens), None


def assert_no_ttnn():
    imported = [name for name in sys.modules if name == "ttnn" or name.startswith("ttnn.")]
    if imported:
        raise RuntimeError(f"HF control unexpectedly imported TTNN: {imported[:5]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("hf", "tt"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    context = json.loads((MODEL_DIR / "doc/context_contract.json").read_text())["supported_context"]
    if not 1 <= args.max_new_tokens < context:
        parser.error("max-new-tokens must fit the supported context contract")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "manifest.json").exists():
        parser.error("Use a fresh output directory to preserve prior evidence")

    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, GenerationConfig

    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    package_root = Path(os.environ["TT_MODEL_BRINGUP_ROOT"])
    prompt_source = package_root / "runtime/readiness_check/vllm_prompts.txt"
    prompts = [prompt.strip() for prompt in prompt_source.read_text().split("\n\n") if prompt.strip()]
    if len(prompts) != 6:
        raise RuntimeError(f"Expected all six shared prompts, found {len(prompts)}")
    checkpoint = Path(snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True))
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    generation_config = GenerationConfig.from_pretrained(checkpoint, local_files_only=True)
    eos_ids = generation_config.eos_token_id
    eos_ids = list(eos_ids) if isinstance(eos_ids, (list, tuple)) else [eos_ids]
    eos_ids = [int(value) for value in eos_ids if value is not None]
    if not eos_ids:
        raise RuntimeError("Pinned checkpoint has no EOS token for logical trimming")
    if not tokenizer.chat_template:
        raise RuntimeError("The pinned Qwen checkpoint must expose its chat template")
    rendered = []
    for index, prompt in enumerate(prompts, 1):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        ids = tokenizer.encode(text, add_special_tokens=False)
        direct_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False
        )
        if ids != direct_ids:
            raise RuntimeError(f"Prompt{index}: rendered and direct chat tokenization disagree")
        if len(ids) + args.max_new_tokens - 1 > context:
            raise RuntimeError(f"Prompt{index}: requested generation exceeds context{context}")
        rendered.append(
            dict(prompt_id=index, prompt=prompt, messages=messages, rendered_prompt=text, prompt_token_ids=ids)
        )
    settings = {
        "greedy": dict(temperature=0.0, top_p=1.0, top_k=1, seed=42),
        "sampled": dict(temperature=0.7, top_p=0.9, top_k=20, seed=42),
    }
    manifest = dict(
        complete=False,
        backend=args.backend,
        model_id=MODEL_ID,
        revision=REVISION,
        checkpoint=str(checkpoint),
        tokenizer_class=type(tokenizer).__name__,
        transformers_version=transformers.__version__,
        torch_version=torch.__version__,
        chat_template_present=True,
        chat_template=tokenizer.chat_template,
        prompt_mode="chat",
        rendering="apply_chat_template(add_generation_prompt=True, enable_thinking=False)",
        chat_template_kwargs=dict(enable_thinking=False),
        prompt_source=str(prompt_source),
        context=context,
        max_batch_size=1,
        max_new_tokens=args.max_new_tokens,
        stopping="generate full budget; logical output ends inclusively at first checkpoint EOS",
        eos_token_ids=eos_ids,
        profiles=settings,
        outputs_completed=0,
        expected_outputs=12,
        scope="matched quality controls; no serving or performance claim",
        command=sys.argv,
    )
    write_json(args.output_dir / "prompts.json", rendered)
    write_json(args.output_dir / "manifest.json", manifest)
    outputs = []
    mesh = gen = None
    started = time.monotonic()
    try:
        if args.backend == "hf":
            assert_no_ttnn()
            from models.autoports.qwen_qwen3_8_flash_next.tests import full_model_hf_reference

            # The exact-row loader writes metadata through this module constant.
            # Redirect that destination so completed full-model artifacts stay intact.
            full_model_hf_reference.EVIDENCE_DIR = args.output_dir / "hf_reference_loader"
            model = full_model_hf_reference.load_exact_text_model()
            assert_no_ttnn()
            manifest["model_class"] = type(model).__name__
            manifest["layers"] = len(model.model.layers)
            manifest["hf_ttnn_imported"] = False
            manifest[
                "sampling_rng_note"
            ] = "torch seed42 per request; HF and TT samplers need not draw identical tokens"
        else:
            import ttnn
            from models.autoports.qwen_qwen3_8_flash_next.tt.generator import build_generator
            from models.common.sampling.sampling_params import SamplingParams

            if not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
                raise RuntimeError("Restore the accepted mesh descriptor after lane-env")
            ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
            mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0, l1_small_size=32768)
            gen = build_generator(MODEL_DIR, mesh, checkpoint=str(checkpoint), max_batch_size=1, max_seq_len=context)
            if tuple(gen.model.layer_indices) != tuple(range(48)):
                raise RuntimeError("TT quality control requires all48 layers")
            gen.prepare()
            manifest["layers"] = list(gen.model.layer_indices)
            manifest["selected_precision"] = gen.model.precision
            manifest["sampling_path"] = "canonical split-sampling token-out, traced decode, host_sampling=False"
            manifest[
                "sampling_rng_note"
            ] = "canonical per-request seed42; HF and TT samplers need not draw identical tokens"
        manifest["loaded_seconds"] = time.monotonic() - started
        write_json(args.output_dir / "manifest.json", manifest)

        for prompt in rendered:
            for profile, parameters in settings.items():
                request_started = time.monotonic()
                torch.manual_seed(parameters["seed"])
                if args.backend == "hf":
                    input_ids = torch.tensor([prompt["prompt_token_ids"]], dtype=torch.long)
                    generation_kwargs = dict(
                        max_new_tokens=args.max_new_tokens,
                        do_sample=profile == "sampled",
                        temperature=parameters["temperature"] if profile == "sampled" else 1.0,
                        top_p=parameters["top_p"],
                        top_k=parameters["top_k"],
                        repetition_penalty=1.0,
                        eos_token_id=None,
                        forced_eos_token_id=None,
                        pad_token_id=tokenizer.pad_token_id or eos_ids[0],
                        use_cache=True,
                    )
                    with torch.inference_mode():
                        sequence = model.generate(
                            input_ids=input_ids, attention_mask=torch.ones_like(input_ids), **generation_kwargs
                        )
                    raw = sequence[0, input_ids.shape[1] :].tolist()
                    assert_no_ttnn()
                    counters = None
                else:
                    before = dict(gen.counters)
                    raw = gen.generate(
                        prompt["prompt_token_ids"],
                        args.max_new_tokens,
                        sampling_params=SamplingParams(**parameters),
                        enable_trace=True,
                        host_sampling=False,
                    )
                    counters = {name: gen.counters[name] - value for name, value in before.items()}
                    generation_kwargs = dict(enable_trace=True, host_sampling=False, sampling_params=parameters)
                if len(raw) != args.max_new_tokens:
                    raise RuntimeError(f"{profile} prompt{prompt['prompt_id']}: raw generation stopped at {len(raw)}")
                logical, eos_index = eos_inclusive(raw, eos_ids)
                result = dict(
                    backend=args.backend,
                    prompt_id=prompt["prompt_id"],
                    profile=profile,
                    settings={**parameters, "max_new_tokens": args.max_new_tokens, "enable_thinking": False},
                    actual_backend_kwargs=generation_kwargs,
                    prompt_token_ids=prompt["prompt_token_ids"],
                    raw_token_ids=raw,
                    logical_token_ids=logical,
                    first_eos_index=eos_index,
                    raw_text=tokenizer.decode(raw, skip_special_tokens=False),
                    text=tokenizer.decode(logical, skip_special_tokens=True),
                    logical_text_with_special_tokens=tokenizer.decode(logical, skip_special_tokens=False),
                    seconds=time.monotonic() - request_started,
                    counter_delta=counters,
                )
                outputs.append(result)
                write_json(args.output_dir / f"prompt_{prompt['prompt_id']:02d}_{profile}.json", result)
                write_json(args.output_dir / "outputs.json", outputs)
                manifest["outputs_completed"] = len(outputs)
                manifest["seconds"] = time.monotonic() - started
                write_json(args.output_dir / "manifest.json", manifest)
                print(
                    json.dumps(
                        dict(
                            completed=len(outputs),
                            backend=args.backend,
                            prompt_id=prompt["prompt_id"],
                            profile=profile,
                            raw_tokens=len(raw),
                            logical_tokens=len(logical),
                            seconds=result["seconds"],
                        )
                    ),
                    flush=True,
                )
        manifest["complete"] = True
    except Exception:
        manifest["error"] = traceback.format_exc()
        raise
    finally:
        try:
            if gen is not None:
                gen.release_persistent_capture()
        except Exception:
            manifest["complete"] = False
            manifest["capture_cleanup_error"] = traceback.format_exc()
            raise
        finally:
            try:
                if mesh is not None:
                    ttnn.close_mesh_device(mesh)
            except Exception:
                manifest["complete"] = False
                manifest["mesh_close_error"] = traceback.format_exc()
                raise
            finally:
                manifest["seconds"] = time.monotonic() - started
                write_json(args.output_dir / "manifest.json", manifest)
    print("QUALITY_CONTROLS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
