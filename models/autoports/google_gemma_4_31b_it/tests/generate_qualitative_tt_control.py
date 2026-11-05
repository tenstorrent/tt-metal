# SPDX-License-Identifier: Apache-2.0
"""Current full-model TT controls for six canonical chat prompts, using device sampling."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

import readiness_check
import ttnn

from ..tt.generator import Gemma4Generator
from ..tt.weights import HF_MODEL, HF_REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/autoports/google_gemma_4_31b_it/readiness_vllm/tt_qualitative_control"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--context", type=int, default=4096)
    args = parser.parse_args()
    assert args.max_new_tokens > 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt_source = Path(readiness_check.__file__).parent / "vllm_prompts.txt"
    prompts = [p.strip() for p in prompt_source.read_text().split("\n\n") if p.strip()]
    assert len(prompts) == 6, "Expected the six canonical readiness prompts"
    source_dir = Path(__file__).resolve().parents[1] / "tt"
    source_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(source_dir.glob("*.py"))
    }
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024)
    gen = None
    try:
        gen = Gemma4Generator(mesh, batch_size=1, max_seq_len=args.context)
        assert len(gen.model.layers) == 60 and list(gen.model.layer_indices) == list(range(60))
        assert gen.tokenizer.chat_template
        cases = []
        for index, prompt in enumerate(prompts):
            messages = [{"role": "user", "content": prompt}]
            tokens = gen.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=False
            )
            assert len(tokens) + args.max_new_tokens <= args.context
            cases.append(
                {
                    "id": f"shared_{index}",
                    "messages": messages,
                    "prompt_tokens": tokens,
                    "rendered_prompt": gen.tokenizer.decode(tokens),
                }
            )
        profiles = {
            "greedy": {"temperature": 0.0, "top_k": 1, "top_p": 1.0, "seed": 42},
            "sampled": {"temperature": 0.7, "top_k": 32, "top_p": 0.9, "seed": 42},
        }
        metadata = {
            "status": "running",
            "source": "standalone_autoregressive_control",
            "hf_model": HF_MODEL,
            "hf_revision": HF_REVISION,
            "tokenizer_revision": HF_REVISION,
            "tokenizer": type(gen.tokenizer).__name__,
            "prompt_mode": "chat",
            "rendering": "apply_chat_template(add_generation_prompt=True,return_dict=False)",
            "prompt_source": str(prompt_source),
            "prompt_source_sha256": hashlib.sha256(prompt_source.read_bytes()).hexdigest(),
            "source_sha256": source_hashes,
            "precision_config": gen.model.precision_config,
            "layer_indices": list(gen.model.layer_indices),
            "mesh": [1, 4],
            "batch_size": 1,
            "concurrency": 1,
            "context": args.context,
            "max_new_tokens": args.max_new_tokens,
            "eos_token_ids": sorted(gen.eos_token_ids),
            "profiles": profiles,
            "sampling_path": "Gemma4Generator.generate canonical traced device sampler",
            "host_sampling": False,
            "quality_verdict": "pending human/agent reading of saved completions",
            "limitations": [
                "Standalone full-model control; not vLLM evidence or a serving performance measurement.",
                "Sampled TT and HF outputs need not match token-for-token because sampling implementations differ.",
                "Context is a control-run allocation limit, not a change to advertised serving capability.",
            ],
        }
        metadata_path = args.output_dir / "qualitative_tt_control_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        gen.prepare_decode()
        for profile, sampling in profiles.items():
            outputs = []
            output_path = args.output_dir / (
                "qualitative_tt.json" if profile == "greedy" else "qualitative_tt_sampled.json"
            )
            for case in cases:
                print("TT_QUALITATIVE_CONTROL", profile, case["id"], flush=True)
                generated = gen.generate(
                    case["prompt_tokens"],
                    args.max_new_tokens,
                    enable_trace=True,
                    host_sampling=False,
                    ignore_eos=False,
                    sampling_params=sampling,
                )
                entry = dict(
                    case,
                    generated_tokens=[int(token) for token in generated],
                    completion=gen.tokenizer.decode(generated),
                    sampling_profile=sampling,
                    prompt_length=len(case["prompt_tokens"]),
                    output_length=len(generated),
                    stop_reason="eos" if generated and generated[-1] in gen.eos_token_ids else "length",
                )
                outputs.append(entry)
                output_path.write_text(json.dumps(outputs, indent=2) + "\n")
                print(entry["completion"], flush=True)
        current_hashes = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(source_dir.glob("*.py"))
        }
        metadata.update(
            status="complete", completed_prompts_per_profile=6, source_unchanged=current_hashes == source_hashes
        )
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        assert metadata["source_unchanged"], "Model source changed during the control run; inspect provenance"
    finally:
        if gen is not None:
            gen.release_persistent_capture()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
