# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Matched six-prompt HF/TT controls and label-hidden review artifacts.

Run backends serially in the configured bringup environment, using one output
directory. Exact matching completed samples may be reused. No quality verdict
is inferred automatically; blind scores must be supplied by an independent
reviewer who receives blind_samples.json and rubric.json, never blind_key.json.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import secrets
import sys
import time
from pathlib import Path

MODEL = "ornith-ai/Ornith-1.5-9B"
REVISION = "489cb97981b8654bcfcf30ce1f94ed1b62e07b53"
GENERAL = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "frequency_penalty": 0.0,
    "repetition_penalty": 1.0,
}
RUBRIC = {
    "fixed_before_generation": True,
    "minimum_matched_seeds_per_prompt": 10,
    "failure_rule": "TT failure rate exceeding HF by more than 20 percentage points on a prompt is stage work.",
    "comparison": "Use identical flags for both hidden sides; unblind only after saving all scores.",
    "failure_flags": [
        "incoherent_or_gibberish",
        "mechanical_repetition_or_loop",
        "unrequested_wrong_language",
        "prompt_echo_or_cross_request_leakage",
        "corrupt_subwords_or_unexpected_control_tokens",
        "materially_wrong_or_irrelevant_answer",
    ],
    "notes": [
        "The checkpoint defaults to thinking; coherent reasoning and think delimiters are expected.",
        "French is requested for the translation prompt; it is not wrong-language drift there.",
        "A token-cap hit, missing final answer while coherently thinking, or absent EOS alone is not a failure.",
        "Record cap-limited incomplete answers separately and compare the same behavior on the other side.",
        "Record concrete excerpts and early divergence; exact sampled token equality is not expected across RNG implementations.",
        "If HF shares a failure within the margin, record a known model limitation and continue.",
    ],
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def require_same(path, value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f"Artifact configuration differs: {path}; use a new output directory")
    write_json(path, value)


def prompt_source(explicit):
    if explicit:
        return Path(explicit).resolve()
    root = os.environ.get("TT_MODEL_BRINGUP_ROOT")
    if root:
        return Path(root).resolve() / "runtime/readiness_check/vllm_prompts.txt"
    spec = importlib.util.find_spec("readiness_check")
    if spec is None or not spec.submodule_search_locations:
        raise ValueError("Set TT_MODEL_BRINGUP_ROOT using the installed skill environment, or pass --prompts")
    return Path(next(iter(spec.submodule_search_locations))) / "vllm_prompts.txt"


def local_checkpoint(explicit):
    if explicit:
        path = Path(explicit).resolve()
    else:
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(MODEL, revision=REVISION, local_files_only=True))
    if not (path / "config.json").is_file():
        raise ValueError(f"Missing local checkpoint config: {path}")
    return path


def checkpoint_identity(path):
    names = (
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "chat_template.jinja",
        "model.safetensors.index.json",
    )
    return {
        "path": str(path),
        "files_sha256": {
            name: hashlib.sha256((path / name).read_bytes()).hexdigest() for name in names if (path / name).is_file()
        },
    }


def export_blind(output, shared):
    pairs = []
    for prompt in shared["prompts"]:
        for seed in shared["seeds"]:
            for backend in ("hf", "tt"):
                path = output / backend / f"{prompt['id']}_seed{seed}.json"
                if not path.exists():
                    return False
                sample = json.loads(path.read_text())
                if sample.get("status") != "complete" or sample["shared_signature"] != digest(shared):
                    return False
                pairs.append((backend, path, sample))
    # Retain existing IDs/order for an exact rerun rather than change a review in progress.
    if (output / "blind_samples.json").exists():
        return True
    random.SystemRandom().shuffle(pairs)
    blind, key = [], {}
    for backend, path, sample in pairs:
        identifier = secrets.token_hex(10)
        blind.append(
            {
                "id": identifier,
                "prompt_id": sample["prompt_id"],
                "seed": sample["seed"],
                "prompt": sample["prompt"],
                "output": sample["text"],
                "stop_reason": sample["stop_reason"],
                "score": {"failure_flags": [], "cap_limited_incomplete": None, "notes": ""},
            }
        )
        key[identifier] = {"backend": backend, "artifact": str(path)}
    write_json(output / "blind_key.json", key)
    write_json(output / "blind_samples.json", blind)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["hf", "tt"], required=True)
    parser.add_argument("--seeds", default=",".join(map(str, range(10))))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--context", type=int, default=2048, help="TT cache context; serving controls use262144")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", help="Original checkpoint, otherwise exact pinned local HF snapshot")
    parser.add_argument(
        "--hf-reference",
        default=os.environ.get("ORNITH_HF_REFERENCE"),
        help="Existing local remapped HF text reference; tokenizer always comes from --checkpoint",
    )
    parser.add_argument("--prompts", help="Defaults to the installed readiness_check/vllm_prompts.txt")
    parser.add_argument("--prompt-ids", help="Comma-separated original suite IDs for a targeted matched-seed control")
    parser.add_argument("--worker-count", type=int, default=1, help="CPU HF task shards; never parallel TT device jobs")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument(
        "--greedy", action="store_true", help="Exploratory paired greedy run; use a separate output directory"
    )
    parser.add_argument("--thinking", choices=["default", "on", "off"], default="default")
    parser.add_argument("--temperature", type=float, help="Override the shared HF/TT temperature (0 selects greedy)")
    parser.add_argument("--top-p", type=float, help="Override the shared HF/TT top-p")
    parser.add_argument("--top-k", type=int, help="Override the shared HF/TT top-k; device supports 1..32")
    parser.add_argument("--presence-penalty", type=float, help="Override the generated-token presence penalty")
    args = parser.parse_args()
    if not 1 <= args.context <= 262144:
        parser.error("--context must be in [1,262144]")
    if args.temperature is not None and (not math.isfinite(args.temperature) or args.temperature < 0):
        parser.error("--temperature must be finite and nonnegative")
    if args.top_p is not None and (not math.isfinite(args.top_p) or not 0 < args.top_p <= 1):
        parser.error("--top-p must be in (0, 1]")
    if args.top_k is not None and not 1 <= args.top_k <= 32:
        parser.error("--top-k must be in the shared device-supported range 1..32")
    if args.presence_penalty is not None and (
        not math.isfinite(args.presence_penalty) or not -2 <= args.presence_penalty <= 2
    ):
        parser.error("--presence-penalty must be in [-2, 2]")
    if args.greedy and args.temperature not in (None, 0):
        parser.error("--greedy conflicts with a nonzero --temperature")
    if not 0 <= args.worker_index < args.worker_count or (args.backend == "tt" and args.worker_count != 1):
        parser.error("Valid worker index/count required; TT jobs must remain serialized")
    seeds = [int(value) for value in args.seeds.split(",")]
    if not seeds or len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        parser.error("--seeds must be distinct nonnegative integers")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    import torch
    import transformers
    from transformers import AutoConfig, AutoTokenizer, GenerationConfig, LogitsProcessor, LogitsProcessorList

    torch.set_num_threads(8)
    checkpoint = local_checkpoint(args.checkpoint)
    reference = local_checkpoint(args.hf_reference) if args.hf_reference else checkpoint
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    source = prompt_source(args.prompts)
    raw_prompts = [value.strip() for value in source.read_text().split("\n\n") if value.strip()]
    if len(raw_prompts) != 6:
        raise ValueError(f"Expected the six shared readiness prompts, got {len(raw_prompts)} from {source}")
    template = tokenizer.chat_template
    if args.thinking != "default" and not template:
        parser.error("--thinking requires a tokenizer chat template")
    template_kwargs = {} if args.thinking == "default" else {"enable_thinking": args.thinking == "on"}
    mode = "chat" if template else "completion"
    prompts = []
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for index, prompt in enumerate(raw_prompts):
        messages = [{"role": "user", "content": prompt}]
        rendered = (
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **template_kwargs)
            if template
            else prompt
        )
        ids = tokenizer(rendered, add_special_tokens=False).input_ids
        if len(ids) + args.max_new_tokens - 1 > args.context:
            raise ValueError("Prompt plus output exceeds the configured context")
        item = {
            "id": f"prompt{index + 1:02}",
            "prompt": prompt,
            "messages": messages if template else None,
            "rendered": rendered,
            "token_ids": ids,
        }
        prompts.append(item)
        require_same(output / "prompts" / f"{item['id']}.json", item)
        (output / "prompts" / f"{item['id']}.txt").write_text(rendered)
    if args.prompt_ids:
        selected = set(args.prompt_ids.split(","))
        if not selected <= {item["id"] for item in prompts}:
            parser.error("Unknown original prompt ID")
        prompts = [item for item in prompts if item["id"] in selected]
    policy = dict(GENERAL)
    if args.greedy:
        policy.update(temperature=0.0, top_k=1, top_p=1.0)
    for name in ("temperature", "top_p", "top_k", "presence_penalty"):
        value = getattr(args, name)
        if value is not None:
            policy[name] = value
    greedy = policy["temperature"] == 0
    original_config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    eos_ids = getattr(original_config, "eos_token_id", None)
    if eos_ids is None:
        eos_ids = tokenizer.eos_token_id
    eos_ids = [eos_ids] if isinstance(eos_ids, int) else list(eos_ids or [])
    shared = {
        "hf_model": MODEL,
        "hf_revision": REVISION,
        "checkpoint": checkpoint_identity(checkpoint),
        "tokenizer_class": type(tokenizer).__name__,
        "chat_template_present": bool(template),
        "chat_template": template,
        "prompt_mode": mode,
        "rendering": (
            "apply_chat_template(tokenize=False, add_generation_prompt=True), "
            + (
                "default thinking"
                if args.thinking == "default"
                else f"enable_thinking={template_kwargs['enable_thinking']}"
            )
            if template
            else "plain completion"
        ),
        "prompt_source": str(source),
        "prompt_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "prompts": prompts,
        "seeds": seeds,
        "sampling": policy,
        "max_new_tokens": args.max_new_tokens,
        "generation": "Both backends run fixed token budget; retain raw IDs and score only through first EOS (excluded)",
        "eos_token_ids": eos_ids,
        "presence_penalty_scope": "generated tokens only, reset for every sample",
        "rubric": RUBRIC,
        "qualifies_for_failure_control": len(seeds) >= 10 and not greedy,
        "seed_note": "Matched request seeds, independently seeded HF/TT RNGs; equal sampled sequences are not assumed",
    }
    if template_kwargs:
        shared["chat_template_kwargs"] = template_kwargs
    require_same(output / "qualitative_prompt_format.json", shared)
    require_same(output / "rubric.json", RUBRIC)
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    backend_identity = {
        "backend": args.backend,
        "shared_signature": digest(shared),
        "runner_sha256": source_hash,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "hf_reference": checkpoint_identity(reference) if args.backend == "hf" else None,
        "hf_loader": "AutoModelForCausalLM.from_pretrained(trust_remote_code=True)" if args.backend == "hf" else None,
        "dtype": "bfloat16" if args.backend == "hf" else "selected optimized decoder policy",
    }
    if args.backend == "tt":
        backend_identity["context"] = args.context
        tt_dir = Path(__file__).resolve().parents[1] / "tt"
        backend_identity["implementation_sha256"] = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(tt_dir.glob("*.py"))
        }
    require_same(output / args.backend / "manifest.json", backend_identity)
    write_json(
        output / args.backend / f"invocation_worker{args.worker_index}.json", {"argv": sys.argv, "cpu_threads": 8}
    )
    tasks = []
    for prompt in prompts:
        for seed_index, seed in enumerate(seeds):
            if seed_index % args.worker_count != args.worker_index:
                continue
            path = output / args.backend / f"{prompt['id']}_seed{seed}.json"
            signature = digest({"backend": backend_identity, "prompt": prompt, "seed": seed})
            if path.exists():
                cached = json.loads(path.read_text())
                if cached.get("signature") != signature:
                    raise ValueError(f"Cached sample is not an exact match: {path}")
                if cached.get("status") == "complete":
                    continue
            tasks.append((prompt, seed, path, signature))
    mesh, gen, hf_model = None, None, None
    try:
        if tasks and args.backend == "hf":
            config = AutoConfig.from_pretrained(reference, local_files_only=True)
            expected_text = getattr(original_config, "text_config", original_config)
            actual_text = getattr(config, "text_config", config)
            for key in ("vocab_size", "hidden_size", "num_hidden_layers", "intermediate_size", "num_attention_heads"):
                if getattr(actual_text, key) != getattr(expected_text, key):
                    raise ValueError(f"HF reference architecture mismatch: {key}")
            # Match the successful readiness HF control's loader semantics.
            hf_model = (
                transformers.AutoModelForCausalLM.from_pretrained(
                    reference, dtype=torch.bfloat16, local_files_only=True, trust_remote_code=True
                )
                .eval()
                .to("cpu")
            )
        elif tasks:
            import ttnn
            from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator import build_generator
            from models.common.sampling.generator import SamplingParams

            ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
            mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=256 * 1024 * 1024)
            gen = build_generator(None, mesh, checkpoint=str(checkpoint), max_batch_size=1, max_seq_len=args.context)

        class PresencePenalty(LogitsProcessor):
            """HF equivalent of common TT output_mask * presence_penalties."""

            def __init__(self, prompt_length, penalty):
                self.prompt_length, self.penalty = prompt_length, penalty

            def __call__(self, input_ids, scores):
                scores = scores.clone()
                for row_index in range(input_ids.shape[0]):
                    seen = torch.unique(input_ids[row_index, self.prompt_length :])
                    scores[row_index, seen] -= self.penalty
                return scores

        for prompt, seed, path, signature in tasks:
            record = {
                "signature": signature,
                "shared_signature": digest(shared),
                "backend": args.backend,
                "prompt_id": prompt["id"],
                "prompt": prompt["prompt"],
                "prompt_tokens": prompt["token_ids"],
                "seed": seed,
                "sampling": policy,
                "max_new_tokens": args.max_new_tokens,
                "status": "running",
            }
            write_json(path, record)
            begin = time.monotonic()
            try:
                if args.backend == "hf":
                    transformers.set_seed(seed)
                    ids = torch.tensor([prompt["token_ids"]], dtype=torch.long)
                    config = GenerationConfig.from_model_config(hf_model.config)
                    config.eos_token_id = config.forced_eos_token_id = None
                    config.pad_token_id = tokenizer.pad_token_id or eos_ids[0]
                    config.do_sample = not greedy
                    config.repetition_penalty = 1.0
                    if not greedy:
                        config.temperature, config.top_p, config.top_k = (
                            policy["temperature"],
                            policy["top_p"],
                            policy["top_k"],
                        )
                        config.min_p = 0.0
                    processor = PresencePenalty(ids.shape[1], policy["presence_penalty"])
                    with torch.inference_mode():
                        sequence = hf_model.generate(
                            input_ids=ids,
                            attention_mask=torch.ones_like(ids),
                            generation_config=config,
                            max_new_tokens=args.max_new_tokens,
                            logits_processor=LogitsProcessorList([processor]),
                        )
                    raw = sequence[0, ids.shape[1] :].tolist()
                else:
                    params = SamplingParams(
                        **{key: value for key, value in policy.items() if key != "min_p"}, seed=seed
                    )
                    raw = [
                        int(value)
                        for value in gen.generate(prompt["token_ids"], args.max_new_tokens, sampling_params=params)
                    ]
                stop = next((index for index, token in enumerate(raw) if token in eos_ids), len(raw))
                effective = raw[:stop]
                record.update(
                    status="complete",
                    elapsed_seconds=time.monotonic() - begin,
                    raw_output_token_ids=raw,
                    output_token_ids=effective,
                    text=tokenizer.decode(effective, skip_special_tokens=False),
                    raw_text=tokenizer.decode(raw, skip_special_tokens=False),
                    stop_reason="eos" if stop < len(raw) else "token_cap",
                    eos_index=stop if stop < len(raw) else None,
                )
                write_json(path, record)
                path.with_suffix(".txt").write_text(record["text"])
                print(f"QUALITATIVE_SAMPLE {args.backend} {prompt['id']} seed={seed} tokens={len(raw)}", flush=True)
            except Exception as error:
                record.update(status="failed", error=repr(error), elapsed_seconds=time.monotonic() - begin)
                write_json(path, record)
                raise
        ready = export_blind(output, shared) if args.worker_index == 0 else False
        write_json(
            output / args.backend / f"completion_worker{args.worker_index}.json",
            {
                "status": "complete",
                "samples": len(prompts) * len(seeds[args.worker_index :: args.worker_count]),
                "blind_artifacts_ready": ready,
                "quality_verdict": "pending independent blind review",
                "qualifies_for_failure_control": shared["qualifies_for_failure_control"],
            },
        )
    finally:
        if gen is not None:
            gen.close()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
