# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark full-model on-device sampling strategies for Llama 3.1 8B.

The optimized-full-model contract keeps greedy decode on the canonical split
top-k/top-p-capable sampler. This harness records which sampler strategies are
actually exposed by the model args, then benchmarks the enabled split-sampling
trace path on real full-model logits.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import ttnn
from transformers import AutoTokenizer

from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.generator import build_generator
from models.common.readiness_check.mesh_device import close_readiness_mesh_device, open_readiness_mesh_device
from models.common.sampling import SamplingParams, format_sampling_params


MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_MODEL_DIR = Path("models/autoports/meta_llama_llama_3_1_8b_instruct")
DEFAULT_PROMPT_FILE = Path("models/common/readiness_check/autoregressive_prompt.txt")


def _build_prompt(prompt_file: Path) -> tuple[list[int], str]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, local_files_only=True)
    prompt_text = prompt_file.read_text(encoding="utf-8").strip()
    return tokenizer.encode(prompt_text, add_special_tokens=True), prompt_text


def _read_first_scalar(tt_tensor: ttnn.Tensor) -> int:
    device_tensor = ttnn.get_device_tensors(tt_tensor)[0]
    return int(ttnn.to_torch(device_tensor).reshape(-1)[0].item())


def _benchmark_sampling_trace(generator, *, replay_count: int) -> dict[str, Any]:
    if replay_count <= 0:
        return {"replay_count": replay_count, "elapsed_s": 0.0, "ms_per_replay": 0.0, "replays_per_s": 0.0}

    start = time.perf_counter()
    for _ in range(replay_count):
        generator.sampling.sample(
            generator._decode_trace.logits,
            enable_trace=True,
            tt_out_tok=generator._decode_trace.tokens,
        )
    ttnn.synchronize_device(generator.mesh_device)
    elapsed_s = time.perf_counter() - start
    return {
        "replay_count": replay_count,
        "elapsed_s": elapsed_s,
        "ms_per_replay": elapsed_s * 1000.0 / replay_count,
        "replays_per_s": replay_count / elapsed_s if elapsed_s > 0 else 0.0,
    }


def _force_argmax_availability(sampling_args) -> dict[str, Any]:
    sampling_ag_config = getattr(sampling_args, "model_config", {}).get("SAMPLING_AG_CONFIG")
    allow_force = bool(sampling_ag_config and sampling_ag_config.get("allow_force_argmax", False))
    return {
        "available": allow_force,
        "reason": None
        if allow_force
        else (
            "Llama31_8B_InstructFullModel.make_sampling_args() exposes model_config={} with no "
            "SAMPLING_AG_CONFIG.allow_force_argmax, and the readiness generator rejects force-argmax "
            "if it activates for greedy decode."
        ),
        "sampling_ag_config": sampling_ag_config,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--mesh-device", default="T3K")
    parser.add_argument("--fabric-config", default="FABRIC_1D_RING")
    parser.add_argument("--replay-count", type=int, default=64)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_MODEL_DIR / "doc" / "optimized_full_model" / "sampling_strategy_benchmark.json",
    )
    args = parser.parse_args()

    prompt_ids, prompt_text = _build_prompt(args.prompt_file)
    prompt = torch.tensor([prompt_ids], dtype=torch.long)

    print(f"Opening {args.mesh_device} {args.fabric_config} mesh")
    mesh_device = open_readiness_mesh_device(args.mesh_device, args.fabric_config)
    generator = None
    try:
        print("Building full generator")
        generator = build_generator(model_dir=args.model_dir, mesh_device=mesh_device)
        sampling_args = generator.model.make_sampling_args()

        generator.reset()
        logits = generator.prefill_forward(
            prompt,
            page_table=generator.page_table,
            kv_cache=generator.kv_cache,
            prompt_lens=[len(prompt_ids)],
            return_all_logits=False,
        )
        ttnn.synchronize_device(mesh_device)
        first_token = int(torch.argmax(logits.reshape(-1)).item())

        print("Capturing model and split-sampling traces")
        generator._decode_trace_sample(
            first_token,
            len(prompt_ids),
            page_table=generator.page_table,
            enable_trace=True,
            token_from_host=True,
            refresh_sampled_hidden=True,
            readback=False,
        )
        ttnn.synchronize_device(mesh_device)

        logits_ref = generator._logits_to_torch(generator._decode_trace.logits)
        first_user_logits = logits_ref.reshape(-1, logits_ref.shape[-1])[0]
        greedy_reference_token = int(torch.argmax(first_user_logits).item())
        greedy_sampled_token = _read_first_scalar(generator._decode_trace.tokens)

        print(f"Benchmarking split greedy sampling trace for {args.replay_count} replays")
        split_greedy = _benchmark_sampling_trace(generator, replay_count=args.replay_count)
        split_greedy.update(
            {
                "available": True,
                "selected": True,
                "sampling_params": {"temperature": 1.0, "top_k": 1, "top_p": 0.0},
                "force_argmax": bool(generator.sampling.tt_sampling.force_argmax_sampling),
                "sampled_token": greedy_sampled_token,
                "reference_argmax_token": greedy_reference_token,
                "reference_logits_shape": list(logits_ref.shape),
                "semantically_greedy": greedy_sampled_token == greedy_reference_token,
            }
        )

        generator.sampling.reset_sampling_params(
            format_sampling_params(
                SamplingParams(temperature=0.7, top_k=8, top_p=0.9),
                generator.sampling.tt_sampling.max_batch_size,
            )
        )
        print(f"Benchmarking split top-k/top-p sampling trace for {args.replay_count} replays")
        split_topk_topp = _benchmark_sampling_trace(generator, replay_count=args.replay_count)
        split_topk_topp.update(
            {
                "available": True,
                "selected_for_greedy": False,
                "sampling_params": {"temperature": 0.7, "top_k": 8, "top_p": 0.9},
                "force_argmax": bool(generator.sampling.tt_sampling.force_argmax_sampling),
                "sampled_token_after_replays": _read_first_scalar(generator._decode_trace.tokens),
            }
        )

        result = {
            "hf_model_id": MODEL_ID,
            "model_dir": str(args.model_dir),
            "mesh_device": args.mesh_device,
            "fabric_config": args.fabric_config,
            "prompt": {
                "file": str(args.prompt_file),
                "num_tokens": len(prompt_ids),
                "text": prompt_text,
            },
            "sampling_contract": {
                "max_top_k": int(generator.sampling.tt_sampling.max_top_k),
                "num_gather_links": int(generator.sampling.tt_sampling.num_gather_links),
                "sampling_dp": int(generator.sampling.tt_sampling._sampling_dp),
                "sampling_all_gather_axis": int(generator.sampling.tt_sampling.sampling_all_gather_axis),
                "pad_logits_to_power_of_2": bool(generator.sampling.tt_sampling.pad_to_power_of_2),
                "force_argmax_available": _force_argmax_availability(sampling_args),
            },
            "strategies": {
                "split_greedy": split_greedy,
                "split_topk_topp": split_topk_topp,
                "force_argmax": {
                    "available": _force_argmax_availability(sampling_args)["available"],
                    "benchmarked": False,
                    "selected": False,
                    "reason": _force_argmax_availability(sampling_args)["reason"],
                },
            },
            "selected_strategy": "split_greedy",
            "status": "pass" if split_greedy["semantically_greedy"] and not split_greedy["force_argmax"] else "fail",
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote {args.output_json}")
    finally:
        if generator is not None:
            generator.teardown()
        close_readiness_mesh_device(mesh_device, args.fabric_config)


if __name__ == "__main__":
    main()
