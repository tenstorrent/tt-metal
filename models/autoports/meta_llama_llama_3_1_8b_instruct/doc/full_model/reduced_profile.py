# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Reduced full-model profiling harness for Llama 3.1 8B Instruct autoport.

This is intentionally a one-real-layer full-model variant. It keeps the real
embedding, RoPE, paged KV/page-table shape, optimized multichip decoder layer,
final norm, split LM head, and canonical split-sampling trace path while keeping
Tracy output small enough for a stage artifact.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import ttnn
from transformers import AutoTokenizer

from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.generator import build_generator
from models.common.readiness_check.mesh_device import close_readiness_mesh_device, open_readiness_mesh_device

try:
    from tracy import signpost
except ImportError:  # pragma: no cover - only absent outside profiler runs

    def signpost(header: str) -> None:
        del header


MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_MODEL_DIR = Path("models/autoports/meta_llama_llama_3_1_8b_instruct")
DEFAULT_PROMPT_FILE = Path("models/common/readiness_check/autoregressive_prompt.txt")


def _build_prompt(prompt_file: Path) -> tuple[list[int], str]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, local_files_only=True)
    prompt_text = prompt_file.read_text(encoding="utf-8").strip()
    return tokenizer.encode(prompt_text, add_special_tokens=True), prompt_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_MODEL_DIR / "doc" / "full_model" / "reduced_profile_summary.json",
    )
    parser.add_argument("--decode-replays", type=int, default=4)
    args = parser.parse_args()

    prompt_ids, prompt_text = _build_prompt(args.prompt_file)
    prompt = torch.tensor([prompt_ids], dtype=torch.long)

    mesh_device = open_readiness_mesh_device("T3K", "FABRIC_1D_RING")
    generator = None
    try:
        generator = build_generator(
            model_dir=args.model_dir,
            mesh_device=mesh_device,
            override_num_layers=1,
            max_seq_len=1024,
            max_num_blocks=16,
            cache_dir=args.model_dir / "tt_cache" / "full_model_reduced_profile",
        )

        generator.reset()
        warm_logits = generator.prefill_forward(
            prompt,
            page_table=generator.page_table,
            kv_cache=generator.kv_cache,
            prompt_lens=[len(prompt_ids)],
            return_all_logits=False,
        )
        ttnn.synchronize_device(mesh_device)
        del warm_logits

        signpost(header="PERF_REDUCED_PREFILL")
        prefill_start = time.perf_counter()
        logits = generator.prefill_forward(
            prompt,
            page_table=generator.page_table,
            kv_cache=generator.kv_cache,
            prompt_lens=[len(prompt_ids)],
            return_all_logits=False,
        )
        ttnn.synchronize_device(mesh_device)
        prefill_ms = (time.perf_counter() - prefill_start) * 1000.0
        signpost(header="PERF_REDUCED_PREFILL_END")

        first_pred = int(torch.argmax(logits.reshape(-1)).item())
        current_pos = len(prompt_ids)
        # Capture model and sampling traces before the profiled steady-state replay window.
        generator._decode_trace_sample(
            first_pred,
            current_pos,
            page_table=generator.page_table,
            enable_trace=True,
            token_from_host=True,
            refresh_sampled_hidden=True,
            readback=False,
        )
        ttnn.synchronize_device(mesh_device)

        replay_ms = []
        for replay_idx in range(args.decode_replays):
            if replay_idx == 0:
                signpost(header="PERF_REDUCED_TOKEN_OUT_DECODE")
            replay_start = time.perf_counter()
            generator._decode_trace_sample(
                0,
                current_pos + 1 + replay_idx,
                page_table=generator.page_table,
                enable_trace=True,
                token_from_host=False,
                refresh_sampled_hidden=True,
                readback=False,
            )
            ttnn.synchronize_device(mesh_device)
            replay_ms.append((time.perf_counter() - replay_start) * 1000.0)
            if replay_idx == 0:
                signpost(header="PERF_REDUCED_TOKEN_OUT_DECODE_END")

        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(
                {
                    "hf_model_id": MODEL_ID,
                    "model_dir": str(args.model_dir),
                    "prompt_file": str(args.prompt_file),
                    "prompt_text": prompt_text,
                    "prompt_tokens": len(prompt_ids),
                    "override_num_layers": 1,
                    "max_seq_len": 1024,
                    "max_num_blocks": 16,
                    "prefill_ms": prefill_ms,
                    "decode_replay_ms": replay_ms,
                    "decode_replay_min_ms": min(replay_ms),
                    "decode_replay_avg_ms": sum(replay_ms) / len(replay_ms),
                    "trace_counters": generator.trace_counters(),
                    "sampling_force_argmax": bool(generator.sampling.tt_sampling.force_argmax_sampling),
                    "sampling_max_top_k": int(generator.sampling.tt_sampling.max_top_k),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    finally:
        if generator is not None:
            generator.teardown()
        close_readiness_mesh_device(mesh_device, "FABRIC_1D_RING")


if __name__ == "__main__":
    main()
