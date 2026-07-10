# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prompt-correct qualitative A/B for the self-conditioning prechunk optimization.

Run in two fresh processes because ``DG_SELFCOND_PRECHUNK_EMBED`` is resolved while
the checkpoint embedding table is moved to device:

    DG_SELFCOND_PRECHUNK_EMBED=0 python -u .../qualitative_prechunk.py --out control.json
    env -u DG_SELFCOND_PRECHUNK_EMBED python -u .../qualitative_prechunk.py --out default.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch

from models.experimental.diffusion_gemma.checkpoint import build_tt_model_from_checkpoint_dir
from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.demo.serving_smoke import _DeviceGenLike
from models.experimental.diffusion_gemma.demo.text_demo import _close_mesh_device, _open_mesh_device
from models.experimental.diffusion_gemma.doc.optimize_perf.sweep_serving import _release_controller
from models.experimental.diffusion_gemma.tt.generate import decode_generation, tokenize_prompt
from models.experimental.diffusion_gemma.tt.self_conditioning import (
    self_conditioning_embedding_prechunk_enabled,
    self_conditioning_logits_l1_mode,
)
from models.experimental.diffusion_gemma.tt.serving import BlockDiffusionServingSession


PROMPTS = (
    ("greeting", "Hello, how are you?"),
    ("explain_diffusion", "Explain what a diffusion language model is in one sentence."),
    ("moon_phases", "Write one concise sentence explaining why the Moon has phases."),
)


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_prompt(
    bundle,
    prompt_id: str,
    user_text: str,
    *,
    steps: int,
    canvas_length: int,
    seed: int,
    gumbel_mode: str,
) -> dict:
    messages = [{"role": "user", "content": user_text}]
    rendered = bundle.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_tokens = tokenize_prompt(bundle.tokenizer, messages)
    config = DiffusionConfig(canvas_length=canvas_length, max_denoise_steps=steps)
    session = BlockDiffusionServingSession(
        bundle.tt_model,
        bundle.state_dict,
        config=config,
        tokenizer=bundle.tokenizer,
        gumbel_mode=gumbel_mode,
        seed=seed,
        stop_token_ids=[],
    )
    try:
        started = time.perf_counter()
        session.prefill(prompt_tokens)
        emission = session.decode_block()
        elapsed_s = time.perf_counter() - started
        committed = emission.tokens
        text = decode_generation(
            bundle.tokenizer,
            prompt_tokens,
            _DeviceGenLike(committed, session.cache_len, session.next_pos),
            skip_prompt=True,
            skip_special_tokens=True,
        )
        text_str = text[0] if text else ""
        committed_sha = hashlib.sha256(committed.to(torch.int64).cpu().numpy().tobytes()).hexdigest()[:16]
        return {
            "prompt_id": prompt_id,
            "messages": messages,
            "rendered_prompt": rendered,
            "prompt_token_ids": prompt_tokens[0].tolist(),
            "prompt_len": int(prompt_tokens.shape[1]),
            "elapsed_s": round(elapsed_s, 4),
            "denoise_steps": emission.num_denoise_steps,
            "committed_sha": committed_sha,
            "text": text_str,
        }
    finally:
        _release_controller(session)
        session.reset()


def _compare(control_path: str, candidate_path: str) -> dict:
    control = json.loads(Path(control_path).read_text(encoding="utf-8"))
    candidate = json.loads(Path(candidate_path).read_text(encoding="utf-8"))
    for result in (control, candidate):
        result.setdefault("DG_SELFCOND_LOGITS_L1", "<unset>")
        result.setdefault("resolved_selfcond_logits_l1", "off")
    if control["resolved_selfcond_logits_l1"] != candidate["resolved_selfcond_logits_l1"]:
        if control["resolved_selfcond_logits_l1"] != "off" or candidate["resolved_selfcond_logits_l1"] != "chain":
            raise RuntimeError("qualitative logits-L1 A/B must compare off against chain")
        if not control["resolved_selfcond_prechunk"] or not candidate["resolved_selfcond_prechunk"]:
            raise RuntimeError("qualitative logits-L1 A/B requires prechunking enabled in both processes")
        gate = "self-conditioning logits L1 prompt-correct qualitative A/B"
    else:
        if control["DG_SELFCOND_PRECHUNK_EMBED"] != "0" or control["resolved_selfcond_prechunk"]:
            raise RuntimeError("qualitative prechunk control must resolve DG_SELFCOND_PRECHUNK_EMBED=0")
        if candidate["DG_SELFCOND_PRECHUNK_EMBED"] != "<unset>" or not candidate["resolved_selfcond_prechunk"]:
            raise RuntimeError("qualitative prechunk candidate must resolve the unset default to prechunk enabled")
        gate = "self-conditioning prechunk prompt-correct qualitative A/B"
    config_fields = (
        "model",
        "checkpoint_config_sha256",
        "tokenizer_config_sha256",
        "tokenizer_class",
        "chat_template_present",
        "prompt_mode",
        "rendering_method",
        "prompt_source",
        "gumbel_mode",
        "seed",
        "canvas_length",
        "max_denoise_steps",
    )
    config_identity = {field: control[field] == candidate[field] for field in config_fields}
    if len(control["prompts"]) != len(candidate["prompts"]):
        raise RuntimeError("qualitative prompt-count mismatch")
    prompts = []
    for control_prompt, candidate_prompt in zip(control["prompts"], candidate["prompts"], strict=True):
        input_fields = ("prompt_id", "messages", "rendered_prompt", "prompt_token_ids", "prompt_len")
        input_exact = {field: control_prompt[field] == candidate_prompt[field] for field in input_fields}
        commit_exact = control_prompt["committed_sha"] == candidate_prompt["committed_sha"]
        text_exact = control_prompt["text"] == candidate_prompt["text"]
        prompts.append(
            {
                **{field: control_prompt[field] for field in input_fields},
                "input_exact": input_exact,
                "control_committed_sha": control_prompt["committed_sha"],
                "candidate_committed_sha": candidate_prompt["committed_sha"],
                "commit_exact": commit_exact,
                "text_exact": text_exact,
                "generated_text": candidate_prompt["text"],
            }
        )
    all_exact = (
        all(config_identity.values())
        and all(all(prompt["input_exact"].values()) for prompt in prompts)
        and all(prompt["commit_exact"] and prompt["text_exact"] for prompt in prompts)
    )
    return {
        "date": "2026-07-10",
        "gate": gate,
        "config": {field: control[field] for field in config_fields},
        "config_identity": config_identity,
        "control": {
            "process_label": Path(control_path).stem,
            "DG_SELFCOND_PRECHUNK_EMBED": control["DG_SELFCOND_PRECHUNK_EMBED"],
            "resolved_selfcond_prechunk": control["resolved_selfcond_prechunk"],
            "DG_SELFCOND_LOGITS_L1": control["DG_SELFCOND_LOGITS_L1"],
            "resolved_selfcond_logits_l1": control["resolved_selfcond_logits_l1"],
        },
        "candidate": {
            "process_label": Path(candidate_path).stem,
            "DG_SELFCOND_PRECHUNK_EMBED": candidate["DG_SELFCOND_PRECHUNK_EMBED"],
            "resolved_selfcond_prechunk": candidate["resolved_selfcond_prechunk"],
            "DG_SELFCOND_LOGITS_L1": candidate["DG_SELFCOND_LOGITS_L1"],
            "resolved_selfcond_logits_l1": candidate["resolved_selfcond_logits_l1"],
        },
        "prompts": prompts,
        "verdict": "PASS" if all_exact else "FAIL",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", default=os.environ.get("DG_CKPT", "/home/zni/dg_models/diffusiongemma-26B-A4B-it")
    )
    parser.add_argument("--mesh", default="P150x4")
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--canvas-length", type=int, default=256)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gumbel-mode", choices=("argmax", "chunked", "host", "device"), default="argmax")
    parser.add_argument("--compare", nargs=2, metavar=("CONTROL", "CANDIDATE"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    if args.compare:
        result = _compare(*args.compare)
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("QUALITATIVE_COMPARISON " + json.dumps(result, ensure_ascii=False), flush=True)
        return 0

    checkpoint = Path(args.checkpoint)
    mesh = _open_mesh_device(args.mesh)
    result = None
    try:
        bundle = build_tt_model_from_checkpoint_dir(
            mesh,
            args.checkpoint,
            max_seq_len=args.max_seq_len,
            create_kv_cache=True,
            tokenizer_kwargs={"local_files_only": True},
        )
        result = {
            "model": "google/diffusiongemma-26B-A4B-it",
            "checkpoint": str(checkpoint),
            "checkpoint_config_sha256": _file_sha256(checkpoint / "config.json"),
            "tokenizer_config_sha256": _file_sha256(checkpoint / "tokenizer_config.json"),
            "tokenizer_class": type(bundle.tokenizer).__name__,
            "chat_template_present": bool(bundle.tokenizer.chat_template),
            "prompt_mode": "chat",
            "rendering_method": "tokenizer.apply_chat_template(add_generation_prompt=True)",
            "prompt_source": "qualitative_prechunk.py::PROMPTS",
            "gumbel_mode": args.gumbel_mode,
            "DG_DENOISE_TRACED": os.environ.get("DG_DENOISE_TRACED", "<unset>"),
            "seed": args.seed,
            "canvas_length": args.canvas_length,
            "max_denoise_steps": args.steps,
            "resolved_selfcond_prechunk": self_conditioning_embedding_prechunk_enabled(),
            "DG_SELFCOND_PRECHUNK_EMBED": os.environ.get("DG_SELFCOND_PRECHUNK_EMBED", "<unset>"),
            "resolved_selfcond_logits_l1": self_conditioning_logits_l1_mode(),
            "DG_SELFCOND_LOGITS_L1": os.environ.get("DG_SELFCOND_LOGITS_L1", "<unset>"),
            "prompts": [
                _run_prompt(
                    bundle,
                    prompt_id,
                    user_text,
                    steps=args.steps,
                    canvas_length=args.canvas_length,
                    seed=args.seed,
                    gumbel_mode=args.gumbel_mode,
                )
                for prompt_id, user_text in PROMPTS
            ],
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    finally:
        _close_mesh_device(mesh)

    print("QUALITATIVE_RESULT " + json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
