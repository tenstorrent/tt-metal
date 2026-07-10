# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact 48-step decision gate for self-conditioning embedding prechunking.

Run the full-depth eager trajectory in two fresh processes so the embedding
representation is selected independently, then compare the compact hashes:

    DG_SELFCOND_PRECHUNK_EMBED=0 python -u .../verify_selfcond_prechunk_decisions.py \
      --out /tmp/selfcond_decisions_control.json
    env -u DG_SELFCOND_PRECHUNK_EMBED python -u .../verify_selfcond_prechunk_decisions.py \
      --out /tmp/selfcond_decisions_default.json
    python -u .../verify_selfcond_prechunk_decisions.py \
      --compare /tmp/selfcond_decisions_control.json /tmp/selfcond_decisions_default.json \
      --out selfcond_prechunk_decisions.json

The decision run is intentionally eager: the eager loop returns every per-step
decision tensor, while the separately benchmarked Metal-traced fixed-step loop
keeps those tensors device-resident and returns only the final commit.
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
from models.experimental.diffusion_gemma.demo.text_demo import _close_mesh_device, _open_mesh_device
from models.experimental.diffusion_gemma.tt import denoise_loop as DL
from models.experimental.diffusion_gemma.tt.generate import (
    denoise_and_commit_block,
    host_canvas_to_device,
    tokenize_prompt,
)
from models.experimental.diffusion_gemma.tt.self_conditioning import (
    self_conditioning_embedding_prechunk_enabled,
    self_conditioning_logits_l1_mode,
)
from models.experimental.diffusion_gemma.tt.serving import BlockDiffusionServingSession


BASE_ENV = {"DG_SPARSE_MOE": "1", "DG_SPARSE_MOE_TUNED": "1", "DG_DEDUP_ARGMAX": "1"}
TENSOR_DECISION_FIELDS = ("argmax", "sampled", "accept_mask", "canvas", "entropy")
DECISION_FIELDS = (*TENSOR_DECISION_FIELDS, "commit_candidate")


def _tensor_sha256(tensor: torch.Tensor) -> str:
    tensor = tensor.detach().cpu().contiguous()
    payload = tensor.numpy().tobytes()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(json.dumps(list(tensor.shape)).encode())
    digest.update(payload)
    return digest.hexdigest()


def _json_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _write_json(path: str, value) -> None:
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_canonical_workload(args) -> None:
    expected = {
        "mesh": (args.mesh, "P150x4"),
        "max_seq_len": (args.max_seq_len, 1024),
        "canvas_length": (args.canvas_length, 256),
        "steps": (args.steps, 48),
        "seed": (args.seed, 0),
        "prompt": (args.prompt, "Explain what a diffusion language model is in one sentence."),
    }
    mismatches = [
        f"{name}={actual!r} (expected {wanted!r})" for name, (actual, wanted) in expected.items() if actual != wanted
    ]
    if mismatches:
        raise RuntimeError("non-canonical exact-decision workload: " + ", ".join(mismatches))


def _run(args) -> dict:
    _validate_canonical_workload(args)
    for key, value in BASE_ENV.items():
        os.environ[key] = value

    mesh = _open_mesh_device(args.mesh)
    session = None
    try:
        bundle = build_tt_model_from_checkpoint_dir(
            mesh,
            args.checkpoint,
            max_seq_len=args.max_seq_len,
            create_kv_cache=True,
            tokenizer_kwargs={"local_files_only": True},
        )
        prompt_tokens = tokenize_prompt(bundle.tokenizer, args.prompt)
        config = DiffusionConfig(canvas_length=args.canvas_length, max_denoise_steps=args.steps)
        session = BlockDiffusionServingSession(
            bundle.tt_model,
            bundle.state_dict,
            config=config,
            tokenizer=bundle.tokenizer,
            gumbel_mode=args.gumbel_mode,
            seed=args.seed,
            stop_token_ids=[],
        )
        session.prefill(prompt_tokens)

        vocab_size = int(session.vocab_size)
        init_generator = torch.Generator().manual_seed(args.seed)
        noise_generator = torch.Generator().manual_seed(args.seed + 1)
        init_canvas = torch.randint(
            0,
            vocab_size,
            (1, args.canvas_length),
            dtype=torch.long,
            generator=init_generator,
        )
        noise_tokens = [
            torch.randint(
                0,
                vocab_size,
                (1, args.canvas_length),
                dtype=torch.long,
                generator=noise_generator,
            )
            for _ in range(args.steps)
        ]

        gumbel_noise_for_step = session._gumbel_noise_fn(0)
        gumbel_identities = []

        def gumbel_noise_fn(step: int):
            noise = gumbel_noise_for_step(step)
            if noise is not None:
                # The selected memory-bounded sampler injects a deterministic
                # descriptor. Seed + chunk size + dtype fully identify the
                # TTNN random stream without materializing a 256 MiB host tensor.
                identity = {
                    "seed": int(noise.seed),
                    "vocab_chunk_size": int(noise.vocab_chunk_size),
                    "dtype": str(noise.dtype),
                }
                gumbel_identities.append(identity)
            return noise

        def noise_tokens_fn(step: int):
            return host_canvas_to_device(mesh, noise_tokens[step].clone())

        started = time.perf_counter()
        block = denoise_and_commit_block(
            session.tt_model,
            session._logits_fn,
            host_canvas_to_device(mesh, init_canvas.clone()),
            config,
            start_pos=session.next_pos,
            gumbel_noise_fn=gumbel_noise_fn,
            noise_tokens_fn=noise_tokens_fn,
            page_table=session.page_table,
            page_tables_per_layer=session.page_tables_per_layer,
            denoise_block_fn=DL.denoise_block,
        )
        elapsed_s = time.perf_counter() - started
        trajectory = block.trajectory
        if trajectory.num_steps != args.steps or len(trajectory.per_step) != args.steps:
            raise RuntimeError(
                f"expected full {args.steps}-step trajectory, got "
                f"num_steps={trajectory.num_steps} records={len(trajectory.per_step)}"
            )
        expected_gumbel_identities = 0 if args.gumbel_mode == "argmax" else args.steps
        if len(gumbel_identities) != expected_gumbel_identities:
            raise RuntimeError(f"expected {expected_gumbel_identities} Gumbel identities, got {len(gumbel_identities)}")

        per_step = []
        for record in trajectory.per_step:
            field_hashes = {field: _tensor_sha256(getattr(record, field)) for field in TENSOR_DECISION_FIELDS}
            # DiffusionGemma commits the last step's clean argmax. Persist the
            # clean commit candidate at every step explicitly rather than
            # relying on readers to infer that contract from ``argmax``.
            field_hashes["commit_candidate"] = field_hashes["argmax"]
            per_step.append(
                {
                    "step": int(record.step),
                    "temperature": record.temperature,
                    "entropy_mean": record.entropy_mean,
                    "num_accepted": int(record.num_accepted),
                    "sha256": field_hashes,
                }
            )

        noise_hashes = [_tensor_sha256(tokens) for tokens in noise_tokens]
        checkpoint = Path(args.checkpoint)
        result = {
            "date": "2026-07-10",
            "model": "google/diffusiongemma-26B-A4B-it",
            "checkpoint": str(checkpoint),
            "checkpoint_config_sha256": _file_sha256(checkpoint / "config.json"),
            "hardware": "QB2 P150x4 TP=4",
            "mesh": args.mesh,
            "mesh_shape": [1, 4],
            "num_layers": len(bundle.tt_model.layers),
            "seed": args.seed,
            "mode": "full-depth eager decision-record trajectory; traced performance is gated separately",
            "base_env": BASE_ENV,
            "DG_SELFCOND_PRECHUNK_EMBED": os.environ.get("DG_SELFCOND_PRECHUNK_EMBED", "<unset>"),
            "resolved_selfcond_prechunk": self_conditioning_embedding_prechunk_enabled(),
            "DG_SELFCOND_LOGITS_L1": os.environ.get("DG_SELFCOND_LOGITS_L1", "<unset>"),
            "resolved_selfcond_logits_l1": self_conditioning_logits_l1_mode(),
            "prompt": args.prompt,
            "prompt_token_ids": prompt_tokens[0].tolist(),
            "max_seq_len": args.max_seq_len,
            "canvas_length": args.canvas_length,
            "max_denoise_steps": args.steps,
            "gumbel_mode": args.gumbel_mode,
            "gumbel_noise": (
                "None at every step"
                if args.gumbel_mode == "argmax"
                else "production seeded ChunkedGumbelNoise descriptor per step"
            ),
            "gumbel_seed": args.seed + 2,
            "gumbel_noise_identity_per_step": gumbel_identities,
            "gumbel_noise_trajectory_sha256": _json_sha256(gumbel_identities),
            "renoise_seed": args.seed + 1,
            "initial_canvas_sha256": _tensor_sha256(init_canvas),
            "noise_tokens_sha256_per_step": noise_hashes,
            "noise_tokens_trajectory_sha256": _json_sha256(noise_hashes),
            "elapsed_s": round(elapsed_s, 4),
            "num_steps": int(trajectory.num_steps),
            "halted": bool(trajectory.halted),
            "committed_sha256": _tensor_sha256(trajectory.committed),
            "per_step": per_step,
            "trajectory_sha256": _json_sha256(per_step),
        }
        return result
    finally:
        if session is not None:
            session.reset()
        _close_mesh_device(mesh)


def _compare(control_path: str, candidate_path: str) -> dict:
    control = json.loads(Path(control_path).read_text(encoding="utf-8"))
    candidate = json.loads(Path(candidate_path).read_text(encoding="utf-8"))
    for result in (control, candidate):
        # Backward-compatible normalization lets the comparator enrich an
        # already-recorded argmax trajectory without rerunning hardware.
        result.setdefault("DG_SELFCOND_LOGITS_L1", "<unset>")
        result.setdefault("resolved_selfcond_logits_l1", "off")
        if result["gumbel_mode"] == "argmax":
            result.setdefault("gumbel_seed", result["renoise_seed"] + 1)
            result.setdefault("gumbel_noise_identity_per_step", [])
            result.setdefault("gumbel_noise_trajectory_sha256", _json_sha256([]))
        for record in result["per_step"]:
            record["sha256"].setdefault("commit_candidate", record["sha256"]["argmax"])
        result["trajectory_sha256"] = _json_sha256(result["per_step"])
    input_keys = (
        "prompt_token_ids",
        "max_seq_len",
        "canvas_length",
        "max_denoise_steps",
        "gumbel_mode",
        "gumbel_noise",
        "gumbel_seed",
        "gumbel_noise_identity_per_step",
        "renoise_seed",
        "initial_canvas_sha256",
        "noise_tokens_sha256_per_step",
    )
    input_identity = {key: control[key] == candidate[key] for key in input_keys}
    if control["num_steps"] != candidate["num_steps"]:
        raise RuntimeError(f"step-count mismatch: {control['num_steps']} != {candidate['num_steps']}")

    per_step = []
    for control_record, candidate_record in zip(control["per_step"], candidate["per_step"], strict=True):
        exact = {
            field: control_record["sha256"][field] == candidate_record["sha256"][field] for field in DECISION_FIELDS
        }
        exact["temperature"] = control_record["temperature"] == candidate_record["temperature"]
        exact["entropy_mean"] = control_record["entropy_mean"] == candidate_record["entropy_mean"]
        exact["num_accepted"] = control_record["num_accepted"] == candidate_record["num_accepted"]
        per_step.append(
            {
                "step": control_record["step"],
                "all_exact": all(exact.values()),
                "exact": exact,
                "control_sha256": control_record["sha256"],
                "candidate_sha256": candidate_record["sha256"],
            }
        )

    committed_exact = control["committed_sha256"] == candidate["committed_sha256"]
    control_final_candidate_is_commit = (
        control["per_step"][-1]["sha256"]["commit_candidate"] == control["committed_sha256"]
    )
    candidate_final_candidate_is_commit = (
        candidate["per_step"][-1]["sha256"]["commit_candidate"] == candidate["committed_sha256"]
    )
    all_exact = (
        all(input_identity.values())
        and all(record["all_exact"] for record in per_step)
        and committed_exact
        and control_final_candidate_is_commit
        and candidate_final_candidate_is_commit
        and control["trajectory_sha256"] == candidate["trajectory_sha256"]
    )
    if control["resolved_selfcond_logits_l1"] != candidate["resolved_selfcond_logits_l1"]:
        gate = "self-conditioning logits L1 exact diffusion decisions"
    elif control["resolved_selfcond_prechunk"] != candidate["resolved_selfcond_prechunk"]:
        gate = "self-conditioning prechunk exact diffusion decisions"
    else:
        gate = "self-conditioning exact diffusion decisions"
    return {
        "date": "2026-07-10",
        "gate": gate,
        "control": {
            "process_label": Path(control_path).stem,
            "DG_SELFCOND_PRECHUNK_EMBED": control["DG_SELFCOND_PRECHUNK_EMBED"],
            "resolved_selfcond_prechunk": control["resolved_selfcond_prechunk"],
            "DG_SELFCOND_LOGITS_L1": control["DG_SELFCOND_LOGITS_L1"],
            "resolved_selfcond_logits_l1": control["resolved_selfcond_logits_l1"],
            "elapsed_s": control["elapsed_s"],
            "initial_canvas_sha256": control["initial_canvas_sha256"],
            "gumbel_noise_trajectory_sha256": control["gumbel_noise_trajectory_sha256"],
            "noise_tokens_trajectory_sha256": control["noise_tokens_trajectory_sha256"],
            "trajectory_sha256": control["trajectory_sha256"],
            "committed_sha256": control["committed_sha256"],
        },
        "candidate": {
            "process_label": Path(candidate_path).stem,
            "DG_SELFCOND_PRECHUNK_EMBED": candidate["DG_SELFCOND_PRECHUNK_EMBED"],
            "resolved_selfcond_prechunk": candidate["resolved_selfcond_prechunk"],
            "DG_SELFCOND_LOGITS_L1": candidate["DG_SELFCOND_LOGITS_L1"],
            "resolved_selfcond_logits_l1": candidate["resolved_selfcond_logits_l1"],
            "elapsed_s": candidate["elapsed_s"],
            "initial_canvas_sha256": candidate["initial_canvas_sha256"],
            "gumbel_noise_trajectory_sha256": candidate["gumbel_noise_trajectory_sha256"],
            "noise_tokens_trajectory_sha256": candidate["noise_tokens_trajectory_sha256"],
            "trajectory_sha256": candidate["trajectory_sha256"],
            "committed_sha256": candidate["committed_sha256"],
        },
        "inputs": {key: control[key] for key in input_keys},
        "config": {
            "model": control["model"],
            "hardware": control["hardware"],
            "max_seq_len": control["max_seq_len"],
            "canvas_length": control["canvas_length"],
            "max_denoise_steps": control["max_denoise_steps"],
            "gumbel_mode": control["gumbel_mode"],
            "gumbel_noise": control["gumbel_noise"],
            "gumbel_seed": control["gumbel_seed"],
            "renoise_seed": control["renoise_seed"],
        },
        "input_identity": input_identity,
        "noise_tokens_trajectory_sha256": control["noise_tokens_trajectory_sha256"],
        "committed_exact": committed_exact,
        "commit_contract": {
            "semantics": "no KV commit occurs inside a denoise step; each step's clean argmax is the commit candidate, and only the final candidate is committed",
            "control_final_candidate_is_commit": control_final_candidate_is_commit,
            "candidate_final_candidate_is_commit": candidate_final_candidate_is_commit,
        },
        "per_step": per_step,
        "steps_all_exact": sum(record["all_exact"] for record in per_step),
        "steps_compared": len(per_step),
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
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gumbel-mode", choices=("argmax", "chunked"), default="argmax")
    parser.add_argument("--prompt", default="Explain what a diffusion language model is in one sentence.")
    parser.add_argument("--compare", nargs=2, metavar=("CONTROL_JSON", "CANDIDATE_JSON"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    result = _compare(*args.compare) if args.compare else _run(args)
    _write_json(args.out, result)
    marker = "SELFCOND_DECISION_COMPARISON" if args.compare else "SELFCOND_DECISION_RUN"
    print(marker + " " + json.dumps(result, ensure_ascii=False), flush=True)
    if args.compare and result["verdict"] != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
