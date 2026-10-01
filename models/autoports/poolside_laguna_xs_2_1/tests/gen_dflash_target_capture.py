# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Layer-streamed CPU capture of real target hidden states for the DFlash draft qualification.

The DFlash draft reads the target's post-layer hidden states.  Synthetic inputs (the
fingerprints in ``tests/test_dflash_reference.py``) prove the draft's arithmetic but not
that the draft is wired to the right target layers.  This tool runs the exact HF
``LagunaDecoderLayer`` one layer at a time over one fixed token sequence (the readiness
AIME24 prompt plus its continuation), keeps every layer's output for all positions, and
records the target's teacher-forced greedy token at every position.  Peak host memory is
one layer (about 30 GB for a Laguna-S MoE layer in fp32), never the 235 GB model.

``tests/test_dflash_reference.py::test_draft_proposals_track_real_target_greedy`` then runs
the published draft on these states and requires it to predict the target's own greedy
tokens, and requires that wrong capture layers do much worse.

Usage (CPU only; no device):
  TT_LAGUNA_MODEL=poolside/Laguna-S-2.1 python -m \
    models.autoports.poolside_laguna_xs_2_1.tests.gen_dflash_target_capture
"""
from __future__ import annotations

import argparse
import gc
import os
import threading
import time
from pathlib import Path

import torch

from models.autoports.poolside_laguna_xs_2_1.tests import laguna_reference as R
from models.autoports.poolside_laguna_xs_2_1.tests import laguna_weights as W
from models.autoports.poolside_laguna_xs_2_1.tt.model_spec import DFLASH_SPEC, MODEL_ID, MODEL_SLUG

TESTS_DIR = Path(__file__).resolve().parent
READINESS_REFERENCES = {
    "poolside/Laguna-S-2.1": TESTS_DIR / "reference_outputs" / "readiness_aime24_chat_s.refpt",
    "poolside/Laguna-XS-2.1": TESTS_DIR / "reference_outputs" / "readiness_aime24_chat.refpt",
}
CAPTURE_ENV = "LAGUNA_DFLASH_TARGET_CAPTURE"


def default_capture_path(model_slug: str = MODEL_SLUG) -> Path:
    override = os.environ.get(CAPTURE_ENV)
    if override:
        return Path(override)
    return Path.home() / ".cache" / "laguna_dflash" / f"{model_slug}_aime24_target_capture.pt"


def _mem_available_gib() -> float:
    with open("/proc/meminfo") as meminfo:
        for line in meminfo:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
    return float("inf")


def _start_memory_guard(floor_gib: float) -> None:
    """Exit hard if the shared host runs low; another job owns most of its memory."""

    def watch():
        while True:
            if _mem_available_gib() < floor_gib:
                print(f"[capture] MemAvailable below {floor_gib} GiB; aborting", flush=True)
                os._exit(3)
            time.sleep(1.0)

    threading.Thread(target=watch, daemon=True).start()


def _rms_norm(x, weight, eps):
    variance = x.pow(2).mean(-1, keepdim=True)
    return weight * (x * torch.rsqrt(variance + eps))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--min-free-gib", type=float, default=36.0, help="required MemAvailable before each layer load")
    ap.add_argument("--abort-free-gib", type=float, default=6.0, help="hard abort if MemAvailable drops below")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    output = args.output or default_capture_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    _start_memory_guard(args.abort_free_gib)

    from models.common.readiness_check.schema import load_reference

    entry = load_reference(READINESS_REFERENCES[MODEL_ID]).entries[0]
    prompt_ids = [int(t) for t in entry.prompt_tokens[0].tolist()]
    continuation_ids = [int(t) for t in entry.generated_tokens[0].tolist()]
    token_ids = prompt_ids + continuation_ids
    reference_top1 = entry.topk_tokens[:, 0].to(torch.int64)
    print(
        f"{MODEL_ID}: {len(prompt_ids)} prompt + {len(continuation_ids)} continuation tokens; "
        f"draft {DFLASH_SPEC.repo_id} targets {DFLASH_SPEC.target_layer_ids}",
        flush=True,
    )

    config = R.build_config()
    from safetensors import safe_open

    snap_dir, weight_map = W._index()
    top_keys = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
    top = {}
    for shard in sorted({weight_map[k] for k in top_keys}):
        with safe_open(W._resolve_shard(snap_dir, shard), "pt") as f:
            for key in top_keys:
                if weight_map[key] == shard:
                    top[key] = f.get_tensor(key).to(torch.float32)
    hidden = top.pop("model.embed_tokens.weight")[torch.tensor(token_ids)].unsqueeze(0)  # [1, S, H] fp32

    layer_outputs = []
    for layer_idx in range(config.num_hidden_layers):
        while _mem_available_gib() < args.min_free_gib:
            print(f"[capture] waiting: MemAvailable {_mem_available_gib():.1f} GiB < {args.min_free_gib}", flush=True)
            time.sleep(30)
        t0 = time.time()
        raw = W.load_layer_tensors(layer_idx)
        state = W.to_hf_layer_state_dict(raw, config, layer_idx)
        del raw
        ctx = R.make_context(config, layer_idx, state_dict=state, dtype=torch.float32)
        del state
        hidden, _ = R.reference_forward(ctx, hidden)
        del ctx
        gc.collect()
        layer_outputs.append(hidden[0].to(torch.bfloat16).clone())
        print(
            f"layer {layer_idx:2d}/{config.num_hidden_layers} rms={hidden.float().pow(2).mean().sqrt():.4f} "
            f"{time.time() - t0:.1f}s free={_mem_available_gib():.1f}GiB",
            flush=True,
        )

    final = _rms_norm(hidden[0], top["model.norm.weight"], config.rms_norm_eps)
    logits = final @ top["lm_head.weight"].t()  # [S, V]: row i predicts token i + 1
    greedy = torch.argmax(logits, dim=-1).to(torch.int64)
    prompt_len = len(prompt_ids)
    agree = (greedy[prompt_len - 1 : len(token_ids) - 1] == reference_top1).float().mean().item()
    print(f"greedy equals the stored readiness top-1 at {agree:.4f} of continuation positions", flush=True)
    torch.save(
        {
            "model": MODEL_ID,
            "token_ids": torch.tensor(token_ids, dtype=torch.int64),
            "prompt_len": prompt_len,
            "layer_outputs": torch.stack(layer_outputs),  # [num_layers, S, H] bf16, post-layer i
            "greedy": greedy,  # [S] teacher-forced target argmax; row i predicts position i + 1
            "readiness_top1_agreement": agree,
        },
        output,
    )
    print(f"saved {output}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
