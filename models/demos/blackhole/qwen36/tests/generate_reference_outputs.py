# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Generate the HF reference ``.refpt`` used by the qwen36 token-accuracy demo case.

Same artifact format and teacher-forcing convention as
``models/tt_transformers/tests/generate_reference_outputs.py``: a natural-text token
sequence plus the HF model's top-5 prediction at every position, so a TT run can be
scored token-by-token without inheriting its own mistakes.

    reference_tokens: [1, total_length]  tokens of the reference text
    top5_tokens:      [total_length, 5]  row i = HF top-5 continuation of position i

The demo prefills the first half and decodes the second half with the *reference* token
fed back each step (teacher forcing), scoring its own prediction against ``top5_tokens``.

This is the qwen36-local generator because the Qwen3.5/3.6 checkpoints need the text-only
HF classes picked explicitly (composite VLM config + a separate MoE class), exactly as
``Qwen36ModelArgs.load_state_dict`` does.

Usage (CPU, ~fp32 reference)::

    export HF_MODEL=Qwen/Qwen3.6-27B
    python models/demos/blackhole/qwen36/tests/generate_reference_outputs.py

    export HF_MODEL=Qwen/Qwen3.6-35B-A3B
    python models/demos/blackhole/qwen36/tests/generate_reference_outputs.py
"""

import argparse
import bz2
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
from loguru import logger

# Anchored to the repo, not the caller's cwd, so the script works from anywhere (the shared
# generator resolves its corpus off __file__ for the same reason).
_REPO_ROOT = Path(__file__).resolve().parents[5]
REFERENCE_OUTPUTS_DIR = _REPO_ROOT / "models/tt_transformers/tests/reference_outputs"
REFERENCE_TEXT = _REPO_ROOT / "models/tt_transformers/tests/tale-of-two-cities.txt.bz2"


def _resolve_checkpoint(hf_model):
    """Snapshot ``hf_model`` if it is a hub id, mirroring ``Qwen36ModelArgs.__init__``."""
    if os.path.isfile(os.path.join(hf_model, "config.json")):
        return hf_model
    from huggingface_hub import snapshot_download

    return snapshot_download(hf_model, local_files_only=os.getenv("HF_HUB_OFFLINE") == "1")


def _model_name(ckpt_dir):
    """``ModelArgs.model_name`` for a checkpoint dir: the HF repo name, not the snapshot sha."""
    parts = os.path.normpath(ckpt_dir).split(os.sep)
    if "snapshots" in parts:
        return parts[parts.index("snapshots") - 1].split("--")[-1]
    return os.path.basename(os.path.normpath(ckpt_dir))


def _load_hf_text_model(ckpt_dir, is_moe, dtype):
    """Load the text-only HF model, mirroring ``Qwen36ModelArgs.load_state_dict``'s class choice."""
    if is_moe:
        from transformers.models.qwen3_5_moe import Qwen3_5MoeForCausalLM as _HFForCausalLM
        from transformers.models.qwen3_5_moe import Qwen3_5MoeTextConfig as _HFTextConfig
    else:
        from transformers.models.qwen3_5 import Qwen3_5ForCausalLM as _HFForCausalLM
        from transformers.models.qwen3_5 import Qwen3_5TextConfig as _HFTextConfig

    text_config = _HFTextConfig.from_pretrained(ckpt_dir)
    model = _HFForCausalLM.from_pretrained(ckpt_dir, config=text_config, dtype=dtype)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Generate a Qwen3.5/3.6 reference .refpt on CPU")
    parser.add_argument("--total-length", type=int, default=1024, help="Reference tokens (half prompt, half scored)")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32", help="HF model dtype on CPU")
    parser.add_argument("--output", default=None, help="Output .refpt path (default: reference_outputs/<model>.refpt)")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    # Deliberately ttnn-free: this is a pure-HF CPU script, runnable on any host.
    ckpt_dir = _resolve_checkpoint(os.environ.setdefault("HF_MODEL", "Qwen/Qwen3.6-27B"))
    model_name = _model_name(ckpt_dir)
    output = Path(args.output) if args.output else REFERENCE_OUTPUTS_DIR / f"{model_name}.refpt"

    with open(os.path.join(ckpt_dir, "config.json")) as f:
        config = json.load(f)
    # Same source of truth as Qwen36ModelArgs (text_config.num_experts), not the outer
    # model_type: on a composite VLM checkpoint the outer type need not mention MoE, and
    # loading a MoE checkpoint through the dense class leaves the experts uninitialized.
    is_moe = bool(config.get("text_config", config).get("num_experts") or 0)

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    logger.info(f"Loading HF reference for {model_name} from {ckpt_dir} ({args.dtype}, moe={is_moe})")
    model = _load_hf_text_model(ckpt_dir, is_moe, dtype)

    with bz2.open(REFERENCE_TEXT, "rt", encoding="utf-8") as f:
        text = f.read()
    # Raw text (no chat template): the reference is next-token prediction over natural prose.
    encoded = tokenizer(text, add_special_tokens=False)["input_ids"][: args.total_length]
    assert len(encoded) == args.total_length, f"reference text too short: {len(encoded)} < {args.total_length}"
    tokens = torch.tensor([encoded], dtype=torch.long)

    # One forward over the whole reference: splitting it would restart positions at 0 for every
    # chunk after the first, so those rows would not be HF's real continuation. A reference long
    # enough to need splitting has to thread past_key_values through instead.
    logger.info(f"Forward pass over {args.total_length} tokens")
    with torch.no_grad():
        logits = model(tokens).logits
    top5_tokens = torch.topk(logits, k=5, dim=-1).indices.squeeze(0).cpu()  # [total_length, 5]
    # Row i predicts token i+1, so the last row has no target to score against.
    targets = tokens[0, 1:]
    scored = top5_tokens[: len(targets)]
    top1_correct = (scored[:, 0] == targets).tolist()
    top5_correct = (scored == targets.unsqueeze(1)).any(dim=1).tolist()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "reference_tokens": tokens,
            "top5_tokens": top5_tokens,
            "metadata": {
                "hf_model": ckpt_dir,
                "model_name": model_name,
                "dtype": args.dtype,
                "total_length": args.total_length,
                "reference_text": str(REFERENCE_TEXT),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        },
        output,
    )
    logger.info(f"Saved {output} (reference_tokens {tuple(tokens.shape)}, top5_tokens {tuple(top5_tokens.shape)})")
    # Self-consistency of the reference itself: how well HF predicts the prose it was scored on.
    # This is the ceiling a TT run is measured against, not a pass/fail gate.
    n = len(top1_correct)
    logger.info(
        f"HF reference vs text: top-1 {100 * sum(top1_correct) / n:.2f}%, top-5 {100 * sum(top5_correct) / n:.2f}%"
    )


if __name__ == "__main__":
    main()
