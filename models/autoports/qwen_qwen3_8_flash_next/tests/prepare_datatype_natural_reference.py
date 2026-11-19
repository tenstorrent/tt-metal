# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only exact-HF natural-text reference for the datatype sweep.

The continuation contains actual consecutive book tokens, not greedy samples.
The pinned original HF model scores every continuation position in one causal
forward, matching readiness_check.generate's reference extraction convention.
"""

from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import os
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from huggingface_hub import snapshot_download
from readiness_check.schema import Reference, ReferenceEntry, load_reference, save_reference
from transformers import AutoTokenizer

from models.autoports.qwen_qwen3_8_flash_next.tests import full_model_hf_reference as exact_hf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--scored-tokens", type=int, default=512)
    parser.add_argument("--token-offset", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path, default=exact_hf.MODEL_DIR / "doc/datatype_sweep/natural_text.refpt")
    args = parser.parse_args()
    if min(args.prompt_len, args.scored_tokens, args.threads) < 1 or args.token_offset < 0:
        raise ValueError("Lengths and threads must be positive; offset must be nonnegative")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    exact_hf.EVIDENCE_DIR = output.parent / "natural_text_hf"
    exact_hf.EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    repo = exact_hf.MODEL_DIR.parents[2]
    book = repo / "models/ttt_compat/tests/tale-of-two-cities.txt.bz2"
    text = bz2.decompress(book.read_bytes()).decode("utf-8")
    snapshot = Path(snapshot_download(exact_hf.MODEL_ID, revision=exact_hf.REVISION, local_files_only=True))
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    encoded = tokenizer.encode(text, add_special_tokens=False)
    count = args.prompt_len + args.scored_tokens
    selected = encoded[args.token_offset : args.token_offset + count]
    if len(selected) != count:
        raise ValueError("Book is too short for requested token range")
    tokens = torch.tensor(selected, dtype=torch.long).unsqueeze(0)
    (exact_hf.EVIDENCE_DIR / "tokens.json").write_text(json.dumps(selected) + "\n")
    (exact_hf.EVIDENCE_DIR / "text.txt").write_text(tokenizer.decode(selected, skip_special_tokens=False))
    model = exact_hf.load_exact_text_model()
    if any(t.device.type != "cpu" for t in model.parameters()):
        raise RuntimeError("Natural reference must remain CPU-only")
    with torch.inference_mode():
        logits = model(tokens).logits[0, args.prompt_len - 1 : count - 1, :]
        if not torch.isfinite(logits).all():
            raise RuntimeError("HF reference contains non-finite logits")
        topk_logits, topk_tokens = logits.topk(100, dim=-1)
    reference = Reference(
        k=100,
        hf_model_id=exact_hf.MODEL_ID,
        entries=[
            ReferenceEntry(
                prompt_text=tokenizer.decode(selected[: args.prompt_len], skip_special_tokens=False),
                prompt_tokens=tokens[:, : args.prompt_len].clone(),
                generated_tokens=tokens[:, args.prompt_len :].clone(),
                topk_tokens=topk_tokens.to(torch.int32),
                tf_prompt_len=args.prompt_len,
            )
        ],
        token_ids_meta={
            "bos_id": tokenizer.bos_token_id,
            "eos_id": tokenizer.eos_token_id,
            "pad_id": tokenizer.pad_token_id,
        },
    )
    save_reference(reference, output)
    # Preserve scored top-100 logit values and margins for later close-call diagnostics.
    torch.save(
        {"topk_logits": topk_logits.cpu(), "topk_tokens": topk_tokens.cpu()},
        exact_hf.EVIDENCE_DIR / "top100_logits.pt",
    )
    loaded = load_reference(output)
    assert loaded.entries[0].topk_tokens.shape == (args.scored_tokens, 100)
    assert torch.equal(loaded.entries[0].generated_tokens, tokens[:, args.prompt_len :])
    assert "ttnn" not in sys.modules, "CPU-only reference unexpectedly imported TTNN"
    provenance = {
        "reference": str(output),
        "schema": "readiness_v1",
        "schema_roundtrip": "passed readiness_check.schema.load_reference",
        "hf_model": exact_hf.MODEL_ID,
        "revision": exact_hf.REVISION,
        "snapshot": str(snapshot),
        "model_implementation": "transformers.models.qwen4_exp.modeling_qwen4_exp.Qwen4ExpForCausalLM",
        "transformers_version": transformers.__version__,
        "loader": str(Path(exact_hf.__file__).resolve()),
        "loader_evidence": str(exact_hf.EVIDENCE_DIR / "loader.json"),
        "prompt_type": "raw-natural-text",
        "chat_template": False,
        "purpose": "per-position teacher-forced HF numerical agreement; not chat quality evaluation",
        "continuation": "actual consecutive book tokens, not HF-greedy output",
        "book": str(book),
        "book_utf8_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "book_tokenization": "add_special_tokens=False; exact original UTF-8 text including line endings",
        "book_token_count": len(encoded),
        "token_offset": args.token_offset,
        "token_ids_sha256": hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),
        "token_ids_path": str(exact_hf.EVIDENCE_DIR / "tokens.json"),
        "decoded_text_path": str(exact_hf.EVIDENCE_DIR / "text.txt"),
        "prompt_tokens": args.prompt_len,
        "scored_tokens": args.scored_tokens,
        "topk_shape": list(topk_tokens.shape),
        "scored_logits_shape": list(logits.shape),
        "scored_logits_dtype": str(logits.dtype),
        "logit_position_slice": [args.prompt_len - 1, count - 1],
        "top100_logits_path": str(exact_hf.EVIDENCE_DIR / "top100_logits.pt"),
        "hf_top1_matches_actual_book_tokens": float((topk_tokens[:, 0] == tokens[0, args.prompt_len :]).float().mean()),
        "device": "cpu",
        "pid": os.getpid(),
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_threads": torch.get_num_threads(),
        "ttnn_imported": "ttnn" in sys.modules,
        "command": shlex.join([sys.executable, *sys.argv]),
        "elapsed_seconds": time.monotonic() - started,
    }
    output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        json.dumps({"reference": str(output), "schema_valid": True, "elapsed_seconds": provenance["elapsed_seconds"]})
    )


if __name__ == "__main__":
    main()
