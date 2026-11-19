# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""HF teacher scores on actual, non-repeating book continuation tokens."""

import bz2
import hashlib
import json
from pathlib import Path


def main():
    import torch
    from huggingface_hub import snapshot_download
    from readiness_check.schema import Reference, ReferenceEntry, save_reference
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(8)
    model_id = "ornith-ai/Ornith-1.5-9B"
    revision = "489cb97981b8654bcfcf30ce1f94ed1b62e07b53"
    checkpoint = snapshot_download(model_id, revision=revision, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    source = Path("models/ttt_compat/tests/tale-of-two-cities.txt.bz2")
    text = bz2.decompress(source.read_bytes()).decode()
    # Skip the Gutenberg front matter; retain a contiguous, unmodified excerpt.
    ids = tokenizer.encode(text, add_special_tokens=False)[4096 : 4096 + 640]
    tokens = torch.tensor([ids])
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, dtype=torch.bfloat16, local_files_only=True, trust_remote_code=True
    ).eval()
    with torch.inference_mode():
        logits = model(input_ids=tokens[:, :-1], use_cache=False).logits
        topk = logits[0, 127:].float().topk(100, dim=-1).indices.int()
    entry = ReferenceEntry(tokenizer.decode(ids[:128]), tokens[:, :128], tokens[:, 128:], topk, 128)
    out = Path("models/autoports/ornith_ai_ornith_1_5_9b/doc/datatype_sweep")
    save_reference(
        Reference(100, model_id, [entry], {"eos_id": tokenizer.eos_token_id, "pad_id": tokenizer.pad_token_id}),
        out / "natural_text.refpt",
    )
    short_entry = ReferenceEntry(
        tokenizer.decode(ids[:32]),
        tokens[:, :32],
        tokens[:, 32:288],
        logits[0, 31:287].float().topk(100, dim=-1).indices.int(),
        32,
    )
    save_reference(
        Reference(100, model_id, [short_entry], {"eos_id": tokenizer.eos_token_id, "pad_id": tokenizer.pad_token_id}),
        out / "short_prefix.refpt",
    )
    (out / "natural_text_provenance.json").write_text(
        json.dumps(
            {
                "model": model_id,
                "revision": revision,
                "source": str(source),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "token_offset": 4096,
                "prompt_tokens": 128,
                "scored_tokens": 512,
                "forced_inputs": "actual contiguous book tokens, not HF greedy predictions",
                "prompt_mode": "raw natural-text continuation stress; not a chat quality verdict",
                "hf_dtype": "bfloat16",
                "tokens_sha256": hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    print("NATURAL_REFERENCE_OK", tuple(topk.shape))


if __name__ == "__main__":
    main()
