"""Reconstruct pinned readiness logits and an aligned natural-text reference."""

import argparse
import bz2
import hashlib
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("Inert without --execute; run only in the serialized model queue")
    import torch
    from readiness_check.generate import _generate_one_entry
    from readiness_check.schema import Reference, save_reference
    from transformers import AutoTokenizer

    from models.autoports.google_gemma_4_31b_it.tests.generate_full_reference import load_reference
    from models.autoports.google_gemma_4_31b_it.tt.weights import HF_MODEL, HF_REVISION

    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    assert not (root / "references.json").exists(), "Do not overwrite completed reference evidence"
    started = time.time()
    source = Path("models/autoports/google_gemma_4_31b_it/doc/optimized_full_model")
    metadata_path = source / "reference_metadata.json"
    prompt_path = source / "aime24_prompt.json"
    book = Path("models/tt_transformers/tests/tale-of-two-cities.txt.bz2")
    metadata = json.loads(metadata_path.read_text())
    prompt = json.loads(prompt_path.read_text())
    assert metadata["revision"] == metadata["tokenizer_revision"] == HF_REVISION
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, revision=HF_REVISION, local_files_only=True)
    rendered_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt["prompt"]}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    assert rendered_ids == prompt["tokens"], "Pinned readiness prompt/template changed"
    natural_ids = tokenizer.encode(bz2.decompress(book.read_bytes()).decode("utf-8"), add_special_tokens=False)
    # Skip the book's opening material, then use a contiguous non-repeating span.
    natural_span = natural_ids[4096:5120]
    assert len(natural_span) == 1024
    specs = [
        (
            "aime24",
            rendered_ids,
            metadata["tokens"],
            "Previously saved pinned HF continuation; logits recomputed, no old TT scores reused",
        ),
        (
            "natural",
            natural_span[:512],
            natural_span[512:],
            "Actual book tokens, offset4096, prefix512 and512 scored positions; continuation stress accuracy, not chat quality",
        ),
    ]
    assert len(metadata["tokens"]) == 100
    model = load_reference()
    records = []
    for name, prefix, continuation, basis in specs:
        begin = time.time()
        with torch.inference_mode():
            entry = _generate_one_entry(
                model=model,
                tokenizer=tokenizer,
                prompt_tokens=torch.tensor(prefix, dtype=torch.long),
                gen_tokens=torch.tensor(continuation, dtype=torch.long),
                top_k=100,
                device=torch.device("cpu"),
            )
        path = root / (name + ".refpt")
        save_reference(
            Reference(
                k=100,
                hf_model_id=HF_MODEL,
                entries=[entry],
                token_ids_meta={
                    "bos_id": tokenizer.bos_token_id,
                    "eos_id": tokenizer.eos_token_id,
                    "pad_id": tokenizer.pad_token_id,
                },
            ),
            path,
        )
        records.append(
            {
                "name": name,
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "prompt_tokens": len(prefix),
                "positions": len(continuation),
                "basis": basis,
                "elapsed_seconds": time.time() - begin,
            }
        )
        print("REFERENCE_COMPLETE", json.dumps(records[-1]), flush=True)
    evidence = {
        "hf_model": HF_MODEL,
        "revision": HF_REVISION,
        "tokenizer_revision": HF_REVISION,
        "reference_only": True,
        "regime": "serial_cpu",
        "records": records,
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (metadata_path, prompt_path, book, Path(__file__))
        },
        "start_epoch": started,
        "end_epoch": time.time(),
        "elapsed_seconds": time.time() - started,
    }
    (root / "references.json").write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()
