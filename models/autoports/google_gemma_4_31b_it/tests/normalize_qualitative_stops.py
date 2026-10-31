"""Apply pinned HF stop IDs to retained greedy controls without regenerating prefixes."""

import json
from pathlib import Path

from transformers import AutoTokenizer, GenerationConfig

from ..tt.weights import HF_MODEL, HF_REVISION


def main():
    root = Path("models/autoports/google_gemma_4_31b_it/doc/full_model")
    path = root / "qualitative_hf.json"
    rows = json.loads(path.read_text())
    assert len(rows) == 6
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, revision=HF_REVISION)
    config = GenerationConfig.from_pretrained(HF_MODEL, revision=HF_REVISION)
    stop_ids = set(config.eos_token_id)
    original = root / "qualitative_hf_unstopped.json"
    if not original.exists():
        original.write_text(path.read_text())
    for row in rows:
        ids = row["generated_tokens"]
        stop = next((i + 1 for i, t in enumerate(ids) if t in stop_ids), len(ids))
        row["generated_tokens"] = ids[:stop]
        row["completion"] = tokenizer.decode(ids[:stop])
        row["stop_normalization"] = dict(
            original_tokens=len(ids),
            retained_tokens=stop,
            eos_token_ids=sorted(stop_ids),
            basis="Exact prefix of deterministic greedy generation; stop criteria cannot change prior token choices",
        )
    path.write_text(json.dumps(rows, indent=2) + "\n")
    meta_path = root / "qualitative_prompt_format.json"
    meta = json.loads(meta_path.read_text())
    meta["generation"]["eos_token_id"] = sorted(stop_ids)
    meta["generation_config_revision"] = HF_REVISION
    meta[
        "stop_correction"
    ] = "Initial text-only constructor inherited eos1; retained exact greedy prefixes end at first pinned generation_config stop ID. Original outputs preserved in qualitative_hf_unstopped.json. Future loader loads pinned GenerationConfig."
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print([(r["id"], len(r["generated_tokens"])) for r in rows])


if __name__ == "__main__":
    main()
