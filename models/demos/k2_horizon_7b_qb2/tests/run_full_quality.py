"""Shared chat suite and packaged autoregressive comparison on all36 layers."""

import json
from pathlib import Path
from unittest.mock import patch

import torch
from readiness_check import run_autoregressive
from readiness_check.schema import load_reference
from transformers import AutoTokenizer

import ttnn

from ..tt.generator import K2Generator
from ..tt.model import HF_MODEL, HF_REVISION

DOC = Path("models/demos/k2_horizon_7b_qb2/doc/full_model")
MODEL_DIR = DOC.parents[1]


def run(gen, *, shared_suite=True, output_dir="autoregressive"):
    mesh = gen.mesh
    controls = json.loads((DOC / "hf_qualitative.json").read_text())
    assert len(controls) == 6, "HF shared controls must finish first"
    ref = load_reference(MODEL_DIR / "readiness_aime24_chat.refpt")
    entry = ref.entries[0]
    tok = AutoTokenizer.from_pretrained(HF_MODEL, revision=HF_REVISION, trust_remote_code=True)
    rendered = tok.decode(entry.prompt_tokens[0], skip_special_tokens=False)
    (DOC / "aime_rendered_prompt.txt").write_text(rendered)
    # The packaged plain-text runner adds BOS itself. Remove only that already
    # rendered BOS so its encoded input exactly matches the fresh chat reference.
    ids = entry.prompt_tokens[0].tolist()
    assert ids[0] == tok.bos_token_id
    prompt = tok.decode(ids[1:], skip_special_tokens=False)
    prompt_path = DOC / "aime_runner_prompt.txt"
    prompt_path.write_text(prompt)
    assert tok.encode(prompt, add_special_tokens=True) == ids
    original_encode = tok.encode

    def preserve_template(text, *args, **kwargs):
        # run_autoregressive.strip() removes the required newline after
        # <ifm|think>. Restore that exact template text before tokenization.
        assert text == prompt.strip()
        encoded = original_encode(prompt, *args, **kwargs)
        assert encoded == ids
        return encoded

    # Reuse the fresh pinned AIME HF continuation, not another model or
    # prompt. The packaged runner still owns the comparison artifacts.
    def fresh_reference(**kwargs):
        assert kwargs["prompt_token_ids"] == entry.prompt_tokens[0].tolist()
        assert kwargs["max_new_tokens"] == 100
        return entry.generated_tokens[0].tolist()

    with (
        patch.object(run_autoregressive, "_import_build_generator", return_value=lambda **kw: gen),
        patch.object(run_autoregressive, "_hf_generate_greedy", side_effect=fresh_reference),
        patch.object(run_autoregressive.AutoTokenizer, "from_pretrained", return_value=tok),
        patch.object(tok, "encode", side_effect=preserve_template),
    ):
        run_autoregressive.run_autoregressive(
            model_dir=MODEL_DIR,
            hf_model_id=HF_MODEL,
            prompt_file=prompt_path,
            max_new_tokens=100,
            mesh_device=mesh,
            output_dir=DOC / output_dir,
        )
    if not shared_suite:
        return
    records = []
    for record in controls:
        tokens = gen.generate(record["prompt_token_ids"], 128)
        records.append(
            {
                **record,
                "tt_token_ids": tokens,
                "tt_text": tok.decode(tokens, skip_special_tokens=False),
                "tt_perf": gen.last_perf,
            }
        )
        (DOC / "qualitative_outputs.json").write_text(json.dumps(records, indent=2) + "\n")
        print("QUALITATIVE", record["id"], records[-1]["tt_text"], flush=True)


def main():
    torch.set_num_threads(16)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200_000_000)
    gen = None
    try:
        gen = K2Generator(mesh)
        run(gen)
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
