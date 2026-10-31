"""Shared readiness runners on one retained generator and pinned HF control."""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from readiness_check import run_autoregressive, run_prefill_check, run_teacher_forcing

import ttnn

from ..tt.generator import Gemma4Generator
from ..tt.weights import HF_MODEL, HF_REVISION
from .full_generation_evidence import performance, qualitative
from .run_full_batch import inventory, memory


def context_boundaries(gen, root):
    prompt = json.loads((root / "aime24_prompt.json").read_text())["tokens"]
    records = []
    for length in (262143, 262144):
        ids = (prompt * ((length + len(prompt) - 1) // len(prompt)))[:length]
        if length < gen.context:
            out = gen.generate(ids, 1)
            last = gen.decode_forward(
                torch.tensor([[out[0]]]), torch.tensor([length]), page_table=gen.page_table, kv_cache=gen.kv_cache
            )
            positions = ttnn.to_torch(ttnn.get_device_tensors(gen.positions)[0]).reshape(-1).tolist()
            assert positions == [262144]
        else:
            gen.reset()
            gen._reserve_requests([length], 0)
            out = gen.prefill_forward(
                torch.tensor([ids]), page_table=gen.page_table, kv_cache=gen.kv_cache, prompt_lens=[length]
            )
        records.append(dict(prompt_tokens=length, output=out))
        print("FINAL_FULL_CONTEXT", records[-1], flush=True)
    with patch.object(gen, "_host_logits", wraps=gen._host_logits) as host_logits:
        host_tokens = gen.generate(
            prompt, 8, host_sampling=True, sampling_params={"temperature": [0.0], "top_k": [1], "top_p": [1.0]}
        )
        assert host_logits.call_count == len(host_tokens)
    compatible = host_tokens == gen.generate(prompt, 8)
    assert compatible
    (root / "full_context_post_barrier.json").write_text(
        json.dumps(
            dict(
                scope="full60layers",
                context=gen.context,
                boundaries=records,
                last_decode_input_position=262143,
                last_decode_output=last,
                positions_after=positions,
                host_sampling_compatibility_equal=compatible,
                host_sampling_logit_reads=len(host_tokens),
            ),
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--long-context", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    model_dir = Path("models/autoports/google_gemma_4_31b_it")
    root = model_dir / "doc/full_model"
    prompt = json.loads((root / "aime24_prompt.json").read_text())
    reference = json.loads((root / "reference_metadata.json").read_text())
    assert reference["revision"] == HF_REVISION and reference["actual_generated_tokens"] == 100
    ttnn.CONFIG.throw_exception_on_fallback = True
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024)
    try:
        start = time.perf_counter()
        gen = Gemma4Generator(mesh)
        constructed = memory(mesh)
        owned = inventory(gen)
        gen.prepare_decode()
        (root / "memory_b1.json").write_text(
            json.dumps(
                dict(
                    construction_stages=gen.model.allocation_bytes,
                    constructed=constructed,
                    inventory=owned,
                    prepared=memory(mesh),
                ),
                indent=2,
            )
            + "\n"
        )
        result = {"scope": "full60layers", "preparation_seconds": time.perf_counter() - start}
        factory = lambda **kwargs: gen
        for name, module, fn in [
            ("prefill", run_prefill_check, run_prefill_check.run_prefill_check),
            ("decode", run_teacher_forcing, run_teacher_forcing.run_teacher_forcing),
        ]:
            with patch.object(module, "_import_build_generator", return_value=factory):
                result[name] = fn(model_dir=model_dir, reference_path=Path(reference["reference"]), mesh_device=mesh)
            (root / "readiness_accuracy.json").write_text(json.dumps(result, indent=2) + "\n")
        # The stock AR runner takes raw text and has no revision/chat-template
        # parameters. Bind its tokenizer boundary to the already rendered pinned
        # prompt and reuse the fresh exact100-token HF control, verifying identity.
        rendered = prompt["rendered"]
        rendered_file = root / "aime24_rendered.txt"
        rendered_file.write_text(rendered)

        class RenderedTokenizer:
            def encode(self, text, add_special_tokens=True):
                assert text.strip() == rendered.strip()
                return prompt["tokens"]

            def decode(self, *args, **kwargs):
                return gen.tokenizer.decode(*args, **kwargs)

        def hf_control(**kwargs):
            assert kwargs["hf_model_id"] == HF_MODEL
            assert kwargs["prompt_token_ids"] == prompt["tokens"] and kwargs["max_new_tokens"] == 100
            return reference["tokens"]

        with (
            patch.object(run_autoregressive, "_import_build_generator", return_value=factory),
            patch.object(
                run_autoregressive,
                "AutoTokenizer",
                SimpleNamespace(from_pretrained=lambda *a, **kw: RenderedTokenizer()),
            ),
            patch.object(run_autoregressive, "_hf_generate_greedy", side_effect=hf_control),
        ):
            run_autoregressive.run_autoregressive(
                model_dir=model_dir,
                hf_model_id=HF_MODEL,
                prompt_file=rendered_file,
                mesh_device=mesh,
                output_dir=root / "autoregressive",
                max_new_tokens=100,
            )
        qualitative(gen, root)
        if args.long_context:
            context_boundaries(gen, root)
        performance(gen, root)
        result["counters"] = gen.counters
        result["traces"] = {"model": str(gen.model_trace), "sampling": str(gen.sample_trace)}
        (root / "readiness_accuracy.json").write_text(json.dumps(result, indent=2) + "\n")
        for phase in ("prefill", "decode"):
            assert all(row["top5"] >= 0.98 and row["top100"] == 1 for row in result[phase]), result
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
