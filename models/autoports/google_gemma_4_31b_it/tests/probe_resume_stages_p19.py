# SPDX-License-Identifier: Apache-2.0
"""Compare P19 layer-zero stages for fresh and resumed 1024-row chunks.

Prepared but not executed. It reads correctness intermediates only; no performance claims.
"""

import argparse
import bz2
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch

import ttnn

from ..tt.generator import Gemma4Generator
from ..tt.optimized_decoder import OptimizedDecoder
from .probe_prefill_geometry import compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--starts", default="32,64,256")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "models/autoports/google_gemma_4_31b_it/doc/agentic_evals/setup/"
            "bf16-hifi4-kv-down-bfp8-accuracy-p19/precision.json"
        ),
    )
    parser.add_argument(
        "--text-file", type=Path, default=Path("models/tt_transformers/tests/tale-of-two-cities.txt.bz2")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "models/autoports/google_gemma_4_31b_it/doc/agentic_evals/setup/"
            "p19-resume-investigation/p19-resume-stages.json"
        ),
    )
    args = parser.parse_args()
    starts = [int(value) for value in args.starts.split(",")]
    assert starts and all(0 < start and start % 32 == 0 and start + 1024 <= 2048 for start in starts)
    torch.set_num_threads(8)
    text = bz2.open(args.text_file, "rt").read() if args.text_file.suffix == ".bz2" else args.text_file.read_text()
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024)
    gen = None
    try:
        gen = Gemma4Generator(
            mesh,
            layer_indices=(0, 5),
            batch_size=1,
            max_seq_len=4096,
            precision_config=args.config,
        )
        if gen.model.precision_config["config_id"] != "stage12_bf16_hifi4_kv_down_bfp8_p19":
            raise ValueError("This probe is pinned to the P19 policy")
        if gen.model.precision_config["ccl_dtype"]["prefill"] != "bfloat16":
            raise ValueError("Expected P19 BF16 prefill CCL")
        gen.prepare_decode()
        compare_rows = 256
        prompt_tokens = max(starts) + 1024
        tokens = torch.tensor(gen.tokenizer.encode(text, add_special_tokens=True)[:prompt_tokens], dtype=torch.int32)
        assert tokens.numel() == prompt_tokens
        layer = gen.model.layers[0]
        captures = {}
        active = False
        run = ""
        absolute_start = 0

        def snapshot(name, tensor):
            if not active:
                return
            assert len(tensor.shape) == 4, (name, tensor.shape)
            # The fresh baseline retains every absolute row under comparison;
            # continuations retain their first 256 rows. All chip shards are read.
            count = max(starts) + compare_rows if run == "fresh_1024" else compare_rows
            shards = []
            for chip in ttnn.get_device_tensors(tensor):
                host = ttnn.to_torch(chip).float()
                shards.append(host.movedim(2, 0).reshape(host.shape[2], -1)[:count].clone())
            stages = captures.setdefault(run, {})
            assert name not in stages, (run, name)
            stages[name] = {
                "rows": torch.cat(shards, dim=1),
                "shape": list(tensor.shape),
                "padded_shape": list(tensor.padded_shape),
                "dtype": str(tensor.dtype),
                "absolute_start": absolute_start,
            }

        original_forward = layer.prefill_chunk_forward
        original_norm = layer._norm
        original_linear = layer._linear
        original_local_linear = OptimizedDecoder._linear
        original_qkv = layer._qkv
        original_finish = layer._finish

        def forward(x, **kwargs):
            nonlocal active, absolute_start
            absolute_start = int(kwargs["chunk_start"])
            active = run.endswith("_suffix") or (run == "fresh_1024" and absolute_start == 0)
            try:
                snapshot("layer.input", x)
                output = original_forward(x, **kwargs)
                snapshot("layer.output", output)
                return output
            finally:
                active = False

        def norm(x, name=None):
            snapshot(f"norm[{name}].input", x)
            output = original_norm(x, name)
            snapshot(f"norm[{name}].output", output)
            return output

        def linear(x, name):
            snapshot(f"linear[{name}].input", x)
            output = original_linear(x, name)
            snapshot(f"linear[{name}].output", output)
            return output

        def local_linear(self, x, name):
            output = original_local_linear(self, x, name)
            if self is layer:
                snapshot(f"linear[{name}].before_ccl", output)
            return output

        def qkv(x, cos, sin, *, decode):
            output = original_qkv(x, cos, sin, decode=decode)
            for name, tensor in zip(("q", "k", "v"), output):
                snapshot(f"qkv.{name}", tensor)
            return output

        def finish(x, attention, *, decode):
            snapshot("finish.residual_input", x)
            snapshot("finish.attention_input", attention)
            return original_finish(x, attention, decode=decode)

        cache_before = mesh.num_program_cache_entries()
        with ExitStack() as stack:
            for owner, name, wrapper in (
                (layer, "prefill_chunk_forward", forward),
                (layer, "_norm", norm),
                (layer, "_linear", linear),
                (OptimizedDecoder, "_linear", local_linear),
                (layer, "_qkv", qkv),
                (layer, "_finish", finish),
            ):
                stack.enter_context(patch.object(owner, name, wrapper))
            run = "fresh_1024"
            gen.reset()
            for _, _, hidden in gen._prefill_hidden_chunks(tokens, slot=0, start=0, end=1024):
                del hidden
            for start in starts:
                run = f"resume_{start}_prefix"
                gen.reset()
                for _, _, hidden in gen._prefill_hidden_chunks(tokens, slot=0, start=0, end=start):
                    del hidden
                run = f"resume_{start}_suffix"
                for _, _, hidden in gen._prefill_hidden_chunks(tokens, slot=0, start=start, end=start + 1024):
                    del hidden
        baseline = captures["fresh_1024"]
        comparisons = []
        for run, stages in captures.items():
            if run == "fresh_1024":
                continue
            assert list(stages) == list(baseline), (run, list(stages), list(baseline))
            for name, capture in stages.items():
                start = capture["absolute_start"]
                reference_rows = baseline[name]["rows"][start : start + compare_rows]
                resumed_rows = capture["rows"][:compare_rows]
                bands = []
                for offset in range(0, compare_rows, 32):
                    end = offset + 32
                    bands.append(
                        {
                            "absolute_range": [start + offset, start + end],
                            **compare(reference_rows[offset:end], resumed_rows[offset:end]),
                        }
                    )
                comparisons.append(
                    {
                        "case": run,
                        "stage": name,
                        "absolute_range": [start, start + compare_rows],
                        "reference_shape": baseline[name]["shape"],
                        "source_shape": capture["shape"],
                        "source_padded_shape": capture["padded_shape"],
                        "dtype": capture["dtype"],
                        **compare(reference_rows, resumed_rows),
                        "bands_32_rows": bands,
                    }
                )
        model_dir = Path(__file__).resolve().parents[1]
        source_files = [
            Path(__file__).resolve(),
            *[
                model_dir / "tt" / name
                for name in ("optimized_decoder.py", "multichip_decoder.py", "prefill.py", "generator.py", "model.py")
            ],
        ]
        report = {
            "scope": "explicit_p19_real_layers_0_5_layer0_intermediates",
            "purpose": "Locate the earliest stage difference between fresh and resumed 1024-row chunks",
            "prompt_tokens": prompt_tokens,
            "fresh_physical_chunks": [[0, 1024]],
            "resume_physical_chunks": {str(start): [start, start + 1024] for start in starts},
            "compared_absolute_rows": {str(start): [start, start + compare_rows] for start in starts},
            "comparison_band_rows": 32,
            "context": 4096,
            "source_text": str(args.text_file),
            "precision_config": gen.model.precision_config,
            "precision_config_path": str(args.config),
            "program_cache_before": cache_before,
            "program_cache_after": mesh.num_program_cache_entries(),
            "comparisons": comparisons,
            "source_sha256": {
                str(path.relative_to(Path.cwd())): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in source_files
            },
            "limitations": [
                "Explicit correctness readbacks on all four chips; no performance measurement.",
                "Only the first 256 resumed rows are compared, both as one interval and in 32-row bands.",
                "Only the first fresh chunk is captured; all requested comparison rows lie inside it.",
                "qkv.k and qkv.v are captured before their paged cache write; this does not compare the stored cache contents.",
                "A reduced layer set is not evidence that later full-model layers are equivalent.",
                "before_ccl names capture the local projection before any output CCL; other projections have no output CCL.",
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"output": str(args.output), "comparisons": len(comparisons)}))
    finally:
        if gen is not None:
            gen.release_persistent_capture()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
