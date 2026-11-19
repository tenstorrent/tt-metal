# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Short-prefix full-model accuracy per user, or two-layer KV operation smoke."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-kwargs-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--lengths", default="31,32,33,127,129")
    args = parser.parse_args()
    import torch
    from readiness_check.schema import load_reference

    import ttnn
    from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator import build_generator
    from models.common.sampling.generator import SamplingParams

    torch.set_num_threads(8)
    root = Path(__file__).resolve().parents[1]
    kwargs = json.loads(args.build_kwargs_file.read_text())
    kwargs.update(max_batch_size=args.batch, max_seq_len=args.context)
    if args.smoke:
        kwargs["layer_indices"] = [0, 3]
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=256 * 1024 * 1024)
    gen = None
    result = {"status": "running", "batch": args.batch, "build_kwargs": kwargs, "smoke": args.smoke}
    try:
        gen = build_generator(root, mesh, **kwargs)
        if args.smoke:
            lengths = [int(value) for value in args.lengths.split(",")]
            programs = captures = None
            for length in lengths:
                gen.generate(list(range(100, 100 + length)), 4)
                if programs is None:
                    programs = mesh.num_program_cache_entries()
                    captures = gen.counters["captures"]
                    mesh.set_program_cache_misses_allowed(False)
                else:
                    assert mesh.num_program_cache_entries() == programs
                    assert gen.counters["captures"] == captures
            result["lengths"] = lengths
            result["programs_after_first_prepared_request"] = programs
            result["captures_after_first_prepared_request"] = captures
        else:
            ref = load_reference(root / "doc/datatype_sweep/short_prefix.refpt").entries[0]
            prompt = ref.prompt_tokens.repeat(args.batch, 1)
            params = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)
            gen.prepare_decode(kv_cache=gen.cache, page_table=gen.page_table, slots=tuple(range(args.batch)))
            first = gen.prefill_forward(
                prompt,
                prompt_lens=[32] * args.batch,
                page_table=gen.page_table,
                kv_cache=gen.cache,
                sampling_params=params,
            ).reshape(args.batch)
            predictions = [first]
            for step in range(1, ref.num_generated):
                forced = torch.full((args.batch, 1), int(ref.generated_tokens[0, step - 1]), dtype=torch.int64)
                pred = gen.decode_forward(
                    forced,
                    [32 + step - 1] * args.batch,
                    page_table=gen.page_table,
                    kv_cache=gen.cache,
                    sampling_params=params,
                    reload_inputs=True,
                    reload_sampling_params=step == 1,
                    reset_sampling_state=step == 1,
                    prompt_tokens=prompt if step == 1 else None,
                    output_tokens=forced if step == 1 else None,
                ).reshape(args.batch)
                predictions.append(pred)
            preds = torch.stack(predictions)
            result["per_user"] = [
                {
                    f"top{k}": float((preds[:, slot, None] == ref.topk_tokens[:, :k]).any(-1).float().mean())
                    for k in (1, 5, 100)
                }
                for slot in range(args.batch)
            ]
            result["all_users_same"] = bool((preds == preds[:, :1]).all())
            result["prompt_tokens"] = 32
            result["scored_tokens"] = ref.num_generated
            assert result["all_users_same"]
            assert all(r["top1"] >= 0.90 and r["top5"] >= 0.98 and r["top100"] == 1.0 for r in result["per_user"])
        result["trace_counters"] = dict(gen.counters)
        result["runtime_policy"] = gen.model.precision_summary()
        result["status"] = "pass"
    except Exception as error:
        result.update(status="fail", error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
