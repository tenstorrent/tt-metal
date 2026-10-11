# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Padded prefill must not write unadmitted pages or another request's cache."""

import argparse
import json
from pathlib import Path

import torch

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tt.generator import build_generator

ROOT = Path(__file__).resolve().parents[1]


def main(args):
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    generator = None
    try:
        generator = build_generator(
            ROOT, mesh, batch_size=args.batch, **({"layer_indices": [0, 3]} if args.reduced else {})
        )
        traces = dict(generator.traces)
        table = torch.stack([torch.arange(4096, dtype=torch.int32) % 2048 + slot * 2048 for slot in range(args.batch)])
        checks = []
        for length in (64, 65, 96, 97, 129, 257, 4129, 4161):
            ids = (torch.arange(length, dtype=torch.int32) % 127 + 32)[None]

            def request(mapping, split, verify_neighbor=False):
                generator.reset()
                if args.batch == 2:
                    generator.prefill_forward(
                        torch.arange(32, 65)[None],
                        page_table=mapping,
                        kv_cache=generator.kv_cache,
                        prompt_lens=[33],
                        slots=[1],
                        sample=False,
                    )

                def neighbor_cache():
                    kinds = [
                        0,
                        next(i for i, layer in enumerate(generator.model.layers) if layer.kind == "full_attention"),
                    ]
                    return [
                        ttnn.to_torch(ttnn.get_device_tensors(cache)[0])[2048:2050].clone()
                        for index in kinds
                        for cache in generator.kv_cache[index]
                    ]

                cache_before = neighbor_cache() if verify_neighbor else None
                start = min(31, length) if split else 0
                if start:
                    generator.prefill_forward(
                        ids[:, :start],
                        page_table=mapping,
                        kv_cache=generator.kv_cache,
                        prompt_lens=[start],
                        slots=[0],
                        sample=False,
                    )
                pref = generator.prefill_forward(
                    ids[:, start:],
                    page_table=mapping,
                    kv_cache=generator.kv_cache,
                    prompt_lens=[length - start],
                    slots=[0],
                    start_pos=[start],
                    sample=False,
                )
                if verify_neighbor:
                    assert all(
                        torch.equal(a, b) for a, b in zip(cache_before, neighbor_cache())
                    ), "Unrelated live cache pages changed"
                values = pref.argmax(-1).flatten().tolist()
                positions = [length] + ([33] if args.batch == 2 else [])
                # At an exact page boundary the scheduler must admit the next
                # logical page for decode; prefill still owned only its pages.
                decode_mapping = mapping.clone()
                decode_mapping[0, length // 32] = table[0, length // 32]
                dec = generator.decode_forward(
                    values, positions, page_table=decode_mapping, kv_cache=generator.kv_cache, sample=False
                )
                return pref, dec

            for split in (False, True):
                baseline = request(table, split)
                for alias in [0, 2048] if args.batch == 2 else [0]:
                    minimal = table.clone()
                    minimal[0, (length + 31) // 32 :] = alias
                    if args.batch == 2:
                        minimal[1, 2:] = 0
                    verify_neighbor = args.batch == 2 and alias == 2048 and length == 65 and not split
                    actual = request(minimal, split, verify_neighbor=verify_neighbor)
                    same = [torch.equal(a, b) for a, b in zip(actual, baseline)]
                    check = dict(
                        length=length,
                        split_at=31 if split else None,
                        unused_page_alias=alias,
                        prefill_equal=same[0],
                        decode_equal=same[1],
                        neighbor_cache_bitwise_preserved=True if verify_neighbor else None,
                        main_cosine=torch.nn.functional.cosine_similarity(
                            actual[0][0].float(), baseline[0][0].float(), dim=-1
                        ).item(),
                    )
                    checks.append(check)
                    args.output.write_text(
                        json.dumps(dict(batch=args.batch, layers=2 if args.reduced else 52, checks=checks), indent=2)
                        + "\n"
                    )
                    print("MINIMAL_ADMISSION", check, flush=True)
                    assert all(same), check
        assert generator.traces == traces and generator.counters["captures"] == generator.TRACE_COUNT
        result = dict(
            batch=args.batch,
            layers=2 if args.reduced else 52,
            checks=checks,
            traces={str(k): str(v) for k, v in traces.items()},
            counters=dict(generator.counters),
            passed=True,
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        if generator:
            generator.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, choices=(1, 2), default=2)
    parser.add_argument("--reduced", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
