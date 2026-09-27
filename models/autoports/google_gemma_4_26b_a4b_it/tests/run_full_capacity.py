# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""All-layer maximum-context allocation and nonaligned public prefill probe."""
import argparse
import json
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator import build_generator


def main():
    torch.set_num_threads(8)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("models/autoports/google_gemma_4_26b_a4b_it/doc/full_model")
    )
    root = parser.parse_args().output_dir
    root.mkdir(parents=True, exist_ok=True)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=100000000)
    gen = None
    rows = []
    try:
        gen = build_generator(None, mesh)
        cache, table = gen.model.allocate_cache(slots=1, context=262144)
        (root / "capacity_precision_runtime.json").write_text(
            json.dumps(gen.model.precision_summary(), indent=2) + "\n"
        )
        for length in (262143, 262144):
            started = time.perf_counter()
            logits = gen.prefill_forward(
                torch.full((1, length), 100, dtype=torch.int64), page_table=table, kv_cache=cache, prompt_lens=[length]
            )
            values = gen._read_logits(logits)
            assert values.shape[-2] == 1 and torch.isfinite(values).all()
            if length < 262144:
                gen.decode_forward(
                    torch.tensor([100]), torch.tensor([length], dtype=torch.int32), page_table=table, kv_cache=cache
                )
                tokens = gen._read_tokens()
                assert 0 <= tokens[0] < gen.model.config.vocab_size
                gen._release_trace()
            rows.append(
                dict(
                    prompt_length=length,
                    full_layers=30,
                    cache_context=262144,
                    logical_tail_sliced=True,
                    finite_logits=True,
                    final_position_decode=length < 262144,
                    elapsed_seconds=time.perf_counter() - started,
                )
            )
            (root / "capacity.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(rows[-1], flush=True)
    finally:
        if gen is not None:
            gen.teardown()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
