# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-seed shared-prompt controls using the pre-optimization generator."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
import types
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "models.autoports.ifm_k2_horizon_mova_36b_a4b.tt"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("standalone", "server"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--control", type=Path)
    p.add_argument("--search-loops", action="store_true")
    args = p.parse_args()
    metadata = json.loads((ROOT / "readiness_vllm/optimized/acceptance/qualitative_prompt_format.json").read_text())
    rows = []

    def save():
        args.output.write_text(json.dumps(dict(mode=args.mode, runs=rows), indent=2) + "\n")

    if args.mode == "server":
        import requests

        assert args.control is not None
        controls = json.loads(args.control.read_text())["runs"]
        for row in controls:
            request = dict(
                model=metadata["model"],
                prompt=row["prompt_token_ids"],
                max_tokens=256,
                temperature=0.7,
                top_k=32,
                top_p=0.9,
                seed=row["seed"],
                ignore_eos=True,
                return_token_ids=True,
            )
            response = requests.post("http://localhost:8000/v1/completions", json=request, timeout=180)
            response.raise_for_status()
            response = response.json()
            tokens = response["choices"][0]["token_ids"]
            result = dict(
                prompt_id=row["prompt_id"],
                seed=row["seed"],
                request=request,
                response=response,
                matches_stage9_standalone=tokens == row["token_ids"],
            )
            rows.append(result)
            save()
            print(row["prompt_id"], row["seed"], result["matches_stage9_standalone"], flush=True)
        assert all(r["matches_stage9_standalone"] for r in rows)
        return

    import torch

    import ttnn
    from models.common.sampling.generator import SamplingParams

    # Load the committed Stage9 class under a private module without mutating
    # the candidate file or borrowing its new capture/readback implementation.
    versions = json.loads((ROOT / "doc/optimized_vllm/source_versions.json").read_text())
    relative = "models/autoports/ifm_k2_horizon_mova_36b_a4b/tt/generator.py"
    source = subprocess.check_output(["git", "show", f"{versions['baseline_commit']}:{relative}"])
    assert hashlib.sha256(source).hexdigest() == versions["baseline_generator_sha256"]
    module = types.ModuleType(PACKAGE + "._stage9_control")
    module.__package__ = PACKAGE
    module.__file__ = str(ROOT / "tt/generator.py")
    sys.modules[module.__name__] = module
    exec(compile(source, "<committed-stage9-generator>", "exec"), module.__dict__)
    ttnn.CONFIG.throw_exception_on_fallback = True
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    g = None
    try:
        g = module.build_generator(ROOT, mesh, max_seq_len=524288, max_batch_size=32, page_size=64)
        g.capture()
        found = set()
        groups = (
            [[(pid, seed) for pid in (2, 5) for seed in range(start, start + 16)] for start in range(0, 128, 16)]
            if args.search_loops
            else [[(pid, seed)] for pid in (2, 5) for seed in (11, 42, 53, 7, 0, 1)]
        )
        for pairs in groups:
            n = len(pairs)
            ids = [metadata["prompts"][pid]["tokens"] for pid, _ in pairs]
            lengths = [len(x) for x in ids]
            prompts = torch.zeros(n, max(lengths), dtype=torch.int64)
            for slot, values in enumerate(ids):
                prompts[slot, : len(values)] = torch.tensor(values)
            params = SamplingParams(
                temperature=[0.7] * n + [0.0] * (32 - n),
                top_k=[32] * n + [1] * (32 - n),
                top_p=[0.9] * n + [1.0] * (32 - n),
                seed=[seed for _, seed in pairs] + [None] * (32 - n),
            )
            g.reset()
            start = time.monotonic()
            first = g.prefill_forward(
                prompts,
                page_table=g.page_table_host[:n],
                kv_cache=g.kv_cache,
                prompt_lens=lengths,
                sampling_params=params,
            )
            result = [[int(first[slot, 0])] for slot in range(n)]
            inputs = torch.zeros(32, 1, dtype=torch.int64)
            inputs[:n] = first[:n]
            for step in range(1, 256):
                positions = torch.full((32,), -1, dtype=torch.int32)
                positions[:n] = torch.tensor(lengths, dtype=torch.int32) + step - 1
                out = g.decode_forward(
                    inputs,
                    positions,
                    page_table=g.page_table_host,
                    kv_cache=g.kv_cache,
                    sampling_params=params,
                    reload_inputs=step == 1,
                    reload_sampling_params=step == 1,
                    reset_sampling_state=step == 1,
                )
                for slot in range(n):
                    result[slot].append(int(out[slot, 0]))
            for slot, (prompt_id, seed) in enumerate(pairs):
                text = g.tokenizer.decode(result[slot], skip_special_tokens=False)
                words = text.split()
                repeated = Counter(" ".join(words[i : i + 10]) for i in range(len(words) - 9)).most_common(1)
                count = repeated[0][1] if repeated else 0
                if count >= (4 if prompt_id == 2 else 3):
                    found.add(prompt_id)
                row = dict(
                    prompt_id=prompt_id,
                    seed=seed,
                    prompt_token_ids=ids[slot],
                    token_ids=result[slot],
                    text=text,
                    max_repeated_10gram=repeated,
                    active_batch=n,
                    slot=slot,
                    seconds=time.monotonic() - start,
                    source_commit=versions["baseline_commit"],
                    source_sha256=versions["baseline_generator_sha256"],
                    layers=48,
                    context=524288,
                    max_batch_size=32,
                    page_size=64,
                    mesh=[1, 4],
                    precision=g.model.precision,
                    sampling=dict(temperature=0.7, top_k=32, top_p=0.9),
                )
                rows.append(row)
                print(prompt_id, seed, repeated, flush=True)
            save()
            if args.search_loops and found == {2, 5}:
                break
        if args.search_loops:
            assert found == {2, 5}, "No matching repetition signatures found; retain evidence and investigate further"
    finally:
        if g is not None:
            g.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
