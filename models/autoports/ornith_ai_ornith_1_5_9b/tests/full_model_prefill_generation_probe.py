# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Public admission atomicity and single-request generation in a fixed B3 pool.

Device execution belongs to the parent serialized lane. This probe rejects bad
requests before helper preparation and verifies peers across warmed generation.
"""


def main():
    import argparse
    import hashlib
    import json
    from dataclasses import replace
    from pathlib import Path
    from unittest.mock import patch

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import torch

    import ttnn
    from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator import build_generator
    from models.autoports.ornith_ai_ornith_1_5_9b.tt.prefill_contract import PrefillContract
    from models.autoports.ornith_ai_ornith_1_5_9b.tt.sampling_state import _raw_parameters
    from models.common.sampling.generator import SamplingParams

    torch.set_num_threads(8)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"layers": [0, 3], "batch": 3, "context": 256, "cases": []}
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024)
    gen = None
    try:
        gen = build_generator(None, mesh, layer_indices=[0, 3], max_batch_size=3, max_seq_len=256)
        params = SamplingParams(temperature=[0.5, 1.0, 2.0], top_k=20, top_p=0.95, seed=[101, 102, 103])
        tokens = torch.arange(100, 100 + 3 * 35).reshape(3, 35).long()
        lengths = [33, 34, 35]

        def record(name, passed, **details):
            report["cases"].append({"case": name, "passed": bool(passed), **details})
            path.write_text(json.dumps(report, indent=2) + "\n")
            assert passed, name

        def forbidden(*args, **kwargs):
            raise AssertionError("Rejected request reached preparation/device mutation")

        bad = [
            {"sampling_params": replace(params, temperature=[float("nan"), 1.0, 1.0])},
            {"sampling_params": replace(params, top_k=65)},
            {"sampling_params": replace(params, top_p=1.1)},
            {"empty_slots": [0, 0, 2]},
            {"empty_slots": [0, 1, 3]},
            {"page_table": gen.page_table[:, :-1]},
            {"page_table": gen.page_table + 1000},
            {"start_pos": [34, 0, 0]},
            {"enable_trace": True},
        ]
        for index, override in enumerate(bad):
            kwargs = {
                "page_table": gen.page_table,
                "kv_cache": gen.cache,
                "prompt_lens": lengths,
                "sampling_params": params,
            }
            kwargs.update(override)
            before = dict(gen.counters)
            with patch.object(PrefillContract, "prepare", side_effect=forbidden):
                try:
                    gen.prefill_forward(tokens, **kwargs)
                except (ValueError, TypeError, NotImplementedError) as error:
                    record(f"invalid_admission_{index}", gen.counters == before, error=str(error))
                else:
                    raise AssertionError("Invalid admission was accepted")
        for capacity in (1, 3, 32):
            source = replace(params, temperature=[0.5, 1.0, 2.0][:capacity], seed=[101, 102, 103][:capacity])
            raw = _raw_parameters(source, capacity)
            record(
                f"raw_sampling_physical32_logical{capacity}", raw.temperature[: min(capacity, 3)] == source.temperature
            )

        first = gen.prefill_forward(
            tokens, page_table=gen.page_table, kv_cache=gen.cache, prompt_lens=lengths, sampling_params=params
        )
        record(
            "sampled_prefill_B3_raw_temperatures",
            tuple(first.shape) == (3, 1) and gen.sampling_params.temperature[:3] == [0.5, 1.0, 2.0],
        )
        state = gen.states[(id(gen.cache), (0, 1, 2))]

        def digest(value):
            return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

        def peers():
            result = {}
            for li, layer in enumerate(gen.model.layers):
                for name in ("conv", "recurrent") if layer.kind == "linear_attention" else ("key", "value"):
                    tensor = getattr(gen.cache[li][0], name)
                    for rank, shard in enumerate(ttnn.get_device_tensors(tensor)):
                        value = ttnn.to_torch(shard)
                        selected = (
                            value[:, :, 1:3] if name == "conv" else value[1:3] if name == "recurrent" else value[8:24]
                        )
                        result[f"{li}.{name}.{rank}"] = digest(selected)
            for name in ("prompt_mask", "output_mask", "output_counts", "output_counts_gathered"):
                for rank, shard in enumerate(ttnn.get_device_tensors(getattr(gen.sampler.tt_penalties, name))):
                    result[f"{name}.{rank}"] = digest(ttnn.to_torch(shard).reshape(32, -1)[1:3])
            result["seeds"] = (
                ttnn.to_torch(ttnn.get_device_tensors(gen.sampler.tt_sampling.seeds_tt_tensor)[0])
                .reshape(-1)[1:3]
                .tolist()
            )
            result["temperatures"] = gen.sampling_params.temperature[1:3]
            return result

        prompt = tokens[0, :33].tolist()
        baseline_peers = peers()
        greedy = SamplingParams(temperature=0.0, top_k=1, top_p=1.0, seed=42)
        first_generation = gen.generate(prompt, 8, sampling_params=greedy)
        record("B3_generation_preserves_inactive_peers", peers() == baseline_peers)
        programs = mesh.num_program_cache_entries()
        counters = dict(gen.counters)
        output = gen.generate(prompt, 8, sampling_params=greedy, readiness_extra=object())
        delta = {key: value - counters[key] for key, value in gen.counters.items()}
        record(
            "B3_generation_native_full_pool_device_feedback",
            output == first_generation
            and len(output) == 8
            and peers() == baseline_peers
            and set(gen.states) == {(id(gen.cache), (0, 1, 2))}
            and state.masked
            and state.host_positions == (33, -1, -1)
            and state.position_advances == 7
            and delta["token_refreshes"] == 0
            and delta["model_replays"] == 7
            and delta["sampling_replays"] == 7
            and mesh.num_program_cache_entries() == programs,
            output=output,
            counters=delta,
            programs=programs,
        )
        report["passed"] = True
    except BaseException as error:
        report["passed"] = False
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        path.write_text(json.dumps(report, indent=2) + "\n")
        mesh.set_program_cache_misses_allowed(True)
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
