# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reduced runtime tripwires for token-out host-logit/argmax/H2D fallback.

Run under the parent's serialized device lease. This is a numerical contract
probe, not qualitative text evaluation or a performance measurement.
"""


def main():
    import argparse
    import hashlib
    import json
    import math
    from pathlib import Path
    from unittest.mock import patch

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    report = {
        "layers": [0, 3],
        "context": 128,
        "generated_tokens": 8,
        "source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("tt/generator.py", "tt/model.py", "tt/prefill_runtime.py", "tt/prefill_ops.py")
        },
        "scope": "Reduced real-model host-boundary tripwires; no full-model throughput/quality claim",
    }
    import torch

    import ttnn
    from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator import build_generator
    from models.common.sampling.generator import SamplingParams

    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024)
    gen = None
    try:
        gen = build_generator(None, mesh, layer_indices=[0, 3], max_batch_size=1, max_seq_len=128)
        prompt = list(range(100, 133))
        greedy = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)
        gen.generate(prompt, 8, sampling_params=greedy)  # Compile/capture before tripwires.
        original_execute, original_read = gen._execute, ttnn.to_torch
        original_upload, original_copy = ttnn.from_torch, ttnn.copy_host_to_device_tensor
        inside_execute = False
        checked_replays, token_reads = 0, 0

        def forbidden(*_args, **_kwargs):
            raise AssertionError("Token-out path crossed a forbidden host logits/argmax boundary")

        def execute(*args, **kwargs):
            nonlocal inside_execute, checked_replays
            inside_execute = True
            try:
                result = original_execute(*args, **kwargs)
                checked_replays += 1
                return result
            finally:
                inside_execute = False

        def read(tensor, *args, **kwargs):
            nonlocal token_reads
            if math.prod(tensor.shape) > 32:
                raise AssertionError("Token-out path read back more than the physical32 token buffer")
            token_reads += 1
            return original_read(tensor, *args, **kwargs)

        def upload(*args, **kwargs):
            if inside_execute and kwargs.get("device") is not None:
                raise AssertionError("H2D tensor construction during decode replay")
            return original_upload(*args, **kwargs)

        def copy(*args, **kwargs):
            if inside_execute:
                raise AssertionError("H2D copy during decode replay")
            return original_copy(*args, **kwargs)

        with (
            patch.object(gen, "_execute", execute),
            patch.object(gen.model, "host_logits", forbidden),
            patch.object(torch, "argmax", forbidden),
            patch.object(torch.Tensor, "argmax", forbidden),
            patch.object(ttnn, "to_torch", read),
            patch.object(ttnn, "from_torch", upload),
            patch.object(ttnn, "copy_host_to_device_tensor", copy),
        ):
            device = gen.generate(prompt, 8, sampling_params=greedy)
        device_metrics = dict(gen.last_generation_metrics)
        host = gen.generate(prompt, 8, sampling_params=greedy, host_sampling=True)
        host_metrics = dict(gen.last_generation_metrics)
        assert checked_replays == 7 and token_reads == 8
        assert device_metrics["counter_delta"]["token_refreshes"] == 0
        assert device_metrics["counter_delta"]["sampling_replays"] == 7
        assert host_metrics["counter_delta"]["sampling_replays"] == 0
        assert device == host, "Device greedy differs from explicit host-argmax compatibility"
        report.update(
            passed=True,
            device_tokens=device,
            host_compatibility_tokens=host,
            checked_decode_replays=checked_replays,
            allowed_physical_token_reads=token_reads,
            device_counter_delta=device_metrics["counter_delta"],
            host_counter_delta=host_metrics["counter_delta"],
            token_out_host_logits_and_argmax_forbidden=True,
            per_replay_h2d_forbidden=True,
        )
    except BaseException as error:
        report.update(passed=False, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        path.write_text(json.dumps(report, indent=2) + "\n")
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
