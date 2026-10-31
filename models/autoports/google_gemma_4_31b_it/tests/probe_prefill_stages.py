"""Read-only tensor observations localize reduced repeated-prefill variation."""

import json
from pathlib import Path
from unittest.mock import patch

import torch

import ttnn

from ..tt.generator import Gemma4Generator


def main():
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024)
    report, baseline = [], {}
    try:
        gen = Gemma4Generator(mesh, layer_indices=(0, 5), batch_size=32, max_seq_len=4096)
        gen.prepare_decode()
        prompts = [[2, 125, 89] * ((31 + slot * 17) // 3) + [2] * ((31 + slot * 17) % 3) for slot in range(32)]
        prompts[-1] = None
        originals = dict(
            embed=gen.model.embed,
            layer0=gen.model.layers[0].prefill_chunk_forward,
            layer5=gen.model.layers[1].prefill_chunk_forward,
            selection=gen.prefill.last_row,
            terminal=gen.model.terminal,
        )
        for repeat in range(2):
            counters, seen = {}, {}

            def observe(name):
                def call(*args, **kwargs):
                    output = originals[name](*args, **kwargs)
                    slot = counters.get(name, 0)
                    counters[name] = slot + 1
                    value = ttnn.to_torch(ttnn.get_device_tensors(output)[0]).float()
                    key = (name, slot)
                    if name == "selection":
                        expected = seen[("layer5", slot)][:, :, len(prompts[slot]) - 1 : len(prompts[slot]), :]
                        assert torch.equal(value, expected), ("selection mismatch", repeat, slot)
                    seen[key] = value
                    if repeat:
                        diff = (value - baseline[key]).abs()
                        report.append(
                            dict(
                                stage=name,
                                slot=slot,
                                logical_length=len(prompts[slot]),
                                equal=torch.equal(value, baseline[key]),
                                max_abs_diff=float(diff.max()),
                                changed_elements=int((diff != 0).sum()),
                            )
                        )
                    return output

                return call

            with (
                patch.object(gen.model, "embed", side_effect=observe("embed")),
                patch.object(gen.model.layers[0], "prefill_chunk_forward", side_effect=observe("layer0")),
                patch.object(gen.model.layers[1], "prefill_chunk_forward", side_effect=observe("layer5")),
                patch.object(gen.prefill, "last_row", side_effect=observe("selection")),
                patch.object(gen.model, "terminal", side_effect=observe("terminal")),
                patch.object(ttnn, "execute_trace", return_value=None),
            ):
                gen.generate_batch(prompts, 1, ignore_eos=True)
            if repeat == 0:
                baseline = seen
        Path("models/autoports/google_gemma_4_31b_it/doc/full_model/prefill_stages_probe.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(json.dumps([r for r in report if not r["equal"]]), flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
