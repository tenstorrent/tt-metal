"""Reduced diagnostic: isolate prefill from sampling and decode state changes."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch

import ttnn

from ..tt.generator import Gemma4Generator
from ..tt.model import Gemma4Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gather-barrier", action="store_true")
    args = parser.parse_args()
    if args.gather_barrier:

        def gather(self, x, dim):
            return ttnn.experimental.all_gather_async(
                x,
                dim=dim,
                multi_device_global_semaphore=self.ccl.get_and_cycle_ag_semaphore_handles(),
                barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(),
                num_links=2,
                topology=ttnn.Topology.Ring,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        Gemma4Model.gather = gather
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024)
    records = []
    try:
        gen = Gemma4Generator(mesh, layer_indices=(0, 5), batch_size=32, max_seq_len=4096)
        gen.prepare_decode()
        prompts = [[2, 125, 89] * ((31 + slot * 17) // 3) + [2] * ((31 + slot * 17) % 3) for slot in range(32)]
        prompts[-1] = None
        execute = ttnn.execute_trace
        previous = None
        for phase in ("prefill_only", "prefill_sample", "prefill_decode_sample", "prefill_only_again"):
            for repeat in range(3):
                captured = []

                def observed_execute(mesh_arg, tid, **kwargs):
                    if tid == gen.sample_trace and not captured:
                        captured.append(gen._host_logits(gen.logits).reshape(32, -1))
                    if phase.startswith("prefill_only"):
                        assert tid == gen.sample_trace
                        return
                    return execute(mesh_arg, tid, **kwargs)

                with patch.object(ttnn, "execute_trace", side_effect=observed_execute):
                    gen.generate_batch(prompts, 4 if phase == "prefill_decode_sample" else 1, ignore_eos=True)
                logits = captured[0]
                diff = None if previous is None else (logits - previous).abs()
                record = dict(
                    phase=phase,
                    repeat=repeat,
                    equal_previous=previous is not None and torch.equal(previous, logits),
                    max_abs_diff=None if diff is None else float(diff.max()),
                    mean_abs_diff=None if diff is None else float(diff.mean()),
                    changed_elements_per_slot=None if diff is None else (diff != 0).sum(-1).tolist(),
                    slot30_max_count=int((logits[30] == logits[30].max()).sum()),
                )
                records.append(record)
                print(record, flush=True)
                previous = logits
        Path(
            "models/autoports/google_gemma_4_31b_it/doc/full_model/prefill_repeat_probe"
            + ("_barrier" if args.gather_barrier else "")
            + ".json"
        ).write_text(json.dumps(records, indent=2) + "\n")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
