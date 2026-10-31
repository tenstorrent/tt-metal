"""Discriminate reduced greedy repeat failures at the logits/sampling boundary."""

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
    rows, previous = [], None
    try:
        gen = Gemma4Generator(mesh, layer_indices=(0, 5), batch_size=32, max_seq_len=4096)
        gen.prepare_decode()
        prompts = [[2, 125, 89] * ((31 + slot * 17) // 3) + [2] * ((31 + slot * 17) % 3) for slot in range(32)]
        prompts[-1] = None
        original = gen.prefill_forward
        for repeat in range(5):
            captured = {}

            def prefill(*args, **kwargs):
                result = original(*args, **kwargs)
                captured["logits"] = gen._host_logits(gen.logits).reshape(32, -1)
                captured["tokens"] = result
                return result

            with patch.object(gen, "prefill_forward", side_effect=prefill):
                output = gen.generate_batch(prompts, 4, ignore_eos=True)
            logits = captured["logits"]
            values, indices = logits[30].topk(8)
            rows.append(
                dict(
                    repeat=repeat,
                    outputs=output,
                    first_tokens=captured["tokens"],
                    logits_equal_previous=None if previous is None else torch.equal(previous, logits),
                    differing_elements=None if previous is None else int((previous != logits).sum()),
                    slot30_max=float(logits[30].max()),
                    slot30_max_count=int((logits[30] == logits[30].max()).sum()),
                    slot30_selected_value=float(logits[30, captured["tokens"][30]]),
                    slot30_top8=list(zip(indices.tolist(), values.tolist())),
                )
            )
            previous = logits
            print(rows[-1], flush=True)
        # Freeze a real prefill-logit tensor and replay only its existing sampler.
        gen.reset()
        gen._reserve_requests([len(p) if p else 0 for p in prompts], 4)
        width = max(len(p) if p else 0 for p in prompts)
        tokens = torch.zeros(32, width, dtype=torch.int32)
        for slot, prompt in enumerate(prompts):
            if prompt:
                tokens[slot, : len(prompt)] = torch.tensor(prompt)
        gen.prefill_forward(
            tokens, page_table=gen.page_table, kv_cache=gen.kv_cache, prompt_lens=[len(p) if p else 0 for p in prompts]
        )
        fixed_logits = gen._host_logits(gen.logits).reshape(32, -1)
        sampled = []
        for _ in range(20):
            ttnn.execute_trace(mesh, gen.sample_trace, cq_id=0, blocking=False)
            result = gen._read_tokens()
            sampled.append(result)
            assert all(fixed_logits[i, result[i]] == fixed_logits[i].max() for i in range(31))
        report = dict(
            scope="reduced layers0/5", repeats=rows, fixed_logits_sampled=sampled, fixed_logits_semantically_greedy=True
        )
        Path("models/autoports/google_gemma_4_31b_it/doc/full_model/repeat_greedy_probe.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
