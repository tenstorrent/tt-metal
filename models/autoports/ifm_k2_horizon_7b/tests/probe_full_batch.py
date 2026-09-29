"""Mixed prompt/cache slots through public low-level model/generator APIs."""

import argparse
import json
from pathlib import Path

import torch

import ttnn

from ..tt.generator import K2Generator


def read(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).clone()


def run(gen, output):
    records = []
    for batch in [2, 32]:
        cache, table = gen.model.allocate_cache(batch_size=batch, capacity=128)
        torch.manual_seed(37)
        table = torch.randperm(table.numel()).reshape_as(table).int()
        lengths = [31 + 2 * (i % 2) for i in range(batch)]
        tokens = torch.arange(batch * 33).reshape(batch, 33) % 1000 + 500
        prefill_tokens = gen.prefill_forward(tokens, page_table=table, kv_cache=cache, prompt_lens=lengths)
        assert prefill_tokens.shape == (batch,)
        assert ((prefill_tokens >= 0) & (prefill_tokens < gen.model.vocab_size)).all()
        inputs = (torch.arange(batch) + 600).reshape(batch, 1)
        positions = torch.tensor(lengths)
        # A fixed inactive row shares the padded batch shape but must never
        # update any cache page or advance its negative position sentinel.
        if batch == 32:
            positions[-1] = -1
        inactive_before = read(cache[0][0])[table[-1].long()].clone()
        logits = gen.decode_forward(inputs, positions, page_table=table, kv_cache=cache, sampling_mode="host")
        assert logits.shape == (batch, gen.model.vocab_size)
        assert torch.isfinite(logits[:-1] if batch == 32 else logits).all()
        actual_pos = read(gen.state["positions"])
        expected_pos = torch.where(positions >= 0, positions + 1, positions)
        assert torch.equal(actual_pos[:batch], expected_pos)
        if batch == 32:
            assert torch.equal(read(cache[0][0])[table[-1].long()], inactive_before)
        # Token-out replay at fixed slots must consume the sampler's own
        # previous output, while position state advances independently.
        gen.decode_forward(inputs, positions, page_table=table, kv_cache=cache)
        sampled = read(gen.state["tokens"]).clone()
        counters = gen.counters.copy()
        gen.replay()
        feedback_positions = read(gen.state["positions"])[:batch]
        assert torch.equal(feedback_positions, torch.where(positions >= 0, positions + 2, positions))
        assert gen.counters["token_refreshes"] == counters["token_refreshes"]
        assert gen.counters["position_refreshes"] == counters["position_refreshes"]
        before_external = read(cache[0][0])
        gen.reset()
        assert torch.equal(read(cache[0][0]), before_external), "reset must not zero external caches"
        # Re-run two slots as batch1 with exactly their original logical page
        # map, prompt, token and position. This catches row/physical-page mixups.
        controls = []
        for i in range(2):
            single = gen.decode_forward(
                inputs[i : i + 1],
                positions[i : i + 1],
                page_table=table[i : i + 1],
                kv_cache=cache,
                sampling_mode="host",
            )[0]
            a, b = single.float(), logits[i].float()
            corr = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
            controls.append({"slot": i, "pcc": corr, "batch_token": int(b.argmax()), "single_token": int(a.argmax())})
            assert corr >= 0.995
        record = {
            "layers": gen.model.num_layers,
            "head_fidelity": gen.model.head_fidelity,
            "batch": batch,
            "prompt_lens": lengths,
            "positions": positions.tolist(),
            "token_out_format": list(sampled.shape),
            "fixed_slot_feedback": True,
            "single_slot_controls": controls,
            "inactive_cache_unchanged": batch == 32,
            "external_cache_survives_reset": True,
        }
        records.append(record)
        print(json.dumps(record), flush=True)
        Path(output).write_text(json.dumps(records, indent=2) + "\n")
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--output", default="models/autoports/ifm_k2_horizon_7b/doc/full_model/batch_probe.json")
    args = p.parse_args()
    torch.set_num_threads(16)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200_000_000)
    gen = None
    try:
        gen = K2Generator(mesh, override_num_layers=args.layers)
        run(gen, args.output)
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
