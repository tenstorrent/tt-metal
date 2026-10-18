"""Discriminate temperature from inverse temperature on canonical traced sampling."""
import json
import math
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt.generator import build_generator

ROOT = Path("models/autoports/qwen_qwen3_8_flash_next")


def main():
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=160000000)
    gen = None
    try:
        gen = build_generator(
            ROOT,
            mesh,
            layer_indices=[],
            batch_size=32,
            max_seq_len=2048,
            checkpoint="/home/vkovacevic/qwen38-flash/hf-local/de4b8e4d43b917e7706784d8bb445c9af86a3540",
        )
        gen.prepare()
        logits = torch.full(tuple(gen.tt_logits.shape), -100.0, dtype=torch.bfloat16)
        logits[..., 0], logits[..., 1] = 1.0, 0.0
        gen._copy(logits, gen.tt_logits)
        rows = []
        for temperature in (0.5, 2.0):
            gen.set_sampling_params(top_k=8, top_p=1.0, temperature=temperature, seed=123)
            encoded = ttnn.to_torch(ttnn.get_device_tensors(gen.sample_params["sample"][2])[0]).flatten().tolist()
            assert encoded == [1 / temperature] * 32, encoded
            tokens = []
            for position in range(128):
                gen._positions([position] * 32)
                gen._replay("sample")
                tokens.extend(gen._read_tokens())
            local = [token % (gen.model.cfg.vocab_size // 4) for token in tokens]
            assert set(local) <= {0, 1}
            frequency = local.count(0) / len(local)
            expected = 1 / (1 + math.exp(-1 / temperature))
            assert abs(frequency - expected) < 0.04, (temperature, frequency, expected)
            rows.append(
                dict(
                    temperature=temperature,
                    encoded_inverse=encoded[0],
                    samples=len(local),
                    observed_top_probability=frequency,
                    expected_top_probability=expected,
                )
            )
        report = dict(
            status="pass",
            scope="canonical full-model split sampling; synthetic nonuniform logits",
            tracking=True,
            rows=rows,
            counters=dict(gen.counters),
        )
        (ROOT / "doc/vllm_integration/temperature_probe.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
