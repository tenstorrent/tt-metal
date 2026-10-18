"""Reduced fixed-logit and interleaved-model controls for seeded sampling."""

import argparse, json
from pathlib import Path
import torch
import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt.generator import build_generator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", default="")
    parser.add_argument("--tag", default="sampling_repro01")
    args = parser.parse_args()
    torch.set_num_threads(8)
    root = Path("models/autoports/qwen_qwen3_8_flash_next")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=160000000)
    gen = None
    try:
        gen = build_generator(
            root,
            mesh,
            layer_indices=[int(v) for v in args.layers.split(",") if v],
            batch_size=3,
            max_seq_len=2048,
            checkpoint="/home/vkovacevic/qwen38-flash/hf-local/de4b8e4d43b917e7706784d8bb445c9af86a3540",
        )
        gen.prepare()
        logits = torch.full(tuple(gen.tt_logits.shape), -10.0, dtype=torch.bfloat16)
        logits[..., :32] = torch.linspace(-0.5, 0.5, 32)
        gen._copy(logits, gen.tt_logits)
        report = {}
        for name, persistent_seed in [("keep_rng", False), ("seed_inside_trace", True)]:
            if persistent_seed:
                gen._copy(torch.arange(123, 155, dtype=torch.int64), gen.seeds)
            trials = []
            for _ in range(3):
                gen.set_sampling_params(top_k=8, top_p=0.8, temperature=0.7, seed=123)
                trial = []
                for _ in range(4):
                    gen._replay("sample")
                    trial.append(gen._read_tokens())
                trials.append(trial)
            report[name] = dict(tokens=trials, repeat=trials[0] == trials[1] == trials[2])
        gen._copy(torch.full((32,), 0xFFFFFFFF, dtype=torch.int64), gen.seeds)
        trials = []
        for _ in range(3):
            gen.set_sampling_params(top_k=8, top_p=0.8, temperature=0.7, seed=123)
            trials.append(gen.generate([100, 101, 102, 103, 104, 105, 106], 4, sampling="sample"))
        report["interleaved_terminal_model"] = dict(tokens=trials, repeat=trials[0] == trials[1] == trials[2])
        gen._copy(torch.arange(123, 155, dtype=torch.int64), gen.seeds)
        trials = []
        for _ in range(3):
            gen.set_sampling_params(top_k=8, top_p=0.8, temperature=0.7, seed=123)
            trials.append(gen.generate([100, 101, 102, 103, 104, 105, 106], 4, sampling="sample"))
        report["interleaved_seed_inside_trace"] = dict(tokens=trials, repeat=trials[0] == trials[1] == trials[2])
        report["args"] = vars(args)
        (root / "doc/full_model" / f"{args.tag}.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
