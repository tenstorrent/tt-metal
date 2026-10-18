"""Reduced full-shape reproduction of the repeated fixed-token trace control."""

import argparse, json, time
from pathlib import Path
import torch, ttnn
from readiness_check.schema import load_reference
from models.autoports.qwen_qwen3_8_flash_next.tt.generator import build_generator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cycles", type=int, default=30)
    ap.add_argument("--tag", default="replay_repro")
    a = ap.parse_args()
    torch.set_num_threads(8)
    ttnn.CONFIG.throw_exception_on_fallback = True
    root = Path("models/autoports/qwen_qwen3_8_flash_next")
    output = root / "doc/optimized_full_model" / f"{a.tag}.json"
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=160000000)
    gen = None
    report = dict(args=vars(a), cycles=[])
    try:
        gen = build_generator(root, mesh, checkpoint=a.checkpoint, layer_indices=[0, 1, 3])
        gen.prepare()
        ids = dict(gen.traces)
        entry = load_reference(root / "readiness_aime24_chat.refpt").entries[0]
        prompt = entry.prompt_tokens[0].tolist()
        gen.prefill_logits(prompt + entry.generated_tokens[0].tolist())
        gen.generate(prompt, 100)
        for _ in range(4):
            gen.generate(prompt[:128], 128)
        for i in range(a.cycles):
            gen.generate(prompt[:128], 1)
            gen._copy(torch.full((1, 1, 1, 1), 393, dtype=torch.int32), gen.tt_token)
            ttnn.synchronize_device(mesh)
            tick = time.perf_counter()
            print("FIXED_REPLAY_START", i, flush=True)
            for _ in range(127):
                gen._replay("decode")
            ttnn.synchronize_device(mesh)
            seconds = time.perf_counter() - tick
            position = ttnn.to_torch(ttnn.get_device_tensors(gen.tt_pos)[0])[:, 0].tolist()
            assert position == [255] and gen.traces == ids
            report["cycles"].append(dict(cycle=i, seconds=seconds, position=position))
            output.write_text(json.dumps(report, indent=2) + "\n")
            print("FIXED_REPLAY_PASS", i, seconds, flush=True)
        report.update(status="pass", trace_ids={k: str(v) for k, v in ids.items()}, counters=dict(gen.counters))
        output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
