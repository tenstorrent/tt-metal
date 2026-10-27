# SPDX-License-Identifier: Apache-2.0
"""Real-weight layer0 grouped/where control versus masked direct state commit."""
import argparse
import copy
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoConfig

import ttnn
from models.autoports.qwen_qwen3_8_27b.tt import multichip_decoder as current
from models.autoports.qwen_qwen3_8_27b.tt.model import Checkpoint
from models.autoports.qwen_qwen3_8_27b.tt.precision import decoder_policy, load_precision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/data/mvasiljev/weights/Qwen3.8-27B")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--masked", action="store_true")
    parser.add_argument("--all-inactive", action="store_true")
    parser.add_argument(
        "--varying-mask", action="store_true", help="Start from nonzero state and exercise activity transitions"
    )
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(982)
    cfg = AutoConfig.from_pretrained(args.checkpoint).text_config
    checkpoint = Checkpoint(args.checkpoint)
    prefix = "model.language_model.layers.0."
    weights = checkpoint.read([k for k in checkpoint.index if k.startswith(prefix)])
    os.environ["QWEN38_GROUPED_RECURRENCE"] = "1"
    os.environ["QWEN38_MASKED_RECURRENCE"] = "0"
    delta_source = None
    result = dict(
        scope="one real-weight linear-attention decoder, TP4; not full-model/SWE performance",
        checkpoint=args.checkpoint,
        batch=args.batch,
        masked=args.masked,
        all_inactive=args.all_inactive,
        varying_mask=args.varying_mask,
        generated_delta_source=delta_source,
        allocation_tracking=os.getenv("TT_METAL_TRACE_ALLOC_TRACKING"),
        program_cache_skip=os.getenv("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"),
        status="running",
    )
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=64 * 1024**2)
    traces = []
    try:
        precision = load_precision()
        model = current.MultichipDecoder.from_state_dict(
            weights, hf_config=cfg, layer_idx=0, mesh_device=mesh, policy=decoder_policy(precision, 0)
        )
        alternate = copy.copy(model)
        alternate.policy = copy.deepcopy(model.policy)
        alternate.policy["masked_recurrence"] = True
        if hasattr(model, "_batched_decode_impl"):
            alternate._batched_decode_impl = copy.copy(model._batched_decode_impl)
            alternate._batched_decode_impl.policy = copy.deepcopy(model._batched_decode_impl.policy)
            alternate._batched_decode_impl.policy["masked_recurrence"] = True
        models = [model, alternate]
        b = args.batch

        def put(x, shard=False):
            return ttnn.from_torch(
                x,
                device=mesh,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=(ttnn.ShardTensorToMesh(mesh, dim=2) if shard else ttnn.ReplicateTensorToMesh(mesh)),
            )

        hidden = put(torch.randn(b, 1, 5120) * 0.012563, shard=True)
        outputs = [put(torch.zeros(b, 1, 5120), shard=True) for _ in models]
        initial = model.allocate_state(batch_size=b, context_length=256)
        if args.varying_mask:
            for field, scale in [("conv", 0.02), ("recurrent", 0.01)]:
                tensor = getattr(initial, field)
                value = torch.randn(tuple(tensor.shape)) * scale
                host = ttnn.from_torch(value, dtype=tensor.dtype, layout=tensor.layout)
                ttnn.copy_host_to_device_tensor(host, tensor)
        states = [model.snapshot(initial) for _ in models]
        schedules = (
            [
                torch.ones(b),
                (torch.arange(b) % 2 == 0).float(),
                torch.zeros(b),
                (torch.arange(b) % 2 != 0).float(),
                torch.ones(b),
            ]
            if args.varying_mask
            else [torch.zeros(b) if args.all_inactive else (torch.arange(b) % 3 != 0).float()] * 3
        )
        active = schedules[0]
        mask = put(active.reshape(b, 1, 1)) if args.masked or args.all_inactive or args.varying_mask else None
        result["semantic_masks"] = [value.tolist() for value in schedules] if mask is not None else None

        def run(index):
            y = models[index].decode_forward(hidden, state=states[index], active_mask=mask)
            ttnn.copy(y, outputs[index])

        def reset():
            for state in states:
                ttnn.copy(initial.conv, state.conv)
                ttnn.copy(initial.recurrent, state.recurrent)

        for index in range(2):
            for _ in range(2):
                run(index)
        reset()
        ttnn.synchronize_device(mesh)
        for index in range(2):
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            traces.append(trace)
            try:
                run(index)
            finally:
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
        reset()
        checks = []
        before = {}

        def compare(step):
            for name, tensors in [
                ("output", outputs),
                ("conv", [s.conv for s in states]),
                ("recurrent", [s.recurrent for s in states]),
            ]:
                for device, pair in enumerate(zip(*(ttnn.get_device_tensors(t) for t in tensors))):
                    left, right = [ttnn.to_torch(t) for t in pair]
                    assert torch.equal(left, right), (step, name, device, (left - right).abs().max().item())
                    preserved = None
                    nonzero = None
                    if name != "output" and mask is not None and (active == 0).any():
                        preserved = torch.equal(left[active == 0], before[name][device][active == 0])
                        assert preserved, (name, device, "inactive state changed")
                        nonzero = int(torch.count_nonzero(before[name][device][active == 0]))
                        if args.varying_mask:
                            assert nonzero > 0, (name, device, "inactive preservation fixture is zero")
                    checks.append(
                        dict(
                            step=step,
                            tensor=name,
                            device=device,
                            exact=True,
                            inactive_preserved=preserved,
                            inactive_nonzero_elements=nonzero,
                        )
                    )

        for step, active in enumerate(schedules):
            if mask is not None:
                host = ttnn.from_torch(active.reshape(b, 1, 1), dtype=mask.dtype, layout=mask.layout)
                ttnn.copy_host_to_device_tensor(host, mask)
                before = {
                    field: [ttnn.to_torch(t) for t in ttnn.get_device_tensors(getattr(states[0], field))]
                    for field in ("conv", "recurrent")
                }
            for trace in traces:
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            compare(step + 1)
        timings = {}
        for name, trace in zip(["baseline", "masked_commit"], traces):
            windows = []
            for _ in range(5):
                start = time.perf_counter()
                for _ in range(args.iterations):
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                windows.append((time.perf_counter() - start) * 1e6 / args.iterations)
            timings[name] = dict(replay_us=windows, median_replay_us=statistics.median(windows))
        compare("after_timing")
        result.update(
            status="pass",
            checks=checks,
            timing=timings,
            precision=precision,
            speedup=timings["baseline"]["median_replay_us"] / timings["masked_commit"]["median_replay_us"],
        )
    except BaseException as exc:
        result.update(status="fail", error=repr(exc))
        raise
    finally:
        root = Path(__file__).resolve().parents[1]
        sources = [
            Path(__file__),
            root / "tt/grouped_recurrence.py",
            root / "tt/precision.py",
            root / "tt/multichip_decoder.py",
            *root.glob("tt/kernels/*grouped_recurrence.cpp"),
        ]
        result["source_sha256"] = {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        for trace in traces:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)
    print(
        json.dumps({k: v for k, v in result.items() if k not in ("checks", "source_sha256", "generated_delta_source")}),
        flush=True,
    )


if __name__ == "__main__":
    main()
