# SPDX-License-Identifier: Apache-2.0
"""Isolated real-weight B32 LM-head layout experiment; never a model benchmark.

Run only after the caller has obtained exclusive device ownership. This probe
uses synchronization for isolated trace timings, not the live phase collector.
"""

import argparse
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.full_model_lm_head import host, measure
from models.autoports.qwen_qwen3_8_flash_next.tests.metrics import pcc_metrics
from models.autoports.qwen_qwen3_8_flash_next.tt.model import QwenFlashNextModel
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import _hidden_parts


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geometry(tensor):
    logical = list(tensor.shape)
    padded = list(tensor.padded_shape)
    return {
        "logical": logical,
        "padded": padded,
        "logical_elements": math.prod(logical),
        "padded_elements": math.prod(padded),
        "physical_m_rows": math.prod(padded[:-1]),
        "dtype": str(tensor.dtype),
        "layout": str(tensor.layout),
    }


def flatten_decode_input(mixed):
    assert tuple(mixed.shape) == (1, 32, 1, 2560)
    # ROW_MAJOR conversion removes each batch lane's 31 padded sequence rows.
    # A tile-only view is not sufficient proof of logical row compaction.
    compact = ttnn.to_layout(mixed, ttnn.ROW_MAJOR_LAYOUT)
    compact = ttnn.reshape(compact, (1, 1, 32, 2560))
    return ttnn.to_layout(compact, ttnn.TILE_LAYOUT)


def compare(reference, actual):
    reference = reference.reshape(32, -1)
    actual = actual.reshape(32, -1)
    per_row = [pcc_metrics(a, b) for a, b in zip(reference, actual)]
    ref_top5 = reference.topk(5, dim=-1).indices
    got_top5 = actual.topk(5, dim=-1).indices
    overlap = (ref_top5.unsqueeze(-1) == got_top5.unsqueeze(-2)).any(-1).sum(-1)
    return {
        "aggregate": pcc_metrics(reference, actual),
        "per_row": per_row,
        "minimum_row_pcc": min(row["pcc"] for row in per_row),
        "argmax_equal_rows": int((reference.argmax(-1) == actual.argmax(-1)).sum()),
        "reference_argmax": reference.argmax(-1).tolist(),
        "candidate_argmax": actual.argmax(-1).tolist(),
        "top5_overlap_count_per_row": overlap.tolist(),
        "all_rows": 32,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--recorded",
        default="/home/vkovacevic/qwen38-flash/tt-metal-cache/qwen_qwen3_8_flash_next/"
        "multichip/recorded_inputs/layer3.pt",
    )
    parser.add_argument("--offsets", type=int, nargs="+", default=[0, 32])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if any(offset < 0 for offset in args.offsets):
        parser.error("offsets must be nonnegative")
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fixture = torch.load(args.recorded, weights_only=True)["inputs"][0]
    assert fixture.ndim == 3 and fixture.shape[0] == 1 and fixture.shape[-1] == 10240
    assert all(offset + 32 <= fixture.shape[1] for offset in args.offsets)
    torch.set_num_threads(8)
    policy_path = Path(__file__).resolve().parents[1] / "doc/datatype_sweep/selected_precision_config.json"
    result = {
        "status": "started",
        "args": vars(args),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "probe_sha256": sha256(__file__),
        "model_source_sha256": sha256(Path(__file__).resolve().parents[1] / "tt/model.py"),
        "fixture_sha256": sha256(args.recorded),
        "precision_sha256": sha256(policy_path),
        "precision": json.loads(policy_path.read_text()),
        "scope": "Real selected-policy head weights and HF-layer3 input rows; no transformer layers executed",
        "timing": {
            "warmup_calls": 2,
            "trace_replays_per_sample": 32,
            "samples": 3,
            "boundary": "host perf_counter through synchronize_device after 32 asynchronous trace replays",
            "includes": "input compaction, projection and public decode shape restoration",
            "excludes": "terminal mixer, transformer layers, sampler, server and request scheduling",
        },
        "cases": [],
    }

    def save():
        dest.write_text(json.dumps(result, indent=2) + "\n")

    save()
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=160000000)
    try:
        result["device_ids"] = list(mesh.get_device_ids())
        model = QwenFlashNextModel(args.checkpoint, mesh, batch_size=32, layer_indices=[])
        assert model.precision_policy == result["precision"]
        assert model.precision_policy["config_id"] == "combined_head4_gdn4_lofi"
        result["weight_geometry_per_chip"] = geometry(model.lm_head)
        for offset in args.offsets:
            residual_host = fixture[:, offset : offset + 32].reshape(1, 32, 1, 10240).contiguous()
            assert torch.unique(residual_host.reshape(32, -1), dim=0).shape[0] == 32
            residual = model.ops.shards(_hidden_parts(residual_host))
            mixed = model.terminal(residual)
            compact = flatten_decode_input(mixed)
            # Host verification occurs outside the timing region and proves lane
            # identity independently of a potentially insensitive logits metric.
            assert torch.equal(host(mixed).reshape(32, -1), host(compact).reshape(32, -1))
            row = {"offset": offset, "native_input": geometry(mixed), "compact_input": geometry(compact)}
            result["cases"].append(row)
            save()
            del compact

            def project(value):
                return ttnn.linear(
                    value,
                    model.lm_head,
                    compute_kernel_config=model.head_compute,
                    dtype=model.logits_dtype,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )

            def native():
                logits = project(mixed)
                return ttnn.reshape(logits, (1, 1, 32, model.cfg.vocab_size // 4))

            def candidate():
                # This already has the public model.forward(decode=True) shape.
                logits = project(flatten_decode_input(mixed))
                return ttnn.reshape(logits, (1, 1, 32, model.cfg.vocab_size // 4))

            reference, native_ms = measure(mesh, native)
            actual, candidate_ms = measure(mesh, candidate)
            row.update(
                native_ms=native_ms,
                candidate_ms=candidate_ms,
                native_median_ms=statistics.median(native_ms),
                candidate_median_ms=statistics.median(candidate_ms),
                measured_speedup=statistics.median(native_ms) / statistics.median(candidate_ms),
                metrics=compare(reference, actual),
            )
            raw_path = dest.with_name(dest.stem + f"_offset{offset}.pt")
            torch.save({"native": reference, "candidate": actual, "residual": residual_host}, raw_path)
            row["raw_outputs"] = {"path": str(raw_path), "sha256": sha256(raw_path)}
            save()
            print("BATCH_HEAD_PROBE", json.dumps(row), flush=True)
            del mixed, residual, reference, actual
        result["status"] = "measured"
    except Exception as error:
        result.update(status="error", error=repr(error))
        raise
    finally:
        save()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
