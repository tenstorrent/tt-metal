# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""All40 fixed-slot AIME teacher forcing, with independent physical KV pages.

After activating python_env and sourcing lane-env.sh, run serialized:
 timeout 1200 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_full_model_batched_teacher

Three copies of the pinned 161-token prompt each produce 100 scored predictions:
one prefill prediction and 99 decode predictions. Decode consumes HF reference
tokens, reads logits for accuracy only, and does not execute a host sampler.
This is batch accuracy evidence, not qualitative or token-out performance evidence.
"""

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen3.6-35B-A3B"
REVISION = "345d467b117d1fd5eda9b879a5988de9ae5f47a7"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_fixture(path):
    """CPU-only exact cache provenance validation before importing TTNN."""
    from readiness_check.reference_cache import ReferenceCache
    from readiness_check.schema import load_reference

    summary_path = ROOT / "doc/full_model/hf_reference/reference_summary.json"
    summary = json.loads(summary_path.read_text())
    key = summary["cache_key"]
    assert summary["model_id"] == MODEL and summary["revision"] == REVISION
    assert key["hf_model_id"] == MODEL and key["revision"] == "hub:" + REVISION
    assert key["settings"] == {
        "chat_template": True,
        "device": "cpu",
        "gen_len": 100,
        "num_entries": 1,
        "prompt_len": None,
        "prompt_source": "aime24",
        "top_k": 100,
    }, key["settings"]
    cache = ReferenceCache.for_model_dir(ROOT)
    assert cache.directory is not None
    stem = cache.directory / (key["kind"] + "-" + ReferenceCache.digest(key))
    sidecar, cached = Path(str(stem) + ".key.json"), Path(str(stem) + ".refpt")
    assert json.loads(sidecar.read_text()) == key, "Reference metadata is not an exact cache match"
    assert sha256(path) == sha256(cached), "Reference bytes differ from the pinned cache artifact"
    reference = load_reference(path)
    assert reference.hf_model_id == MODEL and reference.k == 100 and len(reference.entries) == 1
    entry = reference.entries[0]
    assert tuple(entry.prompt_tokens.shape) == (1, 161)
    assert tuple(entry.generated_tokens.shape) == (1, 100)
    assert tuple(entry.topk_tokens.shape) == (100, 100)
    assert entry.tf_prompt_len == 161
    return entry, {
        "path": str(path),
        "sha256": sha256(path),
        "cache_key": key,
        "cache_sidecar": str(sidecar),
        "cached_reference": str(cached),
        "summary_sha256": sha256(summary_path),
        "prompt_tokens": entry.prompt_tokens[0].tolist(),
        "generated_tokens": entry.generated_tokens[0].tolist(),
        # Packaged generation obtains the continuation with cached HF decode,
        # then computes top-K with a separate full-sequence HF forward. Those
        # two numerical paths need not have identical top-1 at every position.
        "hf_cached_generation_vs_full_forward": score(entry.generated_tokens[0].tolist(), entry.topk_tokens),
    }


def score(predictions, reference_topk):
    import torch

    predicted = torch.tensor(predictions, dtype=torch.int64)
    assert predicted.numel() == reference_topk.shape[0] > 0
    hits = {str(k): (reference_topk[:, :k].long() == predicted[:, None]).any(-1) for k in (1, 5, 100)}
    return {
        "total": len(predictions),
        **{f"top{k}": int(hit.sum()) / len(predictions) for k, hit in hits.items()},
        **{f"top{k}_matches": int(hit.sum()) for k, hit in hits.items()},
        "top5_miss_indices": (~hits["5"]).nonzero().flatten().tolist(),
        "top100_miss_indices": (~hits["100"]).nonzero().flatten().tolist(),
        "pass": int(hits["5"].sum()) * 100 >= 98 * len(predictions) and bool(hits["100"].all()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--reference", type=Path, default=ROOT / "readiness_aime24_chat.refpt")
    parser.add_argument("--output", type=Path, default=ROOT / "doc/full_model/batched_teacher/result.json")
    args = parser.parse_args()
    assert args.context >= 1024 and args.context % 32 == 0
    entry, provenance = load_fixture(args.reference)
    assert os.environ.get("TT_METAL_DEVICE_PROFILER", "0") != "1"
    os.environ.setdefault("TT_METAL_TRACE_ALLOC_TRACKING", "1")
    os.environ.setdefault("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0")
    assert os.environ["TT_METAL_TRACE_ALLOC_TRACKING"] == "1"
    assert os.environ["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] == "0"

    import torch

    import ttnn
    from models.autoports.qwen_qwen3_6_35b_a3b.tt.generator import QwenGenerator
    from models.autoports.qwen_qwen3_6_35b_a3b.tt.model import HF_MODEL, HF_REVISION, QwenModel

    assert (HF_MODEL, HF_REVISION) == (MODEL, REVISION)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "running",
        "command": sys.argv,
        "time_unix": time.time(),
        "model": MODEL,
        "revision": REVISION,
        "layers": list(range(40)),
        "context": args.context,
        "max_batch_size": 3,
        "slots": [0, 1, 2],
        "scope": "Full-stack batch teacher-forcing accuracy; no qualitative or token-out performance claim",
        "reference": provenance,
        "host_sampling": False,
        "diagnostic_logit_readback": True,
        "feedback": "HF generated_tokens[i-1] at logical position 161+i-1; TT argmax is scored only",
        "coverage": {
            "prefill_predictions_per_slot": 1,
            "decode_predictions_per_slot": 99,
            "total_predictions_per_slot": 100,
            "final_next_position": 260,
        },
        "thresholds": {"top5": 0.98, "top100": 1.0, "scope": "prefill, decode, and combined, each slot independently"},
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in [Path(__file__)] + sorted((ROOT / "tt").glob("*.py"))
        },
        "predictions_by_slot": [[], [], []],
    }

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    def observe(logits):
        rows = logits.reshape(3, -1, 248320)
        assert rows.shape[1] == 1 and torch.isfinite(rows).all(), "Invalid diagnostic logits"
        predicted = rows[:, 0].argmax(-1).tolist()
        for slot, token in enumerate(predicted):
            result["predictions_by_slot"][slot].append(token)

    save()
    mesh = generator = None
    torch.set_num_threads(8)
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200000000)
        model = QwenModel(mesh, max_seq_len=args.context)
        assert tuple(model.layer_indices) == tuple(range(40))
        generator = QwenGenerator(model, max_batch_size=3, host_sampling=False)
        cache = generator.cache
        full_states = next(states for states in cache if "page_table" in states[0])
        width = int(full_states[0]["page_table"].shape[-1])
        pages_per_slot = (161 + 99 + 31) // 32
        assert pages_per_slot == 9 and 3 * pages_per_slot <= width
        pages = torch.stack([torch.arange(width).roll(-slot * pages_per_slot) for slot in range(3)]).int()
        live_pages = pages[:, :pages_per_slot].flatten().tolist()
        assert len(set(live_pages)) == 27
        result["physical_pages_by_slot"] = pages[:, :pages_per_slot].tolist()
        result["aggregate_page_pool_size"] = width
        # Compile both host-logit readback shapes before trace allocation locks.
        generator.prefill_forward(torch.ones((1, 33), dtype=torch.int32), prompt_lens=[33])
        model.host_logits(generator.sampler_logits)
        for position in generator.positions:
            ttnn.to_torch(ttnn.get_device_tensors(position)[0])
        generator.prepare()
        generator.reset()
        assert generator.cache is cache and not generator.host_sampling
        assert "model_fixed" in generator.traces
        baseline_programs = int(mesh.num_program_cache_entries())
        baseline = dict(generator.counters)
        observe(
            generator.prefill_forward(
                entry.prompt_tokens.repeat(3, 1).int(),
                prompt_lens=[161] * 3,
                start_pos=[0] * 3,
                page_table=pages,
                kv_cache=cache,
                slots=(0, 1, 2),
            )
        )
        decode_baseline = dict(generator.counters)
        for index in range(1, 100):
            forced = torch.full((3, 1), int(entry.generated_tokens[0, index - 1]), dtype=torch.int32)
            observe(
                generator.decode_forward(
                    forced,
                    [161 + index - 1] * 3,
                    slots=(0, 1, 2),
                    kv_cache=cache,
                    refresh_inputs=True,
                    read_tokens=False,
                    return_logits=True,
                )
            )
            if index % 10 == 0 or index == 99:
                print("BATCH_TEACHER_PROGRESS", index + 1, "/100", flush=True)
                save()
        ttnn.synchronize_device(mesh)
        result["decode_counter_delta"] = {k: generator.counters[k] - v for k, v in decode_baseline.items()}
        result["counter_delta"] = {k: generator.counters[k] - v for k, v in baseline.items()}
        result["program_cache_entries"] = {
            "after_prepare": baseline_programs,
            "after_checks": int(mesh.num_program_cache_entries()),
        }
        result["rows"] = [
            {
                "slot": slot,
                "prefill": score(tokens[:1], entry.topk_tokens[:1]),
                "decode": score(tokens[1:], entry.topk_tokens[1:]),
                "combined": score(tokens, entry.topk_tokens),
            }
            for slot, tokens in enumerate(result["predictions_by_slot"])
        ]
        result["final_positions"] = [
            int(ttnn.to_torch(ttnn.get_device_tensors(p)[0]).flatten()[0]) for p in generator.positions
        ]
        assert result["final_positions"] == [260] * 3
        assert result["counter_delta"]["captures"] == 0
        assert result["decode_counter_delta"]["model_replays"] == 99
        assert result["decode_counter_delta"]["sampling_replays"] == 0
        assert result["decode_counter_delta"]["readbacks"] == 99
        assert result["decode_counter_delta"]["page_table_refreshes"] == 0
        assert int(mesh.num_program_cache_entries()) == baseline_programs, "Unwarmed programs after prepare"
        assert all(row[part]["pass"] for row in result["rows"] for part in ("prefill", "decode", "combined")), result[
            "rows"
        ]
        result["status"] = "pass"
    except BaseException:
        result["status"] = "fail"
        result["error"] = traceback.format_exc()
        raise
    finally:
        result["elapsed_seconds"] = time.time() - result["time_unix"]
        save()
        if generator is not None and not generator.closed:
            generator.close()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        print("BATCH_TEACHER_RESULT", result["status"], str(args.output), flush=True)


if __name__ == "__main__":
    main()
