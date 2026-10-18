# SPDX-License-Identifier: Apache-2.0
"""Standalone TP4 producer/loader compatibility after the pinned TopK rebuild.

Root runs with exclusive reservation ownership. No cache index or blob is
written. Fresh dumps are retained only in the new isolated output directory.
No timing/roofline or full-regeneration claim; exactly12 tensors/48 shards.
"""

import argparse
import json
import os
import time
from pathlib import Path

from models.autoports.qwen_qwen3_8_flash_next.tools import benchmark_migrate_topk_native_cache as guard


def run(args):
    # Device imports occur only after root explicitly invokes this probe.
    import torch

    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tt.checkpoint import CheckpointState
    from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import MeshOps

    started = time.perf_counter()
    root = args.cache_root.resolve()
    output = args.output.resolve()
    guard.require(output != root and root not in output.parents, "Probe output cannot be inside expert cache")
    output.mkdir(parents=True, exist_ok=False)
    supplied = guard.read_pinned(args.attestation, args.attestation_sha256)
    guard.require(supplied == guard.build_attestation(), "Current native build does not match pinned attestation")
    verifier, cache_module, index_path, index = guard.load_baseline(root)
    identity = index["identity"]
    guard.require(identity["native_sha256"] == supplied["native_before"], "Probe old binary/cache identity mismatch")
    for path, expected in identity["transform_sha256"].items():
        guard.require(guard.sha(path) == expected, "Expert transform changed")
    checkpoint = Path(identity["source_seal"]["fingerprints"]["checkpoint"])
    guard.require(
        cache_module.seal_sources(checkpoint, root, "read") == identity["source_seal"], "Checkpoint seal changed"
    )
    report = dict(
        status="incomplete",
        old_identity=guard.OLD_IDENTITY,
        old_index_sha256=guard.OLD_INDEX_SHA256,
        build_attestation_sha256=args.attestation_sha256,
        native_sha256=guard.NATIVE_NEW,
        probe_source_sha256=guard.sha(__file__),
        sample_keys=guard.SAMPLE_KEYS,
        scope="Layers0/47, local experts0/127, gate/up/down, all4 shards:12 distributed tensors/48 shards. Not full regeneration.",
        identity=dict(identity, native_sha256=guard.NATIVE_NEW),
        cache_written_bytes=0,
        comparisons=[],
        pid=os.getpid(),
        source_commit=guard.git("rev-parse", "HEAD").decode().strip(),
    )
    summary = output / "summary.json"

    def save():
        summary.write_text(json.dumps(report, indent=2) + "\n")

    save()
    mesh = None
    try:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
        mesh.enable_program_cache()
        ops = MeshOps(mesh)
        for layer in (0, 47):
            state = CheckpointState(checkpoint, f"model.language_model.layers.{layer}.")
            weights = {
                "gate_up": state["mlp.experts.gate_up_proj"],
                "down": state["mlp.experts.down_proj"],
            }
            for expert in (0, 127):
                for role in ("gate", "up", "down"):
                    key = f"layer{layer}/{expert}/{role}"
                    entry = index["entries"][key]
                    verifier.validate_entry(key, entry)
                    old_path = root / "blobs" / f"{entry['sha256']}.tensorbin"
                    old_state = verifier.file_state(old_path)
                    guard.require(guard.sha(old_path) == entry["sha256"], "Old sample payload changed")
                    # Exact existing ExpertWeightCache.load_layer producer chain.
                    if role == "down":
                        shards = [weights["down"][rank * 128 + expert].T for rank in range(4)]
                    else:
                        region = slice(0, 640) if role == "gate" else slice(640, None)
                        shards = [weights["gate_up"][rank * 128 + expert, region].T for rank in range(4)]
                    fresh = ops.shards(shards, ttnn.bfloat4_b)
                    metadata = cache_module.ExpertWeightCache.metadata(fresh, ops)
                    guard.require(metadata == entry["metadata"], "Fresh packed tensor metadata differs")
                    fresh_path = output / f"layer{layer}_expert{expert}_{role}.tensorbin"
                    ttnn.dump_tensor(fresh_path, fresh)
                    fresh_hash = guard.sha(fresh_path)
                    guard.require(fresh_hash == entry["sha256"], "Fresh packed serialized payload differs")
                    loaded = ttnn.load_tensor(old_path, device=mesh)
                    guard.require(
                        cache_module.ExpertWeightCache.metadata(loaded, ops) == metadata,
                        "Loaded tensor metadata differs",
                    )
                    equal = []
                    for actual, reference in zip(ttnn.get_device_tensors(fresh), ttnn.get_device_tensors(loaded)):
                        actual_host = ttnn.to_torch(actual)
                        reference_host = ttnn.to_torch(reference)
                        equal.append(bool(torch.equal(actual_host, reference_host)))
                    guard.require(equal == [True] * 4, "Fresh/loaded logical shard mismatch")
                    guard.require(verifier.file_state(old_path) == old_state, "Old blob changed during probe")
                    report["comparisons"].append(
                        dict(
                            key=key,
                            old_sha256=entry["sha256"],
                            fresh_sha256=fresh_hash,
                            fresh_path=str(fresh_path),
                            metadata=metadata,
                            logical_equal_per_shard=equal,
                        )
                    )
                    save()
                    del fresh, loaded, shards, actual, reference, actual_host, reference_host
            del weights, state
        guard.require(guard.sha(index_path) == guard.OLD_INDEX_SHA256, "Old index changed during probe")
        guard.require(guard.build_attestation() == supplied, "Native build changed during probe")
        guard.require(
            cache_module.seal_sources(checkpoint, root, "read") == identity["source_seal"], "Checkpoint seal changed"
        )
        for path, expected in identity["transform_sha256"].items():
            guard.require(guard.sha(path) == expected, "Expert transform changed during probe")
        guard.require(
            guard.sha(guard.MODEL / "doc/datatype_sweep/selected_precision_config.json")
            == identity["precision_sha256"],
            "Precision policy changed",
        )
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["failure"] = repr(error)
        raise
    finally:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        report["seconds"] = time.perf_counter() - started
        save()
    guard.validate_compatibility(report, args.attestation_sha256, index["entries"], identity)
    print(
        json.dumps(
            dict(status=report["status"], summary=str(summary), sha256=guard.sha(summary), tensors=12, shards=48)
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("--attestation-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
