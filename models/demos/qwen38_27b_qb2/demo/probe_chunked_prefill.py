# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in full-model TP4 chunk/state diagnostic, without enabling serving chunks."""

import hashlib
import json
import os
import time
from pathlib import Path

from models.demos.qwen38_27b_qb2.tests.chunked_prefill_scenario import run_scenario


def main(args):
    import torch

    import ttnn
    from models.demos.qwen38_27b_qb2.demo.galaxy_serving import qualified_groups, verify_qualified_source
    from models.demos.qwen38_27b_qb2.tt.generator import build_generator, configure_fabric
    from models.demos.qwen38_27b_qb2.tt.generator_vllm import Qwen38ForCausalLM

    args.output.mkdir()
    report = dict(
        state="validating",
        hardware_opened=False,
        cleanup_completed=False,
        scope="Full 64-layer TP4 adapter; identical chunk boundaries, changed request scheduling and physical slots",
        host_logits=True,
        teacher_forced=True,
        sampler_qualified=False,
        plugin_scheduler_qualified=False,
        is_gpqa_score=False,
        is_performance_measurement=False,
        started_at=time.time(),
        passed=False,
        comparisons=[],
        criteria=dict(min_pcc=0.999, max_relative_rms=0.02, require_greedy_agreement=True),
    )

    def save():
        path = args.output / "progress.json.tmp"
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        path.replace(args.output / "progress.json")

    parent = generator = None
    save()
    try:
        qualification = json.loads(args.qualification.read_text())
        groups = qualified_groups(qualification)
        source = Path(__file__).resolve().parents[1]
        os.environ["QWEN_PRECISION_CONFIG"] = str(args.precision.resolve())
        verify_qualified_source(qualification, source)
        # Intermediate plugin chunks already require host logits. The diagnostic
        # keeps this backend fixed on both arms; device RNG is a separate gate.
        os.environ["QWEN_VLLM_HOST_COMPATIBILITY"] = "all"
        os.environ["QWEN_DECODE_BUCKETS"] = "1"
        os.environ["QWEN_BATCHED_PREFILL"] = "1"
        torch.set_num_threads(8)
        if ttnn.cluster.get_cluster_type() != ttnn.cluster.ClusterType.BLACKHOLE_GALAXY:
            raise ValueError("Requires the allocated Blackhole Galaxy")
        report.update(
            state="loading",
            model_source_sha256=qualification["source_sha256"],
            qualification_sha256=hashlib.sha256(args.qualification.read_bytes()).hexdigest(),
            precision=qualification["precision"],
        )
        save()
        configure_fabric(topology=ttnn.Topology.Linear)
        parent = ttnn.open_mesh_device(ttnn.MeshShape(8, 4), trace_region_size=200000000)
        report["hardware_opened"] = True
        mesh = parent.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
        if list(mesh.get_device_ids()) != [int(chip) for chip in groups[0].split(",")]:
            raise ValueError("Physical TP4 group differs from qualification")
        generator = build_generator(source, mesh, topology=ttnn.Topology.Linear)
        if len(generator.model.layers) != 64 or generator.model.precision != qualification["precision"]:
            raise ValueError("Requires all 64 layers and the qualified precision")
        cache = generator._ensure_cache(8, 256)
        adapter = Qwen38ForCausalLM(generator, 8, 256)
        adapter.cache = cache
        passages = {
            "A": "A botanist studies cedar trees and records rainfall, soil chemistry and growth each spring. ",
            "B": "A physicist measures light through a prism and compares wavelengths against a calibrated reference. ",
            "C": "A software engineer tests a queue with independent customers and verifies each customer's receipt. ",
        }
        prompts = {}
        for request, passage in passages.items():
            ids = generator.tokenizer.encode(passage, add_special_tokens=False)
            prompts[request] = torch.tensor((ids * (128 // len(ids) + 1))[:128], dtype=torch.int64)
        pages_per_request = generator.page_host.shape[1]
        page_rows = {
            request: torch.arange(index * pages_per_request, (index + 1) * pages_per_request, dtype=torch.int32)
            for index, request in enumerate(passages)
        }
        report.update(
            state="comparing",
            device_ids=list(mesh.get_device_ids()),
            loaded_at=time.time(),
            prompt_tokens={k: v.tolist() for k, v in prompts.items()},
        )
        save()

        def reset():
            generator._release_traces()
            generator.reset_recurrent_slots(list(range(8)))
            adapter._decode_bound = False
            adapter._sampling_key = None

        def prefill(rows):
            output, _ = adapter.prefill_forward(
                tokens=torch.stack([prompts[request] for request, *_ in rows]),
                page_table=torch.stack([page_rows[request] for request, *_ in rows]),
                kv_cache=cache,
                prompt_lens=[end for _, _, end, _ in rows],
                start_pos=[start for _, start, _, _ in rows],
                empty_slots=[slot for *_, slot in rows],
                sampling_params=None,
            )
            return [row.float().reshape(-1).clone() for row in output]

        def decode(rows, remap):
            tokens = torch.zeros(8, dtype=torch.int64)
            positions = torch.full((8,), -1, dtype=torch.int32)
            table = generator.page_host.clone()
            if remap is not None:
                table = table[remap].clone()
            for slot, (request, position) in enumerate(rows):
                tokens[slot], positions[slot], table[slot] = prompts[request][position], position, page_rows[request]
            output = adapter.decode_forward(
                tokens=tokens,
                start_pos=positions,
                page_table=table,
                kv_cache=cache,
                sampling_params=None,
                reset_batch=True,
                slot_remap=remap,
            )
            return [row.float().reshape(-1).clone() for row in output[: len(rows)]]

        reference, changed = run_scenario(reset, prefill, decode)
        for key, expected in reference.items():
            actual = changed[key]
            if (
                actual.shape != expected.shape
                or actual.numel() != generator.model.config.vocab_size
                or not torch.isfinite(actual).all()
                or not torch.isfinite(expected).all()
            ):
                raise ValueError("Full-vocabulary logits must have matching finite values")
            a, b = actual.double(), expected.double()
            centered_a, centered_b = a - a.mean(), b - b.mean()
            pcc = float(torch.dot(centered_a, centered_b) / (centered_a.norm() * centered_b.norm()).clamp_min(1e-30))
            relative_rms = float((a - b).norm() / b.norm().clamp_min(1e-30))
            greedy_agreement = int(a.argmax()) == int(b.argmax())
            report["comparisons"].append(
                dict(
                    request=key[0],
                    position=key[1],
                    pcc=pcc,
                    relative_rms=relative_rms,
                    reference_top1=int(b.argmax()),
                    actual_top1=int(a.argmax()),
                    passed=pcc >= 0.999 and relative_rms <= 0.02 and greedy_agreement,
                )
            )
            save()
        report.update(
            state="completed", passed=all(row["passed"] for row in report["comparisons"]), finished_at=time.time()
        )
    except BaseException as error:
        report.update(state="failed", error=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        try:
            try:
                if generator is not None:
                    generator.close()
            finally:
                if parent is not None:
                    ttnn.close_mesh_device(parent)
            report["cleanup_completed"] = True
        finally:
            save()
