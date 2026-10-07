# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full-weight aggregated prefill/decode on one TP4 replica of a Galaxy.

This checks repeatability and writes raw output/performance. It does not replace
reference evaluations or the eight-concurrent-replica qualification gate.
"""

import json
import os
import time
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

import ttnn
from models.demos.qwen38_27b_qb2.demo.galaxy_serving import model_source_hashes
from models.demos.qwen38_27b_qb2.tests.galaxy_prompt import qualification_prompt
from models.demos.qwen38_27b_qb2.tt.generator import build_generator, configure_fabric
from models.demos.qwen38_27b_qb2.tt.model import checkpoint_path


@pytest.mark.skipif(os.getenv("QWEN_GALAXY_SMOKE") != "1", reason="explicit allocated-Galaxy hardware test")
def test_full_model_galaxy_replica():
    tokens = qualification_prompt(AutoTokenizer.from_pretrained(checkpoint_path(), local_files_only=True))
    torch.set_num_threads(8)
    assert ttnn.cluster.get_cluster_type() == ttnn.cluster.ClusterType.BLACKHOLE_GALAXY
    output = Path(os.environ["QWEN_GALAXY_RECEIPT"])
    configure_fabric(topology=ttnn.Topology.Linear)
    parent = ttnn.open_mesh_device(ttnn.MeshShape(8, 4), trace_region_size=200000000)
    gen = None
    try:
        mesh = parent.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
        started = time.perf_counter()
        gen = build_generator(Path(__file__).resolve().parents[1], mesh, topology=ttnn.Topology.Linear)
        setup_s = time.perf_counter() - started
        print(f"GALAXY_SETUP_COMPLETE seconds={setup_s:.3f}", flush=True)
        print("GALAXY_FIRST_GENERATION_BEGIN", flush=True)
        first = gen.generate(tokens, 128)
        cold_perf = dict(gen.last_perf)
        print(f"GALAXY_FIRST_GENERATION_COMPLETE {cold_perf}", flush=True)
        second = gen.generate(tokens, 128)
        print(f"GALAXY_WARM_GENERATION_COMPLETE {gen.last_perf}", flush=True)
        assert first == second, "Repeated greedy generation diverged"
        assert first and all(0 <= token < gen.model.config.vocab_size for token in first)
        output.write_text(
            json.dumps(
                {
                    "state": "completed",
                    "passed": True,
                    "precision": gen.model.precision,
                    "source_sha256": model_source_hashes(Path(__file__).resolve().parents[1]),
                    "parent_mesh": [8, 4],
                    "replica_mesh": [1, 4],
                    "replicas_executed": 1,
                    "topology": "linear",
                    "layers": len(gen.model.layers),
                    "setup_s": setup_s,
                    "prompt_tokens": tokens,
                    "output_tokens": first,
                    "text": gen.tokenizer.decode(first),
                    "repeat_equal": True,
                    "first_run_perf": cold_perf,
                    "second_run_perf": gen.last_perf,
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(parent)
