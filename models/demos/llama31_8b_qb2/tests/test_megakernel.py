# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real-checkpoint, paged-cache and trace checks for opt-in decode fusion.

Run device operations serially. QB2_MEGAKERNEL_ARTIFACT_DIR optionally saves
machine-readable results. Timed trace replay includes enqueue/synchronization;
device-profiler measurements must be collected in a separate run.
"""

import json
import os
from pathlib import Path
import time

import pytest
import torch
from transformers import AutoConfig

import ttnn
from models.common.modules.tt_ccl import TT_CCL
from models.demos.llama31_8b_qb2.tt.decoder import LlamaDecoder
from models.demos.llama31_8b_qb2.tt.megakernel.decoder import experimental_layers
from models.demos.llama31_8b_qb2.tt.model import Checkpoint, checkpoint_path
from models.demos.llama31_8b_qb2.tt.precision import load_precision_config
from models.demos.llama31_8b_qb2.tests.test_decoder import to_device, to_host


def compare(actual, expected):
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    a, b = actual.float().flatten(), expected.float().flatten()
    error = a - b
    pcc = torch.corrcoef(torch.stack((a, b)))[0, 1].item()
    relative_l2 = (torch.linalg.vector_norm(error) / torch.linalg.vector_norm(b).clamp_min(1e-12)).item()
    assert pcc >= 0.9999, f"Fused versus traced baseline PCC={pcc}"
    assert relative_l2 < 0.01, f"Fused versus traced baseline relative L2={relative_l2}"
    return {
        "pcc": pcc,
        "relative_l2": relative_l2,
        "max_abs": error.abs().max().item(),
        "exact": torch.equal(actual, expected),
    }


def copy_to(value, target, mesh):
    host = ttnn.from_torch(
        value.contiguous(),
        dtype=target.dtype,
        layout=target.layout,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    ttnn.copy_host_to_device_tensor(host, target)


@pytest.mark.parametrize(
    "mode,reuse_scratch",
    [
        ("swiglu", False),
        ("mlp", False),
        ("mlp", True),
        ("mlp_reduce", False),
        ("mlp_tail", False),
        ("norm_mlp_tail", False),
        ("gather_norm_mlp_tail", False),
        ("post_attention", False),
        ("attention_tail", False),
        ("decoder", False),
    ],
    ids=[
        "swiglu",
        "mlp",
        "mlp_shared",
        "mlp_reduce",
        "mlp_tail",
        "norm_mlp_tail",
        "gather_norm_mlp_tail",
        "post_attention",
        "attention_tail",
        "decoder",
    ],
)
def test_fused_layer_real_weights(qb2_mesh, mode, reuse_scratch):
    torch.set_num_threads(8)
    mesh = qb2_mesh
    folder = checkpoint_path()
    checkpoint = Checkpoint(folder)
    config = AutoConfig.from_pretrained(folder, local_files_only=True)
    weights = checkpoint.load([n for n in checkpoint.index if n.startswith("model.layers.0.")])
    baseline = LlamaDecoder.from_state_dict(
        weights,
        hf_config=config,
        layer_idx=0,
        mesh_device=mesh,
        precision_policy=load_precision_config(),
        ccl=TT_CCL(mesh),
    )
    baseline.prepare_decode(1)
    gu_workers = int(os.environ.get("QB2_GU_WORKERS", "8"))
    prototype = experimental_layers([baseline], mode=mode, reuse_scratch=reuse_scratch, gu_workers=gu_workers)[0]
    embedding = checkpoint.load(["model.embed_tokens.weight"])["model.embed_tokens.weight"]
    ids = torch.randint(0, config.vocab_size, (1, 259), generator=torch.Generator().manual_seed(35))
    hidden = embedding[ids]
    results = {
        "mode": mode,
        "reuse_scratch": reuse_scratch,
        "gu_workers": gu_workers,
        "batch": 1,
        "checkpoint": str(folder),
        "precision": baseline.precision_policy,
        "checks": [],
    }
    # Both sides own independent cache arenas, including unused sentinel pages.
    caches = [baseline.allocate_cache(num_physical_pages=9) for _ in range(2)]
    for prefix in (127, 255):
        for remap in (False, True):
            mapping = torch.tensor([[1, 3, 5, 7]], dtype=torch.int32)
            table = to_device(mapping, mesh)
            prompt = to_device(hidden[:, None, :prefix], mesh, shard=True)
            for cache in caches:
                baseline.prefill_forward(prompt, page_table=table, kv_cache=cache)
            position = to_device(torch.tensor([prefix], dtype=torch.int32), mesh)
            x = to_device(hidden[:, None, prefix : prefix + 1], mesh, shard=True)
            functions = [
                lambda layer=layer, cache=cache: layer.decode_forward(
                    x, current_pos=position, page_table=table, kv_cache=cache
                )
                for layer, cache in zip((baseline, prototype), caches)
            ]
            # Compile both implementations before trace capture owns allocator addresses.
            for call in functions:
                call()
            ttnn.synchronize_device(mesh)
            traces = []
            outputs = []
            try:
                for call in functions:
                    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                    output = call()
                    ttnn.end_trace_capture(mesh, trace, cq_id=0)
                    traces.append(trace)
                    outputs.append(output)
                for offset in range(3):
                    if remap and offset == 1:
                        # Migrate physical pages and mutate the SAME page-table
                        # buffer after capture; both traces must observe it.
                        next_mapping = mapping.flip(1).contiguous()
                        for cache in caches:
                            for tensor in cache:
                                original = to_host(tensor)
                                migrated = original.clone()
                                for old, new in zip(mapping.flatten(), next_mapping.flatten()):
                                    migrated[int(new)] = original[int(old)]
                                host = ttnn.from_torch(
                                    migrated,
                                    dtype=tensor.dtype,
                                    layout=tensor.layout,
                                    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
                                )
                                ttnn.copy_host_to_device_tensor(host, tensor)
                        copy_to(next_mapping, table, mesh)
                        mapping = next_mapping
                    pos = prefix + offset
                    # Shard host hidden state exactly like the original device input.
                    hx = ttnn.from_torch(
                        hidden[:, None, pos : pos + 1].contiguous(),
                        dtype=x.dtype,
                        layout=x.layout,
                        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
                    )
                    ttnn.copy_host_to_device_tensor(hx, x)
                    copy_to(torch.tensor([pos], dtype=torch.int32), position, mesh)
                    for trace in traces:
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    actual_output, expected_output = to_host(outputs[1]), to_host(outputs[0])
                    try:
                        metrics = compare(actual_output, expected_output)
                        for actual, expected in zip(caches[1], caches[0]):
                            torch.testing.assert_close(to_host(actual), to_host(expected), rtol=0, atol=0)
                    except AssertionError:
                        if destination := os.environ.get("QB2_MEGAKERNEL_ARTIFACT_DIR"):
                            folder = Path(destination)
                            folder.mkdir(parents=True, exist_ok=True)
                            torch.save(
                                {
                                    "actual": actual_output,
                                    "expected": expected_output,
                                    "position": pos,
                                    "mapping": mapping,
                                    "cache": [[to_host(t) for t in pair] for pair in caches],
                                },
                                folder / f"layer-{mode}-{pos}-{remap}-failure.pt",
                            )
                        raise
                    saved = to_host(outputs[1]).clone()
                    ttnn.execute_trace(mesh, traces[1], cq_id=0, blocking=True)
                    torch.testing.assert_close(to_host(outputs[1]), saved, rtol=0, atol=0)
                    results["checks"].append({"position": pos, "remapped": remap, **metrics})
                for name, trace in zip(("baseline", "prototype"), traces):
                    for _ in range(5):
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    samples = []
                    for _ in range(5):
                        start = time.perf_counter_ns()
                        for _ in range(20):
                            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                        ttnn.synchronize_device(mesh)
                        samples.append((time.perf_counter_ns() - start) / 20 / 1e3)
                    results.setdefault("host_trace_replay_us", []).append(
                        {"implementation": name, "position": pos, "remapped": remap, "samples": samples}
                    )
            finally:
                for trace in traces:
                    ttnn.release_trace(mesh, trace)
    print(json.dumps(results, indent=2), flush=True)
    if destination := os.environ.get("QB2_MEGAKERNEL_ARTIFACT_DIR"):
        path = Path(destination)
        path.mkdir(parents=True, exist_ok=True)
        (path / f"layer-{mode}{'_shared' if reuse_scratch else ''}.json").write_text(json.dumps(results, indent=2))
