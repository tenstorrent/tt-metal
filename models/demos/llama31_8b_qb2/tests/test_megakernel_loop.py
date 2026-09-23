# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real layer0/31 address-selection, page migration and replay checks for the loop."""

import json
import os
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig
import ttnn
from models.common.modules.tt_ccl import TT_CCL
from models.demos.llama31_8b_qb2.tt.decoder import LlamaDecoder
from models.demos.llama31_8b_qb2.tt.model import Checkpoint, checkpoint_path
from models.demos.llama31_8b_qb2.tt.precision import load_precision_config
from models.demos.llama31_8b_qb2.tt.megakernel.decoder import experimental_layers
from models.demos.llama31_8b_qb2.tt.megakernel.loop import DecoderLoop
from .test_decoder import to_device, to_host
from .test_megakernel import compare, copy_to


@pytest.mark.parametrize("count", [1, 2])
def test_device_layer_loop_real_weights(qb2_mesh, count, tuning=None):
    torch.set_num_threads(8)
    mesh = qb2_mesh
    checkpoint = Checkpoint(checkpoint_path())
    config = AutoConfig.from_pretrained(checkpoint_path(), local_files_only=True)
    layers = []
    workspace = None
    for index in (0, 31)[:count]:
        weights = checkpoint.load([name for name in checkpoint.index if name.startswith(f"model.layers.{index}.")])
        layer = LlamaDecoder.from_state_dict(
            weights,
            hf_config=config,
            layer_idx=index,
            mesh_device=mesh,
            precision_policy=load_precision_config(),
            ccl=TT_CCL(mesh),
        )
        workspace = layer.prepare_decode(1, workspace=workspace)
        layers.append(layer)
    body = experimental_layers(layers, mode="decoder", tuning=tuning)[0].fused_body
    baseline_cache = [layer.allocate_cache(num_physical_pages=9) for layer in layers]
    loop_cache = [layer.allocate_cache(num_physical_pages=9) for layer in layers]
    loop = DecoderLoop(body, loop_cache)
    embedding = checkpoint.load(["model.embed_tokens.weight"])["model.embed_tokens.weight"]
    tokens = torch.randint(0, config.vocab_size, (1, 131), generator=torch.Generator().manual_seed(53))
    hidden = embedding[tokens]
    mapping = torch.tensor([[1, 3, 5, 7]], dtype=torch.int32)
    table = to_device(mapping, mesh)
    for caches in (baseline_cache, loop_cache):
        prompt = to_device(hidden[:, None, :127], mesh, shard=True)
        for layer, cache in zip(layers, caches):
            prompt = layer.prefill_forward(prompt, page_table=table, kv_cache=cache)
    position = to_device(torch.tensor([127], dtype=torch.int32), mesh)
    x = to_device(hidden[:, None, 127:128], mesh, shard=True)

    def baseline():
        value = x
        for layer, cache in zip(layers, baseline_cache):
            value = layer.decode_forward(value, current_pos=position, page_table=table, kv_cache=cache)
        return value

    # The real generator compiles with position -1 after prefill. This must
    # preserve every existing physical cache page, including unused sentinels.
    folder = Path(os.environ.get("QB2_MEGAKERNEL_ARTIFACT_DIR", "/tmp/qb2-megakernel-loop"))
    folder.mkdir(parents=True, exist_ok=True)
    saved_cache = [[to_host(t).clone() for t in pair] for pair in loop_cache]
    rotary_zero = to_device(torch.tensor([0], dtype=torch.int32), mesh)
    copy_to(torch.tensor([-1], dtype=torch.int32), position, mesh)
    for _ in range(3):
        loop(x, position, table, rotary_zero)
    ttnn.synchronize_device(mesh)
    try:
        for pair, saved in zip(loop_cache, saved_cache):
            for tensor, original in zip(pair, saved):
                torch.testing.assert_close(to_host(tensor), original, rtol=0, atol=0)
    except AssertionError:
        torch.save(
            {"before": saved_cache, "after": [[to_host(t) for t in pair] for pair in loop_cache]},
            folder / f"loop-{count}-inactive-failure.pt",
        )
        raise
    copy_to(torch.tensor([127], dtype=torch.int32), position, mesh)

    calls = (baseline, lambda: loop(x, position, table))
    for call in calls:
        call()
    ttnn.synchronize_device(mesh)
    traces, outputs, results = [], [], []
    try:
        for call in calls:
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            traces.append(trace)
            try:
                output = call()
            finally:
                # A host compile/allocation error must not leave capture open
                # and hang mesh teardown. Outer cleanup releases this handle.
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
            outputs.append(output)
        for pos in (127, 128, 129):
            if pos == 128:
                next_mapping = mapping.flip(1).contiguous()
                for caches in (baseline_cache, loop_cache):
                    for pair in caches:
                        for tensor in pair:
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
            host = ttnn.from_torch(
                hidden[:, None, pos : pos + 1],
                dtype=x.dtype,
                layout=x.layout,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
            )
            ttnn.copy_host_to_device_tensor(host, x)
            copy_to(torch.tensor([pos], dtype=torch.int32), position, mesh)
            loop.reserve_invocations()
            for trace in traces:
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            expected, actual = [to_host(output) for output in outputs]
            try:
                record = {"position": pos, "output": compare(actual, expected), "cache": []}
                for native, fused in zip(baseline_cache, loop_cache):
                    record["cache"].append([compare(to_host(b), to_host(a)) for a, b in zip(native, fused)])
                loop.reserve_invocations(8)
                for _ in range(8):
                    ttnn.execute_trace(mesh, traces[1], cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                record["replay"] = compare(to_host(outputs[1]), actual)
                for native, fused in zip(baseline_cache, loop_cache):
                    for a, b in zip(native, fused):
                        assert torch.equal(to_host(a), to_host(b)), "Repeated loop replay changed the cache"
                results.append(record)
            except Exception:
                torch.save(
                    {
                        "expected": expected,
                        "actual": actual,
                        "position": pos,
                        "baseline_cache": [[to_host(t) for t in pair] for pair in baseline_cache],
                        "loop_cache": [[to_host(t) for t in pair] for pair in loop_cache],
                    },
                    folder / f"loop-{count}-failure.pt",
                )
                raise
    finally:
        for trace in traces:
            ttnn.release_trace(mesh, trace)
        (folder / f"loop-{count}.json").write_text(
            json.dumps({"layers": [0, 31][:count], "inactive_warmup_kv_exact": True, "checks": results}, indent=2)
        )
    print(json.dumps(results, indent=2), flush=True)
