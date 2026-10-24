# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device-profiler run of the generic prefill chunk at chosen shapes.

Under ``python -m tracy -r -p -v -m pytest ...`` with ``TT_METAL_DEVICE_PROFILER=1``: the layers
``V41_PROF_LAYERS`` (default 0,1,2,3,20,21,24: the sharing schedule plus an Engram layer) in a generic state
of
``V41_PROF_MAX_SEQ`` tokens, one eager chunk per shape of ``V41_PROF_SHAPES`` (``start:length`` pairs, comma
separated; default ``0:5120,0:1028,0:2052,0:4100,<last chunk of the 1M bucket>:5120``), each run twice (the
first
compiles the profiler kernels) and the second between tracy signposts named ``shape <start>:<length>``, the
device
profiler read after every block (no marker overflow). Synthetic tokens; the state's contents do not matter to
a
chunk's timing. Reports each shape's wall time as ``V41_PROF_RESULT``.

``V41_PROF_COMPARE=<variable>`` (e.g. ``TT_V41_PREFILL_MASK_ONCE``; no profiler needed) runs each shape with
the
variable unset and then set to ``1`` on the same state (built with it set; its index-K caches filled with
seeded
random keys, so the selections are not tie-breaks over zero scores) and compares the logits and every block's
output and selection bit
for bit (``compare`` in the result; the test fails on a difference).
"""

import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import oracle as orc
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.tests.v41.galaxy_meshes import galaxy_meshes
from models.demos.deepseek_v3_d_p.tests.v41.test_v41_depth import parse_layers
from models.demos.deepseek_v3_d_p.tests.v41.weight_cache import weight_cache_dir
from models.demos.deepseek_v3_d_p.tt.v41.engram_checkpoint import build_engram_modules
from models.demos.deepseek_v3_d_p.tt.v41.transformer import TtV41Transformer, _HostChunk, _Request
from models.demos.deepseek_v3_d_p.tt.v41.weights import load_layer, load_layer_dense, resolve_checkpoint

try:
    from tracy import signpost
except ModuleNotFoundError:  # not under the tracy wrapper

    def signpost(_name: str) -> None:
        pass


DEFAULT_LAYERS = "0,1,2,3,20,21,24"
TRACE_REGION = 256 * 1024 * 1024


@pytest.mark.timeout(int(os.environ.get("V41_PROF_TIMEOUT", 2 * 3600)))
@pytest.mark.parametrize("mesh_device, device_params", galaxy_meshes(trace_region_size=TRACE_REGION), indirect=True)
def test_v41_prefill_profile(mesh_device, device_params):
    layers = parse_layers(os.environ.get("V41_PROF_LAYERS", DEFAULT_LAYERS))
    chunk = int(os.environ.get("TT_V41_CHUNK", 5120))
    max_seq = int(os.environ.get("V41_PROF_MAX_SEQ", 1054720))  # the served 1,053,696 rounded up to the chunk
    last_1m = (1048576 - chunk) // chunk * chunk  # the last full chunk whose visible count is in the 1M bucket
    default_shapes = f"0:{chunk},0:1028,0:2052,0:4100,{last_1m}:{chunk}"
    shapes = [tuple(int(v) for v in s.split(":")) for s in os.environ.get("V41_PROF_SHAPES", default_shapes).split(",")]
    profile = os.environ.get("TT_METAL_DEVICE_PROFILER", "0") == "1"
    compare = os.environ.get("V41_PROF_COMPARE", "")
    if compare:
        os.environ[compare] = "1"  # the state is built with the variable set (its optional tensors exist)
    ckpt = resolve_checkpoint()
    if ckpt is None:
        pytest.skip("V4.1 checkpoint shards not downloaded")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(orc.HF_SNAPSHOT)
    spec = orc.real_spec(tuple(layers), max_seq, checkpoint=ckpt.root)
    root = weight_cache_dir(spec, mesh_device.shape)
    t0 = time.time()
    engram, engram_hash = build_engram_modules(mesh_device, C, ckpt, layers, tokenizer)
    top = ckpt.read(["embed.weight", "norm.weight", "head.weight"])
    model = TtV41Transformer(
        mesh_device,
        C,
        layers,
        lambda layer, include_moe: (load_layer if include_moe else load_layer_dense)(ckpt, layer),
        top["embed.weight"],
        top["norm.weight"],
        top["head.weight"],
        max_seq_len=max_seq,
        chunk=chunk,
        engram=engram,
        engram_hash=engram_hash,
        weight_cache_path=root,
    )
    state = model.new_state(generic=True)
    report = {
        "layers": layers,
        "chunk": chunk,
        "max_seq": max_seq,
        "shapes": shapes,
        "build_s": round(time.time() - t0, 1),
    }
    logger.info(f"prefill profile: built {len(layers)} layers in {report['build_s']} s")
    tokens = torch.randint(1, C.VOCAB_SIZE, (chunk,), dtype=torch.int64)
    buffers: dict = {}

    def drain(layer, x, start, length):
        if profile:
            ttnn.ReadDeviceProfiler(mesh_device)

    def one_chunk(start: int, length: int, on_block, outputs: dict | None = None) -> float:
        model.reset_state(state)
        state.start = start
        request = _Request(tokens[:length], length - 1, None, None)
        host = next(model._host_chunks(request))
        host = _HostChunk(start, host.length, host.token_ids, host.engram, host.types)
        token_ids, engram_inputs = model._stage(host, buffers)
        state.tables.prepare_chunk(start, length)
        ttnn.synchronize_device(mesh_device)
        t0 = time.time()
        scored = model._chunk_forward(state, host, token_ids, engram_inputs, None, None, start + length - 1, on_block)
        ttnn.synchronize_device(mesh_device)
        wall = time.time() - t0
        if outputs is not None:
            outputs["logits"] = torch.cat([_down(mesh_device, logits) for logits, *_ in scored])
        return wall

    if compare:
        # seeded random index keys in every source's cache (the chunk's own rows are overwritten by the chunk)
        g = torch.Generator().manual_seed(1)
        for src, t in state.index_k.items():
            keys = torch.randn(tuple(t.shape), generator=g).to(torch.bfloat16)
            targets = [(keys, t)]
            twin = state.index_k_tiled.get(src)
            if twin is not None:  # the tiled twin: the same keys, its zero tile past them
                targets.append((torch.cat([keys, torch.zeros(1, 1, 32, keys.shape[3], dtype=keys.dtype)], dim=2), twin))
            for host, dst in targets:
                ttnn.copy_host_to_device_tensor(
                    ttnn.from_torch(
                        host, dtype=dst.dtype, layout=dst.layout, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
                    ),
                    dst,
                )
        ttnn.synchronize_device(mesh_device)
        logger.info(f"prefill profile: compare {compare} unset vs 1; index keys seeded in {sorted(state.index_k)}")

    if profile:
        ttnn.ReadDeviceProfiler(mesh_device)
    times = {}
    for start, length in shapes:
        name = f"{start}:{length}"
        warm = one_chunk(start, length, drain)  # compiles (the profiler kernels too)
        if profile:
            ttnn.ReadDeviceProfiler(mesh_device)
        signpost(f"shape {name}")
        wall = one_chunk(start, length, drain)
        if profile:
            ttnn.ReadDeviceProfiler(mesh_device)
        signpost(f"end {name}")
        times[name] = {"warm_s": round(warm, 3), "wall_s": round(wall, 3)}
        logger.info(f"prefill profile: shape {name} eager {wall:.3f} s (first {warm:.3f})")
    report["times"] = times
    if compare:
        report["compare"], mismatches = {}, []
        for start, length in shapes:
            name, runs = f"{start}:{length}", {}
            for value in ("", "1"):
                if value:
                    os.environ[compare] = value
                else:
                    os.environ.pop(compare, None)
                blocks: dict = {}

                def record(layer, x, start, length):
                    blocks[f"x[{layer}]"] = _down(mesh_device, x)
                    for key, t in state.selection.items():
                        if t is not None and getattr(t, "ids", t) is not None:
                            blocks[f"{key}[{layer}]"] = _down(mesh_device, getattr(t, "ids", t))

                one_chunk(start, length, record)  # compile (the first graph of each setting)
                wall = one_chunk(start, length, record, blocks)
                runs[value or "unset"] = {"wall_s": round(wall, 3), "outputs": blocks}
            off, on = runs["unset"]["outputs"], runs["1"]["outputs"]
            diff = []
            for key in sorted(off):
                if key not in on:
                    diff.append(f"{key}: missing with {compare}=1")
                elif off[key].shape != on[key].shape:
                    diff.append(f"{key}: shape {tuple(off[key].shape)} vs {tuple(on[key].shape)}")
                elif not torch.equal(off[key], on[key]):
                    a, b = off[key].float(), on[key].float()
                    n = int((a != b).sum())
                    diff.append(f"{key}: {n} of {a.numel()} elements differ, max |d| {float((a - b).abs().max())}")
            report["compare"][name] = {
                "unset_s": runs["unset"]["wall_s"],
                "set_s": runs["1"]["wall_s"],
                "compared": sorted(off),
                "mismatches": diff,
            }
            mismatches += [f"{name} {d}" for d in diff]
            logger.info(f"prefill profile: compare {name}: {report['compare'][name]}")
        os.environ.pop(compare, None)
    logger.info("V41_PROF_RESULT " + json.dumps(report))
    out = os.environ.get("V41_PROF_OUT")
    if out:
        with open(out, "w") as f:
            json.dump(report, f, indent=1)
    if compare:
        assert not mismatches, "\n".join(mismatches)


def _down(mesh_device, t) -> torch.Tensor:
    """Every chip's copy of ``t`` as one host tensor (uint32 and bf16 both compare exactly as torch tensors)."""
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
