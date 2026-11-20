# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""MLP implementation comparison inside the Qwen3.8-27B TP4 model at S=4096 with random weights.

The implementation is the only difference between runs: off (multi-op main path), partial (qwen_mlp_partial plus
reduce-scatter, all-gather and residual add) or single (qwen_mlp_prefill). Each test runs the same model with the
main path first and with the selected implementation second.
"""

import gc
import time
import zlib

import pytest
import torch
from loguru import logger
from safetensors import safe_open

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.qwen38_27b_qb2.tt import model as model_module
from models.demos.qwen38_27b_qb2.tt.generator import configure_fabric
from models.demos.qwen38_27b_qb2.tt.model import Checkpoint, Qwen38Model

SEQUENCE_LENGTH = 4096
WARMUP_ITERATIONS = 3
TIMED_ITERATIONS = 10
PCC_THRESHOLD = 0.99
IMPLEMENTATIONS = ["off", "partial", "single"]
LAYER_OF_KIND = {"linear_attention": 0, "full_attention": 3}


class RandomCheckpoint(Checkpoint):
    """Real tensor names and shapes, deterministic random values."""

    random_head = False

    def tensor(self, name):
        with safe_open(self.path / self.index[name], framework="pt", device="cpu") as f:
            shape = f.get_slice(name).get_shape()
        if "layers" not in name and not self.random_head:
            return torch.zeros(shape, dtype=torch.bfloat16)
        generator = torch.Generator().manual_seed(zlib.crc32(name.encode()))
        scale = 0.1 if len(shape) == 1 else 0.02
        return (scale * torch.randn(shape, generator=generator, dtype=torch.float32)).to(torch.bfloat16)


@pytest.fixture(scope="module")
def mesh():
    configure_fabric()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 2), trace_region_size=200000000)
    mesh.reshape(ttnn.MeshShape(1, 4))
    yield mesh
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def build_model(mesh, monkeypatch, implementation, layer_indices, random_head=False):
    RandomCheckpoint.random_head = random_head
    monkeypatch.setattr(model_module, "Checkpoint", RandomCheckpoint)
    return Qwen38Model(mesh, layer_indices=layer_indices, policy={"qwen_mlp_op": implementation})


def select(model, implementation):
    for layer in model.layers:
        layer.policy["qwen_mlp_op"] = implementation


def timed(call, mesh):
    for _ in range(WARMUP_ITERATIONS):
        call()
    ttnn.synchronize_device(mesh)
    start = time.perf_counter()
    for _ in range(TIMED_ITERATIONS):
        call()
    ttnn.synchronize_device(mesh)
    return (time.perf_counter() - start) / TIMED_ITERATIONS


def host(tensor, mesh):
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).float()


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("kind", list(LAYER_OF_KIND))
def test_layer(mesh, monkeypatch, kind, implementation):
    model = build_model(mesh, monkeypatch, implementation, [LAYER_OF_KIND[kind]])
    layer = model.layers[0]
    assert layer.kind == kind
    cache = model.allocate_cache(batch_size=1, capacity=SEQUENCE_LENGTH)
    state = cache.layers[0]
    page_table = model.upload(
        torch.arange(cache.num_pages, dtype=torch.int32).reshape(1, -1), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    positions = model.upload(
        torch.arange(SEQUENCE_LENGTH, dtype=torch.int32).reshape(1, -1),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    cos, sin = model.rope(positions, batch=1, length=SEQUENCE_LENGTH)
    pos_matrix = ttnn.reshape(ttnn.typecast(positions, ttnn.int32), [SEQUENCE_LENGTH, 1])
    torch.manual_seed(0)
    x_torch = torch.randn(1, SEQUENCE_LENGTH, 5120).bfloat16()
    x = model.upload(x_torch)

    def run():
        return layer.prefill_forward(x, state=state, page_table=page_table, cos=cos, sin=sin, positions=pos_matrix)

    outputs = {}
    for name in ("off", implementation):
        select(model, name)
        model.reset_cache(cache)
        outputs[name] = host(run(), mesh).reshape(SEQUENCE_LENGTH, 5120)
    select(model, implementation)
    seconds = timed(run, mesh)

    x_float = x_torch.reshape(SEQUENCE_LENGTH, 5120).float()
    passed, pcc = comp_pcc(outputs["off"], outputs[implementation], PCC_THRESHOLD)
    _, delta_pcc = comp_pcc(outputs["off"] - x_float, outputs[implementation] - x_float, PCC_THRESHOLD)
    logger.info(f"RESULT layer {kind} {implementation}: {seconds * 1e3:.3f} ms, pcc {pcc}, delta pcc {delta_pcc}")
    assert passed and delta_pcc > PCC_THRESHOLD
    del model
    gc.collect()


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("layers", ["reduced", "full"])
def test_prefill(mesh, monkeypatch, layers, implementation):
    model = build_model(mesh, monkeypatch, implementation, [0, 3] if layers == "reduced" else None, random_head=True)
    cache = model.allocate_cache(batch_size=1, capacity=SEQUENCE_LENGTH)
    page_table = model.upload(
        torch.arange(cache.num_pages, dtype=torch.int32).reshape(1, -1), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    torch.manual_seed(0)
    tokens = model.upload(
        torch.randint(0, model.config.vocab_size, (1, SEQUENCE_LENGTH), dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    def run():
        return model.prefill(tokens, cache=cache, page_table=page_table, length=SEQUENCE_LENGTH)

    def logits():
        model.reset_cache(cache)
        out = ttnn.to_torch(run(), mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float()
        return out.reshape(-1, out.shape[-1])[0]

    results = {}
    for name in ("off", implementation):
        select(model, name)
        results[name] = logits()
    select(model, implementation)
    seconds = timed(run, mesh)

    passed, pcc = comp_pcc(results["off"], results[implementation], PCC_THRESHOLD)
    same_top1 = results["off"].argmax().item() == results[implementation].argmax().item()
    logger.info(f"RESULT prefill {layers} {implementation}: {seconds * 1e3:.3f} ms, logits pcc {pcc}, top1 {same_top1}")
    assert passed and same_top1
    del model
    gc.collect()
