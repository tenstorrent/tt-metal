# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Prefill chunk: the GR read/write over 32 rows against the 1-row read/write, row for row, on a torch-backed ttnn.

No device.  The step-4 fake (``test_mtp_v2_step4_rows_no_device``) is extended with the GR ops the 1-row path
uses (the async gather + fast reduce of the partial, the zero-copy views, the fused sigmoid multiply, the FP32
linear output).  Every op is per row or per element, so ``read_rows``/``write_rows`` must equal 32 ``read``/``write``
calls bitwise.  The source pins hold the 1-row bodies untouched and the rows bodies free of views and host I/O.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from models.demos.blackhole.qwen38_flash_next.ttnn import gr as gr_module
from models.demos.blackhole.qwen38_flash_next.ttnn.gr import (
    BLOCK_ROWS_LOCAL_SHAPE,
    FLAT_LOCAL_WIDTH,
    INJECTION_ROWS_SHAPE,
    PARTIAL_WIDTH,
    RESIDUAL_ROWS_LOCAL_SHAPE,
    Qwen38TTNNGatedResidual,
    Qwen38TTNNGatedResidualWeights,
)

GR_SOURCE = Path(__file__).resolve().parents[1] / "ttnn" / "gr.py"
ROWS = 32
# The tests directory is not a package: load the step-4 fake by path.
_spec = importlib.util.spec_from_file_location(
    "test_mtp_v2_step4_rows_no_device", Path(__file__).with_name("test_mtp_v2_step4_rows_no_device.py")
)
step4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(step4)
BF16, FP32, TILE, TP = step4.BF16, step4.FP32, step4.TILE, step4.TP
FakeChunk, FakeContract, FakeTensor, make_fake_ttnn = (
    step4.FakeChunk,
    step4.FakeContract,
    step4.FakeTensor,
    step4.make_fake_ttnn,
)


def _padded_shape(self) -> tuple[int, ...]:
    shape = self.shape
    if self.layout != TILE or len(shape) < 2:
        return shape
    return (*shape[:-2], -(-shape[-2] // 32) * 32, -(-shape[-1] // 32) * 32)


def _sequential_sum(x: torch.Tensor, dim: int) -> torch.Tensor:
    """fp32 accumulation in index order, one bf16/fp32 rounding at the end (fast_reduce_nc's contract)."""

    pieces = x.float().unbind(dim)
    acc = pieces[0]
    for piece in pieces[1:]:
        acc = acc + piece
    return acc.unsqueeze(dim).to(x.dtype)


def _gr_fake():
    fake = make_fake_ttnn(FakeChunk())
    FakeTensor.padded_shape = property(_padded_shape)
    fake.UnaryOpType.SIGMOID = "SIGMOID"

    def linear(a, w, *, memory_config=None, program_config=None, dtype=None, compute_kernel_config=None):
        out = dtype or a.dtype
        results = []
        for x, wt in zip(a.torch_shards(), w.torch_shards()):
            weight = wt.reshape(wt.shape[-2], wt.shape[-1]).float()
            rows = x.reshape(-1, x.shape[-1])
            product = torch.stack([(row[None].float() @ weight)[0] for row in rows])
            results.append(product.reshape(*x.shape[:-1], weight.shape[-1]).to(out.torch))
        return FakeTensor(results, out, a.layout)

    base_multiply = fake.multiply

    def multiply(a, b, *, memory_config=None, dtype=None, input_tensor_a_activations=None, output_tensor=None):
        if input_tensor_a_activations:
            assert input_tensor_a_activations == [("SIGMOID", 4.0, 0.0)]
            a = FakeTensor([torch.sigmoid(x) for x in a.torch_shards()], a.dtype, a.layout)
        return base_multiply(a, b, memory_config=memory_config, dtype=dtype, output_tensor=output_tensor)

    def post_all_gather(
        tensor, gathered, *, epsilon, weight=None, memory_config=None, compute_kernel_config=None, dtype
    ):
        outputs = []
        for x, stats in zip(tensor.torch_shards(), gathered.torch_shards()):
            total = stats[..., ::32].sum(dim=-1, keepdim=True)
            mean = total / (TP * x.shape[-1])
            outputs.append((x.float() * torch.rsqrt(mean + epsilon)).to(dtype.torch))
        return FakeTensor(outputs, dtype, tensor.layout)

    def all_gather_async(tensor, *, persistent_output_buffer, dim, **kwargs):
        full = torch.cat(tensor.torch_shards(), dim=dim)
        return FakeTensor([full.clone() for _ in range(TP)], tensor.dtype, tensor.layout)

    def fast_reduce_nc(tensor, *, dims, output, compute_kernel_config, memory_config):
        # The device reports the 32-row tile as the row count of a TILE result below 32 rows (zero pad rows):
        # the rows bodies restore the logical rows with ttnn.reshape(t, logical, padded).
        (dim,) = dims
        shards = [_sequential_sum(x, dim) for x in tensor.torch_shards()]
        if tensor.layout == TILE and dim != 2 and shards[0].shape[2] % 32:
            shards = [F.pad(x, (0, 0, 0, 32 - x.shape[2] % 32)) for x in shards]
        return FakeTensor(shards, tensor.dtype, tensor.layout)

    def reshape(tensor, shape, padded_shape=None, memory_config=None):
        shards = tensor.torch_shards()
        if padded_shape is not None and tuple(shape) != tuple(shards[0].shape):
            assert tuple(padded_shape) == tuple(shards[0].shape), (padded_shape, shards[0].shape)
            shards = [x.narrow(2, 0, shape[2]) for x in shards]
        return FakeTensor([x.reshape(tuple(shape)).clone() for x in shards], tensor.dtype, tensor.layout)

    def view(tensor, shape):
        return FakeTensor([x.reshape(tuple(shape)) for x in tensor.torch_shards()], tensor.dtype, tensor.layout)

    fake.linear = linear
    fake.multiply = multiply
    fake.reshape = reshape
    fake.rms_norm_post_all_gather = post_all_gather
    # ttnn.experimental.view as the step-4 fake models it: the same tile pages under the new shape (a torch reshape
    # would give the 1-row shapes the same values but not the rows' flat rows, which is the claim under test).
    fake.experimental = SimpleNamespace(
        all_gather_async=all_gather_async, fast_reduce_nc=fast_reduce_nc, view=step4._view, torch_view=view
    )
    return fake


def _bf16(*shape: int, scale: float = 1.0) -> torch.Tensor:
    return (torch.randn(*shape) * scale).to(torch.bfloat16)


def _gr_module(fake) -> Qwen38TTNNGatedResidual:
    torch.manual_seed(7)
    weights = Qwen38TTNNGatedResidualWeights(
        norm_scale=FakeTensor([(torch.randn(1, 4, 1, 640) * 0.1 + 0.25) for _ in range(TP)], FP32, TILE, 3),
        down_inject=FakeTensor(
            [_bf16(1, 1, FLAT_LOCAL_WIDTH, PARTIAL_WIDTH, scale=FLAT_LOCAL_WIDTH**-0.5) for _ in range(TP)],
            BF16,
            TILE,
            2,
        ),
        up=FakeTensor(
            [_bf16(1, 1, PARTIAL_WIDTH, FLAT_LOCAL_WIDTH, scale=PARTIAL_WIDTH**-0.5) for _ in range(TP)],
            BF16,
            TILE,
            3,
        ),
        replicated_anchor=FakeTensor([torch.zeros(1, 1, 1, 1, dtype=torch.bfloat16) for _ in range(TP)], BF16, TILE),
        epsilon=1e-6,
        layer_index=0,
        block="attn",
        namespace="backbone",
    )
    module = object.__new__(Qwen38TTNNGatedResidual)
    module.mesh_device = "mesh"
    module.mesh_contract = FakeContract()
    module.weights = weights
    module.tt_ccl = SimpleNamespace(
        get_num_links=lambda axis: 1,
        get_and_cycle_ag_semaphore_handles=lambda axis: "ag",
        get_and_cycle_barrier_semaphore_handle=lambda axis: "barrier",
    )
    module.collective_topology = "linear"
    module.compute_config = "compute"
    module.down_inject_act_memory_config = fake.DRAM_MEMORY_CONFIG
    module.down_inject_program_config = "down"
    module.up_act_memory_config = fake.DRAM_MEMORY_CONFIG
    module.up_program_config = "up"
    module.norm_scale_flat = fake.experimental.view(weights.norm_scale, (1, 1, 1, FLAT_LOCAL_WIDTH))
    return module


@pytest.fixture
def fake(monkeypatch):
    fake_ttnn = _gr_fake()
    monkeypatch.setattr(gr_module, "ttnn", fake_ttnn)
    return fake_ttnn


def _rows(tensor: FakeTensor, row: int, dim: int = 2) -> FakeTensor:
    return FakeTensor([x.narrow(dim, row, 1).clone() for x in tensor.torch_shards()], tensor.dtype, tensor.layout)


def _equal(batched: FakeTensor, per_row: list[FakeTensor], *, dim: int, label: str) -> None:
    for device, local in enumerate(batched.torch_shards()):
        stacked = torch.cat([row.torch_shards()[device] for row in per_row], dim=dim)
        assert stacked.shape == local.shape, (label, device, stacked.shape, local.shape)
        assert torch.equal(stacked.view(torch.int16), local.view(torch.int16)), (label, device)


@pytest.mark.parametrize("flat_views", (False, True))
def test_read_rows_and_write_rows_equal_the_one_row_read_and_write_row_for_row(fake, flat_views: bool) -> None:
    module = _gr_module(fake)
    torch.manual_seed(11)
    residual_rows = FakeTensor([_bf16(*RESIDUAL_ROWS_LOCAL_SHAPE) for _ in range(TP)], BF16, TILE, 3)
    block_rows = FakeTensor([_bf16(*BLOCK_ROWS_LOCAL_SHAPE) for _ in range(TP)], BF16, TILE, 3)

    block_input, state = module.read_rows(residual_rows, flat_views=flat_views)
    written = module.write_rows(block_rows, state)
    assert block_input.shape == BLOCK_ROWS_LOCAL_SHAPE and state.injection.shape == INJECTION_ROWS_SHAPE
    assert written.shape == RESIDUAL_ROWS_LOCAL_SHAPE and state.residual is residual_rows

    one_row = [module.read(_rows(residual_rows, row)) for row in range(ROWS)]
    _equal(block_input, [block for block, _ in one_row], dim=2, label="GR read block input")
    _equal(state.injection, [row_state.injection for _, row_state in one_row], dim=2, label="GR read injection")
    _equal(
        written,
        [module.write(_rows(block_rows, row), one_row[row][1]) for row in range(ROWS)],
        dim=2,
        label="GR write",
    )
    # The injection depends on the row: the per-row test is not vacuous.
    injection = state.injection.torch_shards()[0]
    assert not torch.equal(injection[..., 0, :], injection[..., 1, :])


def _method(name: str) -> ast.FunctionDef:
    tree = ast.parse(GR_SOURCE.read_text(encoding="utf-8"))
    (cls,) = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Qwen38TTNNGatedResidual"]
    (function,) = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name]
    return function


def _calls(function: ast.FunctionDef) -> list[str]:
    return [ast.unparse(node.func) for node in ast.walk(function) if isinstance(node, ast.Call)]


@pytest.mark.parametrize("rows", (2, 8, ROWS))
def test_flat_views_read_rows_is_the_permute_form_bitwise(fake, rows: int) -> None:
    """Every branch is one 32-row tile row whatever the row count, so the branch-major tile sequence is the flat rows'
    tile sequence: the view over the same pages reads what the permute + reshape pair materializes (the fake models
    the tile pages, the rows below 32 as their tile padding)."""

    module = _gr_module(fake)
    torch.manual_seed(13 + rows)
    shapes = gr_module.gr_rows_shapes(rows)
    residual_rows = FakeTensor([_bf16(*shapes["residual"]) for _ in range(TP)], BF16, TILE, 3)
    permuted, permuted_state = module.read_rows(residual_rows)
    viewed, viewed_state = module.read_rows(residual_rows, flat_views=True)
    _equal(viewed, [permuted], dim=2, label=f"GR read block input at {rows} rows (flat views vs permutes)")
    _equal(viewed_state.injection, [permuted_state.injection], dim=2, label=f"GR read injection at {rows} rows")
    unit = FakeTensor([_bf16(*shapes["residual"]) for _ in range(TP)], BF16, TILE, 3)
    by_view = fake.experimental.view(unit, shapes["flat"])
    by_permute = fake.reshape(fake.permute(unit, (0, 2, 1, 3)), shapes["flat"])
    _equal(by_view, [by_permute], dim=2, label=f"flat rows view vs permute + reshape at {rows} rows")
    assert by_view.padded_shape == shapes["flat_padded"]
    if rows > 1:
        # A plain torch reshape of the branch-major rows is not the device view: the fake's tile-page model matters.
        assert not torch.equal(
            by_view.torch_shards()[0].view(torch.int16),
            fake.experimental.torch_view(unit, shapes["flat"]).torch_shards()[0].view(torch.int16),
        )


def test_rows_bodies_walk_the_layout_by_permutes_or_flat_views_and_touch_no_host() -> None:
    read_calls = _calls(_method("read_rows"))
    assert read_calls.count("ttnn.permute") == 2 and read_calls.count("ttnn.experimental.view") == 2
    assert read_calls.count("ttnn.reshape") == 5  # stats, flat rows, token-major rows, the two row restores
    assert read_calls.count("ttnn.linear") == 2 and read_calls.count("ttnn.experimental.fast_reduce_nc") == 2
    assert read_calls.count("ttnn.experimental.all_gather_async") == 1 and "ttnn.all_reduce" not in read_calls
    write_calls = _calls(_method("write_rows"))
    assert write_calls.count("ttnn.permute") == 1 and "ttnn.reshape" not in write_calls
    for calls in (read_calls, write_calls):
        for forbidden in ("ttnn.from_torch", "ttnn.to_torch", "ttnn.copy_host_to_device_tensor"):
            assert forbidden not in calls
    # The rows path derives its shapes from the validated row count of its input (1..32 lanes; 32 or 128 = the
    # chunk forms; the slab counts), never from a runtime int of its own: every shape comes from the module's
    # row-shape helpers, and below 32 rows the padded shapes are one tile (tile_rows).
    source = inspect.getsource(Qwen38TTNNGatedResidual.read_rows) + inspect.getsource(
        Qwen38TTNNGatedResidual.write_rows
    )
    assert source.count("residual_rows_shape(rows)") >= 3 and ".shape[" not in source
    assert "tile_rows = max(rows, CHUNK_ROWS)" in source and "block_rows_shape(rows)" in source


def test_one_row_read_and_write_keep_their_views_and_op_walk() -> None:
    read_calls = _calls(_method("read"))
    assert read_calls.count("ttnn.experimental.view") == 1 and "ttnn.permute" not in read_calls
    normalize_calls = _calls(_method("_normalize"))
    assert normalize_calls.count("ttnn.experimental.view") == 1 and "ttnn.permute" not in normalize_calls
    write_calls = [call for call in _calls(_method("write")) if call.startswith("ttnn.")]
    assert write_calls == ["ttnn.reshape", "ttnn.multiply", "ttnn.add"]
    partial = ast.get_source_segment(GR_SOURCE.read_text(encoding="utf-8"), _method("_all_reduce_partial"))
    assert "expected_shape != PARTIAL_REDUCTION_SHAPE" in partial and "CHUNK_ROWS" not in partial
    assert RESIDUAL_ROWS_LOCAL_SHAPE == (1, 4, 32, 640) and BLOCK_ROWS_LOCAL_SHAPE == (1, 1, 32, 640)
