# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import json
import os

import pytest
import torch

import ttnn
from models.common.utility_functions import is_quasar


@pytest.fixture(autouse=True)
def quasar_host_cast_fp32_to_bf16(monkeypatch):
    """On Quasar, cast float32 torch tensors to bfloat16 on the host before ttnn.from_torch uploads them.

    When the source is float32 and the target dtype is bfloat16, from_torch converts on the device: it
    tilizes in float32 and then typecasts. The float32 tilize faults on Quasar with UNPACKER_0
    ILLEGAL_FORMAT_CONVERSION (issue #57780). Casting on the host makes the source dtype match the
    target dtype, so the upload runs neither the float32 tilize nor a device typecast. ttnn.as_tensor
    calls ttnn.from_torch, so weight uploads are covered too.
    """
    if not is_quasar():
        return

    orig_from_torch = ttnn.from_torch

    def _from_torch(tensor, *args, **kwargs):
        dtype = kwargs.get("dtype", args[0] if args else None)
        if (
            isinstance(tensor, torch.Tensor)
            and tensor.dtype == torch.float32
            and dtype == ttnn.bfloat16
            and kwargs.get("device") is not None
        ):
            tensor = tensor.to(torch.bfloat16)
        return orig_from_torch(tensor, *args, **kwargs)

    monkeypatch.setattr(ttnn, "from_torch", _from_torch)


@pytest.fixture(autouse=True)
def quasar_block_float_to_bf16(monkeypatch, quasar_host_cast_fp32_to_bf16, quasar_ops):
    """On Quasar, use bfloat16 wherever the model asks for bfloat8_b or bfloat4_b.

    Quasar does not support the Bfp8_b and Bfp4_b data formats, so any op that sets up a dataflow
    buffer in one of them fails program validation. The model sets these dtypes at many call sites,
    so the wrappers below change them where they reach ttnn: tensor uploads, and the listed ops that take
    an output dtype. This fixture depends on quasar_host_cast_fp32_to_bf16 so that the from_torch wrapper
    of this fixture is the outer one: it changes the dtype to bfloat16, and then the inner wrapper from
    quasar_host_cast_fp32_to_bf16 does the float32 host cast. This fixture depends on quasar_ops so that
    it wraps the Quasar versions of the ops that quasar_ops routes.

    bfloat16 weights take about 1.9x (Bfp8_b) to 3.6x (Bfp4_b) as much DRAM. They are cached in a directory
    of their own. The weight cache directory is named after the dtype that the model requests (bfloat8_b),
    so a Quasar run would otherwise share a directory with a block float run on another device that uses the
    same cache location. If that run marked the directory complete (see ModelArgs.weight_cache_is_complete),
    a Quasar run would skip loading the original weights, and ttnn.as_tensor would then find no bfloat16 file
    for a weight that the model asks for in a block float format. weight_cache_is_complete also gets its
    directory from weight_cache_path, so on Quasar it looks for the marker in the bfloat16 directory, which
    only Quasar runs write to.
    """
    if not is_quasar():
        return

    from models.experimental.ops.quasar.gpt_oss.tt.model_config import ModelArgs

    block_float_dtypes = (ttnn.bfloat8_b, ttnn.bfloat4_b)

    def _to_bf16(dtype):
        return ttnn.bfloat16 if dtype in block_float_dtypes else dtype

    def _wrap(op, dtype_arg_index):
        # dtype_arg_index is the position of dtype among the positional arguments of op, or None when
        # op takes dtype only as a keyword argument.
        def _op(*args, **kwargs):
            if "dtype" in kwargs:
                kwargs["dtype"] = _to_bf16(kwargs["dtype"])
            elif dtype_arg_index is not None and len(args) > dtype_arg_index:
                args = list(args)
                args[dtype_arg_index] = _to_bf16(args[dtype_arg_index])
            return op(*args, **kwargs)

        return _op

    for name, dtype_arg_index in (
        ("from_torch", 1),
        ("as_tensor", 1),
        ("typecast", 1),
        ("embedding", None),
        ("matmul", None),
        ("linear", None),
        ("sparse_matmul", None),
    ):
        monkeypatch.setattr(ttnn, name, _wrap(getattr(ttnn, name), dtype_arg_index))

    orig_weight_cache_path = ModelArgs.weight_cache_path

    def _weight_cache_path(self, dtype):
        path = orig_weight_cache_path(self, dtype)
        quasar_path = path.with_name(f"{path.name}_quasar_bf16")
        quasar_path.mkdir(parents=True, exist_ok=True)
        return quasar_path

    monkeypatch.setattr(ModelArgs, "weight_cache_path", _weight_cache_path)


@pytest.fixture(autouse=True)
def quasar_ops(monkeypatch):
    """On Quasar, route the mainline ops below, which Quasar rejects, to their ttnn.experimental.quasar versions.

    Experimental ops maintain the same API as the mainline ops, so the model code needs no change.
    """
    if not is_quasar():
        return

    quasar = ttnn.experimental.quasar
    routes = (
        # (module the model calls, op name there, Quasar module, Quasar op name)
        (ttnn, "add", quasar, "add"),
        (ttnn, "mul", quasar, "multiply"),
        (ttnn, "multiply", quasar, "multiply"),
        (ttnn, "subtract", quasar, "subtract"),
        (ttnn, "sub", quasar, "subtract"),
        # The mainline linear adds some biases with a C++ call to the mainline binary add. The "add" route above
        # does not reach that call, so linear is routed to experimental version as a whole.
        (ttnn, "linear", quasar, "linear"),
        (ttnn.transformer, "scaled_dot_product_attention", quasar.transformer, "scaled_dot_product_attention"),
        (
            ttnn.transformer,
            "scaled_dot_product_attention_decode",
            quasar.transformer,
            "scaled_dot_product_attention_decode",
        ),
        (
            ttnn.transformer,
            "paged_scaled_dot_product_attention_decode",
            quasar.transformer,
            "paged_scaled_dot_product_attention_decode",
        ),
    )
    for module, name, quasar_module, quasar_name in routes:
        monkeypatch.setattr(module, name, getattr(quasar_module, quasar_name))


def pytest_addoption(parser):
    parser.addoption("--skip-model-load", action="store_true", default=False, help="Skip loading the model state dict")


@pytest.fixture(scope="session")
def state_dict(request):
    load_model = not request.config.getoption("--skip-model-load")
    model_path = os.getenv("HF_MODEL", None)
    if model_path is None or not load_model:
        # Explicit skip: build weights purely from the ttnn cache.
        return {}
    # Defer the (expensive) HF weight load to create_tt_model, which knows the mesh shape +
    # dtype and can skip it when a warm ttnn cache is already on disk (see
    # ModelArgs.weight_cache_is_complete). Returning None signals "load if needed".
    return None


@pytest.fixture
def test_thresholds(request):
    with open("models/experimental/ops/quasar/gpt_oss/unit_test_thresholds.json", "r") as f:
        thresholds = json.load(f)
    return thresholds
