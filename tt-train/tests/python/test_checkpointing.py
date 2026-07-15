# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Round-trip tests for ``ttml.checkpointing``."""

import os
import pickle

import ml_dtypes
import numpy as np
import pytest

import ttml
import ttnn
from ttml import checkpointing
from ttml.common.muon_optimizer import MuonWithAdamW
from ttml.modules import AbstractModuleBase, LinearLayer

DIM = 32  # tile-aligned: LinearLayer weights are (1,1,DIM,DIM), bias (1,1,1,DIM)


class TwoLayer(AbstractModuleBase):
    """Two stacked linear layers -> four bf16 params (two weights, two biases)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = LinearLayer(DIM, DIM)
        self.fc2 = LinearLayer(DIM, DIM)

    def forward(self, x):
        return self.fc2(self.fc1(x))


def _make_optimizer(name: str, params: "ttml.NamedParameters") -> "ttml.optimizers.OptimizerBase":
    """Construct an optimizer by short name. ``MuonWithAdamW`` is the Python composite whose state dict
    nests ``muon``/``adamw`` sub-dicts (note its ``(config, params)`` arg order); the others are flat."""
    if name == "AdamW":
        cfg = ttml.optimizers.AdamWConfig.make(lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0)
        return ttml.optimizers.AdamW(params, cfg)
    if name == "MuonComposite":
        cfg = ttml.optimizers.MuonConfig.make(lr=1e-3, momentum=0.95, ns_steps=5)
        return ttml.optimizers.MuonComposite(params, cfg)
    if name == "MuonWithAdamW":
        return MuonWithAdamW(
            {"muon": {"lr": 1e-2, "momentum": 0.95, "ns_steps": 5}, "adamw": {"lr": 1e-3, "weight_decay": 0.0}},
            params,
        )
    raise ValueError(f"Unknown optimizer name: {name!r}")


def _train_steps(model: TwoLayer, opt, n: int, x_np: np.ndarray) -> None:
    """Run ``n`` deterministic optimizer steps so the optimizer's moments become non-zero.

    Input is rebuilt from the same numpy each step (identical input -> deterministic update). The graph is
    reset after each step; parameters are leaves and persist across resets.
    """
    ctx = ttml.autograd.AutoContext.get_instance()
    for _ in range(n):
        opt.zero_grad()
        x = ttml.autograd.Tensor.from_numpy(x_np, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16)
        loss = ttml.ops.unary.mean(model(x))
        loss.backward(False)
        opt.step()
        ctx.reset_graph()


def _params_np(model: TwoLayer) -> dict:
    """{name: float32 numpy} for every model parameter."""
    return {name: t.to_numpy(ttnn.DataType.FLOAT32) for name, t in model.parameters().items()}


def _moments_np(opt) -> dict:
    """{path: float32 numpy} for every tensor moment in the optimizer state dict (NamedParameters leaves)."""
    out = {}

    def walk(node, prefix):
        if isinstance(node, ttml.NamedParameters):
            for name, t in node.items():
                out[f"{prefix}/{name}"] = t.to_numpy(ttnn.DataType.FLOAT32)
        elif isinstance(node, dict):
            for key, sub in node.items():
                walk(sub, f"{prefix}/{key}")

    for key, value in opt.get_state_dict().items():
        walk(value, key)
    return out


def _scalars(opt) -> dict:
    """{path: value} for every non-tensor leaf in the optimizer state dict (steps, lr, ...), at any depth.

    Optimizer-agnostic: works for flat state dicts (``AdamW``: top-level ``steps``) and nested ones
    (``MuonWithAdamW``: ``muon/steps``, ``adamw/steps``), so step/lr restoration is checked without
    special-casing — `OptimizerBase` exposes no ``get_steps`` to Python.
    """
    out = {}

    def walk(node, prefix):
        if isinstance(node, ttml.NamedParameters):
            return  # tensors are compared by _moments_np
        if isinstance(node, dict):
            for key, sub in node.items():
                walk(sub, f"{prefix}/{key}" if prefix else key)
        else:
            out[prefix] = node

    walk(opt.get_state_dict(), "")
    return out


@pytest.mark.requires_device
@pytest.mark.parametrize("optimizer_name", ["AdamW", "MuonComposite", "MuonWithAdamW"])
def test_save_load_roundtrip(tmp_path, optimizer_name):
    """Save a trained model+optimizer to disk, restore into a fresh pair, and verify exact recovery."""
    ctx = ttml.autograd.AutoContext.get_instance()
    x_np = np.random.default_rng(0).standard_normal((1, 1, DIM, DIM)).astype(np.float32)

    # --- original: train a few steps so optimizer moments are non-zero
    ttml.init.manual_seed(0)
    orig = TwoLayer()
    opt = _make_optimizer(optimizer_name, orig.parameters())
    _train_steps(orig, opt, n=3, x_np=x_np)
    orig_scalars = _scalars(opt)
    step_values = [v for k, v in orig_scalars.items() if k.split("/")[-1] == "steps"]
    assert step_values and all(v == 3 for v in step_values), f"expected every step counter == 3, got {orig_scalars}"

    orig_params = _params_np(orig)
    orig_moments = _moments_np(opt)
    assert orig_moments, "optimizer should expose at least one tensor moment after stepping"
    assert any(np.any(m != 0.0) for m in orig_moments.values()), "moments should be non-zero after training"

    # --- save through the real write path
    path = str(tmp_path / "ckpt.pkl")
    checkpointing.save_checkpoint(path, header={"step": 3}, model_params=orig.parameters(), optimizer=opt)
    assert os.path.exists(path)
    assert not os.path.exists(path + ".tmp"), "temp file should be gone after the atomic replace"

    ctx.reset_graph()

    # --- fresh pair with a DIFFERENT init
    ttml.init.manual_seed(12345)
    restored = TwoLayer()
    opt2 = _make_optimizer(optimizer_name, restored.parameters())

    fresh_params = _params_np(restored)
    for name, orig_arr in orig_params.items():
        assert not np.allclose(
            orig_arr, fresh_params[name], rtol=1e-2, atol=1e-2
        ), f"fresh param {name} matches the original before load — test can't prove load did anything"

    # --- load through the real read path (dispatches NamedParameters -> assign, OptimizerBase -> set_state_dict)
    header = checkpointing.load_checkpoint(path, model_params=restored.parameters(), optimizer=opt2)
    assert header["step"] == 3

    restored_params = _params_np(restored)
    for name, orig_arr in orig_params.items():
        np.testing.assert_array_equal(restored_params[name], orig_arr, err_msg=f"param {name} mismatch after load")

    assert _scalars(opt2) == orig_scalars, "optimizer scalars (steps/lr/...) not restored"
    restored_moments = _moments_np(opt2)
    assert set(restored_moments) == set(orig_moments)
    for key, orig_arr in orig_moments.items():
        np.testing.assert_array_equal(restored_moments[key], orig_arr, err_msg=f"optimizer moment {key} mismatch")

    # --- strongest check: one more identical step must produce identical params on both
    _train_steps(orig, opt, n=1, x_np=x_np)
    _train_steps(restored, opt2, n=1, x_np=x_np)
    orig_after = _params_np(orig)
    restored_after = _params_np(restored)
    for name, orig_arr in orig_after.items():
        np.testing.assert_array_equal(
            restored_after[name], orig_arr, err_msg=f"param {name} diverged after one more step"
        )

    ctx.reset_graph()


@pytest.mark.requires_device
def test_partial_load_skips_absent_groups(tmp_path):
    """Loading only the ``model`` group restores it and silently skips the ``optimizer`` group (inference path)."""
    ctx = ttml.autograd.AutoContext.get_instance()
    x_np = np.random.default_rng(1).standard_normal((1, 1, DIM, DIM)).astype(np.float32)

    ttml.init.manual_seed(0)
    orig = TwoLayer()
    opt = _make_optimizer("AdamW", orig.parameters())
    _train_steps(orig, opt, n=2, x_np=x_np)
    orig_params = _params_np(orig)

    path = str(tmp_path / "ckpt.pkl")
    checkpointing.save_checkpoint(path, header={"step": 2}, model_params=orig.parameters(), optimizer=opt)
    ctx.reset_graph()

    ttml.init.manual_seed(99)
    restored = TwoLayer()
    header = checkpointing.load_checkpoint(path, model_params=restored.parameters())  # no optimizer target
    assert header["step"] == 2
    restored_params = _params_np(restored)
    for name, orig_arr in orig_params.items():
        np.testing.assert_allclose(
            restored_params[name], orig_arr, rtol=1e-2, atol=1e-2, err_msg=f"param {name} mismatch after partial load"
        )

    ctx.reset_graph()


@pytest.mark.requires_device
def test_bf16_dtype_preserved_on_disk(tmp_path):
    """A bf16 weight is stored as bf16 on disk, not widened to float32."""
    ctx = ttml.autograd.AutoContext.get_instance()
    ttml.init.manual_seed(0)
    model = TwoLayer()
    assert all(
        t.get_value(ttml.autograd.PreferredPrecision.NATIVE).dtype == ttnn.DataType.BFLOAT16
        for t in model.parameters().values()
    ), "LinearLayer params should be bf16"

    path = str(tmp_path / "bf16.pkl")
    checkpointing.save_checkpoint(path, header={}, model_params=model.parameters())

    with open(path, "rb") as f:
        pickle.load(f)  # skip the header record (format/header/manifest)
        first_tensor = pickle.load(f)  # first streamed tensor record
    assert first_tensor.dtype == ml_dtypes.bfloat16, f"expected bf16 on disk, got {first_tensor.dtype}"

    ctx.reset_graph()


def test_rng_state_roundtrips_in_header(tmp_path):
    """The opaque header round-trips its payload byte-for-byte through pickle. train.py's saver stores the
    C++ generator string under "rng"; this test also bundles a numpy state tuple to prove the header
    preserves embedded ndarrays exactly, not just strings. Host-only: no model tensors."""
    ctx = ttml.autograd.AutoContext.get_instance()
    ctx.set_seed(7)
    np.random.seed(7)
    rng = {"cpp": ctx.get_generator_state(), "numpy": np.random.get_state()}

    path = str(tmp_path / "rng.pkl")
    checkpointing.save_checkpoint(path, header={"step": 0, "rng": rng})

    restored = checkpointing.load_checkpoint(path)["rng"]
    assert restored["cpp"] == rng["cpp"]
    assert restored["numpy"][0] == rng["numpy"][0]  # algorithm tag ("MT19937")
    np.testing.assert_array_equal(restored["numpy"][1], rng["numpy"][1])  # 624-word state
    assert restored["numpy"][2:] == rng["numpy"][2:]  # position + cached-gaussian fields


def test_bad_file_raises(tmp_path, expect_error):
    """A non-checkpoint, empty, or wrong-format file raises a clear ValueError (host-only — no device)."""
    not_pickle = tmp_path / "garbage.pkl"
    not_pickle.write_bytes(b"not a pickle at all")
    with expect_error(ValueError, "could not read checkpoint header"):
        checkpointing.read_header(str(not_pickle))

    empty = tmp_path / "empty.pkl"
    empty.touch()
    with expect_error(ValueError, "could not read checkpoint header"):
        checkpointing.read_header(str(empty))

    wrong_format = tmp_path / "wrong.pkl"
    with open(wrong_format, "wb") as f:
        pickle.dump({"not": "a checkpoint"}, f)
    with expect_error(ValueError, "not a ttml checkpoint or unsupported format"):
        checkpointing.read_header(str(wrong_format))
