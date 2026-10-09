# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Mesh-topology labels of the moreh optimizer, clip_grad_norm and dot_backward outputs.

The device-operation framework relabels every tensor an op returns with the union of its inputs' placements. For
these ops the returned tensors are frequently the caller's own: moreh_adamw / moreh_adam / moreh_sgd write into a
preallocated ``*_out`` (tt-train's MorehAdamW hands in the parameter itself), moreh_clip_grad_norm scales its
``inputs`` in place, moreh_dot_backward only ever writes caller-provided grads. The union is the right label for a
fresh output, but for a caller-owned one it can drop the caller's distribution shape (a collapsed 1-D label against
an N-D input) or replace the caller's shard dim with an input's.

Contract pinned here (``ttnn::operations::core::caller_owned_output_topology``, shared with the in-place softmax /
layer_norm and KV-cache hooks of PRs #59329-#59332):
  * a preallocated (caller-owned) output keeps the label it arrived with while that label still describes the data:
    no input may be sharded along a mesh axis on which the output is replicated;
  * otherwise -- a gradient sharded along an axis on which the parameter is replicated leaves every device with a
    different parameter -- it takes the union, like an output the op allocates itself; a ``Replicate`` label kept on
    per-device-different data would make the flatbuffer serialiser save one device's copy for all;
  * an output the op allocates itself takes the union of all inputs (it is per-device distinct whenever any input is);
  * absent optional outputs (``amsgrad=False``, ``momentum=0``, one grad of dot_backward) get no entry, so the
    3-vs-4 / 1-vs-2 slot counts are exercised;
  * the topology label does not take part in the program hash, so a cached program is reused across labels.

The ``*_keeps_own_label`` tests pin the kept label where the union would differ (a collapsed ``{2},[Shard(2)]``
parameter against an N-D ``{1,2},[Replicate, Shard(3)]`` gradient: the union is the N-D label). The
``*_follow_the_data`` tests pin the fallback: a replicated parameter updated from a per-device-different gradient
comes back ``['Shard(3)']`` on the caller's own handle, and the per-device results are checked to differ.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose_and_pcc

pytestmark = pytest.mark.parametrize("mesh_device", [2], indirect=True)

LR = 0.1
# The kernels receive every hyperparameter as a bf16 scalar tile (truncated float32), so the betas must survive that:
# these are the nightly test_moreh_adamw / test_moreh_adam values. The usual 0.999 becomes 0.99609375, i.e. 1 - beta2
# is 3.9x too large, and a step computed from it differs from any float32 Adam by O(1) wherever exp_avg_sq is small.
BETAS = (0.5, 0.555)
EPS = 1e-6
WEIGHT_DECAY = 0.3
STEP = 3
RTOL = ATOL = 0.1
PCC = 0.99


# --- helpers ----------------------------------------------------------------------------------------------------


def _replicated(tensor, mesh_device):
    return ttnn.from_torch(
        tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _sharded(tensor, mesh_device, dim):
    """Collapsed 1-D label ``{N},[Shard(dim)]`` (what the default mapper produces)."""
    return ttnn.from_torch(
        tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=dim),
    )


def _sharded_nd(tensor, mesh_device, dim):
    """N-D label ``{1,N},[Replicate, Shard(dim)]``: the same split across the mesh columns, one placement per axis."""
    return ttnn.from_torch(
        tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(None, dim)),
    )


GRAD_LABELS = {"collapsed": ["Shard(3)"], "nd": ["Replicate", "Shard(3)"]}


def _place_grad(tensor, mesh_device, grad_label):
    return (_sharded if grad_label == "collapsed" else _sharded_nd)(tensor, mesh_device, 3)


def _placement_names(tensor):
    return [
        f"Shard({p.dim})" if isinstance(p, ttnn.PlacementShard) else "Replicate"
        for p in tensor.tensor_topology().placements()
    ]


def _assert_label(tensor, expected_names, what):
    names = _placement_names(tensor)
    assert names == expected_names, f"{what}: topology {names}, expected {expected_names}"


def _device_slices(tensor):
    """Each device's slice as float32 torch, in mesh order. No composer: nothing here trusts the label."""
    return [ttnn.to_torch(shard).float() for shard in ttnn.get_device_tensors(tensor.cpu())]


def _assert_close(expected, actual, what):
    passing, out = comp_allclose_and_pcc(expected, actual, pcc=PCC, rtol=RTOL, atol=ATOL)
    assert passing, f"{what}: {out}"


def _assert_devices_differ(slices, what):
    """Precondition for a meaningful test: a per-device-distinct result must actually differ between devices,
    otherwise a Replicate label would be harmless and prove nothing."""
    assert len(slices) >= 2 and not torch.equal(slices[0], slices[1]), f"{what}: identical on both devices"


def _adam_reference(param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq, *, optim_cls, amsgrad):
    """One step of torch.optim.Adam / AdamW (`optim_cls`) seeded with the given moments at step STEP - 1, run in
    bfloat16 as the nightly test_moreh_adam / test_moreh_adamw goldens do, so torch's weight-decay placement and bias
    correction are the specification. The slices arrive as float32 copies of bf16 device data, so the conversion back
    is lossless; the kernels compute in bf16 too (no fp32_dest_acc_en), hence the nightly tests' tolerances.
    """

    def bf16(tensor):
        return None if tensor is None else tensor.to(torch.bfloat16)

    p = torch.nn.Parameter(bf16(param))
    p.grad = bf16(grad)
    optimizer = optim_cls([p], lr=LR, betas=BETAS, eps=EPS, weight_decay=WEIGHT_DECAY, amsgrad=amsgrad)
    state = optimizer.state[p]
    state["step"] = torch.tensor(float(STEP - 1))
    state["exp_avg"] = bf16(exp_avg)
    state["exp_avg_sq"] = bf16(exp_avg_sq)
    if amsgrad:
        state["max_exp_avg_sq"] = bf16(max_exp_avg_sq)
    optimizer.step()
    outputs = (
        p.detach(),
        state["exp_avg"],
        state["exp_avg_sq"],
        state["max_exp_avg_sq"] if amsgrad else None,
    )
    return tuple(None if t is None else t.float() for t in outputs)


def _sgd_reference(
    param,
    grad,
    momentum_buffer,
    *,
    momentum,
    weight_decay=0.0,
    dampening=0.0,
    nesterov=False,
):
    grad = grad + weight_decay * param
    if momentum != 0:
        momentum_buffer = momentum * momentum_buffer + (1 - dampening) * grad
        grad = grad + momentum * momentum_buffer if nesterov else momentum_buffer
    else:
        momentum_buffer = None
    return param - LR * grad, momentum_buffer


class _AdamInputs:
    """param / moments replicated (or sharded on `param_shard_dim`), grad sharded on dim 3 so that every device sees
    a different gradient with the parameter's per-device shape."""

    def __init__(
        self,
        mesh_device,
        amsgrad,
        *,
        param_shard_dim=None,
        preallocate=True,
        grad_label="collapsed",
        caller_label="collapsed",
    ):
        torch.manual_seed(0)
        self.mesh_device = mesh_device
        self.param_shard_dim = param_shard_dim
        self.amsgrad = amsgrad
        self.grad_label = GRAD_LABELS[grad_label]
        self.caller_label = caller_label
        num_devices = mesh_device.get_num_devices()
        local_shape = [1, 1, 32, 64]
        param_shape = list(local_shape)
        if param_shard_dim is None:
            self.param_label = ["Replicate"]
        else:
            param_shape[param_shard_dim] *= num_devices
            self.param_label = [f"Shard({param_shard_dim})"]
        if caller_label == "nd":  # one placement per mesh axis: the row axis is replicated
            self.param_label = ["Replicate"] + self.param_label
        grad_shape = list(local_shape)
        grad_shape[3] *= num_devices

        self.param = torch.randn(param_shape, dtype=torch.bfloat16)
        self.grad = torch.randn(grad_shape, dtype=torch.bfloat16)
        self.exp_avg = torch.rand(param_shape, dtype=torch.bfloat16)
        # Second moments in [0.5, 1.5): the update is exp_avg / sqrt(exp_avg_sq), and torch.rand in bf16 returns an
        # exact 0 for ~0.4% of elements, where that ratio is unbounded and every bf16 rounding in the kernel is
        # amplified without limit. The amsgrad max is drawn from the same range so that it wins on either side.
        self.exp_avg_sq = torch.rand(param_shape, dtype=torch.bfloat16) + 0.5
        self.max_exp_avg_sq = torch.rand(param_shape, dtype=torch.bfloat16) + 0.5

        self.param_in = self._place(self.param)
        self.grad_in = _place_grad(self.grad, mesh_device, grad_label)
        self.exp_avg_in = self._place(self.exp_avg)
        self.exp_avg_sq_in = self._place(self.exp_avg_sq)
        self.max_exp_avg_sq_in = self._place(self.max_exp_avg_sq) if amsgrad else None
        assert _placement_names(self.grad_in) == self.grad_label, "precondition: grad labelled Shard(3)"
        assert _placement_names(self.param_in) == self.param_label

        if preallocate:
            zeros = torch.zeros(param_shape, dtype=torch.bfloat16)
            self.param_out = self._place(zeros)
            self.exp_avg_out = self._place(zeros)
            self.exp_avg_sq_out = self._place(zeros)
            self.max_exp_avg_sq_out = self._place(zeros) if amsgrad else None
        else:
            self.param_out = self.exp_avg_out = self.exp_avg_sq_out = self.max_exp_avg_sq_out = None

    def _place(self, tensor):
        """Parameter-side tensors share one mapper, so their labels agree with each other."""
        if self.caller_label == "nd":
            column = (
                ttnn.PlacementReplicate() if self.param_shard_dim is None else ttnn.PlacementShard(self.param_shard_dim)
            )
            mapper = ttnn.create_mesh_mapper(
                self.mesh_device, ttnn.MeshMapperConfig([ttnn.PlacementReplicate(), column])
            )
            return ttnn.from_torch(
                tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh_device, mesh_mapper=mapper
            )
        if self.param_shard_dim is None:
            return _replicated(tensor, self.mesh_device)
        return _sharded(tensor, self.mesh_device, self.param_shard_dim)

    def check_numerics(self, outputs, *, optim_cls):
        """Per device: the torch.optim step on that device's slices of every input."""
        param_out, exp_avg_out, exp_avg_sq_out, max_exp_avg_sq_out = outputs
        assert (max_exp_avg_sq_out is not None) == self.amsgrad
        slices = {
            name: _device_slices(t)
            for name, t in (
                ("param", self.param_in),
                ("grad", self.grad_in),
                ("exp_avg", self.exp_avg_in),
                ("exp_avg_sq", self.exp_avg_sq_in),
            )
        }
        num_devices = len(slices["param"])
        slices["max"] = _device_slices(self.max_exp_avg_sq_in) if self.amsgrad else [None] * num_devices
        actual = [_device_slices(t) for t in (param_out, exp_avg_out, exp_avg_sq_out)]
        if self.amsgrad:
            actual.append(_device_slices(max_exp_avg_sq_out))
        _assert_devices_differ(actual[0], "param_out")
        for dev in range(num_devices):
            expected = _adam_reference(
                slices["param"][dev],
                slices["grad"][dev],
                slices["exp_avg"][dev],
                slices["exp_avg_sq"][dev],
                slices["max"][dev],
                optim_cls=optim_cls,
                amsgrad=self.amsgrad,
            )
            # zip stops at the three present outputs when amsgrad is off (expected[3] is None then).
            for name, exp, act in zip(("param", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"), expected, actual):
                _assert_close(exp, act[dev], f"device {dev} {name}")


def _run_adamw(inputs):
    return ttnn.operations.moreh.adamw(
        inputs.param_in,
        inputs.grad_in,
        inputs.exp_avg_in,
        inputs.exp_avg_sq_in,
        LR,
        BETAS[0],
        BETAS[1],
        EPS,
        WEIGHT_DECAY,
        STEP,
        inputs.amsgrad,
        max_exp_avg_sq_in=inputs.max_exp_avg_sq_in,
        param_out=inputs.param_out,
        exp_avg_out=inputs.exp_avg_out,
        exp_avg_sq_out=inputs.exp_avg_sq_out,
        max_exp_avg_sq_out=inputs.max_exp_avg_sq_out,
    )


def _run_adam(inputs):
    return ttnn.operations.moreh.adam(
        inputs.param_in,
        inputs.grad_in,
        inputs.exp_avg_in,
        inputs.exp_avg_sq_in,
        lr=LR,
        beta1=BETAS[0],
        beta2=BETAS[1],
        eps=EPS,
        weight_decay=WEIGHT_DECAY,
        step=STEP,
        amsgrad=inputs.amsgrad,
        max_exp_avg_sq_in=inputs.max_exp_avg_sq_in,
        param_out=inputs.param_out,
        exp_avg_out=inputs.exp_avg_out,
        exp_avg_sq_out=inputs.exp_avg_sq_out,
        max_exp_avg_sq_out=inputs.max_exp_avg_sq_out,
    )


# (ttnn runner, torch optimizer with the same weight-decay placement)
ADAM_OPS = [
    pytest.param(_run_adamw, torch.optim.AdamW, id="adamw"),
    pytest.param(_run_adam, torch.optim.Adam, id="adam"),
]


def _check_adam_labels(inputs, outputs, expected):
    """Every present output carries `expected`; the 4th slot exists iff amsgrad."""
    param_out, exp_avg_out, exp_avg_sq_out, max_exp_avg_sq_out = outputs
    assert (max_exp_avg_sq_out is not None) == inputs.amsgrad, "max_exp_avg_sq_out present iff amsgrad"
    _assert_label(param_out, expected, "param_out")
    _assert_label(exp_avg_out, expected, "exp_avg_out")
    _assert_label(exp_avg_sq_out, expected, "exp_avg_sq_out")
    if inputs.amsgrad:
        _assert_label(max_exp_avg_sq_out, expected, "max_exp_avg_sq_out")
    # The inputs are never relabelled either way.
    _assert_label(inputs.param_in, inputs.param_label, "param_in")
    _assert_label(inputs.grad_in, inputs.grad_label, "grad")


# --- adamw / adam -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("run, optim_cls", ADAM_OPS)
@pytest.mark.parametrize("amsgrad", [True, False], ids=["amsgrad", "no_amsgrad"])
def test_adam_preallocated_outputs_follow_the_data(mesh_device, run, optim_cls, amsgrad):
    """(a) Replicated param / moments, every *_out preallocated and replicated, and a Shard(3) grad that holds a
    different slice on every device: after the step every device holds a different parameter, so each *_out takes
    the union ['Shard(3)'] -- on the caller's own handle too, since the returned handles are the caller's tensors.
    Keeping ['Replicate'] here (what a hook that returns the caller's label unconditionally does) would make the
    flatbuffer serialiser save one device's parameter as everyone's."""
    inputs = _AdamInputs(mesh_device, amsgrad)
    outputs = run(inputs)
    _check_adam_labels(inputs, outputs, ["Shard(3)"])
    assert outputs[0].buffer_address() == inputs.param_out.buffer_address()
    _assert_label(inputs.param_out, ["Shard(3)"], "caller's param_out handle")
    inputs.check_numerics(outputs, optim_cls=optim_cls)  # asserts the per-device results differ


@pytest.mark.parametrize("run, optim_cls", ADAM_OPS)
@pytest.mark.parametrize("amsgrad", [True, False], ids=["amsgrad", "no_amsgrad"])
def test_adam_fresh_outputs_take_union(mesh_device, run, optim_cls, amsgrad):
    """(b) Same inputs, nothing preallocated: the op allocates the outputs, and they are per-device distinct
    (each device saw its own grad shard), so the union label ['Shard(3)'] is the correct, dedup-safe one.
    """
    inputs = _AdamInputs(mesh_device, amsgrad, preallocate=False)
    outputs = run(inputs)
    _check_adam_labels(inputs, outputs, ["Shard(3)"])
    assert list(outputs[0].tensor_topology().distribution_shape()) == [mesh_device.get_num_devices()]
    inputs.check_numerics(outputs, optim_cls=optim_cls)


@pytest.mark.parametrize("run, optim_cls", ADAM_OPS)
@pytest.mark.parametrize("preallocate", [True, False], ids=["preallocated", "fresh"])
@pytest.mark.parametrize("grad_label", ["collapsed", "nd"], ids=["collapsed_grad", "nd_grad"])
def test_adam_sharded_param_keeps_own_label(mesh_device, run, optim_cls, preallocate, grad_label):
    """(c) param / moments ``{2},[Shard(2)]``, grad sharded on dim 3 along the same mesh axis: the grad is sharded
    only where the parameter already is, so a preallocated *_out keeps ['Shard(2)']. With a collapsed grad label the
    union agrees (it keeps the earliest-seen shard dim). With the N-D grad label ``{1,2},[Replicate, Shard(3)]`` the
    union is that N-D label (only the inputs of maximal distribution rank contribute), which is what a fresh output
    gets and what a preallocated one used to get before the hook."""
    inputs = _AdamInputs(
        mesh_device,
        amsgrad=False,
        param_shard_dim=2,
        preallocate=preallocate,
        grad_label=grad_label,
    )
    outputs = run(inputs)
    if preallocate or grad_label == "collapsed":
        _check_adam_labels(inputs, outputs, ["Shard(2)"])
    else:
        _check_adam_labels(inputs, outputs, ["Replicate", "Shard(3)"])
    inputs.check_numerics(outputs, optim_cls=optim_cls)


@pytest.mark.parametrize("run, optim_cls", ADAM_OPS)
@pytest.mark.parametrize("param_shard_dim", [2, None], ids=["sharded_param", "replicated_param"])
def test_adam_nd_caller_label(mesh_device, run, optim_cls, param_shard_dim):
    """The caller's tensors carry one placement per mesh axis (an explicit N-D mapper) and the grad the N-D
    ``{1,2},[Replicate, Shard(3)]`` label. ``sharded_param``: param / moments / outs ``[Replicate, Shard(2)]`` are
    sharded along the same mesh axis as the grad, so the preallocated outputs keep that label (the union agrees, every
    label having the mesh's rank). ``replicated_param``: ``[Replicate, Replicate]`` against a grad that differs per
    device, so the outputs take the union ``['Replicate', 'Shard(3)']``. This is the per-axis branch of the rule for
    the caller's own label, which the collapsed-label cases above never reach."""
    inputs = _AdamInputs(
        mesh_device, amsgrad=False, param_shard_dim=param_shard_dim, grad_label="nd", caller_label="nd"
    )
    outputs = run(inputs)
    expected = ["Replicate", "Shard(2)"] if param_shard_dim == 2 else ["Replicate", "Shard(3)"]
    _check_adam_labels(inputs, outputs, expected)
    inputs.check_numerics(outputs, optim_cls=optim_cls)


@pytest.mark.parametrize("run, optim_cls", ADAM_OPS)
@pytest.mark.parametrize("param_shard_dim", [None, 2], ids=["replicated_param", "sharded_param"])
def test_adam_aliased_outputs(mesh_device, run, optim_cls, param_shard_dim):
    """tt-train's call shape (``MorehAdamW::step``): the parameter and the moments are passed as both the inputs and
    the preallocated outputs, so the op updates them in place and the hook's label lands on the caller's only
    handle. ``replicated_param`` with a per-device-different Shard(3) grad: that handle reads ['Shard(3)'] after the
    step. ``sharded_param`` (``{2},[Shard(2)]``, the tensor-parallel layout): it keeps ['Shard(2)']. Numerics are
    checked per device against torch on snapshots taken before the step, since the inputs are overwritten."""
    inputs = _AdamInputs(mesh_device, amsgrad=False, param_shard_dim=param_shard_dim, preallocate=False)
    before = {
        name: _device_slices(t)
        for name, t in (
            ("param", inputs.param_in),
            ("grad", inputs.grad_in),
            ("exp_avg", inputs.exp_avg_in),
            ("exp_avg_sq", inputs.exp_avg_sq_in),
        )
    }
    inputs.param_out, inputs.exp_avg_out, inputs.exp_avg_sq_out = (
        inputs.param_in,
        inputs.exp_avg_in,
        inputs.exp_avg_sq_in,
    )

    param_out, exp_avg_out, exp_avg_sq_out, max_exp_avg_sq_out = run(inputs)

    assert max_exp_avg_sq_out is None
    expected = ["Shard(3)"] if param_shard_dim is None else ["Shard(2)"]
    for name, t in (("param", param_out), ("exp_avg", exp_avg_out), ("exp_avg_sq", exp_avg_sq_out)):
        _assert_label(t, expected, name)
    assert param_out.buffer_address() == inputs.param_in.buffer_address()
    _assert_label(inputs.param_in, expected, "the caller's aliased param handle")
    _assert_label(inputs.exp_avg_in, expected, "the caller's aliased exp_avg handle")
    _assert_label(inputs.grad_in, ["Shard(3)"], "grad")

    actual = [_device_slices(t) for t in (param_out, exp_avg_out, exp_avg_sq_out)]
    _assert_devices_differ(actual[0], "param")
    for dev in range(len(before["param"])):
        expected_values = _adam_reference(
            before["param"][dev],
            before["grad"][dev],
            before["exp_avg"][dev],
            before["exp_avg_sq"][dev],
            None,
            optim_cls=optim_cls,
            amsgrad=False,
        )
        for name, exp, act in zip(("param", "exp_avg", "exp_avg_sq"), expected_values, actual):
            _assert_close(exp, act[dev], f"device {dev} {name}")


@pytest.mark.parametrize("run, optim_cls", ADAM_OPS)
def test_adam_topology_does_not_touch_program_cache(mesh_device, run, optim_cls):
    """Same per-device shapes and the same preallocation pattern under different labels reuse one cached program,
    and each run still reports its own label: the label is applied when the outputs are created, not by the
    program, and is not part of the program hash."""
    # Build every input before the cache window opens: from_torch with TILE_LAYOUT onto a mesh runs a device tilize
    # program, which must not be counted among the optimizer's entries.
    runs = [
        (
            _AdamInputs(mesh_device, amsgrad=True, param_shard_dim=param_shard_dim),
            expected,
        )
        for param_shard_dim, expected in (
            (None, ["Shard(3)"]),
            (2, ["Shard(2)"]),
            (None, ["Shard(3)"]),
        )
    ]
    mesh_device.enable_program_cache()
    mesh_device.clear_program_cache()
    try:
        entries = []
        for inputs, expected in runs:
            outputs = run(inputs)
            entries.append(mesh_device.num_program_cache_entries())
            _check_adam_labels(inputs, outputs, expected)
        assert entries[0] > 0
        assert entries[0] == entries[1] == entries[2], f"program cache entries changed across topologies: {entries}"
    finally:
        mesh_device.disable_and_clear_program_cache()


# --- sgd --------------------------------------------------------------------------------------------------------


class _SgdInputs:
    def __init__(
        self,
        mesh_device,
        momentum,
        *,
        preallocate,
        param_shard_dim=None,
        grad_label="collapsed",
    ):
        torch.manual_seed(1)
        num_devices = mesh_device.get_num_devices()
        param_shape = [1, 1, 32, 64]
        if param_shard_dim is None:
            self.param_label = ["Replicate"]
            place = lambda t: _replicated(t, mesh_device)  # noqa: E731
        else:
            param_shape[param_shard_dim] *= num_devices
            self.param_label = [f"Shard({param_shard_dim})"]
            place = lambda t: _sharded(t, mesh_device, param_shard_dim)  # noqa: E731
        grad_shape = [1, 1, 32, 64 * num_devices]
        self.momentum = momentum
        self.grad_label = GRAD_LABELS[grad_label]
        self.param = torch.randn(param_shape, dtype=torch.bfloat16)
        self.grad = torch.randn(grad_shape, dtype=torch.bfloat16)
        self.momentum_buffer = torch.rand(param_shape, dtype=torch.bfloat16)
        self.param_in = place(self.param)
        self.grad_in = _place_grad(self.grad, mesh_device, grad_label)
        self.momentum_buffer_in = place(self.momentum_buffer) if momentum != 0 else None
        if preallocate:
            zeros = torch.zeros(param_shape, dtype=torch.bfloat16)
            self.param_out = place(zeros)
            self.momentum_buffer_out = place(zeros) if momentum != 0 else None
        else:
            self.param_out = self.momentum_buffer_out = None

    def run(self):
        # momentum_initialized=True: the kernel folds momentum_buffer_in into the new buffer (torch's steady state).
        return ttnn.operations.moreh.sgd(
            self.param_in,
            self.grad_in,
            self.momentum_buffer_in,
            self.param_out,
            self.momentum_buffer_out,
            LR,
            self.momentum,
            0.0,
            0.0,
            False,
            momentum_initialized=self.momentum != 0,
        )

    def check(self, outputs, expected_label):
        param_out, momentum_buffer_out = outputs
        assert (momentum_buffer_out is not None) == (self.momentum != 0), "momentum_buffer_out present iff momentum"
        _assert_label(param_out, expected_label, "param_out")
        if self.momentum != 0:
            _assert_label(momentum_buffer_out, expected_label, "momentum_buffer_out")
        _assert_label(self.param_in, self.param_label, "param_in")
        _assert_label(self.grad_in, self.grad_label, "grad")

        params, grads = _device_slices(self.param_in), _device_slices(self.grad_in)
        buffers = _device_slices(self.momentum_buffer_in) if self.momentum != 0 else [None] * len(params)
        actual_params = _device_slices(param_out)
        actual_buffers = _device_slices(momentum_buffer_out) if self.momentum != 0 else None
        _assert_devices_differ(actual_params, "param_out")
        for dev in range(len(params)):
            expected_param, expected_buffer = _sgd_reference(
                params[dev], grads[dev], buffers[dev], momentum=self.momentum
            )
            _assert_close(expected_param, actual_params[dev], f"device {dev} param")
            if self.momentum != 0:
                _assert_close(
                    expected_buffer,
                    actual_buffers[dev],
                    f"device {dev} momentum_buffer",
                )


@pytest.mark.parametrize("momentum", [0.9, 0.0], ids=["momentum", "no_momentum"])
def test_sgd_preallocated_outputs_follow_the_data(mesh_device, momentum):
    """Replicated param (and momentum buffer), preallocated outputs, and a Shard(3) grad that differs per device:
    every device ends with a different parameter, so the preallocated outputs take the union ['Shard(3)'], on the
    caller's handle too. momentum=0 leaves a single present output (1-vs-2 slot count).
    """
    inputs = _SgdInputs(mesh_device, momentum, preallocate=True)
    outputs = inputs.run()
    inputs.check(outputs, ["Shard(3)"])
    assert outputs[0].buffer_address() == inputs.param_out.buffer_address()
    _assert_label(inputs.param_out, ["Shard(3)"], "caller's param_out handle")


@pytest.mark.parametrize("preallocate", [True, False], ids=["preallocated", "fresh"])
def test_sgd_sharded_param_keeps_own_label(mesh_device, preallocate):
    """param / momentum buffer ``{2},[Shard(2)]``, grad ``{1,2},[Replicate, Shard(3)]`` along the same mesh axis:
    preallocated outputs keep ['Shard(2)'] (one placement), fresh outputs take the N-D union, which is also what a
    preallocated output used to get before the hook."""
    inputs = _SgdInputs(mesh_device, 0.9, preallocate=preallocate, param_shard_dim=2, grad_label="nd")
    outputs = inputs.run()
    inputs.check(outputs, ["Shard(2)"] if preallocate else ["Replicate", "Shard(3)"])


@pytest.mark.parametrize("momentum", [0.9, 0.0], ids=["momentum", "no_momentum"])
def test_sgd_fresh_outputs_take_union(mesh_device, momentum):
    """Nothing preallocated: the op-allocated param (and momentum buffer) are per-device distinct and carry the
    union label ['Shard(3)']."""
    inputs = _SgdInputs(mesh_device, momentum, preallocate=False)
    outputs = inputs.run()
    inputs.check(outputs, ["Shard(3)"])


def test_sgd_mixed_preallocation(mesh_device):
    """Only param_out preallocated, momentum on, param ``{2},[Shard(2)]`` and an N-D grad: the preallocated param_out
    keeps ['Shard(2)'] while the op-allocated momentum buffer takes the N-D union. Exercises both branches in one
    call."""
    inputs = _SgdInputs(mesh_device, 0.9, preallocate=True, param_shard_dim=2, grad_label="nd")
    inputs.momentum_buffer_out = None
    param_out, momentum_buffer_out = inputs.run()
    _assert_label(param_out, ["Shard(2)"], "param_out")
    _assert_label(momentum_buffer_out, ["Replicate", "Shard(3)"], "momentum_buffer_out")


# --- clip_grad_norm ---------------------------------------------------------------------------------------------


def _clip_reference(grads, max_norm, norm_type):
    norms = torch.stack([torch.linalg.vector_norm(g, ord=norm_type) for g in grads])
    total_norm = torch.linalg.vector_norm(norms, ord=norm_type)
    clip_coef = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)
    return total_norm, [g * clip_coef for g in grads]


def _run_clip(mesh_device, grads, *, preallocate_total_norm, max_norm=1.0, norm_type=2.0):
    total_norm_in = (
        _replicated(torch.zeros([1, 1], dtype=torch.bfloat16), mesh_device) if preallocate_total_norm else None
    )
    total_norm = ttnn.operations.moreh.clip_grad_norm(grads, max_norm, norm_type, total_norm=total_norm_in)
    if preallocate_total_norm:
        assert total_norm.buffer_address() == total_norm_in.buffer_address(), "total_norm must be the caller's tensor"
    return total_norm, total_norm_in


def _check_clip_numerics(mesh_device, grads, grad_slices, total_norm, *, max_norm=1.0, norm_type=2.0):
    """The norm runs over each device's own slices (no cross-device reduction), so the reference is per device."""
    actual_total = _device_slices(total_norm)
    actual = [_device_slices(g) for g in grads]
    for dev in range(mesh_device.get_num_devices()):
        expected_total, expected = _clip_reference([slices[dev] for slices in grad_slices], max_norm, norm_type)
        assert expected_total > max_norm, "precondition: clipping must actually scale the gradients"
        _assert_close(
            expected_total.reshape(1),
            actual_total[dev].reshape(-1)[:1],
            f"device {dev} total_norm",
        )
        for index, (exp, act) in enumerate(zip(expected, actual)):
            _assert_close(exp, act[dev], f"device {dev} grad {index}")


@pytest.mark.parametrize(
    "preallocate_total_norm",
    [False, True],
    ids=["fresh_total_norm", "preallocated_total_norm"],
)
def test_clip_grad_norm_labels_follow_the_data(mesh_device, preallocate_total_norm):
    """Two gradients scaled in place, one replicated and one sharded on dim 3 with a different slice per device. The
    norm is computed per device, so total_norm, the clip coefficient and every scaled gradient differ per device:
    all of them take the union ['Shard(3)'], the replicated gradient and a preallocated (replicated) total_norm
    included, on the caller's handles. Keeping ['Replicate'] on them would have the serialiser save one device's
    values for all. (Step1 has no hook: its union over the grads relabels tmp_pow_sum Shard(3); a fresh total_norm
    is left to the framework's union over tmp_pow_sum, Shard(3) as well.)"""
    torch.manual_seed(2)
    num_devices = mesh_device.get_num_devices()
    g0 = torch.randn([1, 1, 32, 32], dtype=torch.bfloat16)
    g1 = torch.randn([1, 1, 32, 32 * num_devices], dtype=torch.bfloat16)
    grads = [_replicated(g0, mesh_device), _sharded(g1, mesh_device, 3)]
    grad_slices = [_device_slices(g) for g in grads]
    _assert_devices_differ(grad_slices[1], "sharded grad")

    total_norm, total_norm_in = _run_clip(mesh_device, grads, preallocate_total_norm=preallocate_total_norm)

    _assert_label(grads[0], ["Shard(3)"], "replicated grad, scaled per device")
    _assert_label(grads[1], ["Shard(3)"], "sharded grad")
    _assert_label(total_norm, ["Shard(3)"], "total_norm")
    if preallocate_total_norm:
        _assert_label(total_norm_in, ["Shard(3)"], "caller's total_norm handle")
    _assert_devices_differ(_device_slices(grads[0]), "replicated grad after a per-device scale")
    _check_clip_numerics(mesh_device, grads, grad_slices, total_norm)


@pytest.mark.parametrize(
    "preallocate_total_norm",
    [False, True],
    ids=["fresh_total_norm", "preallocated_total_norm"],
)
def test_clip_grad_norm_replicated_grads_keep_replicate(mesh_device, preallocate_total_norm):
    """Every gradient replicated: the norm is the same on every device, nothing is sharded anywhere, and every
    handle keeps ['Replicate'] (union and own agree; pinned so the compatible path of the rule is covered).
    """
    torch.manual_seed(2)
    grads = [_replicated(torch.randn([1, 1, 32, 32], dtype=torch.bfloat16), mesh_device) for _ in range(2)]
    grad_slices = [_device_slices(g) for g in grads]

    total_norm, total_norm_in = _run_clip(mesh_device, grads, preallocate_total_norm=preallocate_total_norm)

    for index, g in enumerate(grads):
        _assert_label(g, ["Replicate"], f"grad {index}")
    _assert_label(total_norm, ["Replicate"], "total_norm")
    if preallocate_total_norm:
        _assert_label(total_norm_in, ["Replicate"], "caller's total_norm handle")
    for index, g in enumerate(grads):
        slices = _device_slices(g)
        assert torch.equal(slices[0], slices[1]), f"grad {index}: replicated gradients must stay identical"
    _check_clip_numerics(mesh_device, grads, grad_slices, total_norm)


# --- dot_backward -----------------------------------------------------------------------------------------------


def _dot_backward_inputs(mesh_device, *, want_other_grad, input_grad_label, other_label):
    """output_grad and input replicated; other sharded on dim 3 (collapsed or N-D label), a different slice per
    device; the caller's input_grad buffer replicated or collapsed-sharded on dim 3; other_grad (optional) sharded.
    """
    torch.manual_seed(3)
    num_devices = mesh_device.get_num_devices()
    n = 64
    # Small integers times 1.5 are exact in bfloat16; a non-zero scalar keeps the products device-distinct.
    output_grad = torch.full([1, 1, 1, 1], 1.5, dtype=torch.bfloat16)
    input_ = torch.randint(-2, 3, [1, 1, 1, n], dtype=torch.bfloat16)
    other = torch.randint(-2, 3, [1, 1, 1, n * num_devices], dtype=torch.bfloat16)
    nan_local = torch.full([1, 1, 1, n], float("nan"), dtype=torch.bfloat16)
    nan_wide = torch.full([1, 1, 1, n * num_devices], float("nan"), dtype=torch.bfloat16)
    tensors = {
        "output_grad": _replicated(output_grad, mesh_device),
        "input": _replicated(input_, mesh_device),
        "other": _place_grad(other, mesh_device, other_label),
        "input_grad": (
            _replicated(nan_local, mesh_device)
            if input_grad_label == "replicate"
            else _sharded(nan_wide, mesh_device, 3)
        ),
        "other_grad": _sharded(nan_wide, mesh_device, 3) if want_other_grad else None,
    }
    return output_grad, input_, tensors


def _check_dot_backward_numerics(mesh_device, output_grad, input_, tensors, input_grad_out, other_grad_out):
    # input_grad = output_grad * other (per device: that device's shard of other); other_grad = output_grad * input.
    others = _device_slices(tensors["other"])
    actual_input_grad = _device_slices(input_grad_out)
    _assert_devices_differ(actual_input_grad, "input_grad")
    actual_other_grad = _device_slices(other_grad_out) if other_grad_out is not None else None
    for dev in range(mesh_device.get_num_devices()):
        _assert_close(
            output_grad.float() * others[dev],
            actual_input_grad[dev],
            f"device {dev} input_grad",
        )
        if other_grad_out is not None:
            _assert_close(
                output_grad.float() * input_.float(),
                actual_other_grad[dev],
                f"device {dev} other_grad",
            )


@pytest.mark.parametrize("want_other_grad", [True, False], ids=["both_grads", "input_grad_only"])
def test_dot_backward_input_grad_follows_the_data(mesh_device, want_other_grad):
    """moreh_dot_backward only writes caller-provided grads. input replicated, other sharded on dim 3 with a
    different slice per device: input_grad = output_grad * other differs per device, so the caller's replicated
    input_grad takes the union ['Shard(3)'] (a kept ['Replicate'] would have the serialiser save one device's
    values). other_grad = output_grad * input is produced from two replicated tensors, so the caller's Shard(3)
    other_grad keeps its label: the rule compares labels, and nothing here is sharded where other_grad is
    replicated. Requesting one grad only exercises the 1-vs-2 slot count."""
    output_grad, input_, t = _dot_backward_inputs(
        mesh_device,
        want_other_grad=want_other_grad,
        input_grad_label="replicate",
        other_label="collapsed",
    )
    input_grad_out, other_grad_out = ttnn.operations.moreh.dot_backward(
        t["output_grad"],
        t["input"],
        t["other"],
        input_grad=t["input_grad"],
        other_grad=t["other_grad"],
    )

    _assert_label(input_grad_out, ["Shard(3)"], "input_grad")
    _assert_label(t["input_grad"], ["Shard(3)"], "caller's input_grad handle")
    assert input_grad_out.buffer_address() == t["input_grad"].buffer_address()
    assert (other_grad_out is not None) == want_other_grad
    if want_other_grad:
        _assert_label(other_grad_out, ["Shard(3)"], "other_grad")
        _assert_label(t["other_grad"], ["Shard(3)"], "caller's other_grad handle")
    _check_dot_backward_numerics(mesh_device, output_grad, input_, t, input_grad_out, other_grad_out)


def test_dot_backward_sharded_input_grad_keeps_own_label(mesh_device):
    """The caller's input_grad is ``{2},[Shard(3)]`` and other carries the N-D ``{1,2},[Replicate, Shard(3)]``
    label along the same mesh axis: other is sharded only where input_grad already is, so input_grad keeps its
    collapsed label. The union -- what a fresh output gets, and what input_grad got before the hook -- is the N-D
    label, since only the inputs of maximal distribution rank contribute."""
    output_grad, input_, t = _dot_backward_inputs(
        mesh_device, want_other_grad=True, input_grad_label="shard", other_label="nd"
    )
    _assert_label(t["other"], ["Replicate", "Shard(3)"], "precondition: N-D other")
    input_grad_out, other_grad_out = ttnn.operations.moreh.dot_backward(
        t["output_grad"],
        t["input"],
        t["other"],
        input_grad=t["input_grad"],
        other_grad=t["other_grad"],
    )

    _assert_label(input_grad_out, ["Shard(3)"], "input_grad")
    _assert_label(t["input_grad"], ["Shard(3)"], "caller's input_grad handle")
    assert input_grad_out.tensor_topology() != t["other"].tensor_topology(), "the union label was not applied"
    _assert_label(other_grad_out, ["Shard(3)"], "other_grad")
    _check_dot_backward_numerics(mesh_device, output_grad, input_, t, input_grad_out, other_grad_out)
