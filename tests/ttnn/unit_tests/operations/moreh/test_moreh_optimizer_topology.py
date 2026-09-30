# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Mesh-topology labels of the moreh optimizer, clip_grad_norm and dot_backward outputs.

The device-operation framework relabels every tensor an op returns with the union of its inputs' placements. For
these ops the returned tensors are frequently the caller's own: moreh_adamw / moreh_adam / moreh_sgd write into a
preallocated ``*_out`` (tt-train's MorehAdamW hands in the parameter itself), moreh_clip_grad_norm scales its
``inputs`` in place, moreh_dot_backward only ever writes caller-provided grads. Relabelling those breaks the caller:
a gradient labelled ``Shard(3)`` turned a replicated ``param_out`` into ``Shard(3)``, and the serialiser / mesh
composers then treat one device's copy as a shard of a larger tensor.

Contract pinned here:
  * a preallocated (caller-owned) output keeps the label it arrived with;
  * an output the op allocates itself takes the union of all inputs (it is per-device distinct whenever any input is);
  * absent optional outputs (``amsgrad=False``, ``momentum=0``, one grad of dot_backward) get no entry, so the
    3-vs-4 / 1-vs-2 slot counts are exercised;
  * the topology label does not take part in the program hash, so a cached program is reused across labels.

Negative controls: every ``*_keep_own_topology`` test below failed before this change with the preallocated output
reporting ``['Shard(3)']`` (the union) instead of its own label. The ``*_fresh_outputs_take_union`` tests document
the unchanged union behaviour for op-allocated outputs.
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
    return ttnn.from_torch(
        tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=dim),
    )


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
    is lossless; the kernels compute in bf16 too (no fp32_dest_acc_en), hence the nightly tests' tolerances."""

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
    outputs = (p.detach(), state["exp_avg"], state["exp_avg_sq"], state["max_exp_avg_sq"] if amsgrad else None)
    return tuple(None if t is None else t.float() for t in outputs)


def _sgd_reference(param, grad, momentum_buffer, *, momentum, weight_decay=0.0, dampening=0.0, nesterov=False):
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

    def __init__(self, mesh_device, amsgrad, *, param_shard_dim=None, preallocate=True):
        torch.manual_seed(0)
        self.mesh_device = mesh_device
        self.param_shard_dim = param_shard_dim
        self.amsgrad = amsgrad
        num_devices = mesh_device.get_num_devices()
        local_shape = [1, 1, 32, 64]
        param_shape = list(local_shape)
        if param_shard_dim is None:
            self.param_label = ["Replicate"]
        else:
            param_shape[param_shard_dim] *= num_devices
            self.param_label = [f"Shard({param_shard_dim})"]
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
        self.grad_in = _sharded(self.grad, mesh_device, 3)
        self.exp_avg_in = self._place(self.exp_avg)
        self.exp_avg_sq_in = self._place(self.exp_avg_sq)
        self.max_exp_avg_sq_in = self._place(self.max_exp_avg_sq) if amsgrad else None
        assert _placement_names(self.grad_in) == ["Shard(3)"], "precondition: grad labelled Shard(3)"
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
    _assert_label(inputs.grad_in, ["Shard(3)"], "grad")


# --- adamw / adam -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("run, optim_cls", ADAM_OPS)
@pytest.mark.parametrize("amsgrad", [True, False], ids=["amsgrad", "no_amsgrad"])
def test_adam_preallocated_outputs_keep_own_topology(mesh_device, run, optim_cls, amsgrad):
    """(a) Replicated param / moments, Shard(3) grad, every *_out preallocated and replicated: each *_out keeps
    Replicate. Before this change every one of them came back ['Shard(3)'] (the union of the inputs)."""
    inputs = _AdamInputs(mesh_device, amsgrad)
    outputs = run(inputs)
    _check_adam_labels(inputs, outputs, ["Replicate"])
    # The returned handles are the caller's tensors, so the caller's own handles report the same label.
    assert outputs[0].buffer_address() == inputs.param_out.buffer_address()
    _assert_label(inputs.param_out, ["Replicate"], "caller's param_out handle")
    inputs.check_numerics(outputs, optim_cls=optim_cls)


@pytest.mark.parametrize("run, optim_cls", ADAM_OPS)
@pytest.mark.parametrize("amsgrad", [True, False], ids=["amsgrad", "no_amsgrad"])
def test_adam_fresh_outputs_take_union(mesh_device, run, optim_cls, amsgrad):
    """(b) Same inputs, nothing preallocated: the op allocates the outputs, and they are per-device distinct
    (each device saw its own grad shard), so the union label ['Shard(3)'] is the correct, dedup-safe one."""
    inputs = _AdamInputs(mesh_device, amsgrad, preallocate=False)
    outputs = run(inputs)
    _check_adam_labels(inputs, outputs, ["Shard(3)"])
    assert list(outputs[0].tensor_topology().distribution_shape()) == [mesh_device.get_num_devices()]
    inputs.check_numerics(outputs, optim_cls=optim_cls)


@pytest.mark.parametrize("run, optim_cls", ADAM_OPS)
@pytest.mark.parametrize("preallocate", [True, False], ids=["preallocated", "fresh"])
def test_adam_sharded_param_stays_sharded(mesh_device, run, optim_cls, preallocate):
    """(c) param / moments Shard(2), grad Shard(3): the union keeps the earliest-seen shard dim, so own and union
    agree on ['Shard(2)'] whether or not the outputs are preallocated (the case that was already correct)."""
    inputs = _AdamInputs(mesh_device, amsgrad=False, param_shard_dim=2, preallocate=preallocate)
    outputs = run(inputs)
    _check_adam_labels(inputs, outputs, ["Shard(2)"])
    inputs.check_numerics(outputs, optim_cls=optim_cls)


@pytest.mark.parametrize("run, optim_cls", ADAM_OPS)
def test_adam_topology_does_not_touch_program_cache(mesh_device, run, optim_cls):
    """Same per-device shapes and the same preallocation pattern under different labels reuse one cached program,
    and each run still reports its own label: the label is applied when the outputs are created, not by the
    program, and is not part of the program hash."""
    # Build every input before the cache window opens: from_torch with TILE_LAYOUT onto a mesh runs a device tilize
    # program, which must not be counted among the optimizer's entries.
    runs = [
        (_AdamInputs(mesh_device, amsgrad=True, param_shard_dim=param_shard_dim), expected)
        for param_shard_dim, expected in ((None, ["Replicate"]), (2, ["Shard(2)"]), (None, ["Replicate"]))
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
    def __init__(self, mesh_device, momentum, *, preallocate):
        torch.manual_seed(1)
        num_devices = mesh_device.get_num_devices()
        param_shape = [1, 1, 32, 64]
        grad_shape = [1, 1, 32, 64 * num_devices]
        self.momentum = momentum
        self.param = torch.randn(param_shape, dtype=torch.bfloat16)
        self.grad = torch.randn(grad_shape, dtype=torch.bfloat16)
        self.momentum_buffer = torch.rand(param_shape, dtype=torch.bfloat16)
        self.param_in = _replicated(self.param, mesh_device)
        self.grad_in = _sharded(self.grad, mesh_device, 3)
        self.momentum_buffer_in = _replicated(self.momentum_buffer, mesh_device) if momentum != 0 else None
        if preallocate:
            zeros = torch.zeros(param_shape, dtype=torch.bfloat16)
            self.param_out = _replicated(zeros, mesh_device)
            self.momentum_buffer_out = _replicated(zeros, mesh_device) if momentum != 0 else None
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
        _assert_label(self.param_in, ["Replicate"], "param_in")
        _assert_label(self.grad_in, ["Shard(3)"], "grad")

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
                _assert_close(expected_buffer, actual_buffers[dev], f"device {dev} momentum_buffer")


@pytest.mark.parametrize("momentum", [0.9, 0.0], ids=["momentum", "no_momentum"])
def test_sgd_preallocated_outputs_keep_own_topology(mesh_device, momentum):
    """Replicated param (and momentum buffer), Shard(3) grad, preallocated outputs keep Replicate. Before this
    change they came back ['Shard(3)']. momentum=0 leaves a single present output (1-vs-2 slot count)."""
    inputs = _SgdInputs(mesh_device, momentum, preallocate=True)
    outputs = inputs.run()
    inputs.check(outputs, ["Replicate"])
    assert outputs[0].buffer_address() == inputs.param_out.buffer_address()
    _assert_label(inputs.param_out, ["Replicate"], "caller's param_out handle")


@pytest.mark.parametrize("momentum", [0.9, 0.0], ids=["momentum", "no_momentum"])
def test_sgd_fresh_outputs_take_union(mesh_device, momentum):
    """Nothing preallocated: the op-allocated param (and momentum buffer) are per-device distinct and carry the
    union label ['Shard(3)']."""
    inputs = _SgdInputs(mesh_device, momentum, preallocate=False)
    outputs = inputs.run()
    inputs.check(outputs, ["Shard(3)"])


def test_sgd_mixed_preallocation(mesh_device):
    """Only param_out preallocated with momentum on: param_out keeps Replicate, the op-allocated momentum buffer
    takes the union. Exercises both branches in one call."""
    inputs = _SgdInputs(mesh_device, 0.9, preallocate=True)
    inputs.momentum_buffer_out = None
    param_out, momentum_buffer_out = inputs.run()
    _assert_label(param_out, ["Replicate"], "param_out")
    _assert_label(momentum_buffer_out, ["Shard(3)"], "momentum_buffer_out")


# --- clip_grad_norm ---------------------------------------------------------------------------------------------


def _clip_reference(grads, max_norm, norm_type):
    norms = torch.stack([torch.linalg.vector_norm(g, ord=norm_type) for g in grads])
    total_norm = torch.linalg.vector_norm(norms, ord=norm_type)
    clip_coef = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)
    return total_norm, [g * clip_coef for g in grads]


@pytest.mark.parametrize("preallocate_total_norm", [False, True], ids=["fresh_total_norm", "preallocated_total_norm"])
def test_clip_grad_norm_inputs_keep_own_topology(mesh_device, preallocate_total_norm):
    """Two gradients scaled in place, one replicated and one sharded on dim 3: each keeps its own label. Before
    this change step3 returned the inputs relabelled with their union, so the replicated gradient came back
    ['Shard(3)']. A preallocated total_norm likewise keeps its label (it is the caller's tensor)."""
    torch.manual_seed(2)
    num_devices = mesh_device.get_num_devices()
    max_norm, norm_type = 1.0, 2.0
    g0 = torch.randn([1, 1, 32, 32], dtype=torch.bfloat16)
    g1 = torch.randn([1, 1, 32, 32 * num_devices], dtype=torch.bfloat16)
    grads = [_replicated(g0, mesh_device), _sharded(g1, mesh_device, 3)]
    g0_slices, g1_slices = _device_slices(grads[0]), _device_slices(grads[1])
    total_norm_in = (
        _replicated(torch.zeros([1, 1], dtype=torch.bfloat16), mesh_device) if preallocate_total_norm else None
    )

    total_norm = ttnn.operations.moreh.clip_grad_norm(grads, max_norm, norm_type, total_norm=total_norm_in)

    _assert_label(grads[0], ["Replicate"], "replicated grad")
    _assert_label(grads[1], ["Shard(3)"], "sharded grad")
    if preallocate_total_norm:
        _assert_label(total_norm, ["Replicate"], "preallocated total_norm")
        assert total_norm.buffer_address() == total_norm_in.buffer_address()
        _assert_label(total_norm_in, ["Replicate"], "caller's total_norm handle")
    else:
        # A fresh total_norm takes the union: step1 has no hook, so its union over {Replicate grad, Shard(3) grad,
        # tmp_pow_sum} relabels tmp_pow_sum to Shard(3); step2 returns {} for a fresh output, and the framework's
        # union over its only tensor_arg (tmp_pow_sum) is then Shard(3) too.
        _assert_label(total_norm, ["Shard(3)"], "fresh total_norm")

    # The norm runs over each device's own slices (no cross-device reduction), so the reference is per device.
    actual_total = _device_slices(total_norm)
    actual_g0, actual_g1 = _device_slices(grads[0]), _device_slices(grads[1])
    for dev in range(num_devices):
        expected_total, (expected_g0, expected_g1) = _clip_reference(
            [g0_slices[dev], g1_slices[dev]], max_norm, norm_type
        )
        assert expected_total > max_norm, "precondition: clipping must actually scale the gradients"
        _assert_close(expected_total.reshape(1), actual_total[dev].reshape(-1)[:1], f"device {dev} total_norm")
        _assert_close(expected_g0, actual_g0[dev], f"device {dev} grad 0")
        _assert_close(expected_g1, actual_g1[dev], f"device {dev} grad 1")


# --- dot_backward -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("want_other_grad", [True, False], ids=["both_grads", "input_grad_only"])
def test_dot_backward_preallocated_grads_keep_own_topology(mesh_device, want_other_grad):
    """moreh_dot_backward only writes caller-provided grads. input replicated, other sharded on dim 3: the
    replicated input_grad keeps Replicate (before this change: ['Shard(3)'], the union) and the sharded other_grad
    keeps Shard(3). other_grad = output_grad * input is identical on every device, yet it stays Shard(3): under the
    caller-owned contract the label follows the caller's declaration, not the bytes. Requesting one grad only
    exercises the 1-vs-2 slot count."""
    torch.manual_seed(3)
    num_devices = mesh_device.get_num_devices()
    n = 64
    # Small integers times 1.5 are exact in bfloat16; a non-zero scalar keeps the products device-distinct.
    output_grad = torch.full([1, 1, 1, 1], 1.5, dtype=torch.bfloat16)
    input_ = torch.randint(-2, 3, [1, 1, 1, n], dtype=torch.bfloat16)
    other = torch.randint(-2, 3, [1, 1, 1, n * num_devices], dtype=torch.bfloat16)

    output_grad_tt = _replicated(output_grad, mesh_device)
    input_tt = _replicated(input_, mesh_device)
    other_tt = _sharded(other, mesh_device, 3)
    input_grad_tt = _replicated(torch.full([1, 1, 1, n], float("nan"), dtype=torch.bfloat16), mesh_device)
    other_grad_tt = (
        _sharded(torch.full([1, 1, 1, n * num_devices], float("nan"), dtype=torch.bfloat16), mesh_device, 3)
        if want_other_grad
        else None
    )

    input_grad_out, other_grad_out = ttnn.operations.moreh.dot_backward(
        output_grad_tt, input_tt, other_tt, input_grad=input_grad_tt, other_grad=other_grad_tt
    )

    _assert_label(input_grad_out, ["Replicate"], "input_grad")
    _assert_label(input_grad_tt, ["Replicate"], "caller's input_grad handle")
    assert input_grad_out.buffer_address() == input_grad_tt.buffer_address()
    assert (other_grad_out is not None) == want_other_grad
    if want_other_grad:
        _assert_label(other_grad_out, ["Shard(3)"], "other_grad")
        _assert_label(other_grad_tt, ["Shard(3)"], "caller's other_grad handle")

    # input_grad = output_grad * other (per device: that device's shard of other); other_grad = output_grad * input.
    others = _device_slices(other_tt)
    actual_input_grad = _device_slices(input_grad_out)
    actual_other_grad = _device_slices(other_grad_out) if want_other_grad else None
    _assert_devices_differ(actual_input_grad, "input_grad")
    for dev in range(num_devices):
        _assert_close(output_grad.float() * others[dev], actual_input_grad[dev], f"device {dev} input_grad")
        if want_other_grad:
            _assert_close(output_grad.float() * input_.float(), actual_other_grad[dev], f"device {dev} other_grad")
