# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CPU ownership/capture contracts using the actual Tracer and Euler source.

The fake records capture commands without executing them, as native traces do.
This tests control flow and BF16 operation order, not native kernels/allocators.
"""

import __future__

import ast
import math
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import overload
from unittest.mock import patch

import torch

ROOT = Path(__file__).parents[2]
PIPELINE = ROOT / "pipelines/ltx/pipeline_ltx_distilled.py"


def definitions(path, names, namespace):
    nodes = [
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names
    ]
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec", flags=__future__.annotations.compiler_flag),
        namespace,
    )


class Tensor:
    def __init__(self, value, device):
        self.value, self._device = value.clone(), device
        self.layout = "TILE"

    @property
    def dtype(self):
        return self.value.dtype

    @property
    def shape(self):
        return self.value.shape

    def device(self):
        return self._device

    def buffer_address(self):
        return self.value.data_ptr()

    def memory_config(self):
        return "DRAM_INTERLEAVED"


class NativeTraceFake:
    Tensor = Tensor
    bfloat16 = torch.bfloat16

    def __init__(self):
        self.device = SimpleNamespace(id=lambda: 7)
        self.capture = None
        self.traces = {}
        self.executions = 0
        self.blocking = []
        self.snapshots = []
        self.watch = ()
        self.fail_execute = False

    def clone(self, value):
        assert self.capture is None
        return Tensor(value.value, value.device())

    def operation(self, op):
        if self.capture is None:
            op()
        else:
            self.traces[self.capture].append(op)

    def typecast(self, value, dtype):
        if value.dtype == dtype:
            return value
        result = Tensor(torch.empty(value.shape, dtype=dtype), value.device())
        self.operation(lambda: result.value.copy_(value.value.to(dtype)))
        return result

    def multiply_(self, value, scale):
        self.operation(lambda: value.value.mul_(scale.value if isinstance(scale, Tensor) else scale))
        return value

    def add_(self, value, addition):
        self.operation(lambda: value.value.add_(addition.value))
        return value

    def binary(self, a, b, operation):
        result = Tensor(torch.empty_like(a.value), a.device())
        self.operation(lambda: result.value.copy_(operation(a.value, b.value if isinstance(b, Tensor) else b)))
        return result

    def multiply(self, a, b):
        return self.binary(a, b, torch.mul)

    def subtract(self, a, b):
        return self.binary(a, b, torch.sub)

    def begin_trace_capture(self, device, *, cq_id):
        assert device == self.device and self.capture is None
        self.capture = len(self.traces) + 1
        self.traces[self.capture] = []
        self.snapshots.append(tuple(t.value.clone() for t in self.watch))
        return self.capture

    def end_trace_capture(self, device, trace_id, *, cq_id):
        assert trace_id == self.capture
        self.snapshots.append(tuple(t.value.clone() for t in self.watch))
        self.capture = None

    def execute_trace(self, device, trace_id, *, cq_id, blocking):
        if self.fail_execute:
            raise RuntimeError("synthetic execution failure")
        self.executions += 1
        self.blocking.append(blocking)
        for operation in self.traces[trace_id]:
            operation()

    def release_trace(self, device, trace_id):
        del self.traces[trace_id]


def load(fake):
    ns = dict(
        ttnn=fake,
        math=math,
        os=os,
        overload=overload,
        logger=SimpleNamespace(debug=lambda *a: None, warning=lambda *a: None),
        _kernel_prewarm_capturing=False,
    )
    ns["_TRACER_VALID_INPUT_TYPES"] = (Tensor, int, float, str, bool, type(None))
    definitions(ROOT / "utils/tracing.py", {"Tracer", "_verify_value", "_tree_map", "_clone_tensor"}, ns)
    definitions(ROOT / "utils/ltx_euler.py", {"euler_tail", "EulerTail"}, ns)
    tree = ast.parse(PIPELINE.read_text())
    loop = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "step_idx"
    )
    first = next(
        i
        for i, n in enumerate(loop.body)
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "dt"
    )
    last = next(
        i
        for i in range(first, len(loop.body))
        if isinstance(loop.body[i], ast.If)
        and isinstance(loop.body[i].test, ast.Name)
        and loop.body[i].test.id == "ancestral"
        and loop.body[i].orelse
        and isinstance(loop.body[i].orelse[0], ast.If)
        and isinstance(loop.body[i].orelse[0].test, ast.Name)
        and loop.body[i].orelse[0].test.id == "trace_euler"
    )
    block = compile(ast.Module(body=loop.body[first : last + 1], type_ignores=[]), str(PIPELINE), "exec")
    return ns, block


class EulerTraceContract(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.fake = NativeTraceFake()
        self.ns, self.block = load(self.fake)

    def state(self, tokens=65, real=63):
        make = lambda value: Tensor(value, self.fake.device)
        mask = torch.ones((1, 1, tokens, 1), dtype=torch.bfloat16)
        mask[..., real:, :] = 0
        return SimpleNamespace(
            _euler_tail=None,
            tt_video_lat=make(torch.zeros((1, 1, tokens, 32), dtype=torch.bfloat16)),
            tt_audio_lat=make(torch.zeros((1, 1, 32, 32), dtype=torch.bfloat16)),
            tt_video_pad_mask=make(mask),
            tt_audio_pad_mask=make(torch.cat((torch.ones(1, 1, 29, 1), torch.zeros(1, 1, 3, 1)), 2).bfloat16()),
        )

    def step(
        self,
        state,
        velocities,
        sigma,
        sigma_next,
        *,
        enabled=True,
        traced=True,
        image_cond=False,
        captured=True,
        step_sync=False,
    ):
        transformer = object()
        ns = dict(self.ns)
        ns.update(
            self=SimpleNamespace(
                _trace_euler_tail=enabled,
                transformer=transformer,
                mesh_device=self.fake.device,
                _post_process_latent_tt=lambda x, m, c: x,
            ),
            state=state,
            v_out=velocities[0],
            a_out=velocities[1],
            sigma=sigma,
            sigma_next=sigma_next,
            ancestral=False,
            image_cond=image_cond,
            traced=traced,
            trace_key="stage",
            step_sync=step_sync,
            tt_i2v_mask=None,
            tt_i2v_clean=None,
            LTXTransformerModel=SimpleNamespace(
                inner_step=SimpleNamespace(
                    _tracers_keyed={transformer: {"stage": SimpleNamespace(trace_captured=captured)}}
                )
            ),
        )
        exec(self.block, ns)
        return ns

    def assert_bits(self, a, b):
        self.assertTrue(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))

    def test_all_eleven_steps_changed_velocity_aba_and_padding(self):
        tree = ast.parse(PIPELINE.read_text())
        schedules = [
            ast.literal_eval(n.value)
            for n in tree.body
            if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id in ("_DEFAULT_S1_SIGMAS", "_DEFAULT_S2_SIGMAS")
        ]
        self.assertEqual(sum(len(s) - 1 for s in schedules), 11)
        for dtype in (torch.float32, torch.bfloat16):
            for real in (63, 65):
                for schedule in schedules:
                    with self.subTest(dtype=dtype, real=real, schedule=schedule):
                        base, candidate = self.state(real=real), self.state(real=real)
                        shapes = [base.tt_video_lat.shape, base.tt_audio_lat.shape]
                        velocities = [Tensor(torch.zeros(s, dtype=dtype), self.fake.device) for s in shapes]
                        base_velocities = [Tensor(torch.zeros(s, dtype=dtype), self.fake.device) for s in shapes]
                        torch.manual_seed(100)
                        initial = [torch.randn(s).bfloat16() for s in shapes]
                        saved_a = None
                        captures = None
                        executions = self.fake.executions
                        for prompt in (11, 29, 11):
                            for b, c, value in zip(
                                (base.tt_video_lat, base.tt_audio_lat),
                                (candidate.tt_video_lat, candidate.tt_audio_lat),
                                initial,
                            ):
                                b.value.copy_(value)
                                c.value.copy_(value)
                            sigmas = torch.tensor(schedule, dtype=torch.float32).tolist()
                            for index, (sigma, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
                                torch.manual_seed(prompt + index)
                                for b, c in zip(base_velocities, velocities):
                                    value = torch.randn(c.shape).to(dtype)
                                    b.value.copy_(value)
                                    c.value.copy_(value)
                                self.step(base, base_velocities, sigma, sigma_next, enabled=False)
                                self.step(candidate, velocities, sigma, sigma_next)
                                for b, c in (
                                    (base.tt_video_lat, candidate.tt_video_lat),
                                    (base.tt_audio_lat, candidate.tt_audio_lat),
                                ):
                                    self.assert_bits(b.value, c.value)
                                self.assertEqual(candidate.tt_video_lat.value[..., real:, :].count_nonzero(), 0)
                                self.assertEqual(candidate.tt_audio_lat.value[..., 29:, :].count_nonzero(), 0)
                            if saved_a is None:
                                saved_a = candidate.tt_video_lat.value.clone()
                                captures = len(candidate._euler_tail._tracers)
                            elif prompt == 11:
                                self.assert_bits(saved_a, candidate.tt_video_lat.value)
                            else:
                                self.assertFalse(torch.equal(saved_a, candidate.tt_video_lat.value))
                            self.assertEqual(len(candidate._euler_tail._tracers), captures)
                        self.assertEqual(self.fake.executions - executions, 3 * (len(schedule) - 1))
                        candidate._euler_tail.release()

    def test_first_capture_leaves_real_inputs_unchanged_until_one_execute(self):
        state = self.state()
        velocity = [
            Tensor(torch.full(t.shape, 0.5), self.fake.device) for t in (state.tt_video_lat, state.tt_audio_lat)
        ]
        self.fake.watch = (
            state.tt_video_lat,
            state.tt_audio_lat,
            *velocity,
            state.tt_video_pad_mask,
            state.tt_audio_pad_mask,
        )
        before = [t.value.clone() for t in self.fake.watch]
        self.step(state, velocity, 1.0, 0.5)
        self.assertEqual(self.fake.executions, 1)
        for snapshot in self.fake.snapshots:
            for expected, actual in zip(before, snapshot):
                self.assert_bits(expected, actual)
        self.assertEqual(state.tt_video_lat.value[0, 0, 0, 0], -0.25)
        self.assertEqual(self.ns["Tracer"]._traces_live[7], 1)
        self.assertIsNone(next(iter(state._euler_tail._tracers.values()))._outputs)
        self.assertEqual(len(next(iter(self.fake.traces.values()))), 10)
        state._euler_tail.release()
        self.assertEqual(self.ns["Tracer"]._traces_live[7], 0)

    def test_tail_replay_blocks_only_under_step_sync(self):
        for step_sync in (False, True):
            self.fake.blocking.clear()
            state = self.state()
            velocity = [Tensor(torch.ones(t.shape), self.fake.device) for t in (state.tt_video_lat, state.tt_audio_lat)]
            self.step(state, velocity, 1.0, 0.5, step_sync=step_sync)
            self.step(state, velocity, 1.0, 0.5, step_sync=step_sync)
            self.assertEqual(self.fake.blocking, [step_sync, step_sync])
            state._euler_tail.release()

    def test_routes_without_real_unconditioned_producer_never_own_tail(self):
        for change in (dict(enabled=False), dict(traced=False), dict(image_cond=True), dict(captured=False)):
            state = self.state()
            velocity = [Tensor(torch.ones(t.shape), self.fake.device) for t in (state.tt_video_lat, state.tt_audio_lat)]
            self.step(state, velocity, 1.0, 0.5, **change)
            self.assertIsNone(state._euler_tail)
        with patch.dict(os.environ, {"LTX_DEBUG_STATS": "1"}):
            state = self.state()
            velocity = [Tensor(torch.ones(t.shape), self.fake.device) for t in (state.tt_video_lat, state.tt_audio_lat)]
            self.step(state, velocity, 1.0, 0.5)
            self.assertIsNone(state._euler_tail)
        self.assertFalse(self.fake.traces)

    def test_rebinding_fails_before_execution_and_release_requires_new_owner(self):
        state = self.state()
        velocity = [Tensor(torch.ones(t.shape), self.fake.device) for t in (state.tt_video_lat, state.tt_audio_lat)]
        self.step(state, velocity, 1.0, 0.5)
        owner = state._euler_tail
        velocity[0] = Tensor(velocity[0].value, self.fake.device)
        with self.assertRaisesRegex(AssertionError, "inputs changed"):
            self.step(state, velocity, 1.0, 0.5)
        self.assertEqual(self.fake.executions, 1)
        owner.release()
        self.assertFalse(self.fake.traces)
        with self.assertRaisesRegex(AssertionError, "recreate"):
            self.step(state, velocity, 1.0, 0.5)
        state._euler_tail = None
        self.step(state, velocity, 1.0, 0.5)
        self.assertEqual(self.fake.executions, 2)

    def test_partial_execution_failure_requires_cleanup(self):
        state = self.state()
        velocity = [Tensor(torch.ones(t.shape), self.fake.device) for t in (state.tt_video_lat, state.tt_audio_lat)]
        self.fake.fail_execute = True
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            self.step(state, velocity, 1.0, 0.5)
        self.fake.fail_execute = False
        with self.assertRaisesRegex(AssertionError, "failure"):
            self.step(state, velocity, 1.0, 0.5)
        state._euler_tail.release()
        self.assertFalse(self.fake.traces)

    def test_constructor_policy_is_default_off_and_static_only(self):
        path = ROOT / "pipelines/ltx/pipeline_ltx.py"
        tree = ast.parse(path.read_text())
        assignment = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Attribute)
            and n.targets[0].attr == "_trace_euler_tail"
        )
        code = compile(ast.Module(body=[assignment], type_ignores=[]), str(path), "exec")
        for enabled in ("0", "1"):
            for dynamic in (False, True):
                instance = SimpleNamespace()
                with patch.dict(os.environ, {"LTX_EULER_TAIL_TRACE": enabled}):
                    exec(code, dict(self=instance, os=os, dynamic_load=dynamic))
                self.assertEqual(instance._trace_euler_tail, enabled == "1" and not dynamic)

    def test_pipeline_releases_consumers_before_producer_and_buffers(self):
        path = ROOT / "pipelines/ltx/pipeline_ltx.py"
        tree = ast.parse(path.read_text())
        method = next(
            n
            for c in tree.body
            if isinstance(c, ast.ClassDef) and c.name == "LTXPipeline"
            for n in c.body
            if isinstance(n, ast.FunctionDef) and n.name == "release_traces"
        )
        order = []
        transformer = object()
        states = {"s1": SimpleNamespace(_euler_tail=SimpleNamespace(release=lambda: order.append("tail")))}
        ns = dict(
            LTXTransformerModel=SimpleNamespace(
                inner_step=SimpleNamespace(
                    _tracers_keyed={
                        transformer: {"s1": SimpleNamespace(release_trace=lambda: order.append("producer"))}
                    }
                )
            ),
            StateTensor=lambda: order.append("prompt_clear"),
        )
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), ns)
        pipeline = SimpleNamespace(
            _trace_state=states,
            transformer=transformer,
            tt_vocoder_with_bwe=None,
            tt_mel_decoder=None,
            vae_decoder=None,
        )
        ns["release_traces"](pipeline)
        self.assertEqual(order, ["tail", "producer", "prompt_clear", "prompt_clear"])
        self.assertFalse(states)


if __name__ == "__main__":
    unittest.main()
