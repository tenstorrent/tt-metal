# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CPU control/lifetime contracts using production adapter, pipeline and Tracer methods.

Run directly to avoid device-aware repository conftest imports. Native calls and
checkpoint shells are fakes; these tests establish no device or quality acceptance.
"""

import ast
import functools
import inspect
import json
import os
import unittest
import weakref
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple, overload
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
AUDIO = ROOT / "models/audio_vae/audio_decoder_ltx.py"
PIPELINE = ROOT / "pipelines/ltx/pipeline_ltx.py"


def definitions(path, names, namespace=None, class_name=None):
    tree = ast.parse(path.read_text())
    nodes = tree.body
    if class_name:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name).body
    selected = [n for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    ns = {"cap_torch_threads_to_half_the_cpus": lambda: (lambda fn: fn), **(namespace or {})}
    source = "from __future__ import annotations\n" + ast.unparse(ast.Module(selected, type_ignores=[]))
    exec(compile(source, str(path), "exec"), ns)
    return ns


class Tensor:
    def __init__(self, value=1, shape=(1, 8, 80, 16), dtype="float32", device=None):
        self.value, self.shape, self.dtype, self._device = value, shape, dtype, device
        self.layout = "ROW_MAJOR"

    def reshape(self, *shape):
        return Tensor(self.value, shape, self.dtype, self._device)

    def permute(self, *axes):
        return self.reshape(*(self.shape[i] for i in axes))

    def float(self):
        self.dtype = "float32"
        return self

    def squeeze(self, axis):
        return self.reshape(*(s for i, s in enumerate(self.shape) if i != axis or s != 1))

    def device(self):
        return self._device

    def buffer_address(self):
        return id(self)


class Native:
    Tensor = Tensor
    BufferType = SimpleNamespace(TRACE="TRACE")
    float32, bfloat16 = "float32", "bfloat16"

    def __init__(self):
        self.mesh = SimpleNamespace(id=lambda: 7, shape=(1, 1))
        self.view = SimpleNamespace(num_banks=8, total_bytes_per_bank=1024, largest_contiguous_bytes_free_per_bank=1024)
        self.get_memory_view = Mock(return_value=self.view)
        self.active = None
        self.traces = {}
        self.serial = 0
        self.fail_end = False
        self.fail_execute = False
        self.executions = 0

    def begin_trace_capture(self, device, *, cq_id):
        assert self.active is None
        self.serial += 1
        self.active = self.serial
        self.traces[self.active] = []
        return self.active

    def end_trace_capture(self, device, trace_id, *, cq_id):
        self.active = None
        if self.fail_end:
            raise RuntimeError("trace reservation exhausted")

    def execute_trace(self, device, trace_id, *, cq_id, blocking):
        if self.fail_execute:
            raise RuntimeError("execution failed")
        self.executions += 1
        for op in self.traces[trace_id]:
            op()

    def release_trace(self, device, trace_id):
        del self.traces[trace_id]

    def copy(self, new, prev):
        prev.value = new.value

    def operation(self, x):
        out = Tensor(0, x.shape, x.dtype, self.mesh)
        op = lambda: setattr(out, "value", x.value + 1)
        if self.active is None:
            op()
        else:
            self.traces[self.active].append(op)
        return out


def fixture(*, traced=True, independent=True, env=None):
    native = Native()
    # Load the real decorator and Tracer, without native or torch imports.
    tracing = definitions(
        ROOT / "utils/tracing.py",
        {"Tracer", "traced_function", "_verify_value", "_tree_map", "_clone_tensor", "_is_tracer_valid_value"},
        dict(
            ttnn=native,
            overload=overload,
            logger=Mock(),
            functools=functools,
            inspect=inspect,
            weakref=weakref,
            _kernel_prewarm_capturing=False,
            _OMITTED=object(),
            _MESH_DEVICE_PARAM_NAMES=("mesh_device", "device"),
            _TRACER_VALID_INPUT_TYPES=(Tensor, int, float, str, bool, type(None)),
        ),
    )
    tracing["_TRACER_CALL_KWARGS"] = frozenset(
        n
        for n, p in inspect.signature(tracing["Tracer"].__call__).parameters.items()
        if p.kind == inspect.Parameter.KEYWORD_ONLY
    )
    traced_function = tracing["traced_function"]

    class Leaf:
        def __init__(self, **kwargs):
            self.mesh_device = native.mesh
            self.use_trace = False
            self.z_channels = 8
            self.loaded = True
            self.eager_calls = 0
            self.stats = None

        @traced_function(device=lambda self: self.mesh_device, prep_run=False)
        def _forward_device(self, value):
            return native.operation(value)

        def run(self, value, traced=False):
            # Model the production host bridge: new upload, owned host result.
            uploaded = Tensor(value.value, value.shape, value.dtype, native.mesh)
            out = self._forward_device(uploaded, traced=traced, tracer_trace_key=value.shape)
            return Tensor(out.value, out.shape, out.dtype)

        def __call__(self, value):
            self.eager_calls += 1
            return self.run(value)

        def forward_traced(self, value):
            return self.run(value, traced=True)

        def release_trace(self):
            for t in type(self)._forward_device._tracers_keyed.get(self, {}).values():
                t.release_trace()

        def is_loaded(self):
            return self.loaded

        def set_per_channel_stats(self, *stats):
            self.stats = stats

    class BWE:
        def __init__(self, vocoder, bwe_generator, **kwargs):
            self.vocoder, self.bwe_generator = vocoder, bwe_generator
            self.mesh_device = native.mesh
            self.output_sampling_rate = 24000
            self.device_chain = False
            self.use_trace = self.use_trace_bwe = False
            self.loaded = True

        @traced_function(device=lambda self: self.mesh_device, prep_run=False)
        def _forward_device_chain(self, value):
            return native.operation(value)

        def __call__(self, value):
            if self.device_chain:
                uploaded = Tensor(value.value, value.shape, value.dtype, native.mesh)
                out = self._forward_device_chain(
                    uploaded, traced=self.use_trace or self.use_trace_bwe, tracer_trace_key=value.shape
                )
                return Tensor(out.value, out.shape, out.dtype)
            x = self.vocoder.forward_traced(value) if self.use_trace else self.vocoder(value)
            return self.bwe_generator.forward_traced(x) if self.use_trace_bwe else self.bwe_generator(x)

        def release_trace(self):
            self.vocoder.release_trace()
            self.bwe_generator.release_trace()
            for t in type(self)._forward_device_chain._tracers_keyed.get(self, {}).values():
                t.release_trace()

        def is_loaded(self):
            return self.loaded

    config = {
        "audio_vae": {"model": {"params": {"ddconfig": {"mel_bins": 64}}}},
        "vocoder": {
            "vocoder": {},
            "bwe": dict(
                n_fft=16,
                hop_length=4,
                win_length=16,
                num_mels=64,
                input_sampling_rate=16000,
                output_sampling_rate=24000,
            ),
        },
    }
    checkpoint = Mock()
    checkpoint.__enter__ = Mock(return_value=checkpoint)
    checkpoint.__exit__ = Mock(return_value=False)
    checkpoint.metadata.return_value = {"config": json.dumps(config)}
    checkpoint.get_tensor.return_value = 1
    ns = definitions(
        AUDIO,
        {"_env_flag", "resolve_audio_traced", "LTXAudioDecoderAdapter"},
        dict(
            os=os,
            json=json,
            contextmanager=contextmanager,
            _TRUE_TOKENS={"1", "true", "yes", "on"},
            _FALSE_TOKENS={"0", "false", "no", "off"},
            logger=Mock(),
            ttnn=native,
            safe_open=Mock(return_value=checkpoint),
            MelDecoder=Leaf,
            MelSTFT=Mock(),
            VocoderWithBWE=BWE,
            AudioTCParallelConfig=type("AudioTCParallelConfig", (), {}),
            conv3d_blocking_hash=lambda _: "",
            cache_module=SimpleNamespace(load_model=Mock()),
        ),
    )
    cls = ns["LTXAudioDecoderAdapter"]
    with patch.object(cls, "_build_vocoder", side_effect=lambda *a, **kw: Leaf()), patch.dict(
        os.environ, env or {}, clear=True
    ):
        adapter = cls(
            "audio25.safetensors",
            mesh_device=native.mesh,
            vae_ccl_manager=None,
            dit_parallel_config=None,
            traced=traced,
            independent_trace=independent,
        )
    return adapter, native, ns, tracing


def decode(adapter, value):
    with adapter.decode_scope(value):
        mel = adapter.mel_decoder
        x = mel.forward_traced(value) if mel.use_trace else mel(value)
        return adapter.vocoder_with_bwe(x)


class AudioTraceControlTest(unittest.TestCase):
    def test_pipeline_propagates_independent_control(self):
        adapter = Mock()
        method = definitions(
            PIPELINE, {"_new_audio_decoder"}, {"LTXAudioDecoderAdapter": adapter}, class_name="LTXPipeline"
        )["_new_audio_decoder"]
        for pipeline_traced, audio_traced in [(False, True), (True, True), (False, False), (True, False)]:
            with self.subTest(pipeline=pipeline_traced, audio=audio_traced):
                pipeline = SimpleNamespace(
                    _traced=pipeline_traced,
                    _audio_traced=audio_traced,
                    audio_mesh_device=object(),
                    audio_ccl_manager=object(),
                    parallel_config=object(),
                    _audio_checkpoint_path=lambda: "audio25.safetensors",
                )
                method(pipeline)
                self.assertEqual(adapter.call_args.kwargs["traced"], audio_traced)
                self.assertEqual(adapter.call_args.kwargs["independent_trace"], audio_traced and not pipeline_traced)
                self.assertEqual(pipeline._traced, pipeline_traced)

    def test_policy_defaults_overrides_and_invalid_env(self):
        _, _, ns, _ = fixture()
        resolve = ns["resolve_audio_traced"]
        for traced in (True, False):
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(resolve(traced, None), traced)
            for token, enabled in [("1", True), ("true", True), ("0", False), ("false", False)]:
                with patch.dict(os.environ, {"LTX_AUDIO_TRACE": token}, clear=True):
                    self.assertEqual(resolve(traced, None), enabled)
                    self.assertEqual(resolve(traced, not enabled), not enabled)
        with patch.dict(os.environ, {"LTX_AUDIO_TRACE": "invalid"}, clear=True):
            with self.assertRaises(ValueError):
                resolve(False, None)
            self.assertFalse(resolve(True, False))

    def test_stage_opt_outs_and_legacy_flags(self):
        for traced, independent in ((True, True), (True, False), (False, False)):
            for name, stage in [("LTX_VAE_TRACE", "mel"), ("LTX_VOC_TRACE", "vocoder"), ("LTX_BWE_TRACE", "bwe")]:
                with self.subTest(traced=traced, independent=independent, stage=stage):
                    a, _, _, _ = fixture(traced=traced, independent=independent, env={name: "0"})
                    decode(a, Tensor())
                    decode(a, Tensor())
                    stages = a.trace_status()["stages"]
                    self.assertFalse(stages[stage]["enabled"])
                    self.assertEqual(stages[stage]["captures"], 0)
                    self.assertEqual(stages[stage]["replays"], 0)
                    self.assertEqual(stages["bwe" if stage != "bwe" else "mel"]["enabled"], traced)

    def test_independent_request_with_all_stages_disabled_stays_eager(self):
        a, native, _, _ = fixture(env={name: "0" for name in ("LTX_VAE_TRACE", "LTX_VOC_TRACE", "LTX_BWE_TRACE")})
        for value in (5, 9):
            self.assertEqual(decode(a, Tensor(value)).value, value + 3)
        self.assertTrue(a.trace_status()["requested"])
        self.assertIsNone(a._warm_trace_identity)
        self.assertEqual(a.mel_decoder.eager_calls, 2)
        self.assertEqual(native.serial, 0)
        for stage in a.trace_status()["stages"].values():
            self.assertEqual(stage, dict(enabled=False, captured=0, captures=0, replays=0))

    def test_capacity_zero_unknown_and_exhausted_rejected(self):
        a, native, ns, _ = fixture()
        cls = type(a)
        for view, expected in [
            (SimpleNamespace(num_banks=8, total_bytes_per_bank=0), ValueError),
            (None, RuntimeError),
        ]:
            native.get_memory_view.return_value = view
            with patch.object(cls, "_build") as build, self.assertRaises(expected):
                cls(
                    "checkpoint",
                    mesh_device=native.mesh,
                    vae_ccl_manager=None,
                    dit_parallel_config=None,
                    traced=True,
                    independent_trace=True,
                )
            build.assert_not_called()
        native.get_memory_view.return_value = native.view
        native.view.largest_contiguous_bytes_free_per_bank = 0
        with self.assertRaisesRegex(RuntimeError, "No reserved trace capacity"):
            decode(a, Tensor())
        self.assertEqual(a.mel_decoder.eager_calls, 0)
        self.assertEqual(a.trace_status()["reservation_bytes"], 8192)

    def test_existing_traced_and_eager_modes_do_not_query_new_capacity_contract(self):
        for traced in (True, False):
            a, native, _, _ = fixture(traced=traced, independent=False)
            native.get_memory_view.assert_not_called()
            decode(a, Tensor())
            decode(a, Tensor())
            self.assertEqual(a.trace_status()["stages"]["mel"]["captures"], int(traced))
            self.assertEqual(a.trace_status()["stages"]["mel"]["replays"], int(traced))
            native.get_memory_view.assert_not_called()

    def test_reuse_release_recapture_and_owned_outputs(self):
        a, native, _, _ = fixture()
        first = decode(a, Tensor(5))
        self.assertEqual(first.value, 8)
        self.assertEqual(a.mel_decoder.eager_calls, 1)
        self.assertEqual(decode(a, Tensor(9)).value, 12)
        self.assertEqual(first.value, 8)
        self.assertEqual(a.mel_decoder.eager_calls, 1)
        for stage in ("mel", "vocoder", "bwe"):
            self.assertEqual(a.trace_status()["stages"][stage], dict(enabled=True, captured=1, captures=1, replays=1))
        a.release_traces()
        a.release_traces()
        self.assertEqual(native.traces, {})
        self.assertIsNone(a._warm_trace_identity)
        decode(a, Tensor(3))
        self.assertEqual(a.mel_decoder.eager_calls, 2)
        self.assertEqual(a.trace_status()["stages"]["mel"]["captures"], 2)

    def test_shape_dtype_and_mode_changes_require_release(self):
        for change in ("shape", "dtype", "chain", "disable"):
            with self.subTest(change=change):
                a, native, _, _ = fixture()
                decode(a, Tensor())
                value = Tensor()
                if change == "shape":
                    value.shape = (1, 8, 100, 16)
                elif change == "dtype":
                    value.dtype = "bfloat16"
                elif change == "chain":
                    a.vocoder_with_bwe.device_chain = True
                else:
                    a.mel_decoder.use_trace = False
                    a.vocoder_with_bwe.use_trace = a.vocoder_with_bwe.use_trace_bwe = False
                with self.assertRaisesRegex(RuntimeError, "Release audio traces"):
                    decode(a, value)
                self.assertEqual(native.serial, 3)
                self.assertEqual(len(native.traces), 3)
                self.assertIsNotNone(a._warm_trace_identity)
                a.release_traces()
                decode(a, value)

    def test_chain_status_counts_only_owning_tracer(self):
        a, _, _, _ = fixture()
        a.vocoder_with_bwe.device_chain = True
        decode(a, Tensor())
        decode(a, Tensor())
        stages = a.trace_status()["stages"]
        self.assertEqual(stages["chain"]["replays"], 1)
        self.assertEqual(stages["vocoder"]["captures"], 0)
        self.assertEqual(stages["bwe"]["captures"], 0)

    def test_warm_failure_restores_flags_and_does_not_mark_warm(self):
        a, _, _, _ = fixture()
        with patch.object(type(a.mel_decoder), "__call__", side_effect=RuntimeError("warm failure")):
            with self.assertRaisesRegex(RuntimeError, "warm failure"):
                decode(a, Tensor())
        self.assertIsNone(a._warm_trace_identity)
        self.assertTrue(a.mel_decoder.use_trace)
        self.assertTrue(a.vocoder_with_bwe.use_trace)
        self.assertTrue(a.vocoder_with_bwe.use_trace_bwe)
        self.assertEqual(a.trace_status()["stages"]["mel"]["captures"], 0)

    def test_failed_warmup_clears_persistent_caches_and_allows_retry(self):
        for error in (RuntimeError, KeyboardInterrupt, SystemExit, GeneratorExit):
            for chain in (False, True):
                for direct in (False, True):
                    with self.subTest(error=error.__name__, chain=chain, direct=direct):
                        a, native, _, _ = fixture()
                        voc = a.vocoder_with_bwe
                        voc.device_chain = chain
                        # Exercise the production cache disposal with fake device owners.
                        type(voc.vocoder).release_trace = definitions(
                            ROOT / "models/audio_vae/vocoder_ltx.py", {"release_trace"}, class_name="Vocoder"
                        )["release_trace"]
                        type(voc).release_trace = definitions(
                            ROOT / "models/audio_vae/bwe_ltx.py", {"release_trace"}, class_name="VocoderWithBWE"
                        )["release_trace"]
                        for module in (a.mel_decoder, voc.vocoder, voc.bwe_generator):
                            module._tpad_mask_cache = {}
                        voc.mel_stft = SimpleNamespace(stft_fn=SimpleNamespace(_window_indices={}))
                        caches = (
                            voc.vocoder._tpad_mask_cache,
                            voc.bwe_generator._tpad_mask_cache,
                            voc.mel_stft.stft_fn._window_indices,
                        )
                        failure = error("after eager cache allocation")

                        def fail_after_cache(module, value):
                            for cache in caches:
                                cache[value.shape] = object()
                            raise failure

                        with patch.object(type(voc), "__call__", fail_after_cache):
                            for frames in (80, 96, 112):
                                value = Tensor(shape=(1, 8, frames, 16))
                                with self.assertRaises(error) as caught:
                                    a.prepare_decode(value) if direct else decode(a, value)
                                self.assertIs(caught.exception, failure)
                                self.assertTrue(all(not cache for cache in caches))
                                self.assertIsNone(a._warm_trace_identity)
                                self.assertEqual(native.traces, {})
                                self.assertTrue(a.mel_decoder.use_trace)
                                self.assertTrue(voc.use_trace)
                                self.assertTrue(voc.use_trace_bwe)
                        self.assertEqual(decode(a, Tensor(5)).value, 7 if chain else 8)
                        self.assertEqual(decode(a, Tensor(9)).value, 11 if chain else 12)
                        stage = "chain" if chain else "vocoder"
                        self.assertEqual(a.trace_status()["stages"][stage]["replays"], 1)

    def test_partial_capture_and_execution_failures_release_buffers(self):
        for failure in ("end", "execute", "initial_execute", "late"):
            with self.subTest(failure=failure):
                a, native, _, tracing = fixture()
                if failure == "late":

                    def fail_late(*args, **kwargs):
                        raise RuntimeError("late capture failure")

                    a.vocoder_with_bwe.bwe_generator.forward_traced = fail_late
                elif failure == "end":
                    native.fail_end = True
                else:
                    if failure == "execute":
                        decode(a, Tensor())
                    native.fail_execute = True
                with self.assertRaises(RuntimeError):
                    decode(a, Tensor())
                self.assertEqual(native.traces, {})
                for _, module, method, _ in a._trace_modules():
                    for tracer in getattr(type(module), method)._tracers_keyed.get(module, {}).values():
                        self.assertEqual(tracer.inputs, {})
                        self.assertIsNone(tracer._outputs)
                self.assertIsNone(a._warm_trace_identity)
                self.assertEqual(tracing["Tracer"]._traces_live[7], 0)
                self.assertEqual(a.trace_status()["stages"]["mel"]["replays"], 0)

    def test_interrupted_capture_replay_and_late_stage_release_buffers(self):
        for error in (KeyboardInterrupt, SystemExit, GeneratorExit):
            for phase in ("capture", "replay", "late"):
                with self.subTest(error=error.__name__, phase=phase):
                    a, native, _, tracing = fixture()
                    if phase == "replay":
                        decode(a, Tensor())
                        target, method = native, "execute_trace"
                    elif phase == "late":
                        target, method = a.vocoder_with_bwe.bwe_generator, "forward_traced"
                    else:
                        # Complete eager preparation so the fault occurs inside capture.
                        a.prepare_decode(Tensor())
                        target, method = native, "operation"
                    with patch.object(target, method, side_effect=error("interrupted")):
                        with self.assertRaises(error):
                            decode(a, Tensor())
                    self.assertEqual(native.traces, {})
                    self.assertIsNone(native.active)
                    self.assertIsNone(a._warm_trace_identity)
                    self.assertEqual(tracing["Tracer"]._traces_live.get(7, 0), 0)
                    for _, module, method, _ in a._trace_modules():
                        for tracer in getattr(type(module), method)._tracers_keyed.get(module, {}).values():
                            self.assertEqual(tracer.inputs, {})
                            self.assertIsNone(tracer._outputs)
                    # Cancellation propagates; a later request can warm and capture anew.
                    decode(a, Tensor())
                    self.assertEqual(a.trace_status()["stages"]["mel"]["replays"], 0)

    def test_kernel_prewarm_is_neither_capture_nor_replay(self):
        a, _, _, tracing = fixture()
        tracing["_kernel_prewarm_capturing"] = True
        decode(a, Tensor())
        self.assertEqual(a.trace_status()["stages"]["mel"]["captures"], 0)
        self.assertEqual(a.trace_status()["stages"]["mel"]["replays"], 0)

    def test_failed_capture_can_retry_only_after_cleanup_and_eager_warmup(self):
        a, native, _, _ = fixture()
        native.fail_end = True
        with self.assertRaisesRegex(RuntimeError, "trace reservation exhausted"):
            decode(a, Tensor())
        self.assertEqual(native.traces, {})
        self.assertIsNone(a._warm_trace_identity)
        native.fail_end = False
        self.assertEqual(decode(a, Tensor(5)).value, 8)
        self.assertEqual(a.mel_decoder.eager_calls, 2)
        self.assertEqual(a.trace_status()["stages"]["mel"], dict(enabled=True, captured=1, captures=1, replays=0))
        self.assertEqual(decode(a, Tensor(9)).value, 12)
        self.assertEqual(a.trace_status()["stages"]["mel"]["replays"], 1)

    def test_reload_invalidates_traces_before_loading_new_weight_addresses(self):
        a, native, ns, _ = fixture()
        decode(a, Tensor())
        a.mel_decoder.loaded = False
        ns["cache_module"].load_model.side_effect = lambda *args, **kwargs: self.assertEqual(native.traces, {})
        a.reload_weights()
        ns["cache_module"].load_model.assert_called_once()
        self.assertIsNone(a._warm_trace_identity)
        self.assertEqual(a.mel_decoder.stats, (1, 1))

    def test_residency_guard_uses_final_decoder_and_preserves_existing_modes(self):
        method = definitions(PIPELINE, {"_register_coresident_exclusions"}, class_name="LTXPipeline")[
            "_register_coresident_exclusions"
        ]
        for dynamic, exclusive in [(True, False), (False, True), (True, True)]:
            p = SimpleNamespace(
                _audio_traced=True,
                _traced=False,
                dynamic_load=dynamic,
                vae_decoder=SimpleNamespace(requires_exclusive_residency=exclusive),
            )
            with self.assertRaisesRegex(ValueError, "resident weights"):
                method(p)
        p = SimpleNamespace(
            _audio_traced=True, _traced=False, dynamic_load=False, vae_decoder=None, transformer_states=[]
        )
        method(p)
        p._traced = True
        method(p)

    def test_pipeline_decode_uses_scope_and_reports_real_replay(self):
        a, native, _, _ = fixture()
        log = Mock()
        methods = definitions(
            PIPELINE,
            {"decode_audio", "_decode_mel", "_prepare_audio_decoder"},
            dict(os=os, logger=log, Watchdog=lambda _: nullcontext(), Audio=lambda **kw: SimpleNamespace(**kw)),
            class_name="LTXPipeline",
        )
        p = SimpleNamespace(
            checkpoint_name="checkpoint",
            _audio_adapter=a,
            tt_mel_decoder=a.mel_decoder,
            tt_vocoder_with_bwe=a.vocoder_with_bwe,
        )
        p._prepare_audio_decoder = functools.partial(methods["_prepare_audio_decoder"], p)
        p._decode_mel = functools.partial(methods["_decode_mel"], p)
        with patch.dict(os.environ, {}, clear=True):
            out = methods["decode_audio"](p, Tensor(5, (1, 80, 128)), 80)
            self.assertEqual(out.waveform.value, 8)
            methods["decode_audio"](p, Tensor(9, (1, 80, 128)), 80)
            self.assertEqual(a.trace_status()["stages"]["mel"]["replays"], 1)
            native.fail_execute = True
            with self.assertRaisesRegex(RuntimeError, "execution failed"):
                methods["decode_audio"](p, Tensor(9, (1, 80, 128)), 80)
        self.assertEqual(native.traces, {})
        self.assertTrue(any("Audio trace status:" in str(c) for c in log.info.call_args_list))

    def test_pipeline_constructor_policy_and_kernel_prewarm(self):
        tree = ast.parse(PIPELINE.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LTXPipeline")
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        self.assertIn("audio_traced", [n.arg for n in init.args.args + init.args.kwonlyargs])
        assigns = [
            n
            for n in init.body
            if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Attribute)
            and n.targets[0].attr in ("_traced", "_audio_traced")
        ]
        _, _, ns, _ = fixture()
        p = SimpleNamespace()
        exec(
            compile(ast.Module(assigns, type_ignores=[]), str(PIPELINE), "exec"),
            dict(self=p, traced=False, audio_traced=True, resolve_audio_traced=ns["resolve_audio_traced"]),
        )
        self.assertFalse(p._traced)
        self.assertTrue(p._audio_traced)
        call = next(
            n
            for n in ast.walk(init)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "set_kernel_prewarm_capturing"
            and not isinstance(n.args[0], ast.Constant)
        )
        self.assertTrue(eval(compile(ast.Expression(call.args[0]), str(PIPELINE), "eval"), {"self": p}))

    def test_pipeline_release_delegates_adapter_lifetime(self):
        method = definitions(PIPELINE, {"release_traces"}, {"StateTensor": Mock()}, class_name="LTXPipeline")[
            "release_traces"
        ]
        a, native, _, _ = fixture()
        decode(a, Tensor())
        p = SimpleNamespace(_trace_state={}, transformer=None, _audio_adapter=a, vae_decoder=None, vae_encoder=None)
        method(p)
        self.assertIsNone(a._warm_trace_identity)
        self.assertEqual(native.traces, {})


class AudioGatherLifetimeTest(unittest.TestCase):
    """Exercise the audio helpers and real CCL buffer policy without native kernels."""

    def setUp(self):
        configs = definitions(
            ROOT / "parallel/config.py",
            {"ParallelFactor", "AudioTParallelConfig", "AudioTCParallelConfig"},
            {"NamedTuple": NamedTuple},
        )
        self.pf = configs["ParallelFactor"]
        self.time = configs["AudioTParallelConfig"](self.pf(4, 0), self.pf(8, 1))
        self.tc = configs["AudioTCParallelConfig"](self.pf(4, 0), self.pf(8, 1))
        self.ops = definitions(
            ROOT / "layers/audio_ops.py", {"channel_axis", "all_gather_channel", "_all_gather_t"}, configs
        )
        self.native = SimpleNamespace(
            bfloat16="bf16",
            TILE_LAYOUT="tile",
            DRAM_MEMORY_CONFIG="dram",
            Shape=tuple,
            synchronize_device=Mock(),
            allocate_tensor_on_device=Mock(side_effect=lambda shape, dtype, *args: self.tensor(shape, dtype)),
            reshape=lambda x, shape: self.tensor(shape, x.get_dtype()),
            experimental=SimpleNamespace(all_gather_async=Mock(side_effect=self.gather)),
        )
        methods = definitions(
            ROOT / "parallel/manager.py",
            {"get_dim", "get_ag_ping_pong_buffer", "all_gather", "_check_audio_buffer", "_record_audio_buffer"},
            {"ttnn": self.native},
            class_name="CCLManager",
        )
        manager_type = type("Manager", (), {k: v for k, v in methods.items() if not k.startswith("__")})
        self.manager = manager_type()
        self.manager.mesh_device = SimpleNamespace(shape=(4, 8))
        self.manager.num_links = 1
        self.manager.topology = "ring"
        self.manager._ping_pong_buffer_cache = {}
        self.manager._ping_pong_buffer_indices = {}
        self.manager.get_ag_ping_pong_semaphore = Mock(return_value=["ready", "done"])
        self.manager.get_barrier_semaphore = Mock(side_effect=AssertionError("replay-unsafe barrier requested"))
        self.manager.get_ag_hyperparams = Mock(side_effect=AssertionError("unexpected hyperparameters"))

    @staticmethod
    def tensor(shape=(1, 32, 32), dtype="bf16"):
        return SimpleNamespace(shape=tuple(shape), get_dtype=lambda: dtype)

    def gather(self, x, *, persistent_output_buffer, barrier_semaphore, **kwargs):
        self.assertIsNotNone(persistent_output_buffer)
        self.assertIsNone(barrier_semaphore)
        return persistent_output_buffer

    def test_channel_gather_uses_warmed_buffer_pool(self):
        x = self.tensor()
        buffers = []
        for _ in range(4):
            out = self.ops["all_gather_channel"](self.manager, x, self.tc, dim=2)
            self.assertEqual(out.shape, (1, 32, 256))
            buffers.append(self.native.experimental.all_gather_async.call_args.kwargs["persistent_output_buffer"])
        self.assertIs(buffers[0], buffers[2])
        self.assertIs(buffers[1], buffers[3])
        self.assertIsNot(buffers[0], buffers[1])
        self.assertEqual(self.native.allocate_tensor_on_device.call_count, 2)
        self.assertEqual(self.native.synchronize_device.call_count, 2)
        self.assertEqual(len(self.manager._ping_pong_buffer_cache), 1)

    def test_channel_noop_without_channel_parallelism(self):
        x = self.tensor()
        configs = (None, self.pf(4, 0), self.time, type(self.tc)(self.pf(4, 0), self.pf(1, 1)))
        for config in configs:
            with self.subTest(config=config):
                self.assertIs(self.ops["all_gather_channel"](self.manager, x, config, dim=2), x)
        self.native.experimental.all_gather_async.assert_not_called()
        self.native.allocate_tensor_on_device.assert_not_called()

    def test_time_gather_preserves_axis_order_and_intermediate_shape(self):
        out = self.ops["_all_gather_t"](self.manager, self.tensor(), self.time)
        self.assertEqual(out.shape, (1, 1024, 32))
        calls = self.native.experimental.all_gather_async.call_args_list
        self.assertEqual([c.kwargs["cluster_axis"] for c in calls], [1, 0])
        self.assertEqual([c.args[0].shape for c in calls], [(1, 1, 32, 32), (1, 1, 256, 32)])
        self.assertEqual([c.kwargs["dim"] for c in calls], [2, 2])
        self.assertEqual(self.native.allocate_tensor_on_device.call_count, 4)

    def test_single_axis_time_gather_uses_persistent_buffer(self):
        for config in (self.pf(4, 0), self.tc):
            with self.subTest(config=config):
                out = self.ops["_all_gather_t"](self.manager, self.tensor(), config)
                self.assertEqual(out.shape, (1, 128, 32))
        self.assertEqual(self.native.allocate_tensor_on_device.call_count, 2)

    def test_unit_mesh_axis_remains_identity(self):
        self.manager.mesh_device.shape = (1, 1)
        x = self.tensor()
        self.assertIs(self.ops["_all_gather_t"](self.manager, x, self.time), x)
        self.assertIs(self.ops["all_gather_channel"](self.manager, x, self.tc, dim=2), x)
        self.native.experimental.all_gather_async.assert_not_called()
        self.native.allocate_tensor_on_device.assert_not_called()

    def test_pool_separates_shape_dtype_and_axis(self):
        for x in (self.tensor(), self.tensor(dtype="fp32"), self.tensor(shape=(1, 64, 32))):
            self.ops["all_gather_channel"](self.manager, x, self.tc, dim=2)
        self.ops["_all_gather_t"](self.manager, self.tensor(), self.time)
        self.assertEqual(len(self.manager._ping_pong_buffer_cache), 5)
        self.assertEqual(self.native.allocate_tensor_on_device.call_count, 10)

    def test_collective_errors_propagate_at_every_audio_call_site(self):
        cases = [("channel", 0), ("single", 0), ("two_axes", 0), ("two_axes", 1)]
        for mode, fail_at in cases:
            with self.subTest(mode=mode, fail_at=fail_at):
                failure = RuntimeError("native collective failed")
                calls = []

                def fail(x, **kwargs):
                    calls.append((x, kwargs))
                    if len(calls) == fail_at + 1:
                        raise failure
                    return x

                manager = SimpleNamespace(all_gather=fail)
                with self.assertRaises(RuntimeError) as caught:
                    if mode == "channel":
                        self.ops["all_gather_channel"](manager, self.tensor(), self.tc, dim=2)
                    else:
                        self.ops["_all_gather_t"](manager, self.tensor(), self.time if mode == "two_axes" else self.tc)
                self.assertIs(caught.exception, failure)
                self.assertEqual(len(calls), fail_at + 1)


if __name__ == "__main__":
    unittest.main()
