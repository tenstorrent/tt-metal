# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU contracts; run directly without importing the TTNN/device pytest conftest."""
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location("c12_contract", ROOT / "models/tt_dit/utils/c12.py")
c12 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = c12
spec.loader.exec_module(c12)


class C12Contracts(unittest.TestCase):
    def test_four_cells(self):
        self.assertEqual(
            [(c12.Cell(m).pack, c12.Cell(m).time_block) for m in "ABCD"], [(1, 4), (1, 32), (4, 4), (4, 32)]
        )


# Tensor-shaped CPU shim for the actual analytic helper, not Torch/native evidence.
import copy
import itertools
import tempfile


class Tensor:
    def __init__(self, shape, values=None):
        self.shape = shape
        self.values = dict(values or {})

    def new_zeros(self, shape):
        return Tensor(shape)

    def __getitem__(self, key):
        return self.values.get(key, 0.0)

    def __setitem__(self, key, value):
        self.values[key] = value


class Bias(list):
    @property
    def shape(self):
        return (len(self),)

    def repeat(self, count):
        return Bias(self * count)


def fixture():
    return Tensor((24, 24, 7), {(o, (o + j) % 24, j): ((o + j) % 5 - 2) / 8 for o in range(24) for j in range(7)})


def direct(rows, weight, bias):
    """Independent scatter from every original input tap to its output sample."""
    out = [list(bias or [0.0] * weight.shape[0]) for _ in rows]
    half = weight.shape[2] // 2
    for (o, i, j), value in weight.values.items():
        if value:
            for sample, row in enumerate(rows):
                dst = sample - j + half
                if 0 <= dst < len(rows):
                    out[dst][o] += value * row[i]
    return out


def packed_eval(rows, weight, bias, pack, shards=1, wrong_seams=False):
    transformed, repeated = c12.transform_weight(weight, bias, pack)
    padded = rows + [[0.0] * 24] * (-len(rows) % pack)
    packed = [sum(padded[n : n + pack], []) for n in range(0, len(padded), pack)]
    if shards == 1:
        result = direct(packed, transformed, repeated)
    else:
        assert len(packed) % shards == 0
        local = len(packed) // shards
        half = transformed.shape[2] // 2
        result = []
        for shard in range(shards):
            start = shard * local
            halo = [
                packed[i]
                if 0 <= i < len(packed) and (not wrong_seams or start <= i < start + local)
                else [0.0] * (24 * pack)
                for i in range(start - half, start + local + half)
            ]
            result.extend(direct(halo, transformed, repeated)[half : half + local])
    return [r[n : n + 24] for r in result for n in range(0, pack * 24, 24)][: len(rows)]


def evidence(config=None):
    return {
        "source_id": "fixture-only",
        "weight_sha256": "w",
        "bias_sha256": "b",
        "module_path": c12.MODULE_PATH,
        "config_sha256": c12.digest(config or c12.CONFIG),
        "weight_shape": [24, 24, 7],
        "bias_shape": [24],
    }


class AlgebraAndIdentity(unittest.TestCase):
    def test_actual_helper_direct_scatter_bias_zero_odd_short_and_batch2(self):
        for pack, length, bias_on, batch in itertools.product(
            (1, 4), (1, 3, 7, 31, 65, 128, 132), (False, True), (0, 1)
        ):
            with self.subTest(pack=pack, length=length, bias=bias_on, batch=batch):
                x = [[((t * 7 + c + batch) % 9 - 4) / 8 for c in range(24)] for t in range(length)]
                w = fixture()
                bias = Bias(c / 16 for c in range(24)) if bias_on else None
                self.assertEqual(packed_eval(x, w, bias, pack), direct(x, w, bias))
                zero = Tensor((24, 24, 7))
                self.assertEqual(packed_eval(x, zero, bias, pack), direct(x, zero, bias))

    def test_shards_tails_changed_restored_and_negative_seams(self):
        w = fixture()
        b = Bias(c / 16 for c in range(24))
        for shards, tail in itertools.product((2, 4, 8), (0, 1, 2, 3, 4, 7, 127, 128, 129, 257, 511)):
            if tail >= 128 * shards:
                continue
            rows = [[((t + c) % 11 - 5) / 8 for c in range(24)] for t in range(128 * shards)]
            if tail:
                rows[-tail:] = [[0.0] * 24 for _ in range(tail)]
            expected = direct(rows, w, b)
            self.assertEqual(packed_eval(rows, w, b, 4, shards), expected)
            wrong = packed_eval(rows, w, b, 4, shards, True)
            if len(rows) - tail > 125:
                self.assertTrue(wrong != expected, (shards, tail, "live seam must detect missing halo"))
            else:
                # A near-empty clip has no live sample reaching any interior seam.
                self.assertEqual(wrong, expected)
            changed = [[-v for v in r] for r in rows]
            self.assertEqual(packed_eval(changed, w, b, 4, shards), direct(changed, w, b))
            self.assertNotEqual(packed_eval(changed, w, b, 4, shards), expected)
            self.assertEqual(packed_eval(rows, w, b, 4, shards), expected)

    def test_transform_impulse_columns(self):
        w = fixture()
        v, _ = c12.transform_weight(w, None, 4)
        # Probe an input impulse independently; cross-correlation reverses displacement.
        for source_phase, channel in itertools.product(range(4), range(24)):
            x = [[0.0] * 24 for _ in range(28)]
            x[12 + source_phase][channel] = 1.0
            response = direct(x, w, None)
            for q, phase, out in itertools.product(range(2, 5), range(4), range(24)):
                self.assertEqual(v[phase * 24 + out, source_phase * 24 + channel, 4 - q], response[q * 4 + phase][out])

    def test_shapes_and_modes(self):
        for mode in "ABCD":
            cell = c12.Cell(mode)
            for t in (128, 132, 256):
                self.assertIsNone(c12.shape_reason(cell, (1, t, 24), starts=[0, t], time_factor=2))
            for shape in ((2, 128, 24), (1, 128, 32), (1, 0, 24)):
                self.assertIsNotNone(c12.shape_reason(cell, shape, starts=[0, shape[1]], time_factor=2))
            self.assertIsNotNone(c12.shape_reason(cell, (1, 128, 24), starts=[1, 129], time_factor=2))
        for t in (1, 3, 124, 127, 129, 131):
            self.assertIsNotNone(c12.shape_reason(c12.Cell("D"), (1, t, 24), starts=[0, t], time_factor=2))
        for mode in ("off", "0", "", None):
            self.assertEqual(c12.parse_mode(mode), (None, False))
        self.assertEqual(c12.parse_mode("auto:B"), (c12.Cell("B"), True))
        for mode in ("auto", "pack2", "E", "a"):
            with self.assertRaises(ValueError):
                c12.parse_mode(mode)

    def test_checkpoint_configuration_and_exact_path(self):
        self.assertIsNone(c12.checkpoint_reason(c12.CONFIG, evidence()))
        self.assertEqual(c12.checkpoint_reason(c12.CONFIG, None), "missing checkpoint evidence")
        for key in c12.CONFIG:
            cfg = copy.deepcopy(c12.CONFIG)
            cfg.pop(key)
            self.assertIsNotNone(c12.checkpoint_reason(cfg, evidence(cfg)))
        for key, value in (
            ("module_path", "bwe_generator.resblocks.16.convs2.0"),
            ("weight_shape", [32, 32, 7]),
            ("bias_shape", [32]),
            ("bias_sha256", None),
            ("config_sha256", "stale"),
        ):
            e = evidence()
            e[key] = value
            self.assertIsNotNone(c12.checkpoint_reason(c12.CONFIG, e))

    def test_cache_and_trace_identity(self):
        identities = [c12.Cell(mode).identity() for mode in "ABCD"]
        self.assertEqual(len({c12.cache_suffix(i) for i in identities}), 4)
        self.assertEqual(c12.cache_suffix(None), "")
        self.assertEqual(c12.trace_key((1, 128, 24), None), (1, 128, 24))
        self.assertEqual(len({c12.trace_key((1, 128, 24), i) for i in identities}), 4)
        for key in ("schema", "module_path", "pack", "weight_layout", "blocking", "precision"):
            modified = copy.deepcopy(identities[0])
            modified[key] = "changed"
            self.assertNotEqual(c12.cache_suffix(modified), c12.cache_suffix(identities[0]))
        with tempfile.TemporaryDirectory() as root:
            c12.claim_cache_root(root, identities[0])
            with self.assertRaises(ValueError):
                c12.claim_cache_root(root, identities[0])


# Execute the real wrapper against operation/ownership stubs. This is explicitly
# not a TTNN simulator; the algebra above separately checks numerical equivalence.
import ast
import json
import os
from collections import namedtuple
from types import SimpleNamespace as NS
from unittest.mock import patch

PC = namedtuple("ParallelFactor", "factor mesh_axis")


class DeviceTensor:
    def __init__(self, shape, values=None, dtype="fp32", layout="rm"):
        self.shape = tuple(shape)
        self.padded_shape = (*shape[:-1], (shape[-1] + 31) // 32 * 32)
        self.layout = layout
        self.dtype = dtype
        # Poison physical padding; data-moving stubs must read logical values.
        count = 1
        for dim in shape:
            count *= dim
        logical = list(range(count)) if values is None else list(values)
        self.pages = [
            logical[i : i + shape[-1]] + [999999] * (self.padded_shape[-1] - shape[-1])
            for i in range(0, count, shape[-1])
        ]

    def logical(self):
        return [v for p in self.pages for v in p[: self.shape[-1]]]

    def get_dtype(self):
        return self.dtype

    def memory_config(self):
        return "dram"


def wrapper_namespace():
    calls = []

    class Base:
        def __init__(self, ci=24, co=24, kernel_size=7, **kw):
            self.__dict__.update(kw)
            self.bias_enabled = kw.get("bias", True)
            self.bias = NS(data=None) if self.bias_enabled else None
            self.weight = NS(data=None)
            self.unpadded_in_channels = ci
            self.unpadded_out_channels = co
            self.kernel_size = (kernel_size, 1, 1)
            self.stride = (1, 1, 1)
            self.dilation = 1
            self.same_pad = kernel_size // 2
            self.eff_k = kernel_size
            self.internal_padding = (0, 0, 0)
            self.halo_pad_left = self.halo_pad_right = kernel_size // 2
            self.padding_mode = "zeros"
            self.compute_kernel_config = NS(
                math_fidelity="HiFi4", math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
            )

        def forward(self, x):
            calls.append(("baseline", x.shape))
            return DeviceTensor(x.shape, x.logical())

        def __call__(self, x):
            return self.forward(x)

        def is_loaded(self):
            return False

    def sliced(x, starts, ends, steps=None, **kw):
        calls.append(("slice", starts, ends, steps))
        steps = steps or [1] * len(starts)
        b, t, c = x.shape
        values = x.logical()
        ranges = [range(a, z, s) for a, z, s in zip(starts, ends, steps)]
        return DeviceTensor(
            tuple(len(r) for r in ranges), [values[(i * t + j) * c + k] for i, j, k in itertools.product(*ranges)]
        )

    def concat(parts, dim, **kw):
        calls.append(("concat", dim))
        assert dim == 2
        b, t, c = parts[0].shape
        lists = [p.logical() for p in parts]
        return DeviceTensor(
            (b, t, c * len(parts)), [v for i in range(b * t) for vals in lists for v in vals[i * c : (i + 1) * c]]
        )

    def reshape(x, shape):
        calls.append(("reshape", x.shape, shape))
        return DeviceTensor(shape, x.logical())

    def neighbor(x, **kw):
        calls.append(("halo", kw))
        b, t, c = x.shape
        assert b == 1 and kw["pad_left"] == kw["pad_right"] == 1
        # This stub records the call contract, not real multi-chip halo arithmetic.
        return DeviceTensor((b, t + 2, c), [0] * c + x.logical() + [0] * c)

    def conv(**kw):
        x = kw["input_tensor"]
        calls.append(("conv", kw))
        b, t, _, _, c = x.shape
        # Identity payload tests phase order, trim and unpack across physical pads.
        return DeviceTensor((b, t - 2, 1, 1, c), x.logical()[c:-c])

    def clone(x, **kw):
        calls.append(("clone", x.shape))
        return DeviceTensor(x.shape, x.logical())

    tt = NS(
        float32="fp32",
        ROW_MAJOR_LAYOUT="rm",
        DRAM_MEMORY_CONFIG="dram",
        MathFidelity=NS(HiFi4="HiFi4"),
        Conv3dConfig=lambda **kw: NS(**kw),
        slice=sliced,
        concat=concat,
        reshape=reshape,
        clone=clone,
        experimental=NS(conv3d=conv),
    )
    ns = {
        "__name__": "c12_wrapper_stub",
        "ttnn": tt,
        "_AlignedOutConv1d": Base,
        "_t_neighbor_pad": neighbor,
        "ParallelFactor": PC,
        "DilatedConv1d": Base,
        "logger": NS(info=lambda *args: None),
        "json": json,
        "os": os,
    }
    for name in (
        "cache_suffix",
        "checkpoint_reason",
        "claim_cache_root",
        "digest",
        "parse_mode",
        "shape_reason",
        "transform_weight",
    ):
        ns[name] = getattr(c12, name)
    tree = ast.parse((ROOT / "models/tt_dit/layers/audio_c12.py").read_text())
    tree.body = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
    exec(compile(tree, "audio_c12.py", "exec"), ns)
    return ns, Base, calls


class WrapperContracts(unittest.TestCase):
    def make_base(self, base_class):
        return base_class(
            mesh_device=NS(shape=(2, 1), compute_with_storage_grid_size=lambda: (8, 8)),
            dtype="fp32",
            parallel_config=PC(2, 0),
            ccl_manager=object(),
            split_mode="off",
        )

    def test_pack_layout_and_borrowed_input_ownership(self):
        for mode in "CD":
            ns, base, calls = wrapper_namespace()
            obj = ns["C12Conv1d"](
                self.make_base(base), cell=c12.Cell(mode), automatic=False, identity=c12.Cell(mode).identity()
            )
            obj.bind_input(batch=1, local_t=128, tail_samples=3)
            x = DeviceTensor((1, 128, 24))
            before = copy.deepcopy(x.pages)
            result = obj.forward(x)
            self.assertEqual(result.shape, x.shape)
            self.assertEqual(result.logical(), x.logical())
            self.assertEqual(x.pages, before)
            self.assertIsNot(result.pages, x.pages)
            self.assertEqual(sum(c[0] == "conv" for c in calls), 1)
            self.assertEqual(sum(c[0] == "halo" for c in calls), 1)
            self.assertEqual(calls[-1][0], "clone")
            manifest = next(iter(obj.execution_manifests.values()))
            self.assertEqual(manifest["global_shard_starts"], [0, 128])
            self.assertIsNone(manifest["fallback_reason"])
            self.assertEqual(manifest["upload_partition"]["tail_samples"], 3)
            for values in ([v + 1 for v in x.logical()], x.logical()):
                changed = DeviceTensor(x.shape, values)
                self.assertEqual(obj.forward(changed).logical(), values)

    def test_precision_and_fixed_configs(self):
        for mode in "ABCD":
            ns, base, _ = wrapper_namespace()
            obj = ns["C12Conv1d"](self.make_base(base), cell=c12.Cell(mode), automatic=False, identity={})
            obj._validate_precision()
            for key, expected in c12.Cell(mode).identity()["blocking"].items():
                self.assertEqual(getattr(obj.conv_config, key), expected)
            for key, changed in (
                ("math_fidelity", "LoFi"),
                ("math_approx_mode", True),
                ("fp32_dest_acc_en", False),
                ("packer_l1_acc", False),
            ):
                before = getattr(obj.compute_kernel_config, key)
                setattr(obj.compute_kernel_config, key, changed)
                with self.assertRaises(ValueError):
                    obj._validate_precision()
                setattr(obj.compute_kernel_config, key, before)
            obj.conv_config.T_out_block = 8
            with self.assertRaises(ValueError):
                obj._validate_precision()

    def test_explicit_early_rejection_and_auto_fallback(self):
        ns, base, calls = wrapper_namespace()
        for auto in (False, True):
            obj = ns["C12Conv1d"](self.make_base(base), cell=c12.Cell("D"), automatic=auto, identity={})
            if not auto:
                with self.assertRaises(ValueError):
                    obj.bind_input(batch=1, local_t=127, tail_samples=0)
                with self.assertRaises(ValueError):
                    obj.forward(DeviceTensor((1, 128, 24)))
            else:
                obj.bind_input(batch=1, local_t=127, tail_samples=0)
                obj.forward(DeviceTensor((1, 127, 24)))
                manifest = next(iter(obj.execution_manifests.values()))
                self.assertEqual(manifest["selected"], "baseline")
                self.assertIn("divisible", manifest["fallback_reason"])
        self.assertFalse(any(c[0] == "conv" for c in calls))

    def test_selector_default_exact_module_and_checkpoint(self):
        ns, base, _ = wrapper_namespace()
        install = ns["install_c12"]
        self.assertIsNone(install(object(), mode="off", config=None, checkpoint=None))

        class Modules(list):
            def add_module(self, key, value):
                self[int(key)] = value

        def vocoder():
            target = self.make_base(base)
            blocks = Modules(
                [
                    NS(
                        channels=24,
                        kernel_size=7,
                        num_branches=3,
                        convs2=Modules([self.make_base(base) for _ in range(3)]),
                    )
                    for _ in range(18)
                ]
            )
            blocks[16].convs2[0] = target
            return NS(
                parallel_config=PC(2, 0),
                mesh_device=target.mesh_device,
                dtype="fp32",
                num_upsamples=6,
                num_kernels=3,
                resblocks=blocks,
                is_loaded=lambda: False,
            )

        for mode in "ABCD":
            v = vocoder()
            originals = [[b.convs2[i] for i in range(3)] for b in v.resblocks]
            with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {"TT_DIT_CACHE_DIR": root}):
                install(v, mode=mode, config=c12.CONFIG, checkpoint=evidence())
            for i, j in itertools.product(range(18), range(3)):
                if (i, j) == (16, 0):
                    self.assertIsNot(v.resblocks[i].convs2[j], originals[i][j])
                else:
                    self.assertIs(v.resblocks[i].convs2[j], originals[i][j])
        for mutation in (
            lambda v: setattr(v, "parallel_config", None),
            lambda v: setattr(v.resblocks[16].convs2[0], "dilation", 3),
            lambda v: setattr(v.resblocks[16], "channels", 32),
            lambda v: setattr(v.resblocks[16].convs2[0], "split_mode", "full"),
            lambda v: setattr(v.resblocks[16].convs2[0].compute_kernel_config, "packer_l1_acc", False),
        ):
            v = vocoder()
            mutation(v)
            with self.assertRaises(ValueError):
                install(v, mode="D", config=c12.CONFIG, checkpoint=evidence())
            self.assertIsNone(install(v, mode="auto:D", config=c12.CONFIG, checkpoint=evidence()))
            self.assertEqual(v.c12_selection["selected"], "baseline")
        v = vocoder()
        self.assertIsNone(install(v, mode="auto:D", config=c12.CONFIG, checkpoint=None))
        self.assertEqual(v.c12_selection["fallback_reason"], "missing checkpoint evidence")


if __name__ == "__main__":
    unittest.main()
