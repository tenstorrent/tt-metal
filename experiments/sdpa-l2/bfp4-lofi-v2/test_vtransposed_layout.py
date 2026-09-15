"""CPU/static checks only: no TTNN import and no device execution."""

import ast
import contextlib
import io
import json
import math
import struct
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace as NS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


class Tensor:
    next_address = 4096

    def __init__(self, shape, dtype="bf16"):
        self.shape, self.dtype = tuple(shape), dtype
        self.address = Tensor.next_address
        Tensor.next_address += 4096

    def buffer_address(self):
        return self.address

    def volume(self):
        return math.prod(self.shape)

    def transpose(self, a, b):
        shape = list(self.shape)
        shape[a], shape[b] = shape[b], shape[a]
        return Tensor(shape, self.dtype)

    def contiguous(self):
        return self


class Device:
    def compute_with_storage_grid_size(self):
        return NS(x=11, y=10)

    def worker_core_from_logical_core(self, core):
        return core


def environment(filename):
    calls = []
    ttnn = NS(
        bfloat16="bf16",
        bfloat4_b="b4",
        bfloat8_b="b8",
        float32="fp32",
        TILE_LAYOUT="tile",
        DRAM_MEMORY_CONFIG="dram",
        Shape=tuple,
        MathFidelity=NS(LoFi="LoFi", HiFi4="HiFi4"),
        CoreCoord=lambda x, y: NS(x=x, y=y),
        CoreRange=lambda a, b: (a, b),
        CoreRangeSet=list,
        RuntimeArgs=lambda: defaultdict(lambda: defaultdict(list)),
        allocate_tensor_on_device=lambda shape, dtype, *args: Tensor(shape, dtype),
        from_torch=lambda x, **kwargs: Tensor(x.shape, x.dtype),
        TensorAccessorArgs=lambda t: NS(get_compile_time_args=lambda: [1, t.volume(), t.dtype]),
        generic_op=lambda tensors, desc: calls.append(desc),
    )
    for name in (
        "CBDescriptor",
        "CBFormatDescriptor",
        "ComputeConfigDescriptor",
        "KernelDescriptor",
        "ReaderConfigDescriptor",
        "WriterConfigDescriptor",
        "ProgramDescriptor",
        "SemaphoreDescriptor",
    ):
        setattr(ttnn, name, lambda **kwargs: NS(**kwargs))

    def quantizer(device, src, **kwargs):
        return Tensor(src.shape, kwargs.get("output_format", "b4")), lambda: None, {}

    scope = dict(
        HERE=HERE,
        ROOT=ROOT,
        __file__=str(HERE / filename),
        Path=Path,
        math=math,
        struct=struct,
        json=json,
        ttnn=ttnn,
        calls=calls,
        PREFIX="experiments/sdpa-l2/bfp4-lofi-v2/fullchip/",
        PRIVATE="experiments/sdpa-l2/bfp4-lofi-v2/vtransposed/",
        COMPUTE="experiments/sdpa-l2/bfp4-lofi-v2/vtransposed/compute.cpp",
        PREP=NS(build=quantizer),
        B4_PREP=NS(build=quantizer),
    )
    source = ast.parse((HERE / filename).read_text())
    source.body = [
        node
        for node in source.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("format_info", "build_transpose", "build", "source_files")
    ]
    exec(compile(source, str(HERE / filename), "exec"), scope)
    return scope


class VTransposeLayout(unittest.TestCase):
    def test_tile_mapping(self):
        # Symbolic (head, token, channel) identities check both tile grid AND
        # within-tile transpose; every one of each chunk's 64 slots is unique.
        for nt in (32, 1024, 8192):
            for head in (0, 2):
                for chunk in (0, nt // 16 - 1):
                    slots = set()
                    for p in range(64):
                        d, k = divmod(p, 16)
                        page = head * 4 * nt + d * nt + chunk * 16 + k
                        slot = k * 4 + d
                        slots.add(slot)
                        self.assertEqual(page // (4 * nt), head)
                        self.assertEqual((page % (4 * nt)) // nt, d)
                        self.assertEqual(page % nt, chunk * 16 + k)
                        for row, col in ((0, 0), (15, 16), (31, 31)):
                            # Physical VT[row=channel,col=token], then PV T.
                            original = ((chunk * 16 + k) * 32 + row, d * 32 + col)
                            recovered = ((page % nt) * 32 + row, ((page % (4 * nt)) // nt) * 32 + col)
                            self.assertEqual(original, recovered)
                    self.assertEqual(slots, set(range(64)))

    def test_descriptors(self):
        baseline = environment("grid7_fullchip.py")
        candidate = environment("Vtransposed_fullchip.py")
        inputs = [Tensor([1, 2, 1024, 128]) for _ in range(3)]
        for destination, denom_only in (("main_bf16", False), ("fast_bf16", False), ("fast_bf16", True)):
            for formats in ("b8_b8", "b4_b8", "b8_b4", "b4_b4"):
                for grid7 in (False, True):
                    args = NS(
                        destination=destination,
                        denom_only=denom_only,
                        kv_formats=formats,
                        length=1024,
                        heads=2,
                        cores=4,
                        check_preprocess=False,
                        read_barrier_tiles=2,
                        grid7_exp=grid7,
                        v_transposed=False,
                    )
                    with contextlib.redirect_stdout(io.StringIO()):
                        control = baseline["build"](Device(), args, inputs)
                        ordinary = candidate["build"](Device(), args, inputs)
                        args.v_transposed = True
                        rotated = candidate["build"](Device(), args, inputs)
                    a, b, c = control[-1], ordinary[-1], rotated[-1]
                    for field in ("cb_audit", "input_slots", "assignments", "q_chunk", "k_chunk", "head_dim"):
                        self.assertEqual(a[field], b[field])
                        self.assertEqual(a[field], c[field])
                    self.assertEqual(a["defines"], b["defines"])
                    self.assertEqual(c["defines"], dict(a["defines"], SDPA_V_TRANSPOSED="1"))
                    self.assertEqual(c["input_slots"], 2)
                    self.assertEqual(rotated[1][2].shape, (1, 2, 128, 1024))
                    self.assertEqual(ordinary[1][2].shape, (1, 2, 1024, 128))
                    self.assertEqual(set(rotated[4].stages), {"quantization", "v_transpose"})
                    rotated[3]()
                    desc = candidate["calls"][-1]
                    self.assertIn(("SDPA_V_TRANSPOSED", "1"), desc.kernels[0].defines)
                    self.assertEqual(
                        desc.kernels[2].compile_time_args,
                        [2, struct.unpack("I", struct.pack("f", 1 / math.sqrt(128)))[0], 8],
                    )

    def test_source_paths(self):
        candidate = environment("Vtransposed_fullchip.py")
        for destination in ("main_bf16", "fast_bf16"):
            paths = candidate["source_files"](destination)
            self.assertGreater(len(paths), 40)
            self.assertTrue(all(path.is_file() for path in paths))
            self.assertIn(HERE / "vtransposed/pv_transpose.hpp", paths)
            self.assertIn(ROOT / "tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py", paths)
            self.assertIn(HERE.parent / "frontier-accuracy-v1/run.py", paths)


if __name__ == "__main__":
    unittest.main()
