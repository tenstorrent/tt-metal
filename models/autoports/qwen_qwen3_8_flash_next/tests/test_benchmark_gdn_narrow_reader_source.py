# SPDX-License-Identifier: Apache-2.0
"""CPU-only reconstruction and exhaustive selected-word/alignment proof."""

import ast
import hashlib
import types
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = HERE.parent
REPO = MODEL.parents[2]


def functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in selected} != set(names):
        raise ValueError("Missing pure source transformer")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)


def reconstruct():
    production = ast.parse((MODEL / "tt/multichip_decoder.py").read_text())
    original = next(
        ast.literal_eval(n.value)
        for n in production.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_LOCAL_PREFILL_READER" for t in n.targets)
    )
    ns = {}
    functions(MODEL / "tt/multichip_decoder.py", ["_broadcast_recurrence_reader"], ns)
    functions(MODEL / "tt/gdn_prefill_qk.py", ["_replace_once", "_make_reader"], ns)
    ns["baseline"] = types.SimpleNamespace(
        _RM_QK_READER=ns["_make_reader"](ns["_broadcast_recurrence_reader"](original))
    )
    functions(HERE / "benchmark_gdn_half_columns_candidate.py", ["replace", "reader_source"], ns)
    ns["qualified"] = types.SimpleNamespace(READER=ns["reader_source"](), replace=ns["replace"])
    resident = ast.parse((HERE / "benchmark_gdn_resident_candidate.py").read_text())
    for node in resident.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "READER" for t in node.targets):
            exec(compile(ast.Module(body=[node], type_ignores=[]), "resident_reader", "exec"), ns)
    functions(HERE / "benchmark_gdn_narrow_reader_candidate.py", ["replace_once", "transform_reader"], ns)
    return ns["READER"], ns["transform_reader"], ns["transform_reader"](ns["READER"])


def off(row, col):
    return (row // 16) * 512 + (col // 16) * 256 + (row % 16) * 16 + col % 16


class NarrowReaderTests(unittest.TestCase):
    def test_real_reader_reconstruction_and_only_operand_transfer_changes(self):
        resident, _, candidate = reconstruct()
        for pattern in (
            "noc.async_read_barrier();",
            "noc.async_write_barrier();",
            "cb_push_back",
            "cb_pop_front",
            "cb_reserve_back",
            "cb_wait_front",
        ):
            self.assertEqual(resident.count(pattern), candidate.count(pattern))
        self.assertEqual(candidate.count("CoreLocalMem<uint32_t>(scratch),64,"), 3)
        self.assertEqual(candidate.count("CoreLocalMem<uint32_t>(scratch),512,"), 2)
        self.assertNotIn("CoreLocalMem<uint32_t>(scratch),4096,", candidate)
        for pattern in (
            "if(token==0)",
            "if(token+1==length)",
            "dst[f*512+256+j]=poison",
            "dst[tile*1024+off(0,c+16)]=poison",
        ):
            self.assertEqual(resident.count(pattern), candidate.count(pattern))
        tree = ast.parse((HERE / "benchmark_gdn_narrow_reader_candidate.py").read_text())
        for key in ("COMPUTE", "LAUNCHER_SOURCE"):
            node = next(
                n
                for n in tree.body
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == key for t in n.targets)
            )
            self.assertEqual(ast.unparse(node.value), f"resident.{key}")

    def test_every_v_and_scalar_word_with64B_alignment(self):
        for length in (128, 256, 512):
            for token in range(length):
                for head in range(12):
                    scalar = 4 * off(token % 32, 0)
                    self.assertEqual(scalar % 64, 0)
                    self.assertLessEqual(scalar + 64, 4096)
                    self.assertEqual(scalar + 4 * head, 4 * off(token % 32, head))
                    for half in range(2):
                        value = 4 * off(head, half * 16)
                        self.assertEqual(value % 64, 0)
                        self.assertLessEqual(value + 64, 4096)
                        for col in range(16):
                            self.assertEqual(value + 4 * col, 4 * off(head, col + half * 16))

    def test_blackhole_alignment_and_source_drift(self):
        path = REPO / "tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h"
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            "47fbf6f1be529f34f261d6c3567f4e8c83763fc0b50b5ebf9678f12973705829",
        )
        self.assertIn("#define NOC_DRAM_READ_ALIGNMENT_BYTES 64", path.read_text())
        source, transform, _ = reconstruct()
        with self.assertRaises(ValueError):
            transform(source.replace("src[off(h%32,c+half*16)]", "src[0]"))

    def test_probe_has_qualified_control_all_lengths_and_no_global_patch(self):
        text = (HERE / "benchmark_gdn_narrow_reader_probe.py").read_text()
        ast.parse(text)
        self.assertIn('VARIANTS = ("current_rm", "resident_control", "narrow_zero", "narrow_nan")', text)
        self.assertIn('default="128,256,512"', text)
        self.assertIn("candidate.recurrence(name)", text)
        self.assertIn("trace_released=True", text)
        candidate = (HERE / "benchmark_gdn_narrow_reader_candidate.py").read_text()
        self.assertIn("_namespace = dict(vars(baseline)", candidate)
        self.assertNotIn("setattr(resident", candidate)


if __name__ == "__main__":
    unittest.main()
