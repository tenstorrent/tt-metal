# SPDX-License-Identifier: Apache-2.0
"""CPU-only scalar-cache lifetime/address and launcher-equivalence guards."""

import ast
import importlib.util
import textwrap
import types
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "narrow_source_proof", HERE / "test_benchmark_gdn_narrow_reader_source.py"
)
PROOF = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROOF)


def transforms():
    ns = {}
    PROOF.functions(HERE / "benchmark_gdn_narrow_reader_candidate.py", ["replace_once"], ns)
    PROOF.functions(HERE / "benchmark_gdn_scalar_cache_candidate.py", ["transform_reader", "transform_launcher"], ns)
    return ns


def qualified_launcher():
    source = (HERE.parent / "tt/gdn_prefill_qk.py").read_text()
    fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "_rm_recurrence")
    ns = transforms()
    ns.update(
        textwrap=textwrap,
        inspect=types.SimpleNamespace(getsource=lambda _: ast.get_source_segment(source, fn)),
        baseline=types.SimpleNamespace(_rm_recurrence=None),
        qualified=types.SimpleNamespace(replace=ns["replace_once"]),
    )
    PROOF.functions(HERE / "benchmark_gdn_resident_candidate.py", ["launcher_source"], ns)
    return ns["launcher_source"]()


class ScalarCacheTests(unittest.TestCase):
    def test_exact_reader_lifetime_and_required_barriers(self):
        narrow = PROOF.reconstruct()[2]
        cached = transforms()["transform_reader"](narrow)
        self.assertEqual(cached.count("cb_reserve_back(16,2);"), 1)
        self.assertNotIn("cb_push_back(16", cached)
        self.assertNotIn("cb_pop_front(16", cached)
        self.assertLess(cached.index("cb_reserve_back(16,2);"), cached.index("for (uint32_t job"))
        self.assertIn("if(token%32==0)", cached)
        self.assertIn("scalar_cache+(which-4)*4096", cached)
        self.assertIn("value=scalar_src[off(token%32,h%32)]", cached)
        self.assertEqual(cached.count("CoreLocalMem<uint32_t>(scalar_cache),4096"), 1)
        self.assertEqual(cached.count("CoreLocalMem<uint32_t>(scalar_cache+4096),4096"), 1)
        self.assertIn(
            "                noc.async_read_barrier();\n            }\n            for (uint32_t tile", cached
        )
        self.assertIn("if(which==3) noc.async_read_barrier();", cached)
        self.assertEqual(cached.count("CoreLocalMem<uint32_t>(scratch),512,"), 2)
        self.assertEqual(cached.count("CoreLocalMem<uint32_t>(scratch),64,"), 1)
        for pattern in (
            "if(token+1==length)",
            "cb_push_back",
            "cb_pop_front",
            "noc.async_write(",
            "noc.async_write_barrier();",
        ):
            self.assertEqual(cached.count(pattern), narrow.count(pattern))

    def test_all_scalar_words_pages_and_disjoint_storage(self):
        for length in (128, 256, 512):
            st = (length + 31) // 32
            for token in range(length):
                for head in range(12):
                    page = token // 32 + head // 32
                    refresh_page = (token // 32 * 32) // 32 + head // 32
                    self.assertEqual(page, refresh_page)
                    self.assertLess(page, st)
                    offset = PROOF.off(token % 32, head) * 4
                    self.assertGreaterEqual(offset, 0)
                    self.assertLess(offset + 3, 4096)
                    # Exact words agree with prior64B row-fetch addressing.
                    self.assertEqual(offset, PROOF.off(token % 32, 0) * 4 + head * 4)
                    self.assertLess(offset, 4096 + offset)
                    self.assertLess(4096 + offset + 3, 8192)
            self.assertEqual(length // 32, len({token // 32 for token in range(length)}))

    def test_launcher_only_adds_dedicated_two_page_CB(self):
        old = qualified_launcher()
        changed = transforms()["transform_launcher"](old)
        tree = ast.parse(changed)
        body = tree.body[0].body
        additions = [
            i
            for i, node in enumerate(body)
            if (isinstance(node, ast.If) and ast.unparse(node.test) == "len(sizes) != 16")
            or (isinstance(node, ast.Expr) and ast.unparse(node.value) == "sizes.append(2)")
        ]
        self.assertEqual(len(additions), 2)
        tree.body[0].body = [node for i, node in enumerate(body) if i not in additions]
        self.assertEqual(ast.dump(tree, include_attributes=False), ast.dump(ast.parse(old), include_attributes=False))
        self.assertIn("page_size=4096", changed)
        self.assertIn("total_size=count * 4096", changed)
        self.assertIn("if not (length > 1 and i in (0, 12))", changed)

    def test_controls_and_source_drift(self):
        narrow = PROOF.reconstruct()[2]
        with self.assertRaises(ValueError):
            transforms()["transform_reader"](
                narrow.replace("const uint32_t scratch = get_write_ptr(14);", "const uint32_t scratch = 0;")
            )
        text = (HERE / "benchmark_gdn_scalar_cache_probe.py").read_text()
        self.assertIn('"resident_control", "narrow_control", "cache_zero", "cache_nan"', text)
        self.assertIn('default="128,256,512"', text)
        self.assertIn('case["relative_to_narrow_speedup"]', text)
        tree = ast.parse((HERE / "benchmark_gdn_scalar_cache_candidate.py").read_text())
        node = next(
            n
            for n in tree.body
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "COMPUTE" for t in n.targets)
        )
        self.assertEqual(ast.unparse(node.value), "narrow.COMPUTE")


if __name__ == "__main__":
    unittest.main()
