"""Verify the narrow QSA B1 precision delta supporting retained GDN/B>1 evidence."""

import ast
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/multichip_decoder"
OLD = "457af56586296a43f012bc975c3f7e1906ecb5f8a5592a82c6731e83ff04d9c4"


def main():
    old_path = DOC / "selected_v4_mesh_l0/source.py.txt"
    new_path = ROOT / "tt/multichip_decoder.py"
    before, after = old_path.read_text(), new_path.read_text()
    assert hashlib.sha256(before.encode()).hexdigest() == OLD
    old, new = ast.parse(before), ast.parse(after)
    old_qsa = next(n for n in old.body if isinstance(n, ast.ClassDef) and n.name == "ParallelQSA")
    new_qsa = next(n for n in new.body if isinstance(n, ast.ClassDef) and n.name == "ParallelQSA")
    old.body.remove(old_qsa)
    new.body.remove(new_qsa)
    assert ast.dump(old) == ast.dump(new), "Changes outside QSA invalidate GDN reuse"
    new_methods = {n.name: n for n in new_qsa.body if isinstance(n, ast.FunctionDef)}
    for name, phase in [("prefill_forward", False), ("decode_forward", True)]:
        expected = ast.parse(
            f"def {name}(self, hidden_states, **kwargs):\n    self._decode_phase = {phase}\n    return super().{name}(hidden_states, **kwargs)\n"
        ).body[0]
        assert ast.dump(new_methods[name]) == ast.dump(expected)
        new_qsa.body.remove(new_methods[name])
    build = new_methods["build"]
    removed = []
    for node in list(build.body):
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) in (
            "obj.decode_projection",
            "obj._decode_phase",
        ):
            removed.append(ast.unparse(node))
            build.body.remove(node)
    assert removed == [
        "obj.decode_projection = ttnn.typecast(obj.packed_projection, ttnn.bfloat4_b) if batch_size == 1 else None",
        "obj._decode_phase = False",
    ]
    linear = new_methods["_linear"]
    branch = linear.body.pop(0)
    expected = ast.parse(
        "if self._decode_phase and self.batch_size == 1 and weight is self.packed_projection:\n    return self.ops.linear(x, self.decode_projection, compute_kernel_config=self.projection_compute)\n"
    ).body[0]
    assert ast.dump(branch) == ast.dump(expected)
    assert ast.dump(old_qsa) == ast.dump(new_qsa), "Unexpected QSA changes invalidate B>1 reuse"
    diff = "".join(
        difflib.unified_diff(
            before.splitlines(True), after.splitlines(True), fromfile="qualified_v4", tofile="selected_v6"
        )
    )
    (DOC / "source_reuse.diff").write_text(diff)
    result = dict(
        status="passed",
        prior_source_sha256=OLD,
        selected_source_sha256=hashlib.sha256(after.encode()).hexdigest(),
        prior_source=str(old_path.relative_to(ROOT)),
        selected_source=str(new_path.relative_to(ROOT)),
        diff="source_reuse.diff",
        unchanged_paths=[
            "All GDN and GDN+PLE methods and dependencies",
            "QSA B>1 BF8 weights and HiFi2 multi-token prefill / LoFi singleton compute",
        ],
        changed_path="Only explicit B1 decode uses a separate BF8-to-BF4 packed projection at LoFi. All prefill and B>1 paths retain baseline BF8 policy; wrappers bind phase per invocation.",
        reuse_scope="selected_v4 GDN/B>1 correctness, context, trace, watcher, direct cache/layout, performance and profiles remain applicable. QSA B1 phase transition/mixed-cache path requires new qualification and is not inherited from v4.",
    )
    (DOC / "source_reuse.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
