# SPDX-License-Identifier: Apache-2.0
"""CPU-only allocation and addressing checks; does not import/open TTNN."""
import ast
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def test_initial_allocations_are_physical_not_batch_context():
    source = ast.parse((ROOT / "tt/qsa.py").read_text())
    cls = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "QSA")
    constructor = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "from_state_dict"
    )
    statements = constructor.body
    begin = next(
        i
        for i, node in enumerate(statements)
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "pages" for t in node.targets)
    )
    end = next(i for i in range(begin, len(statements)) if isinstance(statements[i], ast.Return))
    code = compile(
        ast.fix_missing_locations(ast.Module(body=statements[begin:end], type_ignores=[])), "<cache-init>", "exec"
    )

    class Shape:
        def __init__(self, shape):
            self.shape = tuple(shape)

        def float(self):
            return self

        def reshape(self, *shape):
            return Shape(shape)

    torch = SimpleNamespace(zeros=lambda *shape, **_: Shape(shape), arange=lambda n: Shape((n,)), bfloat16="bf16")
    tt = SimpleNamespace(ROW_MAJOR_LAYOUT="rm", float32="fp32")
    for batch in (1, 32):
        context, physical = 262144, 279552
        obj = SimpleNamespace(paged_pooled=True, max_context=context, blocks=context // 4, chunk_size=128)
        env = dict(
            obj=obj,
            batch_size=batch,
            attention_pool_tokens=physical,
            page_size=32,
            allocate_kv_cache=False,
            allocate_block_ids=False,
            upload=lambda x, *_, **__: x,
            torch=torch,
            ttnn=tt,
            max_requests=batch * 2,
        )
        exec(code, env)
        assert obj.pooled_bank is None
        assert obj.block_ids is None
        assert obj.pooled_keys.shape == (1, 1, context // 4, 128)
        assert obj.pooled_cache.shape == (1, 1, physical // 4, 128)
        assert obj.raw_key_cache.shape == (physical // 32, 1, 32, 128)
        assert not hasattr(obj, "k_cache") and not hasattr(obj, "v_cache")
        assert physical * (512 + 128 * 2 + 128 * 2 // 4) == 232587264


def test_noncontiguous_complete_group_mapping_and_suffix():
    # Row tags represent all 128 BF16 elements, independent of value/norm.
    table = [7, 1, 12, 0, 15]
    pool = [-999] * (16 * 8)
    dense = [-999] * 512
    for position in range(133):
        if position % 4 == 3:
            group = position // 4
            pool[table[position // 32] * 8 + position % 32 // 4] = group
            dense[group] = group
        complete = (position + 1) // 4
        gathered = [pool[table[group // 8] * 8 + group % 8] if group < complete else 0 for group in range(512)]
        assert gathered[:complete] == dense[:complete]
        assert all(x == 0 for x in gathered[complete:])
        if position in (3, 4, 31, 32, 127, 128):
            assert complete == (position + 1) // 4
            # The existing packed expander consumes logical IDs and reaches
            # the same physical page as pooled/raw addressing for every token.
            for group in range(complete):
                for offset in range(4):
                    logical = group * 4 + offset
                    physical = table[logical // 32] * 32 + logical % 32
                    assert physical // 4 == table[group // 8] * 8 + group % 8
