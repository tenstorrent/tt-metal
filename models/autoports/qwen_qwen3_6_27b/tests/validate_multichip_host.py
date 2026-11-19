# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks for local recurrence ownership and distributed norm rank."""

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch


def main():
    root = Path(__file__).parents[1]
    source = root / "tt/multichip_decoder.py"
    tree = ast.parse(source.read_text())
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    env = {}
    exec(compile(ast.Module(body=[functions["_local12_head_partition"]], type_ignores=[]), str(source), "exec"), env)
    partition = env["_local12_head_partition"]
    checked = 0
    for batch in [1, 2, 3, 8, 16, 31, 32]:
        for available in [1, 2, 3, 8, 12, 24, 32, 64, 110]:
            for requested in [None, 1, 2, 3, 8, 12, 24, 32, 64, 110]:
                grouped, chunks = partition(batch, available, requested)
                heads = [h for start, count in chunks for h in range(start, start + count)]
                assert heads == list(range(batch * 12))
                assert len(chunks) <= min(available, requested or available)
                assert max(c for _, c in chunks) - min(c for _, c in chunks) <= 1
                checked += 1
    reader = next(
        ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_LOCAL12_READER" for t in n.targets)
    )
    assert "page=user+head/32" in reader and "user=h/12,hv=h%12,hk=hv/3" in reader
    assert "PACKED?5120:(ROW_MAJOR?1024:(NORMALIZED?4096:2048))" in reader
    assert "PACKED?5120:(ROW_MAJOR?3072:2048)" in reader
    assert "(PACKED?1024:0)+hk*256" in reader and "(PACKED?2048:0)+hv*256" in reader
    for batch in [1, 2, 3, 16, 32]:
        for h in range(batch * 12):
            user, head = divmod(h, 12)
            gate_page = user + head // 32
            assert gate_page == user and gate_page < batch
            assert 0 <= head // 3 < 4
            # Global head ownership must commute with local GQA mapping.
            for rank in range(4):
                assert (rank * 12 + head) // 3 == rank * 4 + head // 3
            assert 2048 + head * 256 + 256 <= 5120
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MultichipDecoder")
    norm = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_norm")

    class Tensor:
        def __init__(self, shape):
            self.shape = tuple(shape)

    def reshape(x, shape):
        return Tensor(shape)

    def pre(x, **kwargs):
        assert len(x.shape) == 4
        return Tensor((*x.shape[:3], 32))

    def post(x, stats, **kwargs):
        assert len(x.shape) == len(stats.shape) == 4 and x.shape[:3] == stats.shape[:3]
        assert stats.shape[-1] == 128
        return x

    fake = SimpleNamespace(
        DRAM_MEMORY_CONFIG=None,
        bfloat16=None,
        to_memory_config=lambda x, m: x,
        reshape=reshape,
        rms_norm_pre_all_gather=pre,
        rms_norm_post_all_gather=post,
    )
    env = {"ttnn": fake}
    exec(compile(ast.Module(body=[norm], type_ignores=[]), str(source), "exec"), env)
    obj = SimpleNamespace(
        policy={},
        ckc=None,
        config=SimpleNamespace(rms_norm_eps=1e-6),
        shard_norms={"norm": None},
        _collective=lambda x: Tensor((*x.shape[:-1], x.shape[-1] * 4)),
    )
    cases = []
    for batch, seq in [(1, 1), (2, 1), (32, 1), (1, 31), (3, 33), (1, 512), (3, 513)]:
        for flat in [False, True]:
            obj.policy["flat_norm"] = flat
            out = env["_norm"](obj, Tensor((batch, seq, 1280)), "norm")
            assert out.shape == (batch, seq, 5120)
            cases.append([batch, seq, flat])
    # Exercise the actual pure-host fixture helper without importing TTNN.
    runner = root / "tests/run_multichip_decoder.py"
    runner_tree = ast.parse(runner.read_text())
    remap_fn = next(n for n in runner_tree.body if isinstance(n, ast.FunctionDef) and n.name == "remap_cache_pages")
    remap_env = {"torch": torch}
    exec(compile(ast.Module(body=[remap_fn], type_ignores=[]), str(runner), "exec"), remap_env)
    remap = remap_env["remap_cache_pages"]
    remap_cases = 0
    for batch in [1, 2, 3, 32]:
        for pages in [1, 2, 16, 17]:
            blocks = batch * pages + 3  # Unused physical IDs must also remain valid.
            table = torch.randperm(blocks, generator=torch.Generator().manual_seed(17))[: batch * pages].reshape(
                batch, pages
            )
            for heads in [1, 4]:
                values = [
                    torch.arange(blocks * heads * 32 * 8).reshape(blocks, heads, 32, 8) + rank * 10000000
                    for rank in range(4)
                ]
                mapped_table, mapped_values = remap(table, values)
                assert mapped_table.dtype == torch.int32
                for original, mapped in zip(values, mapped_values):
                    assert torch.equal(original[table], mapped[mapped_table.long()])
                    assert torch.equal(original.sort(dim=0).values, mapped.sort(dim=0).values)
                # Wrongly changing only the page table must fail the semantic check.
                assert not torch.equal(values[0][table], values[0][mapped_table.long()])
                remap_cases += 1
    result = {
        "pass": True,
        "partition_cases": checked,
        "physical_page_remap_cases": remap_cases,
        "runner_sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
        "norm_rank_cases": cases,
        "gate_stride_batches": [1, 2, 3, 16, 32],
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "scope": "host ownership/rank contracts only; no hardware math, synchronization or trace claim",
    }
    (root / "doc/multichip_decoder/host_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print("HOST_CONTRACTS_PASS", checked, "partition cases")


if __name__ == "__main__":
    main()
