# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``sparse_sdpa_tiled`` without a device: the attended-set identity between the block-shared kernel's rule and the
slab's expansion (synthetic ids at every position class, and the captured slab selections when
``QWEN38_QSA_BLOCK_CAPTURES`` names the directory of ``slab*-qsa-blocks.pt`` files), the reader / writer emulation's
invariants (union bound, seed in chunk 0, bands = attended set), the oracle against a per-row loop, the kernel
argument contract (every named compile-time arg a kernel reads is one the builder passes, and vice versa), the L1
budget, and the registry entry."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.ttnn import fused
from models.demos.blackhole.qwen38_flash_next.ttnn import qsa as qsa_module
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import sparse_sdpa_tiled as sst
from models.demos.blackhole.qwen38_flash_next.ttnn.fused.sparse_sdpa_tiled import synthetic as gen

KERNELS = Path(sst.__file__).parent / "kernels"
NAMED = re.compile(r'get_named_compile_time_arg_val\("([A-Z0-9_]+)"\)')
CAPTURES = os.environ.get("QWEN38_QSA_BLOCK_CAPTURES")


def _valid_tokens(row: torch.Tensor, T: int) -> set[int]:
    return {int(t) for t in row.tolist() if int(t) != sst.MASKED_INDEX and int(t) < T}


@pytest.mark.parametrize("P", [0, 4, 2048, 6144, 28672])
@pytest.mark.parametrize("pattern", ["shared", "random", "clustered", "ranges"])
def test_attended_set_is_todays_expansion(P: int, pattern: str) -> None:
    T = max(P + 64, 8192) if pattern == "ranges" else P + 64
    inputs = gen.make_inputs(S=64, T=T, IDS=64, H=1, P=P, pattern=pattern, seed=P + 1)
    mask = sst.attended_mask(inputs.block_ids, inputs.positions, inputs.T)
    today = sst.expand_today(inputs.block_ids, inputs.positions)
    for j in range(64):
        assert {int(t) for t in torch.nonzero(mask[j]).flatten().tolist()} == _valid_tokens(today[j], inputs.T)
        p = int(inputs.positions[j])
        assert int(mask[j].sum()) == 4 * min(64, (p + 1) // 4) + (p + 1) % 4


def test_expansion_matches_the_slab_emulation() -> None:
    """``expand_today`` = the device rule of derive_qsa_chunk_inputs (row_keep_bits / row_fill on the template)."""

    P, S, blocks = 2048, 2048, 8192
    emulated = qsa_module.emulate_qsa_chunk_inputs(P, allocated_compressed_blocks=blocks, rows=S)
    keep = emulated["row_keep_bits"].reshape(S, -1).to(torch.int64)
    fill = emulated["row_fill"].reshape(S, -1).to(torch.int64)
    gen_ = torch.Generator().manual_seed(0)
    ids = gen.select_rows(S, P + S, 512, P, "random", gen_)
    constants = qsa_module.qsa_chunk_constant_rows(blocks, S)
    offsets = constants["block_offsets_rows"].reshape(S, -1).to(torch.int64)
    pad = constants["sentinel_pad_rows"].reshape(S, -1).to(torch.int64)
    expanded = (ids << 2).repeat_interleave(4, dim=1) + offsets
    template = torch.cat([expanded, pad], dim=1)
    device_rule = (template & keep) | fill
    assert torch.equal(device_rule & 0xFFFFFFFF, sst.expand_today(ids, P + torch.arange(S)) & 0xFFFFFFFF)


def test_tile_plan_invariants_and_bands_give_the_attended_set() -> None:
    for pattern, P, T in (
        ("shared", 0, 1024),
        ("random", 512, 2048),
        ("ranges", 4096, 8192),
        ("clustered", 28672, 30720),
    ):
        inputs = gen.make_inputs(S=32, T=T, IDS=64, H=1, P=P, pattern=pattern, seed=3)
        mask = sst.attended_mask(inputs.block_ids, inputs.positions, T)
        for tile in range(2):
            plan = sst.tile_plan(inputs.block_ids, inputs.positions, tile)
            assert plan.U <= sst.DEFAULT.union_max(64)
            assert len(plan.union) == len(set(plan.union))
            assert plan.union[len(plan.seeds) :] == sorted(plan.union[len(plan.seeds) :])
            assert plan.seeds == sorted(plan.seeds) and len(plan.seeds) >= 1
            chunk0 = set(plan.union[: sst.DEFAULT.chunk_blocks])
            for qi in range(16):
                members = {plan.union[s] for s, m in enumerate(plan.member) if (m >> qi) & 1}
                own = {plan.complete[qi]} if plan.tail[qi] else set()
                assert (members | own) & chunk0, f"row {qi} has no attended block in chunk 0"
                assert plan.complete[qi] not in members
            assert torch.equal(sst.tile_attended_from_bands(plan, T=T), mask[plan.t0 : plan.t0 + 16])


def test_reference_matches_a_per_row_loop() -> None:
    inputs = gen.make_inputs(S=16, T=512, IDS=16, H=2, P=128, pattern="random", seed=4)
    ref = sst.reference_fp32(inputs.q, inputs.kv, inputs.block_ids, inputs.positions)
    mask = sst.attended_mask(inputs.block_ids, inputs.positions, inputs.T)
    k, v = inputs.kv[0, 0, :, 256:], inputs.kv[0, 0, :, :256]
    for h in range(2):
        for j in range(16):
            idx = torch.nonzero(mask[j]).flatten()
            s = (k[idx] @ inputs.q[0, h, j]) / 16.0
            expect = torch.softmax(s, 0) @ v[idx]
            assert torch.allclose(ref[0, h, j], expect, atol=1e-5, rtol=1e-4)


def test_kernel_named_args_match_the_builder() -> None:
    src = sst.__file__
    text = Path(src).read_text(encoding="utf-8")
    for kernel, dict_name in (
        ("reader.cpp", "reader_named"),
        ("writer.cpp", "writer_named"),
        ("compute.cpp", "compute_named"),
    ):
        used = set(NAMED.findall((KERNELS / kernel).read_text(encoding="utf-8")))
        block = text[text.index(f"{dict_name} = {{") : text.index("}", text.index(f"{dict_name} = {{"))]
        provided = set(re.findall(r'"([A-Z0-9_]+)":', block))
        if "**shared" in block:
            provided |= set(
                re.findall(
                    r'"([A-Z0-9_]+)":', text[text.index("shared = {") : text.index("}", text.index("shared = {"))]
                )
            )
        assert (
            used == provided
        ), f"{kernel}: reads {sorted(used - provided)} unprovided, builder passes {sorted(provided - used)} unread"
    common = (KERNELS / "common.h").read_text(encoding="utf-8")
    assert f"MSG_WORDS = {sst.MSG_WORDS}" in common
    assert "MASK_FLOOR_BF16 = 0xFF7Fu" in common and struct_bits(sst.MASK_FLOOR) == 0xFF7F


def struct_bits(value: float) -> int:
    import struct

    return struct.unpack("<I", struct.pack("<f", value))[0] >> 16


def test_l1_budget_of_the_slab_shapes() -> None:
    """Config A (TQ 16, CB 64, 6 heads, 32k and 256k caches) fits in both state formats; the design's table is 1210 KB
    with fp32 state and 1034 KB with bf16 (the band template adds 16 KB to each)."""

    class T:
        def __init__(self, shape, dtype, layout):
            self.shape, self.dtype, self.layout = shape, dtype, layout

    import ttnn

    rm = ttnn.ROW_MAJOR_LAYOUT
    for context in (32768, 262144):
        q = T((1, 6, 2048, 256), ttnn.bfloat16, rm)
        kv = T((1, 1, context, 512), ttnn.bfloat16, rm)
        ids = T((1, 1, 2048, 512), ttnn.uint32, rm)
        pos = T((1, 1, 1, 2048), ttnn.uint32, rm)
        for config, ceiling in ((sst.Config(fp32_state=True), 1250 * 1024), (sst.DEFAULT, 1080 * 1024)):
            g = sst.geometry(q, kv, ids, pos, config)
            assert g["num_tiles"] == 128 and g["Sqt"] == 3 and g["qsb"] == 3 and g["Skt"] == 8
            assert sst.l1_bytes(g, config) <= min(ceiling, sst.L1_BUDGET_BYTES), (
                context,
                config.fp32_state,
                sst.l1_bytes(g, config),
            )
        # Config B at CB = 64 (TQ 32) does not fit: the builder refuses it (the design's L1 table).
        wide = sst.Config(tile_queries=32, chunk_blocks=64)
        assert sst.l1_bytes(sst.geometry(q, kv, ids, pos, wide), wide) > sst.L1_BUDGET_BYTES


def test_grid_config_and_shape_admission() -> None:
    """16-query tiles where S / 16 tiles fit the grid one per core (the 130-core p150: 128 tiles), else 32-query
    tiles with 32-block chunks (a 110-core die: 64 tiles); the shape admission is the geometry + L1 contract."""

    assert sst.config_for_grid(2048, 130) == sst.DEFAULT
    small = sst.config_for_grid(2048, 110)
    assert (small.tile_queries, small.chunk_blocks) == (32, 32) and small.fp32_state == sst.DEFAULT.fp32_state
    assert sst.config_for_grid(4096, 130).tile_queries == 32  # 256 tiles of 16 would double up on every core
    for context in (32768, 262144):
        assert sst.admits_shapes(6, 2048, context, 512, 256, 512)
        assert sst.admits_shapes(6, 2048, context, 512, 256, 512, small)
    assert not sst.admits_shapes(6, 2048, 32768, 512, 256, 512, sst.Config(tile_queries=32, chunk_blocks=64))  # L1
    assert not sst.admits_shapes(6, 2040, 32768, 512, 256, 512)  # S not whole 16-query tiles
    assert not sst.admits_shapes(6, 2048, 32770, 512, 256, 512)  # T not whole blocks
    assert not sst.admits_shapes(5, 2048, 32768, 512, 256, 512)  # 80 rows per tile: not whole tile rows


def test_fidelity_switch_resolves_from_the_environment(expect_error) -> None:
    import ttnn

    # Served default = HiFi2 / approximate / bf16 DEST on the bf16 state (the line gate of 2026-09-26); hifi4 = the
    # slab's compute config (the kernel's validation config) as the switch.
    hifi2 = sst.model_config({})
    assert hifi2 == sst.HIFI2 and hifi2 == sst.model_config({sst.FIDELITY_ENV: "hifi2"})
    assert (hifi2.fidelity, hifi2.approx, hifi2.fp32_dest) == (ttnn.MathFidelity.HiFi2, True, False)
    assert hifi2.tile_queries == sst.DEFAULT.tile_queries and hifi2.fp32_state == sst.DEFAULT.fp32_state is False
    hifi4 = sst.model_config({sst.FIDELITY_ENV: "hifi4"})
    assert hifi4 == sst.DEFAULT and (hifi4.fidelity, hifi4.approx, hifi4.fp32_dest) == (
        ttnn.MathFidelity.HiFi4,
        False,
        True,
    )
    assert sst.config_for_grid(2048, 110, hifi2).fidelity == ttnn.MathFidelity.HiFi2
    with expect_error(ValueError):
        sst.model_config({sst.FIDELITY_ENV: "lofi"})


def test_registry_entry() -> None:
    entry = fused.kernel("sparse_sdpa_tiled")
    assert entry.tolerance == fused.COMPONENT
    # Serves by default since the line gate of 2026-09-26 (a COMPONENT kernel needs its recorded proof to be listed).
    assert entry.default_on and entry.component_proof and "2026-09-26" in entry.component_proof
    assert "sparse_sdpa_tiled" in fused.default_names()
    assert (
        fused.resolve_admitted("sparse_sdpa_tiled", {}).kernel is entry
    )  # an AdmittedStep: the chain outside the contract
    assert fused.resolve("sparse_sdpa_tiled", {fused.OFF_ENV: "sparse_sdpa_tiled"}) is sst.sparse_sdpa_tiled_composed
    assert entry.admits is sst.admits and entry.fused is sst.sparse_sdpa_tiled
    assert "sparse_sdpa" in entry.replaces


@pytest.mark.skipif(not CAPTURES, reason="set QWEN38_QSA_BLOCK_CAPTURES to the directory of slab*-qsa-blocks.pt")
def test_captured_slab_selections_keep_the_attended_set_and_the_union_bound() -> None:
    files = sorted(Path(CAPTURES).rglob("slab*-qsa-blocks.pt"))
    assert files, f"no captures under {CAPTURES}"
    for path in files:
        record = torch.load(path)
        P, rows = int(record["offset"]), int(record["rows"])
        positions = P + torch.arange(rows)
        for layer, ids in sorted(record["block_ids"].items())[:2]:
            ids = ids.reshape(1, 1, rows, -1)
            T = P + rows
            for start in (0, rows - 64):
                block = ids[:, :, start : start + 64]
                pos = positions[start : start + 64]
                mask = sst.attended_mask(block, pos, T)
                today = sst.expand_today(block, pos)
                for j in range(64):
                    assert {int(t) for t in torch.nonzero(mask[j]).flatten().tolist()} == _valid_tokens(today[j], T)
            for tile in (0, rows // 16 - 1):
                plan = sst.tile_plan(ids, positions, tile)
                assert plan.U <= sst.DEFAULT.union_max(ids.shape[-1])
