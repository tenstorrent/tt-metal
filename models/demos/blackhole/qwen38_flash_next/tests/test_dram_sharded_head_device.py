# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""QWEN38_MTP_MIXER_SHARDED: the MTP input mixer (fc_embedding 1 row, fc_hidden 4 rows, 2560 x 320 per device) on stock
DRAM-sharded matmuls against the interleaved path, and the LM head (4 chunks, rows 1 and 5) as it runs today.

One die (1x1 mesh): every matmul here is per device.  Same random bf8 weights and activations for old and new; PCC, max abs
diff, per-row argmax agreement over the 31040-wide vocab shard (LM head), and traced us per call (slope of a 4- and a
12-call trace, wall clock).  The LM head chunks already run the DRAM-sharded program (40 activation cores, two readers per bank,
bf8) in the model; the "interleaved" rows time the interleaved-weight ttnn.linear the study assumed, for reference.
Results go to $LH1_RESULTS (jsonl)."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen38_flash_next.tests.moe_experts_support import traced_wall_us
from models.demos.blackhole.qwen38_flash_next.ttnn import decode_matmul as dm
from models.demos.blackhole.qwen38_flash_next.ttnn import mtp as mtp_module

BF8, BF16 = ttnn.bfloat8_b, ttnn.bfloat16
RESULTS = Path(os.environ.get("LH1_RESULTS", "/tmp/lh1.jsonl"))
REPLAYS = int(os.environ.get("LH1_REPLAYS", "100"))
HIDDEN, MIX_N = 2560, 320
CHUNKS = (8000, 8000, 8000, 7040)  # TP=8 LM-head chunk widths (embedding.LM_HEAD_CHUNK_COLUMNS)
DRAM = ttnn.DRAM_MEMORY_CONFIG
PARAMS = [pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)]


def record(row):
    logger.info(json.dumps(row))
    with RESULTS.open("a") as f:
        f.write(json.dumps(row) + "\n")


def us(device, call):
    return round(traced_wall_us(device, call, replays=REPLAYS)[0], 2)


def compute_cfg(device, dtype=BF8):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=getattr(ttnn.MathFidelity, dm.dense_math_fidelity_name(dtype)),
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


def dev(device, t, dtype=BF16, cfg=DRAM):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=cfg)


def host(t):
    return ttnn.to_torch(t).float()


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-30))


def quiet(*tensors):
    for t in tensors:
        ttnn.deallocate(t)


def make_mixer(device, seed=0):
    g = torch.Generator().manual_seed(seed)
    w_emb = (torch.randn(1, 1, HIDDEN, MIX_N, generator=g) * 0.02).to(torch.bfloat16)
    w_hid = (torch.randn(1, 1, HIDDEN, MIX_N, generator=g) * 0.02).to(torch.bfloat16)
    ws_cfg = dm.dram_sharded_weight_memory_config(device, HIDDEN, MIX_N)
    weights = SimpleNamespace(
        fc_embedding=dev(device, w_emb, BF8),
        fc_hidden=dev(device, w_hid, BF8),
        fc_embedding_ws=dev(device, w_emb, BF8, ws_cfg),
        fc_hidden_ws=dev(device, w_hid, BF8, ws_cfg),
        projection_dtype=BF8,
    )
    mixer = object.__new__(mtp_module.Qwen38TTNNMTPInput)
    mixer.mesh_device = device
    mixer.weights = weights
    mixer.projection_compute_config = compute_cfg(device)
    mixer._validate_input = lambda *a, **k: None  # the 1x1 mesh has no TP4/TP8 mesh contract
    return mixer


def mixer_old(mixer, full_embedding, full_hidden):
    pe = ttnn.linear(
        full_embedding,
        mixer.weights.fc_embedding,
        memory_config=DRAM,
        dtype=BF16,
        compute_kernel_config=mixer.projection_compute_config,
    )
    ph = ttnn.linear(
        full_hidden,
        mixer.weights.fc_hidden,
        memory_config=DRAM,
        dtype=BF16,
        compute_kernel_config=mixer.projection_compute_config,
    )
    be = ttnn.repeat(pe, (1, 4, 1, 1), memory_config=DRAM)
    out = ttnn.add(ph, be, memory_config=DRAM)
    quiet(pe, ph, be)
    return out


def mixer_new(mixer, full_embedding, full_hidden):
    temporaries = []
    out = mixer.project_dram_sharded(full_embedding, full_hidden, temporaries)
    for t in temporaries:
        ttnn.deallocate(t)
    return out


@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_mixer_old_vs_new(mesh_device):
    device = mesh_device
    mixer = make_mixer(device)
    row = dict(test="mixer")
    for seed in range(4):
        g = torch.Generator().manual_seed(100 + seed)
        emb = dev(device, torch.randn(1, 1, 1, HIDDEN, generator=g).to(torch.bfloat16))
        hid = dev(device, torch.randn(1, 4, 1, HIDDEN, generator=g).to(torch.bfloat16))
        old = mixer_old(mixer, emb, hid)
        new = mixer_new(mixer, emb, hid)
        for t in (old, new):
            assert tuple(t.shape) == (1, 4, 1, MIX_N), tuple(t.shape)
            assert t.dtype == BF16 and t.layout == ttnn.TILE_LAYOUT and t.memory_config() == DRAM
        a, b = host(old), host(new)
        row[f"seed{seed}"] = dict(pcc=pcc(a, b), max_abs=float((a - b).abs().max()), ref_max=float(a.abs().max()))
        assert row[f"seed{seed}"]["pcc"] > 0.9999, row
        quiet(old, new, emb, hid)
    g = torch.Generator().manual_seed(5)
    emb = dev(device, torch.randn(1, 1, 1, HIDDEN, generator=g).to(torch.bfloat16))
    hid = dev(device, torch.randn(1, 4, 1, HIDDEN, generator=g).to(torch.bfloat16))
    cc = mixer.projection_compute_config

    def lin_emb():
        quiet(ttnn.linear(emb, mixer.weights.fc_embedding, memory_config=DRAM, dtype=BF16, compute_kernel_config=cc))

    def lin_hid():
        quiet(ttnn.linear(hid, mixer.weights.fc_hidden, memory_config=DRAM, dtype=BF16, compute_kernel_config=cc))

    act_cfg, prog = mixer._dram_sharded_programs()

    def ws_chain(act, weight):
        a = ttnn.to_memory_config(act, act_cfg)
        o = ttnn.linear(
            a,
            weight,
            dtype=BF16,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=prog,
            compute_kernel_config=cc,
        )
        d = ttnn.to_memory_config(o, DRAM)
        quiet(a, o, d)

    def ws_emb():
        ws_chain(emb, mixer.weights.fc_embedding_ws)

    hid_rows = ttnn.reshape(hid, (1, 1, 4, HIDDEN))

    def ws_hid():  # without the reshape of the activation
        ws_chain(hid_rows, mixer.weights.fc_hidden_ws)

    a_emb = ttnn.to_memory_config(emb, act_cfg)
    a_hid = ttnn.to_memory_config(hid_rows, act_cfg)

    def kernel_only(act, weight):
        def call():
            quiet(
                ttnn.linear(
                    act,
                    weight,
                    dtype=BF16,
                    memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                    program_config=prog,
                    compute_kernel_config=cc,
                )
            )

        return call

    def section_old():
        quiet(mixer_old(mixer, emb, hid))

    def section_new():
        quiet(mixer_new(mixer, emb, hid))

    row.update(
        old_fc_embedding_us=us(device, lin_emb),
        old_fc_hidden_us=us(device, lin_hid),
        new_fc_embedding_chain_us=us(device, ws_emb),
        new_fc_hidden_chain_us=us(device, ws_hid),
        new_fc_embedding_kernel_us=us(device, kernel_only(a_emb, mixer.weights.fc_embedding_ws)),
        new_fc_hidden_kernel_us=us(device, kernel_only(a_hid, mixer.weights.fc_hidden_ws)),
        old_section_us=us(device, section_old),  # 2 linears + repeat + add
        new_section_us=us(device, section_new),  # shards + 2 linears + to-DRAM + add + 2 reshapes
    )
    record(row)


@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_mixer_core_sweep(mesh_device):
    """Kernel-only fc_hidden / fc_embedding us for other activation grids and K blocks (information; not asserted)."""
    device = mesh_device
    mixer = make_mixer(device)
    cc = mixer.projection_compute_config
    out = {}
    tiles = dm.bank_tiles(device, HIDDEN, MIX_N)
    for cores in (5, 8, 10, 20):
        kt = HIDDEN // 32 // cores
        for block in sorted({w for w in range(1, 17) if kt % w == 0}, reverse=True)[:3]:
            grid = ttnn.CoreGrid(x=cores, y=1) if cores <= 8 else ttnn.CoreGrid(x=8, y=cores // 8 + (cores % 8 > 0))
            if cores > 8 and cores % 8:
                grid = (
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cores - 1, 0))})
                    if cores <= 13
                    else None
                )
            if grid is None:
                continue
            try:
                act_cfg = ttnn.create_sharded_memory_config(
                    (32, HIDDEN // cores),
                    grid,
                    ttnn.ShardStrategy.WIDTH,
                    ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                prog = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=block, per_core_M=1, per_core_N=tiles, fused_activation=None
                )
                act = ttnn.to_memory_config(dev(device, torch.randn(1, 1, 4, HIDDEN).to(torch.bfloat16)), act_cfg)

                def call():
                    quiet(
                        ttnn.linear(
                            act,
                            mixer.weights.fc_hidden_ws,
                            dtype=BF16,
                            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                            program_config=prog,
                            compute_kernel_config=cc,
                        )
                    )

                out[f"cores{cores}_block{block}"] = us(device, call)
            except Exception as error:  # a grid or block the program rejects
                out[f"cores{cores}_block{block}"] = f"fail: {str(error)[:80]}"
    record(dict(test="mixer_core_sweep", us=out))


@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("rows", [1, 5])
def test_lm_head_rows(mesh_device, rows):
    device = mesh_device
    workers = dm.validate_decode_dram_workers(dm.DEFAULT_WORKERS_PER_DRAM_BANK)
    cc = compute_cfg(device)
    g = torch.Generator().manual_seed(11)
    configs = [
        dm.dram_sharded_matmul_configs(device, HIDDEN, n, num_cores=40, num_workers_per_dram_bank=workers)
        for n in CHUNKS
    ]
    host_w = [(torch.randn(1, 1, HIDDEN, n, generator=g) * 0.02).to(torch.bfloat16) for n in CHUNKS]
    w_ws = [
        dev(device, w, BF8, dm.dram_sharded_weight_memory_config(device, HIDDEN, n, num_workers_per_dram_bank=workers))
        for w, n in zip(host_w, CHUNKS)
    ]
    w_il = [dev(device, w, BF8) for w in host_w]
    hidden = dev(device, torch.randn(1, 1, rows, HIDDEN, generator=g).to(torch.bfloat16))
    act_cfg = configs[0][0]

    def head_ws(hid):
        a = ttnn.to_memory_config(hid, act_cfg)
        outs = []
        for w, (_, prog) in zip(w_ws, configs):
            o = ttnn.linear(
                a,
                w,
                dtype=BF16,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                program_config=prog,
                compute_kernel_config=cc,
            )
            outs.append(ttnn.to_memory_config(o, DRAM))
            quiet(o)
        quiet(a)
        full = ttnn.concat(outs, dim=3, memory_config=DRAM)
        quiet(*outs)
        return full

    def head_il(hid):
        outs = [ttnn.linear(hid, w, dtype=BF16, memory_config=DRAM, compute_kernel_config=cc) for w in w_il]
        full = ttnn.concat(outs, dim=3, memory_config=DRAM)
        quiet(*outs)
        return full

    ref, got = head_il(hidden), head_ws(hidden)
    a, b = host(ref).reshape(-1, sum(CHUNKS))[:rows], host(got).reshape(-1, sum(CHUNKS))[:rows]
    row = dict(
        test="lm_head",
        rows=rows,
        pcc=pcc(a, b),
        max_abs=float((a - b).abs().max()),
        argmax_agree=[int(a[r].argmax() == b[r].argmax()) for r in range(rows)],
    )
    quiet(ref, got)
    act_one = ttnn.to_memory_config(hidden, act_cfg)
    for i, n in enumerate(CHUNKS[:1] + CHUNKS[3:]):
        j = 0 if i == 0 else 3

        def k_ws():
            quiet(
                ttnn.linear(
                    act_one,
                    w_ws[j],
                    dtype=BF16,
                    memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                    program_config=configs[j][1],
                    compute_kernel_config=cc,
                )
            )

        def k_il():
            quiet(ttnn.linear(hidden, w_il[j], dtype=BF16, memory_config=DRAM, compute_kernel_config=cc))

        row[f"chunk{n}_dram_sharded_us"] = us(device, k_ws)
        row[f"chunk{n}_interleaved_us"] = us(device, k_il)
    row["full_head_dram_sharded_us"] = us(device, lambda: quiet(head_ws(hidden)))
    row["full_head_interleaved_us"] = us(device, lambda: quiet(head_il(hidden)))
    record(row)
    assert row["pcc"] > 0.9999
