# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""DS41F-0169 replay (tt-blaze postmortem): one chip's exact unified_routed_expert_moe inputs, dumped from the V4.1 prefill
by V41_PREFILL_EXPERT_DUMP (tt/moe/tt_moe.py), re-uploaded to ONE chip and run WS10_REPS times; every run's valid rows are
compared with the first run's and with the production output in the dump.

    WS10_DUMP=/path/moe_l17_chip8_*.pt WS10_REPS=500 pytest -s test_ws10_moe_replay.py
"""

import hashlib
import os

import pytest
import torch
from loguru import logger

import ttnn


def _enum(s, enum):
    return getattr(enum, s.split(".")[-1])


_MESH = os.environ.get("WS10_MESH", "1")
_PARAMS = (
    pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-chip")
    if _MESH == "1"
    else pytest.param(
        tuple(int(v) for v in _MESH.split("x")),
        {"fabric_config": getattr(ttnn.FabricConfig, os.environ.get("WS10_FABRIC", "FABRIC_2D"))},
        id=f"mesh-{_MESH}",
    )
)


@pytest.mark.parametrize("mesh_device, device_params", [_PARAMS], indirect=["mesh_device", "device_params"])
def test_ws10_moe_replay(mesh_device):
    blob = torch.load(os.environ["WS10_DUMP"], weights_only=False)
    reps = int(os.environ.get("WS10_REPS", "100"))
    dev = mesh_device

    def up(t, dtype, layout):
        return ttnn.from_torch(
            t,
            mesh_mapper=ttnn.ReplicateTensorToMesh(dev),
            layout=layout,
            device=dev,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    disp = blob["dispatched"]
    if os.environ.get("WS10_SYNTH", "0") == "1":
        # DS41F-0169 WS10: random data with the dump's exact geometry (same counts / offsets / table / shapes)
        g0 = torch.Generator().manual_seed(int(os.environ.get("WS10_SEED", "0")))
        nz_rows = disp.abs().sum(-1) != 0
        disp = torch.zeros_like(disp)
        disp[nz_rows] = torch.randn(int(nz_rows.sum()), disp.shape[-1], generator=g0).to(disp.dtype)
        for key in ("gate", "up", "down"):
            blob[key] = [torch.randn(w.shape, generator=g0) * 0.02 for w in blob[key]]
        logger.warning("WS10 SYNTH: random activations and weights, dump geometry")
    roll = int(os.environ.get("WS10_ROLL", "0"))
    if roll:
        # DS41F-0169 WS10: rotate local expert 0's valid rows (global 24, rows [0, 610)) by `roll`, so token 580's data
        # lands on another row / M-row core; a flip that follows the data is data / timing, one that stays put is the core.
        n0 = int(blob["counts"].flatten()[int(blob["idx_table"].flatten()[0])])
        disp = disp.clone()
        disp[:n0] = torch.roll(disp[:n0], shifts=roll, dims=0)
        logger.warning(f"WS10 roll: expert 0 rows [0, {n0}) rolled by {roll}: token 580 -> row {(580 + roll) % n0}")
    assert "ROW_MAJOR" in blob["dispatched_layout"], blob["dispatched_layout"]
    tt_disp = up(disp.to(torch.bfloat16), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    u32 = lambda t: up(t.to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    tt_cnt, tt_off, tt_idx = u32(blob["counts"]), u32(blob["offsets"]), u32(blob["idx_table"])
    wdt = _enum(blob["w_dtype"], ttnn.DataType)
    gates = [up(w.float(), wdt, ttnn.TILE_LAYOUT) for w in blob["gate"]]
    ups = [up(w.float(), wdt, ttnn.TILE_LAYOUT) for w in blob["up"]]
    downs = [up(w.float(), wdt, ttnn.TILE_LAYOUT) for w in blob["down"]]
    ck = blob["ckc"]
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=_enum(ck["math_fidelity"], ttnn.MathFidelity),
        math_approx_mode=ck["math_approx_mode"],
        fp32_dest_acc_en=ck["fp32_dest_acc_en"],
        packer_l1_acc=ck["packer_l1_acc"],
        dst_full_sync_en=os.environ.get("WS10_DST_FULL", "0") == "1",
    )
    act = _enum(blob["activation"], ttnn.RoutedExpertActivation)
    assert not any(blob["biases"]), "biases not dumped"

    # valid rows through the expert table (local j -> global g -> counts[g], offsets[g])
    cnt, off = blob["counts"].flatten().long(), blob["offsets"].flatten().long()
    idx = blob["idx_table"].flatten().long()
    rows = disp.reshape(-1, disp.shape[-1]).shape[0]
    valid = torch.zeros(rows, dtype=torch.bool)
    spans = []
    for j, g in enumerate(idx.tolist()):
        c, o = int(cnt[g]), int(off[g])
        if c > 0:
            valid[o : o + c] = True
        spans.append((j, g, o, c))
    logger.warning(
        f"WS10 replay {os.path.basename(os.environ['WS10_DUMP'])}: disp {tuple(disp.shape)} spans (local, global, "
        f"offset, count) {spans}; max_tokens {blob['max_tokens']} act {act} ckc {ck}"
    )

    n_dev = dev.get_num_devices()
    used = int(valid.nonzero().max()) + 1
    if os.environ.get("WS10_IDS", "0") == "1":
        try:
            ids = dev.get_device_ids()
            logger.warning(f"WS10 IDS mesh {dev.shape}: device ids by logical index {list(ids)}")
            for lc in (6, 6), (0, 0), (10, 7):
                pc = dev.worker_core_from_logical_core(ttnn.CoreCoord(*lc))
                logger.warning(f"WS10 IDS logical core {lc} -> NOC {pc}")
            logger.warning(f"WS10 IDS compute grid {dev.compute_with_storage_grid_size()}")
        except Exception as e:  # diagnostic only
            logger.warning(f"WS10 IDS failed: {e}")
    chips_rd = set(int(v) for v in os.environ.get("WS10_CHIPS", ",".join(map(str, range(n_dev)))).split(","))
    empty = torch.zeros(used, disp.shape[-1])

    gap = float(os.environ.get("WS10_GAP_MS", "0")) / 1000.0

    def run_all():
        """Every chip runs the same replicated inputs; returns [n_dev, used, emb] (valid region only)."""
        if gap:
            import time

            ttnn.synchronize_device(dev)
            time.sleep(gap)
        out = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
            tt_disp,
            tt_off,
            tt_cnt,
            tt_idx,
            gates,
            ups,
            downs,
            max_dispatched_tokens_per_expert=blob["max_tokens"],
            compute_kernel_config=ckc,
            activation=act,
            gate_biases=None,
            up_biases=None,
            down_biases=None,
            min_active_tokens=0,
        )
        sl = ttnn.slice(out, [0, 0], [used, disp.shape[-1]])
        dts = ttnn.get_device_tensors(sl)
        ts = [
            ttnn.to_torch(dts[c]).float().reshape(-1, disp.shape[-1])[:used] if c in chips_rd else empty
            for c in range(len(dts))
        ]
        ttnn.deallocate(sl)
        ttnn.deallocate(out)
        return torch.stack(ts)

    epochs = int(os.environ.get("WS10_EPOCHS", "1"))
    if n_dev > 1 and epochs > 1:
        # DS41F-0169 WS10: shift every DRAM buffer per epoch (a random-size pad tensor before re-uploading inputs and
        # weights) to vary bank alignment / arrival timing; flips counted per chip and epoch.
        vu = valid[:used]
        prod_u = blob["first_out"].float().reshape(-1, disp.shape[-1])[:used]
        tot = [0] * n_dev
        g = torch.Generator().manual_seed(int(os.environ.get("WS10_EPOCH_SEED", "1")))
        pads = []
        for ep in range(epochs):
            for t in [tt_disp, tt_cnt, tt_off, tt_idx] + gates + ups + downs:
                ttnn.deallocate(t)
            pad_rows = int(torch.randint(1, 4096, (1,), generator=g)) * 32
            pads.append(up(torch.zeros(pad_rows, 1024), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT))
            tt_disp = up(disp.to(torch.bfloat16), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
            tt_cnt, tt_off, tt_idx = u32(blob["counts"]), u32(blob["offsets"]), u32(blob["idx_table"])
            order = torch.randperm(36, generator=g).tolist()
            ws = [None] * 36
            srcs = blob["gate"] + blob["up"] + blob["down"]
            for i in order:
                ws[i] = up(srcs[i].float(), wdt, ttnn.TILE_LAYOUT)
                if int(torch.randint(0, 3, (1,), generator=g)) == 0:
                    pads.append(
                        up(
                            torch.zeros(int(torch.randint(1, 256, (1,), generator=g)) * 32, 1024),
                            ttnn.bfloat16,
                            ttnn.ROW_MAJOR_LAYOUT,
                        )
                    )
            gates, ups, downs = ws[:12], ws[12:24], ws[24:]
            if ep < int(os.environ.get("WS10_SKIP_EPOCHS", "0")):
                logger.warning(f"WS10 epoch {ep}: pad {pad_rows} rows (placement only, no reps)")
                continue
            ref = run_all()
            nprod = sum(int(((ref[c] != prod_u) & vu[:, None]).sum()) for c in range(n_dev) if c in chips_rd)
            per = [0] * n_dev
            for r in range(reps):
                t = run_all()
                for c in range(n_dev):
                    d = (t[c] != ref[c]) & vu[:, None]
                    if d.any():
                        per[c] += 1
                        nz = d.nonzero()
                        logger.warning(
                            f"WS10 epoch {ep} rep {r} chip {c}: {int(d.sum())} values differ, rows "
                            f"{sorted(set(nz[:, 0].tolist()))[:8]}, cols {int(nz[:, 1].min())}..{int(nz[:, 1].max())}"
                        )
            tot = [a + b for a, b in zip(tot, per)]
            logger.warning(
                f"WS10 epoch {ep}: pad {pad_rows} rows, first run vs production {nprod} values; per-chip "
                f"differing runs (of {reps}): {per}"
            )
        logger.warning(f"WS10 RESULT epochs {epochs} x {reps} reps: per-chip totals {tot}")
        return

    if n_dev > 1:
        vu = valid[:used]
        prod_u = blob["first_out"].float().reshape(-1, disp.shape[-1])[:used]
        ref = run_all()
        for c in range(n_dev):
            dpc = (ref[c] != prod_u) & vu[:, None]
            if dpc.any():
                logger.warning(f"WS10 mesh chip {c}: first run differs from PRODUCTION in {int(dpc.sum())} values")
        per_chip, alts = [0] * n_dev, {}
        for r in range(reps):
            t = run_all()
            for c in range(n_dev):
                d = (t[c] != ref[c]) & vu[:, None]
                if d.any():
                    per_chip[c] += 1
                    nz = d.nonzero()
                    key = (c, hashlib.md5(t[c][d].numpy().tobytes() + nz.numpy().tobytes()).hexdigest()[:8])
                    if key not in alts:
                        rr = sorted(set(nz[:, 0].tolist()))
                        logger.warning(
                            f"WS10 mesh rep {r} chip {c}: NEW alternate {key[1]}: {int(d.sum())} values, rows "
                            f"{rr[:10]}, cols {int(nz[:, 1].min())}..{int(nz[:, 1].max())}"
                        )
                        alts[key] = 0
                    alts[key] += 1
            if (r + 1) % 25 == 0:
                logger.info(f"WS10 mesh progress {r + 1}/{reps}: per-chip differing runs {per_chip}")
        logger.warning(
            f"WS10 RESULT mesh {_MESH}: per-chip runs (of {reps}) differing from that chip's first: {per_chip}; "
            f"alternates {alts}"
        )
        return

    def run():
        out = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
            tt_disp,
            tt_off,
            tt_cnt,
            tt_idx,
            gates,
            ups,
            downs,
            max_dispatched_tokens_per_expert=blob["max_tokens"],
            compute_kernel_config=ckc,
            activation=act,
            gate_biases=None,
            up_biases=None,
            down_biases=None,
            min_active_tokens=0,
        )
        t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(dev, dim=0)).float()
        ttnn.deallocate(out)
        return t.reshape(-1, t.shape[-1])[:rows]

    prod = blob["first_out"].float().reshape(-1, disp.shape[-1])[:rows]
    ref = run()
    dp = (ref != prod) & valid[:, None]
    logger.warning(
        f"WS10 replay vs PRODUCTION output: {int(dp.sum())} differing valid values"
        + (f" rows {sorted(set(dp.nonzero()[:, 0].tolist()))[:12]}" if dp.any() else " (exact)")
    )
    alts, ndiff = {}, 0
    for r in range(reps):
        t = run()
        d = (t != ref) & valid[:, None]
        if d.any():
            ndiff += 1
            nz = d.nonzero()
            key = hashlib.md5(t[d].numpy().tobytes() + nz.numpy().tobytes()).hexdigest()[:8]
            if key not in alts:
                rr = sorted(set(nz[:, 0].tolist()))
                sp = [next((s for s in spans if s[2] <= x < s[2] + max(s[3], 1)), None) for x in rr[:6]]
                vals = ", ".join(
                    f"({int(a)},{int(b)}) {float(ref[a, b]):.6g}->{float(t[a, b]):.6g}" for a, b in nz[:10].tolist()
                )
                logger.warning(
                    f"WS10 rep {r}: NEW alternate {key}: {int(d.sum())} values, rows {rr[:12]} spans {sp}, cols "
                    f"{int(nz[:, 1].min())}..{int(nz[:, 1].max())}; {vals}"
                )
                alts[key] = 0
            alts[key] += 1
    logger.warning(f"WS10 RESULT replay: {ndiff} of {reps} re-runs differ from the first; alternates {alts}")
