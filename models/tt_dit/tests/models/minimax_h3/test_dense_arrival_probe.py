# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Does the dense ring SDPA compute on what the fused all-gather delivers, or on what the K/V buffers held before?

The dense `ring_joint_scaled_dot_product_attention` is launched three times on the SAME inputs and the SAME
persistent K/V buffers (the CCLManager ping-pong pair, obtained exactly as `_run` in test_sol_compact does):

    launch 1   buffers zeroed before the op
    launch 2   buffers pre-filled with the exact gathered K/V (torch cat of the shards, replicated to every device)
    launch 3   buffers zeroed again

For every launch: PCC vs fp32 attention per shard (`_pooled_tail_reference(..., pool_partial=False)`), max |out|,
and the gathered K/V buffers read back AFTER the op, slot by slot (which torch shard each slot holds, and how
many tiles differ from the exact value). The pre-op contents are read back too, so a fill that did not land
cannot be mistaken for an op result.

Reading the `[probe] verdict` line:
    PREMATURE_READS      right only on launch 2: the reader computes on whatever the buffers held before the op
                         (its slab-arrival wait passes before the all-gather has landed the slab)
    OP_OK_AT_THIS_SHAPE  right on all three
    SDPA_SIDE_DEFECT     wrong on all three while the read-back shows every remote slot exact: the data is
                         there when the op ends, the op mis-computes on it (chains / addressing / compute)
    ALL_GATHER_WRONG     wrong on all three and the read-back has wrong or never-written slots

    pytest models/tt_dit/tests/models/minimax_h3/test_dense_arrival_probe.py -s -k 4x8
    PROBE_ROWS=1184 ...   # other per-shard row counts (logical_n = rows * shards)
    TT_SOL_NO_CHAIN=1 ... # same run with the SDPA K/V forwarding chains (row-wide NOC multicast) disabled
"""

from __future__ import annotations

import os

import torch
from loguru import logger

import ttnn

from ....parallel.manager import CCLManager
from ....utils.tensor import bf16_tensor, from_torch
from .common import GALAXY_RING
from .test_ag_dram_banks import _harvesting_masks
from .test_sol_compact_minimax_h3 import D, H, _pcc, _pooled_tail_reference, _qkv, logical_length_tensor

Q_CHUNK, K_CHUNK = 256, 512
GOOD = 0.98  # per-shard PCC vs the fp32 reference that counts as "right" (the sol tests' gate)


def _host_replicated(mesh_device, t: torch.Tensor) -> ttnn.Tensor:
    """Host-side TILE bf16 tensor replicated to every device: the spec twin of a ping-pong buffer."""
    return ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )


def _read_gathered(mesh_device, sp_axis, buf) -> torch.Tensor:
    """[S devices along the ring, H, S*rows, D] from one persistent buffer (first TP row only)."""
    return ttnn.to_torch(
        buf,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            mesh_device, dims=[1, 0] if sp_axis == 1 else [0, 1], mesh_shape=tuple(mesh_device.shape)
        ),
    )[:, :H].float()


def _slot_report(name, got, shards, S, rows, *, local, remote):
    """Per device: which torch shard each slot holds ('?' none above 0.99 PCC, 'z' all zero) and the number of
    tiles that differ from what the slot should hold. `local` / `remote` say what slot d == d / d != d should
    hold: 'zero' or 'exact' (shard s). `shards` is [S, H, rows, D] already rounded to bf16.
    Returns (line, bad tiles in remote slots, bad tiles in local slots)."""
    Dt = D // 32
    tiles_per_slot = H * (rows // 32) * Dt
    lines, remote_bad, local_bad = [], 0, 0
    for d in range(S):
        best, bad = [], []
        for s in range(S):
            blk = got[d, :, s * rows : (s + 1) * rows]
            expected = local if s == d else remote
            ref = torch.zeros_like(blk) if expected == "zero" else shards[s]
            tiles = (H, rows // 32, 32, Dt, 32)
            n_bad = int((blk.reshape(tiles) != ref.reshape(tiles)).any(-1).any(2).sum())
            if s == d:
                local_bad += n_bad
            else:
                remote_bad += n_bad
            bad.append(n_bad)
            if float(blk.abs().max()) == 0.0:
                best.append("z")
            else:
                pccs = [_pcc(blk, shards[t]) for t in range(S)]
                m = max(range(S), key=lambda i: pccs[i])
                best.append(f"{m}" if pccs[m] > 0.99 else "?")
        lines.append(f"d{d}:[{','.join(best)}] bad:[{','.join(map(str, bad))}]")
    header = (
        f"[probe] {name}: slot -> torch shard per device (expect local slot {local}, remote slots {remote}), "
        f"bad tiles per slot of {tiles_per_slot}: "
    )
    return header + " ".join(lines), remote_bad, local_bad


@GALAXY_RING
def test_dense_arrival_probe(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, reset_seeds):
    S = tuple(mesh_device.shape)[sp_axis]
    rows = int(os.environ.get("PROBE_ROWS", "1024"))
    logical_n = rows * S
    grid = mesh_device.compute_with_storage_grid_size()
    logger.info(
        f"[probe] mesh {tuple(mesh_device.shape)} sp_axis {sp_axis} ring {S} links {num_links} "
        f"compute grid {grid.x}x{grid.y} (sdpa {grid.x - 1}x{grid.y}, ccl column {grid.x - 1}) "
        f"rows {rows} logical_n {logical_n} q_chunk {Q_CHUNK} k_chunk {K_CHUNK} "
        f"TT_SOL_NO_CHAIN={os.environ.get('TT_SOL_NO_CHAIN', '')!r}"
    )
    try:
        masks = _harvesting_masks()
        logger.info(
            "[probe] harvesting (chip: tensix/dram): "
            + " ".join(
                f"{c}:{m.get('harvest_mask', 0):#x}/{m.get('dram_harvesting_mask', 0):#x}"
                for c, m in sorted(masks.items())
            )
        )
    except Exception as e:  # host-only diagnostics; never block the probe
        logger.info(f"[probe] harvesting masks unavailable: {e}")

    ccl = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    mesh_axes = [sp_axis, None, None, None]
    q, k, v = _qkv(S, rows, 0)  # seed 0: what _pooled_tail_reference regenerates
    tq, tk, tv = (
        from_torch(x, device=mesh_device, dtype=ttnn.bfloat16, mesh_axes=mesh_axes, layout=ttnn.TILE_LAYOUT)
        for x in (q, k, v)
    )
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x - 1, grid.y),
        q_chunk_size=Q_CHUNK,
        k_chunk_size=K_CHUNK,
        exp_approx_mode=False,
    )
    ckc = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False
    )
    empty_joint = bf16_tensor(torch.zeros((1, H, 0, D)), device=mesh_device)
    logical_n_t = logical_length_tensor(mesh_device, logical_n)

    # Exact gathered K/V: slot s = shard s, as the all-gather lays them out. bf16-rounded copies for the exact checks.
    k_bf, v_bf = k.bfloat16().float(), v.bfloat16().float()
    k_full = k.permute(1, 0, 2, 3).reshape(1, H, S * rows, D)
    v_full = v.permute(1, 0, 2, 3).reshape(1, H, S * rows, D)
    zeros = torch.zeros_like(k_full)
    fill_exact = (_host_replicated(mesh_device, k_full), _host_replicated(mesh_device, v_full))
    fill_zero = (_host_replicated(mesh_device, zeros), _host_replicated(mesh_device, zeros))

    ref = _pooled_tail_reference(S, rows, logical_n, pool_partial=False)

    def per_shard_pcc(t):
        out = []
        for sh in range(S):
            lo, hi = sh * rows, min(sh * rows + rows, logical_n)
            out.append(_pcc(t[:, lo:hi], ref[:, lo:hi]))
        return out

    def launch(idx, fill):
        exact = fill is fill_exact
        pk = ccl.get_ag_ping_pong_buffer(tq.shape, 2, sp_axis, dtype=ttnn.bfloat16)
        pv = ccl.get_ag_ping_pong_buffer(tq.shape, 2, sp_axis, dtype=ttnn.bfloat16)
        addr = []
        for b in (pk, pv):
            try:
                addr.append(hex(b.buffer_address()))
            except Exception:
                addr.append("?")
        ttnn.copy_host_to_device_tensor(fill[0], pk)
        ttnn.copy_host_to_device_tensor(fill[1], pv)
        ttnn.synchronize_device(mesh_device)
        # What the buffers hold going INTO the op (a fill that did not land must not pass as an op result).
        pre_state = "exact" if exact else "zero"
        pre_line, pre_remote, pre_local = _slot_report(
            f"launch {idx} PRE-op K",
            _read_gathered(mesh_device, sp_axis, pk),
            k_bf,
            S,
            rows,
            local=pre_state,
            remote=pre_state,
        )
        logger.info(pre_line)
        pre_bad = pre_remote + pre_local
        if pre_bad:
            logger.warning(f"[probe] launch {idx}: the {pre_state} pre-fill did not land ({pre_bad} tiles differ)")
        out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
            tq,
            tk,
            tv,
            empty_joint,
            empty_joint,
            empty_joint,
            persistent_output_buffer_k=pk,
            persistent_output_buffer_v=pv,
            joint_strategy="rear",
            logical_n=logical_n_t,
            program_config=program_config,
            compute_kernel_config=ckc,
            dim=2,
            multi_device_global_semaphore=ccl.get_ag_ping_pong_semaphore(sp_axis),
            num_links=num_links,
            cluster_axis=sp_axis,
            mesh_device=mesh_device,
            topology=topology,
            subdevice_id=ccl.ccl_sub_device_id,
            ccl_core_grid_offset=(grid.x - 1, 0),
            use_column_major_ccl=True,
        )
        ttnn.synchronize_device(mesh_device)
        host = ttnn.to_torch(
            out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                mesh_device, dims=[1, 2] if sp_axis == 1 else [2, 1], mesh_shape=tuple(mesh_device.shape)
            ),
        )[0, :H].float()
        # After the op every remote slot must be exact; the local slot is never written by the fused all-gather,
        # so it still holds the pre-fill (zero, or exact).
        post = {}
        for name, buf, shards in (("K", pk, k_bf), ("V", pv, v_bf)):
            line, bad, _ = _slot_report(
                f"launch {idx} POST-op {name}",
                _read_gathered(mesh_device, sp_axis, buf),
                shards,
                S,
                rows,
                local=pre_state,
                remote="exact",
            )
            logger.info(line)
            post[name] = bad
        bad_k, bad_v = post["K"], post["V"]
        pccs = per_shard_pcc(host)
        valid = host[:, :logical_n]
        finite = bool(torch.isfinite(valid).all())
        sane = finite and float(valid.abs().max()) < 1e3
        logger.info(
            f"[probe] launch {idx} ({'exact pre-fill' if exact else 'zeroed'}; K/V buffers {addr[0]}/{addr[1]}): "
            f"pcc vs fp32 per shard " + " ".join(f"{p:.4f}" for p in pccs) + f"; min {min(pccs):.4f}; "
            f"max |out| {float(valid.abs().max()):.3g}; finite {finite}; "
            f"remote-slot bad tiles after the op K {bad_k} V {bad_v}; fill landed {pre_bad == 0}"
        )
        return dict(out=host, pccs=pccs, right=min(pccs) > GOOD and sane, bad=bad_k + bad_v, landed=pre_bad == 0)

    r1 = launch(1, fill_zero)
    r2 = launch(2, fill_exact)
    r3 = launch(3, fill_zero)
    d13 = float((r1["out"] - r3["out"])[:, :logical_n].abs().max())
    d12 = float((r1["out"] - r2["out"])[:, :logical_n].abs().max())
    logger.info(f"[probe] max |launch1 - launch3| (both zeroed) {d13:.3e}; max |launch1 - launch2| {d12:.3e}")

    right = [r["right"] for r in (r1, r2, r3)]
    readback_ok = all(r["bad"] == 0 for r in (r1, r2, r3))
    landed = all(r["landed"] for r in (r1, r2, r3))
    if all(right):
        verdict = "OP_OK_AT_THIS_SHAPE"
    elif right[1] and not (right[0] and right[2]):
        verdict = "PREMATURE_READS (right only with the exact K/V pre-filled: the reader uses pre-op buffer contents)"
    elif readback_ok:
        verdict = "SDPA_SIDE_DEFECT (every remote slot reads back exact after the op, output still wrong)"
    else:
        verdict = "ALL_GATHER_WRONG (slots read back wrong or never written after the op)"
    if not landed:
        verdict += " [CAUTION: a pre-fill did not land, see warnings]"
    logger.info(
        f"[probe] verdict: {verdict}; right per launch {right}; read-back exact {readback_ok}; "
        f"launch1==launch3 {d13 == 0.0}"
    )
    assert all(right), f"[probe] {verdict}"
