# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""ttnn.all_to_all_matmul in q-stem mode (qk_rope_head_dim set): the MLA early-Q stem before RoPE in one call.

Per device the input is the unsplit q_b_proj output (1, 1, S, Hl*(K+R)); head j's columns are [j*(K+R), j*(K+R)+K)
(nope) then R rope columns. Over the G=4 ring along cluster_axis 1 the op returns, on device q,
    prefix[q]   = cat_p(nope_p[:, :, q*S/G:(q+1)*S/G], dim=1) @ w   (per head)     (1, H, S/G, N)
    rope_seq[q] = cat_p(rope_p[:, :, q*S/G:(q+1)*S/G], dim=1)                     (1, H, S/G, R), exact copy
Checks: torch reference (PCC for prefix, bit-exact rope_seq), prefix bit-identical to the plain op on the pre-split
nope (the same exchange + matmul), cache hits with new data, L1 placements, caller errors.

Mesh: a 1x4 ring under FABRIC_2D_TORUS_XY (on a LoudBox: one 2x2 half, TT_VISIBLE_DEVICES=0,1,2,3, opened as 1x4).
Shapes: Kimi K3.1 (H=96, K=128, R=64, N=512) and GLM-5.3 (H=64, K=192, R=64, N=512), S=640 per chip as in MLA on
the 1x4 ring; HiFi2, 16-bit DEST as MLA runs it.
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params

G = 4
CLUSTER_AXIS = 1
MESH = (1, 4)
KIMI_PAYLOAD = 7232  # Kimi K3.1's production fabric packet size (3 bf16 tiles); default: the arch maximum

# (H, K, R, N, S)
KIMI = (96, 128, 64, 512, 640)
GLM = (64, 192, 64, 512, 640)

PCC = {ttnn.bfloat16: 0.999, ttnn.bfloat8_b: 0.998}


def _cfg():
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    return cfg


def _mesh_params(payload=None):
    return pytest.param(MESH, torus_xy_device_params(fabric_payload_size=payload), id=f"payload{payload or 'max'}")


MESH_DEFAULT = [_mesh_params()]
MESH_BOTH_PAYLOADS = [_mesh_params(), _mesh_params(KIMI_PAYLOAD)]


def _inputs(shape, seed):
    H, K, R, N, S = shape
    Hl = H // G
    torch.manual_seed(seed)
    q = torch.randn(1, G, S, Hl * (K + R), dtype=torch.float32).to(torch.bfloat16)  # device c: q[:, c]
    w = (torch.randn(1, H, K, N, dtype=torch.float32) / K**0.5).to(torch.bfloat16)
    return q, w


def _split(q, shape):
    """(1, G, S, Hl*(K+R)) -> nope (G, Hl, S, K), rope (G, Hl, S, R): each device's own head split."""
    H, K, R, N, S = shape
    Hl = H // G
    heads = q[0].view(G, S, Hl, K + R).permute(0, 2, 1, 3)
    return heads[..., :K].contiguous(), heads[..., K:].contiguous()


def _reference(q, w, shape, w_dtype):
    """Per destination q: (prefix (1, H, Mq, N) fp32, rope_seq (1, H, Mq, R) bf16)."""
    H, K, R, N, S = shape
    Mq = S // G
    nope, rope = _split(q, shape)
    wf = w.float()
    if w_dtype == ttnn.bfloat8_b:  # the device multiplies the bfloat8_b-rounded weight
        wf = ttnn.to_torch(ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)).float()
    out = []
    for d in range(G):
        a = torch.cat([nope[p, :, d * Mq : (d + 1) * Mq] for p in range(G)], dim=0).unsqueeze(0)  # (1, H, Mq, K)
        r = torch.cat([rope[p, :, d * Mq : (d + 1) * Mq] for p in range(G)], dim=0).unsqueeze(0)
        out.append((a.float() @ wf, r))
    return out


def _to_mesh(t, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16):
    """(1, G, ...) torch -> per device (1, 1, ...) along mesh columns."""
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=memory_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH, dims=(None, 1)),
    )


def _weight(w, mesh_device, w_dtype):
    return ttnn.from_torch(
        w,
        dtype=w_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _per_device(t):
    return [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]


def _qstem(tq, tw, shape, **kw):
    H, K, R, N, S = shape
    return ttnn.all_to_all_matmul(
        tq,
        tw,
        cluster_axis=CLUSTER_AXIS,
        topology=ttnn.Topology.Ring,
        num_links=2,
        compute_kernel_config=_cfg(),
        qk_rope_head_dim=R,
        **kw,
    )


def _check(prefix, rope_seq, ref, w_dtype, tag=""):
    got_p, got_r = _per_device(prefix), _per_device(rope_seq)
    pccs = []
    for d, (want_p, want_r) in enumerate(ref):
        assert tuple(got_p[d].shape) == tuple(want_p.shape), f"{tag} prefix shape {got_p[d].shape} != {want_p.shape}"
        assert tuple(got_r[d].shape) == tuple(want_r.shape), f"{tag} rope shape {got_r[d].shape} != {want_r.shape}"
        assert torch.equal(
            got_r[d].view(torch.int16), want_r.view(torch.int16)
        ), f"{tag} device {d}: rope_seq is not the exact exchanged rope columns"
        ok, pcc = comp_pcc(want_p, got_p[d].float(), PCC[w_dtype])
        pccs.append(pcc)
        assert ok, f"{tag} device {d}: prefix PCC {pcc} < {PCC[w_dtype]}"
    print(f"QSTEM {tag} prefix PCC min {min(pccs):.6f}; rope_seq bit-exact")
    return got_p


@pytest.mark.parametrize("mesh_device, device_params", MESH_BOTH_PAYLOADS, indirect=True)
@pytest.mark.parametrize("w_dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["w_bf16", "w_bfp8"])
@pytest.mark.parametrize("shape", [KIMI, GLM], ids=["kimi_k31", "glm_53"])
def test_qstem_vs_reference_and_plain(mesh_device, device_params, shape, w_dtype):
    """Torch reference, and prefix bit-identical to the plain op on the pre-split nope (old mode unchanged)."""
    H, K, R, N, S = shape
    q, w = _inputs(shape, seed=0)
    ref = _reference(q, w, shape, w_dtype)
    tw = _weight(w, mesh_device, w_dtype)
    prefix, rope_seq = _qstem(_to_mesh(q, mesh_device), tw, shape)
    got_p = _check(prefix, rope_seq, ref, w_dtype, tag=f"H{H} K{K} {w_dtype}")

    # the plain (old) mode on the same nope, split on the host
    nope, _ = _split(q, shape)
    plain = ttnn.all_to_all_matmul(
        _plain_input(nope, mesh_device),
        tw,
        cluster_axis=CLUSTER_AXIS,
        topology=ttnn.Topology.Ring,
        num_links=2,
        compute_kernel_config=_cfg(),
    )
    got_plain = _per_device(plain)
    for d in range(G):
        ok, pcc = comp_pcc(ref[d][0], got_plain[d].float(), PCC[w_dtype])
        assert ok, f"plain mode device {d}: PCC {pcc}"
        assert torch.equal(
            got_plain[d].view(torch.int16), got_p[d].view(torch.int16)
        ), f"device {d}: q-stem prefix differs from the plain op on the split nope"


def _plain_input(nope, mesh_device):
    """nope (G, Hl, S, K) -> per device c: nope[c] as (1, Hl, S, K), DRAM."""
    G_, Hl, S, K = nope.shape
    return ttnn.from_torch(
        nope.reshape(1, G_ * Hl, S, K),  # device c's heads at [c*Hl, (c+1)*Hl): shard dim 1 over the columns
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH, dims=(None, 1)),
    )


@pytest.mark.parametrize("mesh_device, device_params", MESH_DEFAULT, indirect=True)
@pytest.mark.parametrize("shape", [KIMI, GLM], ids=["kimi_k31", "glm_53"])
def test_qstem_cache_hit_with_scratch(mesh_device, device_params, shape):
    """MLA's call: persistent scratch_tensor, three calls on the cached program with different data each time."""
    H, K, R, N, S = shape
    _, w = _inputs(shape, seed=0)
    tw = _weight(w, mesh_device, ttnn.bfloat16)
    q0, _ = _inputs(shape, seed=1)
    tq0 = _to_mesh(q0, mesh_device)
    scratch = ttnn.all_to_all_matmul_scratch_tensor(
        tq0, tw, cluster_axis=CLUSTER_AXIS, compute_kernel_config=_cfg(), qk_rope_head_dim=R
    )
    n0 = mesh_device.num_program_cache_entries()
    for i, seed in enumerate((1, 2, 3)):
        q, _ = _inputs(shape, seed=seed)
        prefix, rope_seq = _qstem(_to_mesh(q, mesh_device), tw, shape, scratch_tensor=scratch)
        _check(prefix, rope_seq, _reference(q, w, shape, ttnn.bfloat16), ttnn.bfloat16, tag=f"call {i} seed {seed}")
        ttnn.deallocate(prefix)
        ttnn.deallocate(rope_seq)
    assert mesh_device.num_program_cache_entries() - n0 == 1, "the q-stem calls should share one cached program"


@pytest.mark.parametrize("mesh_device, device_params", MESH_DEFAULT, indirect=True)
def test_qstem_l1_placements(mesh_device, device_params):
    """q_b_proj output in L1 (as MLA may leave it), prefix and rope_seq in L1."""
    shape = KIMI
    q, w = _inputs(shape, seed=5)
    tw = _weight(w, mesh_device, ttnn.bfloat16)
    tq = _to_mesh(q, mesh_device, memory_config=ttnn.L1_MEMORY_CONFIG)
    prefix, rope_seq = _qstem(
        tq, tw, shape, memory_config=ttnn.L1_MEMORY_CONFIG, rope_memory_config=ttnn.L1_MEMORY_CONFIG
    )
    assert prefix.memory_config().buffer_type == ttnn.BufferType.L1
    assert rope_seq.memory_config().buffer_type == ttnn.BufferType.L1
    _check(prefix, rope_seq, _reference(q, w, shape, ttnn.bfloat16), ttnn.bfloat16, tag="L1")


@pytest.mark.parametrize("mesh_device, device_params", MESH_DEFAULT, indirect=True)
def test_qstem_caller_errors(mesh_device, device_params, expect_error):
    H, K, R, N, S = KIMI
    q, w = _inputs(KIMI, seed=0)
    tw = _weight(w, mesh_device, ttnn.bfloat16)
    tq = _to_mesh(q, mesh_device)
    with expect_error(ValueError, "q-stem input must be"):
        _qstem(tq, tw, (H, K, 2 * R, N, S))  # width != Hl*(K+R)
    with expect_error(ValueError, "multiple of 32"):
        _qstem(tq, tw, (H, K, 48, N, S))
    with expect_error(ValueError, "rope_memory_config needs qk_rope_head_dim"):
        ttnn.all_to_all_matmul(tq, tw, cluster_axis=CLUSTER_AXIS, rope_memory_config=ttnn.DRAM_MEMORY_CONFIG)
    q_odd = torch.randn(1, G, 96, (H // G) * (K + R)).to(torch.bfloat16)  # S/G = 24: not tile-aligned
    with expect_error(ValueError, "S/G to be a multiple of 32"):
        _qstem(_to_mesh(q_odd, mesh_device), tw, KIMI)
