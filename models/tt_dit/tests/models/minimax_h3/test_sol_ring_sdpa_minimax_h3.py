# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Ring-joint SDPA in Sol-Attn mode vs the torch reference, on the 4x8 mesh at small shapes.

Pipeline under test: sol_prepass.SolPrepass (device) -> ring_joint_scaled_dot_product_attention(sol_*)
against sol_ref.sol_attention(granularity="chunk", k_chunk=64, tt_kernel_rules=True) on the same bf16 Q/K/V.
Also checks the degenerate settings: tau=-1e4 (everything exact) must match dense ring SDPA tightly.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch
from loguru import logger

import ttnn

from ....models.transformers.minimax_h3.sol_prepass import SolGeometry, SolPrepass
from ....parallel.manager import CCLManager
from ....utils.tensor import bf16_tensor, from_torch
from .common import GALAXY_RING
from .test_transformer_minimax_h3 import logical_length_tensor

sys.path.insert(0, os.environ.get("SOL_ATTN_WORK", "/home/losullivan/sol_attn_work"))  # /data copy on Slurm nodes
import sol_ref as R  # noqa: E402

_SOL_MODE_LINE = "RingJointSDPA sol mode resolved: "


def _sol_mode_guard(capfd, mode, state):
    """Fail when a requested Sol mode did not reach the kernels (lead's request after the docstring-gate incident).

    The program factory logs "RingJointSDPA sol mode resolved: <mode> ..." once per program creation, i.e. exactly when
    the kernels are created with that mode's named compile-time args. Read it back from the captured stderr after the
    first sparse launch; the captured text is re-emitted so job logs keep every line.
    """
    if state.get("checked"):
        return
    state["checked"] = True
    out, err = capfd.readouterr()
    with capfd.disabled():
        sys.stdout.write(out)
        sys.stderr.write(err)
        sys.stdout.flush()
        sys.stderr.flush()
    lines = [l for l in (out + err).splitlines() if _SOL_MODE_LINE in l]
    assert (
        lines
    ), "no 'RingJointSDPA sol mode resolved' line captured: the factory log is missing (old build) or filtered"
    resolved = {l.split(_SOL_MODE_LINE, 1)[1].split()[0] for l in lines}
    expected = "gather" if mode in (None, "auto") else mode
    assert resolved == {
        expected
    }, f"requested sol mode {expected!r} but the op resolved {sorted(resolved)}: {lines[-1]}"
    logger.info(f"[sol] mode guard: op resolved {expected!r} ({len(lines)} program creation(s))")


def _sol_mode_kwargs(mode):
    """Pass sol_mode only when the installed binding has it (older builds infer gather/chunk from the mask width).

    The op's docstring never lists keyword names (a check on it is always False), so probe the raw nanobind function:
    its argument error text carries the full signature. "compact" cannot be inferred, so a binding without the keyword
    is an error there instead of a silent gather run (that silently happened once; see DESIGN_BLOCK_COMPACT.md).
    """
    if mode in (None, "auto"):
        return {}
    raw = ttnn._ttnn.operations.transformer.ring_joint_scaled_dot_product_attention
    try:
        raw()
        has = False
    except TypeError as e:
        has = "sol_mode" in str(e)
    if not has:
        if mode == "compact":
            raise RuntimeError(
                "the installed ring_joint_scaled_dot_product_attention binding has no sol_mode keyword: "
                "a 'compact' run would silently execute in gather mode"
            )
        logger.warning(f"binding has no sol_mode keyword; mode {mode!r} is inferred from the mask width")
        return {}
    logger.info(f"[sol] passing sol_mode={mode!r} to the ring op")
    return {"sol_mode": mode}


def _structured_qkv(S, H, rows, D, seed=1):
    torch.manual_seed(seed)
    topics = torch.randn(8, D)
    tid = torch.randint(0, 8, (S * rows,))
    base = topics[tid].view(S, 1, rows, D)
    q = (base + 0.7 * torch.randn(S, 1, rows, D)).expand(S, H, rows, D).clone() + 0.3 * torch.randn(S, H, rows, D)
    k = (base + 0.7 * torch.randn(S, 1, rows, D)).expand(S, H, rows, D).clone() + 0.3 * torch.randn(S, H, rows, D)
    v = torch.randn(S, H, rows, D)
    return (x.bfloat16().float() for x in (q, k, v))


@GALAXY_RING
@pytest.mark.parametrize(
    ("rows", "logical_n", "q_chunk", "k_chunk", "tau"),
    [
        pytest.param(1024, 8192, 256, 512, -1e4, id="dense_equiv"),
        pytest.param(1024, 8192, 256, 512, 1.0, id="tau1"),
        pytest.param(1184, 9000, 256, 512, 1.0, id="tau1_ragged"),
        pytest.param(1184, 9472, 256, 256, 1.5, id="tau1.5_k256"),
        # the 5 s production geometry (4736 rows per chip, 37888 padded tokens, module chunks 320x384: 6 blocks per K chunk)
        pytest.param(4736, 37800, 320, 384, 1.0, id="geo5s_tau1"),
        # the 10 s production geometry (rows 9184, logical_n 73425, sink 849) that the e2e runs use
        pytest.param(
            9184,
            73425,
            256,
            512,
            1.0,
            id="geo10s_tau1",
            marks=pytest.mark.skipif(os.environ.get("SOL_RING_FULL") != "1", reason="set SOL_RING_FULL=1"),
        ),
        pytest.param(
            9184,
            73425,
            256,
            512,
            2.0,
            id="geo10s_tau2",
            marks=pytest.mark.skipif(os.environ.get("SOL_RING_FULL") != "1", reason="set SOL_RING_FULL=1"),
        ),
    ],
)
def test_sol_ring_sdpa(
    mesh_device,
    sp_axis,
    tp_axis,
    num_links,
    topology,
    is_fsdp,
    rows,
    logical_n,
    q_chunk,
    k_chunk,
    tau,
    reset_seeds,
    capfd,
):
    S = tuple(mesh_device.shape)[sp_axis]
    TP = tuple(mesh_device.shape)[tp_axis]
    H, D = 14, 128
    scale = D**-0.5
    sink_tokens = int(os.environ.get("SOL_SINK_TOKENS", "300" if rows < 4096 else "849"))
    q, k, v = _structured_qkv(S, H, rows, D)
    geo = SolGeometry(shards=S, rows=rows, logical_n=logical_n, q_chunk=q_chunk, k_chunk=k_chunk)

    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    mesh_axes = [sp_axis, None, None, None]  # tensor dim 0 (shard) fractured over the SP mesh axis

    def upload(x):
        return from_torch(x, device=mesh_device, dtype=ttnn.bfloat16, mesh_axes=mesh_axes, layout=ttnn.TILE_LAYOUT)

    tq, tk, tv = upload(q), upload(k), upload(v)
    mode = os.environ.get(
        "SOL_MODE", "gather"
    )  # "gather" (block gathering), "chunk" (compute skip with chains), "compact"
    block_form = mode in ("gather", "compact")  # block-level mask and reference (full blocks only)
    # SOL_GEN_BIAS: "0" = pre-pass uploads the sol_bias tensor (legacy), "1" = the ring op generates the bias tiles
    # on device (production), "ab" = both, and the two outputs must agree bit for bit.
    gen_bias = os.environ.get("SOL_GEN_BIAS", "0")
    pre = SolPrepass(
        mesh_device=mesh_device, ccl_manager=ccl_manager, sp_axis=sp_axis, tau=tau, mode=mode, emit_bias=gen_bias != "1"
    )
    pre.prepare(geo, sink_tokens, scale)
    sol = pre(tq, tk, tv, scale=scale, report_density=True)
    logger.info(f"[{mode}] device kept density {pre.last_density:.4f}")

    grid = mesh_device.compute_with_storage_grid_size()
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x - 1, grid.y),
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
        exp_approx_mode=False,
    )
    ckc = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False
    )
    dummy_joint = bf16_tensor(torch.zeros((1, H, 0, D)), device=mesh_device)
    kshape = tq.shape

    mode_guard_state = {}

    def run_ring(**sol_kwargs):
        out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
            tq,
            tk,
            tv,
            dummy_joint,
            dummy_joint,
            dummy_joint,
            persistent_output_buffer_k=ccl_manager.get_ag_ping_pong_buffer(kshape, 2, sp_axis, dtype=ttnn.bfloat16),
            persistent_output_buffer_v=ccl_manager.get_ag_ping_pong_buffer(kshape, 2, sp_axis, dtype=ttnn.bfloat16),
            joint_strategy="rear",
            logical_n=logical_length_tensor(mesh_device, logical_n),
            program_config=program_config,
            compute_kernel_config=ckc,
            dim=2,
            multi_device_global_semaphore=ccl_manager.get_ag_ping_pong_semaphore(sp_axis),
            num_links=num_links,
            cluster_axis=sp_axis,
            mesh_device=mesh_device,
            topology=topology,
            subdevice_id=ccl_manager.ccl_sub_device_id,
            ccl_core_grid_offset=(grid.x - 1, 0),
            use_column_major_ccl=True,
            **sol_kwargs,
        )
        if sol_kwargs:
            _sol_mode_guard(capfd, mode, mode_guard_state)
        ttnn.synchronize_device(mesh_device)
        host = ttnn.to_torch(
            out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                mesh_device, dims=[1, 2] if sp_axis == 1 else [2, 1], mesh_shape=tuple(mesh_device.shape)
            ),
        )
        return host[0, :H].float()  # [H, S*rows, D] (TP replicas identical)

    dense = run_ring()
    if os.environ.get("SOL_DENSE_ONLY", "0") == "1":  # fresh-launch stress of the dense ring op (DESIGN.md 9.8)
        logger.info(f"[dense-only] dense op ran; output abs mean {dense.abs().mean().item():.4f}")
        return
    sol_common = dict(
        sol_block_mask=sol["block_mask"],
        sol_pooled_k=sol["pooled_k"],
        sol_pooled_v=sol["pooled_v"],
        **_sol_mode_kwargs(mode),
    )
    if mode == "compact" and os.environ.get("SOL_TEST_SET_PARTIAL", "0") == "1":
        # Producer fault injection: set the bit of every shard's partial tail block (if any) for every Q chunk. The
        # reader and the on-device bias generator must ignore partial blocks, so the output must not change.
        lengths = geo.block_lengths()  # [S, nb_pad]
        partial = [(s_, b_) for s_ in range(S) for b_ in range(geo.nb) if 0 < lengths[s_, b_] < geo.block]
        assert partial, "this case needs a geometry with a partial tail block (rows % 64 != 0 or a logical_n tail)"
        mesh_shape = tuple(mesh_device.shape)
        host_mask = ttnn.to_torch(
            sol["block_mask"],
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                mesh_device, dims=[0, 1] if sp_axis == 0 else [1, 0], mesh_shape=mesh_shape
            ),
        )  # [S, TP*H, nqc_pad, nbg]; TP replicas are identical
        host_mask = host_mask[:, :H].clone()
        for s_, b_ in partial:
            host_mask[..., s_ * geo.nb_pad + b_] = 1
        faulty = (
            ttnn.from_torch(  # same distribution as SolPrepass._upload(shard=True): dim 0 over SP, replicated over TP
                host_mask,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    mesh_device, dims=[None, 0] if sp_axis == 1 else [0, None], mesh_shape=mesh_shape
                ),
            )
        )
        baseline = run_ring(**sol_common)
        with_partial = run_ring(**{**sol_common, "sol_block_mask": faulty})
        n_diff = int((baseline != with_partial).sum())
        max_abs = (baseline - with_partial).abs().max().item()
        # Diagnostics for a mismatch: is the re-uploaded mask itself identical to the original (round trip), and does
        # the faulty run still track the dense reference (small deviation) or is it garbage (wrong upload)?
        rt = ttnn.to_torch(
            faulty,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                mesh_device, dims=[0, 1] if sp_axis == 0 else [1, 0], mesh_shape=mesh_shape
            ),
        )[:, :H]
        logger.info(
            f"[compact] partial-block fault injection on {len(partial)} blocks: differing elements {n_diff} of "
            f"{baseline.numel()}, max |diff| {max_abs:.3e}; re-uploaded mask round trip equal {torch.equal(rt, host_mask)}; "
            f"baseline vs dense rel L2 {((baseline - dense).norm() / dense.norm()).item():.4f}, "
            f"faulty vs dense rel L2 {((with_partial - dense).norm() / dense.norm()).item():.4f}"
        )
        assert torch.equal(
            baseline, with_partial
        ), "partial tail blocks must be ignored by the reader and the bias generator"
    if gen_bias == "0":
        sparse = run_ring(**sol_common, sol_bias=sol["bias"])
    else:
        sparse = run_ring(**sol_common)  # generated bias
        if gen_bias == "ab":
            sparse_tensor_bias = run_ring(**sol_common, sol_bias=sol["bias"])
            max_abs = (sparse_tensor_bias - sparse).abs().max().item()
            n_diff = int((sparse_tensor_bias != sparse).sum())
            logger.info(f"[{mode}] generated vs tensor bias: max |diff| {max_abs:.3e}, differing elements {n_diff}")
            assert torch.equal(
                sparse_tensor_bias, sparse
            ), f"generated bias must reproduce the tensor bias path bit-exactly (max diff {max_abs})"

    # torch references (fp32) on the same bf16-valued inputs
    # gather mode: (Q chunk x 64-token block) granularity, full blocks only; chunk mode: (Q chunk x K chunk),
    # every valid token of a kept chunk is exact.
    rgeo = R.Geometry(shards=S, rows=rows, logical_n=logical_n, q_chunk=q_chunk, k_chunk=64 if block_form else k_chunk)
    qg = q.permute(1, 0, 2, 3).reshape(H, S * rows, D)
    kg = k.permute(1, 0, 2, 3).reshape(H, S * rows, D)
    vg = v.permute(1, 0, 2, 3).reshape(H, S * rows, D)
    ref_dense = R.dense_attention(qg, kg, vg, rgeo)
    # block-form modes (gather, compact): full blocks only and block 0 of every shard forced kept (tt_kernel_rules)
    even_up_bpc = (
        (k_chunk // 64)
        if (
            mode == "compact"
            and (os.environ.get("TT_SOL_EVENUP", "0") == "1" or os.environ.get("TT_SOL_EVENUP_ALL", "0") == "1")
        )
        else 0
    )  # kernel even-up rule
    routing = R.route(
        qg,
        kg,
        rgeo,
        tau=tau,
        scale=scale,
        band=1,
        sink_tokens=sink_tokens,
        tt_kernel_rules=True,
        dense_prefix_queries=True,
        full_blocks_only=block_form,
        even_up_bpc=even_up_bpc,
    )
    ref_sol = R.sol_attention(qg, kg, vg, rgeo, routing, granularity="chunk")
    valid = torch.arange(S * rows) < logical_n

    # Reference driven by the DEVICE routing (mask + bf16 pooled K/V read back), isolating the kernel arithmetic
    # from threshold rounding differences in the pre-pass.
    comp = ttnn.ConcatMesh2dToTensor(
        mesh_device, dims=[0, 1] if sp_axis == 0 else [1, 0], mesh_shape=tuple(mesh_device.shape)
    )
    dev_mask = ttnn.to_torch(sol["block_mask"], mesh_composer=comp)[:, :H].float()  # [S,H,nqc_pad,nbg or nkc_pad]
    if block_form:
        cols = torch.cat([torch.arange(s_ * geo.nb_pad, s_ * geo.nb_pad + geo.nb) for s_ in range(S)])
        ncol = geo.nb
    else:
        cols = torch.cat([torch.arange(s_ * geo.nkc, s_ * geo.nkc + geo.nkc) for s_ in range(S)])
        ncol = geo.nkc
    dev_chunk_mask = dev_mask[:, :, : geo.nqc][..., cols].permute(1, 0, 2, 3).reshape(H, S * geo.nqc, S * ncol).bool()
    mask_agree = (dev_chunk_mask == routing["chunk_mask"]).float().mean().item()
    bcols = torch.cat([torch.arange(s_ * geo.nb_pad, s_ * geo.nb_pad + geo.nb) for s_ in range(S)])
    dev_kc = ttnn.to_torch(sol["pooled_k"], mesh_composer=comp)[0, :H].float()[:, bcols]
    if mode == "compact" and os.environ.get("TT_SOL_Q_PAIRS", "0") == "1":
        # Sol Q pairs: the kernel's effective selection per Q chunk is the OR with its pass partner (same core, same
        # head); both references take the same union so the comparison is against exactly the operator implemented.
        n_cores = (grid.x - 1) * grid.y
        dev_chunk_mask = R.pair_union_rows(dev_chunk_mask, geo.nqc, S, n_cores)
        routing = dict(routing)
        routing["chunk_mask"] = R.pair_union_rows(routing["chunk_mask"], geo.nqc, S, n_cores)
        logger.info(
            f"[pairs] reference selection = pair union ({n_cores} cores); density {routing['chunk_mask'].float().mean().item():.4f}"
        )
    dev_routing = dict(routing)
    dev_routing["chunk_mask"] = R.even_up_blocks(dev_chunk_mask, rgeo, even_up_bpc) if even_up_bpc else dev_chunk_mask
    dev_routing["kc"] = dev_kc
    ref_sol_devmask = R.sol_attention(qg, kg, vg, rgeo, dev_routing, granularity="chunk")
    if even_up_bpc:  # diagnostic: is the kernel closer to the even-up reference or to the plain one?
        plain = dict(routing)
        plain["chunk_mask"] = dev_chunk_mask
        plain["kc"] = dev_kc
        ref_plain = R.sol_attention(qg, kg, vg, rgeo, plain, granularity="chunk")
        e_plain = R.error_stats(ref_plain[:, valid], sparse[:, valid])
        e_refs = R.error_stats(ref_plain[:, valid], ref_sol_devmask[:, valid])
        logger.info(
            f"[even-up diag] kernel vs PLAIN device-mask reference: {e_plain} | plain ref vs even-up ref: {e_refs}"
        )
    e_sol_vs_devref = R.error_stats(ref_sol_devmask[:, valid], sparse[:, valid])
    logger.info(
        f"device mask vs reference mask agreement {mask_agree:.5f}; sol kernel vs device-mask reference: {e_sol_vs_devref}"
    )

    e_dense = R.error_stats(ref_dense[:, valid], dense[:, valid])
    e_sol_vs_ref = R.error_stats(ref_sol[:, valid], sparse[:, valid])
    e_sol_vs_dense = R.error_stats(ref_dense[:, valid], sparse[:, valid])
    e_ref_vs_dense = R.error_stats(ref_dense[:, valid], ref_sol[:, valid])
    logger.info(f"dense kernel vs fp32 dense: {e_dense}")
    logger.info(f"sol kernel vs sol reference: {e_sol_vs_ref}")
    logger.info(f"sol kernel vs fp32 dense:    {e_sol_vs_dense}")
    logger.info(
        f"sol reference vs fp32 dense: {e_ref_vs_dense}  (routing density {routing['chunk_mask'].float().mean():.4f})"
    )
    # The kernel is bf16 (HiFi2); judge the sparse kernel against its own fp32 reference relative to the
    # dense kernel's distance from the fp32 dense reference on the same data.
    assert e_dense["pcc"] > 0.99
    assert e_sol_vs_devref["pcc"] > e_dense["pcc"] - 0.003, (e_sol_vs_devref, e_dense)
    assert e_sol_vs_devref["rel_l2"] < 1.5 * e_dense["rel_l2"] + 0.01, (e_sol_vs_devref, e_dense)
    if tau < -100:
        assert torch.allclose(sparse, dense, atol=1e-3, rtol=1e-2), "tau=-inf must reproduce the dense kernel"
