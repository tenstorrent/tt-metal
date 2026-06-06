# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TP4 unit test for the dots.ocr vision MLP on Blackhole 4×P300c.

Tests the canonical tensor-parallel SwiGLU MLP layout on a Blackhole 4-chip
(1×4) mesh.  All activation tensors (op inputs and outputs) are kept in L1
interleaved memory; weights stay in DRAM (too large for L1 across the BH grid).
The all_reduce output is also written to L1 (all prior activations are freed by
that point, so L1 is empty and the ~287 KB/core output fits with headroom).

TP4 sharding:
    fc1 / fc3  column-parallel  — N-shard (output dim, dim=-1)
                                  each device: [HIDDEN, I/TP] weights
                                  no collective; x replicated
    silu × up  elementwise      — local on each device's I/TP slice
    fc2        row-parallel     — K-shard (input dim, dim=-2)
                                  each device: [I/TP, HIDDEN] weights
                                  single ttnn.all_reduce after (not trace-compatible;
                                  use reduce_scatter + all_gather if tracing needed)

Program config (``_bh_tp4_vision_mlp_pc``):
    Finds the largest effective_grid_y ≤ grid_y that divides m_tiles to use the
    maximum number of BH cores.  The CB-budget check uses:

        cb_budget_kb = 1500 KB − l1_resident_kb

    where l1_resident_kb is the per-core L1 occupied by the tensors that are
    already resident in L1 during each matmul call (in0 tensor + future output
    tensor).  This unlocks larger out_block_h values than the DRAM-mode 1024 KB
    cap allows, reducing the number of outer-M weight-re-read iterations.

    Concretely (8×8 grid, 64 cores):
        gate/up: l1_resident = x(541KB) + gate/up_out(197/99KB) → cb_budget=762/860KB
                 best ob_h=11 (vs DRAM ob_h=4); outer-M iters 4 → vs 11
        down   : l1_resident = gum(197KB) + partial(288KB) → cb_budget=1015KB
                 best ob_h=22 (vs DRAM ob_h=4); outer-M iters 2 vs 11
                 full DST area=8 preserved (subblock h=4, w=2)

Supported hardware (MESH_DEVICE env; default: auto-detect from arch):
  - P150x4  : 4 × P150 Blackhole single-chip cards (1×4 line mesh)
  - P300x2  : 2 × P300 Blackhole dual-chip cards   (1×4 line mesh)

Run::

    export ARCH_NAME=blackhole

    MESH_DEVICE=P150x4 pytest \\
        models/experimental/tt_symbiote/tests/test_vision_mlp_tp4_n_k_shard_blackhole.py -s

    MESH_DEVICE=P300x2 pytest \\
        models/experimental/tt_symbiote/tests/test_vision_mlp_tp4_n_k_shard_blackhole.py -s
"""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, is_blackhole, run_for_blackhole

# ---------------------------------------------------------------------------
# dots.ocr model source + expected vision MLP shapes (asserted at load time).
# ---------------------------------------------------------------------------

DOTS_OCR_MODEL_ID = "rednote-hilab/dots.ocr"

HIDDEN = 1536  # hidden dimension H  (vision encoder width)
INTERMEDIATE = 4224  # vision intermediate I  (4224 / 4 = 1056 — tile-aligned)
TP = 4
TP_MESH_SHAPE = (1, TP)
SEQ_LEN = 11264  # vision bucket (352 tiles, 352 % 8 == 0 for 8-row grids)

PCC_THRESHOLD = 0.95  # guards against wrong shard ordering / missing collective

# ---------------------------------------------------------------------------
# Supported Blackhole 4-chip mesh environments.
# ---------------------------------------------------------------------------

SUPPORTED_MESH_ENVS = frozenset({"P150x4", "P300x2"})


def _resolve_model_path() -> str:
    env_path = os.environ.get("DOTS_OCR_MODEL_PATH")
    if env_path and os.path.isdir(env_path):
        return env_path
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(DOTS_OCR_MODEL_ID)
    except Exception:
        return DOTS_OCR_MODEL_ID


def _load_vision_mlp():
    """Return blocks[0].mlp from the dots.ocr vision tower (bf16, eval mode).

    Uses ``from_config`` so the test works without downloading pretrained weights —
    the architecture (shapes) is real; weights are random. Set ``DOTS_OCR_MODEL_PATH``
    to a local checkpoint directory to test with the actual pretrained weights.
    """
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError:
        pytest.skip("transformers required for dots.ocr weight loading")

    model_path = _resolve_model_path()
    try:
        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception as exc:
        pytest.skip(f"dots.ocr config unavailable at {model_path!r}: {exc}")

    # Reduce layer counts so model init is fast.
    model_config.num_hidden_layers = 1
    vision_config = getattr(model_config, "vision_config", None)
    if vision_config is not None:
        for attr in ("num_hidden_layers", "num_layers", "depth"):
            if hasattr(vision_config, attr):
                setattr(vision_config, attr, 1)

    try:
        hf_model = (
            AutoModelForCausalLM.from_config(model_config, trust_remote_code=True).to(dtype=torch.bfloat16).eval()
        )
    except Exception as exc:
        pytest.skip(f"dots.ocr model init failed: {exc}")

    blocks = getattr(hf_model.vision_tower, "blocks", getattr(hf_model.vision_tower, "layers", None))
    assert blocks is not None, "vision tower must expose .blocks or .layers"
    mlp = blocks[0].mlp

    # Verify expected production shapes.
    assert int(mlp.fc1.weight.shape[1]) == HIDDEN, f"expected vision hidden={HIDDEN}, got {mlp.fc1.weight.shape[1]}"
    assert (
        int(mlp.fc1.weight.shape[0]) == INTERMEDIATE
    ), f"expected vision intermediate={INTERMEDIATE}, got {mlp.fc1.weight.shape[0]}"

    return mlp


def _mesh_env() -> str:
    env = os.environ.get("MESH_DEVICE", "")
    if env in SUPPORTED_MESH_ENVS:
        return env
    return "P150x4" if is_blackhole() else ""


def _device_params():
    if not _mesh_env():
        pytest.skip(f"requires MESH_DEVICE ∈ {sorted(SUPPORTED_MESH_ENVS)} and Blackhole arch")
    return {
        "trace_region_size": 0,
        "num_command_queues": 1,
        # FABRIC_1D_RING: ring connectivity for P300c (also works on linear P150x4).
        "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
        # Blackhole COL dispatch frees row cores for compute.
        "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
    }


# ---------------------------------------------------------------------------
# Program config: max cores + L1-aware CB budget.
# ---------------------------------------------------------------------------

_DTYPE_TILE_BYTES = {
    ttnn.bfloat16: 2048,
    ttnn.bfloat8_b: 1088,
    ttnn.bfloat4_b: 544,
}

_L1_PER_CORE_BYTES = 1500 * 1024  # 1.5 MB per BH tensix core


def _largest_divisor_le(value: int, limit: int) -> int:
    for c in range(min(value, limit), 0, -1):
        if value % c == 0:
            return c
    return 1


def _bh_tp4_vision_mlp_pc(
    device,
    m_dim: int,
    k_dim: int,
    n_dim: int,
    *,
    l1_resident_bytes_per_core: int = 0,
):
    """Optimal 2D-mcast program config for BH TP4 vision MLP.

    Improvements over the generic ``_vision_matmul_program_config``:
    1. Finds the largest ``effective_grid_y`` ≤ ``grid_y`` that divides
       ``m_tiles`` so that BH devices with non-8 row counts still get an
       explicit (max-core) config instead of falling back to auto-config.
    2. Uses ``cb_budget = L1_PER_CORE - l1_resident_bytes_per_core`` for the
       CB-size gate, allowing larger ``out_block_h`` when activations are in L1
       (the existing DRAM-mode 1024 KB cap is too conservative in that case).
       Larger ``out_block_h`` → fewer outer-M iterations → fewer weight DRAM
       re-reads, which dominates runtime on these weight-bandwidth-bound shapes.
    """
    grid = device.compute_with_storage_grid_size()
    grid_x, grid_y = int(grid.x), int(grid.y)
    tile = 32

    if m_dim % tile or k_dim % tile or n_dim % tile:
        return None

    m_tiles = m_dim // tile
    k_tiles = k_dim // tile
    n_tiles = n_dim // tile

    # Largest effective_grid_y ≤ grid_y that divides m_tiles (max-core fit).
    eff_gy = grid_y
    while eff_gy > 1 and m_tiles % eff_gy != 0:
        eff_gy -= 1

    per_core_m = m_tiles // eff_gy
    per_core_n = (n_tiles + grid_x - 1) // grid_x  # ceil-div; 2D mcast pads trailing

    if per_core_n > 24 or per_core_m > 64:
        return None

    in0_block_w = _largest_divisor_le(k_tiles, 8)

    # CB budget for the matmul kernel, reduced by the L1 already occupied by
    # resident activation tensors (the in0 tensor that is already in L1, and
    # the output tensor being written into L1 during this op).
    cb_budget_kb = max(256, (_L1_PER_CORE_BYTES - l1_resident_bytes_per_core) // 1024)

    # Enumerate all divisors of per_core_m (largest first) to find the
    # out_block_h that maximises DST area then minimises outer-M iterations.
    divisors = sorted([h for h in range(1, per_core_m + 1) if per_core_m % h == 0], reverse=True)

    best_area = 0
    best_ob_h = 1
    best_sh = 1
    best_sw = 1

    for ob_h in divisors:
        # BF16 in0 CB: ob_h * in0_block_w tiles × 2 KB each.
        # BF8  interm CB: ob_h * per_core_n tiles × 1 KB each (×2 for out+interm).
        approx_interm_kb = (ob_h * per_core_n * 2048) // 1024
        approx_in0_kb = (ob_h * in0_block_w * 2 * 2048) // 1024
        if approx_interm_kb + approx_in0_kb > cb_budget_kb:
            continue

        cand_area = 0
        cand_h = cand_w = 1
        dst = 8  # tiles in DST register file (LoFi, fp32_dest_acc_en=False)
        for h in range(min(ob_h, dst), 0, -1):
            if ob_h % h != 0:
                continue
            for w in range(min(per_core_n, dst // h), 0, -1):
                if per_core_n % w != 0:
                    continue
                if h * w > cand_area:
                    cand_area = h * w
                    cand_h = h
                    cand_w = w
                break  # inner loop: take first (largest) w

        # Prefer max DST area; break ties by max ob_h (fewest outer-M iters).
        if cand_area > best_area or (cand_area == best_area and ob_h > best_ob_h):
            best_area = cand_area
            best_ob_h = ob_h
            best_sh = cand_h
            best_sw = cand_w

    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid_x, eff_gy),
        in0_block_w=in0_block_w,
        out_subblock_h=best_sh,
        out_subblock_w=best_sw,
        out_block_h=best_ob_h,
        out_block_w=per_core_n,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )


# ---------------------------------------------------------------------------
# Mesh open / close helpers.
# ---------------------------------------------------------------------------


def _open_tp4_mesh(updated_device_params):
    if len(ttnn.get_pcie_device_ids()) < TP:
        pytest.skip(f"TP={TP} requires at least {TP} PCIe devices")

    req_shape = TP_MESH_SHAPE
    sys_shape = tuple(ttnn._ttnn.multi_device.SystemMeshDescriptor().shape())
    sys_num = sys_shape[0] * sys_shape[1]

    if sys_num == TP:
        mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*req_shape), **updated_device_params)
        if sys_shape == (2, 2):
            mesh.reshape(ttnn.MeshShape(*req_shape))
        return mesh, None

    parent_shape = sys_shape
    if not (req_shape[0] <= parent_shape[0] and req_shape[1] <= parent_shape[1]):
        rotated = (parent_shape[1], parent_shape[0])
        if req_shape[0] <= rotated[0] and req_shape[1] <= rotated[1]:
            parent_shape = rotated
        else:
            pytest.skip(f"TP mesh {req_shape} does not fit system mesh {sys_shape}")

    parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*parent_shape), **updated_device_params)
    tp_mesh = parent.create_submesh(ttnn.MeshShape(*req_shape))
    tp_mesh.reshape(ttnn.MeshShape(*req_shape))
    return tp_mesh, parent


@pytest.fixture(scope="function")
def bh_tp4_mesh_device(device_params, silicon_arch_name, silicon_arch_blackhole):
    """1×4 Blackhole mesh (P150x4 or P300x2) with Fabric 1D Ring for CCL."""
    from tests.scripts.common import get_updated_device_params

    _ = silicon_arch_name, silicon_arch_blackhole
    if not is_blackhole():
        pytest.skip("requires Blackhole (P150x4 or P300x2)")

    mesh_env = _mesh_env()
    if not mesh_env:
        pytest.skip(f"requires MESH_DEVICE ∈ {sorted(SUPPORTED_MESH_ENVS)}")

    updated = get_updated_device_params(dict(device_params))
    fabric_config = updated.pop("fabric_config", None)
    fabric_tensix_config = updated.pop("fabric_tensix_config", None)
    reliability_mode = updated.pop("reliability_mode", None)
    if fabric_config:
        ttnn.set_fabric_config(
            fabric_config,
            reliability_mode or ttnn.FabricReliabilityMode.STRICT_INIT,
            None,
            fabric_tensix_config or ttnn.FabricTensixConfig.DISABLED,
            ttnn.FabricUDMMode.DISABLED,
            ttnn.FabricManagerMode.DEFAULT,
        )

    mesh_device, parent_mesh = _open_tp4_mesh(updated)
    if mesh_device.get_num_devices() != TP:
        pytest.skip(f"expected {TP} devices, got {mesh_device.get_num_devices()}")
    if mesh_device.arch() != ttnn.Arch.BLACKHOLE:
        pytest.skip(f"expected Blackhole, got {mesh_device.arch().name}")

    yield mesh_device

    ttnn.close_mesh_device(mesh_device)
    if parent_mesh is not None:
        ttnn.close_mesh_device(parent_mesh)
    if fabric_config:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


# ---------------------------------------------------------------------------
# Weight upload helpers.
# ---------------------------------------------------------------------------


def _to_tt_weight(w: torch.Tensor) -> torch.Tensor:
    """torch Linear [out, in] → TT layout [K=in, N=out]."""
    return w.t().contiguous()


def _upload(t: torch.Tensor, mesh, mapper, dtype=ttnn.bfloat8_b, *, mem=ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor:
    return ttnn.from_torch(t, device=mesh, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=mem, mesh_mapper=mapper)


# ---------------------------------------------------------------------------
# TP4 forward: all activations in L1, weights in DRAM.
# ---------------------------------------------------------------------------


def _mlp_tp4_l1_forward(
    x_tt,  # [1,1,M,H]  bf16  L1    replicated
    w1_tt,
    b1_tt,  # [H, I/TP]  bf8   DRAM  N-shard
    w3_tt,
    b3_tt,  # [H, I/TP]  bf8   DRAM  N-shard
    w2_tt,  # [I/TP, H]  bf8   DRAM  K-shard  (output: L1)
    *,
    gate_pc,
    up_pc,
    down_pc,  # pre-computed program configs (computed once, logged in test)
    compute_cfg,
) -> ttnn.Tensor:
    """TP4 vision SwiGLU with all activation tensors in L1.

    Column-parallel fc1/fc3 (N-shard):
      gate = fc1(x_replicated)  →  [1,1,M,I/TP]  bf8  L1
      up   = fc3(x_replicated)  →  [1,1,M,I/TP]  bf4  L1
      x is freed after both matmuls.

    Elementwise:
      gate_up_mul = silu(gate) * up  →  [1,1,M,I/TP]  bf8  L1

    Row-parallel fc2 (K-shard):
      partial = fc2(gate_up_mul)  →  [1,1,M,H]  bf8  L1  (partial sum per device)
      all_reduce                  →  [1,1,M,H]  bf8  L1  (full sum, every device)
      MLP ends here; no post-reduce bias (fc2 bias would be summed TP times).

    all_reduce L1 budget (64 cores, 8×8 grid):
      By this point gate/up/gum/partial are all deallocated; only the all_reduce
      output is written, at ~287 KB per core — well within 1.5 MB.
    """
    l1 = ttnn.L1_MEMORY_CONFIG

    gate = ttnn.linear(
        x_tt,
        w1_tt,
        bias=b1_tt,
        dtype=ttnn.bfloat8_b,
        memory_config=l1,
        compute_kernel_config=compute_cfg,
        program_config=gate_pc,
    )

    up = ttnn.linear(
        x_tt,
        w3_tt,
        bias=b3_tt,
        dtype=ttnn.bfloat4_b,
        memory_config=l1,
        compute_kernel_config=compute_cfg,
        program_config=up_pc,
    )

    ttnn.deallocate(x_tt)

    gate_up_mul = ttnn.mul(
        gate,
        up,
        input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        fast_and_approximate_mode=True,
        dtype=ttnn.bfloat8_b,
        memory_config=l1,
    )
    ttnn.deallocate(gate)
    ttnn.deallocate(up)

    partial = ttnn.linear(
        gate_up_mul,
        w2_tt,
        bias=None,
        dtype=ttnn.bfloat8_b,
        memory_config=l1,
        compute_kernel_config=compute_cfg,
        program_config=down_pc,
    )
    ttnn.deallocate(gate_up_mul)

    # all_reduce sums partial [M, H] across TP devices; every device gets the
    # full reduced result in L1.  Not trace-compatible (dynamic intermediate
    # allocation); use reduce_scatter + all_gather if tracing is required.
    out = ttnn.all_reduce(partial, num_links=1, cluster_axis=1, topology=ttnn.Topology.Ring, memory_config=l1)
    ttnn.deallocate(partial)
    return out


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
@run_for_blackhole()
def test_vision_mlp_tp4_n_k_shard_blackhole(bh_tp4_mesh_device):
    """Vision SwiGLU TP4 on Blackhole P300c: real dots.ocr weights, L1 activations.

    Column-parallel fc1/fc3 (N-shard, no collective) + row-parallel fc2
    (K-shard + single ttnn.all_reduce).  All local op I/O is L1 interleaved.
    Weights come from the dots.ocr vision tower (blocks[0].mlp).
    PCC is checked against the float torch reference using the same weights.
    """
    mesh_device = bh_tp4_mesh_device
    mesh_env = _mesh_env()

    assert SEQ_LEN % TP == 0
    assert INTERMEDIATE % TP == 0
    itp = INTERMEDIATE // TP  # 1056 — I per TP device

    # ---- load real dots.ocr vision MLP weights ------------------------------
    hf_mlp = _load_vision_mlp()
    # fc2 bias is not applied in the TP forward (K-shard: would be summed TP
    # times across the all_reduce).  Zero it in the reference for a fair PCC.
    if hf_mlp.fc2.bias is not None:
        hf_mlp.fc2.bias.data.zero_()

    torch.manual_seed(0xB1AC_4B0E)
    x = torch.randn(1, 1, SEQ_LEN, HIDDEN, dtype=torch.bfloat16) * 0.1
    with torch.no_grad():
        ref = hf_mlp.float()(x.float()).to(torch.float32)

    # ---- mesh mappers -------------------------------------------------------
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    shard_n = ttnn.ShardTensorToMesh(mesh_device, dim=-1)  # column-parallel (N)
    shard_k = ttnn.ShardTensorToMesh(mesh_device, dim=-2)  # row-parallel    (K)

    has_bias = lambda layer: layer.bias is not None  # noqa: E731

    # ---- weight upload (DRAM) -----------------------------------------------
    # fc1 / fc3 (gate / up): N-shard → [H, I/TP] per device.
    w1 = _upload(_to_tt_weight(hf_mlp.fc1.weight.data), mesh_device, shard_n, dtype=ttnn.bfloat8_b)
    b1 = _upload(hf_mlp.fc1.bias.data.reshape(1, -1), mesh_device, shard_n) if has_bias(hf_mlp.fc1) else None

    w3 = _upload(_to_tt_weight(hf_mlp.fc3.weight.data), mesh_device, shard_n, dtype=ttnn.bfloat8_b)
    b3 = _upload(hf_mlp.fc3.bias.data.reshape(1, -1), mesh_device, shard_n) if has_bias(hf_mlp.fc3) else None

    # fc2 (down): K-shard → [I/TP, H] per device.  No bias (see above).
    w2 = _upload(_to_tt_weight(hf_mlp.fc2.weight.data), mesh_device, shard_k, dtype=ttnn.bfloat8_b)

    # ---- input upload (L1) --------------------------------------------------
    # Vision tower body is replicated → full HIDDEN on every device.
    x_tt = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,  # L1 from the start
        mesh_mapper=rep,
    )

    # ---- program configs (computed once, logged, then passed to forward) ----
    grid = mesh_device.compute_with_storage_grid_size()
    nc = int(grid.x) * int(grid.y)
    tile = 32
    m_tiles = SEQ_LEN // tile
    h_tiles = HIDDEN // tile
    itp_tiles = itp // tile
    xB = ((m_tiles * h_tiles + nc - 1) // nc) * _DTYPE_TILE_BYTES[ttnn.bfloat16]
    gB = ((m_tiles * itp_tiles + nc - 1) // nc) * _DTYPE_TILE_BYTES[ttnn.bfloat8_b]
    uB = ((m_tiles * itp_tiles + nc - 1) // nc) * _DTYPE_TILE_BYTES[ttnn.bfloat4_b]
    partB = ((m_tiles * h_tiles + nc - 1) // nc) * _DTYPE_TILE_BYTES[ttnn.bfloat8_b]

    gate_pc = _bh_tp4_vision_mlp_pc(mesh_device, SEQ_LEN, HIDDEN, itp, l1_resident_bytes_per_core=xB + gB)
    up_pc = _bh_tp4_vision_mlp_pc(mesh_device, SEQ_LEN, HIDDEN, itp, l1_resident_bytes_per_core=xB + uB)
    down_pc = _bh_tp4_vision_mlp_pc(mesh_device, SEQ_LEN, itp, HIDDEN, l1_resident_bytes_per_core=gB + partB)

    compute_cfg = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    def _log_pc(label, pc):
        if pc is not None:
            logger.info(
                f"  {label}: grid={pc.compute_with_storage_grid_size} "
                f"per_core_M={pc.per_core_M} per_core_N={pc.per_core_N} "
                f"in0_block_w={pc.in0_block_w} out_block_h={pc.out_block_h} "
                f"sub=({pc.out_subblock_h},{pc.out_subblock_w}) "
                f"dst_area={pc.out_subblock_h * pc.out_subblock_w}"
            )
        else:
            logger.info(f"  {label}: None (auto-config)")

    logger.info(f"[vision_mlp_tp4_bh grid=({grid.x},{grid.y}) num_cores={nc}]")
    logger.info(f"  x={xB//1024}KB  gate={gB//1024}KB  up={uB//1024}KB  partial={partB//1024}KB per core")
    _log_pc("gate  [M,H,I/TP]", gate_pc)
    _log_pc("up    [M,H,I/TP]", up_pc)
    _log_pc("down  [M,I/TP,H]", down_pc)

    # ---- TP4 forward --------------------------------------------------------
    out_tt = _mlp_tp4_l1_forward(
        x_tt,
        w1,
        b1,
        w3,
        b3,
        w2,
        gate_pc=gate_pc,
        up_pc=up_pc,
        down_pc=down_pc,
        compute_cfg=compute_cfg,
    )
    ttnn.synchronize_device(mesh_device)

    # After all_reduce every device holds the full [1,1,SEQ_LEN,HIDDEN] sum.
    # Check PCC on all 4 devices — they must all agree with the torch reference.
    device_tensors = ttnn.get_device_tensors(out_tt)
    assert len(device_tensors) == TP, f"expected {TP} device tensors, got {len(device_tensors)}"

    failures = []
    for dev_idx, dev_t in enumerate(device_tensors):
        out_dev = dev_t.cpu().to_torch().float().reshape(ref.shape)
        passed_dev, pcc_dev = comp_pcc(ref, out_dev, PCC_THRESHOLD)
        logger.info(
            f"[vision_mlp_tp4_n_k_shard arch={mesh_device.arch().name} env={mesh_env} "
            f"TP={TP} S={SEQ_LEN} H={HIDDEN} I={INTERMEDIATE} I/TP={itp} "
            f"fc1/fc3:N-shard fc2:K-shard all-ops-L1 dev={dev_idx}] "
            f"pcc={float(pcc_dev):.6f} (threshold {PCC_THRESHOLD})"
        )
        if not passed_dev:
            failures.append((dev_idx, float(pcc_dev)))

    ttnn.deallocate(out_tt)

    assert not failures, (
        "Vision MLP TP4 N/K-shard failed PCC on "
        + ", ".join(f"dev{i} pcc={p:.6f}" for i, p in failures)
        + f" (threshold {PCC_THRESHOLD})"
    )
