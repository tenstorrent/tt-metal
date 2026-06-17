# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Unit test for the Qwen3.5 SwiGLU MLP (down(silu(gate(x)) * up(x))) vs the HF golden.

Build ONE HF Qwen3_5MLP with random weights, hand its state_dict to the TT Qwen35MLP, then
PCC-check the TT output against HF.

Every case runs on both a (1, 1) single device and the (1, 4) tensor-parallel mesh (the latter
with the 1D fabric the out-projection reduce-scatter rides), across batch ∈ {1, 32} and
seq ∈ {32, 512}. test_mlp runs the forward eagerly; test_mlp_trace captures it as a ttnn trace
and replays it, so the traced path the demo uses is exercised too.

Run:
    pytest models/demos/blackhole/qwen3_5_9b/tests/unit/test_mlp.py -v
"""
import os

import pytest
import torch
from loguru import logger
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen3_5_9b.tt.mlp import Qwen35MLP
from models.demos.blackhole.qwen3_5_9b.tt.model_config import Qwen35ModelArgs
from models.tt_transformers.tt.ccl import TT_CCL

### Test Parameters & Fixtures ─────────────────────────────────────────────────────────
os.environ.setdefault("HF_MODEL", "Qwen/Qwen3.5-9B")

BATCHES = [1, 32]
SEQ_LENS = [32, 512]
PCC = 0.99
TRACE_REGION_SIZE = 23887872


@pytest.fixture
def setup(mesh_device):
    args = Qwen35ModelArgs(mesh_device, max_batch_size=max(BATCHES), max_seq_len=max(SEQ_LENS))
    cfg = args.hf_config.get_text_config()
    hf_mlp = Qwen3_5MLP(cfg, intermediate_size=cfg.intermediate_size).to(torch.float32).eval()
    return hf_mlp, hf_mlp.state_dict(), args


# ── Helpers (host torch ⇄ possibly-sharded TT tensors) ───────────────────────────────
def _build_tt_mlp(mesh_device, state_dict, args):
    """The TT MLP under test. TT_CCL only on a multi-device mesh (it drives the reduce-scatter);
    tensor_cache_path stays None — caching random weights would corrupt a later re-run."""
    tt_ccl = TT_CCL(mesh_device) if mesh_device.get_num_devices() > 1 else None
    return Qwen35MLP(mesh_device, state_dict=state_dict, args=args, tt_ccl=tt_ccl)


def _to_device(x, mesh_device):
    """Activation as the MLP input layout: [1, 1, M, dim] replicated to every device, bf16 DRAM."""
    return ttnn.from_torch(
        x.to(torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _from_device(out, mesh_device):
    """forward reduce-scatters its [1, 1, M, dim] output along the hidden dim on TP, so concat
    dim=3 reassembles it; on a single device the output already holds the full dim (dim=0 no-op)."""
    nd = mesh_device.get_num_devices()
    composer = ttnn.ConcatMeshToTensor(mesh_device, dim=3 if nd > 1 else 0)
    return ttnn.to_torch(out, mesh_composer=composer)[0, 0].float()  # [M, dim]


# ── Tests ────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        ((1, 1), {"trace_region_size": TRACE_REGION_SIZE}),
        ((1, 4), {"trace_region_size": TRACE_REGION_SIZE, "fabric_config": ttnn.FabricConfig.FABRIC_1D}),
    ],
    indirect=True,
)
@pytest.mark.parametrize("seq_len", SEQ_LENS, ids=lambda s: f"seq{s}")
@pytest.mark.parametrize("batch", BATCHES, ids=lambda b: f"b{b}")
def test_mlp(mesh_device, batch, seq_len, setup, reset_seeds, ensure_gc):
    """Eager SwiGLU MLP PCC vs the HF golden across batch × seq on (1, 1) and the (1, 4) TP mesh."""

    # 1. Build reference MLP and TT MLP
    hf_mlp, state_dict, args = setup
    tt_mlp = _build_tt_mlp(mesh_device, state_dict, args)

    # 2. Instantiate random input tensor
    x = torch.randn(batch, seq_len, args.dim, dtype=torch.float32)  # [B, seq, dim]
    x_tt = _to_device(x.reshape(batch, 1, seq_len, args.dim), mesh_device)  # [B, 1, seq, dim]
    out_ref = hf_mlp(x)

    tt_out = tt_mlp.forward(x_tt)
    tt_out = _from_device(tt_out, mesh_device).reshape(batch, seq_len, args.dim)

    passing, pcc = comp_pcc(out_ref, tt_out, PCC)
    logger.info(f"MLP PCC (mesh={tuple(mesh_device.shape)}, b={batch}, S={seq_len}) = {pcc}")
    assert passing, f"MLP PCC too low (b={batch}, S={seq_len}): {pcc}"


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        ((1, 1), {"trace_region_size": TRACE_REGION_SIZE}),
        ((1, 4), {"trace_region_size": TRACE_REGION_SIZE, "fabric_config": ttnn.FabricConfig.FABRIC_1D}),
    ],
    indirect=True,
)
@pytest.mark.parametrize("seq_len", SEQ_LENS, ids=lambda s: f"seq{s}")
@pytest.mark.parametrize("batch", BATCHES, ids=lambda b: f"b{b}")
def test_mlp_trace(mesh_device, batch, seq_len, setup, reset_seeds, ensure_gc):
    """Same PCC check, but the forward runs as a captured ttnn trace — the path the demo replays to
    collapse the MLP's ops into one device dispatch. Confirms the graph (3 matmuls, SiLU, the TP
    reduce-scatter) captures cleanly and the replay reproduces the HF golden."""

    # 1. Build reference MLP and TT MLP
    hf_mlp, state_dict, args = setup
    tt_mlp = _build_tt_mlp(mesh_device, state_dict, args)

    # 2. Instantiate input tensor
    x = torch.randn(batch, seq_len, args.dim, dtype=torch.float32)  # [B, seq, dim]
    x_tt = _to_device(x.reshape(batch, 1, seq_len, args.dim), mesh_device)  # [B, 1, seq, dim]
    out_ref = hf_mlp(x)

    # 3. Compile the kernels. Trace capture only works when there are compiled kernels.
    _ = tt_mlp.forward(x_tt)

    # 4. Capture once. `out` is the persistent output buffer the trace writes into.
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = tt_mlp.forward(x_tt)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)

    # 5. Replay trace: recomputes MLP(x) into `out`'s buffer with a single dispatch, then read it back.
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    tt_out = _from_device(out, mesh_device).reshape(batch, seq_len, args.dim)
    ttnn.release_trace(mesh_device, tid)

    passing, pcc = comp_pcc(out_ref, tt_out, PCC)
    logger.info(f"MLP TRACE PCC (mesh={tuple(mesh_device.shape)}, b={batch}, S={seq_len}) = {pcc}")
    assert passing, f"MLP trace PCC too low (b={batch}, S={seq_len}): {pcc}"
