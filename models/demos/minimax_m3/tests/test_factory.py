# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0


import json
import os

import pytest
import torch
from transformers import AutoConfig

import ttnn

from ..config import MeshConfig
from ..tt.ccl import L1_SMALL_SIZE, CCLManager
from ..tt.model_config import ModelArgs
from ..utils.general_utils import get_default_num_links

# Bundled MiniMax-M3 config.json (dims only — no model code is vendored). HF-reference
# tests that need the actual modeling classes load them from a real checkpoint via
# HF_MODEL (see requires_hf_reference below).
_CONFIG_JSON = os.path.join(os.path.dirname(__file__), "..", "configs", "MiniMax-M3", "config.json")


def minimax_config_dims() -> dict:
    """Load MiniMax-M3 dimension constants from the bundled config.json (no HF needed)."""
    with open(_CONFIG_JSON) as f:
        return json.load(f)


_HF_MODEL = os.getenv("HF_MODEL")
# Tests that compare against the HuggingFace reference need the model's modeling code,
# which ships WITH the checkpoint (loaded at runtime via trust_remote_code) — we do NOT
# vendor it into the repo. Point HF_MODEL at a downloaded MiniMax-M3 checkpoint to run them.
requires_hf_reference = pytest.mark.skipif(
    not (_HF_MODEL and os.path.isdir(_HF_MODEL)),
    reason="set HF_MODEL to a downloaded MiniMax-M3 checkpoint (carries modeling_minimax_m3.py) to run HF-reference tests",
)


def hf_model_path() -> str:
    return _HF_MODEL


class TestFactory:
    # Common test configurations
    MESH_SHAPES = {"1x4": (1, 4), "4x8": (4, 8), "1x8": (1, 8), "4x4": (4, 4), "2x4": (2, 4), "1x1": (1, 1)}

    BATCH_SEQ_CONFIGS = [
        (1, 1),  # Single token
        (1, 32),  # Small sequence
        (1, 128),  # Medium sequence
    ]

    @staticmethod
    def setup_test(mesh_device, use_real_weights=True, dtype=ttnn.bfloat8_b):
        """Universal test setup - replaces all the duplicated setup code"""

        # Use mesh_device as-is (already created by conftest.py fixture)
        mesh_shape = mesh_device.shape

        # Setup ModelArgs (no import-time loading)
        model_args = ModelArgs(mesh_device=mesh_device, dummy_weights=not use_real_weights)

        # Setup mesh config using actual mesh shape

        mesh_config = MeshConfig(mesh_shape, tp=mesh_shape[1])

        # Setup CCL
        ccl_manager = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device))

        config = AutoConfig.from_pretrained(model_args.model_path, trust_remote_code=True)

        # Only cache tensors to disk when using real weights; with dummy (random) weights
        # there is nothing to persist and the model_path may not be a writable local directory.
        tensor_cache_path = model_args.weight_cache_path(dtype) if use_real_weights else None

        return {
            "mesh_device": mesh_device,
            "model_args": model_args,
            "mesh_config": mesh_config,
            "ccl_manager": ccl_manager,
            "config": config,
            "dtype": dtype,
            "tensor_cache_path": tensor_cache_path,
        }


def parametrize_mesh_with_fabric(mesh_shapes=None, linear_fabric=False):
    """Universal mesh + fabric parametrization for minimax_m3 tests.

    Generates a paired ``(mesh_device, device_params)`` parametrize. Each
    case opens a mesh of the requested shape directly (no submesh carving
    in test bodies) and configures the appropriate fabric for that shape.
    Each parametrize case has a single id like ``1x1`` / ``1x2`` / ``1x4`` /
    ``1x8`` / ``4x8``, so ``pytest -k 1x2`` (or ``-k 4x8``) filters cleanly
    without dual-id confusion from a separate inner ``mesh_shape`` parametrize.

    Auto-filters to the shapes that fit on the current system. Default
    shapes: (1,1) single card, (1,2) 2xP150, (1,4) QuietBox 2 (2xP300),
    (1,8) LoudBox / T3K, (4,8) Galaxy. Pass an explicit ``mesh_shapes`` list
    to override (useful for tests that only make sense at one TP factor).

    Fabric: ``(1,1)`` disables fabric (no inter-chip topology to ring
    around). Multi-device shapes use ``FABRIC_1D_RING`` — minimax_m3's CCL
    operations (reduce_scatter, all_gather, all_reduce) all use the ring
    topology.

    When ``CI=true`` is set in the environment, only the largest mesh shape
    that fits on the current system is parametrized. This lets one yaml
    entry target multiple SKUs without per-SKU ``-k "1xN"`` filters: each
    runner picks the largest mesh its device count supports. Manual / non-CI
    invocations are unchanged (all-shapes-that-fit, ``-k`` available for fast
    iteration).

    Usage:
        @parametrize_mesh_with_fabric()              # all shapes that fit
        @parametrize_mesh_with_fabric([(1, 8)])      # 1x8 only

        pytest -k 1x1   # single card             (manual / non-CI)
        pytest -k 1x2   # 2xP150                  (manual / non-CI)
        pytest -k 1x4   # QuietBox 2 (2xP300)     (manual / non-CI)
        pytest -k 1x8   # LoudBox / T3K           (manual / non-CI)
        pytest -k 4x8   # Galaxy                  (manual / non-CI)
    """
    num_devices = ttnn.get_num_devices()
    if mesh_shapes is None:
        all_shapes = [(1, 1), (1, 2), (1, 4), (1, 8), (4, 8)]
        mesh_shapes = [s for s in all_shapes if s[0] * s[1] <= num_devices]
    else:
        # User-provided shapes: still filter to those that fit, so an explicit
        # mesh_shapes=[(1,8)] decorator gracefully skips on smaller systems.
        mesh_shapes = [s for s in mesh_shapes if s[0] * s[1] <= num_devices]

    # CI mode: pick only the largest fitting shape so that one yaml entry can
    # target multiple SKUs and let each runner select the appropriate mesh.
    if os.getenv("CI") == "true" and len(mesh_shapes) > 1:
        mesh_shapes = [max(mesh_shapes, key=lambda s: s[0] * s[1])]

    if not mesh_shapes:
        params = [
            pytest.param(
                (1, 1),
                {"fabric_config": None, "trace_region_size": 100000000, "l1_small_size": L1_SMALL_SIZE},
                id="1x1",
                marks=pytest.mark.skip(reason="No supported minimax_m3 mesh shape fits on this system"),
            )
        ]
    else:
        # Multi-device fabric: default is FABRIC_1D_RING (torus), but a plain-MESH
        # Galaxy (no wrap-around links) can only do FABRIC_1D — pass linear_fabric=True
        # there (and use ttnn.Topology.Linear in the CCLManager). See galaxy_mesh_smoke.py.
        multidev_fabric = ttnn.FabricConfig.FABRIC_1D if linear_fabric else ttnn.FabricConfig.FABRIC_1D_RING
        params = [
            pytest.param(
                shape,
                {
                    "fabric_config": (None if shape == (1, 1) else multidev_fabric),
                    "trace_region_size": 100000000,
                    "l1_small_size": L1_SMALL_SIZE,
                },
                id=f"{shape[0]}x{shape[1]}",
            )
            for shape in mesh_shapes
        ]

    def decorator(func):
        return pytest.mark.parametrize("mesh_device, device_params", params, indirect=True)(func)

    return decorator


def parametrize_batch_seq(configs=None, ids=None):
    """Universal batch/seq parametrization"""
    configs = configs or [(1, 1), (1, 32)]
    ids = ids or [
        (
            f"prefill_{seq_len//1024 if seq_len > 1024 else seq_len}" + ("k" if seq_len > 1024 else "")
            if seq_len > 1
            else "decode_mode"
        )
        for batch_size, seq_len in configs
    ]
    return pytest.mark.parametrize("batch_size, seq_len", configs, ids=ids)


def parametrize_weights(use_real=False):
    """Universal weight parametrization"""
    return pytest.mark.parametrize("use_real_weights", [use_real], ids=["real" if use_real else "random"])


def compose_tp_hidden(device_tensors, row, cols):
    """Reassemble one mesh row's residual hidden onto the host.

    Under a sharded residual each TP column holds ``emb/tp``; concat them on the last dim.
    Under a replicated residual column 0 already has full emb.
    """
    from models.demos.minimax_m3.tt.residual import use_sharded_residual

    if use_sharded_residual() and cols > 1:
        return torch.cat(
            [ttnn.to_torch(device_tensors[row * cols + c]).float() for c in range(cols)],
            dim=-1,
        )
    return ttnn.to_torch(device_tensors[row * cols]).float()


# MiniMax-M3 attention dims of the random-weight attention-layer tests.
M3_HIDDEN, M3_NQ, M3_NKV, M3_HEAD_DIM, M3_ROTARY_DIM = 6144, 64, 4, 128, 64
M3_NIDX, M3_INDEX_DIM = 4, 128


def random_sp_attention(mesh_device, *, chunk_size, max_seq_len, is_sparse, global_layer_idx, local_layer_idx=None):
    """A random-weight M3 attention layer on an SP=rows x TP=cols mesh, with the runtime's whole-cache indexed RoPE
    (TtPrefillRuntime._build_indexed_rope). Returns a namespace for packed_attention_forward."""
    from types import SimpleNamespace

    from models.common.utils import block_cyclic_reorder

    from ..tt.attention import Attention, AttentionConfig
    from ..tt.attention_configs import MiniMaxM3AttentionProgramConfig
    from ..tt.model import create_rope_setup
    from ..utils.weight_conversion import convert_hf_qkv_to_meta_format_partial

    rows, cols = tuple(mesh_device.shape)
    torch.manual_seed(0)
    hf_state = {
        "q_proj.weight": torch.rand(M3_NQ * M3_HEAD_DIM, M3_HIDDEN) * 0.02,
        "k_proj.weight": torch.rand(M3_NKV * M3_HEAD_DIM, M3_HIDDEN) * 0.02,
        "v_proj.weight": torch.rand(M3_NKV * M3_HEAD_DIM, M3_HIDDEN) * 0.02,
        "o_proj.weight": torch.rand(M3_HIDDEN, M3_NQ * M3_HEAD_DIM) * 0.02,
        "q_norm.weight": torch.randn(M3_HEAD_DIM) * 0.1,
        "k_norm.weight": torch.randn(M3_HEAD_DIM) * 0.1,
    }
    if is_sparse:
        hf_state |= {
            "index_q_proj.weight": torch.rand(M3_NIDX * M3_INDEX_DIM, M3_HIDDEN) * 0.02,
            "index_k_proj.weight": torch.rand(M3_INDEX_DIM, M3_HIDDEN) * 0.02,
            "index_q_norm.weight": torch.randn(M3_INDEX_DIM) * 0.1,
            "index_k_norm.weight": torch.randn(M3_INDEX_DIM) * 0.1,
        }
    state = convert_hf_qkv_to_meta_format_partial(hf_state, M3_HEAD_DIM, M3_ROTARY_DIM)
    ccl = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device), topology=ttnn.Topology.Linear)
    hf_config = SimpleNamespace(
        hidden_size=M3_HIDDEN,
        num_attention_heads=M3_NQ,
        num_key_value_heads=M3_NKV,
        head_dim=M3_HEAD_DIM,
        rotary_dim=M3_ROTARY_DIM,
        rope_theta=5_000_000.0,
        rope_scaling=None,
        rms_norm_eps=1e-6,
        max_position_embeddings=max_seq_len,
        use_qk_norm=True,
        use_gemma_norm=True,
    )
    rope_setup = create_rope_setup(mesh_device=mesh_device, hf_config=hf_config, datatype=ttnn.bfloat16)
    attn = Attention(
        mesh_device=mesh_device,
        config=AttentionConfig(
            hidden_size=M3_HIDDEN,
            num_heads=M3_NQ,
            num_kv_heads=M3_NKV,
            head_dim=M3_HEAD_DIM,
            rotary_dim=M3_ROTARY_DIM,
            rms_norm_eps=hf_config.rms_norm_eps,
            use_qk_norm=True,
            use_gemma_norm=True,
            max_seq_len=max_seq_len,
            max_local_batch_size=1,
            is_sparse=is_sparse,
            sequence_parallel=True,
        ),
        state_dict=state,
        ccl_manager=ccl,
        mesh_config=MeshConfig((rows, cols), tp=cols),
        program_config=MiniMaxM3AttentionProgramConfig(),
        global_layer_idx=global_layer_idx,
        local_layer_idx=local_layer_idx,
        transformation_mats=rope_setup.get_both_trans_mats(),
    )
    seq_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(2, None))

    def indexed_rope(dev_mat):
        full = ttnn.to_torch(ttnn.get_device_tensors(dev_mat)[0])[:, :, :max_seq_len, :]
        bc = block_cyclic_reorder(full, chunk_size // rows, rows, seq_dim=2)
        return ttnn.from_torch(
            bc, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=seq_mapper
        )

    rope = [indexed_rope(rope_setup.cos_matrix_prefill), indexed_rope(rope_setup.sin_matrix_prefill)]
    return SimpleNamespace(
        attn=attn, ccl=ccl, rope=rope, seq_mapper=seq_mapper, mesh_device=mesh_device, chunk_size=chunk_size
    )


def packed_attention_forward(layer, x_chunks, chunks, kv_cache):
    """One attention forward of ``layer`` (random_sp_attention) over ``chunks``; x_chunks are natural-order
    [chunk_size, hidden] rows per chunk. Returns each chunk's output rows chip-major: rank r's rows of a chunk are
    ``pack_forward_tokens``' positions of that rank, in order."""
    from ..tt.prefill_chunk import pack_forward_tokens

    rows, cols = tuple(layer.mesh_device.shape)
    c, n = layer.chunk_size, len(chunks)
    idx = pack_forward_tokens([[k * c + i for i in range(c)] for k in range(n)], [e.actual_start for e in chunks], rows)
    x = torch.cat(x_chunks)[idx].reshape(1, 1, -1, M3_HIDDEN)
    x_tt = ttnn.from_torch(
        x, device=layer.mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=layer.seq_mapper
    )
    out = layer.attn(x_tt, rope_mats=layer.rope, kv_cache=kv_cache, indexed_rope=True, chunks=chunks)
    dts = ttnn.get_device_tensors(out)
    per_chip = [compose_tp_hidden(dts, r, cols).reshape(n, c // rows, M3_HIDDEN) for r in range(rows)]
    ttnn.deallocate(out)
    return [torch.cat([chip[k] for chip in per_chip]) for k in range(n)]


# Test helper functions
def compare_tensors(tt_tensor, torch_tensor, mesh_device, pcc_threshold=0.99):
    """Universal tensor comparison - handles both TT tensors and already-converted torch tensors"""
    from models.common.utility_functions import comp_pcc

    # Check if tt_tensor is already a torch tensor
    if isinstance(tt_tensor, torch.Tensor):
        # Already converted, use directly
        tt_torch = tt_tensor
    else:
        # Convert TT tensor to torch
        mesh_shape = mesh_device.shape
        tt_torch = ttnn.to_torch(
            tt_tensor, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=(-2, -1))
        )

    return comp_pcc(torch_tensor, tt_torch, pcc_threshold)
