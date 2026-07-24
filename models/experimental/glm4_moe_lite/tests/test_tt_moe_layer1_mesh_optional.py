# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import ttnn

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

from models.experimental.glm4_moe_lite.tt.config import Glm4MoeLiteHParams
from models.experimental.glm4_moe_lite.tt.layer_weights import convert_decoder_layer_weights
from models.experimental.glm4_moe_lite.tt.moe_tt import (
    _clear_buffered_moe_all_reduce_cache,
    _moe_all_reduce_across_mesh,
    create_moe_runtime,
    moe_sparse_experts_forward_tt,
)
from models.experimental.glm4_moe_lite.tt.reference_moe import run_layer_moe_reference_from_hidden_states
from models.experimental.glm4_moe_lite.tt.weights import (
    find_missing_shards,
    load_glm_lazy_state_dict,
    resolve_best_effort_snapshot_dir,
)


def _load_hparams(snapshot_dir: Path) -> Glm4MoeLiteHParams:
    cfg = json.loads((Path(snapshot_dir) / "config.json").read_text())
    hparams = Glm4MoeLiteHParams.from_hf_config(SimpleNamespace(**cfg))
    hparams.validate()
    return hparams


def _mesh_shape_from_env(default: tuple[int, int] = (1, 8)) -> tuple[int, int]:
    raw = os.environ.get("TT_TEST_MESH_SHAPE", "").strip().lower()
    if not raw:
        return default
    raw = raw.replace("x", ",")
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"Invalid TT_TEST_MESH_SHAPE={raw!r}; expected 'rowsxcols' (e.g. '1x8')")
    return (int(parts[0]), int(parts[1]))


def _set_galaxy_fabric() -> None:
    if ttnn.cluster.get_cluster_type() != ttnn.cluster.ClusterType.GALAXY:
        return
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    )


@pytest.mark.skipif(
    os.environ.get("TT_ENABLE_HW_TESTS") != "1",
    reason="Enable with TT_ENABLE_HW_TESTS=1 (requires Tenstorrent device access).",
)
@pytest.mark.skipif(
    os.environ.get("TT_ENABLE_MULTI_DEVICE_TESTS") != "1",
    reason="Enable with TT_ENABLE_MULTI_DEVICE_TESTS=1 (opens a multi-device mesh).",
)
@pytest.mark.parametrize("num_links", [1, 3, 4])
@pytest.mark.parametrize("topology", [ttnn.Topology.Linear, ttnn.Topology.Ring], ids=["linear", "ring"])
@pytest.mark.parametrize("buffered", [False, True], ids=["gather_reduce", "buffered_all_reduce"])
def test_collective_epilogue_matches_existing_path_on_galaxy(
    num_links: int, topology: ttnn.Topology, buffered: bool
) -> None:
    mesh_rows, mesh_cols = _mesh_shape_from_env(default=(4, 8))
    if (mesh_rows, mesh_cols) != (4, 8):
        pytest.skip("Collective epilogue validation requires a 4x8 mesh")

    _set_galaxy_fabric()
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(mesh_rows, mesh_cols),
        dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER),
    )
    tensors: list[ttnn.Tensor] = []
    try:
        torch.manual_seed(2026)
        routed = torch.randn((1, 1, 1, 2048), dtype=torch.bfloat16)
        shared = torch.randn_like(routed)
        residual = torch.randn_like(routed)
        mapper = ttnn.ReplicateTensorToMesh(mesh_device)

        def to_mesh(host: torch.Tensor) -> ttnn.Tensor:
            tensor = ttnn.from_torch(
                host,
                device=mesh_device,
                mesh_mapper=mapper,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            tensors.append(tensor)
            return tensor

        baseline_routed = _moe_all_reduce_across_mesh(
            to_mesh(routed),
            device=mesh_device,
            num_links=num_links,
            topology=topology,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tensors.append(baseline_routed)
        baseline = ttnn.add(baseline_routed, to_mesh(shared), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tensors.append(baseline)
        baseline_final = ttnn.add(baseline, to_mesh(residual), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tensors.append(baseline_final)

        fused = _moe_all_reduce_across_mesh(
            to_mesh(routed),
            device=mesh_device,
            num_links=num_links,
            topology=topology,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            epilogue_input_a=to_mesh(shared),
            epilogue_input_b=to_mesh(residual),
            use_buffered_all_reduce=buffered,
        )
        tensors.append(fused)
        ttnn.synchronize_device(mesh_device)

        baseline_shards = [ttnn.to_torch(tensor).float() for tensor in ttnn.get_device_tensors(baseline_final)]
        fused_shards = [ttnn.to_torch(tensor).float() for tensor in ttnn.get_device_tensors(fused)]
        assert len(fused_shards) == mesh_rows * mesh_cols
        for baseline_shard, fused_shard in zip(baseline_shards, fused_shards):
            ok, msg = comp_pcc(baseline_shard[:, :, :1, :], fused_shard[:, :, :1, :], pcc=0.9999)
            assert ok, msg
            assert torch.allclose(fused_shard, fused_shards[0], rtol=0.0, atol=0.0)
    finally:
        for tensor in reversed(tensors):
            try:
                ttnn.deallocate(tensor, force=False)
            except Exception:
                pass
        _clear_buffered_moe_all_reduce_cache(mesh_device)
        ttnn.close_mesh_device(mesh_device)


@pytest.mark.skipif(
    os.environ.get("TT_ENABLE_HW_TESTS") != "1",
    reason="Enable with TT_ENABLE_HW_TESTS=1 (requires Tenstorrent device access).",
)
@pytest.mark.skipif(
    os.environ.get("TT_ENABLE_LARGE_MODEL_TESTS") != "1",
    reason="Enable with TT_ENABLE_LARGE_MODEL_TESTS=1 (loads routed expert weights).",
)
@pytest.mark.skipif(
    os.environ.get("TT_ENABLE_MULTI_DEVICE_TESTS") != "1",
    reason="Enable with TT_ENABLE_MULTI_DEVICE_TESTS=1 (opens a multi-device mesh).",
)
@pytest.mark.parametrize(
    ("fused_epilogue", "buffered"),
    [(False, False), (True, False), (True, True)],
    ids=["baseline", "fused_epilogue", "buffered_fused_epilogue"],
)
def test_layer1_routed_experts_mesh_matches_reference_given_reference_routing(
    monkeypatch: pytest.MonkeyPatch,
    fused_epilogue: bool,
    buffered: bool,
) -> None:
    """Validate local-expert sharding + all-reduce aggregation on a real mesh."""
    snap = resolve_best_effort_snapshot_dir("zai-org/GLM-4.7-Flash")
    missing = find_missing_shards(snap)
    if missing:
        pytest.skip(f"GLM snapshot missing {len(missing)} shards; run ensure_glm47_weights.sh first.")

    # Use BF16 expert weights for accuracy in the comparison.
    monkeypatch.setenv("GLM4_MOE_LITE_EXPERTS_TT_DTYPE", "bf16")

    hparams = _load_hparams(Path(snap))
    layer_idx = 1
    tokens = 32  # sparse kernel requires multiple of 32
    hidden = int(hparams.hidden_size)

    torch.manual_seed(0)
    x = torch.randn((tokens, hidden), dtype=torch.float32).to(torch.bfloat16).to(torch.float32)

    # CPU oracle (routing + experts + shared). We'll compare only routed_out.
    ref = run_layer_moe_reference_from_hidden_states(Path(snap), layer_idx=layer_idx, hidden_states=x)

    mesh_rows, mesh_cols = _mesh_shape_from_env()
    _set_galaxy_fabric()
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(mesh_rows, mesh_cols),
        dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER),
    )
    try:
        state = load_glm_lazy_state_dict(Path(snap), num_layers=int(hparams.num_hidden_layers))
        w = convert_decoder_layer_weights(
            device=mesh_device,
            state=state,
            layer_idx=layer_idx,
            hparams=hparams,
            cache_dir=Path(os.path.expanduser("~/.cache/ttnn/models/glm4_moe_lite/moe_layer1_tt_mesh_cache")),
            enable_moe=True,
        )
        assert w.moe is not None, "Expected routed MoE weights for layer 1"

        rt = create_moe_runtime(device=mesh_device, hparams=hparams)

        x_tt = ttnn.from_torch(
            x.to(torch.bfloat16).view(1, 1, tokens, hidden),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        topk_idx = ref.topk_indices.to(dtype=torch.int16).view(1, 1, tokens, -1)
        topk_w = ref.topk_weights.to(dtype=torch.bfloat16).view(1, 1, tokens, -1)

        topk_idx_tt_rm = ttnn.from_torch(
            topk_idx,
            device=mesh_device,
            dtype=ttnn.uint16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        topk_w_tt_rm = ttnn.from_torch(
            topk_w,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        topk_idx_tt = ttnn.to_layout(topk_idx_tt_rm, ttnn.TILE_LAYOUT)
        topk_w_tt = ttnn.to_layout(topk_w_tt_rm, ttnn.TILE_LAYOUT)
        ttnn.deallocate(topk_idx_tt_rm)
        ttnn.deallocate(topk_w_tt_rm)

        shared = torch.randn((1, 1, tokens, hidden), dtype=torch.bfloat16)
        residual = torch.randn((1, 1, tokens, hidden), dtype=torch.bfloat16)
        shared_tt = ttnn.from_torch(
            shared,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        residual_tt = ttnn.from_torch(
            residual,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        routed_tt = moe_sparse_experts_forward_tt(
            device=mesh_device,
            hidden_states=x_tt,
            topk_expert_indices=topk_idx_tt,
            topk_expert_weights=topk_w_tt,
            moe_w=w.moe,
            rt=rt,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            collective_epilogue_a=shared_tt if fused_epilogue else None,
            collective_epilogue_b=residual_tt if fused_epilogue else None,
            use_buffered_all_reduce=buffered,
        )

        routed_dev0 = ttnn.get_device_tensors(routed_tt)[0]
        routed = ttnn.to_torch(routed_dev0).reshape(tokens, hidden).to(dtype=torch.float32).cpu()
        expected = ref.routed_out
        if fused_epilogue:
            expected = expected + shared.reshape(tokens, hidden).float() + residual.reshape(tokens, hidden).float()
    finally:
        try:
            ttnn.deallocate(routed_tt)
        except Exception:
            pass
        try:
            ttnn.deallocate(topk_idx_tt)
            ttnn.deallocate(topk_w_tt)
            ttnn.deallocate(x_tt)
            ttnn.deallocate(shared_tt)
            ttnn.deallocate(residual_tt)
        except Exception:
            pass
        _clear_buffered_moe_all_reduce_cache(mesh_device)
        ttnn.close_mesh_device(mesh_device)

    ok, msg = comp_pcc(routed, expected, pcc=0.98)
    assert ok, f"layer1 routed experts (mesh) output mismatch vs reference: {msg}"
