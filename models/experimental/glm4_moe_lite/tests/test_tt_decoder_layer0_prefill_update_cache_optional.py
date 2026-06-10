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
from models.experimental.glm4_moe_lite.tt.decoder_layer_tt import run_decoder_layer_prefill_update_cache_tt
from models.experimental.glm4_moe_lite.tt.layer0_tt import (
    _alloc_contiguous_page_table,
    _alloc_paged_kvpe_cache,
    _round_up,
    convert_layer0_weights,
    make_rope_tensors,
    mesh_shard0_to_torch,
    run_layer0_prefill_tt,
)
from models.experimental.glm4_moe_lite.tt.layer_weights import convert_decoder_layer_weights
from models.experimental.glm4_moe_lite.tt.moe_tt import create_moe_runtime
from models.experimental.glm4_moe_lite.tt.tt_embedding import (
    convert_embedding_weight_to_tt,
    prefill_embed_memory_config,
    run_tt_embedding,
)
from models.experimental.glm4_moe_lite.tt.weights import (
    find_missing_shards,
    load_glm_lazy_state_dict,
    resolve_best_effort_snapshot_dir,
)

_PCC_TARGET = 0.999


def _test_seq_len() -> int:
    raw = os.environ.get("GLM4_MOE_LITE_TEST_PREFILL_SEQ_LEN", "").strip()
    if not raw:
        return 5
    seq_len = int(raw)
    if seq_len <= 0:
        raise ValueError(f"GLM4_MOE_LITE_TEST_PREFILL_SEQ_LEN must be > 0, got {seq_len}")
    return seq_len


def _input_ids_for_test() -> torch.Tensor:
    seq_len = _test_seq_len()
    return torch.arange(1, seq_len + 1, dtype=torch.int32).unsqueeze(0)


def _mesh_shape_from_env(default: tuple[int, int]) -> tuple[int, int]:
    raw = os.environ.get("TT_TEST_MESH_SHAPE", "").strip().lower()
    if not raw:
        return default
    raw = raw.replace("x", ",")
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"Invalid TT_TEST_MESH_SHAPE={raw!r}; expected 'rowsxcols' (e.g. '1x4')")
    return (int(parts[0]), int(parts[1]))


def _set_fabric_config_for_mesh(num_devices: int) -> None:
    """Initialize fabric before open_mesh_device; required for multi-device CCL."""
    if int(num_devices) <= 1:
        return
    is_galaxy = ttnn.cluster.get_cluster_type() == ttnn.cluster.ClusterType.GALAXY
    fabric = ttnn.FabricConfig.FABRIC_1D_RING if is_galaxy else ttnn.FabricConfig.FABRIC_1D
    ttnn.set_fabric_config(
        fabric,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    )


def _dispatch_core_config() -> ttnn.DispatchCoreConfig:
    if ttnn.cluster.get_cluster_type() == ttnn.cluster.ClusterType.GALAXY:
        return ttnn.DispatchCoreConfig(axis=ttnn.DispatchCoreAxis.ROW)
    return ttnn.DispatchCoreConfig(ttnn.device.get_default_dispatch_core_type())


def _mesh_tensor_to_torch(
    tt_tensor: ttnn.Tensor,
    *,
    seq_len: int,
    hidden: int,
    check_replicated_shards: bool = True,
) -> torch.Tensor:
    """Materialize [1,S,H] from a mesh or single-device tensor (uses device 0 on mesh)."""
    device_tensors = ttnn.get_device_tensors(tt_tensor)
    out = mesh_shard0_to_torch(tt_tensor).reshape(1, seq_len, hidden).cpu()
    if check_replicated_shards:
        for shard in device_tensors[1:]:
            other = ttnn.to_torch(shard.cpu()).reshape(1, seq_len, hidden).cpu()
            if not torch.allclose(out, other, rtol=0.0, atol=0.0):
                max_diff = (out - other).abs().max().item()
                raise AssertionError(
                    f"mesh device outputs diverged (max_abs_diff={max_diff}); "
                    "expected replicated activations across devices"
                )
    return out


def _run_generic_layer0_prefill_once(
    *,
    mesh_device,
    snap: Path,
    hparams: Glm4MoeLiteHParams,
    input_ids: torch.Tensor,
    cache_dir: Path,
    use_decoder_layer_weights: bool,
    check_replicated_shards: bool = True,
) -> torch.Tensor:
    """Run generic paged prefill for layer 0 and return [1,S,H] on CPU."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    seq_len = int(input_ids.shape[1])
    padded_len = _round_up(seq_len, 128)
    hidden = int(hparams.hidden_size)

    block_size = 64
    blocks_per_seq = max(1, _round_up(seq_len, block_size) // block_size)

    kvpe_cache = _alloc_paged_kvpe_cache(
        device=mesh_device,
        max_num_blocks=int(1 * blocks_per_seq),
        block_size=block_size,
        kvpe_dim=int(hparams.kv_lora_rank + hparams.qk_rope_head_dim),
        dtype=ttnn.bfloat8_b,
    )
    page_table = _alloc_contiguous_page_table(batch=1, blocks_per_seq=blocks_per_seq)
    is_mesh = mesh_device.__class__.__name__ == "MeshDevice"
    page_table_tt = ttnn.from_torch(
        page_table,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )

    state = load_glm_lazy_state_dict(snap, num_layers=int(hparams.num_hidden_layers))
    if use_decoder_layer_weights:
        w = convert_decoder_layer_weights(
            device=mesh_device,
            state=state,
            layer_idx=0,
            hparams=hparams,
            cache_dir=cache_dir,
        )
        embed_w = convert_embedding_weight_to_tt(
            device=mesh_device,
            embed_weight=state["model.embed_tokens.weight"],
            cache_file_name=cache_dir / "embed_w",
        )
    else:
        w = convert_layer0_weights(device=mesh_device, state=state, cache_dir=cache_dir)
        embed_w = w.embed_w

    rope = make_rope_tensors(
        device=mesh_device,
        seq_len=padded_len,
        rope_dim=int(hparams.qk_rope_head_dim),
        rope_theta=float(hparams.rope_theta),
    )

    input_padded = torch.zeros((1, padded_len), dtype=input_ids.dtype)
    input_padded[:, :seq_len] = input_ids
    embed_mc = prefill_embed_memory_config(seq_tokens=padded_len, hidden_dim=hidden)
    x_embed = run_tt_embedding(
        device=mesh_device,
        token_ids=input_padded.to(torch.int32),
        tt_weight=embed_w,
        memory_config=embed_mc,
    )
    if x_embed.layout != ttnn.TILE_LAYOUT:
        x_embed = ttnn.to_layout(x_embed, ttnn.TILE_LAYOUT)
    x_embed = ttnn.reshape(x_embed, (1, 1, padded_len, hidden))

    x_out = run_decoder_layer_prefill_update_cache_tt(
        device=mesh_device,
        x_embed=x_embed,
        page_table_tt=page_table_tt,
        kvpe_cache=kvpe_cache,
        cos_matrix=rope["cos_matrix"],
        sin_matrix=rope["sin_matrix"],
        trans_matrix=rope["trans_matrix"],
        w=w,
        hparams=hparams,
        prompt_len=seq_len,
    )

    x0 = ttnn.slice(x_out, [0, 0, 0, 0], [1, 1, seq_len, hidden])
    out = _mesh_tensor_to_torch(
        x0,
        seq_len=seq_len,
        hidden=hidden,
        check_replicated_shards=check_replicated_shards,
    )

    ttnn.deallocate(x0)
    ttnn.deallocate(x_out)
    ttnn.deallocate(x_embed)
    ttnn.deallocate(page_table_tt)
    ttnn.deallocate(kvpe_cache)
    ttnn.deallocate(rope["cos_matrix"])
    ttnn.deallocate(rope["sin_matrix"])
    ttnn.deallocate(rope["trans_matrix"])

    return out


def _run_generic_layers_prefill_once(
    *,
    mesh_device,
    snap: Path,
    hparams: Glm4MoeLiteHParams,
    input_ids: torch.Tensor,
    cache_dir: Path,
    num_layers: int,
    enable_moe: bool,
    check_replicated_shards: bool = True,
) -> torch.Tensor:
    """Run generic paged prefill through layers [0, num_layers) and return [1,S,H] on CPU."""
    if int(num_layers) < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")

    cache_dir.mkdir(parents=True, exist_ok=True)

    seq_len = int(input_ids.shape[1])
    padded_len = _round_up(seq_len, 128)
    hidden = int(hparams.hidden_size)

    block_size = 64
    blocks_per_seq = max(1, _round_up(seq_len, block_size) // block_size)

    kvpe_caches = [
        _alloc_paged_kvpe_cache(
            device=mesh_device,
            max_num_blocks=int(1 * blocks_per_seq),
            block_size=block_size,
            kvpe_dim=int(hparams.kv_lora_rank + hparams.qk_rope_head_dim),
            dtype=ttnn.bfloat8_b,
        )
        for _ in range(int(num_layers))
    ]
    page_table = _alloc_contiguous_page_table(batch=1, blocks_per_seq=blocks_per_seq)
    is_mesh = mesh_device.__class__.__name__ == "MeshDevice"
    page_table_tt = ttnn.from_torch(
        page_table,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )

    state = load_glm_lazy_state_dict(snap, num_layers=int(hparams.num_hidden_layers))
    embed_w = convert_embedding_weight_to_tt(
        device=mesh_device,
        embed_weight=state["model.embed_tokens.weight"],
        cache_file_name=cache_dir / "embed_w",
    )

    rope = make_rope_tensors(
        device=mesh_device,
        seq_len=padded_len,
        rope_dim=int(hparams.qk_rope_head_dim),
        rope_theta=float(hparams.rope_theta),
    )

    input_padded = torch.zeros((1, padded_len), dtype=input_ids.dtype)
    input_padded[:, :seq_len] = input_ids
    embed_mc = prefill_embed_memory_config(seq_tokens=padded_len, hidden_dim=hidden)
    x = run_tt_embedding(
        device=mesh_device,
        token_ids=input_padded.to(torch.int32),
        tt_weight=embed_w,
        memory_config=embed_mc,
    )
    if x.layout != ttnn.TILE_LAYOUT:
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
    x = ttnn.reshape(x, (1, 1, padded_len, hidden))

    moe_runtime = create_moe_runtime(device=mesh_device, hparams=hparams) if enable_moe else None

    for layer_idx in range(int(num_layers)):
        w = convert_decoder_layer_weights(
            device=mesh_device,
            state=state,
            layer_idx=layer_idx,
            hparams=hparams,
            cache_dir=cache_dir,
            enable_moe=enable_moe,
        )
        layer_moe_runtime = moe_runtime if getattr(w, "moe", None) is not None else None
        x_next = run_decoder_layer_prefill_update_cache_tt(
            device=mesh_device,
            x_embed=x,
            page_table_tt=page_table_tt,
            kvpe_cache=kvpe_caches[layer_idx],
            cos_matrix=rope["cos_matrix"],
            sin_matrix=rope["sin_matrix"],
            trans_matrix=rope["trans_matrix"],
            w=w,
            hparams=hparams,
            prompt_len=seq_len,
            moe_runtime=layer_moe_runtime,
        )
        if layer_idx > 0:
            ttnn.deallocate(x, force=False)
        x = x_next

    x0 = ttnn.slice(x, [0, 0, 0, 0], [1, 1, seq_len, hidden])
    out = _mesh_tensor_to_torch(
        x0,
        seq_len=seq_len,
        hidden=hidden,
        check_replicated_shards=check_replicated_shards,
    )

    ttnn.deallocate(x0)
    ttnn.deallocate(x)
    ttnn.deallocate(page_table_tt)
    for kv in kvpe_caches:
        ttnn.deallocate(kv)
    ttnn.deallocate(rope["cos_matrix"])
    ttnn.deallocate(rope["sin_matrix"])
    ttnn.deallocate(rope["trans_matrix"])

    return out


def _layer0_prefill_pcc_check(
    *,
    mesh_rows: int,
    mesh_cols: int,
    physical_device_ids: list[int] | None,
) -> None:
    """Compare generic paged prefill vs layer0 harness on the given mesh shape."""
    snap = resolve_best_effort_snapshot_dir("zai-org/GLM-4.7-Flash")
    missing = find_missing_shards(snap)
    if missing:
        pytest.skip(f"GLM snapshot missing {len(missing)} shards; run ensure_glm47_weights.sh first.")

    input_ids = _input_ids_for_test()

    cfg = json.loads((Path(snap) / "config.json").read_text())
    hparams = Glm4MoeLiteHParams.from_hf_config(SimpleNamespace(**cfg))
    hparams.validate()

    cache_dir = Path(os.path.expanduser("~/.cache/ttnn/models/glm4_moe_lite/layer0_tt_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    num_devices = int(mesh_rows) * int(mesh_cols)
    _set_fabric_config_for_mesh(num_devices)

    open_kwargs: dict = {
        "mesh_shape": ttnn.MeshShape(mesh_rows, mesh_cols),
        "dispatch_core_config": _dispatch_core_config(),
    }
    if physical_device_ids is not None:
        open_kwargs["physical_device_ids"] = physical_device_ids

    mesh_device = ttnn.open_mesh_device(**open_kwargs)
    try:
        ref = run_layer0_prefill_tt(
            device=mesh_device,
            snapshot_dir=Path(snap),
            input_ids=input_ids,
            cache_dir=cache_dir,
            seq_pad_multiple=128,
        )

        seq_len = int(input_ids.shape[1])
        out = _run_generic_layer0_prefill_once(
            mesh_device=mesh_device,
            snap=Path(snap),
            hparams=hparams,
            input_ids=input_ids,
            cache_dir=cache_dir,
            use_decoder_layer_weights=False,
        )

    finally:
        ttnn.close_mesh_device(mesh_device)

    ok, msg = comp_pcc(out, ref.x_mlp_out, pcc=_PCC_TARGET)
    assert ok, (
        f"generic layer0 prefill mismatch vs specialized harness on {mesh_rows}x{mesh_cols} mesh "
        f"(seq_len={seq_len}): {msg}"
    )


@pytest.mark.skipif(
    os.environ.get("TT_ENABLE_HW_TESTS") != "1",
    reason="Enable with TT_ENABLE_HW_TESTS=1 (requires Tenstorrent device access).",
)
@pytest.mark.skipif(
    os.environ.get("TT_ENABLE_LARGE_MODEL_TESTS") != "1",
    reason="Enable with TT_ENABLE_LARGE_MODEL_TESTS=1 (loads large embedding weights).",
)
def test_generic_layer0_prefill_update_cache_matches_existing_harness() -> None:
    """Sanity-check generic decoder-layer prefill on a 1x1 mesh."""
    _layer0_prefill_pcc_check(mesh_rows=1, mesh_cols=1, physical_device_ids=[0])


@pytest.mark.skipif(
    os.environ.get("TT_ENABLE_HW_TESTS") != "1",
    reason="Enable with TT_ENABLE_HW_TESTS=1 (requires Tenstorrent device access).",
)
@pytest.mark.skipif(
    os.environ.get("TT_ENABLE_LARGE_MODEL_TESTS") != "1",
    reason="Enable with TT_ENABLE_LARGE_MODEL_TESTS=1 (loads large embedding weights).",
)
@pytest.mark.skipif(
    os.environ.get("TT_ENABLE_MULTI_DEVICE_TESTS") != "1",
    reason="Enable with TT_ENABLE_MULTI_DEVICE_TESTS=1 (opens a multi-device mesh).",
)
def test_generic_layer0_prefill_update_cache_matches_existing_harness_mesh_1x4() -> None:
    """PCC-check generic paged prefill vs layer0 harness on a 1x4 mesh (no TP by default)."""
    mesh_rows, mesh_cols = _mesh_shape_from_env(default=(1, 4))
    if mesh_rows * mesh_cols <= 1:
        pytest.skip(f"TT_TEST_MESH_SHAPE={mesh_rows}x{mesh_cols} is single-device; use the 1x1 test instead.")
    _layer0_prefill_pcc_check(mesh_rows=mesh_rows, mesh_cols=mesh_cols, physical_device_ids=None)


def _layer0_prefill_tp_pcc_check(
    *,
    mesh_rows: int,
    mesh_cols: int,
    physical_device_ids: list[int] | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compare TP=1 generic paged prefill vs no-TP reference on the same mesh."""
    snap = resolve_best_effort_snapshot_dir("zai-org/GLM-4.7-Flash")
    missing = find_missing_shards(snap)
    if missing:
        pytest.skip(f"GLM snapshot missing {len(missing)} shards; run ensure_glm47_weights.sh first.")

    input_ids = _input_ids_for_test()

    cfg = json.loads((Path(snap) / "config.json").read_text())
    hparams = Glm4MoeLiteHParams.from_hf_config(SimpleNamespace(**cfg))
    hparams.validate()

    cache_base = Path(os.path.expanduser("~/.cache/ttnn/models/glm4_moe_lite/layer0_tp_pcc_cache"))

    num_devices = int(mesh_rows) * int(mesh_cols)
    if num_devices <= 1:
        pytest.skip("TP PCC check requires a multi-device mesh.")

    monkeypatch.setenv("GLM4_MOE_LITE_CCL_NUM_LINKS", "2")
    monkeypatch.setenv("GLM4_MOE_LITE_CCL_TOPOLOGY", "linear")
    monkeypatch.setenv("GLM4_MOE_LITE_FUSE_MLP_MOE_REDUCE", "1")

    _set_fabric_config_for_mesh(num_devices)

    open_kwargs: dict = {
        "mesh_shape": ttnn.MeshShape(mesh_rows, mesh_cols),
        "dispatch_core_config": _dispatch_core_config(),
    }
    if physical_device_ids is not None:
        open_kwargs["physical_device_ids"] = physical_device_ids

    mesh_device = ttnn.open_mesh_device(**open_kwargs)
    try:
        monkeypatch.delenv("GLM4_MOE_LITE_TP", raising=False)
        ref_out = _run_generic_layer0_prefill_once(
            mesh_device=mesh_device,
            snap=Path(snap),
            hparams=hparams,
            input_ids=input_ids,
            cache_dir=cache_base / "no_tp",
            use_decoder_layer_weights=True,
        )

        monkeypatch.setenv("GLM4_MOE_LITE_TP", "1")
        tp_out = _run_generic_layer0_prefill_once(
            mesh_device=mesh_device,
            snap=Path(snap),
            hparams=hparams,
            input_ids=input_ids,
            cache_dir=cache_base / "tp1",
            use_decoder_layer_weights=True,
        )
    finally:
        ttnn.close_mesh_device(mesh_device)

    seq_len = int(input_ids.shape[1])
    ok, msg = comp_pcc(tp_out, ref_out, pcc=_PCC_TARGET)
    assert ok, (
        f"TP=1 layer0 prefill mismatch vs no-TP reference on {mesh_rows}x{mesh_cols} mesh "
        f"(seq_len={seq_len}): {msg}"
    )


@pytest.mark.skipif(
    os.environ.get("TT_ENABLE_HW_TESTS") != "1",
    reason="Enable with TT_ENABLE_HW_TESTS=1 (requires Tenstorrent device access).",
)
@pytest.mark.skipif(
    os.environ.get("TT_ENABLE_LARGE_MODEL_TESTS") != "1",
    reason="Enable with TT_ENABLE_LARGE_MODEL_TESTS=1 (loads large embedding weights).",
)
@pytest.mark.skipif(
    os.environ.get("TT_ENABLE_MULTI_DEVICE_TESTS") != "1",
    reason="Enable with TT_ENABLE_MULTI_DEVICE_TESTS=1 (opens a multi-device mesh).",
)
def test_generic_layer0_prefill_update_cache_tp_matches_no_tp_mesh_1x4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PCC-check TP=1 paged prefill vs replicated no-TP on a 1x4 mesh."""
    mesh_rows, mesh_cols = _mesh_shape_from_env(default=(1, 4))
    if mesh_rows * mesh_cols <= 1:
        pytest.skip(f"TT_TEST_MESH_SHAPE={mesh_rows}x{mesh_cols} is single-device; TP requires multi-device.")
    _layer0_prefill_tp_pcc_check(
        mesh_rows=mesh_rows,
        mesh_cols=mesh_cols,
        physical_device_ids=None,
        monkeypatch=monkeypatch,
    )


def _layers_moe_prefill_tp_pcc_check(
    *,
    mesh_rows: int,
    mesh_cols: int,
    physical_device_ids: list[int] | None,
    monkeypatch: pytest.MonkeyPatch,
    num_layers: int,
) -> None:
    """Compare TP=1 vs no-TP after prefill through layers [0, num_layers)."""
    if int(num_layers) < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")

    snap = resolve_best_effort_snapshot_dir("zai-org/GLM-4.7-Flash")
    missing = find_missing_shards(snap)
    if missing:
        pytest.skip(f"GLM snapshot missing {len(missing)} shards; run ensure_glm47_weights.sh first.")

    input_ids = _input_ids_for_test()

    cfg = json.loads((Path(snap) / "config.json").read_text())
    hparams = Glm4MoeLiteHParams.from_hf_config(SimpleNamespace(**cfg))
    hparams.validate()

    cache_base = Path(os.path.expanduser("~/.cache/ttnn/models/glm4_moe_lite/layers_moe_tp_pcc_cache"))
    last_layer_idx = int(num_layers) - 1

    num_devices = int(mesh_rows) * int(mesh_cols)
    if num_devices <= 1:
        pytest.skip("TP PCC check requires a multi-device mesh.")

    monkeypatch.setenv("GLM4_MOE_LITE_CCL_NUM_LINKS", "2")
    monkeypatch.setenv("GLM4_MOE_LITE_CCL_TOPOLOGY", "linear")
    monkeypatch.setenv("GLM4_MOE_LITE_FUSE_MLP_MOE_REDUCE", "1")

    _set_fabric_config_for_mesh(num_devices)

    open_kwargs: dict = {
        "mesh_shape": ttnn.MeshShape(mesh_rows, mesh_cols),
        "dispatch_core_config": _dispatch_core_config(),
    }
    if physical_device_ids is not None:
        open_kwargs["physical_device_ids"] = physical_device_ids

    mesh_device = ttnn.open_mesh_device(**open_kwargs)
    try:
        monkeypatch.delenv("GLM4_MOE_LITE_TP", raising=False)
        ref_out = _run_generic_layers_prefill_once(
            mesh_device=mesh_device,
            snap=Path(snap),
            hparams=hparams,
            input_ids=input_ids,
            cache_dir=cache_base / "no_tp",
            num_layers=num_layers,
            enable_moe=True,
        )

        monkeypatch.setenv("GLM4_MOE_LITE_TP", "1")
        tp_out = _run_generic_layers_prefill_once(
            mesh_device=mesh_device,
            snap=Path(snap),
            hparams=hparams,
            input_ids=input_ids,
            cache_dir=cache_base / "tp1",
            num_layers=num_layers,
            enable_moe=True,
        )
    finally:
        ttnn.close_mesh_device(mesh_device)

    seq_len = int(input_ids.shape[1])
    ok, msg = comp_pcc(tp_out, ref_out, pcc=_PCC_TARGET)
    assert ok, (
        f"TP=1 prefill mismatch vs no-TP reference on {mesh_rows}x{mesh_cols} mesh "
        f"(num_layers={num_layers}, last_layer_idx={last_layer_idx}, seq_len={seq_len}): {msg}"
    )


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
def test_layers_through_1_moe_prefill_tp_matches_no_tp_mesh_1x4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PCC-check TP=1 vs no-TP after prefill through layer 1 (first MoE) on a 1x4 mesh.

    First known TP divergence in hidden states (~PCC 0.95); demo argmax at NUM_LAYERS=2
    can still match no-TP while hidden states differ.
    """
    mesh_rows, mesh_cols = _mesh_shape_from_env(default=(1, 4))
    if mesh_rows * mesh_cols <= 1:
        pytest.skip(f"TT_TEST_MESH_SHAPE={mesh_rows}x{mesh_cols} is single-device; TP requires multi-device.")
    _layers_moe_prefill_tp_pcc_check(
        mesh_rows=mesh_rows,
        mesh_cols=mesh_cols,
        physical_device_ids=None,
        monkeypatch=monkeypatch,
        num_layers=2,
    )


def _layers_moe_prefill_tp_only_run(
    *,
    mesh_rows: int,
    mesh_cols: int,
    physical_device_ids: list[int] | None,
    monkeypatch: pytest.MonkeyPatch,
    num_layers: int,
) -> torch.Tensor:
    """Run TP=1 prefill through layers [0, num_layers) once (for Tracy profiling; no no-TP pass)."""
    if int(num_layers) < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")

    snap = resolve_best_effort_snapshot_dir("zai-org/GLM-4.7-Flash")
    missing = find_missing_shards(snap)
    if missing:
        pytest.skip(f"GLM snapshot missing {len(missing)} shards; run ensure_glm47_weights.sh first.")

    input_ids = _input_ids_for_test()

    cfg = json.loads((Path(snap) / "config.json").read_text())
    hparams = Glm4MoeLiteHParams.from_hf_config(SimpleNamespace(**cfg))
    hparams.validate()

    cache_dir = Path(os.path.expanduser("~/.cache/ttnn/models/glm4_moe_lite/layers_moe_tp_only_profile_cache"))

    num_devices = int(mesh_rows) * int(mesh_cols)
    if num_devices <= 1:
        pytest.skip("TP profiling requires a multi-device mesh.")

    monkeypatch.setenv("GLM4_MOE_LITE_CCL_NUM_LINKS", "2")
    monkeypatch.setenv("GLM4_MOE_LITE_CCL_TOPOLOGY", "linear")
    monkeypatch.setenv("GLM4_MOE_LITE_FUSE_MLP_MOE_REDUCE", "1")
    monkeypatch.setenv("GLM4_MOE_LITE_TP", "1")

    _set_fabric_config_for_mesh(num_devices)

    open_kwargs: dict = {
        "mesh_shape": ttnn.MeshShape(mesh_rows, mesh_cols),
        "dispatch_core_config": _dispatch_core_config(),
    }
    if physical_device_ids is not None:
        open_kwargs["physical_device_ids"] = physical_device_ids

    mesh_device = ttnn.open_mesh_device(**open_kwargs)
    try:
        return _run_generic_layers_prefill_once(
            mesh_device=mesh_device,
            snap=Path(snap),
            hparams=hparams,
            input_ids=input_ids,
            cache_dir=cache_dir,
            num_layers=num_layers,
            enable_moe=True,
        )
    finally:
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
def test_layers_through_1_moe_prefill_tp_only_mesh_1x4(monkeypatch: pytest.MonkeyPatch) -> None:
    """TP=1 prefill through layer 1 (dense + first MoE); single pass for Tracy profiling."""
    mesh_rows, mesh_cols = _mesh_shape_from_env(default=(1, 4))
    if mesh_rows * mesh_cols <= 1:
        pytest.skip(f"TT_TEST_MESH_SHAPE={mesh_rows}x{mesh_cols} is single-device; TP requires multi-device.")
    out = _layers_moe_prefill_tp_only_run(
        mesh_rows=mesh_rows,
        mesh_cols=mesh_cols,
        physical_device_ids=None,
        monkeypatch=monkeypatch,
        num_layers=2,
    )
    assert torch.isfinite(out).all()


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
def test_layers_through_2_moe_prefill_tp_matches_no_tp_mesh_1x4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PCC-check TP=1 vs no-TP after prefill through layer 2 (MoE) on a 1x4 mesh.

    Amplifies layer-1 MoE TP error to catastrophic PCC (~0.26) and flips demo argmax.
    """
    mesh_rows, mesh_cols = _mesh_shape_from_env(default=(1, 4))
    if mesh_rows * mesh_cols <= 1:
        pytest.skip(f"TT_TEST_MESH_SHAPE={mesh_rows}x{mesh_cols} is single-device; TP requires multi-device.")
    _layers_moe_prefill_tp_pcc_check(
        mesh_rows=mesh_rows,
        mesh_cols=mesh_cols,
        physical_device_ids=None,
        monkeypatch=monkeypatch,
        num_layers=3,
    )


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
def test_incremental_per_layer_tp_pcc_mesh_1x4(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bisect which layer first diverges: compare TP vs no-TP after each layer depth."""
    snap = resolve_best_effort_snapshot_dir("zai-org/GLM-4.7-Flash")
    missing = find_missing_shards(snap)
    if missing:
        pytest.skip(f"GLM snapshot missing {len(missing)} shards; run ensure_glm47_weights.sh first.")

    cfg = json.loads((Path(snap) / "config.json").read_text())
    hparams = Glm4MoeLiteHParams.from_hf_config(SimpleNamespace(**cfg))
    hparams.validate()
    input_ids = _input_ids_for_test()
    cache_base = Path(os.path.expanduser("~/.cache/ttnn/models/glm4_moe_lite/incremental_tp_pcc_cache"))

    mesh_rows, mesh_cols = _mesh_shape_from_env(default=(1, 4))
    if mesh_rows * mesh_cols <= 1:
        pytest.skip("Incremental TP PCC requires a multi-device mesh.")

    monkeypatch.setenv("GLM4_MOE_LITE_CCL_NUM_LINKS", "2")
    monkeypatch.setenv("GLM4_MOE_LITE_CCL_TOPOLOGY", "linear")
    monkeypatch.setenv("GLM4_MOE_LITE_FUSE_MLP_MOE_REDUCE", "1")

    num_devices = int(mesh_rows) * int(mesh_cols)
    _set_fabric_config_for_mesh(num_devices)
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(mesh_rows, mesh_cols),
        dispatch_core_config=_dispatch_core_config(),
    )
    try:
        for num_layers in (1, 2, 3):
            monkeypatch.delenv("GLM4_MOE_LITE_TP", raising=False)
            ref_out = _run_generic_layers_prefill_once(
                mesh_device=mesh_device,
                snap=Path(snap),
                hparams=hparams,
                input_ids=input_ids,
                cache_dir=cache_base / "no_tp",
                num_layers=num_layers,
                enable_moe=True,
            )
            monkeypatch.setenv("GLM4_MOE_LITE_TP", "1")
            tp_out = _run_generic_layers_prefill_once(
                mesh_device=mesh_device,
                snap=Path(snap),
                hparams=hparams,
                input_ids=input_ids,
                cache_dir=cache_base / "tp1",
                num_layers=num_layers,
                enable_moe=True,
            )
            ok, msg = comp_pcc(tp_out, ref_out, pcc=_PCC_TARGET)
            assert ok, (
                f"TP=1 diverges from no-TP after layer {num_layers - 1} "
                f"(num_layers={num_layers}, seq_len={int(input_ids.shape[1])}): {msg}"
            )
    finally:
        ttnn.close_mesh_device(mesh_device)


def _layer0_prefill_sharded_q_a_pcc_check(monkeypatch: pytest.MonkeyPatch, *, fuse_qkv_a: bool) -> None:
    """Compare sharded q_a matmul→norm path vs interleaved gather baseline."""
    snap = resolve_best_effort_snapshot_dir("zai-org/GLM-4.7-Flash")
    missing = find_missing_shards(snap)
    if missing:
        pytest.skip(f"GLM snapshot missing {len(missing)} shards; run ensure_glm47_weights.sh first.")

    input_ids = _input_ids_for_test()
    seq_len = int(input_ids.shape[1])

    cfg = json.loads((Path(snap) / "config.json").read_text())
    hparams = Glm4MoeLiteHParams.from_hf_config(SimpleNamespace(**cfg))
    hparams.validate()

    cache_dir = Path(os.path.expanduser("~/.cache/ttnn/models/glm4_moe_lite/layer0_sharded_qkv_pcc_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    if fuse_qkv_a:
        monkeypatch.setenv("GLM4_MOE_LITE_FUSE_QKV_A", "1")
    else:
        monkeypatch.delenv("GLM4_MOE_LITE_FUSE_QKV_A", raising=False)

    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 1),
        physical_device_ids=[0],
        dispatch_core_config=_dispatch_core_config(),
    )
    try:
        monkeypatch.setenv("GLM4_MOE_LITE_PREFILL_SHARDED_QKV", "0")
        out_interleaved = _run_generic_layer0_prefill_once(
            mesh_device=mesh_device,
            snap=Path(snap),
            hparams=hparams,
            input_ids=input_ids,
            cache_dir=cache_dir / "interleaved",
            use_decoder_layer_weights=True,
        )

        monkeypatch.setenv("GLM4_MOE_LITE_PREFILL_SHARDED_QKV", "1")
        out_sharded = _run_generic_layer0_prefill_once(
            mesh_device=mesh_device,
            snap=Path(snap),
            hparams=hparams,
            input_ids=input_ids,
            cache_dir=cache_dir / "sharded",
            use_decoder_layer_weights=True,
        )
    finally:
        ttnn.close_mesh_device(mesh_device)

    ok, msg = comp_pcc(out_sharded, out_interleaved, pcc=_PCC_TARGET)
    label = "fused QKV" if fuse_qkv_a else "w_q_a"
    assert ok, f"sharded {label} prefill diverges from interleaved baseline (seq_len={seq_len}): {msg}"


@pytest.mark.skipif(
    os.environ.get("TT_ENABLE_HW_TESTS") != "1",
    reason="Enable with TT_ENABLE_HW_TESTS=1 (requires Tenstorrent device access).",
)
@pytest.mark.skipif(
    os.environ.get("TT_ENABLE_LARGE_MODEL_TESTS") != "1",
    reason="Enable with TT_ENABLE_LARGE_MODEL_TESTS=1 (loads large embedding weights).",
)
def test_layer0_prefill_sharded_qkv_matches_interleaved(monkeypatch: pytest.MonkeyPatch) -> None:
    """PCC-check sharded fused QKV→q_a norm vs interleaved gather on a 1x1 mesh."""
    _layer0_prefill_sharded_q_a_pcc_check(monkeypatch, fuse_qkv_a=True)


@pytest.mark.skipif(
    os.environ.get("TT_ENABLE_HW_TESTS") != "1",
    reason="Enable with TT_ENABLE_HW_TESTS=1 (requires Tenstorrent device access).",
)
@pytest.mark.skipif(
    os.environ.get("TT_ENABLE_LARGE_MODEL_TESTS") != "1",
    reason="Enable with TT_ENABLE_LARGE_MODEL_TESTS=1 (loads large embedding weights).",
)
def test_layer0_prefill_sharded_w_q_a_matches_interleaved(monkeypatch: pytest.MonkeyPatch) -> None:
    """PCC-check sharded w_q_a→q_a norm vs interleaved gather on a 1x1 mesh."""
    _layer0_prefill_sharded_q_a_pcc_check(monkeypatch, fuse_qkv_a=False)
