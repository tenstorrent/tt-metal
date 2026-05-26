# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Isolated 1L Qwen3-32B full-attention decode tracy driver.

Mirrors ``models/demos/qwen3_6_galaxy_v2/demo/tracy_perf_1L_fullattn.py`` in
structure (1L model, prefill T=128 warm-up, 2 warm decode steps inside tracy
signposts) so the per-op CSVs are directly comparable.

Run:

    export HF_MODEL=Qwen/Qwen3-32B \\
        && export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) \\
        && source python_env/bin/activate \\
        && python -m tracy -p -v -r -m pytest --noconftest \\
            models/demos/llama3_70b_galaxy/tests/tracy_perf_1L_qwen32b_fullattn.py \\
            -v -s
"""
from __future__ import annotations

import os
import time

import pytest
import torch

import ttnn

_B = 1
_T_PREFILL = 128
_N_LAYERS = 1
_N_DECODE_STEPS = 2  # 2 warm decode steps inside signposts (matches qwen3.6 1L tracy)


@pytest.fixture(scope="module")
def bh_glx_mesh():
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
    )
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(8, 4),
        trace_region_size=184915840,
        worker_l1_size=1345000,
    )
    yield mesh
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _build_tt_model(mesh, n_layers):
    from models.demos.llama3_70b_galaxy.tt.llama_model import TtTransformer
    from models.demos.llama3_70b_galaxy.tt.qwen_model_config import TtQwenModelArgs
    from models.demos.llama3_70b_galaxy.tt.llama_common import PagedAttentionConfig

    # batch_size=1 to match qwen3.6 v2 batch=1 single-user comparison.
    args = TtQwenModelArgs(mesh, dummy_weights=False, max_batch_size=1, max_seq_len=1024)
    args.n_layers = n_layers

    state_dict = args.load_state_dict()

    paged_attention_config = PagedAttentionConfig(block_size=32, max_num_blocks=64)

    weight_cache_path = args.weight_cache_path(ttnn.bfloat8_b)
    weight_cache_path.mkdir(parents=True, exist_ok=True)

    model = TtTransformer(
        args=args,
        dtype=ttnn.bfloat8_b,
        mesh_device=mesh,
        state_dict=state_dict,
        weight_cache_path=weight_cache_path,
        paged_attention_config=paged_attention_config,
        use_paged_kv_cache=False,
    )
    return model, args, paged_attention_config


def _build_page_table_tt(mesh, args, paged_attention_config):
    permutation = torch.randperm(paged_attention_config.max_num_blocks)
    reverse_permutation = torch.argsort(permutation)
    page_table = reverse_permutation.reshape(
        args.batch_size_per_device_group,
        paged_attention_config.max_num_blocks // args.batch_size_per_device_group,
    )
    return ttnn.from_torch(
        page_table,
        device=mesh,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, None), mesh_shape=args.cluster_shape),
    )


def _build_rope_cos_sin_decode(mesh, args, cur_pos: int):
    """Use the rope_setup that's already attached to the model.

    For Qwen3-32B full RoPE, the rope setup builds cos/sin from cur_pos directly.
    """
    return model_args_rope_setup.get_rm_rot_mats(torch.tensor([cur_pos], dtype=torch.long))


def _embed_tokens_replicated(state_dict, mesh, token_ids: torch.Tensor):
    """Embed tokens via the HF embedding matrix on host, then replicate to mesh.

    Qwen3-32B's TtTransformer.forward expects already-embedded hidden states
    for prefill, or token ids for decode (depending on embedding setup).
    """
    emb_w_key = next(k for k in state_dict if k.endswith("embed_tokens.weight") or k.endswith("tok_embeddings.weight"))
    emb_w = state_dict[emb_w_key]
    return emb_w[token_ids].to(torch.bfloat16)


def _do_prefill(model, mesh, args, x_prefill_cpu: torch.Tensor, page_table_tt):
    """Send a [B, T, H] prefill input through the 70B-galaxy TtTransformer.

    Mirrors the prefill setup in qwen3.6 v2's intrace test.
    """
    B, T, H = x_prefill_cpu.shape
    x_4d = x_prefill_cpu.reshape(1, 1, T, H)
    # Replicate to mesh — TtTransformer.prefill_forward will redistribute internally.
    x_tt = ttnn.from_torch(
        x_4d,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    rope_setup = model.rope_setup
    rot_mats = rope_setup.get_both_trans_mats()

    return model.forward(
        x_tt,
        current_pos=None,
        rot_mats=rot_mats,
        user_id=0,
        mode="prefill",
        page_table=page_table_tt,
        chunk_page_table=None,
        chunk_start_idx=ttnn.from_torch(
            torch.tensor([0], dtype=torch.int32),
            device=mesh,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        ),
        start_pos=0,
        get_last_token=-1,
        kv_cache=None,
        batch_size=1,
    )


def _do_decode(model, mesh, args, x_decode_cpu: torch.Tensor, cur_pos: int, page_table_tt):
    B, T, H = x_decode_cpu.shape
    x_4d = x_decode_cpu.reshape(1, 1, T, H)
    x_tt = ttnn.from_torch(
        x_4d,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    rope_setup = model.rope_setup
    cos, sin = rope_setup.get_rm_rot_mats(torch.tensor([cur_pos] * args.max_batch_size, dtype=torch.long))
    cur_pos_tt = ttnn.from_torch(
        torch.tensor([cur_pos] * args.max_batch_size, dtype=torch.int32),
        device=mesh,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    out = model.forward(
        x_tt,
        current_pos=cur_pos_tt,
        rot_mats=(cos, sin),
        user_id=0,
        mode="decode",
        page_table=page_table_tt,
        chunk_page_table=None,
        chunk_start_idx=None,
        start_pos=0,
        get_last_token=-1,
        kv_cache=None,
        batch_size=1,
    )
    if isinstance(out, list):
        for o in out:
            try:
                _ = ttnn.to_torch(
                    o, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, dims=(1, 3), mesh_shape=args.cluster_shape)
                )
                break
            except Exception:
                continue
    else:
        _ = ttnn.to_torch(
            out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, dims=(1, 3), mesh_shape=args.cluster_shape)
        )


@pytest.mark.hardware
def test_qwen32b_1L_fullattn_decode_perf(bh_glx_mesh):
    """1L Qwen3-32B full-attention prefill T=128 + 2 decode steps with tracy signposts."""
    try:
        from tracy import signpost
    except ImportError:
        signpost = lambda *_args, **_kwargs: None  # noqa: E731

    print(f"[qwen32b-1L-fa] env HF_MODEL = {os.environ.get('HF_MODEL', '(unset)')}")

    print("[qwen32b-1L-fa] building 1L TtTransformer ...")
    model, args, paged_attention_config = _build_tt_model(bh_glx_mesh, _N_LAYERS)
    print(
        f"[qwen32b-1L-fa] TT 1L model built; n_heads={args.n_heads}, n_kv_heads={args.n_kv_heads}, "
        f"hidden={args.dim}, intermediate={args.hidden_dim}, vocab={args.vocab_size}"
    )

    page_table_tt = _build_page_table_tt(bh_glx_mesh, args, paged_attention_config)

    torch.manual_seed(44)
    x_prefill_cpu = torch.randn(_B, _T_PREFILL, args.dim, dtype=torch.bfloat16)
    x_decode_cpu = torch.randn(_B, 1, args.dim, dtype=torch.bfloat16)

    # --- Warmup prefill (compile pass, NOT signposted) ---
    print("[qwen32b-1L-fa] WARMUP prefill (compile pass) ...")
    t0 = time.perf_counter()
    pf_out_w = _do_prefill(model, bh_glx_mesh, args, x_prefill_cpu, page_table_tt)
    if pf_out_w is not None and not isinstance(pf_out_w, list):
        try:
            pf_out_w.deallocate(True)
        except Exception:
            pass
    ttnn.synchronize_device(bh_glx_mesh)
    print(f"[qwen32b-1L-fa] warmup prefill done in {(time.perf_counter() - t0) * 1000.0:.2f} ms")

    print("[qwen32b-1L-fa] flushing device profiler buffers after warmup ...")
    ttnn.ReadDeviceProfiler(bh_glx_mesh)

    # --- Profiled region ---
    signpost("start")

    print("[qwen32b-1L-fa] PROFILED prefill (T=128, warm) ...")
    t0 = time.perf_counter()
    pf_out = _do_prefill(model, bh_glx_mesh, args, x_prefill_cpu, page_table_tt)
    if pf_out is not None and not isinstance(pf_out, list):
        try:
            pf_out.deallocate(True)
        except Exception:
            pass
    ttnn.synchronize_device(bh_glx_mesh)
    prefill_ms = (time.perf_counter() - t0) * 1000.0
    signpost("prefill_done")
    print(f"[qwen32b-1L-fa] PROFILED prefill (warm): {prefill_ms:.2f} ms")

    n_total = _N_DECODE_STEPS + 1
    print(f"[qwen32b-1L-fa] PROFILED decode x {n_total} (#0 = compile, #1..{_N_DECODE_STEPS} = warm) ...")
    decode_times_ms = []
    for step in range(n_total):
        cur_pos = _T_PREFILL + step
        t0 = time.perf_counter()
        _do_decode(model, bh_glx_mesh, args, x_decode_cpu, cur_pos, page_table_tt)
        ttnn.synchronize_device(bh_glx_mesh)
        dt = (time.perf_counter() - t0) * 1000.0
        decode_times_ms.append(dt)
        label = "COMPILE" if step == 0 else "TIMED"
        print(f"[qwen32b-1L-fa]   decode step {step} ({label}, cur_pos={cur_pos}): {dt:.2f} ms")

    signpost("stop")

    print("\n[qwen32b-1L-fa] === summary ===")
    print(f"[qwen32b-1L-fa]   prefill (T={_T_PREFILL}, warm) : {prefill_ms:.2f} ms")
    print(f"[qwen32b-1L-fa]   decode #0 (compile)          : {decode_times_ms[0]:.2f} ms")
    warm = decode_times_ms[1:]
    if warm:
        print(f"[qwen32b-1L-fa]   decode #1..N (warm) mean     : {sum(warm)/len(warm):.2f} ms")
    print(f"[qwen32b-1L-fa]   decode raw                    : {decode_times_ms}")
