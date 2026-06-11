# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Decode L1-residual probe — can the qwen3.6 decode residual stream live in L1?

Background (the 60-vs-20 tok/s investigation): qwen3.6 decode keeps the
residual stream **col-sharded H/4 but in DRAM** (``skip_mem_cfg =
DRAM_MEMORY_CONFIG`` in ``llama_decoder.py``), whereas llama70b/Qwen3-32B keep
the equivalent residual **width-sharded in L1** end-to-end
(``DECODE_RESIDUAL_MEMCFG``) so the activation never round-trips to DRAM. Both
layouts are the SAME shape/sharding ([1,1,T,1280] col-sharded H/4) — they differ
only in placement (DRAM-interleaved vs L1-width-sharded over 10 cores). Earlier
per-op sharding attempts (T1) netted ~0 because the residual reverted to DRAM at
the add boundary.

This is an ITERATION HARNESS, not a pass/fail gate. It exercises the exact decode
residual chain that touches ``skip_mem_cfg``:

    attn_in = attention_norm(x)              # col-sharded norm
    h_new   = x + attn_out                   # <- residual add (skip_mem_cfg)
    ff_in   = ff_norm(h_new)                 # col-sharded norm
    ff_out  = MLP(ff_in)                     # _mlp_decode_qwen36
    out     = ff_out + h_new                 # <- residual add (skip_mem_cfg)

Attention is deliberately replaced by a synthetic ``attn_out`` so the probe needs
no KV cache / recurrent state and does not mutate anything between the two runs.

For each stage it reports, in ONE run:
  - GOLDEN: the chain with residual in DRAM (today's contract).
  - CANDIDATE: the chain with residual placed in DECODE_RESIDUAL_MEMCFG (L1).
  - per stage: did the op accept L1 input, what memcfg did it OUTPUT (L1-sharded
    vs forced back to DRAM), and PCC(candidate, golden).

The ops whose output is forced back to DRAM are the work items that block keeping
the residual in L1. PCC confirms the L1 path is numerically equivalent where it runs.

Run:
    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) && source python_env/bin/activate
    export HF_MODEL=Qwen/Qwen3.6-27B MESH_DEVICE=BH_GLX \\
        QWEN36_FULLATTN_WO_TUNED=1 QWEN36_DELTA_OP_TUNED=1 \\
        QWEN36_CCL_NUM_LINKS_DELTA=2 QWEN36_SEQ_CORES_PER_HEAD=4
    pytest models/demos/qwen3_6_galaxy_v2/tests/test_decode_l1_residual_probe.py -v -s
"""
from __future__ import annotations

import json
import pathlib

import pytest
import torch

import ttnn

_SNAPSHOT = pathlib.Path(
    "/home/tt-admin/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B"
    "/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
)

_B = 1
_H = 5120
# DECODE_RESIDUAL_MEMCFG is (1,1,32,128) — the decode hidden is tile-padded to 32
# rows. The prefill-norm primitive (tt_distributed_rmsnorm) divides by a tile/row
# count that is degenerate at 1 row (SIGFPE), so the probe uses the real 32-row
# tile shape. PCC compares DRAM vs L1 on the identical 32-row input — valid.
_T = 32
_PCC_THRESH = 0.99


@pytest.fixture(scope="module")
def bh_glx_mesh():
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
    )
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(8, 4))
    yield mesh
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _load_state_dict_for_layer(snapshot_dir: pathlib.Path, layer_idx: int) -> dict:
    """Load only the weights needed for one decoder layer, relabeled to layer 0.

    Mirrors test_deltanet_layer_isolated_pcc.py so the 1-layer model build finds
    its keys at index 0.
    """
    with open(snapshot_dir / "model.safetensors.index.json") as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]
    from safetensors.torch import load_file as load_st

    needed_prefixes = [
        "model.language_model.embed_tokens.",
        "model.language_model.norm.",
        "lm_head.",
        f"model.language_model.layers.{layer_idx}.",
    ]
    needed_keys = [k for k in weight_map if any(k.startswith(p) for p in needed_prefixes)]
    files = sorted({weight_map[k] for k in needed_keys})
    sd: dict = {}
    for fn in files:
        shard = load_st(str(snapshot_dir / fn))
        for k in needed_keys:
            if k in shard:
                sd[k] = shard[k]
    if layer_idx != 0:
        old_prefix = f"model.language_model.layers.{layer_idx}."
        new_prefix = "model.language_model.layers.0."
        sd = {(new_prefix + k[len(old_prefix) :] if k.startswith(old_prefix) else k): v for k, v in sd.items()}
    return sd


def _build_tt_model(mesh, state_dict, pattern):
    from models.demos.qwen3_6_galaxy_v2.tt.llama_model import TtTransformer
    from models.demos.qwen3_6_galaxy_v2.tt.qwen36_model_config import TtQwen36ModelArgs

    args = TtQwen36ModelArgs(mesh)
    args.n_layers = 1
    args.linear_attention_pattern = pattern
    weight_cache_path = args.weight_cache_path(ttnn.bfloat8_b)
    weight_cache_path.mkdir(parents=True, exist_ok=True)
    model = TtTransformer(
        args=args,
        dtype=ttnn.bfloat8_b,
        mesh_device=mesh,
        state_dict=state_dict,
        weight_cache_path=weight_cache_path,
        paged_attention_config=None,
        use_paged_kv_cache=False,
    )
    return model, args


def _send_col_sharded(t_full_h: torch.Tensor, mesh, args):
    """[B, T, H] full-H torch -> col-sharded [1,1,T,H/4] per device, DRAM, bf16."""
    B, T, H = t_full_h.shape
    t_4d = t_full_h.reshape(1, 1, T, H)
    return ttnn.from_torch(
        t_4d,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, 3), mesh_shape=args.cluster_shape),
    )


def _gather_full(tt_tensor, mesh, args, T: int):
    out = ttnn.to_torch(
        tt_tensor,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, dims=(0, 3), mesh_shape=args.cluster_shape),
    )
    out = out[0:1]
    while out.dim() > 3 and out.shape[0] == 1:
        out = out.squeeze(0)
    if out.dim() == 3:
        out = out[:, :T, :]
    return out.float()


def _build_partial_rope_cos_sin_tt(mesh, T: int):
    cos_ref = torch.zeros(T, 64, dtype=torch.bfloat16)
    sin_ref = torch.zeros(T, 64, dtype=torch.bfloat16)
    mk = lambda t: ttnn.from_torch(
        t.unsqueeze(0),
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    return mk(cos_ref), mk(sin_ref)


def _warmup_prefill(model, mesh, args, T=128):
    """Run one real prefill forward to lazily wire tt_ccl into layers/norms.

    The norms' CCL (tt_distributed_rmsnorm) is initialized inside model.forward
    (model.py:143-160), NOT at build — calling submodules directly before this
    SIGFPEs (tt_ccl is None). The state this populates is irrelevant: the probe
    uses a synthetic attn_out, never the attention output.
    """
    x = torch.randn(_B, T, _H, dtype=torch.bfloat16) * 0.1
    x_tt = _send_col_sharded(x, mesh, args)
    cos_tt, sin_tt = _build_partial_rope_cos_sin_tt(mesh, T)
    chunk_start_idx_tt = ttnn.from_torch(
        torch.tensor([0], dtype=torch.int32),
        device=mesh,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    model.forward(
        x_tt,
        current_pos=None,
        rot_mats=(cos_tt, sin_tt),
        user_id=0,
        mode="prefill",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=chunk_start_idx_tt,
        start_pos=0,
        get_last_token=-1,
        kv_cache=None,
        batch_size=1,
    )
    x_tt.deallocate(True)


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _memloc(tt_tensor) -> str:
    """Compact description of where a tensor lives: L1-sharded / L1 / DRAM."""
    try:
        mc = tt_tensor.memory_config()
        bt = str(mc.buffer_type).split(".")[-1]
        layout = str(mc.memory_layout).split(".")[-1]
        sharded = "SHARDED" if "SHARD" in layout.upper() or mc.shard_spec is not None else "INTERLEAVED"
        return f"{bt}/{sharded}"
    except Exception as e:  # pragma: no cover - diagnostic only
        return f"?({e})"


def _run_residual_chain(blk, x, attn_out, skip_mem_cfg, stage_log, prefix):
    """Run the decode residual chain. Records per-stage memloc into stage_log.

    Returns (out_tensor, {stage_name: tt_tensor}) for PCC comparison. Each op is
    run defensively: on exception we record it and re-raise (the candidate path
    failing IS a result we want surfaced).
    """
    norm_mode = "prefill"  # qwen36 decode runs norms/MLP via the prefill primitives
    stages = {}

    attn_in, _ = blk.attention_norm(x, None, norm_mode)
    stage_log.append((f"{prefix}.attn_norm_out", _memloc(attn_in)))
    stages["attn_norm"] = attn_in

    h_new = ttnn.add(x, attn_out, memory_config=skip_mem_cfg)
    stage_log.append((f"{prefix}.h_new(resid_add1)", _memloc(h_new)))
    stages["h_new"] = h_new

    ff_in, _ = blk.ff_norm(h_new, None, norm_mode)
    stage_log.append((f"{prefix}.ff_norm_out", _memloc(ff_in)))
    stages["ff_norm"] = ff_in

    ff_out = blk._mlp_decode_qwen36(ff_in, batch_size=1)
    stage_log.append((f"{prefix}.mlp_out", _memloc(ff_out)))
    stages["mlp"] = ff_out

    out = ttnn.add(ff_out, h_new, memory_config=skip_mem_cfg)
    stage_log.append((f"{prefix}.out(resid_add2)", _memloc(out)))
    stages["out"] = out

    return out, stages


@pytest.mark.hardware
@pytest.mark.parametrize(
    "layer_idx,pattern,label",
    [
        (0, ["linear_attention"], "GDN"),
        (3, ["full_attention"], "FA"),
    ],
)
def test_decode_l1_residual_probe(bh_glx_mesh, layer_idx, pattern, label):
    mesh = bh_glx_mesh
    state_dict = _load_state_dict_for_layer(_SNAPSHOT, layer_idx)
    model, args = _build_tt_model(mesh, state_dict, pattern)
    blk = model.layers[0]
    is_lin = getattr(blk, "is_linear_attention_layer", False)
    print(f"\n[{label}] layer_idx={layer_idx} is_linear_attention_layer={is_lin}")

    # Wire tt_ccl into the layer/norms (lazy init inside model.forward).
    _warmup_prefill(model, mesh, args)
    print(f"[{label}] warmup prefill done — tt_ccl wired")

    l1_residual_memcfg = args.model_config["DECODE_RESIDUAL_MEMCFG"]
    print(f"[{label}] DECODE_RESIDUAL_MEMCFG = {l1_residual_memcfg}")

    torch.manual_seed(0)
    x_full = torch.randn(_B, _T, _H, dtype=torch.bfloat16) * 0.1
    a_full = torch.randn(_B, _T, _H, dtype=torch.bfloat16) * 0.1

    stage_log: list[tuple[str, str]] = []

    # ---- GOLDEN: residual in DRAM (today's contract) ----
    x_dram = _send_col_sharded(x_full, mesh, args)
    a_dram = _send_col_sharded(a_full, mesh, args)
    out_dram, golden_stages = _run_residual_chain(
        blk, x_dram, a_dram, ttnn.DRAM_MEMORY_CONFIG, stage_log, f"{label}/DRAM"
    )
    golden = {k: _gather_full(v, mesh, args, _T) for k, v in golden_stages.items()}

    # ---- CANDIDATE: residual placed in L1 (DECODE_RESIDUAL_MEMCFG) ----
    x_l1_src = _send_col_sharded(x_full, mesh, args)
    a_l1_src = _send_col_sharded(a_full, mesh, args)
    cand_error = None
    cand = {}
    try:
        x_l1 = ttnn.to_memory_config(x_l1_src, l1_residual_memcfg)
        a_l1 = ttnn.to_memory_config(a_l1_src, l1_residual_memcfg)
        stage_log.append((f"{label}/L1.x_in", _memloc(x_l1)))
        stage_log.append((f"{label}/L1.attn_out_in", _memloc(a_l1)))
        _, cand_stages = _run_residual_chain(blk, x_l1, a_l1, l1_residual_memcfg, stage_log, f"{label}/L1")
        cand = {k: _gather_full(v, mesh, args, _T) for k, v in cand_stages.items()}
    except Exception as e:  # surfacing WHICH op rejects L1 input is the point
        cand_error = e
        stage_log.append((f"{label}/L1.ERROR", repr(e)[:200]))

    # ---- Report ----
    print(f"\n[{label}] ===== per-stage memory placement =====")
    for name, loc in stage_log:
        print(f"    {name:42s} {loc}")

    print(f"\n[{label}] ===== PCC candidate(L1) vs golden(DRAM) =====")
    if cand_error is not None:
        print(f"    CANDIDATE L1 path FAILED: {cand_error!r}")
        print(f"    -> first op that rejects L1-sharded residual identified above.")
    else:
        for k in golden:
            if k in cand:
                print(f"    {k:12s} PCC={_pcc(golden[k], cand[k]):.6f}  " f"goldenloc/candloc reported above")

    # ---- Diagnostic assertions (informational gate) ----
    # 1) The DRAM golden chain must produce the expected col-sharded H/4 shape.
    assert golden["out"].shape[-1] == _H, f"golden out H mismatch: {golden['out'].shape}"
    # 2) If the L1 candidate ran, it must be numerically equivalent to DRAM.
    if cand_error is None:
        pcc_out = _pcc(golden["out"], cand["out"])
        print(f"\n[{label}] FINAL out PCC (L1 vs DRAM) = {pcc_out:.6f}")
        assert pcc_out > _PCC_THRESH, f"L1 residual chain diverged: PCC {pcc_out} < {_PCC_THRESH}"
    else:
        pytest.fail(
            f"[{label}] L1-residual candidate path did not run end-to-end "
            f"(see per-stage log for the blocking op): {cand_error!r}"
        )
