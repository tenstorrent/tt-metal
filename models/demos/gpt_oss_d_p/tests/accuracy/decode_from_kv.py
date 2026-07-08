# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Load a KV cache dumped by kv_cache_prefill.py --save-tt-kv, then run TT decode
to generate tokens without re-doing prefill.  Comparing the generated tokens
against an HF oracle produced by hf_reference_oracle.py --gen-tokens N
localises decode-time bugs independently of prefill.

Modes:
  --mode check   Round-trip PCC self-check.  Scatter each layer's dump back
                 into a freshly-allocated on-device KV cache, then gather it
                 with the same routine kv_cache_prefill.py uses, and PCC
                 against the source .npy.  Expected PCC ~0.99 (bfloat8_b
                 quantization floor, since the on-device cache is bf8_b).

  --mode decode  (not yet implemented — M3.)

Example:
  export HF_MODEL=/data/jmalone/.cache/huggingface/hub/.../gpt-oss-120b
  export TT_MESH_GRAPH_DESC_PATH=...
  python3 models/demos/gpt_oss_d_p/tests/accuracy/decode_from_kv.py \\
      --kv-dir /data/jmalone/gpt_oss_tt_kv \\
      --oracle-dir /data/jmalone/gpt_oss_ref \\
      --prompt "What are the prime factors of 1?" \\
      --mode check
"""

import argparse
import gc
import json
import pathlib
import sys

import numpy as np
import torch

import ttnn

PCC_THRESHOLD_CHECK = 0.97


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    am = a - a.mean()
    bm = b - b.mean()
    denom = am.norm() * bm.norm()
    return float((am * bm).sum() / (denom + 1e-8))


def scatter_kv_into_cache(k_cache, v_cache, k_np: np.ndarray, v_np: np.ndarray, mesh, real_len: int):
    """Inverse of kv_cache_prefill._gather_kv_cache.

    Takes gathered [num_kv_heads, real_len, head_dim] tensors (Meta interleave,
    as saved by --save-tt-kv), re-shards for the current mesh, and calls
    ttnn.fill_cache on both K and V caches.

    Sharding:
      - dim 1 (heads) across TP cols → each col holds local_kv_heads = num_kv_heads/tp heads
      - dim 2 (seq)   across SP rows → each row holds isl_per_row positions

    The on-device cache is allocated with per-device seq dim = isl_per_row.
    We build a host tensor of shape [1, num_kv_heads, isl_per_row*sp, head_dim],
    fill positions [0, real_len) with data and zero-pad the rest, then let
    ShardTensor2dMesh(dims=(2, 1)) split it correctly across devices.
    """
    sp, tp = tuple(mesh.shape)
    num_kv_heads, gathered_seq_len, head_dim = k_np.shape
    assert gathered_seq_len == real_len, f"KV .npy seq dim ({gathered_seq_len}) does not match real_len ({real_len})"
    assert num_kv_heads % tp == 0, f"num_kv_heads={num_kv_heads} not divisible by tp={tp}"

    # Per-device cache dim 2 is the FULL max_seq_len (see attention/kv_cache.py:48-53),
    # not per-row.  Under SP sharding each row's fill_cache writes only
    # isl_per_row = max_seq_len/sp positions locally (starting at 0); positions
    # [isl_per_row, max_seq_len) stay zero.  Build host tensor sized max_seq_len,
    # shard dim 2 across sp rows so each row gets exactly isl_per_row positions.
    max_seq_len = k_cache.shape[2]
    assert max_seq_len % sp == 0, f"max_seq_len={max_seq_len} not divisible by sp={sp}"
    total_seq = max_seq_len
    assert real_len <= total_seq, f"real_len={real_len} exceeds cache capacity {total_seq}"

    def _prep(np_arr):
        host = torch.zeros(1, num_kv_heads, total_seq, head_dim, dtype=torch.float32)
        host[:, :, :real_len, :] = torch.from_numpy(np_arr).float().unsqueeze(0)
        return ttnn.from_torch(
            host,
            device=mesh,
            dtype=k_cache.dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=mesh.shape, dims=(2, 1)),
        )

    tt_k = _prep(k_np)
    tt_v = _prep(v_np)
    ttnn.fill_cache(k_cache, tt_k, batch_idx=0)
    ttnn.fill_cache(v_cache, tt_v, batch_idx=0)
    tt_k.deallocate(True)
    tt_v.deallocate(True)


def _gather_kv_cache(cache_tensor, mesh, real_len: int, isl_per_row: int) -> torch.Tensor:
    """Copy of kv_cache_prefill._gather_kv_cache to avoid a cross-file import."""
    sp, tp = tuple(mesh.shape)
    device_tensors = ttnn.get_device_tensors(cache_tensor)
    col_slices = []
    for col in range(tp):
        row_shards = []
        for row in range(sp):
            shard_len = min(isl_per_row, max(0, real_len - row * isl_per_row))
            if shard_len == 0:
                break
            t = ttnn.to_torch(device_tensors[row * tp + col]).float()
            row_shards.append(t[0, :, :shard_len, :])
        col_slices.append(torch.cat(row_shards, dim=1))
    return torch.cat(col_slices, dim=0)


def _build_model(mesh, shape, args, real_len_hint: int):
    """Instantiate a non-paged GPT-OSS model matching kv_cache_prefill's config."""
    from transformers import AutoConfig, AutoTokenizer

    from models.demos.gpt_oss.config import MeshConfig, ModeConfig
    from models.demos.gpt_oss.tt.model_config import ModelArgs
    from models.demos.gpt_oss.utils.general_utils import get_default_num_links
    from models.demos.gpt_oss_d_p.tt.ccl import CCLManager
    from models.demos.gpt_oss_d_p.tt.model import Model

    sp = shape[0]
    topology = ttnn.Topology.Ring if sp > 1 else ttnn.Topology.Linear

    model_args = ModelArgs(mesh_device=mesh)
    hf_config = AutoConfig.from_pretrained(model_args.model_path, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(model_args.model_path, trust_remote_code=True)

    # Match kv_cache_prefill's max_seq_len computation but allow headroom for
    # generated tokens.  Callers pass real_len_hint = real_len + gen_tokens.
    block = sp * 32
    max_seq_len = ((real_len_hint + block - 1) // block) * block

    state_dict = ModelArgs.load_state_dict(model_args.weights_path, dummy_weights=False)
    cache = model_args.weight_cache_path(ttnn.bfloat8_b)
    mesh_config = MeshConfig(
        shape,
        decode=ModeConfig(tp=shape[1], ep=shape[0]),
        prefill=ModeConfig(tp=shape[1], sp=shape[0], ep=1),
    )
    ccl = CCLManager(mesh, num_links=get_default_num_links(mesh), topology=topology)
    model = Model(
        mesh_device=mesh,
        hf_config=hf_config,
        state_dict=state_dict,
        ccl_manager=ccl,
        mesh_config=mesh_config,
        tensor_cache_path=str(cache),
        create_kv_cache=True,
        max_local_batch_size=1,
    )
    del state_dict
    gc.collect()
    return model, tok, mesh_config, max_seq_len


def _run_check(args, mesh, shape, record, oracle_dir, kv_dir):
    from models.common.utility_functions import is_blackhole  # noqa: F401  (mesh already opened)

    sp = shape[0]
    real_len = record["n_tokens"]
    if args.num_tokens is not None and args.num_tokens != real_len:
        print(
            f"[decode_from_kv] ERROR: --num-tokens={args.num_tokens} but oracle record "
            f"has n_tokens={real_len}.  Re-run the oracle with matching --num-tokens.",
            flush=True,
        )
        return 1

    available_layers = sorted(int(k) for k in record.get("kv_files", {}))
    if not available_layers:
        print("[decode_from_kv] ERROR: oracle record has no kv_files.", flush=True)
        return 1
    layers_to_check = (
        available_layers if args.max_layers is None else [i for i in available_layers if i < args.max_layers]
    )
    if not layers_to_check:
        print("[decode_from_kv] ERROR: no layers to check (check --max-layers).", flush=True)
        return 1

    tt_layers_present = sorted(int(p.stem.split("layer")[1].split("_")[0]) for p in kv_dir.glob("tt_layer*_k.npy"))
    missing = [i for i in layers_to_check if i not in tt_layers_present]
    if missing:
        print(
            f"[decode_from_kv] ERROR: TT KV dump missing layers {missing[:8]}"
            f"{'...' if len(missing) > 8 else ''}. Re-run kv_cache_prefill.py --save-tt-kv.",
            flush=True,
        )
        return 1

    model, _tok, _mesh_config, max_seq_len = _build_model(mesh, shape, args, real_len_hint=real_len)
    isl_per_row = max_seq_len // sp
    print(
        f"[decode_from_kv] model built; real_len={real_len} max_seq_len={max_seq_len} " f"isl_per_row={isl_per_row}",
        flush=True,
    )

    any_fail = False
    results = []
    for i in layers_to_check:
        k_np = np.load(kv_dir / f"tt_layer{i}_k.npy")
        v_np = np.load(kv_dir / f"tt_layer{i}_v.npy")
        k_cache, v_cache = model.layers[i].self_attn.kv_cache

        scatter_kv_into_cache(k_cache, v_cache, k_np, v_np, mesh, real_len)
        ttnn.synchronize_device(mesh)

        k_round = _gather_kv_cache(k_cache, mesh, real_len, isl_per_row)
        v_round = _gather_kv_cache(v_cache, mesh, real_len, isl_per_row)

        pcc_k = _pcc(torch.from_numpy(k_np), k_round)
        pcc_v = _pcc(torch.from_numpy(v_np), v_round)
        passed = pcc_k >= args.pcc_threshold and pcc_v >= args.pcc_threshold
        if not passed:
            any_fail = True
        results.append((i, pcc_k, pcc_v, passed))
        print(
            f"[decode_from_kv] layer {i:3d}: K PCC={pcc_k:.4f}  V PCC={pcc_v:.4f}  " f"{'PASS' if passed else 'FAIL'}",
            flush=True,
        )

    print("", flush=True)
    print(f"{'Layer':>6}  {'K PCC':>8}  {'V PCC':>8}  Status", flush=True)
    print("-" * 38, flush=True)
    for layer_idx, pcc_k, pcc_v, passed in results:
        print(f"{layer_idx:>6}  {pcc_k:>8.4f}  {pcc_v:>8.4f}  {'PASS' if passed else 'FAIL'}", flush=True)

    return 1 if any_fail else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument(
        "--mode",
        choices=["check", "decode"],
        default="check",
        help="check: round-trip PCC self-check on the loader (M2).  decode: run TT decode from loaded KV (M3, TBD).",
    )
    prompt_group = ap.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt", type=str, default=None)
    prompt_group.add_argument("--prompt-file", type=str, default=None)
    ap.add_argument(
        "--kv-dir",
        type=str,
        required=True,
        help="directory containing tt_layer*_k.npy / tt_layer*_v.npy from --save-tt-kv",
    )
    ap.add_argument(
        "--oracle-dir",
        type=str,
        required=True,
        help="directory containing ref_results.json from hf_reference_oracle.py",
    )
    ap.add_argument(
        "--num-tokens",
        type=int,
        default=None,
        help="optional sanity assertion: expected prompt length after truncation.  "
        "Must match the oracle record's n_tokens (and the --num-tokens used when "
        "generating the KV dump).  Not used for record lookup.",
    )
    ap.add_argument("--max-layers", type=int, default=None)
    ap.add_argument("--pcc-threshold", type=float, default=PCC_THRESHOLD_CHECK)
    args = ap.parse_args()

    if args.prompt_file is not None:
        with open(args.prompt_file) as f:
            args.prompt = f.read()
    elif args.prompt is None:
        args.prompt = "What are the prime factors of 1?"

    oracle_dir = pathlib.Path(args.oracle_dir)
    kv_dir = pathlib.Path(args.kv_dir)

    results_path = oracle_dir / "ref_results.json"
    if not results_path.exists():
        print(f"[decode_from_kv] ERROR: {results_path} not found.", flush=True)
        return 1
    with results_path.open() as f:
        records = json.load(f)
    record = next((r for r in records if r.get("prompt") == args.prompt), None)
    if record is None:
        print(f"[decode_from_kv] ERROR: no oracle record for prompt {args.prompt!r}", flush=True)
        return 1

    shape = (args.rows, args.cols)
    sp = shape[0]
    if shape == (1, 1):
        fabric = None
        topology = ttnn.Topology.Linear
    else:
        from models.common.utility_functions import is_blackhole

        if is_blackhole():
            fabric = ttnn.FabricConfig.FABRIC_1D
            topology = ttnn.Topology.Linear
        else:
            fabric = ttnn.FabricConfig.FABRIC_1D_RING
            topology = ttnn.Topology.Ring
        del topology  # not needed here; _build_model derives its own
    print(f"[decode_from_kv] mesh={shape} SP={shape[0]} TP={shape[1]} mode={args.mode}", flush=True)

    if fabric is not None:
        ttnn.set_fabric_config(fabric)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*shape))
    try:
        if args.mode == "check":
            return _run_check(args, mesh, shape, record, oracle_dir, kv_dir)
        else:
            print("[decode_from_kv] --mode decode not yet implemented (M3).", flush=True)
            return 2
    finally:
        ttnn.close_mesh_device(mesh)
        if fabric is not None:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    sys.exit(main())
