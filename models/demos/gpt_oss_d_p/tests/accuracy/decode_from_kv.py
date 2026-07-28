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
import hashlib
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


def _debug_shard_check(tt_tensor, np_source: np.ndarray, real_len: int, mesh, label: str, layout: str = "sp_sharded"):
    """Read back each device tensor from a freshly-sharded multi-device tensor and
    PCC-compare against the expected shard from np_source.  Isolates whether
    ShardTensor2dMesh produced the right per-device shards.
    """
    sp, tp = tuple(mesh.shape)
    num_kv_heads = np_source.shape[0]
    head_dim = np_source.shape[2]
    local_kv_heads = num_kv_heads // tp
    per_device_seq = tt_tensor.shape[2]
    devs = ttnn.get_device_tensors(tt_tensor)
    print(
        f"[debug] {label} ({layout}): per-device tt_tensor.shape={tuple(tt_tensor.shape)} n_devices={len(devs)}",
        flush=True,
    )
    for r in range(sp):
        for c in range(tp):
            dev_t = ttnn.to_torch(devs[r * tp + c]).float()  # [1, local_kv_heads, per_device_seq, head_dim]
            head_lo = c * local_kv_heads
            head_hi = head_lo + local_kv_heads
            if layout == "sp_sharded":
                seq_lo = r * per_device_seq
                seq_hi = min(seq_lo + per_device_seq, real_len)
            else:  # replicated: every row has the full [0, real_len)
                seq_lo = 0
                seq_hi = min(per_device_seq, real_len)
            if seq_hi <= seq_lo:
                continue
            expected_len = seq_hi - seq_lo
            expected = torch.from_numpy(np_source[head_lo:head_hi, seq_lo:seq_hi, :]).float()
            got = dev_t[0, :, :expected_len, :]
            pcc = _pcc(expected, got)
            print(
                f"[debug] {label} r={r} c={c} (heads[{head_lo}:{head_hi}] seq[{seq_lo}:{seq_hi}]) "
                f"pcc={pcc:.4f}  |exp|={expected.norm():.3f} |got|={got.norm():.3f}",
                flush=True,
            )


def scatter_kv_into_cache(
    k_cache,
    v_cache,
    k_np: np.ndarray,
    v_np: np.ndarray,
    mesh,
    real_len: int,
    layout: str = "sp_sharded",
    debug: bool = False,
):
    """Inverse of kv_cache_prefill._gather_kv_cache.  Two layouts:

    layout="sp_sharded" (matches the original SP=rows prefill dump):
      dim 2 (seq) across SP rows using the original block-padded isl_per_row
      = ceil(real_len / (sp*32)) * 32 / sp; dim 1 (heads) across TP cols.
      Each row's local cache holds positions [0, isl_per_row).  Compatible
      with _gather_kv_cache for round-trip verification, but NOT usable for
      decode (each row only has its own sequence shard).

    layout="replicated" (needed for decode):
      dim 2 (seq) NOT sharded — the sequence is replicated across rows so
      every row's cache holds the full [0, real_len).  dim 1 (heads) still
      TP-sharded across cols.  Decode's SDPA runs per-row (rows are EP
      replicas), and each row needs the full history to attend over.
    """
    sp, tp = tuple(mesh.shape)
    num_kv_heads, gathered_seq_len, head_dim = k_np.shape
    assert gathered_seq_len == real_len, f"KV .npy seq dim ({gathered_seq_len}) does not match real_len ({real_len})"
    assert num_kv_heads % tp == 0, f"num_kv_heads={num_kv_heads} not divisible by tp={tp}"

    max_seq_len_cache = k_cache.shape[2]

    if layout == "sp_sharded":
        # See kv_cache_prefill.py:288-290 for the block-padding math.
        block = sp * 32
        padded_seq = ((real_len + block - 1) // block) * block
        assert padded_seq <= max_seq_len_cache, f"padded_seq={padded_seq} exceeds cache capacity {max_seq_len_cache}"
        total_seq = padded_seq
        mesh_dims = (2, 1)
    elif layout == "replicated":
        # Tile-align only; no per-row split of the sequence.
        padded_seq = ((real_len + 31) // 32) * 32
        assert padded_seq <= max_seq_len_cache, f"padded_seq={padded_seq} exceeds cache capacity {max_seq_len_cache}"
        total_seq = padded_seq
        mesh_dims = (None, 1)
    else:
        raise ValueError(f"unknown layout {layout!r}; use 'sp_sharded' or 'replicated'")

    def _prep(np_arr):
        host = torch.zeros(1, num_kv_heads, total_seq, head_dim, dtype=torch.float32)
        host[:, :, :real_len, :] = torch.from_numpy(np_arr).float().unsqueeze(0)
        return ttnn.from_torch(
            host,
            device=mesh,
            dtype=k_cache.dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=mesh.shape, dims=mesh_dims),
        )

    tt_k = _prep(k_np)
    tt_v = _prep(v_np)
    if debug:
        print(
            f"[debug] scatter: layout={layout} k_np.shape={k_np.shape} real_len={real_len} "
            f"cache_max_seq_len={max_seq_len_cache} sp={sp} tp={tp} "
            f"local_kv_heads={num_kv_heads // tp} padded_seq={padded_seq} mesh_dims={mesh_dims}",
            flush=True,
        )
        _debug_shard_check(tt_k, k_np, real_len, mesh, "K post-shard", layout=layout)
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

    sp, tp = shape[0], shape[1]
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

    model, _tok, _mesh_config, _requested_max_seq_len = _build_model(mesh, shape, args, real_len_hint=real_len)
    # The model's KV cache dim 2 is dictated by its config (often 128k), not the
    # value we compute in _build_model.  Read the actual per-device dim 2 from
    # the first layer's cache.  isl_per_row must match what the original prefill
    # used (block=sp*32, padded_seq=ceil(real_len/block)*block, isl=padded/sp),
    # not cache_dim2/sp.
    cache_max_seq_len = model.layers[0].self_attn.kv_cache[0].shape[2]
    block = sp * 32
    padded_seq = ((real_len + block - 1) // block) * block
    isl_per_row = padded_seq // sp
    print(
        f"[decode_from_kv] model built; real_len={real_len} "
        f"cache_max_seq_len={cache_max_seq_len} padded_seq={padded_seq} "
        f"isl_per_row={isl_per_row}",
        flush=True,
    )

    any_fail = False
    results = []
    for idx, i in enumerate(layers_to_check):
        k_np = np.load(kv_dir / f"tt_layer{i}_k.npy")
        v_np = np.load(kv_dir / f"tt_layer{i}_v.npy")
        k_cache, v_cache = model.layers[i].self_attn.kv_cache

        debug_this = idx == 0  # only for first evaluated layer
        if debug_this:
            print(f"[debug] layer {i}: k_cache.shape={tuple(k_cache.shape)} k_cache.dtype={k_cache.dtype}", flush=True)
            print(f"[debug] layer {i}: k_np.shape={k_np.shape} k_np.dtype={k_np.dtype}", flush=True)
            print(
                f"[debug] layer {i}: |k_np|_L2={float(np.linalg.norm(k_np)):.3f} "
                f"mean={float(k_np.mean()):.4f} std={float(k_np.std()):.4f}",
                flush=True,
            )
        scatter_kv_into_cache(k_cache, v_cache, k_np, v_np, mesh, real_len, layout="sp_sharded", debug=debug_this)
        ttnn.synchronize_device(mesh)

        k_round = _gather_kv_cache(k_cache, mesh, real_len, isl_per_row)
        v_round = _gather_kv_cache(v_cache, mesh, real_len, isl_per_row)

        pcc_k = _pcc(torch.from_numpy(k_np), k_round)
        pcc_v = _pcc(torch.from_numpy(v_np), v_round)
        passed = pcc_k >= args.pcc_threshold and pcc_v >= args.pcc_threshold

        if debug_this:
            print(
                f"[debug] layer {i}: k_round.shape={tuple(k_round.shape)} " f"|k_round|_L2={float(k_round.norm()):.3f}",
                flush=True,
            )
            # Per-row PCC (SP shards) — which rows of the sequence came back wrong?
            k_np_t = torch.from_numpy(k_np).float()
            for r in range(sp):
                lo = r * isl_per_row
                hi = min(lo + isl_per_row, real_len)
                if hi <= lo:
                    continue
                pcc_r = _pcc(k_np_t[:, lo:hi, :], k_round[:, lo:hi, :])
                print(f"[debug] layer {i}: K row {r} (seq[{lo}:{hi}]) PCC={pcc_r:.4f}", flush=True)
            # Per-TP-col PCC (heads) — which head-slices came back wrong?
            local_kv_heads = k_np.shape[0] // tp
            for c in range(tp):
                hlo = c * local_kv_heads
                hhi = hlo + local_kv_heads
                pcc_c = _pcc(k_np_t[hlo:hhi, :, :], k_round[hlo:hhi, :, :])
                print(f"[debug] layer {i}: K col {c} (heads[{hlo}:{hhi}]) PCC={pcc_c:.4f}", flush=True)

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


def _run_decode(args, mesh, shape, record, oracle_dir, kv_dir):
    from transformers import AutoTokenizer

    sp, tp = shape[0], shape[1]
    real_len = record["n_tokens"]
    if args.num_tokens is not None and args.num_tokens != real_len:
        print(
            f"[decode_from_kv] ERROR: --num-tokens={args.num_tokens} but oracle record " f"has n_tokens={real_len}.",
            flush=True,
        )
        return 1

    if "generation" not in record or not record["generation"]:
        print(
            "[decode_from_kv] ERROR: oracle record has no 'generation' list. "
            "Re-run hf_reference_oracle.py with --gen-tokens N.",
            flush=True,
        )
        return 1
    oracle_gen = record["generation"]
    gen_tokens = args.gen_tokens if args.gen_tokens > 0 else len(oracle_gen)
    if gen_tokens > len(oracle_gen):
        print(
            f"[decode_from_kv] ERROR: --gen-tokens={gen_tokens} exceeds oracle's "
            f"generation length ({len(oracle_gen)}).",
            flush=True,
        )
        return 1
    print(
        f"[decode_from_kv] decoding {gen_tokens} tokens from real_len={real_len}, "
        f"seeded from oracle argmax={oracle_gen[0]['argmax_id']} "
        f"({oracle_gen[0]['argmax_text']!r})",
        flush=True,
    )

    # Locate the TT KV dump layers.
    tt_layers_present = sorted(int(p.stem.split("layer")[1].split("_")[0]) for p in kv_dir.glob("tt_layer*_k.npy"))
    if not tt_layers_present:
        print(f"[decode_from_kv] ERROR: no tt_layer*_k.npy files in {kv_dir}", flush=True)
        return 1

    model, _tok_unused, _mesh_config, _requested_max_seq_len = _build_model(
        mesh, shape, args, real_len_hint=real_len + gen_tokens
    )
    n_model_layers = len(model.layers)
    missing = [i for i in range(n_model_layers) if i not in tt_layers_present]
    if missing:
        print(
            f"[decode_from_kv] ERROR: KV dump missing layers {missing[:8]}"
            f"{'...' if len(missing) > 8 else ''} — decode needs all {n_model_layers} layers.",
            flush=True,
        )
        return 1

    # Load HF tokenizer for decoding token ids to text in the output JSON.
    from models.demos.gpt_oss.tt.model_config import ModelArgs

    tok = AutoTokenizer.from_pretrained(ModelArgs(mesh_device=mesh).model_path, trust_remote_code=True)

    # Scatter every layer's KV into the on-device cache with REPLICATED layout —
    # each row's cache must hold the full [0, real_len) so decode's SDPA (which
    # runs per row as an EP replica) sees the full history.
    print(f"[decode_from_kv] loading KV for {n_model_layers} layers (replicated layout)...", flush=True)
    for i in range(n_model_layers):
        k_np = np.load(kv_dir / f"tt_layer{i}_k.npy")
        v_np = np.load(kv_dir / f"tt_layer{i}_v.npy")
        assert k_np.shape[1] == real_len, f"layer {i}: k_np seq={k_np.shape[1]} != real_len={real_len}"
        k_cache, v_cache = model.layers[i].self_attn.kv_cache
        scatter_kv_into_cache(k_cache, v_cache, k_np, v_np, mesh, real_len, layout="replicated", debug=(i == 0))
    ttnn.synchronize_device(mesh)
    print("[decode_from_kv] KV load complete", flush=True)

    # Build the kv_cache list the model expects.
    kv_cache_list = [layer.self_attn.kv_cache for layer in model.layers]

    # Seed decode from the oracle's first-generated token.  All decode work is
    # single-user (batch=1); the model was built with max_local_batch_size=1.
    seed_id = oracle_gen[0]["argmax_id"]
    tt_gen = [
        {
            "pos": real_len,
            "argmax_id": seed_id,
            "argmax_text": oracle_gen[0]["argmax_text"],
            "top5": None,  # seed came from oracle, no TT logits at this step
            "seed": True,
        }
    ]

    out_tok = torch.tensor([seed_id], dtype=torch.int64)
    current_pos = torch.tensor([real_len], dtype=torch.int64)

    for step in range(1, gen_tokens):
        tt_tokens, tt_current_pos, tt_rope_idxs, _ = model.prepare_inputs_decode(out_tok, current_pos, None)
        decode_out = model.ttnn_decode_forward(
            tt_tokens,
            tt_current_pos,
            rot_mat_idxs=tt_rope_idxs,
            page_table=None,
            kv_cache=kv_cache_list,
            on_device_logits=False,
        )
        logits_ttnn = decode_out[0] if isinstance(decode_out, tuple) else decode_out
        logits_host = logits_ttnn.cpu()
        logits_torch = model.process_output_decode(logits_host, B=1, S=1, is_tokens=False)
        logits_flat = logits_torch.view(-1).float()
        topk = torch.topk(logits_flat, 5)
        argmax_id = int(topk.indices[0])
        top5 = [{"id": int(i), "text": tok.decode([int(i)]), "logit": float(logits_flat[int(i)])} for i in topk.indices]
        pos_now = real_len + step
        tt_gen.append(
            {
                "pos": pos_now,
                "argmax_id": argmax_id,
                "argmax_text": tok.decode([argmax_id]),
                "top5": top5,
                "seed": False,
            }
        )
        print(
            f"[decode_from_kv] step {step:3d} pos={pos_now} tt_argmax={argmax_id} "
            f"({tok.decode([argmax_id])!r})  oracle={oracle_gen[step]['argmax_id']} "
            f"({oracle_gen[step]['argmax_text']!r})  "
            f"{'MATCH' if argmax_id == oracle_gen[step]['argmax_id'] else 'MISMATCH'}",
            flush=True,
        )
        out_tok = torch.tensor([argmax_id], dtype=torch.int64)
        current_pos = current_pos + 1

    # Persist output next to the KV dump.
    key = hashlib.sha256(args.prompt.encode()).hexdigest()[:12]
    out_path = kv_dir / f"tt_generation_{key}.json"
    with out_path.open("w") as f:
        json.dump(
            {
                "prompt": args.prompt,
                "real_len": real_len,
                "gen_tokens": gen_tokens,
                "seed_from_oracle": True,
                "generation": tt_gen,
            },
            f,
            indent=2,
        )
    print(f"[decode_from_kv] wrote {out_path}", flush=True)
    return 0


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
    ap.add_argument(
        "--gen-tokens",
        type=int,
        default=0,
        help="decode mode: number of tokens to generate.  Defaults to the length "
        "of the oracle's generation record.  Must not exceed it.",
    )
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
            return _run_decode(args, mesh, shape, record, oracle_dir, kv_dir)
    finally:
        ttnn.close_mesh_device(mesh)
        if fabric is not None:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    sys.exit(main())
