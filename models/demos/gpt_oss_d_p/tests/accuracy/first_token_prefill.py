# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Real-weights prefill -> first token on a BH Galaxy mesh.

Loads the GPT-OSS checkpoint, builds the gpt_oss_d_p Model (SP-sharded for
multi-row meshes), embeds a prompt, runs one ttnn_prefill_forward, and decodes
the argmax first token + top-5. PREFILL ONLY — this tests the prefill engine;
decode lives in the decode worker.

The script handles SP sharding transparently: for a 1×8 mesh (SP=1) tokens are
replicated across the single row; for a 4×8 mesh (SP=4) the sequence is split
across 4 rows.

Run (1×8):
  cd /path/to/tt-metal
  export HF_MODEL=/data/jmalone/.cache/huggingface/hub/models--openai--gpt-oss-120b/gpt-oss-120b
  export TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_1x8_mesh_graph_descriptor.textproto
  python3 models/demos/gpt_oss_d_p/tests/accuracy/first_token_prefill.py \\
      --prompt "What are the prime factors of 1?" \\
      --dump-logits /tmp/tt_logits.npy \\
      --oracle-dir /data/jmalone/gpt_oss_ref

Run (4×8 Galaxy):
  export TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_4x8_mesh_graph_descriptor.textproto
  python3 models/demos/gpt_oss_d_p/tests/accuracy/first_token_prefill.py --rows 4 --cols 8   --prompt "What are the prime factors of 1?"
"""

import argparse
import gc
import json
import os
import sys

import numpy as np
import torch

import ttnn


def _sp_shard_embed(mesh, model, padded_ids, sp, isl_per_row):
    """Embed token ids with SP sharding across mesh rows.

    For SP=1 (single-row mesh) this reduces to a normal embedding with full
    sequence on every device in the row. For SP>1 each mesh row gets
    isl_per_row consecutive tokens.

    Args:
        mesh: MeshDevice
        model: gpt_oss_d_p Model (provides model.embedding_weight)
        padded_ids: list[int] of length sp * isl_per_row
        sp: sequence-parallel factor (== mesh.shape[0])
        isl_per_row: tokens per SP row

    Returns:
        4-D ttnn.Tensor [1, 1, isl_per_row, hidden] sharded across SP rows,
        replicated across TP columns.
    """
    t = torch.tensor(padded_ids, dtype=torch.int64).reshape(sp, 1, 1, isl_per_row)
    tokens = ttnn.from_torch(
        t,
        device=mesh,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=mesh.shape, dims=(0, None)),
    )
    emb = ttnn.embedding(tokens, model.embedding_weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
    tokens.deallocate(True)
    if len(emb.shape) == 3:
        emb = ttnn.unsqueeze_to_4D(emb)
    return emb


def _extract_logit_vec(tt_logits, mesh, real_len, isl_per_row, vocab_size):
    """Pull the last-real-token logit vector from the SP row that holds it.

    With SP>1 the sequence is split across rows; only the row whose shard
    contains position (real_len - 1) has the meaningful last-token logit.

    Returns:
        torch.Tensor of shape [vocab_size], float32
    """
    sp_row = (real_len - 1) // isl_per_row
    pos_within_tile = (real_len - 1) % 32  # get_last_token aligns to 32-boundary
    tp = mesh.shape[1]
    dts = ttnn.get_device_tensors(tt_logits)
    row_tensors = [ttnn.to_torch(dts[sp_row * tp + col]) for col in range(tp)]
    # Concat along vocab dim: [1, 1, 32, padded_vocab]
    full_logits = torch.cat(row_tensors, dim=-1).float()
    return full_logits[0, 0, pos_within_tile, :vocab_size]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--prompt", type=str, default="What are the prime factors of 1?")
    ap.add_argument(
        "--dump-logits", type=str, default=None, help="path to save last-pos logit vector (.npy) for PCC vs oracle"
    )
    ap.add_argument(
        "--oracle-dir", type=str, default=None, help="directory containing ref_results.json from hf_reference_oracle.py"
    )
    args = ap.parse_args()

    from transformers import AutoConfig, AutoTokenizer

    from models.common.utility_functions import is_blackhole
    from models.demos.gpt_oss.config import MeshConfig, ModeConfig
    from models.demos.gpt_oss.tt.model_config import ModelArgs
    from models.demos.gpt_oss.utils.general_utils import get_default_num_links
    from models.demos.gpt_oss_d_p.tt.ccl import CCLManager
    from models.demos.gpt_oss_d_p.tt.model import Model

    shape = (args.rows, args.cols)
    sp = shape[0]  # sequence-parallel factor == mesh rows

    if shape == (1, 1):
        fabric = None
        topology = ttnn.Topology.Linear
    elif is_blackhole() and shape[0] == 1:
        # Single-row BH (e.g. 1×8 T3K): TP chips are in a line, not a ring.
        fabric = ttnn.FabricConfig.FABRIC_1D
        topology = ttnn.Topology.Linear
    elif is_blackhole():
        # Multi-row BH Galaxy (e.g. 4×8): BH plain-mesh MGD has LINE topology on both
        # axes; ring fabric requires torus links that don't exist on BH.
        fabric = ttnn.FabricConfig.FABRIC_1D
        topology = ttnn.Topology.Linear
    else:
        fabric = ttnn.FabricConfig.FABRIC_1D_RING
        topology = ttnn.Topology.Ring

    print(
        f"[prefill] mesh={shape} SP={sp} TP={shape[1]} fabric={fabric} topology={topology} prompt={args.prompt!r}",
        flush=True,
    )

    if fabric is not None:
        ttnn.set_fabric_config(fabric)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*shape))
    print(f"[prefill] mesh opened: {tuple(mesh.shape)} ndev={mesh.get_num_devices()}", flush=True)

    try:
        model_args = ModelArgs(mesh_device=mesh)
        hf_config = AutoConfig.from_pretrained(model_args.model_path, trust_remote_code=True)
        tok = AutoTokenizer.from_pretrained(model_args.model_path, trust_remote_code=True)

        # Tokenize (plain, no chat template — matches GPT-OSS demo instruct=False)
        ids = tok(args.prompt)["input_ids"]
        real_len = len(ids)

        # Pad sequence to multiple of (sp * 32) so SP rows each get a whole-tile shard
        block = sp * 32
        max_seq_len = ((real_len + block - 1) // block) * block
        isl_per_row = max_seq_len // sp
        padded = ids + [tok.pad_token_id or 1] * (max_seq_len - real_len)
        print(f"[prefill] prompt_tokens={real_len} padded_seq={max_seq_len} isl_per_row={isl_per_row}", flush=True)

        print("[prefill] loading + converting real weights (slow on first run) ...", flush=True)
        state_dict = ModelArgs.load_state_dict(model_args.weights_path, dummy_weights=False)
        cache = model_args.weight_cache_path(ttnn.bfloat8_b)
        mesh_config = MeshConfig(
            shape, decode=ModeConfig(tp=shape[1], ep=shape[0]), prefill=ModeConfig(tp=shape[1], sp=shape[0], ep=1)
        )
        ccl = CCLManager(mesh, num_links=get_default_num_links(mesh), topology=topology)

        model = Model(
            mesh_device=mesh,
            hf_config=hf_config,
            state_dict=state_dict,
            ccl_manager=ccl,
            mesh_config=mesh_config,
            tensor_cache_path=str(cache),
            create_kv_cache=False,
            max_local_batch_size=1,
        )
        del state_dict
        gc.collect()
        print("[prefill] model built; weights on device", flush=True)

        # Embed tokens (SP-sharded) and run prefill
        emb = _sp_shard_embed(mesh, model, padded, sp, isl_per_row)

        # get_last_token: 32-tile start within the SP row holding the last real token
        local_last_pos = (real_len - 1) % isl_per_row
        get_last_token = (local_last_pos // 32) * 32

        # kv_cache=None: prefill logit accuracy doesn't depend on KV writes
        tt_logits = model.ttnn_prefill_forward(x=emb, kv_cache=None, get_last_token=get_last_token)
        ttnn.synchronize_device(mesh)
        print("[prefill] forward complete", flush=True)

        logit_vec = _extract_logit_vec(tt_logits, mesh, real_len, isl_per_row, hf_config.vocab_size)

        if not torch.isfinite(logit_vec).all():
            print("[prefill] ERROR: logits contain NaN or Inf", flush=True)
            return 1

        top = torch.topk(logit_vec, 5)
        first_token_id = int(top.indices[0])
        print(f"[prefill] FIRST TOKEN id={first_token_id} -> {tok.decode([first_token_id])!r}", flush=True)
        print(f"[prefill] top-5: {[(int(i), tok.decode([int(i)])) for i in top.indices]}", flush=True)

        if args.dump_logits:
            np.save(args.dump_logits, logit_vec.numpy())
            print(f"[prefill] dumped last-pos logits -> {args.dump_logits}", flush=True)

        if args.oracle_dir:
            oracle_file = os.path.join(args.oracle_dir, "ref_results.json")
            with open(oracle_file) as f:
                records = json.load(f)
            record = next((r for r in records if r["prompt"] == args.prompt), None)
            if record is None:
                print(f"[prefill] WARNING: no oracle record found for prompt {args.prompt!r}", flush=True)
                print(
                    f"[prefill] Run hf_reference_oracle.py --prompt {args.prompt!r} --out {args.oracle_dir} first",
                    flush=True,
                )
            else:
                expected_id = record["argmax_id"]
                expected_text = record["argmax_text"]
                match = first_token_id == expected_id
                in_top5 = first_token_id in [e["id"] for e in record["top5"]]
                print(
                    f"[prefill] ORACLE CHECK: expected id={expected_id} ({expected_text!r}), "
                    f"got id={first_token_id} ({tok.decode([first_token_id])!r}) "
                    f"-> {'MATCH' if match else ('IN TOP-5' if in_top5 else 'MISMATCH')}",
                    flush=True,
                )

                # PCC against saved oracle logits
                oracle_logits_file = os.path.join(args.oracle_dir, record["logits_file"])
                if os.path.exists(oracle_logits_file):
                    oracle_logits = torch.tensor(np.load(oracle_logits_file), dtype=torch.float32)
                    if args.dump_logits:
                        tt_logits_np = torch.tensor(np.load(args.dump_logits), dtype=torch.float32)
                    else:
                        tt_logits_np = logit_vec
                    # Pearson correlation coefficient (PCC)
                    tt_z = tt_logits_np - tt_logits_np.mean()
                    ref_z = oracle_logits - oracle_logits.mean()
                    pcc = float((tt_z * ref_z).sum() / (tt_z.norm() * ref_z.norm() + 1e-8))
                    print(f"[prefill] PCC vs oracle logits: {pcc:.4f}", flush=True)

                if not match and not in_top5:
                    print("[prefill] FAIL — first token not in oracle top-5", flush=True)
                    return 1

        print("[prefill] PASS", flush=True)
        return 0

    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    sys.exit(main())
