# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Quick check: run Code Predictor with a dummy hidden state and CB0 token.

Verifies that the Code Predictor produces reasonable CB1-15 tokens.
"""

import os
import sys
import torch

MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

def main():
    import ttnn
    from models.demos.qwen3_tts.tt.model_config import TalkerModelArgs, CodePredictorModelArgs
    from models.demos.qwen3_tts.tt.code_predictor import CodePredictorTransformer

    os.environ["HF_MODEL"] = MODEL_PATH

    device_ids = ttnn.get_device_ids()
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, len(device_ids)),
        dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER),
    )
    try:
        mesh.enable_program_cache()
    except AttributeError:
        ttnn.enable_program_cache(mesh)

    # Load Talker for codec_embed_weight
    talker_args = TalkerModelArgs(mesh_device=mesh, max_batch_size=1, max_seq_len=768, use_hf_rope=True)
    talker_sd = talker_args.load_state_dict()
    codec_embed_w = talker_sd["talker.tok_embeddings.weight"]  # [3072, 2048]
    print(f"codec_embed_weight: {codec_embed_w.shape}")

    # Build Code Predictor
    cp_args = CodePredictorModelArgs(mesh_device=mesh, max_batch_size=1, max_seq_len=128, use_hf_rope=True)
    cp_sd = cp_args.load_state_dict()
    cp_cache = cp_args.weight_cache_path(ttnn.bfloat16)

    cp = CodePredictorTransformer(
        args=cp_args, dtype=ttnn.bfloat16, mesh_device=mesh,
        state_dict=cp_sd, weight_cache_path=cp_cache,
    )

    # Check codec_embeddings
    print(f"\nCode Predictor has {len(cp.codec_embeddings)} codec embedding tables")
    for i, emb in enumerate(cp.codec_embeddings):
        print(f"  CB{i+1} embedding: {emb.shape}, norm(row0)={emb[0].norm():.4f}")

    # Test with a few CB0 tokens
    test_cb0s = [1995, 215, 1494, 1281]  # from HF generation above
    for cb0_val in test_cb0s:
        cb0_token = torch.tensor([cb0_val], dtype=torch.long)
        # Use a dummy hidden state (zeros)
        hidden = torch.zeros(1, 1, talker_args.dim)
        frame = cp.predict_codebooks(hidden, cb0_token, codec_embed_w)
        print(f"\nCB0={cb0_val} -> frame: {frame[0].tolist()}")
        print(f"  CB0={frame[0, 0].item()} (should be {cb0_val})")
        print(f"  CB1-15: min={frame[0, 1:].min().item()}, max={frame[0, 1:].max().item()}, "
              f"unique={len(set(frame[0, 1:].tolist()))}")

    # Test with random hidden state (more realistic)
    print("\n--- With random hidden state ---")
    for cb0_val in test_cb0s[:2]:
        cb0_token = torch.tensor([cb0_val], dtype=torch.long)
        hidden = torch.randn(1, 1, talker_args.dim) * 0.1  # Small norm
        frame = cp.predict_codebooks(hidden, cb0_token, codec_embed_w)
        print(f"CB0={cb0_val} -> CB1-15: {frame[0, 1:].tolist()}")
        print(f"  min={frame[0, 1:].min().item()}, max={frame[0, 1:].max().item()}")

    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + ":" + os.path.dirname(__file__) + "/../../../.."
    main()
