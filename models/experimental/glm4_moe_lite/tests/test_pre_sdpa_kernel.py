#!/usr/bin/env python3
"""Standalone test for PreSDPA fused kernel debug_stage binary search.

Run inside the dev container (after stopping vLLM):
    docker compose --env-file dev/.env.glm47 -f dev/docker-compose.yml run --rm \
        -e DEV_RUN_CMD=1 vllm-tt \
        python /tt-metal/models/experimental/glm4_moe_lite/tests/test_pre_sdpa_kernel.py --stage 0

This tests the kernel in isolation, avoiding warmup/flash MLA L1 conflicts.
"""
import argparse
import os
import time

os.environ.setdefault("TT_METAL_HOME", "/tt-metal")
os.environ.setdefault("LOGURU_LEVEL", "INFO")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=0, help="debug_stage: 0=noop, 1=rmsnorm, 2=mcast, ... 99=all")
    parser.add_argument("--layer", type=int, default=0, help="Which layer's weights to use")
    args = parser.parse_args()

    print(f"[TEST] PreSDPA kernel test: debug_stage={args.stage}, layer={args.layer}")

    import json
    import torch
    import ttnn
    from pathlib import Path
    from types import SimpleNamespace

    # Open mesh device
    print("[TEST] Opening mesh device...")
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 8),
        dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER),
    )
    mesh.enable_program_cache()
    print(f"[TEST] Mesh device opened: {mesh.get_num_devices()} devices")

    try:
        # Load HF config and create hparams
        snapshot_dir = Path(
            "/cache/huggingface/hub/models--zai-org--GLM-4.7-Flash/snapshots/7dd20894a642a0aa287e9827cb1a1f7f91386b67"
        )
        with open(snapshot_dir / "config.json") as f:
            raw = json.load(f)
        hf_config = SimpleNamespace(**raw)
        from models.experimental.glm4_moe_lite.tt.config import Glm4MoeLiteHParams

        hparams = Glm4MoeLiteHParams.from_hf_config(hf_config)
        print(f"[TEST] HParams loaded: hidden={hparams.hidden_size}, heads={hparams.num_attention_heads}")

        # Load lazy state dict
        from models.experimental.glm4_moe_lite.tt.weights import load_glm_lazy_state_dict

        state = load_glm_lazy_state_dict(snapshot_dir)
        print("[TEST] LazyStateDict loaded")

        # Prepare fused weights
        from models.experimental.glm4_moe_lite.tt.layer_weights import _prepare_fused_pre_sdpa_weights

        cache_dir = Path("/root/.cache/ttnn/models/glm4_moe_lite/vllm")

        print(f"[TEST] Preparing fused pre-SDPA weights for layer {args.layer}...")
        t0 = time.monotonic()
        fps = _prepare_fused_pre_sdpa_weights(
            device=mesh,
            state=state,
            layer_idx=args.layer,
            hparams=hparams,
            cache_dir=cache_dir,
            dense_dtype=ttnn.bfloat8_b,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        t1 = time.monotonic()
        print(f"[TEST] Fused weights prepared in {t1 - t0:.1f}s")
        print(f"[TEST] Weight keys: {sorted(fps.keys())}")

        # Count L1 usage
        l1_bytes = 0
        for k, v in fps.items():
            if hasattr(v, "volume") and hasattr(v, "element_size"):
                try:
                    l1_bytes += v.volume() * v.element_size()
                except Exception:
                    pass
        print(f"[TEST] Approximate L1 usage: {l1_bytes:,} bytes ({l1_bytes / 1024:.1f} KB)")

        # Prepare input tensor
        hidden = int(hparams.hidden_size)
        x_torch = torch.randn(1, hidden, dtype=torch.bfloat16)
        x_for_fused = ttnn.from_torch(
            x_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=fps["input_mem_config"],
            tile=ttnn.Tile((1, 32)),
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        print(f"[TEST] Input tensor created")

        # Run the kernel
        from models.experimental.glm4_moe_lite.fused_ops.pre_sdpa.op import PreSDPA

        print(f"\n[TEST] === Calling PreSDPA.op(debug_stage={args.stage}) ===")
        t0 = time.monotonic()
        sdpa_output = PreSDPA.op(
            x_for_fused,
            fps["intermediate_tensor"],
            fps["gamma"],
            fps["matmul_weights"],
            fps["rmsnorm2_gamma"],
            fps["matmul2_weights"],
            fps["matmul3_weights"],
            fps["sin"],
            fps["cos"],
            fps["trans_mat"],
            fps["krope_cos"],
            fps["krope_sin"],
            fps["dkv_matmul_weights"],
            fps["dkv_rmsnorm_gamma"],
            fps["output_tensor"],
            fps["sender_coord"],
            semaphores=None,
            cluster_axis=0,
            secondary_cluster_axis=1,
            using_persistent_buffers=True,
            epsilon=fps["epsilon"],
            fp32_dest_acc_en=True,
            skip_ccl=True,
            debug_stage=args.stage,
        )
        t1 = time.monotonic()
        print(f"[TEST] PreSDPA.op() returned in {(t1 - t0)*1000:.1f}ms", flush=True)

        # Sync device
        print("[TEST] Syncing device...", flush=True)
        t0 = time.monotonic()
        ttnn.synchronize_device(mesh)
        t1 = time.monotonic()
        print(f"[TEST] Device sync OK in {(t1 - t0)*1000:.1f}ms", flush=True)

        # Second call (should use program cache)
        print(f"\n[TEST] === Second call (cached) ===", flush=True)
        t0 = time.monotonic()
        sdpa_output2 = PreSDPA.op(
            x_for_fused,
            fps["intermediate_tensor"],
            fps["gamma"],
            fps["matmul_weights"],
            fps["rmsnorm2_gamma"],
            fps["matmul2_weights"],
            fps["matmul3_weights"],
            fps["sin"],
            fps["cos"],
            fps["trans_mat"],
            fps["krope_cos"],
            fps["krope_sin"],
            fps["dkv_matmul_weights"],
            fps["dkv_rmsnorm_gamma"],
            fps["output_tensor"],
            fps["sender_coord"],
            semaphores=None,
            cluster_axis=0,
            secondary_cluster_axis=1,
            using_persistent_buffers=True,
            epsilon=fps["epsilon"],
            fp32_dest_acc_en=True,
            skip_ccl=True,
            debug_stage=args.stage,
        )
        t1 = time.monotonic()
        print(f"[TEST] Second call returned in {(t1 - t0)*1000:.1f}ms", flush=True)
        print("[TEST] Syncing device (2nd)...", flush=True)
        ttnn.synchronize_device(mesh)
        print("[TEST] Second sync OK", flush=True)

        # Cleanup
        ttnn.deallocate(sdpa_output, force=True)
        ttnn.deallocate(sdpa_output2, force=True)
        ttnn.deallocate(x_for_fused, force=True)
        for k, v in list(fps.items()):
            if hasattr(v, "is_allocated") and callable(v.is_allocated):
                try:
                    ttnn.deallocate(v, force=True)
                except Exception:
                    pass

        print(f"\n[TEST] *** SUCCESS: debug_stage={args.stage} completed without hang! ***", flush=True)

    finally:
        print("[TEST] Closing mesh device...", flush=True)
        ttnn.close_mesh_device(mesh)
        print("[TEST] Done.", flush=True)


if __name__ == "__main__":
    main()
