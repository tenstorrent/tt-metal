# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Numerical drift between traced and untraced `Pi0_5ModelTTNN.sample_actions`.

Locks all inputs (images, masks, tokens, lang_mask) and writes a fixed
noise into model.x_t_ttnn so that any difference between the two paths
must come from the trace machinery itself (memory aliasing, layout
alignment, etc.) rather than per-call randomness.

Run:
    TT_METAL_HOME=/home/tt-admin/sdawle/pi0/tt-metal TT_VISIBLE_DEVICES=0 \
      QWEN_NLP_CONCAT_HEADS_HEAD_SPLIT=1 QWEN_NLP_CREATE_HEADS_HEAD_SPLIT=1 \
      PYTHONPATH=/home/tt-admin/sdawle/pi0/tt-metal \
      python_env/bin/python models/experimental/pi0_5/tests/pcc/test_trace_vs_untraced_drift.py
"""

import sys
import types as _types

import torch

_fake = _types.ModuleType("transformers.models.siglip.check")
_fake.check_whether_transformers_replace_is_installed_correctly = lambda: True
sys.modules["transformers.models.siglip.check"] = _fake

REPO_ROOT = "/home/tt-admin/sdawle/pi0/tt-metal"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _build_fixed_noise(model):
    """Match Pi0_5ModelTTNN's internal noise build (1, ah_padded, action_dim).
    Seed deterministically so each call returns the same noise.
    """
    ah = model.config.action_horizon
    ah_padded = model._action_horizon_padded
    g = torch.Generator().manual_seed(42)
    noise_padded = torch.zeros(1, ah_padded, model.config.action_dim, dtype=torch.float32)
    noise_padded[:, :ah, :] = torch.randn(1, ah, model.config.action_dim, generator=g)
    return noise_padded


def main():
    import ttnn
    from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
    from models.experimental.pi0_5.tt.ttnn_pi0_5_model import Pi0_5ModelTTNN

    CHECKPOINT = "/storage/sdawle/pi05_weights/pi05_libero_finetuned"

    print("[setup] opening device...")
    device = ttnn.open_device(
        device_id=0,
        l1_small_size=24576,
        trace_region_size=134_217_728,
    )
    print("[setup] loading model...")
    cfg = Pi0_5ModelConfig(action_horizon=50)
    loader = Pi0_5WeightLoader(CHECKPOINT)
    model = Pi0_5ModelTTNN(cfg, loader, device)
    print("[setup] model loaded")

    # Build identical inputs once.
    torch.manual_seed(0)
    img = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    images = [img, img.clone(), torch.zeros_like(img)]
    img_masks = [
        torch.ones(1, dtype=torch.bool),
        torch.ones(1, dtype=torch.bool),
        torch.zeros(1, dtype=torch.bool),
    ]
    LANG_SEQ = 256
    tokens = torch.randint(0, 256000, (1, LANG_SEQ), dtype=torch.int32)
    lang_mask = torch.ones(1, LANG_SEQ, dtype=torch.bool)
    fixed_noise = _build_fixed_noise(model)

    def upload_inputs():
        """Fresh upload of identical inputs."""
        images_ttnn = [
            ttnn.from_torch(
                i, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            for i in images
        ]
        img_masks_ttnn = [
            ttnn.from_torch(
                m.float(),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            for m in img_masks
        ]
        tokens_ttnn = ttnn.from_torch(
            tokens.to(torch.uint32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        lang_mask_ttnn = ttnn.from_torch(
            lang_mask.to(torch.float32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        return images_ttnn, img_masks_ttnn, tokens_ttnn, lang_mask_ttnn

    def write_noise():
        """Overwrite model.x_t_ttnn with fixed_noise via copy_host_to_device."""
        noise_host = ttnn.from_torch(fixed_noise, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        ttnn.copy_host_to_device_tensor(noise_host, model.x_t_ttnn)
        return noise_host

    # === UNTRACED: run sample_actions twice with same inputs + same noise ===
    print("\n=== UNTRACED PATH ===")
    model.resample_noise = False
    untraced_outputs = []
    for trial in range(2):
        images_ttnn, img_masks_ttnn, tokens_ttnn, lang_mask_ttnn = upload_inputs()
        noise_host = write_noise()  # noqa: F841, kept alive
        with torch.no_grad():
            out = model.sample_actions(
                images=images_ttnn,
                img_masks=img_masks_ttnn,
                lang_tokens=tokens_ttnn,
                lang_masks=lang_mask_ttnn,
                state=None,
            )
        ttnn.synchronize_device(device)
        untraced_outputs.append(ttnn.to_torch(out).float())
        print(
            f"  [untraced trial {trial}] shape={tuple(untraced_outputs[-1].shape)} "
            f"mean={untraced_outputs[-1].mean().item():+.5f} "
            f"std={untraced_outputs[-1].std().item():.5f}"
        )
        for t in images_ttnn:
            ttnn.deallocate(t)
        for t in img_masks_ttnn:
            ttnn.deallocate(t)
        ttnn.deallocate(tokens_ttnn)
        ttnn.deallocate(lang_mask_ttnn)
        ttnn.deallocate(out)

    # Determinism check: untraced twice should match.
    d = (untraced_outputs[0] - untraced_outputs[1]).abs()
    print(f"  [untraced determinism] max={d.max().item():.3e}  mean={d.mean().item():.3e}")

    # === TRACED: capture once, replay twice with same inputs + same noise ===
    print("\n=== TRACED PATH ===")
    # Use the same upload functions; but the trace expects persistent buffers.
    # Build them once, write identical data, capture, then replay.
    images_ttnn, img_masks_ttnn, tokens_ttnn, lang_mask_ttnn = upload_inputs()
    noise_host = write_noise()

    # Warmup (one untraced pass to JIT compile).
    with torch.no_grad():
        warm = model.sample_actions(
            images=images_ttnn,
            img_masks=img_masks_ttnn,
            lang_tokens=tokens_ttnn,
            lang_masks=lang_mask_ttnn,
            state=None,
        )
    ttnn.synchronize_device(device)
    ttnn.deallocate(warm)

    # Re-write the noise (warmup consumed it numerically but state-wise nothing
    # changed in model.x_t_ttnn — values still match; this is a safety belt).
    noise_host = write_noise()

    # Capture
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    trace_out = model.sample_actions(
        images=images_ttnn,
        img_masks=img_masks_ttnn,
        lang_tokens=tokens_ttnn,
        lang_masks=lang_mask_ttnn,
        state=None,
    )
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)

    # Read capture-time result (the trace just ran sample_actions, so this is
    # the first traced output)
    traced_outputs = [ttnn.to_torch(trace_out).float()]
    print(
        f"  [traced capture]      shape={tuple(traced_outputs[-1].shape)} "
        f"mean={traced_outputs[-1].mean().item():+.5f} "
        f"std={traced_outputs[-1].std().item():.5f}"
    )

    # Replay twice
    for trial in range(2):
        # Refresh noise to the same fixed value
        noise_host = write_noise()  # noqa: F841
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        traced_outputs.append(ttnn.to_torch(trace_out).float())
        print(
            f"  [traced replay {trial}]    shape={tuple(traced_outputs[-1].shape)} "
            f"mean={traced_outputs[-1].mean().item():+.5f} "
            f"std={traced_outputs[-1].std().item():.5f}"
        )

    # === COMPARE ===
    print("\n=== COMPARISONS ===")
    ref = untraced_outputs[0]
    print(f"reference = untraced trial 0, shape={tuple(ref.shape)}")
    for label, t in (
        ("untraced trial 1", untraced_outputs[1]),
        ("traced capture", traced_outputs[0]),
        ("traced replay 0", traced_outputs[1]),
        ("traced replay 1", traced_outputs[2]),
    ):
        d = (ref - t).abs()
        print(
            f"  ref vs {label:<24} max={d.max().item():.4e}  mean={d.mean().item():.4e}  "
            f"cosim={(ref.flatten()*t.flatten()).sum().item() / (ref.flatten().norm().item()*t.flatten().norm().item() + 1e-12):.6f}"
        )

    # === VARYING-INPUTS COMPARISON ===
    # Simulates the LIBERO rollout: images, tokens, lang_mask, and noise all
    # differ between chunks. Each "chunk" must produce identical actions
    # whether we go through untraced sample_actions or traced replay.
    print("\n=== VARYING-INPUTS TEST (mimics LIBERO rollout) ===")

    # Generate 5 distinct chunks worth of inputs
    chunk_inputs = []
    for c in range(5):
        torch.manual_seed(100 + c)
        chunk_inputs.append(
            dict(
                images=[
                    torch.randn(1, 3, 224, 224, dtype=torch.float32),
                    torch.randn(1, 3, 224, 224, dtype=torch.float32),
                    torch.zeros(1, 3, 224, 224, dtype=torch.float32),
                ],
                tokens=torch.randint(0, 256000, (1, LANG_SEQ), dtype=torch.int32),
                noise=torch.randn(1, model._action_horizon_padded, model.config.action_dim, dtype=torch.float32),
            )
        )

    def upload_chunk(chunk):
        """Fresh upload of chunk inputs (untraced path style)."""
        ims = [
            ttnn.from_torch(
                i, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            for i in chunk["images"]
        ]
        ims_m = [
            ttnn.from_torch(
                m.float(),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            for m in img_masks
        ]
        tks = ttnn.from_torch(
            chunk["tokens"].to(torch.uint32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        lm = ttnn.from_torch(lang_mask.to(torch.float32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        return ims, ims_m, tks, lm

    def write_noise_value(noise_torch):
        noise_host = ttnn.from_torch(noise_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        ttnn.copy_host_to_device_tensor(noise_host, model.x_t_ttnn)
        return noise_host

    # Untraced: run each chunk fresh
    print("\n  -- untraced per chunk --")
    untraced_per_chunk = []
    for c, chunk in enumerate(chunk_inputs):
        ims, ims_m, tks, lm = upload_chunk(chunk)
        _ = write_noise_value(chunk["noise"])
        with torch.no_grad():
            o = model.sample_actions(images=ims, img_masks=ims_m, lang_tokens=tks, lang_masks=lm, state=None)
        ttnn.synchronize_device(device)
        untraced_per_chunk.append(ttnn.to_torch(o).float())
        for t in ims:
            ttnn.deallocate(t)
        for t in ims_m:
            ttnn.deallocate(t)
        ttnn.deallocate(tks)
        ttnn.deallocate(lm)
        ttnn.deallocate(o)

    # Traced: write each chunk via copy_host_to_device_tensor + execute_trace
    # (reusing the persistent buffers built for the capture above).
    print("  -- traced per chunk (copy_host_to_device + execute_trace) --")
    traced_per_chunk = []
    for c, chunk in enumerate(chunk_inputs):
        # Write each image into the persistent device buffer
        for i, img in enumerate(chunk["images"]):
            host_t = ttnn.from_torch(img, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            ttnn.copy_host_to_device_tensor(host_t, images_ttnn[i])
        # Tokens
        host_t = ttnn.from_torch(chunk["tokens"].to(torch.uint32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(host_t, tokens_ttnn)
        # Noise
        _ = write_noise_value(chunk["noise"])
        # Replay
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        traced_per_chunk.append(ttnn.to_torch(trace_out).float())

    print("\n  per-chunk diff (untraced vs traced with same inputs):")
    for c, (u, t) in enumerate(zip(untraced_per_chunk, traced_per_chunk)):
        d = (u - t).abs()
        cs = (u.flatten() * t.flatten()).sum().item() / (u.flatten().norm().item() * t.flatten().norm().item() + 1e-12)
        print(
            f"    chunk {c}: untraced.std={u.std().item():.4f}  traced.std={t.std().item():.4f}  "
            f"max_diff={d.max().item():.4e}  mean_diff={d.mean().item():.4e}  cosim={cs:.6f}"
        )

    # Free
    ttnn.release_trace(device, tid)
    for t in images_ttnn:
        ttnn.deallocate(t)
    for t in img_masks_ttnn:
        ttnn.deallocate(t)
    ttnn.deallocate(tokens_ttnn)
    ttnn.deallocate(lang_mask_ttnn)
    ttnn.close_device(device)


if __name__ == "__main__":
    main()
