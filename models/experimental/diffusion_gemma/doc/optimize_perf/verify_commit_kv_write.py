# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device gate for the batched commit's ONE-OP KV write (#47557), on the real backbone.

``tt/commit_batched.py`` can append the committed canvas into the frozen contiguous KV
cache two ways:

  * ``position`` — the device-proven reference: per committed position, 2 slices + 2
    reshards + 2 ``paged_update_cache`` = ~1536 dispatches per layer. Measured at ~52% of
    the whole commit wall-clock, and almost pure host dispatch.
  * ``fill`` — the default: ONE ``ttnn.fill_cache`` per K/V at the tile-aligned
    ``update_idx=start_pos``. 2 dispatches per layer.

The write span is tile-aligned by construction (``start_pos % 32 == 0``,
``canvas_len = 256``), so FILL is a pure tile copy and the two must agree **bit-for-bit**.
``tests/test_device_commit_kv_write.py`` proves that at the op level on raw tensors; this
proves it on the real DiffusionGemma-26B-A4B backbone, through the full commit, and
measures the commit-step speedup.

WHAT IT DOES (single model build, one prompt, one denoise block):
  1. build the TT model + prefill a prompt (writes the frozen prompt K/V);
  2. run ONE denoise block (argmax) WITHOUT committing (capturing no-op ``commit_fn``),
     so the cache still holds only the prompt prefix;
  3. snapshot every layer's pre-commit K/V, then deep-clone the caches
     (sharing-preserving) so both write modes start from the identical prefix;
  4. run the BATCHED commit twice — ``write_mode="position"`` into the originals,
     ``write_mode="fill"`` into the clones — timing each;
  5. compare the FULL cache tensors (every position, every device shard): they must be
     bit-identical (max_abs_diff == 0.0), and the frozen prefix ``[0, start_pos)`` must
     equal the pre-commit snapshot in both;
  6. print commit_ms(position), commit_ms(fill) and the speedup.

Everything else in the commit (attention, MoE, norms) is identical between the two runs,
so any difference is the write — which is why this asserts bit-identity, not a PCC.

EXIT: 0 iff every layer's K and V are bit-identical between the modes AND both left the
frozen prefix untouched.

*** DEVICE-OWNERSHIP NOTE ***
DO NOT run this while another agent owns the QB2 device — only one process may open the
mesh. Run it only when the device is free.

Run (when device is free):
  DG_CKPT=/home/zni/dg_models/diffusiongemma-26B-A4B-it \
    python models/experimental/diffusion_gemma/doc/optimize_perf/verify_commit_kv_write.py \
    --mesh 1x4 --num-layers 30 --max-seq-len 1024 --prompt "The capital of France is"

Notes:
  * keep ``--max-seq-len`` small (e.g. 1024) and the prompt short: the A/B clones every
    layer's full cache. ``start_pos + canvas_len`` must be ``<= max_seq_len``.
  * ``--num-layers 30`` is the full-depth gate; fewer layers is a faster smoke that still
    exercises the per-layer write.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from loguru import logger
import torch


def _parse_mesh(mesh: str) -> tuple[int, int]:
    rows, cols = mesh.lower().split("x")
    return int(rows), int(cols)


def _open_mesh(mesh: str):
    import ttnn

    rows, cols = _parse_mesh(mesh)
    if rows * cols > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D, ttnn.FabricReliabilityMode.STRICT_INIT, None)
    return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(rows, cols))


def _close_mesh(mesh_device):
    import ttnn

    try:
        ttnn.close_mesh_device(mesh_device)
    finally:
        if mesh_device.get_num_devices() > 1:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _full_shards(kv_cache):
    """Read the WHOLE K and V cache, one host tensor per device shard.

    The full tensor (not just the written region) so a disturbed frozen prefix or tail
    fails the comparison too. The cache is TP-sharded over KV heads across the mesh, so
    every shard is compared.
    """
    import ttnn

    k_cache, v_cache = kv_cache

    def read(cache):
        shards = ttnn.get_device_tensors(cache) if hasattr(ttnn, "get_device_tensors") else [cache]
        return [ttnn.to_torch(shard).clone() for shard in shards]

    return read(k_cache), read(v_cache)


def _clone_caches_sharing_preserving(tt_kv_cache):
    """Clone every unique layer cache once; shared entries reuse the same clone."""
    import ttnn

    by_id: dict[int, list] = {}
    clones = []
    for kv in tt_kv_cache:
        key = id(kv[0])
        if key not in by_id:
            by_id[key] = [ttnn.clone(kv[0]), ttnn.clone(kv[1])]
        clones.append(by_id[key])
    return clones, list(by_id.values())


def _timed_commit(commit_fn, tt_model, committed, *, start_pos, write_mode):
    import ttnn

    ttnn.synchronize_device(tt_model.mesh_device)
    t0 = time.perf_counter()
    commit_fn(tt_model, committed, start_pos=start_pos, page_table=None, write_mode=write_mode)
    ttnn.synchronize_device(tt_model.mesh_device)
    return (time.perf_counter() - t0) * 1000.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mesh", default="1x4")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--num-layers", type=int, default=30)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--canvas-length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--no-warmup",
        action="store_true",
        help="skip the scratch-clone warm-up of both write modes (saves one cache clone's memory; timings then include cold-start cost)",
    )
    args = ap.parse_args()

    checkpoint = os.environ.get("DG_CKPT")
    if not checkpoint:
        logger.error("set DG_CKPT to the diffusiongemma-26B-A4B-it checkpoint dir")
        return 2

    from models.experimental.diffusion_gemma.checkpoint import (
        build_tt_model_from_checkpoint_dir,
        text_generation_prefixes_for_layers,
    )
    from models.experimental.diffusion_gemma.config import DiffusionConfig
    from models.experimental.diffusion_gemma.tt.commit_batched import commit_canvas_tokens_batched
    from models.experimental.diffusion_gemma.tt.generate import denoise_and_commit_block, tokenize_prompt
    from models.experimental.diffusion_gemma.tt.serving import BlockDiffusionServingSession

    mesh_device = _open_mesh(args.mesh)
    try:
        model_inputs = build_tt_model_from_checkpoint_dir(
            mesh_device,
            checkpoint,
            state_prefixes=text_generation_prefixes_for_layers(args.num_layers),
            num_layers=args.num_layers,
            max_seq_len=args.max_seq_len,
        )
        tt_model = model_inputs.tt_model
        config = DiffusionConfig(canvas_length=args.canvas_length)

        prompt_tokens = tokenize_prompt(model_inputs.tokenizer, args.prompt)
        session = BlockDiffusionServingSession(
            tt_model,
            model_inputs.state_dict,
            config=config,
            tokenizer=model_inputs.tokenizer,
            gumbel_mode="argmax",
            seed=args.seed,
        )
        session.prefill(prompt_tokens)
        start_pos = session.next_pos
        logger.info(f"[verify] start_pos={start_pos} canvas_len={config.canvas_length}")
        if start_pos + config.canvas_length > args.max_seq_len:
            logger.error(
                f"start_pos+canvas_len ({start_pos + config.canvas_length}) exceeds max_seq_len "
                f"({args.max_seq_len}); increase --max-seq-len or shorten the prompt"
            )
            return 2

        # One denoise block WITHOUT committing → the clean committed tokens.
        captured: dict = {}

        def _capture_noop(_tt_model, canvas_tokens, **_kw):
            captured["committed"] = canvas_tokens.clone()

        denoise_and_commit_block(
            tt_model,
            session._logits_fn,
            session._init_canvas_fn(0, start_pos),
            config,
            start_pos=start_pos,
            gumbel_noise_fn=session._gumbel_noise_fn(0),
            noise_tokens_fn=session._noise_tokens_fn(0),
            commit_fn=_capture_noop,
        )
        committed = captured["committed"]
        logger.info(f"[verify] captured committed tokens shape={tuple(committed.shape)}")

        orig_caches = tt_model.tt_kv_cache
        pre_commit = [_full_shards(kv) for kv in orig_caches]

        if not args.no_warmup:
            # Warm BOTH modes on a scratch clone first, then free it. Without this the
            # first timed commit absorbs the JIT / program-cache / allocator cold cost
            # and the A/B overstates whichever mode runs second-to-none.
            import ttnn

            warm_caches, warm_unique = _clone_caches_sharing_preserving(orig_caches)
            tt_model.tt_kv_cache = warm_caches
            try:
                for mode in ("position", "fill"):
                    commit_canvas_tokens_batched(
                        tt_model, committed, start_pos=start_pos, page_table=None, write_mode=mode
                    )
                ttnn.synchronize_device(tt_model.mesh_device)
            finally:
                tt_model.tt_kv_cache = orig_caches
            for kv in warm_unique:
                for t in kv:
                    t.deallocate(True)
            logger.info("[verify] warmed both write modes on a scratch cache clone")

        clone_caches, _unique = _clone_caches_sharing_preserving(orig_caches)

        # A: the proven per-position write into the originals.
        pos_ms = _timed_commit(
            commit_canvas_tokens_batched, tt_model, committed, start_pos=start_pos, write_mode="position"
        )
        pos_full = [_full_shards(kv) for kv in orig_caches]

        # B: the one-op fill write into the clones.
        tt_model.tt_kv_cache = clone_caches
        try:
            fill_ms = _timed_commit(
                commit_canvas_tokens_batched, tt_model, committed, start_pos=start_pos, write_mode="fill"
            )
            fill_full = [_full_shards(kv) for kv in clone_caches]
        finally:
            tt_model.tt_kv_cache = orig_caches

        failures = []
        max_abs = 0.0
        for layer_idx, ((pk, pv), (fk, fv), (ak, av)) in enumerate(zip(pos_full, fill_full, pre_commit)):
            for name, p_shards, f_shards, a_shards in (("K", pk, fk, ak), ("V", pv, fv, av)):
                for dev, (p, f, a) in enumerate(zip(p_shards, f_shards, a_shards)):
                    diff = float((p.float() - f.float()).abs().max()) if p.numel() else 0.0
                    max_abs = max(max_abs, diff)
                    if not torch.equal(p, f):
                        nwrong = int((p != f).sum())
                        failures.append((layer_idx, name, dev, f"{nwrong} elems differ, max_abs={diff:.4e}"))
                    # The frozen prefix must survive both writes untouched.
                    for mode_name, written in (("position", p), ("fill", f)):
                        if not torch.equal(written[:, :, :start_pos, :], a[:, :, :start_pos, :]):
                            failures.append(
                                (layer_idx, name, dev, f"{mode_name} disturbed the frozen prefix [0,{start_pos})")
                            )

        print("=" * 78)
        print(
            f"commit_ms  per-position = {pos_ms:9.1f}   one-op fill = {fill_ms:9.1f}   "
            f"speedup = {pos_ms / max(fill_ms, 1e-9):5.2f}x"
        )
        print(f"whole-cache max_abs_diff (position vs fill) = {max_abs:.4e}   (must be 0.0)")
        print(f"layers = {len(pos_full)}  shards/layer = {len(pos_full[0][0])}  start_pos = {start_pos}")
        if failures:
            print(f"FAILURES ({len(failures)}):")
            for layer_idx, name, dev, why in failures[:32]:
                print(f"  layer {layer_idx:2d} {name} dev{dev}: {why}")
            print("RESULT: FAIL")
            return 1
        print("RESULT: PASS — one-op fill KV write is bit-identical to the per-position write")
        return 0
    finally:
        _close_mesh(mesh_device)


if __name__ == "__main__":
    sys.exit(main())
