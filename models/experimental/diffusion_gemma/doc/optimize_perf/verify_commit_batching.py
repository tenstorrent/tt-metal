# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device verify for the batched commit-append (#47557).

Proves the opt-in batched commit (``tt/commit_batched.py``) is equivalent to the
baseline 256 sequential single-token decode-appends, AND measures the commit-step
speedup, on the real DiffusionGemma-26B-A4B backbone.

WHAT IT DOES (single model build, one prompt, one denoise block):
  1. build the TT model + prefill a prompt (writes the frozen prompt K/V);
  2. run ONE denoise block (argmax) to get the clean committed 256 tokens — WITHOUT
     committing (a capturing no-op ``commit_fn``), so the cache still holds only the
     prompt prefix;
  3. deep-clone every layer's K/V cache (sharing-preserving) — the clones get the
     BATCHED writes, the originals get the SEQUENTIAL writes, both starting from the
     identical pre-commit prefix;
  4. run the SEQUENTIAL commit into the originals (timed) and the BATCHED commit into
     the clones (timed);
  5. read the written region ``[start_pos : start_pos+canvas_len]`` of every layer's
     K and V from both, per device shard, and compare with PCC + max-abs-diff;
  6. print commit_ms(seq), commit_ms(batched), speedup, and per-layer / worst PCC.

EXIT: 0 iff every layer's K and V pass ``--pcc`` (default 0.997). See
``commit_batching.md`` for the bit-exactness argument this asserts (algebraic
equivalence; the residual delta is prefill-vs-decode kernel numerics, so this asserts
high PCC, not bit-identity).

*** DEVICE-OWNERSHIP NOTE ***
DO NOT run this while another agent owns the QB2 device — only one process may open
the mesh. Run it only when the device is free.

Run (when device is free):
  DG_CKPT=/path/to/diffusiongemma-26B-A4B-it \
    python models/experimental/diffusion_gemma/doc/optimize_perf/verify_commit_batching.py \
    --mesh 1x4 --num-layers 30 --max-seq-len 1024 --prompt "The capital of France is"

Notes:
  * Use a small ``--max-seq-len`` (e.g. 1024) and a short prompt: the A/B clones the
    full per-layer cache, so keep it small. ``start_pos + canvas_len`` must be
    ``<= max_seq_len``.
  * ``--num-layers 30`` is the real full-depth gate; a smaller count is a faster smoke
    that still exercises the per-layer write/attention mechanism.
  * ``--long-context`` inserts filler so ``start_pos > sliding_window - canvas_len`` to
    exercise the sliding-window mask edge (see the caveat in ``commit_batching.md``).
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


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation of two flattened tensors (matches the repo's PCC gate)."""
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    if a.numel() == 0:
        return 1.0
    if torch.allclose(a, b):
        return 1.0
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom == 0:
        return 1.0 if a.norm() == b.norm() else 0.0
    return float((a @ b) / denom)


def _region_shards(kv_cache, *, start_pos: int, canvas_len: int):
    """Read the ``[start_pos : start_pos+canvas_len]`` region of a mesh K/V cache.

    Returns ``(k_shards, v_shards)`` lists of host tensors, one per device shard (the
    KV cache is TP-sharded over KV heads across the mesh; comparing every shard covers
    all heads).
    """
    import ttnn

    k_cache, v_cache = kv_cache

    def read(cache):
        shards = ttnn.get_device_tensors(cache) if hasattr(ttnn, "get_device_tensors") else [cache]
        out = []
        for shard in shards:
            t = ttnn.to_torch(shard)
            out.append(t[:, :, start_pos : start_pos + canvas_len, :].clone().float())
        return out

    return read(k_cache), read(v_cache)


def _clone_caches_sharing_preserving(tt_kv_cache):
    """Clone every unique layer cache once; shared entries reuse the same clone.

    ``tt_model.tt_kv_cache`` has shared layers pointing at their source layer's cache
    (same object); this preserves that identity so KV-sharing behaves in the clones.
    """
    import ttnn

    by_id: dict[int, list] = {}
    clones = []
    for kv in tt_kv_cache:
        key = id(kv[0])
        if key not in by_id:
            by_id[key] = [ttnn.clone(kv[0]), ttnn.clone(kv[1])]
        clones.append(by_id[key])
    return clones, list(by_id.values())


def _timed_commit(commit_fn, tt_model, committed, *, start_pos, page_table):
    import ttnn

    ttnn.synchronize_device(tt_model.mesh_device)
    t0 = time.perf_counter()
    commit_fn(tt_model, committed, start_pos=start_pos, page_table=page_table)
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
    ap.add_argument("--pcc", type=float, default=0.997)
    ap.add_argument(
        "--long-context",
        action="store_true",
        help="pad the prompt so start_pos exceeds (sliding_window - canvas_len), exercising the sliding-window mask edge",
    )
    ap.add_argument(
        "--write-batch",
        type=int,
        default=1,
        help="batched-commit contiguous KV write granularity (1 = per-position, proven)",
    )
    args = ap.parse_args()

    checkpoint = os.environ.get("DG_CKPT")
    if not checkpoint:
        logger.error("set DG_CKPT to the diffusiongemma-26B-A4B-it checkpoint dir")
        return 2

    # Force the batched-write granularity requested (opt-in fast write is >1).
    os.environ["DG_COMMIT_WRITE_BATCH"] = str(args.write_batch)

    from models.experimental.diffusion_gemma.checkpoint import (
        build_tt_model_from_checkpoint_dir,
        text_generation_prefixes_for_layers,
    )
    from models.experimental.diffusion_gemma.config import DiffusionConfig
    from models.experimental.diffusion_gemma.tt.commit_batched import commit_canvas_tokens_batched
    from models.experimental.diffusion_gemma.tt.generate import (
        commit_canvas_tokens,
        denoise_and_commit_block,
        tokenize_prompt,
    )
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
        if args.long_context:
            # Repeat the prompt so the committed block sits past the sliding window.
            reps = max(1, (1024 // max(1, prompt_tokens.shape[1])) + 1)
            prompt_tokens = prompt_tokens.repeat(1, reps)
            logger.info(f"[long-context] padded prompt to {prompt_tokens.shape[1]} tokens")

        # Prefill + adapter + noise fns via the serving session (single-sequence).
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
        logger.info(f"[verify] prompt cache_len/start_pos={start_pos} canvas_len={config.canvas_length}")
        if start_pos + config.canvas_length > args.max_seq_len:
            logger.error(
                f"start_pos+canvas_len ({start_pos + config.canvas_length}) exceeds max_seq_len "
                f"({args.max_seq_len}); increase --max-seq-len or shorten the prompt"
            )
            return 2

        # One denoise block WITHOUT committing → the clean committed 256 tokens.
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

        # Clone the pre-commit caches: originals ← sequential, clones ← batched.
        orig_caches = tt_model.tt_kv_cache
        clone_caches, _unique = _clone_caches_sharing_preserving(orig_caches)

        # SEQUENTIAL commit into the originals.
        seq_ms = _timed_commit(commit_canvas_tokens, tt_model, committed, start_pos=start_pos, page_table=None)
        seq_region = [_region_shards(kv, start_pos=start_pos, canvas_len=config.canvas_length) for kv in orig_caches]

        # BATCHED commit into the clones (swap the model's cache list, then restore).
        tt_model.tt_kv_cache = clone_caches
        try:
            batched_ms = _timed_commit(
                commit_canvas_tokens_batched, tt_model, committed, start_pos=start_pos, page_table=None
            )
            batched_region = [
                _region_shards(kv, start_pos=start_pos, canvas_len=config.canvas_length) for kv in clone_caches
            ]
        finally:
            tt_model.tt_kv_cache = orig_caches

        # Compare per layer, per K/V, per device shard.
        worst_pcc = 1.0
        worst_where = None
        max_abs = 0.0
        failures = []
        for layer_idx, ((sk, sv), (bk, bv)) in enumerate(zip(seq_region, batched_region)):
            for name, s_shards, b_shards in (("K", sk, bk), ("V", sv, bv)):
                for dev, (s, b) in enumerate(zip(s_shards, b_shards)):
                    pcc = _pcc(s, b)
                    diff = float((s - b).abs().max())
                    max_abs = max(max_abs, diff)
                    if pcc < worst_pcc:
                        worst_pcc, worst_where = pcc, f"layer{layer_idx}.{name}.dev{dev}"
                    if pcc < args.pcc:
                        failures.append((layer_idx, name, dev, pcc, diff))

        print("=" * 72)
        print(
            f"commit_ms  sequential = {seq_ms:9.1f}   batched = {batched_ms:9.1f}   "
            f"speedup = {seq_ms / batched_ms:5.2f}x"
        )
        print(f"KV PCC     worst = {worst_pcc:.6f} @ {worst_where}   max_abs_diff = {max_abs:.4e}")
        print(f"threshold  pcc >= {args.pcc}")
        if failures:
            print(f"FAILURES ({len(failures)}):")
            for layer_idx, name, dev, pcc, diff in failures[:32]:
                print(f"  layer {layer_idx:2d} {name} dev{dev}: pcc={pcc:.6f} max_abs={diff:.4e}")
            print("RESULT: FAIL")
            return 1
        print("RESULT: PASS — batched commit KV matches the 256 sequential appends within PCC")
        return 0
    finally:
        _close_mesh(mesh_device)


if __name__ == "__main__":
    sys.exit(main())
