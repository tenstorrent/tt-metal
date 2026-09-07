# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Does splitting Gemma 4's layers across pipeline stages compute the same thing?

The pipeline's numerics are otherwise unchecked. Gemma 4's context-parallel prefill deleted its CPU
reference in the same refactor that introduced it, so the surviving end-to-end test asserts liveness
and finiteness only -- and PP inherits exactly that. Everything the PP work added is therefore
capable of being silently wrong rather than loudly broken: the pre-norm stage boundary, the
sharding of the activation that crosses it, each rank's own RoPE and chunk metadata, and the
per-rank KV window.

This does not need a golden trace or an HF reference, because it does not ask "is Gemma 4 right?".
It asks the one question PP introduces: **splitting the layer stack must not change the answer.**
The same window of layers, over the same chunks, is run two ways on two different columns of the
galaxy:

    reference   submesh 0:  layers [0, N)                        -> hidden
    split       submesh 1:  layers [0, k)   -> host relay ->
                submesh 2:  layers [k, N)                        -> hidden

Both stages of the split use the same mesh mapper the D2D socket uses
(``[Shard(2), Replicate()]``, sequence CP-sharded and embedding whole), so a wrong hand-off layout
shows up here rather than as plausible-looking garbage at 256k. The two paths run identical ops in
an identical order -- the only difference is that the split makes a round trip through the host --
so anything below ~1.0 PCC is a real defect, not numerical drift.

What this does NOT cover, and needs the TP=4 comparison instead: the TP=1-only code paths, namely
the head-grouped ``concat_heads`` and the per-TP global KV head count. Both sides here are TP=1.

Nor does it cover a final norm: this runtime builds every rank with ``prefill_weights_only=True``
because the KV cache is the product, so no rank has a head to check. ``--last`` still exercises the
``is_last_rank`` path (``prefill_chunk`` returning ``None``, the persistent trace output), just not
a norm.

Usage (galaxy idle; run detached):
    python models/demos/gemma4/tests/perf/pp4/verify_pp_split.py --layers 17 --split 8 --chunks 3
"""

import argparse
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=17, help="window size N; both paths run layers [0, N)")
    ap.add_argument("--split", type=int, default=8, help="k: the split path runs [0,k) then [k,N)")
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--max-seq", type=int, default=32768)
    ap.add_argument("--chunks", type=int, default=3)
    ap.add_argument("--pcc", type=float, default=0.9999)
    ap.add_argument(
        "--last",
        action="store_true",
        help="build the reference and the split's SECOND stage as last ranks. That exercises the "
        "is_last_rank path -- prefill_chunk returns None because the KV cache is the product, so "
        "the hidden is read off the runtime's persistent trace output instead. It does NOT exercise "
        "a final norm: this runtime is KV-only (prefill_weights_only=True), so no rank has a head.",
    )
    args = ap.parse_args()

    import torch
    import ttnn
    from loguru import logger

    from models.common.utility_functions import comp_pcc
    from models.demos.common.prefill.adapter import PrefillRunParams, get_adapter
    from models.demos.common.prefill.runners.runner_utils import _create_fabric_router_config

    assert 0 < args.split < args.layers, "--split must be strictly inside the window"

    adapter = get_adapter("gemma4_31b")
    hf_config = adapter.load_hf_config()
    text_config = getattr(hf_config, "text_config", hf_config)
    types = list(text_config.layer_types)
    glob = lambda a, b: [i for i in range(a, b) if types[i] == "full_attention"]  # noqa: E731
    logger.warning(
        f"[verify] reference [0,{args.layers}) globals={glob(0, args.layers)} | "
        f"split [0,{args.split}) globals={glob(0, args.split)} + "
        f"[{args.split},{args.layers}) globals={glob(args.split, args.layers)}"
    )

    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_2D,
        ttnn.FabricReliabilityMode.RELAXED_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
        _create_fabric_router_config(max_payload_size=adapter.model_config.FABRIC_PAYLOAD_SIZE),
    )
    trace_region = int(os.environ.get("GEMMA4_PREFILL_TRACE_REGION_SIZE", 256 * 1024 * 1024))
    full = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4), trace_region_size=trace_region)
    rc = 1
    try:
        cols = full.create_submeshes(ttnn.MeshShape(8, 1))

        def build(mesh, first_layer, n_layers, is_first, is_last):
            params = PrefillRunParams(
                mesh_shape=(8, 1),
                num_layers=n_layers,
                first_layer_idx=first_layer,
                is_first_rank=is_first,
                is_last_rank=is_last,
                max_seq_len=args.max_seq,
                chunk_size=args.chunk,
                num_users=1,
                capacity_factor=8,
                num_links=2,
                gate_mode_name="DEVICE_FP32",
                kv_only_last_layer=False,
                weight_cache_path=adapter.weight_cache_path((8, 1)),
                use_trace=True,
            )
            rt = adapter.build_runtime(mesh_device=mesh, hf_config=hf_config, params=params)
            kv = adapter.allocate_kv_cache(mesh_device=mesh, hf_config=hf_config, params=params)
            rt.compile(kv)
            rt.capture_trace(kv)
            return rt, kv

        t0 = time.perf_counter()
        ref, ref_kv = build(cols[0], 0, args.layers, True, args.last)
        sa, sa_kv = build(cols[1], 0, args.split, True, False)
        sb, sb_kv = build(cols[2], args.split, args.layers - args.split, False, args.last)
        # No stage may build the final norm. Two independent reasons, and both must hold: a non-last
        # pipeline rank hands the RAW residual stream to the next stage's layer 0 (normalizing here
        # would apply the final norm once per boundary), AND this runtime builds every rank with
        # prefill_weights_only=True because the KV cache is the product -- so even the LAST rank has
        # no head. The two knobs are collapsed into Gemma4Model._build_head; assert on the outcome.
        for name, rt in (("reference", ref), ("split head", sa), ("split tail", sb)):
            assert rt.model.norm is None, f"{name} built a final norm; this service is KV-only"
            assert not rt.model._build_head, f"{name} thinks it should build a head"
        logger.warning(f"[verify] three runtimes built in {time.perf_counter() - t0:.1f}s")

        # Exactly the mapper the runner gives the D2D socket for Gemma 4
        # (pipeline_activation_emb_tp_sharded = False): sequence CP-sharded, embedding whole.
        d2d_mapper_cfg = ttnn.MeshMapperConfig(placements=[ttnn.PlacementShard(2), ttnn.PlacementReplicate()])

        def gather(tensor):
            """CP-sharded [1,1,chunk/cp,H] per device -> full [1,1,chunk,H] on host."""
            shards = ttnn.get_device_tensors(tensor)
            return torch.cat([ttnn.to_torch(s).float() for s in shards], dim=-2)

        torch.manual_seed(1234)
        vocab = int(text_config.vocab_size)
        verdicts = []
        for c in range(args.chunks):
            start = c * args.chunk
            tokens = torch.randint(0, vocab, (args.chunk,), dtype=torch.int32).tolist()
            common = dict(slot_id=0, actual_start=start, actual_end=start + args.chunk, request_id=c)

            out_ref = ref.prefill_chunk(ref.make_chunk_input(tokens), ref_kv, **common)
            host_ref = gather(out_ref if out_ref is not None else ref._trace_output)

            out_a = sa.prefill_chunk(sa.make_chunk_input(tokens), sa_kv, **common)
            relay = gather(out_a).to(torch.bfloat16)
            inp_b = ttnn.from_torch(
                relay,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=cols[2],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.create_mesh_mapper(cols[2], d2d_mapper_cfg),
            )
            out_b = sb.prefill_chunk(inp_b, sb_kv, **common)
            host_split = gather(out_b if out_b is not None else sb._trace_output)

            ok, msg = comp_pcc(host_ref, host_split, pcc=args.pcc)
            exact = bool(torch.equal(host_ref, host_split))
            logger.warning(
                f"[verify] chunk {c} [{start},{start + args.chunk}) {'PASS' if ok else 'FAIL'} "
                f"{msg} exact={exact} ref_std={float(host_ref.std()):.4f}"
            )
            verdicts.append(ok)
            assert torch.isfinite(host_ref).all() and torch.isfinite(host_split).all()
            # A degenerate output would make any PCC meaningless.
            assert float(host_ref.std()) > 1e-3, "reference output is degenerate; the check proves nothing"

        rc = 0 if all(verdicts) else 1
        logger.warning(
            f"[verify] RESULT {'PASS' if rc == 0 else 'FAIL'}: {sum(verdicts)}/{len(verdicts)} chunks "
            f"at PCC >= {args.pcc}  (layers [0,{args.layers}) split at {args.split}, "
            f"{'is_last_rank path exercised' if args.last else 'plain non-last stages'}, "
            f"pre-norm residual stream -- this runtime is KV-only)"
        )
        for rt in (ref, sa, sb):
            rt.release_trace()
    except Exception:
        logger.exception("[verify] FAILED")
    finally:
        ttnn.close_mesh_device(full)
    sys.exit(rc)


if __name__ == "__main__":
    sys.path.insert(0, os.environ.get("TT_METAL_HOME", "."))
    main()
