# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Stage-04: how many ethernet links a *decode* collective should use.

Stage 03 measured ``num_links=1`` at 0.4738 ms against two links' 0.4766 --
0.6%, which it called noise-level, and kept 2 links because prefill needs them
(1.84x at a 2 MB payload). Stage 04's first two runs agreed with that reading
(0.3% and inside spread). The final run of ``layer_levers.py``, against the
shipped path with persistent collective buffers, reads **0.4285 against
0.4334** -- 1.1%, which is 20x the leg-against-itself spread and no longer
dismissable.

So this probe settles it: 2 links against 1 link **for decode only**, six
passes with the leg order alternating so a position effect cannot be mistaken
for a link effect. Prefill is untouched in the 1-link leg, which is the whole
point -- ``all_reduce`` branches on the same ``S <= 32`` test
``_decode_ccl_buffers`` already uses.

Result: **0.42875 ms against 0.43400, 1.22%**, output bit-identical on all four
dies in every leg, and each configuration reads the same at both positions.

    python links_probe.py

Prints ``P|`` lines only.
"""
import statistics
import sys
import time

import torch

import ttnn

sys.path.insert(0, "/home/raahem/tt-metal")
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tests.reference import layer_state_dict, load_config
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt import functional_decoder as F
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt import multichip_decoder as MC
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt.weight_mapping import convert_layer_weights

CTX = 128
hf = load_config()
tw = convert_layer_weights(layer_state_dict(0), hf)

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=90_000_000, l1_small_size=32768)

_all_reduce = MC.all_reduce
_LINKS = {}


def links(n):
    """An all-reduce on ``n`` links, over a **cached** alternate context.

    Cached because the context owns the persistent collective buffers and those
    are allocated on the first call at each shape; a fresh context per call puts
    that allocation inside ``begin_trace_capture``, which raises and leaves the
    trace open."""

    def fn(x, ctx, n=n):
        c = _LINKS.get((id(ctx), n))
        if c is None:
            # ``decode_num_links`` is the field ``_links`` reads at decode, and
            # it must be set explicitly: setting ``num_links`` alone changes
            # only prefill. The first version of this probe set only
            # ``num_links``, so both legs ran on one link and the 1.2% it
            # published could not be reproduced -- see ``_links``.
            c = _LINKS[(id(ctx), n)] = MC.MeshContext(
                ctx.mesh, ctx.ccl, ctx.num_devices, n, ctx.topology, decode_num_links=n
            )
        return _all_reduce(x, c)

    return fn


try:
    cfg = MC.MeshDecoderConfig.from_hf(hf)
    ctx = MC.mesh_context(mesh)
    weights = MC.upload_multichip_weights(tw, mesh, cfg)
    cos, sin = F.build_rope_cache(hf, 1024, mesh)
    kv = MC.create_mesh_kv_cache(mesh, cfg, 1, 1024, block_size=32)

    torch.manual_seed(0)
    tok = ttnn.from_torch(
        torch.randn(1, 1, 1, hf.hidden_size) * 0.02,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    pos = ttnn.from_torch(
        torch.tensor([CTX - 1], dtype=torch.int32),
        dtype=ttnn.int32,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )

    # **Alternating leg order**, added when this probe was repaired.
    #
    # Review found that ``_links`` had stopped honouring an explicit
    # ``num_links=2``, so this probe could no longer tell its own legs apart.
    # Repairing ``_links`` and re-running reproduced the published 1.2% almost
    # exactly -- which is *suspicious* rather than reassuring, because the same
    # gap had appeared while the two legs were identical. Two explanations fit:
    # the lever is real, or the leg that runs first in each pass is simply
    # slower and the number was always an artifact.
    #
    # Swapping the order on even passes separates them, and the answer is
    # unambiguous: each configuration reads the same at both positions
    # (2 links 0.4342/0.4341/0.4340 at A and 0.4341/0.4337/0.4339 at B; 1 link
    # 0.4290/0.4288/0.4286 at A and 0.4291/0.4283/0.4287 at B). The gap follows
    # the link count, not the position. **The lever is real.**
    #
    # The old log predates ``_links`` -- it was produced while ``all_reduce``
    # read ``ctx.num_links`` directly, which did distinguish the legs. Adopting
    # ``NUM_LINKS_DECODE`` is what made the probe unable to re-run, so the
    # published figure was right and only its reproducibility was lost.
    A = ("decode num_links=2", links(2))
    B = ("decode num_links=1", links(1))
    reference = None
    for p in (1, 2, 3, 4, 5, 6):
        for i, (name, ar) in enumerate([A, B] if p % 2 else [B, A]):
            slot = "AB"[i]  # NB: not ``pos`` -- that name is the position tensor
            MC.all_reduce = MC.all_reduce_decode = ar

            def step():
                return MC.decoder_layer_decode_multichip(tok, weights, cfg, ctx, cos, sin, kv, pos, CTX - 1)

            try:
                out = step()
                ttnn.synchronize_device(mesh)
                got = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()
                agree = "reference" if reference is None else f"max|diff| {(got - reference).abs().max().item():.3e}"
                if reference is None:
                    reference = got
                tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                try:
                    step()
                finally:
                    ttnn.end_trace_capture(mesh, tid, cq_id=0)
                for _ in range(10):
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                samples = []
                for _ in range(100):
                    t0 = time.perf_counter()
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                    samples.append((time.perf_counter() - t0) * 1e3)
                ttnn.release_trace(mesh, tid)
                print(f"P|pass{p} pos{slot} {name:22s} {statistics.median(samples):.4f} ms   ({agree})", flush=True)
            except Exception as exc:
                print(f"P|pass{p} pos{slot} {name:22s} FAILED {str(exc)[:160]}", flush=True)
finally:
    MC.all_reduce = MC.all_reduce_decode = _all_reduce
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
print("P|done")
