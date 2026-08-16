# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""In-model PCC against the HF reference, at every context the lever was swept at.

The op-level sweeps (``sdpa_sweep_probe.py``, ``sdpa_sweep_confirm.py``) score
the SDPA op against a float32 reference built from the same cache. That is the
right instrument for choosing between chunkings and the wrong one for deciding
whether the model is still correct -- stage 06 has now been bitten twice by
probes that could not see cache and state interaction. This probe runs the
**real multichip decoder layer**, with a **prefill-primed paged KV cache**,
against the **HF layer**, at the swept depths, with and without the adopted
program config in the same process.

It reuses ``tests/test_multichip_decoder.py``'s fixture, so the weights, the
mesh, the rope caches and the prefill are exactly the tested ones.

    python sdpa_hf_pcc_at_depth.py [--contexts 128 1024 4096]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

import ttnn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[4]))

from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tests import test_multichip_decoder as T  # noqa: E402
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt import multichip_decoder as MC  # noqa: E402
from models.common.utility_functions import comp_pcc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=int, nargs="+", default=[128, 1024, 4096])
    ap.add_argument("--out", default=str(HERE / "sdpa_hf_pcc_at_depth.json"))
    args = ap.parse_args()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*MC.MESH_SHAPE), trace_region_size=T.TRACE_REGION_SIZE)
    results = []
    try:
        reference = T.build_reference_layer(T.LAYER_IDX)
        layer, hf_config = reference
        weights = T.convert_layer_weights(T.layer_state_dict(T.LAYER_IDX), hf_config)
        fixture = T.Fixture(mesh, hf_config, weights)
        cfg = fixture.config
        # The fixture builds its rope cache to MAX_SEQ=1024; this probe decodes
        # well past that, so rebuild it to cover the deepest context asked for.
        fixture.cos, fixture.sin = T.F.build_rope_cache(hf_config, max(args.contexts), mesh)

        for ctx in args.contexts:
            # prompt = ctx-1 so the single decode step lands at position ctx-1
            prompt = ctx - 1
            full = T._hidden(hf_config, prompt + 1)
            ref = T._reference_layer(layer, hf_config, full)

            for leg in ("default", "adopted"):
                kv = MC.create_mesh_kv_cache(mesh, cfg, 1, ctx, block_size=T.BLOCK_SIZE)
                fixture.multichip_prefill(full[:, :, :prompt, :], kv_cache=kv)
                pos = ttnn.from_torch(
                    torch.tensor([prompt], dtype=torch.int32),
                    dtype=ttnn.int32,
                    device=mesh,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                )
                # The only difference between the legs: force the op default by
                # temporarily neutralising the config builder.
                real = MC._sdpa_program_config
                if leg == "default":
                    MC._sdpa_program_config = lambda device, kv_cache=None: (
                        None if (kv_cache is not None and kv_cache.is_paged) else real(device, kv_cache)
                    )
                try:
                    out = MC.decoder_layer_decode_multichip(
                        fixture.rep(full[:, :, prompt : prompt + 1, :]),
                        fixture.multichip,
                        cfg,
                        fixture.ctx,
                        fixture.cos,
                        fixture.sin,
                        kv,
                        pos,
                        prompt,
                    )
                finally:
                    MC._sdpa_program_config = real
                got = fixture.dies(out)[0].reshape(1, -1).float()
                _, message = comp_pcc(ref[:, prompt, :], got, 0.995)
                pcc = float(str(message).split("=")[-1].strip().rstrip(")"))
                k_chunk = MC._sdpa_k_chunk(kv) if leg == "adopted" else None
                print(
                    f"ctx {ctx:6d}  cur_pos {prompt:6d}  {leg:<8} " f"k_chunk={k_chunk}  PCC vs HF {pcc:.6f}",
                    flush=True,
                )
                results.append({"ctx": ctx, "cur_pos": prompt, "leg": leg, "k_chunk": k_chunk, "pcc_vs_hf": pcc})
                for t in (out, pos, kv.k, kv.v, kv.page_table):
                    ttnn.deallocate(t)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
