"""A/B a full DiffVAE decode on the SHIPPED checkpoint: standard NA vs Generalized NA.

The stage-5 parity tests answer this only for synthetic weights (``randomize(seed=1234)``), where an
untrained adaLN gate can make attention contribute far less to the residual stream than it does in the
real network -- so a stride looks cheaper there than it may be. This runs real weights instead, and it
does not need the golden capture: the NA decode IS the reference, since GNA's whole deviation is that
queries share a window. The reported PCC is therefore exactly what a stride costs at the pixels.

One decoder, one latent, one noise seed, two decodes with ``DIFFVAE_GNA`` flipped between them -- na3d
resolves the stride per call, so both arms share weights and inputs bit-for-bit and the only difference
is which keys each query attends.

Requires the block-permuted fused path (``DIFFVAE_BLOCK`` + ``DIFFVAE_SP_FUSED``), the only path that
derives a stride from the Q block. ``_STRIDES_SEEN`` asserts the GNA arm actually got one, so a silent
fallback to stride 1 fails loudly instead of reporting a free 1.00x at PCC 1.0.

Env: ``DIFFVAE_LATENT_T`` (19 -> 145 frames, the 6s target), ``LATENT_HW`` ("34,60" = 1080p),
``GNA_STRIDE`` ("t,h,w") to force an explicit stride instead of the Q block. The parallelism flags
(``DIFFVAE_TP_HEADS``, ``DIFFVAE_STAGES_WSP``, ``DIFFVAE_NUM_LINKS``, ``DIFFVAE_TOPOLOGY``) are read the
same way ``test_decode_wsp_timing`` reads them, so a timing configuration can be quality-checked as-is.

``DIFFVAE_SLAB_FRAMES`` matters more here than it looks: the Q block is picked from a band's padded frame
count, so the band size decides how many queries share a window and therefore how aggressive the stride
is. Quality is only meaningful alongside the band size that produced it.
"""

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import time
from pathlib import Path

import torch

import ttnn
from models.tt_dit.models.vae.diffvae_ltx import DiffVAEDecoder, decoder_config
from models.tt_dit.parallel.manager import CCLManager

CHECKPOINT = Path(
    os.environ.get(
        "DIFFVAE_CHECKPOINT",
        os.path.expanduser("~/.cache/ltx-checkpoints/ltx-2.5/vae/ltx-2.5-video-vae-bf16.safetensors"),
    )
)

os.environ.setdefault("DIFFVAE_BLOCK", "1")
os.environ.setdefault("DIFFVAE_SP_FUSED", "1")

_STRIDES_SEEN: set[tuple[int, ...] | None] = set()


def _install_probe() -> None:
    inner = ttnn.transformer.scaled_dot_product_attention

    def probed(*args, **kwargs):
        stride = kwargs.get("neighborhood_stride")
        _STRIDES_SEEN.add(tuple(stride) if stride is not None else None)
        return inner(*args, **kwargs)

    ttnn.transformer.scaled_dot_product_attention = probed


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    x, y = a.flatten().double(), b.flatten().double()
    x, y = x - x.mean(), y - y.mean()
    denom = (x.norm() * y.norm()).item()
    return 1.0 if denom == 0.0 else (x @ y).item() / denom


def main() -> None:
    assert CHECKPOINT.exists(), f"missing checkpoint {CHECKPOINT}"
    lh, lw = (int(v) for v in os.environ.get("LATENT_HW", "34,60").split(","))
    t_lat = int(os.environ.get("DIFFVAE_LATENT_T", 19))

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(4, 8))
    try:
        config = decoder_config(CHECKPOINT)
        torch.manual_seed(0)
        latent = torch.randn(1, config["in_channels"], t_lat, lh, lw)

        # Mirrors test_decode_wsp_timing's parallelism knobs. Quality has to be read on the configuration
        # that produced the timing, not a nearby one: head TP and the W-sharded det stages change how the
        # volume is split, and a stride interacts with a split -- a group whose members land on different
        # chips is a different attention from one that does not.
        ring = os.environ.get("DIFFVAE_TOPOLOGY", "linear").lower() == "ring"
        ccl = CCLManager(
            mesh,
            num_links=int(os.environ.get("DIFFVAE_NUM_LINKS", 1)),
            topology=ttnn.Topology.Ring if ring else ttnn.Topology.Linear,
        )
        tp_axis = 0 if os.environ.get("DIFFVAE_TP_HEADS") == "1" else None
        stages_wsp = os.environ.get("DIFFVAE_STAGES_WSP") == "1"
        dec = DiffVAEDecoder(
            config,
            mesh_device=mesh,
            ccl_manager=ccl,
            stage5_na3d_backend="op_sp_w_sharded",
            stage5_sp_axis=1,
            stage5_tp_axis=tp_axis,
            stages_na3d_backend="op_sp_w_sharded" if stages_wsp else None,
            stages_sp_axis=1 if stages_wsp else None,
            stages_tp_axis=tp_axis if stages_wsp else None,
        )
        dec.load_checkpoint(CHECKPOINT)
        _install_probe()

        stride_env = os.environ.get("GNA_STRIDE")
        arms = [("NA (stride 1)", {"DIFFVAE_GNA": "0", "DIFFVAE_GNA_STRIDE": ""})]
        if stride_env:
            arms.append((f"GNA (stride {stride_env})", {"DIFFVAE_GNA": "0", "DIFFVAE_GNA_STRIDE": stride_env}))
        else:
            arms.append(("GNA (stride = Q block)", {"DIFFVAE_GNA": "1", "DIFFVAE_GNA_STRIDE": ""}))

        print(f"\n=== decode A/B · latent(1,{config['in_channels']},{t_lat},{lh},{lw}) · 4x8 SPxTP ===", flush=True)

        results = []
        for name, env in arms:
            for key, val in env.items():
                if val == "":
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = val

            dec.decode(latent, seed=0)  # warmup: program cache, and the stride is part of its key
            ttnn.synchronize_device(mesh)
            _STRIDES_SEEN.clear()
            t0 = time.perf_counter()
            pixels = dec.decode(latent, seed=0)
            ttnn.synchronize_device(mesh)
            dt = (time.perf_counter() - t0) * 1000
            strides = sorted(_STRIDES_SEEN, key=str)
            results.append((name, dt, pixels.float(), strides))
            print(f"[{name:24s}] {dt:8.0f} ms   -> {tuple(pixels.shape)}   op saw {strides}", flush=True)

        (na_name, na_ms, na_px, na_strides), (gna_name, gna_ms, gna_px, gna_strides) = results
        assert na_strides == [None], f"NA arm should run at no stride, saw {na_strides}"
        assert gna_strides != [None], (
            "GNA arm ran at stride 1 -- no legal block for this geometry, so this compares NA to itself. "
            "Pass GNA_STRIDE=t,h,w explicitly."
        )

        rms = na_px.pow(2).mean().sqrt().item()
        delta = na_px - gna_px
        print(
            f"\n{na_name:26s} {na_ms:9.0f} ms\n"
            f"{gna_name:26s} {gna_ms:9.0f} ms   {na_ms / gna_ms:.2f}x\n"
            f"\n[quality] REAL weights, NA decode is the reference\n"
            f"[quality] PCC          = {_pcc(na_px, gna_px):.6f}\n"
            f"[quality] rel-rms      = {delta.pow(2).mean().sqrt().item() / rms:.6f}\n"
            f"[quality] max|delta|   = {delta.abs().max().item():.6f}   rms(NA) = {rms:.6f}",
            flush=True,
        )
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
