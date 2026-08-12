"""Why does factor 8 diverge? Test whether it is the T-padding, not the tile edge.

test_audio_minimax_h3.py blames the tile edge: "207 frames padding to 256 makes 256/8 = 32 exactly one
tile per shard, which is the obvious suspect". There is a simpler candidate that also explains why
factor 4 is fine and factor 8 is not.

`_upload_BCT` pads T up to a multiple of 32*factor, so T=207 becomes 256 either way -- t_pad = 49.
Spread over the shards:

  factor 4: 64 rows/chip. Rows 207..255 are pad, so chip 3 is 49/64 pad and chips 0-2 are clean.
            No shard is entirely padding.
  factor 8: 32 rows/chip. Rows 224..255 are pad, so **chip 7 is entirely padding** and chip 6 is
            17/32 pad.

`_set_tpad_tail(..., mode="replicate")` has to materialize "the real last row" into the pad region, and
on a fully-padding shard that row does not exist locally -- it lives on chip 6. If the implementation
reads its own local last row, chip 7 replicates a padding value and the tail of the waveform is wrong.

The discriminator: run T=256, where t_pad = 0 and no shard is padding at all, and T=207 where it is 49.

  T=256 clean, T=207 broken  -> the T-padding path is the bug; the tile edge is innocent
  both broken                -> the tile edge (or the halo) is the bug after all

Reference is factor=1 on the same mesh -- the unsharded graph, so any divergence is sharding's.
"""

import json
import os

import torch
from safetensors.torch import load_file

import ttnn
from models.tt_dit.models.audio_vae.minimax_h3.convert_minimax_h3_audio import convert_minimax_h3_audio_state_dict
from models.tt_dit.models.audio_vae.minimax_h3.decoder_minimax_h3_audio import MiniMaxH3AudioDecoder
from models.tt_dit.parallel.config import ParallelFactor
from models.tt_dit.parallel.manager import CCLManager

MESH_AXIS = int(os.environ.get("MESH_AXIS", "1"))
FACTOR = int(os.environ.get("T_FACTOR", "8"))
TS = [int(x) for x in os.environ.get("PROBE_TS", "207,256").split(",")]

wd = os.path.join(os.environ["MINIMAX_H3_DIFFUSERS_DIR"], "audio_vae")
cfg = {k: v for k, v in json.load(open(os.path.join(wd, "config.json"))).items() if not k.startswith("_")}
converted = convert_minimax_h3_audio_state_dict(load_file(os.path.join(wd, "diffusion_pytorch_model.safetensors")))


def build(d, pc, ccl):
    dec = MiniMaxH3AudioDecoder(
        latent_channels=cfg["latent_channels"],
        latent_dim=cfg["latent_dim"],
        decoder_dim=cfg["decoder_dim"],
        decoder_rates=tuple(cfg["decoder_rates"]),
        decoder_kernel_sizes=tuple(cfg["decoder_kernel_sizes"]),
        resblock_kernel_sizes=tuple(cfg["resblock_kernel_sizes"]),
        resblock_dilation_sizes=tuple(tuple(x) for x in cfg["resblock_dilation_sizes"]),
        mesh_device=d,
        parallel_config=pc,
        ccl_manager=ccl,
    )
    dec.load_torch_state_dict(converted, strict=False)
    return dec


def psnr(a, b):
    a, b = a.float(), b.float()
    mse = float(((a - b) ** 2).mean())
    if mse == 0:
        return float("inf")
    return 10.0 * float(torch.log10(a.abs().max() ** 2 / mse))


ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
d = ttnn.open_mesh_device(ttnn.MeshShape(4, 8), l1_small_size=65536)
try:
    ref_dec = build(d, None, None)
    ccl = CCLManager(d, num_links=1, topology=ttnn.Topology.Linear)
    sh_dec = build(d, ParallelFactor(factor=FACTOR, mesh_axis=MESH_AXIS), ccl)
    print(f"factor={FACTOR} axis={MESH_AXIS}; reference is factor=1 on the same mesh\n", flush=True)
    for T in TS:
        align = 32 * FACTOR
        t_pad = (align - T % align) % align
        padded = T + t_pad
        rows_per_chip = padded // FACTOR
        full_pad_chips = t_pad // rows_per_chip
        torch.manual_seed(2)
        lat = torch.randn(2, cfg["latent_channels"], T) * 0.1
        ref = torch.as_tensor(ref_dec(lat)).float()
        got = torch.as_tensor(sh_dec(lat)).float()
        p = psnr(ref, got)
        print(
            f"PROBE T={T:>4}  t_pad={t_pad:>3}  padded={padded:>4}  {rows_per_chip:>3} rows/chip  "
            f"fully-padding shards={full_pad_chips}  PSNR {p:>7.1f} dB  "
            f"{'OK' if p > 40 else 'DIVERGES'}",
            flush=True,
        )
finally:
    ttnn.close_mesh_device(d)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
