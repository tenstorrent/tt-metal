"""Which stage does T-sharding first diverge at? Bisect the real forward, stage by stage.

T-sharding returns wrong audio at every factor (-10.1 dB at 4, -11.0 dB at 8) and the failure does not
look like boundary math: shards 1+ saturate at the closing tanh/clamp while shard 0 differs, the error
is spread through the interior (boundary/interior ratio 1.23), and correlation is 0.04 at lag 0. So
rather than guess, compare a sharded and an unsharded run of the *same* graph after each stage and find
the first one that differs.

Run at T=256 so `t_pad = 0` and the T-padding fault (separately worth 22 dB) is out of the picture --
this chases one bug, not two.

Instruments the real modules by wrapping their bound `forward` (Module.__call__ dispatches through
`self.forward`, and Module.__setattr__ only special-cases Module/Parameter values, so a plain function
assigns cleanly). Snapshots to host inside the hook because `_forward_device` deallocates its
intermediates -- reading them after the fact would read freed memory.
"""

import json
import os

import torch
from safetensors.torch import load_file

import ttnn
from models.tt_dit.layers.audio_ops import _partition_t
from models.tt_dit.models.audio_vae.minimax_h3.convert_minimax_h3_audio import convert_minimax_h3_audio_state_dict
from models.tt_dit.models.audio_vae.minimax_h3.decoder_minimax_h3_audio import MiniMaxH3AudioDecoder
from models.tt_dit.parallel.config import ParallelFactor
from models.tt_dit.parallel.manager import CCLManager

FACTOR = int(os.environ.get("T_FACTOR", "8"))
MESH_AXIS = int(os.environ.get("MESH_AXIS", "1"))
T = int(os.environ.get("NUM_T", "256"))

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
    if a.shape != b.shape:
        return None
    mse = float(((a - b) ** 2).mean())
    if mse == 0:
        return float("inf")
    return 10.0 * float(torch.log10(a.abs().max() ** 2 / mse))


def shard_order(d, pc):
    """Which get_device_tensors indices hold T-slices 0..factor-1, in order.

    Determined empirically rather than assumed: a 4x8 mesh replicates along the axis that is not being
    sharded, so the flat device list contains each T-slice `32/factor` times and the stride depends on
    whether the list is row- or column-major. Partition a ramp and read off where each slice landed.
    """
    ramp = torch.arange(T, dtype=torch.float32).reshape(1, T, 1).expand(1, T, 32).contiguous()
    t = ttnn.from_torch(ramp, device=d, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32)
    t = ttnn.to_layout(t, ttnn.TILE_LAYOUT)
    t = _partition_t(t, pc)
    t = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
    firsts = []
    for i, sh in enumerate(ttnn.get_device_tensors(t)):
        v = float(ttnn.to_torch(sh)[0, 0, 0])
        firsts.append((i, v))
    per = T // FACTOR
    order = []
    for slice_idx in range(FACTOR):
        want = slice_idx * per
        hit = [i for i, v in firsts if abs(v - want) < 0.5]
        assert hit, f"no device holds T-slice starting at {want}; firsts={firsts[:12]}"
        order.append(hit[0])
    dup = {slice_idx: [i for i, v in firsts if abs(v - slice_idx * per) < 0.5] for slice_idx in range(FACTOR)}
    print(f"shard order (device index per T-slice): {order}")
    print(f"replication groups: {[len(v) for v in dup.values()]} (expect {32 // FACTOR} each)")
    return order


ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
d = ttnn.open_mesh_device(ttnn.MeshShape(4, 8), l1_small_size=65536)
try:
    STAGES = ["conv_pre"] + [f"ups{i}" for i in range(7)] + ["act_post", "conv_post"]

    def instrument(dec, store, take):
        """Wrap each stage's forward to snapshot its output to host. `take` maps a device tensor to
        the full-T torch tensor for this configuration."""
        voc = dec.decoder

        def wrap(mod, name):
            orig = mod.forward

            def wrapped(*a, **kw):
                out = orig(*a, **kw)
                try:
                    store[name] = take(out)
                except Exception as exc:  # a snapshot failure must not change the run being measured
                    store[name] = f"snapshot failed: {str(exc)[:120]}"
                return out

            mod.forward = wrapped

        wrap(voc.conv_pre, "conv_pre")
        for i in range(len(voc.ups)):
            wrap(voc.ups[i], f"ups{i}")
        wrap(voc.act_post, "act_post")
        wrap(voc.conv_post, "conv_post")

    # --- unsharded reference on the same mesh: every device holds the same tensor, so read one ---
    ref_dec = build(d, None, None)
    ref = {}
    instrument(ref_dec, ref, lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float())
    torch.manual_seed(2)
    lat = torch.randn(2, cfg["latent_channels"], T) * 0.1
    ref_out = torch.as_tensor(ref_dec(lat)).float()
    assert ref_dec.decoder._t_pad == 0, f"expected t_pad 0 at T={T}, got {ref_dec.decoder._t_pad}"

    # --- sharded run: reassemble full T from the shards that hold slices 0..factor-1 ---
    pc = ParallelFactor(factor=FACTOR, mesh_axis=MESH_AXIS)
    ccl = CCLManager(d, num_links=1, topology=ttnn.Topology.Linear)
    order = shard_order(d, pc)
    sh_dec = build(d, pc, ccl)
    got = {}

    def gather_shards(t):
        shards = ttnn.get_device_tensors(t)
        parts = [ttnn.to_torch(shards[i]).float() for i in order]
        return torch.cat(parts, dim=1)

    instrument(sh_dec, got, gather_shards)
    sh_out = torch.as_tensor(sh_dec(lat)).float()
    assert sh_dec.decoder._t_pad == 0, f"expected t_pad 0 at T={T}, got {sh_dec.decoder._t_pad}"

    print(f"\nfactor={FACTOR} axis={MESH_AXIS} T={T} (t_pad=0)\n")
    print(f"{'stage':>10} {'ref shape':>22} {'sharded shape':>22} {'PSNR dB':>9}")
    print("-" * 68)
    first_bad = None
    for name in STAGES:
        a, b = ref.get(name), got.get(name)
        if isinstance(a, str) or isinstance(b, str):
            print(f"{name:>10} {'':>22} {'':>22}  {a if isinstance(a, str) else b}")
            continue
        if a is None or b is None:
            print(f"{name:>10} {'(not recorded)':>22}")
            continue
        p = psnr(a, b)
        flag = "" if (p is not None and p > 40) else "  <== DIVERGES"
        if flag and first_bad is None:
            first_bad = name
        print(
            f"{name:>10} {str(tuple(a.shape)):>22} {str(tuple(b.shape)):>22} {('n/a' if p is None else f'{p:8.1f}')}{flag}"
        )

    print(f"\nfinal output PSNR: {psnr(ref_out, sh_out):.1f} dB")
    print(f"first diverging stage: {first_bad or 'none -- divergence is after conv_post'}")
finally:
    ttnn.close_mesh_device(d)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
