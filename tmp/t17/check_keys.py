# CPU-only: list the LTX conv-decoder conv3d keys at a given res/frames and how get_conv3d_config resolves them.
import json, sys
from models.tt_dit.models.vae import vae_ltx as V
from models.tt_dit.utils import conv3d as C

ckpt, nf, H, W = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
with open(ckpt, "rb") as f:
    n = int.from_bytes(f.read(8), "little")
    hdr = json.loads(f.read(n))
cfg = json.loads(hdr["__metadata__"]["config"])["vae"]
blocks = cfg["decoder_blocks"]
print("decoder_blocks", blocks, "base", cfg.get("decoder_base_channels"))
dims = V._compute_ltx_decoder_dims(decoder_blocks=blocks, num_frames=nf, height=H, width=W, h_factor=4, w_factor=8)
for d in dims:
    print(d)
want = {(d.T, d.H, d.W) for d in dims}
print("table entries at these (T,H,W), mesh 4x8:")
for k, v in sorted(C._BLOCKINGS.items(), key=lambda kv: (-kv[0][5], kv[0][2])):
    if k[:2] == (4, 8) and (k[5], k[6], k[7]) in want:
        print(" ", k, v)
# Walk the decoder construction order and resolve each (C_in, C_out) site exactly as get_conv3d_config does.
ch = 1024
sites = [(128, ch)]
for name, p in reversed(blocks):
    if name in V._DECODER_STRIDE_MAP:
        p1, p2, p3 = V._DECODER_STRIDE_MAP[name]
        mult = p.get("multiplier", 2) if isinstance(p, dict) else 2
        out = ch * p1 * p2 * p3 // mult
        sites.append((ch, out))
        ch = out // (p1 * p2 * p3)
    else:
        sites.append((ch, ch))
sites.append((ch, 48))
miss = 0
for (ci, co), d in zip(sites, dims):
    k = (4, 8, ci, co, (3, 3, 3), d.T, d.H, d.W)
    hit = k in C._BLOCKINGS
    miss += not hit
    print("EXACT" if hit else "MISS ", k, C._BLOCKINGS.get(k))
print(f"sites={len(sites)} dims={len(dims)} misses={miss}")
