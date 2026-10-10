"""t376: exact _BLOCKINGS keys of every LTX-2.5 conv VAE decoder conv, 1088x1920 / 145 frames on 4x8 (CPU only).

Mirrors vae_ltx._compute_ltx_decoder_dims + LTXVideoDecoder channel logic, with the decoder_blocks from the
safetensors config of /var/tmp/fasth3/models/ltx-2.5/vae/ltx-2.5-video-vae-conv-bf16.safetensors (blx01).
"""
import ast, math, pathlib, re, sys

BLOCKS = [("res_x", 4), ("compress_space", 2), ("res_x", 6), ("compress_time", 2), ("res_x", 4),
          ("compress_all", 1), ("res_x", 2), ("compress_all", 2), ("res_x", 2)]
STRIDE = {"compress_all": (2, 2, 2), "compress_space": (1, 2, 2), "compress_time": (2, 1, 1)}
h_f, w_f, frames, height, width = 4, 8, 145, 1088, 1920
dev = lambda full, f: (full + f - 1) // f
H, W, T = height // 32, width // 32, (frames - 1) // 8 + 1
k3 = lambda: (T + 2, dev(H, h_f), dev(W, w_f))
convs = [("conv_in", 128, 1024, k3())]
ch = 1024
for name, mult in reversed(BLOCKS):
    if name in STRIDE:
        p = STRIDE[name]
        convs.append((f"up_{name}_x{mult}", ch, math.prod(p) * ch // mult, k3()))
        H, W, T = H * p[1], W * p[2], T * p[0] - (1 if p[0] == 2 else 0)
        ch //= mult
    else:
        for i in range(mult):  # each res block has 2 convs
            convs += [(f"res{ch}", ch, ch, k3())] * 2
convs.append(("conv_out", 128, 48, k3()))

src = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "models/tt_dit/utils/conv3d.py").read_text()
table = {}
for m in re.finditer(r"^\s*\((4, 8, \d+, \d+, \(3, 3, 3\), \d+, \d+, \d+)\): (\([\d, ]+\)),\s*#\s*(\S+)", src, re.M):
    table.setdefault(ast.literal_eval(m.group(1)), (ast.literal_eval(m.group(2)), m.group(3)))
seen = {}
for name, cin, cout, (t, h, w) in convs:
    key = (4, 8, cin, cout, (3, 3, 3), t, h, w)
    seen.setdefault(key, []).append(name)
for key, names in seen.items():
    blk, tag = table.get(key, (None, "-"))
    print(f"{key}  x{len(names):2d} {sorted(set(names))}  table={blk} ({tag})")
