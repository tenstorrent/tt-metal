#!/usr/bin/env python3
# Census of the host's trid-ring decision for every production conv3d blocking in the repo.
#
# Needs the scratch factory hook: with TT_CONV3D_PROBE_ONLY=1 the program factory throws a RuntimeError
# whose message carries "CONV3D_GATHER bytes_per_tile=... gather_trids=... coalesced=... staging=..."
# right after it has made the ring decision and before it builds any kernel. The decision depends only on
# (C_in, C_out, kernel, blocking, dtype, DRAM geometry of the device), not on T/H/W or the grid, so each
# config is probed with the smallest input that holds one output block.
#
# usage: census.py <out.jsonl> [arch-label]   (set TT_METAL_MOCK_CLUSTER_DESC_PATH for a mock WH device)
import json
import os
import re
import sys
import traceback

import torch
import ttnn

from models.tt_dit.utils.conv3d import _BLOCKINGS, _DEFAULT_BLOCKINGS, _FP32_BLOCKINGS, _SPLIT_BLOCKINGS

out_path, arch_label = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "")

entries = []  # (source, arch, dtype, C_in, C_out, kernel, blocking, note)


def add(source, arch, dtype, c_in, c_out, kernel, blocking, note=""):
    entries.append((source, arch, dtype, int(c_in), int(c_out), tuple(kernel), tuple(blocking), note))


# tt_dit exact tables (Wan 2.2 decoder/encoder, LTX-2.3 decoder/upsampler): all swept on Blackhole.
for (hf, wf, c_in, c_out, k, T, H, W), blk in _BLOCKINGS.items():
    add("tt_dit._BLOCKINGS", "bh", "bf16", c_in, c_out, k, blk, f"h{hf}w{wf} T{T} H{H} W{W}")
# Hunyuan Image 3.0 VAE tables: read from source with ast (importing the package downloads the model).
import ast, pathlib  # noqa: E402

_hy = {}
for node in ast.parse(
    pathlib.Path("models/experimental/hunyuan_image_3_0/ttnn/vae/conv3d_blockings.py").read_text()
).body:
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        _hy[node.target.id] = ast.literal_eval(node.value)
for (hf, wf, c_in, c_out, k, T, H, W), blk in _hy["HUNYUAN_EXACT_BLOCKINGS"].items():
    add("hunyuan.exact", "bh", "bf16", c_in, c_out, k, blk, f"h{hf}w{wf} T{T} H{H} W{W}")
for (hf, wf, c_in, c_out, k, T, H, W), blk in _hy["HUNYUAN_DECODER_EXACT_BLOCKINGS"].items():
    add("hunyuan.decoder", "bh", "bf16", c_in, c_out, k, blk, f"h{hf}w{wf} T{T} H{H} W{W}")
for (c_in, c_out, k), blk in _hy["HUNYUAN_CHANNEL_BLOCKINGS"].items():
    add("hunyuan.channel", "bh", "bf16", c_in, c_out, k, blk)
# Per-arch split tables (Qwen-Image / FIBO VAE conv2d-as-conv3d).
for (arch, hf, wf, c_in, c_out, k), blk in _SPLIT_BLOCKINGS.items():
    add("tt_dit._SPLIT_BLOCKINGS", arch, "bf16", c_in, c_out, k, blk, f"h{hf}w{wf}")
# MiniMax-H3 VAE encoder (conv_minimax_h3.py registers these at import; importing it pulls in the model).
_H3 = {
    (32, 128): (32, 128, 3, 2, 16),
    (128, 128): (64, 128, 1, 16, 2),
    (128, 256): (32, 256, 3, 16, 2),
    (256, 256): (64, 128, 1, 16, 2),
    (256, 512): (64, 128, 1, 16, 2),
    (512, 512): (64, 128, 1, 16, 2),
    (512, 1024): (64, 128, 1, 8, 4),
    (1024, 1024): (64, 64, 5, 16, 2),
    (1024, 48): (128, 32, 1, 16, 2),
}
for (c_in, c_out), blk in _H3.items():
    for k in ((3, 3, 3), (1, 3, 3)):
        for dtype in ("bf16", "fp32"):
            add("minimax_h3", "bh", dtype, c_in, c_out, k, blk)
add("minimax_h3", "bh", "bf16", 128, 256, (1, 1, 1), (128, 256, 16, 1, 32))
# Channel fallbacks (LTX encoder, LTX audio mel-VAE) and fp32 audio vocoder blockings.
for (c_in, c_out, k), blk in _DEFAULT_BLOCKINGS.items():
    for arch in ("bh", "wh"):
        add("tt_dit._DEFAULT_BLOCKINGS", arch, "bf16", c_in, c_out, k, blk)
for (c_in, c_out, k), blk in _FP32_BLOCKINGS.items():
    for arch in ("bh", "wh"):
        add("tt_dit._FP32_BLOCKINGS", arch, "fp32", c_in, c_out, k, blk)
# Mochi VAE (models/tt_dit/layers/conv3d.py): keyed by C_in only, same-channel 3x3x3 convs.
for c, blk in {
    768: (128, 96, 1, 2, 16),
    512: (128, 128, 1, 8, 4),
    256: (128, 128, 4, 4, 2),
    128: (128, 128, 1, 2, 16),
}.items():
    for arch in ("bh", "wh"):
        add("mochi.layers_conv3d", arch, "bf16", c, c, (3, 3, 3), blk)
# unet_3d demo (models/demos/unet_3d/ttnn_impl/conv3d.py).
for (c_in, c_out, k), blk in {
    (32, 32, 3): (32, 32, 1, 1, 16),
    (32, 64, 3): (32, 32, 2, 2, 16),
    (64, 64, 3): (32, 32, 1, 1, 16),
    (64, 128, 3): (32, 64, 1, 1, 16),
    (128, 128, 3): (64, 64, 1, 1, 16),
    (128, 256, 3): (64, 64, 1, 1, 8),
    (384, 128, 3): (128, 64, 1, 1, 4),
    (192, 64, 3): (64, 32, 4, 4, 8),
    (96, 32, 3): (32, 32, 2, 2, 8),
    (32, 32, 1): (32, 32, 4, 4, 16),
    (64, 32, 3): (32, 32, 2, 8, 8),
}.items():
    for arch in ("bh", "wh"):
        add("unet_3d", arch, "bf16", c_in, c_out, (k, k, k), blk)

want_arch = arch_label or ("bh" if ttnn.device.is_blackhole() else "wh")
# One probe per distinct (dtype, C_in, C_out, kernel, blocking); remember every entry that maps to it.
probes = {}
for e in entries:
    source, arch, dtype, c_in, c_out, k, blk, note = e
    # CENSUS_ANY_ARCH: the exact/channel tables are not arch-keyed (only _SPLIT_BLOCKINGS is), so on the other
    # arch keep every entry that was labelled for the sweep arch too.
    if arch != want_arch and not (os.environ.get("CENSUS_ANY_ARCH") and source != "tt_dit._SPLIT_BLOCKINGS"):
        continue
    users = probes.setdefault((dtype, c_in, c_out, k, blk), [])
    if (source, note) not in users:
        users.append((source, note))
print(f"{len(entries)} table entries, {len(probes)} distinct probes for arch {want_arch}", file=sys.stderr)

if os.environ.get("CENSUS_ETH_DISPATCH"):
    device = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 1), dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.ETH)
    )
else:
    device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
grid = (grid.x, grid.y)
if os.environ.get("CENSUS_GRID"):
    grid = tuple(int(v) for v in os.environ["CENSUS_GRID"].split("x"))
pat = re.compile(r"CONV3D_GATHER (.*?)(?:\n|$)")


def probe(dtype, c_in, c_out, k, blk):
    cin_blk, cout_blk, t_blk, h_blk, w_blk = blk
    kT, kH, kW = k
    c_pad = (c_in + 31) // 32 * 32
    T, H, W = t_blk + kT - 1, h_blk + kH - 1, w_blk + kW - 1
    tdt = torch.float32
    ddt = ttnn.float32 if dtype == "fp32" else ttnn.bfloat16
    x = ttnn.from_torch(
        torch.zeros(1, T, H, W, c_pad, dtype=tdt), device=device, dtype=ddt, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    w = ttnn.from_torch(torch.zeros(c_out, c_pad, kT, kH, kW, dtype=tdt), dtype=ddt, pad_value=0)
    w = ttnn.experimental.prepare_conv3d_weights(weight_tensor=w, C_in_block=cin_blk, alignment=32, device=device)
    b = ttnn.from_torch(
        torch.zeros(1, c_out, dtype=tdt), device=device, dtype=ddt, layout=ttnn.TILE_LAYOUT, pad_value=0
    )
    cfg = ttnn.Conv3dConfig(
        weights_dtype=ddt,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        T_out_block=t_blk,
        W_out_block=w_blk,
        H_out_block=h_blk,
        C_out_block=cout_blk,
        C_in_block=cin_blk,
        compute_with_storage_grid_size=grid,
    )
    ck = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    try:
        ttnn.experimental.conv3d(
            input_tensor=x,
            weight_tensor=w,
            device=device,
            bias_tensor=b,
            dtype=ddt,
            output_channels=c_out,
            kernel_size=k,
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            dilation=(1, 1, 1),
            padding_mode="zeros",
            config=cfg,
            compute_kernel_config=ck,
        )
        return {"error": "no probe exception (hook missing?)"}
    except RuntimeError as e:
        m = pat.search(str(e))
        if not m:
            return {"error": str(e).splitlines()[0][:300]}
        return {kv.split("=")[0]: int(kv.split("=")[1]) for kv in m.group(1).split()}
    finally:
        for t in (x, w, b):
            ttnn.deallocate(t)


with open(out_path, "w") as f:
    for i, (key, users) in enumerate(sorted(probes.items())):
        dtype, c_in, c_out, k, blk = key
        try:
            r = probe(dtype, c_in, c_out, k, blk)
        except Exception as e:  # noqa: BLE001
            r = {"error": f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"}
            traceback.print_exc(limit=1, file=sys.stderr)
        rec = {
            "arch": want_arch,
            "dtype": dtype,
            "C_in": c_in,
            "C_out": c_out,
            "kernel": list(k),
            "blocking": list(blk),
            "users": users,
            **r,
        }
        f.write(json.dumps(rec) + "\n")
        f.flush()
        print(
            f"[{i + 1}/{len(probes)}] {dtype} {c_in}x{c_out} k{k} blk{blk} -> {r.get('gather_trids', r.get('error'))}",
            file=sys.stderr,
        )
ttnn.close_device(device)
