# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""TRELLIS.2-4B image to textured 3D mesh, with the three flow DiTs on one Tenstorrent Blackhole chip.

Runs the reference TRELLIS.2 pipeline (microsoft/TRELLIS.2 75fbf018 with the CPU overlay in this folder) and replaces
the forward of its structure, shape and texture flow DiTs with the TT models of this branch. Everything else stays on
the host CPU in fp32: DINOv3 conditioning, the structure decoder, the cascade upsampler and the SC-VAE decoders.
Output: a vertex-coloured PLY, a six-view preview PNG and a timing JSON. Setup and usage: README.md next to this file.

    python models/experimental/trellis2/demo/demo.py --trellis-dir <TRELLIS.2 clone> --image <rgba image> [--out <dir>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

TRELLIS2_REPO, TRELLIS2_REV = "microsoft/TRELLIS.2-4B", "af44b45f2e35a493886929c6d786e563ec68364d"
TRELLIS1_REPO, TRELLIS1_REV = "microsoft/TRELLIS-image-large", "25e0d31ffbebe4b5a97464dd851910efc3002d96"
DINOV3_REPO, DINOV3_REV = "facebook/dinov3-vitl16-pretrain-lvd1689m", "ea8dc2863c51be0a264bab82070e3e8836b02d51"
PIPELINE_TYPES = ("512", "1024", "1024_cascade", "1536_cascade")

# pipeline model name -> (checkpoint, TT model: None for the structure DiT, else SLat input channels)
FLOW_MODELS = {
    "sparse_structure_flow_model": ("ss_flow_img_dit_1_3B_64_bf16", None),
    "shape_slat_flow_model_512": ("slat_flow_img2shape_dit_1_3B_512_bf16", 32),
    "tex_slat_flow_model_512": ("slat_flow_imgshape2tex_dit_1_3B_512_bf16", 64),
    "shape_slat_flow_model_1024": ("slat_flow_img2shape_dit_1_3B_1024_bf16", 32),
    "tex_slat_flow_model_1024": ("slat_flow_imgshape2tex_dit_1_3B_1024_bf16", 64),
}
DECODERS = {"shape_slat_decoder": "shape_dec_next_dc_f16c32_fp16", "tex_slat_decoder": "tex_dec_next_dc_f16c32_fp16"}
SS_DECODER = "ss_dec_conv3d_16l8_fp16"


def models_for(ptype: str) -> list[str]:
    flows = {
        "512": ["shape_slat_flow_model_512", "tex_slat_flow_model_512"],
        "1024": ["shape_slat_flow_model_1024", "tex_slat_flow_model_1024"],
        "1024_cascade": ["shape_slat_flow_model_512", "shape_slat_flow_model_1024", "tex_slat_flow_model_1024"],
        "1536_cascade": ["shape_slat_flow_model_512", "shape_slat_flow_model_1024", "tex_slat_flow_model_1024"],
    }[ptype]
    return ["sparse_structure_flow_model", "sparse_structure_decoder", *flows, *DECODERS]


# ---------------------------------------------------------------------------------------------------- weights


def fetch_weights(weights_dir: str, ptype: str) -> str:
    """Downloads what the pipeline type needs into weights_dir (skips files already there); returns the config name.
    DINOv3 is gated: request access on its Hugging Face page and log in (`hf auth login`) or set HF_TOKEN."""
    from huggingface_hub import hf_hub_download, snapshot_download

    t2 = os.path.join(weights_dir, "TRELLIS.2-4B")
    get = lambda repo, rev, f, d: hf_hub_download(repo, f, revision=rev, local_dir=d)
    get(TRELLIS2_REPO, TRELLIS2_REV, "pipeline.json", t2)
    names = [FLOW_MODELS[m][0] for m in models_for(ptype) if m in FLOW_MODELS] + list(DECODERS.values())
    for name in names:
        for ext in (".json", ".safetensors"):
            get(TRELLIS2_REPO, TRELLIS2_REV, f"ckpts/{name}{ext}", t2)
    t1 = os.path.join(weights_dir, "TRELLIS-image-large")
    for ext in (".json", ".safetensors"):
        get(TRELLIS1_REPO, TRELLIS1_REV, f"ckpts/{SS_DECODER}{ext}", t1)
    dinov3 = os.path.join(weights_dir, "dinov3-vitl16-pretrain-lvd1689m")
    if not os.path.exists(os.path.join(dinov3, "config.json")):
        snapshot_download(DINOV3_REPO, revision=DINOV3_REV, local_dir=dinov3)

    cfg = json.load(open(os.path.join(t2, "pipeline.json")))
    keep = set(models_for(ptype))
    cfg["args"]["models"] = {k: v for k, v in cfg["args"]["models"].items() if k in keep}
    cfg["args"]["models"]["sparse_structure_decoder"] = os.path.join(t1, "ckpts", SS_DECODER)
    cfg["args"]["image_cond_model"]["args"]["model_name"] = dinov3
    name = f"pipeline_tt_{ptype}.json"
    with open(os.path.join(t2, name), "w") as f:
        json.dump(cfg, f, indent=1)
    return name


# ---------------------------------------------------------------------------------------------------- host pipeline


class _NoRembg:
    """RMBG-2.0 (the shipped default) is non-commercial and never loaded; inputs carry their own alpha."""

    def __init__(self, *a, **k):
        pass

    def to(self, *a, **k):
        return self

    def cpu(self):
        return self

    def __call__(self, *a, **k):
        raise RuntimeError("the input has no alpha channel; pass an RGBA image or --matte-white")


def _dinov3_features(self, image: torch.Tensor) -> torch.Tensor:
    """TRELLIS.2's DinoV3FeatureExtractor.extract_features for both transformers layouts: 4.x keeps the blocks in
    model.layer, 5.x in model.model.layer."""
    image = image.to(self.model.embeddings.patch_embeddings.weight.dtype)
    h = self.model.embeddings(image, bool_masked_pos=None)
    pe = self.model.rope_embeddings(image)
    layers = self.model.layer if hasattr(self.model, "layer") else self.model.model.layer
    for layer in layers:
        h = layer(h, position_embeddings=pe)
        h = h[0] if isinstance(h, tuple) else h
    return F.layer_norm(h, h.shape[-1:])


def build_pipeline(trellis_dir: str, weights_dir: str, config_name: str):
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_CONV_BACKEND", "torch")
    sys.path.insert(0, trellis_dir)
    from trellis2.modules import image_feature_extractor
    from trellis2.modules.utils import convert_module_to
    from trellis2.pipelines import Trellis2ImageTo3DPipeline, rembg

    if not os.path.exists(os.path.join(trellis_dir, "trellis2", "utils", "fdg_torch.py")):
        raise SystemExit(f"{trellis_dir} is not TRELLIS.2 75fbf018 with trellis2_cpu_overlay.patch applied (README)")
    rembg.BiRefNet = _NoRembg
    image_feature_extractor.DinoV3FeatureExtractor.extract_features = _dinov3_features
    pipe = Trellis2ImageTo3DPipeline.from_pretrained(os.path.join(weights_dir, "TRELLIS.2-4B"), config_file=config_name)
    pipe.low_vram = False
    for m in pipe.models.values():
        if hasattr(m, "convert_to"):
            m.convert_to(torch.float32)
        else:
            for part in ("blocks", "middle_block"):
                if hasattr(m, part):
                    getattr(m, part).apply(partial(convert_module_to, dtype=torch.float32))
            m.dtype = torch.float32
            m.use_fp16 = False
        m.float()
    pipe.image_cond_model.model.to(torch.float32)
    pipe.to(torch.device("cpu"))
    return pipe


def load_image(path: str, matte_white: bool) -> Image.Image:
    image = Image.open(path)
    if image.mode == "RGBA":
        return image
    if not matte_white:
        raise SystemExit(f"{path} is {image.mode}; pass an RGBA image, or --matte-white for an object on plain white")
    from scipy import ndimage

    rgb = np.array(image.convert("RGB"))
    near_white = (rgb >= 245).all(-1)
    labels, _ = ndimage.label(near_white)
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    background = np.isin(labels, border[border > 0])
    alpha = np.where(background, 0, 255).astype(np.uint8)
    return Image.fromarray(np.dstack([rgb, alpha]), "RGBA")


# ---------------------------------------------------------------------------------------------------- TT models


class TtFlows:
    """Loads the TT flow DiTs the pipeline needs and routes the pipeline's flow-model calls to them."""

    def __init__(self, mesh_device, weights_dir: str):
        self.mesh_device = mesh_device
        self.ckpts = os.path.join(weights_dir, "TRELLIS.2-4B", "ckpts")
        self.models, self.tables = {}, {}
        self.calls, self.device_s = 0, 0.0

    def _model(self, name: str):
        if name not in self.models:
            from models.experimental.trellis2.reference import structure_flow_host as host
            from models.experimental.trellis2.tt.slat_flow import TtSLatFlow
            from models.experimental.trellis2.tt.structure_flow import TtStructureFlow

            ckpt, in_channels = FLOW_MODELS[name]
            if in_channels is None:
                model = TtStructureFlow(mesh_device=self.mesh_device)
            else:
                model = TtSLatFlow(mesh_device=self.mesh_device, in_channels=in_channels)
            model.load_torch_state_dict(host.load_state_dict(os.path.join(self.ckpts, f"{ckpt}.safetensors")))
            self.models[name] = model
        return self.models[name]

    def _forward(self, name: str):
        def structure(x, t, cond, **kw):
            t0 = time.time()
            out = self._model(name)(x.float(), t.float(), cond.float())
            self.calls, self.device_s = self.calls + 1, self.device_s + time.time() - t0
            return out

        def slat(x, t, cond, concat_cond=None, **kw):
            t0 = time.time()
            model = self._model(name)
            feats = x.feats.float()
            if concat_cond is not None:
                feats = torch.cat([feats, concat_cond.feats.float()], dim=-1)
            coords = x.coords.int()
            key = (coords.shape[0], hashlib.sha1(coords.numpy().tobytes()).hexdigest(), cond.shape[1])
            if key not in self.tables:
                self.tables[key] = model.prepare(coords, cond.shape[1])
            out = model(feats, t.float(), cond.float(), self.tables[key])
            self.calls, self.device_s = self.calls + 1, self.device_s + time.time() - t0
            return x.replace(out)

        return structure if FLOW_MODELS[name][1] is None else slat

    def patch(self, pipe) -> None:
        for name in FLOW_MODELS:
            if name in pipe.models:
                pipe.models[name].forward = self._forward(name)


# ---------------------------------------------------------------------------------------------------- run


def run_pipeline(pipe, flows: TtFlows, image: Image.Image, ptype: str, seed: int):
    """TRELLIS2ImageTo3DPipeline.run without decode_latent's CUDA-only hole filling. Returns meshes, texture voxels,
    the output resolution, voxel / token counts and per-stage timings."""
    timings, stats = {}, {}

    def stage(name, fn):
        c0, d0, t0 = flows.calls, flows.device_s, time.time()
        out = fn()
        timings[name] = {"wall_s": round(time.time() - t0, 1)}
        if flows.calls > c0:
            timings[name].update(tt_calls=flows.calls - c0, tt_s=round(flows.device_s - d0, 1))
        print(f"  {name}: {timings[name]}", flush=True)
        return out

    torch.manual_seed(seed)
    cond_512 = stage("dinov3_512", lambda: pipe.get_cond([image], 512))
    cond_1024 = stage("dinov3_1024", lambda: pipe.get_cond([image], 1024)) if ptype != "512" else None
    ss_res = {"512": 32, "1024": 64, "1024_cascade": 32, "1536_cascade": 32}[ptype]
    coords = stage("structure", lambda: pipe.sample_sparse_structure(cond_512, ss_res, 1, {}))
    stats["structure_voxels"] = int(coords.shape[0])
    if ptype in ("512", "1024"):
        cond = cond_512 if ptype == "512" else cond_1024
        shape = stage(
            "shape", lambda: pipe.sample_shape_slat(cond, pipe.models[f"shape_slat_flow_model_{ptype}"], coords, {})
        )
        tex = stage(
            "texture", lambda: pipe.sample_tex_slat(cond, pipe.models[f"tex_slat_flow_model_{ptype}"], shape, {})
        )
        res = int(ptype)
    else:
        hr = 1024 if ptype == "1024_cascade" else 1536
        shape, res = stage(
            "shape",
            lambda: pipe.sample_shape_slat_cascade(
                cond_512,
                cond_1024,
                pipe.models["shape_slat_flow_model_512"],
                pipe.models["shape_slat_flow_model_1024"],
                512,
                hr,
                coords,
                {},
                49152,
            ),
        )
        tex = stage(
            "texture", lambda: pipe.sample_tex_slat(cond_1024, pipe.models["tex_slat_flow_model_1024"], shape, {})
        )
    stats["slat_tokens"] = int(shape.coords.shape[0])
    with torch.no_grad():
        meshes, subs = stage("decode_shape", lambda: pipe.decode_shape_slat(shape, res))
        tex_voxels = stage("decode_texture", lambda: pipe.decode_tex_slat(tex, subs))
    return meshes[0], tex_voxels, res, stats, timings


def write_ply(path: str, mesh, tex_voxels, res: int) -> dict:
    """Binary PLY with per-vertex base colour from the texture voxel containing the vertex (raw dual-grid mesh: no
    simplification, hole filling or UVs)."""
    v = mesh.vertices.cpu().numpy().astype(np.float32)
    f = mesh.faces.cpu().numpy().astype(np.int32)
    tc = tex_voxels.coords[:, 1:].cpu().numpy().astype(np.int64)
    base = tex_voxels.feats[:, 0:3].cpu().numpy().clip(0, 1)
    key = lambda c: (c[:, 0] * res + c[:, 1]) * res + c[:, 2]
    order = np.argsort(key(tc))
    sorted_keys = key(tc)[order]

    def lookup(c):
        k = key(c)
        pos = np.searchsorted(sorted_keys, k).clip(max=len(sorted_keys) - 1)
        hit = (sorted_keys[pos] == k) & (c >= 0).all(1) & (c < res).all(1)
        return np.where(hit, order[pos], -1)

    cell = np.floor((v + 0.5) * res).astype(np.int64)
    idx = lookup(cell)
    for off in np.stack(np.meshgrid(*[[-1, 0, 1]] * 3, indexing="ij"), -1).reshape(-1, 3):
        miss = idx < 0
        if not miss.any():
            break
        idx[miss] = lookup(cell[miss] + off)
    rgb = np.full((len(v), 3), 128, np.uint8)
    rgb[idx >= 0] = (base[idx[idx >= 0]] * 255).round().astype(np.uint8)

    vert = np.empty(len(v), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    vert["x"], vert["y"], vert["z"] = v[:, 0], v[:, 1], v[:, 2]
    vert["r"], vert["g"], vert["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    face = np.empty(len(f), dtype=[("n", "u1"), ("i", "<i4", (3,))])
    face["n"], face["i"] = 3, f
    header = (
        "ply\nformat binary_little_endian 1.0\ncomment TRELLIS.2-4B, flow DiTs on Tenstorrent\n"
        f"element vertex {len(v)}\nproperty float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f"element face {len(f)}\nproperty list uchar int vertex_indices\nend_header\n"
    )
    with open(path, "wb") as fh:
        fh.write(header.encode())
        fh.write(vert.tobytes())
        fh.write(face.tobytes())
    return {"vertices": len(v), "faces": len(f), "uncoloured_vertices": int((idx < 0).sum())}


def write_views(path: str, tex_voxels, image: Image.Image, size: int = 256) -> None:
    """Input next to the textured voxels seen from +-x, +-y, +-z (nearest voxel wins, depth-shaded)."""
    c = tex_voxels.coords[:, 1:].cpu().numpy()
    col = tex_voxels.feats[:, 0:3].cpu().numpy().clip(0, 1)
    n = int(c.max()) + 1
    panels = [("input", image.convert("RGB").resize((size, size)))]
    for axis in range(3):
        u, w = [a for a in range(3) if a != axis]
        for sign, name in ((1, "+"), (-1, "-")):
            depth = c[:, axis] * sign
            order = np.argsort(depth)
            img = np.zeros((n, n, 3))
            shade = 0.55 + 0.45 * (depth[order] - depth.min()) / max(1, np.ptp(depth))
            uu = c[order, u] if sign > 0 else n - 1 - c[order, u]
            img[n - 1 - c[order, w], uu] = col[order] * shade[:, None]
            panel = Image.fromarray((img * 255).astype(np.uint8)).resize((size, size), Image.NEAREST)
            panels.append((f"from {name}{'xyz'[axis]}", panel))
    canvas = Image.new("RGB", (size * 4, (size + 18) * 2), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    for i, (name, p) in enumerate(panels):
        x, y = (i % 4) * size, (i // 4) * (size + 18)
        canvas.paste(p, (x, y + 18))
        draw.text((x + 4, y + 3), name, fill=(230, 230, 230))
    canvas.save(path)


def run_e2e(
    mesh_device,
    *,
    trellis_dir: str,
    weights_dir: str,
    image_path: str,
    out_dir: str = "trellis2_out",
    pipeline_type: str = "512",
    seed: int = 42,
    matte_white: bool = False,
) -> dict:
    """Image file in, PLY + preview + summary JSON out (under out_dir). Returns the summary."""
    os.makedirs(out_dir, exist_ok=True)
    t_all = time.time()
    t0 = time.time()
    config_name = fetch_weights(weights_dir, pipeline_type)
    pipe = build_pipeline(trellis_dir, weights_dir, config_name)
    flows = TtFlows(mesh_device, weights_dir)
    flows.patch(pipe)
    load_s = round(time.time() - t0, 1)
    print(f"loaded the pipeline in {load_s} s; running {pipeline_type}", flush=True)

    image = pipe.preprocess_image(load_image(image_path, matte_white))
    stem = os.path.splitext(os.path.basename(image_path))[0]
    image.save(os.path.join(out_dir, f"{stem}_preprocessed.png"))
    mesh, tex_voxels, res, stats, timings = run_pipeline(pipe, flows, image, pipeline_type, seed)

    ply = os.path.join(out_dir, f"{stem}_{pipeline_type}_seed{seed}.ply")
    mesh_stats = write_ply(ply, mesh, tex_voxels, res)
    views = os.path.join(out_dir, f"{stem}_{pipeline_type}_seed{seed}_views.png")
    write_views(views, tex_voxels, image)
    summary = {
        "image": os.path.abspath(image_path),
        "pipeline_type": pipeline_type,
        "resolution": res,
        "seed": seed,
        **stats,
        "mesh": mesh_stats,
        "ply": os.path.abspath(ply),
        "views": os.path.abspath(views),
        "load_s": load_s,
        "stages": timings,
        "tt_flow_s": round(flows.device_s, 1),
        "host_peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1),
        "total_s": round(time.time() - t_all, 1),
    }
    with open(os.path.join(out_dir, f"{stem}_{pipeline_type}_seed{seed}.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1), flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trellis-dir", default=os.environ.get("TRELLIS2_REPO"), help="TRELLIS.2 clone with the overlay")
    ap.add_argument("--image", default=os.environ.get("TRELLIS2_IMAGE"), help="RGBA image (or set TRELLIS2_IMAGE)")
    ap.add_argument("--matte-white", action="store_true", help="cut a plain white background from an RGB image")
    ap.add_argument("--out", default="trellis2_out")
    ap.add_argument("--pipeline-type", default="512", choices=PIPELINE_TYPES)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--weights-dir", default=os.path.expanduser(os.environ.get("TRELLIS2_WEIGHTS_DIR", "~/.cache/trellis2_tt"))
    )
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads for the host stages")
    args = ap.parse_args()
    if not args.trellis_dir:
        raise SystemExit("pass --trellis-dir or set TRELLIS2_REPO (see README.md)")
    if not args.image:
        raise SystemExit("pass --image or set TRELLIS2_IMAGE: an RGBA image of one object (see README.md)")
    if args.threads:
        torch.set_num_threads(args.threads)

    import ttnn

    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        run_e2e(
            mesh_device,
            trellis_dir=os.path.abspath(args.trellis_dir),
            weights_dir=args.weights_dir,
            image_path=args.image,
            out_dir=args.out,
            pipeline_type=args.pipeline_type,
            seed=args.seed,
            matte_white=args.matte_white,
        )
    finally:
        ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
