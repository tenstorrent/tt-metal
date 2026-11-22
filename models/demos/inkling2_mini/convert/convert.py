# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0

"""Inkling-2-Mini checkpoint -> fixed tt weight cache v1. Host only (ttnn host conversions, no device program).

    python -m models.demos.inkling2_mini.convert [--ckpt DIR] [--out DIR] [--layers 0,20] [--skip-global]

Each file is ``ttnn.dump_tensor`` of a host tensor in its final dtype + layout. ``manifest.json`` is rewritten
after every unit (global, L00 ... L23). Each unit records a key over the converter version and the settings that
shape it (expert dtypes, dtype-map entries, stick layout, expert source); a rerun skips a unit only when its key
is unchanged, so changing the dtype map reconverts exactly the layers it touches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import resource
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import ttnn
from models.demos.inkling2_mini.convert.source import Checkpoint, deinterleave_gate_up, dequant_experts
from models.demos.inkling2_mini.convert.stick import write_sticks

P = "model.llm."
# Bump when a change to this file changes what any unit writes: every unit's key changes, so a rerun reconverts.
CONVERTER_VERSION = 2
N_LAYERS = 24
GLOBAL_LAYERS = {5, 11, 17, 23}
LOOP_LAYERS = range(6, 18)
MXFP8_LAYERS = range(20, 24)  # publisher kept these at MXFP8: always stored BFP8
HIDDEN = 2304
N_EXPERTS = 128
EXPERT_CHUNK = 8
VOCAB_REAL, VOCAB_TT = 200058, 200064
QGRS_COLS = 6560
# Layers 0-19 whose experts are all promoted from BFP4 to BFP8 by default; the per-tensor picks (ranked by MoE-block
# output PCC vs the bf16 dequant) live in default_dtype_map.json.
DEFAULT_BFP8_EXPERT_LAYERS: tuple[int, ...] = ()
# Sanity floor on per-tensor weight PCC vs the bf16 dequant (4-bit formats sit near 0.993): BFP4 experts below it
# fall back to BFP8; any other tensor below it is a bug.
MIN_WEIGHT_PCC = 0.99
PCC_CHUNK_ELEMS = 1 << 23

DTYPES = {"BF16": ttnn.bfloat16, "FP32": ttnn.float32, "BFP8": ttnn.bfloat8_b, "BFP4": ttnn.bfloat4_b}
TORCH_DTYPES = {"BF16": torch.bfloat16, "FP32": torch.float32, "BFP8": torch.bfloat16, "BFP4": torch.bfloat16}
LAYOUTS = {"RM": ttnn.ROW_MAJOR_LAYOUT, "TILE": ttnn.TILE_LAYOUT}


class Pcc:
    """Pearson correlation accumulated over chunks (float64 sums)."""

    def __init__(self):
        self.sums = torch.zeros(5, dtype=torch.float64)
        self.n = 0

    def add(self, ref: torch.Tensor, got: torch.Tensor):
        x, y = ref.double().flatten(), got.double().flatten()
        self.sums += torch.stack([x.sum(), y.sum(), (x * x).sum(), (y * y).sum(), (x * y).sum()])
        self.n += x.numel()

    def value(self) -> float:
        sx, sy, sxx, syy, sxy = self.sums.tolist()
        vx, vy = sxx - sx * sx / self.n, syy - sy * sy / self.n
        if vx <= 0 or vy <= 0:
            return 1.0 if vx == vy == 0 else 0.0
        return (sxy - sx * sy / self.n) / (vx * vy) ** 0.5


def roundtrip_pcc(t: torch.Tensor, dtype: str) -> tuple[float, float]:
    """PCC of ``t`` [1, X, R, C] vs its host ``dtype`` TILE round trip, and the min over ``X`` (experts).
    Converted in row blocks of a multiple of 32: tiles are encoded independently, so this equals the full tensor."""
    total, per_x = Pcc(), []
    rows = max(32, PCC_CHUNK_ELEMS // t.shape[-1] // 32 * 32)
    for i in range(t.shape[1]):
        one = Pcc()
        for r in range(0, t.shape[2], rows):
            ref = t[:, i : i + 1, r : r + rows].contiguous()
            got = ttnn.to_torch(ttnn.from_torch(ref, dtype=DTYPES[dtype], layout=ttnn.TILE_LAYOUT))
            total.add(ref, got)
            one.add(ref, got)
        per_x.append(one.value())
    return total.value(), min(per_x)


def rows32(t: torch.Tensor) -> torch.Tensor:
    """Vector(s) [C] or [k, C] -> fp32 [32, C], rows 0..k-1 used, rest zero."""
    t = t.float().reshape(-1, t.shape[-1])
    return F.pad(t, (0, 0, 0, 32 - t.shape[0]))


def lag_taps(conv1d_weight: torch.Tensor) -> torch.Tensor:
    """Depthwise conv1d weight [C, 1, 4] (causal, tap 3 = current) -> [4, C] with row r = lag r."""
    return conv1d_weight[:, 0, :].flip(-1).T


def expected_shape(name: str, layer: int | None) -> tuple:
    """cache shapes, without the leading [1, X] dims."""
    is_global = layer in GLOBAL_LAYERS
    return {
        "embed": (201024, HIDDEN),
        "embed_norm": (72, 32),
        "final_norm": (72, 32),
        "unembed": (HIDDEN, VOCAB_TT),
        "final_res_q": (32, HIDDEN),
        "msg_q": (32, HIDDEN),
        "attn.w_qgrs": (HIDDEN, QGRS_COLS),
        "attn.w_k": (HIDDEN, 1056 if is_global else 1024),
        "attn.w_v": (HIDDEN, 1056 if is_global else 1024),
        "attn.w_o": (3072, HIDDEN),
        "attn.q_clip_scale": (24, 128),
        "attn.rel_clip_scale": (24, 16),
        "attn.rel_proj": (16, 1024),
        "attn.k_conv": (4, 1024),
        "attn.v_conv": (4, 1024),
        "post_attn_conv": (4, HIDDEN),
        "post_mlp_conv": (4, HIDDEN),
        "attn_res_q": (32, HIDDEN),
        "mlp_res_q": (32, HIDDEN),
        "router.w": (HIDDEN, N_EXPERTS),
        "router.bias": (32, N_EXPERTS),
        "experts.w_gate_up": (HIDDEN, HIDDEN),
        "experts.w_down": (1152, HIDDEN),
    }.get(name, (72, 32) if name.endswith("norm") else (4608, 64) if name.endswith(".down") else (64, HIDDEN))


class Writer:
    def __init__(self, out: Path):
        self.out = out
        self.entries: list[dict] = []

    def write(self, name, layer, tensor, dtype, layout, src, pcc=None, pcc_bf4=None, **extra):
        if tensor.dim() == 2:
            tensor = tensor[None, None]
        if tuple(tensor.shape[2:]) != expected_shape(name, layer):
            raise ValueError(
                f"{name} L{layer}: shape {tuple(tensor.shape)} != cache shape {expected_shape(name, layer)}"
            )
        tensor = tensor.to(TORCH_DTYPES[dtype]).contiguous()
        if pcc is None and dtype in ("BFP8", "BFP4"):
            pcc, _ = roundtrip_pcc(tensor, dtype)
        if pcc is not None and pcc < MIN_WEIGHT_PCC:
            raise ValueError(f"{name} L{layer}: {dtype} weight PCC {pcc:.5f} < {MIN_WEIGHT_PCC}")
        rel = f"global/{name}.tensorbin" if layer is None else f"layers/L{layer:02d}/{name}.tensorbin"
        (self.out / rel).parent.mkdir(parents=True, exist_ok=True)
        ttnn.dump_tensor(self.out / rel, ttnn.from_torch(tensor, dtype=DTYPES[dtype], layout=LAYOUTS[layout]))
        self.entries.append(
            {
                "name": name,
                "layer": layer,
                "file": rel,
                "dtype": dtype,
                "layout": layout,
                "shape": list(tensor.shape),
                "src": src,
                "pcc": 1.0 if pcc is None else round(pcc, 7),  # BF16/FP32 copies of bf16/fp32 sources are exact
                "pcc_bf4": None if pcc_bf4 is None else round(pcc_bf4, 7),
                "fallback": False,
                **extra,
            }
        )


def convert_global(ck: Checkpoint, w: Writer):
    w.write("embed", None, ck.get(P + "embed.weight"), "BF16", "RM", [P + "embed.weight"])
    for name, key in (("embed_norm", "embed_norm.weight"), ("final_norm", "norm.weight")):
        w.write(name, None, ck.get(P + key).reshape(72, 32), "BF16", "RM", [P + key])
    unembed = F.pad(ck.get(P + "unembed.weight")[:VOCAB_REAL].T, (0, VOCAB_TT - VOCAB_REAL))
    w.write("unembed", None, unembed, "BFP8", "TILE", [P + "unembed.weight"])
    del unembed
    key = P + "attn_res.output_query"
    w.write("final_res_q", None, rows32(ck.get(key)), "FP32", "TILE", [key])
    # (q, mlp) and (k, v) per loop layer -> (q, k, v, mlp), as transformers' Inkling2LoopMergeMessageQueries
    qk, kvk = P + "middle_cycle_cross_pass_queries", P + "middle_cycle_cross_pass_kv_queries"
    q, kv = ck.get(qk).unflatten(0, (-1, 2)), ck.get(kvk).unflatten(0, (-1, 2))
    msg = torch.cat([q[:, :1], kv, q[:, 1:]], dim=1)
    w.write("msg_q", None, torch.stack([rows32(m) for m in msg])[None], "FP32", "TILE", [qk, kvk])


def expert_tensor_dtype(dtype_map: dict, layer: int, name: str, default: str) -> str:
    """Resolve one expert tensor's dtype from ``--dtype-map``: ``"L08/experts.w_down": "BFP8"`` (per tensor) or
    ``{"default": "BFP4", "BFP8": [3, 17]}`` (per expert). A cache file holds one dtype, so a per-expert map that
    mixes dtypes inside one tensor is refused until the model splits experts into two groups."""
    spec = dtype_map.get(f"L{layer:02d}/{name}", default)
    if isinstance(spec, dict):
        other = [d for d in spec if d != "default" and spec[d] and d != spec.get("default", default)]
        if other:
            raise SystemExit(
                f"L{layer:02d}/{name}: per-expert dtypes {sorted(other)} + {spec.get('default', default)} cannot be "
                "stored in one cache tensor; needs the two-group expert split (later model change)"
            )
        spec = spec.get("default", default)
    if spec not in ("BFP4", "BFP8"):
        raise SystemExit(f"L{layer:02d}/{name}: expert dtype {spec!r} not BFP4/BFP8")
    return spec


def convert_layer(
    ck: Checkpoint,
    w: Writer,
    layer: int,
    expert_dtype: str,
    dtype_map: dict | None = None,
    store: Checkpoint | None = None,
    stick_dtypes: tuple[str, ...] = (),
):
    """Writes every cache tensor of one layer; experts in ``expert_dtype`` unless ``dtype_map`` overrides. Experts
    whose gate_up and down share a dtype in ``stick_dtypes`` are stored in the stick layout (convert/stick.py)."""
    convert_layer_dense(ck, w, layer)
    p = f"{P}layers.{layer}."
    for name, src_name, is_w13 in (("experts.w_gate_up", "w13", True), ("experts.w_down", "w2", False)):
        dtype = expert_tensor_dtype(dtype_map or {}, layer, name, expert_dtype)
        convert_experts(ck, w, layer, name, f"{p}mlp.experts.{src_name}_weight", is_w13, dtype, store)
    stick_layer(w, layer, stick_dtypes)


def convert_layer_dense(ck: Checkpoint, w: Writer, layer: int):
    """Every cache tensor of one layer except the routed experts."""
    p = f"{P}layers.{layer}."
    is_global = layer in GLOBAL_LAYERS

    def get(*keys):
        return [ck.get(p + k) for k in keys], [p + k for k in keys]

    def put(name, tensor, dtype, layout, src):
        w.write(name, layer, tensor, dtype, layout, src)

    keys = ["attn.wq_du.weight", "attn.gate_attn.weight", "attn.wr_du.weight", "attn.attn_sink_proj.weight"]
    ts, src = get(*keys)
    qgrs = torch.cat([t.T for t in ts], dim=1)
    put("attn.w_qgrs", F.pad(qgrs, (0, QGRS_COLS - qgrs.shape[1])), "BFP8", "TILE", src)
    for kv in "kv":
        if is_global:
            (lin, proj, static), src = get(
                f"attn.w{kv}_dv.weight", f"attn.{kv}_sconv.weight_proj", f"attn.{kv}_sconv.static_w"
            )
            put(f"attn.w_{kv}", torch.cat([lin.T, proj.T], dim=1), "BFP8", "TILE", src[:2])
            put(f"attn.{kv}_conv", static, "BF16", "TILE", src[2:])  # already lag order: [0] = current token
        else:
            (lin, conv), src = get(f"attn.w{kv}_dv.weight", f"attn.{kv}_sconv.weight")
            put(f"attn.w_{kv}", lin.T, "BFP8", "TILE", src[:1])
            put(f"attn.{kv}_conv", lag_taps(conv), "BF16", "TILE", src[1:])
    (t,), src = get("attn.wo_ud.weight")
    put("attn.w_o", t.T, "BFP8", "TILE", src)
    for name, key in (
        ("attn.q_clip_scale", "attn.q_l2_clip.scale"),
        ("attn.rel_clip_scale", "attn.rel_l2_clip.scale"),
        ("attn.rel_proj", "attn.rel_logits_proj.proj"),
    ):
        (t,), src = get(key)
        put(name, t, "BF16", "TILE", src)
    for name, key in (("post_attn_conv", "attn_sconv.weight"), ("post_mlp_conv", "mlp_sconv.weight")):
        (t,), src = get(key)
        put(name, lag_taps(t), "FP32", "TILE", src)
    for name in (["q_input_norm", "k_input_norm", "v_input_norm"] if is_global else ["attn_norm"]) + ["mlp_norm"]:
        (t,), src = get(name + ".weight")
        put(name, t.reshape(72, 32), "BF16", "RM", src)
    if layer > 0:
        key = f"{P}attn_res.attn_queries.{layer - 1}"
        q = ck.get(key)
        if q.reshape(-1, HIDDEN).shape[0] != (3 if is_global else 1):
            raise ValueError(f"{key}: shape {tuple(q.shape)} does not match layer type")
        put("attn_res_q", rows32(q), "FP32", "TILE", [key])
    key = f"{P}attn_res.mlp_queries.{layer}"
    put("mlp_res_q", rows32(ck.get(key)), "FP32", "TILE", [key])
    (gw, gb), src = get("mlp.gate.weight", "mlp.gate.bias")
    put("router.w", gw.float().T, "FP32", "TILE", src[:1])
    put("router.bias", rows32(gb), "FP32", "TILE", src[1:])
    if layer in LOOP_LAYERS:
        for m, ctl in (("q", "attn"), ("k", "k"), ("v", "v"), ("mlp", "mlp")):
            (down, forget, gain), src = get(*(f"controllers.{ctl}.{k}_weight" for k in ("down", "forget", "gain")))
            put(f"msg_gate.{m}.down", down.T, "BF16", "TILE", src[:1])
            put(f"msg_gate.{m}.forget", forget.float().T, "FP32", "TILE", src[1:2])
            put(f"msg_gate.{m}.gain", gain.float().T, "FP32", "TILE", src[2:])


def stick_layer(w: Writer, layer: int, stick_dtypes) -> bool:
    """Rewrite the layer's two expert tensors in the stick layout, in place, if they share a dtype in
    ``stick_dtypes`` (the fused MoE kernel needs one dtype for both)."""
    ex = [i for i, e in enumerate(w.entries) if e["layer"] == layer and e["name"].startswith("experts.")]
    if len({w.entries[i]["dtype"] for i in ex}) != 1 or w.entries[ex[0]]["dtype"] not in stick_dtypes:
        return False
    for i in ex:
        f = w.out / w.entries[i]["file"]
        w.entries[i] = write_sticks(f, f, w.entries[i])
    return True


def convert_experts(
    ck: Checkpoint, w: Writer, layer: int, name: str, base: str, is_w13: bool, dtype: str, store: Checkpoint | None
):
    """``store``: a transformers bf16 export (``--bf16-store``) whose ``gate_up_proj`` / ``down_proj`` [E, out, in]
    (gate rows first) replace the dequant of ``ck``'s NVFP4 / MXFP8 experts."""
    nvfp4 = ck.has(base + "_amax")
    if nvfp4 == (layer in MXFP8_LAYERS):
        raise ValueError(f"{base}: format {'NVFP4' if nvfp4 else 'MXFP8'} unexpected for layer {layer}")
    amax = ck.get(base + "_amax") if nvfp4 else None
    denominator = ck.get(base + "_outer_scale_denominator") if nvfp4 else None
    store_key = f"model.language_model.layers.{layer}.mlp.experts.{'gate_up_proj' if is_w13 else 'down_proj'}"
    full = None
    for s in range(0, N_EXPERTS, EXPERT_CHUNK):
        e = slice(s, s + EXPERT_CHUNK)
        if store is not None:
            t = store.experts(store_key, s, e.stop)  # already de-interleaved
        else:
            t = dequant_experts(
                ck.experts(base, s, e.stop),
                ck.experts(base + "_scale", s, e.stop),
                amax[e] if nvfp4 else None,
                denominator,
            )
            if is_w13:
                t = deinterleave_gate_up(t)  # rows gate 0:1152, up 1152:2304
        if full is None:
            full = torch.empty(1, N_EXPERTS, t.shape[2], t.shape[1], dtype=torch.bfloat16)
        full[0, e] = t.transpose(1, 2)  # [in, out]
        del t
    pcc_bf4, min_bf4 = roundtrip_pcc(full, "BFP4")
    reason = None
    if dtype == "BFP4" and pcc_bf4 < MIN_WEIGHT_PCC:
        dtype, reason = "BFP8", f"pcc_bf4 {pcc_bf4:.5f} < {MIN_WEIGHT_PCC}"
    elif dtype == "BFP8" and nvfp4:
        reason = "experts promoted to BFP8 (--bfp8-expert-layers / --dtype-map)"
    pcc, min_pcc = (pcc_bf4, min_bf4) if dtype == "BFP4" else roundtrip_pcc(full, "BFP8")
    src = [base, base + "_scale"] + ([base + "_amax", base + "_outer_scale_denominator"] if nvfp4 else [])
    if store is not None:
        src = [f"{store.root}/{store_key}"]
    w.write(
        name,
        layer,
        full,
        dtype,
        "TILE",
        src,
        pcc=pcc,
        pcc_bf4=pcc_bf4,
        pcc_min_expert=round(min_pcc, 7),
        pcc_bf4_min_expert=round(min_bf4, 7),
        source_format="NVFP4" if nvfp4 else "MXFP8",
    )
    if reason:
        w.entries[-1].update(fallback=True, fallback_reason=reason)


DTYPE_MAP_KEY = re.compile(r"L(\d\d)/(experts\.w_gate_up|experts\.w_down)")


def check_dtype_map(dtype_map: dict):
    """Refuse keys the converter would not apply as written: only ``Lxx/experts.w_gate_up`` and
    ``Lxx/experts.w_down`` of the NVFP4 layers 0-19. Layers 20-23 come as MXFP8 and are always stored BFP8."""
    bad = []
    for k in dtype_map:
        m = DTYPE_MAP_KEY.fullmatch(k)
        if not m or int(m[1]) >= N_LAYERS:
            bad.append(f"{k} (not Lxx/experts.w_gate_up or Lxx/experts.w_down with xx < {N_LAYERS})")
        elif int(m[1]) in MXFP8_LAYERS:
            bad.append(f"{k} (layers {MXFP8_LAYERS.start}-{MXFP8_LAYERS.stop - 1} are MXFP8, always BFP8)")
    if bad:
        raise SystemExit(f"--dtype-map: bad keys: {bad}")


def unit_key(unit: str, expert_dtype: str | None, dtype_map: dict, stick_dtypes, expert_source: str | None) -> str:
    """Hash of everything that decides what ``unit`` writes (besides the checkpoint, checked separately)."""
    spec = {"version": CONVERTER_VERSION, "unit": unit}
    if unit != "global":
        spec |= {
            "expert_dtype": expert_dtype,
            "dtype_map": {k: v for k, v in sorted(dtype_map.items()) if k.startswith(unit + "/")},
            "stick_dtypes": sorted(stick_dtypes),
            "expert_source": expert_source,
        }
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tt_metal_sha() -> str:
    here = Path(__file__).parent
    sha = subprocess.run(["git", "-C", here, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", here, "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True
    ).stdout.strip()
    return sha + ("-dirty" if dirty else "")


def summarize(entries: list[dict]) -> str:
    by_class: dict[str, list] = {}
    for e in entries:
        by_class.setdefault(f"{e['name']} {e['dtype']}", []).append(e["pcc"])
    lines = [f"{'class':40s} {'n':>3s} {'min':>9s} {'max':>9s}"]
    for k, v in sorted(by_class.items()):
        lines.append(f"{k:40s} {len(v):3d} {min(v):9.6f} {max(v):9.6f}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", type=Path, default=os.environ.get("INKLING_CKPT"), help="checkpoint dir ($INKLING_CKPT)")
    ap.add_argument("--out", type=Path, default=os.environ.get("INKLING_CACHE"), help="cache dir ($INKLING_CACHE)")
    ap.add_argument("--layers", default=",".join(map(str, range(N_LAYERS))), help="comma list, default all")
    ap.add_argument("--skip-global", action="store_true")
    ap.add_argument(
        "--bfp8-expert-layers",
        default=",".join(map(str, DEFAULT_BFP8_EXPERT_LAYERS)),
        help="layers 0-19 (or all) whose experts are stored BFP8 instead of BFP4 (layers 20-23 are always BFP8)",
    )
    ap.add_argument(
        "--dtype-map",
        type=Path,
        default=Path(__file__).with_name("default_dtype_map.json"),
        help='JSON {"L08/experts.w_down": "BFP8" | {"default": "BFP4", "BFP8": [expert ids]}}, over --bfp8-expert-layers;'
        " default = the shipped pick (MoE-block PCC ranking); pass an empty file ({}) for none",
    )
    ap.add_argument("--force", action="store_true", help="reconvert listed units even if the manifest has them")
    ap.add_argument(
        "--stick-experts",
        default="",
        help="dtypes (BFP4,BFP8) whose layers, when gate_up and down share that dtype, get the stick expert layout",
    )
    ap.add_argument(
        "--bf16-store",
        type=Path,
        help="read experts from this transformers bf16 export instead of dequantizing --ckpt (e.g. for cache/v1-b8)",
    )
    args = ap.parse_args(argv)
    if args.ckpt is None or args.out is None:
        ap.error("--ckpt and --out are required (or set INKLING_CKPT and INKLING_CACHE)")
    dtype_map = json.loads(args.dtype_map.read_text()) if args.dtype_map else {}
    check_dtype_map(dtype_map)

    ck = Checkpoint(args.ckpt)
    store = Checkpoint(args.bf16_store) if args.bf16_store else None
    header_sha = ck.header_sha256()
    manifest_path = args.out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if manifest and manifest["source_header_sha256"] != header_sha:
        raise SystemExit(f"{manifest_path} was written from another checkpoint; use a new --out")
    manifest = manifest or {
        "schema": 1,
        "source": str(args.ckpt),
        "source_header_sha256": header_sha,
        "config_sha256": sha256_file(args.ckpt / "config.json"),
        "tt_metal_sha": tt_metal_sha(),
        "expert_source": str(args.bf16_store) if args.bf16_store else "dequant of --ckpt",
        "min_weight_pcc": MIN_WEIGHT_PCC,
        "units_done": [],
        "unit_keys": {},
        "unit_seconds": {},
        "tensors": [],
    }
    w = Writer(args.out)
    w.entries = manifest["tensors"]
    stick_dtypes = tuple(x for x in args.stick_experts.split(",") if x)
    if set(stick_dtypes) - {"BFP4", "BFP8"}:
        raise SystemExit(f"--stick-experts: BFP4 and/or BFP8, got {args.stick_experts}")
    if args.bfp8_expert_layers == "all":
        args.bfp8_expert_layers = ",".join(map(str, range(N_LAYERS)))
    bfp8 = {int(x) for x in args.bfp8_expert_layers.split(",") if x} | set(MXFP8_LAYERS)
    layers = [int(x) for x in args.layers.split(",") if x]
    if bad := [x for x in layers + sorted(bfp8) if not 0 <= x < N_LAYERS]:
        raise SystemExit(f"--layers / --bfp8-expert-layers: no layer {bad} (0-{N_LAYERS - 1})")
    units = ([] if args.skip_global else ["global"]) + [f"L{x:02d}" for x in layers]
    expert_source = str(args.bf16_store) if args.bf16_store else None
    manifest.setdefault("unit_keys", {})

    def key(unit):
        layer = None if unit == "global" else int(unit[1:])
        dtype = None if layer is None else "BFP8" if layer in bfp8 else "BFP4"
        return unit_key(unit, dtype, dtype_map, stick_dtypes, expert_source)

    def save():
        manifest["fallback_layers"] = sorted({e["layer"] for e in w.entries if e["fallback"]})
        manifest["expert_dtype"] = {
            f"L{e['layer']:02d}/{e['name']}": e["dtype"] for e in w.entries if e["name"].startswith("experts.")
        }
        manifest["expert_layout"] = {
            f"L{e['layer']:02d}/{e['name']}": e["layout"] for e in w.entries if e["name"].startswith("experts.")
        }
        all_units = {"global"} | {f"L{i:02d}" for i in range(N_LAYERS)}
        manifest["complete"] = set(manifest["units_done"]) >= all_units and all(
            u in manifest["unit_keys"] for u in all_units
        )
        manifest["peak_rss_gib_last_run"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
        args.out.mkdir(parents=True, exist_ok=True)
        tmp = manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest, indent=1))
        tmp.replace(manifest_path)

    for unit in units:
        if unit in manifest["units_done"] and not args.force:
            if manifest["unit_keys"].get(unit) == key(unit):
                print(f"{unit}: already in manifest with the same settings, skipped", flush=True)
                continue
            print(f"{unit}: settings or converter version changed, reconverting", flush=True)
        t0 = time.time()
        layer = None if unit == "global" else int(unit[1:])
        # Not done until rewritten: a crash mid-unit leaves the manifest incomplete, so the loader refuses it.
        manifest["units_done"] = [u for u in manifest["units_done"] if u != unit]
        manifest["unit_keys"].pop(unit, None)
        w.entries[:] = [e for e in w.entries if e["layer"] != layer]  # drop a half-written unit
        save()
        if layer is None:
            convert_global(ck, w)
        else:
            convert_layer(ck, w, layer, "BFP8" if layer in bfp8 else "BFP4", dtype_map, store, stick_dtypes)
            m = manifest.setdefault("dtype_map", {})
            for k in [k for k in m if k.startswith(f"{unit}/")]:
                del m[k]
            m.update({k: v for k, v in dtype_map.items() if k.startswith(f"{unit}/")})
        manifest["units_done"].append(unit)
        manifest["unit_keys"][unit] = key(unit)
        manifest["unit_seconds"][unit] = round(time.time() - t0, 1)
        save()
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
        print(f"{unit}: {time.time() - t0:.0f}s, peak RSS {rss:.2f} GiB", flush=True)
    save()
    print(summarize(w.entries))
    print(f"BFP8-promoted layers: {manifest['fallback_layers']}; complete: {manifest['complete']}")
