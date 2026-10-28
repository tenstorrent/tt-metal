# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""E6b datatype sweep harness for the vision and audio towers (the text sweep is dtype_sweep.py).

    python .../tests/perf/dtype_sweep_towers.py list --tower vision
    python .../tests/perf/dtype_sweep_towers.py run --tower vision --candidates all --out sweep_vision.jsonl
    python .../tests/perf/dtype_sweep_towers.py probe --tower audio --candidates base,x --out probe_audio.jsonl
    python .../tests/perf/dtype_sweep_towers.py report --tower vision --inputs a.jsonl --probe p.jsonl \
        --doc-dir .../doc/datatype_sweep --selected <id>

A candidate is a delta over the tower's E5 policy (`precision.E5_VISION_POLICY` / `E5_AUDIO_POLICY`) in
the `selected_precision_config.json` schema (tt/precision.py); the text section is the selected text
policy. Every candidate is built through `VisionEncoder.from_pretrained(precision=...)` /
`AudioEncoder.from_pretrained(precision=...)` (`probe`: `EmbeddingGemma2.from_pretrained(precision=...)`),
the construction path the default model takes with the selected file.

`run` (screen, one device session per candidate):
  accuracy, eager (trace replay == eager bit for bit: test_vision_trace / test_audio_trace), against the
  live HF fp32 tower reference, on the cases and metrics of tests/pcc/<tower> (bars as the tests):
    vision  patch embedder PCC (every case), layer 0 PCC (every case), every layer on LAYER_CASES,
            attention / MLP of layers 0 / 7 / 15 on LAYER_CASES (>= 0.995); encoder-stack PCC (>= 0.99);
            soft-token PCC per image (>= 0.99); worst soft-token cosine (reported, not gated)
    audio   subsample PCC (every case), layer 0 PCC (every case), every layer on LAYER_CASES, FFN1 / FFN2 /
            attention / lconv of layers 0 / 5 / 11 on LAYER_CASES, output_proj / embed_audio on
            LAYER_CASES (>= 0.995); soft-token PCC per clip (>= 0.99); worst soft-token cosine (reported)
  performance: traced device time per forward (execute_trace + one sync per window, inputs resident):
    vision 1x2560, 4x2560, 8x1280; audio 1x256, 1x768, 4x256. Burst median of `--reps` windows after
    `--idle` s idle, then sustained after a `--soak` s soak. Ranking aggregate (fixed before the sweep):
    vision burst 1x2560 + 4x2560; audio burst 1x768 + 4x256.
`probe` (finalists): the full model with the candidate, on inputs outside the test set (probe_inputs) and
  on the tower's test_full_model items: tower soft-token PCC / worst-token cosine vs the HF fp32 tower,
  end-to-end embedding cosine vs HF fp32 `EmbeddingGemma2Model` + ST pooling, test_retrieval argmax.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import torch

BAR_MODULE = 0.995
BAR_STACK = 0.99
BAR_SOFT = 0.99
BAR_COS = 0.99

PERF_SHAPES = {"vision": ((1, 2560), (4, 2560), (8, 1280)), "audio": ((1, 256), (1, 768), (4, 256))}
RANK_SHAPES = {"vision": ((1, 2560), (4, 2560)), "audio": ((1, 768), (4, 256))}

VISION_GROUPS = ("attn_qkv", "attn_o", "mlp_gate_up", "mlp_down", "patch_embed", "projector")
VISION_LAYER_GROUPS = ("attn_qkv", "attn_o", "mlp_gate_up", "mlp_down")
AUDIO_GROUPS = (
    "subsample_conv",
    "subsample_proj",
    "attn_qkv",
    "attn_o",
    "ffn_up",
    "ffn_down",
    "lconv_start",
    "lconv_end",
    "output_proj",
    "projector",
)
AUDIO_PROJ_GROUPS = AUDIO_GROUPS[1:]  # every audio matmul group except the conv


def _mm(base: dict, **kw) -> dict:
    return {**base, **kw}


def _w(tower: str, groups, **kw) -> dict:
    from models.demos.blackhole.embedding_gemma2.tt.precision import E5_AUDIO_POLICY, E5_VISION_POLICY

    base = (E5_VISION_POLICY if tower == "vision" else E5_AUDIO_POLICY)["weights"]
    return {"weights": {g: _mm(base[g], **kw) for g in groups}}


def _vision_candidates() -> dict:
    c = {"base": ("E5b vision policy: bf16 weights, HiFi2 + fp32 acc (patch / projector HiFi4)", {})}
    t = "vision"
    c["bfp8w_all"] = ("every vision weight bfp8 (same fidelities)", _w(t, VISION_GROUPS, dtype="bfp8"))
    groups = {
        "qkv": ["attn_qkv"],
        "o": ["attn_o"],
        "gu": ["mlp_gate_up"],
        "down": ["mlp_down"],
        "patch": ["patch_embed"],
        "proj": ["projector"],
    }
    for k, gs in groups.items():
        c[f"bfp8_{k}"] = (f"{'+'.join(gs)} bfp8 weights", _w(t, gs, dtype="bfp8"))
        c[f"bfp4_{k}"] = (f"{'+'.join(gs)} bfp4 weights", _w(t, gs, dtype="bfp4"))
        c[f"bfp4_{k}_lofi"] = (f"{'+'.join(gs)} bfp4 weights, LoFi", _w(t, gs, dtype="bfp4", fidelity="LoFi"))
    c["lofi_attn"] = ("attn_qkv + attn_o LoFi (bf16 weights)", _w(t, ["attn_qkv", "attn_o"], fidelity="LoFi"))
    c["lofi_mlp"] = ("mlp_gate_up + mlp_down LoFi (bf16 weights)", _w(t, ["mlp_gate_up", "mlp_down"], fidelity="LoFi"))
    c["lofi_all"] = ("every layer matmul LoFi (bf16 weights)", _w(t, VISION_LAYER_GROUPS, fidelity="LoFi"))
    c["hifi2_patch_proj"] = ("patch_embed + projector HiFi2", _w(t, ["patch_embed", "projector"], fidelity="HiFi2"))
    c["hifi4_all"] = ("every matmul HiFi4 (accuracy-up option)", _w(t, VISION_GROUPS, fidelity="HiFi4"))
    c["nofp32acc"] = ("layer matmuls without fp32 accumulation", _w(t, VISION_LAYER_GROUPS, fp32_acc=False))
    sd = {"fidelity": "HiFi2", "fp32_acc": True, "exp_approx": False}
    c["sdpa_nofp32acc"] = ("SDPA fp32_dest_acc off", {"sdpa": {**sd, "fp32_acc": False}})
    c["sdpa_exp_approx"] = ("SDPA exp_approx_mode on", {"sdpa": {**sd, "exp_approx": True}})
    c["sdpa_lofi"] = ("SDPA LoFi", {"sdpa": {**sd, "fidelity": "LoFi"}})
    c["sdpa_fast"] = ("SDPA fp32 acc off + exp_approx", {"sdpa": {**sd, "fp32_acc": False, "exp_approx": True}})
    c["norms_hifi2"] = ("norms HiFi2", {"norms": {"fidelity": "HiFi2", "fp32_acc": True}})
    c["act_bfp8_qkv"] = ("attn_qkv output bfp8", _w(t, ["attn_qkv"], out_dtype="bfp8"))
    c["act_bfp8_gu"] = ("mlp_gate_up outputs bfp8", _w(t, ["mlp_gate_up"], out_dtype="bfp8"))
    c["act_bfp8_qkv_gu"] = ("attn_qkv + mlp_gate_up outputs bfp8", _w(t, ["attn_qkv", "mlp_gate_up"], out_dtype="bfp8"))
    c["act_bfp8_layers"] = ("every layer projection output bfp8", _w(t, VISION_LAYER_GROUPS, out_dtype="bfp8"))
    c["mask_bf16"] = ("mask bf16", {"mask_dtype": "bf16"})
    return c


def _audio_candidates() -> dict:
    from models.demos.blackhole.embedding_gemma2.tt.precision import AUDIO_ACT_GROUPS

    t = "audio"
    c = {"base": ("E5c audio policy: bf16 weights, HiFi4 + fp32 acc, q/k/v fp32", {})}
    c["bfp8w_all"] = ("every audio projection weight bfp8 (HiFi4)", _w(t, AUDIO_PROJ_GROUPS, dtype="bfp8"))
    groups = {
        "qkv": ["attn_qkv"],
        "o": ["attn_o"],
        "ffn": ["ffn_up", "ffn_down"],
        "lconv": ["lconv_start", "lconv_end"],
        "sub": ["subsample_proj"],
        "proj": ["output_proj", "projector"],
    }
    for k, gs in groups.items():
        c[f"bfp8_{k}"] = (f"{'+'.join(gs)} bfp8 weights", _w(t, gs, dtype="bfp8"))
        c[f"bfp4_{k}"] = (f"{'+'.join(gs)} bfp4 weights", _w(t, gs, dtype="bfp4"))
        c[f"bfp4_{k}_lofi"] = (f"{'+'.join(gs)} bfp4 weights, LoFi", _w(t, gs, dtype="bfp4", fidelity="LoFi"))
    c["bfp8_conv"] = ("subsample conv weights bfp8", _w(t, ["subsample_conv"], dtype="bfp8"))
    c["hifi2_all"] = ("every audio projection HiFi2 (E5c E6 screen)", _w(t, AUDIO_PROJ_GROUPS, fidelity="HiFi2"))
    for k, gs in groups.items():
        c[f"hifi2_{k}"] = (f"{'+'.join(gs)} HiFi2", _w(t, gs, fidelity="HiFi2"))
    c["lofi_ffn"] = ("ffn_up + ffn_down LoFi", _w(t, ["ffn_up", "ffn_down"], fidelity="LoFi"))
    c["hifi2_bfp8_all"] = (
        "every projection bfp8 + HiFi2",
        _w(t, AUDIO_PROJ_GROUPS, dtype="bfp8", fidelity="HiFi2"),
    )
    c["attn_hifi2"] = ("band-attention block matmuls HiFi2", {"attention": {"fidelity": "HiFi2", "fp32_acc": True}})
    c["shift_lofi"] = ("lconv shift matmuls LoFi (E5c c25)", {"lconv_shift": {"fidelity": "LoFi", "fp32_acc": True}})
    c["shift_hifi2"] = ("lconv shift matmuls HiFi2", {"lconv_shift": {"fidelity": "HiFi2", "fp32_acc": True}})
    c["norms_sharded"] = (
        "block-sharded RMSNorm for <= 24 tile rows (E5c c21)",
        {"norms": {"fidelity": "HiFi4", "fp32_acc": True, "sharded": True}},
    )
    c["norms_hifi2"] = ("norms HiFi2", {"norms": {"fidelity": "HiFi2", "fp32_acc": True, "sharded": False}})
    c["qkv_out_bf16"] = ("fused QKV output bf16 (was fp32)", _w(t, ["attn_qkv"], out_dtype="bf16"))
    c["act_bfp8_ffn_up"] = ("ffn_up output bfp8", _w(t, ["ffn_up"], out_dtype="bfp8"))
    c["act_bfp8_lconv_start"] = ("lconv_start output bfp8", _w(t, ["lconv_start"], out_dtype="bfp8"))
    c["act_fp32"] = ("fp32 activations (accuracy-up, E3 act_dtype=float32)", _w(t, AUDIO_ACT_GROUPS, out_dtype="fp32"))
    return c


CANDIDATES = {"vision": _vision_candidates, "audio": _audio_candidates}
_REG: dict[str, dict] = {}


def candidates(tower: str) -> dict:
    if tower not in _REG:
        _REG[tower] = CANDIDATES[tower]()
    return _REG[tower]


def register_combo(tower: str, cid: str, description: str, delta: dict) -> None:
    candidates(tower)[cid] = (description, delta)


def candidate_dict(tower: str, cid: str) -> dict:
    """Full precision config: the selected file with the tower section = E5 policy + the candidate delta."""
    from models.demos.blackhole.embedding_gemma2.tt.precision import (
        DEFAULT_PRECISION_PATH,
        E5_AUDIO_POLICY,
        E5_VISION_POLICY,
        deep_merge,
    )

    description, delta = candidates(tower)[cid]
    d = json.loads(DEFAULT_PRECISION_PATH.read_text())
    base = E5_VISION_POLICY if tower == "vision" else E5_AUDIO_POLICY
    d[tower] = deep_merge(base, delta)
    d["config_id"] = f"{tower}:{cid}"
    d["description"] = description
    return d


# ---------------------------------------------------------------------- helpers
def _pcc(a, b) -> float:
    from models.demos.blackhole.embedding_gemma2.tests.common import pcc

    return float(pcc(a, b))


def _worst(rows):
    r = min(rows, key=lambda t: t[0])
    return {"value": r[0], "where": list(r[1:])}


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001 - provenance only
        return "unknown"


def _dtype_name(t) -> str:
    return str(t.dtype).split(".")[-1]


def _fid(ckc) -> str:
    return str(ckc.math_fidelity).split(".")[-1]


# ---------------------------------------------------------------------- vision screen
class VisionRefs:
    """HF fp32 vision references of the test cases (tests/pcc/vision), computed once per process."""

    def __init__(self, snapshot):
        from models.demos.blackhole.embedding_gemma2.reference.hf_vision_reference import load_reference_vision

        self.snapshot = snapshot
        self.tower, self.embed, _ = load_reference_vision(snapshot)
        self.cache = {}

    def get(self, case):
        from models.demos.blackhole.embedding_gemma2.reference.hf_vision_reference import run_vision_reference
        from models.demos.blackhole.embedding_gemma2.tests.pcc.vision.vision_common import LAYER_CASES, load_case

        if case not in self.cache:
            pv, pid = load_case(str(self.snapshot), case)
            sub = (0, 7, 15) if case in LAYER_CASES else ()
            layers = None if case in LAYER_CASES else (0,)
            with torch.no_grad():
                ref = run_vision_reference(
                    self.tower, self.embed, pv, pid, capture_layers=layers, capture_sublayers=sub
                )
            self.cache[case] = (pv, pid, ref)
        return self.cache[case]

    def tower_ref(self, pv, pid):
        from models.demos.blackhole.embedding_gemma2.reference.hf_vision_reference import run_vision_reference

        with torch.no_grad():
            return run_vision_reference(self.tower, self.embed, pv, pid, capture_layers=())


def vision_accuracy(device, enc, refs: VisionRefs) -> dict:
    from models.demos.blackhole.embedding_gemma2.tests.pcc.vision.vision_common import (
        ALL_CASES,
        LAYER_CASES,
        download_rows,
        upload_rows,
    )
    from models.demos.blackhole.embedding_gemma2.tt.vision.vision_model import key_padding_mask

    module, layer, stack, soft, tokcos = [], [], [], [], []
    m = enc.model
    for case in ALL_CASES:
        pv, pid, ref = refs.get(case)
        N = pid.shape[1]
        host = enc.prepare_inputs(pv, pid)
        x = enc.to_device(host)
        v = ref.valid
        pe = download_rows(m.patch_embedder(x.pixel_values, x.pos_x, x.pos_y), N)
        module.append((_pcc(ref.patch_embeds[v], pe[v]), case, "patch_embed"))
        mask = key_padding_mask(x.keep, m.mask_dtype)
        layers = range(len(m.layers)) if case in LAYER_CASES else (0,)
        for i in layers:
            y = m.layers[i](upload_rows(device, ref.layer_in[i], host.seq_len), mask, x.cos, x.sin)
            layer.append((_pcc(ref.layer_out[i][v], download_rows(y, N)[v]), case, i))
        if case in LAYER_CASES:
            for i in (0, 7, 15):
                a = m.layers[i].self_attn(upload_rows(device, ref.attn_in[i], host.seq_len), mask, x.cos, x.sin)
                module.append((_pcc(ref.attn_out[i][v], download_rows(a, N)[v]), case, f"attn{i}"))
                f = m.layers[i].mlp(upload_rows(device, ref.mlp_in[i], host.seq_len))
                module.append((_pcc(ref.mlp_out[i][v], download_rows(f, N)[v]), case, f"mlp{i}"))
        h = download_rows(m.forward_hidden(x), N)
        stack.append((_pcc(ref.encoder_out[v], h[v]), case))
        out = enc.encode(pv, pid)
        for b, (tt, hf) in enumerate(zip(out, ref.soft_tokens)):
            soft.append((_pcc(hf, tt), case, b))
            c = torch.nn.functional.cosine_similarity(hf, tt, dim=-1)
            tokcos.append((float(c.min()), case, b, int(c.argmin())))
    res = {
        "layer_pcc": _worst(layer),
        "module_pcc": _worst(module),
        "stack_pcc": _worst(stack),
        "soft_pcc": _worst(soft),
        "token_cos_min": _worst(tokcos),
        "soft_pcc_per_case": {f"{c}/{b}": v for v, c, b in soft},
        "token_cos_per_case": {f"{c}/{b}": v for v, c, b, _ in tokcos},
    }
    return _screen(res)


def _screen(res: dict) -> dict:
    fails = []
    for k, bar in (("layer_pcc", BAR_MODULE), ("module_pcc", BAR_MODULE), ("stack_pcc", BAR_STACK)):
        if k in res and res[k]["value"] < bar:
            fails.append(f"{k} {res[k]['value']:.5f} < {bar} at {res[k]['where']}")
    if res["soft_pcc"]["value"] < BAR_SOFT:
        fails.append(f"soft_pcc {res['soft_pcc']['value']:.5f} < {BAR_SOFT} at {res['soft_pcc']['where']}")
    res["screen_pass"] = not fails
    res["failing"] = fails
    return res


def vision_construction(model) -> dict:
    ev = {"layers": {}}
    for i, L in enumerate(model.layers):
        a, f = L.self_attn, L.mlp
        ev["layers"][i] = {
            "wqkv": _dtype_name(a.wqkv),
            "wo": _dtype_name(a.wo),
            "w_gate": _dtype_name(f.w_gate),
            "w_up": _dtype_name(f.w_up),
            "w_down": _dtype_name(f.w_down),
            "qkv_policy": a.qkv_policy.to_dict(),
            "o_policy": a.o_policy.to_dict(),
            "gate_up_policy": f.gate_up_policy.to_dict(),
            "down_policy": f.down_policy.to_dict(),
            "sdpa": {
                "fidelity": _fid(a.sdpa_compute_kernel_config),
                "fp32_acc": bool(a.sdpa_compute_kernel_config.fp32_dest_acc_en),
                "exp_approx": a.sdpa_policy.exp_approx,
            },
            "norm_fidelity": _fid(L.input_layernorm.compute_kernel_config),
            "norm_fp32_acc": bool(L.input_layernorm.compute_kernel_config.fp32_dest_acc_en),
        }
    ev["patch_embed"] = {"weight": _dtype_name(model.patch_embedder.w_in), **model.patch_embedder.policy.to_dict()}
    ev["projector"] = {"weight": _dtype_name(model.embedder.w_proj), **model.embedder.policy.to_dict()}
    ev["mask_dtype"] = str(model.mask_dtype).split(".")[-1]
    return ev


# ---------------------------------------------------------------------- audio screen
class AudioRefs:
    def __init__(self, snapshot):
        from models.demos.blackhole.embedding_gemma2.reference.hf_audio_reference import load_reference_audio

        torch.set_num_threads(max(torch.get_num_threads(), 16))
        self.snapshot = snapshot
        self.tower, self.embed, _ = load_reference_audio(snapshot)
        self.cache = {}

    def get(self, case):
        from models.demos.blackhole.embedding_gemma2.reference.hf_audio_reference import run_audio_reference
        from models.demos.blackhole.embedding_gemma2.tests.pcc.audio.audio_common import (
            LAYER_CASES,
            SUBLAYERS,
            load_case,
        )

        if case not in self.cache:
            feats, mask = load_case(str(self.snapshot), case)
            sub = SUBLAYERS if case in LAYER_CASES else ()
            layers = None if case in LAYER_CASES else (0, 11)
            with torch.no_grad():
                ref = run_audio_reference(
                    self.tower, self.embed, feats, mask, capture_layers=layers, capture_sublayers=sub
                )
            self.cache[case] = (feats, mask, ref)
        return self.cache[case]

    def tower_ref(self, feats, mask):
        from models.demos.blackhole.embedding_gemma2.reference.hf_audio_reference import run_audio_reference

        with torch.no_grad():
            return run_audio_reference(self.tower, self.embed, feats, mask, capture_layers=())


def audio_accuracy(device, enc, refs: AudioRefs) -> dict:
    import ttnn
    from models.demos.blackhole.embedding_gemma2.tests.pcc.audio.audio_common import (
        ALL_CASES,
        LAYER_CASES,
        SUBLAYERS,
        device_inputs,
        download_rows,
        upload_rows,
        valid,
    )
    from models.demos.blackhole.embedding_gemma2.tt.audio.attention import attention_mask

    module, layer, soft, tokcos = [], [], [], []
    m = enc.model
    for case in ALL_CASES:
        feats, mask, ref = refs.get(case)
        host, x = device_inputs(enc, feats, mask)
        N = ref.token_mask.shape[1]
        S = host.seq_len
        tm = ref.token_mask
        sub = download_rows(m.subsample(x), N)
        module.append((_pcc(*valid(ref.subsample_out, sub, tm)), case, "subsample"))
        layers = range(len(m.layers)) if case in LAYER_CASES else (0,)
        for i in layers:
            y = m.layers[i](upload_rows(device, ref.layer_in[i], S), x.attn_mask)
            layer.append((_pcc(*valid(ref.layer_out[i], download_rows(y, N), tm)), case, i))
        if case in LAYER_CASES:
            amask = attention_mask(tm.sum(-1).tolist(), S, enc.config.attention)
            am = ttnn.from_torch(amask, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
            for i in SUBLAYERS:
                L = m.layers[i]
                for which, xin, yref in (
                    ("ffn1", ref.ffn1_in[i], ref.ffn1_out[i]),
                    ("ffn2", ref.ffn2_in[i], ref.ffn2_out[i]),
                ):
                    y = getattr(L, which)(upload_rows(device, xin, S))
                    module.append((_pcc(*valid(yref, download_rows(y, N), tm)), case, f"{which}{i}"))
                y = L.attn(upload_rows(device, ref.attn_in[i], S), am)
                module.append((_pcc(*valid(ref.attn_out[i], download_rows(y, N), tm)), case, f"attn{i}"))
                y = L.lconv(upload_rows(device, ref.lconv_in[i], S))
                module.append((_pcc(*valid(ref.lconv_out[i], download_rows(y, N), tm)), case, f"lconv{i}"))
            last = len(m.layers) - 1
            y = m.output_proj(upload_rows(device, ref.layer_out[last], S))
            module.append((_pcc(*valid(ref.hidden, download_rows(y, N), tm)), case, "output_proj"))
            y = m.embedder(upload_rows(device, ref.hidden, S, dtype=ttnn.float32))
            module.append((_pcc(*valid(ref.soft_all, download_rows(y, N), tm)), case, "embed_audio"))
        out = enc.encode(feats, mask)
        for b, (tt, hf) in enumerate(zip(out, ref.soft_tokens)):
            soft.append((_pcc(hf, tt), case, b))
            c = torch.nn.functional.cosine_similarity(hf, tt, dim=-1)
            tokcos.append((float(c.min()), case, b, int(c.argmin())))
    res = {
        "layer_pcc": _worst(layer),
        "module_pcc": _worst(module),
        "soft_pcc": _worst(soft),
        "token_cos_min": _worst(tokcos),
        "soft_pcc_per_case": {f"{c}/{b}": v for v, c, b in soft},
        "token_cos_per_case": {f"{c}/{b}": v for v, c, b, _ in tokcos},
    }
    return _screen(res)


def audio_construction(model) -> dict:
    def lin(cl):
        return {"weight": _dtype_name(cl.weight), **cl.policy.to_dict(), "out": str(cl.out_dtype).split(".")[-1]}

    ev = {"layers": {}}
    for i, L in enumerate(model.layers):
        ev["layers"][i] = {
            "attn_qkv": {"weight": _dtype_name(L.attn.wqkv), **L.attn.qkv_policy.to_dict()},
            "attn_o": lin(L.attn.post),
            "ffn_up": lin(L.ffn1.ffw1),
            "ffn_down": lin(L.ffn1.ffw2),
            "lconv_start": lin(L.lconv.linear_start),
            "lconv_end": lin(L.lconv.linear_end),
            "attention_fidelity": _fid(L.attn.compute_kernel_config),
            "lconv_shift_fidelity": _fid(L.lconv.compute_kernel_config),
            "norm_fidelity": _fid(L.norm_out.compute_kernel_config),
            "norm_sharded": L.norm_out.sharded,
        }
    s = model.subsample
    ev["subsample"] = {
        "conv": s.conv_policy.to_dict(),
        "proj_weight": _dtype_name(s.w_in),
        "proj": s.proj_policy.to_dict(),
        "act_dtype": str(s.act_dtype).split(".")[-1],
    }
    ev["output_proj"] = {"weight": _dtype_name(model.output_proj.weight), **model.output_proj.policy.to_dict()}
    ev["projector"] = {"weight": _dtype_name(model.embedder.w_proj), **model.embedder.policy.to_dict()}
    return ev


# ---------------------------------------------------------------------- performance
def _window_ms(enc, b, s, n) -> float:
    import ttnn

    t0 = time.perf_counter()
    for _ in range(n):
        enc.execute_captured(b, s)
    ttnn.synchronize_device(enc.device)
    return (time.perf_counter() - t0) * 1e3 / n


def _perf_inputs(tower, snapshot, b, s):
    if tower == "vision":
        from models.demos.blackhole.embedding_gemma2.tests.perf.bench_vision import _inputs
    else:
        from models.demos.blackhole.embedding_gemma2.tests.perf.bench_audio import _inputs
    return _inputs(snapshot, b, s)


def run_perf(tower, enc, snapshot, args) -> dict:
    shapes = PERF_SHAPES[tower]
    enc.capture_all(shapes)
    out = {}
    try:
        for b, s in shapes:
            host = enc.prepare_inputs(*_perf_inputs(tower, snapshot, b, s))
            assert (host.batch, host.seq_len) == (b, s), (host.batch, host.seq_len)
            soft = enc.replay(host)[0]
            counts = host.num_soft_tokens if tower == "vision" else host.num_tokens
            rows = enc.read_rows(soft, counts)
            assert all(torch.isfinite(r).all() for r in rows)
            _window_ms(enc, b, s, 3)
            burst = []
            for _ in range(args.reps):
                time.sleep(args.idle)
                burst.append(_window_ms(enc, b, s, args.burst))
            t_end = time.perf_counter() + args.soak
            while time.perf_counter() < t_end:
                _window_ms(enc, b, s, 5)
            n_sus = max(5, int(args.sustained_ms / max(burst)))
            sus = [_window_ms(enc, b, s, n_sus) for _ in range(args.reps)]
            key = f"{b}x{s}"
            out[key] = {
                "burst_ms_median": statistics.median(burst),
                "burst_windows": burst,
                "sustained_ms_median": statistics.median(sus),
                "sustained_windows": sus,
                "sustained_replays_per_window": n_sus,
            }
            print(
                f"  PERF {key}: burst {out[key]['burst_ms_median']:.3f} sustained {out[key]['sustained_ms_median']:.3f}"
            )
            time.sleep(args.idle)
    finally:
        enc.release_traces()
    out["regime"] = f"{len(shapes)} traces captured in one process: {','.join(f'{b}x{s}' for b, s in shapes)}"
    out["rank_ms"] = sum(out[f"{b}x{s}"]["burst_ms_median"] for b, s in RANK_SHAPES[tower])
    return out


# ---------------------------------------------------------------------- run (screen)
def cmd_run(args) -> None:
    import ttnn
    from models.demos.blackhole.embedding_gemma2.conftest import DEVICE_PARAMS, MODEL_ID
    from models.demos.blackhole.embedding_gemma2.tt.precision import PrecisionConfig
    from models.demos.blackhole.embedding_gemma2.tt.weights import resolve_snapshot

    tower = args.tower
    _load_combos(args)
    ids = list(candidates(tower)) if args.candidates == "all" else args.candidates.split(",")
    snapshot = resolve_snapshot(MODEL_ID)
    if tower == "vision":
        from models.demos.blackhole.embedding_gemma2.tt.vision.vision_encoder import VisionEncoder as Enc
        from models.demos.blackhole.embedding_gemma2.tt.vision.weights import load_vision_state_dict as load_sd

        refs = VisionRefs(snapshot) if not args.no_acc else None
        acc_fn, con_fn = vision_accuracy, vision_construction
    else:
        from models.demos.blackhole.embedding_gemma2.tt.audio.audio_encoder import AudioEncoder as Enc
        from models.demos.blackhole.embedding_gemma2.tt.audio.weights import load_audio_state_dict as load_sd

        refs = AudioRefs(snapshot) if not args.no_acc else None
        acc_fn, con_fn = audio_accuracy, audio_construction
    state_dict = load_sd(snapshot)
    head = _git_head()
    for cid in ids:
        prec = candidate_dict(tower, cid)
        rec = {
            "id": cid,
            "tower": tower,
            "description": prec["description"],
            "delta": candidates(tower)[cid][1],
            "precision": prec[tower],
            "commit": head,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "host": "sjc-snva-tp100 p150a",
        }
        print(f"== {tower}:{cid}: {prec['description']}", flush=True)
        device = ttnn.open_device(device_id=0, **DEVICE_PARAMS)
        try:
            t0 = time.time()
            enc = Enc.from_pretrained(
                device, str(snapshot), state_dict=state_dict, precision=PrecisionConfig.from_dict(prec, cid)
            )
            rec["construction"] = con_fn(enc.model)
            if refs is not None:
                rec["accuracy"] = a = acc_fn(device, enc, refs)
                print(
                    f"  ACC layer {a['layer_pcc']['value']:.5f} module {a['module_pcc']['value']:.5f} "
                    f"soft {a['soft_pcc']['value']:.5f} tokcos {a['token_cos_min']['value']:.4f} -> "
                    f"{'PASS' if a['screen_pass'] else 'FAIL ' + '; '.join(a['failing'])}",
                    flush=True,
                )
            if not args.no_perf:
                rec["perf"] = run_perf(tower, enc, snapshot, args)
                print(f"  RANK {cid}: {rec['perf']['rank_ms']:.3f} ms", flush=True)
            rec["seconds"] = time.time() - t0
        except Exception as e:  # noqa: BLE001 - a candidate the runtime cannot build / run is recorded as blocked
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["traceback"] = traceback.format_exc()[-4000:]
            print(f"  ERROR {cid}: {rec['error'][:400]}", flush=True)
        finally:
            ttnn.close_device(device)
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")


def _load_combos(args) -> None:
    if getattr(args, "combos", ""):
        for cid, spec in json.loads(Path(args.combos).read_text()).items():
            register_combo(args.tower, cid, spec["description"], spec["delta"])


# ---------------------------------------------------------------------- probe (finalists)
def _synthetic_image(kind: str, w: int, h: int, seed: int):
    from PIL import Image

    g = np.random.default_rng(seed)
    if kind == "noise":
        a = g.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    elif kind == "gradient":
        x = np.linspace(0, 255, w)[None, :, None]
        y = np.linspace(0, 255, h)[:, None, None]
        a = np.concatenate(
            [np.broadcast_to(x, (h, w, 1)), np.broadcast_to(y, (h, w, 1)), np.full((h, w, 1), 128.0)], -1
        )
    elif kind == "checker":
        yy, xx = np.mgrid[0:h, 0:w]
        a = (((xx // 16) + (yy // 16)) % 2 * 255)[..., None].repeat(3, -1)
    elif kind == "blobs":
        yy, xx = np.mgrid[0:h, 0:w]
        a = np.zeros((h, w, 3))
        for c in range(3):
            cx, cy, r = g.uniform(0, w), g.uniform(0, h), g.uniform(0.1, 0.5) * min(w, h)
            a[..., c] = 255 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)))
    elif kind == "gray":
        a = np.linspace(0, 255, w)[None, :, None].repeat(h, 0).repeat(3, -1)
    else:
        raise KeyError(kind)
    return Image.fromarray(np.ascontiguousarray(a).astype(np.uint8))


_ROOT = Path(__file__).resolve().parents[6]
PROBE_IMAGE_FILES = {
    "demo_jpeg": "models/sample_data/demo.jpeg",
    "ilsvrc_48736": "models/sample_data/ILSVRC2012_val_00048736.JPEG",
    "cat": "models/sample_data/huggingface_cat_image.jpg",
    "street": "models/experimental/efficientdetd0/resources/vid_5_26800.jpg",
    "group": "models/experimental/yunet/test_images/group1.jpg",
    "chart_png": "models/perf/images/example_perf_report.png",
}
PROBE_SYNTH = {
    "noise_224": ("noise", 224, 224, 1),
    "gradient_wide_1024x256": ("gradient", 1024, 256, 2),
    "checker_tall_256x1024": ("checker", 256, 1024, 3),
    "noise_small_48": ("noise", 48, 48, 4),
    "blobs_640x480": ("blobs", 640, 480, 5),
    "extreme_wide_2000x100": ("gradient", 2000, 100, 6),
    "extreme_tall_100x1500": ("blobs", 100, 1500, 7),
    "gray_3000x2000": ("gray", 3000, 2000, 8),
}


def _video_frames(start_s: float, n: int, step_s: float = 1.0):
    import cv2

    cap = cv2.VideoCapture(str(_ROOT / "tech_reports/ttnn/images/ttnn-dashboard.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(n):
        idx = min(int(round((start_s + i * step_s) * fps)), total - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        if not ok:
            raise RuntimeError(f"cannot read frame {idx}")
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames)


def _synthetic_video(n: int, seed: int):
    g = np.random.default_rng(seed)
    h, w = 360, 640
    yy, xx = np.mgrid[0:h, 0:w]
    out = []
    for i in range(n):
        a = np.stack([(xx + 40 * i) % 256, (yy + 25 * i) % 256, np.full((h, w), 60 + 20 * i)], -1)
        a = a + g.normal(0, 8, size=a.shape)
        out.append(np.clip(a, 0, 255).astype(np.uint8))
    return np.stack(out)


def vision_probe_inputs() -> dict:
    """name -> full-model item (image or video). >= 15 inputs outside tests/pcc/vision's cases."""
    from PIL import Image

    from models.demos.blackhole.embedding_gemma2.reference.full_model_inputs import video

    items = {}
    for name, rel in PROBE_IMAGE_FILES.items():
        items[f"img_{name}"] = {"image": Image.open(_ROOT / rel).convert("RGB")}
    for name, spec in PROBE_SYNTH.items():
        items[f"img_{name}"] = {"image": _synthetic_image(*spec)}
    items["video_1f_t20"] = {"video": video(_video_frames(20.0, 1))}
    items["video_3f_t10"] = {"video": video(_video_frames(10.0, 3))}
    items["video_5f_t5_half"] = {"video": video(_video_frames(5.0, 5, 0.5))}
    items["video_4f_synth"] = {"video": video(_synthetic_video(4, 11))}
    return items


def _wav(i: int) -> np.ndarray:
    from models.demos.blackhole.embedding_gemma2.tests.pcc.audio.audio_common import SR, _read, _speech_files

    w = _read(str(_speech_files()[i]))
    assert SR == 16000
    return w


def audio_probe_inputs() -> dict:
    """name -> waveform (16 kHz mono float32). >= 15 inputs outside tests/pcc/audio's cases."""
    sr = 16000
    g = np.random.default_rng(1234)
    files = [_wav(i) for i in range(5)]
    items = {}
    for i, w in enumerate(files):
        items[f"clip{i}_full"] = w
    items["clip3_0.3s_at1.0"] = files[3][int(1.0 * sr) : int(1.3 * sr)]
    items["clip2_1.3s_at2.2"] = files[2][int(2.2 * sr) : int(3.5 * sr)]
    items["clip4_5s_at1.0"] = files[4][int(1.0 * sr) : int(6.0 * sr)]
    rev = np.concatenate(files[::-1])
    items["concat_rev_20s"] = rev[: 20 * sr]
    items["concat_rev_25s_at1.5"] = rev[int(1.5 * sr) : int(26.5 * sr)]
    items["clip1_x0.1"] = (0.1 * files[1]).astype(np.float32)
    items["clip0_x3_clipped"] = np.clip(3.0 * files[0], -1, 1).astype(np.float32)
    items["white_noise_2s"] = (0.1 * g.standard_normal(2 * sr)).astype(np.float32)
    brown = np.cumsum(g.standard_normal(4 * sr))
    items["brown_noise_4s"] = (0.5 * brown / np.abs(brown).max()).astype(np.float32)
    items["silence1s_speech2s"] = np.concatenate([np.zeros(sr, np.float32), files[2][: 2 * sr]])
    t = np.arange(int(1.5 * sr)) / sr
    items["tone_1khz_1.5s"] = (0.3 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    t = np.arange(8 * sr) / sr
    items["chirp_8s"] = (0.3 * np.sin(2 * np.pi * (100 * t + (6000 - 100) / 16 * t**2))).astype(np.float32)
    return {k: v.astype(np.float32) for k, v in items.items()}


FULL_ITEMS = {
    "vision": (
        "image_house",
        "image_pajamas",
        "image_dog",
        "image_handwriting",
        "image_house_caption",
        "image_dog_caption",
        "image_pajamas_caption",
        "video_text",
        "mixed_two_images",
    ),
    "audio": ("audio_speech2s", "audio_speech10s", "audio_tone3s", "audio_speech2s_text", "audio_speech10s_text"),
}


class FullRefs:
    """HF fp32 full-model embeddings and tower soft tokens of the probe / full-model items, cached on disk."""

    def __init__(self, tower, snapshot, cache_dir: Path):
        self.tower, self.snapshot, self.dir = tower, snapshot, Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._hf_full = None
        self._tower_refs = None
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(str(snapshot))

    def _cached(self, key, fn):
        p = self.dir / f"{self.tower}_{key}.pt"
        if p.exists():
            return torch.load(p)
        v = fn()
        torch.save(v, p)
        return v

    @property
    def hf_full(self):
        if self._hf_full is None:
            from models.demos.blackhole.embedding_gemma2.reference.hf_full_reference import load_reference_full_model

            self._hf_full = load_reference_full_model(self.snapshot)
        return self._hf_full

    @property
    def tower_refs(self):
        if self._tower_refs is None:
            self._tower_refs = (VisionRefs if self.tower == "vision" else AudioRefs)(self.snapshot)
        return self._tower_refs

    def embedding(self, name, item):
        from models.demos.blackhole.embedding_gemma2.reference.hf_full_reference import embed
        from models.demos.blackhole.embedding_gemma2.tt.embedding_model import process_item

        return self._cached(f"emb_{name}", lambda: embed(self.hf_full, process_item(self.processor, item))[0])

    def tower_soft(self, name, inputs):
        return self._cached(f"soft_{name}", lambda: self.tower_refs.tower_ref(*inputs).soft_tokens)


def _tower_inputs(tower, refs, item):
    """Processor tower inputs of one probe item: (pixel_values, position_ids) or (features, mask)."""
    processor = refs.processor
    if tower == "vision":
        if "image" in item:
            out = processor.image_processor(images=[item["image"]], return_tensors="pt")
            return out["pixel_values"].float(), out["image_position_ids"].long()
        out = processor.video_processor(videos=[item["video"]["array"]], return_tensors="pt")
        return out["pixel_values_videos"].float(), out["video_position_ids"].long()
    from models.demos.blackhole.embedding_gemma2.tests.pcc.audio.audio_common import feature_extractor

    fe = feature_extractor(str(refs.snapshot))
    out = fe([item["audio"]["array"]], sampling_rate=16000, return_tensors="pt")
    return out["input_features"].float(), out["input_features_mask"].bool()


def run_probe(tower, full, refs: FullRefs) -> dict:
    """Probe + full-model items: tower soft-token PCC / worst-token cosine, end-to-end embedding cosine."""
    from models.demos.blackhole.embedding_gemma2.reference import full_model_inputs as F

    enc = full.vision if tower == "vision" else full.audio
    if tower == "vision":
        probe = vision_probe_inputs()
    else:
        probe = {k: {"audio": {"array": w, "sampling_rate": 16000}} for k, w in audio_probe_inputs().items()}
    rows = []
    for name, item in probe.items():
        inputs = _tower_inputs(tower, refs, item)
        hf_soft = refs.tower_soft(name, inputs)
        tt_soft = enc.encode(*inputs)
        p = min(_pcc(h, t) for h, t in zip(hf_soft, tt_soft))
        c = min(float(torch.nn.functional.cosine_similarity(h, t, dim=-1).min()) for h, t in zip(hf_soft, tt_soft))
        ref = refs.embedding(name, item)
        emb = full.encode([item])[0]
        e = float(torch.nn.functional.cosine_similarity(ref.float(), emb.float(), dim=-1))
        n_tok = sum(int(h.shape[0]) for h in hf_soft)
        rows.append({"input": name, "soft_pcc": p, "token_cos_min": c, "embedding_cos": e, "soft_tokens": n_tok})
        print(f"  PROBE {name}: soft {p:.5f} tokcos {c:.4f} emb {e:.5f} ({n_tok} tokens)", flush=True)
    # the tower's test_full_model items (end-to-end cosine) and test_retrieval
    items = {**F.items(), **F.IMAGE_RETRIEVAL_TEXTS, **F.AUDIO_RETRIEVAL_TEXTS}
    full_rows = []
    for name in FULL_ITEMS[tower]:
        ref = refs.embedding(f"full_{name}", items[name])
        emb = full.encode([items[name]])[0]
        e = float(torch.nn.functional.cosine_similarity(ref.float(), emb.float(), dim=-1))
        full_rows.append({"item": name, "embedding_cos": e})
        print(f"  FULL {name}: emb {e:.5f}", flush=True)
    caps = F.IMAGE_RETRIEVAL_TEXTS if tower == "vision" else F.AUDIO_RETRIEVAL_TEXTS
    kind = "image" if tower == "vision" else "audio"
    media = [f"{kind}_{k.removeprefix('caption_')}" for k in caps]
    tt_r = full.encode([items[n] for n in caps])
    tt_c = full.encode([items[n] for n in media])
    hf_r = torch.stack([refs.embedding(f"full_{n}", items[n]) for n in caps])
    hf_c = torch.stack([refs.embedding(f"full_{n}", items[n]) for n in media])
    s_tt, s_hf = tt_r.float() @ tt_c.float().T, hf_r.float() @ hf_c.float().T
    top2 = s_tt.topk(2, dim=-1).values
    retrieval = {
        "agree": bool(torch.equal(s_tt.argmax(-1), s_hf.argmax(-1))),
        "tt_margin_min": float((top2[:, 0] - top2[:, 1]).min()),
        "max_abs_diff": float((s_tt - s_hf).abs().max()),
    }
    print(f"  RETRIEVAL {kind}: {retrieval}", flush=True)
    worst = {
        "min_soft_pcc": min(r["soft_pcc"] for r in rows),
        "min_soft_pcc_at": min(rows, key=lambda r: r["soft_pcc"])["input"],
        "min_token_cos": min(r["token_cos_min"] for r in rows),
        "min_token_cos_at": min(rows, key=lambda r: r["token_cos_min"])["input"],
        "min_embedding_cos": min(r["embedding_cos"] for r in rows),
        "min_embedding_cos_at": min(rows, key=lambda r: r["embedding_cos"])["input"],
        "full_min_embedding_cos": min(r["embedding_cos"] for r in full_rows),
        "n_probe": len(rows),
    }
    worst["pass"] = (
        worst["min_soft_pcc"] >= BAR_SOFT
        and worst["min_embedding_cos"] >= BAR_COS
        and worst["full_min_embedding_cos"] >= BAR_COS
        and retrieval["agree"]
    )
    return {"rows": rows, "full_items": full_rows, "retrieval": retrieval, "summary": worst}


def cmd_probe(args) -> None:
    import ttnn
    from models.demos.blackhole.embedding_gemma2.conftest import MODEL_ID
    from models.demos.blackhole.embedding_gemma2.tt.embedding_model import EmbeddingGemma2
    from models.demos.blackhole.embedding_gemma2.tt.precision import PrecisionConfig
    from models.demos.blackhole.embedding_gemma2.tt.weights import resolve_snapshot

    tower = args.tower
    _load_combos(args)
    snapshot = resolve_snapshot(MODEL_ID)
    refs = FullRefs(tower, snapshot, args.ref_dir)
    for cid in args.candidates.split(","):
        prec = candidate_dict(tower, cid)
        rec = {"id": cid, "tower": tower, "commit": _git_head(), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        print(f"== probe {tower}:{cid}", flush=True)
        device = ttnn.open_device(device_id=0, l1_small_size=0, trace_region_size=256 * 1024 * 1024)
        try:
            full = EmbeddingGemma2.from_pretrained(
                device,
                str(snapshot),
                max_seq_len=8192,
                vision=tower == "vision",
                audio=tower == "audio",
                precision=PrecisionConfig.from_dict(prec, cid),
            )
            rec.update(run_probe(tower, full, refs))
            s = rec["summary"]
            print(
                f"  PROBE SUMMARY {cid}: soft {s['min_soft_pcc']:.5f} ({s['min_soft_pcc_at']}) emb "
                f"{s['min_embedding_cos']:.5f} full {s['full_min_embedding_cos']:.5f} tokcos {s['min_token_cos']:.4f} "
                f"retrieval {rec['retrieval']['agree']} -> {'PASS' if s['pass'] else 'FAIL'}",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001 - recorded as blocked
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["traceback"] = traceback.format_exc()[-4000:]
            print(f"  ERROR {cid}: {rec['error'][:400]}", flush=True)
        finally:
            ttnn.close_device(device)
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")


def cmd_list(args) -> None:
    for cid, (desc, delta) in candidates(args.tower).items():
        print(f"{cid:24s} {desc}   {json.dumps(delta)}")


# ---------------------------------------------------------------------- report
def _load(paths) -> list[dict]:
    recs = {}
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                recs[r["id"]] = r
    return list(recs.values())


def _pareto(points):
    pts = sorted(points, key=lambda p: (p[0], -p[1]))
    front, best = [], -1e9
    for p in pts:
        if p[1] > best:
            front.append(p)
            best = p[1]
    return front


def _chart(tower, recs, metric, ylabel, bar, path, selected, title, probe_key=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    def val(r):
        if metric == "embedding_cos":
            return r["probe"]["summary"]["full_min_embedding_cos"] if r.get("probe") else None
        return r["accuracy"][metric]["value"]

    pts = [
        (r["perf"]["rank_ms"], val(r), r["id"], r["accuracy"]["screen_pass"] and not r.get("probe_fail"))
        for r in recs
        if "rank_ms" in r.get("perf", {}) and "accuracy" in r and val(r) is not None
    ]
    probe_min = {r["id"]: r["probe"]["summary"][probe_key] for r in recs if probe_key and r.get("probe")}
    front = _pareto([(x, y, cid) for x, y, cid, _ in pts])
    front_ids = {p[2] for p in front}
    fig, ax = plt.subplots(figsize=(12.5, 6.6), dpi=150)
    ax.set_facecolor("#fbfaf7")
    ys = [y for _, y, *_ in pts] + list(probe_min.values()) + [bar]
    lo, hi = min(ys), max(max(ys), 1.0 if max(ys) > 0.999 else max(ys))
    pad = 0.06 * (hi - lo if hi > lo else 1e-3)
    ax.set_ylim(lo - pad, hi + pad)
    ax.plot([p[0] for p in front], [p[1] for p in front], color="#3b6ea8", lw=1.4, alpha=0.75, zorder=2)
    ax.axhline(bar, linestyle=":", color="#444444", lw=1.3, zorder=1)
    for x, y, cid, ok in pts:
        if cid in probe_min:
            ax.plot([x, x], [y, probe_min[cid]], color="#7a5195", lw=0.9, alpha=0.8, zorder=2)
            ax.scatter(x, probe_min[cid], s=46, facecolor="none", edgecolor="#7a5195", lw=1.3, zorder=4)
        if cid == selected:
            ax.scatter(x, y, s=170, color="#d62728", edgecolor="black", lw=0.8, zorder=5)
        else:
            ax.scatter(x, y, s=46, color="#3b6ea8" if ok else "#b0b0b0", edgecolor="#22323f", lw=0.6, zorder=3)
    fig.canvas.draw()
    placed = [ax.get_window_extent()]
    order = sorted(pts, key=lambda p: (p[2] != selected, p[2] not in front_ids and p[2] != "base", p[0]))
    for x, y, cid, ok in order:
        color = "#d62728" if cid == selected else ("#22323f" if ok else "#8a8a8a")
        for dx, dy in ((6, 3), (6, -10), (-6, 3), (-6, -10), (6, 12), (-6, 12), (6, -19), (-6, -19)):
            t = ax.annotate(
                cid,
                (x, y),
                textcoords="offset points",
                xytext=(dx, dy),
                fontsize=6.6 if cid != selected else 7.8,
                fontweight="bold" if cid == selected else "normal",
                color=color,
                ha="left" if dx > 0 else "right",
                zorder=6,
            )
            bb = t.get_window_extent(renderer=fig.canvas.get_renderer()).expanded(1.04, 1.15)
            inside = placed[0].contains(bb.x0, bb.y0) and placed[0].contains(bb.x1, bb.y1)
            if inside and not any(bb.overlaps(o) for o in placed[1:]):
                placed.append(bb)
                break
            t.remove()
    rank = " + ".join(f"{b}x{s}" for b, s in RANK_SHAPES[tower])
    ax.set_xlabel(f"traced device ms, burst median, {rank} (lower is faster)")
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#e3e0d8", lw=0.6)
    handles = [
        Line2D([], [], color="#3b6ea8", lw=1.4, alpha=0.75, label="Pareto frontier (every evaluated config)"),
        Line2D([], [], color="#444444", linestyle=":", lw=1.3, label=f"gate bar {bar}"),
        Line2D(
            [], [], marker="o", linestyle="", markersize=12, color="#d62728", mec="black", label=f"selected: {selected}"
        ),
        Line2D([], [], marker="o", linestyle="", markersize=7, color="#3b6ea8", mec="#22323f", label="passes screen"),
        Line2D(
            [], [], marker="o", linestyle="", markersize=7, color="#b0b0b0", mec="#22323f", label="fails screen / probe"
        ),
    ]
    if probe_min:
        handles.append(
            Line2D(
                [], [], marker="o", linestyle="", markersize=7, mfc="none", mec="#7a5195", label="min over probe inputs"
            )
        )
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _compact(rec: dict) -> dict:
    r = copy.deepcopy(rec)
    con = r.get("construction")
    if con and "layers" in con:
        groups = {}
        for i, ev in con["layers"].items():
            groups.setdefault(json.dumps(ev, sort_keys=True), []).append(int(i))
        con["layers"] = [{"layers": ls, **json.loads(k)} for k, ls in groups.items()]
    if "traceback" in r:
        r["traceback"] = r["traceback"][-600:]
    return r


def cmd_report(args) -> None:
    import csv

    tower = args.tower
    recs = _load(args.inputs.split(","))
    probes = {}
    for p in filter(None, args.probe.split(",")):
        for line in Path(p).read_text().splitlines():
            if line.strip():
                pr = json.loads(line)
                probes[pr["id"]] = pr
    for r in recs:
        if r["id"] in probes:
            r["probe"] = probes[r["id"]]
            r["probe_fail"] = not probes[r["id"]].get("summary", {}).get("pass", False)
    recs.sort(key=lambda r: r.get("perf", {}).get("rank_ms", 1e9))
    doc = Path(args.doc_dir)
    # sweep_results.json: one list over every tower; this tower's rows are replaced
    pj = doc / "sweep_results.json"
    old = json.loads(pj.read_text()) if pj.exists() else []
    keep = [r for r in old if r.get("tower", "text") != tower]
    new = []
    for r in recs:
        c = _compact(r)
        if c.get("probe"):
            c["probe"] = {k: c["probe"][k] for k in ("summary", "retrieval", "rows", "full_items") if k in c["probe"]}
        new.append(c)
    pj.write_text(json.dumps(keep + new, indent=1) + "\n")
    # sweep_results.csv: the text columns plus `tower` and the tower metrics
    pc = doc / "sweep_results.csv"
    rows_old, cols = [], []
    if pc.exists():
        with open(pc, newline="") as f:
            rd = csv.DictReader(f)
            cols = list(rd.fieldnames)
            rows_old = [r for r in rd if (r.get("tower") or "text") != tower]
    if "tower" not in cols:
        cols = ["tower"] + cols
    for r in rows_old:
        r["tower"] = r.get("tower") or "text"
    extra = ["burst_" + f"{b}x{s}" for b, s in PERF_SHAPES[tower]] + [
        "sustained_" + f"{b}x{s}" for b, s in PERF_SHAPES[tower]
    ]
    extra += ["stack_pcc", "soft_pcc", "soft_pcc_at", "token_cos_min", "token_cos_min_at"]
    extra += ["probe_min_soft_pcc", "probe_min_soft_pcc_at", "probe_min_embedding_cos", "probe_full_min_embedding_cos"]
    extra += ["probe_min_token_cos", "probe_retrieval_agree", "probe_pass"]
    for c in [
        "id",
        "description",
        "screen_pass",
        "failing",
        "rank_ms",
        "layer_pcc_gated",
        "module_pcc",
        "module_pcc_at",
        "perf_regime",
        "error",
        "commit",
    ] + extra:
        if c not in cols:
            cols.append(c)
    out_rows = []
    for r in recs:
        a, p = r.get("accuracy", {}), r.get("perf", {})
        row = {
            "tower": tower,
            "id": r["id"],
            "description": r["description"],
            "error": r.get("error", ""),
            "commit": r["commit"],
        }
        if a:
            row.update(
                screen_pass=a["screen_pass"],
                failing="; ".join(a["failing"]),
                layer_pcc_gated=f"{a['layer_pcc']['value']:.6f}",
                module_pcc=f"{a['module_pcc']['value']:.6f}",
                module_pcc_at=a["module_pcc"]["where"],
                soft_pcc=f"{a['soft_pcc']['value']:.6f}",
                soft_pcc_at=a["soft_pcc"]["where"],
                token_cos_min=f"{a['token_cos_min']['value']:.5f}",
                token_cos_min_at=a["token_cos_min"]["where"],
            )
            if "stack_pcc" in a:
                row["stack_pcc"] = f"{a['stack_pcc']['value']:.6f}"
        if p:
            row["rank_ms"] = f"{p['rank_ms']:.3f}"
            row["perf_regime"] = p.get("regime", "")
            for b, s in PERF_SHAPES[tower]:
                k = f"{b}x{s}"
                if k in p:
                    row[f"burst_{k}"] = f"{p[k]['burst_ms_median']:.3f}"
                    row[f"sustained_{k}"] = f"{p[k]['sustained_ms_median']:.3f}"
        if r.get("probe") and "summary" in r["probe"]:
            s = r["probe"]["summary"]
            row.update(
                probe_min_soft_pcc=f"{s['min_soft_pcc']:.6f}",
                probe_min_soft_pcc_at=s["min_soft_pcc_at"],
                probe_min_embedding_cos=f"{s['min_embedding_cos']:.6f}",
                probe_full_min_embedding_cos=f"{s['full_min_embedding_cos']:.6f}",
                probe_min_token_cos=f"{s['min_token_cos']:.5f}",
                probe_retrieval_agree=r["probe"]["retrieval"]["agree"],
                probe_pass=s["pass"],
            )
        out_rows.append(row)
    with open(pc, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows_old + out_rows:
            w.writerow({k: r.get(k, "") for k in cols})
    _chart(
        tower,
        recs,
        "embedding_cos",
        "min end-to-end embedding cosine vs HF fp32 (test_full_model items)",
        BAR_COS,
        doc / f"{tower}_cosine_perf_pareto.png",
        args.selected,
        f"EmbeddingGemma 2 {tower} tower: embedding cosine vs traced device time (P150a, probed configs)",
        "min_embedding_cos",
    )
    _chart(
        tower,
        recs,
        "soft_pcc",
        "min soft-token PCC vs HF fp32 (test cases)",
        BAR_SOFT,
        doc / f"{tower}_softpcc_perf_pareto.png",
        args.selected,
        f"EmbeddingGemma 2 {tower} tower: soft-token PCC vs traced device time (P150a, every evaluated config)",
        "min_soft_pcc",
    )
    print(f"wrote {len(recs)} {tower} records to {doc}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tower", required=True, choices=("vision", "audio"))
    r.add_argument("--candidates", default="all")
    r.add_argument("--combos", default="")
    r.add_argument("--out", required=True)
    r.add_argument("--no-acc", action="store_true")
    r.add_argument("--no-perf", action="store_true")
    r.add_argument("--burst", type=int, default=20)
    r.add_argument("--reps", type=int, default=5)
    r.add_argument("--idle", type=float, default=3.0)
    r.add_argument("--soak", type=float, default=5.0)
    r.add_argument("--sustained-ms", type=float, default=1500.0)
    pr = sub.add_parser("probe")
    pr.add_argument("--tower", required=True, choices=("vision", "audio"))
    pr.add_argument("--candidates", required=True)
    pr.add_argument("--combos", default="")
    pr.add_argument("--out", required=True)
    pr.add_argument("--ref-dir", required=True)
    ls = sub.add_parser("list")
    ls.add_argument("--tower", required=True, choices=("vision", "audio"))
    rp = sub.add_parser("report")
    rp.add_argument("--tower", required=True, choices=("vision", "audio"))
    rp.add_argument("--inputs", required=True)
    rp.add_argument("--probe", default="")
    rp.add_argument("--doc-dir", required=True)
    rp.add_argument("--selected", default="")
    args = ap.parse_args()
    {"run": cmd_run, "probe": cmd_probe, "list": cmd_list, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
