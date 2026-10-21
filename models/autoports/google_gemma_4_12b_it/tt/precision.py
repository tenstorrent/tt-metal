# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Precision policy of the Gemma 4 12B full model, read from a JSON precision config.

``Gemma4Model`` (and therefore ``build_generator`` and every adapter built on them) loads
``DEFAULT_PRECISION_CONFIG`` (``doc/datatype_sweep/selected_precision_config.json``, the datatype sweep's selection)
unless the caller passes ``precision=`` or sets ``GEMMA4_PRECISION_CONFIG``. The file is required: a missing or
malformed config raises instead of silently falling back to hard-coded dtypes. ``precision="baseline"`` selects the
optimized-full-model (stage 7) policy (``BASELINE_PRECISION_CONFIG``), the safe fallback; ``precision="legacy"`` skips
the config and keeps the decoder's hard-coded per-layer policy (``multichip_layer_policy``; equivalence check only).

Every swept field is turned into decoder / LM-head construction arguments here (``layer_policy``, ``lm_head_policy``),
so a value in the JSON is a value in the built tensors and compute configs:

* ``weights``: weight dtype per matmul group (``qkv``, ``o``, ``gate_up`` = gate and up, ``down``) with per-layer
  exceptions; DRAM reader counts follow the dtype (BFP4 groups read with 2-3 cores per bank, ``_readers``).
* ``compute_fidelity``: decode / prefill matmul fidelity per group with per-layer exceptions, LM head, SDPA decode /
  prefill. ``fixed`` lists the fidelities the decoder does not expose (norms, RoPE, short-chunk SDPA); they are
  validated against the implementation.
* ``kv_cache.dtype``, ``ccl`` (decode all-reduce payloads, prefill o_proj reduce-scatter, prefill norm-output
  all-gathers), ``activations`` (prefill MLP intermediates per layer), ``lm_head.dtype``.
* ``activations.residual``, ``activations.decode_intermediates``, ``logits`` and ``sampling`` record the dtype
  assumptions of the residual stream, decode matmul outputs, logits buffer and sampler. Each has exactly one legal
  value in this implementation (BF16; the split top-k sampler); other values raise. ``decode_intermediates`` covers
  the decode matmul outputs that stay on chip; the o_proj / down outputs feeding the all-reduce take the
  ``ccl.decode_all_reduce`` dtype.
"""

import json
import os
import warnings
from pathlib import Path

import ttnn

SCHEMA = "gemma4_12b_precision/v1"
MODEL_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PRECISION_CONFIG = MODEL_DIR / "doc/datatype_sweep/selected_precision_config.json"
BASELINE_PRECISION_CONFIG = MODEL_DIR / "doc/datatype_sweep/configs/baseline_stage7.json"
PRECISION_ENV = "GEMMA4_PRECISION_CONFIG"

GROUPS = ("qkv", "o", "gate_up", "down")
GROUP_ROLES = {"qkv": ("qkv",), "o": ("o",), "gate_up": ("gate", "up"), "down": ("down",)}
WEIGHT_DTYPES = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
MATMUL_FIDELITIES = ("lofi", "lofifp32", "hifi2", "hifi2fp32", "hifi4", "hifi4fp32")
KV_DTYPES = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
AR_DTYPES = ("bf16", "bfp8", "bfp4")
ACT_DTYPES = {"bf16": None, "bfp8": "bfp8"}  # decoder policy encoding: None = BF16
# fidelities the decoder hard-codes (validated, not swept)
FIXED_FIDELITIES = {"norms": "hifi4fp32", "rope_decode": "hifi4", "sdpa_prefill_short_chunk": "hifi2"}
# dtype assumptions with a single legal value in this implementation
FIXED_ASSUMPTIONS = {
    ("activations", "residual"): "bf16",
    ("activations", "decode_intermediates"): "bf16",
    ("logits", "dtype"): "bf16",
    ("sampling", "sampler"): "split_topk",
    ("sampling", "input_dtype"): "bf16",
}
# DRAM reader cores per bank for BFP4 weights (optimized-decoder README "Per-layer BFP4 MLP"): MLP 3 for every MLP
# matmul of a layer with a BFP4 MLP group, o_proj 3, QKV 2 (sliding: 32 tiles per bank) / 3 (full: 36)
BFP4_READERS = {"mlp": 3, "o": 3, "qkv_sliding": 2, "qkv_full": 3}


def _require(cond, msg):
    """Config validation that also holds under ``python -O`` (a bad config must never build silently)."""
    if not cond:
        raise ValueError(msg)


def parse_layers(spec, num_layers=48):
    """``"13-24,36"`` / ``[13, 14]`` / ``"all"`` -> sorted list of layer ids."""
    if spec == "all":
        return list(range(num_layers))
    if isinstance(spec, int):
        spec = [spec]
    out = set()
    for part in spec if isinstance(spec, list) else str(spec).split(","):
        if isinstance(part, int):
            out.add(part)
            continue
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    bad = [i for i in out if not 0 <= i < num_layers]
    _require(not bad, f"layer ids outside 0..{num_layers - 1}: {bad}")
    return sorted(out)


def _per_layer(section, field, num_layers, legal, what):
    """{"default": {group: v}, "layer_exceptions": [{"layers": spec, field: {group: v}}]} -> [{group: v}] per layer."""
    default = dict(section["default"])
    _require(set(default) == set(GROUPS) or what == "prefill MLP intermediates", f"{what}: default needs {GROUPS}")
    table = [dict(default) for _ in range(num_layers)]
    for exc in section.get("layer_exceptions", []):
        vals = exc[field]
        for i in parse_layers(exc["layers"], num_layers):
            for g, v in vals.items():
                _require(g in default, f"{what}: unknown group {g}")
                table[i][g] = v
    for row in table:
        for g, v in row.items():
            _require(v in legal, f"{what}: illegal value {v!r} for {g} (legal: {sorted(legal)})")
    return table


class PrecisionConfig:
    """A validated precision config (see the module docstring); ``None`` = the decoder's hard-coded policy."""

    def __init__(self, cfg, source="<dict>", num_layers=48):
        self.source = str(source)
        self.legacy = cfg is None
        self.num_layers = num_layers
        self.raw = cfg
        if self.legacy:
            self.config_id = "legacy_hardcoded"
            return
        _require(cfg.get("schema") == SCHEMA, f"{source}: schema must be {SCHEMA}")
        self.config_id = cfg["config_id"]
        self.weights = _per_layer(cfg["weights"], "dtype", num_layers, WEIGHT_DTYPES, "weights")
        fid = cfg["compute_fidelity"]
        self.decode_fid = _per_layer(fid["decode_matmul"], "fidelity", num_layers, MATMUL_FIDELITIES, "decode fid")
        self.prefill_fid = _per_layer(fid["prefill_matmul"], "fidelity", num_layers, MATMUL_FIDELITIES, "prefill fid")
        self.lm_head_fid = fid["lm_head"]
        self.sdpa_decode_fid = fid["sdpa_decode"]
        self.sdpa_prefill_fid = dict(fid["sdpa_prefill"])
        for v in [self.lm_head_fid, self.sdpa_decode_fid, *self.sdpa_prefill_fid.values()]:
            _require(v in MATMUL_FIDELITIES, f"illegal fidelity {v}")
        _require(
            set(self.sdpa_prefill_fid) == {"sliding", "full"}, "compute_fidelity.sdpa_prefill needs sliding / full"
        )
        _require(
            fid.get("fixed", FIXED_FIDELITIES) == FIXED_FIDELITIES,
            f"compute_fidelity.fixed must be {FIXED_FIDELITIES}: the decoder hard-codes these",
        )
        self.lm_head_dtype = cfg["lm_head"]["dtype"]
        _require(
            self.lm_head_dtype in WEIGHT_DTYPES, f"lm_head.dtype {self.lm_head_dtype!r} not in {sorted(WEIGHT_DTYPES)}"
        )
        self.kv_dtype = cfg["kv_cache"]["dtype"]
        _require(self.kv_dtype in KV_DTYPES, f"kv_cache.dtype {self.kv_dtype!r} not in {sorted(KV_DTYPES)}")
        ccl = cfg["ccl"]
        self.ar_dtype = dict(ccl["decode_all_reduce"])
        _require(
            set(self.ar_dtype) == {"o", "down"} and all(v in AR_DTYPES for v in self.ar_dtype.values()),
            f"ccl.decode_all_reduce needs o / down in {AR_DTYPES}: {self.ar_dtype}",
        )
        self.prefill_rs_o = ccl["prefill_reduce_scatter_o"]
        _require(
            self.prefill_rs_o in ACT_DTYPES,
            f"ccl.prefill_reduce_scatter_o {self.prefill_rs_o!r} not in {sorted(ACT_DTYPES)}",
        )
        self.prefill_ag = dict(ccl["prefill_all_gather"])
        _require(
            set(self.prefill_ag) == {"qkv", "gate_up"} and all(v in ACT_DTYPES for v in self.prefill_ag.values()),
            f"ccl.prefill_all_gather needs qkv / gate_up in {sorted(ACT_DTYPES)}: {self.prefill_ag}",
        )
        _require(
            ccl.get("prefill_reduce_scatter_down", "follows_prefill_mlp_product") == "follows_prefill_mlp_product",
            "ccl.prefill_reduce_scatter_down must be 'follows_prefill_mlp_product' (the down output's dtype)",
        )
        act = cfg["activations"]
        mlp = act["prefill_mlp_intermediates"]
        _require(
            set(mlp["default"]) == {"gate_up", "product"},
            "activations.prefill_mlp_intermediates needs gate_up / product",
        )
        self.prefill_mlp = _per_layer(mlp, "dtype", num_layers, ACT_DTYPES, "prefill MLP intermediates")
        for (sec, key), legal in FIXED_ASSUMPTIONS.items():
            v = cfg[sec][key]
            _require(v == legal, f"{sec}.{key} = {v!r}: this implementation supports only {legal!r}")

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, precision=None, num_layers=48):
        """``precision``: None (``$GEMMA4_PRECISION_CONFIG`` or ``DEFAULT_PRECISION_CONFIG``), a path, a dict,
        ``"baseline"`` (the stage-7 policy) or ``"legacy"`` (no config: the decoder's hard-coded policy)."""
        if isinstance(precision, PrecisionConfig):
            return precision
        if precision == "legacy":
            return cls(None, "legacy", num_layers)
        if isinstance(precision, dict):
            return cls(precision, "<dict>", num_layers)
        if precision is None:
            if os.environ.get(PRECISION_ENV):
                warnings.warn(
                    f"{PRECISION_ENV}={os.environ[PRECISION_ENV]} overrides the selected precision config "
                    f"{DEFAULT_PRECISION_CONFIG}",
                    stacklevel=2,
                )
            precision = os.environ.get(PRECISION_ENV) or DEFAULT_PRECISION_CONFIG
        if precision == "baseline":
            precision = BASELINE_PRECISION_CONFIG
        path = Path(precision)
        if not path.is_file():
            raise FileNotFoundError(
                f"precision config {path} not found: the model's dtype / fidelity policy is read from it "
                f"(doc/datatype_sweep/README.md); pass precision='baseline' for the stage-7 policy"
            )
        return cls(json.loads(path.read_text()), path, num_layers)

    # ------------------------------------------------------------------ construction arguments
    def layer_policy(self, layer_idx, layer_type):
        """Decoder policy overrides for one layer (``MultichipDecoder.from_state_dict(policy=...)``): every precision
        field of the decoder policy, so none of its hard-coded per-layer defaults survives. ``{}`` for legacy."""
        if self.legacy:
            return {}
        w = self.weights[layer_idx]
        dtype = {r: WEIGHT_DTYPES[w[g]] for g in GROUPS for r in GROUP_ROLES[g]}
        return {
            "dtype": dtype,
            "readers": self._readers(w, layer_type),
            "decode_fid": {r: self.decode_fid[layer_idx][g] for g in GROUPS for r in GROUP_ROLES[g]},
            "prefill_fid": {r: self.prefill_fid[layer_idx][g] for g in GROUPS for r in GROUP_ROLES[g]},
            "kv_dtype": KV_DTYPES[self.kv_dtype],
            "ar_dtype": dict(self.ar_dtype),
            "prefill_o_dtype": ACT_DTYPES[self.prefill_rs_o],
            "prefill_norm_dtype": {k: ACT_DTYPES[v] for k, v in self.prefill_ag.items()},
            "prefill_mlp_act_dtype": {k: ACT_DTYPES[v] for k, v in self.prefill_mlp[layer_idx].items()},
            "sdpa_decode_fid": self.sdpa_decode_fid,
            "sdpa_prefill_fid": dict(self.sdpa_prefill_fid),
        }

    @staticmethod
    def _readers(w, layer_type):
        mlp4 = w["gate_up"] == "bfp4" or w["down"] == "bfp4"
        mlp = BFP4_READERS["mlp"] if mlp4 else 1
        qkv = BFP4_READERS["qkv_full" if layer_type == "full_attention" else "qkv_sliding"] if w["qkv"] == "bfp4" else 1
        return {"qkv": qkv, "o": BFP4_READERS["o"] if w["o"] == "bfp4" else 1, "gate": mlp, "up": mlp, "down": mlp}

    def lm_head_policy(self):
        """LM head dtype / fidelity (``Gemma4Model``'s ``LM_HEAD_POLICY`` overrides); ``{}`` for legacy."""
        if self.legacy:
            return {}
        return {"dtype": WEIGHT_DTYPES[self.lm_head_dtype], "fidelity": self.lm_head_fid}

    @property
    def logits_dtype(self):
        return ttnn.bfloat16

    @property
    def sampler(self):
        return "split_topk"

    def describe(self):
        return {"config_id": self.config_id, "source": self.source}

    def verify(self, summary):
        """Raise if a built model (``built_precision_summary``) holds other weight / KV / all-reduce / prefill
        activation dtypes, reader counts or matmul / SDPA / LM-head fidelities than this config prescribes."""
        if self.legacy:
            return
        dt = {"bf16": "BFLOAT16", "bfp8": "BFLOAT8_B", "bfp4": "BFLOAT4_B"}
        fid = {f: f.replace("hifi", "HiFi").replace("lofi", "LoFi").replace("fp32", "+fp32") for f in MATMUL_FIDELITIES}
        errs = []

        def check(what, built, want):
            if built != want:
                errs.append(f"{what}: built {built} != config {want}")

        check("lm_head", summary["lm_head"], {"dtype": dt[self.lm_head_dtype], "fidelity": fid[self.lm_head_fid]})
        for row in summary["layers"]:
            i, kind = row["layer"], row["type"]
            w = self.weights[i]
            roles = [(g, r) for g in GROUPS for r in GROUP_ROLES[g]]
            check(f"layer {i} weights", row["weights"], {r: dt[w[g]] for g, r in roles})
            check(f"layer {i} readers", row["readers"], self._readers(w, kind))
            check(f"layer {i} decode fidelity", row["decode_fid"], {r: fid[self.decode_fid[i][g]] for g, r in roles})
            check(f"layer {i} prefill fidelity", row["prefill_fid"], {r: fid[self.prefill_fid[i][g]] for g, r in roles})
            check(f"layer {i} sdpa decode", row["sdpa_decode_fid"], fid[self.sdpa_decode_fid])
            sdpa = self.sdpa_prefill_fid["full" if kind == "full_attention" else "sliding"]
            check(f"layer {i} sdpa prefill", row["sdpa_prefill_fid"], fid[sdpa])
            check(f"layer {i} norms", row["norm_fid"], fid[FIXED_FIDELITIES["norms"]])
            check(f"layer {i} kv cache", row["kv_cache"], [dt[self.kv_dtype]] * 2)
            check(f"layer {i} all-reduce", row["ar_dtype"], {r: dt[d] for r, d in self.ar_dtype.items()})
            check(f"layer {i} prefill rs o", row["prefill_o_dtype"], dt[self.prefill_rs_o])
            check(
                f"layer {i} prefill gathers", row["prefill_norm_dtype"], {r: dt[d] for r, d in self.prefill_ag.items()}
            )
            check(f"layer {i} prefill mlp", row["prefill_mlp_act"], {k: dt[v] for k, v in self.prefill_mlp[i].items()})
        if errs:
            raise AssertionError(f"built model differs from precision config {self.config_id}: " + "; ".join(errs[:8]))


def deep_merge(base, over):
    """``base`` updated by ``over`` (dicts merged key by key, recursively); neither input is modified."""
    out = {k: (deep_merge(v, {}) if isinstance(v, dict) else v) for k, v in (base or {}).items()}
    for k, v in (over or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def built_precision_summary(model):
    """What the built model actually holds: per-layer weight / KV dtypes, reader counts and compute-kernel fidelities
    read back from the constructed tensors and configs (the propagation evidence of the datatype sweep)."""

    def dt(t):
        return str(t.dtype).split(".")[-1]

    def fid(ck):
        return f"{str(ck.math_fidelity).split('.')[-1]}{'+fp32' if ck.fp32_dest_acc_en else ''}"

    layers = []
    for layer in model.layers:
        row = {
            "layer": layer.layer_idx,
            "type": layer.layer_type,
            "weights": {
                "qkv": dt(layer.wqkv),
                "o": dt(layer.wo),
                "gate": dt(layer.w_gate),
                "up": dt(layer.w_up),
                "down": dt(layer.w_down),
            },
            "readers": dict(layer.policy["readers"]),
            "decode_fid": {r: fid(ck) for r, ck in layer.decode_ck.items()},
            "prefill_fid": {r: fid(ck) for r, ck in layer.prefill_ck.items()},
            "sdpa_decode_fid": fid(layer.compute_sdpa_decode),
            "sdpa_prefill_fid": fid(layer.compute_sdpa_prefill),
            "norm_fid": fid(layer.compute_hifi4_fp32),
            # an unattached cache (serving builds the layers first) reports the dtype the owner must allocate;
            # ``Gemma4Model.attach_kv_cache`` re-verifies the attached tensors
            "kv_cache": [dt(t) for t in layer.kv_cache] if layer.kv_cache else [str(layer.kv_dtype).split(".")[-1]] * 2,
            "ar_dtype": {r: str(d).split(".")[-1] for r, d in layer.ar_dtype.items()},
            "prefill_o_dtype": str(layer.prefill_o_dtype).split(".")[-1] if layer.prefill_o_dtype else "BFLOAT16",
            "prefill_norm_dtype": {
                r: (str(d).split(".")[-1] if d else "BFLOAT16") for r, d in layer.prefill_norm_dtype.items()
            },
            "prefill_mlp_act": {
                "gate_up": (
                    str(layer.prefill_gate_up_dtype).split(".")[-1] if layer.prefill_gate_up_dtype else "BFLOAT16"
                ),
                "product": (
                    str(layer.prefill_product_dtype).split(".")[-1] if layer.prefill_product_dtype else "BFLOAT16"
                ),
            },
            "fused_gate_up_packed": dt(layer.w_gate_up_packed) if layer.fused_gate_up else None,
        }
        layers.append(row)
    return {
        "precision": model.precision.describe(),
        "lm_head": {"dtype": dt(model.lm_head), "fidelity": fid(model.lm_ck)},
        "lm_head_program": model.lm_policy["program"],
        "embedding": dt(model.embedding),
        "logits_dtype": str(model.precision.logits_dtype).split(".")[-1],
        "layers": layers,
    }
