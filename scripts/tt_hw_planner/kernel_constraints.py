"""
Kernel-level constraint checker for HuggingFace -> TTNN bring-up.

The companion to compatibility.py: while that module answers "does TT have a
*module* that implements this HF concept?", this module answers "do the TTNN
*kernels* accept this model's exact shapes / dtypes / layouts?".

Each constraint mirrors a real `TT_FATAL` predicate found in the C++ device
ops under `ttnn/cpp/ttnn/operations/`. We hard-code the predicates rather
than calling into ttnn at runtime, so this check is fully static and does
not require hardware.

Predicates cited inline reference the file the rule came from so reviewers
can verify; if the C++ kernel relaxes a constraint, update both the C++ side
and the citation here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, List, Optional


TILE = 32

DEFAULT_TP_GRID = [1, 2, 4, 8, 32]


class Severity(str, Enum):
    BLOCKER = "blocker"
    WARN = "warn"
    INFO = "info"


@dataclass
class KernelFinding:
    op: str
    field: str
    value: Any
    constraint: str
    passes: Optional[bool]
    severity: Severity
    fix: str = ""
    source: str = ""

    def status_glyph(self) -> str:
        if self.passes is None:
            return "[ ?? ]"
        if self.passes:
            return "[ ok ]"
        if self.severity == Severity.WARN:
            return "[warn]"
        if self.severity == Severity.INFO:
            return "[info]"
        return "[FAIL]"

    def to_dict(self) -> dict:
        return {
            "op": self.op,
            "field": self.field,
            "value": self.value,
            "constraint": self.constraint,
            "passes": self.passes,
            "severity": self.severity.value if isinstance(self.severity, Severity) else str(self.severity),
            "fix": self.fix or "",
            "source": self.source or "",
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KernelFinding":
        sev = d.get("severity")
        if isinstance(sev, str):
            try:
                sev = Severity(sev)
            except ValueError:
                sev = Severity.WARN
        return cls(
            op=str(d.get("op") or ""),
            field=str(d.get("field") or ""),
            value=d.get("value"),
            constraint=str(d.get("constraint") or ""),
            passes=(lambda p: p if p is None else bool(p))(d.get("passes", False)),
            severity=sev or Severity.WARN,
            fix=str(d.get("fix") or ""),
            source=str(d.get("source") or ""),
        )


def collect_actionable_findings(report: "KernelReport") -> List[KernelFinding]:
    """Dedup-collect every non-passing WARN/BLOCKER finding across all TPs in
    the report. Most kernel constraints (rotary, RMSNorm, MLP shapes) are
    TP-invariant; a few attention shape rules vary by TP. Dedup by
    (op, field, value, constraint) so the same warning across TPs collapses
    to one entry.

    INFO-level findings are intentionally excluded — they're for the
    planner's own routing decisions, not the LLM's iter context.
    """
    seen = set()
    out: List[KernelFinding] = []
    for findings in report.findings_by_tp.values():
        for f in findings:
            if f.passes is not False:
                continue
            if f.severity not in (Severity.WARN, Severity.BLOCKER):
                continue
            key = (f.op, f.field, repr(f.value), f.constraint)
            if key in seen:
                continue
            seen.add(key)
            out.append(f)
    return out


def _text_cfg(cfg: dict) -> dict:
    """Backwards-compat alias for :func:`compatibility._text_config`."""
    from .compatibility import _text_config

    return _text_config(cfg)


def _g(cfg: dict, *keys, default=None):
    """First non-None lookup across keys, searching text_config too."""
    t = _text_cfg(cfg)
    for k in keys:
        v = t.get(k)
        if v is not None:
            return v
        v = cfg.get(k)
        if v is not None:
            return v
    return default


def normalize_config_value(v):
    """Detect the FORMAT of a config value, then extract a scalar from it.

    Returns ``(kind, scalar, detail)`` where:
      kind   : 'scalar' | 'per_layer' | 'mapping' | 'unknown'
      scalar : a single representative int (the max), or None if undeterminable
      detail : the raw structure, so callers can react to heterogeneity

    Never raises: any shape it does not recognize degrades to
    ``('unknown', None, v)`` rather than crashing the caller.
    """
    if isinstance(v, bool):
        return ("scalar", int(v), v)
    if isinstance(v, (int, float)):
        return ("scalar", int(v), v)
    if isinstance(v, str):
        s = v.strip()
        return ("scalar", int(s), v) if s.lstrip("-").isdigit() else ("unknown", None, v)
    if isinstance(v, dict):
        nums = [normalize_config_value(x)[1] for x in v.values()]
        nums = [n for n in nums if n is not None]
        return ("mapping", max(nums) if nums else None, v)
    if isinstance(v, (list, tuple)):
        nums = [normalize_config_value(x)[1] for x in v]
        nums = [n for n in nums if n is not None]
        return ("per_layer", max(nums) if nums else None, v)
    return ("unknown", None, v)


def _cfg_int(v) -> Optional[int]:
    """Representative scalar int of a config value, or None if undeterminable.

    Thin wrapper over :func:`normalize_config_value` so config reads stay
    shape-agnostic: a scalar passes through, a per-layer list/dict reduces to
    its max, and any unrecognized shape degrades to None instead of raising.
    """
    return normalize_config_value(v)[1]


def _head_dim(cfg: dict) -> Optional[int]:
    hd = _cfg_int(_g(cfg, "head_dim"))
    if hd:
        return hd
    h = _cfg_int(_g(cfg, "hidden_size", "dim"))
    n = _cfg_int(_g(cfg, "num_attention_heads", "n_heads"))
    if h and n:
        return h // n
    return None


def _is_mla(cfg: dict) -> bool:
    """Backwards-compat alias for :func:`compatibility._is_mla`."""
    from .compatibility import _is_mla as _is_mla_canonical

    return _is_mla_canonical(cfg)


def check_attention_shapes(cfg: dict, tp: int) -> List[KernelFinding]:
    """Predicates from sdpa_device_operation.cpp and the matmul backing QKV."""
    out: List[KernelFinding] = []
    h = _cfg_int(_g(cfg, "hidden_size", "dim"))
    nh = _cfg_int(_g(cfg, "num_attention_heads", "n_heads"))
    nkv = _cfg_int(_g(cfg, "num_key_value_heads", "n_kv_heads", default=nh))
    hd = _head_dim(cfg)
    mla = _is_mla(cfg)

    if hd is not None and not mla:
        out.append(
            KernelFinding(
                op="ttnn.transformer.scaled_dot_product_attention",
                field="head_dim",
                value=hd,
                constraint=f"head_dim must be a multiple of TILE({TILE}) - SDPA tilizes inner dim",
                passes=(hd % TILE == 0),
                severity=Severity.BLOCKER,
                fix="Use a TT-supported sibling model with aligned head_dim, or pad heads (custom kernel work).",
                source="ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa_program_factory.cpp (DHt = DH / TILE_WIDTH)",
            )
        )

    if mla:
        t = _text_cfg(cfg)
        rope_hd = t.get("qk_rope_head_dim")
        nope_hd = t.get("qk_nope_head_dim")
        v_hd = t.get("v_head_dim")
        for name, _raw in [("qk_rope_head_dim", rope_hd), ("qk_nope_head_dim", nope_hd), ("v_head_dim", v_hd)]:
            val = _cfg_int(_raw)
            if val is None:
                continue
            out.append(
                KernelFinding(
                    op="ttnn.transformer.flash_multi_latent_attention_decode",
                    field=name,
                    value=val,
                    constraint=f"{name} must be a multiple of TILE({TILE})",
                    passes=(val % TILE == 0),
                    severity=Severity.BLOCKER,
                    fix="MLA head dims are model-defined; mismatched values usually mean an unusual variant.",
                    source="ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_device_operation.cpp",
                )
            )

    if nh and nkv and not mla:
        out.append(
            KernelFinding(
                op="ttnn.transformer.scaled_dot_product_attention",
                field="num_attention_heads / num_key_value_heads",
                value=f"{nh}/{nkv}",
                constraint="num_attention_heads must be divisible by num_key_value_heads (GQA ratio)",
                passes=(int(nh) % int(nkv) == 0 and int(nh) >= int(nkv)),
                severity=Severity.BLOCKER,
                fix="No mitigation - this is a fundamental GQA layout constraint.",
                source="ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_device_operation.cpp",
            )
        )

    if h and tp > 1:
        out.append(
            KernelFinding(
                op="ttnn.matmul (QKV projection)",
                field="hidden_size",
                value=h,
                constraint=f"hidden_size must be divisible by TP({tp}) so the QKV matmul shards evenly",
                passes=(int(h) % tp == 0),
                severity=Severity.BLOCKER,
                fix=f"Pick a different mesh shape, or pad hidden to multiple of {tp}.",
                source="models/tt_transformers/tt/attention.py + matmul program-config validators",
            )
        )

    if nh and tp > 1:
        out.append(
            KernelFinding(
                op="ttnn.matmul (attention output)",
                field="num_attention_heads",
                value=nh,
                constraint=f"num_attention_heads must be divisible by TP({tp}) so each shard owns whole heads",
                passes=(int(nh) % tp == 0),
                severity=Severity.BLOCKER,
                fix=f"Use TP that divides {nh}, or accept replicated attention.",
                source="models/tt_transformers/tt/attention.py",
            )
        )

    if nkv and tp > 1 and not mla:
        out.append(
            KernelFinding(
                op="nlp_concat_heads_decode (KV sharding)",
                field="num_key_value_heads",
                value=nkv,
                constraint=f"num_key_value_heads must be divisible by TP({tp}); GQA/MQA shards KV across the mesh's TP dim",
                passes=(int(nkv) % tp == 0),
                severity=Severity.BLOCKER,
                fix=f"Use TP that divides {nkv} (e.g. {', '.join(str(t) for t in (1, 2, 4, 8) if int(nkv) % t == 0) or '1'}), or replicate KV (not supported in tt_transformers).",
                source="models/tt_transformers/tt/model_config.py: `assert self.n_kv_heads % self.cluster_shape[1] == 0`",
            )
        )

    if h is not None:
        out.append(
            KernelFinding(
                op="ttnn.matmul",
                field="hidden_size",
                value=h,
                constraint=f"inner matmul dim must align to TILE({TILE})",
                passes=(int(h) % TILE == 0),
                severity=Severity.BLOCKER,
                fix="No fix - pick a TT-compatible model variant.",
                source="ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp",
            )
        )

    return out


def check_rope_hf(cfg: dict, _tp: int) -> List[KernelFinding]:
    """RoPE kernel selection. `tt_transformers` has TWO working kernels:

    1. `ttnn.experimental.rotary_embedding_hf` -- requires head_dim % 64 == 0
       (faster on Blackhole; used when `use_hf_rope=True`)
    2. `ttnn.experimental.rotary_embedding_llama` (a.k.a. mllama RoPE) --
       requires head_dim % 32 == 0
       (the **default** in `tt_transformers/tt/model_config.py:500`;
       used when `use_hf_rope=False`)

    2026-05-23 bug fix: previously this checker hard-failed the bring-up
    when (1) didn't fit, even though (2) is the default code path and
    would have worked fine. That produced false-positives like
    "Phi-3.5-mini cannot run on QB2" for a model that actually runs out
    of the box via the mllama RoPE fallback.

    New behavior:
      - If head_dim % 64 == 0  -> HF RoPE works, [ok], no finding for the fallback.
      - elif head_dim % 32 == 0 -> HF RoPE fails BUT runtime auto-falls-back
        to mllama RoPE. Emit a WARN (not BLOCKER) so the compat gate
        doesn't refuse to scaffold a perfectly-supported model. The
        runtime config knob is already `use_hf_rope=False` by default,
        so no user action is required for the common case.
      - else (head_dim % 32 != 0) -> NEITHER kernel accepts; that's a
        true BLOCKER.

    Skipped for MLA models (DeepSeek uses qk_rope_head_dim, not the same kernel)."""
    if _is_mla(cfg):
        return []
    hd = _head_dim(cfg)
    if hd is None:
        return []
    if hd % 64 == 0:
        return [
            KernelFinding(
                op="ttnn.experimental.rotary_embedding_hf",
                field="head_dim",
                value=hd,
                constraint="head_dim must be divisible by 64 (HF RoPE kernel layout)",
                passes=True,
                severity=Severity.BLOCKER,
                fix="",
                source="ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_hf/device/rotary_embedding_hf_device_operation.cpp",
            )
        ]
    if hd % 32 == 0:
        return [
            KernelFinding(
                op="ttnn.experimental.rotary_embedding_hf",
                field="head_dim",
                value=hd,
                constraint=(
                    "head_dim must be divisible by 64 (HF RoPE kernel layout); "
                    "this model has head_dim divisible by 32 so the runtime "
                    "auto-falls-back to rotary_embedding_llama (mllama RoPE), "
                    "which is the default in tt_transformers/tt/model_config.py:500. "
                    "No user action required."
                ),
                passes=False,
                severity=Severity.WARN,
                fix=(
                    "No action required -- `use_hf_rope=False` is already the "
                    "default. If you've explicitly set `use_hf_rope=True`, "
                    "unset it for this model."
                ),
                source="ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_hf/device/rotary_embedding_hf_device_operation.cpp",
            )
        ]
    return [
        KernelFinding(
            op="ttnn.experimental.rotary_embedding_hf",
            field="head_dim",
            value=hd,
            constraint=(
                "head_dim must be divisible by 64 (HF RoPE) OR 32 (mllama RoPE); "
                "this model satisfies neither so both RoPE kernels reject it."
            ),
            passes=False,
            severity=Severity.BLOCKER,
            fix=(
                "Pad the attention heads to align head_dim to a multiple of 32, "
                "or extend rotary_embedding_llama_device_operation.cpp to handle "
                "this head_dim (custom kernel work)."
            ),
            source="ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_hf/device/rotary_embedding_hf_device_operation.cpp",
        )
    ]


def check_rmsnorm(cfg: dict, _tp: int) -> List[KernelFinding]:
    """rmsnorm.cpp / layernorm_device_operation.cpp - TILE layout, last dim aligned."""
    h = _cfg_int(_g(cfg, "hidden_size", "dim"))
    if h is None:
        return []
    return [
        KernelFinding(
            op="ttnn.rms_norm",
            field="hidden_size",
            value=h,
            constraint=f"normalized dim must be a multiple of TILE({TILE})",
            passes=(int(h) % TILE == 0),
            severity=Severity.BLOCKER,
            fix="No fix - this is a kernel layout invariant.",
            source="ttnn/cpp/ttnn/operations/normalization/rmsnorm/rmsnorm.cpp",
        )
    ]


def check_mlp_shapes(cfg: dict, tp: int) -> List[KernelFinding]:
    """MLP / SwiGLU: gate_up and down projections share intermediate_size."""
    out: List[KernelFinding] = []
    inter = _cfg_int(_g(cfg, "intermediate_size", "hidden_dim"))
    if inter is None:
        return out

    out.append(
        KernelFinding(
            op="ttnn.matmul (MLP up/gate)",
            field="intermediate_size",
            value=inter,
            constraint=f"intermediate_size must be a multiple of TILE({TILE})",
            passes=(int(inter) % TILE == 0),
            severity=Severity.BLOCKER,
            fix="No fix.",
            source="ttnn/cpp/ttnn/operations/matmul/...",
        )
    )

    if tp > 1:
        out.append(
            KernelFinding(
                op="ttnn.matmul (MLP up/gate)",
                field="intermediate_size",
                value=inter,
                constraint=f"intermediate_size must be divisible by TP({tp})",
                passes=(int(inter) % tp == 0),
                severity=Severity.BLOCKER,
                fix=f"Pick a mesh whose TP divides {inter}.",
                source="models/tt_transformers/tt/mlp.py",
            )
        )

    return out


def check_embedding(cfg: dict, _tp: int) -> List[KernelFinding]:
    """embedding_device_operation.cpp: BF16 weights only; tilized-output alignment."""
    out: List[KernelFinding] = []
    vocab = _cfg_int(_g(cfg, "vocab_size"))
    if vocab is None:
        return out

    out.append(
        KernelFinding(
            op="ttnn.embedding",
            field="vocab_size",
            value=vocab,
            constraint=f"vocab tile-padding: pads to multiple of {TILE} (no defect, just memory overhead if vocab is odd)",
            passes=(int(vocab) % TILE == 0),
            severity=Severity.INFO,
            fix="No action - padding is automatic, costs at most one tile per shard.",
            source="ttnn/cpp/ttnn/operations/embedding/device/embedding_device_operation.cpp",
        )
    )
    return out


def check_lm_head(cfg: dict, tp: int) -> List[KernelFinding]:
    """LM head: vocab sharded across TP via column-split matmul."""
    out: List[KernelFinding] = []
    vocab = _cfg_int(_g(cfg, "vocab_size"))
    if vocab is None or tp <= 1:
        return out

    pad_to = TILE * tp
    padded = ((int(vocab) + pad_to - 1) // pad_to) * pad_to
    overhead = padded - int(vocab)

    out.append(
        KernelFinding(
            op="ttnn.matmul (LM head)",
            field="vocab_size",
            value=vocab,
            constraint=f"vocab is padded to a multiple of TILE*TP ({TILE}*{tp}={pad_to}); pad overhead = {overhead} tokens",
            passes=True,
            severity=Severity.INFO,
            fix="None - padded_vocab_size handles padding automatically.",
            source="models/tt_transformers/tt/lm_head.py",
        )
    )
    return out


def check_topk(cfg: dict, _tp: int) -> List[KernelFinding]:
    """topk_device_operation.cpp: multi-core requires dim power-of-2, k<=64, dim<65536."""
    vocab = _cfg_int(_g(cfg, "vocab_size"))
    if vocab is None:
        return []
    v = int(vocab)
    pow2 = (v & (v - 1)) == 0 and v > 0
    return [
        KernelFinding(
            op="ttnn.topk (sampling)",
            field="vocab_size",
            value=v,
            constraint=f"multi-core top-k requires vocab to be a power of 2 AND < 65536; else falls back to single-core",
            passes=(pow2 and v < 65536),
            severity=Severity.WARN,
            fix="No action needed - kernel will use the single-core path. Decode throughput will be lower.",
            source="ttnn/cpp/ttnn/operations/reduction/topk/device/topk_device_operation.cpp",
        )
    ]


def check_moe(cfg: dict, _tp: int) -> List[KernelFinding]:
    """Mixtral MoE path: hardcoded num_devices=8 and top-2 (mixtral_moe.py)."""
    t = _text_cfg(cfg)
    is_moe = bool(t.get("num_local_experts") or t.get("num_experts"))
    if not is_moe:
        return []
    out: List[KernelFinding] = []

    tk_kind, tk, tk_raw = normalize_config_value(t.get("num_experts_per_tok") or t.get("moe_topk"))
    if tk_raw is not None:
        tk_hetero = tk_kind in ("per_layer", "mapping")
        out.append(
            KernelFinding(
                op="models/tt_transformers/tt/mixtral_moe.py",
                field="num_experts_per_tok",
                value=tk_raw,
                constraint=(
                    "shared MoE block hardcodes top-2 routing"
                    + (" (per-layer top-k varies; the shared block supports only a single top-2)" if tk_hetero else "")
                ),
                passes=(None if tk is None else (tk == 2)),
                severity=(Severity.INFO if tk is None else Severity.WARN),
                fix="Lift mixtral_moe.py to parameterize top-k, or use the model-specific MoE in models/demos/{gpt_oss,deepseek_v3}.",
                source="models/tt_transformers/tt/mixtral_moe.py",
            )
        )

    ne_kind, ne, ne_raw = normalize_config_value(
        t.get("num_local_experts") or t.get("num_experts") or t.get("n_routed_experts")
    )
    if ne_raw is not None:
        ne_hetero = ne_kind in ("per_layer", "mapping")
        out.append(
            KernelFinding(
                op="models/tt_transformers/tt/mixtral_moe.py",
                field="num_local_experts",
                value=ne_raw,
                constraint=(
                    "shared MoE block assumes num_experts == 8 (one expert per device on a T3K)"
                    + (" (per-layer expert count varies)" if ne_hetero else "")
                ),
                passes=(None if ne is None else (ne == 8)),
                severity=(Severity.INFO if ne is None else Severity.WARN),
                fix="Use a per-model MoE in demos, or refactor the shared block to parameterize expert count.",
                source="models/tt_transformers/tt/mixtral_moe.py",
            )
        )

    return out


def check_sliding_window(cfg: dict, _tp: int) -> List[KernelFinding]:
    """attention.py raises NotImplementedError on chunked prefill + sliding window.

    2026-05-23 audit bug #13/#14:
      - Previously `passes=True` so the table showed `[ ok ]` for an
        active incompatibility. Now `passes=False` (still WARN, not
        BLOCKER, because non-chunked prefill works).
      - Previously only the scalar `sliding_window` field was read;
        Gemma-2/hybrid models use `layer_types` containing
        "sliding_attention" entries. Detect that too."""
    t = _text_cfg(cfg)
    sw = t.get("sliding_window")
    layer_types = t.get("layer_types")
    has_sliding_layer_type = isinstance(layer_types, list) and any(
        isinstance(x, str) and "sliding" in x.lower() for x in layer_types
    )
    if not sw and not has_sliding_layer_type:
        return []
    return [
        KernelFinding(
            op="ttnn.transformer.scaled_dot_product_attention (sliding)",
            field="sliding_window" if sw else "layer_types[*]=sliding_attention",
            value=sw if sw else layer_types,
            constraint="sliding-window attention is not supported in combination with chunked prefill",
            passes=False,
            severity=Severity.WARN,
            fix="If using long-context prefill, disable chunked prefill for sliding-window layers.",
            source="models/tt_transformers/tt/attention.py",
        )
    ]


def check_rope_scaling(cfg: dict, _tp: int) -> List[KernelFinding]:
    """RoPE scaling. tt_transformers' `rotary_embedding_factory`
    (models/tt_transformers/tt/rope.py:304) explicitly handles exactly
    four scaling types: linear, llama3, yarn, longrope. Anything else
    raises `ValueError(\"Invalid rope_scaling: ...\")` at model
    instantiation time.

    2026-05-23 audit fixes (3 bugs):

    1. The supported-set previously listed `dynamic`, but tt_transformers
       does NOT handle `rope_type=dynamic` -- the runtime would raise.
       Listing it as supported produced false-negatives (compat said
       "ok", runtime then crashed). Fix: remove `dynamic` from the
       supported set.

    2. The check read only `cfg.rope_scaling` and ignored the newer HF
       `cfg.rope_parameters` field (see Phi-3.5's deprecation warning:
       `rope_parameters['original_max_position_embeddings']`).
       tt_transformers' model_config.py:2736 also only reads
       `rope_scaling`, so models that migrated to `rope_parameters`
       would silently lose their scaling at runtime (the runtime
       would treat them as no-scaling). Fix: detect the migration
       case explicitly and WARN.

    3. If `rope_scaling` is set as a dict but its `type`/`rope_type`
       field is missing, the previous check silently returned [] --
       hiding a config that the runtime would not interpret correctly.
       Fix: emit a WARN.
    """
    t = _text_cfg(cfg)
    rs = t.get("rope_scaling")
    rp = t.get("rope_parameters")
    out: List[KernelFinding] = []

    if rp and not rs:
        out.append(
            KernelFinding(
                op="models/tt_transformers/tt/rope.py",
                field="rope_parameters (HF-newer field)",
                value="set",
                constraint=(
                    "this HF config uses `rope_parameters` (the newer "
                    "field) but tt_transformers/tt/model_config.py:2736 "
                    "only reads `rope_scaling`. The runtime will silently "
                    "treat this model as having NO scaling, potentially "
                    "diverging from HF reference at long contexts."
                ),
                passes=False,
                severity=Severity.WARN,
                fix=(
                    "Mirror `rope_parameters` into `rope_scaling` in the "
                    "model_params/<model>/config.json override, OR extend "
                    "tt_transformers/tt/model_config.py to also read "
                    "`rope_parameters`. Safe at short contexts; matters "
                    "at the model's full native context."
                ),
                source="models/tt_transformers/tt/model_config.py:2736",
            )
        )

    if not isinstance(rs, dict):
        return out

    rtype = (rs.get("type") or rs.get("rope_type") or "").lower()
    if not rtype:
        out.append(
            KernelFinding(
                op="models/tt_transformers/tt/rope.py",
                field="rope_scaling.type",
                value="(missing)",
                constraint=(
                    "rope_scaling is set as a dict but has no `type` or "
                    "`rope_type` field; tt_transformers' factory won't "
                    "know how to interpret it and will likely fall back "
                    "to a default RoPE, producing wrong positional "
                    "embeddings at long contexts."
                ),
                passes=False,
                severity=Severity.WARN,
                fix=(
                    'Add `"type": "<linear|llama3|yarn|longrope>"` '
                    "to the rope_scaling dict (or to a model_params "
                    "override config.json)."
                ),
                source="models/tt_transformers/tt/rope.py:314",
            )
        )
        return out

    supported = {"linear", "llama3", "yarn", "longrope"}
    out.append(
        KernelFinding(
            op="models/tt_transformers/tt/rope.py",
            field="rope_scaling.type",
            value=rtype,
            constraint=(f"rope_scaling.type must be one of " f"{{{', '.join(sorted(supported))}}} to apply correctly"),
            passes=(rtype in supported),
            severity=Severity.BLOCKER,
            fix=(
                f"tt_transformers/tt/rope.py:304 only handles {sorted(supported)}. "
                f"Type {rtype!r} would raise ValueError at model "
                f"instantiation. Either map this type to one of the "
                f"supported ones in a model_params override, or extend "
                f"`rotary_embedding_factory` with a new branch."
            ),
            source="models/tt_transformers/tt/rope.py:304 (rotary_embedding_factory)",
        )
    )
    return out


def check_alibi(cfg: dict, _tp: int) -> List[KernelFinding]:
    t = _text_cfg(cfg)
    if not t.get("alibi_bias_max"):
        return []
    return [
        KernelFinding(
            op="(attention bias)",
            field="alibi_bias_max",
            value=t.get("alibi_bias_max"),
            constraint="ALiBi positional bias has no TTNN implementation",
            passes=False,
            severity=Severity.BLOCKER,
            fix="Write a new ALiBi-aware attention kernel, or use a sibling model that uses RoPE.",
            source="(no TT op found in survey)",
        )
    ]


CONSTRAINT_CHECKS: List[Callable[[dict, int], List[KernelFinding]]] = [
    check_attention_shapes,
    check_rope_hf,
    check_rope_scaling,
    check_rmsnorm,
    check_mlp_shapes,
    check_embedding,
    check_lm_head,
    check_topk,
    check_moe,
    check_sliding_window,
    check_alibi,
]


@dataclass
class KernelReport:
    findings_by_tp: dict = field(default_factory=dict)
    tp_grid: List[int] = field(default_factory=list)

    @property
    def shape_findings(self) -> List[KernelFinding]:
        """TP-invariant findings - take from any TP slot, they'll be identical."""
        if not self.tp_grid:
            return []
        all_at_tp1 = self.findings_by_tp.get(1, [])
        return [f for f in all_at_tp1 if "TP(" not in f.constraint]

    @property
    def tp_dependent_findings(self) -> dict:
        """tp -> [findings that depend on TP and FAIL at that TP]."""
        out = {}
        for tp, findings in self.findings_by_tp.items():
            tp_findings = [f for f in findings if "TP(" in f.constraint and f.passes is False]
            if tp_findings:
                out[tp] = tp_findings
        return out

    def has_blockers(self, tp: Optional[int] = None) -> bool:
        if tp is not None:
            return any(f.passes is False and f.severity == Severity.BLOCKER for f in self.findings_by_tp.get(tp, []))
        return any(
            f.passes is False and f.severity == Severity.BLOCKER
            for findings in self.findings_by_tp.values()
            for f in findings
        )


def evaluate_kernels(cfg: dict, tp_grid: Optional[List[int]] = None) -> KernelReport:
    grid = tp_grid or DEFAULT_TP_GRID
    report = KernelReport(tp_grid=list(grid))
    for tp in grid:
        all_findings: List[KernelFinding] = []
        for fn in CONSTRAINT_CHECKS:
            all_findings.extend(fn(cfg, tp))
        report.findings_by_tp[tp] = all_findings
    return report
