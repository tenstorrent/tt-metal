# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

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
    passes: bool
    severity: Severity
    fix: str = ""
    source: str = ""

    def status_glyph(self) -> str:
        if self.passes:
            return "[ ok ]"
        if self.severity == Severity.WARN:
            return "[warn]"
        if self.severity == Severity.INFO:
            return "[info]"
        return "[FAIL]"


def _text_cfg(cfg: dict) -> dict:
    return cfg.get("text_config") or cfg


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


def _head_dim(cfg: dict) -> Optional[int]:
    hd = _g(cfg, "head_dim")
    if hd:
        return int(hd)
    h = _g(cfg, "hidden_size", "dim")
    n = _g(cfg, "num_attention_heads", "n_heads")
    if h and n:
        return int(h) // int(n)
    return None


def _is_mla(cfg: dict) -> bool:
    """MLA models use compressed KV (qk_nope_head_dim / kv_lora_rank) and a
    distinct attention kernel; standard SDPA shape checks don't apply."""
    t = _text_cfg(cfg)
    return bool(t.get("kv_lora_rank") or t.get("q_lora_rank") or t.get("qk_rope_head_dim") or t.get("qk_nope_head_dim"))


def check_attention_shapes(cfg: dict, tp: int) -> List[KernelFinding]:
    """Predicates from sdpa_device_operation.cpp and the matmul backing QKV."""
    out: List[KernelFinding] = []
    h = _g(cfg, "hidden_size", "dim")
    nh = _g(cfg, "num_attention_heads", "n_heads")
    nkv = _g(cfg, "num_key_value_heads", "n_kv_heads", default=nh)
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
        for name, val in [("qk_rope_head_dim", rope_hd), ("qk_nope_head_dim", nope_hd), ("v_head_dim", v_hd)]:
            if val is None:
                continue
            out.append(
                KernelFinding(
                    op="ttnn.transformer.flash_multi_latent_attention_decode",
                    field=name,
                    value=val,
                    constraint=f"{name} must be a multiple of TILE({TILE})",
                    passes=(int(val) % TILE == 0),
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
    """rotary_embedding_hf_device_operation.cpp requires last dim divisible by 64.
    Skipped for MLA models (DeepSeek uses qk_rope_head_dim, not the same kernel)."""
    if _is_mla(cfg):
        return []
    hd = _head_dim(cfg)
    if hd is None:
        return []
    return [
        KernelFinding(
            op="ttnn.experimental.rotary_embedding_hf",
            field="head_dim",
            value=hd,
            constraint="head_dim must be divisible by 64 (HF RoPE kernel layout)",
            passes=(hd % 64 == 0),
            severity=Severity.BLOCKER,
            fix="Fall back to use_hf_rope=False (rotary_embedding_llama, multiple of 32) if model permits.",
            source="ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_hf/device/rotary_embedding_hf_device_operation.cpp",
        )
    ]


def check_rmsnorm(cfg: dict, _tp: int) -> List[KernelFinding]:
    """rmsnorm.cpp / layernorm_device_operation.cpp - TILE layout, last dim aligned."""
    h = _g(cfg, "hidden_size", "dim")
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
    inter = _g(cfg, "intermediate_size", "hidden_dim")
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
    vocab = _g(cfg, "vocab_size")
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
    vocab = _g(cfg, "vocab_size")
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
    vocab = _g(cfg, "vocab_size")
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

    top_k = t.get("num_experts_per_tok") or t.get("moe_topk")
    if top_k is not None:
        out.append(
            KernelFinding(
                op="models/tt_transformers/tt/mixtral_moe.py",
                field="num_experts_per_tok",
                value=top_k,
                constraint="shared MoE block hardcodes top-2 routing",
                passes=(int(top_k) == 2),
                severity=Severity.WARN,
                fix="Lift mixtral_moe.py to parameterize top-k, or use the model-specific MoE in models/demos/{gpt_oss,deepseek_v3}.",
                source="models/tt_transformers/tt/mixtral_moe.py",
            )
        )

    num_experts = t.get("num_local_experts") or t.get("num_experts") or t.get("n_routed_experts")
    if num_experts is not None:
        out.append(
            KernelFinding(
                op="models/tt_transformers/tt/mixtral_moe.py",
                field="num_local_experts",
                value=num_experts,
                constraint="shared MoE block assumes num_experts == 8 (one expert per device on a T3K)",
                passes=(int(num_experts) == 8),
                severity=Severity.WARN,
                fix="Use a per-model MoE in demos, or refactor the shared block to parameterize expert count.",
                source="models/tt_transformers/tt/mixtral_moe.py",
            )
        )

    return out


def check_sliding_window(cfg: dict, _tp: int) -> List[KernelFinding]:
    """attention.py raises NotImplementedError on chunked prefill + sliding window."""
    t = _text_cfg(cfg)
    sw = t.get("sliding_window")
    if not sw:
        return []
    return [
        KernelFinding(
            op="ttnn.transformer.scaled_dot_product_attention (sliding)",
            field="sliding_window",
            value=sw,
            constraint="sliding-window attention is not supported in combination with chunked prefill",
            passes=True,
            severity=Severity.WARN,
            fix="If using long-context prefill, disable chunked prefill for sliding-window layers.",
            source="models/tt_transformers/tt/attention.py",
        )
    ]


def check_rope_scaling(cfg: dict, _tp: int) -> List[KernelFinding]:
    """rope.py: warns + drops scaling for 'default' / 'mrope'."""
    t = _text_cfg(cfg)
    rs = t.get("rope_scaling")
    if not isinstance(rs, dict):
        return []
    rtype = (rs.get("type") or rs.get("rope_type") or "").lower()
    if not rtype:
        return []
    supported = {"linear", "llama3", "yarn", "longrope", "dynamic"}
    return [
        KernelFinding(
            op="models/tt_transformers/tt/rope.py",
            field="rope_scaling.type",
            value=rtype,
            constraint=f"rope_scaling.type must be one of {{{', '.join(sorted(supported))}}} to apply correctly",
            passes=(rtype in supported),
            severity=Severity.BLOCKER,
            fix="If type is 'mrope', a new RoPE kernel is required - inference will diverge from HF otherwise.",
            source="models/tt_transformers/tt/rope.py (rope_scaling_model_factory)",
        )
    ]


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
            tp_findings = [f for f in findings if "TP(" in f.constraint and not f.passes]
            if tp_findings:
                out[tp] = tp_findings
        return out

    def has_blockers(self, tp: Optional[int] = None) -> bool:
        if tp is not None:
            return any(not f.passes and f.severity == Severity.BLOCKER for f in self.findings_by_tp.get(tp, []))
        return any(
            not f.passes and f.severity == Severity.BLOCKER
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
