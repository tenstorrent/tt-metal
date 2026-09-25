# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Shape suite for the matmul out-of-box (default program config) benchmark.

Every case is run WITHOUT a program_config, so it measures the config matmul picks on its own. The compute
kernel config is always passed explicitly, because the default fidelity changes depending on whether a
core_grid is given (#55889). Pinning it keeps comparisons between selectors about the config only.

Tiers:
  issues   shapes from filed default-config perf issues (see Case.source)
  models   common LLM decode/prefill linears (weights in bfp8/bfp4)
  generic  a coarse M x K x N grid, bf16 and bfp8 weights, DRAM and L1
  sharded  sharded-input / sharded-output cases on the full device grid
"""

from dataclasses import dataclass, field, replace
from typing import Optional, Tuple, Union

# Memory placements:
#   "dram" / "l1"            interleaved
#   "l1_height" / "l1_width" / "l1_block"
#                            L1 sharded; inputs are sharded across the full device grid, outputs get
#                            the matching ttnn.L1_*_SHARDED_MEMORY_CONFIG (shard spec left to matmul)
MEMS = ("dram", "l1", "l1_height", "l1_width", "l1_block")
DTYPES = ("bf16", "bfp8", "bfp4", "fp32")
FIDELITIES = ("LoFi", "HiFi2", "HiFi3", "HiFi4")


@dataclass(frozen=True)
class Case:
    name: str
    a_shape: Tuple[int, ...]
    b_shape: Tuple[int, ...]
    tier: str
    source: str = ""  # issue/PR the shape comes from
    a_dtype: str = "bf16"
    b_dtype: str = "bf16"
    out_dtype: Optional[str] = None  # None -> matmul default (a's dtype)
    a_mem: str = "dram"
    b_mem: str = "dram"
    out_mem: str = "dram"
    transpose_a: bool = False
    transpose_b: bool = False
    op: str = "matmul"  # "matmul" | "linear"
    bias: bool = False  # linear only
    activation: Optional[str] = None  # linear only, e.g. "silu"
    core_grid: Union[None, str, Tuple[int, int]] = None  # None | "device" | (x, y)
    fidelity: str = "HiFi2"
    fp32_acc: bool = False
    packer_l1_acc: bool = True
    tags: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self):
        assert self.a_mem in MEMS and self.b_mem in MEMS and self.out_mem in MEMS, self
        assert self.a_dtype in DTYPES and self.b_dtype in DTYPES, self
        assert self.out_dtype is None or self.out_dtype in DTYPES, self
        assert self.fidelity in FIDELITIES, self
        assert self.op in ("matmul", "linear"), self
        assert self.op == "linear" or (not self.bias and self.activation is None), self

    @property
    def mkn(self):
        """(batch, M, K, N) after transposes; batch is the product of A's (broadcast) leading dims."""
        a = list(self.a_shape)
        b = list(self.b_shape)
        if self.transpose_a:
            a[-2], a[-1] = a[-1], a[-2]
        if self.transpose_b:
            b[-2], b[-1] = b[-1], b[-2]
        batch = 1
        lead_a, lead_b = a[:-2], b[:-2]
        for i in range(1, max(len(lead_a), len(lead_b)) + 1):
            da = lead_a[-i] if i <= len(lead_a) else 1
            db = lead_b[-i] if i <= len(lead_b) else 1
            batch *= max(da, db)
        return batch, a[-2], a[-1], b[-1]


def _with_mems(base: Case, *triples):
    """One variant of `base` per (a_mem, b_mem, out_mem) triple, named <base>-<a>.<b>.<out>."""
    return [replace(base, name=f"{base.name}-{a}.{b}.{o}", a_mem=a, b_mem=b, out_mem=o) for a, b, o in triples]


# ---------------------------------------------------------------------------
# Tier: issues
# ---------------------------------------------------------------------------


def _issue_cases():
    cases = []

    # #56976: ~4x slower default when any operand/output is L1 (all-DRAM gate). Gemma4-31B prefill MLP.
    base = Case("i56976_gemma4_mlp", (1024, 5376), (5376, 5376), "issues", "#56976", b_dtype="bfp8")
    cases += _with_mems(
        base, ("dram", "dram", "dram"), ("l1", "dram", "dram"), ("dram", "dram", "l1"), ("l1", "dram", "l1")
    )
    cases.append(replace(base, name=f"{base.name}-grid", core_grid="device"))

    # #40845: DeepSeek linear slower with L1 than DRAM (1D chosen where 2D is ~2-5x faster)
    for m in (32, 128):
        base = Case(f"i40845_ds_{m}x7168x256", (m, 7168), (7168, 256), "issues", "#40845", b_dtype="bfp8")
        cases += _with_mems(base, ("dram", "dram", "dram"), ("l1", "dram", "l1"))

    # #35396 / #29716 comments: tt-train nanoGPT (batched attention, fwd/bwd linears)
    cases += [
        Case("i35396_attn_qk", (64, 6, 256, 64), (64, 6, 256, 64), "issues", "#35396", transpose_b=True),
        Case("i35396_attn_av", (64, 6, 256, 256), (64, 6, 256, 64), "issues", "#35396"),
        Case("i35396_attn_bwd_ta", (64, 6, 256, 256), (64, 6, 256, 64), "issues", "#35396", transpose_a=True),
        Case("i35396_fwd_qkv", (16384, 384), (384, 1152), "issues", "#35396"),
        Case("i35396_fwd_proj", (16384, 384), (384, 384), "issues", "#35396"),
        Case("i35396_fwd_up", (16384, 384), (384, 1536), "issues", "#35396"),
        Case("i35396_fwd_down", (16384, 1536), (1536, 384), "issues", "#35396"),
        Case("i35396_bwd_dw_up", (16384, 384), (16384, 1536), "issues", "#35396", transpose_a=True),
        Case("i35396_bwd_dw_proj", (16384, 384), (16384, 384), "issues", "#35396", transpose_a=True),
        Case("i29716_gpt2_ta", (4, 12, 1024, 1024), (4, 12, 1024, 64), "issues", "#29716", transpose_a=True),
        Case("i29716_tinyllama_ta", (1, 32, 2048, 2048), (1, 32, 2048, 64), "issues", "#29716", transpose_a=True),
    ]
    cases += [replace(c, name=f"{c.name}-grid", core_grid="device") for c in list(cases) if c.source == "#35396"]

    # #29716: TT-DiT linears, HiFi2 + fp32 acc. Reported with core_grid set; run both ways.
    dit = [
        (11264, 3072, 4608),
        (11264, 3072, 8192),
        (11264, 4096, 3072),
        (5632, 3072, 2304),
        (5632, 3072, 4096),
        (5632, 2048, 3072),
        (37888, 5120, 1280),
        (37888, 5120, 3456),
        (37888, 3456, 5120),
        (9472, 5120, 1280),
        (9472, 5120, 3456),
        (9472, 5120, 2560),
        (9472, 5120, 6912),
        (9472, 3456, 5120),
        (9472, 6912, 5120),
    ]
    for m, k, n in dit:
        base = Case(f"i29716_dit_{m}x{k}x{n}", (m, k), (k, n), "issues", "#29716", fp32_acc=True, tags=("large",))
        cases += [base, replace(base, name=f"{base.name}-grid", core_grid="device")]

    # #25503: linear picks a small core grid (batched A, small K/N)
    cases += [
        Case("i25503_704", (1, 704, 703, 128), (128, 128), "issues", "#25503", a_dtype="bfp8", b_dtype="bfp8"),
        Case("i25503_768_n4", (768, 768, 128), (128, 4), "issues", "#25503"),
        Case("i25503_768_n512", (768, 768, 128), (128, 512), "issues", "#25503", b_dtype="bfp8"),
        Case("i25503_768_n128", (768, 768, 128), (128, 128), "issues", "#25503"),
    ]

    # #25502: batched x batched in L1, default used 9 cores
    cases.append(
        Case(
            "i25502_bmm704_l1",
            (1, 32, 704, 704),
            (1, 32, 704, 704),
            "issues",
            "#25502",
            a_dtype="bfp8",
            b_dtype="bfp8",
            a_mem="l1",
            b_mem="l1",
            out_mem="l1",
        )
    )

    # #31743: L1 output fell back to MultiCore
    base = Case("i31743", (1, 4, 256, 2048), (1, 4, 2048, 7168), "issues", "#31743", b_dtype="bfp4")
    cases += _with_mems(base, ("dram", "dram", "dram"), ("dram", "dram", "l1"))

    # #45311 / #57806: square-ish bf16, plus fp32 acc and fused silu
    for m, k, n in ((2048, 2048, 2048), (4096, 2048, 2048)):
        base = Case(f"i45311_{m}x{k}x{n}", (m, k), (k, n), "issues", "#45311")
        cases += [
            base,
            replace(base, name=f"{base.name}-fp32acc", fp32_acc=True),
            replace(base, name=f"{base.name}-silu", op="linear", activation="silu"),
        ]

    # #55889: fidelity confound shapes (fidelity pinned here)
    cases += [
        Case("i55889_2048x4096x14336", (2048, 4096), (4096, 14336), "issues", "#55889"),
        Case("i55889_32x4096x14336", (32, 4096), (4096, 14336), "issues", "#55889"),
    ]

    # #36426: transpose_a with large K; OOM with core_grid, slow without
    cases += [
        Case("i36426_ta_11008", (8192, 11008), (8192, 4096), "issues", "#36426", transpose_a=True, tags=("large",)),
        Case("i36426_ta_50272", (16384, 50272), (16384, 384), "issues", "#36426", transpose_a=True, tags=("large",)),
    ]

    # #30405 / #30407: big square, all weight dtypes
    for dt, fid in (("bf16", "HiFi2"), ("bfp8", "LoFi"), ("bfp4", "LoFi")):
        cases.append(
            Case(
                f"i30407_8192cube_{dt}",
                (8192, 8192),
                (8192, 8192),
                "issues",
                "#30407",
                a_dtype=dt,
                b_dtype=dt,
                fidelity=fid,
                tags=("large",),
            )
        )

    return cases


# ---------------------------------------------------------------------------
# Tier: models
# ---------------------------------------------------------------------------

# (name, K, N, weight dtype); A is always bf16
_LLM_LINEARS = [
    # Llama-3.1-8B
    ("llama8b_qkv", 4096, 6144, "bfp8"),
    ("llama8b_wo", 4096, 4096, "bfp8"),
    ("llama8b_w1", 4096, 14336, "bfp4"),
    ("llama8b_w2", 14336, 4096, "bfp8"),
    # Llama-3.1-70B, TP=8 shards
    ("llama70b_tp8_qkv", 8192, 1280, "bfp8"),
    ("llama70b_tp8_w1", 8192, 3584, "bfp4"),
    ("llama70b_tp8_w2", 3584, 8192, "bfp8"),
    # LM head slice
    ("lmhead_16k", 4096, 16384, "bfp8"),
]


def _model_cases():
    cases = []
    for name, k, n, wdt in _LLM_LINEARS:
        fid = "LoFi" if wdt == "bfp4" else "HiFi2"
        for m, phase in ((32, "decode"), (128, "decode128"), (2048, "prefill2k"), (8192, "prefill8k")):
            base = Case(
                f"m_{name}_{phase}",
                (1, 1, m, k),
                (k, n),
                "models",
                "llm",
                b_dtype=wdt,
                op="linear",
                fidelity=fid,
                tags=(phase,),
            )
            cases.append(base)
            if m <= 128:  # decode activations commonly live in L1
                cases.append(replace(base, name=f"{base.name}-l1", a_mem="l1", out_mem="l1"))
    return cases


# ---------------------------------------------------------------------------
# Tier: generic
# ---------------------------------------------------------------------------


def _generic_cases():
    cases = []
    for m in (32, 256, 1024, 4096):
        for k in (1024, 4096):
            for n in (1024, 4096, 16384):
                for wdt in ("bf16", "bfp8"):
                    for mem in ("dram", "l1"):
                        cases.append(
                            Case(
                                f"g_{m}x{k}x{n}_{wdt}_{mem}",
                                (m, k),
                                (k, n),
                                "generic",
                                b_dtype=wdt,
                                a_mem=mem,
                                out_mem=mem,
                            )
                        )
    return cases


# ---------------------------------------------------------------------------
# Tier: sharded
# ---------------------------------------------------------------------------


def _sharded_cases():
    return [
        # decode: width-sharded activation, 1D in0-mcast territory
        Case("s_w_32x4096x4096", (1, 1, 32, 4096), (4096, 4096), "sharded", b_dtype="bfp8", a_mem="l1_width"),
        Case(
            "s_w_32x4096x4096_wout",
            (1, 1, 32, 4096),
            (4096, 4096),
            "sharded",
            b_dtype="bfp8",
            a_mem="l1_width",
            out_mem="l1_width",
        ),
        Case("s_w_32x8192x1280", (1, 1, 32, 8192), (8192, 1280), "sharded", b_dtype="bfp8", a_mem="l1_width"),
        # tall: height-sharded activation, 1D in1-mcast territory
        Case("s_h_8192x256x256", (1, 1, 8192, 256), (256, 256), "sharded", a_mem="l1_height"),
        Case("s_h_8192x256x256_hout", (1, 1, 8192, 256), (256, 256), "sharded", a_mem="l1_height", out_mem="l1_height"),
        Case("s_h_16384x128x512", (1, 1, 16384, 128), (128, 512), "sharded", a_mem="l1_height"),
        # block-sharded 2D
        Case("s_b_2048x2048x2048", (1, 1, 2048, 2048), (2048, 2048), "sharded", a_mem="l1_block"),
        Case(
            "s_b_2048x2048x2048_bout",
            (1, 1, 2048, 2048),
            (2048, 2048),
            "sharded",
            a_mem="l1_block",
            out_mem="l1_block",
        ),
        Case("s_b_4096x1024x4096", (1, 1, 4096, 1024), (1024, 4096), "sharded", b_dtype="bfp8", a_mem="l1_block"),
        # interleaved input, sharded output only
        Case("s_o_h_8192x512x512", (8192, 512), (512, 512), "sharded", out_mem="l1_height"),
        Case("s_o_w_32x4096x8192", (32, 4096), (4096, 8192), "sharded", b_dtype="bfp8", out_mem="l1_width"),
        Case("s_o_b_2048x2048x2048", (2048, 2048), (2048, 2048), "sharded", out_mem="l1_block"),
    ]


TIERS = {
    "issues": _issue_cases,
    "models": _model_cases,
    "generic": _generic_cases,
    "sharded": _sharded_cases,
}


def get_cases(tiers=None):
    tiers = tiers or list(TIERS)
    cases = [c for t in tiers for c in TIERS[t]()]
    names = [c.name for c in cases]
    dupes = {n for n in names if names.count(n) > 1}
    assert not dupes, f"duplicate case names: {sorted(dupes)}"
    return cases
