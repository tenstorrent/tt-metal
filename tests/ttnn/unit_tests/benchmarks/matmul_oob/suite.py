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
# An explicit shard spec (a_shard / b_shard / out_shard) replaces those defaults: a ShardDesc of the shard
# grid as (x0, y0, x1, y1) core ranges, the shard shape in elements, and "row" / "col" orientation.
MEMS = ("dram", "l1", "l1_height", "l1_width", "l1_block")
DTYPES = ("bf16", "bfp8", "bfp4", "fp32")
FIDELITIES = ("LoFi", "HiFi2", "HiFi3", "HiFi4")


ShardDesc = Tuple[Tuple[Tuple[int, int, int, int], ...], Tuple[int, int], str]


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
    a_shard: Optional[ShardDesc] = None
    b_shard: Optional[ShardDesc] = None
    out_shard: Optional[ShardDesc] = None
    transpose_a: bool = False
    transpose_b: bool = False
    op: str = "matmul"  # "matmul" | "linear"
    bias: bool = False  # linear only
    activation: Optional[str] = None  # linear only, e.g. "silu"
    core_grid: Union[None, str, Tuple[int, int]] = None  # None | "device" | (x, y)
    fidelity: str = "HiFi2"
    fp32_acc: bool = False
    packer_l1_acc: bool = True
    dst_full_sync: bool = False  # compute config dst_full_sync_en: twice the DST tiles per subblock
    tags: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self):
        assert self.a_mem in MEMS and self.b_mem in MEMS and self.out_mem in MEMS, self
        assert self.a_dtype in DTYPES and self.b_dtype in DTYPES, self
        assert self.out_dtype is None or self.out_dtype in DTYPES, self
        assert self.fidelity in FIDELITIES, self
        assert self.op in ("matmul", "linear"), self
        for mem, shard in ((self.a_mem, self.a_shard), (self.b_mem, self.b_shard), (self.out_mem, self.out_shard)):
            assert shard is None or mem.startswith("l1_"), self
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


def _gist_cases():
    """The #57884 gist sweeps' Llama shapes (gist/matmul_sweep_2d.py and _1d.py), timed like every other case:
    bf16 in DRAM, HiFi4, fp32 accumulation, packer L1 accumulation off, as the gist harness runs them."""
    import importlib
    import sys

    sys.path.insert(0, f"{__import__('os').path.dirname(__file__)}/gist")
    cases = []
    for kind in ("2d", "1d"):
        sweep = importlib.import_module(f"matmul_sweep_{kind}")
        for s in sweep.llama_shapes(sweep.DEFAULT_MODELS, sweep.TOKENS, sweep.DEFAULT_PASSES):
            if sweep.route(s) not in sweep.KERNELS:
                continue
            cases.append(
                Case(
                    s.name,
                    (s.K, s.M) if s.transpose_a else (s.M, s.K),
                    (s.N, s.K) if s.transpose_b else (s.K, s.N),
                    "gist",
                    source="#57884 gist",
                    out_dtype="bf16",
                    transpose_a=s.transpose_a,
                    transpose_b=s.transpose_b,
                    fidelity="HiFi4",
                    fp32_acc=True,
                    packer_l1_acc=False,
                )
            )
    return cases


TIERS = {
    "issues": _issue_cases,
    "models": _model_cases,
    "generic": _generic_cases,
    "sharded": _sharded_cases,
    "gist": _gist_cases,
}


def get_cases(tiers=None):
    tiers = tiers or [t for t in TIERS if t not in ("traced", "gist")]
    cases = [c for t in tiers for c in TIERS[t]()]
    names = [c.name for c in cases]
    dupes = {n for n in names if names.count(n) > 1}
    assert not dupes, f"duplicate case names: {sorted(dupes)}"
    return cases


def cases_from_csv(path):
    """The cases of an earlier run_suite.py results CSV (one per case name), e.g. to rerun its traced tier
    without the trace JSON."""
    import ast
    import csv
    import re

    def shard(s):
        if not s:
            return None
        grid, shape, orientation = s.split(":")
        ranges = tuple(tuple(int(x) for x in re.findall(r"\d+", part)) for part in grid.split("+"))
        h, w = shape.split("x")
        return (ranges, (int(h), int(w)), orientation)

    cases = {}
    for r in csv.DictReader(open(path)):
        if r["case"] in cases:
            continue
        cg = r["core_grid"]
        cases[r["case"]] = Case(
            name=r["case"],
            a_shape=tuple(int(x) for x in r["a_shape"].split("x")),
            b_shape=tuple(int(x) for x in r["b_shape"].split("x")),
            tier=r["tier"],
            source=r["source"],
            a_dtype=r["a_dtype"],
            b_dtype=r["b_dtype"],
            out_dtype=r["out_dtype"] or None,
            a_mem=r["a_mem"],
            b_mem=r["b_mem"],
            out_mem=r["out_mem"],
            a_shard=shard(r.get("a_shard", "")),
            b_shard=shard(r.get("b_shard", "")),
            out_shard=shard(r.get("out_shard", "")),
            transpose_a=r["transpose_a"] == "1",
            transpose_b=r["transpose_b"] == "1",
            op=r["op"],
            bias=r["bias"] == "1",
            activation=r["activation"] or None,
            core_grid=None if cg == "" else (cg if cg == "device" else ast.literal_eval(cg)),
            fidelity=r["fidelity"],
            fp32_acc=r["fp32_acc"] == "1",
            packer_l1_acc=r["packer_l1_acc"] == "1",
            dst_full_sync=r.get("dst_full_sync", "0") == "1",
            tags=tuple(t for t in r["tags"].split(";") if t),
        )
    return list(cases.values())


# ---------------------------------------------------------------------------
# Tier: traced (real-model matmul/linear calls from the model tracer's master JSON)
# ---------------------------------------------------------------------------

TRACE_JSON_ENV = "MATMUL_OOB_TRACE_JSON"
_TRACE_DTYPES = {"BFLOAT16": "bf16", "BFLOAT8_B": "bfp8", "BFLOAT4_B": "bfp4", "FLOAT32": "fp32"}
_TRACE_SHARD_LAYOUTS = {"HEIGHT_SHARDED": "l1_height", "WIDTH_SHARDED": "l1_width", "BLOCK_SHARDED": "l1_block"}
_TRACE_ACTIVATIONS = {"gelu", "silu", "relu", "gelu_approx"}


def _traced_cases():
    """Single-device ttnn.matmul/ttnn.linear calls that reach the default config selection.

    Reads the master JSON produced by model_tracer (the `ttnn-operations-master-json` CI artifact), path given
    by $MATMUL_OOB_TRACE_JSON. Keeps calls without a program_config whose operands and output are interleaved or
    L1 sharded (with the traced shard specs), and skips anything it can't reproduce (multi-device placements,
    DRAM or ND sharding, global CBs, sub-devices, other dtypes).
    """
    import json
    import os
    import re

    path = os.environ.get(TRACE_JSON_ENV)
    if not path:
        return []
    ops = json.load(open(path))["operations"]
    cases = {}
    for op_name in ("ttnn.matmul", "ttnn.linear"):
        for cfg in ops.get(op_name, {}).get("configurations", []):
            a = cfg["arguments"]
            if a.get("program_config") or a.get("global_cb") or a.get("sub_device_id"):
                continue
            t0, t1 = a.get("arg0"), a.get("arg1")
            if not isinstance(t0, dict) or not isinstance(t1, dict):
                continue
            if any(t.get("tensor_placement", {}).get("mesh_device_shape") != "[1, 1]" for t in (t0, t1)):
                continue

            def mem(mc):
                """(placement, ShardDesc or None), or (None, None) if it can't be reproduced."""
                if mc is None:
                    return "dram", None
                l1 = mc.get("buffer_type") == "BufferType.L1"
                if not (mc.get("is_sharded") or not mc.get("interleaved", True)):
                    return ("l1" if l1 else "dram"), None
                layout = _TRACE_SHARD_LAYOUTS.get(str(mc.get("memory_layout")).replace("TensorMemoryLayout.", ""))
                if not l1 or layout is None or mc.get("nd_shard_spec"):
                    return None, None
                spec = mc.get("shard_spec")
                if spec is None:
                    return layout, None
                grid = tuple((r["start"]["x"], r["start"]["y"], r["end"]["x"], r["end"]["y"]) for r in spec["grid"])
                orientation = "col" if "COL" in str(spec["orientation"]) else "row"
                return layout, (grid, tuple(spec["shape"]), orientation)

            def dtype(t):
                return _TRACE_DTYPES.get(str(t).replace("DataType.", ""))

            (a_mem, a_shard), (b_mem, b_shard) = mem(t0["memory_config"]), mem(t1["memory_config"])
            out_mem, out_shard = mem(a.get("memory_config"))
            a_dt, b_dt = dtype(t0["original_dtype"]), dtype(t1["original_dtype"])
            out_dt = dtype(a["dtype"]["repr"]) if isinstance(a.get("dtype"), dict) else None
            if None in (a_mem, b_mem, out_mem, a_dt, b_dt) or (a.get("dtype") and out_dt is None):
                continue
            ckc = a.get("compute_kernel_config") or {}
            fidelity = str(ckc.get("math_fidelity", "MathFidelity.HiFi2")).replace("MathFidelity.", "")
            activation = a.get("activation")
            if activation is not None and activation not in _TRACE_ACTIVATIONS:
                continue
            core_grid = None
            if isinstance(a.get("core_grid"), dict):
                m = re.search(r"x=(\d+), y=(\d+)", a["core_grid"].get("value", ""))
                core_grid = (int(m[1]), int(m[2])) if m else None
            op = "linear" if op_name == "ttnn.linear" else "matmul"
            case = Case(
                name=f"t_{'s_' if 'l1_' in a_mem + b_mem + out_mem else ''}{op}_{cfg['config_hash'][:10]}",
                a_shape=tuple(t0["original_shape"]),
                b_shape=tuple(t1["original_shape"]),
                tier="traced",
                source=op_name,
                a_dtype=a_dt,
                b_dtype=b_dt,
                out_dtype=out_dt,
                a_mem=a_mem,
                b_mem=b_mem,
                out_mem=out_mem,
                a_shard=a_shard,
                b_shard=b_shard,
                out_shard=out_shard,
                transpose_a=bool(a.get("transpose_a", False)),
                transpose_b=bool(a.get("transpose_b", False)),
                op=op,
                bias=op == "linear" and isinstance(a.get("bias"), dict),
                activation=activation if op == "linear" else None,
                core_grid=core_grid,
                fidelity=fidelity if fidelity in FIDELITIES else "HiFi2",
                fp32_acc=bool(ckc.get("fp32_dest_acc_en", False)),
                packer_l1_acc=bool(ckc.get("packer_l1_acc", True)),
                tags=("core_grid",) if core_grid else ("default",),
            )
            key = tuple(v for k, v in case.__dict__.items() if k != "name")
            cases.setdefault(key, case)
    return list(cases.values())


TIERS["traced"] = _traced_cases
