# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Build the #65148 shape set + roofline for every matmul in #65142's scope.

Scope (#65142): Qwen3.8-27B projections and LM head; FLUX.2-dev DiT projections and text encoder;
Panoptic-DeepLab 1x1 convs and bilinear-upsample matmuls. Inputs are the Tracy ops reports of the
profiling tests in this folder:
    --pdl      models/experimental/panoptic_deeplab/tests/pcc/test_tt_model.py -k panoptic_deeplab_110_cores
    --qwen     test_qwen38_profile.py
    --flux2    test_flux2_profile.py (1 double + 2 single blocks)
    --flux2-te test_flux2_text_encoder_profile.py

Roofline per chip = max(compute, DRAM):
    compute = 2*M*K*N (each dim rounded up to a 32 tile) / (70% of CORES * 4096 / fidelity passes * CLOCK)
    DRAM    = bytes of the operands that live in DRAM (tile formats: bfp8 = 1088 B, bfp4 = 576 B per tile)
              / (90% of 512 GB/s)
Ethernet does not apply: the matmuls themselves move no data between chips (the collectives next to
some of them are separate ops, listed in the `ccl` column).

Tracy logs a cached `minimal_matmul` program's input shapes once, so a later call with a different M
shows the first M. M is therefore taken from the model's structure (prompt length for Qwen prefill;
image / text / joint token counts for FLUX.2), cross-checked against the ops around each matmul.
"""

import argparse
import re
from dataclasses import dataclass, field

import pandas as pd

CORES = 110  # op-visible worker grid per chip (11x10); 120 Tensix, one column runs fast dispatch
CLOCK_HZ = 1.35e9  # CHIP_FREQ[MHz] in profile_log_device.csv during these runs
FLOPS_PER_CYCLE_LOFI = 4096  # 8x16 @ 16x16 per cycle per core, tech_reports/GEMM_FLOPS
FIDELITY_PASSES = {"LoFi": 1, "HiFi2": 2, "HiFi3": 3, "HiFi4": 4}
COMPUTE_UTIL = 0.70
DRAM_PEAK_BPS = 512e9  # Blackhole GDDR6 spec peak per chip
DRAM_UTIL = 0.90
BYTES = {"BFLOAT16": 2.0, "FLOAT32": 4.0, "BFLOAT8_B": 1088 / 1024, "BFLOAT4_B": 576 / 1024}
DT = {"BFLOAT16": "bf16", "FLOAT32": "fp32", "BFLOAT8_B": "bfp8", "BFLOAT4_B": "bfp4"}
MM_CODES = ("MatmulDeviceOperation", "MinimalMatmulDeviceOperation")


def tile(x):
    return -(-int(x) // 32) * 32


def peak_flops(fidelity):
    return CORES * FLOPS_PER_CYCLE_LOFI / FIDELITY_PASSES[fidelity] * CLOCK_HZ


@dataclass
class Mm:
    model: str
    group: str  # unit the calls count per, e.g. "decode token"
    name: str
    m: int  # activation rows (true M), per batch entry
    k: int  # weight rows
    n: int  # weight cols
    dt_in0: str
    dt_in1: str
    dt_out: str
    mem_in0: str
    mem_in1: str
    mem_out: str
    fidelity: str
    op: str
    config: str
    calls: float
    measured_us: float
    b0: int = 1  # activation batch (PDL upsample)
    b1: int = 1  # weight batch, 1 = broadcast
    k_in0: int | None = None  # in0 width when a second activation carries the rest of K
    n_out: int | None = None  # output width when it differs from N (SwiGLU halves it)
    extra: list = field(default_factory=list)  # other operands: (rows, cols, dtype, mem)
    epilogue: str = "-"
    ccl: str = "-"  # separate collective next to the matmul on this mesh
    cores: int = 0

    @property
    def M(self):
        return self.b0 * self.m

    @property
    def flops(self):
        return 2 * self.b0 * tile(self.m) * tile(self.k) * tile(self.n)

    @property
    def dram_bytes(self):
        total = 0.0
        if "DRAM" in self.mem_in0:
            total += self.b0 * tile(self.m) * tile(self.k_in0 or self.k) * BYTES[self.dt_in0]
        if "DRAM" in self.mem_in1:
            total += self.b1 * tile(self.k) * tile(self.n) * BYTES[self.dt_in1]
        for rows, cols, dt, mem in self.extra:
            if "DRAM" in mem:
                total += tile(rows) * tile(cols) * BYTES[dt]
        if "DRAM" in self.mem_out:
            total += self.b0 * tile(self.m) * tile(self.n_out or self.n) * BYTES[self.dt_out]
        return total

    @property
    def compute_us(self):
        return self.flops / (COMPUTE_UTIL * peak_flops(self.fidelity)) * 1e6

    @property
    def dram_us(self):
        return self.dram_bytes / (DRAM_UTIL * DRAM_PEAK_BPS) * 1e6

    @property
    def roofline_us(self):
        return max(self.compute_us, self.dram_us)

    @property
    def bound(self):
        return "compute" if self.compute_us >= self.dram_us else "DRAM"

    @property
    def ratio(self):
        return self.measured_us / self.roofline_us


# ---------------------------------------------------------------- CSV helpers


def load(path):
    df = pd.read_csv(path, low_memory=False)
    stack, sections = [], []
    for _, r in df.iterrows():
        oc = str(r["OP CODE"])
        if pd.isna(r["DEVICE ID"]):  # signpost
            if oc.endswith("_END"):
                if stack:
                    stack.pop()
            else:
                stack.append(oc[:-6] if oc.endswith("_START") else oc)
            sections.append(None)
            continue
        sections.append("/".join(stack) if stack else None)
    df["section"] = sections
    df["instance"] = df["GLOBAL CALL COUNT"] - df["DEVICE ID"].fillna(0)
    return df


def dims(r, t):
    out = []
    for a in "WZYX":
        v = r.get(f"{t}_{a}_PAD[LOGICAL]")
        if not isinstance(v, str):
            return None
        out.append(int(re.match(r"(\d+)\[(\d+)\]", v).group(2)))
    return out


def mean_over_devices(df, r):
    same = df[(df["instance"] == r["instance"]) & (df["OP CODE"] == r["OP CODE"])]
    return same["DEVICE KERNEL DURATION [ns]"].mean() / 1e3


def mem(s):
    s = re.sub(r"DEV_\d+_", "", s if isinstance(s, str) else "-")
    return {
        "DRAM_INTERLEAVED": "DRAM",
        "L1_INTERLEAVED": "L1",
        "DRAM_WIDTH_SHARDED": "DRAM-WS",
        "L1_WIDTH_SHARDED": "L1-WS",
        "L1_HEIGHT_SHARDED": "L1-HS",
        "L1_BLOCK_SHARDED": "L1-BS",
    }.get(s, s)


def attr(a, key):
    m = re.search(rf"'{key}': '([^']*)'", a if isinstance(a, str) else "")
    return m.group(1) if m else None


def describe(r):
    m = re.search(r"'(?:program_config|config)': '(?:std::optional<)?(\w+)\((.*?)\)'", r["ATTRIBUTES"])
    name, kv = (
        (m.group(1), dict(x.strip().split("=", 1) for x in m.group(2).split(";") if "=" in x)) if m else ("-", {})
    )
    if r["OP CODE"] == "MinimalMatmulDeviceOperation":
        chunks = attr(r["ATTRIBUTES"], "chunks")
        op = "minimal_matmul" + (f"_split ({chunks} out)" if chunks not in (None, "1") else "")
        return op, f"M{kv.get('M_block_size')}/K{kv.get('K_block_size')}/N{kv.get('N_block_size')}"
    short = {
        "MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig": "DRAM-sharded",
        "MatmulMultiCoreReuseMultiCast1DProgramConfig": "1D mcast",
        "MatmulMultiCoreReuseMultiCastProgramConfig": "2D mcast",
    }.get(name, name)
    return f"matmul {short}", f"in0_block_w {kv.get('in0_block_w')}"


def epilogue(r):
    a = r["ATTRIBUTES"]
    parts = []
    m = re.search(r"fused_activation=BasicUnaryWithParam<float>\(op_type=UnaryOpType::(\w+)", a) or re.search(
        r"'fused_activation': '.*?UnaryOpType::(\w+)", a
    )
    if m:
        parts.append(m.group(1).lower())
    if attr(a, "fuse_swiglu") == "true":
        parts.append("swiglu")
    if attr(a, "fused_ternary_scalar") not in (None, "std::nullopt"):
        parts.append("addcmul")
    return "+".join(parts) or "-"


def mm_from_row(df, r, *, model, group, name, calls, m=None, ccl="-"):
    """One matmul row. `m` overrides the logged activation rows (stale for cached minimal_matmul)."""
    a, b, o = dims(r, "INPUT_0"), dims(r, "INPUT_1"), dims(r, "OUTPUT_0")
    logged_m = a[2]
    m = m or logged_m
    extra = []
    for t in ("INPUT_2", "INPUT_3", "INPUT_4"):
        x = dims(r, t)
        if not x:
            continue
        rows = x[0] * x[1] * x[2]
        if x[2] == logged_m and x[2] > 1:  # an activation-shaped operand follows the true M
            rows = x[0] * x[1] * m
        extra.append((rows, x[3], r[f"{t}_DATATYPE"], mem(r[f"{t}_MEMORY"])))
    chunks = attr(r["ATTRIBUTES"], "chunks")
    n_out = None if chunks not in (None, "1") else o[3]
    op, cfg = describe(r)
    return Mm(
        model=model,
        group=group,
        name=name,
        m=m,
        k=b[2],
        n=b[3],
        b0=a[0] * a[1],
        b1=b[0] * b[1],
        k_in0=a[3] if a[3] != b[2] else None,
        n_out=n_out if n_out and n_out != b[3] else None,
        extra=extra,
        dt_in0=r["INPUT_0_DATATYPE"],
        dt_in1=r["INPUT_1_DATATYPE"],
        dt_out=r["OUTPUT_0_DATATYPE"],
        mem_in0=mem(r["INPUT_0_MEMORY"]),
        mem_in1=mem(r["INPUT_1_MEMORY"]),
        mem_out=mem(r["OUTPUT_0_MEMORY"]),
        fidelity=r["MATH FIDELITY"],
        op=op,
        config=cfg,
        epilogue=epilogue(r),
        ccl=ccl,
        calls=calls,
        measured_us=mean_over_devices(df, r),
        cores=int(r["CORE COUNT"]),
    )


# ---------------------------------------------------------------- PDL (one chip, no collectives)

PDL_STAGE_IO = {"RES2": (128, 256), "RES3": (256, 512), "RES4": (512, 1024), "RES5": (1024, 2048)}


def pdl_rows(path):
    df = load(path)
    rows = []
    for _, r in df.iterrows():
        sec = r["section"] or ""
        leaf = sec.split("/")[-1]
        if r["OP CODE"] == "MatmulDeviceOperation":
            a, b = dims(r, "INPUT_0"), dims(r, "INPUT_1")
            k, n = a[3], b[3]
            if "RESNET_LAYER_" in leaf:
                stage = leaf.split("_")[-1]
                cin, cout = PDL_STAGE_IO[stage]
                role = "shortcut" if (k, n) == (cin, cout) else ("reduce" if n < k else "expand")
                name = f"{stage.lower()} {role}"
            elif leaf == "ASPP":
                name = "ASPP image pool" if a[2] == 1 else {2048: "ASPP 1x1 branch", 1280: "ASPP projection"}[k]
            elif leaf.endswith("HEAD_DECODER"):
                name = "Decoder skip projection"
            elif leaf == "SEMSEG_HEAD_PREDICTION":
                name = "Semantic logits (19 of 32 cols)"
            elif leaf.startswith("INSEMB_HEAD_") and leaf.endswith("_PREDICTION"):
                name = "Center / offset heads (1, 2 of 32 cols)"
            elif leaf.startswith("FINAL_UPSAMPLE"):
                name = "Bilinear upsample, " + ("W pass" if a[2] < 512 else "H pass")
            else:
                name = leaf
            rows.append(mm_from_row(df, r, model="PDL", group="inference", name=name, calls=1))
        elif r["OP CODE"] == "Conv2dDeviceOperation" and "window_hw=(1;1)" in r["ATTRIBUTES"]:
            a = r["ATTRIBUTES"]
            nhwc = [int(x) for x in re.search(r"'input_tensor_shape': '\{([^}]*)\}'", a).group(1).split(";")]
            sh, sw = (int(x) for x in re.search(r"stride_hw=\((\d+);(\d+)\)", a).groups())
            cout = int(attr(a, "output_channels"))
            m = nhwc[0] * -(-nhwc[1] // sh) * -(-nhwc[2] // sw)
            rows.append(
                Mm(
                    model="PDL",
                    group="inference",
                    name=f"{leaf.split('_')[-1].lower()} shortcut, stride 2 (conv2d path)",
                    m=m,
                    k=nhwc[3],
                    n=cout,
                    dt_in0=r["INPUT_0_DATATYPE"],
                    dt_in1=r["INPUT_1_DATATYPE"],
                    dt_out=attr(a, "dtype").replace("DataType::", ""),
                    mem_in0=mem(r["INPUT_0_MEMORY"]),
                    mem_in1=mem(r["INPUT_1_MEMORY"]),
                    mem_out=mem(r["OUTPUT_0_MEMORY"]),
                    fidelity=r["MATH FIDELITY"],
                    op="conv2d 1x1 stride 2",
                    config="sliding-window conv",
                    extra=[(1, cout, "BFLOAT8_B", "DRAM")] if attr(a, "has_bias") == "true" else [],
                    calls=1,
                    measured_us=r["DEVICE KERNEL DURATION [ns]"] / 1e3,
                    cores=int(r["CORE COUNT"]),
                )
            )
    totals = {"inference": df["DEVICE KERNEL DURATION [ns]"].sum() / 1e3}
    return rows, totals


# ---------------------------------------------------------------- Qwen3.8-27B (1x4 mesh, TP4)

QWEN_LAYERS = {"linear_attention": 48, "full_attention": 16}
# Profiled stack: layer 0 (linear attention, GDN) then layer 3 (full attention), 4 projections each.
QWEN_ORDER = [
    ("GDN in_proj (qkv/z/b/a)", "linear_attention", False),
    ("GDN out_proj", "linear_attention", True),
    ("MLP gate_up", "linear_attention", False),
    ("MLP down", "linear_attention", True),
    ("Full-attn qkvg (q/gate/k/v)", "full_attention", False),
    ("Full-attn o_proj", "full_attention", True),
    ("MLP gate_up", "full_attention", False),
    ("MLP down", "full_attention", True),
]


def qwen_rows(path):
    df = load(path)
    d0 = df[df["DEVICE ID"] == 0]
    rows, totals = [], {}
    for sec, g in d0.groupby("section", sort=False):
        if sec is None:
            continue
        g = g.reset_index()
        decode = sec.startswith("DECODE")
        group = "decode token" if decode else f"prefill M={int(sec.split('_M')[1])}"
        mms = g[g["OP CODE"].isin(MM_CODES)]
        assert len(mms) == 12, (sec, len(mms))
        for j, (_, r) in enumerate(mms.iterrows()):
            if j < 8:
                name, kind, reduced = QWEN_ORDER[j]
                calls = QWEN_LAYERS[kind]
                ccl = ("all-reduce after" if decode else "reduce-scatter after") if reduced else "-"
            else:
                if not decode:
                    continue  # the LM head runs decode-style once per prefill; listed under decode
                name, calls, ccl = "LM head (vocab chunk)", 1, "-"
            m = None if decode else int(sec.split("_M")[1])
            rows.append(mm_from_row(df, r, model="Qwen3.8-27B", group=group, name=name, calls=calls, m=m, ccl=ccl))

        # full-model device time of this section: pre + 48 x GDN layer + 16 x full layer + head/post
        def norm_before(i):
            while i > 0 and "Norm" not in g.loc[i, "OP CODE"]:
                i -= 1
            return i

        idx = list(mms.index)
        b0, b1, b2 = norm_before(idx[0]), norm_before(idx[4]), norm_before(idx[8])
        dur = g["DEVICE KERNEL DURATION [ns]"] / 1e3
        totals[group] = dur[:b0].sum() + 48 * dur[b0:b1].sum() + 16 * dur[b1:b2].sum() + dur[b2:].sum()
    return rows, totals


# ---------------------------------------------------------------- FLUX.2-dev (2x2 mesh, SP2 x TP2)


def flux2_rows(path_dit, path_te):
    rows, totals = [], {}
    df = load(path_dit)
    d0 = df[df["DEVICE ID"] == 0]
    step = d0[d0.section == "DIT_STEP"].reset_index()
    mm = [i for i in step.index if step.loc[i, "OP CODE"] in MM_CODES]
    assert len(mm) == 21, len(mm)  # 6 pre + 8 double + 3 + 3 single + proj_out

    ce = d0[(d0.section == "CONTEXT_EMBEDDER") & d0["OP CODE"].isin(MM_CODES)].iloc[0]
    txt = dims(ce, "INPUT_0")[2]  # prompt tokens per chip (unique program: logged shape is exact)
    img = dims(step.loc[mm[5]], "INPUT_0")[2]  # x_embedder rows = image tokens per chip
    joint = img + txt

    def add(i, name, calls, m=None, ccl="-", group="denoising step"):
        rows.append(mm_from_row(df, step.loc[i], model="FLUX.2-dev", group=group, name=name, calls=calls, m=m, ccl=ccl))

    pre = ["Timestep MLP linear_1", "Timestep MLP linear_2", "Guidance MLP linear_1", "Guidance MLP linear_2"]
    for i, name in zip(mm[:4], pre):
        add(i, name, 1)
    add(mm[4], "time_mod (all 17 modulation sets)", 1)
    add(mm[5], "x_embedder", 1)
    ag, rs = "all-gather before", "reduce-scatter after"
    double = [
        ("Double QKV, img", img, ag),
        ("Double QKV, txt", txt, ag),
        ("Double attn out, img", img, ag),
        ("Double attn out, txt", txt, ag),
        ("Double FF1, img", img, ag),
        ("Double FF2, img", img, rs),
        ("Double FF1, txt", txt, ag),
        ("Double FF2, txt", txt, rs),
    ]
    for i, (name, m, c) in zip(mm[6:14], double):
        add(i, name, 8, m=m, ccl=c)
    # first single block = blocks 1-47; the final block skips the prompt's MLP and proj_out
    add(mm[14], "Single QKV", 47, m=joint, ccl=ag)
    add(mm[15], "Single FF1", 47, m=joint, ccl=ag)
    add(mm[16], "Single out (attn + MLP)", 47, m=joint, ccl=rs)
    add(mm[17], "Single QKV", 1, m=joint, ccl=ag)
    add(mm[18], "Single FF1, final block", 1, m=img, ccl=ag)
    add(mm[19], "Single out, final block", 1, m=img, ccl=rs)
    add(mm[20], "proj_out", 1, ccl=rs)
    rows.append(mm_from_row(df, ce, model="FLUX.2-dev", group="image", name="context_embedder", calls=1))

    # full step: pre + 8 x double + 47 x single + final single + post, cut at the block boundaries
    dur = step["DEVICE KERNEL DURATION [ns]"] / 1e3
    codes = list(step["OP CODE"])
    tern = [i for i, c in enumerate(codes) if c.startswith("Ternary")]
    d_end = [t for t in tern if t > mm[13]][0] + 1  # addcmul closing the double block
    s1_end = [t for t in tern if t > mm[16]][0] + 1
    s2_end = [t for t in tern if t > mm[19]][0] + 1
    pre_t, dbl, s1, s2, post = (
        dur[: mm[5] + 1].sum(),
        dur[mm[5] + 1 : d_end].sum(),
        dur[d_end:s1_end].sum(),
        dur[s1_end:s2_end].sum(),
        dur[s2_end:].sum(),
    )
    totals["denoising step"] = pre_t + 8 * dbl + 47 * s1 + s2 + post
    print(f"[flux2 step] pre {pre_t:.0f} us, double {dbl:.0f}, single {s1:.0f}, final single {s2:.0f}, post {post:.0f}")

    dt = load(path_te)
    t0 = dt[(dt["DEVICE ID"] == 0) & (dt.section == "TEXT_ENCODER")].reset_index()
    te = [i for i in t0.index if t0.loc[i, "OP CODE"] in MM_CODES]
    assert len(te) == 10, len(te)  # 2 layers x (qkv, o_proj, gate, up, down)
    te_names = [
        ("Text encoder qkv_proj", "-"),
        ("Text encoder o_proj", "all-gather before and after"),
        ("Text encoder gate", "-"),
        ("Text encoder up", "-"),
        ("Text encoder down", "reduce-scatter after"),
    ]
    for j, i in enumerate(te):
        name, c = te_names[j % 5]
        rows.append(mm_from_row(dt, t0.loc[i], model="FLUX.2-dev", group="image", name=name, calls=20, ccl=c))
    tdur = t0["DEVICE KERNEL DURATION [ns]"] / 1e3
    norm = [i for i in t0.index if "Norm" in t0.loc[i, "OP CODE"]]
    l0 = max(n for n in norm if n < te[0])
    l1 = max(n for n in norm if n < te[5])
    layer = tdur[l0:l1].sum()
    totals["text encoder"] = tdur[:l0].sum() + 40 * layer + (tdur[l1:].sum() - layer)
    return rows, totals


# ---------------------------------------------------------------- output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pdl", required=True)
    p.add_argument("--qwen", required=True)
    p.add_argument("--flux2", required=True)
    p.add_argument("--flux2-te", required=True)
    p.add_argument("--csv", required=True)
    a = p.parse_args()
    pdl, pdl_tot = pdl_rows(a.pdl)
    qwen, qwen_tot = qwen_rows(a.qwen)
    flux, flux_tot = flux2_rows(a.flux2, a.flux2_te)
    rows = qwen + flux + pdl
    out = pd.DataFrame(
        [
            dict(
                model=x.model,
                group=x.group,
                matmul=x.name,
                calls=x.calls,
                M=x.M,
                act_batch=x.b0,
                K=x.k,
                N=x.n,
                w_batch=x.b1,
                act_dtype=DT[x.dt_in0],
                wt_dtype=DT[x.dt_in1],
                out_dtype=DT[x.dt_out],
                fidelity=x.fidelity,
                act_mem=x.mem_in0,
                wt_mem=x.mem_in1,
                out_mem=x.mem_out,
                extra_operands=";".join(f"{r}x{c} {DT[t]} {m}" for r, c, t, m in x.extra),
                epilogue=x.epilogue,
                ccl=x.ccl,
                stock_op=x.op,
                config=x.config,
                cores=x.cores,
                measured_us=round(x.measured_us, 2),
                gflop=round(x.flops / 1e9, 3),
                dram_mb=round(x.dram_bytes / 1e6, 3),
                compute_roof_us=round(x.compute_us, 2),
                dram_roof_us=round(x.dram_us, 2),
                roofline_us=round(x.roofline_us, 2),
                bound=x.bound,
                measured_over_roofline=round(x.ratio, 3),
            )
            for x in rows
        ]
    )
    out.to_csv(a.csv, index=False)
    for model, tot in (("Qwen3.8-27B", qwen_tot), ("FLUX.2-dev", flux_tot), ("PDL", pdl_tot)):
        sel = out[out.model == model]
        for grp, g in sel.groupby("group", sort=False):
            tm = (g.calls * g.measured_us).sum() / 1e3
            tr = (g.calls * g.roofline_us).sum() / 1e3
            print(
                f"{model:12s} {grp:16s} calls {g.calls.sum():6g}  matmul {tm:8.2f} ms  roofline {tr:8.2f} ms  ({tm / tr:.2f}x)"
            )
        print(
            f"{model:12s} model device time per unit (ms): " + ", ".join(f"{k} {v / 1e3:.2f}" for k, v in tot.items())
        )


if __name__ == "__main__":
    main()
