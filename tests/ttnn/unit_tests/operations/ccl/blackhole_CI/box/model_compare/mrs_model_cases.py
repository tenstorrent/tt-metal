# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-device matmul + reduce-scatter call sites of the models in this repo, as test cases.

Every case is one row-parallel linear of a model, at the shape and settings the model passes on one device of its
tensor-parallel group, followed by the reduce-scatter over that group. Each case lists the paths to compare:

  prod      what the model runs today at this site (a fused op, or the unfused matmul + reduce-scatter)
  alts      other existing paths the model has for the site (e.g. the unfused fallback of a fused site, or an
            opt-in fused path that is off by default)
  ours      ttnn.experimental.matmul_reduce_scatter_prefill, plus the epilogue the model fuses into its op
            (bias, addcmul) as separate ops, and a concat for the concat-over-K input

Paths (implemented in test_matmul_reduce_scatter_model_compare_bh.py):

  mmrs_dit       minimal_matmul_strided_reduce_scatter_async as RowParallelLinear.forward_fused_addcmul calls it
                 (models/tt_dit/layers/linear.py:688), blocking from get_fused_mmrs_config on the 12x10 grid
  unfused_dit    RowParallelLinear.forward + ttnn.addcmul: minimal_matmul (get_matmul_config, 11x10 grid on a BH
                 Galaxy) or ttnn.linear 1D for M <= 128, then CCLManager.reduce_scatter (reduce_scatter_minimal_async)
  unfused_gptoss gpt_oss apply_output_projection + RS: ttnn.matmul (bf8b in, bf16 out) + bias, reduce_scatter_minimal_async
  unfused_m3     minimax_m3 apply_output_projection + apply_reduce_scatter (sharded residual default)
  unfused_muse   muse_glimmer prefill: minimal_matmul (policy config) + typecast to ccl dtype + reduce_scatter_minimal_async
  mrs_async_muse muse_glimmer "fused_reduce" policy: matmul_reduce_scatter_async (multichip_decoder.py:494)
  mrs_async_qwen36  qwen36 GDN out-proj prefill: matmul_reduce_scatter_async (tp_common.py:891)

Mesh: every runnable case is a TP=4 group, run as the 1x4 ring of a Blackhole QuietBox 2 (FABRIC_1D_RING, 2 links).
For the tt_dit models that group is axis 0 of a BH Galaxy 4x8 / 4x32 (also a ring of 4 with 2 links). TP=8 sites
(Flux2 sp0tp1, gpt_oss on WH / BH Galaxy rows) are listed with tp=8 and skipped on a 4-chip box.

Only the per-device shapes matter: M is the rows one device holds after sequence parallelism, K the device's slice of
the contraction dim, N the full output width before the scatter (the scattered output is N / tp).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Case:
    name: str
    model: str
    site: str  # file:line of the call (models/ relative)
    prod_mesh: str  # production mesh and the group axis
    M: int
    K: int  # per device; for a concat-over-K input the sum of the halves
    N: int  # full width before the scatter
    act: str  # activation dtype as the op receives it: bf16 | bf8b | fp32
    weight: str  # bf16 | bf8b | bf4b
    fidelity: str  # LoFi | HiFi2 | HiFi3 | HiFi4
    prod: str  # path the model runs today
    alts: tuple = ()
    tp: int = 4
    topology: str = "ring"  # ring | linear
    fp32_acc: bool = True
    approx: bool = False
    bias: bool = False
    addcmul: str | None = None  # None | "bcast" (gate [1,1,1,N/tp]) | "token" (gate [1,1,M,N/tp])
    k_split: tuple | None = None  # concat-over-K input: (k_prefix, k_suffix)
    out_dtype: str | None = None  # dtype the existing path writes (None: its default)
    ccl_dtype: str | None = None  # muse: dtype of the reduce-scatter
    large: bool = False  # heavy host setup / long device time: deselect with -m "not large"
    notes: str = ""


DIT_RING = "BH Galaxy 4x8 sp1tp0 / ring of 4 on axis 0"


def _flux2():
    # Flux2 (pipeline_flux2.py): D = 48 * 128 = 6144, MLP 3D = 18432. sp1tp0 on a BH 4x8: TP4 (axis 0), SP8.
    # Spatial tokens (W/16)^2 / SP: 1024px 512, 2048px 2048, 4096px 8192, 8192px 32768; prompt 512 / SP8 = 64.
    # ff2 K = 18432 / 4 = 4608; proj_out K = (6144 + 18432) / 4 = 6144 as [x_c_attn 1536, x_c_mlp 4608].
    out = []
    common = dict(model="Flux2", N=6144, act="bf16", weight="bf16", fidelity="HiFi2", addcmul="bcast")
    for px, m in ((1024, 512), (2048, 2048), (4096, 8192), (8192, 32768)):
        large = m >= 8192
        out.append(
            Case(
                f"flux2_ff2_{px}px",
                site="tt_dit/blocks/transformer_block_opt.py:364 (ff.ff2)",
                prod_mesh=DIT_RING,
                M=m,
                K=4608,
                prod="mmrs_dit",
                alts=("unfused_dit",),
                large=large,
                **common,
            )
        )
        out.append(
            Case(
                f"flux2_proj_out_{px}px",
                site="tt_dit/models/transformers/transformer_flux2.py:193 (single blocks 1-47)",
                prod_mesh=DIT_RING,
                M=m + 64,
                K=6144,
                k_split=(1536, 4608),
                prod="mmrs_dit",
                alts=("unfused_dit",),
                large=large,
                notes="concat-over-K input [x_c_attn, x_c_mlp], spatial + prompt rows",
                **common,
            )
        )
        out.append(
            Case(
                f"flux2_proj_out_last_{px}px",
                site="tt_dit/models/transformers/transformer_flux2.py:193 (last single block)",
                prod_mesh=DIT_RING,
                M=m,
                K=6144,
                k_split=(1536, 4608),
                prod="mmrs_dit",
                alts=("unfused_dit",),
                large=large,
                notes="last block drops the prompt rows",
                **common,
            )
        )
    out.append(
        Case(
            "flux2_ff_context",
            site="tt_dit/blocks/transformer_block_opt.py:398 (ff_context.ff2)",
            prod_mesh=DIT_RING,
            M=64,
            K=4608,
            prod="mmrs_dit",
            alts=("unfused_dit",),
            notes="prompt 512 tokens / SP8",
            **common,
        )
    )
    # sp0tp1 on a BH 4x8: TP8 (axis 1), SP4. Needs a ring of 8: listed, skipped on a QuietBox.
    tp8 = "BH Galaxy 4x8 sp0tp1 / ring of 8 on axis 1"
    for px, m in ((1024, 1024), (2048, 4096)):
        out.append(
            Case(
                f"flux2_tp8_ff2_{px}px",
                site="tt_dit/blocks/transformer_block_opt.py:364",
                prod_mesh=tp8,
                M=m,
                K=2304,
                prod="mmrs_dit",
                alts=("unfused_dit",),
                tp=8,
                **common,
            )
        )
        out.append(
            Case(
                f"flux2_tp8_proj_out_{px}px",
                site="tt_dit/models/transformers/transformer_flux2.py:193",
                prod_mesh=tp8,
                M=m + 128,
                K=3072,
                k_split=(768, 2304),
                prod="mmrs_dit",
                alts=("unfused_dit",),
                tp=8,
                **common,
            )
        )
    out.append(
        Case(
            "flux2_tp8_ff_context",
            site="tt_dit/blocks/transformer_block_opt.py:398",
            prod_mesh=tp8,
            M=128,
            K=2304,
            prod="mmrs_dit",
            alts=("unfused_dit",),
            tp=8,
            **common,
        )
    )
    return out


def _wan():
    # Wan2.2 A14B (transformer_wan.py:262): D 5120, FFN 13824 -> ff2 K = 13824 / 4 = 3456, bias. 81 frames -> 21
    # latent frames, 1560 (480p) / 3600 (720p) tokens per frame, padded to 32 * SP.
    out = []
    for mesh, sp, rows in (
        ("BH Galaxy 4x8", 8, ((480, 4096), (720, 9472))),
        ("BH 4x32", 32, ((480, 1024), (720, 2368))),
    ):
        for res, m in rows:
            out.append(
                Case(
                    f"wan_ff2_{res}p_sp{sp}",
                    model="Wan2.2",
                    site="tt_dit/models/transformers/wan2_2/transformer_wan.py:244 (ffn.ff2)",
                    prod_mesh=f"{mesh} / ring of 4 on axis 0, SP{sp}",
                    M=m,
                    K=3456,
                    N=5120,
                    act="bf16",
                    weight="bf16",
                    fidelity="HiFi2",
                    bias=True,
                    addcmul="bcast",
                    prod="mmrs_dit",
                    alts=("unfused_dit",),
                )
            )
    return out


def _ltx():
    # LTX-2.3 (transformer_ltx.py:107): video D 4096 / FFN 16384 -> K 4096; audio D 2048 / FFN 8192 -> K 2048; bias.
    # 1088x1920, 145 frames: 9690 (stage 1) / 38760 (stage 2) tokens -> 9728 / 38912 at SP8; CI 153 frames -> 10240 /
    # 40960. 4x32 (SP32): 320 (stage 1), 1216 / 1280 (stage 2). i2v smoke 512x768x25: 64 / 192.
    out = []
    rows = (
        ("145f_stage1_sp8", "BH Galaxy 4x8", 1216),
        ("145f_stage2_sp8", "BH Galaxy 4x8", 4864),
        ("153f_stage1_sp8", "BH Galaxy 4x8", 1280),
        ("153f_stage2_sp8", "BH Galaxy 4x8", 5120),
        ("stage1_sp32", "BH 4x32", 320),
        ("153f_stage2_sp32", "BH 4x32", 1280),
        ("i2v_smoke_stage1", "BH Galaxy 4x8", 64),
        ("i2v_smoke_stage2", "BH Galaxy 4x8", 192),
    )
    for tag, mesh, m in rows:
        out.append(
            Case(
                f"ltx_video_ff2_{tag}",
                model="LTX-2.3",
                site="tt_dit/models/transformers/ltx/transformer_ltx.py:323 (ffn.ff2)",
                prod_mesh=f"{mesh} / ring of 4 on axis 0",
                M=m,
                K=4096,
                N=4096,
                act="bf16",
                weight="bf16",
                fidelity="HiFi2",
                bias=True,
                addcmul="bcast",
                prod="mmrs_dit",
                alts=("unfused_dit",),
                notes="t2v gate is [1,1,1,D/tp]; i2v uses a per-token gate (same op cost class)",
            )
        )
    out.append(
        Case(
            "ltx_audio_ff2",
            model="LTX-2.3",
            site="tt_dit/models/transformers/ltx/transformer_ltx.py:323 (audio_ff.ff2)",
            prod_mesh="BH Galaxy 4x8 and 4x32 / ring of 4 on axis 0",
            M=32,
            K=2048,
            N=2048,
            act="bf16",
            weight="bf16",
            fidelity="HiFi2",
            bias=True,
            addcmul="bcast",
            prod="mmrs_dit",
            alts=("unfused_dit",),
        )
    )
    return out


def _h3():
    # MiniMax-H3 (minimax_h3/mmrs_config.py:49): ff2 K = 14336 / 4 = 3584, N 5376, no bias, per-token gate. M = packed
    # rows (text + 2 audio + video) bucketed to the 4x8 / 4x32 ladders, divided by SP.
    out = []
    for mesh, sp, ms in (
        ("BH Galaxy 4x8", 8, (2912, 3616, 4864, 5664, 7104, 8256, 9600, 11136, 13952)),
        ("BH 4x32", 32, (704, 992, 1376, 1920, 2688, 3776)),
    ):
        for m in ms:
            out.append(
                Case(
                    f"h3_ff2_m{m}_sp{sp}",
                    model="MiniMax-H3",
                    site="tt_dit/models/transformers/minimax_h3/transformer_block_minimax_h3.py:375 (ff.ff2)",
                    prod_mesh=f"{mesh} / ring of 4 on axis 0, SP{sp}",
                    M=m,
                    K=3584,
                    N=5376,
                    act="bf16",
                    weight="bf16",
                    fidelity="HiFi2",
                    approx=True,
                    addcmul="token",
                    prod="mmrs_dit",
                    alts=("unfused_dit",),
                    large=m >= 9600,
                    notes="4x8 preset runs without persistent CCL buffers; here persistent (as 4x32)",
                )
            )
    return out


def _gpt_oss():
    # gpt_oss (configs/*/config.json): hidden 2880, 64 heads x 64. o_proj row-parallel: K = 4096 / tp; N = padded
    # local hidden * tp: TP4 2880/4 = 720 -> 736 (N 2944); TP8 360 -> 384 (N 3072). Bias real on TP device 0 only.
    # Blackhole: the fused path is off (#46181 race, operations.py:128), every prefill length runs the unfused path.
    out = []
    for m in (128, 1024, 4096, 8192):
        out.append(
            Case(
                f"gpt_oss_o_proj_s{m}",
                model="gpt-oss-20b",
                site="demos/gpt_oss/tt/attention/operations.py:101 (+ apply_allreduce RS half)",
                prod_mesh="BH QB2 1x4 / ring of 4 on axis 1",
                M=m,
                K=1024,
                N=2944,
                act="bf8b",
                weight="bf8b",
                fidelity="LoFi",
                bias=True,
                out_dtype="bf16",
                prod="unfused_gptoss",
                notes="fused MMRS gated off on Blackhole (#46181); the trailing all-gather + slice is not measured",
            )
        )
    out.append(
        Case(
            "gpt_oss_o_proj_s1024_tp8",
            model="gpt-oss-20b/120b",
            site="demos/gpt_oss/tt/attention/operations.py:205",
            prod_mesh="WH LoudBox 1x8 / WH Galaxy 4x8, ring of 8 on axis 1",
            M=1024,
            K=512,
            N=3072,
            act="bf8b",
            weight="bf8b",
            fidelity="LoFi",
            bias=True,
            tp=8,
            prod="mmrs_gptoss",
            alts=("unfused_gptoss",),
            out_dtype="bf16",
            notes="the only live fused LLM site; Wormhole only, ring of 8 (BH Galaxy runs the unfused alt)",
        )
    )
    return out


def _minimax_m3():
    # MiniMax-M3 / M3.1 o_proj (minimax_m3/tt/attention/prefill.py:383): K = 64 * 128 / 4 = 2048, N 6144, no bias, BH
    # Galaxy 8x4 TP4 on axis 1, production topology Linear. Per-chip M = chunk / SP: 5120 / 8 = 640, CI 1024 / 8 = 128.
    out = []
    for m in (640, 128):
        out.append(
            Case(
                f"minimax_m3_o_proj_m{m}",
                model="MiniMax-M3",
                site="demos/minimax_m3/tt/attention/prefill.py:411",
                prod_mesh="BH Galaxy 8x4 / line of 4 on axis 1",
                M=m,
                K=2048,
                N=6144,
                act="bf8b",
                weight="bf8b",
                fidelity="LoFi",
                fp32_acc=False,
                out_dtype="bf16",
                topology="linear",
                prod="unfused_m3",
                notes="precise_matmuls off: bf8b cast, ttnn.matmul default config",
            )
        )
        out.append(
            Case(
                f"minimax_m31_o_proj_m{m}",
                model="MiniMax-M3.1",
                site="demos/minimax_m3/tt/attention/prefill.py:411",
                prod_mesh="BH Galaxy 8x4 / line of 4 on axis 1",
                M=m,
                K=2048,
                N=6144,
                act="bf16",
                weight="bf8b",
                fidelity="HiFi3",
                out_dtype="bf16",
                topology="linear",
                prod="unfused_m3",
                notes="precise_matmuls on: bf16, HiFi3",
            )
        )
    return out


def _muse():
    # Muse-Glimmer-30B on a QB2 1x4 (multichip_decoder.py:99): hidden 6656, intermediate 19968, 32 x 128 heads.
    # o_proj K = 4096 / 4 = 1024; down_proj K = 19968 / 4 = 4992; N 6656; bf4b weights, LoFi, fp32 acc off.
    # Prefill chunk 4096. CCL dtype: bf8b on full-attention layers (matmul output bf8b too), bf16 on sliding layers.
    out = []
    for lin, k in (("o_proj", 1024), ("down_proj", 4992)):
        for kind, out_dtype, ccl in (("sliding", "bf16", "bf16"), ("full", "bf8b", "bf8b")):
            for m in (1024, 4096):
                out.append(
                    Case(
                        f"muse_{lin}_{kind}_m{m}",
                        model="Muse-Glimmer-30B",
                        site="demos/muse_glimmer_30b_qb2/tt/multichip_decoder.py:454 (_linear -> _reduce)",
                        prod_mesh="BH QB2 1x4 / ring of 4 on axis 1",
                        M=m,
                        K=k,
                        N=6656,
                        act="bf16",
                        weight="bf4b",
                        fidelity="LoFi",
                        fp32_acc=False,
                        out_dtype=out_dtype,
                        ccl_dtype=ccl,
                        prod="unfused_muse",
                        alts=("mrs_async_muse",),
                        notes="fused_reduce (matmul_reduce_scatter_async) is opt-in by policy, off by default",
                    )
                )
    return out


def _qwen36():
    # Qwen3.6-27B GDN out_proj prefill (tp_common.py:891): K = 48 * 128 / 4 = 1536, N 5120, bf8b weight, fp32
    # activation, HiFi2 approx. Reached only by test_gdn_out_agmm_vs_mmrs (production runs column-parallel AG + MM).
    out = []
    for m in (64, 512, 2048):
        out.append(
            Case(
                f"qwen36_gdn_out_proj_t{m}",
                model="Qwen3.6-27B",
                site="demos/blackhole/qwen36/tt/gdn/tp.py:1010 -> tp_common.py:921",
                prod_mesh="BH QB2 1x4 / ring of 4 on axis 1",
                M=m,
                K=1536,
                N=5120,
                act="fp32",
                weight="bf8b",
                fidelity="HiFi2",
                approx=True,
                prod="mrs_async_qwen36",
                notes="ours takes a bf16 activation (fp32 unsupported); production default is column-parallel AG+MM",
            )
        )
    return out


CASES = _flux2() + _wan() + _ltx() + _h3() + _gpt_oss() + _minimax_m3() + _muse() + _qwen36()
CASES_BY_NAME = {c.name: c for c in CASES}
assert len(CASES_BY_NAME) == len(CASES), "duplicate case names"
