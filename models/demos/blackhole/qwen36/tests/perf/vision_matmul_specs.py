# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Ground-truth inventory of every matmul the Qwen3.5 VISION TOWER issues.

Two independent views of the same seven matmuls, so a tuning sweep can never silently drift
from what the model actually runs:

1. :func:`derive_specs` -- ANALYTIC. Builds the (M, K, N) / dtype / program-config / fidelity
   table straight from ``VisionModelArgs`` (the same object the tower is built from), so it is
   correct for any model + mesh combination: Qwen3.5-9B on N300 (TP=2, activations FRACTURED)
   and Qwen3.5-27B on T3K (TP=8, activations REPLICATED -- see ``vision_ccl``).

2. :func:`capture_specs` -- OBSERVED. Runs one real ``DropInVisionTransformer.forward`` with
   ``ttnn.linear`` monkey-patched and records what every call site really passed.

``assert_specs_match`` diffs the two. That is the gate that makes an isolated sweep the SAME
experiment as the model (see the module-perf-optimization skill's rule 9): if someone changes a
reshape granularity, a weight dtype or a fidelity in the tower, the sweep fails instead of
optimizing a shape nothing runs.

The seven matmuls, per device, for the demo image (grid 1x86x128 = 11008 patches -> 12288 padded)
on N300 (TP=2), as the tower runs them after ``VisionModelArgs.vision_mm_plan`` was tuned:

    patch_embed  5504 x 1536 x  576   bf16 x bf16   HiFi2       2D g6x8 ibw6  (+bias)
    qkv          1536 x 1152 x 2304   bf16 x bf8b   HiFi2       2D g8x8 ibw18 (+bias folded in)
    wo           4096 x  768 x 1152   bf8b x bf8b   LoFi        2D g6x8 ibw24
    mlp_fc1      3072 x 1152 x 2176   bf16 x bf8b   HiFi2_fp16  2D g8x8 ibw6  (+bias, GELU fused)
    mlp_fc2      1536 x 2176 x 1152   bf16 x bf8b   HiFi2_fp16  2D g6x8 ibw4  (L1 output)
    merger_fc1   2752 x 4608 x 2304   bf16 x bf8b   HiFi2_fp16  auto (+bias, separate GELU)
    merger_fc2   2752 x 2304 x 4096   bf16 x bf8b   HiFi2_fp16  auto

``M`` is the PER-CHUNK row count: the tower reshapes ``[1, 1, S, K] -> [1, S/C, C, K]`` before each
matmul (metadata-only on a TILE tensor) and the program config is sized for one chunk, so `qkv` runs
as 8 chunks of 1536 rows, `wo` as 3 of 4096, and so on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from loguru import logger

import ttnn

SEQ_LEN_PAD = 2048

# Call sites of the tower's seven matmuls, in the order each site issues them.
# (module basename, ordinal within that basename) -> canonical family name.
_CALL_SITES = {
    ("patch_embed.py", 0): "patch_embed",
    ("vision_attention.py", 0): "qkv",
    ("vision_attention.py", 1): "wo",
    ("vision_mlp.py", 0): "mlp_fc1",
    ("vision_mlp.py", 1): "mlp_fc2",
    ("patch_merger.py", 0): "merger_fc1",
    ("patch_merger.py", 1): "merger_fc2",
}

FAMILIES = ("patch_embed", "qkv", "wo", "mlp_fc1", "mlp_fc2", "merger_fc1", "merger_fc2")


@dataclass
class MatmulSpec:
    """One matmul family, with per-device shapes -- i.e. what the device op sees."""

    name: str
    rows: int  # total activation rows the family processes (before chunking)
    chunk: int  # rows per matmul batch element (`M` of the program config)
    k: int
    n: int
    in0_dtype: Any
    in1_dtype: Any
    out_dtype: Any
    fidelity_name: str  # key into ModelArgs' compute_kernel_config_* attributes
    has_bias: bool  # a bias the model folds into ttnn.linear via `bias=`
    separate_bias: bool  # a bias the model adds as its own op AFTER the matmul
    activation: str | None  # the activation this matmul applies, whatever the mechanism
    # True when the tower folds `activation` into the program config's `fused_activation` instead of
    # passing ttnn.linear's `activation=` kwarg (which dispatches a separate unary op). Either way the
    # activation IS part of this matmul's cost -- a sweep that dropped it would compare a bare matmul
    # against the model's matmul+GELU and report a bogus 1.3x.
    activation_fused: bool = False
    # True when the tower lands this matmul's OUTPUT in L1 rather than DRAM. The baseline candidate
    # has to reproduce it, or the sweep compares a DRAM baseline against L1 candidates and reports
    # the L1 win twice.
    out_l1: bool = False
    baseline_progcfg: Any = None  # None == the model leaves this matmul on `auto`
    # True for a ROW-parallel matmul: its output is a partial sum, and the model adds the bias only
    # after the all-reduce / reduce-scatter. Folding such a bias into the matmul is numerically
    # WRONG (the collective would sum it TP times), so the sweep must not offer it.
    bias_after_collective: bool = False
    notes: str = ""

    @property
    def batch(self) -> int:
        assert self.rows % self.chunk == 0, f"{self.name}: rows {self.rows} % chunk {self.chunk}"
        return self.rows // self.chunk

    @property
    def m_tiles(self) -> int:
        return self.chunk // ttnn.TILE_SIZE

    @property
    def k_tiles(self) -> int:
        return self.k // ttnn.TILE_SIZE

    @property
    def n_tiles(self) -> int:
        return self.n // ttnn.TILE_SIZE

    @property
    def in0_shape(self) -> tuple[int, int, int, int]:
        return (1, self.batch, self.chunk, self.k)

    @property
    def in1_shape(self) -> tuple[int, int, int, int]:
        return (1, 1, self.k, self.n)

    def flops(self) -> int:
        return 2 * self.rows * self.k * self.n

    def summary(self) -> str:
        return (
            f"{self.name:<11} {self.rows}x{self.k}x{self.n} "
            f"(batch {self.batch} x M {self.chunk}) "
            f"{_DTYPE_NAMES[self.in0_dtype]} x {_DTYPE_NAMES[self.in1_dtype]} -> "
            f"{_DTYPE_NAMES[self.out_dtype]} {self.fidelity_name}"
            f"{' +bias' if self.has_bias else ''}"
            f"{' +sep_bias' if self.separate_bias else ''}"
            f"{(' +fused_' if self.activation_fused else ' +') + self.activation if self.activation else ''}"
            f"{' progcfg' if self.baseline_progcfg is not None else ' AUTO'}"
            f"{' L1out' if self.out_l1 else ''}"
        )


_DTYPE_NAMES = {
    ttnn.bfloat16: "bf16",
    ttnn.bfloat8_b: "bf8b",
    ttnn.bfloat4_b: "bf4b",
    ttnn.float32: "fp32",
}


def padded_seq_len(n_patches: int) -> int:
    """The rounding ``DropInVisionTransformer.forward`` applies to every image."""
    return ((n_patches // SEQ_LEN_PAD) + 1) * SEQ_LEN_PAD


def derive_specs(model_args, n_patches: int) -> dict[str, MatmulSpec]:
    """Analytic per-device matmul table for one image of ``n_patches`` patches.

    Every dimension is read off ``model_args`` (which the tower itself is built from), so this
    is automatically right for 9B/N300 (TP=2, fractured acts) and 27B/T3K (TP=8, replicated
    acts) without a per-model table.
    """
    vcfg = model_args.hf_config.vision_config
    tile = model_args.tile_size
    tp = model_args.cluster_shape[1]
    merge = vcfg.spatial_merge_size

    seq_len = padded_seq_len(n_patches)
    dim = model_args.dim
    # The block I/O contract: fractured along dim=3 unless TP cannot split dim into whole tiles.
    dim_local = dim if model_args.vision_replicated_acts else dim // tp

    # --- patch embed: Conv3d folded to a linear over the already-patchified pixels ---
    pixel_dim = vcfg.in_channels * vcfg.temporal_patch_size * vcfg.patch_size**2
    embed_rows = math.ceil(n_patches / tile) * tile  # VisionEmbed rounds uploaded rows to a tile

    # --- attention ---
    padded_head_dim = model_args.padded_head_dim
    n_local_heads = model_args.n_heads // tp
    n_local_kv_heads = model_args.n_kv_heads // tp
    local_qkv = (n_local_heads + 2 * n_local_kv_heads) * padded_head_dim
    wo_k = n_local_heads * padded_head_dim

    # --- MLP ---
    hidden_local = model_args.hidden_dim // tp

    # --- merger (consumes the tower output sliced back to the REAL patch count) ---
    merged_rows = n_patches // (merge**2)
    mlp_size = vcfg.hidden_size * (merge**2)
    mlp_local = mlp_size // tp

    # Every matmul's chunk / program config / fidelity now comes from the tower's own planner, so
    # the sweep automatically re-baselines whenever the tuning table changes.
    plans = {
        "patch_embed": model_args.vision_mm_plan(
            "patch_embed",
            rows=embed_rows,
            k=pixel_dim,
            n=dim_local,
            in0_dtype=ttnn.bfloat16,
            in1_dtype=ttnn.bfloat16,
            out_dtype=ttnn.bfloat16,
        ),
        "qkv": model_args.vision_mm_plan(
            "qkv",
            rows=seq_len,
            k=dim,
            n=local_qkv,
            in0_dtype=ttnn.bfloat16,
            in1_dtype=ttnn.bfloat8_b,
            out_dtype=ttnn.bfloat16,
        ),
        "wo": model_args.vision_mm_plan(
            "wo",
            rows=seq_len,
            k=wo_k,
            n=dim,
            in0_dtype=ttnn.bfloat8_b,
            in1_dtype=ttnn.bfloat8_b,
            out_dtype=ttnn.bfloat8_b,
        ),
        "mlp_fc1": model_args.vision_mm_plan(
            "mlp_fc1",
            rows=seq_len,
            k=dim,
            n=hidden_local,
            in0_dtype=ttnn.bfloat16,
            in1_dtype=ttnn.bfloat4_b if model_args.optimizations.bfp4_mlp else ttnn.bfloat8_b,
            out_dtype=ttnn.bfloat16,
            fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, False),
        ),
        "mlp_fc2": model_args.vision_mm_plan(
            "mlp_fc2",
            rows=seq_len,
            k=hidden_local,
            n=dim,
            in0_dtype=ttnn.bfloat16,
            in1_dtype=ttnn.bfloat8_b,
            out_dtype=ttnn.bfloat16,
        ),
        "merger_fc1": model_args.vision_mm_plan(
            "merger_fc1",
            rows=merged_rows,
            k=mlp_size,
            n=mlp_local,
            in0_dtype=ttnn.bfloat16,
            in1_dtype=ttnn.bfloat8_b,
            out_dtype=ttnn.bfloat16,
            fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, False),
        ),
        "merger_fc2": model_args.vision_mm_plan(
            "merger_fc2",
            rows=merged_rows,
            k=mlp_local,
            n=vcfg.out_hidden_size,
            in0_dtype=ttnn.bfloat16,
            in1_dtype=ttnn.bfloat8_b,
            out_dtype=ttnn.bfloat16,
        ),
    }

    def _spec(name, *, rows, k, n, in0, in1, out, has_bias, separate_bias, after_ccl, act, notes):
        plan = plans[name]
        return MatmulSpec(
            name=name,
            rows=rows,
            chunk=plan.chunk,
            k=k,
            n=n,
            in0_dtype=in0,
            in1_dtype=in1,
            out_dtype=out,
            fidelity_name=plan.fidelity,
            has_bias=has_bias,
            separate_bias=separate_bias,
            bias_after_collective=after_ccl,
            activation=act,
            # With a program config the tower fuses the activation INTO the matmul; the `activation=`
            # kwarg (a separate unary op) only survives on the auto-config fallback.
            activation_fused=bool(act) and plan.program_config is not None,
            out_l1=plan.memory_config is ttnn.L1_MEMORY_CONFIG,
            baseline_progcfg=plan.program_config,
            notes=notes,
        )

    specs = {
        "patch_embed": _spec(
            "patch_embed",
            rows=embed_rows,
            k=pixel_dim,
            n=dim_local,
            in0=ttnn.bfloat16,
            in1=ttnn.bfloat16,
            out=ttnn.bfloat16,
            has_bias=True,
            separate_bias=False,
            after_ccl=False,
            act=None,
            notes="VisionEmbed.forward -- runs once per image, not per block",
        ),
        # The qkv bias is folded into the matmul (column-parallel: the output is final).
        "qkv": _spec(
            "qkv",
            rows=seq_len,
            k=dim,
            n=local_qkv,
            in0=ttnn.bfloat16,
            in1=ttnn.bfloat8_b,
            out=ttnn.bfloat16,
            has_bias=True,
            separate_bias=False,
            after_ccl=False,
            act=None,
            notes="VisionAttention.forward_prefill",
        ),
        "wo": _spec(
            "wo",
            rows=seq_len,
            k=wo_k,
            n=dim,
            in0=ttnn.bfloat8_b,
            in1=ttnn.bfloat8_b,
            out=ttnn.bfloat8_b,
            has_bias=False,
            separate_bias=True,
            after_ccl=True,
            act=None,
            notes="VisionAttention.forward_prefill (row-parallel; output is a partial sum)",
        ),
        "mlp_fc1": _spec(
            "mlp_fc1",
            rows=seq_len,
            k=dim,
            n=hidden_local,
            in0=ttnn.bfloat16,
            in1=ttnn.bfloat4_b if model_args.optimizations.bfp4_mlp else ttnn.bfloat8_b,
            out=ttnn.bfloat16,
            has_bias=True,
            separate_bias=False,
            after_ccl=False,
            act="gelu",
            notes="MLP.forward",
        ),
        "mlp_fc2": _spec(
            "mlp_fc2",
            rows=seq_len,
            k=hidden_local,
            n=dim,
            in0=ttnn.bfloat16,
            in1=ttnn.bfloat8_b,
            out=ttnn.bfloat16,
            has_bias=False,
            separate_bias=True,
            after_ccl=True,
            act=None,
            notes="MLP.forward (row-parallel; output is a partial sum)",
        ),
        "merger_fc1": _spec(
            "merger_fc1",
            rows=merged_rows,
            k=mlp_size,
            n=mlp_local,
            in0=ttnn.bfloat16,
            in1=ttnn.bfloat8_b,
            out=ttnn.bfloat16,
            has_bias=True,
            separate_bias=False,
            after_ccl=False,
            act="gelu",
            notes="PatchMerger.forward -- runs once per image, not per block",
        ),
        "merger_fc2": _spec(
            "merger_fc2",
            rows=merged_rows,
            k=mlp_local,
            n=vcfg.out_hidden_size,
            in0=ttnn.bfloat16,
            in1=ttnn.bfloat8_b,
            out=ttnn.bfloat16,
            has_bias=False,
            separate_bias=True,
            after_ccl=True,
            act=None,
            notes="PatchMerger.forward (row-parallel; output is a partial sum)",
        ),
    }
    return specs


# --------------------------------------------------------------------------------------- capture


@dataclass
class CapturedCall:
    """What one ``ttnn.linear`` call site really passed, as the device op sees it."""

    family: str
    in0_shape: tuple
    in1_shape: tuple
    in0_dtype: str
    in1_dtype: str
    out_dtype: str
    in0_buffer: str
    out_buffer: str
    bias_shape: tuple | None
    activation: str | None
    progcfg: str | None
    compute_kernel: str | None
    kwargs: dict = field(default_factory=dict)


def _local_shape(tensor) -> tuple:
    """Per-device shape of a (possibly mesh-sharded) tensor -- what the device op sees."""
    try:
        shards = ttnn.get_device_tensors(tensor)
        if shards:
            return tuple(shards[0].padded_shape)
    except Exception:  # pragma: no cover - single-device / API drift
        pass
    return tuple(tensor.padded_shape)


def capture_specs(model, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> dict[str, CapturedCall]:
    """Run one real tower forward with ``ttnn.linear`` patched; return what each site issued."""
    import inspect

    real_linear = ttnn.linear
    seen: dict[str, int] = {}
    calls: dict[str, CapturedCall] = {}

    def wrapper(a, b, *args, **kwargs):
        site = "?"
        for frame in inspect.stack()[1:6]:
            base = Path(frame.filename).name
            if base != "vision_matmul_specs.py":
                site = base
                break
        ordinal = seen.get(site, 0)
        seen[site] = ordinal + 1
        family = _CALL_SITES.get((site, ordinal))
        out = real_linear(a, b, *args, **kwargs)
        if family is not None:
            bias = kwargs.get("bias")
            pc = kwargs.get("program_config")
            calls[family] = CapturedCall(
                family=family,
                in0_shape=_local_shape(a),
                in1_shape=_local_shape(b),
                in0_dtype=str(a.dtype).split(".")[-1],
                in1_dtype=str(b.dtype).split(".")[-1],
                out_dtype=str(out.dtype).split(".")[-1],
                in0_buffer=str(a.memory_config().buffer_type).split(".")[-1],
                out_buffer=str(out.memory_config().buffer_type).split(".")[-1],
                bias_shape=None if bias is None else _local_shape(bias),
                activation=kwargs.get("activation"),
                progcfg=None if pc is None else repr(pc),
                compute_kernel=repr(kwargs.get("compute_kernel_config")),
            )
        return out

    ttnn.linear = wrapper
    try:
        out = model(pixel_values, grid_thw)
        ttnn.deallocate(out)
    finally:
        ttnn.linear = real_linear
    return calls


def assert_specs_match(specs: dict[str, MatmulSpec], calls: dict[str, CapturedCall]) -> None:
    """Fail loudly if the analytic table and the real tower disagree on ANY matmul."""
    problems = []
    missing = sorted(set(specs) - set(calls))
    extra = sorted(set(calls) - set(specs))
    if missing:
        problems.append(f"analytic families the tower never issued: {missing}")
    if extra:
        problems.append(f"tower matmuls missing from the analytic table: {extra}")

    for name in sorted(set(specs) & set(calls)):
        spec, call = specs[name], calls[name]
        got_in0, got_in1 = call.in0_shape, call.in1_shape
        want_in0, want_in1 = spec.in0_shape, spec.in1_shape
        # in0 is 4D [1, batch, chunk, K]; a batch of 1 may be reported as rank<4 by some ops.
        if tuple(got_in0[-2:]) != (spec.chunk, spec.k) or math.prod(got_in0[:-2]) != spec.batch:
            problems.append(f"{name}: in0 {got_in0} != analytic {want_in0}")
        if tuple(got_in1[-2:]) != (spec.k, spec.n):
            problems.append(f"{name}: in1 {got_in1} != analytic {want_in1}")
        for label, got, want in (
            ("in0_dtype", call.in0_dtype, _DTYPE_NAMES[spec.in0_dtype]),
            ("in1_dtype", call.in1_dtype, _DTYPE_NAMES[spec.in1_dtype]),
            ("out_dtype", call.out_dtype, _DTYPE_NAMES[spec.out_dtype]),
        ):
            want_full = {"bf16": "BFLOAT16", "bf8b": "BFLOAT8_B", "bf4b": "BFLOAT4_B", "fp32": "FLOAT32"}[want]
            if got.upper() != want_full:
                problems.append(f"{name}: {label} {got} != analytic {want}")
        if (call.bias_shape is not None) != spec.has_bias:
            problems.append(f"{name}: bias folded={call.bias_shape is not None} != analytic {spec.has_bias}")
        want_act = None if spec.activation_fused else spec.activation
        if (call.activation or None) != want_act:
            problems.append(f"{name}: activation kwarg {call.activation!r} != analytic {want_act!r}")
        pc_act = None
        if call.progcfg is not None and "fused_activation=std::nullopt" not in call.progcfg:
            pc_act = "gelu" if "GELU" in call.progcfg else "?"
        want_pc_act = spec.activation if spec.activation_fused else None
        if pc_act != want_pc_act:
            problems.append(f"{name}: program config fused_activation {pc_act!r} != analytic {want_pc_act!r}")
        if (call.progcfg is not None) != (spec.baseline_progcfg is not None):
            problems.append(
                f"{name}: program_config present={call.progcfg is not None} "
                f"!= analytic {spec.baseline_progcfg is not None}"
            )
        elif call.progcfg is not None and call.progcfg != repr(spec.baseline_progcfg):
            problems.append(
                f"{name}: program_config\n  model:    {call.progcfg}\n  analytic: {spec.baseline_progcfg!r}"
            )

    if problems:
        raise AssertionError(
            "vision matmul specs do not match the model -- the sweep would tune shapes the tower "
            "never runs:\n  - " + "\n  - ".join(problems)
        )
    logger.info(f"vision matmul specs match the model on all {len(specs)} matmuls")
