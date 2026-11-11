# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The one configuration object of the PUP prefill runner (Qwen3.5-9B / the FLUX.3 Klein prompt upsampler on one
(1, 4) Blackhole column). Every value is fixed in code; nothing is read from the environment.

Two sources:
- the session-5 column profile (``Qwen36ModelArgs.set_column_prefill_profile``: token-parallel residual stream, LoFi
  projections, bf8 CCLs, L1 placement, per-site AGMM launch configs), measured in tt-blaze's co_prefill harness;
- the shipped MLP precision fix on top of it: the dense MLP at HiFi2 with fp32 accumulation (``QWEN36_MLP_FIDELITY=hifi2``
  and no ``QWEN36_TOKPAR_MLP_CFG2``), and the MLP / residual activations in DRAM for the large buckets (the 3200-row
  bucket of the 10-reference requests: the fp32-accumulating MLP's circular buffers do not fit next to an L1 placement
  there). Teacher-forced on the PUP caption set this halves the tokens that differ from HF (27 -> 13 at a logit gap
  >= 0.1) for +4 ms on a 1 MP request and +20 ms on 10 x 512^2.

The d-gen prefill runs no LM head: decode computes the last prompt token, so the LM-head precision choice of the
co-resident serving path (hifi4_fp32) does not apply here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PupPrefillConfig:
    # --- serving shape (the manifest may override num_users / max_seq_len per deployment) ---
    tp: int = 4
    """One (1, 4) column: KV head h and GDN heads [8h, 8h + 8) live on chip h."""
    chunk_size: int = 4096
    """One chunk per PUP request: the longest request (10 references + a 450-token instruction) is 3502 tokens."""
    max_seq_len: int = 4096
    num_users: int = 4
    block_size: int = 64
    """The model's paged-attention block; each slot owns max_seq_len / block_size consecutive blocks."""
    segment_tokens: int = 4096
    """The model's chunk trace size: a prompt shorter than this runs as ONE segment (a traced short bucket up to 2048,
    else one masked bucket of the next 128 multiple), as today's prefill runs a request's tail as one bucket. 2048 (the
    model default) splits a 3177-token 10-reference request into a 2048 chunk trace + a tail, which gave different
    image rows past position 2048 (DGEN_PREFILL_LOG h2)."""

    # --- session-5 column profile ---
    column_profile: bool = True

    # --- the shipped MLP precision fix ---
    mlp_fidelity: str = "hifi2"
    """Dense MLP math fidelity, fp32 accumulation (the column profile alone runs LoFi)."""
    mlp_bf16_dest: bool = False
    """``prefill_tokpar_mlp_cfg2`` (bf16 dest accumulation in gate / up / down): off."""
    mlp_l1_max_rows: int = 1152
    """Token-parallel MLP activations in L1 up to this many prefill rows (the 1 MP bucket), DRAM above."""
    res_l1_max_rows: int = 3072
    """Token-parallel residual stream in L1 up to this many rows; the 3200 bucket keeps it in DRAM."""

    # --- fast runtime (fast_runtime.py): system-turn snapshot + traced buckets + per-layer-group publish ---
    fast: bool = True
    """The grouped (session-5) prefill with a system-turn snapshot; False = the generic B=1 path (runtime.py)."""
    fast_buckets: tuple = (
        128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1536, 1792, 2048, 2304, 2560, 2688, 2816, 3072, 3328,
        3584,
    )  # fmt: skip
    """Traced tail buckets (rows; multiples of 128). A tail runs in the smallest bucket >= its length: 128 apart up to
    1280 (1 MP tail 1088 -> 1152), then 256 (10 refs: 2656 -> 2688). No 1408: its one-row (non token-parallel) body
    is slower than the token-parallel 1536 (5 refs: 75.4 vs 72.0 ms)."""
    publish_segments: int = 4
    """Traces per bucket = layer groups published (and reported to the engine) as each finishes."""
    prefix_max: int = 1024
    """Longest system turn kept as a snapshot (rows; the snapshot's K / V buffers are this long)."""
    prefix_align: int = 128
    """Snapshot boundary: the system turn floored to this (the GDN sub-chunk)."""
    prefix_slots: int = 4
    """Snapshots kept (LRU): one per distinct system turn (PUP: t2i, i2i)."""
    tokpar_min_rows: int = 1536
    """Token-parallel residual stream from this many rows (session 5: costs more than it saves below)."""

    # --- media ---
    media_source: str = "spliced"
    """Image embeddings come from the encoder service's embedding store; prefill runs no vision tower."""

    def __post_init__(self):
        if self.mlp_fidelity not in ("lofi", "hifi2", "hifi4"):
            raise ValueError(f"mlp_fidelity {self.mlp_fidelity!r}")
        if self.max_seq_len % self.block_size or self.chunk_size % self.block_size:
            raise ValueError("max_seq_len and chunk_size must be whole paged-attention blocks")
        if self.segment_tokens % 128 or self.segment_tokens > self.max_seq_len:
            raise ValueError(f"segment_tokens {self.segment_tokens}: a multiple of 128, <= max_seq_len")
        if self.chunk_size > self.max_seq_len:
            raise ValueError(f"chunk_size {self.chunk_size} > max_seq_len {self.max_seq_len}")
        if any(b % 128 for b in self.fast_buckets) or self.prefix_max % self.block_size or self.prefix_align % 128:
            raise ValueError("fast buckets / prefix_align: multiples of 128; prefix_max: whole blocks")
        if self.media_source not in ("spliced",):
            raise ValueError("the d-gen prefill takes images only from the embedding store")

    @property
    def blocks_per_slot(self) -> int:
        return self.max_seq_len // self.block_size

    def with_serving(
        self,
        *,
        num_users: int | None = None,
        max_seq_len: int | None = None,
        chunk_size: int | None = None,
        fast: bool | None = None,
    ):
        """The same model settings at a deployment's slot count / sequence length (from the runner's params)."""
        from dataclasses import replace

        kw = {}
        if fast is not None:
            kw["fast"] = bool(fast)
        if num_users is not None:
            kw["num_users"] = int(num_users)
        if max_seq_len is not None:
            kw["max_seq_len"] = int(max_seq_len)
        if chunk_size is not None:
            kw["chunk_size"] = int(chunk_size)
        return replace(self, **kw)

    def apply(self, args) -> None:
        """Set the model arguments (``Qwen36ModelArgs``) before the layers are built."""
        import ttnn

        if self.column_profile:
            args.set_column_prefill_profile()
        args.mlp_math_fidelity = {
            "lofi": ttnn.MathFidelity.LoFi,
            "hifi2": ttnn.MathFidelity.HiFi2,
            "hifi4": ttnn.MathFidelity.HiFi4,
        }[self.mlp_fidelity]
        args.prefill_tokpar_mlp_cfg2 = self.mlp_bf16_dest
        # layer._l1_rows_ok: an int switch = L1 only while the prefill's total rows are <= it
        args.prefill_tokpar_mlp_l1 = int(self.mlp_l1_max_rows)
        args.prefill_tokpar_res_l1 = int(self.res_l1_max_rows)
        args.prefill_media_source = self.media_source


DEFAULT_CONFIG = PupPrefillConfig()
