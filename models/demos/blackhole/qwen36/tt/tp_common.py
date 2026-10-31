# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TP helpers for Qwen3.5/3.6 on Blackhole (9B single-device + 27B TP=4 / TP=8).

Used only when num_devices > 1. DRAM-sharded matmul cfgs, prefill progcfgs,
mesh shard/replicate, FP8 dequant, HF weight reorder for per-device sharding.
"""

import math
import os

import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.blackhole.qwen36.tt import weight_prefetch

# Hardware constants
TILE_SIZE = 32
DRAM_CORES = 8
DRAM_GRID = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(DRAM_CORES - 1, 0))})


# Compute kernel configs
COMPUTE_HIFI2 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=True,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)

# Same as COMPUTE_HIFI2 except LoFi (used for BFP4 GDN in-proj in decode only).
COMPUTE_LOFI = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=True,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)


FABRIC_PAYLOAD_ENV = "QWEN36_FABRIC_PAYLOAD_BYTES"


def fabric_router_config():
    """QWEN36_FABRIC_PAYLOAD_BYTES (default 8192; 0 = the fabric default of 4352): max fabric packet payload for the
    multi-device meshes, the value qwen38_27b_qb2 uses. Larger packets cut per-packet overhead in the decode
    all-reduces: TP=8 decode 20.25 -> 19.87 ms/token at B=1 and 25.31 -> 24.92 at B=32 (15232, the Blackhole maximum,
    is slower than 8192). Returns a ttnn.FabricRouterConfig for set_fabric_config, or None for the fabric default."""
    payload = int(os.environ.get(FABRIC_PAYLOAD_ENV, "8192") or 0)
    if payload <= 0:
        return None
    router = ttnn.FabricRouterConfig()
    router.max_packet_payload_size_bytes = payload
    return router


def verify_device_topk_m():
    """Device-side top-k for the sampled verify (QWEN36_VERIFY_DEVICE_TOPK, default 1 = on; 0 turns it off): 0 when off, else
    M = QWEN36_VERIFY_TOPK_M (default 64) candidates per vocab shard. Read by the model (third verify trace),
    the engine (memory costs, spec_mode "topk") and the vLLM class (SpecPlan accept_modes / topk_candidates), so
    one process sees one answer. Note 26 (work/notes/26_device_topk_sampled_verify.md) is the design."""
    if not bool(int(os.environ.get("QWEN36_VERIFY_DEVICE_TOPK", "1"))):
        return 0
    return int(os.environ.get("QWEN36_VERIFY_TOPK_M", "64"))


def verify_dar_enabled():
    """QWEN36_VERIFY_DAR (default "1"; "0" = today's verify, byte for byte): the traced verify body runs on the replicated
    RES40 residual of the plain decode (DecodeResidualAllReduce: one AG at entry, one all_reduce_async per row-parallel
    output, local sharded norms), 2 CCLs per layer instead of 4. It still needs QWEN36_DECODE_ALLREDUCE=1 and a bucket of
    <= 32 rows; the model decides the rest (see Qwen36Model._verify_dar). Note 42 has the design."""
    return os.environ.get("QWEN36_VERIFY_DAR", "1") == "1"


def verify_attn_lean_enabled():
    """QWEN36_VERIFY_ATTN_LEAN (default "1"; "0" = today's verify attention, byte for byte): the full-attention layers of the
    traced verify take the lean decode op sequence (TPAttention._forward_verify_lean) and the model hands them cos/sin
    already in the layout the rope wants (one reshape per trace, not four per layer). Note 42 has the op list."""
    return os.environ.get("QWEN36_VERIFY_ATTN_LEAN", "1") == "1"


def kv_write_multirow_enabled():
    """QWEN36_KV_WRITE_MULTIROW (default "0" = today's T fused calls per layer): the traced verify's KV write
    (TPAttention._write_kv_fused_groups) goes out as ONE paged_fused_update_cache(rows_per_user=T) per layer: each core
    owns one user's T candidate rows and read-modify-writes each cache tile they touch once, instead of T calls of
    one row per user (each with 2 slices + 2 to_memory_config). Cache bytes are identical. Needs a ttnn build whose
    paged_fused_update_cache takes rows_per_user (probed in attention/tp.py; otherwise the flag is ignored)."""
    return os.environ.get("QWEN36_KV_WRITE_MULTIROW", "0") == "1"


def spec_sdpa_packed_enabled():
    """QWEN36_SPEC_SDPA_PACKED (default "0" = today's fused spec-verify SDPA, byte for byte): the fused spec-verify SDPA
    (spec_multi_pos_tiles = Tg) packs each user group's Tg candidates x NH local q heads into ONE 32-row q tile
    (spec_pack_heads = NH; 4 x 3 = 12 rows at TP8) instead of Tg padded row-tiles that use NH of their 32 rows, so every
    KV chunk costs one q row-tile of QK, softmax and PV work instead of Tg, with the causal mask built per row. The KV
    chunking, core split and reduction order are unchanged, so the attention output rows are bit-identical. Needs a
    ttnn build whose paged SDPA decode takes spec_pack_heads (probed in attention/tp.py; otherwise ignored)."""
    return os.environ.get("QWEN36_SPEC_SDPA_PACKED", "0") == "1"


def verify_layout_cuts_enabled():
    """QWEN36_VERIFY_LAYOUT_CUTS (default "1"; "0" = today's verify, byte for byte): drop layout copies that the plain
    decode does not have (GDN verify projection: no sharded-to-interleaved + copy of the norm output). Note 42."""
    return os.environ.get("QWEN36_VERIFY_LAYOUT_CUTS", "1") == "1"


def attn_glue_cuts_enabled():
    """QWEN36_ATTN_GLUE_CUTS (default "0" = today's ops, byte for byte): drop pure data-movement ops in the B > 1
    decode attention (the traced verify, the drafter steps and reseed, plain batched decode): the partial-RoPE
    pass-through slice lands in DRAM itself instead of slice-to-L1 + a to_memory_config copy (one op less per q / k
    rope call). Same bytes into the same concat, so every output is bit-identical."""
    return os.environ.get("QWEN36_ATTN_GLUE_CUTS", "0") == "1"


def verify_heads_dense_enabled():
    """QWEN36_VERIFY_HEADS_DENSE (default "0" = today's traced-verify attention, byte for byte): the lean verify
    attention (TPAttention._forward_verify_lean) splits heads with the prefill nlp_create_qkv_heads into the dense
    [1, heads, rows, HD] layout (rows = users x candidates in one tile-row per head: 32 tiles for q|k instead of 256
    mostly-padding tiles) and runs the q and k RMSNorm, the (1+w) weight multiply and the partial RoPE ONCE over q|k
    stacked as heads, then transposes q / k / v into the decode layout the spec SDPA and the KV write read. Same
    arithmetic ops on the same values (each is row-wise or element-wise), only data movement changes: 15 ops per
    attention layer instead of 24 (no decode head split + 3 sharded-to-interleaved, one norm / multiply / RoPE instead
    of two, no per-head-group transposes and copies). After the SDPA the heads are concatenated the same way round
    (transpose + prefill nlp_concat_heads: 2 ops instead of reshard + nlp_concat_heads_decode + sharded-to-interleaved).
    Read when the attention weights are loaded (the stacked [1, NH+NKV, 1, HD] norm weight) and at trace capture."""
    return os.environ.get("QWEN36_VERIFY_HEADS_DENSE", "0") == "1"


_PREFILL_RS_LINKS = None


def prefill_rs_links(rows=None):
    """QWEN36_PREFILL_RS_LINKS (default unset = None = tt_all_reduce auto-detects, today's behaviour): when set to an
    int N > 0, the prefill MLP-down and attention-wo tt_all_reduce reduce-scatters use N Ethernet links. On the 1x8 TP
    mesh those calls pass cluster_axis=0 (the size-1 axis), so the auto-detect gives 1 link although the 8-wide axis
    has 2 (the fused GDN-out matmul_reduce_scatter_prefill already runs 2). Read once. rows <= TILE_SIZE (decode /
    verify widths that share the MLP code path) always gets None so decode and its traces are unchanged."""
    global _PREFILL_RS_LINKS
    if _PREFILL_RS_LINKS is None:
        raw = os.environ.get("QWEN36_PREFILL_RS_LINKS", "").strip()
        n = int(raw) if raw else 0
        _PREFILL_RS_LINKS = n if n > 0 else 0
    if rows is not None and rows <= TILE_SIZE:
        return None
    return _PREFILL_RS_LINKS or None


_PREFILL_RS_WORKERS = None


def prefill_rs_workers(rows=None):
    """QWEN36_PREFILL_RS_WORKERS (default unset = None = tt_all_reduce's num_workers_per_link=2, today's behaviour):
    when set to an int N > 0, the prefill MLP-down and attention-wo reduce-scatters (tt_all_reduce ->
    reduce_scatter_minimal_async) run N workers per link and direction. These reduce-scatters are worker-bound, and the
    op's own Blackhole default for more than 1 MB per link is 8 workers; tt_all_reduce pins 2. Device time on the 8-chip
    ring (16 calls in one trace), [rows, 5120] bf16 partial per device, 1 link x 2 workers -> 2 links x 8: 8192 rows
    1.78 -> 0.90 ms (2 links x 2: 1.25), 2048 (L1) 0.445 -> 0.229, 1024 0.223 -> 0.121, 512 0.123 -> 0.069, 128 0.045 ->
    0.034. The output is bit-identical to today's in every combination tried (the ring reduction order of a tile does
    not depend on the worker split). Meant for use with QWEN36_PREFILL_RS_LINKS=2.
    rows <= TILE_SIZE (decode / verify widths that share the MLP code path) always get None, so decode and its traces
    are unchanged. Read once."""
    global _PREFILL_RS_WORKERS
    if _PREFILL_RS_WORKERS is None:
        raw = os.environ.get("QWEN36_PREFILL_RS_WORKERS", "").strip()
        n = int(raw) if raw else 0
        _PREFILL_RS_WORKERS = n if n > 0 else 0
    if rows is not None and rows <= TILE_SIZE:
        return None
    return _PREFILL_RS_WORKERS or None


def prefill_rs_kwargs(rows):
    """Extra tt_all_reduce kwargs of a prefill MLP-down / attention-wo reduce-scatter of `rows` rows: {} (today's call,
    byte for byte) unless QWEN36_PREFILL_RS_WORKERS is set. The link count stays with prefill_rs_links."""
    w = prefill_rs_workers(rows)
    return {} if w is None else {"num_workers_per_link": w}


_R6_FLAGS = {}


def _r6_flag(name):
    """Round-6 prefill / join flags: "1" turns the lever on; default "0" = the r5 code path, byte for byte. Read once."""
    v = _R6_FLAGS.get(name)
    if v is None:
        v = _R6_FLAGS[name] = os.environ.get(name, "0") == "1"
    return v


def prefill_host_tolist_enabled():
    """QWEN36_PREFILL_HOST_TOLIST: the spec engine's prefill (MTPSpecEngine._prefill_row) converts the whole prompt
    prefix to Python ints with ONE tolist() instead of a 0-d tensor per token (~1 us/token: 32 ms at 32K, 130 ms at
    128K, paid by every vLLM chunk call of a long prompt while the device idles), builds the int32 prompt row from the
    tensor and hashes the chunk-resume digests from its bytes. Same ints, same row, same digests."""
    return _r6_flag("QWEN36_PREFILL_HOST_TOLIST")


def join_logits_one_copy_enabled():
    """QWEN36_JOIN_LOGITS_ONE_COPY: the replicated final prefill logits (all-gathered: every device holds the same
    [1, 1, vocab] tile row) are read from ONE device instead of concatenating all of them (the engine kept only the
    first copy anyway): 1/8 of the D2H. The chunk-trace / masked-bucket ends of a prompt (2K-aligned prompts, the
    eager path) read them."""
    return _r6_flag("QWEN36_JOIN_LOGITS_ONE_COPY")


def join_topp_candidates_enabled():
    """QWEN36_JOIN_TOPP_CANDIDATES: the vLLM TT plugin's host sampler (a join's first token) gets candidate paths for
    requests with top-k disabled, instead of vLLM's full 248K-logit sort (21-30 ms per join on the pod CPU; the
    candidate path 1.9 ms): exact for top_p >= 1, the top-M candidates with an exact-or-full-path check for top_p < 1.
    Installed into the plugin's sampler class at model init (see host_topp.py)."""
    return _r6_flag("QWEN36_JOIN_TOPP_CANDIDATES")


def prefill_gdn_glue_enabled():
    """QWEN36_PREFILL_GDN_GLUE: GDN-layer prefill glue cuts, same values everywhere.

    * KDA conv input: the projection's TILE qkv goes straight into qkv_causal_conv1d_silu (the op reads TILE in place).
      Today it is untilized to a ROW_MAJOR copy that the op tilizes right back (UntilizeCodegen + Tilize, ~0.22 ms per
      layer per 8K chunk at the profiler clock). The carried conv state (the chunk's last K-1 input rows) is cut from
      the last tile row of qkv instead of the ROW_MAJOR copy.
    * beta: b = qkvzab columns [Nv, 2 Nv) starts mid-tile, so ttnn.slice untilizes, slices and re-tilizes it (~83 us
      per layer). It is taken with an exact one-hot column-select matmul of the one-tile-wide [a | b] block instead
      (HiFi4, fp32 accumulation: one nonzero term per output element, the bf16 value reproduced exactly; only a -0.0
      can become +0.0, and sigmoid(+-0) is the same 0.5).

    ttsim-perf, one chip = one TP8 shard (gdn/tp.py ops around the KDA conv): 8192-row chunk (DRAM qkvzab) 468 -> 136 us
    per GDN layer, 2048-row chunk (L1) 172 -> 67 us, with q / k / v / conv carry / a / b bit-identical."""
    return _r6_flag("QWEN36_PREFILL_GDN_GLUE")


_PREFILL_MTP_LEAN = None


def prefill_mtp_lean_enabled():
    """QWEN36_PREFILL_MTP_LEAN (default "0"; "0" = today's prompt warm-up, byte for byte). Two cuts to the drafter's
    PROMPT warm-up, the eager per-chunk forward that runs after every big-chunk replay and the warm inside the spec
    short traces (Qwen36MTP.forward_prefill):

    * the drafter layer stops right after its paged K/V fill (the prefill twin of QWEN36_MTP_KV_ONLY): no Q norm /
      RoPE, paged SDPA, gate, wo + reduce-scatter, residual add, ffn_norm or MLP + reduce-scatter. Every caller freed
      that block output unread; the K/V bytes come from the same ops as before.
    * a fast-prefill segment whose rows are an exact multiple of the chunk size no longer warms its last chunk twice
      (the replay loop already ran on_chunk for it; the second call wrote the same K/V and captured the same feed row),
      and a non-final vLLM chunk, whose logits are never read, skips the last-row LM head.

    Drafter K/V, the captured feed row and every returned logit are unchanged. TP only; read once (the short traces
    capture whichever body is active at warm-up)."""
    global _PREFILL_MTP_LEAN
    if _PREFILL_MTP_LEAN is None:
        _PREFILL_MTP_LEAN = os.environ.get("QWEN36_PREFILL_MTP_LEAN", "0") == "1"
    return _PREFILL_MTP_LEAN


_PREFILL_INTRACE_SNAP = None


def prefill_intrace_snap_enabled():
    """QWEN36_PREFILL_INTRACE_SNAP (default "0"; "0" = today's planner and traces, byte for byte). The spec fast prefill
    runs every prefill call as ONE segment and takes the GDN prefix-cache snapshots (learn / end-of-prompt / stride /
    forced points) INSIDE the replays instead of splitting the prompt at them into 2048-row chunks and short traces:

    * every prefill trace body (2048 / big chunk traces, spec short traces) also hands each GDN layer's
      chunk_gated_delta_rule a persistent snapshot index page and an [S, Nv, Dk, Dv] snapshot buffer (the op copies
      the carried state after the staged 32-row chunks, the same fp32 bytes a segment end would leave), and gathers
      the three conv-input rows before each staged position out of the chunk's row-major conv input (an embedding
      gather: the cross-chunk conv carry a segment ending there would keep). S = GDN_SNAP_SLOTS per replay;
    * the host stages the snapshot rows of each replay with its other inputs and, after the replay, copies each staged
      snapshot into its prefix-cache slot (GdnPrefixStateCache.save with a snapshot source); a point at a replay end
      is saved from the bound state as before.

    Prefill outputs (logits, feed rows, GDN state) and the base KV do not change (the same rows run through the same
    per-row ops; a snapshot is a copy of the scan state). The drafter K/V can change in its low bits for the rows a
    split plan warmed in 2048-row chunk / short-trace drafter passes and one segment warms in a big-chunk pass (r67
    stack, 2026-10-31: drafter K/V differed at 5 of 7 prompt lengths, acceptance moved at 1 of 7, 41021 tokens
    2.824 -> 2.706 committed/iter; the 64 greedy tokens were identical everywhere). A planner difference: the state at
    a former split point no longer passes through a masked short-trace tail (whose trailing masked GDN chunks hand on a
    state re-packed by one more identity scan step), so
    such a saved snapshot differs in its low bits; the downstream prefill outputs were bit-identical in every device
    test (see the module docstring of prefill_plan.py). Needs the KDA conv (QWEN_GDN_CONV=kda) and the fused chunk op;
    TP spec engine with B > 1 only. Read once (the traces capture whichever body is active at warm-up)."""
    global _PREFILL_INTRACE_SNAP
    if _PREFILL_INTRACE_SNAP is None:
        _PREFILL_INTRACE_SNAP = os.environ.get("QWEN36_PREFILL_INTRACE_SNAP", "0") == "1"
    return _PREFILL_INTRACE_SNAP


# Snapshot slots per replay under QWEN36_PREFILL_INTRACE_SNAP: a prefill call saves at most two learn points, one
# end-of-prompt point and one forced point, and a stride point always sits at a replay end.
GDN_SNAP_SLOTS = 4


def prefill_intrace_snap_min_rows():
    """QWEN36_PREFILL_INTRACE_SNAP_MIN_ROWS (default 0): traced prefill bodies of fewer rows than this get no
    in-replay snapshot ops (their GDN chunk op keeps the default dispatch: the phased program where it is predicted
    faster), and the planner never snapshots inside such a replay. 0 = every trace body. Read when the snapshot
    buffers are allocated."""
    return int(os.environ.get("QWEN36_PREFILL_INTRACE_SNAP_MIN_ROWS", "0") or 0)


_PREFILL_SNAP_TILE = None


def prefill_snap_tile_enabled():
    """QWEN36_PREFILL_SNAP_TILE (default "0"; r67 stack). With QWEN36_PREFILL_GDN_GLUE and QWEN36_PREFILL_INTRACE_SNAP
    both on, a traced prefill body that takes in-replay snapshots needs the snapshot conv rows (the three conv-input
    rows before each staged position). "0": they are gathered from the ROW_MAJOR conv input, so such a body keeps the
    untilize + row-major conv (the merge guard in TPGatedDeltaNet.forward_prefill: GDN_GLUE's conv cut is lost on
    every snapshot-carrying body, about 261 of 332 us per GDN layer at 8K rows). "1": the glue conv stays and the rows
    are gathered from the TILE input instead: in-replay snapshot rows sit on 128-row boundaries (prefill_plan.ALIGN),
    so rows r-3..r-1 are rows 29..31 of the last tile row of 128-row block r/128 - 1; every block's last tile row is
    sliced (a quarter of the rows, conv columns only), untilized and embedding-gathered (gdn/tp.py
    snap_rows_from_tile). The same bf16 values, so outputs and snapshots are unchanged. Needs both flags above; no-op
    otherwise. Read once (the traces capture whichever body is active at warm-up)."""
    global _PREFILL_SNAP_TILE
    if _PREFILL_SNAP_TILE is None:
        _PREFILL_SNAP_TILE = os.environ.get("QWEN36_PREFILL_SNAP_TILE", "0") == "1"
    return _PREFILL_SNAP_TILE


_PREFILL_MASKED_BIG = None


def prefill_masked_big_enabled():
    """QWEN36_PREFILL_MASKED_BIG (default "0"; "0" = today's split, byte for byte; needs QWEN36_PREFILL_BIG_CHUNK=8192
    and the spec fast prefill). After the full big-chunk replays of a fast-prefill segment, the remainder r rows run as
    ONE masked 4096- or 8192-row spec short trace (padding rows: identity GDN updates, K/V to the trash block, MTP
    padding to its scratch block) when the cost model (prefill_plan.plan_remainder, depth-aware) finds it cheaper than
    r // 2048 chunk replays + a short-trace tail. Port of the masked-big remainder of atupe 3247d9311b9. The two extra
    spec short traces are captured at warm-up with the default buckets (their persistent inputs and trace region are
    the cost). Numerics: bit-identical where it replaces 2048-row chunks + a short-trace tail; where the split plan's
    tail is 1-2 rows (the eager masked bucket, r % 2048 in {1, 2}) the prefill logits change (PCC 0.99978 measured at
    2049 tokens). Read once."""
    global _PREFILL_MASKED_BIG
    if _PREFILL_MASKED_BIG is None:
        _PREFILL_MASKED_BIG = os.environ.get("QWEN36_PREFILL_MASKED_BIG", "0") == "1"
    return _PREFILL_MASKED_BIG


def mtp_kv_only_enabled():
    """QWEN36_MTP_KV_ONLY (default "0"; "0" = today's drafter, byte for byte): an MTP KV-maintenance forward
    (Qwen36MTP.forward_decode with need_logits=False: the batched reseed inside the drafter trace, the eager reseeds and
    the join warms) stops right after the drafter layer's paged KV write. It skips the paged SDPA, the gate, wo, the
    attention all-reduce, the residual add, ffn_norm (all-gather) and the MLP with its all-reduce, all of which only
    produced an output the callers dropped. The K/V bytes written are unchanged (same ops up to the write), so the
    drafts and acceptance are unchanged. Read once when the MTP head is built (TP only)."""
    return os.environ.get("QWEN36_MTP_KV_ONLY", "0") == "1"


# -------------------------------------------------------------------------------------------------------------------- #
# r6 drafter bundle: MTP drafter micro-optimizations, each behind its own flag. Every default is off, and off is today's
# drafter byte for byte. They change only the DRAFTER's programs: the verify arbitrates every draft, so the output stays
# lossless and only acceptance can move. All are read when the MTP head is built (Qwen36MTP.__init__), TP only.
# Measured on a BH Galaxy TP8 row (r5 stack, K=3, 32K KV, production-loop harness): FC_PROGCFG + EMBED_ROW (+ the
# existing QWEN36_DECODE_AR_FUSE_RESIDUAL) 33.8 -> 33.05 ms per step at 8 users, + LAYER_BFP4 32.9; output md5 and
# acceptance unchanged on the 32K / 4K A/Bs. A sliding-window drafter SDPA was tried and dropped: W = 2K..16K of a 32K
# context cost 10-13% of the accepted drafts (W = 28K: 3%), far more than its 3% shorter step.
# -------------------------------------------------------------------------------------------------------------------- #
MTP_FC_PROGCFG_ENV = "QWEN36_MTP_FC_PROGCFG"
MTP_FC_IBW_ENV = "QWEN36_MTP_FC_IBW"
MTP_FC_CORES_ENV = "QWEN36_MTP_FC_CORES"
MTP_LAYER_BFP4_ENV = "QWEN36_MTP_LAYER_BFP4"
MTP_EMBED_ROW_ENV = "QWEN36_MTP_EMBED_ROW"


def _env_int(name, default=0):
    """Integer env value; unset or blank -> ``default``. A non-integer raises ValueError (no silent default)."""
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def mtp_fc_decode_progcfg(k, n, grid_w):
    """QWEN36_MTP_FC_PROGCFG (default "0"): the MTP fc (eh_proj, [rows, 2*dim] x [2*dim, dim/tp] BFP8, HiFi2) of a
    decode-width call (rows <= 32: the draft steps, the batched reseed, the join warms) runs on an explicit 1D mcast
    program config, its concat input in L1, instead of ttnn's auto config on a DRAM input. Auto picks in0_block_w=2 for
    this K (2*dim = 320 k-tiles): 160 in0 multicast rounds, 145-148 us per call (7 MB at ~48 GB/s), 4 calls per K=3
    iteration (sp5 drafter profile).
    QWEN36_MTP_FC_IBW (default 32) caps in0_block_w (largest divisor of the k-tiles <= the cap; 32 -> 10 rounds) and
    QWEN36_MTP_FC_CORES (default 20 = one 32-wide output column per core at dim/tp = 640) is the core budget, shaped
    wide-first on the device grid width ``grid_w``. The K-block partial sums reassociate, so a near-tie can draft a
    different token: only acceptance can move (measured: 143 -> 28 us per call at 8 rows, bit-identical to auto at <= 8
    rows, 1 bf16 ulp at 32 rows; acceptance unchanged). Returns the program config (built once by the caller), or None
    when off."""
    if os.environ.get(MTP_FC_PROGCFG_ENV, "0") != "1":
        return None
    ibw = _env_int(MTP_FC_IBW_ENV, 32)
    cores = _env_int(MTP_FC_CORES_ENV, 20)
    if ibw <= 0 or cores <= 0:
        raise ValueError(f"{MTP_FC_IBW_ENV}={ibw} / {MTP_FC_CORES_ENV}={cores}: both must be > 0")
    return create_matmul_1d_decode_progcfg(TILE_SIZE, k, n, num_cores=cores, grid_w=grid_w, max_in0_block_w=ibw)


def mtp_layer_bfp4_parts():
    """QWEN36_MTP_LAYER_BFP4 (default "0"): the MTP drafter layer's (mtp.layers.0) weights in bfloat4_b with LoFi
    decode matmuls, for the drafter only. The drafter layer has its own checkpoint weights and its own cache files (the
    dtype is in the file name), so the base model and the verify keep their dtypes and no extra DRAM is needed (the
    BFP4 copies replace the drafter layer's BFP8 ones). "1" / "all": attention (fused q|k|v|gate and wo) and MLP
    (gate/up/down); "attn" or "mlp" (or "attn,mlp") picks parts. Measured with "1": -0.15 ms per K=3 step; accepted
    drafts unchanged at 32K / 8 users, 4K / 4 users (K=7) and 4K / 8 users, -0.01 committed per verify on a 32K
    verbatim-retrieval prompt. Returns a frozenset of {"attn", "mlp"}, empty when off."""
    raw = os.environ.get(MTP_LAYER_BFP4_ENV, "0").strip().lower()
    if raw in ("", "0", "off", "none"):
        return frozenset()
    if raw in ("1", "all"):
        return frozenset({"attn", "mlp"})
    parts = frozenset(p.strip() for p in raw.split(",") if p.strip())
    bad = parts - {"attn", "mlp"}
    if bad or not parts:
        raise ValueError(f"{MTP_LAYER_BFP4_ENV}={raw!r}: expected 0, 1, all, attn, mlp or attn,mlp")
    return parts


def mtp_embed_row_enabled():
    """QWEN36_MTP_EMBED_ROW (default "0"): the MTP drafter's token embedding (Qwen36MTP.forward_decode) is looked up
    row-major, viewed as [1,1,rows,dim/tp] and tilized as ONE tile row, instead of the TILE lookup over [rows,1] ids
    that gives every id its own padded 32-row tile and then re-packs the rows with a reshape copy (sp5: a 27 us tilize
    + a 6 us reshape per draft step, 30 + 20 us for the 32-row reseed; 8.8 us at 1-16 rows with this flag). The same
    bf16 rows reach the same place, so the drafter's input, its drafts and the output are byte-identical."""
    return os.environ.get(MTP_EMBED_ROW_ENV, "0") == "1"


def spec_hostfix_enabled():
    """Host-side spec-step fixes (QWEN36_SPEC_HOSTFIX, default 1; 0 = the old code paths): conflict check without
    tolist, one read of the drafter ids, vectorized row loops, drafter page tables skipped when unchanged.
    Same device inputs and outputs either way."""
    return bool(int(os.environ.get("QWEN36_SPEC_HOSTFIX", "1")))


def spec_hostset_enabled():
    """QWEN36_SPEC_HOSTSET (default 1; 0 = today's exact code paths, for A/B): the host set of the MTP spec step
    (work/notes/39_hostset_impl.md). Same device-visible values, fewer host copies and less host Python:
    packed per-step int inputs with the RoPE cos/sin looked up on the device (verify and drafter traces), the
    next verify's position/ctrl inputs staged while the drafter runs, vectorized per-row host plans, and the
    narrow (no-draft) step reading one logits column per user. The trace-changing part (packed inputs, in-trace
    RoPE lookup) is read at trace capture; it also needs QWEN36_SPEC_HOSTSET_TRACE (default 1)."""
    return os.environ.get("QWEN36_SPEC_HOSTSET", "1") != "0"


def spec_fused_enabled():
    """QWEN36_SPEC_FUSED (default "0"): the MTP spec step may run the greedy accept on the device (spec_mode
    "fused_sample", work/notes/45 stage S1): verify, accept trace and drafter trace in one enqueue and one synchronize.
    Read at prepare_decode (allocates the accept buffers and trace) and by the vLLM class's SpecPlan."""
    return os.environ.get("QWEN36_SPEC_FUSED", "0") == "1"


def spec_fused_sampled_enabled():
    """QWEN36_SPEC_FUSED_SAMPLED (default "1"; needs QWEN36_SPEC_FUSED=1 and the verify top-k trace): stage S2, the engine also
    prepares and captures the SAMPLED accept trace, so a fused_sample step that carries ``spec_sampling`` runs on the device.
    Off: such a step takes the full-logits host walk (reason "no_sampled_trace")."""
    return os.environ.get("QWEN36_SPEC_FUSED_SAMPLED", "1") != "0"


def spec_row_delta_enabled():
    """QWEN36_SPEC_ROW_DELTA (default "1"; "0" = today's decline): the fused / sampled / chain device bodies add a PER-ROW M-RoPE
    delta (fused aux row 6, chain overlay last row), so a batch of rows with different deltas (an image row plus text rows)
    stays on the fused path. "0" keeps the "mixed_rope_delta" fallback for such a batch (the words are still laid out the
    same way). Read once when the engine is built."""
    return os.environ.get("QWEN36_SPEC_ROW_DELTA", "1") != "0"


def spec_chain_sampled_enabled():
    """QWEN36_SPEC_CHAIN_SAMPLED (default "0"): a deferred (defer_read=True) fused step may carry a non-greedy row in
    ``spec_sampling``. Off, decode_forward raises ValueError for it: sampled overflow is not handled for deferred steps yet, so
    the plugin runs a sampled step as the synchronous S2 step and defers only greedy-only steps."""
    return os.environ.get("QWEN36_SPEC_CHAIN_SAMPLED", "0") == "1"


def spec_prefill_overlap_enabled():
    """QWEN36_SPEC_PREFILL_OVERLAP (default on; "0" turns it off; needs QWEN36_SPEC_CHAIN=1 and QWEN36_SPEC_FUSED=1): the plugin may call prefill_forward
    while one deferred (defer_read=True) verify step is still pending, and finish that step (propose_draft_tokens with its handle) after
    the prefill. Read as the model capability supports_prefill_over_pending_spec."""
    return os.environ.get("QWEN36_SPEC_PREFILL_OVERLAP", "1") != "0"


def spec_fused_audit_enabled():
    """QWEN36_SPEC_FUSED_AUDIT (default "1" in stage S2): every sampled fused step reads back the top-k union (two extra
    reads) and checks the device commit against spec_fused_sampled.fused_sampled_reference."""
    return os.environ.get("QWEN36_SPEC_FUSED_AUDIT", "1") != "0"


def spec_fused_audit_every():
    """QWEN36_SPEC_FUSED_AUDIT_EVERY (default "1" = every sampled fused step; integer >= 1): the S2 audit runs only on
    sampled fused steps whose per-engine sampled-step counter % N == 0."""
    return max(1, int(os.environ.get("QWEN36_SPEC_FUSED_AUDIT_EVERY", "1")))


SPEC_BUCKETS_DEFAULT = "1,2,4,8"


def spec_buckets_flag():
    """The raw QWEN36_SPEC_BUCKETS text (default SPEC_BUCKETS_DEFAULT = on; "8" or "" = off), stripped."""
    return os.environ.get("QWEN36_SPEC_BUCKETS", SPEC_BUCKETS_DEFAULT).strip()


def spec_buckets(B=8):
    """QWEN36_SPEC_BUCKETS (default "1,2,4,8" = on; "8" = one 8-row bucket = off): the verify/drafter row buckets as a sorted tuple.
    Every value must be in {1, 2, 4, 8} and the engine width B must be the largest; else ValueError."""
    from models.demos.blackhole.qwen36.tt.spec_bucket import parse_buckets

    return parse_buckets(spec_buckets_flag(), B)


def spec_bucket_hold():
    """QWEN36_SPEC_BUCKET_HOLD (default 16; integer >= 0, 0 = no delay): a step whose live rows fit a smaller row bucket than the
    current one keeps the current bucket until this many consecutive steps fit the smaller one (or the step breaks the chain
    anyway), so churn does not pay a bucket_change chain break per down-shift. An up-shift is always immediate."""
    return max(0, int(os.environ.get("QWEN36_SPEC_BUCKET_HOLD", "16")))


def engine_spec_buckets(B=8):
    """The engine's view of QWEN36_SPEC_BUCKETS: as spec_buckets, but an engine narrower than 8 (B < 8) keeps only the buckets
    <= B plus B itself, instead of raising; the warning is logged only when QWEN36_SPEC_BUCKETS is explicitly set (unset: clip
    silently). B >= 8 is as spec_buckets (strict). Bad values still raise.
    """
    from models.demos.blackhole.qwen36.tt.spec_bucket import parse_buckets

    text = spec_buckets_flag()
    if B >= 8:
        return parse_buckets(text, B)
    if str(text).strip() in ("", "8"):  # the default: off at any engine width, nothing to clip
        return (B,)
    kept = parse_buckets(text, B, clip=True)  # validates every value; drops those > B and adds B
    dropped = sorted({int(x) for x in str(text).split(",") if x.strip() and int(x) > B})
    if dropped and "QWEN36_SPEC_BUCKETS" in os.environ:
        logger.warning(
            f"QWEN36_SPEC_BUCKETS={text!r}: engine width B={B} is below 8, dropping buckets {dropped}; using {kept}"
        )
    return kept


def spec_chain_enabled():
    """QWEN36_SPEC_CHAIN (default "0"; needs QWEN36_SPEC_FUSED=1): stage S3c, chain mode in sync serving. A fused step that follows
    a fused step on the same live rows lets the device build the next verify's token and aux buffers (the chain trace, tt/spec_chain.py)
    instead of the host staging them. Read at prepare_decode (allocates the chain device) and capture_decode (captures its traces).
    """
    return os.environ.get("QWEN36_SPEC_CHAIN", "0") == "1"


def spec_chain_audit_enabled():
    """QWEN36_SPEC_CHAIN_AUDIT (default "1"): on a chain step, the host builds the words today's path would stage and, after the
    step's synchronize, requires them bit-equal to what the device wrote into the verify token / aux buffers."""
    return os.environ.get("QWEN36_SPEC_CHAIN_AUDIT", "1") != "0"


def spec_chain_audit_every():
    """QWEN36_SPEC_CHAIN_AUDIT_EVERY (default 1; integer >= 1): the chain audit runs on chain steps whose counter % N == 0."""
    return max(1, int(os.environ.get("QWEN36_SPEC_CHAIN_AUDIT_EVERY", "1")))


def spec_chain_kernel_enabled():
    """QWEN36_SPEC_CHAIN_KERNEL (default "0" = the 126-op ttnn graph, byte for byte; needs QWEN36_SPEC_CHAIN=1): the chain trace
    body (spec_chain.ChainDevice.body) runs as ONE single-core generic_op kernel (tt/spec_fused_kernels.py,
    kernels/spec_chain_fused.cpp) on the same input / output buffers, writing the same words. Read when a ChainDevice is
    allocated (so the eager warm-up, the trace capture and every replay agree)."""
    return os.environ.get("QWEN36_SPEC_CHAIN_KERNEL", "0") == "1"


def spec_accept_kernel_enabled():
    """QWEN36_SPEC_ACCEPT_KERNEL (default "0" = the 115-op ttnn graph, byte for byte; needs QWEN36_SPEC_FUSED=1): the GREEDY accept
    trace body (spec_fused.FusedGreedyDevice.body) runs as ONE generic_op kernel (tt/spec_fused_kernels.py,
    kernels/spec_accept_fused.cpp) on the same buffers, writing the same words. The sampled accept (FusedSampledDevice) keeps
    its op graph. Read when a FusedGreedyDevice is allocated."""
    return os.environ.get("QWEN36_SPEC_ACCEPT_KERNEL", "0") == "1"


def spec_accept_kernel_cores():
    """QWEN36_SPEC_ACCEPT_KERNEL_CORES (default 32; integer >= 1): cores of the accept kernel (the reseed page-table rows are
    split into column chunks over them; capped by the grid and by the number of 64 B chunks of one row)."""
    return max(1, int(os.environ.get("QWEN36_SPEC_ACCEPT_KERNEL_CORES", "32")))


def spec_accept_kernel_sampled_enabled():
    """QWEN36_SPEC_ACCEPT_KERNEL_SAMPLED (default "0" = the sampled accept op graph, byte for byte; needs QWEN36_SPEC_FUSED=1
    and the stage-S2 sampled accept, QWEN36_SPEC_FUSED_SAMPLED + the verify top-k trace): the SAMPLED accept trace body
    (spec_fused_sampled.FusedSampledDevice.body) runs as ONE generic_op kernel on R cores (tt/spec_fused_kernels.py,
    kernels/spec_accept_sampled_fused.cpp; bit-exact host model tt/spec_sampled_kernel_math.py) on the same buffers. Its
    float32 math differs from fused_sampled_reference only by rounding (decisions flip within ~1e-7 of a threshold), so
    the S2 audit also counts a top-p tail edge on an examined row as a float32 boundary while this flag is on. Read when
    a FusedSampledDevice is allocated."""
    return os.environ.get("QWEN36_SPEC_ACCEPT_KERNEL_SAMPLED", "0") == "1"


def spec_hostset_trace_enabled():
    """The part of QWEN36_SPEC_HOSTSET that changes the captured traces: one packed uint32 input buffer per trace with
    in-trace slices, and cos/sin rows looked up from the resident RoPE tables by position (ttnn.embedding) instead of
    uploaded. QWEN36_SPEC_HOSTSET_TRACE=0 keeps the host-only parts of the set and today's trace inputs. Read when the
    verify / drafter traces are prepared, so one process sees one answer."""
    return spec_hostset_enabled() and os.environ.get("QWEN36_SPEC_HOSTSET_TRACE", "1") != "0"


HOSTSET_PACK_ALIGN = 32  # words; a packed row is padded to a multiple of this (128 B)


def pack_u32_rows(segments):
    """Pack 1-D non-negative integer segments into ONE host buffer for a single H2D copy: row i holds segment i, zero
    padded to W = max segment length rounded up to HOSTSET_PACK_ALIGN words. Returns torch.int32 [len(segments), W]
    (staged as ttnn.uint32). The trace slices row i, columns [0, len_i), with ttnn.slice (the same start-at-0 RM slice
    the batched decode loop uses), so every segment starts at a row page boundary. Raises on a negative value: the
    buffer is uint32 and a negative would become a huge index."""
    segs = [x.reshape(-1).tolist() if isinstance(x, torch.Tensor) else [int(v) for v in x] for x in segments]
    width = -(-max(len(x) for x in segs) // HOSTSET_PACK_ALIGN) * HOSTSET_PACK_ALIGN
    out = torch.tensor([x + [0] * (width - len(x)) for x in segs], dtype=torch.int64)
    if int(out.min()) < 0:
        raise ValueError("pack_u32_rows: negative value in a packed uint32 input")
    return out.to(torch.int32)


def rope_row_index(positions, rope_delta, table_rows):
    """RoPE table row per position for the in-trace lookup: position + rope_delta, clamped into [0, table_rows).
    ``rope_delta`` is a scalar (every position the same) or one delta per position (a list / tensor of the same length:
    the rows' own M-RoPE deltas). Equal to the host-computed cos/sin for every position inside the table. Outside it only
    rows the engine never reads are affected: positions >= table_rows lie past max_model_len (junk draft / padding window
    rows that no commit uses), and the engine never passes a negative position (inactive rows run at position 0)."""
    pos = torch.as_tensor(positions).to(torch.int64).reshape(-1)
    if isinstance(rope_delta, (list, tuple, torch.Tensor)):
        delta = torch.as_tensor(rope_delta).to(torch.int64).reshape(-1)
        if delta.numel() != pos.numel():
            raise ValueError(f"rope_row_index: {delta.numel()} deltas for {pos.numel()} positions")
    else:
        delta = int(rope_delta)
    idx = pos + delta
    return idx.clamp(0, int(table_rows) - 1).to(torch.int32)


def spec_fast_join_enabled():
    """QWEN36_SPEC_FAST_JOIN (default "1"; "0" opts out): a fresh row's first verify (the seed, or a join into a running
    batch) runs INSIDE the normal verify replay as an ordinary 0-draft row, after its ring slot 0 / conv window are
    seeded from the prefill state with a few exact data-movement ops per GDN layer (TPGatedDeltaNet.seed_spec_rows_select).
    "0" opts out and keeps the eager T = 1 seed forward (model.seed_spec_step) plus the per-row ring rebuild, byte for byte.
    """
    return os.environ.get("QWEN36_SPEC_FAST_JOIN", "1") == "1"


def spec_join_fast_enabled():
    """QWEN36_SPEC_JOIN_FAST (default "1"; "0" = the old join host path, byte for byte): a join (fresh rows seeded
    into a running spec batch, MTPSpecEngine._seed_fresh_rows) stages only the fresh rows' MTP page tables (the old
    path re-staged the [B, nb] table and every row's [1, nb] table) and, with two or more fresh rows, warms their
    MTP slot Tp-1 KV in ONE batched drafter decode call instead of one call per row (work/notes/55_mtp_join_fast.md).
    The KV written is the same; the staged tables of the other rows are refreshed by the next propose."""
    return os.environ.get("QWEN36_SPEC_JOIN_FAST", "1") != "0"


def spec_join_trace_enabled():
    """QWEN36_SPEC_JOIN_TRACE (default on; "0" turns it off): the GDN part of a join seed (MTPSpecEngine._seed_fresh_rows,
    QWEN36_SPEC_FAST_JOIN=1) replays one captured trace per conv-window parity (a fixed 48-layer op sequence,
    TPGatedDeltaNet._seed_masked_impl, selected by a fresh-row mask) instead of enqueueing seed_spec_rows_select eagerly
    for every layer. A join the trace cannot serve (a layer's taps route is not "hist", mixed parity, no trace) runs
    the eager loop as before. Read when the traces are captured and on every join; "0" keeps the eager seed byte for byte.
    """
    return os.environ.get("QWEN36_SPEC_JOIN_TRACE", "1") != "0"


def spec_fast_prefill_enabled():
    """QWEN36_SPEC_FAST_PREFILL (default "0"; "1" opts in): the MTP engine's prompt prefill replays captured traces
    like the non-spec class instead of the eager masked-bucket forward: one short-prompt bucket trace per request
    (base forward + the MTP drafter's KV warm-up + the feed row of slot Tp-1 in the SAME trace), the 2048-token chunk
    trace for long prompts (also at B > 1, on the B=1 GDN scratch), and the per-slot GDN slot-write trace. "0" keeps
    the eager prefill_for_spec path byte for byte."""
    return os.environ.get("QWEN36_SPEC_FAST_PREFILL", "0") == "1"


def kv_cache_bf8_enabled():
    """bf8 paged KV cache (the qwen38_27b_qb2 KV dtype): QWEN36_KV_BF8 (default 1; 0 = bf16 cache) keeps the
    prefill Q in bf16, QWEN_SDPA_BF8=1 additionally casts the prefill Q to bf8. Halves KV memory and the SDPA reads
    that grow with context: TP=8 B=1 decode 22.75 -> 21.55 ms/token at 64k and 24.19 -> 22.57 at 103k, unchanged
    at 128 tokens; accuracy_512 96.88 -> 97.27, accuracy_1536 95.96 -> 95.70 (QWEN_SDPA_BF8: 96.29 / 95.64)."""
    return os.environ.get("QWEN_SDPA_BF8", "0") == "1" or os.environ.get("QWEN36_KV_BF8", "1") == "1"


def prefill_l1_rows_ok(rows):
    """L1 placement of prefill activations is validated up to 2048 rows (the serving chunk); a larger prefill chunk
    (QWEN36_PREFILL_L1_MAX_ROWS, default 2048) keeps those tensors in DRAM so their L1 banks do not clash with the
    matmul circular buffers that grow with rows."""
    return int(rows) <= int(os.environ.get("QWEN36_PREFILL_L1_MAX_ROWS", "2048"))


def l1_or_dram(rows):
    """L1 for prefill tensors of at most QWEN36_PREFILL_L1_MAX_ROWS rows, else DRAM."""
    return ttnn.L1_MEMORY_CONFIG if prefill_l1_rows_ok(rows) else ttnn.DRAM_MEMORY_CONFIG


def sdpa_prefill_exp_approx():
    """QWEN36_SDPA_PREFILL_EXP_APPROX (default 1): prefill SDPA uses the fast approximate exponential. Since tt-metal
    #57180 the fp32-dest SDPA kernel honours exp_approx_mode=False with the accurate SFPU exponential (~20x the SFPU cost
    per QK tile); single-chip chunked SDPA at depth 128k, 2048 rows: 63.7 -> 29.5 ms, PCC vs fp32 0.99900 -> 0.99883.
    0 restores the accurate exponential."""
    return os.environ.get("QWEN36_SDPA_PREFILL_EXP_APPROX", "1") == "1"


def _probe_sdpa_prefix_reuse():
    """Whether this ttnn build's SDPAProgramConfig takes prefix_reuse (QWEN36_SDPA_PREFIX_REUSE); probed without a
    device, like _probe_spec_sdpa_pack in attention/tp.py."""
    try:
        return hasattr(ttnn.SDPAProgramConfig, "prefix_reuse")
    except Exception:  # noqa: BLE001 - a stub ttnn (host tests) has no SDPAProgramConfig
        return False


SDPA_PREFIX_REUSE_SUPPORTED = _probe_sdpa_prefix_reuse()
_SDPA_PREFIX_REUSE = None
_PREFIX_REUSE_AUTO = -1  # QWEN36_SDPA_PREFIX_REUSE=auto


def sdpa_prefix_reuse_bits():
    """QWEN36_SDPA_PREFIX_REUSE (default "0" = today's chunked prefill SDPA program, byte for byte): the
    SDPAProgramConfig.prefix_reuse of the chunked (paged, causal) prefill SDPA. Bits, each only moving K/V reads or Q
    chunks between cores so every Q row still meets its K chunks in the same order (bit-identical outputs):
      1  K/V read batches sized by the cores that read K/V from DRAM instead of by the grid (more reads in flight);
      2  K/V prefix chains: the cores whose Q chunks read the same (batch, KV head) K/V, across Q chunks and across the
         KV head's 3 query heads, take each K/V chunk from the previous core of their chain (store and forward), so
         only one core per chain reads the prefix from DRAM;
      4  one Q chunk per core when there are no more Q chunks than cores (no causal light/heavy pairs: 2048-row chunks
         and the short traces at depth otherwise run two Q chunks' prefixes back to back on half the cores).
    Bits 8-15 cap the chain length in cores (0 = 8). "auto" (recommended) picks per call, see sdpa_prefill_reuse_bits.
    Read once (traces capture the program built at warm-up); 0 when the ttnn build has no prefix_reuse."""
    global _SDPA_PREFIX_REUSE
    if _SDPA_PREFIX_REUSE is None:
        raw = os.environ.get("QWEN36_SDPA_PREFIX_REUSE", "0").strip() or "0"
        if raw.lower() == "auto":
            bits = _PREFIX_REUSE_AUTO
        else:
            bits = int(raw, 0)
            if bits < 0 or bits > 0xFFFF:
                raise ValueError(f"QWEN36_SDPA_PREFIX_REUSE={raw!r}: expected auto or 0..0xFFFF")
        _SDPA_PREFIX_REUSE = bits if SDPA_PREFIX_REUSE_SUPPORTED else 0
    return _SDPA_PREFIX_REUSE


def sdpa_prefill_reuse_bits(rows, q_chunk, n_heads, n_cores, k_chunk=128):
    """prefix_reuse of one chunked prefill SDPA call of `rows` Q rows in Q chunks of q_chunk rows (K chunks of k_chunk
    rows). An explicit QWEN36_SDPA_PREFIX_REUSE value applies to every call. "auto" uses what measured best on one TP8
    row (op level, prefix 0-96K):
      * more Q chunks than cores (the 8192-row chunks, causal light/heavy pairs on 96 cores): the prefix chains (2:
        0.98-1.27x on the SDPA alone, 2-7% off the model's 16K steps, the most at depth; read batches were slower);
      * Q chunks that fit the grid (the 2048-row chunks, the short traces): one Q chunk per core (4: 1.8-2.0x at depth);
      * split calls on few cores (at most sdpa_prefill_read_batch_units Q chunks: the 128- and 256-row short traces in
        32-row Q chunks, 12 / 24 cores): also the read batches (5: 1.84-1.86x over 4 alone at 24K-96K)."""
    bits = sdpa_prefix_reuse_bits()
    if bits != _PREFIX_REUSE_AUTO:
        return bits
    units = n_heads * -(-rows // q_chunk)
    if units > n_cores:
        return 2
    if q_chunk < k_chunk and units <= sdpa_prefill_read_batch_units(n_cores):
        return 5
    return 4


_SDPA_PREFILL_READ_BATCH_UNITS = None


def sdpa_prefill_read_batch_units(n_cores):
    """QWEN36_SDPA_PREFILL_READ_BATCH_UNITS (default 1/5 of the grid, 24 on the 12x10 Galaxy grid; 0 = never): with
    QWEN36_SDPA_PREFIX_REUSE=auto, a split call (Q chunk smaller than the K chunk) of at most this many Q chunks also
    sizes its K/V read batches by the cores that read DRAM (prefix_reuse bit 1). Measured on one TP8 row (op level):
    128 / 256 rows in 32-row Q chunks at 24K-96K, 1.84-1.86x on the SDPA over one Q chunk per core alone; at 48 Q chunks
    it is a wash (+-4%). Read once."""
    global _SDPA_PREFILL_READ_BATCH_UNITS
    if _SDPA_PREFILL_READ_BATCH_UNITS is None:
        raw = os.environ.get("QWEN36_SDPA_PREFILL_READ_BATCH_UNITS", "").strip()
        v = int(raw) if raw else -1
        if raw and v < 0:
            raise ValueError(f"QWEN36_SDPA_PREFILL_READ_BATCH_UNITS={raw!r}: expected a count >= 0")
        _SDPA_PREFILL_READ_BATCH_UNITS = v
    return n_cores // 5 if _SDPA_PREFILL_READ_BATCH_UNITS < 0 else _SDPA_PREFILL_READ_BATCH_UNITS


_SDPA_PREFILL_QSPLIT = None
_SDPA_PREFILL_QSPLIT_UNITS = None


def sdpa_prefill_qsplit_min():
    """QWEN36_SDPA_PREFILL_QSPLIT (default "0" = Q chunk equals the K chunk, today's call): the smallest Q chunk (32 or
    64 rows) the chunked prefill SDPA may use. See sdpa_prefill_q_chunk. Read once."""
    global _SDPA_PREFILL_QSPLIT
    if _SDPA_PREFILL_QSPLIT is None:
        raw = os.environ.get("QWEN36_SDPA_PREFILL_QSPLIT", "0").strip() or "0"
        v = int(raw)
        if v not in (0, 32, 64):
            raise ValueError(f"QWEN36_SDPA_PREFILL_QSPLIT={raw!r}: expected 0, 32 or 64")
        _SDPA_PREFILL_QSPLIT = v
    return _SDPA_PREFILL_QSPLIT


def sdpa_prefill_qsplit_units(n_cores):
    """QWEN36_SDPA_PREFILL_QSPLIT_UNITS (default 2/5 of the grid, 48 on the 12x10 Galaxy grid): the most Q chunks a split
    may make. Measured on one TP8 row (op level, prefix 24K-96K): the best split of the 128-2048-row calls lands at
    about 48 Q chunks; more, smaller ones run slower (a 32- or 64-row Q chunk costs nearly a 128-row one per K chunk,
    and every Q chunk reads the whole K/V prefix from DRAM: 48 such readers already draw about 350 GB/s per chip).
    Read once."""
    global _SDPA_PREFILL_QSPLIT_UNITS
    if _SDPA_PREFILL_QSPLIT_UNITS is None:
        raw = os.environ.get("QWEN36_SDPA_PREFILL_QSPLIT_UNITS", "").strip()
        v = int(raw) if raw else 0
        if v < 0:
            raise ValueError(f"QWEN36_SDPA_PREFILL_QSPLIT_UNITS={raw!r}: expected a positive count")
        _SDPA_PREFILL_QSPLIT_UNITS = v
    return _SDPA_PREFILL_QSPLIT_UNITS or max(1, (2 * n_cores) // 5)


def sdpa_prefill_q_chunk(k_chunk, rows, n_heads, n_cores, q_min=None, chunk_start=0, max_units=None):
    """Q chunk rows of a chunked prefill SDPA whose K chunk is k_chunk rows (QWEN36_SDPA_PREFILL_QSPLIT).

    A causal Q chunk of k_chunk rows runs on one core and walks the whole K/V prefix. Short calls at depth (the
    128-1024-row short traces) have far fewer such chunks than cores, so most of the grid idles while a few cores walk
    the prefix. A Q chunk of k_chunk / 2 or k_chunk / 4 rows keeps the K chunks (each row still meets K chunks
    [0, its diagonal chunk] in order, with the same per-row masks: bit-identical) and multiplies the cores working on
    the call. Picks the smallest Q chunk >= q_min whose count n_heads * ceil(rows / q) stays within max_units
    (sdpa_prefill_qsplit_units); k_chunk when the flag is off, k_chunk is not 128, or no smaller chunk fits. A split
    needs chunk_start to be a multiple of the Q chunk (it always is: chunk starts are multiples of 128)."""
    q_min = sdpa_prefill_qsplit_min() if q_min is None else q_min
    if not q_min or k_chunk != 128 or rows < k_chunk:
        return k_chunk
    max_units = sdpa_prefill_qsplit_units(n_cores) if max_units is None else max_units
    best = k_chunk
    q = k_chunk // 2
    while q >= q_min:
        if chunk_start % q:
            break
        if n_heads * -(-rows // q) > max_units:
            break
        best = q
        q //= 2
    return best


LMHEAD_CFG_ENV = "QWEN36_LMHEAD_CFG"
_LMHEAD_LOGGED = False


def lm_head_compute_config():
    """Compute kernel config for LM-head matmuls, chosen by QWEN36_LMHEAD_CFG.
    "default" (or unset): None (ttnn default: HiFi2, bf16 dest accumulation for bf16 output).
    "hifi2_fp32" / "hifi4_fp32": fp32 dest accumulation, packer_l1_acc."""
    global _LMHEAD_LOGGED
    mode = os.environ.get(LMHEAD_CFG_ENV, "default").strip().lower()
    if mode == "default":
        cfg = None
    elif mode in ("hifi2_fp32", "hifi4_fp32"):
        cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2 if mode == "hifi2_fp32" else ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
    else:
        raise ValueError(f"{LMHEAD_CFG_ENV}={mode!r} invalid (default|hifi2_fp32|hifi4_fp32)")
    if not _LMHEAD_LOGGED:
        _LMHEAD_LOGGED = True
        logger.info(f"LM head compute config ({LMHEAD_CFG_ENV}) = {mode}")
    return cfg


def lm_head_kwargs():
    """ttnn.linear kwargs for LM head: empty for "default" so behavior is bit-identical."""
    cfg = lm_head_compute_config()
    return {} if cfg is None else {"compute_kernel_config": cfg}


DRAFT_LMHEAD_BFP4_ENV = "QWEN36_DRAFT_LMHEAD_BFP4"
_DRAFT_LMHEAD_BFP4_LOGGED = False


def draft_lmhead_bfp4_enabled():
    """QWEN36_DRAFT_LMHEAD_BFP4 (default 0, opt-in pending an acceptance A/B): the MTP DRAFTER gets its own
    bfloat4_b copy of the LM head; the base/verify head stays bfloat8_b, so output is unchanged (verify
    arbitrates every draft) and only acceptance can move. Halves the drafter head's DRAM read per draft
    step for about +89 MB DRAM per device. Logs the choice once per process."""
    global _DRAFT_LMHEAD_BFP4_LOGGED
    on = os.environ.get(DRAFT_LMHEAD_BFP4_ENV, "0") == "1"
    if not _DRAFT_LMHEAD_BFP4_LOGGED:
        _DRAFT_LMHEAD_BFP4_LOGGED = True
        logger.info(
            f"MTP drafter LM head: {'BFP4 copy (verify head stays BFP8)' if on else 'shared BFP8 head'} "
            f"({DRAFT_LMHEAD_BFP4_ENV}={int(on)})"
        )
    return on


# DRAM-sharded LM head for the decode-shaped (<= 32 rows) head calls, after the qwen38_27b_qb2 head
# (nstamatovic/qwen38-27b-qb2-opts, Qwen38Model._dram_logits). The plain head is an auto-configured
# 1D mcast matmul on an interleaved weight; this streams a second, DRAM-width-sharded copy of the same
# vocab-sharded weight from bank-adjacent readers. Each device's 970 vocab tiles (Qwen3.8 TP8) sit in
# 8 banks x 122 = 976 tiles; the 6 pad tiles exist only in the last bank's shard, past the tensor's
# logical width, so they never reach the logits (no -inf fill, no slice, no argmax change).
LMHEAD_DSHARD_ENV = "QWEN36_LMHEAD_DSHARD"
LMHEAD_DSHARD_READERS_ENV = "QWEN36_LMHEAD_DSHARD_READERS"
# K6 knobs of the DRAM-sharded head. Each defaults to today's behaviour and only acts with QWEN36_LMHEAD_DSHARD on;
# OUT_PCN / IBW / FP32_DEST apply to the "mc" pipeline only.
LMHEAD_DSHARD_SCOPE_ENV = "QWEN36_LMHEAD_DSHARD_SCOPE"
LMHEAD_DSHARD_OUT_PCN_ENV = "QWEN36_LMHEAD_DSHARD_OUT_PCN"
LMHEAD_DSHARD_IBW_ENV = "QWEN36_LMHEAD_DSHARD_IBW"
LMHEAD_DSHARD_FP32_DEST_ENV = "QWEN36_LMHEAD_DSHARD_FP32_DEST"
LMHEAD_DSHARD_SCOPES = ("all", "draft", "base")


def lm_head_dshard_mode():
    """QWEN36_LMHEAD_DSHARD (default "0" = off): "1" / "exact" runs the single-reader DRAM-sharded program with the
    plain head's K blocking (in0_block_w 2), Dest mode and packer L1 accumulation, so every logit is meant to be the
    plain head's bits; "mc" runs the qb2 multi-core pipeline, which accumulates 8 K tiles per Dest pass and is
    therefore NOT bit-identical (A/B only). Returns None, "exact" or "mc"."""
    mode = os.environ.get(LMHEAD_DSHARD_ENV, "0").strip().lower()
    if mode in ("", "0", "off"):
        return None
    if mode in ("1", "exact"):
        return "exact"
    if mode == "mc":
        return "mc"
    raise ValueError(f"{LMHEAD_DSHARD_ENV}={mode!r} invalid (0|1|exact|mc)")


def lm_head_dshard_readers(mode=None):
    """QWEN36_LMHEAD_DSHARD_READERS: DRAM readers per bank. Unset = 2 for "exact", the qb2 rule for "mc".

    mode "exact" with QWEN36_WEIGHT_PREFETCH on and the variable unset: 3 readers. Each prefetch GCB takes about
    70 KB of L1 on every worker core (about 350 KB at "all"), and the fp32 drafter head needs about 1180 KB of the
    1400 KB storage-core budget at 2 readers but about 977 KB at 3. An explicit value always wins."""
    raw = os.environ.get(LMHEAD_DSHARD_READERS_ENV, "").strip()
    if raw:
        return int(raw)
    if mode == "exact" and weight_prefetch.enabled_targets():
        logger.info(
            f"{LMHEAD_DSHARD_ENV}: {weight_prefetch.ENV} is on, so the DRAM-sharded LM head uses 3 readers/bank "
            f"(its GCBs share L1; set {LMHEAD_DSHARD_READERS_ENV} to override)"
        )
        return 3
    return None


def lm_head_dshard_scope():
    """QWEN36_LMHEAD_DSHARD_SCOPE (default "all" = today): which LM-head calls take the DRAM-sharded copy.
      all   - every decode-shaped call (base/verify/decode/prefill last row and the MTP drafter);
      draft - only the MTP drafter's calls (they pass the head weight explicitly). The base/verify calls keep the plain
              interleaved head, so the verify logits (and the greedy output) stay bit-identical; only the drafter's
              proposals, and with them acceptance, can move;
      base  - only the base calls (no explicit weight); the drafter keeps the plain head."""
    raw = os.environ.get(LMHEAD_DSHARD_SCOPE_ENV, "all").strip().lower() or "all"
    if raw not in LMHEAD_DSHARD_SCOPES:
        raise ValueError(f"{LMHEAD_DSHARD_SCOPE_ENV}={raw!r} invalid ({'|'.join(LMHEAD_DSHARD_SCOPES)})")
    return raw


def lm_head_dshard_scope_takes(scope, draft):
    """Whether a head call takes the DRAM-sharded copy under QWEN36_LMHEAD_DSHARD_SCOPE ``scope``. draft: the call is
    the MTP drafter's (Qwen36Model._lm_head_linear sees an explicit weight), else a base call (weight None)."""
    return scope == "all" or (scope == "draft") == bool(draft)


def _env_positive_int(name):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    v = int(raw)
    if v < 1:
        raise ValueError(f"{name}={raw!r}: expected a positive integer")
    return v


def lm_head_dshard_out_pcn():
    """QWEN36_LMHEAD_DSHARD_OUT_PCN (unset = the bank shard width, today): logits tiles per core of the "mc" head's L1
    width-sharded output (the op lays it on ceil(N / OUT_PCN) cores, row-major). Every L1 buffer takes its address
    range on every core (one free list for all L1 banks), so the default 122-tile shard holds 244 KB (bf16) / 488 KB
    (fp32 drafter head) of the L1 the pipeline's circular buffers need; 10 spreads the 970 TP8 vocab tiles over 97
    cores (20 / 40 KB) and makes the copy to DRAM 97-core wide. Output values do not depend on it."""
    return _env_positive_int(LMHEAD_DSHARD_OUT_PCN_ENV)


def lm_head_dshard_ibw():
    """QWEN36_LMHEAD_DSHARD_IBW (unset = 2, the plain head's K blocking): the "mc" pipeline's weight block depth in K
    tiles (in0_block_w). Its Dest pass stays ceil(8 / ibw) * ibw K tiles (8 for ibw <= 8), but the per-group K rotation
    is in whole blocks, so the bits depend on it."""
    return _env_positive_int(LMHEAD_DSHARD_IBW_ENV)


def lm_head_dshard_fp32_dest():
    """QWEN36_LMHEAD_DSHARD_FP32_DEST (default 0): the "mc" head's bf16-output (base/verify) calls keep each column
    pass's whole K in fp32 Dest instead of 8 K tiles per 16-bit Dest pass + a Float16_b L1 accumulator: one rounding
    to bf16 per logit instead of ~20 L1 accumulations and per-MVMUL 16-bit Dest rounding. Not bit-identical to the
    plain head (neither is "mc" itself); closer to fp32. Calls that pass their own compute config (the drafter's fp32
    head, QWEN36_LMHEAD_CFG) keep it."""
    raw = os.environ.get(LMHEAD_DSHARD_FP32_DEST_ENV, "0").strip().lower()
    if raw in ("", "0", "off", "false", "no"):
        return False
    if raw in ("1", "on", "true", "yes"):
        return True
    raise ValueError(f"{LMHEAD_DSHARD_FP32_DEST_ENV}={raw!r} invalid (0|1)")


def dram_multicore_workers(bank_tiles):
    """qb2's workers per DRAM bank for the multi-core pipeline: about one per 8 tiles of a bank, 2 to 4."""
    return min(4, max(2, (bank_tiles + 7) // 8))


# L1 the single-reader ("exact") program may take on one storage core (Blackhole: 1.5 MB per Tensix). The rest is
# left for l1_small, the kernel config buffer and whatever the decode trace already holds in L1.
LMHEAD_DSHARD_L1_BUDGET = 1400 * 1024
_TILE_BYTES = {"bfloat16": 2048, "float32": 4096, "bfloat8_b": 1088, "bfloat4_b": 576}
# The "mc" pipeline (create_program_dram_sharded_multicore_spec) puts its circular buffers on every core of its work
# rectangle, and every L1 buffer (sharded or not) takes its address range on every core: one free list serves all L1
# banks. So the pipeline's CBs, the head's activation shard, its logits shard and the L1 the decode trace holds (the
# DecodeResidualAllReduce staging buffer: [32, dim * tp] bf16 over 40 cores = 64 KB at Qwen3.8 TP8, persistent) all
# share one Blackhole L1 bank: 1536 KB - 112000 B unreserved base (MEM_MAP_END 41344 B + the 69 KB kernel config
# buffer, 64 B aligned) - the production l1_small_size (24576 B) = 1436288 B.
LMHEAD_DSHARD_MC_L1_BANK = 1536 * 1024 - 112000 - 24576
LMHEAD_DSHARD_MC_L1_RESERVED = 64 * 1024  # the decode all-reduce staging buffer at TP8 (allocated before any trace)
_MC_DST_TILES = 8  # the pipeline's tiles per column pass (dst_tiles)
_MC_L1ACC_K_TILES = 8  # K tiles per 16-bit Dest pass without fp32_dest_acc_en (l1acc_k_tiles)
_DRAM_ALIGN = 64  # Blackhole DRAM alignment (the pipeline aligns its weight entries to it)


class LMHeadDShardPlan:
    """Shapes of the DRAM-sharded head (pure arithmetic, no device). k / n: the per-device weight [k, n]."""

    def __init__(self, k, n, banks, mode="exact", readers=None, out_pcn=None, ibw=None):
        if mode not in ("exact", "mc"):
            raise ValueError(f"LM-head DRAM-shard mode {mode!r} invalid (exact|mc)")
        if k % TILE_SIZE:
            raise ValueError(f"LM-head K {k} is not a whole number of tiles")
        if mode != "mc" and (out_pcn is not None or ibw is not None):
            raise ValueError(
                f"{LMHEAD_DSHARD_OUT_PCN_ENV} / {LMHEAD_DSHARD_IBW_ENV} apply to {LMHEAD_DSHARD_ENV}=mc only "
                "(exact keeps the plain head's blocking and output layout)"
            )
        self.mode = mode
        self.multicore = mode == "mc"
        self.banks = banks
        self.k_tiles = k // TILE_SIZE
        self.n_tiles = math.ceil(n / TILE_SIZE)
        # The plain head's auto 1D-mcast config streams K in 2-tile blocks when K is even (get_mcast_1d_config);
        # the same blocking keeps the per-tile accumulation order, which "exact" relies on.
        self.in0_block_w = 2 if self.k_tiles % 2 == 0 else 1
        self.k6_ibw = ibw is not None
        if self.k6_ibw:  # QWEN36_LMHEAD_DSHARD_IBW (mc): the pipeline takes a ragged last K block, any width works
            self.in0_block_w = ibw
        if self.multicore:
            self.bank_tiles = math.ceil(self.n_tiles / banks)
            self.readers = dram_multicore_workers(self.bank_tiles) if readers is None else readers
            if self.readers < 2:
                raise ValueError("the multi-core pipeline needs at least 2 readers per bank")
        else:
            # 2 readers per bank by default: one reader per bank needs ~1.85 MB of L1 per storage core for the
            # drafter's fp32 head (in1 triple buffer 732 bfp8 tiles + 124 fp32 out tiles + the 122-tile fp32 shard),
            # more than a Blackhole core has. The bank shard stays 122 tiles (2 x 61) and the per-tile K order
            # does not change, so the result is the same bits.
            self.readers = 2 if readers is None else readers
            if not 1 <= self.readers <= 3:
                raise ValueError(f"single-reader DRAM-sharded head takes 1-3 readers per bank, got {self.readers}")
            # Several readers per bank need the bank's shard to be readers x ceil(N / (banks x readers)) tiles.
            self.bank_tiles = self.readers * math.ceil(self.n_tiles / (banks * self.readers))
        self.padded_tiles = self.bank_tiles * banks
        if self.padded_tiles - self.n_tiles >= self.bank_tiles:
            raise ValueError(f"{self.n_tiles} vocab tiles leave a padding-only bank of {self.bank_tiles}")
        # Activation: width-sharded over one core per bank (the qb2 layout: output cores == activation cores),
        # each shard a whole number of K blocks. The mc pipeline reads activation rows straight from their shards
        # (any block width), so with an explicit QWEN36_LMHEAD_DSHARD_IBW only the shards themselves must be even.
        self.act_cores = banks
        if self.k6_ibw:
            if self.k_tiles % self.act_cores:
                raise ValueError(f"K {self.k_tiles} tiles do not split into {banks} activation shards")
        elif self.k_tiles % (self.act_cores * self.in0_block_w):
            raise ValueError(
                f"K {self.k_tiles} tiles do not split into {banks} shards of {self.in0_block_w}-tile blocks"
            )
        self.act_shard_w = self.k_tiles // self.act_cores * TILE_SIZE
        self.per_core_N = self.bank_tiles
        if out_pcn is not None:  # QWEN36_LMHEAD_DSHARD_OUT_PCN (mc): the logits shard width, not the bank shard
            self.per_core_N = out_pcn
        # Cores of the L1 width-sharded logits (matmul_device_operation: ceil(N / per_core_N), row-major on the grid).
        self.out_cores = math.ceil(self.n_tiles / self.per_core_N)
        if self.multicore:
            self.mc_layout()  # raises on a worker count the pipeline factory rejects

    def mc_layout(self):
        """The multi-core pipeline's layout for this head (create_program_dram_sharded_multicore_spec), as a dict:
        cn column groups and ck K-row groups per bank, p column passes of ncp tiles per core, kb K tiles per streamed
        weight block, max_own output columns per core, work items. Raises on the layouts the factory TT_FATALs."""
        if not self.multicore:
            raise ValueError("mc_layout models the multi-core pipeline only")
        nbt, kt, w = self.bank_tiles, self.k_tiles, self.readers
        cn = min(w, nbt)
        ck = w // nbt if w > nbt else 1
        ncmax = math.ceil(nbt / cn)
        p = math.ceil(ncmax / _MC_DST_TILES)
        ncp = math.ceil(ncmax / p)
        if not (ck == 1 or p == 1):
            raise ValueError(f"{w} readers/bank: K row groups need a single column pass ({nbt} tiles/bank)")
        if nbt // cn < p:
            raise ValueError(f"{w} readers/bank leave a core fewer columns than its {p} passes ({nbt} tiles/bank)")
        kb = min(self.in0_block_w, math.ceil(kt / ck))
        max_own = 0
        for c in range(cn):
            width = nbt * (c + 1) // cn - nbt * c // cn
            for kidx in range(ck):
                max_own = max(max_own, width * (kidx + 1) // ck - width * kidx // ck)
        work = math.ceil(self.n_tiles / nbt) * cn * ck
        if work >= 256:
            raise ValueError(f"{work} work items exceed the pipeline's one-byte role table")
        return dict(cn=cn, ck=ck, p=p, ncp=ncp, kb=kb, max_own=max_own, max_blocks=math.ceil(math.ceil(kt / ck) / kb),
                    work=work)

    def mc_l1_bytes(self, out_dtype="bfloat16", act_dtype="bfloat16", weight_dtype="bfloat8_b", fp32_dest=False,
                    two_streams=True):
        """L1 of one mc call in bytes, from the factory's dataflow buffers: ``cb`` (per core of the work rectangle:
        the whole-K activation copy x, two weight streams of depth-3 blocks, the output staging, the Float16_b
        accumulator without fp32 Dest, the K-group partials when ck > 1), the L1 buffers ``act_shard`` and
        ``out_shard`` (each takes its size on every core), and ``total``. Dtypes are ttnn dtype names."""
        lay = self.mc_layout()
        out_b, act_b = _TILE_BYTES[out_dtype], _TILE_BYTES[act_dtype]
        w_b = -(-_TILE_BYTES[weight_dtype] // _DRAM_ALIGN) * _DRAM_ALIGN
        part_b = _TILE_BYTES["float32"] if fp32_dest else _TILE_BYTES["bfloat16"]
        streams, depth = (2, 3) if two_streams else (1, 5)
        x = lay["max_blocks"] * lay["kb"] * act_b
        w = streams * depth * lay["kb"] * lay["ncp"] * w_b
        out = lay["max_own"] * out_b
        acc = lay["max_own"] * part_b if (not fp32_dest and lay["ck"] == 1) else 0
        part = (lay["ncp"] + lay["ck"] * lay["max_own"]) * part_b if lay["ck"] > 1 else 0
        cb = x + w + out + acc + part
        act_shard = (self.act_shard_w // TILE_SIZE) * act_b
        out_shard = self.per_core_N * out_b
        return dict(cb=cb, act_shard=act_shard, out_shard=out_shard, total=cb + act_shard + out_shard)

    def mc_k_tiles_per_dest_pass(self, fp32_dest=False):
        """K tiles one mc column pass accumulates in Dest before anything leaves it: the whole K range of the core with
        fp32 Dest (one rounding at the pack), else ceil(8 / kb) streamed blocks in 16-bit Dest (8 K tiles for kb <= 8)
        before each packer add into the Float16_b L1 accumulator. The plain head's 1D program does the same with 2."""
        lay = self.mc_layout()
        if fp32_dest:
            return math.ceil(self.k_tiles / lay["ck"])
        return math.ceil(_MC_L1ACC_K_TILES / lay["kb"]) * lay["kb"]

    def mc_l1_need(self, weight_dtype="bfloat8_b", fp32_dest=False):
        """Worst of the two mc calls the model makes: the drafter's fp32 head (fp32 Dest) and the bf16 base head
        (fp32 Dest only with QWEN36_LMHEAD_DSHARD_FP32_DEST)."""
        draft = self.mc_l1_bytes("float32", "bfloat16", weight_dtype, fp32_dest=True)
        base = self.mc_l1_bytes("bfloat16", "bfloat16", weight_dtype, fp32_dest=fp32_dest)
        return max(draft, base, key=lambda d: d["total"])

    def l1_bytes_per_core(self, out_dtype="float32", act_dtype="bfloat16", weight_dtype="bfloat8_b"):
        """Upper bound on one storage core's L1 for the single-reader program (matmul_multicore_reuse_mcast_dram_
        sharded_program_factory): its buffers sit on every core of the worker/storage bounding box, and a storage
        core also holds its input shard and its L1 output shard. Out/interm share one buffer (same format for the
        fp32 head and for the bf16 head with packer L1 acc). Dtypes are ttnn dtype names."""
        if self.multicore:
            raise ValueError("the L1 bound models the single-reader program only")
        out_b, act_b, w_b = _TILE_BYTES[out_dtype], _TILE_BYTES[act_dtype], _TILE_BYTES[weight_dtype]
        per_worker = math.ceil(self.n_tiles / (self.banks * self.readers))
        max_subblock_w = 4 if out_dtype == "float32" else 8
        out_tiles = math.ceil(per_worker / max_subblock_w) * max_subblock_w  # subblock widening pads per_core_N
        in1 = per_worker * self.in0_block_w * 3 * w_b  # triple-buffered K block
        in0 = self.in0_block_w * 2 * act_b  # double-buffered K block
        act_shard = (self.act_shard_w // TILE_SIZE) * act_b
        out_shard = self.per_core_N * out_b
        return in1 + out_tiles * out_b + in0 + act_shard + out_shard

    def check_l1(self, weight_dtype="bfloat8_b", budget=LMHEAD_DSHARD_L1_BUDGET, fp32_dest=False, reserved=None):
        """Raise at init (not inside the decode trace) when the fp32 drafter head would not fit in L1.

        mc: the call's circular buffers + activation and logits shards (mc_l1_need) must fit the L1 bank
        (LMHEAD_DSHARD_MC_L1_BANK) minus ``reserved`` bytes the trace already holds (default the decode all-reduce
        staging buffer, plus one QWEN36_WEIGHT_PREFETCH GCB budget per enabled target: each GCB is a lockstep L1
        allocation). A layout that does not fit would only fail later, at warm-up, with a circular-buffer / L1-buffer
        clash."""
        if self.multicore:
            if reserved is None:
                reserved = LMHEAD_DSHARD_MC_L1_RESERVED + weight_prefetch.gcb_budget_bytes() * len(
                    weight_prefetch.enabled_targets()
                )
            need = self.mc_l1_need(weight_dtype, fp32_dest)
            avail = LMHEAD_DSHARD_MC_L1_BANK - reserved
            if need["total"] > avail:
                raise ValueError(
                    f"LM-head DRAM-shard mc needs ~{need['total'] // 1024} KB of L1 (circular buffers "
                    f"{need['cb'] // 1024} KB + activation shard {need['act_shard'] // 1024} KB + logits shard "
                    f"{need['out_shard'] // 1024} KB) of {avail // 1024} KB free beside {reserved // 1024} KB the "
                    f"trace holds; shrink the logits shard with {LMHEAD_DSHARD_OUT_PCN_ENV} (10 at Qwen3.8 TP8: 97 "
                    f"cores x 10 tiles), use fewer {LMHEAD_DSHARD_READERS_ENV} or a smaller {LMHEAD_DSHARD_IBW_ENV}"
                )
            return
        need = self.l1_bytes_per_core("float32", "bfloat16", weight_dtype)
        if need > budget:
            raise ValueError(
                f"LM-head DRAM-shard with {self.readers} reader(s)/bank needs ~{need // 1024} KB of L1 per storage "
                f"core for the fp32 head (budget {budget // 1024} KB); use {LMHEAD_DSHARD_READERS_ENV}=2 or 3"
            )


def lm_head_default_fidelity(a_dtype, b_dtype):
    """The fidelity ttnn.linear picks with no program config and no compute config (matmul_device_operation):
    LoFi when both operands are block-float, else HiFi2. A program config switches that default to LoFi, so the
    DRAM-sharded head passes it explicitly."""
    low = (ttnn.bfloat8_b, ttnn.bfloat4_b)
    return ttnn.MathFidelity.LoFi if (a_dtype in low and b_dtype in low) else ttnn.MathFidelity.HiFi2


class LMHeadDShard:
    """DRAM-width-sharded copy of the (vocab-sharded) LM-head weight + the program that reads it.

    linear(x, ...) takes the same arguments the plain head call does and returns the same DRAM-interleaved
    [.., rows, vocab/TP] logits. ``applies(x)`` says whether x is decode-shaped (one tile of rows)."""

    def __init__(self, weight, mesh_device, mode, readers=None, out_pcn=None, ibw=None, fp32_dest=None):
        k, n = weight.shape[-2], weight.shape[-1]
        self.mesh_device = mesh_device
        # K6 knobs (QWEN36_LMHEAD_DSHARD_OUT_PCN / _IBW / _FP32_DEST, mc only; the plan rejects them for exact): None
        # reads the environment.
        out_pcn = lm_head_dshard_out_pcn() if out_pcn is None else out_pcn
        ibw = lm_head_dshard_ibw() if ibw is None else ibw
        self.fp32_dest = lm_head_dshard_fp32_dest() if fp32_dest is None else bool(fp32_dest)
        if self.fp32_dest and mode != "mc":
            raise ValueError(
                f"{LMHEAD_DSHARD_FP32_DEST_ENV} applies to {LMHEAD_DSHARD_ENV}=mc only "
                "(exact keeps the plain head's Dest mode)"
            )
        self.plan = p = LMHeadDShardPlan(k, n, mesh_device.dram_grid_size().x, mode, readers, out_pcn, ibw)
        if out_pcn is not None:
            grid = mesh_device.compute_with_storage_grid_size()
            if p.out_cores > grid.x * grid.y:
                raise ValueError(
                    f"{LMHEAD_DSHARD_OUT_PCN_ENV}={out_pcn}: {p.out_cores} logits cores exceed the "
                    f"{grid.x}x{grid.y} grid"
                )
        _w_names = {ttnn.bfloat8_b: "bfloat8_b", ttnn.bfloat4_b: "bfloat4_b", ttnn.bfloat16: "bfloat16"}
        _w_name = _w_names.get(weight.dtype, "bfloat16")  # unknown dtype: assume the largest weight tile
        p.check_l1(_w_name, fp32_dest=self.fp32_dest)
        self.k = k
        self.weight_dtype = weight.dtype
        bank_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(p.banks - 1, 0))})
        memory = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(bank_grid, [k, p.bank_tiles * TILE_SIZE], ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.weight = ttnn.to_memory_config(weight, memory)
        self.act_memcfg = ttnn.create_sharded_memory_config(
            shape=(TILE_SIZE, p.act_shard_w),
            core_grid=ttnn.CoreGrid(x=p.act_cores, y=1),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.progcfg = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=p.in0_block_w,
            per_core_M=1,
            per_core_N=p.per_core_N,
            fused_activation=None,
            num_workers_per_dram_bank=p.readers,
            use_multicore_pipeline=p.multicore,
        )
        logger.info(
            f"LM head DRAM-sharded ({LMHEAD_DSHARD_ENV}={mode}): K {p.k_tiles} x N {p.n_tiles} tiles in "
            f"{p.banks} banks x {p.bank_tiles} (= {p.padded_tiles}), in0_block_w {p.in0_block_w}, "
            f"{p.readers} reader(s)/bank, multicore={p.multicore}"
        )
        if p.multicore and (out_pcn is not None or ibw is not None or self.fp32_dest):
            need = p.mc_l1_need(_w_name, self.fp32_dest)
            logger.info(
                f"LM head mc K6: logits {p.per_core_N} tiles x {p.out_cores} cores, base-call fp32 Dest "
                f"{self.fp32_dest}; worst call L1 {need['total'] // 1024} KB (CBs {need['cb'] // 1024} + act "
                f"{need['act_shard'] // 1024} + logits {need['out_shard'] // 1024}) of "
                f"{LMHEAD_DSHARD_MC_L1_BANK // 1024} KB"
            )

    def applies(self, x):
        shape = x.shape
        return (
            len(shape) == 4
            and shape[0] == 1
            and shape[1] == 1
            and shape[-2] <= TILE_SIZE
            and shape[-1] == self.k
            and x.layout == ttnn.TILE_LAYOUT
        )

    def compute_config(self, x_dtype, out_dtype):
        """The compute config the plain head call ends up with when it passes none (matmul_device_operation):
        default fidelity, no approximation, fp32 Dest only for a float32 output, packer L1 acc otherwise.
        QWEN36_LMHEAD_DSHARD_FP32_DEST (mc): fp32 Dest for the bf16 output too (the pipeline then keeps each pass's
        whole K in Dest and never accumulates in L1)."""
        fp32 = out_dtype == ttnn.float32 or getattr(self, "fp32_dest", False)
        return ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=lm_head_default_fidelity(x_dtype, self.weight_dtype),
            math_approx_mode=False,
            fp32_dest_acc_en=fp32,
            packer_l1_acc=not fp32,
        )

    def linear(self, x, dtype=None, compute_kernel_config=None):
        out_dtype = x.dtype if dtype is None else dtype
        cfg = compute_kernel_config if compute_kernel_config is not None else self.compute_config(x.dtype, out_dtype)
        already = x.memory_config() == self.act_memcfg
        x_sh = x if already else ttnn.to_memory_config(x, self.act_memcfg)
        kw = {} if dtype is None else {"dtype": dtype}
        out = ttnn.linear(
            x_sh,
            self.weight,
            program_config=self.progcfg,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=cfg,
            **kw,
        )
        if not already:
            ttnn.deallocate(x_sh)
        logits = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out)
        return logits


def _gptq_default(off_value, preset_value):
    """Default of a weight-dtype flag: `preset_value` under QWEN36_GPTQ_BFP4=1 (GPTQ checkpoint preset, see
    utils/gptq_bfp4.py), else `off_value`. An explicitly set flag always wins over either default."""
    from models.demos.blackhole.qwen36.utils.gptq_bfp4 import gptq_preset_enabled

    return preset_value if gptq_preset_enabled() else off_value


BFP4_GDN_IN_ENV = "QWEN36_BFP4_GDN_IN"
_BFP4_GDN_IN_LOGGED = False


def bfp4_gdn_in_enabled():
    """QWEN36_BFP4_GDN_IN (default 0; 1 under QWEN36_GPTQ_BFP4=1): GDN input-projection weights in bfloat4_b (+ LoFi in decode only).
    Opt-in because BFP4 here costs about 1 pt of accuracy_512 top-1 (98.63 -> 97.66) and about 0.065 of
    B32 decode PCC, for +3.6% decode. Logs the choice once per process."""
    global _BFP4_GDN_IN_LOGGED
    on = os.environ.get(BFP4_GDN_IN_ENV, _gptq_default("0", "1")) == "1"
    if not _BFP4_GDN_IN_LOGGED:
        _BFP4_GDN_IN_LOGGED = True
        if on:
            logger.info(f"GDN in-proj weights: BFP4 + LOFI decode ({BFP4_GDN_IN_ENV}=1)")
        else:
            logger.info(f"GDN in-proj weights: BFP8 ({BFP4_GDN_IN_ENV}=0, default; =1 for BFP4 + LoFi decode)")
    return on


BFP4_GDN_OUT_ENV = "QWEN36_BFP4_GDN_OUT"
_BFP4_GDN_OUT_LOGGED = False


def bfp4_gdn_out_enabled():
    """QWEN36_BFP4_GDN_OUT (default 0; 1 under QWEN36_GPTQ_BFP4=1): GDN out-projection weights (decode row-parallel
    and prefill column-parallel copies) in bfloat4_b. Math fidelity unchanged (HiFi2). RTN BFP4 here cost -2.3 pt
    emulated top-1 on Qwen3.8, so use it with a GPTQ checkpoint. Logs the choice once per process."""
    global _BFP4_GDN_OUT_LOGGED
    on = os.environ.get(BFP4_GDN_OUT_ENV, _gptq_default("0", "1")) == "1"
    if not _BFP4_GDN_OUT_LOGGED:
        _BFP4_GDN_OUT_LOGGED = True
        logger.info(f"GDN out-proj weights: {'BFP4' if on else 'BFP8'} ({BFP4_GDN_OUT_ENV}={int(on)})")
    return on


BFP4_ATTN_ENV = "QWEN36_BFP4_ATTN"
_BFP4_ATTN_LOGGED = False


def bfp4_attn_enabled():
    """QWEN36_BFP4_ATTN (default 0, opt-in pending task evals): attention QKV and wo weights in bfloat4_b
    (+ LoFi in decode only). +2.0% decode; accuracy_512 top-1 -0.19 on Qwen3.6, larger on Qwen3.8 (see
    QWEN38_HANDOFF.md). Logs the choice once per process."""
    global _BFP4_ATTN_LOGGED
    on = os.environ.get(BFP4_ATTN_ENV, "0") == "1"
    if not _BFP4_ATTN_LOGGED:
        _BFP4_ATTN_LOGGED = True
        logger.info(f"Attention QKV/wo weights: {'BFP4 + LOFI decode' if on else 'BFP8'} ({BFP4_ATTN_ENV}={int(on)})")
    return on


BFP4_MLP_DOWN_ENV = "QWEN36_BFP4_MLP_DOWN"
_BFP4_MLP_DOWN_LOGGED = False


def bfp4_mlp_down_enabled():
    """QWEN36_BFP4_MLP_DOWN (default 0, opt-in pending task evals; 1 under QWEN36_GPTQ_BFP4=1): dense-MLP down_proj (w2)
    weights in bfloat4_b instead of bfloat8_b. +4.4% decode; accuracy-neutral on Qwen3.6, not on Qwen3.8 (see QWEN38_HANDOFF.md).
    """
    return os.environ.get(BFP4_MLP_DOWN_ENV, _gptq_default("0", "1")) == "1"


def resolve_mlp_down_dtype(down_dtype=None):
    """Explicit `down_dtype` wins (MoE shared expert passes bfloat8_b); None -> decide from the flag.
    Logs the dense-MLP choice once per process."""
    global _BFP4_MLP_DOWN_LOGGED
    if down_dtype is not None:
        return down_dtype
    on = bfp4_mlp_down_enabled()
    if not _BFP4_MLP_DOWN_LOGGED:
        _BFP4_MLP_DOWN_LOGGED = True
        logger.info(f"MLP down weights: {'BFP4' if on else 'BFP8'} ({BFP4_MLP_DOWN_ENV}={'1' if on else '0'})")
    return ttnn.bfloat4_b if on else ttnn.bfloat8_b


BFP8_MLP_GATEUP_ENV = "QWEN36_MLP_GATEUP_BFP8"
_BFP8_MLP_GATEUP_LOGGED = False


def bfp8_mlp_gateup_enabled():
    """QWEN36_MLP_GATEUP_BFP8 (default 1 (on); 0 reverts to bfloat4_b; default 0 under QWEN36_GPTQ_BFP4=1): dense-MLP gate/up (w1/w3, packed gate_up) weights in bfloat8_b instead of bfloat4_b."""
    return os.environ.get(BFP8_MLP_GATEUP_ENV, _gptq_default("1", "0")) == "1"


def resolve_mlp_gateup_dtype(gateup_dtype=None):
    """Explicit `gateup_dtype` wins (MoE shared expert passes bfloat4_b); None -> decide from the flag.
    Logs the dense-MLP choice once per process."""
    global _BFP8_MLP_GATEUP_LOGGED
    if gateup_dtype is not None:
        return gateup_dtype
    on = bfp8_mlp_gateup_enabled()
    if not _BFP8_MLP_GATEUP_LOGGED:
        _BFP8_MLP_GATEUP_LOGGED = True
        logger.info(f"MLP gate/up weights: {'BFP8' if on else 'BFP4'} ({BFP8_MLP_GATEUP_ENV}={'1' if on else '0'})")
    return ttnn.bfloat8_b if on else ttnn.bfloat4_b


MLP_FIDELITY_ENV = "QWEN36_MLP_FIDELITY"
_MLP_FIDELITY_LOGGED = False


def mlp_fidelity(math_fidelity=None):
    """QWEN36_MLP_FIDELITY (default hifi2; lofi still selectable): math fidelity of the dense-MLP matmuls (gate/up, down, fused swiglu AGMM;
    prefill and decode). Values: lofi | hifi2 | hifi3 | hifi4. An explicit math_fidelity wins (MoE shared expert
    passes LoFi). Logs the dense-MLP choice once per process."""
    global _MLP_FIDELITY_LOGGED
    if math_fidelity is not None:
        return math_fidelity
    name = os.environ.get(MLP_FIDELITY_ENV, "hifi2").strip().lower()
    table = {
        "lofi": ttnn.MathFidelity.LoFi,
        "hifi2": ttnn.MathFidelity.HiFi2,
        "hifi3": ttnn.MathFidelity.HiFi3,
        "hifi4": ttnn.MathFidelity.HiFi4,
    }
    if name not in table:
        raise ValueError(f"{MLP_FIDELITY_ENV}={name!r} invalid; expected one of {sorted(table)}")
    if not _MLP_FIDELITY_LOGGED:
        _MLP_FIDELITY_LOGGED = True
        logger.info(f"MLP math fidelity: {name} ({MLP_FIDELITY_ENV})")
    return table[name]


# QWEN36_MLP_GU_FUSED (default 0 = off): the dense MLP's decode gate and up projections as ONE matmul on the prefill
# fused-swiglu AGMM weight (w_gate_up: per device [dim, 2*hidden/tp], tile-pair interleaved [g0 u0 g1 u1 ...]), then
# the split-SwiGLU kernel (tt/mlp_swiglu.py) and the unchanged down projection. The decode-only w1/w3 copies are not
# loaded (frees 2 x dim x hidden/tp BFP8 bytes per layer, ~1.5 GB per chip on Qwen3.8 TP=8). Modes:
#   1d   - the 1D mcast matmul (QWEN36_MLP_GU_CORES cores, default 88; in0_block_w as gate/up, QWEN36_MLP_DECODE_IBW)
#          reads w_gate_up directly: no extra DRAM, no extra cache file.
#   dsmc - a DRAM-width-sharded copy of w_gate_up made on device at load (bit-identical tiles, 17 per bank at TP=8;
#          same DRAM as r5's w1+w3) on the multi-core DRAM-sharded pipeline: QWEN36_MLP_GU_DSMC_W workers per bank
#          (default 2), QWEN36_MLP_GU_DSMC_IBW (default 8), QWEN36_MLP_GU_DSMC_PCN output tiles per core (default 4).
#          In-trace on the 8-chip TP mesh (k3bench2): the fused matmul is 63.2 us at W2 ibw8, 68.1 at W4, 81.2 at ibw16,
#          against 39.7 + 37.2 us for r5's gate and up and 73.1 us for the 1d fused matmul.
# Only the dense TP MLP on the 1D decode path with the AGMM weight takes it; everything else keeps today's path.
MLP_GU_FUSED_ENV = "QWEN36_MLP_GU_FUSED"
MLP_GU_FUSED_MODES = ("1d", "dsmc")
_MLP_GU_FUSED_LOGGED = False
# Tiles per column pass in the DRAM-sharded multi-core pipeline's compute (its factory's dst_tiles).
_DSMC_DST_TILES = 8


def parse_mlp_gu_fused(value):
    """QWEN36_MLP_GU_FUSED value -> None (off) | "1d" | "dsmc". 1/on/true mean 1d."""
    v = (value or "").strip().lower()
    if v in ("", "0", "off", "false", "no"):
        return None
    if v in ("1", "on", "true", "yes"):
        return "1d"
    if v not in MLP_GU_FUSED_MODES:
        raise ValueError(f"{MLP_GU_FUSED_ENV}={value!r} invalid; expected 0, 1d or dsmc")
    return v


def mlp_gu_fused_mode():
    global _MLP_GU_FUSED_LOGGED
    mode = parse_mlp_gu_fused(os.environ.get(MLP_GU_FUSED_ENV, "0"))
    if mode is not None and not _MLP_GU_FUSED_LOGGED:
        _MLP_GU_FUSED_LOGGED = True
        logger.info(f"{MLP_GU_FUSED_ENV}={mode}: fused [gate|up] decode matmul on the AGMM weight + split-SwiGLU kernel")
    return mode


def dram_sharded_mc_layout(bank_tiles, workers):
    """(CN, CK, P) of the multi-core DRAM-sharded pipeline: column groups per bank, K row groups per column group,
    column passes (from anatarajan/qwen38-parity-bfp4-matmul)."""
    cn = min(workers, bank_tiles)
    ck = workers // bank_tiles if workers > bank_tiles else 1
    passes = math.ceil(math.ceil(bank_tiles / cn) / _DSMC_DST_TILES)
    return cn, ck, passes


def dram_sharded_mc_workers_ok(bank_tiles, workers):
    """The pipeline factory's layout rules (its TT_FATALs) for `workers` per bank on a bank of `bank_tiles` columns."""
    if workers < 2:
        return False
    cn, ck, passes = dram_sharded_mc_layout(bank_tiles, workers)
    return (ck == 1 or passes == 1) and bank_tiles // cn >= passes


def create_dram_sharded_mc_progcfg(k, n, out_per_core_n, workers, ibw):
    """Multi-core DRAM-sharded decode matmul config (M = 1 tile). per_core_N is the OUTPUT shard width (the op lays
    its width-sharded output on ceil(N / per_core_N) cores, row-major on the compute grid, whatever shard spec the
    caller passes), so it must divide the N tiles."""
    n_tiles = math.ceil(n / TILE_SIZE)
    if n_tiles % out_per_core_n:
        raise ValueError(f"DRAM-sharded mc: per_core_N {out_per_core_n} does not divide {n_tiles} N tiles")
    bank_tiles = _roundup(n, TILE_SIZE * DRAM_CORES) // TILE_SIZE // DRAM_CORES
    if not dram_sharded_mc_workers_ok(bank_tiles, workers):
        raise ValueError(f"DRAM-sharded mc: {workers} workers per bank is not a valid layout for {bank_tiles} tiles/bank")
    if ibw < 1:
        raise ValueError(f"DRAM-sharded mc: in0_block_w {ibw} must be at least 1")
    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=ibw,
        per_core_M=1,
        per_core_N=out_per_core_n,
        fused_activation=None,
        num_workers_per_dram_bank=workers,
        use_multicore_pipeline=True,
    )


def mlp_gu_dsmc_settings():
    """(workers per DRAM bank, in0_block_w, output tiles per core) of QWEN36_MLP_GU_FUSED=dsmc."""
    return (
        int(os.environ.get("QWEN36_MLP_GU_DSMC_W", "2")),
        int(os.environ.get("QWEN36_MLP_GU_DSMC_IBW", "8")),
        int(os.environ.get("QWEN36_MLP_GU_DSMC_PCN", "4")),
    )


# Grid helpers
def prefill_grid_default():
    """BH P150: (8,10); WH: (8,8). y capped at 10 on BH (grid_x=10 breaks matmul)."""
    return (8, 10) if is_blackhole() else (8, 8)


# Max grid COLUMNS a tuned prefill config may use. A Blackhole galaxy reports a 12-wide worker
# grid, but harvested P150s expose only 11, so tuning to 12 would not port. 11 x 10 = 110 cores.
PREFILL_MAX_COLS_PORTABLE = 11

# Why TP=8 wants different values (measured at S=2048, 27B, 1x8 Ring):
#   * widest_cols -- `_best_prefill_cols` ranks candidate widths by (out_subblock_w, cols), i.e.
#     subblock first. At TP=8 the halved N makes wide grids yield a small per_core_N and hence a
#     narrow subblock, so that ranking retreats to fewer columns and leaves cores idle. Measured
#     device time is monotonically decreasing in column count instead: attn_wo went 1944us @ 60
#     cores -> 700us @ 110, and mlp_gate 2943us @ 60 -> 1935us @ 110. So take the width.
#   * in0_block_w_divisor -- `min(cap, k_tiles // grid_x)` is a function of the per-device K, which
#     halves. attn_wo/gdn_out go k_tiles 48 -> 24 and `24 // 11 = 2`, but in0_block_w only has to
#     DIVIDE k_tiles, so a larger block is legal and much faster (attn_wo @ 11 cols, from the sweep:
#     bw2 786us, bw4 719us, bw6 700us, bw8 705us).
#
# in0_block_w_cap is L1-BOUND, NOT just a legality bound. in0_block_w sizes the in0 circular
# buffer, and `_wo_proj` / the MLP prefill arm write their OUTPUT to L1 (attention/tp.py:246,
# mlp.py:284) -- so the CBs and a resident L1 output tensor compete for the same 1536 KB. Measured
# on the real model: cap=8 overflows and test_model_tp_long_prefill dies with
#   "Statically allocated circular buffers in program N clash with L1 buffers on core range
#    [0-0 - 10-8]. L1 buffer allocated at 1314560 and static circular buffer region ends at 1372032"
# from attention/tp.py:241. A standalone per-op sweep CANNOT see this: in isolation the only L1
# tenant is the op under test, so it reports a win that the full model has no room for. Any future
# raise of this cap must be validated by test_model_tp_long_prefill, not by the sweep alone.
_PREFILL_TUNING = {
    4: dict(widest_cols=False, in0_block_w_divisor=False, in0_block_w_cap=4),
    8: dict(widest_cols=True, in0_block_w_divisor=True, in0_block_w_cap=4),
}


def prefill_tuning(num_devices):
    """Prefill matmul tuning for this TP; unknown TP falls back to the frozen TP=4 values."""
    return _PREFILL_TUNING.get(num_devices, _PREFILL_TUNING[4])


def _roundup(a, b):
    return b * math.ceil(a / b)


def aliases(a, b):
    """Whether two handles name the same device buffer (a metadata re-view, not a copy).

    Returns True when the address is unavailable, so the caller skips the extra free: that leaks a
    copy at worst and never double-frees an alias."""
    try:
        return a.buffer_address() == b.buffer_address()
    except Exception:  # pragma: no cover
        return True


def _find_largest_divisor(n, max_div=8):
    for d in range(max_div, 0, -1):
        if n % d == 0:
            return d
    return 1


def _find_grid(n_tiles, target=32):
    max_r, max_c = 8, 8
    possible = [k for k in range(1, max_r * max_c + 1) if n_tiles % k == 0]
    possible.sort(key=lambda x: abs(x - target))
    for cores in possible:
        for rows in range(1, max_r + 1):
            if cores % rows == 0:
                cols = cores // rows
                if cols <= max_c:
                    return rows, cols
    raise ValueError(f"Cannot find grid for {n_tiles} tiles")


# DRAM-sharded config builders
def create_dram_sharded_mem_config(k, n):
    """WIDTH_SHARDED DRAM memory config for a weight matrix [k, n]."""
    padded_n = _roundup(n, TILE_SIZE * DRAM_CORES)
    shard_spec = ttnn.ShardSpec(
        DRAM_GRID,
        (k, padded_n // DRAM_CORES),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        shard_spec,
    )


def create_dram_sharded_matmul_program_config(m, k, n, num_cores=None):
    """DRAM-sharded matmul program config (decode, small M)."""
    m_tiles = math.ceil(m / TILE_SIZE)
    k_tiles = math.ceil(k / TILE_SIZE)
    n_padded = _roundup(n, TILE_SIZE * DRAM_CORES)
    n_tiles = n_padded // TILE_SIZE

    if num_cores is None:
        rows, cols = _find_grid(k_tiles)
        num_cores = rows * cols

    k_tiles_per_core = k_tiles // num_cores
    if k_tiles_per_core == 0:
        k_tiles_per_core = k_tiles
        num_cores = 1
    in0_block_w = _find_largest_divisor(k_tiles_per_core)
    per_core_N = n_tiles // num_cores if n_tiles >= num_cores else 1

    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w,
        per_core_M=m_tiles,
        per_core_N=per_core_N,
        fused_activation=None,
    )


def create_matmul_1d_decode_progcfg(
    m, k, n, num_cores, fused_activation=None, fp32_acc=True, grid_w=8, max_in0_block_w=8
):
    """Explicit-grid 1D (mcast_in0) decode matmul progcfg on ~`num_cores` cores — small grids beat
    the ~80-core DRAM-sharded grid on the bandwidth-bound skinny decode matmuls. Weight must be interleaved.

    Grid is shaped WIDE-first (cols up to `grid_w`, the device worker-grid width — 11 on BH P150, 8 on
    WH): for a fixed core budget a wide-short grid shortens the in0 multicast column and beats a
    tall-narrow one (~2% on this matmul; see test_mlp_matmul_sweep wide1d_* vs forced1d_*). Default
    grid_w=8 preserves the legacy shaping for callers that don't pass the device width.

    `max_in0_block_w` caps in0_block_w (largest divisor of k_tiles <= cap); default 8 = legacy."""
    cols = min(grid_w, num_cores)
    rows = math.ceil(num_cores / cols)
    m_tiles = math.ceil(m / TILE_SIZE)
    k_tiles = math.ceil(k / TILE_SIZE)
    n_tiles = math.ceil(n / TILE_SIZE)
    # mcast_in0: every core streams the full K, so in0_block_w must divide the full k_tiles.
    per_core_k = _find_largest_divisor(k_tiles, max_in0_block_w)
    per_core_n = math.ceil(n_tiles / (cols * rows))
    cap = 4 if fp32_acc else 8  # fp32_dest_acc caps subblock area at 4
    sub_w = max(i for i in range(1, cap + 1) if per_core_n % i == 0)
    sub_h = max(i for i in range(1, cap + 1) if m_tiles % i == 0 and i * sub_w <= cap)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(cols, rows),
        in0_block_w=per_core_k,
        out_subblock_h=sub_h,
        out_subblock_w=sub_w,
        per_core_M=m_tiles,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=fused_activation,
        mcast_in0=True,
    )


def matmul_1d_decode(x, weight, decode_1d_progcfg, compute_cfg, out_memory_config=ttnn.L1_MEMORY_CONFIG):
    """Small-grid 1D (mcast_in0) decode matmul on an interleaved weight; interleaves the K-sharded
    activation first since mcast_in0 needs the full K per core. See test_mlp_matmul_sweep."""
    x_il = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
    # QWEN36_WEIGHT_PREFETCH: a weight registered for this progcfg streams through the Tensor prefetcher; else plain.
    out = weight_prefetch.linear(
        x_il,
        weight,
        decode_1d_progcfg,
        compute_kernel_config=compute_cfg,
        memory_config=out_memory_config,
    )
    if x_il is not x:
        ttnn.deallocate(x_il)
    return out


def create_activation_shard_config(k):
    """WIDTH_SHARDED L1 activation config for a [*, k] activation."""
    k_tiles = k // TILE_SIZE
    rows, cols = _find_grid(k_tiles)
    num_cores = rows * cols
    width_per_core = k // num_cores
    return ttnn.create_sharded_memory_config(
        shape=(TILE_SIZE, width_per_core),
        core_grid=ttnn.CoreGrid(x=cols, y=rows),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


# 2D prefill matmul config
def _get_out_subblock_w(per_core_n, out_subblock_h):
    for w in range(min(per_core_n, 4 // out_subblock_h), 0, -1):
        if per_core_n % w == 0:
            return w
    return 1


def _full_grid_crs(grid):
    """Full-grid allowed_worker_cores for CCL-fused matmuls, which bypass ttnn::prim::matmul()'s normalize_program_config()."""
    gx, gy = grid
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})


def create_prefill_matmul_program_config(m, k, n, grid_size=None, fused_activation=None, tuning=None):
    """2D prefill matmul progcfg (DRAM-interleaved).

    fused_activation in packer; sharded kernel rejects ttnn.linear(activation=...) with progcfg.
    tuning: a `_PREFILL_TUNING` entry (see `prefill_tuning`); None = the frozen TP=4 behavior."""
    if grid_size is None:
        grid_size = prefill_grid_default()
    tuning = tuning or _PREFILL_TUNING[4]
    per_core_M = max(1, math.ceil(m / TILE_SIZE / grid_size[1]))
    per_core_N = max(1, math.ceil(n / TILE_SIZE / grid_size[0]))
    # Above the validated 2048-row chunk the per-core block (and its circular buffers) grows with rows past L1:
    # block the output at the 2048-row block height and round per_core_M up to a multiple of it.
    blocked = {}
    if not prefill_l1_rows_ok(m):
        ob = max(1, math.ceil(2048 / TILE_SIZE / grid_size[1]))
        if per_core_M > ob:
            per_core_M = math.ceil(per_core_M / ob) * ob
            blocked = {"out_block_h": ob}

    out_subblock_h = 1
    out_subblock_w = _get_out_subblock_w(per_core_N, out_subblock_h)

    k_tiles = math.ceil(k / TILE_SIZE)
    cap = tuning["in0_block_w_cap"]
    if tuning["in0_block_w_divisor"]:
        # in0_block_w only has to divide k_tiles (no K tail in the 2D mcast kernel), so take the
        # largest legal block rather than scaling with grid width -- see _PREFILL_TUNING.
        in0_block_w = _find_largest_divisor(k_tiles, cap)
    else:
        in0_block_w = min(cap, max(1, k_tiles // grid_size[0]))

    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=fused_activation,
        fuse_batch=False,
        **blocked,
    )


def _widest_prefill_cols(n, max_cols, subblock_slack=1):
    """Widest grid whose output subblock stays within `subblock_slack` of the best achievable.

    The TP=8 counterpart to `_best_prefill_cols`. More columns is usually a win at TP=8 (the halved
    per-device N leaves cores idle), but NOT when the extra width collapses the subblock: measured
    at S=2048, mlp_gate (N=2176 -> 68 tiles) goes cols 9 -> 11, per_core_N 8 -> 7, and 7 is prime so
    out_subblock_w drops 4 -> 1 -- a 2058us -> 2118us REGRESSION, i.e. the subblock-first ranking
    was right for that shape. Guarding on the subblock keeps the wide grid exactly where it pays:

        matmul     default        this rule       measured
        attn_wo    c10_bw2_sw4    c11_bw4_sw3     803.5 -> 718.7us
        gdn_out    c10_bw2_sw4    c11_bw4_sw3     802.3 -> 719.9us
        mlp_down   c10_bw4_sw4    c11_bw4_sw3    1787.4 -> 1724.9us
        mlp_gate   c9_bw4_sw4     c9_bw4_sw4     2058.1us (unchanged -- already optimal)
    """
    n_tiles = math.ceil(n / TILE_SIZE)
    sw = {cols: _get_out_subblock_w(math.ceil(n_tiles / cols), 1) for cols in range(1, max_cols + 1)}
    floor = max(sw.values()) - subblock_slack
    return max((cols for cols, w in sw.items() if w >= floor), default=1)


def _best_prefill_cols(n, max_cols):
    """Grid width (<=max_cols) maximizing the output subblock, tie-broken to more cores — avoids the
    1x1-subblock stall (e.g. gate/up N=4352 -> 7-wide -> 1x4) the default full width can force."""
    n_tiles = math.ceil(n / TILE_SIZE)
    best_cols, best_key = 1, None
    for cols in range(1, max_cols + 1):
        sw = _get_out_subblock_w(math.ceil(n_tiles / cols), 1)
        key = (sw, cols)  # prefer wider subblock, then more columns (more compute cores)
        if best_key is None or key > best_key:
            best_key, best_cols = key, cols
    return best_cols


def create_prefill_mlp_matmul_program_config(m, k, n, fused_activation=None, max_cols=None, tuning=None):
    """FPU-tuned 2D prefill progcfg for MLP matmuls: picks the grid width that maximizes the output
    subblock (drives prefill FPU) instead of the default full width.

    max_cols caps the grid width. Default = prefill_grid_default()[0] (8). Pass the device worker-grid
    width (11 on BH P150) to let the subblock heuristic go wide -> the measured prefill winners
    (gate 9-wide, down/wo 10-wide, gdn_qkvz 11-wide; test_mlp_matmul_sweep_prefill). Fused AG/RS paths
    pin 8-wide separately and are unaffected.

    tuning: a `_PREFILL_TUNING` entry. With `widest_cols` (TP=8) the subblock-first width heuristic
    is replaced by "take the width, clamped to PREFILL_MAX_COLS_PORTABLE" -- measured device time at
    TP=8 falls monotonically with column count, so trading cores for a wider subblock loses."""
    grid = prefill_grid_default()
    tuning = tuning or _PREFILL_TUNING[4]
    limit = max_cols or grid[0]
    if tuning["widest_cols"]:
        # Cap the width at PREFILL_MAX_COLS_PORTABLE (harvested parts expose 11, not 12) and never
        # exceed the output tile count -- columns beyond it get per_core_N=1 with nothing to compute,
        # paying mcast cost for no work.
        cols = _widest_prefill_cols(n, max(1, min(limit, PREFILL_MAX_COLS_PORTABLE, math.ceil(n / TILE_SIZE))))
    else:
        cols = _best_prefill_cols(n, limit)
    return create_prefill_matmul_program_config(
        m, k, n, grid_size=(cols, grid[1]), fused_activation=fused_activation, tuning=tuning
    )


# Mesh tensor helpers
def shard_w(torch_tensor, mesh, dim, memory_config, cache_path, dtype=ttnn.bfloat8_b):
    """Torch weight [out,in] -> sharded mesh tensor. Transpose to [in,out]; dim=-1 column, dim=0 row.

    The bf16 cast + transpose runs as the as_tensor preprocess, so it only executes on a tensor-cache
    miss. On a hit the checkpoint tensor is never materialised (it may be a memory-mapped safetensor
    on a network mount, and reading 27B parameters through it is what pushed the CI weight load past
    its 1200 s pytest timeout)."""
    return ttnn.as_tensor(
        torch_tensor,
        preprocess=lambda t: t.to(torch.bfloat16).T.contiguous(),
        dtype=dtype,
        device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=dim),
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config,
        cache_file_name=cache_path,
    )


def agmm_k_block_size(k_local, default=8):
    """Largest power-of-2 K_block_size <= `default` that divides K_tiles/device (AGMM Ring has no tail).

    TP=4: 1280->40 tiles->8; TP=8: 640->20 tiles->4. Odd divisors (e.g. 5|20) are unsafe on Ring.
    """
    k_tiles = k_local // TILE_SIZE
    b = 1 << (min(default, max(1, k_tiles)).bit_length() - 1)
    while b > 1 and k_tiles % b:
        b //= 2
    return b


def agmm_gather_buffer(tt_ccl, x, cluster_axis=1):
    """Persistent gather buffer for all_gather_minimal_matmul_async on activation x [.,S,K/tp].

    TODO(#57458): once all_gather_minimal_matmul_async honours barrier_semaphore (an in-kernel receiver-ready
    handshake), drop this buffer and pass barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis)
    instead; test_gdn_out_agmm_deterministic_under_device_skew must keep passing."""
    cache = tt_ccl.__dict__.setdefault("_agmm_gather_buffers", {})
    mesh = tt_ccl.mesh_device
    S, K_local = x.shape[-2], x.shape[-1]
    key = (S, K_local, cluster_axis, x.dtype)
    if key not in cache:
        shape = ttnn.Shape([1, 1, S, K_local * mesh.shape[cluster_axis]])
        pair = [
            ttnn.allocate_tensor_on_device(shape, x.dtype, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG)
            for _ in range(2)
        ]
        cache[key] = [pair, 0]
    pair, idx = cache[key]
    cache[key][1] = idx ^ 1
    return pair[idx]


_PREFILL_AGMM_TUNED = None


def prefill_agmm_tuned_enabled():
    """QWEN36_PREFILL_AGMM_TUNED (default "0" = today's AGMM grid and blocks, byte for byte; read once): "1" runs the
    prefill fused all-gather + matmul ops (MLP gate|up SwiGLU, GDN qkvzab, attention QKV, GDN-out column-parallel) with
    the per-shape, per-row-count configs of _AGMM_TUNED: a 12- (or 11-) wide compute grid where today's is 8 wide, and
    for short buckets (128 - 1024 rows; MLP gate|up, and at 128 rows the GDN / attention in-projections) the
    non-transposed core grid, which streams the weight through a 12-wide row of readers instead of a 9-high column. K_block_size stays agmm_k_block_size(K/tp), so every output tile accumulates the
    same K blocks in the same order: the outputs are bit-identical to today's (checked per config on the 8-chip ring,
    tests/test_prefill_r5_ops.py). Shapes / row counts without an entry keep today's config."""
    global _PREFILL_AGMM_TUNED
    if _PREFILL_AGMM_TUNED is None:
        _PREFILL_AGMM_TUNED = os.environ.get("QWEN36_PREFILL_AGMM_TUNED", "0") == "1"
    return _PREFILL_AGMM_TUNED


# (swiglu, K, N/tp) with K and N rounded up to a tile (the GDN in-proj weight is 2060 wide -> 2080) -> {rows: (grid_x,
# grid_y, M_block, N_block, transpose)} for the row counts the prefill traces run (short buckets 128 / 512 / 1024 /
# 2048, chunk traces 2048 / 8192); any other shape or row count keeps today's config. num_links 2, workers per link =
# ceil(in0 sender axis / 2) (grid_x transposed, grid_y not), subblock (1, 4), K_block = agmm_k_block_size(K/tp).
# Device ms per op (16 calls captured in one trace) on the 8-chip ring (P150x8 Galaxy row, bf16 x, bfp8 weight),
# today -> tuned, all bit-identical:
#   gate|up   8192: 3.03 -> 2.21  2048: 0.714 -> 0.586  1024: 0.414 -> 0.369  512: 0.297 -> 0.265  128: 0.286 -> 0.236
#   qkvzab    8192: 1.64 -> 1.49  128: 0.109 -> 0.099  (512 - 2048: every wider or non-transposed grid was slower)
#   attn qkv  8192: 1.63 -> 1.49  128: 0.107 -> 0.100  (512 - 2048 slower)
#   gdn_out   8192: 1.65 -> 1.56  2048: 0.428 -> 0.409  1024: 0.226 -> 0.214  (128 / 512 slower)
_AGMM_TUNED = {
    (True, 5120, 4352): {
        128: (12, 9, 8, 16, False),
        512: (12, 9, 8, 16, False),
        1024: (12, 9, 8, 16, False),
        2048: (11, 9, 8, 16, True),
        8192: (12, 9, 8, 16, True),
    },
    (False, 5120, 2080): {128: (12, 9, 4, 8, False), 8192: (12, 9, 8, 8, True)},
    (False, 5120, 2048): {128: (12, 9, 4, 8, False), 8192: (12, 9, 8, 16, True)},
    (False, 6144, 640): {1024: (12, 9, 4, 8, True), 2048: (11, 9, 4, 8, True), 8192: (11, 9, 4, 8, True)},
}
_AGMM_TUNED_LOGGED = set()


def agmm_tuned_config(rows, weight, k_local, swiglu, mesh_device):
    """(MinimalMatmulConfig, num_links, workers_per_link, force_transpose) of the tuned AGMM for this shape and row count,
    or None to keep today's config (flag off, no entry, a device grid too small for the entry, or a non-transposed entry
    that the op would transpose anyway because rows > N)."""
    if rows <= TILE_SIZE or not prefill_agmm_tuned_enabled():
        return None
    key = (bool(swiglu), _roundup(int(weight.shape[-2]), TILE_SIZE), _roundup(int(weight.shape[-1]), TILE_SIZE))
    entry = _AGMM_TUNED.get(key, {}).get(int(rows))
    if entry is None:
        return None
    gx, gy, mb, nb, transpose = entry
    full = mesh_device.compute_with_storage_grid_size()
    if gx > full.x or gy >= full.y:  # the fabric muxes take the row below the compute grid
        return None
    if not transpose and rows > (key[2] // 2 if swiglu else key[2]):
        return None  # force_transpose=False only means "do not force": the op transposes M > N itself
    num_links = 2
    workers = -(-(gx if transpose else gy) // num_links)
    if key not in _AGMM_TUNED_LOGGED:
        _AGMM_TUNED_LOGGED.add(key)
        logger.info(f"AGMM tuned configs (QWEN36_PREFILL_AGMM_TUNED) for swiglu={key[0]} K={key[1]} N={key[2]}")
    cfg = ttnn.MinimalMatmulConfig(
        M_block_size=mb,
        K_block_size=agmm_k_block_size(k_local),
        N_block_size=nb,
        subblock_h=1,
        subblock_w=4,
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
    )
    return cfg, num_links, workers, transpose


def all_gather_matmul_prefill(
    x,
    weight,
    tt_ccl,
    compute_cfg,
    topology,
    grid=(7, 9),
    cluster_axis=1,
    fused_activation=None,
    out_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    persistent_output_buffer=None,
):
    """Fused all-gather(dim=3) + column-parallel matmul for prefill (all_gather_minimal_matmul_async).

    x: K-sharded activation [.,S,K/tp]; weight: [K,N] col-sharded (K full). Gathers x to full K and
    matmuls in one op, replacing a separate all_gather + linear. fused_activation applied per tile
    before pack (non-parametrized op, e.g. ttnn.UnaryOpType.SILU). out_memory_config places the result
    (default DRAM; L1 keeps it resident for downstream slices). persistent_output_buffer: the gather buffer
    (agmm_gather_buffer); None lets the op allocate one per call."""
    S, K_local = x.shape[-2], x.shape[-1]
    x4 = ttnn.reshape(x, (1, 1, S, K_local))
    tuned = agmm_tuned_config(S, weight, K_local, False, tt_ccl.mesh_device)  # QWEN36_PREFILL_AGMM_TUNED
    if tuned is not None:
        cfg, num_links, workers, transpose = tuned
    else:
        # AG-bound: 2 ethernet links parallelize the gather (P150x4 max; traced_8k TTFT win). grid.x must
        # = num_links*workers, and the 7-wide default (prime) forces 1 link -> widen to 8 (2 links, 4 workers).
        num_links = 2
        grid = (8, grid[1])
        workers = grid[0] // num_links
        transpose = True
        cfg = ttnn.MinimalMatmulConfig(
            M_block_size=4,
            K_block_size=agmm_k_block_size(K_local),
            N_block_size=8,
            subblock_h=1,
            subblock_w=4,
            compute_with_storage_grid_size=ttnn.CoreCoord(grid[0], grid[1]),
        )
    out = ttnn.experimental.all_gather_minimal_matmul_async(
        input_tensor=x4,
        weight_tensor=weight,
        config=cfg,
        fused_activation=fused_activation,
        compute_kernel_config=compute_cfg,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
        num_links=num_links,
        topology=topology,
        cluster_axis=cluster_axis,
        memory_config=out_memory_config,
        dtype=ttnn.bfloat16,
        force_transpose=transpose,
        num_workers_per_link=workers,
        num_buffers_per_channel=8,
        persistent_output_buffer=persistent_output_buffer,
    )[0]

    return out


def mlp_gateup_agmm_enabled(num_devices):
    """Fuse the ff_norm all-gather into the MLP gate/up matmul (prefill). TP-only (needs the gather)."""
    return num_devices > 1


def all_gather_swiglu_prefill(
    x, weight, tt_ccl, compute_cfg, topology, grid=(7, 9), cluster_axis=1, out_memory_config=ttnn.DRAM_MEMORY_CONFIG
):
    """Fused all-gather + col-parallel gate/up matmul + SwiGLU for prefill (packing gate+up lets ff_norm's AG fuse in).

    x: K-sharded [.,S,K/tp]; weight: tile-pair-interleaved [gate|up] [K, 2N/tp]. Emits silu(gate)*up of width N/tp."""
    S, K_local = x.shape[-2], x.shape[-1]
    x4 = ttnn.reshape(x, (1, 1, S, K_local))
    tuned = agmm_tuned_config(S, weight, K_local, True, tt_ccl.mesh_device)  # QWEN36_PREFILL_AGMM_TUNED
    if tuned is not None:
        cfg, num_links, workers, transpose = tuned
    else:
        num_links = 2
        grid = (8, grid[1])
        workers = grid[0] // num_links
        transpose = True
        cfg = ttnn.MinimalMatmulConfig(
            M_block_size=8,
            K_block_size=agmm_k_block_size(K_local),
            N_block_size=16,
            subblock_h=1,
            subblock_w=4,
            compute_with_storage_grid_size=ttnn.CoreCoord(grid[0], grid[1]),
        )
    return ttnn.experimental.all_gather_minimal_matmul_async(
        input_tensor=x4,
        weight_tensor=weight,
        config=cfg,
        compute_kernel_config=compute_cfg,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
        num_links=num_links,
        topology=topology,
        cluster_axis=cluster_axis,
        memory_config=out_memory_config,
        dtype=ttnn.bfloat16,
        force_transpose=transpose,
        num_workers_per_link=workers,
        num_buffers_per_channel=8,
        fuse_swiglu=True,
    )[0]


def build_mmrs_decode_state(mesh_device, M, K_local, N, nd, dtype=ttnn.bfloat16):
    """Build (progcfg, intermediate_buffer, output_buffer) for a decode matmul_reduce_scatter out-proj.

    M = LOGICAL decode batch (max_batch_size) — the op returns the persistent buffer with its logical
    shape, so an oversized (tile-padded) M leaks into the residual stream. TILE layout pads M<32.
    dtype MUST match the out-proj input activation (bf16 for MLP/attn; FLOAT32 for GDN, which keeps
    fp32 for stability) — the op's default output dtype is the input's, and writing it into a
    mismatched buffer corrupts the output. Matmul on reduced grid (8,6); RS workers at offset (0,6).
    interm [1,1,M,N], out [1,1,M,N/nd]."""
    cg = (8, 6)
    per_core_N = max(1, math.ceil(N / TILE_SIZE / cg[0]))
    pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=cg,
        in0_block_w=min(4, max(1, K_local // TILE_SIZE // cg[0])),
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=max(1, math.ceil(M / TILE_SIZE / cg[1])),
        per_core_N=per_core_N,
        out_block_w=max(1, per_core_N // 2),
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
        allowed_worker_cores=_full_grid_crs(cg),
    )
    mk = lambda w: ttnn.from_torch(
        torch.zeros(1, 1, M, w),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    return pc, mk(N), mk(N // nd)


def matmul_reduce_scatter_decode(
    x, weight, tt_ccl, interm_buf, out_buf, progcfg, compute_cfg, topology, rs_offset=(0, 6)
):
    """Fused row-parallel matmul + reduce-scatter(dim=3) for decode (matmul_reduce_scatter_async).

    x: K-sharded [.,M,K_local]; weight: [K_local,N] K-sharded. Matmul runs on progcfg's (reduced)
    grid; RS workers land at rs_offset (disjoint rows) to avoid the collision that deadlocks a
    full-grid fused CCL. Persistent buffers are caller-owned. Returns [.,M,N/nd] (fractured, DRAM)."""
    _, rs_out = ttnn.experimental.matmul_reduce_scatter_async(
        x,
        weight,
        persistent_intermediate_buffer=interm_buf,
        persistent_output_buffer=out_buf,
        dim=3,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(),
        reduce_scatter_core_grid_offset=rs_offset,
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
        num_links=1,
        memory_config_rs=ttnn.DRAM_MEMORY_CONFIG,
        topology=topology,
        subdevice_id=None,
        memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
        program_config=progcfg,
        compute_kernel_config=compute_cfg,
    )
    # rs_out IS the persistent output buffer; clone so the caller can deallocate its copy while the
    # persistent buffer survives for the next token (else layer.py's deallocate frees it -> corruption).
    return ttnn.clone(rs_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _mmrs_prefill_shared_bufs(tt_ccl, M, N, nd, dtype):
    """Lazily allocate (and cache on tt_ccl) shared persistent buffers for the prefill fused out-proj.

    Prefill M (=chunk seq, e.g. 2048) makes per-layer buffers huge (fp32 [1,1,2048,5120]≈42MB × 64
    layers = infeasible). Prefill runs layers sequentially and each op's output is cloned before the
    next layer reuses the buffer, so ONE shared set per (M,N,nd,dtype) is safe. Allocated during the
    pre-capture warmup forward (eager), reused inside the trace. Keyed so variable M/dtype coexist."""
    cache = getattr(tt_ccl, "_qwen36_mmrs_prefill_bufs", None)
    if cache is None:
        cache = {}
        tt_ccl._qwen36_mmrs_prefill_bufs = cache
    key = (M, N, nd, str(dtype))
    if key not in cache:
        mesh = tt_ccl.mesh_device
        mk = lambda w: ttnn.from_torch(
            torch.zeros(1, 1, M, w),
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        cache[key] = (mk(N), mk(N // nd))
    return cache[key]


def matmul_reduce_scatter_prefill(x, weight, tt_ccl, compute_cfg, topology, nd, dtype, grid=(8, 8), rs_offset=(0, 8)):
    """Fused row-parallel out-proj matmul + reduce-scatter for PREFILL (matmul_reduce_scatter_async).

    Unlike decode (M=1, where the 2D matmul collapses to ~8 cores and this loses), at prefill M>>1 the
    2D matmul fills the grid, so overlapping the RS with the matmul is a WIN (biggest for the fp32
    GDN-out with its large RS). grid=(8,8): matmul rows 0-7, RS workers rows 8-9. x: K-sharded
    [.,M,K_local]; weight [K_local,N]. Returns [1,1,M,N/nd] (cloned; shared buffer survives)."""
    M, K_local = x.shape[-2], x.shape[-1]
    N = weight.shape[-1]
    interm, out_buf = _mmrs_prefill_shared_bufs(tt_ccl, M, N, nd, dtype)
    x4 = ttnn.reshape(x, (1, 1, M, K_local))
    # RS-bound: 2 ethernet links parallelize the fp32 cross-device reduce (P150x4 max; traced_8k win).
    # grid (8,8) leaves rows 8-9 for the 2 RS worker rows.
    num_links = 2
    per_core_N = max(1, math.ceil(N / TILE_SIZE / grid[0]))
    pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=min(4, max(1, K_local // TILE_SIZE // grid[0])),
        out_subblock_h=1,
        # Keep 1x1: op242 is RS-bound and this op is pipelined to overlap the matmul with the RS.
        # Widening the subblock desyncs that overlap and measured net-negative on traced_8k TTFT.
        out_subblock_w=1,
        per_core_M=max(1, math.ceil(M / TILE_SIZE / grid[1])),
        per_core_N=per_core_N,
        out_block_w=max(1, per_core_N // 2),
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
        allowed_worker_cores=_full_grid_crs(grid),
    )
    _, rs = ttnn.experimental.matmul_reduce_scatter_async(
        x4,
        weight,
        persistent_intermediate_buffer=interm,
        persistent_output_buffer=out_buf,
        dim=3,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(),
        reduce_scatter_core_grid_offset=rs_offset,
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
        num_links=num_links,
        memory_config_rs=ttnn.DRAM_MEMORY_CONFIG,
        topology=topology,
        subdevice_id=None,
        memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
        program_config=pc,
        compute_kernel_config=compute_cfg,
    )
    return ttnn.clone(rs, memory_config=ttnn.DRAM_MEMORY_CONFIG)


# ---------------------------------------------------------------------------------------------
# Decode-only replicated-residual path: ONE all_reduce_async per row-parallel output instead of
# [DistributedNorm all-gather + reduce-scatter] (4 CCLs/layer -> 2).
#
#   QWEN36_DECODE_ALLREDUCE=1 (default)  decode residual is REPLICATED full-width ([1,1,B,dim]) in L1
#                                        WIDTH_SHARDED on a 10x4 = 40-core grid (shard [32, dim/40]).
#                                        Norms are local sharded rms_norms, residual adds stay in L1, each
#                                        row-parallel output is resharded to that layout and summed with
#                                        ttnn.experimental.all_reduce_async into a persistent L1 buffer.
#                                        The embedding is all-gathered once at model entry.
#   QWEN36_DECODE_ALLREDUCE=0            restores the previous behaviour: decode residual is hidden-fractured
#                                        ([1,1,B,dim/tp], DRAM); every norm all-gathers, every
#                                        row-parallel output reduce-scatters.
#   QWEN36_DECODE_AR_GDN_FP32=0 (default) GDN out-proj partial (fp32 in A) is cast to bf16 before the
#                                        all-reduce (as the qwen38 QB2 port does).
#   QWEN36_DECODE_AR_GDN_FP32=1          GDN partial all-reduced in fp32 (fp32 persistent buffer,
#                                        fp32 dest accumulation); the residual add then mixes bf16 + fp32
#                                        exactly like the reduce-scatter path does today.
#   QWEN36_DECODE_AR_FUSE_RESIDUAL=0 (default) all_reduce_async, then a separate RES40 ttnn.add of the residual.
#   QWEN36_DECODE_AR_FUSE_RESIDUAL=1     the residual add runs inside all_reduce_async's reduction kernel
#                                        (residual_tensor=: the residual's RES40 shard is a CB the kernel adds onto
#                                        the reduced tile in DST before packing). Removes the 2 RES40 adds per layer
#                                        and their L1 output. bf16 partials only (an fp32 GDN partial under
#                                        QWEN36_DECODE_AR_GDN_FP32=1 keeps the separate add). The layer arms the
#                                        residual (arm_residual) before attention / MLP and takes the sum with
#                                        add_residual; any other all_reduce caller is unaffected.
#   QWEN36_DECODE_AR_LIVE_ROWS=0 (default) every bf16 all-reduce sends all 32 rows of each tile (full 2 KB tiles).
#   QWEN36_DECODE_AR_LIVE_ROWS=1         a bf16 all-reduce whose partial has <= 16 logical rows (plain decode at <= 16
#                                        users, spec verify / drafter buckets of <= 4 users x 4 rows) passes
#                                        live_rows_packing=True: the op sends only the top half (rows 0..15) of each
#                                        tile, 4 half-tiles per 4 KB fabric packet, half the fabric bytes. Rows 0..15
#                                        of the result are byte-identical; rows 16..31 (padding) hold stale partial
#                                        sums, so the persistent buffer is zero-filled at construction (no NaN/Inf
#                                        bit patterns from an uninitialised buffer ever reach the padding rows).
#                                        Inputs with more rows / fp32 keep the full-tile path inside the op.
#   FUSE_RESIDUAL=1 + LIVE_ROWS=1        untested together: an all_reduce that fuses an armed residual keeps the
#                                        full-tile path (no live_rows_packing) unless
#                                        QWEN36_DECODE_AR_FUSE_LIVE_ROWS=1 (default 0; for the both-on op test only).
#   QWEN36_DECODE_AR_ONESHOT=0 (default) all_reduce_async broadcasts every device's whole partial to every peer.
#   QWEN36_DECODE_AR_ONESHOT=1           every bf16 all-reduce passes one_shot=True: a reduce-scatter + all-gather in
#                                        one op (each device unicasts each peer's 1/tp slice of its partial straight
#                                        into that peer's buffer, the slice owner reduces it with the default kernel
#                                        and slot order, then multicasts the reduced slice to every peer). The result
#                                        is byte-identical; the busiest fabric link carries ~2.5x fewer packets. Needs
#                                        a 1D fabric and tp dividing the 40 RES40 cores; takes precedence over
#                                        LIVE_ROWS (no live-rows packing on a one-shot all-reduce). fp32 GDN partials
#                                        (QWEN36_DECODE_AR_GDN_FP32=1) keep the default op.
#
# Prefill never reads these flags: it keeps the fractured residual and produces only KV / GDN state,
# so the residual format is private to each forward. Dispatch inside the layer / final norm is on
# the tensor itself (last dim == dim -> replicated), so a fractured input always takes the old path.
# ---------------------------------------------------------------------------------------------
DECODE_AR_ENV = "QWEN36_DECODE_ALLREDUCE"
DECODE_AR_GDN_FP32_ENV = "QWEN36_DECODE_AR_GDN_FP32"
DECODE_AR_FUSE_ENV = "QWEN36_DECODE_AR_FUSE_RESIDUAL"
DECODE_AR_FUSE_LIVE_ROWS_ENV = "QWEN36_DECODE_AR_FUSE_LIVE_ROWS"
DECODE_AR_LIVE_ROWS_ENV = "QWEN36_DECODE_AR_LIVE_ROWS"
DECODE_AR_ONESHOT_ENV = "QWEN36_DECODE_AR_ONESHOT"
DECODE_AR_LIVE_ROWS_MAX = TILE_SIZE // 2  # rows that fit faces 0/1 of a 32x32 tile (the op's own eligibility bound)


def decode_ar_live_rows_enabled():
    """QWEN36_DECODE_AR_LIVE_ROWS (default "0" = off): <= 16-row decode all-reduces send only each tile's live half."""
    return os.environ.get(DECODE_AR_LIVE_ROWS_ENV, "0") == "1"


_DAR_GRID = (10, 4)  # RES40 core grid (x, y)
_DAR_UNSET = object()


class DecodeResidualAllReduce:
    """Replicated 40-core L1 decode residual + its norm / all-reduce / entry-gather helpers.

    One instance per TT_CCL (see get_decode_allreduce), shared by every layer and the model so the
    persistent all-reduce buffer is allocated exactly once, at construction (never inside a trace).
    """

    # QWEN36_DECODE_AR_ONESHOT, set per instance in __init__; the class default keeps instances built without
    # __init__ (host-test fakes) on the default all-reduce.
    one_shot = False

    def __init__(
        self,
        mesh_device,
        args,
        tt_ccl,
        gdn_fp32=False,
        fuse_residual=False,
        live_rows=False,
        fuse_live_rows=False,
        one_shot=False,
    ):
        self.mesh = mesh_device
        self.tt_ccl = tt_ccl
        self.dim = args.dim
        self.tp = list(mesh_device.shape)[1]
        self.topology = args.ccl_topology()
        # Same link count the fractured path's AG/RS use on this mesh (tt_ccl.get_num_links(axis)).
        self.num_links = tt_ccl.get_num_links(1)
        self.gdn_fp32 = gdn_fp32
        # QWEN36_DECODE_AR_FUSE_RESIDUAL: residual armed by the layer for the next all_reduce, and the output of the
        # all_reduce that consumed it (add_residual returns that output instead of adding again).
        self.fuse_residual = fuse_residual
        self._armed = None
        self._fused_out = None
        self.live_rows = live_rows
        # Both-on guard (QWEN36_DECODE_AR_FUSE_RESIDUAL x QWEN36_DECODE_AR_LIVE_ROWS): until the combined op passes
        # its byte-identity test, an all_reduce that fuses a residual does not also pack live rows, unless
        # fuse_live_rows (QWEN36_DECODE_AR_FUSE_LIVE_ROWS=1) opts in.
        self.fuse_live_rows = bool(fuse_live_rows and fuse_residual and live_rows)
        if fuse_residual and live_rows and not self.fuse_live_rows:
            logger.warning(
                f"{DECODE_AR_FUSE_ENV}=1 with {DECODE_AR_LIVE_ROWS_ENV}=1: residual-fused all-reduces keep full tiles "
                f"(the combination is untested; {DECODE_AR_FUSE_LIVE_ROWS_ENV}=1 enables it)"
            )

        ncores = _DAR_GRID[0] * _DAR_GRID[1]
        # QWEN36_DECODE_AR_ONESHOT: the op splits the RES40 cores into tp equal slices (one owner device each).
        self.one_shot = bool(one_shot)
        if self.one_shot and ncores % self.tp:
            logger.warning(f"{DECODE_AR_ONESHOT_ENV}=1 ignored: tp={self.tp} does not divide the {ncores} RES40 cores")
            self.one_shot = False
        grid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(_DAR_GRID[0] - 1, _DAR_GRID[1] - 1))}
        )

        def width_mem(width):
            return ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(grid, [TILE_SIZE, width], ttnn.ShardOrientation.ROW_MAJOR),
            )

        shard_w = self.dim // ncores
        self.res_mem = width_mem(shard_w)  # RES40: [1,1,B<=32,dim] replicated residual
        block_w = shard_w // TILE_SIZE
        self.norm_pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=_DAR_GRID,
            subblock_w=min(4, block_w),
            block_h=1,
            block_w=block_w,
            inplace=False,
        )
        # all_reduce_async stages every device's partial in this buffer: [32, dim*tp] over the same 40
        # cores (shard [32, dim*tp/40]); it must contain the output grid and be tp x the output shard.
        buf_mem = width_mem(self.dim * self.tp // ncores)

        def alloc(dtype):
            if self.live_rows:
                # Live-rows packing never rewrites rows 16..31 of the remote slots: start them at zero, not at whatever
                # an uninitialised L1 buffer holds, so the reduction's padding rows stay finite.
                return ttnn.from_torch(
                    torch.zeros([1, 1, TILE_SIZE, self.dim * self.tp]),
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh_device,
                    memory_config=buf_mem,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                )
            return ttnn.empty(
                [1, 1, TILE_SIZE, self.dim * self.tp],
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=buf_mem,
            )

        self.buffer = alloc(ttnn.bfloat16)
        self.buffer_fp32 = alloc(ttnn.float32) if gdn_fp32 else None

    def enter(self, x):
        """Model entry: hidden-fractured embedding [1,1,B,dim/tp] -> replicated [1,1,B,dim] in RES40."""
        if x.shape[-2] > TILE_SIZE:
            raise ValueError(f"QWEN36_DECODE_ALLREDUCE supports decode batch <= {TILE_SIZE}, got {x.shape[-2]}")
        gathered = ttnn.experimental.all_gather_async(
            x,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
            num_links=self.num_links,
            topology=self.topology,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )
        out = ttnn.to_memory_config(gathered, self.res_mem)
        if out is not gathered:
            ttnn.deallocate(gathered)
        return out

    def rms_norm(self, x, norm):
        """Local sharded rms_norm on a RES40 input with the replicated full-width (1+w) gamma of the
        framework `RMSNorm` `norm` (same eps / HiFi2 + fp32-acc kernel config as the fractured path)."""
        return ttnn.rms_norm(
            x,
            epsilon=norm.eps,
            weight=norm.weight,
            program_config=self.norm_pc,
            memory_config=self.res_mem,
            compute_kernel_config=norm.compute_kernel_config_hifi2,
        )

    def final_norm(self, x, norm, output_memory_config=ttnn.DRAM_MEMORY_CONFIG):
        """Final norm before the LM head: local sharded norm, then the layout the LM head expects."""
        normed = self.rms_norm(x, norm)
        out = ttnn.to_memory_config(normed, output_memory_config)
        if out is not normed:
            ttnn.deallocate(normed)
        return out

    def add(self, a, b):
        """Residual add; both operands (and the result) live in RES40."""
        return ttnn.add(a, b, memory_config=self.res_mem)

    def arm_residual(self, x):
        """QWEN36_DECODE_AR_FUSE_RESIDUAL=1: the next all_reduce adds `x` (the RES40 residual) in its reduction
        kernel. No-op with the flag off. Pair with add_residual(x, out) on the module's output."""
        self._armed = x if self.fuse_residual else None
        self._fused_out = None

    def disarm(self):
        """Drop any armed residual and fused output (a forward that returns before its add_residual, e.g. the
        MTP kv_only path). No-op when nothing is armed, so always safe with QWEN36_DECODE_AR_FUSE_RESIDUAL=0."""
        self._armed = None
        self._fused_out = None

    def add_residual(self, x, out):
        """Residual add after an armed module: `out` itself when the all_reduce that produced it already added `x`
        (fused), else add(x, out). Disarms. Callers must not deallocate `out` when the result `is out`."""
        fused, self._fused_out, self._armed = self._fused_out, None, None
        if fused is None:
            return self.add(x, out)
        if out is not fused:
            raise RuntimeError(
                f"{DECODE_AR_FUSE_ENV}=1: the module output is not the residual-fused all_reduce output "
                "(an op ran on it after the all-reduce, or the module all-reduced twice); set the flag to 0"
            )
        return out

    def _take_armed(self, packed, dtype):
        """The armed residual if it can be fused into this all_reduce (same dtype / shape / RES40 layout), else
        None. Disarms either way: one armed residual feeds at most one all_reduce."""
        res, self._armed = self._armed, None
        if res is None:
            return None
        if (
            dtype != ttnn.bfloat16
            or res.dtype != dtype
            or list(res.shape) != list(packed.shape)
            or res.memory_config() != self.res_mem
        ):
            return None
        return res

    def all_reduce(self, partial, keep_fp32=False):
        """Sum the per-device row-parallel partial [1,1,B,dim] across the TP axis -> replicated RES40.

        keep_fp32: the caller's partial is fp32 (GDN out-proj); honoured only when QWEN36_DECODE_AR_GDN_FP32=1,
        otherwise it is cast to bf16 (the persistent buffer / CCL dtype) after the reshard to RES40.
        QWEN36_DECODE_AR_FUSE_RESIDUAL=1 with a residual armed (arm_residual): returns residual + sum, computed in
        the reduction kernel; add_residual then hands this tensor back instead of adding again.
        """
        use_fp32 = keep_fp32 and self.gdn_fp32
        dtype = ttnn.float32 if use_fp32 else ttnn.bfloat16
        shape = list(partial.shape)
        if shape[0] != 1 or shape[1] != 1:
            partial = ttnn.reshape(partial, (1, 1, shape[-4] * shape[-3] * shape[-2], shape[-1]))
        packed = ttnn.to_memory_config(partial, self.res_mem)
        if packed is not partial:
            ttnn.deallocate(partial)
        if packed.dtype != dtype:
            cast = ttnn.typecast(packed, dtype)
            ttnn.deallocate(packed)
            packed = cast
        extra = dict(fp32_dest_acc=True) if use_fp32 else {}
        residual = self._take_armed(packed, dtype)
        if residual is not None:
            extra["residual_tensor"] = residual
        if self.one_shot and dtype == ttnn.bfloat16:
            extra["one_shot"] = True
        elif self.live_rows and self.live_rows_eligible(packed) and (residual is None or self.fuse_live_rows):
            extra["live_rows_packing"] = True
        out = ttnn.experimental.all_reduce_async(
            packed,
            self.buffer_fp32 if use_fp32 else self.buffer,
            cluster_axis=1,
            mesh_device=self.mesh,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(1),
            topology=self.topology,
            num_links=self.num_links,
            memory_config=self.res_mem,
            **extra,
        )
        ttnn.deallocate(packed)
        if residual is not None:
            self._fused_out = out
        return out

    @staticmethod
    def live_rows_eligible(packed):
        """True when the op's live-rows packing can engage on this (RES40-packed, cast) partial: bf16 and at most
        DECODE_AR_LIVE_ROWS_MAX logical rows. Passing the flag on an ineligible input is harmless (the op falls back);
        filtering here only keeps the full-row calls' arguments, and so their program-cache entries, unchanged."""
        return packed.dtype == ttnn.bfloat16 and packed.shape[-2] <= DECODE_AR_LIVE_ROWS_MAX


def get_decode_allreduce(args, tt_ccl):
    """Shared DecodeResidualAllReduce for this TT_CCL (default on), or None when QWEN36_DECODE_ALLREDUCE=0 or the
    configuration is unsupported (those keep the fractured-residual decode path unchanged).

    Created on first call (layer / model construction) and cached on `tt_ccl`; every later caller gets the
    same instance, so layers and the model always agree on the decode residual format.
    """
    if tt_ccl is None:
        return None
    cached = getattr(tt_ccl, "_qwen36_decode_ar", _DAR_UNSET)
    if cached is not _DAR_UNSET:
        return cached
    inst = None
    if os.environ.get(DECODE_AR_ENV, "1") == "1":
        mesh = tt_ccl.mesh_device
        shape = list(mesh.shape)
        grid = mesh.compute_with_storage_grid_size()
        ncores = _DAR_GRID[0] * _DAR_GRID[1]
        reason = None
        if len(shape) != 2 or shape[0] != 1 or shape[1] < 2 or shape[1] % 2:
            reason = f"needs a (1, even N>=2) mesh, got {shape}"  # all_reduce_async: cluster_axis=1, even ring
        elif args.dim % (ncores * TILE_SIZE):
            reason = f"hidden dim {args.dim} is not tile-shardable over {ncores} cores"
        elif getattr(args, "tile_padded_batch_rows", TILE_SIZE) != TILE_SIZE:
            reason = f"max_batch_size {args.max_batch_size} > {TILE_SIZE} (RES40 shard height is one tile)"
        elif getattr(args, "moe_num_experts", 0) > 0:
            reason = "MoE layers' decode MLP still reduce-scatters into the fractured residual"
        elif grid.x < _DAR_GRID[0] or grid.y < _DAR_GRID[1]:
            reason = f"compute grid {grid.x}x{grid.y} is smaller than {_DAR_GRID[0]}x{_DAR_GRID[1]}"
        if reason is not None:
            logger.warning(f"{DECODE_AR_ENV}=1 ignored ({reason}); keeping the fractured-residual decode path")
        else:
            inst = DecodeResidualAllReduce(
                mesh,
                args,
                tt_ccl,
                gdn_fp32=os.environ.get(DECODE_AR_GDN_FP32_ENV, "0") == "1",
                fuse_residual=os.environ.get(DECODE_AR_FUSE_ENV, "0") == "1",
                live_rows=decode_ar_live_rows_enabled(),
                fuse_live_rows=os.environ.get(DECODE_AR_FUSE_LIVE_ROWS_ENV, "0") == "1",
                one_shot=os.environ.get(DECODE_AR_ONESHOT_ENV, "0") == "1",
            )
            logger.info(
                f"{DECODE_AR_ENV}=1: decode uses the replicated 40-core L1 residual + all_reduce_async "
                f"(gdn_fp32={inst.gdn_fp32}, fuse_residual={inst.fuse_residual}, num_links={inst.num_links}, "
                f"topology={inst.topology}, live_rows={inst.live_rows})"
                + (f" one_shot={inst.one_shot}" if inst.one_shot else "")
            )
    tt_ccl._qwen36_decode_ar = inst
    return inst


def sharded_decode_matmul(
    x,
    weight,
    compute_cfg,
    decode_progcfg,
    act_shard_cfg,
    prefill_progcfg_fn,
    prefill_k,
    decode_out_memory_config=ttnn.DRAM_MEMORY_CONFIG,
):
    """DRAM-WIDTH_SHARDED weight matmul; branches on M (decode vs prefill).

    Decode (M<=32): L1-sharded act + DRAM-sharded kernel. Prefill: 2D matmul.
    Gate on x.shape[-2] (seq/M), not x.shape[1] (Z=1 in both modes). Decode result placement is
    `decode_out_memory_config` (default DRAM-interleaved; pass L1 to keep the small decode
    activation resident). Prefill result is always DRAM-interleaved."""
    seq = x.shape[-2]
    if seq <= TILE_SIZE:
        # Reshard act to L1 if needed; skip dealloc when x already sharded (GDN reuses x).
        already_sharded = x.memory_config() == act_shard_cfg
        x_sh = x if already_sharded else ttnn.to_memory_config(x, act_shard_cfg)
        out = ttnn.linear(
            x_sh,
            weight,
            compute_kernel_config=compute_cfg,
            program_config=decode_progcfg,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        if not already_sharded:
            ttnn.deallocate(x_sh)
        return ttnn.to_memory_config(out, decode_out_memory_config)
    pc = prefill_progcfg_fn(seq, prefill_k, weight.shape[-1])
    return ttnn.linear(
        x, weight, compute_kernel_config=compute_cfg, program_config=pc, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def replicate(torch_tensor, mesh, cache_path, dtype=ttnn.bfloat16):
    """Small tensor (norm/bias) -> replicated on every device."""
    if torch_tensor.dim() == 1:
        torch_tensor = torch_tensor.unsqueeze(0).unsqueeze(0)
    elif torch_tensor.dim() == 2:
        torch_tensor = torch_tensor.unsqueeze(0)
    return ttnn.as_tensor(
        torch_tensor.to(torch.bfloat16),
        dtype=dtype,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=cache_path,
    )


def shard_small(torch_tensor, mesh, cache_path, dim=-1, dtype=ttnn.bfloat16):
    """Small per-head tensor (conv taps, A_log, dt_bias) -> sharded."""
    if torch_tensor.dim() == 1:
        torch_tensor = torch_tensor.unsqueeze(0).unsqueeze(0)
    elif torch_tensor.dim() == 2:
        torch_tensor = torch_tensor.unsqueeze(0)
    return ttnn.as_tensor(
        torch_tensor.to(torch.bfloat16),
        dtype=dtype,
        device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=dim),
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=cache_path,
    )


def replicate_kv_weight(weight, n_kv_heads, tp, head_dim):
    """Replicate KV weight so each device gets >=1 head. No-op when tp <= n_kv_heads."""
    if tp <= n_kv_heads:
        return weight
    chunks = weight.reshape(n_kv_heads, head_dim, -1)
    parts = []
    for d in range(tp):
        kv_idx = (d * n_kv_heads) // tp
        parts.append(chunks[kv_idx])
    return torch.cat(parts, dim=0).reshape(tp * head_dim, -1)


# FP8 dequantization
def dequant_fp8_block(weight_fp8, scale_inv, block_size=128):
    """Dequantize a block-wise FP8 weight tensor to bfloat16."""
    out_f, in_f = weight_fp8.shape
    weight_bf16 = weight_fp8.to(torch.bfloat16).reshape(out_f // block_size, block_size, in_f // block_size, block_size)
    weight_bf16 = weight_bf16 * scale_inv[:, None, :, None].to(torch.bfloat16)
    return weight_bf16.reshape(out_f, in_f)


# Weight-prep (reorder HF weights for per-device sharding)
def prepare_attn_qkv(q_w, k_w, v_w, qg_per, kv_per, tp):
    """Fuse attn q+gate/k/v for column-parallel shard: each device gets [qg_d|k_d|v_d].

    q_w: [n_heads*head_dim*2, in]; k_w/v_w: [n_kv_heads*head_dim, in].
    qg_per/kv_per: per-device out block sizes."""
    parts = []
    for d in range(tp):
        parts.append(q_w[d * qg_per : (d + 1) * qg_per, :])
        parts.append(k_w[d * kv_per : (d + 1) * kv_per, :])
        parts.append(v_w[d * kv_per : (d + 1) * kv_per, :])
    return torch.cat(parts, dim=0)


def prepare_attn_qkv_deint(q_w, k_w, v_w, nh_local, hd, kv_per, tp):
    """Like prepare_attn_qkv but de-interleaves [q,g] per head -> [all_q|all_gate|k|v] per device.

    Avoids prefill relayout in _make_heads (column perm only; numerically identical).
    q_w: [nh_total*hd*2, in]; nh_local/kv_per: per-device block sizes."""
    hd2 = hd * 2
    parts = []
    for d in range(tp):
        base = d * nh_local * hd2
        q_rows = [q_w[base + h * hd2 : base + h * hd2 + hd, :] for h in range(nh_local)]
        g_rows = [q_w[base + h * hd2 + hd : base + h * hd2 + hd2, :] for h in range(nh_local)]
        # Per-device layout [all_q | k | v | all_gate]: q/k/v contiguous so _make_heads* can hand
        # the fused q|k|v block straight to nlp_create_qkv_heads (no re-concat); gate trails, applied
        # post-SDPA. (Column perm only; numerically identical to [q|gate|k|v].)
        parts.append(torch.cat(q_rows, dim=0))  # all_q
        parts.append(k_w[d * kv_per : (d + 1) * kv_per, :])
        parts.append(v_w[d * kv_per : (d + 1) * kv_per, :])
        parts.append(torch.cat(g_rows, dim=0))  # all_gate (last)
    return torch.cat(parts, dim=0)


def prepare_attn_qkvg_pair(q_w, k_w, v_w, nh_local, hd, kv_per, tp):
    """QWEN36_ATTN_KV_SPLIT decode weight: per device [q of the PAIR's heads | k | v | gate of the pair's heads].

    Device d = 2p + r of KV pair p gets the q / gate rows of global heads 2p*nh_local .. (2p+2)*nh_local - 1 (device 2p's
    heads, then 2p+1's: the pair-global order, the same on both members) and its own k / v rows (the replicated KV
    head p). Same per-head [q, gate] de-interleave as prepare_attn_qkv_deint. q_w: [nh_total*hd*2, in] (interleaved
    [q, gate] per head); k_w / v_w: the KV-replicated [tp*kv_per, in] weights. Returns [tp*(4*nh_local*hd + 2*kv_per),
    in] for shard_w(dim=-1)."""
    assert tp % 2 == 0, f"KV-pair split needs an even TP, got {tp}"
    hd2 = hd * 2
    parts = []
    for d in range(tp):
        heads = range((d // 2) * 2 * nh_local, (d // 2 + 1) * 2 * nh_local)
        parts.append(torch.cat([q_w[g * hd2 : g * hd2 + hd, :] for g in heads], dim=0))
        parts.append(k_w[d * kv_per : (d + 1) * kv_per, :])
        parts.append(v_w[d * kv_per : (d + 1) * kv_per, :])
        parts.append(torch.cat([q_w[g * hd2 + hd : (g + 1) * hd2, :] for g in heads], dim=0))
    return torch.cat(parts, dim=0)


def prepare_attn_wo_pair(o_w, nh_local, hd, tp):
    """QWEN36_ATTN_KV_SPLIT decode o-projection: device d of pair p gets the wo input rows of the pair's 2*nh_local heads
    (pair-global order). o_w: torch o_proj.weight [dim, nh_total*hd]. Returns [dim, tp*2*nh_local*hd] for
    shard_w(dim=0) (which transposes and gives device d rows [d*2W, (d+1)*2W), W = nh_local*hd)."""
    assert tp % 2 == 0, f"KV-pair split needs an even TP, got {tp}"
    W2 = 2 * nh_local * hd
    return torch.cat([o_w[:, (d // 2) * W2 : (d // 2 + 1) * W2] for d in range(tp)], dim=1)


# QWEN36_ATTN_KV_SPLIT one-hot row selections (q rows of the own users, attention rows back to their global row): HiFi4
# with fp32 accumulation, so 1.0 * x lands on x exactly (every output element has a single non-zero product).
COMPUTE_SELECT = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)


def prepare_gdn_qkv(qkv_w, key_dim, value_dim, nk, dk, nv, dv, tp):
    """Interleave GDN Q/K/V heads for row-parallel shard (contiguous q/k/v block per device).

    qkv_w: [key_dim*2 + value_dim, hidden]."""
    q_part = qkv_w[:key_dim, :]
    k_part = qkv_w[key_dim : 2 * key_dim, :]
    v_part = qkv_w[2 * key_dim :, :]

    q_per = nk // tp
    v_per = nv // tp
    shards = []
    for s in range(tp):
        q_s = q_part[s * q_per * dk : (s + 1) * q_per * dk, :]
        k_s = k_part[s * q_per * dk : (s + 1) * q_per * dk, :]
        v_s = v_part[s * v_per * dv : (s + 1) * v_per * dv, :]
        shards.append(torch.cat([q_s, k_s, v_s], dim=0))
    return torch.cat(shards, dim=0)


def prepare_conv_taps(conv_w, key_dim, nk, dk, nv, dv, kernel_size, tp):
    """Split fused conv1d into kernel taps, reordered to match prepare_gdn_qkv grouping."""
    cw = conv_w.float()
    q_per = nk // tp
    v_per = nv // tp
    taps = []
    for j in range(kernel_size):
        tap = cw[:, 0, j]
        q_tap = tap[:key_dim]
        k_tap = tap[key_dim : 2 * key_dim]
        v_tap = tap[2 * key_dim :]
        shards = []
        for s in range(tp):
            q_s = q_tap[s * q_per * dk : (s + 1) * q_per * dk]
            k_s = k_tap[s * q_per * dk : (s + 1) * q_per * dk]
            v_s = v_tap[s * v_per * dv : (s + 1) * v_per * dv]
            shards.append(torch.cat([q_s, k_s, v_s]))
        taps.append(torch.cat(shards))
    return taps
