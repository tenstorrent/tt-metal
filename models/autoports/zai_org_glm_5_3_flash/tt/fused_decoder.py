# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused TTNN decoder layer for zai-org/GLM-5.3-Flash (`glm5_next`), single 1x1 Blackhole mesh.

Same public contract as `functional_decoder.FunctionalDecoder` (prefill/decode semantics, chunk planner, paged MLA
latent cache, persistent per-slot KDA state, trace capture), with the op graph rewritten onto dedicated fused ops
(defaults = `FusedDecoderConfig`, chosen by whole-step A/B; doc/fused_decoder/README.md):

    mHC decode          up to 4 rows: ttnn.experimental.deepseek.fused_hyperconnection + mix_streams (fp32) over the
                        stacked [T, 1, 4, 4096] layout (fn projection from the stacked streams up to 2 rows, 16-way
                        K-split above); from 5 rows: the stream-major [1, 4, T, 4096] composite (mhc_split_sinkhorn,
                        broadcast collapse / mix). Prefill keeps Stage 1's composite mHC.
    KDA decode          ttnn.experimental.kda.fused_recurrent_kda (in-place state, decode-row layout) over ONE
                        persistent [slots, 64, 128, 128] fp32 state, users-as-rows conv history [1, 3, slots, 24576]
                        bf16, merged gate multiplies, sigmoid_gated_rms_norm from 2 rows.
    KDA prefill         Stage 1's exact chunked path (qkv_causal_conv1d_silu, prepare_chunk_recurrence, fp32 carry,
                        sigmoid_gated_rms_norm) reading / writing the slot through the batched state (fill_cache).
    MoE                 ttnn.experimental.deepseek.moe.fused_experts (gather -> clamped SwiGLU -> down -> weighted
                        accumulate, union-of-selections weight streaming, runtime hit count) for decode AND prefill,
                        fed by the fp32 sigmoid router + bias top-8 (uint16 ids, unbiased bf16 scores); shared expert /
                        dense MLP with ttnn.clamped_silu_glu.
    MLA                 nlp_create_qkv_heads head split, full-grid per-head absorption matmuls, nlp_concat_heads (two
                        32-head groups) head concat, one 64-head paged flash-MLA decode call from 16 rows (K chunk 128
                        up to 2 rows).
    Norms               width-sharded multi-core rms_norm for decode rows.

Residual stream contract (unchanged): `[1, 1, T, 16384]` float32, stream i in columns [4096 i, 4096 (i + 1)).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt.config import LayerSpec, round_up
from models.autoports.zai_org_glm_5_3_flash.tt.functional_decoder import (
    BLOCK_SIZE,
    PREFILL_CHUNK,
    DecoderCache,
    FunctionalDecoder,
    bucket_for,
    plan_prefill_chunks,
)
from models.autoports.zai_org_glm_5_3_flash.tt.hyperconnection import HyperConnection
from models.autoports.zai_org_glm_5_3_flash.tt.kda import CHUNK, CONV_K, KDA, chunk_live_masks, keep_if_dead
from models.autoports.zai_org_glm_5_3_flash.tt.mla import MLA
from models.autoports.zai_org_glm_5_3_flash.tt.mlp import SwiGLUMLP
from models.autoports.zai_org_glm_5_3_flash.tt.moe import PADDED_EXPERTS
from models.autoports.zai_org_glm_5_3_flash.tt.precision import ACT_DTYPE, KDA_STATE_DTYPE, WEIGHT_DTYPES, hifi4_fp32
from models.autoports.zai_org_glm_5_3_flash.tt.weights import TileCacheSource, TorchStateDictSource, WeightSource

DRAM = ttnn.DRAM_MEMORY_CONFIG
TILE = 32
FP32 = ttnn.float32
BF16 = ttnn.bfloat16

__all__ = [
    "FusedDecoder",
    "FusedDecoderConfig",
    "plan_prefill_chunks",
    "bucket_for",
    "PREFILL_CHUNK",
]


@dataclass
class FusedDecoderConfig:
    """Fusion knobs (defaults = the measured best; see doc/fused_decoder/README.md)."""

    # fused_experts: tokens per call and the static expert bound per call (min(E, top_k * micro_batch)); the bound
    # keeps every call one program (trace-safe) and always covers the union of the call's selections, and the op
    # loops over the real union only (runtime hit count), so one call per 32 tokens is the fastest
    moe_decode_micro_batch: int = 32
    moe_prefill_micro_batch: int = 32
    experts_block_size: int = 1  # decode calls
    experts_block_size_prefill: int = 1  # prefill calls (32-token micro-batches)
    # mHC implementation per mode: "fused" (fused_hyperconnection + mix_streams on the stacked [T,1,4,4096] layout),
    # "stream_major" (StreamMajorHyperConnection on [1,4,T,4096]), "composite" (Stage 1 HyperConnection)
    mhc_decode: str = "auto"  # auto: "fused" up to mhc_auto_fused_max_rows decode rows, else "stream_major"
    mhc_prefill: str = "composite"
    mhc_auto_fused_max_rows: int = 4
    # fused-mHC fn projection: "ksplit" (from the flat streams, batched matmul over mhc_ksplit K blocks +
    # fast_reduce_nc, T <= 32), "single" (one flat matmul), "stacked" (from the stacked [T,1,4,4096] streams, no flat
    # copy), "rmsnorm" (rms_norm op, then one matmul); "auto": "stacked" up to mhc_stacked_max_rows rows, else "ksplit"
    mhc_proj: str = "auto"
    mhc_stacked_max_rows: int = 2
    mhc_ksplit: int = 16
    # stream-major (B > 4) fn projection: cores for the 4 per-stream products (0 = default program)
    mhc_sm_proj_cores: int = 4
    # router: sigmoid folded into the logits matmul (matmul fused activation); top-k width (320 = Stage 1 padding)
    router_sigmoid_in_linear: bool = False
    router_topk_width: int = 320
    # router selection: "topk" (sigmoid, + bias, ttnn.topk, id typecast) | "grouped_topk" (moe_grouped_topk
    # single-group: sigmoid + bias + fp32 top-8 in one op, uint16 ids; unbiased bf16 scores from one sigmoid of the same
    # logits)
    router_impl: str = "topk"
    # with router_impl="topk": use the grouped_topk selection up to this many rows (0 = never)
    router_grouped_max_rows: int = 0
    # router logits from the fp32 post-attention RMSNorm output (the experts / shared expert keep the bf16 copy) instead
    # of the bf16-rounded one: one rounding step fewer before the near-tie-sensitive top-8 selection
    router_fp32_input: bool = False
    # decode-row RMSNorms (input / post-attention): width-sharded multi-core rms_norm instead of one core per tile row
    norm_sharded: bool = True
    # MLA decode attention: one 64-head paged flash-MLA call from this many decode rows (it fits L1 from B = 16; below,
    # two 32-head calls), and its K chunk
    mla_single_call_min_batch: int = 16
    mla_decode_k_chunk: int = 32
    mla_decode_k_chunk_small: int = 128  # K chunk up to mla_decode_small_max_rows rows (0 = mla_decode_k_chunk)
    mla_decode_small_max_rows: int = 2
    # MLA output head concat [1,64,T,256] -> [1,1,T,16384]: "permute" (permute + reshape) | "nlp2" (nlp_concat_heads on
    # two 32-head groups + concat; one 64-head call overflows L1)
    mla_concat: str = "nlp2"
    # shared expert / dense MLP gate|up: "slice" (one matmul + 2 slices) | "split_mm" (minimal_matmul_split, 2 outputs)
    swiglu_impl: str = "slice"
    # KDA decode conv: reduction of the 3 weighted history rows over dim 1 ("sum" | "fast_reduce_nc")
    kda_conv_reduce: str = "fast_reduce_nc"
    # fused_recurrent_kda: final state written in place into the persistent state (no copy-back), and the
    # [1,1,B,H*D] / [1,1,B,H] decode layout for q / beta (no user-split relayout)
    kda_inplace_state: bool = True
    kda_decode_rows: bool = True
    # decode gate chain: sigmoid folded into the multiply's input activation, the -5 bound and the inactive-slot mask
    # merged into one multiplier; beta = one sigmoid-activated (mask) multiply straight from the bf16 projection
    kda_merge_gates: bool = True
    # f_b projection with dt_bias as the linear's bias (matmul + bias -> linear) instead of a separate fp32 add
    kda_fb_bias_in_linear: bool = False
    # decode gated RMSNorm via the dedicated sigmoid_gated_rms_norm op (users as its tile-aligned sequence, padded to
    # 32) from kda_gated_norm_op_min_rows decode rows (below: rms_norm over the [B*64, 128] view + sigmoid-gated
    # multiply)
    kda_gated_norm_op: bool = True
    kda_gated_norm_op_min_rows: int = 2
    # decode conv as one 4-row window: concat(history, u) -> one tap multiply -> fast_reduce_nc -> SiLU
    kda_conv_window: bool = False
    # prefill chunk scan: "stage1" (Stage 1's loop: 5 matmuls per 32-token chunk) | "hoisted" (state-independent
    # products batched over all chunks, 2 matmuls per chunk; same prepare_chunk_recurrence terms and fp32 carry;
    # -8 ms per 2048-token dense chunk, same aggregate PCC, but it moves the L4/4096 position-3245 routing near-tie of
    # the prefill path so Stage 1's per-user batched-decode gate fails: kept off, see the README)
    kda_prefill_scan: str = "stage1"
    # derived gate_up cache (tile-permuted [gate_c | up_c] per core, the fused_experts weight contract)
    fused_experts_cache_dir: str = field(
        default_factory=lambda: os.environ.get(
            "GLM53_FUSED_EXPERTS_CACHE", "/home/alex/glm53/tt-cache-fused/glm53-flash"
        )
    )


# ======================================================================================================================
# fused_experts weights: per-expert DRAM ND-sharded tensors (one [rows, 64] shard per compute core, round robin over the
# 8 DRAM banks); gate_up tile columns permuted [gate | up] -> per-core [gate_c | up_c].
FE_CORES = 64
DRAM_BANKS = 8


def _nd_dram(rows: int, shard_cols: int) -> ttnn.MemoryConfig:
    banks = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(b, 0), ttnn.CoreCoord(b, 0)) for b in range(DRAM_BANKS)])
    spec = ttnn.NdShardSpec(
        shard_shape=[rows, shard_cols],
        grid=banks,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    return ttnn.MemoryConfig(ttnn.BufferType.DRAM, spec)


def gate_up_tile_perm(n_tile_cols: int):
    """[gate | up] tile columns -> [gate_0 | up_0 | gate_1 | up_1 | ...] (column tile order)."""
    half = n_tile_cols // 2
    perm = [0] * n_tile_cols
    for c in range(half):
        perm[2 * c] = c
        perm[2 * c + 1] = half + c
    return perm


def _permuted_gate_up_file(src: str, dst: str, shape, tile_bytes: int) -> None:
    """Byte-exact tile permutation of a bfloat4_b `.tensorbin` (header kept, 576-B tiles reordered per tile row)."""
    import numpy as np

    with open(src, "rb") as f:
        raw = f.read()
    hdr = int(np.frombuffer(raw[:8], dtype="<u8")[0])
    head, data = raw[: 8 + hdr], np.frombuffer(raw[8 + hdr :], dtype=np.uint8)
    kt, nt = shape[0] // TILE, shape[1] // TILE
    assert data.size == kt * nt * tile_bytes, (src, data.size, kt * nt * tile_bytes)
    out = data.reshape(kt, nt, tile_bytes)[:, gate_up_tile_perm(nt), :]
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = f"{dst}.tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(head)
        f.write(np.ascontiguousarray(out).tobytes())
    os.replace(tmp, dst)


def gate_up_column_perm(two_i: int) -> torch.Tensor:
    """Element-column permutation of [gate | up] (width 2I) into the fused_experts per-core [gate_c | up_c] tile
    order."""
    return torch.tensor(gate_up_tile_perm(two_i // TILE)).repeat_interleave(TILE) * TILE + torch.arange(TILE).repeat(
        two_i // TILE
    )


def load_fused_experts(source: WeightSource, layer_idx: int, num_experts: int, mesh_device, cache_dir: str):
    """(gate_up list, down list) in the fused_experts weight contract, bfloat4_b, from any weight source. The tile
    cache is permuted byte-exactly into a derived file per expert; host sources (state dict / synthetic) are permuted
    in torch and packed once."""
    if not isinstance(source, TileCacheSource):
        if isinstance(source, TorchStateDictSource):
            gu_all = source.host(layer_idx, "mlp.experts.gate_up_proj")  # [E, H, 2I] = [gate | up] columns
            dn_all = source.host(layer_idx, "mlp.experts.down_proj")  # [E, I, H]
            get = lambda e: (gu_all[e], dn_all[e])
        else:  # per-expert host tensors in the cache layout (e.g. the synthetic-weight test source)
            get = lambda e: (
                source.host(layer_idx, f"mlp.experts.{e}.gate_up_proj"),
                source.host(layer_idx, f"mlp.experts.{e}.down_proj"),
            )
        gate_up, down = [], []
        for e in range(num_experts):
            gu, dn = get(e)
            if e == 0:
                cols = gate_up_column_perm(gu.shape[-1])
                gu_cfg = _nd_dram(gu.shape[-2], gu.shape[-1] // FE_CORES)
                dn_cfg = _nd_dram(dn.shape[-2], dn.shape[-1] // FE_CORES)
            pack = lambda w: ttnn.from_torch(
                w.to(torch.bfloat16), dtype=WEIGHT_DTYPES["routed_expert"], layout=ttnn.TILE_LAYOUT
            )
            gate_up.append(ttnn.to_device(pack(gu[:, cols]), mesh_device, memory_config=gu_cfg))
            down.append(ttnn.to_device(pack(dn), mesh_device, memory_config=dn_cfg))
        return gate_up, down
    assert isinstance(source, TileCacheSource), type(source)
    from models.autoports.zai_org_glm_5_3_flash.tt.weights import cache_dtype  # noqa: F401  (dtype map check below)

    gate_up, down = [], []
    gu_cfg = dn_cfg = None
    for e in range(num_experts):
        ge = source._entry(layer_idx, f"mlp.experts.{e}.gate_up_proj")
        de = source._entry(layer_idx, f"mlp.experts.{e}.down_proj")
        assert ge["dtype"] == "bfloat4_b" and de["dtype"] == "bfloat4_b", (ge["dtype"], de["dtype"])
        src = os.path.join(source.cache_dir, ge["path"])
        dst = os.path.join(cache_dir, f"L{layer_idx:02d}", f"{e:03d}.gate_up_proj.{ge['sha256'][:16]}.tensorbin")
        if not os.path.exists(dst):
            _permuted_gate_up_file(src, dst, ge["shape"], 576)
        if gu_cfg is None:
            gu_cfg = _nd_dram(ge["shape"][0], ge["shape"][1] // FE_CORES)
            dn_cfg = _nd_dram(de["shape"][0], de["shape"][1] // FE_CORES)
        gate_up.append(ttnn.to_device(ttnn.load_tensor(dst), mesh_device, memory_config=gu_cfg))
        down.append(
            ttnn.to_device(
                ttnn.load_tensor(os.path.join(source.cache_dir, de["path"])), mesh_device, memory_config=dn_cfg
            )
        )
    return gate_up, down


def _xsl_slice_bytes(path: str, shape, tp: int, c: int, kind: str, tile_bytes: int = 576) -> bytes:
    """Tile-exact slice c of tp of a bfloat4_b expert `.tensorbin` (Stage 7a MKX_XSL): gate_up = the PERMUTED file
    ([g0|u0|g1|u1..] tile columns), tile columns [2W c, 2W (c + 1)); down = tile rows [W c, W (c + 1)); W = I / 32 / tp.
    """
    return _xsl_slices_bytes(path, shape, tp, [c], kind, tile_bytes)[0]


def _xsl_slices_bytes(path: str, shape, tp: int, cs, kind: str, tile_bytes: int = 576) -> list:
    """the slices `cs` of one expert file from ONE read (14b 2026-10-20 19:58Z: one read per slice re-read every
    .tensorbin tp times -> 1 TB through read() for a 180-GB TP8 load)"""
    import numpy as np

    with open(path, "rb") as f:
        raw = f.read()
    hdr = int(np.frombuffer(raw[:8], dtype="<u8")[0])
    data = np.frombuffer(raw[8 + hdr :], dtype=np.uint8)
    kt, nt = shape[0] // TILE, shape[1] // TILE
    assert data.size == kt * nt * tile_bytes, (path, data.size, kt * nt * tile_bytes)
    t = data.reshape(kt, nt, tile_bytes)
    out = []
    for c in cs:
        if kind == "gate_up":
            w2 = nt // tp
            out.append(np.ascontiguousarray(t[:, w2 * c : w2 * (c + 1), :]).tobytes())
        else:
            w = kt // tp
            out.append(np.ascontiguousarray(t[w * c : w * (c + 1), :, :]).tobytes())
    return out


_XSL_HEADERS: dict = {}
_XSL_LOCK = __import__("threading").Lock()


def _xsl_host_tensor(data: bytes, shape, tag: str) -> ttnn.Tensor:
    """A host bfloat4_b TILE tensor of `shape` from its tile bytes: the `.tensorbin` header of a template tensor of that
    spec (dumped once per shape) + the bytes, loaded through /dev/shm (no cache files: 42 layers of slices = the
    experts' whole size)."""
    key = tuple(int(v) for v in shape)
    with _XSL_LOCK:
        _xsl_header(key, len(data))
    p = f"/dev/shm/xsl_{os.getpid()}_{tag}.tensorbin"  # tag: unique per (layer, expert, slice) -> thread-safe
    with open(p, "wb") as f:
        f.write(_XSL_HEADERS[key])
        f.write(data)
    t = ttnn.load_tensor(p)
    os.remove(p)
    return t


def _xsl_header(key, nbytes: int) -> None:
    import numpy as np

    if key not in _XSL_HEADERS:
        tmpl = ttnn.from_torch(torch.zeros(key, dtype=torch.bfloat16), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)
        p = f"/dev/shm/xsl_tmpl_{os.getpid()}.tensorbin"
        ttnn.dump_tensor(p, tmpl)
        with open(p, "rb") as f:
            raw = f.read()
        os.remove(p)
        hdr = int(np.frombuffer(raw[:8], dtype="<u8")[0])
        assert len(raw) - 8 - hdr == nbytes, ("template data size", key, len(raw) - 8 - hdr, nbytes)
        _XSL_HEADERS[key] = raw[: 8 + hdr]


def load_xsl_experts(source: WeightSource, layer_idx: int, num_experts: int, mesh_device, cache_dir: str, tp: int,
                     slices=None):  # fmt: skip
    """Batched expert TP weights (Stage 7a MKX_XSL, mk_experts.ReaderExperts(xsl_sliced=True)): per expert ONE mesh
    tensor per projection whose device k (row-major mesh order = the TP rank) holds slice slices[k] of tp (default
    k): gate_up [H, 2I / tp] (fused [g|u] tile columns), down [I / tp, H], ND-sharded [rows, 64] like the full tensors —
    the same bytes per chip as EP's E / tp experts, one address on every chip. Tile-exact cuts of the tile cache (the
    gate_up permutation of load_fused_experts first); slices=[c] on a 1-chip mesh = TP rank c emulated (virtual TP)."""
    assert isinstance(source, TileCacheSource), type(source)
    ndev = int(mesh_device.get_num_devices())
    slices = list(range(ndev)) if slices is None else [int(v) for v in slices]
    assert len(slices) == ndev and all(0 <= v < tp for v in slices), (slices, ndev, tp)
    from concurrent.futures import ThreadPoolExecutor

    def host(e):  # one expert's host slices (file reads + numpy cuts + host tensors: a worker thread)
        ge = source._entry(layer_idx, f"mlp.experts.{e}.gate_up_proj")
        de = source._entry(layer_idx, f"mlp.experts.{e}.down_proj")
        assert ge["dtype"] == "bfloat4_b" and de["dtype"] == "bfloat4_b", (ge["dtype"], de["dtype"])
        gsrc = os.path.join(cache_dir, f"L{layer_idx:02d}", f"{e:03d}.gate_up_proj.{ge['sha256'][:16]}.tensorbin")
        if not os.path.exists(gsrc):
            _permuted_gate_up_file(os.path.join(source.cache_dir, ge["path"]), gsrc, ge["shape"], 576)
        dsrc = os.path.join(source.cache_dir, de["path"])
        gshape = (ge["shape"][0], ge["shape"][1] // tp)
        dshape = (de["shape"][0] // tp, de["shape"][1])
        gh = [_xsl_host_tensor(b, gshape, f"L{layer_idx}e{e}g{k}")
              for k, b in enumerate(_xsl_slices_bytes(gsrc, ge["shape"], tp, slices, "gate_up"))]  # fmt: skip
        dh = [_xsl_host_tensor(b, dshape, f"L{layer_idx}e{e}d{k}")
              for k, b in enumerate(_xsl_slices_bytes(dsrc, de["shape"], tp, slices, "down"))]  # fmt: skip
        return gh, dh, gshape, dshape

    gate_up, down = [], []
    gu_cfg = dn_cfg = None
    # host work in a pool, device writes in expert order on this thread (the lock-step DRAM allocation order, and
    # therefore every expert's address, stays the serial loader's); GLM53_XSL_LOAD_THREADS=1: serial
    with ThreadPoolExecutor(max_workers=max(1, int(os.environ.get("GLM53_XSL_LOAD_THREADS", "16")))) as pool:
        for gh, dh, gshape, dshape in pool.map(host, range(num_experts)):
            if gu_cfg is None:
                gu_cfg, dn_cfg = _nd_dram(gshape[0], 64), _nd_dram(dshape[0], 64)
            g = ttnn.from_host_shards(gh, mesh_device.shape) if ndev > 1 else gh[0]
            d = ttnn.from_host_shards(dh, mesh_device.shape) if ndev > 1 else dh[0]
            gate_up.append(ttnn.to_device(g, mesh_device, memory_config=gu_cfg))
            down.append(ttnn.to_device(d, mesh_device, memory_config=dn_cfg))
    return gate_up, down


# ======================================================================================================================
def _row(t: ttnn.Tensor) -> ttnn.Tensor:
    return ttnn.reshape(t, (1, 1, 1, t.shape[-1]))


def to_stacked(x: ttnn.Tensor, n: int) -> ttnn.Tensor:
    """[1, 1, T, n*D] -> [T, 1, n, D] (the fused mHC ops' stream stack)."""
    t, w = x.shape[-2], x.shape[-1]
    return ttnn.reshape(x, (t, 1, n, w // n))


def reduce_streams(x: ttnn.Tensor) -> ttnn.Tensor:
    """Sum over dim 1 ([1, n, T, W] -> [1, 1, T, W]) with fast_reduce_nc, whose output carries the padded tile height
    as its logical row count: slice back to T rows so it broadcasts against [1, 1, T, *] operands for any T."""
    t, w = x.shape[-2], x.shape[-1]
    return ttnn.slice(ttnn.experimental.fast_reduce_nc(x, dims=[1]), (0, 0, 0, 0), (1, 1, t, w))


def to_flat(x: ttnn.Tensor) -> ttnn.Tensor:
    """[T, 1, n, D] -> [1, 1, T, n*D]."""
    return ttnn.reshape(x, (1, 1, x.shape[0], x.shape[-2] * x.shape[-1]))


class FusedHyperConnection:
    """mHC sublayer wrapper on fused_hyperconnection + mix_streams (fp32 mode: coefficient path, Sinkhorn and mix in
    fp32 on the SFPU / exact-split FPU products, P2a)."""

    def __init__(
        self,
        mesh_device,
        fn_t,
        base: torch.Tensor,
        scale: torch.Tensor,
        *,
        hc_mult,
        hidden_size,
        sinkhorn_iters,
        hc_eps,
        norm_eps,
        ksplit: int,
        fcfg=None,
    ):
        self.device = mesh_device
        self.fcfg = fcfg or FusedDecoderConfig()
        self.n = hc_mult
        self.hidden = hidden_size
        self.width = hc_mult * hidden_size
        self.iters = sinkhorn_iters
        self.hc_eps = hc_eps
        self.norm_eps = norm_eps
        self.ckc = hifi4_fp32(mesh_device)
        self.n_mix = (2 + hc_mult) * hc_mult  # 24
        fn32 = ttnn.typecast(fn_t, FP32)  # [16384, 24] (bf16 checkpoint weight used in fp32, modeling:303)
        self.fn_t = ttnn.reshape(fn32, (1, 1, self.width, self.n_mix))
        self.ksplit = ksplit if ksplit > 1 and self.width % (ksplit * TILE) == 0 else 1
        if self.ksplit > 1:
            fn_pad = ttnn.pad(self.fn_t, ((0, 0), (0, 0), (0, 0), (0, TILE - self.n_mix)), 0.0)
            # [1, ksplit, width / ksplit, 32]: rows of block k are the fn^T rows of input columns [k w/ks, (k+1) w/ks)
            self.fn_split = ttnn.reshape(fn_pad, (1, self.ksplit, self.width // self.ksplit, TILE))
        # stacked projection (mhc_proj="stacked"): fn_cat [4096, 4*32], tile block i = fn^T rows of stream i (24 used
        # columns); per token Y = X_stk [4(32), 4096] @ fn_cat, keep row i of block i (diag_mask), sum rows and blocks
        fn_h = ttnn.to_torch(fn32).float().reshape(hc_mult, hidden_size, self.n_mix)  # setup-time host relayout
        cat = torch.zeros(hidden_size, hc_mult * TILE)
        for i in range(hc_mult):
            cat[:, i * TILE : i * TILE + self.n_mix] = fn_h[i]
        mask = torch.zeros(1, 1, hc_mult, hc_mult * TILE)
        for i in range(hc_mult):
            mask[0, 0, i, i * TILE : (i + 1) * TILE] = 1.0
        mk32 = lambda t: ttnn.from_torch(t, dtype=FP32, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=DRAM)
        self.fn_cat, self.diag_mask = mk32(cat), mk32(mask)
        b = base.float()
        n = hc_mult
        mk = lambda t, shape: ttnn.from_torch(
            t.reshape(shape), dtype=FP32, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=DRAM
        )
        self.pre_bias = mk(b[:n], (1, 1, 1, n))
        self.post_bias = mk(b[n : 2 * n], (1, 1, 1, n))
        self.comb_bias = mk(b[2 * n :], (1, 1, n, n))  # pre-shaped: no per-call reshape (P2a N3)
        self.scales = [float(s) for s in scale.float()]

    def proj_mode(self, t: int) -> str:
        p = self.fcfg.mhc_proj
        if p == "auto":
            p = "stacked" if t <= self.fcfg.mhc_stacked_max_rows else "ksplit"
        return p

    def mixes(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x [1,1,T,16384] fp32 -> m = rsqrt(mean(x^2) + eps) * (x @ fn^T) [1,1,T,24] fp32."""
        t = x.shape[-2]
        proj = self.proj_mode(t)
        if proj == "rmsnorm":  # graph-fusing RMSNorm pattern: unweighted rms_norm op, then the matmul
            xn = ttnn.rms_norm(x, epsilon=self.norm_eps, compute_kernel_config=self.ckc)
            return ttnn.matmul(xn, self.fn_t, compute_kernel_config=self.ckc, dtype=FP32)
        if self.ksplit > 1 and t <= TILE and proj == "ksplit":
            ks, kw = self.ksplit, self.width // self.ksplit
            # K-split: [1,1,T,W] -> [1,T,ks,kw] -> [1,ks,T,kw]; the batched matmul runs on ks cores instead of one
            xv = ttnn.permute(ttnn.reshape(x, (1, t, ks, kw)), (0, 2, 1, 3))
            m = ttnn.experimental.fast_reduce_nc(
                ttnn.matmul(xv, self.fn_split, compute_kernel_config=self.ckc, dtype=FP32), dims=[1]
            )
            ss = ttnn.experimental.fast_reduce_nc(
                ttnn.sum(ttnn.multiply(xv, xv), dim=-1, keepdim=True), dims=[1]
            )  # [1,1,T,1]
            m = ttnn.slice(m, (0, 0, 0, 0), (1, 1, t, self.n_mix))
            ss = ttnn.slice(ss, (0, 0, 0, 0), (1, 1, t, 1))
        else:
            m = ttnn.matmul(x, self.fn_t, compute_kernel_config=self.ckc, dtype=FP32)
            ss = ttnn.sum(ttnn.multiply(x, x), dim=-1, keepdim=True)
        inv = ttnn.rsqrt(ttnn.add(ttnn.multiply(ss, 1.0 / self.width), self.norm_eps))
        return ttnn.multiply(m, inv)

    def mixes_stacked(self, x_stk: ttnn.Tensor) -> ttnn.Tensor:
        """Projection straight from the stacked streams x_stk [T,1,4,4096] -> m [1,1,T,24] (no flat copy needed): one
        batched matmul against fn_cat (128 K-tiles per output tile instead of 512), the diagonal blocks kept by a mask,
        rows summed, the 4 tile-aligned blocks added in fp32; sum(x^2) over the token's 4 x 4096 values."""
        t = x_stk.shape[0]
        y = ttnn.matmul(x_stk, self.fn_cat, compute_kernel_config=self.ckc, dtype=FP32)  # [T,1,4,128]
        z = ttnn.sum(ttnn.multiply(y, self.diag_mask), dim=-2, keepdim=True)  # [T,1,1,128]
        m = ttnn.slice(z, (0, 0, 0, 0), (t, 1, 1, TILE))
        for i in range(1, self.n):
            m = ttnn.add(m, ttnn.slice(z, (0, 0, 0, i * TILE), (t, 1, 1, (i + 1) * TILE)))
        ss = ttnn.sum(ttnn.sum(ttnn.multiply(x_stk, x_stk), dim=-1, keepdim=True), dim=-2, keepdim=True)  # [T,1,1,1]
        inv = ttnn.rsqrt(ttnn.add(ttnn.multiply(ss, 1.0 / self.width), self.norm_eps))
        m = ttnn.multiply(ttnn.slice(m, (0, 0, 0, 0), (t, 1, 1, self.n_mix)), inv)
        return ttnn.reshape(m, (1, 1, t, self.n_mix))

    def pre(self, x_flat: ttnn.Tensor | None, x_stk: ttnn.Tensor):
        """-> (h [T,1,1,4096] fp32 collapsed sublayer input, post [T,1,4,1], comb [T,1,4,4]). x_flat may be None with
        mhc_proj="stacked" (the projection then reads x_stk only)."""
        stacked = x_flat is None or self.proj_mode(x_stk.shape[0]) == "stacked"
        post, comb, h = ttnn.experimental.deepseek.fused_hyperconnection(
            x_stk,
            fused_w=self.mixes_stacked(x_stk) if stacked else self.mixes(x_flat),
            pre_bias=self.pre_bias,
            post_bias=self.post_bias,
            comb_bias=self.comb_bias,
            num_streams=self.n,
            sinkhorn_iters=self.iters,
            pre_scale=self.scales[0],
            post_scale=self.scales[1],
            comb_scale=self.scales[2],
            eps=self.hc_eps,
        )
        return h, post, comb

    def post(self, y: ttnn.Tensor, x_stk: ttnn.Tensor, post: ttnn.Tensor, comb: ttnn.Tensor) -> ttnn.Tensor:
        """y [1,1,T,4096] (bf16 sublayer output), x_stk [T,1,4,4096] fp32 -> new streams [T,1,4,4096] fp32."""
        t = y.shape[-2]
        return ttnn.experimental.deepseek.mix_streams(post, comb, ttnn.reshape(y, (t, 1, 1, y.shape[-1])), x_stk)


class CompositeHyperConnection(HyperConnection):
    """Stage 1's composite mHC (same ops and numerics) with the coefficient columns of pre / post [1,1,T,4] and comb
    [1,1,T,16] taken by ONE permute to [1,k,T,1] + tile-aligned dim-1 slices, instead of k one-column slices of a tile
    tensor (each a sub-tile untilize + slice + tilize). Same values (pure data movement)."""

    @staticmethod
    def _cols(t, k):
        rows = t.shape[-2]
        tp = ttnn.permute(t, (0, 3, 2, 1))  # [1,k,rows,1]
        return [ttnn.slice(tp, (0, j, 0, 0), (1, j + 1, rows, 1)) for j in range(k)]


def to_stream_major(x: ttnn.Tensor, n: int) -> ttnn.Tensor:
    """[1, 1, T, n*D] -> [1, n, T, D] (stream i = columns [D i, D (i+1))): n tile-aligned column slices + one concat."""
    t, d = x.shape[-2], x.shape[-1] // n
    return ttnn.concat([ttnn.slice(x, (0, 0, 0, i * d), (1, 1, t, (i + 1) * d)) for i in range(n)], dim=1)


def from_stream_major(x: ttnn.Tensor) -> ttnn.Tensor:
    """[1, n, T, D] -> [1, 1, T, n*D]."""
    n, t, d = x.shape[1], x.shape[-2], x.shape[-1]
    return ttnn.concat([ttnn.slice(x, (0, i, 0, 0), (1, i + 1, t, d)) for i in range(n)], dim=-1)


class StreamMajorHyperConnection:
    """mHC sublayer on the stream-major `[1, 4, T, 4096]` residual layout (no tile padding, unlike the fused ops'
    stacked `[T, 1, 4, 4096]`): the fn projection is 4 per-stream batched products (x_s @ fn_s, one output tile each,
    spread over `mhc_sm_proj_cores` cores) + fast_reduce_nc over the streams, straight off this layout; the
    coefficients come from the fused `mhc_split_sinkhorn` kernel (fp32), the collapse is one broadcast multiply +
    fast_reduce_nc over the streams, the mix is broadcast multiply-adds. All fp32 (Stage 1 numerics policy)."""

    fcfg = None  # FusedDecoderConfig, set by FusedDecoderLayer

    def __init__(
        self,
        mesh_device,
        fn_t,
        base: torch.Tensor,
        scale: torch.Tensor,
        *,
        hc_mult,
        hidden_size,
        sinkhorn_iters,
        hc_eps,
        norm_eps,
    ):
        from models.autoports.zai_org_glm_5_3_flash.tt.hyperconnection import build_sinkhorn_consts

        self.n, self.hidden, self.width = hc_mult, hidden_size, hc_mult * hidden_size
        self.iters, self.hc_eps, self.norm_eps = sinkhorn_iters, hc_eps, norm_eps
        self.ckc = hifi4_fp32(mesh_device)
        self.n_mix = (2 + hc_mult) * hc_mult
        fn32 = ttnn.reshape(ttnn.typecast(fn_t, FP32), (1, 1, self.width, self.n_mix))
        fn_pad = ttnn.pad(fn32, ((0, 0), (0, 0), (0, 0), (0, TILE - self.n_mix)), 0.0)
        self.fn_sm = ttnn.reshape(fn_pad, (1, hc_mult, hidden_size, TILE))  # block i = fn^T rows of stream i
        self.consts = ttnn.from_torch(
            build_sinkhorn_consts(hc_mult, scale, base),
            dtype=FP32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=DRAM,
        )

    def pre(self, x_sm: ttnn.Tensor):
        """x_sm [1,4,T,D] fp32 -> (h [1,1,T,D] fp32, post [1,4,T,1], comb [1,1,T,16])."""
        t = x_sm.shape[-2]
        m = ttnn.experimental.fast_reduce_nc(self._sm_project(x_sm), dims=[1])  # [1,1,T,32]
        ss = ttnn.experimental.fast_reduce_nc(ttnn.sum(ttnn.multiply(x_sm, x_sm), dim=-1, keepdim=True), dims=[1])
        inv = ttnn.rsqrt(
            ttnn.add(ttnn.multiply(ttnn.slice(ss, (0, 0, 0, 0), (1, 1, t, 1)), 1.0 / self.width), self.norm_eps)
        )
        m = ttnn.multiply(ttnn.slice(m, (0, 0, 0, 0), (1, 1, t, self.n_mix)), inv)
        pre, post, comb = ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn(
            m, self.consts, self.n, self.iters, self.hc_eps
        )
        r4 = lambda u: ttnn.reshape(u, (1, 1, t, u.shape[-1]))
        pre, post, comb = r4(pre), r4(post), r4(comb)
        pre_sm = ttnn.permute(pre, (0, 3, 2, 1))  # [1,4,T,1]
        h = reduce_streams(ttnn.multiply(x_sm, pre_sm))  # [1,1,T,D]
        return h, ttnn.permute(post, (0, 3, 2, 1)), comb

    def _sm_project(self, x_sm: ttnn.Tensor) -> ttnn.Tensor:
        """4 batched [T, 4096] @ [4096, 32] products (one per stream): one output tile each, so at most 4 cores; the
        default program puts all of them on one core (Tracy: 1 core, ~130 us at B=32) -> core_grid. [1,4,T,32] fp32."""
        cores = self.fcfg.mhc_sm_proj_cores if self.fcfg is not None else 0
        grid = dict(core_grid=ttnn.CoreGrid(y=1, x=cores)) if cores else {}
        return ttnn.matmul(x_sm, self.fn_sm, compute_kernel_config=self.ckc, dtype=FP32, **grid)

    def post(self, y: ttnn.Tensor, x_sm: ttnn.Tensor, post_sm: ttnn.Tensor, comb: ttnn.Tensor) -> ttnn.Tensor:
        """y [1,1,T,D] (bf16 sublayer output) -> new streams [1,4,T,D] fp32: X'_j = post_j y + sum_i C[i,j] X_i."""
        t, n, d = y.shape[-2], self.n, x_sm.shape[-1]
        out = ttnn.multiply(ttnn.typecast(y, FP32), post_sm)  # [1,4,T,D]
        # one permute [1,1,T,16] -> [1,16,T,1] (row i*n + j = C[i, j]); per stream i a dim-1 slice, so no sub-tile
        # column slice (untilize / tilize) and no per-stream permute
        comb_sm = ttnn.permute(comb, (0, 3, 2, 1))
        for i in range(n):
            c_i = ttnn.slice(comb_sm, (0, i * n, 0, 0), (1, (i + 1) * n, t, 1))  # [1,4,T,1] over j
            x_i = ttnn.slice(x_sm, (0, i, 0, 0), (1, i + 1, t, d))  # [1,1,T,D]
            out = ttnn.add(out, ttnn.multiply(x_i, c_i))
        return out


# ======================================================================================================================
class _SlotView:
    """Read-only per-slot view of a batched state tensor (Stage 1's `cache.kda.recurrent[slot]` /
    `cache.kda.conv[slot]` access pattern, for inspection outside the measured passes)."""

    def __init__(self, n: int, get):
        self._n, self._get = n, get

    def __len__(self):
        return self._n

    def __getitem__(self, slot: int) -> ttnn.Tensor:
        if not 0 <= slot < self._n:
            raise IndexError(slot)
        return self._get(slot)


@dataclass
class FusedKDAState:
    """Persistent per-user KDA state of one layer for `slots` users, batched in ONE tensor each (decode reads/writes
    them whole; prefill reads/writes one slot in place)."""

    S: ttnn.Tensor  # [slots, H, K, V] fp32 TILE recurrent state
    C: ttnn.Tensor  # [1, 3, slots, 3*H*K] bf16 TILE: pre-conv rows t-3, t-2, t-1, users as rows
    slot_mask: ttnn.Tensor  # [1, 1, slots, 1] bf16 one-hot of the prefill slot (refreshed at the chunk boundary)
    zero_state: ttnn.Tensor  # [1, H, K, V] fp32 zeros (slot reset)
    slots: int

    def recurrent_of(self, slot: int) -> ttnn.Tensor:
        """[1, H, K, V] copy of one slot's recurrent state (dim-0 slice)."""
        return ttnn.slice(self.S, (slot, 0, 0, 0), (slot + 1, *tuple(self.S.shape)[1:]))

    def conv_rows_of(self, slot: int) -> ttnn.Tensor:
        """[1, 3, 3HK] bf16 HOST tensor of one slot's conv history rows t-3..t-1 (inspection helper: a device read, no
        device program, so it is safe while a decode trace is live)."""
        rows = ttnn.to_torch(self.C)[0, :, slot, :]
        return ttnn.from_torch(rows.reshape(1, CONV_K - 1, -1), dtype=BF16)

    @property
    def recurrent(self) -> _SlotView:
        return _SlotView(self.slots, self.recurrent_of)

    @property
    def conv(self) -> _SlotView:
        return _SlotView(self.slots, self.conv_rows_of)


class FusedKDA(KDA):
    """KDA with the fused decode path; prefill = Stage 1's exact chunked path through the batched state (the hoisted
    chunk scan is a measured option, kda_prefill_scan="hoisted")."""

    def allocate_state(self, slots: int) -> FusedKDAState:
        if slots > TILE:
            raise ValueError(f"fused KDA decode supports up to {TILE} user slots, got {slots}")
        z = lambda shape, dtype: ttnn.zeros(
            shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=DRAM
        )
        return FusedKDAState(
            S=z((slots, self.h, self.d, self.d), KDA_STATE_DTYPE),
            C=z((1, CONV_K - 1, slots, 3 * self.width), BF16),
            slot_mask=z((1, 1, slots, 1), BF16),
            zero_state=z((1, self.h, self.d, self.d), KDA_STATE_DTYPE),
            slots=slots,
        )

    def set_decode_weights(self):
        """fp32 conv taps [1, 3, 1, 3HK] (history rows) and [1, 1, 1, 3HK] (current row) for the decode conv."""
        rows = [ttnn.reshape(t, (1, 1, 1, t.shape[-1])) for t in self.taps_decode]
        self.taps_hist = ttnn.concat(rows[: CONV_K - 1], dim=1, memory_config=DRAM)  # [1,3,1,3HK]
        self.tap_cur = rows[CONV_K - 1]
        self.taps_all = ttnn.concat(rows, dim=1, memory_config=DRAM)  # [1,4,1,3HK]
        self.o_norm_row32 = ttnn.reshape(ttnn.typecast(self.o_norm, FP32), (1, 1, 1, self.d))

    # ------------------------------------------------------------------------------------------------ prefill scan
    def _chunk_recurrence(self, q, k, v, g, beta, s0, actual_end, g32, valid_end_f):
        """Stage 1's chunked delta rule (same `prepare_chunk_recurrence` terms, same exact-fp32 state carry) with the
        state-independent products hoisted out of the sequential chunk loop (kda_prefill_scan="hoisted"):
            U_n = A_n - B_n S_n,  Y_n = D_n + C_n S_n   with  A = T^-1 v_beta, B = T^-1 kd, C = q_decay - intra B,
            D = intra A  (batched over all chunks), so per chunk ONE matmul [-B_n; C_n] S_n gives [U_n; Y_n] after one
            add of [A_n; D_n], and S_n+1 = decay_n * S_n + [k_dec_n^T | 0] [U_n; Y_n] (the zero block drops Y_n
            exactly).
        Per chunk 2 matmuls + 3 eltwise + 4 slices instead of 5 matmuls + 4 eltwise + 7 slices."""
        if self.fcfg.kda_prefill_scan != "hoisted":
            return super()._chunk_recurrence(q, k, v, g, beta, s0, actual_end, g32, valid_end_f)
        c, h, d = q.shape[-2], self.h, self.d
        n_chunks = c // CHUNK
        beta_by_chunk = ttnn.reshape(ttnn.permute(beta, (0, 2, 1)), (h, n_chunks, CHUNK, 1))
        v_beta, kd, q_decay, intra, k_dec_t, final_decay, t_inv = ttnn.experimental.kda.prepare_chunk_recurrence(
            q,
            k,
            v,
            g,
            beta_by_chunk,
            h,
            memory_config=DRAM,
            compute_kernel_config=self.ckc_kda,
            output_bf16_mask=0,
            actual_start=self.zero_u32,
            actual_end=actual_end,
            sequence_parallel_axis=0,
        )
        ttnn.deallocate(final_decay)  # recomputed in fp32 below (Stage 1: the op's value is biased for slow channels)
        rows = ttnn.slice(self.row_ids, (0, 0, 0, 0), (1, 1, c, 1))
        g_valid = ttnn.multiply(g32, ttnn.lt(rows, valid_end_f))  # padded rows are identity steps
        g_sum = ttnn.sum(ttnn.reshape(g_valid, (n_chunks, CHUNK, h * d)), dim=1, keepdim=True)  # [N, 1, HD]
        chunk_decay = ttnn.exp(
            ttnn.reshape(ttnn.permute(ttnn.reshape(g_sum, (n_chunks, h, d)), (1, 0, 2)), (h, n_chunks, d, 1))
        )
        mm = self._mm
        a_t = mm(t_inv, v_beta)  # [H, N, 32, D]
        b_t = mm(t_inv, kd)
        nbc = ttnn.concat([ttnn.neg(b_t), ttnn.subtract(q_decay, mm(intra, b_t))], dim=2)  # [H, N, 64, D]
        ad = ttnn.concat([a_t, mm(intra, a_t)], dim=2)  # [H, N, 64, D]
        kz = ttnn.pad(k_dec_t, ((0, 0), (0, 0), (0, 0), (0, CHUNK)), 0.0)  # [H, N, D, 64]
        for t in (a_t, b_t, v_beta, kd, q_decay, intra, k_dec_t, t_inv):
            ttnn.deallocate(t)
        live = chunk_live_masks(rows, valid_end_f, n_chunks)  # chunks past the valid end keep the state (kda.py)
        state = ttnn.reshape(s0, (h, d, d))
        outs = []
        for n in range(n_chunks):
            sl = lambda t, r, cols: ttnn.reshape(ttnn.slice(t, (0, n, 0, 0), (h, n + 1, r, cols)), (h, r, cols))
            uy = ttnn.add(sl(ad, 2 * CHUNK, d), mm(sl(nbc, 2 * CHUNK, d), state))  # [H, 64, D] = [U_n; Y_n]
            outs.append(uy)
            new_state = ttnn.add(ttnn.multiply(state, sl(chunk_decay, d, 1)), mm(sl(kz, d, 2 * CHUNK), uy))
            state = keep_if_dead(live[n], new_state, state)
        uy_all = ttnn.reshape(outs[0] if n_chunks == 1 else ttnn.concat(outs, dim=1), (h, n_chunks, 2 * CHUNK, d))
        y = ttnn.reshape(ttnn.slice(uy_all, (0, 0, CHUNK, 0), (h, n_chunks, 2 * CHUNK, d)), (h, c, d))
        return y, ttnn.reshape(state, (1, h, d, d))

    # ------------------------------------------------------------------------------------------------ prefill I/O
    def read_slot(self, state: FusedKDAState, slot: int):
        """(conv history [1, 3, 3HK] bf16 ROW_MAJOR, recurrent state [1, H, K, V] fp32) of the prefill slot. The conv
        rows are selected by the device one-hot `slot_mask` with a where (the other slots contribute exact zeros
        whatever they hold, NaN / Inf included: value-independent, like Stage 1's per-slot tensors), then summed over
        the slots; the state with a dim-0 slice."""
        rows = ttnn.sum(ttnn.where(state.slot_mask, state.C, 0.0), dim=2, keepdim=True)  # [1,3,1,3HK]
        carry = ttnn.to_layout(ttnn.reshape(rows, (1, CONV_K - 1, 3 * self.width)), ttnn.ROW_MAJOR_LAYOUT)
        return carry, state.recurrent_of(slot)

    def write_slot(self, state: FusedKDAState, slot: int, carry: ttnn.Tensor, recurrent: ttnn.Tensor):
        """In place: the slot's recurrent state with fill_cache (batch_idx is a runtime arg), its conv rows with a
        one-hot blend of the whole (small) history tensor followed by an in-place copy."""
        ttnn.fill_cache(state.S, recurrent, slot)
        rows = ttnn.reshape(ttnn.to_layout(carry, ttnn.TILE_LAYOUT), (1, CONV_K - 1, 1, 3 * self.width))
        blended = ttnn.where(state.slot_mask, rows, state.C)
        ttnn.copy(blended, state.C)

    def reset_slot(self, state: FusedKDAState, slot: int):
        ttnn.fill_cache(state.S, state.zero_state, slot)
        ttnn.copy(ttnn.where(state.slot_mask, 0.0, state.C), state.C)

    # ---------------------------------------------------------------------------------------------------- decode
    def decode(self, x: ttnn.Tensor, state: FusedKDAState, active=None) -> ttnn.Tensor:
        """x [1,1,B,4096] bf16, one token per slot 0..B-1 (B == state.slots). Updates state in place.
        active: optional (fp32, bf16) pair of [1,1,B,1] masks (1 = decode the slot, 0 = inactive, cur_pos == -1). An
        inactive slot gets g = 0 and beta = 0, which makes the fused step the identity on its state (bitwise:
        exp(0) = 1, delta = 0), and keeps its conv history rows (where-select); its output row is don't-care."""
        bsz = x.shape[-2]
        h, d, w = self.h, self.d, self.width
        conv_width = 3 * w
        if active is not None:
            # an inactive row is don't-care and may hold NaN / Inf (in a full model, whatever an earlier layer
            # produced): zero it first, selected by the cur_pos-derived mask (never by the row's values), so everything
            # derived from it below is finite and the g = 0 / beta = 0 identity step holds bitwise
            x = ttnn.where(active[1], x, 0.0)
        qkv = ttnn.linear(x, self.w_qkv, compute_kernel_config=self.ckc_proj, memory_config=DRAM, dtype=BF16)
        f_a, g_a, b = self._lowrank(x)
        gate_out = ttnn.linear(g_a, self.w_gb, compute_kernel_config=self.ckc_fp32, memory_config=DRAM, dtype=FP32)
        sig = [ttnn.UnaryOpType.SIGMOID]
        if self.fcfg.kda_merge_gates:
            if self.fcfg.kda_fb_bias_in_linear:
                f = ttnn.linear(
                    f_a,
                    self.w_fb,
                    bias=self.dt_bias,
                    compute_kernel_config=self.ckc_fp32,
                    memory_config=DRAM,
                    dtype=FP32,
                )
            else:
                f = ttnn.add(
                    ttnn.linear(f_a, self.w_fb, compute_kernel_config=self.ckc_fp32, memory_config=DRAM, dtype=FP32),
                    self.dt_bias,
                )
            z = ttnn.multiply(f, self.decay_rate)
            # g = sigmoid(z) * (lower_bound * active); beta = sigmoid(b) * active  (active = 1 without masks)
            g = ttnn.multiply(z, self.lower_bound if active is None else active[2], input_tensor_a_activations=sig)
            beta = ttnn.multiply(b, 1.0 if active is None else active[0], input_tensor_a_activations=sig, dtype=FP32)
        else:
            g = self._decay(f_a)  # [1,1,B,HK] fp32 log-decay
            beta = ttnn.sigmoid(ttnn.typecast(b, FP32))  # [1,1,B,H]
            if active is not None:  # inactive slot: g = 0 (exp(+-0) = 1) and beta = 0 -> identity step on its state
                g = ttnn.multiply(g, active[0])
                beta = ttnn.multiply(beta, active[0])

        # depthwise causal conv over [hist(t-3), hist(t-2), hist(t-1), u_t] (users as rows) + SiLU, fp32
        hist = state.C  # [1,3,B,3HK] bf16
        if self.fcfg.kda_conv_window:
            win = ttnn.concat([hist, ttnn.reshape(qkv, (1, 1, bsz, conv_width))], dim=1)  # [1,4,B,3HK] bf16
            conv = ttnn.silu(reduce_streams(ttnn.multiply(win, self.taps_all, dtype=FP32)))
            new_hist = ttnn.slice(win, (0, 1, 0, 0), (1, CONV_K, bsz, conv_width))
        else:
            prod = ttnn.multiply(hist, self.taps_hist, dtype=FP32)  # [1,3,B,3HK]
            if self.fcfg.kda_conv_reduce == "fast_reduce_nc":
                acc = reduce_streams(prod)  # [1,1,B,3HK]
            else:
                acc = ttnn.sum(prod, dim=1, keepdim=True)
            acc = ttnn.addcmul(acc, ttnn.typecast(qkv, FP32), self.tap_cur)
            conv = ttnn.silu(acc)
            new_hist = ttnn.concat(
                [
                    ttnn.slice(hist, (0, 1, 0, 0), (1, CONV_K - 1, bsz, conv_width)),
                    ttnn.reshape(qkv, (1, 1, bsz, conv_width)),
                ],
                dim=1,
            )
        if active is not None:
            new_hist = ttnn.where(active[1], new_hist, hist)
        ttnn.copy(new_hist, state.C)

        q = ttnn.slice(conv, (0, 0, 0, 0), (1, 1, bsz, w))
        k = ttnn.slice(conv, (0, 0, 0, w), (1, 1, bsz, 2 * w))
        v = ttnn.slice(conv, (0, 0, 0, 2 * w), (1, 1, bsz, 3 * w))
        if not self.fcfg.kda_decode_rows:  # B, T from q's leading dims: q [B, 1, HK], beta [B, 1, H]
            q, beta = ttnn.reshape(q, (bsz, 1, w)), ttnn.reshape(beta, (bsz, 1, h))
        inplace = self.fcfg.kda_inplace_state
        o, s_new = ttnn.experimental.kda.fused_recurrent_kda(
            q,
            k,
            v,
            g,
            beta,
            initial_state=state.S,
            output_final_state=True,
            use_qk_l2norm=True,
            inplace_state=inplace,
        )  # o [B,1,H,V] fp32; the final state [B,H,K,V] (in place: state.S itself)
        if not inplace:
            ttnn.copy(s_new, state.S)

        # gated RMSNorm over D per (user, head) row, weight o_norm, gate sigmoid(g_b(g_a(x)))
        if self.fcfg.kda_gated_norm_op and bsz >= self.fcfg.kda_gated_norm_op_min_rows:
            tu = round_up(bsz, TILE)  # the op wants a tile-aligned sequence: users padded to 32 rows
            y = ttnn.reshape(ttnn.permute(o, (2, 1, 0, 3)), (h, bsz, d))  # [H, B, V] head-major
            gate = ttnn.reshape(ttnn.typecast(gate_out, BF16), (1, bsz, w))
            if tu != bsz:
                y = ttnn.pad(y, ((0, 0), (0, tu - bsz), (0, 0)), 0.0)
                gate = ttnn.pad(gate, ((0, 0), (0, tu - bsz), (0, 0)), 0.0)
            on = ttnn.experimental.kda.sigmoid_gated_rms_norm(
                y,
                gate,
                self.o_norm,
                h,
                epsilon=self.eps,
                memory_config=DRAM,
                compute_kernel_config=self.ckc_kda,
                output_dtype=BF16,
            )  # [1, Tu, H*V]
            if tu != bsz:
                on = ttnn.slice(on, (0, 0, 0), (1, bsz, w))
            return ttnn.linear(
                ttnn.reshape(on, (1, 1, bsz, w)),
                self.w_o,
                compute_kernel_config=self.ckc_proj,
                memory_config=DRAM,
                dtype=BF16,
            )
        rows = ttnn.reshape(o, (1, 1, bsz * h, d))
        normed = ttnn.rms_norm(rows, weight=self.o_norm_row32, epsilon=self.eps, compute_kernel_config=self.ckc_fp32)
        gated = ttnn.multiply(
            normed, ttnn.reshape(gate_out, (1, 1, bsz * h, d)), input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]
        )
        o = ttnn.typecast(ttnn.reshape(gated, (1, 1, bsz, w)), BF16)
        return ttnn.linear(o, self.w_o, compute_kernel_config=self.ckc_proj, memory_config=DRAM, dtype=BF16)


# ======================================================================================================================
class FusedMLA(MLA):
    """MLA with dedicated head ops: nlp_create_qkv_heads (q head split), full-grid per-head absorption matmuls, one
    64-head flash-MLA decode call from mla_single_call_min_batch rows, nlp_concat_heads head concat in two 32-head
    groups."""

    def _head_matmul(self, a, b):
        """Per-head [64, T, K] @ [64, K, N]. Decode-sized T: spread the 64 heads over the full grid (P3:
        W_uk 234 -> 79 us, W_uv 392 -> 77 us); prefill-sized T keeps the default program (the full-grid config
        overflows L1)."""
        if a.shape[-2] > TILE:
            return ttnn.matmul(a, b, compute_kernel_config=self.ckc_fp32, dtype=BF16)
        grid = self.device.compute_with_storage_grid_size()
        return ttnn.matmul(
            a, b, compute_kernel_config=self.ckc_fp32, dtype=BF16, core_grid=ttnn.CoreGrid(y=grid.y, x=grid.x)
        )

    def _project(self, x):
        """x [1,1,T,4096] -> (q_lat [1,64,T,512] bf16, latent [1,1,T,512] bf16)."""
        t = x.shape[-2]
        qk = ttnn.linear(x, self.w_qa_kva, compute_kernel_config=self.ckc_proj, memory_config=DRAM, dtype=BF16)
        q_r = ttnn.slice(qk, (0, 0, 0, 0), (1, 1, t, self.q_lora))
        c = ttnn.slice(qk, (0, 0, 0, self.q_lora), (1, 1, t, self.q_lora + self.kv_lora))
        ttnn.deallocate(qk)
        q_r = ttnn.rms_norm(q_r, weight=self.qa_norm, epsilon=self.eps, compute_kernel_config=self.ckc_fp32)
        latent = ttnn.rms_norm(c, weight=self.kva_norm, epsilon=self.eps, compute_kernel_config=self.ckc_fp32)
        q = ttnn.linear(q_r, self.w_qb, compute_kernel_config=self.ckc_proj, memory_config=DRAM, dtype=BF16)
        q, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            q, num_heads=self.h, num_kv_heads=0, transpose_k_heads=False, memory_config=DRAM
        )  # [1,64,T,256]
        q_lat = self._head_matmul(q, self.w_uk)  # [1,64,T,512]
        return q_lat, latent

    def decode(self, x: ttnn.Tensor, cache: ttnn.Tensor, page_table: ttnn.Tensor, cur_pos: ttnn.Tensor) -> ttnn.Tensor:
        """Stage 1's MLA decode (latent write at cur_pos, attention over 0..cur_pos; cur_pos -1 = inactive slot) with
        one 64-head flash-MLA call for B >= mla_single_call_min_batch (two 32-head calls + concat below)."""
        bsz = x.shape[-2]
        q_lat, latent = self._project(x)
        q_lat = ttnn.permute(q_lat, (0, 2, 1, 3))  # [1,B,64,512]
        grid = self.device.compute_with_storage_grid_size()
        shard = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, self.kv_lora),
            core_grid=ttnn.num_cores_to_corerangeset(bsz, grid, row_wise=True),
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        lat = ttnn.to_memory_config(ttnn.reshape(latent, (1, bsz, 1, self.kv_lora)), shard)
        ttnn.experimental.paged_update_cache(cache, lat, update_idxs_tensor=cur_pos, page_table=page_table)
        program = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid,
            q_chunk_size=0,
            k_chunk_size=(
                self.fcfg.mla_decode_k_chunk_small
                if self.fcfg.mla_decode_k_chunk_small and bsz <= self.fcfg.mla_decode_small_max_rows
                else self.fcfg.mla_decode_k_chunk
            ),
            exp_approx_mode=False,
        )
        heads = self.h if bsz >= self.fcfg.mla_single_call_min_batch else self.decode_heads
        groups = []
        for h0 in range(0, self.h, heads):
            q_g = q_lat if heads == self.h else ttnn.slice(q_lat, (0, 0, h0, 0), (1, bsz, h0 + heads, self.kv_lora))
            o_g = ttnn.transformer.paged_flash_multi_latent_attention_decode(
                q_g,
                cache,
                head_dim_v=self.kv_lora,
                page_table_tensor=page_table,
                cur_pos_tensor=cur_pos,
                scale=self.scale,
                program_config=program,
                compute_kernel_config=self.ckc_sdpa,
                memory_config=DRAM,
            )
            groups.append(ttnn.slice(o_g, (0, 0, 0, 0), (1, bsz, heads, self.kv_lora)))
        o_lat = groups[0] if len(groups) == 1 else ttnn.concat(groups, dim=2)
        return self._output(ttnn.permute(o_lat, (0, 2, 1, 3)))  # [1,64,B,512]

    def _output(self, o_lat):
        """o_lat [1,64,T,512] -> [1,1,T,4096]."""
        t = o_lat.shape[-2]
        o = self._head_matmul(o_lat, self.w_uv)  # [1,64,T,256]
        if self.fcfg.mla_concat == "nlp2":
            # nlp_concat_heads stages a whole head row per core: 64 x 256 needs 2.2 MB of CBs (> 1.5 MB L1), so two
            # 32-head groups (dim-1 slices) and one concat along W
            g = self.h // 2
            parts = [
                ttnn.experimental.nlp_concat_heads(ttnn.slice(o, (0, i * g, 0, 0), (1, (i + 1) * g, t, self.dv)))
                for i in range(2)
            ]
            o = ttnn.concat(parts, dim=-1)  # [1,1,T,16384]
        else:
            o = ttnn.reshape(ttnn.permute(o, (0, 2, 1, 3)), (1, 1, t, self.h * self.dv))  # [1,1,T,16384]
        return ttnn.linear(o, self.w_o, compute_kernel_config=self.ckc_proj, memory_config=DRAM, dtype=BF16)


# ======================================================================================================================
class FusedRouter:
    """noaux_tc sigmoid router -> (ids [1,1,T,8] uint16, unbiased scores [1,1,T,288] bf16) for fused_experts.
    Selection ranks the fp32 choice (scores + bias) as the reference; fused_experts renormalises the k selected
    unbiased scores in-kernel and scales by routed_scaling_factor."""

    def __init__(self, mesh_device, weight_t, bias: torch.Tensor, *, num_experts: int, top_k: int, fcfg=None):
        self.num_experts = num_experts
        self.top_k = top_k
        self.fcfg = fcfg or FusedDecoderConfig()
        mk = lambda t: ttnn.from_torch(t, dtype=FP32, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=DRAM)
        pad = PADDED_EXPERTS - num_experts
        self.weights = {num_experts: weight_t, PADDED_EXPERTS: ttnn.pad(weight_t, ((0, 0), (0, pad)), 0.0)}
        self.biases = {
            num_experts: mk(bias.float().reshape(1, 1, 1, num_experts)),
            PADDED_EXPERTS: mk(torch.cat([bias.float(), torch.full((pad,), -1e9)]).reshape(1, 1, 1, PADDED_EXPERTS)),
        }
        self.ckc = hifi4_fp32(mesh_device)

    def __call__(self, x: ttnn.Tensor):
        t = x.shape[-2]
        if self.fcfg.router_impl == "grouped_topk" or 0 < t <= self.fcfg.router_grouped_max_rows:
            e = self.num_experts
            logits = ttnn.linear(
                x, self.weights[e], compute_kernel_config=self.ckc, dtype=FP32, memory_config=DRAM
            )  # [1,1,T,E]
            _, idx = ttnn.experimental.deepseek_prefill.moe_grouped_topk(
                logits,
                ttnn.repeat(self.biases[e], (1, 1, t, 1)),  # the op wants the bias at the scores' full shape
                n_groups=1,
                summed_experts_per_group=1,
                topk_groups=1,
                n_activated_experts=self.top_k,
                route_scale=1.0,
                epsilon=1e-20,
            )
            return idx, ttnn.typecast(ttnn.sigmoid(logits), BF16)
        width = self.fcfg.router_topk_width
        w, bias = self.weights[width], self.biases[width]
        if self.fcfg.router_sigmoid_in_linear:
            scores = ttnn.linear(
                x, w, compute_kernel_config=self.ckc, dtype=FP32, memory_config=DRAM, activation="sigmoid"
            )
        else:
            scores = ttnn.sigmoid(ttnn.linear(x, w, compute_kernel_config=self.ckc, dtype=FP32, memory_config=DRAM))
        choice = ttnn.add(scores, bias)
        _, idx = ttnn.topk(choice, k=self.top_k, dim=-1, largest=True, sorted=True)
        if idx.dtype != ttnn.uint16:
            idx = ttnn.typecast(idx, ttnn.uint16)
        if width != self.num_experts:
            scores = ttnn.slice(scores, (0, 0, 0, 0), (1, 1, t, self.num_experts))
        return idx, ttnn.typecast(scores, BF16)


class FusedSwiGLU(SwiGLUMLP):
    """Shared expert / dense MLP: one [gate | up] matmul and the dedicated clamped-SiLU GLU op (bitwise equal to
    Stage 1's clamp, clamp, multiply-with-SiLU chain)."""

    fcfg = None  # set by FusedDecoderLayer

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        i = self.intermediate
        if self.fcfg is not None and self.fcfg.swiglu_impl == "split_mm":
            gate, up = ttnn.experimental.minimal_matmul_split(
                x, self.gate_up, chunks=2, dim=-1, compute_kernel_config=self.ckc, dtype=BF16
            )
        else:
            gu = ttnn.linear(x, self.gate_up, compute_kernel_config=self.ckc, memory_config=DRAM, dtype=BF16)
            shape = list(gu.shape)
            gate = ttnn.slice(gu, [0, 0, 0, 0], shape[:-1] + [i])
            up = ttnn.slice(gu, [0, 0, 0, i], shape)
        act = ttnn.clamped_silu_glu(gate, up, self.limit)  # silu(min(gate, L)) * clamp(up, -L, L) in one op
        return ttnn.linear(act, self.down, compute_kernel_config=self.ckc, memory_config=DRAM, dtype=BF16)


class FusedMoE:
    def __init__(
        self,
        mesh_device,
        router: FusedRouter,
        gate_up,
        down,
        shared: FusedSwiGLU,
        *,
        intermediate,
        hidden,
        limit,
        scaling,
        cfg: FusedDecoderConfig,
    ):
        self.device = mesh_device
        self.router = router
        self.gate_up, self.down = gate_up, down
        self.shared = shared
        self.num_experts = router.num_experts
        self.top_k = router.top_k
        self.intermediate, self.hidden, self.limit, self.scaling = intermediate, hidden, limit, scaling
        self.cfg = cfg

    def bound(self, rows: int) -> int:
        """Static expert bound of one fused_experts call over `rows` tokens: covers every possible union."""
        return min(self.num_experts, self.top_k * rows)

    def _call(self, x, idx, scores, block: int):
        rows = x.shape[-2]
        return ttnn.experimental.deepseek.moe.fused_experts(
            x,
            routing_indices=idx,
            routing_scores=scores,
            gate_up_weights=self.gate_up,
            down_weights=self.down,
            num_experts=self.bound(rows),
            intermediate_size=self.intermediate,
            swiglu_limit=self.limit,
            top_k=self.top_k,
            routed_scaling_factor=self.scaling,
            routing_eps=1e-20,
            experts_block_size=min(block, self.bound(rows)),
        )  # [1, rows, H] bf16

    def routed(self, x, idx, scores, micro: int, block: int = 1) -> ttnn.Tensor:
        t = x.shape[-2]
        if t <= micro:
            return ttnn.reshape(self._call(x, idx, scores, block), (1, 1, t, self.hidden))
        parts = []
        for s0 in range(0, t, micro):
            s1 = min(t, s0 + micro)
            sl = lambda a: ttnn.slice(a, (0, 0, s0, 0), (1, 1, s1, a.shape[-1]))
            parts.append(ttnn.reshape(self._call(sl(x), sl(idx), sl(scores), block), (1, 1, s1 - s0, self.hidden)))
        return ttnn.concat(parts, dim=2)

    def __call__(self, x: ttnn.Tensor, *, decode: bool, x_router: ttnn.Tensor | None = None) -> ttnn.Tensor:
        idx, scores = self.router(x if x_router is None else x_router)
        if decode:
            micro, block = self.cfg.moe_decode_micro_batch, self.cfg.experts_block_size
        else:
            micro, block = self.cfg.moe_prefill_micro_batch, self.cfg.experts_block_size_prefill
        routed = self.routed(x, idx, scores, micro, block)
        return ttnn.add(routed, self.shared(x))


# ======================================================================================================================
class FusedDecoderLayer:
    """attn_hc -> input norm -> KDA|MLA -> attn mix -> ffn_hc -> post norm -> dense MLP|MoE -> ffn mix."""

    def __init__(
        self, cfg, layer_idx: int, mesh_device, source: WeightSource, *, block_size: int, fcfg: FusedDecoderConfig
    ):
        self.cfg = cfg
        self.fcfg = fcfg
        self.spec = LayerSpec.from_config(cfg, layer_idx)
        self.layer_idx = layer_idx
        self.device = mesh_device
        L = layer_idx
        dev = lambda name: source.device(L, name, mesh_device)
        host = lambda name: source.host(L, name)
        hc_kwargs = dict(
            hc_mult=cfg.hc_mult,
            hidden_size=cfg.hidden_size,
            sinkhorn_iters=cfg.hc_sinkhorn_iters,
            hc_eps=cfg.hc_eps,
            norm_eps=cfg.rms_norm_eps,
            ksplit=fcfg.mhc_ksplit,
        )
        comp_kwargs = {k: v for k, v in hc_kwargs.items() if k != "ksplit"}
        hc_kwargs["fcfg"] = fcfg
        self.attn_hc = FusedHyperConnection(
            mesh_device, dev("hc_attn_fn"), host("hc_attn_base"), host("hc_attn_scale"), **hc_kwargs
        )
        self.ffn_hc = FusedHyperConnection(
            mesh_device, dev("hc_ffn_fn"), host("hc_ffn_base"), host("hc_ffn_scale"), **hc_kwargs
        )
        # Stage-1 composite mHC (prefill, and decode with mhc_decode="composite"), coefficient columns relaid once
        self.attn_hc_comp = CompositeHyperConnection(
            mesh_device, dev("hc_attn_fn"), host("hc_attn_base"), host("hc_attn_scale"), **comp_kwargs
        )
        sm_kwargs = dict(comp_kwargs)
        self.attn_hc_sm = StreamMajorHyperConnection(
            mesh_device, dev("hc_attn_fn"), host("hc_attn_base"), host("hc_attn_scale"), **sm_kwargs
        )
        self.ffn_hc_sm = StreamMajorHyperConnection(
            mesh_device, dev("hc_ffn_fn"), host("hc_ffn_base"), host("hc_ffn_scale"), **sm_kwargs
        )
        self.attn_hc_sm.fcfg = self.ffn_hc_sm.fcfg = fcfg
        self.ffn_hc_comp = CompositeHyperConnection(
            mesh_device, dev("hc_ffn_fn"), host("hc_ffn_base"), host("hc_ffn_scale"), **comp_kwargs
        )
        self.input_norm = _row(dev("input_layernorm.weight"))
        self.post_attn_norm = _row(dev("post_attention_layernorm.weight"))
        # width-sharded multi-core RMSNorm for decode rows (norm_sharded): 32 cores x [32, 128], ROW_MAJOR weights
        d = cfg.hidden_size
        self.norm_cores = ttnn.CoreGrid(x=8, y=4)
        self.norm_shard = ttnn.create_sharded_memory_config(
            (TILE, d // self.norm_cores.num_cores),
            self.norm_cores,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        bw = d // self.norm_cores.num_cores // TILE
        self.norm_program = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[self.norm_cores.x, self.norm_cores.y],
            subblock_w=max(sw for sw in (4, 3, 2, 1) if bw % sw == 0),
            block_h=1,
            block_w=bw,
            inplace=False,
        )
        rm_w = lambda w: ttnn.to_layout(ttnn.reshape(w, (1, 1, d // TILE, TILE)), ttnn.ROW_MAJOR_LAYOUT)
        self.input_norm_rm = rm_w(self.input_norm)
        self.post_attn_norm_rm = rm_w(self.post_attn_norm)
        self.ckc_norm = hifi4_fp32(mesh_device)

        if self.spec.attention == "kda":
            cat = lambda names: ttnn.concat([dev(n) for n in names], dim=-1, memory_config=DRAM)
            self.attn = FusedKDA(
                mesh_device,
                w_qkv=cat(["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"]),
                w_lowrank=cat(["self_attn.f_a_proj.weight", "self_attn.g_a_proj.weight", "self_attn.b_proj.weight"]),
                w_fb=dev("self_attn.f_b_proj.weight"),
                w_gb=dev("self_attn.g_b_proj.weight"),
                w_o=dev("self_attn.o_proj.weight"),
                taps=cat(["self_attn.q_conv1d.weight", "self_attn.k_conv1d.weight", "self_attn.v_conv1d.weight"]),
                o_norm=dev("self_attn.o_norm.weight"),
                dt_bias=host("self_attn.dt_bias"),
                a_log=host("self_attn.A_log"),
                num_heads=cfg.linear_num_heads,
                head_dim=cfg.linear_head_dim,
                hidden=cfg.hidden_size,
                lower_bound=cfg.linear_lower_bound,
                eps=cfg.rms_norm_eps,
            )
            self.attn.set_decode_weights()
            self.attn.fcfg = fcfg
        else:
            self.attn = FusedMLA(
                mesh_device,
                w_qa_kva=ttnn.concat(
                    [dev("self_attn.q_a_proj.weight"), dev("self_attn.kv_a_proj_with_mqa.weight")],
                    dim=-1,
                    memory_config=DRAM,
                ),
                qa_norm=_row(dev("self_attn.q_a_layernorm.weight")),
                kva_norm=_row(dev("self_attn.kv_a_layernorm.weight")),
                w_qb=dev("self_attn.q_b_proj.weight"),
                w_uk=dev("self_attn.kv_b_proj.W_uk"),
                w_uv=dev("self_attn.kv_b_proj.W_uv"),
                w_o=dev("self_attn.o_proj.weight"),
                num_heads=cfg.num_attention_heads,
                q_lora_rank=cfg.q_lora_rank,
                kv_lora_rank=cfg.kv_lora_rank,
                qk_head_dim=cfg.qk_head_dim,
                v_head_dim=cfg.v_head_dim,
                eps=cfg.rms_norm_eps,
                block_size=block_size,
            )
            self.attn.fcfg = fcfg

        if self.spec.mlp == "dense":
            self.mlp = FusedSwiGLU(
                mesh_device,
                dev("mlp.gate_proj.weight"),
                dev("mlp.up_proj.weight"),
                dev("mlp.down_proj.weight"),
                limit=cfg.swiglu_limit,
            )
            self.mlp.fcfg = fcfg
        else:
            shared = FusedSwiGLU(
                mesh_device,
                dev("mlp.shared_experts.gate_proj.weight"),
                dev("mlp.shared_experts.up_proj.weight"),
                dev("mlp.shared_experts.down_proj.weight"),
                limit=cfg.swiglu_limit,
            )
            shared.fcfg = fcfg
            router = FusedRouter(
                mesh_device,
                dev("mlp.gate.weight"),
                host("mlp.gate.e_score_correction_bias"),
                num_experts=cfg.n_routed_experts,
                top_k=cfg.num_experts_per_tok,
                fcfg=fcfg,
            )
            gate_up, down = load_fused_experts(
                source, L, cfg.n_routed_experts, mesh_device, fcfg.fused_experts_cache_dir
            )
            self.mlp = FusedMoE(
                mesh_device,
                router,
                gate_up,
                down,
                shared,
                intermediate=cfg.moe_intermediate_size,
                hidden=cfg.hidden_size,
                limit=cfg.swiglu_limit,
                scaling=cfg.routed_scaling_factor,
                cfg=fcfg,
            )

    def _norm(self, h, weight, dtype=ACT_DTYPE):
        """h [1,1,T,4096] fp32 -> RMSNorm(h) * weight in bf16 (fp32 math; dtype=FP32 returns the fp32 result)."""
        if self.fcfg.norm_sharded and h.shape[-2] <= TILE:
            w_rm = self.input_norm_rm if weight is self.input_norm else self.post_attn_norm_rm
            out = ttnn.rms_norm(
                ttnn.to_memory_config(h, self.norm_shard),
                weight=w_rm,
                epsilon=self.cfg.rms_norm_eps,
                program_config=self.norm_program,
                memory_config=self.norm_shard,
                compute_kernel_config=self.ckc_norm,
            )
            out = ttnn.sharded_to_interleaved(out, DRAM)
            return ttnn.typecast(out, dtype) if out.dtype != dtype else out
        out = ttnn.rms_norm(h, weight=weight, epsilon=self.cfg.rms_norm_eps, compute_kernel_config=self.ckc_norm)
        return ttnn.typecast(out, dtype) if out.dtype != dtype else out

    def _ffn(self, h, decode: bool):
        """post-attention RMSNorm + dense MLP | MoE of the collapsed ffn input h [1,1,T,4096] fp32."""
        if self.spec.mlp != "moe":
            return self.mlp(self._norm(h, self.post_attn_norm))
        if self.fcfg.router_fp32_input:
            x32 = self._norm(h, self.post_attn_norm, dtype=FP32)
            return self.mlp(ttnn.typecast(x32, ACT_DTYPE), decode=decode, x_router=x32)
        return self.mlp(self._norm(h, self.post_attn_norm), decode=decode)

    def forward(self, streams: ttnn.Tensor, attention, *, decode: bool) -> ttnn.Tensor:
        """streams [1,1,T,16384] fp32 -> [1,1,T,16384] fp32."""
        mode = self.fcfg.mhc_decode if decode else self.fcfg.mhc_prefill
        if mode == "auto":
            mode = "fused" if streams.shape[-2] <= self.fcfg.mhc_auto_fused_max_rows else "stream_major"
        if mode == "composite":
            h, post, comb = self.attn_hc_comp.pre(streams)
            a = attention(self._norm(h, self.input_norm))
            streams = self.attn_hc_comp.post(a, streams, post, comb)
            h, post, comb = self.ffn_hc_comp.pre(streams)
            m = self._ffn(h, decode)
            return self.ffn_hc_comp.post(m, streams, post, comb)
        if mode == "stream_major":
            x_sm = to_stream_major(streams, self.cfg.hc_mult)
            h, post, comb = self.attn_hc_sm.pre(x_sm)
            a = attention(self._norm(h, self.input_norm))
            s1 = self.attn_hc_sm.post(a, x_sm, post, comb)
            ttnn.deallocate(x_sm)
            h, post, comb = self.ffn_hc_sm.pre(s1)
            m = self._ffn(h, decode)
            s2 = self.ffn_hc_sm.post(m, s1, post, comb)
            ttnn.deallocate(s1)
            return from_stream_major(s2)
        assert mode == "fused", mode
        t = streams.shape[-2]
        n, d = self.cfg.hc_mult, self.cfg.hidden_size
        x_stk = to_stacked(streams, n)
        h, post, comb = self.attn_hc.pre(streams, x_stk)
        a = attention(self._norm(ttnn.reshape(h, (1, 1, t, d)), self.input_norm))
        s1 = self.attn_hc.post(a, x_stk, post, comb)
        ttnn.deallocate(x_stk)
        s1_flat = None if self.ffn_hc.proj_mode(t) == "stacked" else to_flat(s1)
        h, post, comb = self.ffn_hc.pre(s1_flat, s1)
        if s1_flat is not None:
            ttnn.deallocate(s1_flat)
        m = self._ffn(ttnn.reshape(h, (1, 1, t, d)), decode)
        s2 = self.ffn_hc.post(m, s1, post, comb)
        ttnn.deallocate(s1)
        return to_flat(s2)


# ======================================================================================================================
class FusedDecoder(FunctionalDecoder):
    """Fused decoder layer with the FunctionalDecoder public contract (see module docstring)."""

    def __init__(self, layer: FusedDecoderLayer, hf_config, mesh_device, *, block_size: int = BLOCK_SIZE):
        super().__init__(layer, hf_config, mesh_device, block_size=block_size)
        self.fcfg = layer.fcfg

    @classmethod
    def from_weight_cache(
        cls,
        *,
        hf_config,
        layer_idx: int,
        mesh_device,
        cache_dir: str | None = None,
        manifest_path: str | None = None,
        block_size: int = BLOCK_SIZE,
        source=None,
        fused_config: FusedDecoderConfig | None = None,
    ):
        source = source or TileCacheSource(cache_dir, manifest_path)
        layer = FusedDecoderLayer(
            hf_config, layer_idx, mesh_device, source, block_size=block_size, fcfg=fused_config or FusedDecoderConfig()
        )
        return cls(layer, hf_config, mesh_device, block_size=block_size)

    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx: int,
        mesh_device,
        block_size: int = BLOCK_SIZE,
        fused_config: FusedDecoderConfig | None = None,
    ):
        source = TorchStateDictSource({layer_idx: state_dict}, hf_config)
        return cls.from_weight_cache(
            hf_config=hf_config,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            block_size=block_size,
            source=source,
            fused_config=fused_config,
        )

    # --------------------------------------------------------------------------------------------------------- state
    def allocate_cache(self, *, max_batch: int, max_seq_len: int) -> DecoderCache:
        if max_seq_len > self.cfg.max_position_embeddings:
            raise ValueError(f"max_seq_len {max_seq_len} > advertised context {self.cfg.max_position_embeddings}")
        if self.is_kda:
            cache = DecoderCache(max_batch=max_batch, max_seq_len=max_seq_len)
            cache.kda = self.layer.attn.allocate_state(max_batch)
            return cache
        return super().allocate_cache(max_batch=max_batch, max_seq_len=max_seq_len)

    def snapshot_state(self, cache: DecoderCache):
        if self.is_kda:
            return ttnn.clone(cache.kda.S), ttnn.clone(cache.kda.C)
        return None

    def restore_state(self, cache: DecoderCache, snap):
        if self.is_kda:
            ttnn.copy(snap[0], cache.kda.S)
            ttnn.copy(snap[1], cache.kda.C)

    def _set_slot_mask(self, cache: DecoderCache, slot: int):
        m = torch.zeros(1, 1, cache.max_batch, 1)
        m[0, 0, slot, 0] = 1.0
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(m, dtype=BF16, layout=ttnn.TILE_LAYOUT), cache.kda.slot_mask)

    def reset_slot(self, cache: DecoderCache, slot: int):
        if self.is_kda:
            self._set_slot_mask(cache, slot)
            self.layer.attn.reset_slot(cache.kda, slot)

    def begin_prefill_chunk(self, cache: DecoderCache, *, user_slot: int, chunk_start: int, n_valid: int):
        if self.is_kda and 0 <= user_slot < cache.max_batch:
            self._set_slot_mask(cache, user_slot)
        super().begin_prefill_chunk(cache, user_slot=user_slot, chunk_start=chunk_start, n_valid=n_valid)

    def prepare_prefill(self, cache: DecoderCache):
        """Compile every prefill program (each reachable bucket, and the slot-indexed state read of every slot) before
        any decode trace is captured. Overwrites slot 0's state/pages (call at setup, before requests)."""
        super().prepare_prefill(cache)
        if self.is_kda:  # the slot read is a dim-0 slice whose program is keyed on the slot
            for slot in range(cache.max_batch):
                self._set_slot_mask(cache, slot)
                for t in self.layer.attn.read_slot(cache.kda, slot):
                    ttnn.deallocate(t)
            ttnn.synchronize_device(self.device)

    # -------------------------------------------------------------------------------------------------------- decode
    def _active_masks(self, cur_pos: ttnn.Tensor):
        """cur_pos [B] int32 -> ([1,1,B,1] fp32, [1,1,B,1] bf16, lower_bound * mask fp32): 1 where cur_pos >= 0."""
        b = cur_pos.shape[0]
        cp = ttnn.to_layout(ttnn.reshape(cur_pos, (1, 1, b, 1)), ttnn.TILE_LAYOUT)
        act = ttnn.typecast(ttnn.ge(ttnn.typecast(cp, FP32), 0.0), FP32)
        return act, ttnn.typecast(act, BF16), ttnn.multiply(act, self.layer.attn.lower_bound)

    def decode_forward(self, streams, *, cache, page_table=None, cur_pos=None):
        """streams [1, 1, B, 16384] fp32 -> [1, 1, B, 16384] fp32, device-only. cur_pos[u] == -1 marks slot u inactive
        (KDA state and conv history untouched, MLA skips its cache write and attention; its output row is don't-care).
        """
        if cur_pos is None:
            raise ValueError("decode_forward needs cur_pos [B] int32 (absolute position per slot, -1 = inactive)")
        # host metadata only (empty rows -> SIGFPE); KDA decodes every slot of the batched state (rows == slots), MLA
        # any 1 <= rows <= max_batch
        rows = streams.shape[-2]
        ok = 0 < rows == cur_pos.shape[0] and (rows == cache.max_batch if self.is_kda else rows <= cache.max_batch)
        if not ok:
            raise ValueError(f"decode rows {rows} / cur_pos {cur_pos.shape[0]} / max_batch {cache.max_batch}")
        if self.is_kda:
            active = self._active_masks(cur_pos)
            attn = lambda h: self.layer.attn.decode(h, cache.kda, active)
        else:
            attn = lambda h: self.layer.attn.decode(h, cache.latent, page_table, cur_pos)
        return self.layer.forward(streams, attn, decode=True)
