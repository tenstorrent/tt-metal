# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The Qwen3.5-9B Gated DeltaNet layer — composes config/weights/state/prefill/decode.

Wraps the experimental ``gated_deltanet_forward_ttnn()`` and the on-device GDN prefill
kernel into a module that manages weight tensors, recurrent state, and conv state.
"""
import os

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt.gdn.config import GDNConfig
from models.demos.blackhole.qwen36.tt.gdn.decode import recurrent_forward
from models.demos.blackhole.qwen36.tt.gdn.state import init_recurrent_state, restore_split_conv_from_fused
from models.demos.blackhole.qwen36.tt.gdn.weights import load_gdn_weights

# QWEN36_GDN_PCFG (PR #57440 port, 2026-09-26): program_config for the fused chunk-prefill op
# (ttnn.transformer.chunk_gated_delta_rule), passed through gdn/decode.py. It replaces the removed
# QWEN_GDN_NP / QWEN_GDN_NV / QWEN_GDN_PLACEMENT C++ env knobs (#57440 ignores them).
#   unset / "auto" -> None: the op's cost model picks fused/phased and the fused geometry.
#   "nv1np6"       -> NV=1, NP=6, row-major (the pre-#57440 geometry; like-for-like A/B).
#   "nv1np5"       -> NV=1, NP=5, row-local (the cost-model pick on P150 13x10 at BH=16, NC=64).
#   "nv2np4"       -> NV=2, NP=4, row-local.
#   "nv2np6"       -> NV=2, NP=6, row-major.
#   "nv1np4"       -> NV=1, NP=4, row-local (11x10 grids, e.g. p300c dies in SP prefill).
#   "nv2np3"       -> NV=2, NP=3, row-local (11x10 grids).
_GDN_PCFG_GEOMETRIES = {
    "nv1np6": dict(num_producers=6, num_receivers=1, row_local=False),
    "nv1np5": dict(num_producers=5, num_receivers=1, row_local=True),
    "nv2np4": dict(num_producers=4, num_receivers=2, row_local=True),
    "nv2np6": dict(num_producers=6, num_receivers=2, row_local=False),
    # 11x10 grids (p300c dies under sequence-parallel prefill, tt/sp_prefill_sc.py): fewer producer columns.
    "nv1np4": dict(num_producers=4, num_receivers=1, row_local=True),
    "nv2np3": dict(num_producers=3, num_receivers=2, row_local=True),
}


def gdn_program_config_from_env():
    """QWEN36_GDN_PCFG -> ttnn.ChunkGdnFusedProgramConfig, or None for the op's cost-model default."""
    name = os.environ.get("QWEN36_GDN_PCFG", "auto").strip().lower()
    if name in ("", "auto"):
        return None
    if name not in _GDN_PCFG_GEOMETRIES:
        raise ValueError(f"QWEN36_GDN_PCFG={name!r}: expected auto or one of {sorted(_GDN_PCFG_GEOMETRIES)}")
    return ttnn.ChunkGdnFusedProgramConfig(**_GDN_PCFG_GEOMETRIES[name])


class Qwen36GatedDeltaNet:
    """Gated DeltaNet (linear attention) layer for Qwen3.5-9B.

    Maintains fixed-size recurrent state [B, H, K, V] that replaces the KV cache.
    Also maintains conv states [B, kernel_size-1, D] for causal conv1d history.
    Supports two modes:
      - "recurrent": single-token decode (T=1), O(1) memory
      - "chunk": multi-token prefill (T>1), chunked parallel processing
    """

    def __init__(self, mesh_device, config: GDNConfig, state_dict, tensor_cache_path=None):
        self.device = mesh_device
        self.cfg = config

        # Mirror config-derived scalar dims so the forward bodies read them directly.
        self.num_heads = config.num_heads
        self.num_v_heads = config.num_v_heads
        self.head_k_dim = config.head_k_dim
        self.head_v_dim = config.head_v_dim
        self.conv_kernel_size = config.conv_kernel_size
        self.norm_eps = config.norm_eps
        self.long_prefill_chunk_size = config.long_prefill_chunk_size

        # step2 (2026-09-22): routed through tpc.prefill_matmul_ckc() -- see QWEN36_PREFILL_MM_*
        # flags in tp_common.py. Legacy values (packer_l1_acc=False, fp32_dest_acc_en=True) unless
        # overridden.
        self.compute_kernel_config = tpc.prefill_matmul_ckc()
        self.compute_kernel_config_decode = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        self.weights = load_gdn_weights(mesh_device, config, state_dict, tensor_cache_path)

        # Native ttnn.conv1d depthwise prefill (replaces the FIR MAC fallback for chunk prefill,
        # T>512, valid_len None — E2). Set QWEN36_GDN_NATIVE_CONV1D=0 to force the FIR fallback.
        # weights.fused_conv_w1d_chunks is None when no allow-listed chunk width
        # (_NATIVE_CONV_CHUNK_WIDTHS in gdn/weights.py) evenly divides C — keep the FIR path then.
        self._native_conv1d_fn = None
        if os.environ.get("QWEN36_GDN_NATIVE_CONV1D", "1") != "0" and self.weights.fused_conv_w1d_chunks is not None:
            from models.demos.blackhole.qwen36.tt.gdn.conv1d_native import make_native_conv1d_fn

            self._native_conv1d_fn = make_native_conv1d_fn(
                mesh_device,
                self.weights.fused_conv_w1d_chunks,
                config.conv_kernel_size,
                q_dim=config.q_dim,
                k_dim=config.k_dim,
                v_dim=config.v_dim,
                # Largest T for which the n_cc=3 (width==q_dim==k_dim==v_dim) qkv-tuple fast path
                # fits L1 (measured with scripts/step1/test_conv_native.py); n_cc=2 (3072-wide,
                # always fits) is used above this. Default 0 (never): on P150 with C=6144
                # (cw=2048), the L1 overflow ("grow to 1684544 B beyond max L1 1572864 B") happens
                # identically at T=512/1024/2048 — HEIGHT_SHARDED grows the core grid with T, so
                # the per-core CB footprint depends on channel width alone, not T; there is no T
                # at which n_cc=3 fits here. QWEN36_GDN_CONV_CHUNKS overrides both n_cc and t3_max.
                t3_max=int(os.environ.get("QWEN36_GDN_CONV_T3_MAX", "0")),
            )

            # step2 (2026-09-22): ttnn.experimental.kda.qkv_causal_conv1d_silu fuses the 4-tap
            # depthwise conv + SiLU + q/k/v split into ONE device op, replacing the native
            # HEIGHT_SHARDED conv1d chunk-loop + concat/slice above for the T>1 chunk-prefill
            # path only (~18 fewer device ops/layer/chunk at T=2048 — see
            # gdn/conv1d_kda.py and qwen35_2b_handoff/scripts/step2/kda_conv_probe.py). Falls back
            # to the native fn built above for masked-tail (valid_len)/non-tile-aligned-T/T<32/
            # channel-width-mismatch calls (guards live inside conv1d_kda.fn). Decode (T=1) is
            # untouched either way — this only swaps the T>1 chunk-prefill callable.
            #   QWEN36_GDN_CONV_KDA=1 (default) — use the fused KDA op.
            #   QWEN36_GDN_CONV_KDA=0           — keep today's native conv1d path, bit-identical.
            #   QWEN36_GDN_CONV_KDA_CCS=<n>     — program_config.channel_chunk_size for the fused
            #     op (must be tile-aligned and evenly divide q_dim+k_dim+v_dim); default 768, read
            #     inside conv1d_kda.py (see its _DEFAULT_CCS comment for the L1-footprint sweep).
            #   QWEN36_GDN_CONV_KDA_FP32ACC=1/0 (default 0) — fp32-accumulate compute_kernel_config
            #     for the fused op (validate()-confirmed legal, unlike packer_l1_acc/math_approx);
            #     read inside conv1d_kda.py. Default 0 (op's own fp32_dest_acc_en=False default):
            #     measured to win on whole-model logits PCC despite fp32acc=1 winning the isolated
            #     op-level PCC comparison — see conv1d_kda.py's docstring for the numbers.
            #   QWEN36_GDN_CONV_KDA_TILED=1/0 (default 1, INT-3) — the op's tiled kernel takes the TILE
            #     in-proj output and TILE conv state directly and returns new_state (no untilize/
            #     zeros/slice/tilize glue); 0 = the ROW_MAJOR KDA path. Bit-identical; read inside
            #     conv1d_kda.py.
            if os.environ.get("QWEN36_GDN_CONV_KDA", "1") != "0" and config.conv_kernel_size == 4:
                from models.demos.blackhole.qwen36.tt.gdn.conv1d_kda import make_kda_conv1d_fn

                self._native_conv1d_fn = make_kda_conv1d_fn(
                    mesh_device,
                    self.weights.fused_conv_weight_taps,
                    config.conv_kernel_size,
                    q_dim=config.q_dim,
                    k_dim=config.k_dim,
                    v_dim=config.v_dim,
                    native_fn=self._native_conv1d_fn,
                )

        # Fused chunk-prefill constants (eye/tril/ones/quadrant masks); built once so traced prefill
        # never uploads from host. None when the fused path is disabled.
        self._fused_const_tiles = None
        if os.environ.get("QWEN36_GDN_FUSED_PREFILL", "1") != "0":
            from models.demos.blackhole.qwen36.tt.gdn.fused_chunk import build_fused_const_tiles

            # HV: only used to build the gb_flat head selector when QWEN_GDN_FLAT_GB=1 (T7).
            self._fused_const_tiles = build_fused_const_tiles(mesh_device, HV=self.num_v_heads)
        # program_config for the fused chunk-prefill op (QWEN36_GDN_PCFG, see the top of this file);
        # gdn/decode.py passes it to chunk_gated_delta_rule_fused_adapter. None = the op's cost model.
        self.gdn_program_config = gdn_program_config_from_env()

        self._prefill_progcfg_fn = tpc.make_prefill_progcfg_fn(mesh_device)
        # Decode (T==1) 1D matmul progcfg (see tp_common measured table): GDN out-proj and the
        # mega in-proj were swept; other GDN projections fall through the same shape-class lookup.
        self._decode_progcfg_fn = tpc.make_decode_progcfg_fn(mesh_device)

        # ---- Runtime state (plain instance attributes, exact same names as before;
        # poked directly by the trace machinery in model.py / qwen36_vllm.py) ----
        self.recurrent_state = None
        # Conv states: ttnn tensors on device [B, kernel_size-1, D]
        self.conv_state_q = None
        self.conv_state_k = None
        self.conv_state_v = None
        # Fused conv state [B, kernel_size-1, D_total] where D_total = q_dim + k_dim + v_dim
        self.fused_conv_state = None
        self.split_conv_state = None
        # Trace capture support
        self.use_inplace_state = False
        # When True (set during chunk-outer traced-prefill capture), the chunk (prefill)
        # path writes recurrent + conv state into the persistent external buffers IN PLACE
        # (ttnn.copy) instead of reassigning a fresh tensor, so the state carries across
        # execute_trace() replays (each replay re-runs the same baked buffer addresses).
        # Eager prefill keeps the reassign path. See Qwen36Model.capture_prefill_trace_chunked.
        self._chunk_inplace_state = False

        # QWEN36_GDN_DECODE_FUSED=2 (default on; 0 = composite): T == 1 decode runs mega linear ->
        # ttnn.experimental.kda.gdn_decode_step -> out-proj (gdn/decode_fused.py). Needs an FP32
        # recurrent state (allocated FP32 everywhere while this is on), the packed conv taps (host
        # pack, once, here) and the packed conv history `conv_hist` (rebuilt on device from
        # fused_conv_state by refresh_conv_hist after prefill / restore / reset). Flag 0: nothing
        # below is created and every path is the pre-T8 one.
        self._decode_fused = False
        self.conv_taps_packed = None
        self.conv_hist = None
        from models.demos.blackhole.qwen36.tt.gdn import decode_fused as _df

        if _df.decode_fused_enabled():
            ok, why = _df.fused_supported(config, self.weights)
            if ok:
                self._decode_fused = True
                self.conv_taps_packed = _df.pack_conv_taps(config, self.weights.fused_conv_weight_taps, mesh_device)
            else:
                _df.warn_unsupported(why)

    def ensure_conv_hist(self):
        """Allocate the packed conv history once (outside any trace; its address is baked into the decode trace)."""
        from models.demos.blackhole.qwen36.tt.gdn import decode_fused as _df

        if self._decode_fused and self.conv_hist is None:
            self.conv_hist = _df.alloc_conv_hist(self.cfg, self.device)
        return self.conv_hist

    def refresh_conv_hist(self):
        """Rebuild conv_hist from fused_conv_state on device (eager, trace safe: no host reads)."""
        from models.demos.blackhole.qwen36.tt.gdn import decode_fused as _df

        if not self._decode_fused:
            return
        self.ensure_conv_hist()
        if self.fused_conv_state is None:
            ttnn.copy(ttnn.zeros_like(self.conv_hist), self.conv_hist)
            return
        _df.repack_conv_hist(self.fused_conv_state, self.conv_hist, self.cfg)

    def forward(self, x, mode="recurrent", chunk_size=None, valid_len=None):
        return recurrent_forward(self, x, mode=mode, chunk_size=chunk_size, valid_len=valid_len)

    def set_external_state(self, recurrent_state, conv_state):
        """Point layer at externally-allocated state buffers.
        Sets use_inplace_state=True so all forward passes write state inplace (preserving buffer addresses).
        Does NOT create split_conv_state — that happens after prefill when there is real data to split.
        """
        expected_rec = [1, self.num_v_heads, self.head_k_dim, self.head_v_dim]
        assert (
            list(recurrent_state.shape) == expected_rec
        ), f"recurrent_state shape mismatch: {list(recurrent_state.shape)} != {expected_rec}"
        assert (
            conv_state.shape[1] == self.conv_kernel_size - 1
        ), f"conv_state dim 1 mismatch: {conv_state.shape[1]} != {self.conv_kernel_size - 1}"
        self.recurrent_state = recurrent_state
        self.fused_conv_state = conv_state
        self.use_inplace_state = True

    def _restore_split_conv_from_fused(self):
        """Copy fused_conv_state slices into existing split_conv_state buffers.
        Preserves device addresses (critical for trace replay).
        Kept as a method because model.py calls it on the instance.
        """
        restore_split_conv_from_fused(self)

    def reset_state(self, batch_size=None):
        if batch_size is not None:
            init_recurrent_state(self, batch_size)
        else:
            self.recurrent_state = None
        self.conv_state_q = None
        self.conv_state_k = None
        self.conv_state_v = None
        self.fused_conv_state = None
        self.split_conv_state = None
