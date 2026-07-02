# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tensor-parallel (TP>1) Gated DeltaNet decode for Qwen3.5.

The delta-rule recurrence is per-value-head, so TP shards by value heads with NO
cross-device comms inside the recurrence — only an all-reduce after the
row-parallel output projection. The recurrence itself reuses the validated
composed-op `recurrent_gated_delta_rule_decode_ttnn` from the experimental
backend (it does L2-norm + scale + the delta step), driven on each device's
local heads. Projections/conv are sharded; weights kept interleaved (auto matmul).

GDN output uses the *gated* RMSNorm (weight, NO +1) followed by a SiLU(z) gate — distinct from the +1 QK/layer norms.
"""
import os

import torch

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import (
    recurrent_gated_delta_rule_decode_ttnn,
)
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_seq import (
    chunk_gated_delta_rule_seq_adapter,
    create_chunk_masks_seq,
)
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_deltanet import _causal_conv1d_fir
from models.tt_transformers.tt.ccl import tt_all_reduce


def load_gdn_weights_tp(mesh, sd, args, cache_dir=None):
    """Shard one GDN layer's RAW linear_attn.* weights across the mesh.

    sd keys (raw, from FP8 loader): linear_attn.in_proj_qkv/in_proj_z/in_proj_a/
    in_proj_b/out_proj/conv1d/A_log/dt_bias/norm.weight.
    """
    tp = args.num_devices
    nk, dk, nv, dv = args.gdn_nk, args.gdn_dk, args.gdn_nv, args.gdn_dv
    key_dim, value_dim = args.gdn_key_dim, args.gdn_value_dim
    qkv_per = args.gdn_qkv_dim_tp
    z_per = args.gdn_z_dim_tp
    nv_per = args.gdn_nv_tp

    if cache_dir is not None:
        import os

        os.makedirs(cache_dir, exist_ok=True)

    def c(n):
        return str(cache_dir / n) if cache_dir is not None else None

    # Keys may arrive in three layouts depending on the loader/path:
    #   - raw FP8 loader, unstripped:          linear_attn.in_proj_qkv.weight / linear_attn.conv1d.weight
    #   - raw FP8 loader, substate()-stripped: in_proj_qkv.weight            / conv1d.weight
    #   - bf16 remap (remap_qwen36_state_dict): qkv_proj.weight (renamed) +
    #     q_conv/k_conv/v_conv.weight (conv1d split per Q/K/V stream)
    # The optional `linear_attn.` prefix is detected; the fused QKV tensor is the
    # same under either name, and the fused conv1d is the concat of the three
    # per-stream conv weights in [Q, K, V] order (the exact split remap applied).
    P = "linear_attn." if any(k.startswith("linear_attn.") for k in sd) else ""

    def first_key(*names):
        for n in names:
            if (P + n) in sd:
                return sd[P + n]
        raise KeyError(f"none of {[P + n for n in names]} found in GDN state dict")

    # ---- fused QKV+Z (column-parallel) ----
    qkv_w = first_key("in_proj_qkv.weight", "qkv_proj.weight")
    if (P + "conv1d.weight") in sd:
        conv1d_w = sd[P + "conv1d.weight"]
    else:  # bf16 remap split conv1d into per-stream q/k/v; reassemble the fused weight
        conv1d_w = torch.cat([sd[P + "q_conv.weight"], sd[P + "k_conv.weight"], sd[P + "v_conv.weight"]], dim=0)
    qkv_re = tpc.prepare_gdn_qkv(qkv_w, key_dim, value_dim, nk, dk, nv, dv, tp)
    z_w = sd[P + "in_proj_z.weight"]
    a_w, b_w = sd[P + "in_proj_a.weight"], sd[P + "in_proj_b.weight"]
    tw = {}
    # Column-parallel qkvz in-projection (the large GDN input matmul) stored DRAM-WIDTH_SHARDED so
    # the M=1 decode matmul reads it at near-peak DRAM bandwidth (mirrors MLP w1/w3). On by default
    # for TP (the memcfg exists). Distinct `.dramshard` cache (ttnn.as_tensor reloads a cache as-is,
    # ignoring memory_config). `out` stays interleaved.
    qkvz_sharded = getattr(args, "gdn_qkvz_weight_memcfg", None) is not None
    # Fold the tiny a/b (decay/beta) projection into qkvz so ONE column-parallel matmul produces
    # [qkv|z|a|b] — eliminates a whole N=2*nv_per decode matmul (the separate `ab` below), K=dim
    # unchanged so the qkvz matmul stays efficient. Per-device block order [qkv|z|a|b] matches the
    # slices in _project_qkvzab. Default on whenever the qkvz weight is DRAM-sharded (TP); the
    # interleaved single-device fallback below keeps the separate ab matmul.
    fuse_ab = qkvz_sharded
    if fuse_ab:
        fused = torch.cat(
            [
                torch.cat(
                    [
                        qkv_re[d * qkv_per : (d + 1) * qkv_per],
                        z_w[d * z_per : (d + 1) * z_per],
                        a_w[d * nv_per : (d + 1) * nv_per],
                        b_w[d * nv_per : (d + 1) * nv_per],
                    ],
                    dim=0,
                )
                for d in range(tp)
            ],
            dim=0,
        )
        tw["qkvz"] = tpc.shard_w(
            fused,
            mesh,
            dim=-1,
            memory_config=args.gdn_qkvzab_weight_memcfg,
            cache_path=c("qkvzab.dramshard"),
            dtype=ttnn.bfloat8_b,
        )
    else:
        fused = torch.cat(
            [
                torch.cat([qkv_re[d * qkv_per : (d + 1) * qkv_per], z_w[d * z_per : (d + 1) * z_per]], dim=0)
                for d in range(tp)
            ],
            dim=0,
        )
        qkvz_mc = args.gdn_qkvz_weight_memcfg if qkvz_sharded else ttnn.DRAM_MEMORY_CONFIG
        tw["qkvz"] = tpc.shard_w(
            fused,
            mesh,
            dim=-1,
            memory_config=qkvz_mc,
            cache_path=c("qkvz" + (".dramshard" if qkvz_sharded else "")),
            dtype=ttnn.bfloat8_b,
        )
        # ---- separate A+B (column-parallel), per-device [a(nv_per), b(nv_per)] ----
        ab = torch.cat(
            [
                torch.cat([a_w[d * nv_per : (d + 1) * nv_per], b_w[d * nv_per : (d + 1) * nv_per]], dim=0)
                for d in range(tp)
            ],
            dim=0,
        )
        tw["ab"] = tpc.shard_w(
            ab, mesh, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG, cache_path=c("ab"), dtype=ttnn.bfloat8_b
        )
    # ---- out projection (row-parallel) ----
    tw["out"] = tpc.shard_w(
        sd[P + "out_proj.weight"],
        mesh,
        dim=0,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_path=c("out"),
        dtype=ttnn.bfloat8_b,
    )
    # ---- per-head params ----
    tw["dt_bias"] = tpc.shard_small(sd[P + "dt_bias"].float(), mesh, c("dt_bias"))
    A_log = tpc.shard_small(sd[P + "A_log"].float(), mesh, c("A_log"))
    tw["neg_exp_A"] = ttnn.neg(ttnn.exp(A_log))
    tw["norm_w"] = tpc.replicate(sd[P + "norm.weight"].float(), mesh, c("norm_w"))
    # ---- conv taps (4), sharded per Q/K/V head grouping ----
    taps = tpc.prepare_conv_taps(conv1d_w, key_dim, nk, dk, nv, dv, args.gdn_conv_kernel_size, tp)
    tw["conv_taps"] = [tpc.shard_small(taps[j], mesh, c(f"tap{j}")) for j in range(args.gdn_conv_kernel_size)]
    return tw


class TPGatedDeltaNet:
    """Standalone TP GDN decode (per-device value-head recurrence + all-reduce)."""

    def __init__(self, mesh, args, tw, tt_ccl):
        self.mesh = mesh
        self.args = args
        self.tw = tw
        self.tt_ccl = tt_ccl
        self.B = args.max_batch_size
        self.Nk = args.gdn_nk_tp
        self.Nv = args.gdn_nv_tp
        self.Dk = args.gdn_dk
        self.Dv = args.gdn_dv
        self.qkv_dim_tp = args.gdn_qkv_dim_tp
        self.qkvz_dim_tp = args.gdn_qkvz_dim_tp
        self.key_dim_tp = args.gdn_key_dim_tp
        self.value_dim_tp = args.gdn_value_dim_tp
        self.K = args.gdn_conv_kernel_size
        self.scale = self.Dk**-0.5
        self.cfg = tpc.COMPUTE_HIFI2
        # Mirror the weight-load gate: qkvz stored DRAM-WIDTH_SHARDED → DRAM-sharded decode matmul
        # (see _col_proj). Must match qkvz_sharded in load_gdn_weights_tp.
        self._dram_sharded = getattr(args, "gdn_qkvz_weight_memcfg", None) is not None
        # The qkvz weight is the fused [qkv|z|a|b]; one matmul replaces qkvz + ab. Default on
        # whenever the qkvz weight is DRAM-sharded (TP). Must match `fuse_ab` in load_gdn_weights_tp.
        self._fuse_ab = self._dram_sharded
        # Pre-build the chunk-prefill masks ONCE (replicated across the mesh) so the seq kernel
        # reads them from cache instead of rebuilding eye/triu/tril via from_torch on every call.
        # The from_torch fallback is a host write that TT_FATALs inside the captured chunk-outer
        # trace; pre-allocating here (outside any trace) makes the GDN prefill trace-safe. Mirrors
        # the single-device gdn/weights.py create_chunk_masks_seq usage.
        self.chunk_seq_masks = create_chunk_masks_seq(args.gdn_chunk_size, mesh)
        self.conv_states = None
        self.rec_state = None
        # When True, decode/prefill update rec_state IN PLACE (ttnn.copy into the fixed
        # buffer) instead of reassigning — required for the decode trace (a trace bakes in
        # tensor addresses). Set by the model's TP allocate_kv_caches; the demo path leaves
        # it False (reassign), so the demo's behavior is unchanged.
        self._stable_state = False
        self.conv_carry = None  # [1, K-1, qkv_dim_tp] cross-chunk prefill conv carry
        # Persistent zero-source buffers, allocated in reset_state (always BEFORE any trace is
        # parked). reset_state_inplace copies from these so it allocates NOTHING — allocating a
        # device buffer while the chunk-outer prefill trace is parked is unsafe (see
        # reset_state_inplace). _zero_conv0 doubles as the trace-safe zero source for
        # conv_states[0] in capture_state.
        self._zero_conv0 = None  # zero source, shape (1, B, qkv_dim_tp)
        self._zero_conv_carry = None  # zero source, shape (1, K-1, qkv_dim_tp)
        self._zero_rec = None  # zero source, shape (B, Nv, Dk, Dv)

    def reset_state(self):
        def z(shape):
            return ttnn.from_torch(
                torch.zeros(*shape, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )

        self.conv_states = [z((1, self.B, self.qkv_dim_tp)) for _ in range(self.K)]
        # fp32 recurrent state by DEFAULT: per-step decode noise compounds through the state
        # carry over hundreds of decode steps; bf16 carry measurably degrades long generations.
        # QWEN35_GDN_STATE_BF16=1 reverts to bf16 (memory A/B).
        if os.environ.get("QWEN35_GDN_STATE_BF16") != "1":  # fp32 DEFAULT (decode-drift mitigation)
            self.rec_state = ttnn.from_torch(
                torch.zeros(self.B, self.Nv, self.Dk, self.Dv, dtype=torch.float32),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )
        else:
            self.rec_state = z((self.B, self.Nv, self.Dk, self.Dv))
        # Cross-chunk prefill carry buffers (chunk-outer trace): conv_carry holds the last
        # K-1 conv inputs of the previous chunk (= conv_new_state, fed as the next chunk's
        # left context); _zero_conv0 is a persistent zero source for conv_states[0] so the
        # capture_state writeback stays trace-safe (no host->device transfer inside a trace).
        self.conv_carry = z((1, self.K - 1, self.qkv_dim_tp))
        # Persistent zero sources for trace-safe in-place resets (reset_state_inplace copies
        # from these, never allocating). Created here because reset_state always runs during
        # warmup/setup, before any trace is parked. All bf16 like the conv/rec buffers; ttnn.copy
        # converts into the (possibly fp32) rec_state, exactly as the old transient zr did.
        self._zero_conv0 = z((1, self.B, self.qkv_dim_tp))
        self._zero_conv_carry = z((1, self.K - 1, self.qkv_dim_tp))
        self._zero_rec = z((self.B, self.Nv, self.Dk, self.Dv))

    def reset_state_inplace(self):
        """Zero conv + recurrent state IN PLACE (preserves buffer addresses for tracing).

        Mirrors qwen35_27b gdn.py reset_state_inplace — used between sequences / trace
        replays so the captured decode/prefill traces' baked addresses stay valid.

        Allocates NOTHING: every zero is copied from the persistent _zero_* source buffers
        created in reset_state (which always runs before a trace is parked). This is required
        for correctness, not just tidiness — once the chunk-outer prefill trace is parked, the
        per-sequence reset still runs (via _reset_gdn_state_for_new_sequence), and allocating a
        device buffer while a trace is active is unsafe: the allocator can hand back an address
        inside the trace's reserved region, which the trace then corrupts on replay. At 256k,
        where the 4096-block KV cache leaves DRAM nearly full, those transient reset buffers were
        forced into the trace region and the corruption surfaced as a bus error during the
        per-chunk replays. (Single-device _reset_dn_state_inplace uses the same copy-from-
        preallocated-zero pattern, which is why it never hit this.)
        """
        if self.conv_states is None:
            self.reset_state()
            return
        # reset_state always runs first (it sets conv_states), so the zero sources exist. Assert
        # rather than lazily allocate — a lazy allocation here would reintroduce the active-trace
        # allocation this method exists to avoid.
        assert (
            self._zero_conv0 is not None and self._zero_conv_carry is not None and self._zero_rec is not None
        ), "zero sources missing; reset_state must run before reset_state_inplace"
        for cs in self.conv_states:
            ttnn.copy(self._zero_conv0, cs)
        ttnn.copy(self._zero_rec, self.rec_state)
        # Zero the cross-chunk conv carry so each new sequence starts from scratch.
        ttnn.copy(self._zero_conv_carry, self.conv_carry)

    def _col_proj(self, x, weight, decode_progcfg):
        """Column-parallel qkvz in-projection. DRAM-sharded decode matmul when enabled, else the
        validated plain interleaved ttnn.linear. Output is DRAM-interleaved either way so the
        downstream slice into qkv/z is unchanged."""
        if not self._dram_sharded:
            return ttnn.linear(x, weight, compute_kernel_config=self.cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return tpc.sharded_decode_matmul(
            x,
            weight,
            self.cfg,
            decode_progcfg,
            self.args.act_shard_hidden,
            self.args.prefill_progcfg,
            self.args.dim,
        )

    def _project_qkvzab(self, x, S):
        """Project x -> (qkv, z, a, b). Fused (P1-gdn-fuse-ab): ONE column-parallel matmul over the
        [qkv|z|a|b] weight, then slice — folds the tiny separate `ab` matmul (N=2*nv_per) into qkvz.
        Default: the qkvz matmul plus a separate `ab` matmul (the validated baseline). S is the
        seq/batch dim (B in decode, T in prefill). Output slices are DRAM-interleaved either way."""
        Nv, qz, az = self.Nv, self.qkv_dim_tp, self.qkvz_dim_tp
        if self._fuse_ab:
            qkvzab = self._col_proj(x, self.tw["qkvz"], self.args.gdn_qkvzab_progcfg)
            qkv = ttnn.slice(qkvzab, (0, 0, 0), (1, S, qz))
            z = ttnn.slice(qkvzab, (0, 0, qz), (1, S, az))
            a = ttnn.slice(qkvzab, (0, 0, az), (1, S, az + Nv))
            b = ttnn.slice(qkvzab, (0, 0, az + Nv), (1, S, az + 2 * Nv))
            ttnn.deallocate(qkvzab)
            return qkv, z, a, b
        qkvz = self._col_proj(x, self.tw["qkvz"], self.args.gdn_qkvz_progcfg)
        qkv = ttnn.slice(qkvz, (0, 0, 0), (1, S, qz))
        z = ttnn.slice(qkvz, (0, 0, qz), (1, S, az))
        ttnn.deallocate(qkvz)
        ab = ttnn.linear(x, self.tw["ab"], compute_kernel_config=self.cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        a = ttnn.slice(ab, (0, 0, 0), (1, S, Nv))
        b = ttnn.slice(ab, (0, 0, Nv), (1, S, 2 * Nv))
        ttnn.deallocate(ab)
        return qkv, z, a, b

    def forward_prefill(self, x, chunk_size=128, valid_len=None, capture_state=False):
        """Causal chunk-prefill over a sequence (from scratch, zero init state).

        x: [1,1,T,dim] replicated. Reuses the shared gated_delta_attn_seq kernel
        (via chunk_gated_delta_rule_seq_adapter) and the FIR conv on per-device
        value heads. Output fractured along dim=3.

        valid_len: number of REAL tokens (< T means the rest is right-padding);
        the recurrence + conv state are masked to reflect exactly valid_len tokens.
        capture_state: when True, store the final recurrent state + the last K-1
        real conv inputs into self.rec_state / self.conv_states so decode continues.
        """
        tw, Nk, Nv, Dk, Dv = self.tw, self.Nk, self.Nv, self.Dk, self.Dv
        if len(x.shape) == 4:
            x = ttnn.reshape(x, (1, x.shape[-2], x.shape[-1]))  # [1,T,dim]
        T = x.shape[1]
        # Pass the RAW valid_len (may be None) to the conv-FIR / seq kernels below — NOT a
        # `valid_len or T` coercion. A full chunk (valid_len is None) must take the kernels'
        # valid_len-None path (a static last-(K-1) slice for the conv state), which is trace-safe;
        # the valid_len-set path builds a one-hot via ttnn.from_torch (a host write) that TT_FATALs
        # ("Writes are not supported during trace capture") inside the captured chunk-outer trace.
        # Masked buckets still pass a real valid_len (< T) so their exact masking is unchanged, and
        # for a full chunk the None slice and the valid_len==T one-hot select the identical rows.

        # Cross-chunk carry (chunk-outer prefill): when _stable_state, the recurrent + conv
        # state continue from the persistent buffers (zeroed at sequence start by
        # reset_state_inplace, so a from-scratch single pass reads zeros == None). The demo
        # path (_stable_state False) is unchanged: no carry, reassign state.
        carry = self._stable_state
        if carry and self.conv_carry is None:
            self.reset_state()

        qkv, z, a, b = self._project_qkvzab(x, T)

        # FIR causal conv1d + SiLU over the sequence. conv_state = the carried last K-1 conv
        # inputs of the previous chunk (left context); zero == None for a from-scratch pass.
        # valid_len makes new_state capture the last K-1 REAL tokens (not padding).
        conv, conv_new_state = _causal_conv1d_fir(
            qkv,
            None,
            None,
            self.K,
            self.mesh,
            # OPT (prefill): run the conv MAC in L1 ([1,T,2560]~=10.5MB @T=2048 -> ~95KB/core). The conv
            # output is freed before the chunk kernel, so no CB clash. _causal_conv1d_fir lands new_state
            # (which IS alive across the kernel) in DRAM internally to avoid the clash. Was DRAM.
            memory_config=ttnn.L1_MEMORY_CONFIG,
            conv_state=self.conv_carry if carry else None,
            weight_taps=tw["conv_taps"],
            bias_dev=None,
            valid_len=valid_len,
        )
        ttnn.deallocate(qkv)

        # NOTE: q/k/v (and beta/g below) must stay DRAM — they are held by this frame's locals and
        # passed into the adapter, so they are ALIVE across the gated_delta_attn_seq kernel call. Putting
        # them in L1 clashes with the kernel's static circular buffers (verified OOM). Unlike out_f, there
        # is no transient tail here: the relayout OUTPUT itself must survive to the kernel.
        kd = self.key_dim_tp
        q = ttnn.reshape(ttnn.slice(conv, (0, 0, 0), (1, T, kd)), (1, T, Nk, Dk))
        k = ttnn.reshape(ttnn.slice(conv, (0, 0, kd), (1, T, 2 * kd)), (1, T, Nk, Dk))
        v = ttnn.reshape(ttnn.slice(conv, (0, 0, 2 * kd), (1, T, self.qkv_dim_tp)), (1, T, Nv, Dv))
        ttnn.deallocate(conv)
        # GQA late-expand (default): leave q/k at Nk heads here — chunk_gated_delta_rule_seq_adapter
        # L2-norms + transforms them at Nk and expands to Nv AFTER (a cheap dim-0 block-repeat),
        # doing ~3x less q/k transform work than pre-expanding to Nv before the L2-norm + permute.

        # beta/g are also adapter inputs held across the kernel -> keep DRAM (same reason as q/k/v).
        beta = ttnn.reshape(ttnn.sigmoid(b), (1, T, Nv))
        ttnn.deallocate(b)
        g = ttnn.reshape(ttnn.multiply(tw["neg_exp_A"], ttnn.softplus(ttnn.add(a, tw["dt_bias"]))), (1, T, Nv))
        ttnn.deallocate(a)

        o, final_state = chunk_gated_delta_rule_seq_adapter(
            q,
            k,
            v,
            beta,
            g,
            chunk_size=chunk_size,
            scale=self.scale,
            initial_state=self.rec_state if carry else None,
            device=self.mesh,
            cached_masks=self.chunk_seq_masks,
            valid_len=valid_len,
        )
        B, D = 1, self.qkv_dim_tp
        # ---- Carry recurrent + conv state for the NEXT chunk (chunk-outer prefill). ----
        # In place (ttnn.copy) when _stable_state so the addresses the prefill/decode traces
        # baked in stay valid across execute_trace replays and across sequences.
        if carry:
            ttnn.copy(final_state, self.rec_state)
            ttnn.deallocate(final_state)
            ttnn.copy(conv_new_state, self.conv_carry)  # [1, K-1, D] last-K-1 conv inputs
        else:
            self.rec_state = final_state
        # ---- Finalize the decode conv window (last chunk / short prompt). ----
        # conv_states[1..K-1] = the last K-1 real conv inputs; [0] is the (shifted-out) zero.
        # Harmless to refresh every chunk — the last chunk's values are the ones decode reads.
        if capture_state:
            if self.conv_states is None:
                self.reset_state()
            if self._zero_conv0 is not None:
                ttnn.copy(self._zero_conv0, self.conv_states[0])
            else:
                zero = ttnn.from_torch(
                    torch.zeros(1, B, D, dtype=torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
                )
                ttnn.copy(zero, self.conv_states[0])
                ttnn.deallocate(zero)
            for j in range(self.K - 1):
                src = ttnn.reshape(ttnn.slice(conv_new_state, (0, j, 0), (1, j + 1, D)), (1, B, D))
                ttnn.copy(src, self.conv_states[j + 1])
        ttnn.deallocate(conv_new_state)
        # o: [1, T, Nv, Dv] -> gated RMSNorm over Dv + SiLU(z) gate
        # OPT (prefill): run the post-kernel norm + the head-flatten reshape (~1.4ms, the biggest single
        # relayout) in L1. These are freed before the out-proj, so no GDN-kernel CB clash. But LAND
        # `gated` in DRAM: it's the out-proj matmul's activation, alive across that matmul's CBs.
        _L1 = ttnn.L1_MEMORY_CONFIG
        out_n = ttnn.rms_norm(o, weight=tw["norm_w"], epsilon=1e-6, memory_config=_L1)
        ttnn.deallocate(o)
        out_f = ttnn.reshape(out_n, (1, T, self.value_dim_tp), memory_config=_L1)
        ttnn.deallocate(out_n)
        gated = ttnn.multiply(out_f, ttnn.silu(z, memory_config=_L1), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out_f)
        ttnn.deallocate(z)
        partial = ttnn.linear(gated, tw["out"], compute_kernel_config=self.cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(gated)
        partial = ttnn.reshape(partial, (1, 1, T, partial.shape[-1]))
        return tt_all_reduce(
            partial,
            self.mesh,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=self.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward_decode(self, x):
        tw, B, Nk, Nv, Dk, Dv = self.tw, self.B, self.Nk, self.Nv, self.Dk, self.Dv
        _L1 = ttnn.L1_MEMORY_CONFIG  # OPT (decode): keep the conv->recurrence->norm/gate chain L1-resident
        if self.conv_states is None:
            self.reset_state()
        if len(x.shape) == 4:
            x = ttnn.reshape(x, (1, x.shape[-2], x.shape[-1]))  # [1,B,dim]

        qkv, z, a, b = self._project_qkvzab(x, B)

        # conv1d shift-register (copy(src, dst): dst is 2nd arg) + weighted sum + silu
        st = self.conv_states
        for j in range(self.K - 1):
            ttnn.copy(st[j + 1], st[j])
        ttnn.copy(qkv, st[self.K - 1])
        ttnn.deallocate(qkv)
        conv = ttnn.multiply(st[0], tw["conv_taps"][0], memory_config=_L1)
        for j in range(1, self.K):
            conv = ttnn.mac(st[j], tw["conv_taps"][j], conv)
        conv = ttnn.silu(conv, memory_config=_L1)

        kd = self.key_dim_tp
        q = ttnn.reshape(ttnn.slice(conv, (0, 0, 0), (1, B, kd)), (B, Nk, Dk))
        k = ttnn.reshape(ttnn.slice(conv, (0, 0, kd), (1, B, 2 * kd)), (B, Nk, Dk))
        v = ttnn.reshape(ttnn.slice(conv, (0, 0, 2 * kd), (1, B, self.qkv_dim_tp)), (B, Nv, Dv))
        ttnn.deallocate(conv)

        # expand Q/K from Nk to Nv heads (GQA); recurrence fn L2-norms + scales internally
        rf = Nv // Nk
        q = ttnn.repeat_interleave(q, rf, dim=1)
        k = ttnn.repeat_interleave(k, rf, dim=1)
        q = ttnn.reshape(q, (B, 1, Nv, Dk))
        k = ttnn.reshape(k, (B, 1, Nv, Dk))
        v = ttnn.reshape(v, (B, 1, Nv, Dv))

        beta = ttnn.reshape(ttnn.sigmoid(b, memory_config=_L1), (B, 1, Nv))
        ttnn.deallocate(b)
        g = ttnn.multiply(tw["neg_exp_A"], ttnn.softplus(ttnn.add(a, tw["dt_bias"])), memory_config=_L1)
        ttnn.deallocate(a)
        g = ttnn.reshape(g, (B, 1, Nv))

        # fp32 decode step DEFAULT (decode-drift mitigation). The per-step gated-delta recurrence
        # h = decay*h + beta*(k⊗delta) compounds every token; in bf16, `decay = exp(g)` (near 1.0)
        # quantizes coarsely and the error accumulates over a long decode → late-generation
        # repetition collapse. high_precision runs the whole step in fp32 (pairs with the fp32
        # rec_state default). QWEN35_GDN_DECODE_BF16=1 reverts to the bf16 step.
        o, new_rec = recurrent_gated_delta_rule_decode_ttnn(
            q,
            k,
            v,
            beta,
            g,
            scale=self.scale,
            initial_state=self.rec_state,
            device=self.mesh,
            high_precision=(os.environ.get("QWEN35_GDN_DECODE_BF16") != "1"),
        )
        if self._stable_state:
            # In-place update keeps rec_state's address fixed so the decode trace
            # (captured at pos 0) replays correctly across steps and sequences.
            ttnn.copy(new_rec, self.rec_state)
            ttnn.deallocate(new_rec)
        else:
            self.rec_state = new_rec

        out_r = ttnn.reshape(o, (B, Nv, Dv))
        out_n = ttnn.rms_norm(
            out_r, weight=tw["norm_w"], epsilon=1e-6, memory_config=_L1
        )  # gated norm: weight only (no +1)
        ttnn.deallocate(out_r)
        out_f = ttnn.reshape(out_n, (1, B, self.value_dim_tp))
        ttnn.deallocate(out_n)
        gated = ttnn.multiply(out_f, ttnn.silu(z, memory_config=_L1), memory_config=_L1)
        ttnn.deallocate(out_f)
        ttnn.deallocate(z)

        partial = ttnn.linear(gated, tw["out"], compute_kernel_config=self.cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(gated)
        partial = ttnn.reshape(partial, (1, 1, B, partial.shape[-1]))
        return tt_all_reduce(
            partial,
            self.mesh,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=self.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
