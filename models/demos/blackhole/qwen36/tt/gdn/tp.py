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
    l2_norm_ttnn,
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
    fused = torch.cat(
        [
            torch.cat([qkv_re[d * qkv_per : (d + 1) * qkv_per], z_w[d * z_per : (d + 1) * z_per]], dim=0)
            for d in range(tp)
        ],
        dim=0,
    )
    tw = {}
    tw["qkvz"] = tpc.shard_w(
        fused, mesh, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG, cache_path=c("qkvz"), dtype=ttnn.bfloat8_b
    )
    # ---- fused A+B (column-parallel), per-device [a(nv_per), b(nv_per)] ----
    a_w, b_w = sd[P + "in_proj_a.weight"], sd[P + "in_proj_b.weight"]
    ab = torch.cat(
        [torch.cat([a_w[d * nv_per : (d + 1) * nv_per], b_w[d * nv_per : (d + 1) * nv_per]], dim=0) for d in range(tp)],
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
        self._zero_conv0 = None  # persistent zero source for conv_states[0] (trace-safe)
        # Batched-decode seeding (Step 1): when B>1, each of the B sequences is
        # prefilled independently and its captured recurrent/conv state is stashed
        # per batch slot here (kept ON DEVICE — the state's per-device head shards
        # differ across the mesh, so a host round-trip would collapse the mesh dim),
        # then concatenated into the [B,...] decode buffers by finalize_seed().
        # Seeding happens once per sequence, not in the decode hot loop.
        self._seed_rec = {}  # slot -> device rec state  [1, Nv, Dk, Dv]
        self._seed_conv = {}  # slot -> list of K-1 device conv windows, each [1, 1, qkv_dim_tp]

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
        self._zero_conv0 = z((1, self.B, self.qkv_dim_tp))

    def reset_state_inplace(self):
        """Zero conv + recurrent state IN PLACE (preserves buffer addresses for tracing).

        Mirrors qwen35_27b gdn.py reset_state_inplace — used between sequences / trace
        replays so the captured decode trace's baked addresses stay valid.
        """
        if self.conv_states is None:
            self.reset_state()
            return

        def z(shape):
            return ttnn.from_torch(
                torch.zeros(*shape, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )

        if self.conv_carry is None:
            # Older state created before the carry buffers existed; allocate them once.
            self.conv_carry = z((1, self.K - 1, self.qkv_dim_tp))
            self._zero_conv0 = z((1, self.B, self.qkv_dim_tp))
        zc = z((1, self.B, self.qkv_dim_tp))
        for cs in self.conv_states:
            ttnn.copy(zc, cs)
        ttnn.deallocate(zc)
        zr = z((self.B, self.Nv, self.Dk, self.Dv))
        ttnn.copy(zr, self.rec_state)
        ttnn.deallocate(zr)
        # Zero the cross-chunk conv carry so each new sequence starts from scratch.
        zcc = z((1, self.K - 1, self.qkv_dim_tp))
        ttnn.copy(zcc, self.conv_carry)
        ttnn.deallocate(zcc)

    def finalize_seed(self, batch_size):
        """Assemble the per-slot stashed prefill state (see forward_prefill seed_slot)
        into the [B,...] decode buffers. Call once after all `batch_size` sequences
        have been prefilled with their seed_slot. Clears the stash.

        Everything stays on device: ttnn.concat runs per-device, so each device's own
        GDN head shard is preserved (the recurrent/conv state is NOT replicated — it is
        computed independently per device). Concatenation is along the batch (slot) axis.
        """
        assert len(self._seed_rec) == batch_size, f"seeded {len(self._seed_rec)} slots, expected {batch_size}"
        if self.conv_states is None:
            self.reset_state()
        # Recurrent state: [B, Nv, Dk, Dv] over the batch axis (dim=0). Match the demo
        # semantics (self.rec_state = final_state) by reassigning; copy in place only when
        # the buffer address must stay fixed (trace path, Step 2).
        slots = [self._seed_rec[b] for b in range(batch_size)]
        rec = slots[0] if batch_size == 1 else ttnn.concat(slots, dim=0)
        if self._stable_state:
            ttnn.copy(rec, self.rec_state)  # data → fixed buffer; rec itself no longer needed
            ttnn.deallocate(rec)
        else:
            self.rec_state = rec  # rec becomes the buffer
        for s in slots:
            if s is not rec and s is not self.rec_state:
                ttnn.deallocate(s)
        # Conv window: conv_states[0] = zero; conv_states[j+1] = the j-th window over all slots.
        D = self.qkv_dim_tp
        zc = ttnn.from_torch(
            torch.zeros(1, batch_size, D, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        ttnn.copy(zc, self.conv_states[0])
        ttnn.deallocate(zc)
        for j in range(self.K - 1):
            wins = [self._seed_conv[b][j] for b in range(batch_size)]  # each [1, 1, D]
            if batch_size == 1:
                win = wins[0]
            else:
                # Concat along the batch axis (dim=1). That is a TILE sub-dim, so detour
                # through ROW_MAJOR where a size-1 stack is unambiguous, then back to TILE.
                rm = [ttnn.to_layout(w, ttnn.ROW_MAJOR_LAYOUT) for w in wins]
                win_rm = ttnn.concat(rm, dim=1)  # [1, B, D] row-major
                win = ttnn.to_layout(win_rm, ttnn.TILE_LAYOUT)
                ttnn.deallocate(win_rm)
                for r in rm:
                    ttnn.deallocate(r)
            ttnn.copy(win, self.conv_states[j + 1])
            if win is not wins[0]:
                ttnn.deallocate(win)
            for w in wins:
                ttnn.deallocate(w)
        self._seed_rec.clear()
        self._seed_conv.clear()

    def reseed_slot(self, b):
        """Replace ONLY batch lane b's recurrent + conv state with the sequence stashed
        for slot b (a prior forward_prefill(seed_slot=b)), leaving the other B-1 lanes'
        running state untouched. In-place (ttnn.copy into the fixed buffers) so a captured
        decode trace's baked addresses survive — this is the continuous-batching join.

        Slice-per-lane + concat (per-device, so each device keeps its own head shard),
        replacing lane b with the seed. Clears the slot's stash.
        """
        assert b in self._seed_rec, f"call forward_prefill(seed_slot={b}) before reseed_slot({b})"
        B = self.B

        def _match(t, ref):
            # After decode steps the running buffers may be a different dtype (fp32 recurrent
            # default) than the freshly-prefilled seed pieces; concat requires uniform dtype.
            return t if t.dtype == ref.dtype else ttnn.typecast(t, ref.dtype)

        # Recurrent state [B, Nv, Dk, Dv]: replace lane b along dim 0.
        seed_rec = _match(self._seed_rec[b], self.rec_state)
        lanes = [
            seed_rec if i == b else ttnn.slice(self.rec_state, (i, 0, 0, 0), (i + 1, self.Nv, self.Dk, self.Dv))
            for i in range(B)
        ]
        new = lanes[0] if B == 1 else ttnn.concat(lanes, dim=0)
        ttnn.copy(new, self.rec_state)
        if new is not lanes[0]:
            ttnn.deallocate(new)
        for i, s in enumerate(lanes):
            if i != b:  # slices we made; the seed (i==b) is freed with the stash below
                ttnn.deallocate(s)
        if seed_rec is not self._seed_rec[b]:
            ttnn.deallocate(seed_rec)
        # Conv window [1, B, D]: conv_states[0] lane b = 0; conv_states[j+1] lane b = window j.
        D = self.qkv_dim_tp
        zero_b = ttnn.from_torch(
            torch.zeros(1, 1, D, dtype=torch.bfloat16),
            dtype=self.conv_states[0].dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        for k in range(self.K):
            seed_lane = zero_b if k == 0 else _match(self._seed_conv[b][k - 1], self.conv_states[k])  # [1,1,D]
            wins = [seed_lane if i == b else ttnn.slice(self.conv_states[k], (0, i, 0), (1, i + 1, D)) for i in range(B)]
            if B == 1:
                new_c = wins[0]
            else:
                rm = [ttnn.to_layout(w, ttnn.ROW_MAJOR_LAYOUT) for w in wins]
                new_rm = ttnn.concat(rm, dim=1)
                new_c = ttnn.to_layout(new_rm, ttnn.TILE_LAYOUT)
                ttnn.deallocate(new_rm)
                for r in rm:
                    ttnn.deallocate(r)
            ttnn.copy(new_c, self.conv_states[k])
            if new_c is not wins[0]:
                ttnn.deallocate(new_c)
            for i, w in enumerate(wins):
                if i != b:  # slices; seed windows freed with the stash below
                    ttnn.deallocate(w)
            if k > 0 and seed_lane is not self._seed_conv[b][k - 1]:
                ttnn.deallocate(seed_lane)  # a typecast temp (not the stashed window)
        ttnn.deallocate(zero_b)
        # Free the slot's stashed device tensors and clear it.
        ttnn.deallocate(self._seed_rec[b])
        for w in self._seed_conv[b]:
            ttnn.deallocate(w)
        del self._seed_rec[b]
        del self._seed_conv[b]

    def forward_prefill(self, x, chunk_size=128, valid_len=None, capture_state=False, seed_slot=None):
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
        # seed_slot prefills a fresh single sequence into a batch lane; it is always
        # from-scratch (initial_state=None), even while the batch runs in-place-state
        # (_stable_state) mode — a running batch reseeds a slot via prefill(seed_slot=b)
        # while _stable_state is True, and self.rec_state is then [B,...] (wrong shape for
        # this batch-1 prefill). So force no-carry when seeding.
        carry = self._stable_state and seed_slot is None
        if carry and self.conv_carry is None:
            self.reset_state()

        qkvz = ttnn.linear(x, tw["qkvz"], compute_kernel_config=self.cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        qkv = ttnn.slice(qkvz, (0, 0, 0), (1, T, self.qkv_dim_tp))
        z = ttnn.slice(qkvz, (0, 0, self.qkv_dim_tp), (1, T, self.qkvz_dim_tp))
        ttnn.deallocate(qkvz)
        ab = ttnn.linear(x, tw["ab"], compute_kernel_config=self.cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        a = ttnn.slice(ab, (0, 0, 0), (1, T, Nv))
        b = ttnn.slice(ab, (0, 0, Nv), (1, T, 2 * Nv))
        ttnn.deallocate(ab)

        # FIR causal conv1d + SiLU over the sequence. conv_state = the carried last K-1 conv
        # inputs of the previous chunk (left context); zero == None for a from-scratch pass.
        # valid_len makes new_state capture the last K-1 REAL tokens (not padding).
        conv, conv_new_state = _causal_conv1d_fir(
            qkv,
            None,
            None,
            self.K,
            self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            conv_state=self.conv_carry if carry else None,
            weight_taps=tw["conv_taps"],
            bias_dev=None,
            valid_len=valid_len,
        )
        ttnn.deallocate(qkv)

        kd = self.key_dim_tp
        q = ttnn.reshape(ttnn.slice(conv, (0, 0, 0), (1, T, kd)), (1, T, Nk, Dk))
        k = ttnn.reshape(ttnn.slice(conv, (0, 0, kd), (1, T, 2 * kd)), (1, T, Nk, Dk))
        v = ttnn.reshape(ttnn.slice(conv, (0, 0, 2 * kd), (1, T, self.qkv_dim_tp)), (1, T, Nv, Dv))
        ttnn.deallocate(conv)
        rf = Nv // Nk
        q = ttnn.repeat_interleave(q, rf, dim=2)
        k = ttnn.repeat_interleave(k, rf, dim=2)

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
        # ---- Batched-decode seeding (Step 1): stash this sequence's state per slot. ----
        # When seed_slot is set we are prefilling ONE of B sequences whose decode buffers
        # are [B,...]; a single-pass prefill produces batch-1 state, so we stash it (keyed
        # by slot, kept on device) and DON'T touch the shared [B,...] buffers here.
        # finalize_seed() concats all B slots into rec_state/conv_states once every
        # sequence is prefilled. Falls through to the shared output tail below.
        if seed_slot is not None:
            # Keep the state ON DEVICE (per-device head shards differ across the mesh;
            # a host round-trip would collapse the mesh dim). finalize_seed concats these
            # per-slot device tensors along the batch axis — ttnn.concat is per-device, so
            # each device's own head shard is preserved. Slices are fresh tensors, so the
            # shared tail's deallocate(conv_new_state) is still correct.
            self._seed_rec[seed_slot] = final_state  # [1, Nv, Dk, Dv]; NOT freed here
            windows = []
            for j in range(self.K - 1):
                windows.append(ttnn.reshape(ttnn.slice(conv_new_state, (0, j, 0), (1, j + 1, D)), (1, B, D)))
            self._seed_conv[seed_slot] = windows  # K-1 tensors, each [1, 1, D]
            # conv_new_state is freed by the shared tail below; final_state is retained.
        # ---- Carry recurrent + conv state for the NEXT chunk (chunk-outer prefill). ----
        # In place (ttnn.copy) when _stable_state so the addresses the prefill/decode traces
        # baked in stay valid across execute_trace replays and across sequences.
        elif carry:
            ttnn.copy(final_state, self.rec_state)
            ttnn.deallocate(final_state)
            ttnn.copy(conv_new_state, self.conv_carry)  # [1, K-1, D] last-K-1 conv inputs
        else:
            self.rec_state = final_state
        # ---- Finalize the decode conv window (last chunk / short prompt). ----
        # conv_states[1..K-1] = the last K-1 real conv inputs; [0] is the (shifted-out) zero.
        # Harmless to refresh every chunk — the last chunk's values are the ones decode reads.
        if seed_slot is None and capture_state:
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
        out_n = ttnn.rms_norm(o, weight=tw["norm_w"], epsilon=1e-6)
        ttnn.deallocate(o)
        out_f = ttnn.reshape(out_n, (1, T, self.value_dim_tp))
        ttnn.deallocate(out_n)
        gated = ttnn.multiply(out_f, ttnn.silu(z))
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
        if self.conv_states is None:
            self.reset_state()
        if len(x.shape) == 4:
            x = ttnn.reshape(x, (1, x.shape[-2], x.shape[-1]))  # [1,B,dim]

        qkvz = ttnn.linear(x, tw["qkvz"], compute_kernel_config=self.cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        qkv = ttnn.slice(qkvz, (0, 0, 0), (1, B, self.qkv_dim_tp))
        z = ttnn.slice(qkvz, (0, 0, self.qkv_dim_tp), (1, B, self.qkvz_dim_tp))
        ttnn.deallocate(qkvz)
        ab = ttnn.linear(x, tw["ab"], compute_kernel_config=self.cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        a = ttnn.slice(ab, (0, 0, 0), (1, B, Nv))
        b = ttnn.slice(ab, (0, 0, Nv), (1, B, 2 * Nv))
        ttnn.deallocate(ab)

        # conv1d shift-register (copy(src, dst): dst is 2nd arg) + weighted sum + silu
        st = self.conv_states
        for j in range(self.K - 1):
            ttnn.copy(st[j + 1], st[j])
        ttnn.copy(qkv, st[self.K - 1])
        ttnn.deallocate(qkv)
        conv = ttnn.multiply(st[0], tw["conv_taps"][0])
        for j in range(1, self.K):
            conv = ttnn.mac(st[j], tw["conv_taps"][j], conv)
        conv = ttnn.silu(conv)

        kd = self.key_dim_tp
        q = ttnn.reshape(ttnn.slice(conv, (0, 0, 0), (1, B, kd)), (B, Nk, Dk))
        k = ttnn.reshape(ttnn.slice(conv, (0, 0, kd), (1, B, 2 * kd)), (B, Nk, Dk))
        v = ttnn.reshape(ttnn.slice(conv, (0, 0, 2 * kd), (1, B, self.qkv_dim_tp)), (B, Nv, Dv))
        ttnn.deallocate(conv)

        rf = Nv // Nk
        if os.environ.get("QWEN_GDN_FUSED_DECODE") == "1":
            # ---- Fused C++ kernel: gated-delta STATE update + raw read, both bit-exact vs torch
            # (test_deltanet_decode_kernel_pcc: state 1.0, PROBE-RAW 0.99998). The op outputs the
            # RAW pre-norm o (its internal norm/gate is unused); the gated-RMSNorm + silu(z) run in
            # ttnn in the shared tail below — matching recurrent_gated_delta_rule_decode_ttnn's
            # raw-o contract. FLATTENED-HEAD batching (Stage 7): the B decode lanes are folded into
            # the head axis (H = B*Nv heads, ONE kernel call, one core per (batch,head)), so batching
            # is ~free. qkv_proj is packed [all_q | all_k | all_v] (batch-major within each block) so
            # the op's per-head offset math (k_head_idx = h/head_expand_ratio) addresses batch b's
            # block; state / z / beta / decay are batch-major [.., B*..] = free reshapes. ----
            H = B * Nv
            qn = ttnn.multiply(l2_norm_ttnn(q, dim=-1), self.scale)  # [B, Nk, Dk]
            kn = l2_norm_ttnn(k, dim=-1)
            qkv_proj = ttnn.reshape(
                ttnn.concat(
                    [ttnn.reshape(qn, (1, 1, B * kd)), ttnn.reshape(kn, (1, 1, B * kd)),
                     ttnn.reshape(v, (1, 1, B * self.value_dim_tp))], dim=-1,
                ), (1, 1, 1, B * self.qkv_dim_tp),
            )
            z_p = ttnn.reshape(z, (1, 1, 1, B * self.value_dim_tp))
            beta_p = ttnn.reshape(ttnn.sigmoid(b), (1, 1, 1, H))
            ttnn.deallocate(b)
            decay_p = ttnn.reshape(
                ttnn.exp(ttnn.multiply(tw["neg_exp_A"], ttnn.softplus(ttnn.add(a, tw["dt_bias"])))), (1, 1, 1, H)
            )
            ttnn.deallocate(a)
            # Pass-through / unused kernel args (kept in the op signature) — allocate once and reuse
            # (a per-step from_torch would be a host write, killing decode perf + breaking trace).
            if getattr(self, "_kdummy_conv", None) is None:
                rep = ttnn.ReplicateTensorToMesh(self.mesh)
                self._kdummy_conv = ttnn.from_torch(
                    torch.zeros(1, 1, B * self.qkv_dim_tp, 32, dtype=torch.bfloat16),
                    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh, mesh_mapper=rep,
                )
                self._kdummy_h = ttnn.from_torch(
                    torch.zeros(1, 1, 1, H, dtype=torch.bfloat16),
                    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh, mesh_mapper=rep,
                )
                self._knorm_w = ttnn.reshape(tw["norm_w"], (1, 1, 1, Dv))
            # State: the batch-flatten reshape [B,Nv,Dk,Dv] -> [1,B*Nv,Dk,Dv] mangles an fp32 TILE
            # tensor (yields NaNs); bf16 reshapes correctly. The kernel's recurrence + output are bf16
            # regardless (CBs are bf16; output dtype = qkv_proj.dtype), so a bf16 fused-path state is
            # not a precision regression vs the kernel itself — only the pure-python path keeps the
            # fp32 high_precision carry. After step 1, self.rec_state is bf16 (reassigned below).
            rec_src = self.rec_state
            if rec_src.dtype != ttnn.bfloat16:
                rec_src = ttnn.typecast(rec_src, ttnn.bfloat16)
            rec_flat = ttnn.reshape(rec_src, (1, H, Dk, Dv))
            _out, new_rec, _cs = ttnn.experimental.deltanet_decode_full(
                qkv_proj, z_p, beta_p, decay_p, self._kdummy_conv, rec_flat, self._kdummy_conv,
                self._kdummy_h, self._kdummy_h, self._knorm_w,
                num_heads=H, num_k_heads=B * Nk, k_head_dim=Dk, v_head_dim=Dv,
                conv_dim=B * self.qkv_dim_tp, conv_kernel_size=self.K, head_expand_ratio=rf,
            )
            ttnn.deallocate(_cs)
            if rec_src is not self.rec_state:
                ttnn.deallocate(rec_src)
            # Kernel outputs the RAW pre-norm read (q @ S_new); the gated-RMSNorm(norm_w) + silu(z)
            # runs in the ttnn tail below. Folding norm/gate into the kernel was measured SLOWER on
            # Blackhole (one-core-per-head serializes the RMSNorm reduce: +8.4ms/step at B=8 vs the
            # parallel ttnn tail) — see memory note. So we keep the raw-o contract + ttnn norm/gate.
            o = _out  # [1,1,1,B*Nv*Dv] RAW; shared tail does norm/gate + reshape to (B,Nv,Dv)
            new_rec = ttnn.reshape(new_rec, (B, Nv, Dk, Dv))  # bf16
            if self._stable_state:
                ttnn.copy(new_rec, self.rec_state)
                ttnn.deallocate(new_rec)
            else:
                self.rec_state = new_rec
        else:
            # expand Q/K from Nk to Nv heads (GQA); recurrence fn L2-norms + scales internally
            q = ttnn.repeat_interleave(q, rf, dim=1)
            k = ttnn.repeat_interleave(k, rf, dim=1)
            q = ttnn.reshape(q, (B, 1, Nv, Dk))
            k = ttnn.reshape(k, (B, 1, Nv, Dk))
            v = ttnn.reshape(v, (B, 1, Nv, Dv))

            beta = ttnn.reshape(ttnn.sigmoid(b), (B, 1, Nv))
            ttnn.deallocate(b)
            g = ttnn.multiply(tw["neg_exp_A"], ttnn.softplus(ttnn.add(a, tw["dt_bias"])))
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

        # Gated-RMSNorm(norm_w) over Dv + silu(z) gate, in ttnn (both the fused-kernel raw-o path
        # and the recurrent_gated_delta_rule_decode_ttnn path output raw o). Folding this into the
        # kernel is a NET LOSS on Blackhole (one-core-per-head serializes the reduce) — see memory.
        out_r = ttnn.reshape(o, (B, Nv, Dv))
        out_n = ttnn.rms_norm(out_r, weight=tw["norm_w"], epsilon=1e-6)  # gated norm: weight only (no +1)
        ttnn.deallocate(out_r)
        out_f = ttnn.reshape(out_n, (1, B, self.value_dim_tp))
        ttnn.deallocate(out_n)
        gated = ttnn.multiply(out_f, ttnn.silu(z))
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
