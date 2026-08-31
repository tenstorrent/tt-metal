# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8 MTP (multi-token prediction) drafter head.

One full-attention decoder layer fed by fc(cat[norm(embed(tok)), norm(hidden)]),
per vLLM's Qwen3_5MultiTokenPredictor (HF transformers ignores mtp.* weights, so
vLLM is the reference). The head shares the target's embedding + LM head
(mtp_use_dedicated_embeddings=false) and keeps its own paged KV cache.

Position convention (vLLM llm_base_proposer): the pair (target_hidden[i],
token[i+1]) sits at drafter position i (RoPE + KV slot i) and predicts
token[i+2]. Every forward is a T=1 paged decode step — a 1-layer step is cheap
enough that drafting, catch-up, and prompt seeding all reuse the same program.

On a TP mesh the head runs fully REPLICATED (weights, KV, and inputs on every
device; no CCL): a 1-layer step is small enough that redundant compute beats
sharding it. The vocab-sharded target LM head is the one non-replicated piece —
logits come back per-shard and are concatenated on host.

See docs/mtp_design.md.
"""

import torch

import ttnn
from models.demos.blackhole.qwen36.tt.attention import AttentionConfig, Qwen36GatedAttention
from models.demos.blackhole.qwen36.tt.mlp import Qwen36MLP
from models.demos.blackhole.qwen36.tt.rms_norm import rms_norm_ttnn
from models.demos.blackhole.qwen36.utils.substate import substate


class _SingleDeviceArgsView:
    """args view forcing num_devices=1 so shared modules take their
    single-device code paths (the replicated head must not reduce-scatter)."""

    num_devices = 1

    def __init__(self, args):
        object.__setattr__(self, "_args", args)

    def __getattr__(self, name):
        return getattr(self._args, name)


class Qwen36MTPHead:
    """MTP drafter head (1 full-attention layer + fc fuse); replicated on TP.

    Args:
        mesh_device: single-device or TP mesh (replicated execution on TP).
        args: Qwen36ModelArgs (attention geometry, dims, eps).
        mtp_state_dict: mtp.*-stripped weights (load_qwen36_mtp_state_dict).
        embedding: callable mapping a [1,1] uint32 device token tensor to its
            REPLICATED full-dim embedding [1,1,dim] (the target's Embedding on a
            single device — hidden-sharded on TP, so TP callers pass their own
            replicated table — or a test stub).
        lm_head_weight: [dim, vocab-ish] device weight shared with the target
            (any output width works — tests pass a small random head).
        rope: Qwen36RoPESetup shared with the target (tables are replicated).
        lm_head_sharded: True when lm_head_weight is vocab-sharded across the
            mesh (step() then concatenates the logit shards on host).
    """

    def __init__(
        self,
        mesh_device,
        args,
        mtp_state_dict,
        embedding,
        lm_head_weight,
        rope,
        tensor_cache_path=None,
        lm_head_sharded=False,
    ):
        self.device = mesh_device
        self.args = args
        self.embedding = embedding
        self.lm_head_weight = lm_head_weight
        self.rope = rope
        self.norm_eps = args.norm_eps
        self.num_devices = mesh_device.get_num_devices()
        self.lm_head_sharded = lm_head_sharded
        self._mesh_kwargs = dict(mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)) if self.num_devices > 1 else {}
        single_view = _SingleDeviceArgsView(args) if self.num_devices > 1 else args

        cache = (tensor_cache_path / "mtp") if tensor_cache_path else None

        def _norm_1p(name):
            # Zero-centered RMSNorm weight, pre-offset by +1 (rms_norm_ttnn contract).
            t = mtp_state_dict[f"{name}.weight"] + 1.0
            return ttnn.as_tensor(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=(cache / f"{name}.weight_offset") if cache else None,
                **self._mesh_kwargs,
            )

        self.pre_fc_norm_embedding = _norm_1p("pre_fc_norm_embedding")
        self.pre_fc_norm_hidden = _norm_1p("pre_fc_norm_hidden")
        self.final_norm = _norm_1p("norm")

        # fc: [2*dim, dim] as ttnn.linear weight ([in, out] = HF [out, in].T).
        self.fc_weight = ttnn.as_tensor(
            mtp_state_dict["fc.weight"].T.contiguous(),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=(cache / "fc.weight") if cache else None,
            **self._mesh_kwargs,
        )

        # Decoder block, mirroring Qwen36DecoderLayer's single-device full-attn
        # path. The block norms ride the same bare-ttnn.rms_norm helper as the
        # fc norms: the class RMSNorm's program variant carries a ~20 KB larger
        # static CB region, which is exactly the margin the batched graph does
        # not have (56275).
        self.attn_norm_w = _norm_1p("layers.0.input_layernorm")
        self.ffn_norm_w = _norm_1p("layers.0.post_attention_layernorm")
        self.attention = Qwen36GatedAttention(
            mesh_device,
            AttentionConfig.from_args(args),
            substate(mtp_state_dict, "layers.0.self_attn"),
            cache,
        )
        self.feed_forward = Qwen36MLP(mesh_device, substate(mtp_state_dict, "layers.0.mlp"), cache, args=single_view)

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        self._k_cache = None
        self._v_cache = None
        self._page_table_tt = None
        self._page_tables = []
        self._page_tables_host = None
        self._win = None  # traced window-step machinery (ensure_window/stage_window)
        self._bwin = None  # traced BATCHED window machinery (ensure_batched_window)
        self._last_normed = None

    def allocate_kv_cache(self, num_blocks, block_size=64, dtype=ttnn.bfloat16, users=1):
        """Own paged KV (one layer) + per-user identity page tables.

        `num_blocks` is PER USER; the shared cache holds users*num_blocks blocks
        and user u's page table routes to blocks [u*num_blocks, (u+1)*num_blocks).
        """
        assert self._k_cache is None, "MTP KV cache already allocated"
        shape = [users * num_blocks, self.args.n_kv_heads, block_size, self.args.head_dim]
        self._k_cache = ttnn.zeros(shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.device)
        self._v_cache = ttnn.zeros(shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.device)
        self.attention.set_paged_kv_cache(self._k_cache, self._v_cache)
        self._page_tables_host = torch.stack(
            [torch.arange(u * num_blocks, (u + 1) * num_blocks, dtype=torch.int32) for u in range(users)]
        )
        self._page_tables = [
            ttnn.from_torch(
                self._page_tables_host[u : u + 1].contiguous(),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                **self._mesh_kwargs,
            )
            for u in range(users)
        ]
        self._page_table_tt = self._page_tables[0]

    def free_kv_cache(self):
        if self._k_cache is None:
            return
        self.release_window_traces()
        for t in (self._k_cache, self._v_cache, *self._page_tables):
            ttnn.deallocate(t)
        self._k_cache = self._v_cache = self._page_table_tt = None
        self._page_tables = []
        self.attention.paged_kv_cache_key = None
        self.attention.paged_kv_cache_value = None
        self.attention.use_paged_attention = False

    def step(self, token_id, hidden_row, position, user=0):
        """One drafter step: (hidden at pos, token at pos+1) -> logits for pos+2.

        Args:
            token_id: int — the input token (target token at position+1).
            hidden_row: torch [dim] — target post-final-norm hidden at `position`
                (or the drafter's own hidden from the previous chained step).
            position: int — drafter RoPE position and KV slot.
            user: which user's drafter KV blocks to read/write (batched spec).

        Returns:
            (logits torch [vocab_out], hidden torch [dim]) — hidden is the
            post-norm drafter output, the chained-draft input for the next step.
        """
        assert self._k_cache is not None, "call allocate_kv_cache first"
        tok = ttnn.from_torch(
            torch.tensor([[token_id]], dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            **self._mesh_kwargs,
        )
        h_in = ttnn.from_torch(
            hidden_row.reshape(1, 1, -1).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            **self._mesh_kwargs,
        )
        cos, sin = self.rope.get_rot_mats(torch.tensor([[position]], dtype=torch.long))
        cur_pos = ttnn.from_torch(
            torch.tensor([position], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            **self._mesh_kwargs,
        )
        logits, normed = self._step_graph(tok, h_in, cur_pos, cos, sin, page_table=self._page_tables[user])
        for t in (tok, h_in, cur_pos):
            ttnn.deallocate(t)

        logits_host = self._logits_to_host(logits)
        if self.num_devices > 1:
            hidden_host = ttnn.to_torch(ttnn.get_device_tensors(normed)[0]).reshape(-1).float()
        else:
            hidden_host = ttnn.to_torch(normed).reshape(-1).float()
        ttnn.deallocate(normed)
        ttnn.deallocate(logits)
        return logits_host, hidden_host

    def _step_graph(self, tok, h_in, cur_pos, cos, sin, page_table=None):
        """The drafter-step op graph over device inputs. Returns (logits, normed).

        Shared by the eager step() and the trace capture/replay path — everything
        here is device ops only (trace-capture safe). Activations live in DRAM:
        a resident L1 activation at batched width ([8,1,5120] pads to
        [8,32,5120] = 20,480 B reserved in EVERY L1 bank) drops the allocator
        watermark under the whole-row rms_norm programs' static CB region — the
        56275 clash fired dispatching a block norm while the fc output sat
        L1-pinned exactly 19,712 B over the margin. With DRAM activations every
        norm dispatches at the baseline watermark, which the same run proved
        sufficient with 20,480 B to spare.
        """
        if page_table is None:
            page_table = self._page_table_tt
        dram = ttnn.DRAM_MEMORY_CONFIG
        emb = self.embedding(tok)  # [B,1,dim]
        e_n = rms_norm_ttnn(emb, self.pre_fc_norm_embedding, eps=self.norm_eps, memory_config=dram)
        ttnn.deallocate(emb)
        h_n = rms_norm_ttnn(h_in, self.pre_fc_norm_hidden, eps=self.norm_eps, memory_config=dram)

        fused = ttnn.concat([e_n, h_n], dim=-1)  # [B,1,2*dim] — embedding first (vLLM order)
        ttnn.deallocate(e_n)
        ttnn.deallocate(h_n)
        x = ttnn.linear(fused, self.fc_weight, compute_kernel_config=self.compute_kernel_config, memory_config=dram)
        ttnn.deallocate(fused)

        # Decoder block (decode-mode residual pattern, as in Qwen36DecoderLayer).
        attn_in = rms_norm_ttnn(x, self.attn_norm_w, eps=self.norm_eps, memory_config=dram)
        attn_out = self.attention.forward(attn_in, cos, sin, position_tensor=cur_pos, page_table=page_table)
        ttnn.deallocate(attn_in)
        h1 = ttnn.add(x, attn_out, memory_config=dram)
        ttnn.deallocate(x)
        ttnn.deallocate(attn_out)

        ff_in = rms_norm_ttnn(h1, self.ffn_norm_w, eps=self.norm_eps, memory_config=dram)
        ff_out = self.feed_forward.forward(ff_in)
        ttnn.deallocate(ff_in)
        out = ttnn.add(h1, ff_out, memory_config=dram)
        ttnn.deallocate(h1)
        ttnn.deallocate(ff_out)

        normed = rms_norm_ttnn(out, self.final_norm, eps=self.norm_eps, memory_config=dram)
        ttnn.deallocate(out)
        logits = ttnn.linear(normed, self.lm_head_weight, compute_kernel_config=self.compute_kernel_config)
        return logits, normed

    def _logits_to_host(self, logits):
        if self.num_devices > 1:
            if self.lm_head_sharded:
                return (
                    ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=-1))
                    .reshape(-1)
                    .float()
                )
            return ttnn.to_torch(ttnn.get_device_tensors(logits)[0]).reshape(-1).float()
        return ttnn.to_torch(logits).reshape(-1).float()

    # ── traced window stepping ───────────────────────────────────────────────
    # An iteration's drafter legs sit at CONSECUTIVE positions (catch-up pairs,
    # then the chained drafts), so per-iteration host work collapses to ONE
    # window upload (pos/cos/sin for the whole run of legs) plus one tiny token
    # upload per leg. One trace is captured PER WINDOW INDEX j — each bakes a
    # static slice of the window buffers — and every leg is a single replay.
    # The greedy pick happens ON DEVICE (per-vocab-shard argmax + max), so a leg
    # reads back only a [2]-value score per device instead of the logit shard.

    def _argmax_score(self, logits):
        """[1,1,Vs] logits -> ((idx [1,1,1] uint32), (val [1,1,1] bf16)) local greedy score.

        Multicore ttnn.argmax needs ROW_MAJOR input and exactly one 32-row tile
        (fewer/more rows silently degrade), so the single logit row is padded to
        32 rows first. The max value reduces the SAME padded tensor along dim=-1
        (only row 0 is read, so the pad value is irrelevant) — data-layout
        reshapes like the old RxC max grid are host-fallback candidates (a read,
        illegal under trace capture), so everything here is pad/untilize/reduce
        only, matching the verify-score graph's programs. Two separate small
        outputs: no cross-dtype concat, no uint32 typecast.
        """
        vs = logits.shape[-1]
        l4 = ttnn.reshape(logits, (1, 1, 1, vs))
        padded = ttnn.pad(l4, [(0, 0), (0, 0), (0, 31), (0, 0)], value=0.0)
        ttnn.deallocate(l4)
        rm = ttnn.untilize(padded, use_multicore=True)
        idx32 = ttnn.argmax(rm, dim=-1, keepdim=False)  # [1,1,32] uint32
        ttnn.deallocate(rm)
        idx = ttnn.slice(idx32, [0, 0, 0], [1, 1, 1])
        ttnn.deallocate(idx32)
        val = ttnn.max(padded, dim=-1)  # [1,1,32] bf16 (row 0 is the real row; host reads [.., 0])
        ttnn.deallocate(padded)
        return idx, val

    def token_from_scores(self, idxs, vals):
        """Host per-shard idx/val vectors [D] -> global greedy token id."""
        if self.num_devices > 1 and self.lm_head_sharded:
            per_shard = self.args.vocab_size // self.num_devices
            d = int(vals.argmax())
            return d * per_shard + int(idxs[d])
        return int(idxs[0])  # replicated logits: every row agrees

    def _init_window(self, w_max):
        mk = self._mesh_kwargs
        rd = self.rope.head_dim
        rm, tile = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT

        def dev(t, dtype, layout):
            return ttnn.from_torch(t, dtype=dtype, layout=layout, device=self.device, **mk)

        self._win = {
            "w_max": w_max,
            "pos": dev(torch.zeros(w_max, dtype=torch.int32), ttnn.int32, rm),
            "cos": dev(torch.zeros(1, w_max, rd, dtype=torch.bfloat16), ttnn.bfloat16, rm),
            "sin": dev(torch.zeros(1, w_max, rd, dtype=torch.bfloat16), ttnn.bfloat16, rm),
            "tok": dev(torch.zeros(1, 1, dtype=torch.int64), ttnn.uint32, rm),
            "h": dev(torch.zeros(1, 1, self.args.dim, dtype=torch.bfloat16), ttnn.bfloat16, tile),
            "traces": {},
        }
        self._last_normed = None

    def stage_window(self, start_pos, width):
        """Upload pos/cos/sin for drafter positions [start_pos, start_pos+width).

        One host->device copy per table per ITERATION; the per-index traces then
        slice their leg's row statically.
        """
        w_max = self._win["w_max"]
        assert width <= w_max, f"window {width} exceeds w_max {w_max}"
        pos = torch.zeros(w_max, dtype=torch.int32)
        pos[:width] = torch.arange(start_pos, start_pos + width, dtype=torch.int32)
        cos = torch.zeros(1, w_max, self.rope.head_dim, dtype=torch.bfloat16)
        sin = torch.zeros(1, w_max, self.rope.head_dim, dtype=torch.bfloat16)
        cos[0, :width] = self.rope.cos_cpu[start_pos : start_pos + width]
        sin[0, :width] = self.rope.sin_cpu[start_pos : start_pos + width]
        for host_t, dtype, layout, dst in (
            (pos, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, "pos"),
            (cos, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, "cos"),
            (sin, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, "sin"),
        ):
            src = ttnn.from_torch(host_t, dtype=dtype, layout=layout, **self._mesh_kwargs)
            ttnn.copy_host_to_device_tensor(src, self._win[dst])

    def _win_step_body(self, j):
        """Drafter step graph for window index j (static window slices + score)."""
        win = self._win
        rd = self.rope.head_dim
        cur_pos = ttnn.slice(win["pos"], [j], [j + 1])
        cos = ttnn.to_layout(ttnn.slice(win["cos"], [0, j, 0], [1, j + 1, rd]), ttnn.TILE_LAYOUT)
        sin = ttnn.to_layout(ttnn.slice(win["sin"], [0, j, 0], [1, j + 1, rd]), ttnn.TILE_LAYOUT)
        logits, normed = self._step_graph(win["tok"], win["h"], cur_pos, cos, sin)
        idx, val = self._argmax_score(logits)
        ttnn.deallocate(logits)
        return idx, val, normed

    def capture_window_traces(self):
        """Capture one step trace per window index, all up front.

        Must run before any OTHER trace is parked (post-park compiles clobber
        parked traces): the spec loop calls this at iteration 1, before the
        verify trace exists. Two phases — ALL eager compile passes first, then
        all captures — so no compile ever runs after the first window trace is
        parked (per-index slice offsets may compile distinct programs). The
        capture-time KV writes land at the staged window slots and are
        rewritten in order by the real legs before any leg attends to them.
        """
        if self._win["traces"]:
            return
        for j in range(self._win["w_max"]):
            idx, val, normed = self._win_step_body(j)
            for t in (idx, val, normed):
                ttnn.deallocate(t)
        ttnn.synchronize_device(self.device)
        for j in range(self._win["w_max"]):
            tid = ttnn.begin_trace_capture(self.device, cq_id=0)
            idx, val, normed = self._win_step_body(j)
            ttnn.end_trace_capture(self.device, tid, cq_id=0)
            self._win["traces"][j] = {"id": tid, "idx": idx, "val": val, "normed": normed}

    def step_win(self, j, token_id, hidden_row=None, chain_hidden=False, want_token=False):
        """One drafter leg at window index j (1-2 tiny uploads + one replay).

        chain_hidden feeds the previous leg's on-device hidden. want_token reads
        back the [2]-value per-shard scores and returns the global greedy token;
        catch-up legs skip the readback entirely.
        """
        if self._win is None:
            raise RuntimeError("call _init_window/stage_window first")
        if not self._win["traces"]:
            self.capture_window_traces()
        tr = self._win["traces"][j]
        tok_h = ttnn.from_torch(
            torch.tensor([[token_id]], dtype=torch.int64),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            **self._mesh_kwargs,
        )
        ttnn.copy_host_to_device_tensor(tok_h, self._win["tok"])
        if chain_hidden:
            ttnn.copy(self._last_normed, self._win["h"])
        else:
            h_h = ttnn.from_torch(
                hidden_row.reshape(1, 1, -1).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                **self._mesh_kwargs,
            )
            ttnn.copy_host_to_device_tensor(h_h, self._win["h"])
        ttnn.execute_trace(self.device, tr["id"], cq_id=0, blocking=False)
        self._last_normed = tr["normed"]
        if not want_token:
            return None
        if self.num_devices > 1:
            comp = ttnn.ConcatMeshToTensor(self.device, dim=0)
            idxs = ttnn.to_torch(tr["idx"], mesh_composer=comp).reshape(-1)
            # val rows beyond 0 are argmax padding; element 0 per device is real.
            vals = ttnn.to_torch(tr["val"], mesh_composer=comp).float().reshape(self.num_devices, -1)[:, 0]
        else:
            idxs = ttnn.to_torch(tr["idx"]).reshape(-1)
            vals = ttnn.to_torch(tr["val"]).float().reshape(1, -1)[:, 0]
        return self.token_from_scores(idxs, vals)

    def ensure_window(self, w_max):
        if self._win is None:
            self._init_window(w_max)

    def release_window_traces(self):
        if self._win is not None:
            for tr in self._win["traces"].values():
                ttnn.release_trace(self.device, tr["id"])
            self._win["traces"] = {}
        if self._bwin is not None:
            for tr in self._bwin["traces"].values():
                ttnn.release_trace(self.device, tr["id"])
            self._bwin["traces"] = {}
        self._last_normed = None

    # ── traced batched (c8) window stepping ──────────────────────────────────
    # All B users advance one drafter leg per replay: a [B]-row T=1 step with
    # per-user positions/rope/KV-blocks as data, the greedy pick per user on
    # device (rows of ONE 32-row argmax), and 2 tiny readbacks per DRAFT leg for
    # all users at once. The caller end-aligns per-user schedules (static K:
    # every user's chained legs are the LAST K-1 indices; shorter catch-ups
    # left-pad by replaying their first pending — an idempotent KV rewrite).

    def ensure_batched_window(self, w_max, users):
        if self._bwin is not None:
            return
        mk = self._mesh_kwargs
        rd = self.rope.head_dim
        rm, tile = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT

        def dev(t, dtype, layout):
            return ttnn.from_torch(t, dtype=dtype, layout=layout, device=self.device, **mk)

        stacked_pt = self._page_tables_host[:users].contiguous()
        self._bwin = {
            "w_max": w_max,
            "B": users,
            "pos": dev(torch.zeros(users, w_max, dtype=torch.int32), ttnn.int32, rm),
            "cos": dev(torch.zeros(users, w_max, rd, dtype=torch.bfloat16), ttnn.bfloat16, rm),
            "sin": dev(torch.zeros(users, w_max, rd, dtype=torch.bfloat16), ttnn.bfloat16, rm),
            "tok": dev(torch.zeros(users, 1, dtype=torch.int64), ttnn.uint32, rm),
            "h": dev(torch.zeros(users, 1, self.args.dim, dtype=torch.bfloat16), ttnn.bfloat16, tile),
            "pt": dev(stacked_pt.to(torch.int32), ttnn.int32, rm),  # [B, blocks_per_user] constant
            "traces": {},
        }

    def stage_batched_window(self, pos_table):
        """Upload the per-(user, leg) position table + matching rope rows.

        pos_table: torch [B, width] int32 — the caller owns the end-aligned
        schedule (padding legs replay a real position). One upload per table
        per ITERATION.
        """
        bw = self._bwin
        B, w_max = bw["B"], bw["w_max"]
        width = pos_table.shape[1]
        assert pos_table.shape[0] == B and width <= w_max
        # Unused columns replicate column 0: only the compile/capture passes
        # execute those indices, and their KV writes land where the real leg-0
        # replay rewrites them (a zero-pad would leak capture garbage to KV
        # position 0 instead).
        pos = pos_table[:, :1].expand(B, w_max).contiguous().to(torch.int32)
        pos[:, :width] = pos_table.to(torch.int32)
        flat = pos.reshape(-1).to(torch.long)
        cos = self.rope.cos_cpu[flat].reshape(B, w_max, -1).to(torch.bfloat16)
        sin = self.rope.sin_cpu[flat].reshape(B, w_max, -1).to(torch.bfloat16)
        for host_t, dtype, dst in ((pos, ttnn.int32, "pos"), (cos, ttnn.bfloat16, "cos"), (sin, ttnn.bfloat16, "sin")):
            src = ttnn.from_torch(host_t, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, **self._mesh_kwargs)
            ttnn.copy_host_to_device_tensor(src, bw[dst])

    def _batched_step_body(self, j):
        """B-row drafter leg at window index j (static window slices; batched scores)."""
        bw = self._bwin
        B = bw["B"]
        rd = self.rope.head_dim
        cur_pos = ttnn.reshape(ttnn.slice(bw["pos"], [0, j], [B, j + 1]), (B,))
        cos = ttnn.to_layout(ttnn.slice(bw["cos"], [0, j, 0], [B, j + 1, rd]), ttnn.TILE_LAYOUT)
        sin = ttnn.to_layout(ttnn.slice(bw["sin"], [0, j, 0], [B, j + 1, rd]), ttnn.TILE_LAYOUT)
        logits, normed = self._step_graph(bw["tok"], bw["h"], cur_pos, cos, sin, page_table=bw["pt"])
        # The traces RETAIN (idx, val, normed) for the process lifetime, one set
        # per window index — all three must live in DRAM (a retained L1 tensor
        # drops the allocator watermark under a parked trace for every later
        # program dispatch: the batched normed alone would reserve 20,480 B of
        # every L1 bank). normed is DRAM at the source (_step_graph); idx/val
        # get explicit DRAM configs below. No post-hoc migration: ttnn's
        # to_memory_config on an already-DRAM tensor returns a distinct handle
        # ALIASING the same buffer, so a same-config "migrate + free the
        # original" frees the buffer out from under the alias (56610).
        # Per-user greedy scores: rows of ONE 32-row argmax. [B,1,Vs] TILE has
        # per-batch row padding, so the row-merge reshape must go through
        # ROW_MAJOR (a padded-TILE reshape is a host-fallback read — illegal
        # under capture).
        vs = logits.shape[-1]
        l_rm = ttnn.untilize(logits, use_multicore=True)
        ttnn.deallocate(logits)
        if l_rm.shape[-2] != 1:
            # untilize carries the padded tile rows ([B,1,Vs] -> [B,32,Vs]);
            # row 0 of each user's block is the real one (56397).
            l_v = ttnn.slice(l_rm, (0, 0, 0), (B, 1, vs))
            ttnn.deallocate(l_rm)
            l_rm = l_v
        l4 = ttnn.reshape(l_rm, (1, 1, B, vs))
        lp = ttnn.pad(l4, [(0, 0), (0, 0), (0, 32 - B), (0, 0)], value=0.0)
        ttnn.deallocate(l_rm)
        # rows 0..B-1 = users; retained -> DRAM (see above)
        idx = ttnn.argmax(lp, dim=-1, keepdim=False, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1,1,32] uint32
        lt = ttnn.to_layout(lp, ttnn.TILE_LAYOUT)
        ttnn.deallocate(lp)
        val = ttnn.max(lt, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1,1,32] (rows 0..B-1 real)
        ttnn.deallocate(lt)
        return idx, val, normed

    def compile_batched_window(self):
        """Eager compile pass for every window index — run BEFORE any trace is
        parked anywhere (per-index slice offsets may compile distinct programs)."""
        if self._bwin["traces"]:
            return
        normed = None
        for j in range(self._bwin["w_max"]):
            if normed is not None:
                ttnn.deallocate(normed)
            idx, val, normed = self._batched_step_body(j)
            for t in (idx, val):
                ttnn.deallocate(t)
        # Warm the chain-hidden copy with the exact normed spec: the first
        # chained leg must not compile a program after traces are parked.
        ttnn.copy(normed, self._bwin["h"])
        ttnn.deallocate(normed)
        ttnn.synchronize_device(self.device)

    def capture_batched_window(self):
        """Capture every window-index trace (compile_batched_window ran earlier)."""
        if self._bwin["traces"]:
            return
        for j in range(self._bwin["w_max"]):
            tid = ttnn.begin_trace_capture(self.device, cq_id=0)
            idx, val, normed = self._batched_step_body(j)
            ttnn.end_trace_capture(self.device, tid, cq_id=0)
            self._bwin["traces"][j] = {"id": tid, "idx": idx, "val": val, "normed": normed}

    def step_batched(self, j, tokens, hiddens=None, chain_hidden=False, want_tokens=False):
        """One drafter leg for ALL users at window index j.

        tokens: list[B] input token ids. hiddens: torch [B, dim] target hiddens
        (catch-up legs) — or chain_hidden=True to feed every user's previous
        on-device hidden. want_tokens reads the batched scores back and returns
        the per-user global greedy picks.
        """
        bw = self._bwin
        B = bw["B"]
        tr = bw["traces"][j]
        tok_h = ttnn.from_torch(
            torch.tensor(tokens, dtype=torch.int64).reshape(B, 1),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            **self._mesh_kwargs,
        )
        ttnn.copy_host_to_device_tensor(tok_h, bw["tok"])
        if chain_hidden:
            ttnn.copy(self._last_normed, bw["h"])
        else:
            h_h = ttnn.from_torch(
                hiddens.reshape(B, 1, -1).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                **self._mesh_kwargs,
            )
            ttnn.copy_host_to_device_tensor(h_h, bw["h"])
        ttnn.execute_trace(self.device, tr["id"], cq_id=0, blocking=False)
        self._last_normed = tr["normed"]
        if not want_tokens:
            return None
        if self.num_devices > 1:
            comp = ttnn.ConcatMeshToTensor(self.device, dim=0)
            idxs = ttnn.to_torch(tr["idx"], mesh_composer=comp).reshape(self.num_devices, -1)
            vals = ttnn.to_torch(tr["val"], mesh_composer=comp).float().reshape(self.num_devices, -1)
        else:
            idxs = ttnn.to_torch(tr["idx"]).reshape(1, -1)
            vals = ttnn.to_torch(tr["val"]).float().reshape(1, -1)
        out = []
        for u in range(B):
            out.append(self.token_from_scores(idxs[:, u], vals[:, u]))
        return out
