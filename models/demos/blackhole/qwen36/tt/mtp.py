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
from models.common.rmsnorm import RMSNorm
from models.demos.blackhole.qwen36.tt.attention import AttentionConfig, Qwen36GatedAttention
from models.demos.blackhole.qwen36.tt.mlp import Qwen36MLP
from models.demos.blackhole.qwen36.tt.rms_norm import rms_norm_ttnn
from models.demos.blackhole.qwen36.utils.substate import substate
from models.tt_transformers.tt.common import Mode


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

        # Decoder block, mirroring Qwen36DecoderLayer's single-device full-attn path.
        self.attention_norm = RMSNorm(
            device=mesh_device,
            dim=args.dim,
            state_dict=mtp_state_dict,
            weight_key="input_layernorm",
            state_dict_prefix="layers.0.",
            weight_cache_path=cache,
            weight_dtype=ttnn.bfloat16,
            add_unit_offset=True,
            eps=args.norm_eps,
        )
        self.ffn_norm = RMSNorm(
            device=mesh_device,
            dim=args.dim,
            state_dict=mtp_state_dict,
            weight_key="post_attention_layernorm",
            state_dict_prefix="layers.0.",
            weight_cache_path=cache,
            weight_dtype=ttnn.bfloat16,
            add_unit_offset=True,
            eps=args.norm_eps,
        )
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
        self._step_trace = None  # persistent traced-step I/O (capture on first step_traced)

    def allocate_kv_cache(self, num_blocks, block_size=64, dtype=ttnn.bfloat16):
        """Own paged KV (one layer) + identity page table over its own blocks."""
        assert self._k_cache is None, "MTP KV cache already allocated"
        shape = [num_blocks, self.args.n_kv_heads, block_size, self.args.head_dim]
        self._k_cache = ttnn.zeros(shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.device)
        self._v_cache = ttnn.zeros(shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.device)
        self.attention.set_paged_kv_cache(self._k_cache, self._v_cache)
        self._page_table_tt = ttnn.from_torch(
            torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            **self._mesh_kwargs,
        )

    def free_kv_cache(self):
        if self._k_cache is None:
            return
        self.release_step_trace()
        for t in (self._k_cache, self._v_cache, self._page_table_tt):
            ttnn.deallocate(t)
        self._k_cache = self._v_cache = self._page_table_tt = None
        self.attention.paged_kv_cache_key = None
        self.attention.paged_kv_cache_value = None
        self.attention.use_paged_attention = False

    def step(self, token_id, hidden_row, position):
        """One drafter step: (hidden at pos, token at pos+1) -> logits for pos+2.

        Args:
            token_id: int — the input token (target token at position+1).
            hidden_row: torch [dim] — target post-final-norm hidden at `position`
                (or the drafter's own hidden from the previous chained step).
            position: int — drafter RoPE position and KV slot.

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
        logits, normed = self._step_graph(tok, h_in, cur_pos, cos, sin)
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

    def _step_graph(self, tok, h_in, cur_pos, cos, sin):
        """The drafter-step op graph over device inputs. Returns (logits, normed).

        Shared by the eager step() and the trace capture/replay path — everything
        here is device ops only (trace-capture safe).
        """
        emb = self.embedding(tok)  # [1,1,dim]
        e_n = rms_norm_ttnn(emb, self.pre_fc_norm_embedding, eps=self.norm_eps, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(emb)
        h_n = rms_norm_ttnn(h_in, self.pre_fc_norm_hidden, eps=self.norm_eps, memory_config=ttnn.L1_MEMORY_CONFIG)

        fused = ttnn.concat([e_n, h_n], dim=-1)  # [1,1,2*dim] — embedding first (vLLM order)
        ttnn.deallocate(e_n)
        ttnn.deallocate(h_n)
        x = ttnn.linear(
            fused, self.fc_weight, compute_kernel_config=self.compute_kernel_config, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        ttnn.deallocate(fused)

        # Decoder block (decode-mode residual pattern, as in Qwen36DecoderLayer).
        attn_in = self.attention_norm(x, mode=Mode.DECODE, norm_config={"output_mem_config": ttnn.L1_MEMORY_CONFIG})
        attn_out = self.attention.forward(attn_in, cos, sin, position_tensor=cur_pos, page_table=self._page_table_tt)
        ttnn.deallocate(attn_in)
        h1 = ttnn.add(x, attn_out)
        ttnn.deallocate(x)
        ttnn.deallocate(attn_out)

        ff_in = self.ffn_norm(h1, mode=Mode.DECODE, norm_config={"output_mem_config": ttnn.L1_MEMORY_CONFIG})
        ff_out = self.feed_forward.forward(ff_in)
        ttnn.deallocate(ff_in)
        out = ttnn.add(h1, ff_out)
        ttnn.deallocate(h1)
        ttnn.deallocate(ff_out)

        normed = rms_norm_ttnn(out, self.final_norm, eps=self.norm_eps, memory_config=ttnn.L1_MEMORY_CONFIG)
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

    # ── traced step ──────────────────────────────────────────────────────────
    def _host_step_inputs(self, token_id, position):
        """Host-side (tok, pos, cos, sin) tensors for copy_host_to_device_tensor."""
        # int64 host source for the uint32 tensor avoids the int32->uint32
        # row-major conversion warning (#18536).
        tok = ttnn.from_torch(
            torch.tensor([[token_id]], dtype=torch.int64), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        pos = ttnn.from_torch(
            torch.tensor([position], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        cos = ttnn.from_torch(
            self.rope.cos_cpu[position : position + 1].unsqueeze(0).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        sin = ttnn.from_torch(
            self.rope.sin_cpu[position : position + 1].unsqueeze(0).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        return tok, pos, cos, sin

    def _capture_step_trace(self, token_id, hidden_row, position):
        """Capture the drafter step at the REAL first-call inputs.

        Persistent inputs get stable addresses; the compile + capture runs write
        the SAME KV slot with the SAME values as the first replay, so the writes
        are idempotent. Must run before any other trace is parked (an eager
        compile after a parked trace can clobber it — the spec loop guarantees
        this by capturing the drafter before the verify trace).
        """
        mk = self._mesh_kwargs
        tr = {
            "tok": ttnn.from_torch(
                torch.tensor([[token_id]], dtype=torch.int64),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                **mk,
            ),
            "pos": ttnn.from_torch(
                torch.tensor([position], dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                **mk,
            ),
            "cos": ttnn.from_torch(
                self.rope.cos_cpu[position : position + 1].unsqueeze(0).contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                **mk,
            ),
            "sin": ttnn.from_torch(
                self.rope.sin_cpu[position : position + 1].unsqueeze(0).contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                **mk,
            ),
            "h": ttnn.from_torch(
                hidden_row.reshape(1, 1, -1).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                **mk,
            ),
        }
        # Compile run (warm the program cache), then capture.
        logits, normed = self._step_graph(tr["tok"], tr["h"], tr["pos"], tr["cos"], tr["sin"])
        ttnn.synchronize_device(self.device)
        ttnn.deallocate(logits)
        ttnn.deallocate(normed)
        tid = ttnn.begin_trace_capture(self.device, cq_id=0)
        logits, normed = self._step_graph(tr["tok"], tr["h"], tr["pos"], tr["cos"], tr["sin"])
        ttnn.end_trace_capture(self.device, tid, cq_id=0)
        tr.update({"id": tid, "logits": logits, "normed": normed})
        self._step_trace = tr

    def step_traced(self, token_id, hidden_row, position, chain_hidden=False):
        """Traced drafter step: refresh persistent inputs, replay, read logits.

        chain_hidden=True feeds the PREVIOUS traced step's output hidden (kept on
        device — ttnn.copy into the persistent input, no host round-trip); else
        hidden_row (torch [dim]) is uploaded. Returns host logits [vocab].
        """
        if self._step_trace is None:
            self._capture_step_trace(token_id, hidden_row if not chain_hidden else torch.zeros(self.args.dim), position)
            # fall through: replay with the real inputs
        tr = self._step_trace
        tok_h, pos_h, cos_h, sin_h = self._host_step_inputs(token_id, position)
        ttnn.copy_host_to_device_tensor(tok_h, tr["tok"])
        ttnn.copy_host_to_device_tensor(pos_h, tr["pos"])
        ttnn.copy_host_to_device_tensor(cos_h, tr["cos"])
        ttnn.copy_host_to_device_tensor(sin_h, tr["sin"])
        if chain_hidden:
            # Recurrence stays on device: previous replay's normed output -> input.
            ttnn.copy(tr["normed"], tr["h"])
        else:
            h_h = ttnn.from_torch(
                hidden_row.reshape(1, 1, -1).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            )
            ttnn.copy_host_to_device_tensor(h_h, tr["h"])
        ttnn.execute_trace(self.device, tr["id"], cq_id=0, blocking=False)
        return self._logits_to_host(tr["logits"])

    def release_step_trace(self):
        if self._step_trace is not None:
            ttnn.release_trace(self.device, self._step_trace["id"])
            self._step_trace = None
